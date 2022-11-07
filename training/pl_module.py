import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from models.qagnn import LM_QAGNN
from training import get_loss, get_optimizer
from .metrics import calculate_confusion_matrix, calculate_f1_score


class QAModule(LightningModule):
    def __init__(self, config: DictConfig):
        self.save_hyperparameters()
        self.config = config
        super().__init__()

        node_embeddings = np.load(config.data.node_embeddings_path)
        node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)

        num_nodes, embedding_dim = node_embeddings.shape[0], node_embeddings.shape[1]
        print(f'| num_nodes: {num_nodes} | embedding_dim: {embedding_dim}')

        self.dataset_config, self.training_config = config.data, config.training
        self.encoder_config, self.decoder_config = config.model.encoder, config.model.decoder
        self.optimizer_config = config.optimization

        self.model = LM_QAGNN(
            encoder_name=self.encoder_config.name,
            gnn_name=self.decoder_config.name,
            n_gnn_layers=self.decoder_config.num_layers,
            n_vertex_types=3,
            n_edge_types=self.dataset_config.num_relation,
            n_concept=num_nodes,
            concept_dim=self.decoder_config.gnn_dim,
            concept_in_dim=embedding_dim,
            n_attn_head=self.decoder_config.att_head_num,
            fc_dim=self.decoder_config.fc_dim,
            n_fc_layers=self.decoder_config.fc_layer_num,
            dropout_prob_emb=self.decoder_config.dropouti,
            dropout_prob_gnn=self.decoder_config.dropoutg,
            dropout_prob_fc=self.decoder_config.dropoutf,
            pretrained_concept_emb=node_embeddings,
            freeze_ent_emb=self.training_config.freeze_ent_emb,
            init_range=self.decoder_config.init_range,
        )

        self.loss = get_loss(self.training_config, ignore_index=-1)

    def training_step(self, batch, batch_idx):
        logits, _ = self.model(batch, layer_id=self.encoder_config.layer)
        loss = self.loss(logits, batch.labels)
        self.log_dict({"train/loss": loss})
        return loss

    def on_train_start(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.training_config.unfreeze_epoch:
            for p in self.model.encoder.parameters():
                p.requires_grad = True
        if self.current_epoch == self.training_config.refreeze_epoch:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def validation_step(self, batch, batch_idx):
        logits, _ = self.model(batch, layer_id=self.encoder_config.layer)
        loss = self.loss(logits, batch.labels)
        predictions = logits.argmax(dim=1)[batch.labels != -1].detach().cpu()
        true_labels = batch.labels[batch.labels != -1].detach().cpu()
        confusion_matrix = calculate_confusion_matrix(predictions, true_labels)
        return {"loss": loss, **confusion_matrix}

    def validation_epoch_end(self, outputs):
        loss, tp, tn, fp, fn = 0, 0, 0, 0, 0
        for o in outputs:
            batch_size = o["TP"] + o["TN"] + o["FP"] + o["FN"]
            loss += o["loss"] * batch_size
            tp += o["TP"]
            tn += o["TN"]
            fp += o["FP"]
            fn += o["FN"]

        n_examples = tp + tn + fp + fn
        loss = loss / n_examples
        acc = (tp + tn) / n_examples
        f1_score = calculate_f1_score(tp, tn, fp, fn)
        self.log_dict({"val/loss": loss, "val/accuracy": acc, "val/f1-score": f1_score})

    def test_step(self, batch, batch_idx):
        logits, _ = self.model(batch, layer_id=self.encoder_config.layer)
        loss = self.loss(logits, batch.labels)
        predictions = logits.argmax(dim=1)[batch.labels != -1].detach().cpu()
        true_labels = batch.labels[batch.labels != -1].detach().cpu()
        confusion_matrix = calculate_confusion_matrix(predictions, true_labels)
        return {"loss": loss, **confusion_matrix}

    def test_epoch_end(self, outputs):
        loss, tp, tn, fp, fn = 0, 0, 0, 0, 0
        for o in outputs:
            batch_size = o["TP"] + o["TN"] + o["FP"] + o["FN"]
            loss += o["loss"] * batch_size
            tp += o["TP"]
            tn += o["TN"]
            fp += o["FP"]
            fn += o["FN"]

        n_examples = tp + tn + fp + fn
        loss = loss / n_examples
        acc = (tp + tn) / n_examples
        f1_score = calculate_f1_score(tp, tn, fp, fn)
        self.log_dict({"test/loss": loss, "test/accuracy": acc, "test/f1-score": f1_score})

    def configure_optimizers(self):
        return get_optimizer(
            config=self.optimizer_config,
            model=self.model,
            total_steps=self.trainer.estimated_stepping_batches
        )
