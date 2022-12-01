import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from models.two_tower_MLP_node_classification import TwoTowerMLPNodeClassifier
from models.MLP_node_classification import MLPNodeClassifier
from models.QAGNN_node_classification import QAGNNNodeClassifier
from models.LM_graph_classification import LMGraphClassifier
from models.QAGNN_graph_classification import QAGNNGraphClassifier
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

        if config.model.name == "two_tower_MLP_node_classification":
            self.model = TwoTowerMLPNodeClassifier(
                encoder_name=self.encoder_config.name,
                n_concept=num_nodes,
                concept_dim=self.decoder_config.hidden_dim,
                concept_in_dim=embedding_dim,
                dropout_prob_emb=self.decoder_config.dropout_emb,
                pretrained_concept_emb=node_embeddings,
                freeze_ent_emb=self.training_config.freeze_ent_emb,
                init_range=self.decoder_config.init_range,
            )
        elif config.model.name == "MLP_node_classification":
            self.model = MLPNodeClassifier(
                encoder_name=self.encoder_config.name,
                fc_dim=self.decoder_config.fc_dim,
                n_fc_layers=self.decoder_config.fc_layer_num,
                dropout_prob_fc=self.decoder_config.dropout_fc,
                n_concept=num_nodes,
                concept_dim=self.decoder_config.hidden_dim,
                concept_in_dim=embedding_dim,
                dropout_prob_emb=self.decoder_config.dropout_emb,
                pretrained_concept_emb=node_embeddings,
                freeze_ent_emb=self.training_config.freeze_ent_emb,
                init_range=self.decoder_config.init_range,
            )
        elif config.model.name == "QAGNN_node_classification":
            self.model = QAGNNNodeClassifier(
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
                dropout_prob_emb=self.decoder_config.dropout_emb,
                dropout_prob_gnn=self.decoder_config.dropout_gnn,
                dropout_prob_fc=self.decoder_config.dropout_fc,
                pretrained_concept_emb=node_embeddings,
                freeze_ent_emb=self.training_config.freeze_ent_emb,
                init_range=self.decoder_config.init_range,
            )
        elif config.model.name == "LM_graph_classification":
            self.model = LMGraphClassifier(
                encoder_name=self.encoder_config.name,
                fc_dim=self.decoder_config.fc_dim,
                n_fc_layers=self.decoder_config.fc_layer_num,
                dropout_prob_fc=self.decoder_config.dropout_fc,
            )
        elif config.model.name == "QAGNN_graph_classification":
            self.model = QAGNNGraphClassifier(
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
                dropout_prob_emb=self.decoder_config.dropout_emb,
                dropout_prob_gnn=self.decoder_config.dropout_gnn,
                dropout_prob_fc=self.decoder_config.dropout_fc,
                pretrained_concept_emb=node_embeddings,
                freeze_ent_emb=self.training_config.freeze_ent_emb,
                init_range=self.decoder_config.init_range,
            )
        else:
            raise ValueError(f"Unknown model name {config.model.name}")

        self.loss = get_loss(self.training_config, ignore_index=-1)

    def predictions_from_logits(self, logits, mask):
        if self.training_config.loss == 'binary_cross_entropy':
            predictions = (logits[mask] > 0.5).long().detach().cpu()
        else:
            predictions = logits.argmax(dim=1)[mask].detach().cpu()

        return predictions

    def mask_labels(self, labels, pos_p: int = 0.0, neg_p: int = 0.0):
        """
        Drop positive labels with pos_p% probability and
        negative labels with neg_p% probability
        """

        negative_mask = (labels == 0)
        positive_mask = (labels == 1)
        drop_probabilities = torch.zeros(labels.shape)
        drop_probabilities[negative_mask] = neg_p
        drop_probabilities[positive_mask] = pos_p
        drop_instances = torch.bernoulli(drop_probabilities).bool()
        labels[drop_instances] = -1
        return labels

    def training_step(self, batch, batch_idx):
        logits = self.model(batch, layer_id=self.encoder_config.layer)

        if self.training_config.label_masking:
            # Drop positive labels with 5% probability and negative labels 
            # with 80% probability so that the model sees a balanced dataset.
            batch.labels = self.mask_labels(batch.labels, pos_p=0.05, neg_p=0.80)

        loss = self.loss(logits, batch.labels)
        self.log_dict({"train/loss": loss})
        return loss
        
    def on_train_start(self) -> None:
        if self.training_config.unfreeze_epoch != -1:
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
        logits = self.model(batch, layer_id=self.encoder_config.layer)
        loss = self.loss(logits, batch.labels)

        mask = (batch.labels != -1)
        predictions = self.predictions_from_logits(logits, mask)
        true_labels = batch.labels[mask].detach().cpu()
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
        f1_score, precision, recall = calculate_f1_score(tp, tn, fp, fn)
        self.log_dict({
            "val/loss": loss,
            "val/accuracy": acc,
            "val/f1-score": f1_score,
            "val/precision": precision,
            "val/recall": recall
        })

    def test_step(self, batch, batch_idx):
        logits = self.model(batch, layer_id=self.encoder_config.layer)
        loss = self.loss(logits, batch.labels)

        mask = (batch.labels != -1)
        predictions = self.predictions_from_logits(logits, mask)
        true_labels = batch.labels[mask].detach().cpu()
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
        f1_score, precision, recall = calculate_f1_score(tp, tn, fp, fn)
        self.log_dict({
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1-score": f1_score,
            "test/precision": precision,
            "test/recall": recall
        })

    def configure_optimizers(self):
        return get_optimizer(
            config=self.optimizer_config,
            model=self.model,
            total_steps=self.trainer.estimated_stepping_batches
        )
