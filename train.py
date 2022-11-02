from os.path import join
from pathlib import Path
import argparse

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets import ComFactDataModule
from training import QAModule


def train(config: DictConfig):
    seed_everything(config.seed, workers=True)

    log_path = join(config.save_dir, "log.csv")
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    with open(log_path, "w+") as fout:
        fout.write("step,dev_acc,test_acc\n")

    dm = ComFactDataModule(config)
    model = QAModule(config)

    # define loggers
    wandb_logger = WandbLogger(
        project=f"qagnn-{config.task}",
        save_dir=config.save_dir,
        log_model=True,
        offline=False
    )
    wandb_logger.watch(model)
    # list of callbacks
    callbacks = []
    if config.checkpoint:
        # define model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=wandb_logger.experiment.dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=-1,
        )
        callbacks.append(checkpoint_callback)
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    callbacks.append(lr_logger)
    trainer = Trainer(
        max_epochs=config.training.num_epochs,
        val_check_interval=0.5,
        log_every_n_steps=config.training.log_interval,
        logger=wandb_logger,
        accelerator="auto",     # uses GPU if available
        callbacks=callbacks,
        gradient_clip_val=config.training.max_grad_norm,
        accumulate_grad_batches=config.data.accumulate_grad_batches,
        precision=16 if config.training.fp16 else 32
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


def update_from_cli(args: argparse.Namespace, config: DictConfig):
    if args.encoder_lr is not None:
        config.optimization.encoder_lr = args.encoder_lr
    if args.decoder_lr is not None:
        config.optimization.decoder_lr = args.decoder_lr
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.gnn_dim is not None:
        config.model.decoder.gnn_dim = args.gnn_dim
        config.model.decoder.fc_dim = args.gnn_dim
    if args.num_layers is not None:
        config.model.decoder.num_layers = args.num_layers

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=join("configs", "qagnn.yaml"), type=str)
    parser.add_argument("--encoder_lr", default=None, type=float)
    parser.add_argument("--decoder_lr", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--gnn_dim", default=None, type=int)
    parser.add_argument("--num_layers", default=None, type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config = update_from_cli(args, config)
    train(config=config)
