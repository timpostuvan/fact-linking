import os
from pathlib import Path
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets import ComFactDataModule
from training import QAModule


def train(config_path: str):
    config = OmegaConf.load(config_path)

    seed_everything(config.seed, workers=True)

    log_path = os.path.join(config.save_dir, "log.csv")
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
        precision=16 if config.training.fp16 else 32,
        deterministic=True
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    args = arg_parser.parse_args()
    train(config_path=args.config_path)
