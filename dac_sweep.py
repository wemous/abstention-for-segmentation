import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import DSAD, CaDIS, NoisyCaDIS, NoisyDSAD
from models import SegmentationModel

pl.seed_everything(1, workers=True)
torch.set_float32_matmul_precision("high")

wandb.login()


def main():
    run = wandb.init()
    config = wandb.config

    max_epochs = 50
    batch_size = 128
    num_classes = 8

    train_dataset = NoisyCaDIS(noise_level=5, setup=1)
    valid_dataset = CaDIS(split="valid", setup=1)

    # train_dataset = NoisyDSAD(noise_level=3)
    # valid_dataset = DSAD(split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    loss_config = {
        "name": "DACLoss",
        "args": {
            "max_epochs": max_epochs,
            "warmup_epochs": config.warmup_epochs,
            "alpha_final": config.alpha_final,
        },
    }

    lr = config.lr

    model = SegmentationModel(
        num_classes + 1,
        loss_config,
        lr,
        model_name="UNet",
        include_background=True,
    )

    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        deterministic="warn",
        log_every_n_steps=len(train_loader) // 3,
        logger=WandbLogger(id=run.id),
    )

    trainer.fit(model, train_loader, valid_loader)
    wandb.finish(quiet=True)


main()
