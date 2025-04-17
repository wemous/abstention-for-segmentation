import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import DSAD, CaDIS, NoisyCaDIS, NoisyDSAD
from models import SegmentationModel

pl.seed_everything(0, workers=True)
torch.set_float32_matmul_precision("high")

wandb.login()


def main():
    run = wandb.init()
    config = wandb.config

    max_epochs = 50
    batch_size = 128

    train_dataset = NoisyCaDIS(noise_level=5, setup=1)
    valid_dataset = CaDIS(split="valid", setup=1)
    test_dataset = CaDIS(split="test", setup=1)
    num_classes = test_dataset.num_classes[1]

    # train_dataset = NoisyDSAD(noise_level=3)
    # valid_dataset = DSAD(split="valid")
    # test_dataset = DSAD(split="test")
    # num_classes = 8

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    wandb.log({"noise rate": train_dataset.noise_rate})

    loss_config = {
        "name": "SCELoss",
        "args": {
            "alpha": config.alpha,
            "A": config.A,
        },
    }

    lr = 3e-3

    model = SegmentationModel(
        num_classes,
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
