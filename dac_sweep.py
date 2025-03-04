import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import CaDIS, NoisyCaDIS, DSAD, NoisyDSAD
from models import SegmentationModel

torch.use_deterministic_algorithms(mode=True, warn_only=True)
pl.seed_everything(1, workers=True)
torch.set_float32_matmul_precision("high")

wandb.login()


def main():
    run = wandb.init()
    config = wandb.config

    max_epochs = 50

    train_dataset = NoisyCaDIS(noise_level=5, setup=1)
    valid_dataset = CaDIS(split="valid", setup=1)
    test_dataset = CaDIS(split="test", setup=1)
    num_classes = test_dataset.num_classes[1]
    batch_size = config.batch_size

    # train_dataset = NoisyDSAD(noise_level=3)
    # valid_dataset = DSAD(split="valid")
    # test_dataset = DSAD(split="test")
    # num_classes = 8
    # batch_size = 50

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

    loss_config = {
        "name": "DACLoss",
        "args": {
            "max_epochs": max_epochs,
            "warmup_epochs": config.warmup_epochs,
            "alpha_final": config.alpha_final,
        },
    }
    optimizer_args = {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-3,
    }

    model = SegmentationModel(
        num_classes + 1,
        loss_config,
        model_name="UNet",
        window_size=16,
        include_background=True,
        **optimizer_args,
    )

    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=len(train_loader) // 3,
        logger=WandbLogger(id=run.id),
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    wandb.finish(quiet=True)


main()
