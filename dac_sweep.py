import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import CaDIS, NoisyCaDIS, DSAD, NoisyDSAD
from models import UNet, DeepLabV3Plus, FPN

pl.seed_everything(13)
torch.set_float32_matmul_precision("high")

wandb.login()


def main():
    run = wandb.init()
    config = wandb.config

    max_epochs = 30

    train_dataset = NoisyCaDIS(noise_level=3, setup=1)
    valid_dataset = CaDIS(split="valid", setup=1)
    test_dataset = CaDIS(split="test", setup=1)
    num_classes = test_dataset.num_classes[1]
    batch_size = 128

    # train_dataset = NoisyDSAD(noise_level=3)
    # valid_dataset = DSAD(split="valid")
    # test_dataset = DSAD(split="test")
    # num_classes = 8
    # batch_size = 64

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
            "warmup_rate": config.warmup_rate,
            "alpha_final": config.alpha_final,
            "alpha_init_factor": config.alpha_init_factor,
            "mu": config.mu,
        },
    }
    optimizer_args = {
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 5e-3,
    }

    model = UNet(num_classes + 1, loss_config, **optimizer_args)
    # model = DeepLabV3Plus(num_classes+1, loss_config)
    # model = FPN(num_classes+1, loss_config)

    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=len(train_loader) // 5,
        logger=WandbLogger(id=run.id),
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    wandb.finish(quiet=True)


main()
