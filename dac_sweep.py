from importlib import import_module

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import ADE20K, CaDIS

pl.seed_everything(13)
torch.set_float32_matmul_precision("high")

wandb.login()


def main():
    run = wandb.init()
    config = wandb.config

    if config.dataset == "ade20k":
        setup = 1
        batch_size = 16
        max_epochs = 30
        train_dataset = ADE20K(split=0, setup=setup)
        valid_dataset = ADE20K(split=1, setup=setup)
        test_dataset = ADE20K(split=2, setup=setup)
    elif config.dataset == "cadis":
        setup = 3
        batch_size = 64
        max_epochs = 50
        train_dataset = CaDIS(
            split=0,
            setup=setup,
        )
        valid_dataset = CaDIS(split=1, setup=setup)
        test_dataset = CaDIS(split=2, setup=setup)

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

    num_classes = train_dataset.num_classes[setup] + 1
    warmup_epochs = int(max_epochs * config.warmup_rate)

    loss_config = {
        "module": "losses",
        "name": "DACLoss",
        "args": {
            "max_epochs": max_epochs,
            "warmup_epochs": warmup_epochs,
            "alpha_final": config.alpha_final,
            "alpha_init_factor": config.alpha_init_factor,
            "mu": config.mu,
        },
    }
    optimizer_config = {
        "module": "torch.optim",
        "name": "SGD",
        "args": {
            "lr": config.lr,
            "nesterov": True,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
    }

    model = getattr(import_module("models"), config.model)(
        num_classes,
        loss_config,
        optimizer_config,
    )

    logger = WandbLogger(project="DAC Segmentation", id=run.id)
    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=len(train_loader) // 5,
        logger=logger,
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    wandb.finish(quiet=True)


main()
