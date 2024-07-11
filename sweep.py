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
    dataset = config.dataset
    setup = dataset["setup"]
    image_size = dataset["image_size"]
    batch_size = dataset["batch_size"]
    max_epochs = dataset["max_epochs"]

    if dataset["name"] == "ade20k":
        train_dataset = ADE20K(split=0, setup=setup)
        valid_dataset = ADE20K(split=1, setup=setup)
        test_dataset = ADE20K(split=2, setup=setup)
    elif dataset["name"] == "cadis":
        train_dataset = CaDIS(split=0, setup=setup)
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

    num_classes = train_dataset.num_classes[setup]
    is_dac = config.loss == "DACLoss"
    num_classes = num_classes + 1 if is_dac else num_classes

    loss_config = {
        "module": "losses",
        "name": config.loss,
        "args": {"max_epochs": max_epochs} if is_dac else {},
    }

    model = getattr(import_module("models"), config.model)(
        num_classes,
        loss_config,
        config.optimizer,
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
