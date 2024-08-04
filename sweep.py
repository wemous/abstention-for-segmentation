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

    noise_rate = config.noise_rate
    noise_type = config.noise_type
    dataset = config.dataset
    setup = dataset["setup"]
    image_size = dataset["image_size"]
    batch_size = dataset["batch_size"]
    max_epochs = dataset["max_epochs"]

    if dataset["name"] == "ade20k":
        train_dataset = ADE20K(
            split=0,
            setup=setup,
            image_size=image_size,
            noise_rate=noise_rate,
            noise_type=noise_type,
        )
        valid_dataset = ADE20K(split=1, setup=setup, image_size=image_size)
        test_dataset = ADE20K(split=2, setup=setup, image_size=image_size)
    elif dataset["name"] == "cadis":
        train_dataset = CaDIS(
            split=0,
            setup=setup,
            image_size=image_size,
            noise_rate=noise_rate,
            noise_type=noise_type,
        )
        valid_dataset = CaDIS(split=1, setup=setup, image_size=image_size)
        test_dataset = CaDIS(split=2, setup=setup, image_size=image_size)

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

    if config.loss == "DACLoss":
        loss_args = {"max_epochs": max_epochs}
        num_classes += 1
    elif config.loss == "IDACLoss":
        loss_args = {"noise_rate": noise_rate, "warmup_epochs": max_epochs // 5}
        num_classes += 1
    else:
        loss_args = {}

    loss_config = {
        "module": "losses",
        "name": config.loss,
        "args": loss_args,
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
        log_every_n_steps=len(train_loader) // 4 if dataset["name"] == "cadis" else None,
        logger=logger,
        gradient_clip_val=0.5,
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    wandb.finish(quiet=True)


main()
