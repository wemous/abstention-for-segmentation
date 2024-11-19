import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import CaDIS, NoisyCaDIS, DSAD, NoisyDSAD
import models

pl.seed_everything(13)
torch.set_float32_matmul_precision("high")

wandb.login()


def main():
    run = wandb.init(project="thesis")
    config = wandb.config

    max_epochs = 100
    noise_level = config.noise_level
    dataset_name = config.dataset["name"]
    augmentations = config.dataset["augmentations"]
    batch_size = config.dataset["batch_size"]

    if dataset_name == "cadis":
        setup = config.dataset["setup"]
        train_dataset = NoisyCaDIS(noise_level=noise_level, setup=setup, **augmentations)
        valid_dataset = CaDIS(split="valid", setup=setup)
        test_dataset = CaDIS(split="test", setup=setup)
        num_classes = test_dataset.num_classes[setup]
    elif dataset_name == "dsad":
        train_dataset = NoisyDSAD(noise_level=noise_level, **augmentations)
        valid_dataset = DSAD(split="valid")
        test_dataset = DSAD(split="test")
        num_classes = 8

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

    noise_rate = round(train_dataset.noise_rate, 2)

    if "DAC" in config.loss:
        num_classes += 1

    loss_config = {
        "name": config.loss,
        "args": {"noise_rate": noise_rate, "max_epochs": max_epochs},
    }

    optimizer_args = {
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 5e-3,
    }

    model = getattr(models, config.model)(
        num_classes,
        loss_config,
        **optimizer_args,
    )

    wandb.log({"noise rate": noise_rate})

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
