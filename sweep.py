from importlib import import_module

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from yaml import full_load

import wandb
from datasets import ADE20K, CaDIS

config = full_load(open("configs/sweep_config.yaml"))
pl.seed_everything(config["seed"])
torch.set_float32_matmul_precision("high")

models = config["models"]
losses = config["losses"]
optimizers = config["optimizers"]
datasets = config["datasets"]

wandb.login()


def train(
    model_name: str,
    optimizer_config: dict,
    loss_name: str,
    num_classes: int,
    max_epochs: int,
):
    loss_args = {"max_epochs": max_epochs} if loss_name == "DACLoss" else {}
    loss_args["num_classes"] = num_classes
    loss_config = {
        "module": "losses",
        "name": loss_name,
        "args": loss_args,
    }
    if loss_name == "DACLoss":
        num_classes += 1
    model = getattr(import_module("models"), model_name)(
        num_classes,
        loss_config,
        optimizer_config,
    )
    run_name = f"{model_name.lower()}/{dataset_name}/{loss_name[:-4].lower()}/{optimizer['name'].lower()}"

    earlystop = EarlyStopping(
        monitor="valid/loss",
        mode="min",
        min_delta=0.03,
        patience=5,
        check_finite=True,
        verbose=False,
    )

    logger = WandbLogger(project="DAC Segmentation", name=run_name)
    trainer = pl.Trainer(
        # devices=config["devices"],
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=len(train_loader) // 5,
        logger=logger,
        # callbacks=[earlystop],
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    wandb.finish(quiet=True)


for dataset in datasets:
    dataset_name = dataset["name"]
    image_size = dataset["image_size"]
    batch_size = dataset["batch_size"]
    setup = dataset["setup"]
    max_epochs = dataset["max_epochs"]

    if dataset_name == "ade20k":
        train_dataset = ADE20K(split=0, setup=setup, image_size=image_size)
        valid_dataset = ADE20K(split=1, setup=setup, image_size=image_size)
        test_dataset = ADE20K(split=2, setup=setup, image_size=image_size)
    elif dataset_name == "cadis":
        transforms = dataset["transforms"]
        train_dataset = CaDIS(
            split=0,
            setup=setup,
            image_size=image_size,
            transforms=transforms,
        )
        valid_dataset = CaDIS(split=1, setup=setup, image_size=image_size)
        test_dataset = CaDIS(split=2, setup=setup, image_size=image_size)

    num_classes = train_dataset.num_classes[setup]

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

    for model_name in models:
        for optimizer in optimizers:
            optimizer_config = {
                "module": "torch.optim",
                "name": optimizer["name"],
                "args": optimizer["args"],
            }
            for loss_name in losses:
                train(model_name, optimizer_config, loss_name, num_classes, max_epochs)
