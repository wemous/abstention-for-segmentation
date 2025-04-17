import os
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import DSAD, CaDIS, NoisyCaDIS, NoisyDSAD
from models import SegmentationModel

torch.set_float32_matmul_precision("high")
wandb.login()


def main():
    run = wandb.init()
    config = run.config
    pl.seed_everything(config.seed, workers=True)

    max_epochs = 50
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
    wandb.log({"noise rate": noise_rate})

    if "DAC" in config.loss:
        num_classes += 1

    loss_config = {
        "name": config.loss,
        "args": {
            "noise_rate": noise_rate,
            "max_epochs": max_epochs,
        },
    }

    lr = 3e-3

    model = SegmentationModel(
        num_classes,
        loss_config,
        lr,
        model_name=config.model,
        window_size=16,
        include_background=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/mIoU",
        mode="max",
        save_top_k=1,
        filename="{epoch}",
    )

    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,
        enable_progress_bar=True,
        deterministic="warn",
        log_every_n_steps=len(train_loader) // 3,
        logger=WandbLogger(id=run.id),
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.logger = None
    checkpoint_path = checkpoint_callback.best_model_path
    checkpoint = SegmentationModel.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
    )
    wandb.log({"best epoch": int(Path(checkpoint_path).stem[6:])})
    #
    final_metrics = trainer.test(model, test_loader)[0]
    wandb.log(
        {
            "test/accuracy_final": final_metrics["test/accuracy"],
            "test/dice_final": final_metrics["test/dice"],
            "test/miou_final": final_metrics["test/miou"],
        }
    )

    best_metrics = trainer.test(checkpoint, test_loader)[0]
    wandb.log(
        {
            "test/accuracy_best": best_metrics["test/accuracy"],
            "test/dice_best": best_metrics["test/dice"],
            "test/miou_best": best_metrics["test/miou"],
        }
    )
    os.remove(checkpoint_path)

    wandb.finish(quiet=True)


main()
