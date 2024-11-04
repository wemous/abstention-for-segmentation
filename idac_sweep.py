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

    batch_size = 128
    max_epochs = 50

    train_dataset = NoisyCaDIS(noise_level=3, setup=1)
    valid_dataset = CaDIS(split="valid", setup=1)
    test_dataset = CaDIS(split="test", setup=1)
    num_classes = test_dataset.num_classes[1]

    # train_dataset = NoisyDSAD(noise_level=3)
    # valid_dataset = DSAD(split="valid")
    # test_dataset = DSAD(split="test")
    # num_classes = 8

    noise_rate = train_dataset.noise_rate
    wandb.log({"noise rate": noise_rate})

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

    warmup_epochs = int(max_epochs * config.warmup_rate) + 1

    loss_config = {
        "name": "IDACLoss",
        "args": {
            "noise_rate": round(noise_rate, 2),
            "warmup_epochs": warmup_epochs,
            "alpha": config.alpha,
        },
    }

    model = UNet(num_classes + 1, loss_config)
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
