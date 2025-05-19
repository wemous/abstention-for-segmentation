import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import DSAD, CaDIS, NoisyCaDIS, NoisyDSAD
from models import SegmentationModel

pl.seed_everything(1, workers=True)
torch.set_float32_matmul_precision("high")

wandb.login()
wandb.Settings(quiet=True)


def main():
    run = wandb.init()
    config = wandb.config

    num_classes = 8
    max_epochs = 50

    train_dataset = NoisyCaDIS(noise_level=config.noise_level, setup=1)
    valid_dataset = CaDIS(split="valid", setup=1)
    test_dataset = CaDIS(split="test")
    batch_size = 128

    # train_dataset = NoisyDSAD(noise_level=5)
    # valid_dataset = DSAD(split="valid")
    # test_dataset = DSAD(split="test")
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

    noise_rate = train_dataset.noise_rate.round(decimals=2)

    loss_config = {
        "name": "GACLoss",
        "args": {
            "max_epochs": max_epochs,
            "noise_rate": noise_rate,
            "alpha_final": config.alpha_final,
            "gamma": config.gamma,
            "warmup_epochs": config.warmup_epochs,
            "q": config.q,
        },
    }

    lr = 3e-3

    model = SegmentationModel(
        num_classes,
        loss_config,
        lr=lr,
        model_name="UNet",
        include_background=isinstance(train_dataset, NoisyCaDIS),
    )

    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        deterministic="warn",
        log_every_n_steps=len(train_loader) // 4,
        logger=WandbLogger(id=run.id),
    )

    trainer.fit(model, train_loader, valid_loader)
    wandb.log({"valid/miou_final": run.summary.get("valid/miou")})
    trainer.logger = None
    final_metrics = trainer.test(model, test_loader)[0]
    wandb.log(
        {
            "test/accuracy_final": final_metrics["test/accuracy"],
            "test/dice_final": final_metrics["test/dice"],
            "test/miou_final": final_metrics["test/miou"],
        }
    )
    wandb.finish()


main()
