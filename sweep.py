import os
from pathlib import Path

import lightning as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.data import DataLoader

import wandb
from datasets import DSAD, CaDIS, NoisyCaDIS, NoisyDSAD
from models import SegmentationModel

torch.set_float32_matmul_precision("high")
wandb.login()
wandb.Settings(quiet=True)


def mask_to_pil(mask: torch.Tensor):
    mask_array = mask.squeeze().cpu().numpy()
    cmap = plt.get_cmap("gnuplot2", 8)
    mask_array = cmap(mask_array / 7)[:, :, :3]
    mask_array = (mask_array * 255).astype("uint8")
    pil_mask = Image.fromarray(mask_array)
    return pil_mask


def main():
    run = wandb.init()
    config = run.config
    pl.seed_everything(config.seed, workers=True)

    max_epochs = 50
    num_classes = 8

    if config.dataset == "cadis":
        train_dataset = NoisyCaDIS(noise_level=config.noise_level)
        valid_dataset = CaDIS(split="valid")
        test_dataset = CaDIS(split="test")
        batch_size = 128
    else:
        train_dataset = NoisyDSAD(noise_level=config.noise_level)
        valid_dataset = DSAD(split="valid")
        test_dataset = DSAD(split="test")
        batch_size = 50

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
    class_noise = train_dataset.class_noise

    loss_config = {
        "name": config.loss,
        "args": {
            "max_epochs": max_epochs,
            "noise_rate": noise_rate,
            "class_noise": class_noise,
        },
    }

    lr = 3e-3

    model = SegmentationModel(
        num_classes,
        loss_config,
        lr=lr,
        decoder=config.model,
        include_background=isinstance(train_dataset, NoisyCaDIS),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/miou",
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
        log_every_n_steps=len(train_loader) // 4,
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

    index = 186 if config.dataset == "cadis" else 212
    sample = model.predict(test_dataset[index][0])
    pil_sample = mask_to_pil(sample)
    pil_sample.save(f"images/{config.dataset}-{config.loss[:-4].lower()}.png")
    wandb.log({"Sample": wandb.Image(pil_sample)})
    wandb.finish()


main()
