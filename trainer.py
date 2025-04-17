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

seed = 1
# torch.use_deterministic_algorithms(True, warn_only=True)
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision("high")

max_epochs = 50
batch_size = 128
num_classes = 8

train_dataset = NoisyCaDIS(noise_level=5, setup=1)
valid_dataset = CaDIS(split="valid", setup=1)
test_dataset = CaDIS(split="test", setup=1)

# train_dataset = NoisyDSAD(noise_level=5, normalized=True, rotated=True)
# valid_dataset = DSAD(split="valid")
# test_dataset = DSAD(split="test")


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

# loss = {
#     "name": "CELoss",
#     "args": {},
# }
loss = {
    "name": "DiceLoss",
    "args": {},
}
# loss = {
#     "name": "ADSLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "warmup_epochs": 10,
#         "alpha_final": 1.0,
#     },
# }
# loss = {
#     "name": "DACLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "warmup_epochs": 10,
#         "alpha_final": 1.0,
#     },
# }
# loss = {
#     "name": "IDACLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "warmup_epochs": 10,
#         "alpha": 1.0,
#         "noise_rate": noise_rate,
#     },
# }

num_classes += 1

lr = 3e-3

model = SegmentationModel(
    num_classes,
    loss,
    lr,
    model_name="UNet",
    include_background=True,
    window_size=16,
)

use_wandb = True

if use_wandb:
    wandb.login()
    wandb.init(project="playground", name=f"dice")
# wandb.log({"noise rate": train_dataset.noise_rate})
# wandb.log({"seed": seed})

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
    log_every_n_steps=len(train_loader) // 3,
    logger=WandbLogger() if use_wandb else None,
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
