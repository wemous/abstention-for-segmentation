import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from datasets import DSAD, CaDIS, NoisyCaDIS, NoisyDSAD
from models import SegmentationModel

seed = 1
torch.use_deterministic_algorithms(True, warn_only=True)
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision("high")

max_epochs = 50
batch_size = 128

train_dataset = NoisyCaDIS(noise_level=5, setup=1)
valid_dataset = CaDIS(split="valid", setup=1)
test_dataset = CaDIS(split="test", setup=1)
num_classes = 8

# train_dataset = NoisyDSAD(noise_level=5, normalized=True, rotated=True)
# valid_dataset = DSAD(split="valid")
# test_dataset = DSAD(split="test")
# num_classes = 8


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
# loss = {
#     "name": "DiceLoss",
#     "args": {},
# }
loss = {
    "name": "ADLoss",
    "args": {
        "max_epochs": max_epochs,
        "warmup_epochs": 20,
        "alpha_final": 2.5,
        "gamma": 1.75,
    },
}
# loss = {
#     "name": "DACLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "warmup_epochs": 5,
#         "alpha_final": 2.0,
#     },
# }
# loss = {
#     "name": "IDACLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "warmup_epochs": 15,
#         "noise_rate": noise_Rate,
#     },
# }
optimizer_args = {
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-3,
}

model = SegmentationModel(
    num_classes,
    loss,
    model_name="UNet",
    window_size=16,
    include_background=True,
    **optimizer_args,
)

use_wandb = True

if use_wandb:
    wandb.login()
    wandb.init(project="xdac", name="adl")
# wandb.log({"noise rate": train_dataset.noise_rate})
wandb.log({"seed": seed})

trainer = pl.Trainer(
    devices=1,
    max_epochs=max_epochs,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    log_every_n_steps=10,
    logger=WandbLogger() if use_wandb else None,
)

trainer.fit(model, train_loader, valid_loader)
trainer.test(model, test_loader)

wandb.finish(quiet=True)
