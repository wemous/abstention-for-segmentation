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
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision("high")

num_classes = 8
max_epochs = 50
lr = 3e-3

train_dataset = NoisyCaDIS(noise_level=3, setup=1)
valid_dataset = CaDIS(split="valid", setup=1)
test_dataset = CaDIS(split="test", setup=1)
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
class_noise = train_dataset.class_noise

# loss_config = {"name": "GCELoss", "args": {"q": 0.3}}
# loss_config = {"name": "SCELoss", "args": {"alpha": 0.5, "beta": 1.0}}
# loss_config = {"name": "CELoss", "args": {}}
# loss_config = {"name": "DiceLoss", "args": {}}

# loss_config = {
#     "name": "ADSLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "noise_rate": noise_rate,
#         "class_noise": class_noise,
#         "alpha_final": 4.0,
#         "gamma": 2.0,
#         "warmup_epochs": 10,
#     },
# }
loss_config = {
    "name": "GACLoss",
    "args": {
        "max_epochs": max_epochs,
        "noise_rate": noise_rate,
        "alpha_final": 3.0,
        "gamma": 4.0,
        "warmup_epochs": 10,
        "q": 0.5,
    },
}
# loss_config = {
#     "name": "SACLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "noise_rate": noise_rate,
#         "alpha_final": 2.0,
#         "gamma": 2.0,
#         "warmup_epochs": 15,
#         "omega": 2.0,
#         "beta": 0.5,
#     },
# }
# loss_config = {
#     "name": "GACLoss",
#     "args": {
#         "max_epochs": max_epochs,
#         "noise_rate": noise_rate,
#         "alpha_final": 4.42974035265131,
#         "gamma": 2.3135905751656525,
#         "warmup_epochs": 6,
#         "q": 0.4866162809498411,
#     },
# }
# loss_config = {
#     "name": "IGACLoss",
#     "args": {"noise_rate": noise_rate, "warmup_epochs": 10, "alpha": 1.0, "q": 0.3},
# }
# loss_config = {
#     "name": "DACLoss",
#     "args": {"max_epochs": max_epochs, "warmup_epochs": 18, "alpha_final": 1.0},
# }
# loss_config = {
#     "name": "IDACLoss",
#     "args": {"noise_rate": noise_rate, "warmup_epochs": 10, "alpha": 1.0},
# }

model = SegmentationModel(
    num_classes,
    loss_config,
    lr=lr,
    model_name="UNet",
    include_background=isinstance(train_dataset, NoisyCaDIS),
)

use_wandb = True

if use_wandb:
    wandb.login()
    wandb.Settings(quiet=True)
    wandb.init(project="playground", name=f"gac")
    wandb.log({"noise rate": noise_rate})
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
    detect_anomaly=True,
    # precision=32,
    # gradient_clip_val=2.0,
    log_every_n_steps=len(train_loader) // 4,
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

wandb.finish()
