import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import UNet, DeepLabV3Plus, FPN, PlainUNet, DeepLabV3

import wandb
from datasets import CaDIS, DSAD, NoisyCaDIS, NoisyDSAD

pl.seed_everything(13)
torch.set_float32_matmul_precision("high")

batch_size = 64

train_dataset = NoisyCaDIS(noise_level=5, setup=1)
valid_dataset = CaDIS(split="valid", setup=1)
test_dataset = CaDIS(split="test", setup=1)
num_classes = test_dataset.num_classes[1]

# train_dataset = NoisyDSAD(noise_level=1)
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


# loss = {
#     "name": "ASCELoss",
#     "args": {
#         "noise_rate": train_dataset.noise_rate,
#         "max_epochs": 30,
#         "warmup_rate": 0.2,
#         "alpha": 0.75,
#         "beta": 0.3,
#         "gamma": 1,
#     },
# }
# loss = {
#     "name": "ASCELoss",
#     "args": {
#         "noise_rate": train_dataset.noise_rate,
#         "max_epochs": 30,
#         "warmup_rate": 0.2,
#         "A": -2,
#         # "alpha": 1,
#         # "beta": 1,
#         # "gamma": 5,
#     },
# }
loss = {
    "name": "DACLoss",
    "args": {
        "max_epochs": 30,
        "warmup_rate": 0.2,
    },
}
# loss = {
#     "name": "SCELoss",
#     "args": {"alpha": 0.5, "beta": 0.75},
# }
optimizer_args = {
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 5e-3,
}


# model = PlainUNet(num_classes, loss, **optimizer_args, bilinear=True)
# model = DeepLabV3(num_classes, loss, **optimizer_args, pretrained=True)
# model = DeepLabV3Plus(num_classes, loss, **optimizer_args)
# model = FPN(num_classes, loss, **optimizer_args)
model = UNet(num_classes + 1, loss, **optimizer_args)

# wandb.login()
# wandb.init(project="thesis")
# wandb.log({"noise rate": train_dataset.noise_rate})

trainer = pl.Trainer(
    # accelerator="cpu",
    max_epochs=30,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    log_every_n_steps=len(train_loader) // 5,
    # logger=WandbLogger(),
    logger=None,
    # strategy="ddp_find_unused_parameters_true",
    # gradient_clip_val=0.5,
)

trainer.fit(model, train_loader, valid_loader)
trainer.test(model, test_loader)

# wandb.finish(quiet=True)
