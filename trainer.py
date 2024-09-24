import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import UNet, DeepLabV3Plus, FPN, PlainUNet, DeepLabV3

import wandb
from datasets import CaDIS

pl.seed_everything(13)
torch.set_float32_matmul_precision("high")

batch_size = 64

train_dataset = CaDIS(split="valid", setup=1)
augmented_dataset = CaDIS(
    split="train", setup=1, normalized=True, jitter=True, gaussian=True, flipped=True
)
valid_dataset = CaDIS(split="valid", setup=1)
test_dataset = CaDIS(split="test", setup=1)

num_classes = train_dataset.num_classes[1]


train_loader = DataLoader(
    augmented_dataset,
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


loss = {"module": "losses", "name": "CELoss", "args": {}}
# optimizer = {"module": "torch.optim", "name": "AdamW", "args": {"lr": 1e-4}}
optimizer = {
    "module": "torch.optim",
    "name": "SGD",
    "args": {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "nesterov": "True",
    },
}


# model = PlainUNet(num_classes, loss, optimizer, bilinear=True)
# model = DeepLabV3(num_classes, loss, optimizer)
# model = DeepLabV3Plus(num_classes, loss, optimizer)
# model = FPN(num_classes, loss, optimizer)
model = UNet(num_classes, loss, optimizer)

# wandb.login()
# logger = WandbLogger(project="DAC Segmentation")
trainer = pl.Trainer(
    max_epochs=20,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    log_every_n_steps=len(train_loader) // 5,
    logger=None,
    strategy="ddp_find_unused_parameters_true",
    # gradient_clip_val=0.5,
)

trainer.fit(model, train_loader, valid_loader)
trainer.test(model, test_loader)

wandb.finish(quiet=True)
