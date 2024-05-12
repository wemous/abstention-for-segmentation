from random import seed
import lightning as pl
import torch
from os import cpu_count
from torch.utils.data import DataLoader

from datasets.ade20k import ADE20K
from models.unet import UNet

torch.set_float32_matmul_precision("high")
seed(13)
torch.manual_seed(13)
torch.cuda.manual_seed_all(13)

train_dataset = ADE20K(True)
valid_dataset = ADE20K(False)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False,
    num_workers=16,
    # num_workers=cpu_count() - 1,  # type: ignore
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=16,
    # num_workers=cpu_count() - 1,  # type: ignore
)

model = UNet(num_classes=151)

trainer = pl.Trainer(
    accelerator="gpu",
    # devices=-1,
    strategy="ddp",
    max_epochs=10,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    logger=False,
)

# trainer.fit(model, train_loader, valid_loader)
trainer.fit(model, valid_loader)
