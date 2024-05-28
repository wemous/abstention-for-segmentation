from importlib import import_module
from random import seed

import lightning as pl
import torch
from torch.utils.data import DataLoader
from yaml import full_load

from datasets import ADE20K, CaDIS

config = full_load(open("train_config.yaml"))

seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.set_float32_matmul_precision("high")

run_name = config["run"]
dataset_name = config["dataset"]
image_size = config["image_size"]
transforms = config["transforms"]
batch_size = config["batch_size"]

if dataset_name == "ade20k":
    train_dataset = ADE20K(train=True, image_size=image_size)
    valid_dataset = ADE20K(train=False, image_size=image_size)
    num_classes = 151
elif dataset_name == "cadis":
    train_dataset = CaDIS(train=True, image_size=image_size, transforms=transforms)
    valid_dataset = CaDIS(train=False, image_size=image_size)
    num_classes = 8

model_config = config["model"]
module = model_config["module"]
model_name = model_config["name"]
model_args = model_config["args"]
model = getattr(import_module(module), model_name)(num_classes, **model_args)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

trainer = pl.Trainer(
    devices=config["devices"],
    max_epochs=config["max_epochs"],
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    logger=False,
)

# trainer.fit(model, train_loader, valid_loader)
trainer.fit(model, valid_loader)
