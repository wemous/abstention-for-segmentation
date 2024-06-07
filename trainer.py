from importlib import import_module

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from yaml import full_load

import wandb
from datasets import ADE20K, CaDIS

config = full_load(open("configs/train_config.yaml"))
pl.seed_everything(config["seed"])
torch.set_float32_matmul_precision("high")

dataset_name = config["dataset"]
image_size = config["image_size"]
transforms = config["transforms"]
batch_size = config["batch_size"]
setup = config["setup"]

if dataset_name == "ade20k":
    train_dataset = ADE20K(split=0, setup=setup, image_size=image_size)
    valid_dataset = ADE20K(split=1, setup=setup, image_size=image_size)
    test_dataset = ADE20K(split=2, setup=setup, image_size=image_size)
elif dataset_name == "cadis":
    train_dataset = CaDIS(
        split=0,
        setup=setup,
        image_size=image_size,
        transforms=transforms,
    )
    valid_dataset = CaDIS(split=1, setup=setup, image_size=image_size)
    test_dataset = CaDIS(split=2, setup=setup, image_size=image_size)

num_classes = train_dataset.num_classes[setup]

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

model_config = config["model"]
module = model_config["module"]
model_name = model_config["name"]
model_args = model_config["args"]
loss_config = model_args["loss"]
optimizer_name = model_args["optimizer"]["name"]

if loss_config["name"] == "DACLoss":
    loss_config["args"]["max_epochs"] = config["max_epochs"]
    num_classes += 1
loss_config["args"]["num_classes"] = num_classes

model_args["loss"] = loss_config
model = getattr(import_module(module), model_name)(num_classes, **model_args)

run_name = f"{model_name.lower()}/{dataset_name}/{loss_config['name'][:-4].lower()}/{optimizer_name.lower()}"

wandb.login()

logger = WandbLogger(name=run_name, project="DAC Segmentation")
trainer = pl.Trainer(
    devices=config["devices"],
    max_epochs=config["max_epochs"],
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    log_every_n_steps=len(train_loader) // 5,
    logger=logger,
)

trainer.fit(model, train_loader, valid_loader)
trainer.test(model, test_loader)

wandb.finish(quiet=True)
