from abc import ABC, abstractmethod
from importlib import import_module

from lightning import LightningModule
from torch import Tensor, clamp
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU


class BaseModel(LightningModule, ABC):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.loss_function = getattr(import_module(loss["module"]), loss["name"])(
            **loss["args"]
        ).to(self.device)
        self.optimizer = getattr(import_module(optimizer["module"]), optimizer["name"])
        self.optimizer_args = optimizer["args"]

        self.gcd = GeneralizedDiceScore(num_classes).to(self.device)
        self.miou = MeanIoU(num_classes).to(self.device)

        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor: ...

    def training_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device)
        preds = self.forward(images)
        preds = clamp(preds, min=1e-7)
        loss = self.loss_function(preds, targets)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device)
        preds = self.forward(images)
        preds = clamp(preds, min=1e-7)
        loss = self.loss_function(preds, targets)
        self.log(
            "valid/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        gcd = self.gcd(preds.argmax(1), targets)
        self.log("Generalized Dice Score", gcd, on_epoch=True)
        miou = self.miou(preds.argmax(1), targets)
        self.log("mIoU Score", miou, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_args)
