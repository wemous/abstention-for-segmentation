from abc import ABC, abstractmethod

from lightning import LightningModule
from torch import Tensor, softmax
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim.sgd import SGD
from torchmetrics.functional.classification import (
    multiclass_f1_score,
    multiclass_jaccard_index,
)

import losses


class BaseModel(LightningModule, ABC):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-3,
        model_name: str = "UNet",
    ):
        super().__init__()
        self.is_abstaining = False
        self.loss_name = loss["name"]
        abstaining_losses = ["DACLoss", "IDACLoss", "ASCELoss"]
        if self.loss_name in abstaining_losses:
            self.is_abstaining = True
            num_classes -= 1
        self.num_classes = num_classes
        self.loss_function = getattr(losses, loss["name"])(**loss["args"]).to(self.device)
        self.optimizer_args = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        self.model_name = model_name

        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor: ...

    def training_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        preds = self.forward(images)
        if self.is_abstaining:
            output = self.loss_function(
                preds,
                targets,
                training=True,
                epoch=self.current_epoch,
            )

            loss = output.pop("loss")
            for k, v in output.items():
                self.log(k, v, sync_dist=True)
        else:
            loss = self.loss_function(preds, targets)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        preds = self.forward(images)
        loss = self.loss_function(preds, targets)
        self.log(
            "valid/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        if self.is_abstaining:
            preds = preds[:, :-1, :, :]
        preds = softmax(preds, 1).argmax(1).detach()
        dice = multiclass_f1_score(preds, targets, self.num_classes)
        miou = multiclass_jaccard_index(preds, targets, self.num_classes)
        accuracy = (preds == targets).float().mean()
        self.log("valid/Accuracy", accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/Dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/mIoU", miou, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        preds = self.forward(images)
        if self.is_abstaining:
            preds = preds[:, :-1, :, :]
        preds = softmax(preds, 1).argmax(1).detach()
        dice = multiclass_f1_score(preds, targets, self.num_classes)
        miou = multiclass_jaccard_index(preds, targets, self.num_classes)
        accuracy = (preds == targets).float().mean()
        self.log("test/Accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("test/Dice", dice, on_epoch=True, sync_dist=True)
        self.log("test/mIoU", miou, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), **self.optimizer_args, nesterov=True)  # type: ignore
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": MultiplicativeLR(optimizer, lambda _: 0.2),
                "interval": "epoch",
                "frequency": self.trainer.max_epochs // 4,  # type: ignore
            },
        }
