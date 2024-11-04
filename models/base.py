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
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        model_name: str = "UNet",
    ):
        super().__init__()
        num_classes = num_classes - 1 if "DAC" in loss["name"] else num_classes
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
        if "DAC" in self.loss_function._get_name():
            loss, ce_loss, regularization, abstention_rate = self.loss_function(
                preds,
                targets,
                training=True,
                epoch=self.current_epoch,
            )
            self.log("CE term", ce_loss, sync_dist=True)
            self.log("Regularization", regularization, sync_dist=True)
            self.log("Abstention rate", abstention_rate, sync_dist=True)
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
        if "DAC" in self.loss_function._get_name():
            preds = preds[:, :-1, :, :]
        preds = softmax(preds, 1).argmax(1).detach()
        dice = multiclass_f1_score(preds, targets, self.num_classes)
        miou = multiclass_jaccard_index(preds, targets, self.num_classes)
        accuracy = (preds == targets).float().mean().item()
        self.log("valid/Accuracy", accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/Dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/mIoU", miou, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        preds = self.forward(images)
        if "DAC" in self.loss_function._get_name():
            preds = preds[:, :-1, :, :]
        preds = softmax(preds, 1).argmax(1).detach()
        dice = multiclass_f1_score(preds, targets, self.num_classes)
        miou = multiclass_jaccard_index(preds, targets, self.num_classes)
        accuracy = (preds == targets).float().mean().item()
        self.log("test/Accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("test/Dice", dice, on_epoch=True, sync_dist=True)
        self.log("test/mIoU", miou, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), **self.optimizer_args, nesterov=True)  # type: ignore
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": MultiplicativeLR(optimizer, lambda _: 1 / 3),
                "interval": "epoch",
                "frequency": self.trainer.max_epochs // 5,  # type: ignore
            },
        }
