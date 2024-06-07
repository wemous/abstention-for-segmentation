from abc import ABC, abstractmethod
from importlib import import_module

from lightning import LightningModule
from torch import Tensor, softmax
from torch.nn.functional import one_hot
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU


class BaseModel(LightningModule, ABC):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, model_name: str):
        super().__init__()
        num_classes = num_classes - 1 if loss["name"] == "DACLoss" else num_classes
        self.num_classes = num_classes
        self.loss_function = getattr(import_module(loss["module"]), loss["name"])(
            **loss["args"]
        ).to(self.device)
        self.optimizer = getattr(import_module(optimizer["module"]), optimizer["name"])
        self.optimizer_args = optimizer["args"]

        self.gds = GeneralizedDiceScore(num_classes).to(self.device)
        self.miou = MeanIoU(num_classes).to(self.device)
        self.model_name = model_name

        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor: ...

    def training_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device)
        targets = one_hot(targets.long(), self.num_classes).movedim(-1, 1)
        preds = self.forward(images)
        if self.loss_function._get_name() == "DACLoss":
            loss, abstention_rate = self.loss_function(
                preds,
                targets.float(),
                train=True,
                epoch=self.current_epoch,
            )
            self.log("abstention rate", abstention_rate, sync_dist=True)
        else:
            loss = self.loss_function(preds, targets.float())
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
        targets = batch[1].to(self.device)
        targets = one_hot(targets.long(), self.num_classes).movedim(-1, 1)
        preds = self.forward(images)
        loss = self.loss_function(preds, targets.float())
        self.log(
            "valid/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        if self.loss_function._get_name() in ["DACLoss", "IDACLoss"]:
            preds = preds[:, :-1, :, :]
        preds = softmax(preds, 1).argmax(1).detach()
        preds = one_hot(preds, self.num_classes).movedim(-1, 1)
        gds = self.gds(preds, targets)
        self.log("valid/GDS", gds, on_epoch=True, sync_dist=True)
        miou = self.miou(preds, targets)
        self.log("valid/mIoU", miou, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device)
        targets = one_hot(targets.long(), self.num_classes).movedim(-1, 1)
        preds = self.forward(images)
        if self.loss_function._get_name() in ["DACLoss", "IDACLoss"]:
            preds = preds[:, :-1, :, :]
        preds = softmax(preds, 1).argmax(1).detach()
        preds = one_hot(preds, self.num_classes).movedim(-1, 1)
        gds = self.gds(preds, targets)
        self.log("test/GDS", gds, sync_dist=True)
        miou = self.miou(preds, targets)
        self.log("test/mIoU", miou, sync_dist=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_args)
