import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim.adamw import AdamW
from torchmetrics.functional.segmentation import dice_score, mean_iou

import losses
import models


class SegmentationModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        lr=0.003,
        model_name: str = "UNet",
        window_size=16,
        include_background=True,
    ):
        super().__init__()
        self.model = getattr(models, model_name)(num_classes).to(self.device)
        self.loss_name = loss["name"]

        self.num_classes = num_classes
        if "DAC" in self.loss_name:
            self.num_classes -= 1

        if "ADS" in self.loss_name:
            self.segmentation_head = self.model.net.segmentation_head
            in_features = self.segmentation_head[0].in_channels * window_size**2
            self.model.net.segmentation_head = nn.Identity()
            self.abstention_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(window_size),
                nn.Flatten(1),
                nn.Linear(in_features, num_classes),
            )

        self.loss_function = getattr(losses, loss["name"])(**loss["args"]).to(self.device)
        self.lr = lr
        self.include_background = include_background
        self.save_hyperparameters()

    def forward(self, x: Tensor, *args, **kwargs):
        output = self.model(x)
        if "ADS" in self.loss_name:
            preds = self.segmentation_head(output)
            if self.training:
                abstention = self.abstention_head(output)
                return preds, abstention
            else:
                return preds
        else:
            return output

    def training_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        if "ADS" in self.loss_name:
            preds, abstention = self.forward(images)
            output = self.loss_function(preds, targets, abstention, epoch=self.current_epoch)
            loss = output.pop("loss")
            for k, v in output.items():
                self.log(k, v)
        else:
            preds = self.forward(images)
            if "DAC" in self.loss_name:
                output = self.loss_function(preds, targets, training=True, epoch=self.current_epoch)

                loss = output.pop("loss")
                for k, v in output.items():
                    self.log(k, v)
            else:
                loss = self.loss_function(preds, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        preds = self.forward(images)
        loss = self.loss_function(preds, targets)
        self.log("valid/loss", loss, prog_bar=True, on_epoch=True)
        if "DAC" in self.loss_name:
            preds = preds[:, :-1, :, :]

        preds = preds.argmax(1).detach()
        accuracy = (preds == targets).float().mean()
        preds = F.one_hot(preds, self.num_classes).movedim(-1, 1)
        targets = F.one_hot(targets, self.num_classes).movedim(-1, 1)
        dice = dice_score(
            preds,
            targets,
            num_classes=self.num_classes,
            include_background=self.include_background,
            average="none",
        ).mean()
        miou = mean_iou(
            preds,
            targets,
            num_classes=self.num_classes,
            include_background=self.include_background,
        ).mean()
        self.log("valid/accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("valid/dice", dice, on_epoch=True, prog_bar=True)
        self.log("valid/miou", miou, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device).squeeze().long()
        preds = self.forward(images)
        if "DAC" in self.loss_name:
            preds = preds[:, :-1, :, :]

        preds = preds.argmax(1).detach()
        accuracy = (preds == targets).float().mean().cpu().item()
        preds = F.one_hot(preds, self.num_classes).movedim(-1, 1)
        targets = F.one_hot(targets, self.num_classes).movedim(-1, 1)
        dice = dice_score(
            preds,
            targets,
            num_classes=self.num_classes,
            include_background=self.include_background,
            average="none",
        )
        dice = dice.mean().cpu().item()
        miou = mean_iou(
            preds,
            targets,
            num_classes=self.num_classes,
            include_background=self.include_background,
        )
        miou = miou.mean().cpu().item()
        self.log("test/accuracy", accuracy, on_epoch=True)
        self.log("test/dice", dice, on_epoch=True)
        self.log("test/miou", miou, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": MultiplicativeLR(optimizer, lambda _: 1 / 5),
                "interval": "epoch",
                "frequency": 10,
            },
        }
