import lightning as pl
from mmseg.models.backbones.unet import InterpConv
from mmseg.models.backbones.unet import UNet as U
from torch import Tensor
from torch.nn import Conv2d, CrossEntropyLoss
from torch.optim.adam import Adam
from torchvision.transforms.v2.functional import resize


class UNet(pl.LightningModule):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        upsample_cfg = {"type": InterpConv}
        self.unet = U(upsample_cfg=upsample_cfg, **kwargs)
        self.out_conv = Conv2d(self.unet.base_channels, num_classes, kernel_size=1)
        self.metric = CrossEntropyLoss().to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        downsampling_factor = 2 ** (self.unet.num_stages - 1)
        if h % downsampling_factor == 0 and w % downsampling_factor == 0:
            out = self.unet(x)[-1]
            out = self.out_conv(out)
        else:
            new_size = [
                round(h / downsampling_factor) * downsampling_factor,
                round(w / downsampling_factor) * downsampling_factor,
            ]
            x = resize(x, new_size, antialias=True)
            out = self.unet(x)[-1]
            out = self.out_conv(out)
            out = resize(out, [h, w], antialias=True)
        return out

    def training_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device)
        preds = self.forward(images)
        loss = self.metric(preds, targets)
        self.log("training loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        images = batch[0].to(self.device)
        targets = batch[1].to(self.device)
        preds = self.forward(images)
        loss = self.metric(preds, targets)
        self.log("validation loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4)
