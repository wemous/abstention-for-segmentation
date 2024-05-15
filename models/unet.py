from mmseg.models.backbones.unet import InterpConv
from mmseg.models.backbones.unet import UNet as U
from torch import Tensor
from torch.nn import Conv2d
from torchvision.transforms.v2.functional import resize

from models.base import BaseModel


class UNet(BaseModel):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, **kwargs):
        super().__init__(num_classes, loss, optimizer, **kwargs)
        upsample_cfg = {"type": InterpConv}
        self.unet = U(upsample_cfg=upsample_cfg, **kwargs)
        self.out_conv = Conv2d(self.unet.base_channels, num_classes, kernel_size=1)

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
