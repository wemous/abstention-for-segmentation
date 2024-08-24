from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus as deeplab
from segmentation_models_pytorch.decoders.fpn.model import FPN as fpn
from segmentation_models_pytorch.decoders.unet.model import Unet as unet
from torch import Tensor

from models.base import BaseModel


class FPN(BaseModel):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, **kwargs):
        super().__init__(
            num_classes=num_classes,
            loss=loss,
            optimizer=optimizer,
            model_name="FPN",
        )
        self.net = fpn(classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore


class DeepLabV3Plus(BaseModel):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, **kwargs):
        super().__init__(
            num_classes=num_classes,
            loss=loss,
            optimizer=optimizer,
            model_name="DeepLabV3+",
        )
        self.net = deeplab(classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore


class UNet(BaseModel):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, **kwargs):
        super().__init__(
            num_classes=num_classes,
            loss=loss,
            optimizer=optimizer,
            model_name="UNet",
        )
        self.net = unet(classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore
