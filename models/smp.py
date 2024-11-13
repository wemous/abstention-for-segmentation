from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus as deeplab
from segmentation_models_pytorch.decoders.fpn.model import FPN as fpn
from segmentation_models_pytorch.decoders.unet.model import Unet as unet
from torch import Tensor

from models.base import BaseModel


class FPN(BaseModel):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-3,
        **kwargs
    ):
        super().__init__(
            num_classes,
            loss,
            lr,
            momentum,
            weight_decay,
            model_name="FPN",
        )
        self.net = fpn(encoder_name="resnet50", classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore


class DeepLabV3Plus(BaseModel):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-3,
        **kwargs
    ):
        super().__init__(
            num_classes,
            loss,
            lr,
            momentum,
            weight_decay,
            model_name="DeepLabV3+",
        )
        self.net = deeplab(encoder_name="resnet50", classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore


class UNet(BaseModel):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-3,
        **kwargs
    ):
        super().__init__(
            num_classes,
            loss,
            lr,
            momentum,
            weight_decay,
            model_name="UNet",
        )
        self.net = unet(encoder_name="resnet50", classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore
