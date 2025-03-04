from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus as deeplab
from segmentation_models_pytorch.decoders.fpn.model import FPN as fpn
from segmentation_models_pytorch.decoders.unet.model import Unet as unet
from torch import Tensor, nn


class FPN(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = fpn(encoder_name="resnet50", classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = deeplab(encoder_name="resnet50", classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = unet(encoder_name="resnet50", classes=num_classes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
