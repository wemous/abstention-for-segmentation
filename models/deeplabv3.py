from torch import Tensor
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.nn import Conv2d

from models.base import BaseModel


class DeepLabV3(BaseModel):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        optimizer: dict,
        pretrained=True,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            loss=loss,
            optimizer=optimizer,
            model_name="DeepLabV3",
        )
        self.is_pretrained = pretrained
        self.net = deeplabv3_resnet50(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT",
            num_classes=None if pretrained else num_classes,
            **kwargs,
        )
        self.out_conv = Conv2d(21, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)["out"]
        if self.is_pretrained:
            out = self.out_conv(out)
        return out
