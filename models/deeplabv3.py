from torch import Tensor
from torchvision.models.segmentation import deeplabv3_resnet50

from models.base import BaseModel


class DeepLabV3(BaseModel):
    def __init__(self, num_classes: int, loss: dict, optimizer: dict, **kwargs):
        super().__init__(num_classes, loss, optimizer, **kwargs)
        self.net = deeplabv3_resnet50(weights_backbone="DEFAULT", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return out["out"]
