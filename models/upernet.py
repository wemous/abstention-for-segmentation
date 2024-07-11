from torch import Tensor
from torchvision.transforms.v2.functional import resize
from transformers.models.upernet import UperNetConfig, UperNetForSemanticSegmentation

from models.base import BaseModel


class UPerNet(BaseModel):
    def __init__(
        self,
        num_classes: int,
        loss: dict,
        optimizer: dict,
        backbone: str = "microsoft/resnet-50",
        use_pretrained_backbone: bool = True,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            loss=loss,
            optimizer=optimizer,
            model_name="UPerNet",
            **kwargs,
        )
        self.net = UperNetForSemanticSegmentation(
            UperNetConfig(
                backbone=backbone,
                use_pretrained_backbone=use_pretrained_backbone,
                use_auxiliary_head=False,
                num_labels=num_classes,
                **kwargs,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        h_in, w_in = x.shape[-2:]
        out = self.net(x)["logits"]
        h_out, w_out = out.shape[-2:]
        if h_out == h_in and w_out == w_in:
            return out
        else:
            return resize(out, [h_in, w_in], antialias=True)
