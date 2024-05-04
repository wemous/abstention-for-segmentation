from os import listdir
from os.path import join
import torch
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype, to_image
from PIL import Image
from torch.utils.data import Dataset

DATA_PATH = "/data/wesam/thesis/ADE20K/"


class ADE20K(Dataset):
    def __init__(self, train=True):
        super().__init__()
        split = "training" if train else "validation"

        self.image_paths = [
            join(DATA_PATH, "images", split, _)
            for _ in sorted(listdir(join(DATA_PATH, "images", split)))
        ]
        self.mask_paths = [
            join(DATA_PATH, "annotations", split, _)
            for _ in sorted(listdir(join(DATA_PATH, "annotations", split)))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple:
        image = to_dtype(
            to_image(Image.open(self.image_paths[index])), torch.float, scale=True
        )
        mask = pil_to_tensor(Image.open(self.mask_paths[index])).squeeze()

        return image, mask
