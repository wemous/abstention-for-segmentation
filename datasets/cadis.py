from os import makedirs
from pathlib import Path
import random

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ColorJitter
from torchvision.transforms.v2.functional import (
    gaussian_noise,
    normalize,
    pil_to_tensor,
    resize,
    rotate,
    to_dtype,
    vertical_flip,
)
from tqdm import tqdm

root_path = Path("/data/wesam/datasets/CaDIS/")

video_splits = {
    "train": [
        "Video01",
        "Video03",
        "Video04",
        "Video06",
        "Video08",
        "Video09",
        "Video10",
        "Video11",
        "Video13",
        "Video14",
        "Video15",
        "Video17",
        "Video18",
        "Video19",
        "Video20",
        "Video21",
        "Video23",
        "Video24",
        "Video25",
    ],
    "valid": ["Video05", "Video07", "Video16"],
    "test": ["Video02", "Video12", "Video22"],
}

mean = [0.5737, 0.3461, 0.1954]
std = [0.1593, 0.1558, 0.1049]

tf_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3).cuda()

setup_2_class_map = {
    7: [8, 10, 20, 27, 32],
    8: [9, 22],
    9: [11, 33],
    10: [12, 28],
    11: [13, 21],
    12: [14, 24],
    13: [15, 18],
    14: [16, 23],
    15: [17],
    16: [19],
    17: [25, 26, 29, 30, 31, 34, 35],
}


def mask_to_setup(mask: Tensor, setup: int) -> Tensor:
    if setup == 1:
        mask[mask > 7] = 7
    if setup == 2:
        for k, values in setup_2_class_map.items():
            for v in values:
                mask[mask == v] = k
    if setup == 3:
        mask[mask > 25] = 25
    return mask


def to_tensor(path, dtype=torch.float, scale=False) -> Tensor:
    return to_dtype(pil_to_tensor(Image.open(path)), dtype=dtype, scale=scale)


def transform(image: Tensor, mask: Tensor, tf: str) -> tuple[Tensor, Tensor]:
    if tf == "flipped":
        image = vertical_flip(image)
        mask = vertical_flip(mask)
    elif tf == "gaussian":
        image = gaussian_noise(image)
    elif tf == "jitter":
        image = tf_jitter(image)
    elif tf == "normalized":
        image = normalize(image, mean, std)
    elif tf == "rotated":
        angle = random.uniform(-60, 60)
        image = rotate(image, angle, fill=35)  
        mask = rotate(mask, angle, fill=35)
    return image, mask


def build_dataset(split: str, tf: str = ""):
    image_paths, mask_paths = [], []
    split_path = root_path.joinpath(f"transformed/{tf if tf else split}")
    if not split_path.exists():
        print(f"Building {tf if tf else split} images and masks")
        length = 534 if split == "valid" else 586 if split == "test" else 3550
        p_bar = tqdm(total=length, desc="images")
        for v in video_splits[split]:
            source = root_path.joinpath(v)
            destination = split_path.joinpath(v)
            makedirs(destination.joinpath("Images"))
            makedirs(destination.joinpath("Labels"))
            images = sorted([*source.joinpath("Images").iterdir()])
            masks = sorted([*source.joinpath("Labels").iterdir()])
            for i_path, m_path in zip(images, masks):
                image = to_tensor(i_path, scale=True)
                image = resize(image, [256, 480])
                mask = to_tensor(m_path, dtype=torch.long)
                mask = resize(mask, [256, 480])
                if tf:
                    image, mask = transform(image.cuda(), mask.cuda(), tf)
                torch.save(image.cpu(), destination.joinpath(f"Images/{i_path.stem}.pt"))
                torch.save(mask.cpu(), destination.joinpath(f"Labels/{m_path.stem}.pt"))
                p_bar.update()
        p_bar.close()

    for v in video_splits[split]:
        video_path = split_path.joinpath(v)
        image_paths.extend(sorted([*video_path.joinpath("Images").iterdir()]))
        mask_paths.extend(sorted([*video_path.joinpath("Labels").iterdir()]))
    return image_paths, mask_paths


class CaDIS(Dataset):
    def __init__(
        self,
        split: str = "train",
        setup: int = 1,
        flipped=False,
        gaussian=False,
        jitter=False,
        normalized=False,
        rotated=False,
    ):
        super().__init__()
        assert setup >= 1 and setup <= 3
        self.setup = setup
        self.num_classes = {1: 8, 2: 18, 3: 26}
        self.image_paths, self.mask_paths = [], []
        transforms = {
            "flipped": flipped,
            "gaussian": gaussian,
            "jitter": jitter,
            "normalized": normalized,
            "rotated": rotated,
        }

        if split == "train":
            images, masks = build_dataset(split="train")
            self.image_paths.extend(images)
            self.mask_paths.extend(masks)
            for k, v in transforms.items():
                if v:
                    images, masks = build_dataset(split="train", tf=k)
                    self.image_paths.extend(images)
                    self.mask_paths.extend(masks)
        else:
            self.image_paths, self.mask_paths = build_dataset(split)
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        mask = mask_to_setup(mask, self.setup)
        return image, mask
