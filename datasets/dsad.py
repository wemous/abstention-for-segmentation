import random
from os import makedirs
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm
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

ROOT = Path("/data/wesam/datasets/DSAD/")
multilabel = ROOT.joinpath("multilabel")

video_splits = {
    "train": [
        "02",
        "03",
        "04",
        "05",
        "07",
        "10",
        "11",
        "12",
        "20",
        "21",
        "22",
        "23",
        "25",
        "27",
        "29",
    ],
    "valid": ["16", "30", "31"],
    "test": ["08", "17", "18", "24", "26"],
}

mean = [0.4944, 0.2892, 0.2332]
std = [0.2423, 0.1864, 0.1665]

tf_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)


def to_tensor(path) -> Tensor:
    return to_dtype(
        pil_to_tensor(Image.open(path)),
        dtype=torch.float,
        scale=True,
    )


def build_image_and_mask(parent_path, index) -> tuple[Tensor, Tensor]:
    image_path = Path(parent_path).joinpath(f"image{index}.png")
    image = to_tensor(image_path)
    image = resize(image, [384, 480])

    mask_paths = [
        p for p in sorted(Path(parent_path).iterdir()) if f"mask{index}" in p.name
    ]
    organ_masks = torch.stack([to_tensor(p) for p in mask_paths])
    organ_masks = resize(organ_masks, [384, 480])
    background = 1 - organ_masks.max(0)[0].unsqueeze(0)
    mask = torch.concat([background, organ_masks]).argmax(0)

    return image, mask


def transform(image: Tensor, mask: Tensor, tf: str) -> tuple[Tensor, Tensor]:
    if tf == "normalized":
        image = normalize(image, mean, std)
    elif tf == "rotated":
        angle = random.uniform(-60, 60)
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    elif tf == "flipped":
        image = vertical_flip(image)
        mask = vertical_flip(mask)
    elif tf == "jitter":
        image = tf_jitter(image)
    elif tf == "noise":
        image = gaussian_noise(image)

    image = normalize(image, mean, std)
    return image, mask


def build_dataset(split: str, tf: str = ""):
    image_paths, mask_paths = [], []
    split_path = ROOT.joinpath(f"transformed/{tf if tf else split}")
    if not split_path.exists():
        print(f"Building {tf if tf else split} images and masks")
        length = 142 if split == "valid" else 288 if split == "test" else 1000
        p_bar = tqdm(total=length, desc="images")
        for v in video_splits[split]:
            source = multilabel.joinpath(v)
            destination = split_path.joinpath(v)
            makedirs(destination)
            images = [f for f in sorted(source.iterdir()) if "image" in f.name]
            for image_path in images:
                index = image_path.name[5:7]
                image, mask = build_image_and_mask(source, index)
                if tf:
                    image, mask = transform(image, mask, tf)
                torch.save(image, destination.joinpath(f"image{index}.pt"))
                torch.save(mask, destination.joinpath(f"mask{index}.pt"))
                p_bar.update()
        p_bar.close()

    for v in video_splits[split]:
        video_path = split_path.joinpath(v)
        image_paths.extend([f for f in sorted(video_path.iterdir()) if "image" in f.name])
        mask_paths.extend([f for f in sorted(video_path.iterdir()) if "mask" in f.name])
    return image_paths, mask_paths


class DSAD(Dataset):
    def __init__(
        self,
        split: str,
        normalized=False,
        rotated=False,
        flipped=False,
        jitter=False,
        noise=False,
    ):
        super().__init__()
        self.image_paths, self.mask_paths = [], []
        transforms = {
            "normalized": normalized,
            "rotated": rotated,
            "flipped": flipped,
            "jitter": jitter,
            "noise": noise,
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
            self.image_paths, self.mask_path = build_dataset(split)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        return image, mask
