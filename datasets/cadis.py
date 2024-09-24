from os import makedirs
from pathlib import Path

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


def to_tensor(path, scale=False) -> Tensor:
    return to_dtype(
        pil_to_tensor(Image.open(path)),
        dtype=torch.float,
        scale=scale,
    )


def transform(x: Tensor, tf: str) -> Tensor:
    if tf == "normalized":
        x = normalize(x, mean, std)
    elif tf == "flipped":
        x = vertical_flip(x)
    elif tf == "jitter":
        x = tf_jitter(x)
    elif tf == "noise":
        x = gaussian_noise(x)

    return x


def build_dataset(split: str, image_size: tuple, tf: str = ""):
    image_paths, label_paths = [], []
    split_path = root_path.joinpath(
        f"transformed_{image_size[0]}_{image_size[1]}/{tf if tf else split}"
    )
    if not split_path.exists():
        print(f"Building {tf if tf else split} images and masks")
        # length = 534 if split == "valid" else 586 if split == "test" else 3550
        for v in tqdm(video_splits[split], desc="videos"):
            source = root_path.joinpath(v)
            destination = split_path.joinpath(v)

            makedirs(destination.joinpath("Images"))
            images = [f for f in source.joinpath("Images").iterdir()]
            for image_path in images:
                image = to_tensor(image_path, scale=True)
                image = resize(image, list(image_size))
                if tf:
                    image = transform(image.cuda(), tf)
                torch.save(
                    image.cpu(), destination.joinpath(f"Images/{image_path.stem}.pt")
                )

            makedirs(destination.joinpath("Labels"))
            labels = [f for f in source.joinpath("Labels").iterdir()]
            for label_path in labels:
                label = to_tensor(label_path, scale=False)
                label = resize(label, list(image_size))
                if tf == "flipped":
                    image = transform(image.cuda(), "flipped")
                torch.save(
                    label.cpu(), destination.joinpath(f"Labels/{label_path.stem}.pt")
                )

    for v in video_splits[split]:
        video_path = split_path.joinpath(v)
        image_paths.extend([f for f in sorted(video_path.joinpath("Images").iterdir())])
        label_paths.extend([f for f in sorted(video_path.joinpath("Labels").iterdir())])
    return image_paths, label_paths


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


class CaDIS(Dataset):
    def __init__(
        self,
        split: str,
        setup: int = 1,
        image_size=(256, 480),
        normalized=False,
        flipped=False,
        jitter=False,
        gaussian=False,
    ):
        super().__init__()
        assert setup >= 1 and setup <= 3
        self.setup = setup
        self.num_classes = {1: 8, 2: 18, 3: 26}
        self.image_paths, self.label_paths = [], []
        transforms = {
            "normalized": normalized,
            "flipped": flipped,
            "jitter": jitter,
            "gaussian": gaussian,
        }

        if split == "train":
            images, labels = build_dataset(split="train", image_size=image_size)
            self.image_paths.extend(images)
            self.label_paths.extend(labels)
            for k, v in transforms.items():
                if v:
                    images, labels = build_dataset(
                        split="train", image_size=image_size, tf=k
                    )
                    self.image_paths.extend(images)
                    self.label_paths.extend(labels)
        else:
            self.image_paths, self.label_paths = build_dataset(split, image_size)
        assert len(self.image_paths) == len(self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.label_paths[index], weights_only=True)
        mask = mask_to_setup(mask, self.setup)
        return image, mask
