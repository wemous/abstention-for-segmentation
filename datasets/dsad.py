from os import listdir, path
from os.path import join

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import (
    resize,
    to_dtype,
    to_image,
)

DATA_PATH = "/data/wesam/datasets/DSAD/"

organs = [
    "abdominal_wall",
    "colon",
    "liver",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach",
]

organs_id_map = {
    "abdominal_wall": 0,
    "colon": 1,
    "liver": 2,
    "pancreas": 3,
    "small_intestine": 4,
    "spleen": 5,
    "stomach": 6,
}


class DSAD_Organ(Dataset):
    def __init__(self, organ: str, image_size: "tuple[int, int]" = (256, 320)):
        super().__init__()
        self.image_size = image_size
        organ_path = join(DATA_PATH, organ)
        assert path.exists(organ_path)

        self.image_paths, self.mask_paths = [], []
        for video in sorted(listdir(organ_path)):
            video_path = join(organ_path, video)
            self.image_paths.extend(
                [
                    join(video_path, file_name)
                    for file_name in sorted(listdir(video_path))
                    if file_name.startswith("image")
                ]
            )
            self.mask_paths.extend(
                [
                    join(video_path, file_name)
                    for file_name in sorted(listdir(video_path))
                    if file_name.startswith("mask")
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = to_dtype(
            to_image(Image.open(self.image_paths[index])),
            dtype=torch.float,
            scale=True,
        )
        mask = to_dtype(
            to_image(Image.open(self.mask_paths[index])),
            dtype=torch.long,
            scale=False,
        )

        image = resize(image, self.image_size)  # type: ignore
        mask = resize(mask, self.image_size)  # type: ignore

        return image, mask


class DSAD_Multi_train(Dataset):
    def __init__(self, image_size: "tuple[int, int]" = (256, 320)):
        super().__init__()
        self.image_size = image_size
        self.image_paths, self.mask_paths, self.labels = [], [], []

        for organ in organs:
            organ_path = join(DATA_PATH, organ)
            for video in sorted(listdir(organ_path)):
                video_path = join(organ_path, video)
                images = [
                    join(video_path, file_name)
                    for file_name in sorted(listdir(video_path))
                    if file_name.startswith("image")
                ]
                self.image_paths.extend(images)

                masks = [
                    join(video_path, file_name)
                    for file_name in sorted(listdir(video_path))
                    if file_name.startswith("mask")
                ]
                self.mask_paths.extend(masks)

                self.labels.extend([organs_id_map[organ]] * len(images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor, str]":
        image = to_dtype(
            to_image(Image.open(self.image_paths[index])),
            dtype=torch.float,
            scale=True,
        )
        mask = to_dtype(
            to_image(Image.open(self.mask_paths[index])),
            dtype=torch.long,
            scale=False,
        )
        label = self.labels[index]

        image = resize(image, self.image_size)  # type: ignore
        mask = resize(mask, self.image_size)  # type: ignore

        return image, mask, label


class DSAD_Multi_valid(Dataset):
    def __init__(self, image_size: "tuple[int, int]" = (256, 320)):
        super().__init__()
        self.image_size = image_size
        organ_path = join(DATA_PATH, "multilabel")
        self.image_paths, self.mask_paths = [], []

        for video in sorted(listdir(organ_path)):
            video_path = join(organ_path, video)
            self.image_paths.extend(
                [
                    join(video_path, file_name)
                    for file_name in sorted(listdir(video_path))
                    if file_name.startswith("image")
                ]
            )
            self.mask_paths.extend(
                [
                    join(video_path, file_name)
                    for file_name in sorted(listdir(video_path))
                    if file_name.startswith("mask")
                ]
            )
        assert len(self.mask_paths) == len(self.image_paths) * 7

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = to_dtype(
            to_image(Image.open(self.image_paths[index])),
            dtype=torch.float,
            scale=True,
        )
        channels = self.mask_paths[index * 7 : 7 + index * 7]
        mask = torch.concat(
            [
                to_dtype(
                    to_image(Image.open(c)),
                    dtype=torch.long,
                    scale=False,
                )
                for c in channels
            ]
        )

        image = resize(image, self.image_size)  # type: ignore
        mask = resize(mask, self.image_size)  # type: ignore

        return image, mask
