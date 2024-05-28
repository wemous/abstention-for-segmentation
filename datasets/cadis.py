from os import listdir
from os.path import join

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    ColorJitter,
    ElasticTransform,
    GaussianBlur,
)
from torchvision.transforms.v2.functional import (
    equalize_image,
    pil_to_tensor,
    resize,
    to_dtype,
    to_image,
    vertical_flip_image,
)

DATA_PATH = "/data/wesam/thesis/CaDIS/"


class CaDIS(Dataset):
    def __init__(
        self, split: str = "train", transform: str = "", image_size=(270, 480)
    ):
        super().__init__()
        assert split in ["train", "valid"]
        self.split = split
        self.transform = transform
        self.image_size = image_size

        videos = {
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
                "Video12",
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
            "valid": [
                "Video02",
                "Video05",
                "Video07",
                "Video16",
                "Video22",
            ],
        }

        self.image_paths, self.mask_paths = [], []
        for video in videos[split]:
            video_path = join(DATA_PATH, video)

            self.image_paths.extend(
                [
                    join(video_path, "Images", file_name)
                    for file_name in sorted(listdir(join(video_path, "Images")))
                ]
            )
            self.mask_paths.extend(
                [
                    join(video_path, "Labels", file_name)
                    for file_name in sorted(listdir(join(video_path, "Labels")))
                ]
            )
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple:
        image = to_dtype(
            to_image(Image.open(self.image_paths[index])),
            dtype=torch.float,
            scale=True,
        )
        mask = pil_to_tensor(Image.open(self.mask_paths[index]))
        mask[mask > 6] = 7

        if self.split == "train":
            if self.transform == "blur":
                image = GaussianBlur(kernel_size=[9, 9], sigma=[1, 20])(image)
            elif self.transform == "elastic":
                image = ElasticTransform(alpha=20, sigma=1)(image)
            elif self.transform == "jitter":
                image = ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.5,
                )(image)
            elif self.transform == "equalize":
                image = equalize_image(image)
            elif self.transform == "flip":
                image = vertical_flip_image(image)
                mask = vertical_flip_image(mask)

        image = resize(image, self.image_size)
        mask = resize(mask, self.image_size).squeeze()

        return image, mask
