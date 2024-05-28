from collections.abc import Iterable
from os import listdir, makedirs
from os.path import exists, join

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    ColorJitter,
    ElasticTransform,
    GaussianBlur,
    Identity,
)
from torchvision.transforms.v2.functional import (
    equalize_image,
    pil_to_tensor,
    resize,
    to_dtype,
    to_image,
    vertical_flip_image,
)
from tqdm import tqdm

DATA_PATH = "/data/wesam/datasets/CaDIS/"

TRANSFORMS = {
    "original": Identity(),
    "blur": GaussianBlur(kernel_size=[9, 9], sigma=[1, 20]),
    "elastic": ElasticTransform(alpha=20, sigma=1),
    "jitter": ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5),
    "equalize": equalize_image,
    "flip": vertical_flip_image,
    "validation": Identity(),
    "testing": Identity(),
}

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
        split: int,
        setup: int,
        transforms: Iterable = {},
        image_size=(270, 480),
    ):
        super().__init__()
        assert split >= 0 and split <= 2
        assert setup >= 1 and split <= 3
        assert set(TRANSFORMS.keys()).issuperset(transforms)
        self.setup = setup
        self.num_classes = {1: 8, 2: 18, 3: 26}
        split_ids = {0: "training", 1: "validation", 2: "testing"}
        # add resize transform if passed an empty set
        if not transforms:
            if split == 0:
                transforms = {"original"}
            elif split == 1:
                transforms = {"validation"}
            if split == 2:
                transforms = {"testing"}
        video_split = {
            "training": [
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
            "validation": [
                "Video05",
                "Video07",
                "Video16",
            ],
            "testing": [
                "Video02",
                "Video12",
                "Video22",
            ],
        }
        videos = video_split[split_ids[split]]
        trasnformed_path = join(DATA_PATH, f"transformed_{image_size[0]}_{image_size[1]}")

        for tf in transforms:
            tf_folder = join(trasnformed_path, tf)

            if not exists(tf_folder):
                makedirs(tf_folder)

                for v in tqdm(videos, desc=f"{tf} transform"):
                    image_source = join(DATA_PATH, v, "Images")
                    image_target = join(tf_folder, v, "Images")
                    makedirs(image_target)
                    image_names = sorted(listdir(image_source))

                    for n in image_names:
                        image = to_dtype(
                            to_image(Image.open(join(image_source, n))),
                            torch.float,
                            scale=True,
                        )
                        image = resize(image, image_size, antialias=True)  # type: ignore
                        image = TRANSFORMS[tf](image)
                        torch.save(image, join(image_target, n[:-4] + ".pt"))

                    mask_source = join(DATA_PATH, v, "Labels")
                    mask_target = join(tf_folder, v, "Labels")
                    makedirs(mask_target)
                    mask_names = sorted(listdir(mask_source))

                    for n in mask_names:
                        mask = pil_to_tensor(Image.open(join(mask_source, n)))
                        mask = resize(mask, image_size, antialias=True).squeeze().long()  # type: ignore
                        if tf == "flip":
                            mask = TRANSFORMS["flip"](mask)
                        torch.save(mask, join(mask_target, n[:-4] + ".pt"))

        self.image_paths, self.mask_paths = [], []

        for tf in transforms:
            for v in videos:
                video_path = join(trasnformed_path, tf, v)
                self.image_paths.extend(
                    [
                        join(video_path, "Images", f)
                        for f in sorted(listdir(join(video_path, "Images")))
                    ]
                )
                self.mask_paths.extend(
                    [
                        join(video_path, "Labels", f)
                        for f in sorted(listdir(join(video_path, "Labels")))
                    ]
                )
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index])
        mask = torch.load(self.mask_paths[index])
        mask = mask_to_setup(mask, self.setup)
        return image, mask
