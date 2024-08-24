from collections.abc import Iterable
from os import listdir, makedirs
from os.path import exists, join
import random

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
    vertical_flip,
    normalize,
    rotate,
)
from tqdm import tqdm
from .utils import make_noise, mask_to_setup_cadis

ROOT = "/data/wesam/datasets/CaDIS/"

TRANSFORMS = {
    "original": Identity(),
    "blur": GaussianBlur(kernel_size=[9, 9], sigma=[1, 20]),
    "elastic": ElasticTransform(alpha=20, sigma=1),
    "jitter": ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5),
    "equalize": equalize_image,
    "flip": vertical_flip,
    "validation": Identity(),
    "testing": Identity(),
}

# tf_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.5)
tf_jitter = ColorJitter(saturation=0.3, hue=0.5)


class CaDIS(Dataset):
    def __init__(
        self,
        split: int,
        setup: int,
        transforms: Iterable = {},
        image_size=(256, 480),
        noise_rate=0.0,
        noise_type=None,
        normalize=False,
        jitter=False,
        rotate=False,
        flip=False,
    ):
        super().__init__()
        assert split >= 0 and split <= 2
        assert setup >= 1 and split <= 3
        assert set(TRANSFORMS.keys()).issuperset(transforms)
        self.setup = setup
        self.noise_rate = noise_rate
        self.noise_type = noise_type
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
        trasnformed_path = join(ROOT, f"transformed_{image_size[0]}_{image_size[1]}")

        for tf in transforms:
            tf_folder = join(trasnformed_path, tf)

            if not exists(tf_folder):
                makedirs(tf_folder)

                for v in tqdm(videos, desc=f"{tf} transform"):
                    image_source = join(ROOT, v, "Images")
                    image_target = join(tf_folder, v, "Images")
                    makedirs(image_target)
                    image_names = sorted(listdir(image_source))

                    for n in image_names:
                        image = to_dtype(
                            pil_to_tensor(Image.open(join(image_source, n))),
                            torch.float,
                            scale=True,
                        )
                        image = resize(image, image_size, antialias=True)  # type: ignore
                        image = TRANSFORMS[tf](image)
                        torch.save(image, join(image_target, n[:-4] + ".pt"))

                    mask_source = join(ROOT, v, "Labels")
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

        self.do_normalize = normalize
        self.do_jitter = jitter
        self.do_rotate = rotate
        self.do_flip = flip
        self.split = split

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        mask = mask_to_setup_cadis(mask, self.setup)
        # chance = torch.rand(1).item()
        # if chance < self.noise_rate:
        if index < self.noise_rate * len(self):
            mask = make_noise(mask.unsqueeze(0), self.noise_type, "cadis", self.setup)  # type: ignore
            # mask = torch.randint_like(mask, self.num_classes[self.setup])

        mean = [0.5737, 0.3461, 0.1954]
        std = [0.1593, 0.1558, 0.1049]
        if self.split == 0:
            if self.do_normalize:
                image = normalize(image, mean, std)
            elif self.do_jitter:
                image = tf_jitter(image)
            elif self.do_rotate:
                angle = random.choice(range(-10, 11))
                image = rotate(image, angle, fill=0)  # type: ignore
                mask = rotate(
                    mask.unsqueeze(0), angle, fill=self.num_classes[self.setup] - 1  # type: ignore
                )
            elif self.do_flip:
                image = vertical_flip(image)
                mask = vertical_flip(mask)
        return image, mask.squeeze()
