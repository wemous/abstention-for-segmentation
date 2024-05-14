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
}


class CaDIS(Dataset):
    def __init__(
        self,
        train=True,
        transforms: Iterable[str] = {"original"},
        image_size=(270, 480),
    ):
        super().__init__()
        assert set(TRANSFORMS.keys()).issuperset(transforms)
        # add resize transform if passed an empty list
        transforms = transforms if transforms else {"original"}
        # use only resize transform for validation
        transforms = set(transforms) if train else {"validation"}
        split = "training" if train else "validation"

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
            "validation": [
                "Video02",
                "Video05",
                "Video07",
                "Video16",
                "Video22",
            ],
        }
        videos = video_split[split]
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
                        mask = resize(mask, image_size, antialias=True)  # type: ignore
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
        return image, mask
