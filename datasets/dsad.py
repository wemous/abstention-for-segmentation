# https://www.nature.com/articles/s41597-022-01719-2

from os import makedirs
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import (
    normalize,
    pil_to_tensor,
    resize_image,
    resize_mask,
    to_dtype,
)
from tqdm import tqdm

# dataset path
root_path = Path("/data/wesam/datasets/DSAD/")
multilabel = root_path.joinpath("multilabel")

# our video splits.
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

# mean and std for normalization calculated over the train split
mean = [0.4944, 0.2892, 0.2332]
std = [0.2423, 0.1864, 0.1665]


def to_tensor(path, dtype=torch.float) -> Tensor:
    return to_dtype(pil_to_tensor(Image.open(path)), dtype=dtype, scale=True).cuda()


def build_image_and_mask(video_path, index) -> tuple[Tensor, Tensor]:
    image_path = Path(video_path).joinpath(f"image{index}.png")
    image = to_tensor(image_path)
    image = resize_image(image, [384, 480])
    image = normalize(image, mean=mean, std=std)

    # combine all organ masks
    mask_paths = sorted([*Path(video_path).glob(f"mask{index}*")])
    organ_masks = torch.stack([to_tensor(p, dtype=torch.long) for p in mask_paths])
    organ_masks = resize_mask(organ_masks, [384, 480])
    # create background channel
    background = 1 - organ_masks.max(0)[0].unsqueeze(0)
    mask = torch.concat([background, organ_masks]).argmax(0)

    return image, mask


def build_dataset(split: str):
    image_paths, mask_paths = [], []
    split_path = root_path.joinpath(f"transformed/{split}")

    # check if dataset exists
    if not split_path.exists():
        print(f"Building {split} images and masks")
        length = 142 if split == "valid" else 288 if split == "test" else 1000

        p_bar = tqdm(total=length, desc="images")
        for v in video_splits[split]:
            source = multilabel.joinpath(v)
            destination = split_path.joinpath(v)
            makedirs(destination)

            for image_path in [*source.glob("image*")]:
                index = image_path.stem[-2:]
                image, mask = build_image_and_mask(source, index)
                torch.save(image.cpu(), destination.joinpath(f"image{index}.pt"))
                torch.save(mask.cpu(), destination.joinpath(f"mask{index}.pt"))
                p_bar.update()
        p_bar.close()

    image_paths.extend(sorted([*split_path.rglob("image*")]))
    mask_paths.extend(sorted([*split_path.rglob("mask*")]))
    return image_paths, mask_paths


class DSAD(Dataset):
    def __init__(self, split: str = "train"):
        super().__init__()
        self.image_paths, self.mask_paths = build_dataset(split)
        # sanity check
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        return image, mask

    def denorm(self, image: Tensor) -> Tensor:
        """Denormalize the image tensor."""
        return image * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
