# https://arxiv.org/pdf/1906.11586
# https://cataracts.grand-challenge.org/CaDIS/

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
root_path = Path("/data/wesam/datasets/CaDIS/")

# video splits specified by the authors
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

# mean and std for normalization calculated over the train split
mean = [0.5737, 0.3461, 0.1954]
std = [0.1593, 0.1558, 0.1049]

# setup 2 mapping
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


# covert raw masks to a setup mapping
def mask_to_setup(mask: Tensor, setup: int) -> Tensor:
    if setup == 1:
        mask[mask > 7] = 7
        # shift classes by 1 so the 'Instrument' class becomes 0
        mask = (mask + 1) % 8
    if setup == 2:
        for k, values in setup_2_class_map.items():
            for v in values:
                mask[mask == v] = k
        # shift classes by 1 so the ignored classes become 0
        mask = (mask + 1) % 18
    if setup == 3:
        mask[mask > 25] = 25
        # shift classes by 1 so the ignored classes become 0
        mask = (mask + 1) % 26
    return mask


def to_tensor(path, dtype=torch.float, scale=False) -> Tensor:
    return to_dtype(pil_to_tensor(Image.open(path)), dtype=dtype, scale=scale).cuda()


def build_dataset(split: str):
    assert split in ["train", "valid", "test"]
    image_paths, mask_paths = [], []
    split_path = root_path.joinpath(f"transformed/{split}")

    # check if dataset exists
    if not split_path.exists():
        print(f"Building {split} images and masks")
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
                image = resize_image(image, [256, 480])
                image = normalize(image, mean=mean, std=std)

                mask = to_tensor(m_path, dtype=torch.long)
                mask = resize_mask(mask, [256, 480])

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
    def __init__(self, split: str = "train", setup: int = 1):
        super().__init__()
        assert setup in [1, 2, 3]
        self.setup = setup
        self.num_classes = {1: 8, 2: 18, 3: 26}
        self.image_paths, self.mask_paths = build_dataset(split)
        # sanity check
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        mask = mask_to_setup(mask, self.setup)
        return image, mask

    def denorm(self, image: Tensor) -> Tensor:
        """Denormalize the image tensor."""
        return image * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
