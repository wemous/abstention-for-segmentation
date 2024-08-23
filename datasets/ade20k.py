from os import listdir, makedirs
from os.path import exists, join

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import (
    pil_to_tensor,
    resize,
    to_dtype,
    to_image,
)
from tqdm import tqdm
from .utils import make_noise, mask_to_setup_ade

DATA_PATH = "/data/wesam/datasets/ADE20K/"


class ADE20K(Dataset):
    def __init__(
        self,
        split: int,
        setup: int,
        image_size: "tuple[int, int]" = (480, 512),
        noise_rate=0.0,
        noise_type=None,
        dataset_size_factor: float = 1,
    ):
        super().__init__()
        assert split >= 0 and split <= 2
        assert setup >= 1 and setup <= 3
        self.setup = setup
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.num_classes = {1: 22, 2: 77, 3: 151}
        split_ids = {0: "training", 1: "validation", 2: "testing"}
        trasnformed_path = join(DATA_PATH, f"transformed_{image_size[0]}_{image_size[1]}")

        image_folder = join(DATA_PATH, "images", split_ids[split])
        transformed_image_folder = join(trasnformed_path, "images", split_ids[split])
        if not exists(transformed_image_folder):
            makedirs(transformed_image_folder)
            file_names = sorted(listdir(image_folder))
            for f in tqdm(file_names, desc=f"transforming {split_ids[split]} images"):
                image = to_dtype(
                    to_image(Image.open(join(image_folder, f))),
                    torch.float,
                    scale=True,
                )
                if image.shape[0] == 1:
                    image = image.expand(3, -1, -1)
                image = resize(image, image_size, antialias=True)  # type: ignore
                torch.save(image, join(transformed_image_folder, f[:-4] + ".pt"))

        mask_folder = join(DATA_PATH, "annotations", split_ids[split])
        transformed_mask_folder = join(trasnformed_path, "annotations", split_ids[split])
        if not exists(transformed_mask_folder):
            makedirs(transformed_mask_folder)
            file_names = sorted(listdir(mask_folder))
            for f in tqdm(
                file_names, desc=f"transforming {split_ids[split]} annotations"
            ):
                mask = pil_to_tensor(Image.open(join(mask_folder, f)))
                mask = resize(mask, image_size, antialias=True).squeeze().long()  # type: ignore
                torch.save(mask, join(transformed_mask_folder, f[:-4] + ".pt"))

        self.image_paths = [
            join(transformed_image_folder, _)
            for _ in sorted(listdir(transformed_image_folder))
        ]
        self.mask_paths = [
            join(transformed_mask_folder, _)
            for _ in sorted(listdir(transformed_mask_folder))
        ]

        dataset_size_factor = 1 if split == 2 else min(1, dataset_size_factor)
        dataset_size = int(len(self.image_paths) * dataset_size_factor)
        self.image_paths = self.image_paths[:dataset_size]
        self.mask_paths = self.mask_paths[:dataset_size]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index])
        mask = torch.load(self.mask_paths[index])
        mask = mask_to_setup_ade(mask, self.setup)
        chance = torch.rand(1).item()
        if chance < self.noise_rate:
            mask = make_noise(mask.unsqueeze(0), self.noise_type, "ade", self.setup)  # type: ignore
            # mask = torch.randint_like(mask, self.num_classes[self.setup])
        return image, mask.squeeze()
