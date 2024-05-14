from os import cpu_count, listdir, makedirs
from os.path import exists, join

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import (
    pil_to_tensor,
    resize,
    to_dtype,
    to_image,
)
from tqdm import tqdm

DATA_PATH = "/data/wesam/datasets/ADE20K/"


class ADE20K(Dataset):
    def __init__(self, train=True, image_size: "tuple[int, int]" = (480, 512)):
        super().__init__()
        split = "training" if train else "validation"
        trasnformed_path = join(DATA_PATH, f"transformed_{image_size[0]}_{image_size[1]}")

        image_folder = join(DATA_PATH, "images", split)
        transformed_image_folder = join(trasnformed_path, "images", split)
        if not exists(transformed_image_folder):
            makedirs(transformed_image_folder)
            file_names = sorted(listdir(image_folder))
            for f in tqdm(file_names, desc=f"transforming {split} images"):
                image = to_dtype(
                    to_image(Image.open(join(image_folder, f))),
                    torch.float,
                    scale=True,
                )
                if image.shape[0] == 1:
                    image = image.expand(3, -1, -1)
                image = resize(image, image_size, antialias=True)  # type: ignore
                torch.save(image, join(transformed_image_folder, f[:-4] + ".pt"))

        mask_folder = join(DATA_PATH, "annotations", split)
        transformed_mask_folder = join(trasnformed_path, "annotations", split)
        if not exists(transformed_mask_folder):
            makedirs(transformed_mask_folder)
            file_names = sorted(listdir(mask_folder))
            for f in tqdm(file_names, desc=f"transforming {split} annotations"):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> "tuple[Tensor, Tensor]":
        image = torch.load(self.image_paths[index])
        mask = torch.load(self.mask_paths[index])
        return image, mask


# class ADE20KLoader(DataLoader):
#     def __init__(self, train=True, batch_size=64):
#         ade = ADE20K(train)
#         super().__init__(
#             ade,
#             batch_size=batch_size,
#             shuffle=train,
#             drop_last=False,
#             collate_fn=self.collate_fn,
#             num_workers=cpu_count(),  # type: ignore
#         )

#     def collate_fn(self, batch: "list[tuple[torch.Tensor, torch.Tensor]]"):
#         images, masks = zip(*batch)
#         sizes = torch.tensor([_.shape[-2:] for _ in images])
#         h, w = sizes.median(0)[0]
#         images = torch.stack(
#             [resize(_, [int(h), int(w)], antialias=True) for _ in images]
#         )
#         masks = torch.stack([resize(_, [int(h), int(w)], antialias=True) for _ in masks])
#         return images, masks.squeeze()
