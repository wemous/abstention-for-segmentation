import random
from pathlib import Path

import torch
from kornia.morphology import dilation, erosion
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .dsad import DSAD


def flip_label(mask: torch.Tensor):
    if len(mask.unique()) > 1:
        label_to_flip = random.choice(mask.unique()[1:].tolist())
        targets = [*range(1, 8)]
        targets.remove(label_to_flip)
        mask[mask == label_to_flip] = random.choice(targets)
    return mask


class NoisyDSAD(Dataset):
    def __init__(self, noise_level: int):
        super().__init__()
        assert noise_level > -1 and noise_level < 6

        dsad = DSAD(split="train")
        self.num_classes = 8
        self.image_paths = dsad.image_paths

        if noise_level == 0:
            self.mask_paths = dsad.mask_paths
            self.noise_rate = torch.tensor(0.0).cuda()
            self.class_noise = torch.zeros(self.num_classes).float().cuda()
        else:
            self.mask_paths = []
            noise_config = {
                1: (torch.ones([7, 7]).cuda(), 0.06),
                2: (torch.ones([9, 9]).cuda(), 0.53),
                3: (torch.ones([13, 13]).cuda(), 0.8),
                4: (torch.ones([19, 19]).cuda(), 0.95),
                5: (torch.ones([29, 29]).cuda(), 1.0),
            }

            root_path = Path(f"/data/wesam/datasets/DSAD/noisy/{noise_level}")
            if not root_path.exists():
                print(f"Building noisy labels at noise level {noise_level}")
                morph = [dilation] * (len(dsad) // 2) + [erosion] * (len(dsad) // 2)
                random.shuffle(morph)

                pixel_count = torch.zeros(self.num_classes, dtype=torch.long).cuda()
                noise_count = torch.zeros(self.num_classes, dtype=torch.long).cuda()
                noise_rate = 0

                for i, p in tqdm(enumerate(dsad.mask_paths), total=len(dsad)):
                    mask = torch.load(p, weights_only=True).cuda()
                    noisy_mask = mask.clone()
                    kernel, flip_rate = noise_config[noise_level]
                    noisy_mask = morph[i](noisy_mask.unsqueeze(0), kernel)[0]
                    if i < len(dsad) * flip_rate:
                        noisy_mask = flip_label(noisy_mask)
                    noise_rate += (mask != noisy_mask).float().mean()
                    for j in range(self.num_classes):
                        noise_count[j] += ((noisy_mask == j) > (mask == j)).sum()
                        pixel_count[j] += (noisy_mask == j).sum()

                    noisy_mask_path = Path(
                        str(p).replace("transformed/train", f"noisy/{noise_level}")
                    )
                    noisy_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(noisy_mask.cpu(), noisy_mask_path)

                torch.save(noise_rate / len(dsad), root_path.joinpath("noise_rate.pt"))
                torch.save(noise_count / pixel_count, root_path.joinpath("class_noise.pt"))

            self.mask_paths = [
                str(p).replace("transformed/train", f"noisy/{noise_level}") for p in dsad.mask_paths
            ]
            self.noise_rate = torch.load(root_path.joinpath("noise_rate.pt"), weights_only=True)
            self.class_noise = torch.load(root_path.joinpath("class_noise.pt"), weights_only=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        return image, mask
