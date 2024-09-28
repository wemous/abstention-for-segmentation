import random
from pathlib import Path

from kornia.morphology import dilation, erosion
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dsad import DSAD


def flip_label(mask: torch.Tensor):
    if len(mask.unique()) > 1:
        label_to_flip = random.choice(mask.unique()[1:].tolist())
        targets = [*range(1, 8)]
        targets.remove(label_to_flip)
        mask[mask == label_to_flip] = random.choice(targets)
    return mask


class NoisyDSAD(Dataset):
    def __init__(
        self,
        noise_level: int,
        flipped=False,
        gaussian=False,
        jitter=False,
        normalized=False,
        rotated=False,
    ):
        super().__init__()
        assert noise_level > 0 and noise_level < 6

        dsad = DSAD(
            split="train",
            flipped=flipped,
            gaussian=gaussian,
            jitter=jitter,
            normalized=normalized,
            rotated=rotated,
        )

        self.image_paths = dsad.image_paths
        self.mask_paths = []
        self.noise_rate = 0

        noise_config = {
            1: (torch.ones([3, 3]).cuda(), 0.02),
            2: (torch.ones([8, 8]).cuda(), 0.0),
            3: (torch.ones([9, 9]).cuda(), 0.35),
            4: (torch.ones([11, 11]).cuda(), 0.6),
            5: (torch.ones([11, 11]).cuda(), 1.0),
        }

        root_path = Path(f"/data/wesam/datasets/DSAD/noisy/{noise_level}")
        if not root_path.exists():
            print(f"Building noisy labels at noise level {noise_level}")
            morph = [dilation] * (len(dsad) // 2) + [erosion] * (len(dsad) // 2)
            random.shuffle(morph)

            for i, p in tqdm(enumerate(dsad.mask_paths), total=len(dsad)):
                mask = torch.load(p, weights_only=True).cuda()
                noisy_mask = mask.clone()
                kernel, flip_rate = noise_config[noise_level]
                noisy_mask = morph[i](noisy_mask.unsqueeze(0), kernel)
                if i % 1000 < 1000 * flip_rate:
                    noisy_mask = flip_label(noisy_mask)
                self.noise_rate += (mask != noisy_mask).float().mean()
                noisy_mask_path = Path(
                    str(p).replace("transformed", f"noisy/{noise_level}")
                )
                noisy_mask_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(noisy_mask.cpu(), noisy_mask_path)

            torch.save(self.noise_rate / len(dsad), root_path.joinpath("noise_rate.pt"))

        self.mask_paths = [
            str(p).replace("transformed", f"noisy/{noise_level}") for p in dsad.mask_paths
        ]
        self.noise_rate = torch.load(
            root_path.joinpath("noise_rate.pt"), weights_only=True
        ).item()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        return image, mask
