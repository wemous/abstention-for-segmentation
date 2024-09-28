import random
from pathlib import Path

from kornia.morphology import dilation, erosion
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .cadis import CaDIS, mask_to_setup


def flip_label(mask: torch.Tensor, num_classes: int):
    label_to_flip = random.choice(mask.unique().tolist())
    targets = [*range(num_classes)]
    targets.remove(label_to_flip)
    mask[mask == label_to_flip] = random.choice(targets)
    return mask


class NoisyCaDIS(Dataset):
    def __init__(
        self,
        noise_level: int,
        setup: int = 1,
        flipped=False,
        gaussian=False,
        jitter=False,
        normalized=False,
        rotated=False,
    ):
        super().__init__()
        assert noise_level > 0 and noise_level < 6
        assert setup > 0 and setup < 4

        cadis = CaDIS(
            split="train",
            setup=setup,
            flipped=flipped,
            gaussian=gaussian,
            jitter=jitter,
            normalized=normalized,
            rotated=rotated,
        )

        self.image_paths = cadis.image_paths
        self.mask_paths = []
        self.noise_rate = 0

        noise_config = {
            1: (torch.ones([2, 3]).cuda(), 0.04),
            2: (torch.ones([5, 6]).cuda(), 0.01),
            3: (torch.ones([8, 8]).cuda(), 0.13),
            4: (torch.ones([9, 9]).cuda(), 0.48),
            5: (torch.ones([9, 9]).cuda(), 0.92),
        }

        root_path = Path(f"/data/wesam/datasets/CaDIS/noisy/setup {setup}/{noise_level}")
        if not root_path.exists():
            print(f"Building noisy labels at noise level {noise_level}")
            morph = [dilation] * (len(cadis) // 2) + [erosion] * (len(cadis) // 2)
            random.shuffle(morph)

            for i, p in tqdm(enumerate(cadis.mask_paths), total=len(cadis)):
                mask = torch.load(p, weights_only=True).cuda()
                mask = mask_to_setup(mask, setup)
                noisy_mask = mask.clone()
                kernel, flip_rate = noise_config[noise_level]
                noisy_mask = morph[i](noisy_mask.unsqueeze(0), kernel)
                if i % 3550 < 3550 * flip_rate:
                    noisy_mask = flip_label(noisy_mask, cadis.num_classes[setup])
                self.noise_rate += (mask != noisy_mask).float().mean()
                noisy_mask_path = Path(
                    str(p).replace("transformed", f"noisy/setup {setup}/{noise_level}")
                )
                noisy_mask_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(noisy_mask.cpu(), noisy_mask_path)

            torch.save(self.noise_rate / len(cadis), root_path.joinpath("noise_rate.pt"))

        self.mask_paths = [
            str(p).replace("transformed", f"noisy/setup {setup}/{noise_level}")
            for p in cadis.mask_paths
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
