import random
from pathlib import Path

import torch
from kornia.morphology import dilation, erosion
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .cadis import CaDIS, mask_to_setup


def flip_label(mask: torch.Tensor, num_classes: int):
    """Flip a random label in the mask to another radom label in the dataset."""

    label_to_flip = random.choice(mask.unique().tolist())
    targets = [*range(num_classes)]
    targets.remove(label_to_flip)
    mask[mask == label_to_flip] = random.choice(targets)
    return mask


class NoisyCaDIS(Dataset):
    def __init__(self, noise_level: int, setup: int = 1):
        super().__init__()
        assert noise_level > -1 and noise_level < 6
        assert setup in [1, 2, 3]

        cadis = CaDIS(split="train", setup=setup)
        self.noise_level = noise_level
        self.setup = setup
        self.num_classes = cadis.num_classes[setup]
        self.image_paths = cadis.image_paths

        # return clean dataset if noise level is 0
        if noise_level == 0:
            self.mask_paths = cadis.mask_paths
            self.noise_rate = torch.tensor(0.0).cuda()
            self.class_noise = torch.zeros(self.num_classes).float().cuda()
        else:
            self.mask_paths = []

            # erosion and dilation kernels and flip rates for different noise levels
            noise_config = {
                1: (torch.ones([5, 5]).cuda(), 0.05),
                2: (torch.ones([9, 9]).cuda(), 0.1),
                3: (torch.ones([9, 9]).cuda(), 0.46),
                4: (torch.ones([11, 11]).cuda(), 0.67),
                5: (torch.ones([13, 13]).cuda(), 0.95),
            }

            # root path for noisy dataset
            # should be consistent with the clean dataset root path
            root_path = Path(f"/data/wesam/datasets/CaDIS/noisy/setup_{setup}/{noise_level}")

            if not root_path.exists():
                print(f"Building noisy labels at noise level {noise_level}")
                morph = [dilation] * (len(cadis) // 2) + [erosion] * (len(cadis) // 2)
                random.shuffle(morph)

                pixel_count = torch.zeros(self.num_classes, dtype=torch.long).cuda()
                noise_count = torch.zeros(self.num_classes, dtype=torch.long).cuda()
                noise_rate = 0

                for i, p in tqdm(enumerate(cadis.mask_paths), total=len(cadis)):
                    mask = torch.load(p, weights_only=True).cuda()
                    mask = mask_to_setup(mask, setup)
                    noisy_mask = mask.clone()
                    kernel, flip_rate = noise_config[noise_level]

                    if i < len(cadis) * flip_rate:
                        noisy_mask = flip_label(noisy_mask, cadis.num_classes[setup])
                    noisy_mask = morph[i](noisy_mask.unsqueeze(0), kernel)[0]

                    # accumulate dataset noise rate and class noise
                    noise_rate += (mask != noisy_mask).float().mean()
                    for j in range(self.num_classes):
                        noise_count[j] += ((noisy_mask == j) > (mask == j)).sum()
                        pixel_count[j] += (noisy_mask == j).sum()

                    noisy_mask_path = Path(
                        str(p).replace("transformed/train", f"noisy/setup_{setup}/{noise_level}")
                    )
                    noisy_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(noisy_mask.cpu(), noisy_mask_path)

                torch.save(noise_rate / len(cadis), root_path.joinpath("noise_rate.pt"))
                torch.save(noise_count / pixel_count, root_path.joinpath("class_noise.pt"))

            self.mask_paths = [
                str(p).replace("transformed/train", f"noisy/setup_{setup}/{noise_level}")
                for p in cadis.mask_paths
            ]
            self.noise_rate = torch.load(root_path.joinpath("noise_rate.pt"), weights_only=True)
            self.class_noise = torch.load(root_path.joinpath("class_noise.pt"), weights_only=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image = torch.load(self.image_paths[index], weights_only=True)
        mask = torch.load(self.mask_paths[index], weights_only=True)
        if self.noise_level == 0:
            mask = mask_to_setup(mask, self.setup)
        return image, mask
