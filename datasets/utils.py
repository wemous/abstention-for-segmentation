import random
from torch import Tensor
from torchvision.transforms.v2 import GaussianBlur, ElasticTransform


cadis_setup_2_class_map = {
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


def mask_to_setup_cadis(mask: Tensor, setup: int) -> Tensor:
    if setup == 1:
        mask[mask > 7] = 7
    if setup == 2:
        for k, values in cadis_setup_2_class_map.items():
            for v in values:
                mask[mask == v] = k
    if setup == 3:
        mask[mask > 25] = 25
    return mask


def mask_to_setup_ade(mask: Tensor, setup: int) -> Tensor:
    if setup == 1:
        mask = mask
    if setup == 2:
        mask[mask > 76] = 0
    if setup == 3:
        mask[mask > 21] = 0
    return mask


def swap(mask: Tensor) -> Tensor:
    x, y = random.sample(mask.unique().tolist(), k=2)
    z = -1
    mask[mask == x] = z
    mask[mask == y] = x
    mask[mask == z] = y
    return mask


def blur(mask: Tensor, setup: int, dataset: str) -> Tensor:
    tf = GaussianBlur(kernel_size=[9, 9], sigma=10)
    mask = tf(mask)
    if dataset == "cadis":
        mask = mask_to_setup_cadis(mask, setup)
    elif dataset == "ade":
        mask = mask_to_setup_ade(mask, setup)
    return mask


def make_noise(mask: Tensor, noise_type: str, dataset: str, setup: int = 1) -> Tensor:
    if noise_type == "blur":
        mask = blur(mask, setup, dataset)
    elif noise_type == "elastic":
        mask = ElasticTransform()(mask)
    elif noise_type == "swap":
        mask = swap(mask)
    elif noise_type == "all":
        mask = swap(mask)
        mask = ElasticTransform()(mask)
        mask = blur(mask, setup, dataset)
    return mask
