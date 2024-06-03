import os
from tqdm import tqdm
from datasets import ADE20K

image_old_path = "/data/wesam/datasets/ADE20K/images/training/"
image_new_path = "/data/wesam/datasets/ADE20K/images/testing/"
anno_old_path = "/data/wesam/datasets/ADE20K/annotations/training/"
anno_new_path = "/data/wesam/datasets/ADE20K/annotations/testing/"

os.makedirs(image_new_path, exist_ok=True)
os.makedirs(anno_new_path, exist_ok=True)

image_names = sorted(os.listdir(image_old_path))[-2000:]
anno_names = sorted(os.listdir(anno_old_path))[-2000:]
for i, old_name in tqdm(enumerate(image_names)):
    new_name = f"ADE_test_{i+1:08d}{old_name[-4:]}"
    os.replace(image_old_path + old_name, image_new_path + new_name)
for i, old_name in tqdm(enumerate(anno_names)):
    new_name = f"ADE_test_{i+1:08d}{old_name[-4:]}"
    os.replace(anno_old_path + old_name, anno_new_path + new_name)

ade_train = ADE20K(0, 1)
ade_valid = ADE20K(1, 1)
ade_test = ADE20K(2, 1)
