from sklearn.model_selection import train_test_split
from monai.data import Dataset
import monai.transforms as mt
from utils import add_channel, lambda_transform, generate_masks_and_noise
import torch
import glob
import os
from tqdm import tqdm 

# MASKS_AND_NOISE = generate_masks_and_noise(amount=60000)


def create_data_dicts(data_dir, mask_dir, edge_dir, left_dir, right_dir):
    png_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    mask_files = set(glob.glob(os.path.join(mask_dir, "*.png")))
    edge_files = set(glob.glob(os.path.join(edge_dir, "*.png")))
    left_files = set(glob.glob(os.path.join(left_dir, "*.png")))
    right_files = set(glob.glob(os.path.join(right_dir, "*.png")))

    data_dicts = []
    for img_file in tqdm(png_files, desc="Creating data dict"):
        base_name = os.path.basename(img_file)
        mask_file = os.path.join(mask_dir, base_name)
        edge_file = os.path.join(edge_dir, base_name)
        left_file = os.path.join(left_dir, base_name)
        right_file = os.path.join(right_dir, base_name)
        if mask_file in mask_files and edge_file in edge_files and left_file in left_files and right_file in right_files:
            data_dicts.append({"img": img_file, "concat": [mask_file, left_file, right_file]})
    
    return data_dicts

def create_transforms(img_size, resize_size):
    return mt.Compose([
        mt.LoadImageD(keys=["img", "concat"]),
        mt.SelectItemsD(keys=["img", "concat"]),
        mt.LambdaD(keys=["img", "concat"], func=lambda x: add_channel(x)),
        mt.RandRotateD(keys=["img", "concat"], prob=0.5, range_x=0.25),
        mt.RandFlipD(keys=["img", "concat"], prob=0.5, spatial_axis=1),
        mt.ResizeWithPadOrCropD(keys=["img", "concat"], spatial_size=(img_size, img_size)),
        mt.ResizeD(keys=["img", "concat"], spatial_size=(resize_size, resize_size)),
        mt.Lambda(func=lambda x: lambda_transform(x)),
        mt.ToTensorD(keys=["img", "concat"], dtype=torch.float, track_meta=False),
    ])

def create_dataset(data_dir, mask_dir, edge_dir, left_dir, right_dir, img_size, resize_size):
    data_dicts = create_data_dicts(data_dir, mask_dir, edge_dir, left_dir, right_dir)
    train_data_dicts, val_data_dicts = train_test_split(data_dicts, test_size=0.05, random_state=42)

    transforms = create_transforms(img_size, resize_size)

    train_ds = Dataset(data=train_data_dicts, transform=transforms)
    val_ds = Dataset(data=val_data_dicts, transform=transforms)

    return train_ds, val_ds

def get_datasamplers(train_ds, val_ds, total_images_seen):
    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=total_images_seen)
    val_sampler = torch.utils.data.RandomSampler(val_ds, replacement=True, num_samples=total_images_seen)

    return train_sampler, val_sampler
