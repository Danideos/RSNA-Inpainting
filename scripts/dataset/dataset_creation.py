from sklearn.model_selection import train_test_split
from monai.data import Dataset
import monai.transforms as mt
from utils import add_channel, lambda_transform_single_square, lambda_transform_ano, deselect_key
import torch
import glob
import os
from tqdm import tqdm 
import csv
import ast


CONCATS_TO_RESIZE = {
    "mask_png", "edge_png", "left_png", "right_png"
}

def load_sop_uids_from_csv(csv_file_path):
    sop_uids = {}
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data_dict = ast.literal_eval(row["bounding_box"]) 
            if row['id'] in sop_uids:
                sop_uids[row['id']].append(data_dict)
            else:
                sop_uids[row['id']] = [data_dict]
    return sop_uids

def create_file_paths(file_dirs, required):
    files = {concat_type: set(glob.glob(os.path.join(file_dir, "*.png"))) 
             for concat_type, file_dir in list(zip(required, file_dirs))[1:]}
    files["img"] = sorted(glob.glob(os.path.join(file_dirs[0], "*.png")))
    return files

def get_concat_files(file_dirs, base_name, required):
    concat_files = {concat_type: os.path.join(file_dir, base_name) for concat_type, file_dir in list(zip(required, file_dirs))[1:]}
    return concat_files

def has_required_concat_files(files, concat_files, required=None):
    if required is None:
        return True 
    for required_file_type in required[1:]:
        if concat_files[required_file_type] not in files[required_file_type]:
            return False
    return True

def create_file_dict(img_file, concat_files, required):
    concat = [concat_files[concat_type] for concat_type in required[1:] if concat_type in CONCATS_TO_RESIZE]
    concat256 = [concat_files[concat_type] for concat_type in required[1:] if concat_type not in CONCATS_TO_RESIZE]
    file_dict = {"img": img_file, "concat": concat, "concat256": concat256}
    return file_dict

def create_data_dicts(file_dirs, required): 
    files = create_file_paths(file_dirs, required=required)
    data_dicts = []
    for img_file in tqdm(files["img"], desc="Creating data dict"):
        base_name = os.path.basename(img_file)
        concat_files = get_concat_files(file_dirs, base_name, required=required)
        if has_required_concat_files(files, concat_files, required=required):
            data_dicts.append(create_file_dict(img_file, concat_files, required))
        else:
            print(f"ano file not found: {base_name}")
    
    return data_dicts

def create_transforms(img_size, resize_size):
    return mt.Compose([
        # Load
        mt.LoadImageD(keys=["img", "concat", "concat256"]),
        mt.SelectItemsD(keys=["img", "concat", "concat256"]),
        mt.LambdaD(keys=["img", "concat", "concat256"], func=lambda x: add_channel(x)),
        # Resize
        mt.ResizeWithPadOrCropD(keys=["img", "concat"], spatial_size=(img_size, img_size)),
        mt.ResizeD(keys=["img", "concat"], spatial_size=(resize_size, resize_size)),
        # Augmentation
        mt.Lambda(func=lambda x: lambda_transform_ano(x)),
        mt.Lambda(func=lambda x: deselect_key(x, key_to_deselect="concat256")),
        mt.RandRotateD(keys=["img", "concat"], prob=0.5, range_x=0.25),
        mt.RandFlipD(keys=["img", "concat"], prob=0.5, spatial_axis=1),
        # Convert to tensors
        mt.ToTensorD(keys=["img", "concat"], dtype=torch.float, track_meta=False),
    ])

def create_dataset(file_dirs, required=["bet_png"], img_size=512, resize_size=256):
    data_dicts = create_data_dicts(file_dirs, required)
    train_data_dicts, val_data_dicts = train_test_split(data_dicts, test_size=0.05, random_state=42)

    transforms = create_transforms(img_size, resize_size)

    train_ds = Dataset(data=train_data_dicts, transform=transforms)
    val_ds = Dataset(data=val_data_dicts, transform=transforms)

    return train_ds, val_ds

def get_datasamplers(train_ds, val_ds, total_images_seen):
    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=total_images_seen)
    val_sampler = torch.utils.data.RandomSampler(val_ds, replacement=True, num_samples=total_images_seen)

    return train_sampler, val_sampler
