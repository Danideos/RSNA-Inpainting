import yaml
import torch
import glob
import os
import random
import matplotlib.pyplot as plt
import random
from monai.data import Dataset
import monai.transforms as mt
import pydicom
import numpy as np
from mediffusion.ddpm import DiffusionModule
from joblib import Parallel, delayed
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def get_dicom_file_series_info(file_path):
    try:
        dicom = pydicom.dcmread(file_path, force=True)
        series_uid = dicom.SeriesInstanceUID
        slice_id = os.path.splitext(os.path.basename(file_path))[0].split("_")[1]  # Extract the ID from the filename (remove .dcm)
        return series_uid, slice_id
    except Exception as e:
        return None, None

def get_series_slice_mapping(input_dir, n_jobs=-1):
    file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.dcm')]
    
    results = Parallel(n_jobs=n_jobs)(delayed(get_dicom_file_series_info)(file_path) for file_path in tqdm(file_paths))
    
    series_to_slices = defaultdict(list)
    for series_uid, slice_id in results:
        series_to_slices[series_uid].append(slice_id)
    
    return series_to_slices

def analyze_csv(csv_file, series_to_slices):
    df = pd.read_csv(csv_file)

    # Filter for 'any' type labels
    df_any = df[df['ID'].str.contains('_any')]
    df_any['SliceID'] = df_any['ID'].apply(lambda x: x.split('_')[1])

    # Create sets for fast lookup
    healthy_ids = set(df_any[df_any['Label'] == 0]['SliceID'])
    unhealthy_ids = set(df_any[df_any['Label'] == 1]['SliceID'])

    # Create a dictionary to track counts
    series_label_counts = defaultdict(lambda: {'0': 0, '1': 0})

    for series_uid, slice_ids in series_to_slices.items():
        for slice_id in slice_ids:
            if slice_id in healthy_ids:
                series_label_counts[series_uid]['0'] += 1
            elif slice_id in unhealthy_ids:
                series_label_counts[series_uid]['1'] += 1

    return series_label_counts

def generate_single_mask(img_shape, square_length):
    h, w = img_shape
    square_length = square_length
    grid_h = h // square_length
    grid_w = w // square_length
    
    i = random.randint(0, 2)
    j = random.randint(0, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(j, grid_h, 3):
        for x in range(i, grid_w, 3):
            mask[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = 1
    
    return mask

def apply_random_shift(mask, square_length):
    shift_x = random.choice([0, square_length // 2])
    shift_y = random.choice([0, square_length // 2])
    shifted_mask = np.roll(np.roll(mask, shift_x, axis=1), shift_y, axis=0)
    return shifted_mask

def plot_image(tensor, title):
    img = tensor.squeeze().detach().cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def lambda_transform_with_grid(data, grid):
    img = data['img']
    concat = data['concat']
    
    mask = grid
    
    mask_tensor = torch.tensor(mask, dtype=torch.float).unsqueeze(0)  # Add channel dimension
    img_tensor = torch.tensor(img, dtype=torch.float)
    masked_img = img_tensor * (1 - mask_tensor)
    
    combined = torch.cat([concat, masked_img], dim=0)
    
    data['concat'] = combined / 127.5 - 1
    data['img'] = img / 127.5 - 1

    return data

def lambda_transform(data):
    img = data['img']
    concat = data['concat']
    
    square_length = random.choice([8, 16, 32])
    mask = generate_single_mask(img.shape[-2:], square_length)
    mask = apply_random_shift(mask, square_length)
    
    mask_tensor = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
    img_tensor = img.clone().detach()
    
    r = random.randint(15, 25)
    noisy_img = add_gaussian_noise(img_tensor, mean=0, variance=r / 1000) 
    masked_img = img_tensor * (1 - mask_tensor) + noisy_img * mask_tensor
    
    combined = torch.cat([concat, masked_img], dim=0)
    
    data['concat'] = combined / 127.5 - 1
    data['img'] = img / 127.5 - 1

    return data


def add_gaussian_noise(image, mean=0, variance=0.02):
    """
    Add Gaussian noise to a grayscale image.
    
    Parameters:
        image (numpy.ndarray): Grayscale image.
        mean (float): Mean of the Gaussian noise.
        variance (float): Variance of the Gaussian noise.
        
    Returns:
        numpy.ndarray: Noisy image.
    """
    sigma = variance ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian * 255
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure values are within [0, 1]
    return torch.tensor(noisy_image, dtype=torch.float)

def load_config(config_path: str):
    '''
    Loads the configuration file from the given path.
    '''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_tensor_shape(x: torch.Tensor):
    '''
    Prints the shape of the given tensor for debugging purposes.
    '''
    print("Tensor Shape:", x.shape)
    return x

def add_channel(x: torch.Tensor):
    '''
    Adds a channel to the tensor if it is not present.
    '''
    if len(x.shape) != 3:
        x = x.unsqueeze(0)
    return x

def get_dicom_file(data_dir: str, index: int=-1):
    '''
    Returns a random DICOM file from the given directory.
    Random index is selected if not provided.
    '''
    dicom_files = glob.glob(os.path.join(data_dir, "*.dcm"))
    if index == -1:
        index = random.randint(0, len(dicom_files) - 1)
    return dicom_files[index]

def apply_window(img, center, width):
    '''
    Applies the windowing technique to the given image tensor or numpy array.
    '''
    min_value = center - width // 2
    max_value = center + width // 2
    
    if isinstance(img, torch.Tensor):
        img = torch.clamp(img, min_value, max_value)
        img = (img - min_value) / (max_value - min_value)
    elif isinstance(img, np.ndarray):
        img = np.clip(img, min_value, max_value)
        img = (img - min_value) / (max_value - min_value)
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
    
    return img

def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)

def rescale_image(image, slope, intercept):
    return image * slope + intercept

def apply_mask(image, mask):
    mask = mask > 0
    masked_image = np.where(mask, image, image.min())
    return masked_image

def extract_base_name(full_name):
    return "_".join(full_name.split("_")[:-1])

def visualize_random_dataset_samples(dataset: Dataset, num_samples: int = 5):
    """
    Visualizes a few samples from the dataset.

    Parameters:
    - dataset: MONAI Dataset, the dataset to visualize samples from.
    - num_samples: int, number of samples to visualize.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        sample = dataset[random.randrange(len(dataset))]["img"][0].permute(1, 0).numpy() # Assuming the channel is the first dimension
        axes[i].imshow(sample, cmap="gray")
        axes[i].axis("off")
    plt.show()

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_divisible_masks(image_shape, square_length, divisibility_factor=2, device='cuda'):
    batch_size, channels, height, width = image_shape
    masks = []

    for div_x in range(divisibility_factor):
        for div_y in range(divisibility_factor):
            mask = torch.zeros((batch_size, channels, height, width), dtype=torch.uint8, device=device)
            for y in range(div_y * square_length, height, square_length * divisibility_factor):
                for x in range(div_x * square_length, width, square_length * divisibility_factor):
                    mask[:, :, y:y + square_length, x:x + square_length] = 1
            masks.append(mask.to(device))
    
    return masks

def prepare_model(config_path, model_path, device="cpu"):
    model = DiffusionModule(config_path)
    model.load_ckpt(model_path, ema=True)
    model.eval().to(device).half() if device == 'cuda' else model.eval().to(device).float()
    return model
