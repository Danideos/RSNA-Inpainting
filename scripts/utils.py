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
from noise import snoise3
from scipy.ndimage import convolve
import cv2
from testing.get_edges import process_image


def generate_masks_and_noise(amount=12000):
    
    masks = Parallel(n_jobs=-1)(delayed(generate_single_mask_and_noise)(square_length, shift, i * shift)
                                for square_length in [8, 16, 32, 48]
                                for shift in range(4)
                                for i in tqdm(range(amount // 12)))
    return masks

def generate_single_mask_and_noise(square_length, shift, seed=None):
    noise_low = generate_simplex_noisy_img((256, 256), square_length, amplitude=0.1, seed=seed)
    noise_high = generate_simplex_noisy_img((256, 256), square_length, amplitude=0.2, seed=seed)
    mask, noise = generate_single_mask((256, 256), square_length, shift, noise_high, noise_low)
    return (mask, noise)

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

def generate_single_mask(img_shape, square_length, shift=None):
    h, w = img_shape
    grid_h = h // square_length
    grid_w = w // square_length
    
    i = random.randint(0, 2)
    j = random.randint(0, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    # if random.random() < 0.5:
    #     noise_high = noise_low.copy()
    # noise = noise_low.copy()
    
    for y in range(j, grid_h, 3):
        for x in range(i, grid_w, 3):
            mask[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = 1
            # if random.random() < 0.5:
            #     noise[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = noise_low[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length]
            # else:
            #     noise[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = noise_high[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length]

    # all_squares = [(y, x) for y in range(grid_h) for x in range(grid_w)]
    # random.shuffle(all_squares)
    # n = random.choice(0, len(all_squares) // 10)
    # selected_squares = all_squares[:n]
    
    # for y, x in selected_squares:
    #     mask[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = 1
    #     if random.random() < 0.5:
    #         noise[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = noise_low[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length]
    #     else:
    #         noise[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length] = noise_high[y*square_length:(y+1)*square_length, x*square_length:(x+1)*square_length]
    mask = apply_random_shift(mask, square_length, shift)
    # mask = smooth_mask_edges(mask, square_length)
    
    return mask

def apply_random_shift(mask, square_length, shift=None):
    if shift is None:
        shift_x = random.choice([0, square_length // 2])
        shift_y = random.choice([0, square_length // 2])
    else:
        shift_x, shift_y = (shift % 2) * square_length // 2, (shift // 2) * square_length // 2
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
    
    mask = np.asarray(grid)
    # mask = smooth_mask_edges(mask, square_length)

    mask_tensor = torch.tensor(mask, dtype=torch.float).unsqueeze(0)  # Add channel dimension
    img_tensor = torch.tensor(img, dtype=torch.float)
    
    # noise = generate_simplex_noisy_img((256, 256), square_length, amplitude=0.225)
    # simplex_img = img + noise * 255
    # simplex_img = np.clip(simplex_img, 0, 255)
    # simplex_tensor = torch.tensor(simplex_img, dtype=torch.float)

    masked_img = img_tensor * (1 - mask_tensor) #+ simplex_tensor * mask_tensor
    # plt.imshow(masked_img.squeeze().numpy(), cmap='gray')
    # plt.title('Masked Image with Simplex Noise')
    # plt.show()\
    combined = torch.cat([concat[0].unsqueeze(0), concat[1].unsqueeze(0), masked_img], dim=0)
    
    
    data['concat'] = combined / 127.5 - 1
    data['img'] = img / 127.5 - 1

    return data

def lambda_transform(data):
    img = data['img']
    concat = data['concat']
    
    square_length = random.choice([8, 16, 32, 64])
    mask = generate_single_mask(img.shape[-2:], square_length)
    # random_mask = random.randint(0, len(masks_and_noise) - 1)
    # mask, noise = masks_and_noise[random_mask]
    mask_tensor = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
    img_tensor = img.clone().detach()

    # simplex_img = img + noise * 255
    # simplex_img = np.clip(simplex_img, 0, 255)
    # simplex_tensor = torch.tensor(simplex_img, dtype=torch.float)
    
    masked_img = img_tensor * (1 - mask_tensor) # + simplex_tensor * mask_tensor

    edge_image = process_image(img[0].detach().cpu().numpy()) 
    edge_image = concat[1] if edge_image is None else edge_image
    combined = torch.cat([concat[0].unsqueeze(0), torch.tensor(edge_image, dtype=torch.float).unsqueeze(0), masked_img], dim=0)
    
    data['concat'] = combined / 127.5 - 1
    data['img'] = img / 127.5 - 1

    return data

def smooth_mask_edges(mask, square_length):
    # Create a convolution kernel that smooths the edges
    kernel_size = max(1, square_length // 4)
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    
    # Apply the convolution to the mask
    smoothed_mask = convolve(mask.astype(float), kernel, mode='constant', cval=0.0)
    
    return smoothed_mask

def generate_simplex_noisy_img(img_shape, square_length, frequency=2**-6, amplitude=0.225, octaves=6, decay=0.8, seed=None):
    height, width = img_shape
    noise = np.zeros((height, width))
    frequency = 32 / square_length * frequency
    if seed is None:
        seed = random.randint(0, 10000)
    
    for i in range(height):
        for j in range(width):
            noise_value = 0.0
            frequency_i = frequency
            amplitude_i = amplitude
            for _ in range(octaves):
                noise_value += amplitude_i * snoise3(i * frequency_i, j * frequency_i, seed)
                frequency_i *= 2
                amplitude_i *= decay
            noise[i][j] = noise_value
   
    return noise

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
