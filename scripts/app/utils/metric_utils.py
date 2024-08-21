# utils/metric_utils.py
from scipy.stats import entropy
from app.utils.state_utils import update_inpainted_square
from app.utils.general_utils import ensure_3_channels, is_square_inpainted, get_keys
from app.utils.general_utils import apply_func_to_grid
from scripts.app.data_manager import DataManager

import lpips
import torch
import numpy as np
import os
from PIL import Image
from pyemd import emd
import streamlit as st
from joblib import Parallel, delayed


# lpips_model = lpips.LPIPS(net='alex')

emd_cost_matrix = np.zeros((78, 78))
for i in range(0,80-2):
    for j in range(0,80-2):
        emd_cost_matrix[i, j] = abs(i - j) 
        


# def calculate_lpips(original_array, inpainted_array, min_size=32):
#     min_size = 32
#     if original_array.shape[0] < min_size or original_array.shape[1] < min_size:
#         original_array_lpips = np.array(Image.fromarray(original_array).resize((min_size, min_size), Image.BICUBIC))
#         inpainted_array_lpips = np.array(Image.fromarray(inpainted_array).resize((min_size, min_size), Image.BICUBIC))
#     else:
#         original_array_lpips = original_array
#         inpainted_array_lpips = inpainted_array

#     original_tensor_lpips = torch.tensor(original_array_lpips.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
#     inpainted_tensor_lpips = torch.tensor(inpainted_array_lpips.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

#     lpips_value = lpips_model(original_tensor_lpips, inpainted_tensor_lpips)
#     return lpips_value

def calculate_square_metrics(inpainted_x, inpainted_y, image, image_path, square_length, offset, img_index, index=None):
    square = (inpainted_x, inpainted_y, square_length)
    grid_key, square_key = get_keys(square, offset)
    metric_index = len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['inpainted_square_image']) - 1 if index is None else index

    inpainted_square = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['inpainted_square_image'][metric_index]
    original_square = image.crop((inpainted_x, inpainted_y, inpainted_x + square_length, inpainted_y + square_length))

    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name

    # Load and extract contour square
    contour_array = DataManager.load_contour_array(contour_path)
    contour_square = contour_array[inpainted_y:inpainted_y + square_length, inpainted_x:inpainted_x + square_length]

    # Ensure both images have 3 channels
    # original_array, inpainted_array = ensure_3_channels(original_square, inpainted_square)
    # lpips_value = calculate_lpips(original_array, inpainted_array)
    # lpips_value = 0 # ignore LPIPS for now

    # Apply histogram calculation only to the white pixels in the contour mask
    mask = (contour_square == 255)
    original_scaled = (np.asarray(original_square).astype(np.float64) / 255.0) * 80
    inpainted_scaled = (np.asarray(inpainted_square).astype(np.float64) / 255.0) * 80
    original_histogram_values = original_scaled[mask].flatten()
    inpainted_histogram_values = inpainted_scaled[mask].flatten()
    original_hist = np.histogram(original_histogram_values, bins=80, range=(0, 80), density=False)[0].astype(np.float64)
    inpainted_hist = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80), density=False)[0].astype(np.float64)

    if original_histogram_values.size == 0 or inpainted_histogram_values.size == 0:
        update_inpainted_square(img_index, grid_key, square_key, metrics=False, index=index)
        return
    
    # Calculate difference in means
    mean_inpainted_hist = np.mean(inpainted_histogram_values)
    mean_original_hist = np.mean(original_histogram_values)
    mean_diff = mean_original_hist - mean_inpainted_hist
    
    # Calculate EMD
    # emd_value = emd(original_histogram_values, inpainted_histogram_values, emd_distance_matrices[square_length], extra_mass_penalty=emd_alpha[square_length])
    emd_value = emd(original_hist[1:-1], inpainted_hist[1:-1], emd_cost_matrix) / np.sum(original_hist[1:-1])

    # Calculate additional metrics as differences
    std_dev_diff = np.std(original_histogram_values) - np.std(inpainted_histogram_values)

    mse = np.mean((original_histogram_values[original_histogram_values < 80] - inpainted_histogram_values[original_histogram_values < 80]) ** 2) if np.any(original_histogram_values < 80) else 0


    # Store metrics with square key
    metrics = {
        "lpips": None, # lpips_value.item(),
        "original_histogram_data": original_histogram_values,
        "inpainted_histogram_data": inpainted_histogram_values,
        "histogram_image": None,
        "mean_diff": mean_diff,
        "emd": emd_value,
        "std_dev_diff": std_dev_diff,
        "mse": mse
    }

    update_inpainted_square(img_index, grid_key, square_key, metrics=metrics, index=index)

def calculate_grid_metrics(image, image_path, square, offset, img_index, index=None):
    apply_func_to_grid(square[2], offset, image.size[0], calculate_square_metrics, image, image_path, square[2], offset, img_index, index)

def calculate_series_metrics(series, series_image_paths, square_lengths):
    def process_image(img_index):
        for square_length in square_lengths:
            for offset in range(4):  # Adjust range as needed
                apply_func_to_grid(
                    square_length,
                    offset,
                    series[img_index].size[0],
                    calculate_square_metrics,
                    series[img_index],
                    series_image_paths[img_index],
                    square_length,
                    offset,
                    img_index
                )
    
    # Parallel processing
    Parallel(n_jobs=1)(delayed(process_image)(img_index) for img_index in range(len(series)))

def navigate_metrics(img_index, grid_key, square_key, direction):
    metrics_amount = len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'])
    metrics_index = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['index']
    st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['index'] = (metrics_index + direction) % metrics_amount

def handle_metric_toggle_buttons(square, offset, img_index):
    grid_key, square_key = get_keys(square, offset)

    is_inpainted = is_square_inpainted(img_index, grid_key, square_key)
    if not is_inpainted:
        return
    metric_amount = len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'])
    if not metric_amount > 1:
        return
    
    with st.sidebar:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('Previous'):
                navigate_metrics(img_index, grid_key, square_key, -1)
        with col2:
            if st.button('Next'):
                navigate_metrics(img_index, grid_key, square_key, 1)

# @st.cache_data
# def compute_emd_2d_distance_matrix(size):
#     distance_matrix = np.zeros((size * size, size * size))
#     for i in range(size):
#         for j in range(size):
#             for k in range(size):
#                 for l in range(size):
#                     distance_matrix[i * size + j, k * size + l] = (i - k)**2 + (j - l)**2
#     return distance_matrix

# emd_distance_matrices = {
#     64: compute_emd_2d_distance_matrix(64),
#     48: compute_emd_2d_distance_matrix(48),
#     32: compute_emd_2d_distance_matrix(32),
#     16: compute_emd_2d_distance_matrix(16),
#     8: compute_emd_2d_distance_matrix(8),
# }

# emd_alpha = {
#     64: 512,
#     48: 256,
#     32: 128,
#     16: 32,
#     8: 8,
# }
