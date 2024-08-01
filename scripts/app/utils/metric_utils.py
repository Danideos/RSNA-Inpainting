# utils/metric_utils.py
from scipy.stats import entropy
from app.utils.state_utils import update_inpainted_square
from app.utils.general_utils import ensure_3_channels, is_square_inpainted, get_keys
from app.utils.general_utils import get_inpainted_square_index
from app.loader import Loader

import lpips
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import streamlit as st

lpips_model = lpips.LPIPS(net='alex')


def calculate_lpips(original_array, inpainted_array, min_size=32):
    min_size = 32
    if original_array.shape[0] < min_size or original_array.shape[1] < min_size:
        original_array_lpips = np.array(Image.fromarray(original_array).resize((min_size, min_size), Image.BICUBIC))
        inpainted_array_lpips = np.array(Image.fromarray(inpainted_array).resize((min_size, min_size), Image.BICUBIC))
    else:
        original_array_lpips = original_array
        inpainted_array_lpips = inpainted_array

    original_tensor_lpips = torch.tensor(original_array_lpips.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inpainted_tensor_lpips = torch.tensor(inpainted_array_lpips.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    lpips_value = lpips_model(original_tensor_lpips, inpainted_tensor_lpips)
    return lpips_value

def calculate_square_metrics(image, image_path, square, offset):
    inpainted_x, inpainted_y, square_length = square
    grid_key, square_key = get_keys(square, offset)
    index = len(st.session_state['all_inpainted_square_images'][grid_key][square_key]['inpainted_square_image']) - 1

    inpainted_square = st.session_state['all_inpainted_square_images'][grid_key][square_key]['inpainted_square_image'][index]
    original_square = image.crop((inpainted_y, inpainted_x, inpainted_y + square_length, inpainted_x + square_length))

    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name

    # Load and and extract contour square
    contour_array = Loader.load_contour_array(contour_path)
    contour_square = contour_array[inpainted_x:inpainted_x + square_length, inpainted_y:inpainted_y + square_length]

    # Ensure both images have 3 channels
    original_array, inpainted_array = ensure_3_channels(original_square, inpainted_square)
    lpips_value = calculate_lpips(original_array, inpainted_array)

    # Apply histogram calculation only to the white pixels in the contour mask
    mask = (contour_square == 255)
    original_scaled = (original_array.astype(np.float32) / 255.0) * 80
    inpainted_scaled = (inpainted_array.astype(np.float32) / 255.0) * 80
    original_histogram_values = original_scaled[mask].flatten()
    inpainted_histogram_values = inpainted_scaled[mask].flatten()

    original_hist, _ = np.histogram(original_histogram_values, bins=80, range=(0, 80), density=True)
    inpainted_hist, _ = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80), density=True)
    # buf = create_histogram_plot(original_histogram_values, inpainted_histogram_values)

    # Calculate difference in means
    mean_diff = np.abs(np.mean(original_histogram_values) - np.mean(inpainted_histogram_values))
    
    # Calculate KL divergence
    kl_div = entropy(original_hist + 1e-10, inpainted_hist + 1e-10)  # Add small value to avoid division by zero

    # Store metrics with square key
    metrics = {
        "lpips": lpips_value.item(),
        "original_histogram_data": original_histogram_values,
        "inpainted_histogram_data": inpainted_histogram_values,
        "histogram_image": None,
        "mean_diff": mean_diff,
        "kl_div": kl_div,
    }

    update_inpainted_square(grid_key, square_key, metrics=metrics)

def calculate_grid_metrics(image, image_path, square, offset):
    dx, dy = (offset % 2) * square[2] // 2, (offset // 2) * square[2] // 2
    for i in range(dy, image.size[0], square[2]):
        for j in range(dx, image.size[1], square[2]):
            if i + square[2] > image.size[0] or j + square[2] > image.size[1]:
                continue
            calculate_square_metrics(image, image_path, (j, i, square[2]), offset)

def navigate_metrics(grid_key, square_key, direction):
    metrics_amount = len(st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'])
    metrics_index = st.session_state['all_inpainted_square_images'][grid_key][square_key]['index']
    st.session_state['all_inpainted_square_images'][grid_key][square_key]['index'] = (metrics_index + direction) % metrics_amount

def handle_metric_toggle_buttons(square, offset):
    grid_key, square_key = get_keys(square, offset)
    is_inpainted = is_square_inpainted(grid_key, square_key)
    if is_inpainted and len(st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics']) > 1:
        with st.sidebar:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button('Previous'):
                    navigate_metrics(grid_key, square_key, -1)
            with col2:
                if st.button('Next'):
                    navigate_metrics(grid_key, square_key, 1)
