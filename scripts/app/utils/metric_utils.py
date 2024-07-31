# utils/metric_utils.py
import lpips
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from scipy.stats import entropy
import os

lpips_model = lpips.LPIPS(net='alex')

def resize_and_pad(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BICUBIC))

    # Pad the image to make it square if necessary
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded_image = np.pad(resized_image, ((pad_h, target_size - new_h - pad_h), (pad_w, target_size - new_w - pad_w), (0, 0)), mode='constant', constant_values=0)
    return padded_image

def calculate_square_metrics(st, image, image_path, square):
    inpainted_square, inpainted_x, inpainted_y = st.session_state['inpainted_square_image']
    original_square = image.crop((inpainted_y, inpainted_x, inpainted_y + square[2], inpainted_x + square[2]))

    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name

    # Load and resize the contour image
    contour_image = Image.open(contour_path).convert('L')  # Convert to grayscale
    contour_image = contour_image.resize((256, 256), Image.NEAREST)  # Use NEAREST to keep values 0 and 255
    contour_array = np.array(contour_image)
    contour_array = (contour_array > 127).astype(np.uint8) * 255

    # Extract the square region from the contour image
    contour_square = contour_array[inpainted_x:inpainted_x + square[2], inpainted_y:inpainted_y + square[2]]

    # Ensure images have channels
    original_array = np.array(original_square)
    inpainted_array = np.array(inpainted_square)

    if original_array.ndim == 2:  # Grayscale image
        original_array = np.stack([original_array]*3, axis=-1)
    if inpainted_array.ndim == 2:  # Grayscale image
        inpainted_array = np.stack([inpainted_array]*3, axis=-1)

    # Calculate LPIPS
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

    # Apply histogram calculation only to the white pixels in the contour mask
    mask = (contour_square == 255)
    original_scaled = (original_array.astype(np.float32) / 255.0) * 80
    inpainted_scaled = (inpainted_array.astype(np.float32) / 255.0) * 80

    original_histogram_values = original_scaled[mask].flatten()
    inpainted_histogram_values = inpainted_scaled[mask].flatten()

    # Plot overlapping histograms
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(original_histogram_values, bins=80, range=(0, 80), alpha=0.5, label='Original')
    ax.hist(inpainted_histogram_values, bins=80, range=(0, 80), alpha=0.5, label='Inpainted')
    ax.set_xlabel('Value', size=14)
    ax.set_ylabel('Count', size=14)
    ax.set_title('Overlapping Histogram')
    ax.legend(loc='upper right')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Calculate difference in means
    mean_diff = np.abs(np.mean(original_histogram_values) - np.mean(inpainted_histogram_values))
    
    # Calculate KL divergence
    original_hist, _ = np.histogram(original_histogram_values, bins=80, range=(0, 80), density=True)
    inpainted_hist, _ = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80), density=True)
    kl_div = entropy(original_hist + 1e-10, inpainted_hist + 1e-10)  # Add small value to avoid division by zero

    # Store metrics with coordinates of the square and size
    metrics = {
        "lpips": lpips_value.item(),
        "histogram_image": buf,
        "mean_diff": mean_diff,
        "kl_div": kl_div,
    }

    if square not in st.session_state:
        st.session_state[square] = []
    st.session_state[square].append(metrics)

def calculate_grid_metrics(st):
    pass

def navigate_metrics(st, key, direction):
    if key in st.session_state and len(st.session_state[key]) > 1:
        if 'metrics_index' not in st.session_state:
            st.session_state.metrics_index = {}
        if key not in st.session_state.metrics_index:
            st.session_state.metrics_index[key] = 0
        st.session_state.metrics_index[key] = (st.session_state.metrics_index[key] + direction) % len(st.session_state[key])
        return st.session_state[key][st.session_state.metrics_index[key]]
    return None

def handle_metric_toggle_buttons(st, square, right_col):

    key = ((square[0], square[1]), square[2])
    if key in st.session_state and len(st.session_state[key]) > 1:
        with right_col:
            # Navigation buttons for metrics
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button('Previous'):
                    metrics = navigate_metrics(st, key, -1)
            with col2:
                if st.button('Next'):
                    metrics = navigate_metrics(st, key, 1)
