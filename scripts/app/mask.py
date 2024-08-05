from app.utils.general_utils import apply_func_to_grid, get_keys, get_current_index

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st


def create_masks(img_size, square_sizes):
    masks = {}

    for size in square_sizes:
        size = size 
        all_masks = []

        for i in range(3):
            for j in range(3):
                for dx in [0, size // 2]:
                    for dy in [0, size // 2]:
                        mask = np.zeros((img_size, img_size), dtype=np.uint8)
                        for x in range(i * size, img_size, 3 * size):
                            for y in range(j * size, img_size, 3 * size):
                                x_off = x + dx
                                y_off = y + dy
                                if x_off + size <= img_size and y_off + size <= img_size:
                                    mask[x_off:x_off + size, y_off:y_off + size] = 1
                        all_masks.append(mask)

        masks[size] = all_masks

    return masks

@st.cache_data
def create_grid_overlay(img_size, square_size, offset):
    overlay = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw the grid with offset
    dx, dy = (offset % 2) * square_size, offset // 2 * square_size
    grid_color = (128, 128, 128, 128)  # Light gray with some transparency
    for x in range(0, img_size, square_size):
        draw.line([(x, 0), (x, img_size)], fill=grid_color)
    for y in range(0, img_size, square_size):
        draw.line([(0, y), (img_size, y)], fill=grid_color)

    return overlay

@st.cache_data
def get_or_create_grid_overlay(img_size, square_size, offset):
    grid_key = (square_size, offset)
    if grid_key not in st.session_state['grid_overlays']:
        st.session_state['grid_overlays'][grid_key] = create_grid_overlay(img_size, square_size, offset)
    return st.session_state['grid_overlays'][grid_key]

def overlay_mask(image, mask, square, offset):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))

    if st.session_state['show_selection']:
        # Use the stored grid overlay
        grid_overlay = get_or_create_grid_overlay(image.size[0], square[2], offset)
        overlay = Image.alpha_composite(overlay, grid_overlay)
        draw = ImageDraw.Draw(overlay)

        x, y, size = square
        draw.rectangle([x, y, x + size - 1, y + size - 1], outline=(200, 69, 0, 96), width=1)  # Orange-red border

    if st.session_state['show_correct_grid']:
        mask_overlay = Image.fromarray((mask * 128).astype(np.uint8), mode='L').convert('RGBA')
        mask_overlay = mask_overlay.point(lambda p: p * 0.5)
        overlay = Image.alpha_composite(overlay, mask_overlay)

    if st.session_state['show_thresholds']:
        apply_func_to_grid(square[2], offset, image.size[0], overlay_thresholds, square[2], offset, overlay)

    combined = Image.alpha_composite(image, overlay)
    return combined

def overlay_thresholds(x, y, square_length, offset, overlay):
    square = (x, y, square_length)
    grid_key, square_key = get_keys(square, offset)
    index = get_current_index(square, offset)
    if index is None: return
    threshold = st.session_state['all_inpainted_square_images'][grid_key][square_key]['thresholds'][index]
    is_thresholded = threshold['is_beyond_threshold']
    if is_thresholded:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x + 1, y + 1, x + square_length - 2, y + square_length - 2], fill=(255, 0, 0, 64), width=1)  # Red border