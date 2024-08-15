import streamlit as st
import numpy as np
import os


def get_contour_path(image_path):
    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name
    edge_path = "/".join(image_path.split("/")[:-2]) + "/edge_png/" + file_name
    return (contour_path, edge_path)

def get_keys(square, offset):
    x, y, square_length = square
    grid_key = (square_length, offset)
    square_key = (x, y)
    return (grid_key, square_key)

def get_current_index(img_index, square, offset):
    grid_key, square_key = get_keys(square, offset)
    if square_key in st.session_state['all_inpainted_square_images'][img_index][grid_key]:
        index = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['index']
        return index
    return None

def set_x_and_y(value, img_size, square_length):
    x, y = value['x'], value['y']
    x, y = x * (img_size / value['width']), y * (img_size / value['height'])
    st.session_state['x_index'], st.session_state['y_index'] = int(x // square_length), int(y // square_length)

def get_inpainted_square_index(grid_key, square_key):
    return st.session_state['all_inpainted_square_images'][grid_key][square_key]['index']

def is_square_inpainted(img_index, grid_key, square_key):
    return (square_key in st.session_state['all_inpainted_square_images'][img_index][grid_key] 
            and st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['index'] is not None)

def ensure_3_channels(original_square, inpainted_square):
    original_array = np.asarray(original_square)
    inpainted_array = np.asarray(inpainted_square)

    if original_array.ndim == 2:  # Grayscale image
        original_array = np.stack([original_array]*3, axis=-1)
    if inpainted_array.ndim == 2:  # Grayscale image
        inpainted_array = np.stack([inpainted_array]*3, axis=-1)
    return original_array, inpainted_array

def apply_func_to_grid(square_length, offset, img_size, function, *args, **kwargs):
    dx, dy = (offset % 2) * square_length // 2, (offset // 2) * square_length // 2
    for i in range(dy, img_size, square_length):
        for j in range(dx, img_size, square_length):
            if i + square_length > img_size or j + square_length > img_size:
                continue
            function(j, i, *args, **kwargs)
