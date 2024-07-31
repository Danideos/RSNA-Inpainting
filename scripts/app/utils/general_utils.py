import streamlit as st
import numpy as np
import os


def get_contour_path(image_path):
    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name
    return contour_path

def get_keys(square, offset):
    x, y, square_length = square
    grid_key = (square_length, offset)
    square_key = (x, y)
    return (grid_key, square_key)

def get_inpainted_square_index(grid_key, square_key):
    return st.session_state['all_inpainted_square_images'][grid_key][square_key]['index']

def is_square_inpainted(grid_key, square_key):
    return square_key in st.session_state['all_inpainted_square_images'][grid_key] and st.session_state['all_inpainted_square_images'][grid_key][square_key]['index'] is not None

def ensure_3_channels(original_square, inpainted_square):
    original_array = np.asarray(original_square)
    inpainted_array = np.asarray(inpainted_square)

    if original_array.ndim == 2:  # Grayscale image
        original_array = np.stack([original_array]*3, axis=-1)
    if inpainted_array.ndim == 2:  # Grayscale image
        inpainted_array = np.stack([inpainted_array]*3, axis=-1)
    return original_array, inpainted_array
