# utils/slider_utils.py
import streamlit as st

def get_square_and_mask(square_length, x_index, y_index, offset_option, masks):
    offset_map = {0: (0, 0), 1: (square_length // 2, 0), 2: (0, square_length // 2), 3: (square_length // 2, square_length // 2)}
    dx, dy = offset_map[offset_option]

    selected_square_coords = (y_index * square_length + dy, x_index * square_length + dx)
    square = (selected_square_coords[0], selected_square_coords[1], square_length)

    mask_index = (x_index % 3 + 3 * (y_index % 3)) * 4 + offset_option
    mask = masks[square_length][mask_index]

    return square, mask


def get_slider_parameters(st, square_lengths, img_size, middle):
    with middle:
        with st.expander("Slider Parameters"):
            square_length = st.selectbox('Select Square Size:', square_lengths)
            offset_option = st.slider('Select an Offset:', 0, 3, 0)

            grid_dim = (img_size // square_length)

            x_index = st.slider('Select X Index:', 0, grid_dim - 1, 0)
            y_index = st.slider('Select Y Index:', 0, grid_dim - 1, 0)

        with st.expander("Inpainting Parameters"):
            start_denoise_step = st.slider('Select Start Denoise Step:', 0, 99, 99)
            resampling_steps = st.slider('Select Resampling Steps:', 1, 10, 1)
            jump_length = st.slider('Select Jump Length:', 1, 10, 1)
            inpaint_parameters = (start_denoise_step, resampling_steps, jump_length)
    return square_length, offset_option, x_index, y_index, inpaint_parameters
