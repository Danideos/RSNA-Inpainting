# utils/slider_utils.py
import streamlit as st
from app.thresholding import ThresholdingPipeline
from app.utils.general_utils import get_current_index


def get_square_and_mask(square_length, x_index, y_index, offset_option):
    offset_map = {0: (0, 0), 1: (square_length // 2, 0), 2: (0, square_length // 2), 3: (square_length // 2, square_length // 2)}
    dx, dy = offset_map[offset_option]

    selected_square_coords = (x_index * square_length + dx, y_index * square_length + dy)
    square = (selected_square_coords[0], selected_square_coords[1], square_length)

    mask_index = (x_index % 3 + 3 * (y_index % 3)) * 4 + offset_option
    mask = st.session_state['masks'][square_length][mask_index]

    return square, mask

def get_slider_parameters(square_lengths, img_size, middle):
    with st.sidebar:
        with st.expander("Slider Parameters"):
            square_length = st.selectbox('Select Square Size:', square_lengths)
            offset_option = st.slider('Select an Offset:', 0, 3, 0)

            grid_dim = (img_size // square_length)

            x_index = st.slider('Select X Index:', 0, grid_dim - 1, 0)
            y_index = st.slider('Select Y Index:', 0, grid_dim - 1, 0)


        with st.expander("Inpainting Parameters"):
            start_denoise_step = st.slider('Select Start Denoise Step:', 1, 100, 100)
            resampling_steps = st.slider('Select Resampling Steps:', 1, 10, 1)
            jump_length = st.slider('Select Jump Length:', 1, 10, 1)
            inpaint_parameters = (start_denoise_step, resampling_steps, jump_length)

        with st.expander("Thresholding Parameters"):
            threshold_percent = st.number_input('Select Original Boundary Percent', 0.0, 100.0, 97., step=0.1)
            difference_percent = st.number_input('Select Difference Percent', 0.0, 100.0, 10.0, step=0.1)
            valid_square_percent = st.number_input('Select Valid Square Percent', 0.0, 100.0, 25.0, step=0.1)
            pre_boundary_count = st.slider('Select Pre-Boundary Count', 0, 80, 5, step=1)
            ThresholdingPipeline.change_threshold_params(
                threshold_percent=threshold_percent,
                difference_percent=difference_percent,
                valid_square_percent=valid_square_percent,
                pre_boundary_count=pre_boundary_count,
            )
            
            if st.button('Recalculate Thresholds'):
                square = (x_index * square_length, y_index * square_length, square_length)
                index = get_current_index(square, offset_option)
                print("current index for threshold change", index)
                ThresholdingPipeline.calculate_grid_thresholds(img_size, square_length, offset_option, index)
            
    return square_length, offset_option, x_index, y_index, inpaint_parameters
