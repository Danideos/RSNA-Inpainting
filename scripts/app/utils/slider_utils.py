# utils/slider_utils.py
import streamlit as st
from app.thresholding import ThresholdingPipeline
from app.utils.general_utils import get_current_index
from app.utils.metric_utils import calculate_grid_metrics


def get_square_and_mask(square_length, x_index, y_index, offset_option):
    offset_map = {0: (0, 0), 1: (square_length // 2, 0), 2: (0, square_length // 2), 3: (square_length // 2, square_length // 2)}
    dx, dy = offset_map[offset_option]

    selected_square_coords = (x_index * square_length + dx, y_index * square_length + dy)
    square = (selected_square_coords[0], selected_square_coords[1], square_length)

    mask_index = (x_index % 3 + 3 * (y_index % 3)) * 4 + offset_option
    mask = st.session_state['masks'][square_length][mask_index]

    return square, mask

def get_slider_parameters(square_lengths, img_size, image, image_path, middle):
    with st.sidebar:
        with st.expander("Slider Parameters"):
            square_length = st.selectbox('Select Square Size:', square_lengths)
            offset_option = st.slider('Select an Offset:', 0, 3, 0)

            grid_dim = (img_size // square_length)

            img_index = st.slide('Select Image Index:', 0, 100, st.session_state.image_index)
            x_index = st.slider('Select X Index:', 0, grid_dim - 1, st.session_state.x_index)
            y_index = st.slider('Select Y Index:', 0, grid_dim - 1, st.session_state.y_index)

        with st.expander("Inpainting Parameters"):
            start_denoise_step = st.slider('Select Start Denoise Step:', 1, 100, 100)
            resampling_steps = st.slider('Select Resampling Steps:', 1, 10, 1)
            jump_length = st.slider('Select Jump Length:', 1, 10, 1)
            inpaint_parameters = (start_denoise_step, resampling_steps, jump_length)

        with st.expander("Thresholding Parameters"):
            threshold_percent = st.number_input('Select Original Boundary Percent', 0.0, 100.0, 97.5, step=0.1)
            difference_percent = st.number_input('Select Difference Percent', -100.0, 100.0, -100.0, step=0.1)
            valid_square_percent = st.number_input('Select Valid Square Percent', 0.0, 100.0, 75.0, step=0.1)
            std_dev_diff_amount = st.number_input('Select Standard Deviation Diff Amount', -20.0, 20.0, -20.0, step=0.1)
            emd = st.number_input('Select EMD', 0.0, 10000.0, 0.0, step=1.0)
            mse = st.number_input('Select MSE', -1000.0, 1000.0, 0.0, step=1.0)
            pixel_dist = st.number_input('Select Pixel Distance', 0, 80, 8, step=1)
            pixel_exceed_count = st.number_input('Select Pixel Exceed Count', 0, 250, 0, step=1)
            pre_boundary_count = st.slider('Select Pre-Boundary Count', 0, 80, 5, step=1)
            ThresholdingPipeline.change_threshold_params(
                threshold_percent=threshold_percent,
                difference_percent=difference_percent,
                valid_square_percent=valid_square_percent,
                pre_boundary_count=pre_boundary_count,
                std_dev_diff_amount=std_dev_diff_amount,
                emd=emd,
                mse=mse,
                pixel_dist=pixel_dist,
                pixel_exceed_count=pixel_exceed_count
            )
            
            if st.button('Recalculate Thresholds'):
                dx, dy = offset_option % 2 * square_length // 2, offset_option // 2 * square_length // 2
                square = (x_index * square_length + dx, y_index * square_length + dy, square_length)
                index = get_current_index(square, offset_option)
                ThresholdingPipeline.calculate_grid_thresholds(image, square_length, offset_option, index)
            if st.button('Recalculate Metrics'):
                dx, dy = offset_option % 2 * square_length // 2, offset_option // 2 * square_length // 2
                square = (x_index * square_length + dx, y_index * square_length + dy, square_length)
                index = get_current_index(square, offset_option)
                calculate_grid_metrics(image, image_path, square, offset_option, index)
                ThresholdingPipeline.calculate_grid_thresholds(image, square_length, offset_option, index)
            
    return square_length, offset_option, x_index, y_index, img_index, inpaint_parameters
