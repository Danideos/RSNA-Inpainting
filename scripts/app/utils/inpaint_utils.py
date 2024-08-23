from inpaint import inpaint_square, inpaint_grid, inpaint_series
from app.utils.metric_utils import calculate_square_metrics, calculate_grid_metrics, calculate_series_metrics
from app.utils.state_utils import reset_session_state
from scripts.app.data_manager import DataManager

import os

import streamlit as st
from thresholding import ThresholdingPipeline


def handle_inpaint_toggle_buttons(series, series_image_paths, square_lengths, square, mask, img_size, inpaint_parameters, offset, img_index):
    with st.sidebar:
        with st.expander("Inpainting Options"):
            image, image_path = series[img_index], series_image_paths[img_index]

            if st.button('Inpaint Square'):
                inpaint_square(image_path, square, mask, img_size, offset, img_index, inpaint_parameters=inpaint_parameters)
                calculate_square_metrics(image, image_path, square, offset, img_index)
                st.rerun()
        
            if st.button('Inpaint Grid'):
                inpaint_grid(image_path, img_size, square, offset, img_index, inpaint_parameters=inpaint_parameters)
                calculate_grid_metrics(image, image_path, square, offset, img_index) 
                ThresholdingPipeline.calculate_grid_thresholds(image, square[2], offset, img_index)
                st.rerun()

            if st.button('Inpaint Series'):
                inpaint_series(series, series_image_paths, square_lengths, img_index, img_size, inpaint_parameters=inpaint_parameters)
                calculate_series_metrics(series, series_image_paths, square_lengths)
                ThresholdingPipeline.calculate_series_thresholds(series, square_lengths)

            if st.button('Predict Directory'):
                directory = '/research/Data/DK_RSNA_HM/series_stage_1_test/healthy/parameter_train'
                for series_id in os.listdir(directory)[4:]:
                    # Cleanup statedict
                    reset_session_state(square_lengths, img_size, series_id)

                    series_path = os.path.join(directory, series_id)
                    series, series_image_paths = DataManager.load_series(series_path + "/bet_png")

                    # Series Inpainting Pipeline
                    inpaint_series(series, series_image_paths, square_lengths, img_index, img_size, inpaint_parameters=inpaint_parameters)
                    calculate_series_metrics(series, series_image_paths, square_lengths)
                    ThresholdingPipeline.calculate_series_thresholds(series, square_lengths)

                    # Save metrics
                    DataManager.save_metrics_state(series_id)
                    

        
            if st.button('Toggle Inpainted Square'):
                st.session_state['show_inpainted_square'] = not st.session_state['show_inpainted_square']

