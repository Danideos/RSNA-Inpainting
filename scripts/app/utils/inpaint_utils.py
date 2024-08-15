from inpaint import inpaint_square, inpaint_grid
from app.utils.metric_utils import calculate_square_metrics, calculate_grid_metrics
from app.utils.general_utils import get_keys
from scripts.app.data_manager import DataManager

import streamlit as st
from thresholding import ThresholdingPipeline


def handle_inpaint_toggle_buttons(image_path, image, square, mask, img_size, inpaint_parameters, offset, img_index):
    with st.sidebar:
        with st.expander("Inpainting Options"):
            grid_key, square_key = get_keys(square, offset)

            if st.button('Inpaint Square'):
                inpaint_square(image_path, square, mask, img_size, offset, img_index, inpaint_parameters=inpaint_parameters)
                calculate_square_metrics(image, image_path, square, offset, img_index)
                st.rerun()
        
            if st.button('Inpaint Grid'):
                inpaint_grid(image_path, img_size, square, offset, img_index, inpaint_parameters=inpaint_parameters)
                calculate_grid_metrics(image, image_path, square, offset, img_index) 
                ThresholdingPipeline.calculate_grid_thresholds(image, square[2], offset, img_index)
                st.rerun()
        
            if st.button('Toggle Inpainted Square'):
                st.session_state['show_inpainted_square'] = not st.session_state['show_inpainted_square']

