from inpaint import inpaint_square, inpaint_grid
from app.utils.metric_utils import calculate_square_metrics, calculate_grid_metrics
from app.loader import Loader

import streamlit as st


def handle_inpaint_toggle_buttons(image_path, image, square, mask, img_size, inpaint_parameters, offset):
    with st.sidebar:
        with st.expander("Inpainting Options"):
            if st.button('Inpaint Square'):
                inpaint_square(image_path, square, mask, img_size, offset, inpaint_parameters=inpaint_parameters)
                calculate_square_metrics(image, image_path, square, offset)
                st.rerun()
        
            if st.button('Inpaint Grid'):
                inpaint_grid(image_path, img_size, square, offset, inpaint_parameters=inpaint_parameters)
                calculate_grid_metrics(image, image_path, square, offset)
                Loader.save_config()
                st.rerun()
        
            if st.button('Toggle Inpainted Square'):
                st.session_state['show_inpainted_square'] = not st.session_state['show_inpainted_square']
