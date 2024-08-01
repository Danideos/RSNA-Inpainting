from inpaint import inpaint_square, inpaint_grid
from app.utils.metric_utils import calculate_square_metrics, calculate_grid_metrics
from app.utils.general_utils import get_keys
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
                grid_key, _ = get_keys(square, offset)
                st.rerun()
        
            if st.button('Toggle Inpainted Square'):
                st.session_state['show_inpainted_square'] = not st.session_state['show_inpainted_square']

            save_path = st.text_input('Enter the path to save configuration:')
            if st.button('Save Configuration'):
                if save_path:
                    Loader.save_config(grid_key, save_path)
                else:
                    st.error('Please enter a valid save path.')
