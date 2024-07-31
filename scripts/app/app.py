import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False

from app.loader import Loader
from app.show_images import show_image
from app.utils.slider_utils import get_slider_parameters, get_square_and_mask
from app.utils.state_utils import initialize_states, handle_visibility_toggle_buttons
from app.utils.metric_utils import handle_metric_toggle_buttons
from app.utils.inpaint_utils import handle_inpaint_toggle_buttons


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def display_image_selector():
    # Slit streamlit into parts
    left, middle, right = st.columns([1, 2, 2])

    img_size = 256
    square_lengths = [48, 32, 16, 8]
    # Initialize streamlit states
    if not st.session_state.is_initialized:
        initialize_states(square_lengths=square_lengths, img_size=img_size)
        Loader.load_config()
        st.session_state.is_initialized = True

    loader = Loader(image_size=img_size) 
    with middle:
        st.title('Inpainting Thresholding Tool')
        image, image_path = loader.load_image_from_input()

    # Handle input from slider params
    square_size, offset_option, x_index, y_index, inpaint_parameters = get_slider_parameters(square_lengths, img_size, middle)
    square, grid_mask = get_square_and_mask(square_size, x_index, y_index, offset_option)  

    # Handle toggle buttons
    handle_visibility_toggle_buttons()
    handle_inpaint_toggle_buttons(image_path, image, square, grid_mask, img_size, inpaint_parameters, offset_option)
    handle_metric_toggle_buttons(square, offset_option)
    
    # Show the image
    show_image(image, square, grid_mask, offset_option, middle, right)

if __name__ == "__main__":
    display_image_selector()
