import sys
import os
import importlib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False

from scripts.app.data_manager import DataManager, handle_datamanagement_toggle_buttons
from app.show_images import show_image
from app.utils.slider_utils import get_slider_parameters, get_square_and_mask
from app.utils.state_utils import initialize_states, handle_visibility_toggle_buttons
from app.utils.metric_utils import handle_metric_toggle_buttons
from app.utils.inpaint_utils import handle_inpaint_toggle_buttons
import app.utils.general_utils
import app.thresholding 
import app.utils.slider_utils   
import app.show_images
import scripts.app.data_manager

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def reload_modules():
    if st.button("Reload Module"):
        importlib.reload(app.thresholding)
        importlib.reload(app.Datamanager)
        importlib.reload(app.utils.slider_utils)
        importlib.reload(app.utils.state_utils)
        importlib.reload(app.utils.general_utils)
        importlib.reload(app.utils.inpaint_utils)
        importlib.reload(app.show_images)
        importlib.reload(app.mask)
        st.success("Module reloaded successfully")


def display_image_selector():
    # Slit streamlit into parts
    left, middle, right = st.columns([1, 2, 2])

    img_size = 256
    square_lengths = [48, 32, 16, 8]

    data_manager = DataManager(image_size=img_size) 
    with middle:
        st.title('Inpainting Thresholding Tool')
        reload_modules()
    
    image_path = data_manager.ask_for_image_path()
    image = data_manager.load_image(image_path)
    # series_path = ''
    # series = DataManager.load_series(series_path)

    # Initialize streamlit states
    initialize_states(square_lengths=square_lengths, img_size=img_size)

    # Handle input from slider params
    square_size, offset_option, x_index, y_index, inpaint_parameters = get_slider_parameters(square_lengths, img_size, image, image_path, middle)
    square, grid_mask = get_square_and_mask(square_size, x_index, y_index, offset_option)  

    # Handle toggle buttons
    handle_visibility_toggle_buttons()
    handle_inpaint_toggle_buttons(image_path, image, square, grid_mask, img_size, inpaint_parameters, offset_option)
    handle_metric_toggle_buttons(square, offset_option)
    handle_datamanagement_toggle_buttons()
    
    # Show the image
    show_image(image, square, grid_mask, offset_option, middle, right)


if __name__ == "__main__":
    display_image_selector()
