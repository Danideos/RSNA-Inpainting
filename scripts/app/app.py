# %%
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)

import streamlit as st
from loader import Loader
from mask import create_masks
from show_images import show_image
from app.utils.slider_utils import get_slider_parameters, get_square_and_mask
from app.utils.state_utils import check_states, handle_visibility_toggle_buttons
from app.utils.metric_utils import handle_metric_toggle_buttons
from app.utils.inpaint_utils import handle_inpaint_toggle_buttons

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
st.set_page_config(layout="wide")


def display_image_selector():
    left, middle, right = st.columns([1, 1, 1])

    img_size = 256
    square_lengths = [48, 32, 16, 8]
     # Create masks for inpainting during initialization
    if 'masks' not in st.session_state:
        st.session_state['masks'] = create_masks(img_size, square_lengths)
    masks = st.session_state['masks']

    with middle:
        st.title('Inpainting Thresholding Tool')
        # Load the image
        loader = Loader(image_size=img_size)
        image, image_path = loader.load_image_from_path(st)

    # Handle input from slider params
    square_size, offset_option, x_index, y_index, inpaint_parameters = get_slider_parameters(st, square_lengths, img_size, middle)
    # Get the square and mask according to the slider params input
    square, mask = get_square_and_mask(square_size, x_index, y_index, offset_option, masks) 

    # Check if the states are initialized
    check_states(st)    

    # Handle toggle buttons
    handle_visibility_toggle_buttons(st, middle)
    handle_inpaint_toggle_buttons(st, image_path, image, square, mask, img_size, inpaint_parameters, masks, offset_option, middle)
    left_image_col, middle_image_col, right_image_col = st.columns([1, 1, 1])
    handle_metric_toggle_buttons(st, square, right_image_col)
    
    # Show the image
    show_image(st, image, square, mask, middle_image_col, right_image_col)

if __name__ == "__main__":
    display_image_selector()
