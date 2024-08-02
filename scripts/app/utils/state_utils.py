# utils/state_utils.py
import streamlit as st
from app.mask import create_masks


def initialize_states(square_lengths=[48, 32, 16, 8], img_size=256):
    if 'masks' not in st.session_state:
        st.session_state['masks'] = create_masks(img_size, square_lengths)
    if 'show_inpainted_square' not in st.session_state:
        st.session_state['show_inpainted_square'] = False
    if 'metrics_index' not in st.session_state:
        st.session_state['metrics_index'] = {}
    if 'all_inpainted_square_images' not in st.session_state:
        st.session_state['all_inpainted_square_images'] = {}
        st.session_state['inpainting_settings'] = {}
        for square_length in square_lengths:
            for offset in [0, 1, 2, 3]:
                grid_key = (square_length, offset)
                st.session_state['all_inpainted_square_images'][grid_key] = {}
                st.session_state['inpainting_settings'][grid_key] = {}    

def init_inpainted_square(grid_key, square_key):
    if square_key not in st.session_state['all_inpainted_square_images'][grid_key]:
        st.session_state['all_inpainted_square_images'][grid_key][square_key] = {}
    st.session_state['all_inpainted_square_images'][grid_key][square_key] = {'inpainted_square_image': [], 'metrics': [], 'index': None, 'parameters': [], 'threshold': False}

def update_inpainted_square(grid_key, square_key, inpainted_square=None, metrics=None, index=None, inpaint_parameters=None, threshold=None):
    if square_key not in st.session_state['all_inpainted_square_images'][grid_key]:
        init_inpainted_square(grid_key, square_key)
    if inpainted_square:
        st.session_state['all_inpainted_square_images'][grid_key][square_key]['inpainted_square_image'].append(inpainted_square)
    if metrics:
        st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'].append(metrics)
    if inpaint_parameters:
        st.session_state['all_inpainted_square_images'][grid_key][square_key]['parameters'].append(inpaint_parameters)
    if index:
        st.session_state['all_inpainted_square_images'][grid_key][square_key]['index'] = index
    elif len(st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics']) == 1 or len(st.session_state['all_inpainted_square_images'][grid_key][square_key]['inpainted_square_image']) == 1:
        st.session_state['all_inpainted_square_images'][grid_key][square_key]['index'] = 0
    if threshold:
        st.session_state['all_inpainted_square_images'][grid_key][square_key]['threshold'] = threshold

def handle_visibility_toggle_buttons():
    with st.sidebar:
        if 'show_selection' not in st.session_state:
            st.session_state['show_selection'] = True
        if 'show_correct_grid' not in st.session_state:
            st.session_state['show_correct_grid'] = False
        if 'show_thresholds' not in st.session_state:
            st.session_state['show_thresholds'] = False

        with st.expander("Visibility Options"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Toggle Selection'):
                    st.session_state['show_selection'] = not st.session_state['show_selection']

            with col2:
                if st.button('Toggle Correct Grid'):
                    st.session_state['show_correct_grid'] = not st.session_state['show_correct_grid']
            
            if st.button('Toggle Thresholds'):
                st.session_state['show_thresholds'] = not st.session_state['show_thresholds']

