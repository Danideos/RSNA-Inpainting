# utils/state_utils.py
import streamlit as st
from app.mask import create_masks

@st.cache_data
def initialize_states(square_lengths=[48, 32, 16, 8], img_size=256, series_id=None):
    if 'masks' not in st.session_state:
        st.session_state['masks'] = create_masks(img_size, square_lengths)
    if 'grid_overlays' not in st.session_state:
        st.session_state['grid_overlays'] = {}
    if 'show_inpainted_square' not in st.session_state:
        st.session_state['show_inpainted_square'] = False
    if 'metrics_index' not in st.session_state:
        st.session_state['metrics_index'] = {}
    if 'all_inpainted_square_images' not in st.session_state:
        st.session_state['all_inpainted_square_images'] = []
        for img_index in range(100):
            st.session_state['all_inpainted_square_images'].append({})
            for square_length in square_lengths:
                for offset in [0, 1, 2, 3]:
                    grid_key = (square_length, offset)
                    st.session_state['all_inpainted_square_images'][img_index][grid_key] = {}
    if 'x_index' not in st.session_state:
        st.session_state['x_index'] = 0
    if 'y_index' not in st.session_state:
        st.session_state['y_index'] = 0
    if 'img_index' not in st.session_state:
        st.session_state['img_index'] = 0

def reset_session_state(square_lengths=[48, 32, 16, 8], img_size=256, series_id=None):
    st.session_state.clear()
    initialize_states(square_lengths, img_size, series_id)
    if 'show_selection' not in st.session_state:
        st.session_state['show_selection'] = True
    if 'show_correct_grid' not in st.session_state:
        st.session_state['show_correct_grid'] = False
    if 'show_thresholds' not in st.session_state:
        st.session_state['show_thresholds'] = False

def init_inpainted_square(img_index, grid_key, square_key):
    if square_key not in st.session_state['all_inpainted_square_images'][img_index][grid_key]:
        st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key] = {}
    st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key] = {'inpainted_square_image': [], 'metrics': [], 'index': None, 'parameters': [], 'thresholds': []}

def update_inpainted_square(img_index, grid_key, square_key, inpainted_square=None, metrics=None, index=None, inpaint_parameters=None, threshold=None):
    if square_key not in st.session_state['all_inpainted_square_images'][img_index][grid_key]:
        init_inpainted_square(img_index, grid_key, square_key)
    if inpainted_square is not None:
        st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['inpainted_square_image'].append(inpainted_square)
    if metrics is not None:
        if index is None or len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics']) == 0:
            st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'].append(metrics)
        else:
            st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'][index] = metrics
    if inpaint_parameters is not None:
        st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['parameters'].append(inpaint_parameters)
    if index is not None:
        st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['index'] = index
    elif len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics']) == 1 or len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['inpainted_square_image']) == 1:
        st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['index'] = 0
    if threshold is not None:
        if index is None or len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['thresholds']) == 0:
            st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['thresholds'].append(threshold)
        else:
            st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['thresholds'][index] = threshold

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

