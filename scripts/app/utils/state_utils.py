# utils/state_utils.py
import streamlit as st

def check_states(st):
    if 'show_inpainted_square' not in st.session_state:
        st.session_state['show_inpainted_square'] = False
    if 'metrics_index' not in st.session_state:
        st.session_state['metrics_index'] = {}
    if 'all_inpainted_square_images' not in st.session_state:
        st.session_state['all_inpainted_square_images'] = {}

def handle_visibility_toggle_buttons(st, middle):
    with middle:
        if 'show_selection' not in st.session_state:
            st.session_state['show_selection'] = True
        if 'show_correct_grid' not in st.session_state:
            st.session_state['show_correct_grid'] = False

        with st.expander("Visibility Options"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Toggle Selection'):
                    st.session_state['show_selection'] = not st.session_state['show_selection']

            with col2:
                if st.button('Toggle Correct Grid'):
                    st.session_state['show_correct_grid'] = not st.session_state['show_correct_grid']
