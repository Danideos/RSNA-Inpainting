# data_manager.py
from PIL import Image
import numpy as np
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import pickle 
import streamlit as st

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


class DataManager:
    def __init__(self):
        self.default_image = os.getenv('UNHEALTHY_PATH')

    def ask_for_image_path(self):
        image_path = st.text_input('Enter the path to your image:', self.default_image)
        return image_path

    @staticmethod
    @st.cache_data
    def load_image(path, image_size=256):
        try:
            image = Image.open(path)
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def load_series(series_path, image_size=256):
        series = []
        series_image_paths = []
        for file in os.listdir(series_path):
            if file.endswith(".png"):
                image = Image.open(os.path.join(series_path, file))
                image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                series.append(image)
                series_image_paths.append(os.path.join(series_path, file))
        return series, series_image_paths

    
    @staticmethod
    def load_contour_array(contour_path):
        contour_image = Image.open(contour_path).convert('L')  # Convert to grayscale
        contour_image = contour_image.resize((256, 256), Image.NEAREST)  # Use NEAREST to keep values 0 and 255
        contour_array = np.array(contour_image)
        contour_array = (contour_array > 127).astype(np.uint8) * 255
        return contour_array
    
    @staticmethod
    def save_session_state(save_name):
        config_save_dir = os.getenv("INPAINTING_CONFIG_SAVE_DIR")
        config_file_name = save_name
        config_save_path = os.path.join(config_save_dir, config_file_name)
        state_dict = {key: value for key, value in st.session_state.items()}
        with open(config_save_path, 'wb') as file:
            pickle.dump(state_dict, file)

    @staticmethod
    def load_session_state(load_name):
        # config_path = load_name #os.getenv("INPAINTING_CONFIG_LOAD_PATH")

        config_save_dir = os.getenv("INPAINTING_CONFIG_SAVE_DIR")
        config_file_name = load_name
        config_path = os.path.join(config_save_dir, config_file_name)
        if os.path.exists(config_path):
            with open(config_path, 'rb') as file:
                state_dict = pickle.load(file)
                for key, value in state_dict.items():
                    st.session_state[key] = value

def handle_datamanagement_toggle_buttons():
    with st.expander("Data Management"):
        save_path = st.text_input('Enter the path to save session state:')
        if st.button('Save Session State'):
            if save_path:
                DataManager.save_session_state(save_path)
                st.success("Save successful")
            else:
                st.error('Please enter a valid save path.')

        load_path = st.text_input('Enter the path to load session state:')
        if st.button('Load Session State'):
            if load_path:
                DataManager.load_session_state(load_path)
                st.success("Load successfull")
            else:
                st.error('Please enter a valid load path.')
