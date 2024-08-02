# loader.py
from PIL import Image
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import os
from app.utils.state_utils import update_inpainted_square
import pickle 

load_dotenv()


class Loader:
    def __init__(self, image_size):
        self.image_size = image_size

    def load_image(self, path):
        try:
            image = Image.open(path)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
    def load_image_from_input(self):
        unhealthy = os.getenv('UNHEALTHY_PATH')
        image_path = st.text_input('Enter the path to your image:', unhealthy)
        image = None
        if image_path:
            image = self.load_image(image_path)
        return image, image_path    
    
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
        with open(config_save_path, 'wb') as f:
            pickle.dump(st.session_state, f)
        st.success('Session state saved successfully.')

    @staticmethod
    def load_session_state():
        config_path = os.getenv("INPAINTING_CONFIG_LOAD_PATH")
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                session_state_data = pickle.load(f)
            for key, value in session_state_data.items():
                st.session_state[key] = value
            st.success('Session state loaded successfully.')
        else:
            st.error('No configuration file found.')
