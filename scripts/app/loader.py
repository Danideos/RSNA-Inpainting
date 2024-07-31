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
    def save_config(grid_key):
        config_data = {}
        for grid_key, squares in st.session_state['all_inpainted_square_images'].items():
            config_data[grid_key] = {}
            for square_key, data in squares.items():
                if data['inpainted_square_image']:
                    config_data[grid_key][square_key] = {
                        'inpainted_square_image': data['inpainted_square_image'][-1],
                        'metrics': data['metrics'][-1],
                        'inpainting_parameters': data['parameters'][-1]
                    }
        config_save_dir = os.getenv("CONFIG_SAVE_DIR")
        config_file_name = f'config.pkl'
        config_save_path = os.path.join(config_save_dir, config_file_name)
        with open(config_save_path, 'wb') as f:
            pickle.dump(config_data, f)
        st.success('Configuration saved successfully.')

    @staticmethod
    def load_config():
        config_path = os.getenv("CONFIG_LOAD_PATH")
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config_data = pickle.load(f)
            for grid_key, squares in config_data.items():
                if grid_key not in st.session_state['all_inpainted_square_images']:
                    st.session_state['all_inpainted_square_images'][grid_key] = {}
                for square_key, data in squares.items():
                    st.session_state['all_inpainted_square_images'][grid_key][square_key] = {
                        'inpainted_square_image': [],
                        'metrics': [],
                        'index': None,
                        'parameters': []
                    }
                    update_inpainted_square(
                        grid_key,
                        square_key,
                        inpainted_square=data['inpainted_square_image'],
                        metrics=data['metrics'],
                        inpaint_parameters=data['inpainting_parameters']
                    )
            st.success('Configuration loaded successfully.')
        else:
            st.error('No configuration file found.')
