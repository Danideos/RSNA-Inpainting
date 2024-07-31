# loader.py
from PIL import Image


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
        
    def load_image_from_path(self, st):
        unhealthy = "/research/Data/DK_RSNA_HM/series_stage_1_test/unhealthy/parameter_train/ID_0ac08adb64/bet_png/s27_ID_85117a302_1.png"
        # healthy = '/home/bje01/Documents/Data/prepared_data_test_series/healthy/train/ID_0a0f590be8/bet_png/s19_ID_833ff904f_0.png'
        image_path = st.text_input('Enter the path to your image:', unhealthy)
        image = None
        if image_path:
            image = self.load_image(image_path)
        return image, image_path    