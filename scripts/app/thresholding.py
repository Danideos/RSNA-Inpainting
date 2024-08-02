from app.utils.general_utils import apply_func_to_grid, get_keys
from app.utils.state_utils import update_inpainted_square

import streamlit as st
import numpy as np


class ThresholdingPipeline:
    @staticmethod
    def calculate_grid_thresholds(img_size, square, offset):
        apply_func_to_grid(square[2], offset, img_size, ThresholdingPipeline.calculate_square_threshold, square[2], offset)

    @staticmethod
    def calculate_square_threshold(x, y, square_length, offset):
        threshold_percent, difference_percent = 97, 20
        grid_key, square_key = get_keys((x, y, square_length), offset)
        metrics = st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'][-1]

        original_histogram_values = metrics['original_histogram_data']
        inpainted_histogram_values = metrics['inpainted_histogram_data']

        if len(original_histogram_values) == 0:
            return
        
        original_hist, _ = np.histogram(original_histogram_values, bins=80, range=(0, 80))
        inpainted_hist, _ = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80))

        # Calculate the boundary for 95% of intensity counts in inpainted histogram
        cumulative_counts = np.cumsum(inpainted_hist)
        total_counts = cumulative_counts[-1]
        boundary_index = np.searchsorted(cumulative_counts, threshold_percent / 100 * total_counts)

        total_difference = 0
        # Calculate the total difference beyond the boundary 
        for i in range(boundary_index - 5, 80):
            original_count = original_hist[i]
            inpainted_count = inpainted_hist[i]

            # Linear multiplier for the 5 indices before the boundary
            multiplier = max(1, (boundary_index - i) * 0.2)
            total_difference += multiplier * (original_count - inpainted_count)

        # Check if the total difference exceeds 10% of the total counts in original histogram
        total_original_counts = np.sum(original_hist)
        is_beyond_threshold = total_difference > (difference_percent / 100 * total_original_counts)
        update_inpainted_square(grid_key, square_key, threshold=is_beyond_threshold)
  

    def calculate_all_thresholds():
        pass