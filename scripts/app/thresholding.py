from app.utils.general_utils import apply_func_to_grid, get_keys
from app.utils.state_utils import update_inpainted_square

import streamlit as st
import numpy as np


class ThresholdingPipeline:
    @staticmethod
    def calculate_grid_threshold(img_size, square, offset):
        apply_func_to_grid(square[2], offset, img_size, ThresholdingPipeline.calculate_square_thresholds, square[2], offset)

    @staticmethod
    def calculate_square_thresholds(square, offset, threshold_percent=95, difference_percent=10):
        inpainted_x, inpainted_y, square_length = square
        grid_key, square_key = get_keys(square, offset)
        metrics = st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'][-1]

        original_hist = metrics['original_histogram_data']
        inpainted_hist = metrics['inpainted_histogram_data']

        # Calculate the boundary for 95% of intensity counts in inpainted histogram
        sorted_indices = np.argsort(inpainted_hist)
        cumulative_counts = np.cumsum(inpainted_hist[sorted_indices])
        total_counts = cumulative_counts[-1]
        boundary_index = np.searchsorted(cumulative_counts, threshold_percent / 100 * total_counts)

        boundary_intensity = sorted_indices[boundary_index]
        total_difference = 0

        # Calculate the total difference beyond the boundary with linearly decreasing multiplier
        for i in range(boundary_index - 5, len(sorted_indices)):
            intensity = sorted_indices[i]
            original_count = original_hist[intensity]
            inpainted_count = inpainted_hist[intensity]

            # Linear multiplier
            multiplier = max(1, i * 0.2)
            total_difference += multiplier * original_count - inpainted_count

        # Check if the total difference exceeds 10% of the total counts in original histogram
        total_original_counts = np.sum(original_hist)
        if total_difference > (difference_percent / 100 * total_original_counts):
            update_inpainted_square(grid_key, square_key, threshold=True)
        else:
            update_inpainted_square(grid_key, square_key, threshold=False)

    def calculate_all_thresholds():
        pass