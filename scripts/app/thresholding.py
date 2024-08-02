from app.utils.general_utils import apply_func_to_grid, get_keys
from app.utils.state_utils import update_inpainted_square

import streamlit as st
import numpy as np


class ThresholdingPipeline:
    threshold_percent = 97
    difference_percent = 10
    valid_square_percent = 25
    pre_boundary_count = 5

    @staticmethod
    def calculate_grid_thresholds(img_size, square_length, offset, index=None):
        apply_func_to_grid(square_length, offset, img_size, ThresholdingPipeline.calculate_square_threshold, square_length, offset, index)

    @staticmethod
    def calculate_square_threshold(x, y, square_length, offset, index=None):
        grid_key, square_key = get_keys((x, y, square_length), offset)
        threshold = {
            'is_beyond_threshold': False,
            'difference_percent': None
        }

        metrics_index = -1 if index is None else index
        metrics = st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'][metrics_index]
        original_histogram_values = metrics['original_histogram_data']
        inpainted_histogram_values = metrics['inpainted_histogram_data']
        original_hist, _ = np.histogram(original_histogram_values, bins=80, range=(0, 80))
        inpainted_hist, _ = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80))

        if not ThresholdingPipeline._is_valid_square(original_histogram_values, square_length): 
            update_inpainted_square(grid_key, square_key, threshold=threshold)
            return 

        boundary_index = ThresholdingPipeline._get_boundary_index(inpainted_hist)
        total_difference = ThresholdingPipeline._get_total_difference(boundary_index, original_hist, inpainted_hist)
        is_beyond_threshold = total_difference > (ThresholdingPipeline.difference_percent / 100 * np.sum(original_hist))
        
        threshold = {
            'is_beyond_threshold': is_beyond_threshold,
            'difference_percent': total_difference / np.sum(original_hist) * 100
        }
        print(x, y, threshold, index)
        update_inpainted_square(grid_key, square_key, threshold=threshold, index=index)
  
    @staticmethod
    def calculate_all_thresholds():
        pass
    
    @staticmethod
    def _is_valid_square(original_histogram_values, square_length):
        square_area = square_length ** 2
        is_big_enough = len(original_histogram_values) / square_area > ThresholdingPipeline.valid_square_percent / 100
        return is_big_enough
    
    @staticmethod
    def _get_boundary_index(inpainted_hist):
        cumulative_counts = np.cumsum(inpainted_hist)
        total_counts = cumulative_counts[-1]
        boundary_index = np.searchsorted(cumulative_counts, ThresholdingPipeline.threshold_percent / 100 * total_counts)
        return boundary_index

    def _get_total_difference(boundary_index, original_hist, inpainted_hist):
        total_difference = 0
        # Calculate the total difference beyond the boundary 
        start_index = boundary_index - ThresholdingPipeline.pre_boundary_count
        for i in range(start_index, 80):
            original_count = original_hist[i]
            inpainted_count = inpainted_hist[i]

            multiplier = max(1, (i - start_index + 1) * (1 / ThresholdingPipeline.pre_boundary_count))
            total_difference += multiplier * (original_count - inpainted_count)
        return total_difference
    
    @classmethod
    def change_threshold_params(cls, threshold_percent=None, difference_percent=None, valid_square_percent=None, pre_boundary_count=None):
        if threshold_percent:
            cls.threshold_percent = threshold_percent
        if difference_percent:
            cls.difference_percent = difference_percent
        if valid_square_percent:
            cls.valid_square_percent = valid_square_percent
        if pre_boundary_count:
            cls.pre_boundary_count = pre_boundary_count