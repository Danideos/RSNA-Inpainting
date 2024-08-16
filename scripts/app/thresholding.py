from app.utils.general_utils import apply_func_to_grid, get_keys
from app.utils.state_utils import update_inpainted_square

import streamlit as st
import numpy as np
from joblib import Parallel, delayed


class ThresholdingPipeline:
    threshold_percent = 97.5
    difference_percent = 20.0
    valid_square_percent = 75.0
    pre_boundary_count = 0
    std_dev_diff_amount = 3.0
    mse = 0.0
    emd = 50.0
    pixel_dist = 10
    pixel_exceed_count = 32

    @staticmethod
    def calculate_series_thresholds(series, square_lengths, index):
        def process_image(img_index):
            for square_length in square_lengths:
                for offset in range(3, 4):  # Adjust range as needed
                    ThresholdingPipeline.calculate_grid_thresholds(
                        series[img_index],
                        square_length,
                        offset,
                        img_index,
                        index
                    )
        
        # Parallel processing
        Parallel(n_jobs=1)(delayed(process_image)(img_index) for img_index in range(len(series)))

    @staticmethod
    def calculate_grid_thresholds(image, square_length, offset, img_index, index=None):
        apply_func_to_grid(square_length, offset, image.size[0], ThresholdingPipeline.calculate_square_threshold, image, square_length, offset, img_index, index)

    @staticmethod
    def calculate_square_threshold(x, y, image, square_length, offset, img_index, index=None):
        grid_key, square_key = get_keys((x, y, square_length), offset)
        threshold = {
            'is_beyond_threshold': False,
            'difference_percent': None
        }

        metrics_index = -1 if index is None else index
        metrics = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'][metrics_index]
        if not metrics:
            update_inpainted_square(img_index, grid_key, square_key, threshold=threshold, index=index)
            return
    
        original_histogram_values = metrics['original_histogram_data']
        inpainted_histogram_values = metrics['inpainted_histogram_data']
        original_hist, _ = np.histogram(original_histogram_values, bins=80, range=(0, 80))
        inpainted_hist, _ = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80))

        if not ThresholdingPipeline._is_valid_square(original_hist, square_length): 
            update_inpainted_square(img_index, grid_key, square_key, threshold=threshold, index=index)
            return 

        boundary_index = ThresholdingPipeline._get_boundary_index(inpainted_hist)
        total_difference = ThresholdingPipeline._get_total_difference(boundary_index, original_hist, inpainted_hist)
        pixel_exceed_count = np.sum(np.abs(original_histogram_values - inpainted_histogram_values) > ThresholdingPipeline.pixel_dist)

        is_deviant = (metrics['std_dev_diff'] >= ThresholdingPipeline.std_dev_diff_amount 
                      and metrics['emd'] >= ThresholdingPipeline.emd 
                      and metrics['mse'] >= ThresholdingPipeline.mse 
                      and pixel_exceed_count >= ThresholdingPipeline.pixel_exceed_count
                      and total_difference > ThresholdingPipeline.difference_percent / 100 * np.sum(original_hist))

        threshold = {
            'is_beyond_threshold': is_deviant,
            'difference_percent': total_difference / np.sum(original_hist) * 100
        }
        print('Threshold:', threshold)
        update_inpainted_square(img_index, grid_key, square_key, threshold=threshold, index=index)

    @staticmethod
    def _is_valid_square(original_hist, square_length):
        square_area = square_length ** 2
        is_big_enough = np.sum(original_hist) / square_area > ThresholdingPipeline.valid_square_percent / 100
        return is_big_enough
    
    @staticmethod
    def _get_boundary_index(inpainted_hist):
        cumulative_counts = np.cumsum(inpainted_hist)
        total_counts = cumulative_counts[-1]
        boundary_index = np.searchsorted(cumulative_counts, ThresholdingPipeline.threshold_percent / 100 * total_counts)
        return boundary_index

    @staticmethod
    def _get_total_difference(boundary_index, original_hist, inpainted_hist):
        total_difference = 0
        # Calculate the total difference beyond the boundary 
        start_index = boundary_index - ThresholdingPipeline.pre_boundary_count
        for i in range(start_index, 80):
            original_count = original_hist[i]
            inpainted_count = inpainted_hist[i]

            multiplier = min(1, (i - start_index + 1) * (1 / (ThresholdingPipeline.pre_boundary_count + 0.0001)))
            total_difference += multiplier * (original_count - inpainted_count)
        return total_difference
    
    @classmethod
    def change_threshold_params(cls, threshold_percent=None, difference_percent=None, valid_square_percent=None, pre_boundary_count=None, std_dev_diff_amount=None, emd=None, mse=mse, pixel_dist=None, pixel_exceed_count=None):
        if threshold_percent is not None:
            cls.threshold_percent = threshold_percent
        if difference_percent is not None:
            cls.difference_percent = difference_percent
        if valid_square_percent is not None:
            cls.valid_square_percent = valid_square_percent
        if pre_boundary_count is not None:
            cls.pre_boundary_count = pre_boundary_count
        if std_dev_diff_amount is not None:
            cls.std_dev_diff_amount = std_dev_diff_amount
        if emd is not None:
            cls.emd = emd
        if mse is not None:
            cls.mse = mse
        if pixel_dist is not None:
            cls.pixel_dist = pixel_dist
        if pixel_exceed_count is not None:
            cls.pixel_exceed_count = pixel_exceed_count
