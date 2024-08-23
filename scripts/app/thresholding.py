from app.utils.general_utils import apply_func_to_grid, get_keys
from app.utils.state_utils import update_inpainted_square

import streamlit as st
import numpy as np
from joblib import Parallel, delayed


class ThresholdingPipeline:
    threshold_percent = {8: 97.5, 16: 97.5, 32: 97.5, 64: 97.5}
    difference_percent = {8: -100, 16: -100, 32: -100, 64: -100}
    valid_square_percent = {8: 75, 16: 50, 32: 40, 64: 30}
    pre_boundary_count = {8: 5, 16: 5, 32: 5, 64: 5}
    std_dev_diff_amount = {8: 2, 16: 2, 32: 1.5, 64: -20}
    mse = {8: 60, 16: 50, 32: 30, 64: 30}
    emd = {8: 4.5, 16: 4.0, 32: 4.0, 64: 3.0}
    pixel_dist = {8: 10, 16: 10, 32: 10, 64: 10}
    pixel_exceed_count = {8: 0, 16: 0, 32: 0, 64: 0}
    mean_diff = {8: -20, 16: -20, 32: -20, 64: -20}

    @staticmethod
    def calculate_series_thresholds(series, square_lengths, index=None):
        def process_image(img_index):
            for square_length in square_lengths:
                for offset in range(4):  # Adjust range as needed
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

        metric_index = len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['inpainted_square_image']) - 1 if index is None else index
        
        metrics = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'][metric_index]
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

        beyond_threshold, threshold_sum = ThresholdingPipeline._threshold_calc(metrics, square_length, original_hist, inpainted_hist, original_histogram_values, inpainted_histogram_values)
        is_deviant = beyond_threshold >= 6# and threshold_sum >= 6.5
        threshold = {
            'is_beyond_threshold': is_deviant,
            'difference_percent': f"{beyond_threshold}, {threshold_sum}"
        }
        update_inpainted_square(img_index, grid_key, square_key, threshold=threshold, index=index)

    @staticmethod
    def _threshold_calc(metrics, square_length, original_hist, inpainted_hist, original_histogram_values, inpainted_histogram_values):
        boundary_index = ThresholdingPipeline._get_boundary_index(inpainted_hist, square_length)
        total_difference = ThresholdingPipeline._get_total_difference(boundary_index, original_hist, inpainted_hist, square_length)
        pixel_exceed_count = np.sum(np.abs(original_histogram_values - inpainted_histogram_values) > ThresholdingPipeline.pixel_dist[square_length])

     
        beyond_threshold = (int(metrics['std_dev_diff'] >= ThresholdingPipeline.std_dev_diff_amount[square_length] )
                      + int(metrics['emd'] >= ThresholdingPipeline.emd[square_length] )
                      + int(metrics['mse'] >= ThresholdingPipeline.mse[square_length] )
                      + int(pixel_exceed_count >= ThresholdingPipeline.pixel_exceed_count[square_length])
                      + int(total_difference >= ThresholdingPipeline.difference_percent[square_length] / 100 * np.sum(original_hist))
                      + int(metrics['mean_diff'] >= -5))
        
        if (ThresholdingPipeline.std_dev_diff_amount[square_length] != 0
                and ThresholdingPipeline.emd[square_length] != 0
                and ThresholdingPipeline.mse[square_length] != 0
                and ThresholdingPipeline.difference_percent[square_length] != 0
                and ThresholdingPipeline.pixel_exceed_count[square_length] != 0):
            threshold_sum = (min(metrics['std_dev_diff'] / ThresholdingPipeline.std_dev_diff_amount[square_length], 2)
                        + min(metrics['emd'] / ThresholdingPipeline.emd[square_length], 2)
                        + min(metrics['mse'] / ThresholdingPipeline.mse[square_length], 2)
                        + min(pixel_exceed_count / ThresholdingPipeline.pixel_exceed_count[square_length], 2)
                        + min(total_difference / ThresholdingPipeline.difference_percent[square_length] / 100 * np.sum(original_hist), 2)
                        + min(metrics['mean_diff'] / 3, 2))
        else:
            threshold_sum = 100
        
        return beyond_threshold, threshold_sum

    @staticmethod
    def _is_valid_square(original_hist, square_length):
        square_area = square_length ** 2
        is_big_enough = np.sum(original_hist) / square_area > ThresholdingPipeline.valid_square_percent[square_length] / 100
        return is_big_enough
    
    @staticmethod
    def _get_boundary_index(inpainted_hist, square_length):
        cumulative_counts = np.cumsum(inpainted_hist)
        total_counts = cumulative_counts[-1]
        boundary_index = np.searchsorted(cumulative_counts, ThresholdingPipeline.threshold_percent[square_length] / 100 * total_counts)
        return boundary_index

    @staticmethod
    def _get_total_difference(boundary_index, original_hist, inpainted_hist, square_length):
        total_difference = 0
        # Calculate the total difference beyond the boundary 
        start_index = boundary_index - ThresholdingPipeline.pre_boundary_count[square_length]
        for i in range(start_index, 80):
            original_count = original_hist[i]
            inpainted_count = inpainted_hist[i]

            multiplier = min(1, (i - start_index + 1) * (1 / (ThresholdingPipeline.pre_boundary_count[square_length] + 0.0001)))
            total_difference += multiplier * (original_count - inpainted_count)
        return total_difference
    
    @classmethod
    def change_threshold_params(cls, square_length,
                                threshold_percent=None, 
                                difference_percent=None, 
                                valid_square_percent=None, 
                                pre_boundary_count=None, 
                                std_dev_diff_amount=None, 
                                emd=None, 
                                mse=None, 
                                pixel_dist=None, 
                                pixel_exceed_count=None,
                                mean_diff=None):
        if threshold_percent is not None:
            cls.threshold_percent[square_length] = threshold_percent
        if difference_percent is not None:
            cls.difference_percent[square_length] = difference_percent
        if valid_square_percent is not None:
            cls.valid_square_percent[square_length] = valid_square_percent
        if pre_boundary_count is not None:
            cls.pre_boundary_count[square_length] = pre_boundary_count
        if std_dev_diff_amount is not None:
            cls.std_dev_diff_amount[square_length] = std_dev_diff_amount
        if emd is not None:
            cls.emd[square_length] = emd
        if mse is not None:
            cls.mse[square_length] = mse
        if pixel_dist is not None:
            cls.pixel_dist[square_length] = pixel_dist
        if pixel_exceed_count is not None:
            cls.pixel_exceed_count[square_length] = pixel_exceed_count
        if mean_diff is not None:
            cls.mean_diff[square_length] = mean_diff
