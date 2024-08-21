import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from app.utils.general_utils import is_square_inpainted, get_keys, set_x_and_y, get_current_index, apply_func_to_grid
from app.mask import overlay_mask

import matplotlib.pyplot as plt
from io import BytesIO
import base64


def show_image(series, series_image_paths, 
               square_lengths, image, image_path, 
               square, mask, offset, img_index, 
               _left_col, _middle_col, _right_col):
    grid_key, square_key = get_keys(square, offset)
    is_inpainted = is_square_inpainted(img_index, grid_key, square_key)
    index = get_current_index(img_index, square, offset)

    value = display_image(square, offset, image, image_path, mask, is_inpainted, grid_key, square_key, index, img_index, _middle_col)
    display_metrics(img_index, grid_key, square_key, index, is_inpainted, _right_col)
    display_series_statistics(series, series_image_paths, is_inpainted, square_lengths, image.size[0], _left_col)
    
    if value:
        set_x_and_y(value, image.size[0], square[2])
        value = None

@st.cache_data
def display_series_statistics(series, series_image_paths, is_inpainted_, square_lengths, img_size, _left_col):
    results = {"FP": [],
               "TP": []
               }
    for img_index in range(len(series)):
        for square_length in square_lengths:
            for offset in range(4):
                apply_func_to_grid(square_length, offset, img_size, get_threshold, series_image_paths, square_length, offset, img_index, results)
    
    # Display the FP and TP results in the left column
    with _left_col:
        st.write("True Positives (TP):")
        for tp in results["TP"]:
            st.write(tp)
        
        st.write("False Positives (FP):")
        for fp in results["FP"]:
            st.write(fp)


def get_threshold(x, y, series_image_paths, square_length, offset, img_index, results):
    square = (x, y, square_length)
    grid_key, square_key = get_keys(square, offset)
    index = get_current_index(img_index, square, offset)
    if index is None: return
    threshold = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['thresholds'][index]
    is_thresholded = threshold['is_beyond_threshold']
    if is_thresholded:
        if int(series_image_paths[img_index].split(".")[0].split("_")[-1]) == 1: 
            results["TP"].append(f"img index: {img_index}, key: {grid_key}, {square_key}")
        else:
            results["FP"].append(f"img index: {img_index}, key: {grid_key}, {square_key}")

def display_image(square, offset, image, image_path, mask, is_inpainted, grid_key, square_key, index, img_index, _middle_col):
    with _middle_col:
        x, y, square_length = square

        image_copy = cache_image(image)
        add_info = f'\nInpainting toggled off'
        if is_inpainted and st.session_state['show_inpainted_square']:
            inpainted_square_image = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['inpainted_square_image'][index]
            image_copy.paste(inpainted_square_image, (x, y))
            add_info = '\nInpainting toggled on'

        overlay_image = overlay_mask(image_copy, mask, square, offset, img_index)
        st.write(f'Square at X: {x // square_length}, Y: {y // square_length}, Index: {index},{add_info}')
        st.write(f'{image_path}')
        st.image(overlay_image, use_column_width=True)
        value = None #streamlit_image_coordinates(overlay_image, use_column_width=True)

    return value

@st.cache_data
def cache_image(image):
    return image.copy()

def display_metrics(img_index, grid_key, square_key, index, is_inpainted, _column):
    with _column:
        _, centre, _ = st.columns([1, 3, 1])
        with centre:    
            if not is_inpainted: 
                return
            metrics_amount = len(st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'])
            for i in range(metrics_amount):
                expanded = True if i == index else False
                with st.expander(f"Metrics for inpainted index {i}", expanded=expanded):
                    metrics = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['metrics'][i]
                    if not metrics: 
                        continue
                    threshold = st.session_state['all_inpainted_square_images'][img_index][grid_key][square_key]['thresholds'][i]
                    # Generate the histogram plot if it doesn't exist
                    if metrics['histogram_image'] is None:
                        metrics['histogram_image'] = generate_histogram_plot(metrics['original_histogram_data'], metrics['inpainted_histogram_data'])
                    # Metric stats
                    st.image(metrics['histogram_image'], caption="Histogram Comparison")
                    st.write(f"LPIPS: {metrics['lpips']}")
                    st.write(f"Mean Difference: {metrics['mean_diff']}")
                    st.write(f"EMD: {metrics['emd']}")
                    st.write(f"Standard Deviation: {metrics['std_dev_diff']}")
                    st.write(f"MSE: {metrics['mse']}")
                    
                    # Threshold stats
                    st.write(f"Threshold Difference Percent: {threshold['difference_percent']}")

def generate_histogram_plot(original_histogram_values, inpainted_histogram_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(original_histogram_values, bins=80, range=(0, 80), alpha=0.5, label='Original')
    ax.hist(inpainted_histogram_values, bins=80, range=(0, 80), alpha=0.5, label='Inpainted')
    ax.set_xlabel('Value', size=14)
    ax.set_ylabel('Count', size=14)
    ax.set_title('Overlapping Histogram')
    ax.legend(loc='upper right')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plt.close(fig)

    return buf