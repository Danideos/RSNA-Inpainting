import os
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count

# Define threshold parameters (adjust these values based on your needs)
THRESHOLD_PARAMS = {
    8: {
        "mse": 100, 
        "emd": 7.5, 
        "std_dev_diff": 3,
        "mean_diff": 3,
        "pixel_dist": 10,           # Pixel distance threshold
        "pixel_exceed_count": 0,    # Count threshold for pixels that exceed the pixel_dist
        "total_difference": -100,
        "valid_square_percent": 75,
    },
    16: {
        "mse": 75, 
        "emd": 8, 
        "std_dev_diff": 2.5,
        "mean_diff": 3,
        "pixel_dist": 10,           
        "pixel_exceed_count": 0,    
        "total_difference": -100,
        "valid_square_percent": 50,
    },
    32: {
        "mse": 50, 
        "emd": 6.0, 
        "std_dev_diff": 2,
        "mean_diff": 3,
        "pixel_dist": 10,           
        "pixel_exceed_count": 0,    
        "total_difference": -100,
        "valid_square_percent": 40,
    },
    64: {
        "mse": 50, 
        "emd": 4.0, 
        "std_dev_diff": 1.5,
        "mean_diff": 3,
        "pixel_dist": 10,           
        "pixel_exceed_count": 0,    
        "total_difference": -100,
        "valid_square_percent": 30,
    }
}

# Function to load metrics from a pickle file
def load_metrics(metric_file_path):
    with open(metric_file_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

# Function to check if a square is valid based on the histogram
def is_valid_square(original_hist, square_length):
    square_area = square_length ** 2
    valid_square_percent = THRESHOLD_PARAMS[square_length]["valid_square_percent"]
    return np.sum(original_hist) / square_area > valid_square_percent / 100

# Function to calculate total difference beyond a boundary index
def get_total_difference(boundary_index, original_hist, inpainted_hist, square_length):
    total_difference = 0
    pre_boundary_count = 5  # Adjust as necessary
    start_index = boundary_index - pre_boundary_count

    for i in range(start_index, 80):
        original_count = original_hist[i]
        inpainted_count = inpainted_hist[i]
        multiplier = min(1, (i - start_index + 1) * (1 / (pre_boundary_count + 0.0001)))
        total_difference += multiplier * (original_count - inpainted_count)

    return total_difference

# Function to calculate if a square is beyond the threshold
def calculate_square_threshold(metrics, square_length):
    if metrics[0] is False:
        return False

    original_histogram_values = metrics[0]['original_histogram_data']
    inpainted_histogram_values = metrics[0]['inpainted_histogram_data']
    original_hist, _ = np.histogram(original_histogram_values, bins=80, range=(0, 80))
    inpainted_hist, _ = np.histogram(inpainted_histogram_values, bins=80, range=(0, 80))

    if not is_valid_square(original_hist, square_length):
        return False
    
    boundary_index = get_boundary_index(inpainted_hist, square_length)
    total_difference = get_total_difference(boundary_index, original_hist, inpainted_hist, square_length)
    
    # Calculate pixel exceed count based on pixel_dist
    pixel_dist = THRESHOLD_PARAMS[square_length]['pixel_dist']
    pixel_exceed_count = np.sum(np.abs(original_histogram_values - inpainted_histogram_values) > pixel_dist)

    mse = metrics[0].get('mse', 0)
    emd = metrics[0].get('emd', 0)
    std_dev_diff = metrics[0].get('std_dev_diff', 0)
    mean_diff = metrics[0].get('mean_diff', 0)

    threshold_params = THRESHOLD_PARAMS[square_length]
    
    is_beyond_threshold = (
        mse >= threshold_params["mse"] and
        emd >= threshold_params["emd"] and
        std_dev_diff >= threshold_params["std_dev_diff"] and
        mean_diff >= threshold_params["mean_diff"] and
        pixel_exceed_count >= threshold_params["pixel_exceed_count"] and
        total_difference >= threshold_params["total_difference"]
    )
    
    return is_beyond_threshold

# Function to get the boundary index for a given histogram
def get_boundary_index(inpainted_hist, square_length):
    cumulative_counts = np.cumsum(inpainted_hist)
    total_counts = cumulative_counts[-1]
    threshold_percent = 97.5  # Adjust as necessary
    return np.searchsorted(cumulative_counts, threshold_percent / 100 * total_counts)

# Function to process a single series and calculate TP/FP
def process_series(metric_file, series_image_paths, square_lengths):
    metrics = load_metrics(metric_file)
    
    results = {"FP": {8: 0, 16: 0, 32: 0, 64: 0}, "TP": {8: 0, 16: 0, 32: 0, 64: 0}}

    # Iterate through each image index
    for img_index, metric_data in metrics.items():
        # Iterate through each grid and square in the metrics
        for grid_key, grid_data in metric_data.items():
            for square_key, square_data in grid_data.items():
                square_length, offset = grid_key
                metrics = square_data
                is_beyond_threshold = calculate_square_threshold(metrics, square_length)
                
                # Determine if the current square is TP or FP based on the file name
                label = int(series_image_paths[img_index].split(".")[0].split("_")[-1])
                
                if is_beyond_threshold:
                    if label == 1:
                        results["TP"][square_length] += 1
                    else:
                        results["FP"][square_length] += 1
    
    return results

# Wrapper function for parallel processing
def process_series_wrapper(args):
    return process_series(*args)

# Main function to process all series
def main(metrics_dir, series_dir, square_lengths):
    metric_files = [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir)]
    
    # Prepare arguments for multiprocessing
    tasks = []
    for metric_file in metric_files:
        # Get series image paths
        series_id = os.path.basename(metric_file)
        series_path = os.path.join(series_dir, series_id, "bet_png")
        
        # Ensure that the directory exists and has images
        if not os.path.exists(series_path) or not os.listdir(series_path):
            print(f"Series path {series_path} does not exist or has no images. Skipping...")
            continue
        
        file_paths = sorted(os.listdir(series_path), key=lambda x: int(x.split("_")[0][1:]))
        series_image_paths = [os.path.join(series_path, file) for file in file_paths]

        tasks.append((metric_file, series_image_paths, square_lengths))
    
    # Use multiprocessing to process the tasks in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_series_wrapper, tasks)
    
    # Print results
    c = 0
    for result, metric_file in zip(results, metric_files):
        print(f"Results for {os.path.basename(metric_file)}:")
        print(f"True Positives (TP): {result['TP']}")
        print(f"False Positives (FP): {result['FP']}")
        print()
        detected = False
        for square_length in square_lengths:
            if square_length != 8:
                if result["FP"][square_length] >= 1 or result["TP"][square_length] >= 1:
                    detected = True
        c += int(detected)
    print(f"{c}/{len(results)}")

if __name__ == "__main__":
    metrics_dir = "/research/projects/DanielKaiser/RSNA_Inpainting/scripts/app/outputs/metrics_unhealthy"
    series_dir = "/research/Data/DK_RSNA_HM/series_stage_1_test/unhealthy/parameter_train"
    square_lengths = [32, 64]

    main(metrics_dir, series_dir, square_lengths)
