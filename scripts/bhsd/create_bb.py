import os
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import multiprocessing as mp

# Define the label dictionary
label_dict = {
    "0": "Background",
    "1": "Epidural",
    "2": "Intraparenchymal",
    "3": "Intraventricular",
    "4": "Subarachnoid",
    "5": "Subdural"
}

# Define colors for different label types
label_colors = {
    "1": 'red',            # Epidural
    "2": 'green',          # Intraparenchymal
    "3": 'blue',           # Intraventricular
    "4": 'yellow',         # Subarachnoid
    "5": 'magenta'         # Subdural
}

# Define paths
input_dir = "/research/Data/DK_RSNA_HM/BHSD/NIFTI_ANNOTATIONS"
images_dir = "/research/Data/DK_RSNA_HM/BHSD/NIFTI_IMAGES"
output_csv = "/research/Data/DK_RSNA_HM/BHSD/bounding_boxes.csv"

# Dataframe to store bounding box information
df = pd.DataFrame(columns=["id", "bounding_box", "labeltype"])

def apply_window(img, center, width):
    '''
    Applies the windowing technique to the given image numpy array.
    '''
    min_value = center - width // 2
    max_value = center + width // 2
    
    img = np.clip(img, min_value, max_value)
    img = (img - min_value) / (max_value - min_value)
    
    return img

def do_boxes_overlap(box1, box2):
    # box1 and box2 are tuples in the format (min_col, min_row, max_col, max_row)
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])


def merge_boxes(box1, box2):
    # Merge the two overlapping boxes by selecting the smallest topleft and largest bottomright coordinates
    min_col = min(box1[0], box2[0])
    min_row = min(box1[1], box2[1])
    max_col = max(box1[2], box2[2])
    max_row = max(box1[3], box2[3])
    return (min_col, min_row, max_col, max_row)

# Merge overlapping boxes
def merge_all_boxes(bounding_boxes):
    merged_boxes = []
    
    while bounding_boxes:
        # Start with the first bounding box
        current_box = bounding_boxes.pop(0)

        min_col, min_row, max_col, max_row, area, label_type = current_box

        has_merged = False
        
        # Iterate through the rest of the bounding boxes
        i = 0
        while i < len(bounding_boxes):
            box = bounding_boxes[i]
            if do_boxes_overlap(current_box, box):
                # Determine the label of the merged box based on the larger area
                if box[4] > area:  # box[4] is the area of the current box in the list
                    label_type = box[5]  # box[5] is the label type of the current box in the list
                
                # Merge the boxes
                current_box = merge_boxes(current_box, box)
                area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])  # Update the area after merge
                current_box = (*current_box[:4], area, label_type)
                bounding_boxes.pop(i)  # Remove the merged box from the list
                has_merged = True
            else:
                i += 1
        
        if has_merged:
            # If a merge happened, check again from the beginning
            bounding_boxes.insert(0, current_box)
        else:
            # No more merges, add the current box to the final list
            merged_boxes.append((*current_box[:4], label_type))
    
    return merged_boxes

# Function to perform dilation, find bounding boxes, and merge overlapping boxes
def process_slice(slice_file):
    annotation_path = os.path.join(input_dir, slice_file)
    image_path = os.path.join(images_dir, slice_file)  # Corresponding image file in the images directory
    
    # Load the annotation and image slices
    annotation_img = nib.load(annotation_path)
    image_img = nib.load(image_path)
    
    annotation_data = annotation_img.get_fdata().squeeze()  # Annotation data (2D)
    image_data = image_img.get_fdata().squeeze()  # Image data (2D)
    
    unique_labels = np.unique(annotation_data)
    brain_windowed_img = apply_window(image_data, center=40, width=80)
    
    bounding_boxes = []  # Store bounding boxes as tuples (min_col, min_row, max_col, max_row, area, label)
    
    # Iterate over each unique label (ignoring background, i.e., label 0)
    for label_type in unique_labels:
        if label_type == 0:
            continue  # Skip background
        
        # Create a binary mask for the current label
        label_mask = (annotation_data == label_type).astype(np.uint8)
        
        # Perform dilation to connect components of the same label
        dilated_mask = ndimage.binary_dilation(label_mask, structure=np.ones((20, 20)))
        
        # Label connected components within the current label
        labeled_mask, num_labels = label(dilated_mask, return_num=True, connectivity=2)
        
        # Iterate through each labeled region
        for region in regionprops(labeled_mask):
            if region.area > 750:  # Ignore very small regions if needed
                # Bounding box coordinates
                min_row, min_col, max_row, max_col = region.bbox
                # Adjust coordinates to account for dilation (add 9 to min, subtract 9 from max)
                min_row = max(min_row + 8, 0)
                min_col = max(min_col + 8, 0)
                max_row = min(max_row - 8, annotation_data.shape[0])
                max_col = min(max_col - 8, annotation_data.shape[1])
                
                bounding_boxes.append((min_col, min_row, max_col, max_row, region.area, label_type))
    
    # Plot original bounding boxes before merging
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].imshow(brain_windowed_img, cmap='gray')
    # for box in bounding_boxes:
    #     min_col, min_row, max_col, max_row, _, label_type = box
    #     rect = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, linewidth=1, edgecolor=label_colors[str(int(label_type))], facecolor='none')
    #     ax[0].add_patch(rect)
    # ax[0].set_title("Original Bounding Boxes")
    merged_boxes = merge_all_boxes(bounding_boxes)
    
    # # Merge overlapping boxes
    
    # # Plot merged bounding boxes
    # ax[1].imshow(brain_windowed_img, cmap='gray')
    # for box in merged_boxes:
    #     min_col, min_row, max_col, max_row, label_type = box
    #     rect = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, linewidth=2, edgecolor=label_colors[str(int(label_type))], facecolor='none')
    #     ax[1].add_patch(rect)
    # ax[1].set_title("Merged Bounding Boxes")
    
    # # Show the plot
    # plt.show()
    
    # Return the merged bounding boxes for further processing
    results = []
    for box in merged_boxes:
        min_col, min_row, max_col, max_row, label_type = box
        label_type_str = str(int(label_type))
        
        # Add bounding box information to the results list
        bounding_box = f"{{topleft_x: {min_col}, topleft_y: {min_row}, bottomright_x: {max_col}, bottomright_y: {max_row}}}"
        parts = slice_file.split('_')
        id1 = parts[1]
        id2 = parts[3][:-7]
        
        # Construct the new filename with the format ID{id1}ID{id2}.png
        id = f"ID{id1}ID{id2}{parts[5][:-6]}png"
        results.append([id, bounding_box, label_dict[label_type_str]])
    
    return results

# Function to process slices in parallel
def process_slices_in_parallel():
    slice_files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
    
    # Use multiprocessing Pool to process slices in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_slice, slice_files), total=len(slice_files)))
    
    # Flatten the list of results and add to the dataframe
    for result in results:
        for row in result:
            df.loc[len(df)] = row

# Main function to run the processing
if __name__ == "__main__":
    process_slices_in_parallel()
    
    # Save bounding box information to CSV
    df.to_csv(output_csv, index=False)
    print(f"Bounding box information saved to {output_csv}")
