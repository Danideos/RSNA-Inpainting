import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Define paths
input_dir = "/research/Data/DK_RSNA_HM/BHSD"
ground_truth_dir = os.path.join(input_dir, "ground truths")
images_dir = os.path.join(input_dir, "images")
output_images_dir = "/research/Data/DK_RSNA_HM/BHSD/NIFTI_IMAGES"
output_annotations_dir = "/research/Data/DK_RSNA_HM/BHSD/NIFTI_ANNOTATIONS"

# Create the output directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Function to check if a slice has any annotation (not completely black)
def has_annotation(slice_data):
    return np.any(slice_data > 0)

# Save slice using nibabel
def save_slice(slice_data, file_name, output_path):
    # Ensure the slice has the correct shape [H, W] (2D)
    if slice_data.ndim == 2:  # If it's 2D, no need to modify
        slice_nii = nib.Nifti1Image(slice_data, affine=np.eye(4))  # Create NIfTI image with identity affine

    # Define the full path for saving the file
    output_file = os.path.join(output_path, file_name)

    # Save the slice using nibabel
    nib.save(slice_nii, output_file)
    print(f"Saved: {output_file}")

# Iterate over the ground truth files
for gt_file in tqdm(os.listdir(ground_truth_dir)):
    gt_path = os.path.join(ground_truth_dir, gt_file)
    img_file = os.path.join(images_dir, gt_file)  # Corresponding image file in 'images' directory
    
    # Load the ground truth and image files
    gt_img = nib.load(gt_path)
    img_img = nib.load(img_file)
    
    gt_data = gt_img.get_fdata()
    img_data = img_img.get_fdata()
    
    # Loop through each slice in the 3D volume
    for i in range(gt_data.shape[2]):
        gt_slice = gt_data[:, :, i]
        img_slice = img_data[:, :, i]
        
        # Check if the slice has any annotation
        if has_annotation(gt_slice):
            print(np.unique(gt_slice))
            
            slice_file_name = f"{gt_file}"
            
            # Save both the image slice and the annotation slice
            save_slice(img_slice, slice_file_name, output_images_dir)
            save_slice(gt_slice, slice_file_name, output_annotations_dir)

print("Slice extraction complete.")

