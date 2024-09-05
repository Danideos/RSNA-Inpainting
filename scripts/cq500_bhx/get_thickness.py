import nibabel as nib

# Load the NIfTI file
nii_file_path = ''
nii_img = nib.load(nii_file_path)

# Extract the header information
header = nii_img.header

# Extract the pixel dimensions (voxel size)
voxel_sizes = header.get_zooms()

# The slice thickness is typically the third element in voxel_sizes
slice_thickness = voxel_sizes[2]

print(f"Slice Thickness: {slice_thickness} mm")
