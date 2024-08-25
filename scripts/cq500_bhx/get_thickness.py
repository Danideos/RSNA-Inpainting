import nibabel as nib

# Load the NIfTI file
nii_file_path = '/research/Data/DK_RSNA_HM/cq500_prepared_data/test/CT 4cc sec 150cc D3D on-3/CT_4cc_sec_150cc_D3D_on-3_4cc_sec_150cc_D3D_on_0_5_Tilt_1.nii'
nii_img = nib.load(nii_file_path)

# Extract the header information
header = nii_img.header

# Extract the pixel dimensions (voxel size)
voxel_sizes = header.get_zooms()

# The slice thickness is typically the third element in voxel_sizes
slice_thickness = voxel_sizes[2]

print(f"Slice Thickness: {slice_thickness} mm")
