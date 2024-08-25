import os
import subprocess
import shutil
import pydicom
import csv
from tempfile import mkdtemp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Function to load SOPInstanceUIDs from CSV file
def load_sop_uids_from_csv(csv_file_path):
    sop_uids = set()
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sop_uids.add(row['SOPInstanceUID'])
    return sop_uids

# Function to check if the DICOM file's SOPInstanceUID is in the CSV file
def is_in_csv(dicom_data, sop_uids):
    sop_instance_uid = dicom_data.SOPInstanceUID
    return sop_instance_uid in sop_uids

# Function to process a single DICOM file
def process_dicom_file(dicom_file, output_dir, sop_uids):
    # Read the DICOM file to extract SOPInstanceUID and SeriesInstanceUID
    dicom_data = pydicom.dcmread(dicom_file, stop_before_pixels=True)
    
    # Check if the SOPInstanceUID is in the CSV file
    if not is_in_csv(dicom_data, sop_uids):
        return  # Skip processing if the SOPInstanceUID is not in the CSV file

    sop_instance_uid = dicom_data.SOPInstanceUID
    series_instance_uid = dicom_data.SeriesInstanceUID

    # Create a temporary directory
    temp_dir = mkdtemp()

    try:
        # Copy the DICOM file to the temporary directory
        temp_dicom_path = os.path.join(temp_dir, os.path.basename(dicom_file))
        shutil.copy(dicom_file, temp_dicom_path)

        # Run dcm2niix on the temporary directory
        output_filename = f"{sop_instance_uid}_{series_instance_uid}"
        subprocess.run(["dcm2niix", "-z", "y", "-f", output_filename, "-o", output_dir, temp_dir])

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

# Wrapper function for multiprocessing
def process_dicom_wrapper(args):
    dicom_file, output_dir, sop_uids = args
    process_dicom_file(dicom_file, output_dir, sop_uids)

# Function to walk through the directory and process each DICOM file using multiprocessing
def convert_dicom_to_nifti(input_dir, output_dir, num_workers, sop_uids):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))

    # Use multiprocessing to process the DICOM files in parallel
    with Pool(num_workers) as pool:
        # Wrap the arguments in tuples for the worker function
        args = [(dicom_file, output_dir, sop_uids) for dicom_file in dicom_files]

        # Use tqdm to display the progress bar
        list(tqdm(pool.imap(process_dicom_wrapper, args), total=len(dicom_files), desc="Processing DICOM files"))

# Define input and output directories
input_directory = "/research/Data/CQ500_Brain_Hemorrhage_Dataset/CT_DICOMs"
output_directory = "/research/Data/DK_RSNA_HM/cq500_prepared_data/nifti_all_v2"
csv_file_path = "/research/Data/CQ500_Brain_Hemorrhage_Dataset/Bounding_Box_Labels/3_Extrapolation_to_Selected_Series.csv"  # Replace with the path to your CSV file

# Load SOPInstanceUIDs from the CSV file
sop_uids = load_sop_uids_from_csv(csv_file_path)

# Number of workers to use for multiprocessing
num_workers = cpu_count()

# Run the conversion with multiprocessing and tqdm progress bar
convert_dicom_to_nifti(input_directory, output_directory, num_workers, sop_uids)

print("Conversion complete!")
