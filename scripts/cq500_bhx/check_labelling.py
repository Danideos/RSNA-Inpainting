import os
import csv
import ast  # To parse the data field from the CSV
import nibabel as nib

def load_csv_annotations(csv_file_path):
    """
    Load the CSV file and extract annotations into a dictionary.
    The dictionary keys will be SOPInstanceUID_SeriesInstanceUID and the values will be the annotation data.
    """
    annotations = set()
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        c = 0
        for row in reader:
            sop_instance_uid = row['SOPInstanceUID']
            series_instance_uid = row['SeriesInstanceUID']
            key = sop_instance_uid
            annotations.add(key)
            c += 1
    print(c)
    return annotations

def check_nii_files(directory, annotations):
    """
    Check how many .nii.gz files in the directory have corresponding annotations in the CSV file.
    """
    nii_files = [f[:-6] for f in os.listdir(directory) if f.endswith('.nii.gz')]
    matching_count = 0

    for nii_file in nii_files:
        # Remove the file extension to get the key
        key = os.path.splitext(nii_file)[0]
        if key.split("_")[0] in annotations:
            matching_count += 1

    print(f"Number of .nii.gz files with corresponding annotations: {matching_count}/{len(nii_files)}")


def main(csv_file_path, nii_directory):
    # Load annotations from CSV
    annotations = load_csv_annotations(csv_file_path)

    # Check .nii.gz files for corresponding annotations
    matching_count = check_nii_files(nii_directory, annotations)


if __name__ == "__main__":
    csv_file_path = "/research/Data/CQ500_Brain_Hemorrhage_Dataset/Bounding_Box_Labels/2_Extrapolation_to_All_Series.csv"  # Replace with your CSV file path
    nii_directory = "/research/Data/DK_RSNA_HM/cq500_prepared_data/mask_nifti"  # Replace with your .nii.gz directory path

    main(csv_file_path, nii_directory)
