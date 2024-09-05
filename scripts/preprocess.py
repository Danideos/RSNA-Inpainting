import os
import argparse
import tempfile
import subprocess
from dataset.dicom_to_nifti import dicom_to_nifti
from dataset.prepare_data import process_nifti_directory
from dataset.filter_data import filter_dicom_images
from CT_BET_GPU.unet_CT_SS import main as unet_ct_ss_main
# from utils import get_series_slice_mapping, analyze_csv
import pickle


def preprocess_dicom_dir(dcm_dir, png_dir, filter, csv_path, series):
    with tempfile.TemporaryDirectory() as tmpdir:
        nii_healthy_dir = os.path.join(tmpdir, "nii_healthy/")
        os.makedirs(nii_healthy_dir, exist_ok=True)
        nii_unhealthy_dir = os.path.join(tmpdir, "nii_unhealthy/")
        os.makedirs(nii_unhealthy_dir, exist_ok=True)
        mask_healthy_dir = os.path.join(tmpdir, "mask_healthy/")
        os.makedirs(mask_healthy_dir, exist_ok=True)
        mask_unhealthy_dir = os.path.join(tmpdir, "mask_unhealthy/")
        os.makedirs(mask_unhealthy_dir, exist_ok=True)
        weight_path = "./unet_CT_SS_20171114_170726.h5" # Download weights from CT_BET repo
        if not series:
            filtered_dir = os.path.join(tmpdir, "filtered/")
            os.makedirs(filtered_dir, exist_ok=True)
            if filter:
                assert(csv_path is not None)
                filter_dicom_images(csv_path, dcm_dir, filtered_dir, 1.0, -1)
            else:
                filtered_dir = dcm_dir
            

            unhealthy_dir = os.path.join(filtered_dir, "unhealthy")
            dicom_to_nifti(unhealthy_dir, nii_unhealthy_dir, num_jobs=-1)
            unet_ct_ss_main(nii_unhealthy_dir, mask_unhealthy_dir, weight_path)
            process_nifti_directory(png_dir, "unhealthy", nii_unhealthy_dir, mask_unhealthy_dir, dcm_dir, threshold_fraction=0.03, test_fraction=1.0, n_jobs=-1)

            healthy_dir = os.path.join(filtered_dir, "healthy")
            dicom_to_nifti(healthy_dir, nii_healthy_dir, num_jobs=-1)
            unet_ct_ss_main(nii_healthy_dir, mask_healthy_dir, weight_path)
            process_nifti_directory(png_dir, "healthy", nii_healthy_dir, mask_healthy_dir, dcm_dir, threshold_fraction=0.03, test_fraction=0.0, n_jobs=-1)
        else:
            # Load the series_to_slices dictionary
            with open('./assets/series_to_slices_test.pkl', 'rb') as f:
                series_to_slices = pickle.load(f)

            # Load the series_label_counts dictionary
            with open('./assets/series_label_counts_test.pkl', 'rb') as f:
                series_label_counts = pickle.load(f)

            dicom_to_nifti(dcm_dir, nii_healthy_dir, num_jobs=-1, series=True, healthy=True, series_to_slices=series_to_slices, series_label_counts=series_label_counts)
            unet_ct_ss_main(nii_healthy_dir, mask_healthy_dir, weight_path, series=True)
            process_nifti_directory(png_dir, "healthy", nii_healthy_dir, mask_healthy_dir, dcm_dir, threshold_fraction=0.03, test_fraction=0.0, n_jobs=-1, series=True)

            dicom_to_nifti(dcm_dir, nii_unhealthy_dir, num_jobs=-1, series=True, healthy=False, series_to_slices=series_to_slices, series_label_counts=series_label_counts)
            nii_dir = '/home/bje01/Documents/Data/archive/images'
            unet_ct_ss_main(nii_dir, mask_unhealthy_dir, weight_path, series=True)
            process_nifti_directory(png_dir, "unhealthy", nii_dir, mask_unhealthy_dir, dcm_dir, threshold_fraction=0.03, test_fraction=0.0, n_jobs=-1, series=True)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DICOM directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory to save the preprocessed png images")
    parser.add_argument("--filter", action="store_true", help="Flag whether to filter out the unhealthy data prior to preprocessing")
    parser.add_argument("--csv_path", type=str, required=False, default=None, help="Path to csv label file for input files for filtering")
    parser.add_argument("--series", action="store_true", help="Whether to to treat slices in the input directory as series")
    args = parser.parse_args()

    preprocess_dicom_dir(args.input_dir, args.preprocessed_dir, args.filter, args.csv_path, args.series)
