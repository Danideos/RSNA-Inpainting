from pathlib import Path
import pydicom
import argparse
import os
import tempfile
from tqdm import tqdm
from joblib import Parallel, delayed
import shutil
import subprocess
import warnings



# Suppress all annoying harmless preprocessing noise
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def remove_duplicates(output_dir):
    removed = 0
    for file in os.listdir(output_dir):
        if (file.endswith('.nii') and not file.endswith('.nii.gz')) or file.count("_") >= 3:
            os.remove(os.path.join(output_dir, file))
            removed += 1
    print(f"Removed count: {removed}")

def run_dcm2niix(input_dir, output_dir, filename):
    cmd = ['dcm2niix', '-z', 'y', '-f', filename, '-b', 'n', '-o', output_dir, input_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running dcm2niix: {result.stderr}")

def dicom_to_nifti(dcm_dir, nii_dir, num_jobs=-1, series=False, healthy=False, series_to_slices=None, series_label_counts=None, range=None):
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir)

    def convert_file_to_nifti(file, new_dir=None, name=None):
            chosen_dir = nii_dir if new_dir is None else new_dir
            file_path = os.path.join(dcm_dir, file)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy(file_path, tmp_dir)
                name = name if name is not None else os.path.basename(file).split(".")[0]
                run_dcm2niix(tmp_dir, chosen_dir, name)
    
    def convert_series_to_nifti(files, series_uid):
        for i, file in enumerate(files):
            nii_series_dir = os.path.join(nii_dir, series_uid)
            os.makedirs(nii_series_dir, exist_ok=True)
            label = '1' if file[1] == True else '0'
            name =  f"s{i}" + " ID_" + file[0] + f"_{label}"
            file = "ID_" + file[0] + ".dcm"
            convert_file_to_nifti(file, nii_series_dir, name)
    
    if series:
        if range is not None:
            start, end = range
        else:
            start, end = 0, len(series_to_slices)
        for series, files in tqdm(list(series_to_slices.items()), desc="Converting to NIfTI"):
            if healthy:
                if series_label_counts[series]['1'] == 0:
                    if start == 0:
                        end -= 1
                        convert_series_to_nifti(files, series)
                    else:
                        start -= 1
            else:
                if series_label_counts[series]['1'] > 0:
                    if start == 0:
                        convert_series_to_nifti(files, series)
                        end -= 1
                    else:
                        start -= 1
            if end == 0:
                break
        if healthy:
            series_lengths = {k: len(v) for k, v in series_to_slices.items() if series_label_counts[k]['1'] == 0}      
        else:
            series_lengths = {k: len(v) for k, v in series_to_slices.items() if series_label_counts[k]['1'] > 0}
        print(f"{len(os.listdir(nii_dir))}/{len(series_lengths)} series available as NifTI.")
        
    else:
        files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]

        Parallel(n_jobs=num_jobs)(
            delayed(convert_file_to_nifti)(file)
            for file in tqdm(files, desc="Converting to NIfTI")
        )

        remove_duplicates(output_dir=nii_dir)
        print(f"{len(os.listdir(nii_dir))}/{len(files)} of original slices available as NifTI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dcm_dir", "--dcm_dir", type=str, required=True, help="Path to the directory containing DICOM files")
    parser.add_argument("-nii_dir", "--nii_dir", type=str, required=True, help="Path to the directory to save NIfTI files")
    parser.add_argument("-num_jobs", "--num_jobs", type=int, default=-1, help="Number of parallel jobs for conversion")

    args = parser.parse_args()
    dcm_dir = args.dcm_dir
    nii_dir = args.nii_dir
    num_jobs = args.num_jobs

    dicom_to_nifti(dcm_dir, nii_dir, num_jobs)
