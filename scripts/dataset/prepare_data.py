import os
import argparse
import nibabel as nib
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import apply_window, apply_mask


def is_slice_valid(slice_img, threshold_fraction):
    total_pixels = slice_img.size
    valid_pixels = np.sum(slice_img != 0)
    fraction_in_range = valid_pixels / total_pixels
    return fraction_in_range >= threshold_fraction

def save_img_single(slice_img, output_path):
    img = Image.fromarray((slice_img * 255).astype(np.uint8))
    img.save(output_path)

def save_img_gray(masked_slice_img, unmasked_slice_img, output_path):
    unmasked_slice_gray = (unmasked_slice_img * 255).astype(np.uint8)
    masked_slice_gray = (masked_slice_img * 255).astype(np.uint8)
    
    concatenated_img = np.concatenate((unmasked_slice_gray, masked_slice_gray), axis=1)
    img = Image.fromarray(concatenated_img)
    img.save(output_path)


def process_nifti_file(nifti_path, mask_path, dcm_dir, threshold_fraction):
    valid_slices = []
    invalid_slices = []
    invalid_count = 0
    
    nifti_img = nib.load(nifti_path)
    mask_img = nib.load(mask_path)

    nifti_data = nifti_img.get_fdata()
    mask_data = mask_img.get_fdata()

    masked_data = apply_mask(nifti_data, mask_data)

    for i in range(masked_data.shape[2]):
        masked_img = masked_data[:, :, i]
        unmasked_img = nifti_data[:, :, i]
        mask_img_slice = mask_data[:, :, i]

        unmasked_img = apply_window(unmasked_img, 40, 80)
        windowed_slice = apply_window(masked_img, 40, 80)
        
        base_name = os.path.basename(nifti_path).split('.')[0]
        if is_slice_valid(windowed_slice, threshold_fraction):
            valid_slices.append((windowed_slice, unmasked_img, base_name, i, mask_img_slice))
        else:
            invalid_slices.append((windowed_slice, unmasked_img, base_name, i, masked_img))
            invalid_count += 1

    return invalid_count, masked_data.shape[2], valid_slices, invalid_slices

def save_valid_slices(slices, bet_png_dir, mask_png_dir, description):
    for masked_slice, unmasked_slice, base_name, i, mask_img in tqdm(slices, desc=description):
        name = f"{base_name}.png"
        img_output_path = os.path.join(bet_png_dir, name)
        mask_output_path = os.path.join(mask_png_dir, name)
        save_img_single(masked_slice, img_output_path)
        save_img_single(mask_img, mask_output_path)

def save_invalid_slices(slices, invalid_dir, description):
    for masked_slice, unmasked_slice, base_name, i, mask_img in tqdm(slices, desc=description):
        name = f"{base_name}.png"
        img_output_path = os.path.join(invalid_dir, name)
        save_img_gray(masked_slice, unmasked_slice, img_output_path)

def create_output_dirs(base_dir, test_fraction, health):
    bet_png_dir_train = os.path.join(base_dir, health, 'train', 'bet_png')
    mask_png_dir_train = os.path.join(base_dir, health, 'train', 'mask_png')
    bet_png_dir_test = os.path.join(base_dir, health, 'test', 'bet_png') if test_fraction > 0.0 else None
    mask_png_dir_test = os.path.join(base_dir, health, 'test', 'mask_png') if test_fraction > 0.0 else None
    invalid_dir = os.path.join(base_dir, health, 'invalid')

    os.makedirs(invalid_dir, exist_ok=True)
    for directory in [bet_png_dir_train, mask_png_dir_train, bet_png_dir_test, mask_png_dir_test]:
        if directory: 
            os.makedirs(directory, exist_ok=True)

    return (bet_png_dir_train, mask_png_dir_train, bet_png_dir_test, mask_png_dir_test, invalid_dir)

def create_series_output_dirs(base_dir, health, series_id, test_fraction):
    bet_png_dir_train = os.path.join(base_dir, health, 'train', series_id, 'bet_png')
    mask_png_dir_train = os.path.join(base_dir, health, 'train', series_id, 'mask_png')
    bet_png_dir_test = os.path.join(base_dir, health, 'test', series_id, 'bet_png') if test_fraction > 0.0 else None
    mask_png_dir_test = os.path.join(base_dir, health, 'test', series_id, 'mask_png') if test_fraction > 0.0 else None

    for directory in [bet_png_dir_train, mask_png_dir_train, bet_png_dir_test, mask_png_dir_test]:
        if directory:
            os.makedirs(directory, exist_ok=True)

    return (bet_png_dir_train, mask_png_dir_train, bet_png_dir_test, mask_png_dir_test)


def process_files(nifti_dir, mask_dir, dcm_dir, threshold_fraction, files, n_jobs):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_nifti_file)(
            os.path.join(nifti_dir, nifti_file),
            os.path.join(mask_dir, nifti_file),
            dcm_dir,
            threshold_fraction
        ) for nifti_file in files if os.path.exists(os.path.join(mask_dir, nifti_file))
    )
    return results

def process_nifti_directory(output_dir, health, nifti_dir, mask_dir, dcm_dir, threshold_fraction=0.05, test_fraction=0.0, n_jobs=-1, series=False):
    if series:
        series_dirs = [d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))]
        total_invalid_slices = 0
        total_slices = 0
        
        for series_dir in tqdm(series_dirs, desc="Processing series"):
            series_nifti_dir = os.path.join(nifti_dir, series_dir)
            series_mask_dir = os.path.join(mask_dir, series_dir)
            bet_png_dir_train_series, mask_png_dir_train_series, bet_png_dir_test_series, mask_png_dir_test_series = create_series_output_dirs(output_dir, health, series_dir, test_fraction)
            
            nifti_files = [f for f in os.listdir(series_nifti_dir) if f.endswith('.nii.gz')]
            print(f"Processing series {series_dir} with {len(nifti_files)} files")
            
            batch_size = 100  # Adjust this value based on your memory constraints
            for i in range(0, len(nifti_files), batch_size):
                batch_files = nifti_files[i:i + batch_size]
                batch_results = process_files(series_nifti_dir, series_mask_dir, dcm_dir, threshold_fraction, batch_files, n_jobs)
                
                valid_slices = []
                invalid_slices = []
                for result in batch_results:
                    valid_slices.extend(result[2])
                    invalid_slices.extend(result[3])
                    total_invalid_slices += result[0]
                    total_slices += result[1]
                
                if test_fraction > 0.0 and test_fraction < 1.0:
                    train_slices, test_slices = train_test_split(valid_slices, test_size=test_fraction, random_state=42)
                    save_valid_slices(train_slices, bet_png_dir_train_series, mask_png_dir_train_series, f"Saving train slices (batch {i//batch_size + 1})")
                    save_valid_slices(test_slices, bet_png_dir_test_series, mask_png_dir_test_series, f"Saving test slices (batch {i//batch_size + 1})")
                elif test_fraction == 0.0:
                    save_valid_slices(valid_slices, bet_png_dir_train_series, mask_png_dir_train_series, f"Saving pred slices (batch {i//batch_size + 1})")
                else:
                    save_valid_slices(valid_slices, bet_png_dir_test_series, mask_png_dir_test_series, f"Saving test slices (batch {i//batch_size + 1})")
                # save_invalid_slices(invalid_slices, invalid_dir, f"Saving invalid slices (batch {i//batch_size + 1})")
        
        print(f"Invalid slices: {total_invalid_slices} - out of total: {total_slices}")
    else:
        bet_png_dir_train, mask_png_dir_train, bet_png_dir_test, mask_png_dir_test, invalid_dir = create_output_dirs(output_dir, test_fraction, health)
        nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')]
        print("nifti amount of files:", len(nifti_files))
        batch_size = 100  # Adjust this value based on your memory constraints
        total_invalid_slices = 0
        total_slices = 0

        for i in tqdm(range(0, len(nifti_files), batch_size), desc="Processing batches"):
            batch_files = nifti_files[i:i + batch_size]
            batch_results = process_files(nifti_dir, mask_dir, dcm_dir, threshold_fraction, batch_files, n_jobs)

            valid_slices = []
            invalid_slices = []
            for result in batch_results:
                valid_slices.extend(result[2])
                invalid_slices.extend(result[3])
                total_invalid_slices += result[0]
                total_slices += result[1]

            if test_fraction > 0.0 and test_fraction < 1.0:
                train_slices, test_slices = train_test_split(valid_slices, test_size=test_fraction, random_state=42)
                save_valid_slices(train_slices, bet_png_dir_train, mask_png_dir_train, f"Saving train slices (batch {i//batch_size + 1})")
                save_valid_slices(test_slices, bet_png_dir_test, mask_png_dir_test, f"Saving test slices (batch {i//batch_size + 1})")
            elif test_fraction == 0.0:
                save_valid_slices(valid_slices, bet_png_dir_train, mask_png_dir_train, f"Saving pred slices (batch {i//batch_size + 1})")
            else:
                save_valid_slices(valid_slices, bet_png_dir_test, mask_png_dir_test, f"Saving test slices (batch {i//batch_size + 1})")
            save_invalid_slices(invalid_slices, invalid_dir, f"Saving invalid slices (batch {i//batch_size + 1})")

        print(f"Invalid slices: {total_invalid_slices} - out of total: {total_slices}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIfTI files and save slices with masks applied.")
    parser.add_argument('--nifti_dir', required=True, help="Directory containing NIfTI files.")
    parser.add_argument('--mask_dir', required=True, help="Directory containing mask NIfTI files.")
    parser.add_argument('--dcm_dir', required=False, help="Directory containing DICOM files.")
    parser.add_argument('--output_dir', required=True, help="Base output directory for processed images.")
    parser.add_argument('--threshold_fraction', type=float, default=0.05, help="Minimum fraction of pixels within the intensity range 50-200.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs to run. -1 means using all processors.")
    parser.add_argument('--test_fraction', type=float, default=0.05, help="Fraction of data to be used as test set.")
    args = parser.parse_args()
    dcm_dir = "ahoooj"
    process_nifti_directory(args.output_dir, "healthy", args.nifti_dir, args.mask_dir, dcm_dir, args.threshold_fraction, args.test_fraction, args.n_jobs)
