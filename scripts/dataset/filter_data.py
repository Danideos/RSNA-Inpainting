import os
import pandas as pd
import shutil
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count, get_context, set_start_method

# Mapping of hemorrhage types to numbers
HEMORRHAGE_TYPES = {
    "epidural": 1,
    "intraparenchymal": 2,
    "intraventricular": 3,
    "subarachnoid": 4,
    "subdural": 5,
    "any": 6
}

def get_hemorrhage_subsets(train, fraction=0.05):
    train["PureID"] = ["ID_" + id.split("_")[1] for id in train["ID"]]
    train["LabelType"] = [id.split("_")[2] for id in train["ID"]]
    train_pivot = train.pivot_table(index='PureID', columns='LabelType', values='Label').fillna(0)

    zero_hemorrhage_ids = train_pivot[(train_pivot.sum(axis=1) == 0)]
    subset_zero_hemorrhage = zero_hemorrhage_ids.sample(frac=fraction, random_state=42)
    
    non_zero_hemorrhage_ids = train_pivot[(train_pivot.sum(axis=1) > 0)]
    subset_non_zero_hemorrhage = non_zero_hemorrhage_ids.sample(frac=fraction, random_state=42)
    
    healthy_ids = subset_zero_hemorrhage.index.to_list()
    unhealthy_ids = subset_non_zero_hemorrhage.index.to_list()
    
    healthy_types = ['0'] * len(healthy_ids)  # No hemorrhage
    unhealthy_types_dict = subset_non_zero_hemorrhage.apply(
        lambda row: ''.join([str(HEMORRHAGE_TYPES[col]) for col in row[row > 0].index]), axis=1
    ).to_dict()

    return pd.DataFrame({'ID': healthy_ids, 'Types': healthy_types}), unhealthy_types_dict

def copy_file_pool(src_dst):
    src, dst = src_dst
    try:
        shutil.copy(src, dst)
        return True
    except Exception as e:
        return False, str(e)
    
def filter_dicom_images(csv_path, raw_path, filtered_path, fraction=0.05, num_jobs=-1):
    train = pd.read_csv(csv_path)
    healthy_subset, unhealthy_types_dict = get_hemorrhage_subsets(train, fraction)
    
    healthy_ids = set(healthy_subset['ID'].tolist())
    unhealthy_ids = set(unhealthy_types_dict.keys())
    print(f"Amount of healthy train images to process: {len(healthy_ids)}")
    print(f"Amount of unhealthy train images to process: {len(unhealthy_ids)}")
    all_train_images = os.listdir(raw_path)
    
    healthy_images = [f for f in tqdm(all_train_images, desc="Processing healthy files", unit="file") if f.split(".")[0] in healthy_ids]
    unhealthy_images = [f for f in tqdm(all_train_images, desc="Processing unhealthy files", unit="file") if f.split(".")[0] in unhealthy_ids]
    
    healthy_path = os.path.join(filtered_path, 'healthy')
    unhealthy_path = os.path.join(filtered_path, 'unhealthy')
    
    if not os.path.exists(healthy_path):
        os.makedirs(healthy_path)
        
    if not os.path.exists(unhealthy_path):
        os.makedirs(unhealthy_path)
        
    num_jobs = cpu_count() if num_jobs == -1 else num_jobs
    with get_context("spawn").Pool(processes=num_jobs) as pool:
        healthy_args = [(os.path.join(raw_path, image_file), os.path.join(healthy_path, f"{image_file.split('.')[0]}_0.dcm")) for image_file in healthy_images]
        unhealthy_args = [(os.path.join(raw_path, image_file), os.path.join(unhealthy_path, f"{image_file.split('.')[0]}_{unhealthy_types_dict[image_file.split('.')[0]]}.dcm")) for image_file in unhealthy_images]
        
        list(tqdm(pool.imap_unordered(copy_file_pool, healthy_args), total=len(healthy_images), desc="Copying healthy files", unit="file"))
        list(tqdm(pool.imap_unordered(copy_file_pool, unhealthy_args), total=len(unhealthy_images), desc="Copying unhealthy files", unit="file"))
  
    print("Filtering and copying completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter DICOM images with zero hemorrhage labels.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing labels.')
    parser.add_argument('--raw_path', type=str, required=True, help='Path to the directory containing raw DICOM images.')
    parser.add_argument('--filtered_path', type=str, required=True, help='Path to the directory to save filtered DICOM images.')
    parser.add_argument('--fraction', type=float, default=0.05, help='Fraction of zero hemorrhage images to filter.')
    parser.add_argument('--num_jobs', type=int, default=-1, help='Number of parallel jobs to run.')

    args = parser.parse_args()

    set_start_method("spawn", force=True)

    filter_dicom_images(args.csv_path, args.raw_path, args.filtered_path, args.fraction, args.num_jobs)
