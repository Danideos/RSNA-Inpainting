import os
import shutil

# Define source and destination directories
source_dir = '/research/Data/DK_RSNA_HM/BHSD/healthy/train/mask_png'
destination_dir = '/research/Data/DK_RSNA_HM/cq500_prepared_data/unhealthy/train/mask_png'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Copy all files from source to destination
for filename in os.listdir(source_dir):
    # Construct full file path
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(source_file):
        shutil.copy2(source_file, destination_file)
        print(f'Copied {filename} to {destination_dir}')

print("File copy completed.")
