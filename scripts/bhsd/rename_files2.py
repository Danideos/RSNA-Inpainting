import os

# Path to the directory containing the files
directory = ""

# Iterate over all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith("..png"):
        # Split the filename to extract the {id} part
        id_part = filename.split('_')[0]
        
        # Construct the new filename with just {id}.png
        new_filename = f"{id_part}.png"
        
        # Get the full paths for the old and new filenames
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")
