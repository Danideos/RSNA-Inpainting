import os

# Path to the directory containing the files
directory = ""

# Iterate over all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        # Split the filename to extract the {id1}, {id2} and ignore the slice part
        parts = filename.split('_')
        if len(parts) == 6 and parts[0] == 'ID' and parts[2] == 'ID':
            id1 = parts[1]
            id2 = parts[3][:-7]
            
            # Construct the new filename with the format ID{id1}ID{id2}.png
            new_filename = f"ID{id1}ID{id2}{parts[5][:-4]}png"
            
            # Get the full paths for the old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")
        else:
            print(filename, parts)