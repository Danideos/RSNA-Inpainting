
csv_file_1 = "/research/Data/DK_RSNA_HM/BHSD/bounding_boxes.csv"
csv_file_2 = "/research/Data/DK_RSNA_HM/cq500_prepared_data/bounding_boxes_bhx.csv"
output_csv_path = "/research/Data/DK_RSNA_HM/cq500_prepared_data/bounding_boxes_concat.csv"

import pandas as pd
import re
import ast

# Function to correct the bounding_box format by adding quotes around keys
def correct_bounding_box(bbox_str):
    try:
        # Add quotes around keys that don't have them
        corrected_str = re.sub(r'(\w+):', r'"\1":', bbox_str)
        # Convert the corrected string to a dictionary
        bbox_dict = eval(corrected_str)  # Using eval since we've manually corrected the string
        return bbox_dict
    except Exception as e:
        # Handle any parsing errors
        print(f"Error processing bounding_box entry: {bbox_str} - {e}")
        return bbox_str
    
def correct_bounding_box2(bbox_str):
    try:
        
        bbox_dict = ast.literal_eval(bbox_str)
        a= 512 - bbox_dict['bottomright_y']
        b=512 - bbox_dict['topleft_y']
        bbox_dict['topleft_y'] = a
        bbox_dict['bottomright_y'] = b
        return bbox_dict
    except Exception as e:
        # Handle any parsing errors
        print(f"Error processing bounding_box entry: {bbox_str} - {e}")
        return bbox_str

def correct_id(id_str):
    try:
        id_str_new = "".join(id_str.split("_")[:4]).replace(".nii.gz", "").replace(".png", "")
        return id_str_new
    except Exception as e:
        print(f"Error processing id entry: {id_str} - {e}")
        return id_str
    


# Read the first CSV file into a DataFrame
df1 = pd.read_csv(csv_file_1)

# Rename the 'labeltype' column to 'labelType'
if 'labeltype' in df1.columns:
    df1 = df1.rename(columns={'labeltype': 'labelType'})

if 'id' in df1.columns:
    df1['id'] = df1['id'].apply(correct_id)

# # Correct the bounding_box format by adding quotes around keys
if 'bounding_box' in df1.columns:
    df1['bounding_box'] = df1['bounding_box'].apply(correct_bounding_box)
# Read the second CSV file into a DataFrame
df2 = pd.read_csv(csv_file_2)
if 'bounding_box' in df1.columns:
    df2['bounding_box'] = df2['bounding_box'].apply(correct_bounding_box2)



# Combine the two DataFrames by concatenating them (stack them on top of each other)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(output_csv_path, index=False)

print(f"Combined CSV file saved to {output_csv_path}")
