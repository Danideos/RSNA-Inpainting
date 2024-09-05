import os
import torch
from libs.Mediffusion_Fork.ddpm import DiffusionModule
import monai.transforms as mt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from utils import add_channel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preprocess_images(input_dir, base_names, img_size=512, resize_size=256):
    device = 'cuda'

    bet_png_dir = os.path.join(input_dir, 'bet_png')
    left_png_dir = os.path.join(input_dir, 'left_png')
    right_png_dir = os.path.join(input_dir, 'right_png')

    # Create dictionaries for all the images in the batch
    data_dicts = []
    for base_name in base_names:
        data_dicts.append({
            'img': os.path.join(bet_png_dir, base_name + '.png'),
            'left': os.path.join(left_png_dir, base_name + '.png'),
            'right': os.path.join(right_png_dir, base_name + '.png')
        })

    # Define the data transforms for all images at once
    data_transforms = mt.Compose([
        mt.LoadImageD(keys=["img", "left", "right"]),
        mt.LambdaD(keys=["img", "left", "right"], func=lambda x: add_channel(x)),
        mt.ResizeWithPadOrCropD(keys=["img", "left", "right"], spatial_size=(img_size, img_size)), 
        mt.ResizeD(keys=["img", "left", "right"], spatial_size=(resize_size, resize_size)),
        mt.ToTensorD(keys=["img", "left", "right"]),
    ])

    # Apply the transforms to the batch of data
    transformed_data = data_transforms(data_dicts)

    # Apply Gaussian blur only to the bet image for conditional input
    blur_transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5))
    bet_img_blurred = torch.stack([blur_transform(transformed_data[i]['img']) for i in range(len(base_names))], dim=0).to(device)

    # Combine all images into tensors (original images, not blurred)
    bet_img_tensor = torch.stack([transformed_data[i]['img'] for i in range(len(base_names))], dim=0).to(device) 
    left_img_tensor = torch.stack([transformed_data[i]['left'] for i in range(len(base_names))], dim=0).to(device) 
    right_img_tensor = torch.stack([transformed_data[i]['right'] for i in range(len(base_names))], dim=0).to(device) 

    # Concatenate along the channel dimension (bet blurred, left, right)
    concat_input = torch.cat([bet_img_blurred, left_img_tensor, right_img_tensor], dim=1) / 127.5 - 1

    return bet_img_tensor, concat_input

def save_comparison_images(original_img, predicted_img, output_path):
    # Convert tensors to numpy arrays
    original_img = original_img.squeeze(0).cpu().numpy() 
    predicted_img = predicted_img.squeeze(0).cpu().numpy() * 127.5 + 127.5
    
    # Calculate the absolute difference and normalize it
    abs_diff = np.abs(original_img - predicted_img)
    abs_diff = (abs_diff / abs_diff.max()) * 255.0

    # Convert arrays to images
    original_img = Image.fromarray(original_img.astype(np.uint8))
    predicted_img = Image.fromarray(predicted_img.astype(np.uint8))
    abs_diff_img = Image.fromarray(abs_diff.astype(np.uint8))

    # Concatenate images side by side (horizontally)
    concatenated_img = Image.new('L', (original_img.width * 3, original_img.height))
    concatenated_img.paste(original_img, (0, 0))
    concatenated_img.paste(predicted_img, (original_img.width, 0))
    concatenated_img.paste(abs_diff_img, (original_img.width * 2, 0))

    # Save the concatenated image
    concatenated_img.save(output_path)

def batch_inpaint(input_dir, output_dir, model, batch_size=100):
    bet_dir = os.path.join(input_dir, 'bet_png')
    file_names = os.listdir(bet_dir)
    base_names = [os.path.splitext(f)[0] for f in file_names]  # Remove the .png extension

    for count in tqdm(range(0, len(base_names), batch_size), desc="Batch processing"):
        batch_base_names = base_names[count:count + batch_size]

        # Preprocess images as a batch
        img_tensors, concat_tensors = preprocess_images(input_dir, batch_base_names)

        # Create a noise tensor for inpainting
        noise = torch.randn(img_tensors.shape, device=img_tensors.device)

        # Inpaint the images
        inpainted_batch = model.predict(
            noise,
            model_kwargs={"concat": concat_tensors},
            inference_protocol="DDIM50",
            mask=None,
        )

        # Save the inpainted images along with original and abs diff
        for i, base_name in enumerate(batch_base_names):
            inpainted_img = inpainted_batch[i].unsqueeze(0)[0, 0]

            # Save original, inpainted, and absolute difference concatenated
            output_path = os.path.join(output_dir, base_name + '_comparison.png')
            save_comparison_images(img_tensors[i], inpainted_img, output_path)

if __name__ == "__main__":
    input_dir = ""  # Replace with your directory
    output_dir = ""  # Replace with your output directory

    model = DiffusionModule("./config.yaml")
    model.load_ckpt("./model-path.ckpt", ema=True)
    model.cuda().half()
    model.eval()

    # Batch prediction
    batch_inpaint(input_dir, output_dir, model)
