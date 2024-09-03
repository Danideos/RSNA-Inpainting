import os
import numpy as np
import torch
import random
from mediffusion import DiffusionModule
import monai.transforms as mt
from PIL import Image, ImageDraw, ImageOps
from utils import add_channel
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preprocess_images(png_paths, mask_paths, img_size=512, resize_size=256):
    device = 'cuda'

    def combined_transform(data, min_box_size=7, max_box_size=70):
        img = data['img'].as_tensor()
        concat = data['concat'].as_tensor()

        img_tensor = img.clone().detach()

        # Initialize mask tensor and loop until a valid box is found
        c = 0
        while c <= 30:
            # Randomly select the size and location of the box
            box_width = random.randint(min_box_size, max_box_size)
            box_height = random.randint(min_box_size, max_box_size)
            
            # Ensure the box is within the bounds of the image
            topleft_x = random.randint(0, resize_size - box_width)
            topleft_y = random.randint(0, resize_size - box_height)
            bottomright_x = topleft_x + box_width
            bottomright_y = topleft_y + box_height

            # Check if at least 50% of the box overlaps with the mask
            mask_array = concat.squeeze().cpu().numpy()
            mask_area = np.sum(mask_array[topleft_y:bottomright_y, topleft_x:bottomright_x])
            box_area = box_width * box_height

            if mask_area >= 0.5 * box_area * 255:
                break
            c += 1

        # Create a mask with the same size as the transformed image
        mask = np.zeros(img_tensor.shape[-2:], dtype=np.uint8)
        mask[topleft_y:bottomright_y, topleft_x:bottomright_x] = 1  
        mask_tensor = torch.tensor(mask, dtype=torch.float).unsqueeze(0).to(img_tensor.device)

        masked_img = img_tensor * (1 - mask_tensor) 

        combined = torch.cat([concat, masked_img], dim=0)

        data['concat'] = combined / 127.5 - 1
        data['img'] = img_tensor / 127.5 - 1
        
        return data, (topleft_x, topleft_y, bottomright_x, bottomright_y)

    data_transforms = mt.Compose([
        mt.LoadImageD(keys=["img", "concat"]),
        mt.LambdaD(keys=["img", "concat"], func=lambda x: add_channel(x)),
        mt.ResizeWithPadOrCropD(keys=["img", "concat"], spatial_size=(img_size, img_size)), 
        mt.ResizeD(keys=["img", "concat"], spatial_size=(resize_size, resize_size)),
    ])

    batch_data = [{"img": png_path, "concat": mask_path} for png_path, mask_path in zip(png_paths, mask_paths)]
    transformed_imgs = data_transforms(batch_data)

    img_tensors = []
    mask_tensors = []
    adjusted_box_coords_list = []

    for img_dict in transformed_imgs:
        img_dict, adjusted_box_coords = combined_transform(img_dict)
        img_tensors.append(img_dict["img"])
        mask_tensors.append(img_dict["concat"])
        adjusted_box_coords_list.append(adjusted_box_coords)

    img_tensors = torch.stack(img_tensors, dim=0).to(device)
    mask_tensors = torch.stack(mask_tensors, dim=0).to(device)
    
    return img_tensors, mask_tensors, adjusted_box_coords_list

def create_inpainted_images(input_dir, output_dir, model, num_images=100, batch_size=100, n=1, partitions=[0]):
    bet_dir = os.path.join(input_dir, 'bet_png')
    mask_dir = os.path.join(input_dir, 'mask_png')
    ano_dir = os.path.join(output_dir, 'ano_png')
    os.makedirs(ano_dir, exist_ok=True)

    file_names = os.listdir(bet_dir)
    file_names.sort()  # Ensure files are sorted for consistent partitioning

    total_images = len(file_names)

    for partition in partitions:
        # Calculate the range of files for this partition
        images_per_partition = total_images // n
        start_idx = partition * images_per_partition
        if partition == 9:
            start_idx += 17000
        end_idx = start_idx + images_per_partition if partition < n - 1 else total_images

        partition_files = file_names[start_idx:end_idx]
        count = 0
        with tqdm(total=len(partition_files), desc=f'Processing partition {partition + 1}/{n}', unit='img') as pbar:
            while count < len(partition_files):
                batch_files = partition_files[count:count+batch_size]
                png_paths = [os.path.join(bet_dir, file_name) for file_name in batch_files]
                mask_paths = [os.path.join(mask_dir, file_name) for file_name in batch_files]

                img_tensors, mask_tensors, adjusted_box_coords_list = preprocess_images(png_paths, mask_paths)

                noise = torch.randn(img_tensors.shape, device=img_tensors.device)
                mask = torch.zeros(img_tensors.shape, dtype=torch.uint8, device=img_tensors.device)

                for i, box_coords in enumerate(adjusted_box_coords_list):
                    topleft_x, topleft_y, bottomright_x, bottomright_y = box_coords
                    mask[i, :, topleft_y:bottomright_y, topleft_x:bottomright_x] = 1 

                inpainted_batch = model.predict(
                    noise,
                    model_kwargs={"concat": mask_tensors},
                    inference_protocol="DDIM50",
                    mask=mask,
                    original_image=img_tensors,
                )

                for i, file_name in enumerate(batch_files):
                    # Get the inpainted image
                    inpainted_img = inpainted_batch[i].unsqueeze(0)[0, 0].cpu().numpy() * 127.5 + 127.5
                    inpainted_img = Image.fromarray(inpainted_img.astype(np.uint8))

                    inpainted_img.save(os.path.join(ano_dir, file_name))

                count += len(batch_files)
                pbar.update(len(batch_files))  # Update the progress bar by the number of files processed in this batch


if __name__ == "__main__":
    input_dir = "/research/Data/DK_RSNA_HM/prepared_data_stage_1/healthy/train"  # Replace with your directory
    output_dir = input_dir  # Assuming the output dir is the same as input dir
    num_images = 100  # Total number of images to inpaint
    batch_size = 100  # Process 100 images at a time
    n = 11  # Number of partitions
    partitions = [9,10]  # Partitions to process (0-indexed)

    model = DiffusionModule("/research/projects/DanielKaiser/RSNA_Inpainting/config.yaml")
    model.load_ckpt("/research/projects/DanielKaiser/RSNA_Inpainting/outputs/pl/ICH_Inpainting_4A100-epoch=359-step=30600-val_loss=0.004333.ckpt", ema=True)
    model.cuda().half()
    model.eval()

    create_inpainted_images(input_dir, output_dir, model, num_images=num_images, batch_size=batch_size, n=n, partitions=partitions)
