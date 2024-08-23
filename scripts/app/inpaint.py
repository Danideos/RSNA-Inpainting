
from app.utils.state_utils import update_inpainted_square
from app.utils.general_utils import get_keys, get_contour_path, apply_func_to_grid
from predict import prepare_model, preprocess_images, inpaint_images

from PIL import Image
import torch
import argparse
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_path = os.getenv('MODEL_CONFIG_PATH')
# model_path = '/research/projects/DanielKaiser/RSNA_Inpainting/outputs/pl/epoch=97-step=238238-val_loss=0.0014.ckpt'
model_path = os.getenv('MODEL_PATH')

model = prepare_model(config_path, model_path, device=device)


def create_args(image_path, square_length, inpaint_parameters, num_images=1):
    args = argparse.Namespace(
        use_cpu=False,
        num_images=num_images,
        square_length=square_length,
        divisibility_factor=3,
        resample_steps=inpaint_parameters[1],
        inference_protocol="DDIM20",
        average=1,
        batch_size=1,
        jump_length=inpaint_parameters[2],
        start_denoise_step=None,
        preprocessed_dir=os.path.dirname(image_path),
        output_dir=os.path.dirname(image_path)
    )
    return args


def inpaint_square(image_path, square, mask, img_size, offset, img_index, inpaint_parameters):
    contour_path, edge_path = get_contour_path(image_path)
    inpainted_square = get_inpainted_square(square, image_path, contour_path, edge_path, mask, img_size=img_size, inpaint_parameters=inpaint_parameters)
    
    grid_key, square_key = get_keys(square, offset)
    update_inpainted_square(img_index, grid_key, square_key, inpainted_square=Image.fromarray(inpainted_square), inpaint_parameters=inpaint_parameters)
    st.session_state['show_inpainted_square'] = True

def inpaint_grid(image_path, img_size, square, offset, img_index, inpaint_parameters):
    contour_path, edge_path, left_path, right_path = get_contour_path(image_path)
    square_length = square[2]
    all_inpainted_squares = get_inpainted_image_squares(image_path, contour_path, edge_path, left_path, right_path, img_index, img_size, square_length, offset, inpaint_parameters)

    update_inpainted_squares(all_inpainted_squares, inpaint_parameters=inpaint_parameters)
    st.session_state['show_inpainted_square'] = True

def inpaint_series(series, series_image_paths, square_lengths, img_index, img_size, inpaint_parameters=None):
    import torch

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)} (Device ID: {device_id})")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device_id) / 1024 ** 3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device_id) / 1024 ** 3:.2f} GB")
    else:
        print("No GPU available, using CPU.")
    batch_size = 100
    batch_images, batch_paths, batch_masks, batch_params = [], [], [], []
    for img_index in range(len(series)):
        for square_length in square_lengths:
            for offset in range(4):
                for i in range(3):
                    for j in range(3):
                        image, image_path = series[img_index], series_image_paths[img_index]
                        mask = st.session_state['masks'][square_length][(i % 3 + 3 * (j % 3)) * 4 + offset]

                        batch_images.append(image)
                        batch_paths.append(image_path)
                        batch_masks.append(mask)
                        batch_params.append((i, j, square_length, offset, img_index))
                        if len(batch_images) >= batch_size or (img_index == len(series) - 1 and square_length == square_lengths[-1] and offset == 3 and i == 2 and j == 2):
                            inpaint_batch(batch_images, batch_paths, batch_masks, batch_params, img_index, img_size, inpaint_parameters)
                            batch_images, batch_paths, batch_masks, batch_params = [], [], [], []

    st.session_state['show_inpainted_square'] = True

def inpaint_batch(batch_images, batch_paths, batch_masks, batch_params, img_index, img_size, inpaint_parameters):
    concat_paths = [get_contour_path(image_path) for image_path in batch_paths]
    contour_paths, edge_paths, left_paths, right_paths = [path[0] for path in concat_paths], [path[1] for path in concat_paths], [path[2] for path in concat_paths], [path[3] for path in concat_paths]
    batch_inpainted_squares = get_inpainted_batch_squares(batch_images, batch_paths, contour_paths, edge_paths, left_paths, right_paths, batch_masks, batch_params, img_size, inpaint_parameters)

    update_inpainted_squares(batch_inpainted_squares, inpaint_parameters=inpaint_parameters)
    st.session_state['show_inpainted_square'] = True


def get_inpainted_square(square, image_path, contour_path, edge_path, grid_mask, img_size=256, inpaint_parameters=None):
    x, y, square_length = square
    args = create_args(image_path, square_length, inpaint_parameters)

    # Preprocess image and mask
    grid_mask = torch.tensor(grid_mask).transpose(1, 0)
    img_tensors, concat_tensors, img_ids = preprocess_images([image_path], [contour_path], [edge_path], args, img_size=512, resize_size=img_size, grid=grid_mask)

    # Inpaint image
    grid_mask = grid_mask.unsqueeze(0).unsqueeze(0).float()
    inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_mask, args, noise_shape=(1, img_size, img_size), device=device, inpaint_parameters=inpaint_parameters)

    # Convert the inpainted image tensor to a numpy array
    inpainted_square = inpainted_imgs[0].cpu().numpy().transpose(2, 1, 0).squeeze(2)
    inpainted_square = ((inpainted_square + 1) * 127.5).astype(np.uint8)
    inpainted_square = inpainted_square[y:y + square_length, x:x + square_length]
    
    return inpainted_square

def get_inpainted_image_squares(image_path, contour_path, edge_path, left_path, right_path, img_index, img_size, square_length, offset, inpaint_parameters):
    args = create_args(image_path, square_length, inpaint_parameters)

    inpainted_mask_squares = []
    for i in range(3):
        for j in range(3):
            # Select the correct grid
            grid_mask = st.session_state['masks'][square_length][(i % 3 + 3 * (j % 3)) * 4 + offset]
             # Preprocess image and mask
            grid_mask = torch.tensor(grid_mask).transpose(1, 0)
            img_tensors, concat_tensors, img_ids = preprocess_images([image_path], [contour_path], [edge_path], [left_path], [right_path], args, img_size=512, resize_size=img_size, grid=grid_mask)

            # Inpaint image
            grid_mask = grid_mask.unsqueeze(0).unsqueeze(0).float()
            inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_mask, args, noise_shape=(1, img_size, img_size), device=device)

            inpainted_img = inpainted_imgs[0].cpu().numpy().transpose(2, 1, 0).squeeze(2)
            inpainted_img = ((inpainted_img + 1) / 2 * 255).astype(np.uint8)

            inpainted_mask_squares.extend(extract_inpainted_squares(i, j, square_length, offset, img_index, img_size, inpainted_img))

    return inpainted_mask_squares

def get_inpainted_batch_squares(batch_images, batch_paths, contour_paths, edge_paths, left_paths, right_paths, batch_masks, batch_params, img_size, inpaint_parameters):
    args = create_args(batch_paths[0], None, inpaint_parameters, num_images=len(batch_images))
    grid_masks = torch.stack([torch.tensor(mask).transpose(1, 0) for mask in batch_masks], dim=0)
    img_tensors, concat_tensors, img_ids = preprocess_images(batch_paths, contour_paths, edge_paths, left_paths, right_paths, args, img_size=512, resize_size=img_size, grid=grid_masks)
    grid_masks = grid_masks.unsqueeze(1).float()

    inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_masks, args, noise_shape=(1, img_size, img_size), device=device)

    inpainted_mask_squares = []
    for i in range(len(batch_images)):
        inpainted_img = inpainted_imgs[i].cpu().numpy().transpose(2, 1, 0).squeeze(2)
        inpainted_img = ((inpainted_img + 1) / 2 * 255).astype(np.uint8)
        i, j, square_length, offset, img_index = batch_params[i]
        inpainted_mask_squares.extend(extract_inpainted_squares(i, j, square_length, offset, img_index, img_size, inpainted_img))
    return inpainted_mask_squares 
            
def extract_inpainted_squares(i, j, square_length, offset, img_index, img_size, inpainted_img):
    inpainted_mask_squares = []
    dx, dy = (offset % 2) * square_length // 2, (offset // 2) * square_length // 2
    for y in range(i * square_length, img_size, 3 * square_length):
        for x in range(j * square_length, img_size, 3 * square_length):
            x_off = x + dy
            y_off = y + dx
            if x_off + square_length <= img_size and y_off + square_length <= img_size:
                inpainted_square = inpainted_img[x_off:x_off + square_length, y_off:y_off + square_length]
                inpainted_mask_squares.append((inpainted_square, y_off, x_off, square_length, offset, img_index))
    return inpainted_mask_squares

def update_inpainted_squares(all_inpainted_squares, inpaint_parameters):
    for inpainted_square, inpainted_x, inpainted_y, square_length, offset, img_index in all_inpainted_squares:
        grid_key, square_key = (square_length, offset), (inpainted_x, inpainted_y)
        update_inpainted_square(img_index, grid_key, square_key, inpainted_square=Image.fromarray(inpainted_square), inpaint_parameters=inpaint_parameters)