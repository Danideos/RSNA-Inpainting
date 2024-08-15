
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


def create_args(image_path, square_length, inpaint_parameters):
    args = argparse.Namespace(
        use_cpu=False,
        num_images=1,
        square_length=square_length,
        divisibility_factor=3,
        resample_steps=inpaint_parameters[1],
        inference_protocol="DDIM100",
        average=1,
        batch_size=1,
        jump_length=inpaint_parameters[2],
        start_denoise_step=inpaint_parameters[0],
        preprocessed_dir=os.path.dirname(image_path),
        output_dir=os.path.dirname(image_path)
    )
    return args


def inpaint_square(image_path, square, mask, img_size, offset, img_index, inpaint_parameters):
    contour_path = get_contour_path(image_path)
    inpainted_square = get_inpainted_square(square, image_path, contour_path, mask, img_size=img_size, inpaint_parameters=inpaint_parameters)
    
    grid_key, square_key = get_keys(square, offset)
    update_inpainted_square(img_index, grid_key, square_key, inpainted_square=Image.fromarray(inpainted_square), inpaint_parameters=inpaint_parameters)
    st.session_state['show_inpainted_square'] = True

def inpaint_grid(image_path, img_size, square, offset, img_index, inpaint_parameters):
    contour_path = get_contour_path(image_path)
    square_length = square[2]
    all_inpainted_squares = get_inpainted_image_squares(image_path, contour_path, img_size, square_length, offset, inpaint_parameters)

    grid_key, _ = get_keys(square, offset)
    for inpainted_square, inpainted_x, inpainted_y, square_length, offset in all_inpainted_squares:
        square_key = (inpainted_x, inpainted_y)
        update_inpainted_square(img_index, grid_key, square_key, inpainted_square=Image.fromarray(inpainted_square), inpaint_parameters=inpaint_parameters)
    st.session_state['show_inpainted_square'] = True

def get_inpainted_square(square, image_path, contour_path, grid_mask, img_size=256, inpaint_parameters=None):
    x, y, square_length = square
    args = create_args(image_path, square_length, inpaint_parameters)

    # Preprocess image and mask
    grid_mask = torch.tensor(grid_mask).transpose(1, 0)
    img_tensors, concat_tensors, img_ids = preprocess_images([image_path], [contour_path], args, img_size=512, resize_size=img_size, grid=grid_mask, square_length=square_length)

    # Inpaint image
    grid_mask = grid_mask.unsqueeze(0).unsqueeze(0).float()
    inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_mask, args, noise_shape=(1, img_size, img_size), device=device, inpaint_parameters=inpaint_parameters)

    # Convert the inpainted image tensor to a numpy array
    inpainted_square = inpainted_imgs[0].cpu().numpy().transpose(2, 1, 0).squeeze(2)
    inpainted_square = ((inpainted_square + 1) * 127.5).astype(np.uint8)
    inpainted_square = inpainted_square[y:y + square_length, x:x + square_length]
    
    return inpainted_square

def get_inpainted_image_squares(image_path, contour_path, img_size, square_length, offset, inpaint_parameters):
    args = create_args(image_path, square_length, inpaint_parameters)

    inpainted_mask_squares = []
    for i in range(3):
        for j in range(3):
            # Select the correct grid
            grid_mask = st.session_state['masks'][square_length][(i % 3 + 3 * (j % 3)) * 4 + offset]
             # Preprocess image and mask
            grid_mask = torch.tensor(grid_mask).transpose(1, 0)
            img_tensors, concat_tensors, img_ids = preprocess_images([image_path], [contour_path], args, img_size=512, resize_size=img_size, grid=grid_mask, square_length=square_length)

            # Inpaint image
            grid_mask = grid_mask.unsqueeze(0).unsqueeze(0).float()
            inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_mask, args, noise_shape=(1, img_size, img_size), device=device)

            inpainted_img = inpainted_imgs[0].cpu().numpy().transpose(2, 1, 0).squeeze(2)
            inpainted_img = ((inpainted_img + 1) / 2 * 255).astype(np.uint8)

            # Extract the inpainted squares
            dx, dy = (offset % 2) * square_length // 2, (offset // 2) * square_length // 2
            for y in range(i * square_length, img_size, 3 * square_length):
                for x in range(j * square_length, img_size, 3 * square_length):
                    x_off = x + dy
                    y_off = y + dx
                    if x_off + square_length <= img_size and y_off + square_length <= img_size:
                        inpainted_square = inpainted_img[x_off:x_off + square_length, y_off:y_off + square_length]
                        inpainted_mask_squares.append((inpainted_square, y_off, x_off, square_length, offset))

    return inpainted_mask_squares
            
