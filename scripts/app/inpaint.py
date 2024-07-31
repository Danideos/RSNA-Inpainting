from scripts.predict import prepare_model, preprocess_images, inpaint_images
import torch
import argparse
import os
import numpy as np
from PIL import Image


def get_inpainted_square(square, image_path, contour_path, grid_mask, img_size=256, inpaint_parameters=None):
    x, y, square_length = square
    args = argparse.Namespace(
        use_cpu=False,
        num_images=1,
        square_length=square_length,
        divisibility_factor=3,
        resample_steps=inpaint_parameters,
        inference_protocol="DDIM100",
        average=1,
        batch_size=1,
        jump_length=inpaint_parameters,
        preprocessed_dir=os.path.dirname(image_path),
        output_dir=os.path.dirname(image_path)
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = "/research/projects/DanielKaiser/RSNA_Inpainting/config.yaml"  # Update with the actual path
    model_path = "/research/projects/DanielKaiser/RSNA_Inpainting/outputs/pl/epoch=0-step=30000-val_loss=0.0015.ckpt" # /home/bje01/Documents/inpainting/outputs/pl/epoch=0-step=111000-val_loss=0.0014.ckpt"  # Update with the actual path

    model = prepare_model(config_path, model_path, device=device)

    # Preprocess image and mask
    grid_mask = torch.tensor(grid_mask).transpose(1, 0)
    img_tensors, concat_tensors, img_ids = preprocess_images([image_path], [contour_path], args, img_size=512, resize_size=img_size, grid=grid_mask)


    # Inpaint image
    grid_mask = grid_mask.unsqueeze(0).unsqueeze(0).float()
    inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_mask, args, noise_shape=(1, img_size, img_size), device=device, inpaint_parameters=inpaint_parameters)

    # Convert the inpainted image tensor to a numpy array
    inpainted_square = inpainted_imgs[0].cpu().numpy().transpose(2, 1, 0).squeeze(2)
    inpainted_square = ((inpainted_square + 1) / 2 * 255).astype(np.uint8)
    inpainted_square = inpainted_square[x:x + square_length, y:y + square_length]
    
    return inpainted_square

def inpaint_square(st, image_path, square, mask, img_size, inpaint_parameters):
    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name
    inpainted_square = get_inpainted_square(square, image_path, contour_path, mask, img_size, inpaint_parameters=inpaint_parameters)
    if inpainted_square is not None:
        inpainted_x, inpainted_y, _ = square
        inpainted_square_image = Image.fromarray(inpainted_square)
        square_length = square[2]
        key = (inpainted_x, inpainted_y, square_length)
        st.session_state['all_inpainted_square_images'][key] = inpainted_square_image
        st.session_state['show_inpainted_square'] = True

def inpaint_grid(st, image_path, img_size, masks, square, offset, inpaint_parameters):
    file_name = os.path.basename(image_path)
    contour_path = "/".join(image_path.split("/")[:-2]) + "/mask_png/" + file_name
    all_inpainted_squares = get_inpainted_image_squares(image_path, contour_path, masks, img_size, square[2], offset, inpaint_parameters)

    for inpainted_square, x_off, y_off, square_length in all_inpainted_squares:
        key = (x_off, y_off, square_length)
        print("storing key:", key)
        st.session_state['all_inpainted_square_images'][key] = Image.fromarray(inpainted_square)

      

def get_inpainted_image_squares(image_path, contour_path, grid_masks, img_size, square_length, offset, inpaint_parameters):
    args = argparse.Namespace(
        use_cpu=False,
        num_images=1,
        square_length=square_length,
        divisibility_factor=3,
        resample_steps=inpaint_parameters,
        inference_protocol="DDIM100",
        average=1,
        batch_size=1,
        jump_length=inpaint_parameters,
        preprocessed_dir=os.path.dirname(image_path),
        output_dir=os.path.dirname(image_path)
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    config_path = "/research/projects/DanielKaiser/RSNA_Inpainting/config.yaml"  # Update with the actual path
    model_path = "/research/projects/DanielKaiser/RSNA_Inpainting/outputs/pl/epoch=0-step=30000-val_loss=0.0015.ckpt" # /home/bje01/Documents/inpainting/outputs/pl/epoch=0-step=111000-val_loss=0.0014.ckpt"  # Update with the actual path

    model = prepare_model(config_path, model_path, device=device)

    inpainted_mask_squares = []
    for i in range(3):
        for j in range(3):
            # Select the correct grid
            grid_mask = grid_masks[square_length][(i % 3 + 3 * (j % 3)) * 4 + offset]
             # Preprocess image and mask
            grid_mask = torch.tensor(grid_mask).transpose(1, 0)
            img_tensors, concat_tensors, img_ids = preprocess_images([image_path], [contour_path], args, img_size=512, resize_size=img_size, grid=grid_mask)

            # Inpaint image
            grid_mask = grid_mask.unsqueeze(0).unsqueeze(0).float()
            inpainted_imgs = inpaint_images(model, img_tensors, concat_tensors, grid_mask, args, noise_shape=(1, img_size, img_size), device=device, inpaint_parameters=inpaint_parameters)

            inpainted_img = inpainted_imgs[0].cpu().numpy().transpose(2, 1, 0).squeeze(2)
            inpainted_img = ((inpainted_img + 1) / 2 * 255).astype(np.uint8)

            # Extract the inpainted squars
            dx, dy = (offset % 2) * square_length, (offset // 2) * square_length
            for x in range(i * square_length, img_size, 3 * square_length):
                for y in range(j * square_length, img_size, 3 * square_length):
                    x_off = x + dx
                    y_off = y + dy
                    if x_off + square_length <= img_size and y_off + square_length <= img_size:
                        inpainted_square = inpainted_img[x_off:x_off + square_length, y_off:y_off + square_length]
                        inpainted_mask_squares.append((inpainted_square, x_off, y_off, square_length))

    return inpainted_mask_squares

