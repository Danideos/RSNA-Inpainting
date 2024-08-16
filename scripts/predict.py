import os
import argparse
import lpips
import torch
import monai.transforms as mt
import numpy as np
from glob import glob
from utils import add_channel
from utils import create_divisible_masks, prepare_model, lambda_transform_with_grid
# from save_images import save_image, save_difference_image

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class LambdaTransformWithOptionalGrid(mt.Transform):
    def __init__(self, func, grid):
        self.func = func
        self.grid = grid

    def __call__(self, data):
        current_grid = self.grid  

        if len(self.grid.shape) > 2 and 'index' in data:
            current_grid = self.grid[data['index']]  

        data = self.func(data, current_grid)
        return data

def preprocess_images(png_paths, mask_paths, edge_paths, args, img_size=512, resize_size=256, grid=None):
    device = 'cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu'
    data_transforms = mt.Compose([
        mt.LoadImageD(keys=["img", "concat"]),
        mt.LambdaD(keys=["img", "concat"], func=lambda x: add_channel(x)),
        mt.ResizeWithPadOrCropD(keys=["img", "concat"], spatial_size=(img_size, img_size)), 
        mt.ResizeD(keys=["img", "concat"], spatial_size=(resize_size, resize_size)), 
        LambdaTransformWithOptionalGrid(func=lambda_transform_with_grid, grid=grid),
        # mt.ScaleIntensityD(keys=["img", "concat"], minv=-1, maxv=1),
        mt.ToTensorD(keys=["img", "concat"], dtype=torch.float, track_meta=False),
    ])
    png_imgs = [{"img": png_paths[i], "concat": (mask_paths[i], edge_paths[i]), "index": i} for i in range(args.num_images)]
    img_ids = [os.path.basename(png_path).split(".")[0] for png_path in png_paths[:args.num_images]]
    transformed_imgs = data_transforms(png_imgs)
    img_tensors = torch.stack([img_dict["img"] for img_dict in transformed_imgs], dim=0).to(device)
    mask_tensors = torch.stack([img_dict["concat"] for img_dict in transformed_imgs], dim=0).to(device)
    return img_tensors, mask_tensors, img_ids

def inpaint_images(model, img_tensors, mask_tensors, masks, args, noise_shape, device="cpu"):
    batch_size = 100
    inpainted_imgs = []
    for i in range(0, len(img_tensors), batch_size):
        max_index = min(len(img_tensors), i + batch_size)
        original_imgs = img_tensors[i:max_index].clone().detach().to(device).half() if device == 'cuda' else img_tensors[i:max_index].clone().detach().to(device).float()
        inp_noise = torch.randn((max_index - i, *noise_shape), device=device).half() if device == 'cuda' else torch.randn((max_index - i, *noise_shape), device=device).float()
        mask_batch = masks[i:max_index].to(device).half() if device == 'cuda' else masks[i:max_index].to(device).float()
        mask_tensors_batch = mask_tensors[i:max_index].to(device).half() if device == 'cuda' else mask_tensors[i:max_index].to(device).float()
      
        inpainted_batch = model.predict(
            inp_noise,
            model_kwargs={"concat": mask_tensors_batch},
            inference_protocol=args.inference_protocol,
            mask=mask_batch,
            original_image=original_imgs,
            resampling_steps=args.resample_steps,
            jump_length=args.jump_length,
            start_denoise_step=args.start_denoise_step
        )
     
        inpainted_imgs.extend(inpainted_batch)
    
    return inpainted_imgs

def process_tensors(img_tensors, mask_tensors, model, img_ids, args, device="cpu"):
    image_shape = img_tensors.shape
    masks = create_divisible_masks(image_shape, args.square_length, args.divisibility_factor, device=device)
    channels, height, width = img_tensors.shape[1:]

    out_relative_dir = f"SqL{args.square_length}_DF{args.divisibility_factor}_RS{args.resample_steps}_Avg{args.average}_JL{args.jump_length}_IP{args.inference_protocol[4:]}_{args.preprocessed_dir.split('/')[-3]}"

    batch_size = args.batch_size
    average = args.average
    for i in range(0, len(img_tensors), batch_size):
        max_index = min(len(img_tensors), i + batch_size)
        img_batch = img_tensors[i:max_index]
        mask_tensors_batch = mask_tensors[i:max_index]
        img_ids_batch = img_ids[i:max_index]

        for _ in range(average):
            torch.manual_seed(_ + 1)
            all_inpainted_imgs = [[] for _ in range(len(img_batch))]
            for mask_index, mask in enumerate(masks):
                mask_batch = mask[i:max_index]

                inpainted_imgs = inpaint_images(model, img_batch, mask_tensors_batch, mask_batch, args, noise_shape=(channels, height, width), device=device)
                for j in range(len(inpainted_imgs)):
                    all_inpainted_imgs[j].append(inpainted_imgs[j].to(device))
                    if _ != average - 1:
                        continue

                    output_dir = os.path.join(args.output_dir, out_relative_dir, img_ids_batch[j])
                    os.makedirs(output_dir, exist_ok=True)
                    output_img_path = os.path.join(output_dir, f"mask_{mask_index}.png")
                    # save_image(img_batch[j], inpainted_imgs[j], mask_batch[j], output_img_path, mask_index)

        for j in range(len(img_batch)):
            output_dir = os.path.join(args.output_dir, out_relative_dir, img_ids_batch[j])
            difference_img_path_rb = os.path.join(output_dir, "difference_rb.png")
            # save_difference_image(img_batch[j].to(device), all_inpainted_imgs[j], [masks[k][j].to(device) for k in range(len(masks))], difference_img_path_rb, device)
            lpips_scores = compute_metrics(img_batch[j].to(device), all_inpainted_imgs[j])
            # save_metrics(lpips_scores, output_dir)

def compute_metrics(original_img, inpainted_imgs):
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net='alex')

    # Ensure images are in the correct format (1, C, H, W)
    original_img = original_img.unsqueeze(0)
    if original_img.size(1) == 1:
        original_img = original_img.repeat(1, 3, 1, 1)  # Convert to 3 channels

    lpips_scores = []
    for inpainted_img in inpainted_imgs:
        inpainted_img = inpainted_img.unsqueeze(0)
        if inpainted_img.size(1) == 1:
            inpainted_img = inpainted_img.repeat(1, 3, 1, 1)  # Convert to 3 channels

        # Compute LPIPS
        lpips_score = loss_fn(original_img, inpainted_img)
        lpips_scores.append(lpips_score.item())

    return lpips_scores

def main(args, img_size=512, resize_size=256):
    assert(args.output_dir is not None)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = "/home/bje01/Documents/inpainting/config.yaml"
    model_path = "/home/bje01/Documents/inpainting/outputs/pl/last.ckpt"
    device = 'cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu'
    model = prepare_model(config_path, model_path, device=device)
    print("Model loaded successfully!")
    
    png_paths = sorted(glob(os.path.join(os.path.join(args.preprocessed_dir, "bet_png"), "*.png")))
    mask_paths = sorted(glob(os.path.join(os.path.join(args.preprocessed_dir, "mask_png"), "*.png")))
    img_tensors, mask_tensors, img_ids = preprocess_images(png_paths, mask_paths, args, img_size, resize_size)

    process_tensors(img_tensors, mask_tensors, model, img_ids, args, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpainting with Diffusion Model")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory to save/load the png bet/mask images") #
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images") #
    parser.add_argument("--num_images", type=int, default=20, help="Number of images to inpaint") #
    parser.add_argument("--use_cpu", action='store_true', help="Use CPU instead of GPU") #
    parser.add_argument("--square_length", type=int, default=16, help="Length of the square mask") #
    parser.add_argument("--divisibility_factor", type=int, default=2, help="Divisibility factor for the square mask") #
    parser.add_argument("--resample_steps", type=int, default=3, help="Number of resampling steps") #
    parser.add_argument("--inference_protocol", type=str, default="DDIM100", help="Inference protocol for inpainting") #
    parser.add_argument("--average", type=int, default=4, help="Number of inpainting averages") #
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inpainting") #
    parser.add_argument("--jump_length", type=int, default=1, help="Jump length for inpainting") #
    args = parser.parse_args()

    main(args)
