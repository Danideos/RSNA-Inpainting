from libs.Mediffusion_Fork.ddpm import DiffusionModule
from libs.Mediffusion_Fork.trainer import Trainer
from dataset.dataset_creation import create_dataset, get_datasamplers
from utils import load_config, visualize_random_dataset_samples

import os
import torch
import argparse

torch.set_float32_matmul_precision('medium')

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ['WANDB_API_KEY'] = "1ad7e01bcd34b7a32fbc85cfe575bb29cf1b3e5c"   

# Load configuration
CONFIG_PATH = "./config.yaml"
config = load_config(CONFIG_PATH)

# Extract parameters
hyperparams = config['hyperparameters']
TOTAL_IMAGE_SEEN = int(float(hyperparams['total_image_seen']))
BATCH_SIZE = hyperparams['batch_size']
NUM_DEVICES = hyperparams['num_devices']
IMG_SIZE = hyperparams['img_size']
ACCUMULATE_GRAD_BATCHES = hyperparams["accumulate_grad_batches"]

TRAIN_ITERATIONS = int(TOTAL_IMAGE_SEEN / (BATCH_SIZE * NUM_DEVICES))

model_params = config['model']
RESIZE_SIZE = model_params['input_size'] 


def create_file_dirs(input_dir, dir_names=["bet_png"]):
    file_dirs = [os.path.join(input_dir, dir_name) for dir_name in dir_names]
    return file_dirs

def train(input_dir, resume=False):    
    dir_names=["bet_png", "ano_png", "left_png", "right_png"]
    file_dirs = create_file_dirs(input_dir, dir_names=dir_names)
    train_ds, val_ds = create_dataset(file_dirs, required=dir_names, img_size=IMG_SIZE, resize_size=RESIZE_SIZE)# 
    train_sampler, val_sampler = get_datasamplers(train_ds, val_ds, TOTAL_IMAGE_SEEN)
    print(f"train dataset size: {len(train_ds)}")
    torch.cuda.empty_cache()

    if resume is False:
        model = DiffusionModule(
            "./config.yaml",
            train_ds=train_ds,
            val_ds=val_ds,
            dl_workers=12,
            train_sampler=train_sampler,
            batch_size=BATCH_SIZE,
            val_batch_size=max(1, BATCH_SIZE // 2),
            with_condition=True,
        )
    else:
        model_path = "/research/projects/DanielKaiser/RSNA_Inpainting/outputs/pl/EM(FE,G5,T1.5)_2A100-epoch=7-step=36472-val_loss=0.000855.ckpt"
        model = DiffusionModule(
            "./config.yaml",
            train_ds=train_ds,
            val_ds=val_ds,
            dl_workers=16,
            train_sampler=train_sampler,
            batch_size=BATCH_SIZE,
            val_batch_size=max(1, BATCH_SIZE // 2),
            with_condition=True,
        )
        model.load_ckpt(model_path, ema=True)
    model.cuda()

    trainer = Trainer(
        max_steps=TRAIN_ITERATIONS,
        val_check_interval=1, # Epochs, not steps now, for DDP protocol
        root_directory="./outputs/",
        precision="16-mixed",
        devices=-1,
        nodes=1,
        wandb_project="cranial_ct_inpainting",
        logger_instance="AnoRemove+2.5D_2A100",
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES
    )
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-r", "--resume", action='store_true')

    args = parser.parse_args()

    train(args.input_dir, args.resume)
