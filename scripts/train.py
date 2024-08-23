from mediffusion import DiffusionModule, Trainer
from dataset.dataset_creation import create_dataset, get_datasamplers
from utils import load_config, visualize_random_dataset_samples
import os
import torch
import argparse

torch.set_float32_matmul_precision('medium')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
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


def train(input_dir, mask_dir=None):    
    data_dir = os.path.join(input_dir, "bet_png")
    mask_dir = os.path.join(input_dir, "mask_png")
    edge_dir = os.path.join(input_dir, "edge_png")
    left_dir = os.path.join(input_dir, "left_png")
    right_dir = os.path.join(input_dir, "right_png")
    train_ds, val_ds = create_dataset(data_dir, mask_dir, edge_dir, left_dir, right_dir, IMG_SIZE, RESIZE_SIZE)
    train_sampler, val_sampler = get_datasamplers(train_ds, val_ds, TOTAL_IMAGE_SEEN)
    print(f"train dataset size: {len(train_ds)}")
    torch.cuda.empty_cache()

    model = DiffusionModule(
        "./config.yaml",
        train_ds=train_ds,
        val_ds=val_ds,
        dl_workers=8,
        train_sampler=train_sampler,
        batch_size=BATCH_SIZE,
        val_batch_size=max(1, BATCH_SIZE // 2),
        with_condition=True,
    )
    # model_path = "/home/bje01/Documents/RSNA-Inpainting/outputs/pl/cranial_ct_inpainting-epoch=0-step=60000-val_loss=0.001024.ckpt"
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
    # model.load_ckpt(model_path, ema=True)
    model.cuda()

    trainer = Trainer(
        max_steps=TRAIN_ITERATIONS,
        val_check_interval=1, # Epochs, not steps now, for DDP protocol
        root_directory="./outputs/",
        precision="16-mixed",
        devices=-1,
        nodes=1,
        wandb_project="cranial_ct_inpainting",
        logger_instance="EM(FEv3,G5,T1.5)_2A100",
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES
    )

    # visualize_random_dataset_samples(train_ds, num_samples=5)
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-input_dir", "--input_dir", type=str)
    args = parser.parse_args()

    train(args.input_dir)
