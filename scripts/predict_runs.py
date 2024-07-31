import subprocess
import yaml
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run_inpainting(run_config):
    command = [
        "python", "/research/projects/DanielKaiser/RSNA_Inpainting/scripts/predict.py",  # replace "inpainting_script.py" with the actual name of your inpainting script
        "--preprocessed_dir", run_config["preprocessed_dir"],
        "--output_dir", run_config["output_dir"],
        "--num_images", str(run_config["num_images"]),
        "--square_length", str(run_config["square_length"]),
        "--divisibility_factor", str(run_config["divisibility_factor"]),
        "--resample_steps", str(run_config["resample_steps"]),
        "--inference_protocol", run_config["inference_protocol"],
        "--average", str(run_config["average"]),
        "--batch_size", str(run_config["batch_size"]),
        "--jump_length", str(run_config["jump_length"])
    ]

    if run_config["use_cpu"]:
        command.append("--use_cpu")
    if run_config["offset"]:
        command.append("--offset")

    subprocess.run(command)

def main():
    with open("/research/projects/DanielKaiser/RSNA_Inpainting/predict_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    for run_config in config["runs"]:
        run_inpainting(run_config)

if __name__ == "__main__":
    main()