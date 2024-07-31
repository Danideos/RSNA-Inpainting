# Preprocessing steps
Data is split into `stage_2_train` and `stage_2_test` folders of .dcm CT brain images, train split having `stage_2_train.csv` hemorrhage type labelling. 

Filtering is performed for only the hemorrhage free (label 0) images as we want to inpaint healthy regions of the brain. Images are then processed to capture only the brain window intensities as in this project subdural and skeletal windows are irrelevant. To ignore useless images containing no relevant information after this preprocessing, thresholding for proportion of pixels in midrange intensities is applied. 

For training, images are resized with padding or cropping to maintain original proportions and basic augmentations (horizontal flip, rotation from -15 to 15 deg) are applied. 

### filter_data.py
Specify `--csv_path` containing train labels, `--raw_path` containing raw .dcm train data, `--filtered_path` directory where only healthy .dcm images from train directory will be stored, `--fraction=1.0` representing fraction of healthy images to keep(for experimental purposes), `--n_jobs=-1` specifying amount of cores to use for filtering.

### prepare_data.py
Specify `--dcm_path` containing healthy .dcm images, `--png_path` where brain windowed and midrange thresholded images will be stored. Choice of .png, which has only 8bit encoding, because of using small brain intensity window so smaller bit size should have no impact on precision.

# Training

### train.py
Specify `--png_path` where images for training are stored, dataset is created with `./scripts/dataset_creation.py` applying desired transforms. Use mediffusion github repo for furher usage, with the `config.yaml` additions:

| Section        | Field                   | Description                                  |
|----------------|-------------------------|----------------------------------------------|
| model          | `input_size`             | Size that images should be resized to for training(same as `img_size` to keep the size) |
| hyperparameters| `img_size`               | Size of the images used in the model (in pixels). |
| hyperparameters| `total_image_seen`       | Total number of images the model has seen during training. |
| hyperparameters| `batch_size`             | Number of images processed in a single batch. |
| hyperparameters| `accumulate_grad_batches`| Number of batches for which gradients are accumulated before performing a backward pass. |
| hyperparameters| `num_devices`            | Number of devices used for training. |
