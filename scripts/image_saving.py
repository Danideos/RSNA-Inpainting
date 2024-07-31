import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def save_inpainted_image(original_img, inpainted_img, mask, output_img_path, mask_index):
    original_img = original_img.permute(1, 2, 0).squeeze(2).cpu().numpy()
    inpainted_img = inpainted_img.permute(1, 2, 0).squeeze(2).cpu().numpy()
    mask = mask.permute(1, 2, 0).squeeze(2).cpu().numpy()
    
    inpainted_with_mask = inpainted_img * (1 - mask) + original_img * mask * 0.1

    concatenated_img = np.concatenate((original_img, inpainted_img, inpainted_with_mask), axis=1)

    plt.imsave(output_img_path, concatenated_img, cmap="gray")

def save_difference_image(original_img, inpainted_images, masks, output_img_path_rgb, output_img_path_rb, device):
    combined_inpainted = torch.zeros_like(original_img, device=device)
    mask_count = torch.zeros_like(original_img, device=device)
    
    for inpainted_img, mask in zip(inpainted_images, masks):
        combined_inpainted += inpainted_img.to(device) * mask.to(device)
        mask_count += mask.to(device)

    # Prevent division by zero by setting any zero entries in mask_count to 1
    mask_count[mask_count == 0] = 1
    
    # Average the combined inpainted image by the number of contributing masks
    combined_inpainted /= mask_count
    combined_inpainted_np = (combined_inpainted.squeeze().cpu().numpy() + 1) / 2

    difference_img = original_img - combined_inpainted
    difference_img_np = difference_img.squeeze().cpu().numpy() / 2
    
    original_img_np = (original_img.squeeze().cpu().numpy() + 1) / 2

    # Create a visualization of the differences for RB
    diff_visualization_rb = np.zeros((difference_img_np.shape[0], difference_img_np.shape[1], 3))
    diff_visualization_rb[difference_img_np > 0, 0] = 1.0  # Red for positive differences 
    diff_visualization_rb[:, :, 0] *= np.abs(difference_img_np)
    diff_visualization_rb[difference_img_np < 0, 2] = 1.0  # Blue for negative differences
    diff_visualization_rb[:, :, 2] *= np.abs(difference_img_np)

    min_val = np.min(diff_visualization_rb)
    max_val = np.max(diff_visualization_rb)
    if max_val - min_val > 0:
        diff_visualization_rb = (diff_visualization_rb - min_val) / (max_val - min_val)

    # Save the RB visualization image
    plt.imsave(output_img_path_rb, diff_visualization_rb)

    return original_img_np, combined_inpainted_np

def save_histogram(original_img_np, combined_inpainted_np, mask_tensors, output_histogram_path, device):
    # Rescale the images to HU (0 to 80)
    original_img_hu = original_img_np * 80
    inpainted_img_hu = combined_inpainted_np * 80

    # Use the mask tensors to ignore regions outside the brain
    brain_mask = mask_tensors.squeeze().cpu().numpy() > 0

    # Create histograms for the original and inpainted images inside the brain region
    original_img_hu_brain = original_img_hu[brain_mask]
    inpainted_img_hu_brain = inpainted_img_hu[brain_mask]

    plt.figure()
    plt.hist(original_img_hu_brain.flatten(), bins=80, range=(0, 80), alpha=0.5, label='Original')
    plt.hist(inpainted_img_hu_brain.flatten(), bins=80, range=(0, 80), alpha=0.5, label='Inpainted')
    plt.xlabel('HU Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of HU Intensities in Brain Region')
    plt.savefig(output_histogram_path)
    plt.close()

    original_hist, _ = np.histogram(original_img_hu_brain.flatten(), bins=80, range=(0, 80))
    inpainted_hist, _ = np.histogram(inpainted_img_hu_brain.flatten(), bins=80, range=(0, 80))
    
    return original_hist, inpainted_hist

def save_summary_metrics(output_dir, metrics_summary):
    avg_kl_div = np.mean(metrics_summary['kl_div_values'])
    avg_ssim = np.mean(metrics_summary['ssim_values'])
    avg_emd = np.mean(metrics_summary["emd_values"])
    avg_lpips = np.mean(metrics_summary["lpips_values"])

    log_file_path = os.path.join(output_dir, 'log.txt')
    with open(log_file_path, 'w') as log_file:  # Open in write mode to ensure a new log file
        log_file.write('Individual Statistics:\n')
        log_file.write('Image ID, KL Divergence, SSIM, EMD, LPIPS\n')
        for img_id, kl_div, ssim_val, emd_value, lpips_value in metrics_summary['individual_stats']:
            log_file.write(f'{img_id}, {kl_div}, {ssim_val}, {emd_value}, {lpips_value}\n')
        
        log_file.write('\nSummary Statistics:\n')
        log_file.write(f'Average KL Divergence: {avg_kl_div}\n')
        log_file.write(f'Average SSIM: {avg_ssim}\n')
        log_file.write(f'Average EMD: {avg_emd}\n')
        log_file.write(f'Average LPIPS: {avg_lpips}\n')

def save_metrics(output_dir, kl_div_value, ssim_value, emd_value, lpips_value):
    log_file_path = os.path.join(output_dir, 'log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f'KL Divergence: {kl_div_value}\n')
        log_file.write(f'SSIM: {ssim_value}\n')
        log_file.write(f'EMD: {emd_value}\n')
        log_file.write(f'LPIPS: {lpips_value}\n')
        log_file.write('\n')