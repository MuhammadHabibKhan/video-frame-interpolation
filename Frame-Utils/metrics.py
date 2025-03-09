# using real-esrgan miniforge environment for this one

import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_average_psnr_ssim(gt_folder, interp_folder):
    """
    Compute the average PSNR and SSIM for all images in two given folders.

    :param gt_folder: Path to the folder containing ground truth images
    :param interp_folder: Path to the folder containing interpolated images
    :return: (average PSNR, average SSIM)
    """
    psnr_values = []
    ssim_values = []
    
    # Get sorted list of files (assuming filenames match in both folders)
    gt_files = sorted(os.listdir(gt_folder))
    interp_files = sorted(os.listdir(interp_folder))

    # Ensure both folders have the same number of images
    if len(gt_files) != len(interp_files):
        raise ValueError("Mismatch in number of images between the two folders.")

    for gt_file, interp_file in zip(gt_files, interp_files):
        gt_path = os.path.join(gt_folder, gt_file)
        interp_path = os.path.join(interp_folder, interp_file)

        # Read images
        gt_image = cv2.imread(gt_path)
        interp_image = cv2.imread(interp_path)

        if gt_image is None or interp_image is None:
            print(f"Skipping {gt_file} due to read error.")
            continue  # Skip unreadable images

        # Convert to grayscale for SSIM calculation
        gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        interp_gray = cv2.cvtColor(interp_image, cv2.COLOR_BGR2GRAY)

        # Compute PSNR
        mse = np.mean((gt_image - interp_image) ** 2)
        psnr_value = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)
        psnr_values.append(psnr_value)

        # Compute SSIM
        ssim_value = ssim(gt_gray, interp_gray, data_range=255)
        ssim_values.append(ssim_value)

    # Compute averages
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0

    return avg_psnr, avg_ssim

# Usage
gt_folder = "../data/ground_truth"

interp_folder = "../data/cnn-gan/interpolated_frames"
interp_gan_folder = "../data/cnn-gan/interpolated_gan_frames_no_upscale"

interp_lk_folder = "../data/lucas-kanade/interpolated_frames"

interp_gf_folder = "../data/gunnar-farneback/interpolated_frames"

# avg_psnr, avg_ssim = calculate_average_psnr_ssim(gt_folder, interp_folder)
# print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

# avg_psnr, avg_ssim = calculate_average_psnr_ssim(gt_folder, interp_gan_folder)
# print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

# avg_psnr, avg_ssim = calculate_average_psnr_ssim(gt_folder, interp_lk_folder)
# print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

# avg_psnr, avg_ssim = calculate_average_psnr_ssim(gt_folder, interp_gf_folder)
# print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")