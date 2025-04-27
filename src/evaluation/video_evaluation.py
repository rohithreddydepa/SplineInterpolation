import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def evaluate_video(original_frames, interpolated_frames, interpolation_factor):
    ssim_list, psnr_list = [], []
    sampled_frames = interpolated_frames[::interpolation_factor]

    for orig, interp in zip(original_frames, sampled_frames):
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        interp_gray = cv2.cvtColor(interp, cv2.COLOR_BGR2GRAY)

        ssim = structural_similarity(orig_gray, interp_gray)
        psnr = peak_signal_noise_ratio(orig_gray, interp_gray)

        ssim_list.append(ssim)
        psnr_list.append(psnr)

    return np.mean(ssim_list), np.mean(psnr_list)
