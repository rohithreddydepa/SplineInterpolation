import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def evaluate_video(original_frames, interpolated_frames, interpolation_factor, output_folder="results/video_charts"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # ✅ Correctly initialize empty lists
    ssim_list = []
    psnr_list = []

    sampled_frames = interpolated_frames[::interpolation_factor]

    for idx, (orig, interp) in enumerate(zip(original_frames, sampled_frames)):
        # Progress print
        print(f"Evaluating frame {idx + 1}/{len(sampled_frames)}...")

        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        interp_gray = cv2.cvtColor(interp, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim = structural_similarity(orig_gray, interp_gray)

        # Manual PSNR calculation to avoid divide by zero
        mse = np.mean((orig_gray.astype(np.float32) - interp_gray.astype(np.float32)) ** 2)
        if mse == 0:
            psnr = float('inf')  # PSNR is infinite if no error
        else:
            psnr = 10 * np.log10((255 ** 2) / mse)

        ssim_list.append(ssim)
        psnr_list.append(psnr)

    # Compute averages
    avg_ssim = np.mean(ssim_list)
    avg_psnr = np.mean([p if p != float('inf') else 100 for p in psnr_list])  # Treat infinite PSNR as 100 dB for averaging

    # 📈 Plot 1: SSIM over frames
    plt.figure(figsize=(12, 6))
    plt.plot(ssim_list, label='SSIM per Frame', color='blue')
    plt.title('SSIM vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylabel('SSIM Value')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ssim_plot_path = os.path.join(output_folder, "ssim_per_frame.png")
    plt.savefig(ssim_plot_path)
    plt.close()

    # 📈 Plot 2: PSNR over frames
    plt.figure(figsize=(12, 6))
    plt.plot(psnr_list, label='PSNR per Frame (dB)', color='green')
    plt.title('PSNR vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    psnr_plot_path = os.path.join(output_folder, "psnr_per_frame.png")
    plt.savefig(psnr_plot_path)
    plt.close()

    # 📄 Save detailed CSV
    csv_path = os.path.join(output_folder, "video_metrics_per_frame.csv")
    with open(csv_path, 'w') as f:
        f.write("Frame,SSIM,PSNR\n")
        for i, (s, p) in enumerate(zip(ssim_list, psnr_list)):
            p_val = p if p != float('inf') else 100.00  # Save 100 for infinite PSNR frames
            f.write(f"{i},{s:.6f},{p_val:.2f}\n")

    print(f"✅ Video evaluation plots and CSV saved to {output_folder}/")

    return avg_ssim, avg_psnr
