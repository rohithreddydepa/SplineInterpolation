import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def evaluate_video(original_frames, enhanced_frames, output_folder="results/video_charts",
                   max_frames=100, parallel=True):
    os.makedirs(output_folder, exist_ok=True)

    assert len(original_frames) > 0 and len(enhanced_frames) > 0, "Empty frame list"
    assert original_frames[0].shape == enhanced_frames[0].shape, "Frame dimension mismatch"

    n_frames = min(len(original_frames), len(enhanced_frames), max_frames)
    orig_frames = original_frames[:n_frames]
    enh_frames = enhanced_frames[:n_frames]

    metrics = {'ssim': [], 'psnr': [], 'mse': [], 'color_psnr': {'y': [], 'u': [], 'v': []}}

    def process_frame_pair(args):
        idx, orig, enh = args
        orig_yuv = cv2.cvtColor(orig, cv2.COLOR_BGR2YUV)
        enh_yuv = cv2.cvtColor(enh, cv2.COLOR_BGR2YUV)

        frame_metrics = {
            'ssim': structural_similarity(orig_yuv[..., 0], enh_yuv[..., 0], data_range=255, gaussian_weights=True),
            'psnr': peak_signal_noise_ratio(orig, enh),
            'color_psnr': {
                'y': peak_signal_noise_ratio(orig_yuv[..., 0], enh_yuv[..., 0]),
                'u': peak_signal_noise_ratio(orig_yuv[..., 1], enh_yuv[..., 1]),
                'v': peak_signal_noise_ratio(orig_yuv[..., 2], enh_yuv[..., 2])
            },
            'mse': np.mean((orig.astype(np.float32) - enh.astype(np.float32)) ** 2)
        }

        if idx < 5:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Original Frame {idx}')
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Enhanced Frame {idx}')
            axes[1].axis('off')

            diff = cv2.absdiff(orig, enh)
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            axes[2].imshow(diff_color)
            axes[2].set_title('Difference Map')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"frame_comparison_{idx}.png"))
            plt.close()

        return frame_metrics

    frame_args = [(i, o, e) for i, (o, e) in enumerate(zip(orig_frames, enh_frames))]

    if parallel:
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_frame_pair, frame_args), total=n_frames, desc="Evaluating Frames"))
    else:
        results = [process_frame_pair(args) for args in tqdm(frame_args, desc="Evaluating Frames")]

    for res in results:
        metrics['ssim'].append(res['ssim'])
        metrics['psnr'].append(res['psnr'])
        metrics['mse'].append(res['mse'])
        for c in ['y', 'u', 'v']:
            metrics['color_psnr'][c].append(res['color_psnr'][c])

    # Plot Combined Metrics
    plt.figure(figsize=(14, 10))

    # SSIM Plot
    plt.subplot(3, 1, 1)
    plt.plot(range(len(metrics['ssim'])), metrics['ssim'], label='SSIM', color='blue')
    plt.title('SSIM over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.legend()

    # PSNR Plot
    plt.subplot(3, 1, 2)
    plt.plot(range(len(metrics['psnr'])), metrics['psnr'], label='PSNR', color='green')
    plt.title('PSNR over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()

    # Color Channel PSNR Plot
    plt.subplot(3, 1, 3)
    for c, color in zip(['y', 'u', 'v'], ['gold', 'magenta', 'cyan']):
        plt.plot(range(len(metrics['color_psnr'][c])), metrics['color_psnr'][c], label=f'{c.upper()} PSNR', color=color)
    plt.title('Color Channels PSNR')
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "video_quality_timeseries.png"))
    plt.close()

    # Separate SSIM plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(metrics['ssim'])), metrics['ssim'], label='SSIM per Frame', color='blue')
    plt.title('SSIM vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "ssim_per_frame.png"))
    plt.close()

    # Separate PSNR plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(metrics['psnr'])), metrics['psnr'], label='PSNR per Frame (dB)', color='green')
    plt.title('PSNR vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "psnr_per_frame.png"))
    plt.close()

    # Summarize metrics
    summary = {
        'mean_ssim': np.mean(metrics['ssim']),
        'mean_psnr': np.mean(metrics['psnr']),
        'mean_mse': np.mean(metrics['mse']),
        'mean_color_psnr': {c: np.mean(metrics['color_psnr'][c]) for c in ['y', 'u', 'v']}
    }

    # Safe JSON dump (convert numpy types)
    def convert_numpy(obj):
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(os.path.join(output_folder, "video_metrics.json"), 'w') as f:
        json.dump({**metrics, **summary}, f, indent=2, default=convert_numpy)

    print(f"✅ Video evaluation completed. Results saved to {output_folder}")
    return {**metrics, **summary}
