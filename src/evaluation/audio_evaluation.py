import numpy as np
from scipy import signal
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
from typing import Tuple


def evaluate_audio(original: np.ndarray,
                   enhanced: np.ndarray,
                   sample_rate: int = 44100,
                   output_folder: str = "results/audio_charts",
                   max_samples: int = 100000) -> Tuple[dict, dict]:
    """
    Evaluate audio enhancement results with comprehensive metrics and visualizations.

    Args:
        original (np.ndarray): Original audio signal (1D array).
        enhanced (np.ndarray): Enhanced audio signal (1D array).
        sample_rate (int): Sampling rate in Hz.
        output_folder (str): Directory to save plots and metrics.
        max_samples (int): Maximum samples to use for visualization.

    Returns:
        Tuple[dict, dict]: (metrics_dict, plot_paths_dict)
    """
    os.makedirs(output_folder, exist_ok=True)

    assert original.ndim == 1 and enhanced.ndim == 1, "Audio signals must be 1D arrays"
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]

    error = original - enhanced
    epsilon = 1e-10  # small number to avoid division by zero

    # Safe dynamic range calculation
    min_nonzero_orig = np.min(np.abs(original[np.abs(original) > 0]))
    min_nonzero_enh = np.min(np.abs(enhanced[np.abs(enhanced) > 0]))

    dynamic_range_original = 20 * np.log10(np.max(np.abs(original)) / (min_nonzero_orig + epsilon))
    dynamic_range_enhanced = 20 * np.log10(np.max(np.abs(enhanced)) / (min_nonzero_enh + epsilon))

    # Calculate metrics
    metrics = {
        'MSE': mean_squared_error(original, enhanced),
        'MAE': mean_absolute_error(original, enhanced),
        'SNR': 10 * np.log10(np.mean(original ** 2) / (np.mean(error ** 2) + epsilon)),
        'PSNR': 10 * np.log10(np.max(original ** 2) / (mean_squared_error(original, enhanced) + epsilon)),
        'Dynamic_Range_Original': dynamic_range_original,
        'Dynamic_Range_Enhanced': dynamic_range_enhanced
    }

    # Spectral analysis
    f_orig, p_orig = signal.welch(original, fs=sample_rate)
    f_enh, p_enh = signal.welch(enhanced, fs=sample_rate)

    metrics['Spectral_Convergence'] = np.mean(np.abs(p_orig - p_enh))
    metrics['Spectral_Centroid_Original'] = np.sum(f_orig * p_orig) / (np.sum(p_orig) + epsilon)
    metrics['Spectral_Centroid_Enhanced'] = np.sum(f_enh * p_enh) / (np.sum(p_enh) + epsilon)

    # Visualization
    plot_paths = {}
    plt.figure(figsize=(16, 10))

    nsamples = int(0.5 * sample_rate)

    plt.subplot(3, 2, 1)
    plt.plot(original[:nsamples], label='Original', alpha=0.7)
    plt.plot(enhanced[:nsamples], label='Enhanced', alpha=0.7, linestyle='--')
    plt.title('Waveform Comparison (First 0.5s)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.semilogy(f_orig, p_orig, label='Original')
    plt.semilogy(f_enh, p_enh, label='Enhanced', alpha=0.7)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(error[:max_samples], color='red')
    plt.title('Error Signal')
    plt.xlabel('Samples')
    plt.ylabel('Error')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(error, bins=100, density=True, color='purple', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error Value')
    plt.ylabel('Probability Density')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.specgram(enhanced[:max_samples] + epsilon, Fs=sample_rate, cmap='viridis')
    plt.title('Enhanced Audio Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    plt.subplot(3, 2, 6)
    plt.bar(['Original', 'Enhanced'],
            [metrics['Dynamic_Range_Original'], metrics['Dynamic_Range_Enhanced']],
            color=['blue', 'orange'])
    plt.title('Dynamic Range Comparison')
    plt.ylabel('dB')

    plt.tight_layout()
    plot_file = os.path.join(output_folder, "audio_analysis.png")
    plt.savefig(plot_file)
    plt.close()
    plot_paths['main_analysis'] = plot_file

    metrics_file = os.path.join(output_folder, "audio_metrics.txt")
    with open(metrics_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"Audio evaluation completed. Results saved in {output_folder}")
    return metrics, plot_paths
