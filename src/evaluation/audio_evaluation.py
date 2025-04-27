import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def evaluate_audio(original, enhanced, output_folder="results/audio_charts"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Match lengths
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]

    # Compute MSE and SNR
    mse = mean_squared_error(original, enhanced)
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum((original - enhanced) ** 2))

    # 📈 Plot 1: Waveforms (Original vs Enhanced)
    plt.figure(figsize=(14, 5))
    plt.plot(original, label='Original Audio', alpha=0.7)
    plt.plot(enhanced, label='Enhanced Audio', alpha=0.7, linestyle='--')
    plt.title('Waveform Comparison: Original vs Enhanced')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    waveform_path = os.path.join(output_folder, "waveform_comparison.png")
    plt.savefig(waveform_path)
    plt.close()

    # 📈 Plot 2: Error Signal
    error_signal = original - enhanced
    plt.figure(figsize=(14, 4))
    plt.plot(error_signal, color='red')
    plt.title('Error Signal (Original - Enhanced)')
    plt.xlabel('Sample Index')
    plt.ylabel('Error Amplitude')
    plt.grid(True)
    plt.tight_layout()
    error_path = os.path.join(output_folder, "error_signal.png")
    plt.savefig(error_path)
    plt.close()

    # 📈 Plot 3: Histogram of Error Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(error_signal, bins=100, color='purple', alpha=0.7)
    plt.title('Histogram of Error Distribution')
    plt.xlabel('Error Amplitude')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    histogram_path = os.path.join(output_folder, "error_histogram.png")
    plt.savefig(histogram_path)
    plt.close()

    print(f"Graphs saved to {output_folder}/")

    return mse, snr
