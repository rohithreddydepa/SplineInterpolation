
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
#
# # Import the SplineInterpolator class
# # (Make sure the previous code is saved in a file named spline_interpolator.py)
from spline_interpolator import SplineInterpolator


def test_audio_interpolation():
    """Test audio interpolation with a synthetic signal"""
    print("Testing audio interpolation...")

    # Generate a test signal (combine two sine waves with noise)
    duration = 1.0  # seconds
    original_sr = 8000
    target_sr = 44100

    t_original = np.linspace(0, duration, int(original_sr * duration), endpoint=False)
    test_signal = (
            np.sin(2 * np.pi * 440 * t_original) +  # 440 Hz tone
            0.5 * np.sin(2 * np.pi * 880 * t_original) +  # 880 Hz harmonic
            0.1 * np.random.normal(size=len(t_original))  # Some noise
    )

    # Create interpolator and process
    interpolator = SplineInterpolator(spline_type='cubic')
    upsampled_signal = interpolator.interpolate_audio(test_signal, original_sr, target_sr)
    denoised_signal = interpolator.denoise_audio(test_signal, noise_reduction=0.2)

    # Create time axes for visualization
    t_upsampled = np.linspace(0, duration, len(upsampled_signal), endpoint=False)

    # Plot only a small segment (10ms) for clarity
    segment_duration = 0.01
    segment_samples_original = int(segment_duration * original_sr)
    segment_samples_upsampled = int(segment_duration * target_sr)

    plt.figure(figsize=(12, 8))

    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(t_original[:segment_samples_original], test_signal[:segment_samples_original], 'r.-')
    plt.title(f'Original Signal ({original_sr} Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Upsampled signal
    plt.subplot(3, 1, 2)
    plt.plot(t_upsampled[:segment_samples_upsampled], upsampled_signal[:segment_samples_upsampled], 'b-')
    plt.title(f'Upsampled Signal ({target_sr} Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Denoised signal
    plt.subplot(3, 1, 3)
    plt.plot(t_original[:segment_samples_original], test_signal[:segment_samples_original], 'r-', alpha=0.5,
             label='Original')
    plt.plot(t_original[:segment_samples_original], denoised_signal[:segment_samples_original], 'g-', label='Denoised')
    plt.title('Denoised Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('audio_interpolation_test.png')
    plt.show()

    # Display some statistics
    print(f"Original signal: {len(test_signal)} samples")
    print(f"Upsampled signal: {len(upsampled_signal)} samples")
    print(f"Upsampling ratio: {len(upsampled_signal) / len(test_signal):.2f}x")
    print(f"Target ratio: {target_sr / original_sr:.2f}x")

    return test_signal, upsampled_signal, denoised_signal


def test_image_upscaling():
    """Test image upscaling with a synthetic pattern"""
    print("\nTesting image upscaling...")

    # Create a test pattern (circular wave pattern)
    size = 80
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x ** 2 + y ** 2)
    test_image = 127.5 * (1 + np.sin(10 * r))
    test_image = test_image.astype(np.uint8)

    # Convert to RGB and add some noise
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    noise = np.random.normal(0, 10, test_image_rgb.shape).astype(np.int8)
    test_image_noisy = np.clip(test_image_rgb + noise, 0, 255).astype(np.uint8)

    # Create interpolator and upscale
    interpolator = SplineInterpolator()
    scale_factor = 3.0
    upscaled_image = interpolator.upscale_image(test_image_noisy, scale_factor)

    # Also test with basic nearest-neighbor interpolation for comparison
    basic_upscaled = cv2.resize(test_image_noisy,
                                (int(size * scale_factor), int(size * scale_factor)),
                                interpolation=cv2.INTER_NEAREST)

    # Display the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(test_image_noisy)
    plt.title(f'Original Noisy Image ({size}x{size})')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(basic_upscaled)
    plt.title(f'Nearest Neighbor ({basic_upscaled.shape[1]}x{basic_upscaled.shape[0]})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(upscaled_image)
    plt.title(f'Cubic Spline ({upscaled_image.shape[1]}x{upscaled_image.shape[0]})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('image_upscaling_test.png')
    plt.show()

    print(f"Original image size: {test_image_noisy.shape}")
    print(f"Upscaled image size: {upscaled_image.shape}")
    print(f"Scale factor: {scale_factor:.1f}x")

    return test_image_noisy, upscaled_image, basic_upscaled


def main():
    print("=== SPLINE INTERPOLATION TEST ===")
    choice = input("Test audio (a), image (i), or both (b)? ").lower()

    if choice == 'a' or choice == 'b':
        test_audio_interpolation()

    if choice == 'i' or choice == 'b':
        test_image_upscaling()

    print("\nTesting complete!")


if __name__ == "__main__":
    main()