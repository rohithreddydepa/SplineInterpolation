# Audio and Video Enhancement using Spline Interpolation
# ---------------------------------------------------------

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.io import wavfile
import cv2


class SplineInterpolator:
    """
    A class for enhancing audio and video using spline interpolation techniques.
    """

    def __init__(self, spline_type='cubic'):
        """
        Initialize the SplineInterpolator.

        Parameters:
        -----------
        spline_type : str
            Type of spline to use ('linear', 'cubic', 'quintic')
        """
        self.spline_type = spline_type
        self.supported_types = ['linear', 'cubic', 'quintic']

        if spline_type not in self.supported_types:
            raise ValueError(f"Spline type must be one of {self.supported_types}")

    def interpolate_audio(self, audio_data, original_sr, target_sr):
        """
        Upsample audio using spline interpolation.

        Parameters:
        -----------
        audio_data : ndarray
            Input audio data
        original_sr : int
            Original sample rate
        target_sr : int
            Target sample rate

        Returns:
        --------
        ndarray: Upsampled audio data
        """
        # Calculate the number of samples for target sample rate
        duration = len(audio_data) / original_sr
        target_length = int(duration * target_sr)

        # Create time axis for original and target audio
        x_original = np.linspace(0, duration, len(audio_data))
        x_target = np.linspace(0, duration, target_length)

        # Handle multi-channel audio
        if len(audio_data.shape) > 1:
            result = np.zeros((target_length, audio_data.shape[1]))
            for channel in range(audio_data.shape[1]):
                # Create spline function
                if self.spline_type == 'linear':
                    spline_func = interpolate.interp1d(x_original, audio_data[:, channel], kind='linear')
                elif self.spline_type == 'cubic':
                    spline_func = interpolate.splrep(x_original, audio_data[:, channel], k=3)
                else:  # quintic
                    spline_func = interpolate.splrep(x_original, audio_data[:, channel], k=5)

                # Apply interpolation
                if self.spline_type == 'linear':
                    result[:, channel] = spline_func(x_target)
                else:
                    result[:, channel] = interpolate.splev(x_target, spline_func)

            return result
        else:
            # Create spline function for mono audio
            if self.spline_type == 'linear':
                spline_func = interpolate.interp1d(x_original, audio_data, kind='linear')
                return spline_func(x_target)
            else:
                k = 3 if self.spline_type == 'cubic' else 5
                spline_func = interpolate.splrep(x_original, audio_data, k=k)
                return interpolate.splev(x_target, spline_func)

    def denoise_audio(self, audio_data, noise_reduction=0.2):
        """
        Simple denoising using spline smoothing

        Parameters:
        -----------
        audio_data : ndarray
            Input audio data
        noise_reduction : float
            Smoothing factor (higher values = more smoothing)

        Returns:
        --------
        ndarray: Denoised audio data
        """
        x = np.arange(len(audio_data))

        # Handle multi-channel audio
        if len(audio_data.shape) > 1:
            result = np.zeros_like(audio_data)
            for channel in range(audio_data.shape[1]):
                # Higher smoothing factor (s) means more smoothing
                smoothing_factor = len(audio_data) * noise_reduction
                spline = interpolate.splrep(x, audio_data[:, channel], s=smoothing_factor)
                result[:, channel] = interpolate.splev(x, spline)
            return result
        else:
            # For mono audio
            smoothing_factor = len(audio_data) * noise_reduction
            spline = interpolate.splrep(x, audio_data, s=smoothing_factor)
            return interpolate.splev(x, spline)

    def interpolate_video(self, frames, target_fps_multiplier):
        """
        Interpolate video frames to increase apparent frame rate

        Parameters:
        -----------
        frames : list of ndarray
            List of video frames (numpy arrays)
        target_fps_multiplier : int
            Multiplier for the frame rate (e.g., 2 = double the frames)

        Returns:
        --------
        list: Interpolated video frames
        """
        if len(frames) < 2:
            return frames

        result_frames = []

        for i in range(len(frames) - 1):
            # Add the current frame
            result_frames.append(frames[i])

            # Create intermediate frames
            for j in range(1, target_fps_multiplier):
                t = j / target_fps_multiplier

                # Interpolate each channel of each pixel
                height, width, channels = frames[i].shape
                intermediate_frame = np.zeros((height, width, channels), dtype=np.uint8)

                for c in range(channels):
                    # Create spline for this channel
                    for y in range(height):
                        for x in range(width):
                            # Linear interpolation for efficiency (can be changed to cubic)
                            intermediate_frame[y, x, c] = int((1 - t) * frames[i][y, x, c] +
                                                              t * frames[i + 1][y, x, c])

                result_frames.append(intermediate_frame)

        # Add the last frame
        result_frames.append(frames[-1])

        return result_frames

    def upscale_image(self, image, scale_factor):
        """
        Upscale an image using bicubic spline interpolation

        Parameters:
        -----------
        image : ndarray
            Input image
        scale_factor : float
            Factor by which to scale the image

        Returns:
        --------
        ndarray: Upscaled image
        """
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)

        # Use bicubic interpolation (which uses cubic splines)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def demo_audio_upsampling():
    """Demo function to show audio upsampling with visualization"""
    # Generate a simple audio signal
    original_sr = 8000
    target_sr = 44100
    duration = 0.5  # seconds

    # Create a sample audio signal (440Hz sine wave with noise)
    t_original = np.linspace(0, duration, int(original_sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t_original) + 0.2 * np.random.normal(size=len(t_original))

    # Interpolate
    interpolator = SplineInterpolator(spline_type='cubic')
    upsampled = interpolator.interpolate_audio(signal, original_sr, target_sr)

    # Create time axis for visualization
    t_target = np.linspace(0, duration, len(upsampled), endpoint=False)

    # Plot only a small segment for clarity
    segment_duration = 0.01
    original_segment = signal[t_original < segment_duration]
    t_original_segment = t_original[t_original < segment_duration]
    upsampled_segment = upsampled[t_target < segment_duration]
    t_target_segment = t_target[t_target < segment_duration]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_original_segment, original_segment, 'r.-', label=f'Original ({original_sr} Hz)')
    plt.title('Audio Upsampling using Cubic Spline Interpolation')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_target_segment, upsampled_segment, 'b-', label=f'Upsampled ({target_sr} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return signal, upsampled, original_sr, target_sr


def demo_image_upscaling():
    """Demo function to show image upscaling"""
    # Create a simple test image
    size = 50
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    image = np.sin(10 * np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2))
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)

    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Upscale
    interpolator = SplineInterpolator()
    upscaled = interpolator.upscale_image(image_rgb, 4.0)

    # Display
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f'Original ({size}x{size})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(upscaled)
    plt.title(f'Upscaled ({upscaled.shape[1]}x{upscaled.shape[0]})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return image_rgb, upscaled


def main():
    print("Audio and Video Enhancement using Spline Interpolation")
    print("-----------------------------------------------------")
    print("1. Demo audio upsampling")
    print("2. Demo image upscaling")
    choice = input("Enter your choice (1-2): ")

    if choice == '1':
        demo_audio_upsampling()
    elif choice == '2':
        demo_image_upscaling()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()