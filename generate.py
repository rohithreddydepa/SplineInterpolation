import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import wavfile
import os


def generate_test_audio():
    """Generate a test audio file with multiple frequencies"""
    print("Generating test audio file...")

    # Set parameters
    sample_rate = 44100  # CD quality
    duration = 5.0  # seconds

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a frequency-varying signal (chirp + tone)
    signal = np.zeros_like(t)

    # Add a chirp (frequency sweep from 200Hz to 1000Hz)
    start_freq, end_freq = 200, 1000
    freq = start_freq + (end_freq - start_freq) * t / duration
    signal += 0.5 * np.sin(2 * np.pi * freq * t)

    # Add a constant tone at 440Hz (A4 note)
    signal += 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add some harmonics at 880Hz
    signal += 0.15 * np.sin(2 * np.pi * 880 * t)

    # Add noise burst in the middle
    mid_point = len(signal) // 2
    window = 0.5 * sample_rate  # half-second noise burst
    noise = 0.1 * np.random.normal(size=int(window))
    window_envelope = np.hanning(len(noise))  # smooth window
    signal[mid_point:mid_point + len(noise)] += noise * window_envelope

    # Normalize to 16-bit range (-32768 to 32767)
    signal = signal / np.max(np.abs(signal)) * 32767 * 0.9  # 90% of max to avoid clipping
    signal = signal.astype(np.int16)

    # Save as WAV file
    filename = "test_audio.wav"
    wavfile.write(filename, sample_rate, signal)

    # Create a downsampled version for testing upsampling
    downsample_rate = 11025  # 1/4 of original sample rate
    downsample_factor = sample_rate // downsample_rate
    downsampled = signal[::downsample_factor]

    downsampled_filename = "test_audio_lowres.wav"
    wavfile.write(downsampled_filename, downsample_rate, downsampled)

    # Plot waveform and spectrogram
    plt.figure(figsize=(12, 8))

    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Test Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(signal, Fs=sample_rate, cmap='viridis')
    plt.title('Test Audio Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')

    plt.tight_layout()
    plt.savefig('test_audio_plot.png')
    plt.close()

    print(f"Created audio files:")
    print(f" - {filename} (Full quality: {sample_rate}Hz, {duration:.1f}s)")
    print(f" - {downsampled_filename} (Low quality: {downsample_rate}Hz, {duration:.1f}s)")

    return filename, downsampled_filename


def generate_test_video():
    """Use 'test.mp4', downscale it to 5 fps and lower resolution."""
    print("\nUsing 'test.mp4' to create test videos...")

    # Input and output paths
    input_filename = os.path.join(os.getcwd(), "test.mp4")
    output_filename = "test_video_lowfps.mp4"

    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found!")
        return None, None

    # Target parameters
    target_width = 640
    target_height = 480
    target_fps = 5.0

    # Open the original video
    cap = cv2.VideoCapture(input_filename)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, target_fps, (target_width, target_height))

    frame_idx = 0
    step = max(1, int(original_fps // target_fps))  # Step to pick frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)

        frame_idx += 1

    cap.release()
    out.release()

    print(f"Created downscaled video: {output_filename}")
    return input_filename, output_filename

def main():
    print("=== TEST MEDIA GENERATOR ===")

    # Create output directory if it doesn't exist
    output_dir = "test_media"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Change to output directory
    os.chdir(output_dir)

    # Generate test files
    audio_file, audio_lowres = generate_test_audio()
    video_file, video_lowfps = generate_test_video()

    # Create a test script
    test_script = """
import numpy as np
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt
from spline_interpolator import SplineInterpolator  # Make sure this is in the parent directory

def test_audio_upsampling():
    print("Testing audio upsampling...")
    sample_rate, audio_data = wavfile.read("test_audio_lowres.wav")

    interpolator = SplineInterpolator(spline_type='cubic')

    target_sr = 44100
    upsampled = interpolator.interpolate_audio(audio_data, sample_rate, target_sr)

    wavfile.write("upsampled_audio.wav", target_sr, upsampled.astype(np.int16))

    print(f"Original audio: {sample_rate}Hz, {len(audio_data)} samples")
    print(f"Upsampled audio: {target_sr}Hz, {len(upsampled)} samples")
    print(f"Saved as 'upsampled_audio.wav'")

    return upsampled

def test_video_interpolation():
    print("\\nTesting video frame interpolation...")
    video_path = "test_video_lowfps.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    interpolator = SplineInterpolator()

    target_fps_multiplier = 12  # Upscale 5fps -> 60fps
    interpolated_frames = interpolator.interpolate_video(frames, target_fps_multiplier)

    output_path = "interpolated_video.mp4"
    out_fps = fps * target_fps_multiplier
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))

    for frame in interpolated_frames:
        out.write(frame)

    out.release()

    print(f"Original video: {fps}fps, {frame_count} frames")
    print(f"Interpolated video: {out_fps}fps, {len(interpolated_frames)} frames")
    print(f"Saved as '{output_path}'")

    return interpolated_frames

if __name__ == "__main__":
    print("=== TESTING WITH GENERATED MEDIA ===")
    choice = input("Test audio (a), video (v), or both (b)? ").lower()

    if choice == 'a' or choice == 'b':
        test_audio_upsampling()

    if choice == 'v' or choice == 'b':
        test_video_interpolation()

    print("\\nTesting complete!")
"""

    with open("test_spline_with_media.py", "w") as f:
        f.write(test_script)

    print("\nGenerated test files in directory:", os.path.abspath(output_dir))
    print("A test script has been created: test_spline_with_media.py")
    print("\nInstructions:")
    print("1. Put your 'test.mp4' file into the 'test_media' folder")
    print("2. Copy your 'spline_interpolator.py' file to 'test_media'")
    print("3. Run the test script: python test_spline_with_media.py")

if __name__ == "__main__":
    main()
