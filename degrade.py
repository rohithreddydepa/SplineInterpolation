import os
import cv2
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal

# Ensure output directories exist
os.makedirs('data/input', exist_ok=True)

# Paths
input_path = 'data/input/original_video.mp4'
low_quality_video_path = 'data/input/input_video.mp4'
low_quality_audio_path = 'data/input/input_audio.wav'

try:
    # Process video
    print("Creating low-quality video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 10  # Reduced FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps

    max_frames = frame_count

    # Create video writer
    writer = cv2.VideoWriter(low_quality_video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             target_fps,
                             (width, height))

    # Process frames
    frame_interval = int(original_fps / target_fps)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        # Only write every nth frame to achieve lower FPS
        if frame_count % frame_interval == 0:
            writer.write(frame)

        frame_count += 1

    cap.release()
    writer.release()

    # Create degraded audio
    print("Creating low-quality audio...")
    # Generate a simple tone for noise
    duration = min(15.0, duration)  # Limit to 15 seconds
    sample_rate = 11025
    t = np.linspace(0, duration, int(sample_rate * duration))
    noise = 0.02 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Create some low-quality audio (simple sine wave + noise)
    base_audio = np.sin(2 * np.pi * 200 * t)  # 200 Hz base tone
    noisy_audio = base_audio + noise

    # Normalize audio
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))

    # Convert to 16-bit PCM
    audio_16bit = (noisy_audio * 32767).astype(np.int16)

    # Save the audio
    wavfile.write(low_quality_audio_path, sample_rate, audio_16bit)

    print("Successfully created degraded samples!")

except Exception as e:
    print(f"Error processing files: {str(e)}")
    raise
