import os
import cv2
import numpy as np
from scipy.io import wavfile

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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps

    max_frames = frame_count

    target_fps = original_fps * 0.07
    print(f"Original FPS: {original_fps:.2f}, Target FPS (7%): {target_fps:.2f}")

    writer = cv2.VideoWriter(low_quality_video_path,
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           target_fps,
                           (width, height))

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_counter >= max_frames:
            break

        if frame_counter % int(round(original_fps / target_fps)) == 0:
            writer.write(frame)

        frame_counter += 1

    cap.release()
    writer.release()

    # === Audio Extraction and Degradation ===
    print("Extracting and degrading audio...")
    # Use cv2 to extract audio
    cap = cv2.VideoCapture(input_path)
    
    # Note: OpenCV doesn't directly support audio extraction
    print("Note: Audio processing is skipped as OpenCV doesn't support audio extraction.")
    print("If audio processing is crucial, consider:")
    print("1. Using ffmpeg directly through subprocess")
    print("2. Installing moviepy package")
    print("3. Using a pre-extracted audio file")

    print("Successfully created low-quality video!")

except Exception as e:
    print(f"Error processing files: {str(e)}")
    raise