import os
import cv2
import numpy as np
import subprocess

# Ensure input directories exist
os.makedirs('data/input', exist_ok=True)

# Paths
input_path = 'data/input/original_video.mp4'
low_quality_video_path = 'data/input/input_video.mp4'
low_quality_audio_path = 'data/input/input_audio.wav'

def extract_and_degrade_audio_ffmpeg(input_video_path, output_audio_path):
    print("Extracting and degrading audio using ffmpeg...")

    try:
        subprocess.run([
            'ffmpeg',
            '-i', input_video_path,   # Input video
            '-ar', '11025',            # Downsample audio to 11025 Hz (bad quality)
            '-ac', '1',                # Convert to mono (optional, more degraded)
            '-vn',                     # No video
            '-y',                      # Overwrite output if exists
            output_audio_path
        ], check=True)

        print(f"Degraded audio saved at {output_audio_path}")

    except subprocess.CalledProcessError as e:
        print(f" Error using ffmpeg: {e}")
        raise

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

    target_fps = original_fps * 0.07  # Reduce FPS to 7% of original
    print(f"Original FPS: {original_fps:.2f}, Target FPS (7%): {target_fps:.2f}")

    writer = cv2.VideoWriter(low_quality_video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             target_fps,
                             (width, height))

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_counter >= frame_count:
            break

        if frame_counter % int(round(original_fps / target_fps)) == 0:
            writer.write(frame)

        frame_counter += 1

    cap.release()
    writer.release()
    print(f" Degraded video saved at {low_quality_video_path}")

    # Extract and degrade real audio using FFmpeg
    extract_and_degrade_audio_ffmpeg(input_path, low_quality_audio_path)

    print("Successfully created degraded video and audio!")

except Exception as e:
    print(f"Error processing files: {str(e)}")
    raise
