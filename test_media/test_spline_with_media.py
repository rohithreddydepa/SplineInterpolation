
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
    print("\nTesting video frame interpolation...")
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

    print("\nTesting complete!")
