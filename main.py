import os
from src.audio.audio_utils import load_audio, save_audio
from src.audio.audio_interpolation import interpolate_audio
from src.video.video_utils import load_video_frames, save_video
from src.video.video_interpolation import interpolate_frames
from src.evaluation.audio_evaluation import evaluate_audio
from src.evaluation.video_evaluation import evaluate_video
from src.integration.av_merge import merge_audio_video

# Paths
input_audio_path = 'data/input/input_audio.wav'
input_video_path = 'data/input/input_video.mp4'
output_audio_path = 'data/output/enhanced_audio.wav'
output_video_path = 'data/output/enhanced_video.mp4'
final_output_path = 'data/output/final_enhanced_video.mp4'

# Parameters
audio_interpolation_factor = 2
video_interpolation_factor = 3

# Ensure output directories exist
os.makedirs('data/output', exist_ok=True)
os.makedirs('results/audio_charts', exist_ok=True)
os.makedirs('results/video_charts', exist_ok=True)

print("Step 1: Loading and Enhancing Audio")

audio, sr = load_audio(input_audio_path)
interpolated_audio = interpolate_audio(audio, audio_interpolation_factor)
save_audio(output_audio_path, interpolated_audio, sr * audio_interpolation_factor)

print(f"Enhanced Audio saved to {output_audio_path}")

print("Step 2: Loading and Enhancing Video")

frames, fps = load_video_frames(input_video_path)
enhanced_frames = []
for i in range(len(frames) - 1):
    enhanced_frames.append(frames[i])
    interpolated = interpolate_frames(frames[i], frames[i + 1], video_interpolation_factor - 1)
    enhanced_frames.extend(interpolated)
enhanced_frames.append(frames[-1])

save_video(enhanced_frames, fps * video_interpolation_factor, output_video_path)
print(f"Enhanced Video saved to {output_video_path}")

print("Step 3: Evaluating Audio Quality")

metrics_audio, plots_audio = evaluate_audio(audio, interpolated_audio, sample_rate=sr)

with open('results/metrics_audio.csv', 'w') as f:
    f.write("MSE,SNR\n{:.5f},{:.2f}\n".format(metrics_audio['MSE'], metrics_audio['SNR']))

print(f"Audio Evaluation Done. MSE: {metrics_audio['MSE']:.5f}, SNR: {metrics_audio['SNR']:.2f} dB")

print("Step 4: Evaluating Video Quality")

metrics_video = evaluate_video(frames, enhanced_frames)

with open('results/metrics_video.csv', 'w') as f:
    f.write("Mean SSIM,Mean PSNR\n{:.5f},{:.2f}\n".format(metrics_video['mean_ssim'], metrics_video['mean_psnr']))

print(f"Video Evaluation Done. Mean SSIM: {metrics_video['mean_ssim']:.5f}, Mean PSNR: {metrics_video['mean_psnr']:.2f} dB")

print("Step 5: Merging Enhanced Audio and Video")

merge_audio_video(output_video_path, output_audio_path, final_output_path)

print(f"Final Output saved at {final_output_path}")
