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
os.makedirs('results', exist_ok=True)

print("--- Starting Audio Enhancement ---")
audio, sr = load_audio(input_audio_path)
interpolated_audio = interpolate_audio(audio, audio_interpolation_factor)
save_audio(output_audio_path, interpolated_audio, sr * audio_interpolation_factor)
print("Audio enhancement complete. Saved to:", output_audio_path)

print("--- Starting Video Enhancement ---")
frames, fps = load_video_frames(input_video_path)
enhanced_frames = []
for i in range(len(frames) - 1):
    enhanced_frames.append(frames[i])
    interpolated = interpolate_frames(frames[i], frames[i + 1], video_interpolation_factor - 1)
    enhanced_frames.extend(interpolated)
enhanced_frames.append(frames[-1])
save_video(enhanced_frames, fps * video_interpolation_factor, output_video_path)
print("Video enhancement complete. Saved to:", output_video_path)

print("--- Evaluating Audio Quality ---")
mse_audio, snr_audio = evaluate_audio(audio, interpolated_audio)
with open('results/metrics_audio.csv', 'w') as f:
    f.write("MSE,SNR\n{:.5f},{:.2f}\n".format(mse_audio, snr_audio))
print(f"Audio Evaluation - MSE: {mse_audio:.5f}, SNR: {snr_audio:.2f} dB")

print("--- Evaluating Video Quality ---")
ssim_video, psnr_video = evaluate_video(frames, enhanced_frames, video_interpolation_factor)
with open('results/metrics_video.csv', 'w') as f:
    f.write("SSIM,PSNR\n{:.4f},{:.2f}\n".format(ssim_video, psnr_video))
print(f"Video Evaluation - SSIM: {ssim_video:.4f}, PSNR: {psnr_video:.2f} dB")

print("--- Merging Audio and Video ---")
merge_audio_video(output_video_path, output_audio_path, final_output_path)
print("Audio and Video successfully merged. Final file saved at:", final_output_path)
