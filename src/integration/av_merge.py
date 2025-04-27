import subprocess

def merge_audio_video(video_path, audio_path, output_path):
    try:
        # Using FFmpeg through subprocess to merge video and audio
        command = [
            'ffmpeg',
            '-i', video_path,  # Input video file
            '-i', audio_path,  # Input audio file
            '-c:v', 'copy',    # Copy video codec
            '-c:a', 'aac',     # Use AAC for audio
            '-strict', 'experimental',
            '-map', '0:v:0',   # Map video from first input
            '-map', '1:a:0',   # Map audio from second input
            '-y',              # Overwrite output file if it exists
            output_path
        ]
        
        print(f"Merging video and audio into {output_path}...")
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Merge complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error merging video and audio: {e}")
        print(f"FFmpeg error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error merging video and audio: {e}")