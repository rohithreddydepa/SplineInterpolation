import cv2

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()
    return frames, fps

def save_video(frames, fps, output_path):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
