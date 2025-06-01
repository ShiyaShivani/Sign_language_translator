import os
import glob

video_folder = "videos/raw_videos"
frames_root = "videos/frames"

os.makedirs(frames_root, exist_ok=True)
video_paths = glob.glob(f"{video_folder}/*.mp4")

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(frames_root, video_name)
    os.makedirs(output_folder, exist_ok=True)
    os.system(f'ffmpeg -i "{video_path}" -vf fps=5 "{output_folder}/frame_%04d.jpg"')
