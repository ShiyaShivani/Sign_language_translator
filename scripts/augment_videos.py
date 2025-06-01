import os
import cv2
import albumentations as A

input_folder = 'videos/raw_videos'
output_folder = 'videos/augmented_videos'
os.makedirs(output_folder, exist_ok=True)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def save_video(frames, save_path, fps=25):
    if not frames: return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3)
])

for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.mp4', '.avi')):
            path = os.path.join(root, file)
            rel = os.path.relpath(root, input_folder)
            out_dir = os.path.join(output_folder, rel)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"aug_{file}")
            frames = extract_frames(path)
            aug_frames = [aug(image=f)['image'] for f in frames]
            save_video(aug_frames, out_path)
