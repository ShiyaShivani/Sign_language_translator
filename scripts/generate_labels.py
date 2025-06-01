import os
import cv2
import glob
import mediapipe as mp

frames_root = "videos/frames/Z"
out_img_dir = "videos/datasets/images/Z"
out_lbl_dir = "videos/datasets/labels/Z"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True)
label_id = 36
image_id = 0

frame_paths = glob.glob(f"{frames_root}/**/*.jpg", recursive=True)

for path in frame_paths:
    img = cv2.imread(path)
    if img is None: continue
    h, w, _ = img.shape
    results = mp_holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    x, y = [], []
    for lm_set in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if lm_set:
            for lm in lm_set.landmark:
                x.append(lm.x)
                y.append(lm.y)

    if results.pose_landmarks:
        for i in [0, 11, 12, 23, 24]:
            lm = results.pose_landmarks.landmark[i]
            x.append(lm.x)
            y.append(lm.y)

    if results.face_landmarks:
        for i in [1, 9, 10]:
            lm = results.face_landmarks.landmark[i]
            x.append(lm.x)
            y.append(lm.y)

    if x:
        xmin, xmax = max(min(x), 0), min(max(x), 1)
        ymin, ymax = max(min(y), 0), min(max(y), 1)
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        bw, bh = xmax - xmin, ymax - ymin

        label = f"{label_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        img_out = os.path.join(out_img_dir, f"frame_{image_id:05d}.jpg")
        lbl_out = os.path.join(out_lbl_dir, f"frame_{image_id:05d}.txt")
        cv2.imwrite(img_out, img)
        with open(lbl_out, "w") as f: f.write(label)
        image_id += 1

mp_holistic.close()
