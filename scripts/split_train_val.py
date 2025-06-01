import os
import shutil
from sklearn.model_selection import train_test_split

img_dir = "videos/datasets/images/Z"
lbl_dir = "videos/datasets/labels/Z"

train_img = "videos/datasets/images/train"
val_img = "videos/datasets/images/val"
train_lbl = "videos/datasets/labels/train"
val_lbl = "videos/datasets/labels/val"

for d in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
train, val = train_test_split(images, test_size=0.2, random_state=42)

for split, dst_img, dst_lbl in [(train, train_img, train_lbl), (val, val_img, val_lbl)]:
    for fname in split:
        shutil.move(os.path.join(img_dir, fname), os.path.join(dst_img, fname))
        shutil.move(os.path.join(lbl_dir, fname.replace(".jpg", ".txt")), os.path.join(dst_lbl, fname.replace(".jpg", ".txt")))
