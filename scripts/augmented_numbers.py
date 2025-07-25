import os
import cv2
import albumentations as A

# 📌 STEP 1: Define input and output paths
input_root = '/content/drive/MyDrive/BTP/Datasets/Numbers/Numbers'
output_root = '/content/drive/MyDrive/BTP/Datasets/Augmented Numbers'

# 📌 STEP 2: Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.4),
    A.Blur(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
])

# 📌 STEP 3: Augment images
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_file in os.listdir(class_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        # Save the original image to output folder
        cv2.imwrite(os.path.join(output_class_path, img_file), image)

        # Generate 100 augmented images
        for i in range(100):
            augmented = transform(image=image)
            aug_image = augmented['image']
            aug_filename = img_file.rsplit('.', 1)[0] + f'_aug{i}.jpg'
            aug_path = os.path.join(output_class_path, aug_filename)
            cv2.imwrite(aug_path, aug_image)

print("✅ Augmentation complete! Check your output folder.")
