# Sign Language Translator using YOLOv8

This project detects sign language gestures using a custom-trained YOLOv8 model and translates them into text/speech in real-time.

## Dataset
- Collected custom sign gesture videos and photos
- Extracted video frames using `extract_frames.py`
- Applied augmentation (flip, brightness, rotation)
- Labeled using mediapipe
- Dataset split into train/val/test

## YOLOv8 Training
- Used Ultralytics YOLOv8
- Config: `data.yaml`, trained using `train.py`
- Model: `best.pt`

## Real-Time Detection
- `real_time/detect.py` uses webcam input to detect and show predicted gesture

## How to Run
```bash
pip install -r requirements.txt
python real_time/detect.py --weights models/best.pt
