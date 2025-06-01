from ultralytics import YOLO

model = YOLO('yolov8_weights/yolov8s.pt')
model.train(
    data='assets/data.yaml',
    epochs=25,
    imgsz=416,
    batch=32,
    name='sign_yolo_model',
    patience=10
)
