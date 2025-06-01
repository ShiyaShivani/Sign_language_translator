import cv2
from ultralytics import YOLO

model = YOLO("./m2/weights/best.pt")
# model = YOLO("./signlanguage5-20250519T211107Z-1-001/signlanguage5/weights/best.pt")
print("Model classes:", model.names)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 640))  # optional based on training
    results = model(resized_frame, conf=0.1)

    annotated_frame = results[0].plot()

    # Log detections
    for r in results:
        boxes = r.boxes
        if boxes:
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                print("Detected:", label)
        else:
            print("No detections")

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
