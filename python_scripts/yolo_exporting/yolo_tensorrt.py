import cv2
import time
from ultralytics import YOLO

# Load TensorRT-optimized model
model = YOLO("yolo11n.engine")  # Or "yolov8n.engine" if you prefer

# Open USB camera (usually /dev/video0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if camera opened successfully
if not cap.isOpened():
    print("❌ Unable to open camera.")
    exit()

print("🔍 Starting inference (Persons Only). Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        break

    start = time.time()

    # Run inference with YOLO
    results = model.predict(source=frame, conf=0.25, show=False, stream=False, verbose=False)

    # Loop through detections and draw only persons
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])  # class ID
            # COCO dataset: 'person' is class 0
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    end = time.time()
    fps = 1 / (end - start)

    # Overlay FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 + TensorRT (Persons Only)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

