import cv2
import time
from ultralytics import YOLO

# Load YOLO11 Nano
model = YOLO("yolo11n.pt")

# Open Webcam
cap = cv2.VideoCapture(0)

# Camera Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detect Objects
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.5,
        verbose=False
    )

    object_count = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            object_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            # Green Rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # FPS
    current = time.time()
    fps = 1 / (current - prev_time) if prev_time else 0
    prev_time = current

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Objects : {object_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLO11 Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()