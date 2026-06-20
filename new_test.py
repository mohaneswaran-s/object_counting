import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("./anpr_17_06_2026 (1).pt")
video_source = "./a1_out.mp4"
output_path = "./output_detection.mp4"
cap = cv2.VideoCapture(video_source)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

#cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model.predict(
        source=frame,
        conf=0.4,
        iou=0.4,
        imgsz=640,
        verbose=False,
       # device="cpu"
    )

    # Draw detections
    for r in results:
        for box in r.boxes:

            class_id = int(box.cls.item())
            class_name = r.names[class_id]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # Save frame to output video
    out.write(frame)

    #cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved: {output_path}")

print("check for new changes in github - tagging version 1.0.1")