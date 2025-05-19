from ultralytics import YOLO
import cv2
import time
import threading

# PARAMETERS
MODEL_PATH   = "best100.pt"
VIDEO_SOURCE = 0
THRESHOLD    = 15

# SHARED STATE FOR API
detected_cards = set()
frame_counters = {}
lock = threading.Lock()

def detection_loop():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        results = model(
            frame,
            stream=True,
            imgsz=640,
            conf=0.5,
            device="0"    # cuda:0 / mps / cpu
        )

        current = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name   = model.names[cls_id]
                current.add(name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (224, 94, 34),
                    2
                )

        # update shared state
        with lock:
            # reset counters for cards no longer seen
            for nm in list(frame_counters):
                if nm not in current:
                    frame_counters[nm] = 0
            for nm in current:
                cnt = frame_counters.get(nm, 0) + 1
                frame_counters[nm] = cnt
                if cnt >= THRESHOLD:
                    detected_cards.add(nm)

        cv2.imshow("WebCam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detection_loop()
