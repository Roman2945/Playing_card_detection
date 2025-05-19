from ultralytics import YOLO
import cv2
import numpy as np
import mss
import time

MODEL_PATH = "best100.pt"

def detection_loop_screen():
    model = YOLO(MODEL_PATH)
    sct   = mss.mss()
    monitor = sct.monitors[0]

    while True:
        img   = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

        results = model(
            frame,
            stream=True,
            imgsz=max(monitor["width"], monitor["height"]),
            conf=0.5,
            device="mps"  # cuda:0 / mps / cpu
        )

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name   = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (34, 94, 224), 2)

        cv2.imshow("Screen Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detection_loop_screen()
