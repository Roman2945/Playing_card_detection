from ultralytics import YOLO
import cv2
import time
import threading
import platform

MODEL_PATH   = "best100.pt"
VIDEO_SOURCE = 0
THRESHOLD    = 10

detected_cards = set()
frame_counters = {}
lock = threading.Lock()

# нова спільна змінна для стриму
latest_jpeg = None


def detection_loop(*, show_window: bool = False):
    print("⏳ Завантажую модель...")
    model = YOLO(MODEL_PATH)
    print("✅ Модель завантажено")

    cap = cv2.VideoCapture(
        VIDEO_SOURCE,
        cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else 0
    )
    if not cap.isOpened():
        print(f"⛔️ Не вдалося відкрити джерело відео: {VIDEO_SOURCE}")
        return

    main_thread = threading.current_thread() is threading.main_thread()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        # ── детекція ────────────────────────────────────────────
        results = model(frame, stream=True, imgsz=640, conf=0.5, device="mps")
        current = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name   = model.names[cls_id]
                current.add(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (224, 94, 34), 2)

        # ── оновлюємо спільний стан ─────────────────────────────
        with lock:
            for nm in list(frame_counters):
                if nm not in current:
                    frame_counters[nm] = 0
            for nm in current:
                cnt = frame_counters.get(nm, 0) + 1
                frame_counters[nm] = cnt
                if cnt >= THRESHOLD:
                    detected_cards.add(nm)
        # ────────────────────────────────────────────────────────

        # ── готуємо кадр для стриму ─────────────────────────────
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            global latest_jpeg
            with lock:
                latest_jpeg = buf.tobytes()
        # ────────────────────────────────────────────────────────

        if show_window and main_thread:
            cv2.imshow("WebCam Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_window and main_thread:
        cv2.destroyAllWindows()


def get_latest_frame() -> bytes | None:
    with lock:
        return latest_jpeg
