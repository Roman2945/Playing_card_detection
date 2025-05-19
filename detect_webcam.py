from ultralytics import YOLO
import cv2
import time
import threading
import platform

# PARAMETERS
MODEL_PATH   = "best100.pt"
VIDEO_SOURCE = 0
THRESHOLD    = 15

# SHARED STATE FOR API
detected_cards = set()
frame_counters = {}
lock = threading.Lock()


def detection_loop(*, show_window: bool = True):
    """
    Детекція гральних карт.
    show_window=True  – показуємо зображення (лише у головному потоці);
    show_window=False – працюємо без GUI (для бекграунд-режиму/FastAPI).
    """
    print("⏳ Завантажую модель...")
    model = YOLO(MODEL_PATH)
    print("✅ Модель завантажено")

    # Відкриття камери
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"⛔️ Не вдалося відкрити джерело відео: {VIDEO_SOURCE}")
        return

    print("🎥 Камеру відкрито, починаю детекцію (натисніть q для виходу)")

    main_thread = threading.current_thread() is threading.main_thread()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ read() повернуло False – очікування 0.1 сек")
            time.sleep(0.1)
            continue

        results = model(frame, stream=True, imgsz=640, conf=0.5, device="mps")

        current = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name   = model.names[cls_id]
                current.add(name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (224, 94, 34), 2)

        # ── оновлюємо спільний стан ──────────────────────────────
        with lock:
            for nm in list(frame_counters):
                if nm not in current:
                    frame_counters[nm] = 0
            for nm in current:
                cnt = frame_counters.get(nm, 0) + 1
                frame_counters[nm] = cnt
                if cnt >= THRESHOLD:
                    detected_cards.add(nm)
        # ─────────────────────────────────────────────────────────

        # Показуємо зображення лише, якщо (а) GUI дозволено, (б) ми у Main-потоці
        if show_window and main_thread:
            cv2.imshow("WebCam Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # У фон-потоці без GUI немає способу «натиснути q», тому вихід реалізуйте власним флагом.

    cap.release()
    if show_window and main_thread:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Запуск напряму з консолі – показуємо вікно
    detection_loop(show_window=True)