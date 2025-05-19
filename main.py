from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import threading

import detect_webcam

@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=detect_webcam.detection_loop, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/detected")
def get_detected():
    with detect_webcam.lock:
        cards = list(detect_webcam.detected_cards)

    total = 0
    for c in cards:
        value = c[:-1] if len(c) > 1 else c

        if value == "A":
            total += 1
        elif value in ("J", "Q", "K"):
            total += {"J": 11, "Q": 12, "K": 13}[value]
        else:
            try:
                total += int(value)
            except ValueError:
                pass

    return {"cards": cards, "sum": total}

@app.post("/reset")
def reset():
    with detect_webcam.lock:
        detect_webcam.detected_cards.clear()
        detect_webcam.frame_counters.clear()
    return JSONResponse({"message": "reset complete"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)