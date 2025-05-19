from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from contextlib import asynccontextmanager
import asyncio, threading

import detect_webcam

# ──────────── запуск детектора ────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(
        target=detect_webcam.detection_loop,
        kwargs={"show_window": False},
        daemon=True
    ).start()
    yield
app = FastAPI(lifespan=lifespan)

# ───────────── MJPEG стрім ────────────────
async def mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
        frame = detect_webcam.get_latest_frame()
        if frame:
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame + b"\r\n")
        await asyncio.sleep(0.04)   # ~25 fps

@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ───────────── API карт та ресет ───────────
@app.get("/detected")
def get_detected():
    with detect_webcam.lock:
        cards = list(detect_webcam.detected_cards)

    total = 0
    for c in cards:
        v = c[:-1] if len(c) > 1 else c
        if v == "A":
            total += 1
        elif v in ("J", "Q", "K"):
            total += {"J":11, "Q":12, "K":13}[v]
        else:
            try:
                total += int(v)
            except ValueError:
                pass
    return {"cards": cards, "sum": total}

@app.post("/reset")
def reset():
    with detect_webcam.lock:
        detect_webcam.detected_cards.clear()
        detect_webcam.frame_counters.clear()
    return JSONResponse({"message": "reset complete"})

# ────────── найпростіший фронт ─────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html><html lang="uk">
<head>
<meta charset="utf-8"><title>Card Detector</title>
<style>
body{font-family:sans-serif;text-align:center;background:#222;color:#eee}
#cards{margin-top:12px;font-size:1.2em}
button{margin-top:8px;padding:6px 14px;font-size:1em}
img{border:2px solid #555}
</style>
</head>
<body>
<h2>Live WebCam Detection</h2>
<img id="cam" src="/video" width="640" height="480"/>
<div id="cards">Завантаження…</div>
<button id="btn">Reset</button>
<script>
async function poll(){
  const r = await fetch('/detected'); const j = await r.json();
  document.getElementById('cards').innerText =
      `Карти: ${j.cards.join(', ')} | Сума: ${j.sum}`;
}
setInterval(poll, 500);
document.getElementById('btn').onclick = async ()=>{
  await fetch('/reset', {method:'POST'});
};
poll();
</script>
</body></html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)