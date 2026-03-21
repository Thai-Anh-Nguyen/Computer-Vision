import asyncio
import base64
import time
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from camera import VideoCamera
from process import ImageProcessor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="CV Camera App")

# Serve everything in /static at the URL path /static
# e.g. /static/main.js → file at static/main.js
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 looks for .html files inside the /templates folder
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
# Two camera instances — one per column in the UI.
# They live at module level so every request shares the same objects.
# (In Flask you'd do the same thing with globals.)

cameras: dict[int, VideoCamera] = {
    1: VideoCamera(),
    2: VideoCamera(),
}


# ---------------------------------------------------------------------------
# Pydantic models  (this is what FastAPI uses instead of request.get_json())
# ---------------------------------------------------------------------------
# Pydantic automatically:
#   - parses the JSON body
#   - validates types (cam_id must be int, source must be str)
#   - returns a clear 422 error if the client sends wrong data
# You get this for free just by defining the class.

class SourcePayload(BaseModel):
    cam_id: int
    source: str = ""          # empty string means "stop this camera"

class CapturePayload(BaseModel):
    cam_id: int
    # Literal restricts the value to only these strings.
    # If the client sends "banana" FastAPI rejects it automatically.
    step: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/filters")
def get_filters():
    """
    Return the list of filters that actually loaded successfully.

    The frontend calls this once on page load and builds the dropdown
    from the response — so the UI always matches what the backend
    actually has available. No hardcoded filter lists in HTML.
    """
    processor = ImageProcessor()
    return {"filters": processor.available_steps()}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main page.

    'request' must be passed into TemplateResponse — Jinja2 needs it
    to build things like url_for() inside the template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------

def _mjpeg_generator(cam_id: int):
    """
    Generator function that yields JPEG frames forever.

    This is a regular (sync) generator — not async — because cv2
    operations are CPU-bound and don't benefit from async.
    FastAPI's StreamingResponse handles running it in a thread pool.

    The boundary string is what separates frames in the MJPEG protocol.
    The browser knows to render each JPEG chunk as it arrives.
    """
    cam = cameras.get(cam_id)
    if cam is None:
        return

    boundary = b"--frame"

    while True:
        jpeg = cam.get_frame_jpeg(quality=80)

        if jpeg is None:
            # Camera not connected yet — send a gray placeholder
            # so the <img> tag doesn't show a broken image icon.
            jpeg = _blank_jpeg()

        # MJPEG frame format:
        #   --boundary\r\n
        #   Content-Type: image/jpeg\r\n
        #   Content-Length: <size>\r\n
        #   \r\n
        #   <jpeg bytes>\r\n
        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\n"
            + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
            + b"\r\n"
            + jpeg
            + b"\r\n"
        )

        time.sleep(0.033)   # ~30 fps ceiling


def _blank_jpeg() -> bytes:
    """Return a small gray JPEG as a placeholder when no camera is connected."""
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: int):
    """
    MJPEG streaming endpoint.

    The browser points an <img src="/video_feed/1"> at this route.
    FastAPI keeps the connection open and streams frames continuously.

    Note: this is a sync route (no async) because the generator is sync.
    FastAPI runs sync routes in a thread pool automatically.
    """
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Camera connect / disconnect
# ---------------------------------------------------------------------------

@app.post("/set_source")
async def set_source(payload: SourcePayload):
    """
    Connect or disconnect a camera.

    POST /set_source
    Body: { "cam_id": 1, "source": "0" }          ← connect webcam
          { "cam_id": 1, "source": "http://..." }  ← connect iPhone
          { "cam_id": 1, "source": "" }            ← disconnect
    """
    if payload.cam_id not in cameras:
        # HTTPException in FastAPI = automatic JSON error response
        raise HTTPException(status_code=400, detail="Invalid cam_id")

    cam = cameras[payload.cam_id]

    if payload.source == "":
        cam.stop()
        return {"ok": True, "msg": "stopped"}

    # Run start() in a thread pool because VideoCapture.open() can block
    # for several seconds on slow RTSP sources. We don't want that to
    # freeze the entire async event loop.
    await asyncio.get_event_loop().run_in_executor(
        None, cam.start, payload.source
    )

    return {"ok": True}


# ---------------------------------------------------------------------------
# Capture + process
# ---------------------------------------------------------------------------

@app.post("/capture")
async def capture(payload: CapturePayload):
    """
    Grab the current frame, run it through ImageProcessor, return both
    the original and processed image as base64 data URIs.

    Why base64? The frontend can set <img src="data:image/jpeg;base64,...">
    directly — no need for a separate image-serving route.
    """
    if payload.cam_id not in cameras:
        raise HTTPException(status_code=400, detail="Invalid cam_id")

    cam = cameras[payload.cam_id]
    frame = cam.get_frame_bgr()

    if frame is None:
        raise HTTPException(status_code=400, detail="No frame yet — is the camera connected?")

    # --- Encode original frame to base64 ---
    original_b64 = _frame_to_b64(frame)

    # --- Process frame ---
    # Run in thread pool for the same reason as cam.start() above:
    # cv2 operations are CPU-bound and would block the event loop.
    try:
        processor = ImageProcessor()
        processed_frame, results, process_time_ms = await asyncio.get_event_loop().run_in_executor(
            None, lambda: processor.process_frame(frame, step=payload.step)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # --- Encode processed frame to base64 ---
    processed_b64 = _frame_to_b64(processed_frame)

    return {
        "ok": True,
        "image": original_b64,
        "processed": processed_b64,
        "process_time_ms": round(process_time_ms, 2),
        "results": results,
        "step": payload.step,
    }


def _frame_to_b64(frame: np.ndarray) -> str:
    """
    Encode a BGR numpy frame as a JPEG data URI string.

    Returns something like:
        "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."

    The <img> tag can use this string directly as its src attribute.
    """
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # uvicorn is the ASGI server that runs FastAPI.
    # (Flask used its own built-in server — FastAPI needs an external one.)
    # host="0.0.0.0" means accept connections from any device on the network,
    # not just localhost — important for testing from your phone.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)