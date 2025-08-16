from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import threading

app = FastAPI(title="Camera Tracking API")

camera_url = 'http://null:6677'
current_mode = None
cap = None
lock = threading.Lock()

# -----------------------
# Video generator
# -----------------------
def video_feed_generator(mode):
    global cap, current_mode
    if cap is None:
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while current_mode == mode:
        success, frame = cap.read()
        if not success:
            break

        if mode == "face_tracking":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -----------------------
# API endpoints
# -----------------------
@app.get("/")
def index():
    html_content = """
    <html>
    <head><title>Camera Tracking</title></head>
    <body>
        <h2>Camera Tracking Feed</h2>
        <img src="/video_feed">
        <br><br>
        <a href="/start/face">Start Face Tracking</a><br>
        <a href="/start/object">Start Object Tracking</a><br>
        <a href="/stop">Stop Tracking</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/start/{mode}")
def start_tracking(mode: str, background_tasks: BackgroundTasks):
    global current_mode
    if mode not in ["face", "object"]:
        raise HTTPException(status_code=400, detail="Invalid mode")

    with lock:
        current_mode = f"{mode}_tracking"
    
    return {"status": f"{mode} tracking started", "stream_url": "/video_feed"}

@app.get("/stop")
def stop_tracking():
    global current_mode
    with lock:
        current_mode = None
    return {"status": "Tracking stopped"}

@app.get("/video_feed")
def video_feed():
    global current_mode
    if not current_mode:
        raise HTTPException(status_code=400, detail="No tracking running")
    return StreamingResponse(video_feed_generator(current_mode),
                             media_type="multipart/x-mixed-replace; boundary=frame")
