

import cv2
import json
import os
import time
import numpy as np
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urljoin
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

# ----------------------------------------------------------
# FASTAPI INIT
# ----------------------------------------------------------

app = FastAPI(title="IRIS AI Server")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

print("Starting IRIS server...")

# ----------------------------------------------------------
# LOAD YOLO MODEL
# ----------------------------------------------------------

model = YOLO("yolov8m.pt")

# ----------------------------------------------------------
# CLASS LIST
# ----------------------------------------------------------

CLASSES = [
"person","bicycle","car","motorcycle","bus","truck",
"dog","cat","chair","couch","bed","tv",
"laptop","cell phone","bottle","cup",
"book","backpack","umbrella","handbag",
"traffic light","stop sign","bench",
"parking meter","potted plant",
"sink","refrigerator","clock",
"scissors","teddy bear","toothbrush"
]

# ----------------------------------------------------------
# DANGER OBJECTS
# ----------------------------------------------------------

DANGER_CLASSES = [
"car","motorcycle","bus","truck","dog"
]

# ----------------------------------------------------------
# TRACK MEMORY
# ----------------------------------------------------------

track_memory = {}

# ----------------------------------------------------------
# POSITION DETECTION
# ----------------------------------------------------------

def get_position(x1,x2,width):

    center = (x1 + x2) / 2

    if center < width/3:
        return "left"

    elif center < 2*width/3:
        return "center"

    else:
        return "right"

# ----------------------------------------------------------
# SPEED ESTIMATION
# ----------------------------------------------------------

def estimate_speed(track_id, center):

    now = time.time()

    speed = 0

    if track_id in track_memory:

        prev_center, prev_time = track_memory[track_id]

        dist = np.linalg.norm(
            np.array(center) - np.array(prev_center)
        )

        dt = now - prev_time

        if dt > 0:
            speed = dist / dt

    track_memory[track_id] = (center, now)

    return round(speed,2)

# ----------------------------------------------------------
# OBJECT DETECTION ENDPOINT
# ----------------------------------------------------------

@app.post("/detect")

async def detect(file: UploadFile = File(...)):

    data = await file.read()

    img = cv2.imdecode(
        np.frombuffer(data,np.uint8),
        cv2.IMREAD_COLOR
    )

    height,width,_ = img.shape

    # YOLO with ByteTrack
    results = model.track(
        img,
        conf=0.5,
        persist=True,
        tracker="bytetrack.yaml"
    )

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            label = model.names[int(box.cls[0])]

            if label not in CLASSES:
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            pos = get_position(x1,x2,width)

            center = ((x1+x2)/2,(y1+y2)/2)

            # Track ID
            track_id = -1

            if box.id is not None:
                track_id = int(box.id[0])

            # Speed
            speed = estimate_speed(track_id, center)

            # Danger detection
            danger = label in DANGER_CLASSES

            detections.append({

                "id": track_id,
                "object": label,
                "position": pos,
                "speed": speed,
                "danger": danger

            })

    return {"detections": detections}

# ----------------------------------------------------------
# OLLAMA CHAT ENDPOINT
# ----------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str

def call_ollama(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        urljoin(OLLAMA_BASE_URL, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))

def ollama_chat(prompt: str) -> str:
    try:
        try:
            body = call_ollama(
                "api/chat",
                {
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            reply = body.get("message", {}).get("content", "").strip()
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

            body = call_ollama(
                "api/generate",
                {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            reply = body.get("response", "").strip()

        if not reply:
            raise RuntimeError("Ollama returned an empty response.")

        return reply

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=exc.code,
            detail=f"Ollama HTTP error: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama server at {OLLAMA_BASE_URL}: {exc.reason}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama chat failed: {exc}"
        ) from exc

@app.post("/chat")

async def chat(chat_request: Optional[ChatRequest] = None, prompt: Optional[str] = None):

    final_prompt = prompt

    if chat_request and chat_request.prompt:
        final_prompt = chat_request.prompt

    if not final_prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")

    reply = ollama_chat(final_prompt)

    return {"response": reply}

# ----------------------------------------------------------
# SERVER STATUS
# ----------------------------------------------------------

@app.get("/")

def root():

    return {
        "status":"IRIS AI server running",
        "model":"YOLOv8m",
        "tracking":"ByteTrack",
        "assistant":"Ollama Llama3"
    }
