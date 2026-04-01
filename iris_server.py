import cv2
import numpy as np
import torch
import json
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(
    "Running on:",
    DEVICE,
    "| torch:",
    torch.__version__,
    "| cuda:",
    torch.version.cuda,
)

model = YOLO("yolov8m.pt")
model.to(DEVICE)

IMPORTANT = {
    "person",
    "dog",
    "cat",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "chair",
    "bottle",
    "cup",
    "cell phone"
}


def get_position(x1,x2,width):

    center = (x1+x2)/2

    if center < width/3:
        return "left"

    elif center < 2*width/3:
        return "center"

    return "right"


@app.websocket("/stream")

async def stream(ws:WebSocket):

    await ws.accept()

    while True:

        data = await ws.receive_bytes()

        frame = cv2.imdecode(
            np.frombuffer(data,np.uint8),
            cv2.IMREAD_COLOR
        )

        h,w,_ = frame.shape

        results = model.predict(
            frame,
            imgsz=416,
            conf=0.5,
            verbose=False,
            device=DEVICE,
        )

        detections = []

        for r in results:

            for box in r.boxes:

                label = model.names[int(box.cls[0])]

                if label not in IMPORTANT:
                    continue

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                pos = get_position(x1,x2,w)

                detections.append({
                    "object":label,
                    "position":pos
                })

        await ws.send_text(json.dumps(detections))


@app.get("/")

def root():

    return {"status":"IRIS server running"}
