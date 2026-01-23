from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
import pickle
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(title="Sign Language Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# MediaPipe (same as your app)
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
hand_detector = vision.HandLandmarker.create_from_options(options)

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    np_img = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        return {"prediction": None, "confidence": 0}

    landmarks = []
    for lm in result.hand_landmarks[0]:
        landmarks.extend([lm.x, lm.y, lm.z])

    data = np.array(landmarks, dtype=np.float32).reshape(1, -1)

    preds = model.predict(data, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "prediction": class_names[idx],
        "confidence": float(preds[idx])
    }
