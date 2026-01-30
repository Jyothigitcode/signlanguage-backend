from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import time

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("model.h5")
print("MODEL LOADED")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# MediaPipe Tasks (same as desktop)
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(options)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received"})

    file = request.files["image"]
    img_bytes = file.read()

    # Decode image
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify({"class": "No hand detected"})

    # Extract landmarks
    landmarks = []
    for lm in result.hand_landmarks[0]:
        landmarks.extend([lm.x, lm.y, lm.z])

    arr = np.array(landmarks, dtype=np.float32).reshape(1, 63)

    # Predict
    pred = model.predict(arr)
    idx = np.argmax(pred)
    label = class_names[idx]

    print("Predicted:", label)
    return jsonify({"class": label})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)