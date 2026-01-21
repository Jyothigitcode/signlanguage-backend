from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
model = tf.keras.models.load_model(
    "model.h5"
)

with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# ==========================================================
# MEDIAPIPE HAND LANDMARKER (STATIC IMAGE MODE)
# ==========================================================
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,   # ✅ STATIC IMAGE
    num_hands=1,
    min_hand_detection_confidence=0.5
)

hand_detector = vision.HandLandmarker.create_from_options(options)

# ==========================================================
# HEALTH CHECK
# ==========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Sign Language Detection API is live"
    })

# ==========================================================
# API INFO
# ==========================================================
@app.route("/predict", methods=["GET"])
def predict_info():
    return jsonify({
        "endpoint": "/predict",
        "method": "POST",
        "field_name": "image",
        "usage": "Send image using multipart/form-data"
    })

# ==========================================================
# PREDICTION ENDPOINT
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    # Decode image
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detect hand landmarks
    result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify({
            "gesture": None,
            "confidence": 0.0
        })

    # Extract landmarks (63 values)
    landmarks = []
    for lm in result.hand_landmarks[0]:
        landmarks.extend([lm.x, lm.y, lm.z])

    X = np.expand_dims(
        np.array(landmarks, dtype=np.float32),
        axis=0
    )

    # Predict
    preds = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(preds))

    return jsonify({
        "gesture": class_names[idx],
        "confidence": float(preds[idx])
    })

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
