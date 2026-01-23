from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image

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

@app.get("/")
def root():
    return {"status": "running", "message": "Sign Language Detection API is live"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    label = class_names[class_id]

    return {"prediction": label}
