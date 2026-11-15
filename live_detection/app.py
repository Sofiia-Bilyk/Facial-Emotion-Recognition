from io import BytesIO
from threading import Lock

from flask import Flask, render_template, request
from PIL import Image

from .face_detection import crop_largest_face, load_face_detector
from .inference import load_model, predict_face


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

model, class_names = load_model()
detector = load_face_detector()
prediction_lock = Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        with Image.open(BytesIO(request.get_data()), formats=["JPEG"]) as image:
            snapshot = image.convert("RGB")
    except (OSError, Image.DecompressionBombError):
        return {"error": "Expected a readable JPEG snapshot."}, 400

    with prediction_lock:
        face = crop_largest_face(snapshot, detector)
        expression = None if face is None else predict_face(face, model, class_names)

    return {"expression": expression}
