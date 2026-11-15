import cv2
import numpy as np
from PIL import Image


def load_face_detector() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Could not load the frontal-face detector.")
    return detector


def crop_largest_face(
    snapshot: Image.Image, detector: cv2.CascadeClassifier
) -> Image.Image | None:
    grayscale = np.asarray(snapshot.convert("L"))
    faces = detector.detectMultiScale(grayscale)

    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    return snapshot.crop((x, y, x + width, y + height))
