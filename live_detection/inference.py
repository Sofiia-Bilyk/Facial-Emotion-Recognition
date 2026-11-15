from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from model import build_model


CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "best_model.pt"


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["class_names"]


def prepare_face(face: Image.Image) -> torch.Tensor:
    grayscale = face.convert("L")
    resized = grayscale.resize((48, 48), Image.Resampling.BILINEAR)
    return ToTensor()(resized).unsqueeze(0)


def predict_face(
    face: Image.Image, model: torch.nn.Module, class_names: tuple[str, ...]
) -> str:
    pixels = prepare_face(face)

    with torch.inference_mode():
        scores = model(pixels)

    class_index = scores.argmax(dim=1).item()
    return class_names[class_index]
