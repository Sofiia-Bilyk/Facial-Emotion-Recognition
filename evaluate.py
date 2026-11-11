from pathlib import Path

import torch

from torch.utils.data import DataLoader

from dataset import BATCH_SIZE, FaceDataset
from model import build_model


def load_best_model():
    checkpoint_path = Path(__file__).resolve().with_name("best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["class_names"]


def build_test_loader(class_names):
    test_dir = Path(__file__).resolve().parent / "data" / "test"
    test_samples = []

    for label, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        image_paths = sorted(class_dir.glob("*.jpg"))

        if not image_paths:
            raise FileNotFoundError(f"No test JPEGs found in {class_dir}")

        test_samples.extend((path, label) for path in image_paths)

    return DataLoader(
        FaceDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


if __name__ == "__main__":
    model, class_names = load_best_model()
    test_loader = build_test_loader(class_names)

    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)

            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[actual, predicted] += 1

    correct_per_class = confusion.diag()
    actual_counts = confusion.sum(dim=1)
    predicted_counts = confusion.sum(dim=0)

    accuracy = correct_per_class.sum().item() / confusion.sum().item()
    f1_per_class = (
        2 * correct_per_class / (actual_counts + predicted_counts).clamp_min(1)
    )

    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test macro F1: {f1_per_class.mean().item():.4f}")
