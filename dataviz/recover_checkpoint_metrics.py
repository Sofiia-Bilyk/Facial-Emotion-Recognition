"""Recompute per-expression metrics on the original validation split."""

import csv
import math
from pathlib import Path
import re
import sys

import torch

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = OUTPUT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from dataset import CLASS_NAMES, val_loader
from model import build_model


def recover_metrics():
    rows = []
    seen = {}
    for path in sorted((PROJECT_DIR / "best_models").glob("best_model_*epoch_*.pt")):
        budget, date = re.fullmatch(r"best_model_(\d+)epoch_(\d{8})", path.stem).groups()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        epoch = checkpoint["epoch"]
        if tuple(checkpoint["class_names"]) != CLASS_NAMES:
            raise ValueError(f"Unexpected class order in {path.name}")

        model = build_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images).argmax(dim=1)
                for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                    confusion[actual, predicted] += 1

        correct = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        precision = correct / predicted.clamp_min(1)
        recall = correct / support.clamp_min(1)
        f1 = 2 * correct / (support + predicted).clamp_min(1)
        accuracy = correct.sum().item() / support.sum().item()

        logs = list((PROJECT_DIR / "logs").glob(f"{budget}epoch_train_{date}*/epoch_metrics.csv"))
        if len(logs) != 1:
            raise ValueError(f"Expected one training log for {path.name}")
        with logs[0].open(newline="", encoding="utf-8") as file:
            logged = next(row for row in csv.DictReader(file) if int(row["epoch"]) == epoch)
        if not (
            math.isclose(accuracy, float(logged["val_accuracy"]), abs_tol=1e-7)
            and math.isclose(f1.mean().item(), checkpoint["macro_f1"], abs_tol=1e-7)
            and math.isclose(f1.mean().item(), float(logged["val_macro_f1"]), abs_tol=1e-7)
        ):
            raise ValueError(f"Validation results differ from the original run for {path.name}")
        if epoch in seen and not torch.equal(seen[epoch], confusion):
            raise ValueError(f"Checkpoints at epoch {epoch} produced different confusion matrices")
        seen[epoch] = confusion

        for label, expression in enumerate(CLASS_NAMES):
            rows.append((int(budget), path.name, epoch, expression, precision[label].item(),
                         recall[label].item(), f1[label].item(), support[label].item()))
        print(f"{path.name}: epoch {epoch}, accuracy {accuracy:.2%}, macro F1 {f1.mean():.4f}")

    if not rows:
        raise FileNotFoundError("No dated checkpoints found in best_models/")
    with (OUTPUT_DIR / "checkpoint_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch_budget", "checkpoint", "epoch", "expression",
                         "precision", "recall", "f1", "support"])
        writer.writerows(sorted(rows))


if __name__ == "__main__":
    torch.set_num_threads(4)
    recover_metrics()
