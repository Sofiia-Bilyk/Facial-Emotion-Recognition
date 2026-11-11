from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from dataset import CLASS_NAMES, train_loader, val_loader
from model import build_model


if __name__ == "__main__":
    torch.manual_seed(42)
    model = build_model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    best_macro_f1 = -1.0
    checkpoint_path = Path(__file__).resolve().with_name('best_model.pt')

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        model.train()

        total_loss = 0.0
        total_images = 0
        old_batch_size = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size
            if total_images > old_batch_size+1000:
                old_batch_size = total_images
                print(f"Training loss on {total_images:,} images: {total_loss / total_images:.4f}")

        print(f"Training loss: {total_loss / total_images:.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        num_classes = len(CLASS_NAMES)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                loss = loss_fn(logits, labels)
                predictions = logits.argmax(dim=1)
                for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                    confusion[actual, predicted] += 1

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (predictions == labels).sum().item()
                val_total += batch_size

        print(f"Validation loss: {val_loss_sum / val_total:.4f}")
        print(f"Validation accuracy: {val_correct / val_total:.2%}")

        true_positives = confusion.diag()
        actual_counts = confusion.sum(dim=1)
        predicted_counts = confusion.sum(dim=0)

        f1_per_class = 2 * true_positives / (actual_counts + predicted_counts).clamp_min(1)
        macro_f1 = f1_per_class.mean().item()
        print(f"Validation macro F1: {macro_f1:.4f}")

        precision = true_positives / predicted_counts.clamp_min(1)
        recall = true_positives / actual_counts.clamp_min(1)

        print(f"{'Expression':<12} {'Precision':>9} {'Recall':>9} {'F1':>9}")
        for label, class_name in enumerate(CLASS_NAMES):
            print(
                f"{class_name:<12} {precision[label].item():>9.2%} "
                f"{recall[label].item():>9.2%} {f1_per_class[label].item():>9.2%}"
            )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "macro_f1": macro_f1,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )
            print(f"Saved {checkpoint_path.name} from epoch {epoch}")
