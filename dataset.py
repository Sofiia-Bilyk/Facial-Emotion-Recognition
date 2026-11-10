from pathlib import Path
from random import Random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

TRAIN_DIR = Path(r"C:\Coding\ML\Facial Emotion Recognition\data\train")
CLASS_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

samples = []
for label, class_name in enumerate(CLASS_NAMES):
    class_dir = TRAIN_DIR / class_name
    for image_path in sorted(class_dir.glob("*.jpg")):
        samples.append((image_path, label))

print(f"Found {len(samples):,} training images")
print(samples[0])

rng = Random(42)
train_samples = []
val_samples = []

for label in range(len(CLASS_NAMES)):
    class_samples = [sample for sample in samples if sample[1] == label]
    rng.shuffle(class_samples)

    val_count = round(len(class_samples) * 0.15)
    val_samples.extend(class_samples[:val_count])
    train_samples.extend(class_samples[val_count:])

print(f"Training: {len(train_samples):,}")
print(f"Validation: {len(val_samples):,}")


class FaceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            pixels = self.to_tensor(image.convert("L"))

        return pixels, label


BATCH_SIZE = 64

train_loader = DataLoader(
    FaceDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    FaceDataset(val_samples), batch_size=BATCH_SIZE, shuffle=False
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)
