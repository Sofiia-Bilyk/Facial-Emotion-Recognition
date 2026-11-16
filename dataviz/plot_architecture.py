from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


def draw_stack(ax, x, size, layers, color):
    y = 3 - size / 2
    for layer in reversed(range(layers)):
        offset = layer * 0.075
        ax.add_patch(Rectangle(
            (x - size / 2 - offset, y + offset), size, size,
            facecolor=color, edgecolor="#294764", linewidth=1,
        ))


def main():
    fig, ax = plt.subplots(figsize=(14, 5.1), layout="constrained")
    ax.set(xlim=(-0.25, 14), ylim=(0, 5.6), aspect="equal")
    ax.axis("off")
    ax.set_title("Facial expression CNN", fontsize=19, fontweight="bold", loc="left")

    draw_stack(ax, 0.85, 1.35, 1, "#ededed")
    ax.add_patch(Ellipse((0.85, 3), 0.78, 1.02, facecolor="#a7a7a7"))
    for x in (0.69, 1.01):
        ax.add_patch(Ellipse((x, 3.13), 0.10, 0.07, facecolor="#383838"))
    ax.plot([0.69, 0.85, 1.01], [2.89, 2.82, 2.89], color="#383838", lw=1.5)

    blocks = [(3.10, 1.28, 5, 32, 24), (5.22, 0.98, 7, 64, 12), (7.25, 0.71, 9, 128, 6)]
    for n, (x, size, layers, channels, width) in enumerate(blocks, start=1):
        draw_stack(ax, x, size, layers, "#bcd3e8")
        ax.text(x, 4.25, f"Conv block {n}", ha="center", weight="bold", fontsize=12)
        ax.text(x, 1.83, f"{channels} × {width} × {width}", ha="center", fontsize=11)

    ax.text(0.85, 4.25, "Grayscale input", ha="center", weight="bold", fontsize=12)
    ax.text(0.85, 1.83, "1 × 48 × 48", ha="center", fontsize=11)
    for start, end in [(1.60, 2.02), (3.82, 4.20), (5.79, 6.20), (7.68, 9.0)]:
        ax.annotate("", (end, 3), (start, 3), arrowprops={"arrowstyle": "->", "color": "#516579"})

    ax.text(9.45, 4.72, "Global average pooling", ha="center", weight="bold", fontsize=11)
    ax.text(9.45, 4.42, "Flatten → dropout (0.3)", ha="center", fontsize=10)
    feature_y = [2.08, 2.45, 2.82, 3.19, 3.56, 3.93]
    output_y = [4.1 - i * 0.37 for i in range(7)]
    labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    for y in feature_y:
        for target_y in output_y:
            ax.plot([9.45, 11.47], [y, target_y], color="#7896b3", alpha=0.38, lw=0.7)
    ax.scatter([9.45] * len(feature_y), feature_y, color="#588548", s=66, zorder=3)
    ax.scatter([11.47] * 7, output_y, color=plt.get_cmap("tab10").colors[:7], s=66, zorder=3)
    for label, y in zip(labels, output_y):
        ax.text(11.72, y, label, va="center", fontsize=11)
    ax.text(9.45, 1.48, "128 features", ha="center", fontsize=11)
    ax.text(11.47, 4.72, "Linear layer", ha="center", weight="bold", fontsize=11)
    ax.text(11.47, 1.48, "7 logits", ha="center", fontsize=11)

    ax.text(0.18, 0.82, "Each block: 3 × 3 convolution (padding 1) → batch normalization → ReLU → 2 × 2 max pooling",
            fontsize=10.5, color="#294764")
    ax.text(0.18, 0.45, "Shapes are channels × height × width. Stacks and feature nodes show a subset for clarity.",
            fontsize=10, color="#516579")
    fig.savefig(Path(__file__).with_name("model_architecture.png"), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
