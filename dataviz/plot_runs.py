import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
LOGS_DIR = OUTPUT_DIR.parent / "logs"
BUDGETS = (10, 25, 50, 100, 200, 500)
BLUE, ORANGE = "#2166ac", "#b84e16"


def load_runs():
    runs = {}
    rows = []
    for budget in BUDGETS:
        paths = sorted(LOGS_DIR.glob(f"{budget}epoch_train_*/epoch_metrics.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one CSV for {budget} epochs, found {len(paths)}")
        metrics = pd.read_csv(paths[0]).sort_values("epoch").reset_index(drop=True)
        if metrics["epoch"].tolist() != list(range(1, budget + 1)):
            raise ValueError(f"Missing or duplicate epochs in {paths[0]}")
        runs[budget] = metrics
        # Training saves only on a strict improvement, so ties keep the first epoch.
        best = metrics.loc[metrics["val_macro_f1"].idxmax()]
        rows.append({
            "epoch_budget": budget,
            "selected_epoch": int(best["epoch"]),
            "best_val_macro_f1": best["val_macro_f1"],
            "selected_val_accuracy": best["val_accuracy"],
            "final_val_macro_f1": metrics.iloc[-1]["val_macro_f1"],
            "elapsed_hours": metrics["elapsed_seconds"].sum() / 3600,
        })
    return runs, pd.DataFrame(rows)


def save(fig, filename):
    fig.savefig(OUTPUT_DIR / filename, dpi=200, facecolor="white")
    plt.close(fig)


def plot_distribution(counts, title, filename):
    total = sum(counts.values())
    shares = [count / total for count in counts.values()]

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    bars = ax.bar(list(counts), shares, color=plt.get_cmap("tab10").colors[:len(counts)])
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n{share:.1%}"
            for count, share in zip(counts.values(), shares)
        ],
        padding=5,
    )
    ax.set(
        title=title,
        ylabel="Share of the full dataset",
        ylim=(0, max(shares) * 1.25),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, filename)


def plot_learning_curves(metrics):
    best = metrics.loc[metrics["val_macro_f1"].idxmax()]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    fig.suptitle("Learning curves for the 500-epoch run", fontsize=16)
    for ax, metric in zip(axes[:2], ("loss", "accuracy")):
        ax.plot(metrics["epoch"], metrics[f"train_{metric}"], color=BLUE, label="Training")
        ax.plot(metrics["epoch"], metrics[f"val_{metric}"], color=ORANGE, label="Validation")
        ax.set_ylabel(metric.title())
        ax.legend(loc="upper right")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].plot(metrics["epoch"], metrics["val_macro_f1"], color=ORANGE)
    axes[2].scatter(best["epoch"], best["val_macro_f1"], color=BLUE, zorder=3)
    axes[2].annotate(
        f"Selected epoch {int(best['epoch'])} | F1 {best['val_macro_f1']:.3f}",
        (best["epoch"], best["val_macro_f1"]), xytext=(8, 9),
        textcoords="offset points", color=BLUE,
    )
    axes[2].set(ylabel="Validation macro F1", xlabel="Epoch")
    axes[2].margins(y=0.18)
    for ax in axes:
        ax.axvline(best["epoch"], color=BLUE, linestyle="--", alpha=0.45)
    save(fig, "01_learning_curves.png")


def plot_epoch_budget(summary):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    positions = list(range(len(summary)))
    ax.plot(positions, summary["best_val_macro_f1"], "o-", color=BLUE, label="Selected checkpoint")
    ax.plot(positions, summary["final_val_macro_f1"], "s--", color=ORANGE,
            markerfacecolor="none", label="Final epoch")
    for x, row in zip(positions, summary.itertuples()):
        ax.annotate(f"Epoch {row.selected_epoch}", (x, row.best_val_macro_f1),
                    xytext=(0, 10), textcoords="offset points", ha="center", color=BLUE)
    ax.set_xticks(positions, summary["epoch_budget"])
    ax.set(title="Best and final validation scores by epoch budget",
           xlabel="Requested epochs", ylabel="Validation macro F1")
    ax.set_ylim(0.32, 0.59)
    ax.legend(loc="lower right")
    save(fig, "02_epoch_budget.png")


def plot_runtime(summary):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    ax.plot(summary["elapsed_hours"], summary["best_val_macro_f1"], "o-", color=BLUE)
    for row in summary.itertuples():
        dx = -8 if row.epoch_budget == 500 else 8
        dy = -15 if row.epoch_budget == 100 else 8
        ax.annotate(str(row.epoch_budget), (row.elapsed_hours, row.best_val_macro_f1),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left")
    ax.set(title="Validation gains versus training time", xlabel="Recorded epoch time (hours)",
           ylabel="Best validation macro F1")
    ax.margins(x=0.08, y=0.18)
    save(fig, "03_runtime_tradeoff.png")


def plot_class_recall(metrics_path, checkpoints=False):
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"No completed epochs in {metrics_path}")
    if checkpoints:
        metrics = metrics.drop_duplicates(subset=["epoch", "expression", "recall"])
    curves = metrics.pivot(index="epoch", columns="expression", values="recall").sort_index()
    if curves.isna().any().any():
        raise ValueError(f"Missing expression metrics in {metrics_path}")

    combined, combined_axis = plt.subplots(figsize=(9, 5), layout="constrained")
    prefix = "checkpoint_recall" if checkpoints else "recall"
    xlabel = "Saved checkpoint epoch" if checkpoints else "Epoch"
    line_style = {"linestyle": "-", "marker": "o", "markersize": 5} if checkpoints else {
        "marker": ".", "markersize": 3,
    }
    for expression, color in zip(curves.columns, plt.get_cmap("tab10").colors):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.plot(curves.index, curves[expression], color=color, **line_style)
        ax.set(title=expression.title(),
               xlabel=xlabel, ylabel="Validation recall", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        save(fig, f"{prefix}_{expression}.png")
        combined_axis.plot(curves.index, curves[expression], color=color,
                           label=expression, **line_style)

    combined_axis.set(title="Validation by expression",
                      xlabel=xlabel, ylabel="Validation recall", ylim=(0, 1))
    combined_axis.yaxis.set_major_formatter(PercentFormatter(1))
    combined_axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(combined, f"{prefix}_all_expressions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-metrics", type=Path, help="Plot recall from this run's class_metrics.csv")
    parser.add_argument("--checkpoint-metrics", type=Path, help="Plot recovered checkpoint recall")
    args = parser.parse_args()
    plt.rcParams.update({
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.22,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    if args.checkpoint_metrics:
        plot_class_recall(args.checkpoint_metrics, checkpoints=True)
    elif args.class_metrics:
        plot_class_recall(args.class_metrics)
    else:
        runs, summary = load_runs()
        plot_distribution(
            {"Training": 24_403, "Validation": 4_306, "Test": 7_178},
            "FER2013 dataset split",
            "00_dataset_split.png",
        )
        plot_distribution(
            {
                "Angry": 4_953, "Disgust": 547, "Fear": 5_121, "Happy": 8_989,
                "Neutral": 6_198, "Sad": 6_077, "Surprise": 4_002,
            },
            "Expression distribution in FER2013",
            "00_expression_distribution.png",
        )
        plot_learning_curves(runs[500])
        plot_epoch_budget(summary)
        plot_runtime(summary)
