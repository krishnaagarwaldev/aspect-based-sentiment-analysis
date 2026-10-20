"""
Plotting and visualization utilities for ABSA runs:
- Epoch-wise training & validation loss curves
- Epoch-wise accuracy curves
- Epoch-wise Macro & Weighted F1 curves
- Confusion matrix heatmap
- Model comparison multi-metric bar chart
"""

from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_training_curves(history_df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Saves three distinct curves:
    1. loss_curve.png (train_loss vs val_loss)
    2. accuracy_curve.png (train_acc vs val_acc)
    3. f1_curve.png (train & val macro_f1 and weighted_f1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history_df["epoch"].values

    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(epochs, history_df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2.2, marker="o")
    ax.plot(epochs, history_df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2.2, marker="s")
    ax.set_title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Loss (Cross-Entropy)", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    # 2. Accuracy Curve
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(epochs, history_df["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2.2, marker="o")
    ax.plot(epochs, history_df["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2.2, marker="s")
    ax.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()

    # 3. F1 Curve (Macro-F1 & Weighted-F1)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(epochs, history_df["train_macro_f1"], label="Train Macro-F1", color="#9467bd", linewidth=2, marker="o")
    ax.plot(epochs, history_df["val_macro_f1"], label="Val Macro-F1", color="#8c564b", linewidth=2, marker="s")
    if "train_weighted_f1" in history_df.columns and "val_weighted_f1" in history_df.columns:
        ax.plot(epochs, history_df["train_weighted_f1"], label="Train Weighted-F1", color="#e377c2", linestyle="--", linewidth=1.8)
        ax.plot(epochs, history_df["val_weighted_f1"], label="Val Weighted-F1", color="#7f7f7f", linestyle="--", linewidth=1.8)
    ax.set_title("Training and Validation F1 Score", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("F1 Score", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_curve.png")
    plt.close()


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
):
    """Plots and saves confusion matrix heatmap with counts and percentages."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)

    # Compute row-normalized percentages
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm_arr / row_sums * 100, 0)

    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot[i, j] = f"{cm_arr[i, j]}\n({cm_norm[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.8,
        linecolor="#dddddd",
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison_chart(comparison_df: pd.DataFrame, output_path: Union[str, Path]):
    """
    Generates a grouped bar chart comparing all models across
    Accuracy, Macro-F1, and Weighted-F1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    if not available_metrics or comparison_df.empty:
        return

    models = comparison_df["model_name"].tolist()
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for i, metric in enumerate(available_metrics):
        offset = (i - (len(available_metrics) - 1) / 2) * width
        vals = comparison_df[metric].values
        label = metric.replace("test_", "").replace("_", " ").title()
        rects = ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)], alpha=0.9)
        # Add value labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="semibold")
    ax.set_title("Cross-Model Performance Comparison on Test Set", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
