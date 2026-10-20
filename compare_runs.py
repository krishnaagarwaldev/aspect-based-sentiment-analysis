"""
Compare all experiment runs stored in runs/
Generates:
- runs/comparison/model_comparison.csv
- runs/comparison/model_comparison.png
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.visualizer import plot_model_comparison_chart

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ABSA_Compare")


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame()

    for model_dir in runs_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ["comparison"]:
            continue

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metrics_file = run_dir / "metrics.json"
            config_file = run_dir / "config.json"

            if not metrics_file.exists():
                continue

            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                continue

            config = {}
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception:
                    pass

            row = {
                "model_name": config.get("model_name", model_dir.name),
                "run_id": run_dir.name,
                "domain": config.get("domain", "all"),
                "test_accuracy": round(metrics.get("test_accuracy", 0.0), 4),
                "test_macro_f1": round(metrics.get("test_macro_f1", 0.0), 4),
                "test_weighted_f1": round(metrics.get("test_weighted_f1", 0.0), 4),
                "best_val_macro_f1": round(metrics.get("best_val_macro_f1", 0.0), 4),
                "best_epoch": metrics.get("best_epoch", -1),
                "epochs": config.get("epochs", None),
                "batch_size": config.get("batch_size", None),
                "learning_rate": config.get("learning_rate", None),
                "class_weighted": config.get("class_weighted", True),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="test_macro_f1", ascending=False).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare ABSA training runs")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing all runs")
    parser.add_argument("--output_dir", type=str, default="runs/comparison", help="Output comparison directory")
    args = parser.parse_args()

    runs_path = Path(args.runs_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = collect_runs(runs_path)

    if df.empty:
        logger.warning(f"No completed runs found with metrics.json in '{runs_path}'")
        return

    csv_path = out_path / "model_comparison.csv"
    png_path = out_path / "model_comparison.png"

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved run comparison table to: {csv_path}")

    # Generate comparison plot
    plot_model_comparison_chart(df, png_path)
    logger.info(f"Saved run comparison plot to: {png_path}")

    # Display clean table
    print("\n" + "=" * 80)
    print("MODEL EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    print(df[["model_name", "run_id", "domain", "test_accuracy", "test_macro_f1", "test_weighted_f1"]].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
