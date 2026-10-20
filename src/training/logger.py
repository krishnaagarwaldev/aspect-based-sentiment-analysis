"""
Run logger managing directory structures, epoch histories, metric artifacts, and plots.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.visualizer import plot_confusion_matrix, plot_training_curves


class RunLogger:
    def __init__(
        self,
        base_dir: str = "runs",
        model_name: str = "bert",
        run_name: str = None,
        config: Dict[str, Any] = None,
    ):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.config = config or {}

        # Generate unique run ID if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Also find existing runs to determine sequential index
            model_dir = self.base_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            existing_runs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            run_idx = len(existing_runs) + 1
            self.run_id = f"run_{run_idx:03d}_{timestamp}"
        else:
            self.run_id = run_name

        self.run_dir = self.base_dir / self.model_name / self.run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []

        # Save initial config
        self.save_config()

    def save_config(self):
        """Saves experiment configuration."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def log_epoch(self, epoch_metrics: Dict[str, Any]):
        """Records metrics for an epoch and updates training_history.csv."""
        self.history.append(epoch_metrics)
        df = pd.DataFrame(self.history)
        df.to_csv(self.run_dir / "training_history.csv", index=False)

    def finish_run(
        self,
        test_metrics: Dict[str, Any],
        classification_rep_text: str,
        class_names: List[str] = None,
    ):
        """
        Finalizes run artifacts:
        - metrics.json
        - classification_report.txt
        - loss_curve.png
        - accuracy_curve.png
        - f1_curve.png
        - confusion_matrix.png
        """
        if class_names is None:
            class_names = ["negative", "neutral", "positive"]

        # 1. Save final test metrics
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=4)

        # 2. Save text classification report
        report_path = self.run_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classification_rep_text)

        # 3. Generate curves if history exists
        if self.history:
            df = pd.DataFrame(self.history)
            plot_training_curves(df, self.run_dir)

        # 4. Generate confusion matrix plot
        if "confusion_matrix" in test_metrics:
            cm = test_metrics["confusion_matrix"]
            cm_path = self.run_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                cm=cm,
                class_names=class_names,
                output_path=cm_path,
                title=f"Test Confusion Matrix ({self.model_name})",
            )
