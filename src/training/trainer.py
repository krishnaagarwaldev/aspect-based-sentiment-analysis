"""
Unified Trainer for Aspect-Based Sentiment Analysis.
Supports Vanilla RNN, BiLSTM-Attn, BERT, and BERT-LoRA.
Includes class-weighted loss, validation on Macro-F1, early stopping, and checkpoint resuming.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics, get_classification_report_text
from .logger import RunLogger

logger = logging.getLogger(__name__)


class ABSATrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        run_logger: Optional[RunLogger] = None,
        patience: int = 5,
        epochs: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_logger = run_logger
        self.patience = patience
        self.epochs = epochs

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.best_val_macro_f1 = -1.0
        self.best_epoch = -1
        self.start_epoch = 1
        self.use_amp = (device == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch_gpu["labels"]

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(**batch_gpu)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / len(all_labels)
        metrics = compute_classification_metrics(all_labels, all_preds)
        return {
            "train_loss": epoch_loss,
            "train_acc": metrics["accuracy"],
            "train_macro_f1": metrics["macro_f1"],
            "train_weighted_f1": metrics["weighted_f1"],
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "Eval") -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(loader, desc=desc, leave=False):
            batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch_gpu["labels"]

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(**batch_gpu)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        eval_loss = total_loss / max(len(all_labels), 1)
        metrics = compute_classification_metrics(all_labels, all_preds)
        metrics["loss"] = eval_loss
        metrics["all_preds"] = all_preds
        metrics["all_labels"] = all_labels
        return metrics

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_macro_f1": self.best_val_macro_f1,
            "history": self.run_logger.history if self.run_logger else [],
        }
        torch.save(state, path)
        if is_best and self.run_logger:
            torch.save(state, self.run_logger.checkpoint_dir / "best_model.pt")

    def resume_from_checkpoint(self, checkpoint_path: str):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_macro_f1 = checkpoint.get("best_val_macro_f1", -1.0)
        if self.run_logger and checkpoint.get("history"):
            self.run_logger.history = checkpoint["history"]
        logger.info(f"Resumed at epoch {self.start_epoch} (Best Val Macro-F1: {self.best_val_macro_f1:.4f})")

    def train(self) -> Dict[str, float]:
        epochs_no_improve = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_res = self.train_epoch(epoch)
            val_res = self.evaluate(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")

            val_macro_f1 = val_res["macro_f1"]
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_res["train_loss"],
                "val_loss": val_res["loss"],
                "train_acc": train_res["train_acc"],
                "val_acc": val_res["accuracy"],
                "train_macro_f1": train_res["train_macro_f1"],
                "val_macro_f1": val_macro_f1,
                "train_weighted_f1": train_res["train_weighted_f1"],
                "val_weighted_f1": val_res["weighted_f1"],
            }

            logger.info(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_res['train_loss']:.4f} Acc: {train_res['train_acc']:.4f} F1: {train_res['train_macro_f1']:.4f} | "
                f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']:.4f} F1: {val_macro_f1:.4f}"
            )

            if self.run_logger:
                self.run_logger.log_epoch(epoch_log)

            # Checkpoint saving & Early stopping
            is_best = val_macro_f1 > self.best_val_macro_f1
            if is_best:
                self.best_val_macro_f1 = val_macro_f1
                self.best_epoch = epoch
                epochs_no_improve = 0
                if self.run_logger:
                    self.save_checkpoint(self.run_logger.checkpoint_dir / "best_model.pt", epoch, is_best=True)
            else:
                epochs_no_improve += 1

            if self.run_logger:
                self.save_checkpoint(self.run_logger.checkpoint_dir / "latest_checkpoint.pt", epoch)

            if epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (Patience={self.patience})")
                break

        # Final evaluation using best model checkpoint
        if self.run_logger and (self.run_logger.checkpoint_dir / "best_model.pt").exists():
            logger.info(f"Loading best checkpoint from epoch {self.best_epoch} for test evaluation...")
            best_ckpt = torch.load(self.run_logger.checkpoint_dir / "best_model.pt", map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])

        test_res = self.evaluate(self.test_loader, desc="[Test Set Evaluation]")
        rep_text = get_classification_report_text(test_res["all_labels"], test_res["all_preds"])

        logger.info("\n" + "=" * 50 + "\nFINAL TEST EVALUATION REPORT:\n" + rep_text + "=" * 50)

        # Remove raw prediction arrays from test_res dictionary before saving JSON
        clean_test_metrics = {
            "test_accuracy": test_res["accuracy"],
            "test_macro_f1": test_res["macro_f1"],
            "test_weighted_f1": test_res["weighted_f1"],
            "test_loss": test_res["loss"],
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "per_class": test_res["per_class"],
            "confusion_matrix": test_res["confusion_matrix"],
            "total_test_samples": test_res["total_samples"],
        }

        if self.run_logger:
            self.run_logger.finish_run(
                test_metrics=clean_test_metrics,
                classification_rep_text=rep_text,
            )

        return clean_test_metrics
