"""
Main Training Script for Aspect-Based Sentiment Analysis.
Usage:
  python train.py --model rnn --epochs 10
  python train.py --model bilstm --epochs 10
  python train.py --model bert --epochs 5 --batch_size 16
  python train.py --model bert_lora --epochs 5 --batch_size 16
"""

import argparse
import logging
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import (
    ABSADataPipeline,
    ABSADataset,
    NormalizedSample,
    ReviewLevelSplitter,
    Vocabulary,
    collate_fn_bert,
    collate_fn_rnn,
)
from src.models import ABSABert, ABSABertLoRA, BiLSTMWithAttention, VanillaRNN
from src.training import ABSATrainer, RunLogger

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ABSA_Train")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train ABSA Models")
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        choices=["rnn", "bilstm", "bert", "bert_lora"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (auto-selected if None)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Root dataset directory")
    parser.add_argument("--domain", type=str, default="all", help="Target domain or 'all'")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class-weighted loss")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (for debugging)")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="Hugging Face model")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load dataset (fast load from data_splits if available)
    splits_dir = Path("data_splits")
    pipeline = ABSADataPipeline(dataset_dir=args.dataset_dir, drop_conflict=True)

    if (splits_dir / "train.csv").exists() and (splits_dir / "val.csv").exists() and (splits_dir / "test.csv").exists():
        logger.info("Loading pre-split datasets from data_splits/...")
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        if args.domain != "all":
            train_df = train_df[train_df["domain"].str.lower() == args.domain.lower()]
            val_df = val_df[val_df["domain"].str.lower() == args.domain.lower()]
            test_df = test_df[test_df["domain"].str.lower() == args.domain.lower()]

        def df_to_samples(df):
            return [
                NormalizedSample(
                    id=row["id"],
                    review_id=row["review_id"],
                    text=str(row["text"]),
                    aspect=str(row["aspect"]),
                    sentiment=row["sentiment"],
                    sentiment_id=int(row["sentiment_id"]),
                    domain=row["domain"],
                    start_char=row.get("start_char"),
                    end_char=row.get("end_char"),
                )
                for _, row in df.iterrows()
            ]

        train_samples = df_to_samples(train_df)
        val_samples = df_to_samples(val_df)
        test_samples = df_to_samples(test_df)
    else:
        all_samples = pipeline.load_all_domains()
        if args.domain != "all":
            all_samples = [s for s in all_samples if s.domain.lower() == args.domain.lower()]
        splitter = ReviewLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
        train_samples, val_samples, test_samples = splitter.split(all_samples)

    if args.max_samples and len(train_samples) > args.max_samples:
        train_samples = train_samples[: args.max_samples]
        val_samples = val_samples[: max(args.max_samples // 4, 10)]
        test_samples = test_samples[: max(args.max_samples // 4, 10)]
        logger.info(f"Truncated datasets for fast debugging: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # 3. Class weights
    if not args.no_class_weights:
        weights_dict = pipeline.compute_class_weights(train_samples)
        # Class order: 0=negative, 1=neutral, 2=positive
        class_weights = torch.tensor(
            [weights_dict["negative"], weights_dict["neutral"], weights_dict["positive"]],
            dtype=torch.float,
        )
        logger.info(f"Computed Class Weights: Neg={class_weights[0]:.3f}, Neu={class_weights[1]:.3f}, Pos={class_weights[2]:.3f}")
    else:
        class_weights = None
        logger.info("Class weights disabled.")

    # 4. Prepare DataLoaders & Models
    is_transformer = args.model in ["bert", "bert_lora"]
    default_lr = 2e-5 if args.model == "bert" else (1e-4 if args.model == "bert_lora" else 1e-3)
    lr = args.lr if args.lr is not None else default_lr

    config = {
        "model_name": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "domain": args.domain,
        "class_weighted": not args.no_class_weights,
        "pretrained_model": args.pretrained_model if is_transformer else "none",
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples),
        "total_test_samples": len(test_samples),
    }

    run_logger = RunLogger(
        base_dir="runs",
        model_name=args.model,
        run_name=args.run_name,
        config=config,
    )
    logger.info(f"Run directory created at: {run_logger.run_dir}")

    train_ds = ABSADataset(train_samples)
    val_ds = ABSADataset(val_samples)
    test_ds = ABSADataset(test_samples)

    if is_transformer:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        collate_fn = partial(collate_fn_bert, tokenizer=tokenizer, max_len=128)

        if args.model == "bert":
            model = ABSABert(pretrained_model_name=args.pretrained_model, num_classes=3)
        else:
            model = ABSABertLoRA(pretrained_model_name=args.pretrained_model, num_classes=3, r=8, lora_alpha=16)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    else:
        # Build vocabulary from training texts and aspects
        vocab = Vocabulary(min_freq=2)
        vocab.build_vocab([s.text for s in train_samples] + [s.aspect for s in train_samples])
        logger.info(f"Built Vocabulary size: {len(vocab):,} tokens")
        vocab.save(run_logger.run_dir / "vocab.json")

        collate_fn = partial(collate_fn_rnn, vocab=vocab, max_len=128)

        if args.model == "rnn":
            model = VanillaRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)
        else:
            model = BiLSTMWithAttention(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler if is_transformer else None,
        class_weights=class_weights,
        device=device,
        run_logger=run_logger,
        patience=args.patience,
        epochs=args.epochs,
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    final_results = trainer.train()
    logger.info(f"Run completed! Best Val Macro-F1: {final_results['best_val_macro_f1']:.4f} | Test Macro-F1: {final_results['test_macro_f1']:.4f}")
    logger.info(f"All artifacts saved in: {run_logger.run_dir}")


if __name__ == "__main__":
    main()
