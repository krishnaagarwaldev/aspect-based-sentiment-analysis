"""
Preprocesses and exports normalized ABSA datasets into leakage-free splits:
- data_splits/train.csv (70%)
- data_splits/val.csv (15%)
- data_splits/test.csv (15%)
- data_splits/all_normalized.csv (100%)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.normalizer import ABSADataPipeline
from src.data.splitter import ReviewLevelSplitter

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ABSA_PrepareData")


def main():
    parser = argparse.ArgumentParser(description="Normalize and export ABSA dataset splits")
    parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Raw dataset directory")
    parser.add_argument("--output_dir", type=str, default="data_splits", help="Directory to save CSV splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and normalize
    pipeline = ABSADataPipeline(dataset_dir=args.dataset_dir, drop_conflict=True)
    all_samples = pipeline.load_all_domains()

    df_all = pipeline.to_dataframe(all_samples)
    all_path = out_dir / "all_normalized.csv"
    df_all.to_csv(all_path, index=False)
    logger.info(f"Saved all normalized samples ({len(df_all):,}) to: {all_path}")

    # 2. Leakage-free group split
    splitter = ReviewLevelSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_samples, val_samples, test_samples = splitter.split(all_samples)

    df_train = pipeline.to_dataframe(train_samples)
    df_val = pipeline.to_dataframe(val_samples)
    df_test = pipeline.to_dataframe(test_samples)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(f"Saved Train split ({len(df_train):,} samples) to: {train_path}")
    logger.info(f"Saved Val split ({len(df_val):,} samples) to: {val_path}")
    logger.info(f"Saved Test split ({len(df_test):,} samples) to: {test_path}")

    # Print summary statistics
    print("\n" + "=" * 65)
    print("DATASET SUMMARY BY SPLIT AND SENTIMENT")
    print("=" * 65)
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = df["sentiment"].value_counts().to_dict()
        dom_counts = df["domain"].value_counts().to_dict()
        print(f"[{name.upper()}] Total: {len(df):,}")
        print(f"   Sentiments: {counts}")
        print(f"   Domains:    {dom_counts}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
