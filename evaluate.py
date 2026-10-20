"""
Standalone Checkpoint Evaluation Script for ABSA.
Usage:
  python evaluate.py --checkpoint runs/bilstm/run_001/checkpoints/best_model.pt --domain all
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import (
    ABSADataPipeline,
    ABSADataset,
    ReviewLevelSplitter,
    Vocabulary,
    collate_fn_bert,
    collate_fn_rnn,
)
from src.evaluation.metrics import compute_classification_metrics, get_classification_report_text
from src.evaluation.visualizer import plot_confusion_matrix
from src.models import ABSABert, ABSABertLoRA, BiLSTMWithAttention, VanillaRNN

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ABSA_Eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ABSA Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Root dataset directory")
    parser.add_argument("--domain", type=str, default="all", help="Evaluation domain or 'all'")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "full"])
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for reports/plots")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Determine run directory and load config if present
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_type = config.get("model_name", "bilstm")
        seed = config.get("seed", 42)
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
    else:
        # Fallback heuristic from path
        model_type = "bert" if "bert" in str(ckpt_path) else "bilstm"
        seed = 42
        pretrained_model = "bert-base-uncased"

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pipeline = ABSADataPipeline(dataset_dir=args.dataset_dir, drop_conflict=True)
    all_samples = pipeline.load_all_domains()

    if args.domain != "all":
        all_samples = [s for s in all_samples if s.domain.lower() == args.domain.lower()]

    splitter = ReviewLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    train_samples, val_samples, test_samples = splitter.split(all_samples)

    if args.split == "train":
        eval_samples = train_samples
    elif args.split == "val":
        eval_samples = val_samples
    elif args.split == "test":
        eval_samples = test_samples
    else:
        eval_samples = all_samples

    eval_ds = ABSADataset(eval_samples)

    # Initialize model
    is_transformer = model_type in ["bert", "bert_lora"]
    if is_transformer:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        collate_fn = partial(collate_fn_bert, tokenizer=tokenizer, max_len=128)
        if model_type == "bert":
            model = ABSABert(pretrained_model_name=pretrained_model, num_classes=3)
        else:
            model = ABSABertLoRA(pretrained_model_name=pretrained_model, num_classes=3)
    else:
        vocab_path = run_dir / "vocab.json"
        if vocab_path.exists():
            vocab = Vocabulary.load(vocab_path)
            logger.info(f"Loaded trained Vocabulary ({len(vocab):,} tokens) from {vocab_path}")
        else:
            vocab = Vocabulary(min_freq=2)
            vocab.build_vocab([s.text for s in train_samples] + [s.aspect for s in train_samples])
            logger.info(f"Built Vocabulary ({len(vocab):,} tokens) from split")
        collate_fn = partial(collate_fn_rnn, vocab=vocab, max_len=128)
        if model_type == "rnn":
            model = VanillaRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)
        else:
            model = BiLSTMWithAttention(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)

    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load checkpoint weights
    logger.info(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(**batch_gpu)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_gpu["labels"].cpu().numpy())

    metrics = compute_classification_metrics(all_labels, all_preds)
    rep_text = get_classification_report_text(all_labels, all_preds)

    logger.info(f"\nEvaluation on split='{args.split}' domain='{args.domain}':\n" + rep_text)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | Weighted-F1: {metrics['weighted_f1']:.4f}")

    # Save results
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(out_dir / "eval_report.txt", "w", encoding="utf-8") as f:
        f.write(rep_text)

    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=["negative", "neutral", "positive"],
        output_path=out_dir / "eval_confusion_matrix.png",
        title=f"Evaluation Matrix ({model_type.upper()} - {args.domain})",
    )
    logger.info(f"Saved evaluation results to {out_dir}")


if __name__ == "__main__":
    main()
