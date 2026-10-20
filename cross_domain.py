"""
Cross-Domain ABSA Experiment Runner.
Evaluates models across domains (In-Domain and Zero-Shot Cross-Domain).
Outputs: runs/comparison/cross_domain_results.csv
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import (
    ABSADataPipeline,
    ABSADataset,
    NormalizedSample,
    ReviewLevelSplitter,
    Vocabulary,
    collate_fn_bert,
    collate_fn_rnn,
)
from src.evaluation.metrics import compute_classification_metrics
from src.models import ABSABert, ABSABertLoRA, BiLSTMWithAttention, VanillaRNN

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ABSA_CrossDomain")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Domain ABSA Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to evaluate across domains")
    parser.add_argument("--model", type=str, default="bert_lora", choices=["rnn", "bilstm", "bert", "bert_lora"])
    parser.add_argument("--domains", nargs="+", default=["restaurant", "laptop", "college", "hospitality"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--splits_dir", type=str, default="data_splits")
    parser.add_argument("--output_csv", type=str, default="runs/comparison/cross_domain_results.csv")
    return parser.parse_args()


def load_test_samples_by_domain(splits_dir: Path):
    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found! Run prepare_data.py first.")
    df = pd.read_csv(test_csv)
    by_domain = {}
    for dom in df["domain"].unique():
        dom_df = df[df["domain"] == dom]
        samples = [
            NormalizedSample(
                id=row["id"],
                review_id=row["review_id"],
                text=str(row["text"]),
                aspect=str(row["aspect"]),
                sentiment=row["sentiment"],
                sentiment_id=int(row["sentiment_id"]),
                domain=row["domain"],
            )
            for _, row in dom_df.iterrows()
        ]
        by_domain[dom.lower()] = samples
    return by_domain


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    splits_dir = Path(args.splits_dir)
    domain_samples = load_test_samples_by_domain(splits_dir)

    # Determine checkpoint
    if args.checkpoint is None:
        # Default to best bert_lora run
        runs_dir = Path("runs/bert_lora")
        candidates = sorted(runs_dir.glob("*/checkpoints/best_model.pt"))
        if candidates:
            ckpt_path = candidates[-1]
            model_type = "bert_lora"
        else:
            raise ValueError("No checkpoint specified and no bert_lora checkpoint found.")
    else:
        ckpt_path = Path(args.checkpoint)
        model_type = args.model

    run_dir = ckpt_path.parent.parent
    logger.info(f"Evaluating checkpoint: {ckpt_path} (Model: {model_type})")

    # Load model architecture
    if model_type in ["bert", "bert_lora"]:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collate_fn = partial(collate_fn_bert, tokenizer=tokenizer, max_len=128)
        if model_type == "bert_lora":
            model = ABSABertLoRA(pretrained_model_name="bert-base-uncased", num_classes=3)
        else:
            model = ABSABert(pretrained_model_name="bert-base-uncased", num_classes=3)
    else:
        vocab_path = run_dir / "vocab.json"
        if vocab_path.exists():
            vocab = Vocabulary.load(vocab_path)
        else:
            raise FileNotFoundError(f"Vocab not found at {vocab_path}")
        collate_fn = partial(collate_fn_rnn, vocab=vocab, max_len=128)
        if model_type == "rnn":
            model = VanillaRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)
        else:
            model = BiLSTMWithAttention(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    results = []
    print("\n" + "=" * 75)
    print(f"DOMAIN-WISE EVALUATION RESULTS ({model_type.upper()})")
    print("=" * 75)

    for dom in args.domains:
        samples = domain_samples.get(dom.lower(), [])
        if not samples:
            logger.warning(f"No samples found for domain: {dom}")
            continue

        loader = DataLoader(ABSADataset(samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits = model(**batch_gpu)
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                all_labels.extend(batch_gpu["labels"].cpu().numpy())

        metrics = compute_classification_metrics(all_labels, all_preds)
        results.append(
            {
                "model": model_type,
                "checkpoint": ckpt_path.name,
                "domain": dom.upper(),
                "num_samples": len(samples),
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "negative_f1": round(metrics["per_class"]["negative"]["f1"], 4),
                "neutral_f1": round(metrics["per_class"]["neutral"]["f1"], 4),
                "positive_f1": round(metrics["per_class"]["positive"]["f1"], 4),
            }
        )
        print(
            f"[{dom.upper():<12}] N={len(samples):<5} | "
            f"Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {metrics['weighted_f1']:.4f} | "
            f"Neg-F1: {metrics['per_class']['negative']['f1']:.3f} | Neu-F1: {metrics['per_class']['neutral']['f1']:.3f} | Pos-F1: {metrics['per_class']['positive']['f1']:.3f}"
        )

    print("=" * 75 + "\n")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved cross-domain evaluation results to: {out_csv}")


if __name__ == "__main__":
    main()
