"""
PyTorch Dataset and Collator implementations for RNN, BiLSTM, and BERT/BERT-LoRA.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .normalizer import NormalizedSample


def simple_tokenize(text: str) -> List[str]:
    """Tokenizes text using regex word patterns."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)


class Vocabulary:
    """Vocabulary mapping for word-based recurrent models (RNN, BiLSTM)."""

    def __init__(self, pad_token: str = "<pad>", unk_token: str = "<unk>", min_freq: int = 1):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            tokens = simple_tokenize(text)
            counter.update(tokens)

        for word, count in counter.items():
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str) -> List[int]:
        tokens = simple_tokenize(text)
        unk_idx = self.word2idx[self.unk_token]
        return [self.word2idx.get(w, unk_idx) for w in tokens]

    def save(self, file_path):
        import json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "min_freq": self.min_freq,
                "word2idx": self.word2idx,
            }, f, indent=2)

    @classmethod
    def load(cls, file_path):
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(pad_token=data["pad_token"], unk_token=data["unk_token"], min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)


class ABSADataset(Dataset):
    """General ABSA Dataset wrapping normalized samples."""

    def __init__(self, samples: List[NormalizedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NormalizedSample:
        return self.samples[idx]


def collate_fn_rnn(batch: List[NormalizedSample], vocab: Vocabulary, max_len: int = 128):
    """
    Collator for Vanilla RNN and BiLSTM-Attention.
    Returns:
        text_ids: (batch_size, max_seq_len)
        text_lens: (batch_size,)
        aspect_ids: (batch_size, max_aspect_len)
        aspect_lens: (batch_size,)
        labels: (batch_size,)
    """
    text_encoded = [vocab.encode(s.text)[:max_len] for s in batch]
    aspect_encoded = [vocab.encode(s.aspect)[:32] for s in batch]

    text_lens = [max(len(t), 1) for t in text_encoded]
    aspect_lens = [max(len(a), 1) for a in aspect_encoded]

    batch_max_text_len = max(text_lens)
    batch_max_aspect_len = max(aspect_lens)

    padded_texts = []
    for t in text_encoded:
        padded = t + [0] * (batch_max_text_len - len(t)) if len(t) > 0 else [0] * batch_max_text_len
        padded_texts.append(padded)

    padded_aspects = []
    for a in aspect_encoded:
        padded = a + [0] * (batch_max_aspect_len - len(a)) if len(a) > 0 else [0] * batch_max_aspect_len
        padded_aspects.append(padded)

    labels = [s.sentiment_id for s in batch]

    return {
        "text_ids": torch.tensor(padded_texts, dtype=torch.long),
        "text_lens": torch.tensor(text_lens, dtype=torch.long),
        "aspect_ids": torch.tensor(padded_aspects, dtype=torch.long),
        "aspect_lens": torch.tensor(aspect_lens, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "domains": [s.domain for s in batch],
    }


def collate_fn_bert(batch: List[NormalizedSample], tokenizer, max_len: int = 128):
    """
    Collator for BERT and BERT-LoRA.
    Encodes: [CLS] text [SEP] aspect [SEP]
    Returns input_ids, attention_mask, token_type_ids, labels
    """
    texts = [s.text for s in batch]
    aspects = [s.aspect for s in batch]
    labels = [s.sentiment_id for s in batch]

    encoding = tokenizer(
        text=texts,
        text_pair=aspects,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    encoding["labels"] = torch.tensor(labels, dtype=torch.long)
    encoding["domains"] = [s.domain for s in batch]
    return encoding
