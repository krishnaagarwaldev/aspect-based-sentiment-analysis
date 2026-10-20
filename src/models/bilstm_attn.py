"""
Bidirectional LSTM with Aspect-Conditioned Interactive Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # Sentence BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Aspect BiLSTM
        self.aspect_bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        bi_hidden = hidden_dim * 2

        # Attention projection layers
        self.w_h = nn.Linear(bi_hidden, bi_hidden, bias=False)
        self.w_a = nn.Linear(bi_hidden, bi_hidden, bias=False)
        self.v = nn.Linear(bi_hidden, 1, bias=False)

        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(bi_hidden * 2, bi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bi_hidden, num_classes),
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lens: torch.Tensor,
        aspect_ids: torch.Tensor,
        aspect_lens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Embeddings
        text_embed = self.dropout(self.embedding(text_ids))      # (B, T, E)
        aspect_embed = self.dropout(self.embedding(aspect_ids))  # (B, A, E)

        # 2. Text BiLSTM encoding -> H: (B, T, 2H)
        H, _ = self.bilstm(text_embed)

        # 3. Aspect BiLSTM encoding -> aspect_h: (B, A, 2H)
        aspect_h, _ = self.aspect_bilstm(aspect_embed)
        # Pool aspect hidden states over aspect length
        aspect_mask = (aspect_ids != 0).unsqueeze(-1).float()  # (B, A, 1)
        aspect_rep = (aspect_h * aspect_mask).sum(dim=1) / aspect_lens.unsqueeze(-1).clamp(min=1).float()  # (B, 2H)

        # 4. Aspect-conditioned Attention over sentence hidden states H
        # aspect_rep_expanded: (B, 1, 2H) -> (B, T, 2H)
        aspect_proj = self.w_a(aspect_rep).unsqueeze(1)  # (B, 1, 2H)
        text_proj = self.w_h(H)                          # (B, T, 2H)

        # score: (B, T, 1)
        scores = self.v(torch.tanh(text_proj + aspect_proj))

        # Mask padding tokens in sentence
        text_mask = (text_ids != 0).unsqueeze(-1)  # (B, T, 1)
        scores = scores.masked_fill(~text_mask, -1e4)
        alpha = F.softmax(scores, dim=1)           # (B, T, 1)

        # Weighted context vector r: (B, 2H)
        context = (alpha * H).sum(dim=1)

        # 5. Concatenate context vector and aspect representation
        features = torch.cat([context, aspect_rep], dim=1)  # (B, 4H)
        logits = self.fc(features)                         # (B, num_classes)
        return logits
