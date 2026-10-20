"""
Vanilla RNN Baseline for Aspect-Based Sentiment Analysis.
"""

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        # Combine sentence hidden state + aspect mean embedding
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lens: torch.Tensor,
        aspect_ids: torch.Tensor,
        aspect_lens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Text encoding via RNN
        text_embed = self.dropout(self.embedding(text_ids))  # (B, T, E)
        rnn_out, _ = self.rnn(text_embed)  # (B, T, H)

        # Gather last non-padded hidden state
        batch_size = text_ids.size(0)
        idx = (text_lens - 1).clamp(min=0).view(batch_size, 1, 1).expand(-1, 1, rnn_out.size(2))
        text_rep = rnn_out.gather(1, idx).squeeze(1)  # (B, H)

        # 2. Aspect representation: average of aspect token embeddings
        aspect_embed = self.embedding(aspect_ids)  # (B, A, E)
        mask = (aspect_ids != 0).unsqueeze(-1).float()  # (B, A, 1)
        aspect_sum = (aspect_embed * mask).sum(dim=1)
        aspect_len = aspect_lens.unsqueeze(-1).clamp(min=1).float()
        aspect_rep = aspect_sum / aspect_len  # (B, E)

        # 3. Concatenate and classify
        combined = torch.cat([text_rep, aspect_rep], dim=1)  # (B, H + E)
        logits = self.fc(combined)
        return logits
