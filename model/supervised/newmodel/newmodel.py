"""Placeholder NewModel for the unified benchmarking workflow.

A simple Conv1d -> bidirectional GRU -> mean-pool -> linear head. Accepts the
bench's (B, 1, T, F) tensors as well as (B, T, F) and produces logits
(B, num_classes). Swap the body later; the wrapper interface is what the
unified pipeline depends on.
"""
import torch
import torch.nn as nn


class NewModel(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_size=232,
        win_len=500,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.win_len = win_len
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.proj = nn.Conv1d(feature_size, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def get_init_params(self):
        return {
            "num_classes": self.num_classes,
            "feature_size": self.feature_size,
            "win_len": self.win_len,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    def _to_btf(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.shape[-1] != self.feature_size and x.shape[-2] == self.feature_size:
            x = x.transpose(1, 2)
        return x

    def forward(self, x):
        x = self._to_btf(x)            # (B, T, F)
        h = self.proj(x.transpose(1, 2))  # (B, hidden, T)
        h = self.drop(self.act(h)).transpose(1, 2)  # (B, T, hidden)
        h, _ = self.rnn(h)             # (B, T, 2*hidden)
        h = h.mean(dim=1)
        return self.head(h)
