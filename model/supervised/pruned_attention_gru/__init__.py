"""CSI-Bench wrapper around prunedAttentionGRU.

The bench's BenchmarkCSIDataset emits inputs of shape (B, 1, T, F) (channel-first
after permuting (T, F, 1) -> (1, T, F)). prunedAttentionGRU expects (B, T, F),
so the wrapper squeezes the channel axis. CrossEntropyLoss with class indices
is used by TaskTrainer, matching the GRU's logits output.
"""
import torch
import torch.nn as nn

from .masked_attention import MaskedAttention, MaskedLinear
from .pruned_attention_gru import PruningModule, prunedAttentionGRU
from .pruned_gru import CustomGRU


class PrunedAttentionGRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_size=232,
        win_len=500,
        hidden_dim=128,
        attention_dim=32,
        input_layout="BCTF",
    ):
        super().__init__()
        self.feature_size = feature_size
        self.win_len = win_len
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.input_layout = input_layout
        self.net = prunedAttentionGRU(
            input_dim=feature_size,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            output_dim=num_classes,
        )

    def get_init_params(self):
        return {
            "num_classes": self.num_classes,
            "feature_size": self.feature_size,
            "win_len": self.win_len,
            "hidden_dim": self.hidden_dim,
            "attention_dim": self.attention_dim,
            "input_layout": self.input_layout,
        }

    def _to_btf(self, x):
        if x.dim() == 4:
            # (B, 1, T, F) from BenchmarkCSIDataset
            x = x.squeeze(1)
        elif x.dim() == 3 and self.input_layout == "BFT":
            x = x.transpose(1, 2)
        if x.shape[-1] != self.feature_size and x.shape[-2] == self.feature_size:
            x = x.transpose(1, 2)
        return x

    def forward(self, x):
        x = self._to_btf(x)
        return self.net(x)

    def apply_pruning(self, method="std", s=0.5, k=0.2, connectivity=0.2):
        if method == "std":
            self.net.prune_by_std(s=s, k=k)
        elif method == "random":
            self.net.prune_by_random(connectivity=connectivity)
        else:
            raise ValueError(f"Unknown pruning method: {method}")

    def sparsity_report(self):
        nonzero = total = 0
        for name, p in self.net.named_parameters():
            if "mask" in name:
                continue
            t = p.data.detach().cpu().numpy()
            nonzero += int((t != 0).sum())
            total += int(t.size)
        return {
            "total_params": total,
            "nonzero_params": nonzero,
            "zero_fraction": 1.0 - nonzero / max(total, 1),
        }


__all__ = [
    "PrunedAttentionGRUClassifier",
    "prunedAttentionGRU",
    "PruningModule",
    "MaskedAttention",
    "MaskedLinear",
    "CustomGRU",
]
