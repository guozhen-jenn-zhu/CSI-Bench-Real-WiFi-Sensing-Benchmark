"""prunedAttentionGRU model + PruningModule. Vendored from parkis2002/prunedAttentionGRU."""
import random

import numpy as np
import torch
import torch.nn as nn

from .masked_attention import MaskedAttention, MaskedLinear
from .pruned_gru import CustomGRU


class PruningModule(nn.Module):
    def prune_by_std(self, s, k):
        for name, module in self.named_modules():
            if isinstance(module, (MaskedLinear, CustomGRU, MaskedAttention)):
                self._prune_weights(module, s, k, name)

    def _prune_weights(self, module, s, k, name):
        for attr in ["weight_ih", "weight_hh", "weight"]:
            if hasattr(module, attr):
                weight = getattr(module, attr)
                threshold = float(np.std(weight.data.abs().cpu().numpy())) * s
                while not module.prune(threshold, k):
                    threshold *= 0.99
                    if threshold < 1e-12:
                        break

    def prune_by_random(self, connectivity):
        for name, module in self.named_modules():
            if isinstance(module, (MaskedLinear, CustomGRU, MaskedAttention)):
                self._random_prune_weights(module, connectivity, name)

    def _random_prune_weights(self, module, connectivity, name):
        for attr in ["weight_ih", "weight_hh", "weight"]:
            if hasattr(module, attr):
                weight = getattr(module, attr)
                row, column = weight.shape[0], weight.shape[1]
                weight_mask = torch.tensor(self._make_mask((row, column), connectivity)).float()
                weight_data = nn.init.orthogonal_(weight.data) * weight_mask.to(weight.device)
                weight.data = weight_data

    @staticmethod
    def _make_mask(shape, connection):
        random.seed(1)
        s = np.random.uniform(size=shape)
        s_flat = np.sort(s.flatten())
        threshold = s_flat[int(shape[0] * shape[1] * (1 - connection))]
        return (s >= threshold).astype(np.float32)


class prunedAttentionGRU(PruningModule):
    """Pruned attention GRU classifier. Input: (B, T, F). Output: logits (B, num_classes)."""

    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = CustomGRU(input_dim, hidden_dim, bias=True, batch_first=True)
        self.attention = MaskedAttention(hidden_dim, attention_dim)
        self.fc = MaskedLinear(attention_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        ctx = self.attention(gru_out)
        return self.fc(ctx)
