"""Masked Linear and Masked Attention modules.

Vendored from parkis2002/prunedAttentionGRU (PrunedAttentionGRU.py / MaskedAttention.py)
with relative imports so it works inside CSI-Bench without installing the upstream package.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_check = bias
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_check:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def prune(self, threshold, k):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        nz = np.count_nonzero(new_mask)
        if k <= nz / (self.in_features * self.out_features):
            self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
            return True
        return False


class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.query = MaskedLinear(hidden_dim, attention_dim)
        self.key = MaskedLinear(hidden_dim, attention_dim)
        self.value = MaskedLinear(hidden_dim, attention_dim)
        self.context_vector = MaskedLinear(attention_dim, 1, bias=False)

    def forward(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        attn_scores = torch.tanh(q + k)
        attn_scores = self.context_vector(attn_scores).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return (v * attn_weights.unsqueeze(-1)).sum(dim=1)

    def prune(self, threshold, k):
        weight_dev = self.query.weight.device
        for linear in [self.query, self.key, self.value, self.context_vector]:
            tensor = linear.weight.data.cpu().numpy()
            mask = linear.mask.data.cpu().numpy()
            new_mask = np.where(abs(tensor) < threshold, 0, mask)
            nz = np.count_nonzero(new_mask)
            if k <= nz / (linear.in_features * linear.out_features):
                linear.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                linear.mask.data = torch.from_numpy(new_mask).to(weight_dev)
            else:
                return False
        return True
