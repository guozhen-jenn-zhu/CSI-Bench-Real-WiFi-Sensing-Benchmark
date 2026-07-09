"""CustomGRU using MaskedLinear gates. Vendored from parkis2002/prunedAttentionGRU/prunedGRU.py."""
import numpy as np
import torch
import torch.nn as nn

from .masked_attention import MaskedLinear


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, mask=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.update_gate = MaskedLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.reset_gate = MaskedLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.new_memory = MaskedLinear(input_size + hidden_size, hidden_size, bias=bias)

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        output = []
        for t in range(seq_len):
            xt = x[t]
            combined = torch.cat((xt, hx), dim=1)
            z_t = torch.sigmoid(self.update_gate(combined))
            r_t = torch.sigmoid(self.reset_gate(combined))
            combined_reset = torch.cat((xt, r_t * hx), dim=1)
            n_t = torch.tanh(self.new_memory(combined_reset))
            hx = (1 - z_t) * n_t + z_t * hx
            output.append(hx)
        output = torch.stack(output, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, hx.unsqueeze(0)

    def prune(self, threshold, k):
        weight_dev = self.update_gate.weight.device
        for linear in [self.update_gate, self.reset_gate, self.new_memory]:
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
