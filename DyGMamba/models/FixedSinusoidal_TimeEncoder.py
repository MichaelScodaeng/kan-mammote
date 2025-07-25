import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class FixedSinusoidalTimeEncoder(nn.Module):
    """
    Fixed sinusoidal time encoder (original modules.TimeEncoder, now in its own file).
    """
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super(FixedSinusoidalTimeEncoder, self).__init__()

        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, current_times: torch.Tensor,
                neighbor_times: torch.Tensor,
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        # Calculate time differences from absolute times
        delta_t = current_times - neighbor_times
        
        # Unsqueeze delta_t to (batch_size, seq_len, 1) for the linear layer
        timestamps_for_encoding = delta_t.unsqueeze(dim=2)

        output = torch.cos(self.w(timestamps_for_encoding))
        
        return output, {} # No regularization losses for this fixed encoder