# models/SPE_TimeEncoder.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

from DyGMamba.models.FixedSinusoidal_TimeEncoder import FixedSinusoidalTimeEncoder


class SPE_TimeEncoder(nn.Module):
    """
    A fixed (non-learnable) time encoding module based on Sinusoidal Positional Embeddings.
    Designed to replace modules.TimeEncoder.
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim

        # Use the imported FixedSinusoidalTimeEncoder directly
        self.spe_time_encoder = FixedSinusoidalTimeEncoder(time_dim=self.time_dim, parameter_requires_grad=False)

    def forward(self, current_times: torch.Tensor,
                neighbor_times: torch.Tensor,
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        # Calculate time differences: delta_t = current_times - neighbor_times
        delta_t = current_times - neighbor_times
        
        # FixedSinusoidalTimeEncoder's forward already computes delta_t internally.
        # We pass delta_t as the "current_times" and a zero tensor as "neighbor_times"
        # to ensure it computes delta_t - 0, effectively.
        output_embeddings, _ = self.spe_time_encoder(delta_t, torch.zeros_like(delta_t))
        
        regularization_losses = {} # SPE has no regularization losses
        return output_embeddings, regularization_losses