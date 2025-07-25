# models/LeTE_TimeEncoder.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.LETE.LeTE import CombinedLeTE # Assuming LeTE.py is in the same directory


class LeTE_TimeEncoder(nn.Module):
    """
    A learnable time encoding module based on CombinedLeTE.
    It takes absolute current and previous timestamps, calculates delta_t, and encodes it.
    Designed to replace modules.TimeEncoder.
    """
    def __init__(self, time_dim: int, p: float = 0.5, layer_norm: bool = True, scale: bool = True, parameter_requires_grad: bool = True):
        super().__init__()
        self.time_dim = time_dim

        self.lete_time_encoder = CombinedLeTE(dim=self.time_dim, p=p,
                                              layer_norm=layer_norm, scale=scale,
                                              parameter_requires_grad=parameter_requires_grad)

    def forward(self, current_times: torch.Tensor,
                neighbor_times: torch.Tensor,
                auxiliary_features: Optional[torch.Tensor] = None # Not used
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes time embeddings using CombinedLeTE on time differences.

        Args:
            current_times (torch.Tensor): Absolute timestamps of current events. Shape (batch_size, seq_len).
            neighbor_times (torch.Tensor): Absolute timestamps of historical neighbor events. Shape (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, dict]:
                time_embeddings (torch.Tensor): The computed time embeddings. Shape (batch_size, seq_len, time_dim).
                regularization_losses (dict): An empty dictionary.
        """
        # Calculate time differences: delta_t = current_times - neighbor_times
        # This aligns with how models typically pass time info to modules.TimeEncoder.
        delta_t = current_times - neighbor_times
        
        time_embeddings = self.lete_time_encoder(delta_t)

        regularization_losses = {}
        return time_embeddings, regularization_losses