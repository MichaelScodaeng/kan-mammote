# models/NoTime_TimeEncoder.py

import torch
import torch.nn as nn
from typing import Optional, Tuple


class NoTime_TimeEncoder(nn.Module):
    """
    A dummy time encoding module that returns zeros, effectively representing "no time embedding".
    It takes absolute current and previous timestamps, but ignores them.
    Designed to replace modules.TimeEncoder.
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, current_times: torch.Tensor,
                neighbor_times: torch.Tensor, # Not used
                auxiliary_features: Optional[torch.Tensor] = None # Not used
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns a tensor of zeros as time embeddings.

        Args:
            current_times (torch.Tensor): Absolute timestamps of current events. Used for shape and device. Shape (batch_size, seq_len).
            neighbor_times (torch.Tensor): Absolute timestamps of historical neighbor events. Not used.

        Returns:
            Tuple[torch.Tensor, dict]:
                time_embeddings (torch.Tensor): A tensor of zeros with shape (batch_size, seq_len, time_dim).
                regularization_losses (dict): An empty dictionary.
        """
        batch_size, seq_len = current_times.shape
        time_embeddings = torch.zeros(batch_size, seq_len, self.time_dim,
                                      device=current_times.device, dtype=current_times.dtype)
        regularization_losses = {}
        return time_embeddings, regularization_losses