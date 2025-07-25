# models/LPE_TimeEncoder.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LPE_TimeEncoder(nn.Module):
    """
    A learnable time encoding module based on Learned Positional Embeddings.
    It takes absolute current and previous timestamps, calculates delta_t, discretizes it, and encodes it.
    Designed to replace modules.TimeEncoder.
    """
    def __init__(self, time_dim: int, num_time_bins: int = 1000, max_time_diff: float = 2.6e7):
        super().__init__()
        self.time_dim = time_dim
        self.num_time_bins = num_time_bins
        self.max_time_diff = max_time_diff

        self.lpe_embedding = nn.Embedding(num_embeddings=self.num_time_bins + 1, embedding_dim=self.time_dim)
        nn.init.uniform_(self.lpe_embedding.weight, -0.01, 0.01)

    def discretize_time_diffs(self, time_diffs: torch.Tensor) -> torch.Tensor:
        """
        Discretizes continuous time differences into integer bins.
        Time differences are clamped to [0, max_time_diff] and then mapped to [0, num_time_bins].
        """
        clamped_time_diffs = torch.clamp(time_diffs, min=0.0, max=self.max_time_diff)
        normalized_time_diffs = clamped_time_diffs / self.max_time_diff
        time_bins = (normalized_time_diffs * self.num_time_bins).long()
        time_bins = torch.clamp(time_bins, min=0, max=self.num_time_bins)
        return time_bins

    def forward(self, current_times: torch.Tensor,
                neighbor_times: torch.Tensor,
                auxiliary_features: Optional[torch.Tensor] = None # Not used
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes time embeddings using Learned Positional Embeddings on time differences.

        Args:
            current_times (torch.Tensor): Absolute timestamps of current events. Shape (batch_size, seq_len).
            neighbor_times (torch.Tensor): Absolute timestamps of historical neighbor events. Shape (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, dict]:
                time_embeddings (torch.Tensor): The computed time embeddings. Shape (batch_size, seq_len, time_dim).
                regularization_losses (dict): An empty dictionary.
        """
        # Calculate time differences: delta_t = current_times - neighbor_times
        delta_t = current_times - neighbor_times
        
        time_bins = self.discretize_time_diffs(delta_t)
        time_embeddings = self.lpe_embedding(time_bins)

        regularization_losses = {}
        return time_embeddings, regularization_losses