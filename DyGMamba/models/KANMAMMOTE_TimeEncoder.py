# models/KANMAMMOTE_TimeEncoder.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

# Import your K-MOTE and FasterKAN modules
from src.utils.config import KANMAMMOTEConfig
from src.models.k_mote import K_MOTE
from src.layers.kan_base_layer import KANLayer # FasterKAN uses KANLayer
from src.models.regularization import KANMAMMOTE_RegularizationLosses # For K-MOTE's regularization loss

# Re-defining FasterKAN locally for clarity, as it's part of your time encoding method.
class FasterKAN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__()
        self.kan_transform = KANLayer(in_features=input_dim, out_features=output_dim,
                                      basis_type='rkhs_gaussian', # Default to RKHS Gaussian for FasterKAN as per your code
                                      config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan_transform(x)


class KANMAMMOTE_TimeEncoder(nn.Module):
    """
    A learnable time encoding module based on KAN-MAMMOTE's K-MOTE and FasterKAN.
    It takes absolute current and previous timestamps to calculate the delta_t embedding.
    Designed to replace the time encoding component within dynamic graph models.
    """
    def __init__(self, time_dim: int, kan_mammote_config: KANMAMMOTEConfig): # Renamed config to kan_mammote_config for clarity
        super().__init__()
        self.time_dim = time_dim # This will be the output dimension of this encoder
        self.kan_mammote_config = kan_mammote_config # Store the KAN-MAMMOTE specific config

        # The core K-MOTE module for absolute time embedding
        self.k_mote = K_MOTE(config=self.kan_mammote_config)

        # FasterKAN for transforming absolute time embeddings before differencing
        self.faster_kan_transform = FasterKAN(input_dim=self.kan_mammote_config.D_time,
                                              output_dim=self.kan_mammote_config.D_time,
                                              config=self.kan_mammote_config)

        # A linear projection to match the desired `time_dim` output if kan_mammote_config.D_time != time_dim
        self.output_projection = nn.Identity() # Default to identity
        if self.kan_mammote_config.D_time != self.time_dim:
            self.output_projection = nn.Linear(self.kan_mammote_config.D_time, self.time_dim)

        # Regularization handler for load balancing loss from K-MOTE
        self.regularization_handler = KANMAMMOTE_RegularizationLosses(self.kan_mammote_config)


    def forward(self, current_times: torch.Tensor,
                neighbor_times: torch.Tensor,
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes time embeddings based on absolute current and neighbor timestamps.

        Args:
            current_times (torch.Tensor): Absolute timestamps of the current events. Shape (batch_size, seq_len).
            neighbor_times (torch.Tensor): Absolute timestamps of the historical neighbor events. Shape (batch_size, seq_len).
                                           Note: Padded values in neighbor_times (where neighbor_id is 0) should be handled.
            auxiliary_features (Optional[torch.Tensor]): Auxiliary features for K-MOTE router. Shape (batch_size, seq_len, F_aux).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                time_embeddings (torch.Tensor): The computed time embeddings. Shape (batch_size, seq_len, time_dim).
                regularization_losses (Dict[str, torch.Tensor]): Dictionary containing regularization losses (e.g., load_balance_loss).
        """
        batch_size, seq_len = current_times.shape
        
        # Flatten current_times and neighbor_times for K-MOTE input
        current_times_flat = current_times.view(-1, 1) # (B*L, 1)
        neighbor_times_flat = neighbor_times.view(-1, 1) # (B*L, 1)

        aux_features_flat = None
        if auxiliary_features is not None:
            aux_features_flat = auxiliary_features.reshape(-1, auxiliary_features.shape[-1])

        # K-MOTE for tk (current_times)
        abs_time_embedding_tk_flat, expert_weights_for_loss_flat, _ = self.k_mote(
            current_times_flat, aux_features_flat
        )

        # K-MOTE for tk-1 (neighbor_times)
        abs_time_embedding_tk_minus_1_flat, _, _ = self.k_mote(
            neighbor_times_flat, aux_features_flat # Pass aux_features if applicable for neighbors too
        )

        # FasterKAN transformation
        transformed_tk_flat = self.faster_kan_transform(abs_time_embedding_tk_flat)
        transformed_tk_minus_1_flat = self.faster_kan_transform(abs_time_embedding_tk_minus_1_flat)

        # Calculate delta_t_embedding
        delta_t_embedding_flat = transformed_tk_flat - transformed_tk_minus_1_flat # (B*L, D_time)

        # Reshape back to (B, L, D_time)
        delta_t_embedding = delta_t_embedding_flat.view(batch_size, seq_len, self.kan_mammote_config.D_time)

        # Apply output projection if D_time != time_dim
        time_embeddings = self.output_projection(delta_t_embedding) # (batch_size, seq_len, time_dim)

        # Regularization Losses
        regularization_losses = {
            "load_balance_loss": self.regularization_handler.compute_load_balance_loss(expert_weights_for_loss_flat),
            "sobolev_l2_loss": self.regularization_handler.compute_sobolev_l2_loss(self.k_mote),
            "total_variation_loss": self.regularization_handler.compute_total_variation_loss(self.k_mote)
        }

        return time_embeddings, regularization_losses