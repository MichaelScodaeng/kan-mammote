# kan_mamote/src/models/kan_mammote.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

# Import your custom modules
from src.utils.config import KANMAMMOTEConfig
from src.models.continuous_mamba_block import ContinuousMambaBlock
from src.models.regularization import KANMAMMOTE_RegularizationLosses

class KANMAMMOTE(nn.Module):
    """
    The full KAN-MAMMOTE model for continuous-time sequence modeling.
    Composed of multiple ContinuousMambaBlocks stacked sequentially.
    """
    def __init__(self, config: KANMAMMOTEConfig):
        super().__init__()
        self.config = config
        print(f"Initializing KAN-MAMMOTE with config: {self.config}")
        
        self.initial_feature_proj = nn.Linear(config.input_feature_dim, config.d_model)
        #print(f"Initial feature projection weights on device: {self.initial_feature_proj.weight.device}, dtype: {self.initial_feature_proj.weight.dtype}")
        
        self.mamba_blocks = nn.ModuleList([
            ContinuousMambaBlock(
                d_model=config.d_model,
                config=config,
                layer_idx=i 
            )
            for i in range(config.num_layers)
        ])

        self.prediction_head = nn.Linear(config.d_model, config.output_dim_for_task)
        self.regularization_handler = KANMAMMOTE_RegularizationLosses(config)

        # These are now informational/for reference only as states are managed internally by vectorized Mamba.
        d_inner_effective = self.config.mamba_expand * self.config.d_model
        d_ssm_effective = d_inner_effective if self.config.mamba_d_ssm is None else self.config.mamba_d_ssm 
        ngroups_effective = 1 
        self.conv_channels_for_state = d_ssm_effective + 2 * ngroups_effective * self.config.mamba_d_state
        self.nheads_for_state = d_ssm_effective // self.config.mamba_headdim
        #print(f"KANMAMMOTE init: Pre-calculated conv_channels_for_state={self.conv_channels_for_state}, nheads_for_state={self.nheads_for_state}")

    def forward(self,
                timestamps: torch.Tensor, # (batch_size, sequence_length)
                features: torch.Tensor,   # (batch_size, sequence_length, input_feature_dim)
                auxiliary_features: Optional[torch.Tensor] = None # (batch_size, sequence_length, aux_feature_dim)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the full KAN-MAMMOTE model, processing the entire sequence in a vectorized manner.
        """
        batch_size, seq_len = timestamps.shape
        assert features.shape[1] == seq_len, "Feature sequence length must match timestamp sequence length."

        timestamps = timestamps.to(self.config.device, dtype=self.config.dtype)
        features = features.to(self.config.device, dtype=self.config.dtype)
        if auxiliary_features is not None:
            auxiliary_features = auxiliary_features.to(self.config.device, dtype=self.config.dtype)
        
        processed_input_features = self.initial_feature_proj(features) # (B, L, d_model)

        # --- K-MOTE and FasterKAN for delta_t_embedding (vectorized) ---
        timestamps_flat_for_kmote = timestamps.view(-1, 1) # (B*L, 1)
        
        aux_features_flat_for_kmote = None # <--- FIX 1: Initialize this variable
        if auxiliary_features is not None:
            aux_features_flat_for_kmote = auxiliary_features.view(-1, auxiliary_features.shape[-1]) # (B*L, F_aux)

        abs_time_embedding_tk_flat, expert_weights_for_loss_flat, expert_selection_mask_flat = self.mamba_blocks[0].k_mote(
            timestamps_flat_for_kmote, aux_features_flat_for_kmote # Guaranteed to be defined
        )
        
        # For previous timestamps (tk_minus_1):
        timestamps_previous_shifted = torch.roll(timestamps, shifts=1, dims=1)
        timestamps_previous_shifted[:, 0] = timestamps[:, 0] # Set t_prev for first step to t_current
        timestamps_previous_flat_for_kmote = timestamps_previous_shifted.view(-1, 1)
        
        aux_features_previous_flat_for_kmote = None # <--- FIX 2: Initialize this variable (This is the one from the error traceback)
        if auxiliary_features is not None:
            aux_features_previous_shifted = torch.roll(auxiliary_features, shifts=1, dims=1)
            aux_features_previous_shifted[:, 0] = auxiliary_features[:, 0]
            aux_features_previous_flat_for_kmote = aux_features_previous_shifted.view(-1, auxiliary_features.shape[-1])

        abs_time_embedding_tk_minus_1_flat, _, _ = self.mamba_blocks[0].k_mote(
            timestamps_previous_flat_for_kmote, aux_features_previous_flat_for_kmote # Guaranteed to be defined
        )

        # FasterKAN for tk and tk_minus_1 embeddings (applied to flattened data)
        transformed_tk_flat = self.mamba_blocks[0].faster_kan_transform(abs_time_embedding_tk_flat)
        transformed_tk_minus_1_flat = self.mamba_blocks[0].faster_kan_transform(abs_time_embedding_tk_minus_1_flat)
        
        # Calculate delta_t_embedding (B*L, D_time)
        delta_t_embedding_flat = transformed_tk_flat - transformed_tk_minus_1_flat 
        
        # Reshape to (B, L, D_time) for passing to `ContinuousMambaBlock.forward_sequence`
        delta_t_embedding = delta_t_embedding_flat.view(batch_size, seq_len, self.config.D_time)

        # --- Pass through Mamba blocks (vectorized) ---
        current_hidden_states = processed_input_features # (B, L, d_model)

        for l_idx, block in enumerate(self.mamba_blocks):
            # Pass hidden states and the pre-computed delta_t_embedding sequence
            current_hidden_states = block.forward_sequence(
                hidden_states=current_hidden_states, # (B, L, d_model)
                delta_t_embedding=delta_t_embedding  # (B, L, D_time)
            )
            # current_hidden_states is now the output of this block (B, L, d_model)

        final_sequence_embedding = current_hidden_states # Output from the last Mamba block (B, L, d_model)
        model_output = self.prediction_head(final_sequence_embedding) # (B, L, output_dim_for_task)

        # --- Regularization Losses ---
        # expert_weights_for_loss_flat is already (B*L, num_experts)
        load_balance_loss = self.regularization_handler.compute_load_balance_loss(
            expert_weights_for_loss_flat
        )

        # Sobolev and Total Variation losses are stubs and return 0.0 as requested
        sobolev_l2_loss = self.regularization_handler.compute_sobolev_l2_loss(self)
        total_variation_loss = self.regularization_handler.compute_total_variation_loss(self)

        regularization_losses = {
            "load_balance_loss": load_balance_loss,
            "sobolev_l2_loss": sobolev_l2_loss,
            "total_variation_loss": total_variation_loss
        }

        return model_output, regularization_losses