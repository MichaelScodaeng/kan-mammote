# kan_mamote/src/models/continuous_mamba_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from src.utils.config import KANMAMMOTEConfig
from src.models.k_mote import K_MOTE
from src.layers.dynamic_mamba_ssm import DynamicMambaSSM
from src.layers.kan_base_layer import KANLayer
from src.models.kan.MatrixKANLayer import MatrixKANLayer

class FasterKAN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__()
        # Use RKHS Gaussian as an example, as per your config
        self.kan_transform = KANLayer(in_features=input_dim, out_features=output_dim,
                                      basis_type='rkhs_gaussian', # Or 'fourier', 'wavelet'
                                      config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan_transform(x)

class ContinuousMambaBlock(nn.Module):
    def __init__(self, d_model: int, config: KANMAMMOTEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.layer_idx = layer_idx 

        # K-MOTE is now part of this block, and its outputs will be used
        # in KANMAMMOTE.forward before calling ContinuousMambaBlock.forward_sequence
        self.k_mote = K_MOTE(config)
        self.faster_kan_transform = FasterKAN(input_dim=config.D_time, output_dim=config.D_time, config=config)
        self.input_proj = nn.Linear(d_model, d_model) 

        # Dynamic Mamba SSM layer
        mamba_kwargs = {
            "d_state": config.mamba_d_state,
            "d_conv": config.mamba_d_conv,
            "expand": config.mamba_expand,
            "headdim": config.mamba_headdim,
            "d_ssm": config.mamba_d_ssm,
            "ngroups": 1, # Default ngroups
            "A_init_range": (1, 16), # Default
            "D_has_hdim": False, # Default
            "rmsnorm": True, # Default
            "norm_before_gate": False, # Default
            "dt_min": config.mamba_dt_min,
            "dt_max": config.mamba_dt_max,
            "dt_init_floor": config.mamba_dt_init_floor,
            "dt_limit": (0.0, float("inf")), # Default
            "bias": config.mamba_bias,
            "conv_bias": config.mamba_conv_bias,
            "chunk_size": config.mamba_chunk_size,
            "use_mem_eff_path": config.mamba_use_mem_eff_path,
            "layer_idx": layer_idx,
            "device": config.device,
            "dtype": config.dtype,
        }
        self.mamba_ssm = DynamicMambaSSM(
            d_model=d_model,
            k_mote_delta_t_embedding_dim=config.D_time,
            **mamba_kwargs
        )
    
    # Removed the `step` method. If single-step inference is needed,
    # it would be a separate, explicit implementation or path, as it
    # conflicts with vectorized training.

    def forward_sequence(self,
                        hidden_states: torch.Tensor, # (batch_size, seq_len, d_model)
                        delta_t_embedding: torch.Tensor # (batch_size, seq_len, D_time)
    ) -> torch.Tensor:
        processed_hidden_states = self.input_proj(hidden_states)

        #print(f"DEBUG_CMB_FWD_SEQ: delta_t_embedding shape (input to proj): {delta_t_embedding.shape}") # ADD THIS
        
        # This is the critical line: it should project from D_time (64) to nheads (8)
        dt_modulation_for_mamba_sequence = self.mamba_ssm.dt_modulation_proj(delta_t_embedding)
        
        #print(f"DEBUG_CMB_FWD_SEQ: dt_modulation_for_mamba_sequence shape (output of proj): {dt_modulation_for_mamba_sequence.shape}") # ADD THIS

        output_sequence = self.mamba_ssm(
            u=processed_hidden_states,
            dt_modulation_sequence=dt_modulation_for_mamba_sequence # Pass the correctly projected tensor
        )
        return output_sequence