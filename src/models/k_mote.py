# kan_mamote/src/models/k_mote.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

from src.utils.config import KANMAMMOTEConfig
from src.layers.kan_base_layer import KANLayer
from src.models.moe_router import MoERouter
from src.models.kan.MatrixKANLayer import MatrixKANLayer

class K_MOTE(nn.Module):
    def __init__(self, config: KANMAMMOTEConfig):
        super().__init__()
        self.config = config
        self.num_experts = 4

        router_input_dim = 1
        if config.use_aux_features_router:
            router_input_dim += config.raw_event_feature_dim
        
        self.router = MoERouter(input_dim=router_input_dim, num_experts=self.num_experts, config=config)

        self.experts = nn.ModuleDict({
            "fourier": KANLayer(in_features=1, out_features=config.D_time, basis_type='fourier', config=config),
            "spline": MatrixKANLayer(
                in_dim=1,
                out_dim=config.D_time,
                num=config.spline_grid_size,
                k=config.spline_degree,
                noise_scale=getattr(config, 'kan_noise_scale', 0.1),
                scale_base_mu=getattr(config, 'kan_scale_base_mu', 0.0),
                scale_base_sigma=getattr(config, 'kan_scale_base_sigma', 1.0),
                grid_eps=getattr(config, 'kan_grid_eps', 0.02),
                grid_range=getattr(config, 'kan_grid_range', [-1, 1]),
                sp_trainable=getattr(config, 'kan_sp_trainable', True),
                sb_trainable=getattr(config, 'kan_sb_trainable', True),
                device=config.device
            ),
            "gaussian": KANLayer(in_features=1, out_features=config.D_time, basis_type='rkhs_gaussian', config=config),
            "wavelet": KANLayer(in_features=1, out_features=config.D_time, basis_type='wavelet', config=config),
        })
        self.expert_names = list(self.experts.keys())

        self.final_projection = nn.Identity()
        self.layer_norm = nn.LayerNorm(config.D_time)

    def forward(self,
                timestamp_input: torch.Tensor,
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = timestamp_input.shape[0]

        router_logits, raw_weights = self.router(timestamp_input, auxiliary_features)
        expert_weights_for_loss = raw_weights # For load balancing loss

        # IMPORTANT FIX: Simplify dispatch_weights for gradient flow when K_top == num_experts
        # If K_top is equal to num_experts (as in your config: K_top=4, num_experts=4),
        # all experts are always selected. Using topk and then re-softmax can be numerically redundant
        # and has issues with `scatter` and gradient flow for indices.
        # Instead, directly use the raw_weights (from softmax) as dispatch weights.
        if self.config.K_top == self.num_experts: # This condition should be true based on your config
            dispatch_weights = raw_weights
            expert_selection_mask = torch.ones_like(raw_weights, dtype=torch.bool) # All experts are "selected"
        else: # Original logic for true Top-K selection (if K_top < num_experts)
            topk_weights, topk_indices = torch.topk(raw_weights, self.config.K_top, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1) # Re-normalize top-k weights
            expert_selection_mask = torch.zeros_like(raw_weights, dtype=torch.bool).scatter_(1, topk_indices, True)
            dispatch_weights = torch.zeros_like(raw_weights).scatter(1, topk_indices, topk_weights)

        processed_expert_outputs = []
        for name in self.expert_names:
            expert_module = self.experts[name]
            output = expert_module(timestamp_input)
            if isinstance(expert_module, MatrixKANLayer):
                processed_expert_outputs.append(output[0])
            else:
                processed_expert_outputs.append(output)
        
        all_expert_outputs = torch.stack(processed_expert_outputs, dim=1)

        weighted_expert_outputs = all_expert_outputs * dispatch_weights.unsqueeze(-1)

        combined = weighted_expert_outputs.sum(dim=1)
        
        final_embedding = self.layer_norm(self.final_projection(combined))

        return final_embedding, expert_weights_for_loss, expert_selection_mask