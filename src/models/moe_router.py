# kan_mamote/src/models/moe_router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.utils.config import KANMAMMOTEConfig

class MoERouter(nn.Module):
    def __init__(self, input_dim: int, num_experts: int = 4, config: Optional[KANMAMMOTEConfig] = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.config = config
        
        self.router_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        
        self.noise_scale = 1e-2 if config is None else getattr(config, 'router_noise_scale', 1e-2)
        self.use_load_balancing = True if config is None else getattr(config, 'use_load_balancing', True)
        self.balance_coefficient = 0.01 if config is None else getattr(config, 'balance_coefficient', 0.01)
        
    def forward(self, 
                timestamp_input: torch.Tensor, 
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = timestamp_input.shape[0]
        
        if timestamp_input.dim() == 1:
            timestamp_input = timestamp_input.unsqueeze(1)
        elif timestamp_input.dim() > 2:
            timestamp_input = timestamp_input.squeeze()
            if timestamp_input.dim() == 1:
                timestamp_input = timestamp_input.unsqueeze(1)
        
        router_input = timestamp_input
        
        # Combine with auxiliary features if configured to use them
        if self.config and self.config.use_aux_features_router:
            aux_dim = getattr(self.config, 'raw_event_feature_dim', 0)
            
            # If auxiliary features are expected but not provided for this call, create zero-padded features
            if auxiliary_features is None and aux_dim > 0:
                auxiliary_features_to_use = torch.zeros(batch_size, aux_dim,
                                                        device=timestamp_input.device,
                                                        dtype=timestamp_input.dtype)
            else: # Use provided aux features, or None if aux_dim is 0
                auxiliary_features_to_use = auxiliary_features
            
            if auxiliary_features_to_use is not None and auxiliary_features_to_use.shape[-1] > 0:
                if auxiliary_features_to_use.dim() == 1:
                    auxiliary_features_to_use = auxiliary_features_to_use.unsqueeze(1)
                elif auxiliary_features_to_use.dim() > 2:
                    auxiliary_features_to_use = auxiliary_features_to_use.view(batch_size, -1)
                
                router_input = torch.cat([timestamp_input, auxiliary_features_to_use], dim=1)
        
        # Assert that the resulting router_input dimension matches the dimension
        # the router_network was initialized with. This catches any remaining mismatches.
        if router_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"MoERouter: Router input dimension mismatch. "
                f"Expected {self.input_dim}, got {router_input.shape[-1]}."
                f"Timestamp input shape: {timestamp_input.shape}, "
                f"Auxiliary features shape: {auxiliary_features.shape if auxiliary_features is not None else 'None'}."
                f"Resulting router_input shape: {router_input.shape}."
            )

        logits = self.router_network(router_input)
        
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(logits) * self.noise_scale
            logits = logits + noise
        
        weights = F.softmax(logits, dim=-1)
        
        if self.training and self.use_load_balancing:
            mean_expert_activation = weights.mean(dim=0)
            target_distribution = torch.ones_like(mean_expert_activation) / self.num_experts
            load_balance_loss = F.kl_div(
                mean_expert_activation.log(), 
                target_distribution, 
                reduction='batchmean'
            ) * self.balance_coefficient
            self.load_balance_loss = load_balance_loss
            weights = weights + 0 * load_balance_loss
        
        return logits, weights
    
    def get_load_balance_loss(self) -> torch.Tensor:
        if hasattr(self, 'load_balance_loss'):
            return self.load_balance_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)