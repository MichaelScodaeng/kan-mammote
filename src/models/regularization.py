# kan_mamote/src/models/regularization.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import KANMAMMOTEConfig
from src.layers.kan_base_layer import KANLayer
# You will need to import MatrixKANLayer if you want to apply regularization to it
# from MatrixKANLayer import MatrixKANLayer 


class KANMAMMOTE_RegularizationLosses(nn.Module):
    """
    Module to compute regularization losses for KAN-MAMMOTE.
    """
    def __init__(self, config: KANMAMMOTEConfig):
        super().__init__()
        self.config = config

    def compute_sobolev_l2_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Computes the Sobolev L2 regularization loss for all relevant KANLayers in the model.
        This encourages smoothness by penalizing the L2 norm of derivatives.
        
        NOTE: This is a conceptual stub. Actual implementation requires access to
              derivative computation or spline coefficients from the KAN libraries.
        """
        loss = torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)
        # Iterate through all modules in the model
        # You'd typically collect all KANLayer instances and compute/approximate their derivative norms.
        for name, module in model.named_modules():
            if isinstance(module, KANLayer):
                # How to get derivatives depends on kan_base_layer.KANLayer's basis_function.
                # Example for FourierBasis (conceptual):
                # if module.basis_type == 'fourier':
                #     # You'd need to compute derivatives of A_j * cos(omega_j * x + phi_j)
                #     # with respect to x. Then sum their squares.
                #     # Or penalize the magnitude of frequencies directly.
                pass
            # If you want to regularize MatrixKANLayer directly:
            # if isinstance(module, MatrixKANLayer):
            #    # Check MatrixKAN library for how to compute derivatives or regularization terms
            #    # Example: loss += module.get_sobolev_loss() or similar API
            pass
        return loss

    def compute_total_variation_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Computes the Total Variation (TV) regularization loss.
        This encourages sparsity in derivatives (piecewise constant), preserving sharp changes.
        
        NOTE: This is a conceptual stub. Actual implementation requires access to
              derivative computation from the KAN libraries.
        """
        loss = torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)
        # Similar to Sobolev L2, iterate through KANLayers and compute/approximate TV of their functions.
        return loss

    def compute_load_balance_loss(self, expert_weights_for_loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the Load Balance Loss for the K-MOTE router.
        This encourages uniform usage of experts by penalizing concentration
        of expert assignments to a few experts.

        Args:
            expert_weights_for_loss: (batch_size, num_experts) raw softmax weights from MoERouter.
                                     These are the 'P' in KL(P || Q) where Q is uniform.
        """
        if not self.config.use_load_balancing:
            return torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)

        if expert_weights_for_loss.numel() == 0: # Handle empty tensor case
            return torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)

        num_experts = expert_weights_for_loss.shape[-1]
        
        # Average probability of selecting each expert across the batch
        # P_e = E_x [p(expert=e | x)]
        mean_expert_prob = expert_weights_for_loss.mean(dim=0) # (num_experts,)

        # Target uniform distribution
        target_distribution = torch.ones_like(mean_expert_prob) / num_experts

        # KL-divergence (KL(P || Q) = sum(P_i * log(P_i / Q_i)))
        # Here P is mean_expert_prob, Q is target_distribution
        # F.kl_div expects log-probabilities for the first argument, so we take log() of mean_expert_prob.
        # reduction='batchmean' means sum over dimensions, then divide by batch_size implicitly.
        # But here, mean_expert_prob is already an average, so we want sum over experts.
        # Alternatively, sum(P * (log P - log Q))
        
        # Ensure log(0) doesn't occur
        mean_expert_prob_safe = mean_expert_prob.clamp(min=1e-9)
        target_distribution_safe = target_distribution.clamp(min=1e-9)

        # Sum over experts for the KL-divergence
        load_balance_loss = (mean_expert_prob_safe * (mean_expert_prob_safe.log() - target_distribution_safe.log())).sum()
        
        # Apply the balance coefficient
        return load_balance_loss * self.config.balance_coefficient