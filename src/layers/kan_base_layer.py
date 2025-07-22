# kan_mamote/src/layers/kan_base_layer.py

import torch
import torch.nn as nn
from typing import Literal

from src.utils.config import KANMAMMOTEConfig
from .basis_functions import FourierBasis, GaussianKernelBasis, WaveletBasis, BaseBasisFunction

class KANLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 basis_type: Literal['fourier', 'rkhs_gaussian', 'wavelet'],
                 config: KANMAMMOTEConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_type = basis_type
        self.config = config

        self.alpha_weights = nn.Parameter(torch.randn(in_features, out_features) * 0.01) # Reduced init scale
        self.alpha_bias = nn.Parameter(torch.zeros(out_features))

        self.norm_after_linear = nn.LayerNorm(out_features) # ADDED LayerNorm for stability

        if basis_type == 'fourier':
            self.basis_function: BaseBasisFunction = FourierBasis(out_features, config)
        elif basis_type == 'rkhs_gaussian':
            self.basis_function: BaseBasisFunction = GaussianKernelBasis(out_features, config)
        elif basis_type == 'wavelet':
            self.basis_function: BaseBasisFunction = WaveletBasis(out_features, config)
        else:
            raise ValueError(f"Unsupported basis_type: {basis_type}. "
                             "Choose from 'fourier', 'rkhs_gaussian', 'wavelet'. "
                             "The 'spline' expert is handled directly by MatrixKANLayer in K_MOTE.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prime = torch.matmul(x, self.alpha_weights) + self.alpha_bias
        x_prime = self.norm_after_linear(x_prime) # APPLY LayerNorm
        return self.basis_function(x_prime)