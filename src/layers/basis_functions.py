# kan_mamote/src/layers/basis_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Literal

# Import configuration and spline matrix utility
from src.utils.config import KANMAMMOTEConfig
from src.utils.spline_matrix_utils import compute_bspline_matrix

class BaseBasisFunction(nn.Module, ABC):
    """
    Abstract base class for all learnable basis functions in K-MOTE.
    Each basis function takes a (linearly transformed) time input and
    produces an embedding of `output_dim`.
    """
    def __init__(self, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__()
        self.output_dim = output_dim
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the basis function value for input x.
        Args:
            x: A tensor of shape (batch_size, self.output_dim).
               Each element x[b, d] is a linearly transformed timestamp
               that the basis function for output dimension 'd' operates on.
        Returns:
            A tensor of shape (batch_size, self.output_dim), representing
            the basis function's output.
        """
        pass

class FourierBasis(BaseBasisFunction):
    """
    Implements a learnable Fourier basis function.
    phi(x) = sum_{j=1}^{K'} (A_j * cos(omega_j * x + phi_j))
    where A_j, omega_j, phi_j are learnable parameters.
    """
    def __init__(self, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__(output_dim, config)
        self.num_harmonics = config.fourier_k_prime

        # Parameters are initialized as nn.Parameters if learnable, or registered as buffers if fixed.
        if config.fourier_learnable_params:
            # Frequencies (omega_j): Initialized to encourage learning from common patterns.
            # Shape: (output_dim, num_harmonics)
            self.frequencies = nn.Parameter(
                torch.randn(output_dim, self.num_harmonics) * 0.1 + # Small random noise
                (torch.arange(1, self.num_harmonics + 1).float() * torch.pi).unsqueeze(0).repeat(output_dim, 1) # Bias towards integer multiples of pi
            )
            # Amplitudes (A_j)
            self.amplitudes = nn.Parameter(torch.ones(output_dim, self.num_harmonics) * 0.5) # Initialized smaller
            # Phases (phi_j)
            self.phases = nn.Parameter(torch.zeros(output_dim, self.num_harmonics))
        else:
            # Fixed parameters for comparison/ablation
            self.register_buffer('frequencies', 
                                 (torch.arange(1, self.num_harmonics + 1).float() * torch.pi).unsqueeze(0).repeat(output_dim, 1))
            self.register_buffer('amplitudes', torch.ones(output_dim, self.num_harmonics) * 0.5)
            self.register_buffer('phases', torch.zeros(output_dim, self.num_harmonics))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # x_expanded: (batch_size, output_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Parameters expanded to (1, output_dim, num_harmonics) for broadcasting
        freq_expanded = self.frequencies.unsqueeze(0)
        amp_expanded = self.amplitudes.unsqueeze(0)
        phase_expanded = self.phases.unsqueeze(0)

        # Compute A_j * cos(omega_j * x + phi_j)
        # Result shape: (batch_size, output_dim, num_harmonics)
        term = amp_expanded * torch.cos(freq_expanded * x_expanded + phase_expanded)
        
        # Sum over harmonics to get final output: (batch_size, output_dim)
        return term.sum(dim=-1)


from src.utils.config import KANMAMMOTEConfig

# --- CORRECTED IMPORT: Import MatrixKANLayer from your MatrixKAN library ---
# Assuming MatrixKANLayer.py is correctly accessible (e.g., in the same package
# or directly in your PYTHONPATH).
try:
    from src.models.kan.MatrixKANLayer import MatrixKANLayer # This might need adjustment based on your exact file structure/import path
except ImportError:
    print("WARNING: Could not import MatrixKANLayer. Using a dummy placeholder.")
    # Dummy placeholder if the actual library isn't found
    class MatrixKANLayer(nn.Module):
        def __init__(self, in_dim=1, out_dim=1, num=5, k=3, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            # print(f"Using Dummy MatrixKANLayer: in={in_dim}, out={out_dim}, num={num}, k={k}")
        def forward(self, x):
            return self.linear(x)

# ... (Your existing BaseBasisFunction, FourierBasis, GaussianKernelBasis, WaveletBasis classes) ...

# Paste all your other basis function classes here
# (BaseBasisFunction, FourierBasis, GaussianKernelBasis, WaveletBasis)
# just before the corrected SplineBasis.

'''class SplineBasis(BaseBasisFunction):
    """
    Implements a learnable B-spline basis function by wrapping the MatrixKANLayer.
    """
    def __init__(self, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__(output_dim, config)
        
        # MatrixKANLayer takes:
        # in_dim: The number of input features this KAN layer processes.
        #         Here, it's `output_dim` because `x` is (batch_size, output_dim),
        #         and each of these `output_dim` entries is treated as an independent
        #         input to a separate KAN function.
        # out_dim: The number of output features. Since each input maps to one output
        #          in this basis function, it's also `output_dim`.
        # num: The number of grid intervals (G).
        # k: The piecewise polynomial order of splines.
        
        self.matrix_kan_layer = MatrixKANLayer(
            in_dim=output_dim,  # Each of the 'output_dim' time series components is an input to a KAN
            out_dim=output_dim, # Each KAN produces one output
            num=config.spline_grid_size, # Number of grid intervals (G)
            k=config.spline_degree,      # Spline order (k)
            # Other parameters from MatrixKANLayer's __init__ can be passed if needed:
            # noise_scale=config.kan_noise_scale,
            # scale_base_mu=config.kan_scale_base_mu,
            # scale_base_sigma=config.kan_scale_base_sigma,
            # base_fun=torch.nn.SiLU(), # Default SiLU from MatrixKAN, or configure
            grid_eps=config.kan_grid_eps, # For adaptive grid update
            grid_range=config.kan_grid_range, # Initial grid range
            sp_trainable=config.kan_sp_trainable, # Scale spline trainable
            sb_trainable=config.kan_sb_trainable, # Scale base trainable
            device=config.device # Ensure it's on the correct device
        )
        
        # Note: The MatrixKANLayer's forward method returns (y, preacts, postacts, postspline)
        # We only need 'y' (the main output) for our basis function.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
               Each element x[b, d] is a scalar time value that forms an input
               to one of the 'output_dim' KAN functions within the MatrixKANLayer.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # The MatrixKANLayer's forward method expects input x of shape (batch_size, in_dim).
        # Your `x` is already (batch_size, output_dim) which matches our `in_dim` setting.
        
        # It returns multiple values (y, preacts, postacts, postspline).
        # We only need the main output `y`.
        y, _, _, _ = self.matrix_kan_layer(x)

        return y'''


class GaussianKernelBasis(BaseBasisFunction):
    def __init__(self, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__(output_dim, config)
        self.num_mixture_components = config.rkhs_num_mixture_components
        
        self.raw_weights = nn.Parameter(torch.randn(output_dim, self.num_mixture_components) * 0.01) # Reduce init scale
        self.means = nn.Parameter(torch.randn(output_dim, self.num_mixture_components) * 1.0) # Reduce scale of means
        self.raw_stds = nn.Parameter(torch.randn(output_dim, self.num_mixture_components) * 0.01) # Reduce init scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(self.raw_weights.unsqueeze(0)) 
        means = self.means.unsqueeze(0)                     
        stds = F.softplus(self.raw_stds.unsqueeze(0)) + 1e-6 

        # Ensure stds don't become too small, e.g., enforce a minimum:
        # stds = torch.clamp(F.softplus(self.raw_stds.unsqueeze(0)), min=1e-3) # Example: min std of 0.001

        x_expanded = x.unsqueeze(-1)
        gaussian_terms = torch.exp(-0.5 * ((x_expanded - means) / stds)**2) # This is the main potential source of NaN if stds is tiny or (x-mean)/stds is huge
        
        return (weights * gaussian_terms).sum(dim=-1)

class WaveletBasis(BaseBasisFunction):
    def __init__(self, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__(output_dim, config)
        self.num_wavelets = config.wavelet_num_wavelets
        self.mother_wavelet_type = config.wavelet_mother_type

        if config.wavelet_learnable_params:
            self.weights = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.01) # Reduce init scale
            self.raw_scales = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.01) # Reduce init scale
            self.translations = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.01) # Reduce init scale
        # ... (rest same)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scales = F.softplus(self.raw_scales.unsqueeze(0)) + 1e-6
        # Similar to Gaussian, enforce minimum scale for stability
        # scales = torch.clamp(F.softplus(self.raw_scales.unsqueeze(0)), min=1e-3)
        
        x_expanded = x.unsqueeze(-1)
        transformed_x = (x_expanded - self.translations.unsqueeze(0)) / scales
        
        # Check for NaNs here: if scales are zero, transformed_x can be NaN/inf
        # if torch.isnan(transformed_x).any():
        #     print("DEBUG_NAN: Wavelet: NaN in transformed_x")
        # if torch.isinf(transformed_x).any():
        #     print("DEBUG_NAN: Wavelet: Inf in transformed_x")

        wavelet_terms = self._mother_wavelet_func(transformed_x) 
        
        return (self.weights.unsqueeze(0) * wavelet_terms).sum(dim=-1)


class WaveletBasis(BaseBasisFunction):
    """
    Implements a learnable Wavelet basis function.
    phi(x) = sum (W_j * mother_wavelet_func((x - T_j) / S_j))
    where W_j, T_j (translation), S_j (scale) are learnable parameters.
    """
    def __init__(self, output_dim: int, config: KANMAMMOTEConfig):
        super().__init__(output_dim, config)
        self.num_wavelets = config.wavelet_num_wavelets
        self.mother_wavelet_type = config.wavelet_mother_type

        # Learnable parameters for each wavelet component
        # Shape: (output_dim, num_wavelets)
        if config.wavelet_learnable_params:
            self.weights = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.1)
            self.raw_scales = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.1) # Will apply softplus/exp
            self.translations = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.1)
        else:
            self.register_buffer('weights', torch.ones(output_dim, self.num_wavelets) * 0.1)
            self.register_buffer('raw_scales', torch.ones(output_dim, self.num_wavelets) * 0.1)
            self.register_buffer('translations', torch.zeros(output_dim, self.num_wavelets))

        # Define the mother wavelet function
        if self.mother_wavelet_type == 'mexican_hat':
            # Mexican Hat wavelet (2nd derivative of Gaussian) is simple and differentiable
            # psi(t) = (1 - t^2) * exp(-0.5 * t^2)
            self._mother_wavelet_func = lambda t: (1 - t**2) * torch.exp(-0.5 * t**2)
        elif self.mother_wavelet_type == 'morlet':
            # Morlet wavelet: exp(-t^2/2) * cos(omega_0 * t)
            # A common choice for omega_0 is ~5-6 for good time-frequency localization
            self._mother_wavelet_func = lambda t: torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)
        else:
            raise NotImplementedError(f"Wavelet type {self.mother_wavelet_type} not implemented for direct computation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # Ensure scales are positive for numerical stability
        scales = F.softplus(self.raw_scales.unsqueeze(0)) + 1e-6 # (1, output_dim, num_wavelets)
        
        x_expanded = x.unsqueeze(-1) # (batch_size, output_dim, 1)

        # Apply transformation: (x - translation) / scale
        # (batch_size, output_dim, 1) - (1, output_dim, num_wavelets) -> (batch_size, output_dim, num_wavelets)
        transformed_x = (x_expanded - self.translations.unsqueeze(0)) / scales

        # Apply mother wavelet function
        wavelet_terms = self._mother_wavelet_func(transformed_x) # (batch_size, output_dim, num_wavelets)

        # Weighted sum over wavelets
        # (1, output_dim, num_wavelets) * (batch_size, output_dim, num_wavelets) -> sum over last dim
        return (self.weights.unsqueeze(0) * wavelet_terms).sum(dim=-1)