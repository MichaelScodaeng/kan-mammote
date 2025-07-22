# kan_mamote/tests/test_kan_layers.py

import unittest
import torch
import torch.nn as nn

# Import the necessary modules from your project structure
from src.utils.config import KANMAMMOTEConfig
from src.layers.kan_base_layer import KANLayer
from src.layers.basis_functions import (
    FourierBasis, SplineBasis, GaussianKernelBasis, WaveletBasis
)

class TestKANLayers(unittest.TestCase):

    def setUp(self):
        """
        Set up common configurations and parameters for all tests.
        """
        self.config = KANMAMMOTEConfig()
        self.batch_size = 4
        self.in_features = 1 # For time input
        self.out_features_per_expert = self.config.D_time_per_expert
        self.device = self.config.device
        print(f"\n--- Running tests on device: {self.device} ---")

    def _test_kan_layer_properties(self, layer: KANLayer, input_data: torch.Tensor):
        """
        Helper method to test common properties of any KANLayer.
        - Output shape check
        - Differentiability check (gradient existence)
        """
        layer.to(self.device)
        input_data = input_data.to(self.device)

        # Ensure input requires gradients for backprop check
        input_data.requires_grad_(True)

        # Forward pass
        output = layer(input_data)

        # Assert output shape
        expected_shape = (self.batch_size, self.out_features_per_expert)
        self.assertEqual(output.shape, expected_shape, 
                         f"Output shape mismatch for {layer.basis_type} layer. Expected {expected_shape}, got {output.shape}")

        # Backward pass: Sum the output to get a scalar for backward()
        try:
            # Clear existing gradients
            layer.zero_grad()
            output.sum().backward()
        except RuntimeError as e:
            self.fail(f"Backward pass failed for {layer.basis_type} layer: {e}")

        # Assert gradients exist for learnable parameters
        found_learnable_params = False
        for name, param in layer.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, 
                                     f"Gradient is None for learnable parameter {name} in {layer.basis_type} layer.")
                found_learnable_params = True
        
        self.assertTrue(found_learnable_params, 
                        f"No learnable parameters found or tested for gradients in {layer.basis_type} layer.")
        print(f"  {layer.basis_type} Layer: Output shape OK. Gradients exist for learnable params.")


    def test_fourier_kan_layer(self):
        """Test KANLayer with FourierBasis."""
        print("Testing FourierBasis...")
        layer = KANLayer(self.in_features, self.out_features_per_expert, 'fourier', self.config)
        dummy_input = torch.randn(self.batch_size, self.in_features) # Time input
        self._test_kan_layer_properties(layer, dummy_input)
        
        # Specific check for FourierBasis parameters
        self.assertIsInstance(layer.basis_function, FourierBasis)
        self.assertTrue(hasattr(layer.basis_function, 'frequencies'))
        self.assertTrue(hasattr(layer.basis_function, 'amplitudes'))
        self.assertTrue(hasattr(layer.basis_function, 'phases'))


    def test_spline_kan_layer(self):
        """Test KANLayer with SplineBasis (MatrixKAN principles)."""
        print("Testing SplineBasis...")
        # Ensure MatrixKAN optimization is enabled in config for this test
        original_use_matrix_kan = self.config.use_matrix_kan_optimized_spline
        self.config.use_matrix_kan_optimized_spline = True # Explicitly enable for test

        layer = KANLayer(self.in_features, self.out_features_per_expert, 'spline', self.config)
        dummy_input = torch.randn(self.batch_size, self.in_features) # Time input
        self._test_kan_layer_properties(layer, dummy_input)
        
        # Specific check for SplineBasis parameters
        self.assertIsInstance(layer.basis_function, SplineBasis)
        self.assertTrue(hasattr(layer.basis_function, 'control_points'))
        self.assertTrue(hasattr(layer.basis_function, 'knots'))
        if layer.basis_function.use_matrix_kan_optimized_spline:
            self.assertTrue(hasattr(layer.basis_function, 'psi_k_matrix'))
        else:
            print("  SplineBasis: MatrixKAN optimization NOT active for test (likely fallback mode).")
        
        # Restore original config setting
        self.config.use_matrix_kan_optimized_spline = original_use_matrix_kan


    def test_gaussian_kernel_kan_layer(self):
        """Test KANLayer with GaussianKernelBasis."""
        print("Testing GaussianKernelBasis...")
        layer = KANLayer(self.in_features, self.out_features_per_expert, 'rkhs_gaussian', self.config)
        dummy_input = torch.randn(self.batch_size, self.in_features) # Time input
        self._test_kan_layer_properties(layer, dummy_input)
        
        # Specific check for GaussianKernelBasis parameters
        self.assertIsInstance(layer.basis_function, GaussianKernelBasis)
        self.assertTrue(hasattr(layer.basis_function, 'raw_weights'))
        self.assertTrue(hasattr(layer.basis_function, 'means'))
        self.assertTrue(hasattr(layer.basis_function, 'raw_stds'))


    def test_wavelet_kan_layer(self):
        """Test KANLayer with WaveletBasis."""
        print("Testing WaveletBasis...")
        layer = KANLayer(self.in_features, self.out_features_per_expert, 'wavelet', self.config)
        dummy_input = torch.randn(self.batch_size, self.in_features) # Time input
        self._test_kan_layer_properties(layer, dummy_input)
        
        # Specific check for WaveletBasis parameters
        self.assertIsInstance(layer.basis_function, WaveletBasis)
        self.assertTrue(hasattr(layer.basis_function, 'weights'))
        self.assertTrue(hasattr(layer.basis_function, 'raw_scales'))
        self.assertTrue(hasattr(layer.basis_function, 'translations'))


if __name__ == '__main__':
    # To run these tests:
    # 1. Navigate to the root of your project (kan_mamote_project/) in your terminal.
    # 2. Make sure your virtual environment is activated.
    # 3. Run: python -m unittest tests/test_kan_layers.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)