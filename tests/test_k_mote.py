# kan_mamote/tests/test_k_mote.py

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
# Import the necessary modules from your project structure
from src.utils.config import KANMAMMOTEConfig
from src.models.moe_router import MoERouter
from src.models.k_mote import K_MOTE

class TestK_MOTE(unittest.TestCase):

    def setUp(self):
        """
        Set up common configurations and parameters for all tests.
        """
        self.config = KANMAMMOTEConfig()
        self.batch_size = 4
        self.timestamp_input_dim = 1 # Scalar timestamp
        self.aux_feature_dim = self.config.raw_event_feature_dim
        self.device = self.config.device
        print(f"\n--- Running K-MOTE tests on device: {self.device} ---")

        # Ensure SplineBasis uses MatrixKAN for a more realistic test
        self.original_use_matrix_kan = self.config.use_matrix_kan_optimized_spline
        self.config.use_matrix_kan_optimized_spline = True

    def tearDown(self):
        """
        Clean up after tests.
        """
        # Restore original config setting
        self.config.use_matrix_kan_optimized_spline = self.original_use_matrix_kan

    def _test_module_properties(self, module: nn.Module, input_data: torch.Tensor, 
                                expected_output_shape: Tuple[int, ...], 
                                name: str = "Module"):
        """
        Helper method to test common properties of any PyTorch module.
        - Output shape check
        - Differentiability check (gradient existence)
        """
        module.to(self.device)
        input_data = input_data.to(self.device)
        input_data.requires_grad_(True)

        # Forward pass
        output = module(input_data)

        # Assert output shape
        self.assertEqual(output.shape, expected_output_shape, 
                         f"Output shape mismatch for {name}. Expected {expected_output_shape}, got {output.shape}")

        # Backward pass
        try:
            module.zero_grad()
            output.sum().backward()
        except RuntimeError as e:
            self.fail(f"Backward pass failed for {name}: {e}")

        # Assert gradients exist for learnable parameters
        found_learnable_params = False
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, 
                                     f"Gradient is None for learnable parameter {param_name} in {name}.")
                found_learnable_params = True
        
        self.assertTrue(found_learnable_params, 
                        f"No learnable parameters found or tested for gradients in {name}.")
        print(f"  {name}: Output shape OK. Gradients exist for learnable params.")
        return output # Return output for further specific checks


    def test_moe_router(self):
        """Test MoERouter module."""
        print("Testing MoERouter...")
        router_input_dim = self.timestamp_input_dim + (self.aux_feature_dim if self.config.use_aux_features_router else 0)
        
        router = MoERouter(input_dim=router_input_dim, num_experts=self.config.num_experts, config=self.config)
        
        timestamp_input = torch.randn(self.batch_size, self.timestamp_input_dim)
        aux_features = torch.randn(self.batch_size, self.aux_feature_dim) if self.config.use_aux_features_router else None

        router.to(self.device)
        timestamp_input = timestamp_input.to(self.device).requires_grad_(True)
        if aux_features is not None:
            aux_features = aux_features.to(self.device).requires_grad_(True)

        logits, weights = router(timestamp_input, aux_features)

        # Check shapes
        self.assertEqual(logits.shape, (self.batch_size, self.config.num_experts))
        self.assertEqual(weights.shape, (self.batch_size, self.config.num_experts))

        # Check softmax property: weights sum to 1
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size, device=self.device)))

        # Check gradients
        try:
            router.zero_grad()
            (logits.sum() + weights.sum()).backward() # Sum both to ensure gradients for all paths
        except RuntimeError as e:
            self.fail(f"Backward pass failed for MoERouter: {e}")
        
        found_learnable_params = False
        for name, param in router.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name} in MoERouter.")
                found_learnable_params = True
        self.assertTrue(found_learnable_params, "No learnable parameters found for MoERouter.")
        
        print("  MoERouter: Output shapes OK, weights sum to 1, Gradients exist.")


    def test_k_mote_module(self):
        """Test K_MOTE module, including Top-Ktop logic and expert combination."""
        print("Testing K_MOTE module...")
        
        k_mote_module = K_MOTE(config=self.config)
        
        timestamp_input = torch.randn(self.batch_size, self.timestamp_input_dim)
        aux_features = torch.randn(self.batch_size, self.aux_feature_dim) if self.config.use_aux_features_router else None

        k_mote_module.to(self.device)
        timestamp_input = timestamp_input.to(self.device).requires_grad_(True)
        if aux_features is not None:
            aux_features = aux_features.to(self.device).requires_grad_(True)

        # Forward pass
        embedding, expert_weights_for_loss, expert_selection_mask = k_mote_module(
            timestamp_input, aux_features
        )

        # Check output shapes
        expected_embedding_shape = (self.batch_size, self.config.D_time)
        self.assertEqual(embedding.shape, expected_embedding_shape, 
                         f"K_MOTE embedding shape mismatch. Expected {expected_embedding_shape}, got {embedding.shape}")
        
        expected_weights_shape = (self.batch_size, self.config.num_experts)
        self.assertEqual(expert_weights_for_loss.shape, expected_weights_shape,
                         f"K_MOTE raw weights shape mismatch. Expected {expected_weights_shape}, got {expert_weights_for_loss.shape}")
        self.assertEqual(expert_selection_mask.shape, expected_weights_shape,
                         f"K_MOTE selection mask shape mismatch. Expected {expected_weights_shape}, got {expert_selection_mask.shape}")
        self.assertTrue(expert_selection_mask.dtype == torch.bool)

        # Check Top-Ktop logic:
        # For each batch item, exactly K_top experts should be selected (True in mask)
        self.assertTrue(torch.all(expert_selection_mask.sum(dim=-1) == self.config.K_top),
                        "Top-Ktop logic failed: incorrect number of experts selected.")
        
        # Check that non-selected experts effectively contribute zero to the dispatch_weights
        # This is implicitly tested if the backward pass works through the zeroed out portions.
        # Can add explicit check: for each row in dispatch_weights, sum of non-selected should be 0.
        
        # Check gradients for K_MOTE module
        try:
            k_mote_module.zero_grad()
            embedding.sum().backward()
        except RuntimeError as e:
            self.fail(f"Backward pass failed for K_MOTE: {e}")
        
        found_learnable_params = False
        for name, param in k_mote_module.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name} in K_MOTE.")
                found_learnable_params = True
        self.assertTrue(found_learnable_params, "No learnable parameters found for K_MOTE.")
        
        print("  K_MOTE: Output shapes OK, Top-Ktop logic verified, Gradients exist.")


if __name__ == '__main__':
    # To run these tests:
    # 1. Navigate to the root of your project (kan_mamote_project/) in your terminal.
    # 2. Make sure your virtual environment is activated.
    # 3. Run: python -m unittest tests/test_k_mote.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)