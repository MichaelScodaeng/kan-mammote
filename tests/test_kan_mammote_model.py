# kan_mamote/tests/test_kan_mamote_model.py

import unittest
import torch
import torch.nn as nn

# Import the necessary modules from your project structure
from src.utils.config import KANMAMMOTEConfig
from src.models.kan_mammote import KAN_MAMOTE_Model

class TestKANMAMMOTEModel(unittest.TestCase):

    def setUp(self):
        """
        Set up common configurations and parameters for the full model test.
        """
        self.config = KANMAMMOTEConfig()
        self.batch_size = 2 # Use a small batch size for testing
        self.seq_len = 5    # Test with a short sequence
        self.timestamp_dim = 1 # Scalar timestamp
        self.raw_event_feature_dim = self.config.raw_event_feature_dim
        self.device = self.config.device
        
        # Ensure MatrixKAN optimization is enabled for SplineBasis within K-MOTE
        self.original_use_matrix_kan = self.config.use_matrix_kan_optimized_spline
        self.config.use_matrix_kan_optimized_spline = True
        
        print(f"\n--- Running KAN_MAMOTE_Model test on device: {self.device} ---")
        print(f"  Batch Size: {self.batch_size}, Sequence Length: {self.seq_len}")
        print(f"  D_time: {self.config.D_time}, Hidden Dim Mamba: {self.config.hidden_dim_mamba}")

    def tearDown(self):
        """
        Clean up after tests.
        """
        # Restore original config setting
        self.config.use_matrix_kan_optimized_spline = self.original_use_matrix_kan

    def test_full_model_forward_backward(self):
        """
        Tests the full KAN_MAMOTE_Model's forward and backward passes,
        including output shapes and gradient existence.
        """
        model = KAN_MAMOTE_Model(config=self.config)
        model.to(self.device)

        # Create dummy input data
        timestamps = torch.randn(self.batch_size, self.seq_len, self.timestamp_dim, device=self.device)
        event_features = torch.randn(self.batch_size, self.seq_len, self.raw_event_feature_dim, device=self.device)

        # Ensure inputs require gradients
        timestamps.requires_grad_(True)
        event_features.requires_grad_(True)

        print("  Performing forward pass...")
        final_embeddings, moe_losses_info = model(timestamps, event_features)

        # --- Assert Output Shapes ---
        # 1. final_embeddings
        expected_embeddings_shape = (self.batch_size, self.seq_len, self.config.hidden_dim_mamba)
        self.assertEqual(final_embeddings.shape, expected_embeddings_shape,
                         f"Final embeddings shape mismatch. Expected {expected_embeddings_shape}, got {final_embeddings.shape}")
        print(f"  Final embeddings shape OK: {final_embeddings.shape}")

        # 2. moe_losses_info (tuple of abs_expert_weights_for_loss, rel_expert_weights_for_loss)
        abs_expert_weights, rel_expert_weights = moe_losses_info
        expected_moe_weights_shape = (self.batch_size, self.seq_len, self.config.num_experts)
        self.assertEqual(abs_expert_weights.shape, expected_moe_weights_shape,
                         f"Abs expert weights shape mismatch. Expected {expected_moe_weights_shape}, got {abs_expert_weights.shape}")
        self.assertEqual(rel_expert_weights.shape, expected_moe_weights_shape,
                         f"Rel expert weights shape mismatch. Expected {expected_moe_weights_shape}, got {rel_expert_weights.shape}")
        print(f"  MoE weights shapes OK: {abs_expert_weights.shape}")

        # --- Assert Differentiability (Backward Pass) ---
        print("  Performing backward pass...")
        try:
            # Sum a loss to backpropagate. For testing, a simple sum is enough.
            # Combine main output and one of the MoE loss components for full gradient check.
            loss = final_embeddings.sum() + abs_expert_weights.sum()
            model.zero_grad()
            loss.backward()
        except RuntimeError as e:
            self.fail(f"Backward pass failed for KAN_MAMOTE_Model: {e}")

        # Check gradients for model's learnable parameters
        found_learnable_params = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for learnable parameter {name}.")
                found_learnable_params = True
        
        self.assertTrue(found_learnable_params, "No learnable parameters found or tested for gradients in KAN_MAMOTE_Model.")
        print("  KAN_MAMOTE_Model: All learnable parameters have gradients.")
        print("Test `test_full_model_forward_backward` passed successfully.")

if __name__ == '__main__':
    # To run this test:
    # 1. Navigate to the root of your project (kan_mamote_project/) in your terminal.
    # 2. Make sure your virtual environment is activated.
    # 3. Run: python -m unittest tests/test_kan_mammote_model.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)