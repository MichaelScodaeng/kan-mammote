# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time

# Import your components
from src.utils.config import KANMAMMOTEConfig
from src.models.kan_mammote import KANMAMMOTE

# --- Dummy Dataset ---
class DummyContinuousTimeDataset(Dataset):
    def __init__(self, num_samples, sequence_length, input_feature_dim, output_dim_for_task):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_feature_dim = input_feature_dim
        self.output_dim_for_task = output_dim_for_task

        # Generate synthetic data
        self.timestamps = torch.linspace(0, 100, sequence_length).unsqueeze(0).repeat(num_samples, 1) # (num_samples, seq_len)
        self.features = torch.randn(num_samples, sequence_length, input_feature_dim) # (num_samples, seq_len, input_feature_dim)
        # Simple target: sum of features + sine wave based on timestamp
        self.targets = self.features.sum(dim=-1, keepdim=True) + torch.sin(self.timestamps / 5).unsqueeze(-1) + torch.randn(num_samples, sequence_length, output_dim_for_task) * 0.1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'timestamps': self.timestamps[idx],
            'features': self.features[idx],
            'targets': self.targets[idx]
        }

# --- Main Training Function ---
def train_kan_mammote():
    config = KANMAMMOTEConfig()
    
    print(f"Using device: {config.device}")
    print(f"D_time: {config.D_time}, Num Layers: {config.num_layers}")

    # Initialize model
    model = KANMAMMOTE(config).to(config.device, dtype=config.dtype)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Loss function for the main task (e.g., MSE for regression)
    criterion = nn.MSELoss()

    # Create dummy dataset and dataloader
    train_dataset = DummyContinuousTimeDataset(
        num_samples=config.batch_size * 50, # More samples for training
        sequence_length=config.sequence_length,
        input_feature_dim=config.input_feature_dim,
        output_dim_for_task=config.output_dim_for_task
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    print("\nStarting conceptual training loop...")
    # Training loop use tqdm for progress bar
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        model.train() # Set model to training mode
        total_loss = 0.0
        main_task_loss_sum = 0.0
        load_balance_loss_sum = 0.0
        sobolev_l2_loss_sum = 0.0
        total_variation_loss_sum = 0.0
        
        start_time = time.time()

        for batch_idx, data in enumerate(train_dataloader):
            timestamps = data['timestamps'] # (B, L)
            features = data['features']     # (B, L, D_in)
            targets = data['targets']       # (B, L, D_out)
            timestamps = timestamps.to(config.device, dtype=config.dtype)
            features = features.to(config.device, dtype=config.dtype)
            targets = targets.to(config.device, dtype=config.dtype)

            optimizer.zero_grad() # Clear gradients

            # Forward pass
            predictions, regularization_losses = model(timestamps, features)

            # Calculate main task loss
            main_loss = criterion(predictions, targets)

            # Retrieve regularization losses
            load_balance_loss = regularization_losses["load_balance_loss"]
            sobolev_l2_loss = regularization_losses["sobolev_l2_loss"]
            total_variation_loss = regularization_losses["total_variation_loss"]

            # Combine all losses
            combined_loss = (
                main_loss
                + load_balance_loss # Load balance loss already has its coefficient applied inside
                + config.lambda_sobolev_l2 * sobolev_l2_loss
                + config.lambda_total_variation * total_variation_loss
            )

            # Backward pass and optimization
            combined_loss.backward()
            optimizer.step()

            total_loss += combined_loss.item()
            main_task_loss_sum += main_loss.item()
            load_balance_loss_sum += load_balance_loss.item()
            sobolev_l2_loss_sum += sobolev_l2_loss.item()
            total_variation_loss_sum += total_variation_loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_dataloader)} - "
                      f"Combined Loss: {combined_loss.item():.6f}, "
                      f"Main Loss: {main_loss.item():.6f}, "
                      f"LB Loss: {load_balance_loss.item():.6f}")


        avg_total_loss = total_loss / len(train_dataloader)
        avg_main_loss = main_task_loss_sum / len(train_dataloader)
        avg_load_balance_loss = load_balance_loss_sum / len(train_dataloader)
        avg_sobolev_l2_loss = sobolev_l2_loss_sum / len(train_dataloader)
        avg_total_variation_loss = total_variation_loss_sum / len(train_dataloader)
        
        end_time = time.time()
        
        print(f"Epoch {epoch+1}/{config.num_epochs} finished in {end_time - start_time:.2f}s")
        print(f"  Avg Total Loss: {avg_total_loss:.6f}")
        print(f"  Avg Main Loss: {avg_main_loss:.6f}")
        print(f"  Avg Load Balance Loss: {avg_load_balance_loss:.6f}")
        print(f"  Avg Sobolev L2 Loss: {avg_sobolev_l2_loss:.6f} (Stub)")
        print(f"  Avg Total Variation Loss: {avg_total_variation_loss:.6f} (Stub)")

    print("\nConceptual training loop finished.")

if __name__ == '__main__':
    # Make sure you are in the kan_mamote directory
    # or adjust sys.path to include kan_mamote.src if running from elsewhere
    # For example:
    # import sys
    # sys.path.append('./kan_mamote/src')
    
    train_kan_mammote()