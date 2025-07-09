# kan_mamote/src/models/immediate_fasterkan_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add faster_kan to path
faster_kan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'faster_kan')
if faster_kan_path not in sys.path:
    sys.path.append(faster_kan_path)

from src.utils.config import KANMAMOTEConfig
from src.models.k_mote import K_MOTE
from src.models.c_mamba import SimplifiedContinuousMambaBlock

# Import Faster-KAN from the cloned repository
try:
    from faster_kan.fasterkan import FasterKAN, FasterKANLayer
    FASTER_KAN_AVAILABLE = True
    print("✓ Faster-KAN successfully imported from cloned repository")
except ImportError as e:
    print(f"Warning: Faster-KAN not available. Error: {e}")
    print("Using MLP fallback.")
    FASTER_KAN_AVAILABLE = False
    # Define dummy classes to avoid unbound errors
    FasterKAN = None  # type: ignore
    FasterKANLayer = None  # type: ignore

class FasterKANTemporalLayer(nn.Module):
    """
    Wrapper around the cloned Faster-KAN layer for temporal difference processing.
    Uses the actual FasterKANLayer from the repository.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 grid_min: float = -2.0,
                 grid_max: float = 2.0,
                 num_grids: int = 8,
                 spline_weight_init_scale: float = 0.667):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if FASTER_KAN_AVAILABLE and FasterKANLayer is not None:
            # Use actual Faster-KAN layer from cloned repository
            try:
                self.kan_layer = FasterKANLayer(
                    input_dim=input_dim,  # FasterKANLayer uses input_dim/output_dim
                    output_dim=output_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    exponent=2,
                    inv_denominator=0.5,
                    train_grid=False,
                    train_inv_denominator=False,
                    base_activation=F.silu,
                    spline_weight_init_scale=spline_weight_init_scale
                )
                self.use_kan = True
                print(f"✓ Using FasterKANLayer: {input_dim}→{output_dim}, grids={num_grids}")
            except Exception as e:
                print(f"Failed to create FasterKANLayer: {e}")
                self.use_kan = False
        else:
            self.use_kan = False
        
        if not self.use_kan:
            # Enhanced MLP fallback with spline-like behavior
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim * 2, input_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            print(f"⚠ Using MLP fallback: {input_dim}→{output_dim}")
    
    def forward(self, x):
        """
        Forward pass through Faster-KAN or MLP fallback.
        
        Args:
            x: (batch_size, seq_len, input_dim) - temporal difference embeddings
            
        Returns:
            output: (batch_size, seq_len, output_dim) - processed embeddings
        """
        if self.use_kan:
            # Faster-KAN expects 2D input: (batch_size * seq_len, input_dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.view(-1, dim)
            
            # Process through Faster-KAN layer
            output_flat = self.kan_layer(x_flat)
            
            # Reshape back to sequence format
            output = output_flat.view(batch_size, seq_len, -1)
        else:
            # MLP can handle 3D input directly
            output = self.mlp(x)
        
        return output

class FasterKANTemporalNetwork(nn.Module):
    """
    Multi-layer Faster-KAN network for sophisticated temporal difference processing.
    Uses the actual FasterKAN from the cloned repository.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 grid_min: float = -2.0,
                 grid_max: float = 2.0,
                 num_grids: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        if FASTER_KAN_AVAILABLE and FasterKAN is not None:
            # Create layer dimensions
            if num_layers == 1:
                layers_hidden = [input_dim, output_dim]
            else:
                layers_hidden = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            
            # Use actual FasterKAN network from cloned repository
            try:
                self.kan_network = FasterKAN(
                    layers_hidden=layers_hidden,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    exponent=2,
                    inv_denominator=0.5,
                    train_grid=False,
                    train_inv_denominator=False,
                    base_activation=F.silu,
                    spline_weight_init_scale=0.667
                )
                self.use_kan = True
                print(f"✓ Using FasterKAN Network: {layers_hidden}, grids={num_grids}")
            except Exception as e:
                print(f"Failed to create FasterKAN: {e}")
                self.use_kan = False
        else:
            self.use_kan = False
        
        if not self.use_kan:
            # Enhanced MLP fallback
            layers = []
            dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            
            for i in range(len(dims) - 1):
                layers.extend([
                    nn.LayerNorm(dims[i]),
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.SiLU() if i < len(dims) - 2 else nn.Identity(),
                    nn.Dropout(0.1) if i < len(dims) - 2 else nn.Identity()
                ])
            
            self.mlp_network = nn.Sequential(*layers)
            self.use_kan = False
            print(f"⚠ Using MLP Network fallback: {dims}")
    
    def forward(self, x):
        """
        Forward pass through multi-layer Faster-KAN or MLP network.
        
        Args:
            x: (batch_size, seq_len, input_dim) - temporal difference embeddings
            
        Returns:
            output: (batch_size, seq_len, output_dim) - processed embeddings
        """
        if self.use_kan:
            # FasterKAN expects 2D input: (batch_size * seq_len, input_dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.view(-1, dim)
            
            # Process through Faster-KAN network
            output_flat = self.kan_network(x_flat)
            
            # Reshape back to sequence format
            output = output_flat.view(batch_size, seq_len, -1)
        else:
            # MLP can handle 3D input directly
            output = self.mlp_network(x)
        
        return output

class ImmediateFasterKANLayer(nn.Module):
    """
    Your proposed architecture:
    1. K-MOTE on current and previous times separately
    2. Compute difference of embeddings
    3. Process through Faster-KAN
    4. Feed to C-Mamba
    """
    
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config
        self.D_time = config.D_time
        
        # K-MOTE for current times
        self.k_mote_current = K_MOTE(config)
        
        # K-MOTE for previous times (separate instance for potentially different learning)
        self.k_mote_previous = K_MOTE(config)
        
        # Faster-KAN layer for temporal difference processing
        # Input: temporal difference embeddings, Output: same dimension
        self.temporal_difference_kan = FasterKANTemporalLayer(
            input_dim=config.D_time,
            output_dim=config.D_time,  # Keep same dimension
            grid_min=getattr(config, 'kan_grid_min', -2.0),
            grid_max=getattr(config, 'kan_grid_max', 2.0),
            num_grids=getattr(config, 'kan_grid_size', 8),
            spline_weight_init_scale=getattr(config, 'kan_spline_scale', 0.667)
        )
        
        # C-Mamba for sequence modeling (using simplified version for stability)
        self.c_mamba = SimplifiedContinuousMambaBlock(
            input_dim=config.D_time,
            config=config,
            kmote_layer=self.k_mote_current,
            faster_kan_layer=self.temporal_difference_kan
        )
        
    def compute_previous_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute previous timestamps for each position.
        For the first position, use the same timestamp (no previous).
        
        Args:
            timestamps: (batch_size, seq_len, 1) - current timestamps
            
        Returns:
            previous_timestamps: (batch_size, seq_len, 1) - previous timestamps
        """
        batch_size, seq_len, _ = timestamps.shape
        
        # Shift timestamps to get previous times
        previous_timestamps = torch.zeros_like(timestamps)
        previous_timestamps[:, 0:1, :] = timestamps[:, 0:1, :]  # First position = current
        if seq_len > 1:
            previous_timestamps[:, 1:, :] = timestamps[:, :-1, :]   # Rest = previous
        
        return previous_timestamps
    
    def forward(self, 
                timestamps: torch.Tensor,
                event_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through your proposed architecture.
        
        Args:
            timestamps: (batch_size, seq_len, 1) - current timestamps
            event_features: (batch_size, seq_len, feature_dim) - event features
            
        Returns:
            final_embeddings: (batch_size, seq_len, hidden_dim_mamba)
            detailed_info: Dictionary with all intermediate information
        """
        batch_size, seq_len, _ = timestamps.shape
        device = timestamps.device
        
        # Step 1: Compute previous timestamps
        previous_timestamps = self.compute_previous_timestamps(timestamps)
        
        # Flatten for K-MOTE processing
        timestamps_flat = timestamps.view(-1, 1)  # (batch_size * seq_len, 1)
        previous_timestamps_flat = previous_timestamps.view(-1, 1)
        
        # Handle empty event features properly
        if event_features.shape[-1] == 0:
            # For empty features, pass None to K-MOTE (it can handle this)
            event_features_flat = None
        else:
            event_features_flat = event_features.view(-1, event_features.shape[-1])
        
        # Step 2: Apply K-MOTE to current times
        current_embeddings, current_weights, current_masks = self.k_mote_current(
            timestamps_flat, 
            event_features_flat
        )
        # Reshape back to sequence format
        # Debug: Check shapes before reshaping
        expected_total = batch_size * seq_len * self.D_time
        actual_total = current_embeddings.numel()
        if expected_total != actual_total:
            print(f"Warning: Shape mismatch in current_embeddings reshape")
            print(f"  Expected total elements: {expected_total}")
            print(f"  Actual total elements: {actual_total}")
            print(f"  current_embeddings shape: {current_embeddings.shape}")
            print(f"  Target shape: ({batch_size}, {seq_len}, {self.D_time})")
        
        try:
            current_embeddings = current_embeddings.view(batch_size, seq_len, self.D_time)
        except RuntimeError as e:
            print(f"Error reshaping current_embeddings: {e}")
            # Create zero tensor as fallback
            current_embeddings = torch.zeros(batch_size, seq_len, self.D_time, device=timestamps.device)
            
        current_weights = current_weights.view(batch_size, seq_len, -1)
        current_masks = current_masks.view(batch_size, seq_len, -1)
        
        # Step 3: Apply K-MOTE to previous times
        previous_embeddings, previous_weights, previous_masks = self.k_mote_previous(
            previous_timestamps_flat,
            event_features_flat
        )
        # Reshape back to sequence format
        # Debug: Check shapes before reshaping
        expected_total = batch_size * seq_len * self.D_time
        actual_total = previous_embeddings.numel()
        if expected_total != actual_total:
            print(f"Warning: Shape mismatch in previous_embeddings reshape")
            print(f"  Expected total elements: {expected_total}")
            print(f"  Actual total elements: {actual_total}")
            print(f"  previous_embeddings shape: {previous_embeddings.shape}")
            print(f"  Target shape: ({batch_size}, {seq_len}, {self.D_time})")
        
        try:
            previous_embeddings = previous_embeddings.view(batch_size, seq_len, self.D_time)
        except RuntimeError as e:
            print(f"Error reshaping previous_embeddings: {e}")
            # Create zero tensor as fallback
            previous_embeddings = torch.zeros(batch_size, seq_len, self.D_time, device=timestamps.device)
            
        previous_weights = previous_weights.view(batch_size, seq_len, -1)
        previous_masks = previous_masks.view(batch_size, seq_len, -1)
        
        # Step 4: Compute temporal difference embeddings
        time_diff_embeddings = current_embeddings - previous_embeddings
        
        # Step 5: Process through Faster-KAN
        kan_processed_diffs = self.temporal_difference_kan(time_diff_embeddings)
        
        # Step 6: Feed to C-Mamba (using original timestamps for internal time difference computation)
        final_embeddings = self.c_mamba(kan_processed_diffs, timestamps)
        
        # Prepare detailed information for analysis and visualization
        detailed_info = {
            'temporal_differences': time_diff_embeddings,
            'kmote_current': current_embeddings,
            'kmote_previous': previous_embeddings,
            'fasterkan_output': kan_processed_diffs,
            'mamba_output': final_embeddings,
            'kmote_expert_mask': current_masks,  # For expert usage analysis
            'kmote_info': {
                'expert_weights': current_weights,
                'expert_mask': current_masks,
                'router_logits': current_weights
            }
        }
        
        return final_embeddings, detailed_info

class ImprovedKANMAMOTE(nn.Module):
    """
    Complete improved KAN-MAMMOTE model using the immediate Faster-KAN architecture.
    """
    
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config
        
        # Use the new Immediate Faster-KAN architecture
        self.immediate_fasterkan_layer = ImmediateFasterKANLayer(config)
        
    def forward(self, 
                timestamps: torch.Tensor,
                event_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through improved KAN-MAMMOTE.
        
        Args:
            timestamps: (batch_size, seq_len, 1)
            event_features: (batch_size, seq_len, feature_dim)
            
        Returns:
            embeddings: (batch_size, seq_len, hidden_dim_mamba)
            detailed_info: Dictionary with all intermediate information
        """
        return self.immediate_fasterkan_layer(timestamps, event_features)
