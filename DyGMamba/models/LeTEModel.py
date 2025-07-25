# models/LeTEModel.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DyGMamba.utils.utils import NeighborSampler
# Import the CombinedLeTE time embedding you provided
from src.LETE.LeTE import CombinedLeTE # Assuming LeTE.py is in the same directory for simplicity, or adjust path if it's in a subfolder like src/baselines/
from DyGMamba.models.modules import TimeEncoder, MergeLayer # MergeLayer might be useful for combining features

class LeTEModel(nn.Module):
    """
    LeTE wrapper model for the dynamic graph link prediction framework.
    This class acts as a 'dynamic_backbone' and implements the
    compute_src_dst_node_temporal_embeddings interface.
    It uses CombinedLeTE to encode time differences derived from edge timestamps.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim # This will be the dim for CombinedLeTE output
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        # Instantiate CombinedLeTE.
        # Its 'dim' parameter will be our time_feat_dim.
        # 'p' controls split between Fourier and Spline (e.g., 0.5 for equal split).
        # We can expose 'p', 'grid_size_fourier', 'grid_size_spline' etc. as args if needed.
        self.lete_time_encoder = CombinedLeTE(dim=self.time_feat_dim, p=0.5,
                                              layer_norm=True, scale=True,
                                              parameter_requires_grad=True)
        
        # Standard TimeEncoder from modules.py for internal feature extraction (e.g., `get_features`)
        # Used for raw time differences before feeding to LeTE, if LeTE doesn't handle raw timestamps.
        # However, CombinedLeTE expects raw timestamps. So, TimeEncoder is NOT used for LeTE's input.
        # But `get_features` helper method needs it for `padded_nodes_neighbor_time_features`.
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)


        # To align with how models like TGAT, DyGFormer etc. process node embeddings:
        # They typically apply graph attention/mixer layers to features (node_raw_features + time_features + edge_features).
        # For simplicity in this baseline, we can use a basic aggregation (e.g., concatenation + linear layer)
        # or a simple MLP after gathering all features for a node's sequence.

        # Input dimension for the node embedding computation: node_feat_dim + edge_feat_dim + time_feat_dim
        # Assuming we concatenate raw node features (of current node and neighbors), raw edge features,
        # and LeTE's time embedding for the sequence.
        # The output from get_features will be (B, L, node_feat_dim), (B, L, edge_feat_dim), (B, L, time_feat_dim)
        # We'll concatenate these and pass through some layers to get a final node embedding.
        
        # Simplistic aggregation: Average neighbor features and concatenate with node_raw_feature
        # Then apply the LeTE time embedding for the *current interaction* as a separate input.
        # This will be more aligned with how existing models process, where time is an aspect of messages.

        # Let's align it closer to TGAT's simple approach:
        # Combine node_features, edge_features, and the LeTE-encoded time features
        # for neighbors, then aggregate them.

        # This baseline will effectively replace the Graph Attention / Transformer logic of models
        # like TGAT/DyGFormer with a simpler MLP on aggregated features that *includes* LeTE's time encoding.
        
        # Input to the first layer for node features: node_feat_dim
        # Input to the first layer for edge features: edge_feat_dim
        # Input to the first layer for time features: time_feat_dim (from CombinedLeTE)

        # Output projection to get the final node embedding dimension
        # A simple linear layer might be sufficient after aggregating the sequence features.
        # After gathering features for a node's sequence, combine them, e.g., via simple average or sum.
        # Then, project to node_feat_dim.

        # Example: Input to a simple aggregation layer might be (B, L, node_feat_dim + edge_feat_dim + time_feat_dim)
        # After averaging/pooling (B, node_feat_dim + edge_feat_dim + time_feat_dim)
        # Then project this to node_feat_dim.
        
        self.output_proj = nn.Linear(self.node_feat_dim + self.edge_feat_dim + self.time_feat_dim, self.node_feat_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Computes source and destination node temporal embeddings using LeTE.
        This method conforms to the interface expected by train_link_prediction.py.

        Args:
            src_node_ids (np.ndarray): Source node IDs for the current batch. Shape (batch_size,).
            dst_node_ids (np.ndarray): Destination node IDs for the current batch. Shape (batch_size,).
            node_interact_times (np.ndarray): Interaction timestamps for the current batch. Shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                src_node_embeddings (torch.Tensor): Temporal embeddings for source nodes. Shape (batch_size, node_feat_dim).
                dst_node_embeddings (torch.Tensor): Temporal embeddings for destination nodes. Shape (batch_size, node_feat_dim).
                dummy_time_diff_embedding (torch.Tensor): Zeros tensor for compatibility. Shape (batch_size, node_feat_dim).
        """
        # 1. Get historical neighbors and pad sequences for source and destination nodes
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # Define a max sequence length for padding. We can use a default or an arg from config.
        # For simplicity, let's use the same fixed length as DyGMamba's default for padding helper:
        max_seq_length_for_padding = 32 # Default from DyGMamba/DyGFormer load_configs.py
        
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=max_seq_length_for_padding)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=max_seq_length_for_padding)

        # 2. Extract features for the padded sequences
        # These are raw node features of neighbors, raw edge features, and time features (from modules.TimeEncoder)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features_modules = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder) # Using modules.TimeEncoder for this helper

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features_modules = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder) # Using modules.TimeEncoder for this helper

        # 3. Apply LeTE to time differences
        # CombinedLeTE expects timestamps (scalar values). Here we feed it the time differences.
        # This aligns with how TGAT's TimeEncoder is used.
        # src_padded_nodes_neighbor_times (np.ndarray) contains (batch_size, max_seq_length) time values relative to current interaction.
        # It needs to be converted to torch.Tensor and possibly unsqueezed if CombinedLeTE expects (B,L,1)
        
        # CombineLeTE expects (B, L) input for timestamps
        src_lete_time_features = self.lete_time_encoder(
            torch.from_numpy(src_padded_nodes_neighbor_times).float().to(self.device)
        ) # Output: (batch_size, max_seq_length, time_feat_dim)

        dst_lete_time_features = self.lete_time_encoder(
            torch.from_numpy(dst_padded_nodes_neighbor_times).float().to(self.device)
        ) # Output: (batch_size, max_seq_length, time_feat_dim)

        # 4. Combine all features and aggregate for node embeddings
        # Concatenate: (B, L, node_feat_dim), (B, L, edge_feat_dim), (B, L, time_feat_dim from LeTE)
        src_combined_features = torch.cat([
            src_padded_nodes_neighbor_node_raw_features,
            src_padded_nodes_edge_raw_features,
            src_lete_time_features # Use LeTE's output for time features
        ], dim=-1) # (batch_size, max_seq_length, node_feat_dim + edge_feat_dim + time_feat_dim)

        dst_combined_features = torch.cat([
            dst_padded_nodes_neighbor_node_raw_features,
            dst_padded_nodes_edge_raw_features,
            dst_lete_time_features # Use LeTE's output for time features
        ], dim=-1) # (batch_size, max_seq_length, node_feat_dim + edge_feat_dim + time_feat_dim)

        # Simple aggregation: Mean pooling over the sequence dimension (L)
        src_aggregated_features = torch.mean(src_combined_features, dim=1) # (batch_size, combined_feature_dim)
        dst_aggregated_features = torch.mean(dst_combined_features, dim=1) # (batch_size, combined_feature_dim)

        # Project aggregated features to node_feat_dim
        src_node_embeddings = self.dropout_layer(self.relu(self.output_proj(src_aggregated_features))) # (batch_size, node_feat_dim)
        dst_node_embeddings = self.dropout_layer(self.relu(self.output_proj(dst_aggregated_features))) # (batch_size, node_feat_dim)
        
        # Dummy time_diff_embedding for compatibility with MergeLayerTD
        dummy_time_diff_embedding = torch.zeros_like(src_node_embeddings).to(self.device)

        return src_node_embeddings, dst_node_embeddings, dummy_time_diff_embedding

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int):
        """
        Pads sequences of historical interactions for nodes. Copied from models/KANMAMMOTE.py.
        This method handles left-padding to a fixed `max_input_sequence_length`.
        The "current" interaction is placed at the very end of the sequence.
        """
        # max_seq_length will include the target node itself
        max_seq_length = max_input_sequence_length
        # No patch_size consideration in this LeTE model, but keeping it in the signature for consistency
        # with KANMAMMOTE's helper or if a patch_size argument is later passed.
        # For this model, max_seq_length is simply max_input_sequence_length.
        
        # First cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1
        for idx in range(len(nodes_neighbor_ids_list)):
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]

        # Pad the sequences
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            # Place current node at the very end of the sequence
            padded_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            padded_nodes_edge_ids[idx, -1] = 0 # No edge ID for self-loop
            padded_nodes_neighbor_times[idx, -1] = node_interact_times[idx]

            # Left-pad historical neighbors
            num_historical = len(nodes_neighbor_ids_list[idx])
            if num_historical > 0:
                padded_nodes_neighbor_ids[idx, -(num_historical + 1):-1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -(num_historical + 1):-1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -(num_historical + 1):-1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        Extracts node, edge, and time features for the padded sequences. Copied from models/KANMAMMOTE.py.
        """
        device = self.device
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        
        # Calculate time differences for the standard time encoder (from modules.py)
        delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(device)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=delta_times)

        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Sets the neighbor sampler and resets its random state if applicable.
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()