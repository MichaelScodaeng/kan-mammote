# models/SPETimeEmbeddingModel.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DyGMamba.utils.utils import NeighborSampler
from DyGMamba.models.modules import TimeEncoder # This module already implements SPE-like encoding
from DyGMamba.models.modules import MergeLayer # For combining features

class SPETimeEmbeddingModel(nn.Module):
    """
    A baseline model that uses Sinusoidal Positional Embeddings (SPE) for time encoding.
    It applies SPE to time differences and integrates these into node embeddings.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim 
        self.num_layers = num_layers # Accepted for interface, but not used by this simple model
        self.num_heads = num_heads # Accepted for interface, but not used by this simple model
        self.dropout = dropout
        self.device = device

        # The TimeEncoder from modules.py already implements the fixed sinusoidal encoding
        # (specifically, it's cos(w*t) where w are fixed geometric progression frequencies and biases are 0).
        # We ensure parameter_requires_grad=False to represent a truly non-learnable SPE.
        self.spe_time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)

        # Output projection. Input will be: node_feat_dim (current node) +
        # (node_feat_dim + edge_feat_dim + time_feat_dim) (from averaged neighbors with SPE).
        self.output_proj = nn.Linear(self.node_feat_dim + self.node_feat_dim + self.edge_feat_dim + self.time_feat_dim, self.node_feat_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Computes source and destination node temporal embeddings using Sinusoidal Positional Embeddings (SPE).
        SPE is applied to time differences (relative time).

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
        # 1. Get historical neighbors and pad sequences
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        max_seq_length_for_padding = 32 # Consistent padding length
        
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

        # 2. Extract features, including SPE-encoded time features
        # The `get_features` helper method uses a `TimeEncoder` (here, our `spe_time_encoder`)
        # to generate the time features from time differences.
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_spe_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              time_encoder=self.spe_time_encoder) # Use the non-learnable SPE encoder

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_spe_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.spe_time_encoder) # Use the non-learnable SPE encoder

        # 3. Combine all features and aggregate for node embeddings
        # Concatenate: raw node features, raw edge features, and SPE time features
        src_combined_features = torch.cat([
            src_padded_nodes_neighbor_node_raw_features,
            src_padded_nodes_edge_raw_features,
            src_spe_time_features # Use SPE's output for time features
        ], dim=-1) # (batch_size, max_seq_length, node_feat_dim + edge_feat_dim + time_feat_dim)

        dst_combined_features = torch.cat([
            dst_padded_nodes_neighbor_node_raw_features,
            dst_padded_nodes_edge_raw_features,
            dst_spe_time_features # Use SPE's output for time features
        ], dim=-1) # (batch_size, max_seq_length, node_feat_dim + edge_feat_dim + time_feat_dim)

        # Simple aggregation: Mean pooling over the sequence dimension (L)
        src_aggregated_features = torch.mean(src_combined_features, dim=1)
        dst_aggregated_features = torch.mean(dst_combined_features, dim=1)

        # Add the current node's own raw static feature
        current_src_node_raw_features = self.node_raw_features[torch.from_numpy(src_node_ids)]
        current_dst_node_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]

        src_final_input = torch.cat([current_src_node_raw_features, src_aggregated_features], dim=-1)
        dst_final_input = torch.cat([current_dst_node_raw_features, dst_aggregated_features], dim=-1)

        # Project combined features to node_feat_dim
        src_node_embeddings = self.dropout_layer(self.relu(self.output_proj(src_final_input)))
        dst_node_embeddings = self.dropout_layer(self.relu(self.output_proj(dst_final_input)))
        
        # Dummy time_diff_embedding for compatibility with MergeLayerTD
        dummy_time_diff_embedding = torch.zeros_like(src_node_embeddings).to(self.device)

        return src_node_embeddings, dst_node_embeddings, dummy_time_diff_embedding

    # Helper methods (copied from KANMAMMOTE.py and LeTEModel.py for consistency)
    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int):
        """
        Pads sequences of historical interactions for nodes.
        This method handles left-padding to a fixed `max_input_sequence_length`.
        The "current" interaction is placed at the very end of the sequence.
        """
        max_seq_length = max_input_sequence_length
        for idx in range(len(nodes_neighbor_ids_list)):
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            padded_nodes_edge_ids[idx, -1] = 0 
            padded_nodes_neighbor_times[idx, -1] = node_interact_times[idx]

            num_historical = len(nodes_neighbor_ids_list[idx])
            if num_historical > 0:
                padded_nodes_neighbor_ids[idx, -(num_historical + 1):-1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -(num_historical + 1):-1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -(num_historical + 1):-1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        Extracts node, edge, and time features for the padded sequences.
        """
        device = self.device
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        
        # This is where SPE is applied to the time differences
        delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(device)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=delta_times)

        # Set the time features to all zeros for the padded timestamp (where neighbor_id is 0)
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