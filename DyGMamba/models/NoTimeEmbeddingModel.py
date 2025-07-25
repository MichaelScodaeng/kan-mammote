# models/NoTimeEmbeddingModel.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DyGMamba.utils.utils import NeighborSampler
# We still import TimeEncoder, but its output for time features will be ignored.
# It's here mainly because get_features helper function returns it.
from DyGMamba.models.modules import TimeEncoder 

class NoTimeEmbeddingModel(nn.Module):
    """
    A baseline model that explicitly uses NO time embedding.
    It computes node embeddings based only on static node and edge features,
    while still conforming to the dynamic graph framework's interface.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        # time_feat_dim is accepted for interface compatibility but not used for time encoding.
        self.time_feat_dim = time_feat_dim 
        self.num_layers = num_layers # Accepted for interface, but not used by this simple model
        self.num_heads = num_heads # Accepted for interface, but not used by this simple model
        self.dropout = dropout
        self.device = device

        # Initialize a dummy TimeEncoder, its output will be discarded.
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)

        # Output projection.
        # This model will just average static node/edge features from neighbors
        # and combine with the current node's static feature.
        # Input to this projection will be: node_feat_dim (current node) +
        # (node_feat_dim + edge_feat_dim) (from averaged neighbors).
        self.output_proj = nn.Linear(self.node_feat_dim + self.node_feat_dim + self.edge_feat_dim, self.node_feat_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Computes source and destination node embeddings WITHOUT using time information.
        It uses only static node and edge features.

        Args:
            src_node_ids (np.ndarray): Source node IDs for the current batch. Shape (batch_size,).
            dst_node_ids (np.ndarray): Destination node IDs for the current batch. Shape (batch_size,).
            node_interact_times (np.ndarray): Interaction timestamps for the current batch. Shape (batch_size,).
                                              This parameter is IGNORED for time encoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                src_node_embeddings (torch.Tensor): Node embeddings for source nodes. Shape (batch_size, node_feat_dim).
                dst_node_embeddings (torch.Tensor): Node embeddings for destination nodes. Shape (batch_size, node_feat_dim).
                dummy_time_diff_embedding (torch.Tensor): Zeros tensor for compatibility. Shape (batch_size, node_feat_dim).
        """
        # 1. Get historical neighbors and pad sequences for source and destination nodes
        # The `node_interact_times` is passed to `get_all_first_hop_neighbors` to filter neighbors
        # by time (i.e., only neighbors *before* the current interaction).
        # However, the *values* of `node_interact_times` are not used for features in this model.
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # Use a fixed max sequence length for padding. Matches DyGMamba's default.
        max_seq_length_for_padding = 32 
        
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, _ = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, # node_interact_times only used for padding current node
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list, # times not directly used in features later
                               max_input_sequence_length=max_seq_length_for_padding)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, _ = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=max_seq_length_for_padding)

        # 2. Extract static features for the padded sequences
        # We explicitly IGNORE the time features that `get_features` would typically produce.
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, _ = \
            self.get_features(node_interact_times=node_interact_times, # Only passed for compatibility, not used for time encoding
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, # Only passed for compatibility, not used for time encoding
                              time_encoder=self.time_encoder) # The TimeEncoder output is discarded

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, _ = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder) # The TimeEncoder output is discarded

        # 3. Combine static features and aggregate for node embeddings
        # For this "No Embedding" baseline, we concatenate raw node features and raw edge features.
        src_combined_static_features = torch.cat([
            src_padded_nodes_neighbor_node_raw_features,
            src_padded_nodes_edge_raw_features
        ], dim=-1) # (batch_size, max_seq_length, node_feat_dim + edge_feat_dim)

        dst_combined_static_features = torch.cat([
            dst_padded_nodes_neighbor_node_raw_features,
            dst_padded_nodes_edge_raw_features
        ], dim=-1) # (batch_size, max_seq_length, node_feat_dim + edge_feat_dim)

        # Simple aggregation: Mean pooling over the sequence dimension (L)
        src_aggregated_features = torch.mean(src_combined_static_features, dim=1)
        dst_aggregated_features = torch.mean(dst_combined_static_features, dim=1)

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
        Extracts node, edge, and time features for the padded sequences. Time features are returned but not used.
        """
        device = self.device
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        
        # We still call the time_encoder here for compatibility of the helper function,
        # but the output `padded_nodes_neighbor_time_features` will be discarded by the main logic.
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