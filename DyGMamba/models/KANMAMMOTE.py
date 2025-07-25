# models/KANMAMMOTE.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DyGMamba.utils.utils import NeighborSampler
# Import your core KAN-MAMMOTE model and its configuration
from src.utils.config import KANMAMMOTEConfig
from src.models.kan_mammote import KANMAMMOTE as KANMAMMOTE_Core
from DyGMamba.models.modules import TimeEncoder # Used for common time encoding if needed in feature construction

class KANMAMMOTE(nn.Module):
    """
    KANMAMMOTE wrapper model for the dynamic graph link prediction framework.
    This class acts as the 'dynamic_backbone' and implements the
    compute_src_dst_node_temporal_embeddings interface.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 config: KANMAMMOTEConfig): # Config object passed directly
        super().__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(config.device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(config.device)

        self.neighbor_sampler = neighbor_sampler
        self.config = config # Store the KANMAMMOTEConfig object

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]

        # Instantiate your core KAN-MAMMOTE model
        self.kan_mammote_core = KANMAMMOTE_Core(config=self.config)

        # Output projection for main node embeddings if d_model != node_feat_dim
        # The kan_mammote_core's prediction_head outputs to config.output_dim_for_task (scalar for link pred)
        # We need the last layer's hidden state (d_model) to become node_feat_dim
        # So, we'll need an additional projection from d_model to node_feat_dim
        self.node_embedding_proj = nn.Linear(self.config.d_model, self.node_feat_dim)

        # Projection for delta_t_embedding to match node_feat_dim for MergeLayerTD
        self.delta_t_embedding_proj = nn.Linear(self.config.D_time, self.node_feat_dim)
        
        # DyGMamba's `compute_src_dst_node_temporal_embeddings` uses a time_encoder
        # when constructing node sequences. We'll use this standard one.
        self.time_encoder = TimeEncoder(time_dim=self.config.time_feat_dim, parameter_requires_grad=False)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Computes source and destination node temporal embeddings by leveraging the KAN-MAMMOTE core.
        This method conforms to the interface expected by train_link_prediction.py.

        Args:
            src_node_ids (np.ndarray): Source node IDs for the current batch. Shape (batch_size,).
            dst_node_ids (np.ndarray): Destination node IDs for the current batch. Shape (batch_size,).
            node_interact_times (np.ndarray): Interaction timestamps for the current batch. Shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                src_node_embeddings (torch.Tensor): Temporal embeddings for source nodes. Shape (batch_size, node_feat_dim).
                dst_node_embeddings (torch.Tensor): Temporal embeddings for destination nodes. Shape (batch_size, node_feat_dim).
                time_diff_embedding (torch.Tensor): An aggregated time difference embedding for the pair. Shape (batch_size, node_feat_dim).
        """
        # 1. Get historical neighbors and pad sequences for source and destination nodes
        # Each list contains np.ndarrays: [src/dst_neighbor_ids, src/dst_edge_ids, src/dst_neighbor_times]
        # (batch_size, num_neighbors_in_sequence)
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # Pad sequences to a fixed max_input_sequence_length
        # Output: np.ndarrays with shape (batch_size, max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.config.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.config.max_input_sequence_length)

        # 2. Extract features for the padded sequences
        # Output: torch.Tensors with shape (batch_size, max_seq_length, feature_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder)

        # Concatenate all features into a single sequence feature tensor for KAN-MAMMOTE_Core
        # KANMAMMOTE_Core's forward expects (batch_size, sequence_length, input_feature_dim)
        # So, we need to decide what `input_feature_dim` means.
        # Let's align with what Mamba usually takes: a fixed d_model.
        # This implies we combine node_raw_features, edge_raw_features, and time_features
        # and project them to a consistent `input_feature_dim` for the core model.
        # For simplicity, let's directly use the concatenation as the input for now,
        # assuming config.input_feature_dim is node_feat_dim + edge_feat_dim + time_feat_dim.
        # Or, the initial_feature_proj in kan_mammote_core should handle this.
        
        # As per kan_mammote.py: `self.initial_feature_proj = nn.Linear(config.input_feature_dim, config.d_model)`
        # `features` to kan_mammote_core.forward is `(B, L, input_feature_dim)`
        # Here, `input_feature_dim` is node_raw_features.shape[1] (172)
        # So, we should ensure the `features` passed to kan_mammote_core is `(B, L, node_feat_dim)`.
        # For now, let's simplify and just use the node_raw_features for features input,
        # and feed all relevant time info through the timestamp argument and delta_t_embedding.

        # The `features` argument to KANMAMMOTE_Core.forward is typically the node/edge features
        # that are then projected to d_model. Here, we can treat the node raw features of the sequence
        # elements as the `features`.

        # For the `timestamps` input to KANMAMMOTE_Core: It expects (B, L). This should be the padded_nodes_neighbor_times.
        
        # Apply KAN-MAMMOTE_Core for source nodes
        # kan_mammote_core.forward(timestamps=(B,L), features=(B,L,input_feature_dim))
        src_model_output, _, src_delta_t_embedding_sequence = self.kan_mammote_core(
            timestamps=torch.from_numpy(src_padded_nodes_neighbor_times).to(self.config.device),
            features=src_padded_nodes_neighbor_node_raw_features # Assuming raw node features as input_feature_dim
        )
        # Apply KAN-MAMMOTE_Core for destination nodes
        dst_model_output, _, dst_delta_t_embedding_sequence = self.kan_mammote_core(
            timestamps=torch.from_numpy(dst_padded_nodes_neighbor_times).to(self.config.device),
            features=dst_padded_nodes_neighbor_node_raw_features
        )

        # 3. Aggregate KAN-MAMMOTE outputs into final node embeddings
        # model_output shape is (batch_size, seq_len, output_dim_for_task=1) after prediction_head
        # We want the output of the last Mamba block (before prediction_head)
        # The final_sequence_embedding (current_hidden_states after loop) in KANMAMMOTE_Core.forward
        # is (B, L, d_model). We take the last element for the "current" node's representation.
        src_node_embeddings = self.node_embedding_proj(src_model_output.squeeze(dim=-1)[:, -1, :]) # Take last element and project from d_model to node_feat_dim
        dst_node_embeddings = self.node_embedding_proj(dst_model_output.squeeze(dim=-1)[:, -1, :]) # Take last element and project from d_model to node_feat_dim
        
        # 4. Aggregate delta_t_embedding for the pair
        # delta_t_embedding_sequence shape is (batch_size, seq_len, D_time)
        # Aggregate across sequence length (L) for a single time_diff_embedding per batch instance
        # Taking the mean is a common approach.
        aggregated_delta_t_embedding_src = torch.mean(src_delta_t_embedding_sequence, dim=1) # (batch_size, D_time)
        aggregated_delta_t_embedding_dst = torch.mean(dst_delta_t_embedding_sequence, dim=1) # (batch_size, D_time)
        
        # For the "time_diff_embedding" for the pair, we can combine src and dst's aggregated delta_t.
        # A simple approach is to sum or concatenate and project. Given DyGMamba's output, it implies a single
        # time difference embedding for the pair. The original KAN-MAMMOTE concept calculates delta_t_embedding
        # from tk - tk-1 for each node's sequence. Here, we can average them or take the difference of averages.
        # Let's take the sum of the average delta_t for source and destination, and then project.
        combined_delta_t_embedding = aggregated_delta_t_embedding_src + aggregated_delta_t_embedding_dst
        time_diff_embedding = self.delta_t_embedding_proj(combined_delta_t_embedding) # (batch_size, node_feat_dim)

        return src_node_embeddings, dst_node_embeddings, time_diff_embedding

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int):
        """
        Pads sequences of historical interactions for nodes. Copied and adapted from DyGMamba.py.
        This method handles left-padding to a fixed `max_input_sequence_length`.
        The "current" interaction is placed at the very end of the sequence.

        Args:
            node_ids (np.ndarray): Source/destination node IDs for the current batch. Shape (batch_size,).
            node_interact_times (np.ndarray): Interaction timestamps for the current batch. Shape (batch_size,).
            nodes_neighbor_ids_list (list): List of np.ndarrays, each containing neighbor IDs for a node.
            nodes_edge_ids_list (list): List of np.ndarrays, each containing edge IDs for a node.
            nodes_neighbor_times_list (list): List of np.ndarrays, each containing neighbor interaction timestamps for a node.
            max_input_sequence_length (int): Maximal length of the input sequence for each node.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                padded_nodes_neighbor_ids (np.ndarray): Padded neighbor IDs. Shape (batch_size, max_seq_length).
                padded_nodes_edge_ids (np.ndarray): Padded edge IDs. Shape (batch_size, max_seq_length).
                padded_nodes_neighbor_times (np.ndarray): Padded neighbor timestamps. Shape (batch_size, max_seq_length).
        """
        # max_seq_length will include the target node itself
        # This is `self.config.max_input_sequence_length` from the config
        max_seq_length = max_input_sequence_length
        # DyGMamba uses `max_input_sequence_length` as the total length of the padded sequence,
        # which means `max_input_sequence_length - 1` historical neighbors + 1 current node.
        # If the number of historical neighbors is more than `max_input_sequence_length - 1`, it cuts them.
        
        # We need to account for patch_size in the final max_seq_length to ensure it's divisible.
        # But for padding the raw sequences, let's use what DyGMamba does first, then patch later.
        
        # First cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1
        for idx in range(len(nodes_neighbor_ids_list)):
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length - 1 interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]

        # Max sequence length for padding, including the current node itself
        # If max_input_sequence_length is from config, it should be the final desired length.
        # We adjust if `patch_size` is not 1 and `max_input_sequence_length` isn't divisible.
        final_seq_length_after_padding = max_input_sequence_length
        if final_seq_length_after_padding % self.config.patch_size != 0:
            final_seq_length_after_padding += (self.config.patch_size - final_seq_length_after_padding % self.config.patch_size)
        
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), final_seq_length_after_padding)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), final_seq_length_after_padding)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), final_seq_length_after_padding)).astype(np.float32)

        for idx in range(len(node_ids)):
            # Place current node at the very end of the sequence
            padded_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            padded_nodes_edge_ids[idx, -1] = 0 # No edge ID for self-loop
            padded_nodes_neighbor_times[idx, -1] = node_interact_times[idx]

            # Left-pad historical neighbors
            num_historical = len(nodes_neighbor_ids_list[idx])
            if num_historical > 0:
                # Fill from `-(num_historical + 1)` up to `-1` (excluding the last element which is the current node)
                padded_nodes_neighbor_ids[idx, -(num_historical + 1):-1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -(num_historical + 1):-1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -(num_historical + 1):-1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        Extracts node, edge, and time features for the padded sequences. Copied from DyGMamba.py.
        
        Args:
            node_interact_times (np.ndarray): Interaction timestamps for the current batch. Shape (batch_size,).
            padded_nodes_neighbor_ids (np.ndarray): Padded neighbor IDs. Shape (batch_size, max_seq_length).
            padded_nodes_edge_ids (np.ndarray): Padded edge IDs. Shape (batch_size, max_seq_length).
            padded_nodes_neighbor_times (np.ndarray): Padded neighbor timestamps. Shape (batch_size, max_seq_length).
            time_encoder (TimeEncoder): The time encoder module to generate time features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                padded_nodes_neighbor_node_raw_features (torch.Tensor): Raw node features.
                padded_nodes_edge_raw_features (torch.Tensor): Raw edge features.
                padded_nodes_neighbor_time_features (torch.Tensor): Time features.
        """
        device = self.config.device
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        
        # Calculate time differences for the time encoder
        # node_interact_times[:, np.newaxis] makes it (batch_size, 1) to broadcast
        # padded_nodes_neighbor_times is (batch_size, max_seq_length)
        # Resulting delta_times is (batch_size, max_seq_length)
        delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(device)
        
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=delta_times)

        # Set the time features to all zeros for the padded timestamp (where neighbor_id is 0)
        # This masks out contributions from non-existent neighbors in padded spots
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    # The `get_patches` method is not directly used by KAN-MAMMOTE_Core's forward
    # because KAN-MAMMOTE_Core expects the full sequence. It's more relevant for DyGFormer/DyGMamba's internal Transformer/Mamba blocks.
    # However, for consistency with the DyGMamba structure that these helper functions come from,
    # and if future modifications might require it, we can include it, but it won't be called directly by `compute_src_dst_node_temporal_embeddings`.
    # Let's omit it for now to keep the code focused, but remember it comes from DyGMamba.py.

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Sets the neighbor sampler and resets its random state if applicable.
        This method is called by the training/evaluation scripts.
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()