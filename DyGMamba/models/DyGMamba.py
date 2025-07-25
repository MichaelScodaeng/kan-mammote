import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the new TimeEncoder wrapper from modules
from DyGMamba.models.modules import TimeEncoder 
from DyGMamba.utils.utils import NeighborSampler
from mamba_ssm import Mamba


class DyGMamba(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1,gamma: float = 0.5, max_input_sequence_length: int = 512, max_interaction_times: int = 10, device: str = 'cpu',
                 time_encoder_type: str = 'FixedSinusoidal', time_encoder_config: dict = None): # ADDED time_encoder_type and time_encoder_config
        """
        DyGMamba model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param gamma: float, gamma
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param max_interaction_times: int, maximal interactions for src and dst to consider
        :param device: str, device
        :param time_encoder_type: str, type of time encoder to use (e.g., 'KANMAMMOTE', 'LeTE', 'SPE', 'LPE', 'NoTime', 'FixedSinusoidal')
        :param time_encoder_config: dict, configuration dictionary for the time encoder
        """
        super(DyGMamba, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.gamma = gamma
        self.max_input_sequence_length = max_input_sequence_length
        self.max_interaction_times = max_interaction_times
        self.device = device

        # Instantiate the new TimeEncoder wrapper
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, time_encoder_type=time_encoder_type, time_encoder_config=time_encoder_config) # UPDATED

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NIFEncoder(nif_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })


        self.num_channels = 4
        feature_expansion_size = 2

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, out_features=self.node_feat_dim, bias=True)
        self.output_layer_t_diff = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                             out_features=self.node_feat_dim, bias=True)

        self.mamba = nn.ModuleList([
            Mamba(d_model=self.num_channels * self.channel_embedding_dim // feature_expansion_size,  # Model dimension d_model
                  d_state=16,  # SSM state expansion factor
                  d_conv=4,  # Local convolution width
                  expand=1,  # Block expansion factor
                  )
            for _ in range(self.num_layers)
        ])

        self.mamba_t_diff = nn.ModuleList([
            Mamba(d_model=int(self.gamma*self.channel_embedding_dim),  # Model dimension d_model
                  d_state=16,  # SSM state expansion factor
                  d_conv=4,  # Local convolution width
                  expand=1,  # Block expansion factor
                  )
            for _ in range(self.num_layers)
        ])

        self.projection_layer_t_diff = nn.Linear(in_features=self.time_feat_dim, out_features=int(self.gamma*self.channel_embedding_dim), bias=True)
        self.projection_layer_t_diff_up = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                                    out_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, bias=True)


        self.weightagg = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, 1)

        self.reduce_layer = nn.Linear(self.num_channels * self.channel_embedding_dim, self.num_channels * self.channel_embedding_dim // feature_expansion_size)

        self.channel_norm = nn.LayerNorm(self.num_channels * self.channel_embedding_dim // feature_expansion_size)
        self.channel_feedforward = FeedForwardNet(input_dim=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                                                  dim_expansion_factor=4,
                                                  dropout=self.dropout)

        self.neighbor_selection_layer = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, self.num_channels * self.channel_embedding_dim // feature_expansion_size)


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, ) (Absolute current interaction time)
        :return:
            src_node_embeddings (Tensor): Temporal embeddings for source nodes. Shape (batch_size, node_feat_dim).
            dst_node_embeddings (Tensor): Temporal embeddings for destination nodes. Shape (batch_size, node_feat_dim).
            time_diff_embedding (Tensor): A time difference embedding for the pair. Shape (batch_size, node_feat_dim).
            time_encoder_reg_losses (dict): Dictionary of regularization losses from the time encoder.
        """
        total_time_encoder_reg_losses = {}
        device = self.device

        # get the first-hop neighbors of source and destination nodes
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # time_modeling for the specific (src,dst) pair interactions
        padded_time_diff_emb, reg_losses_time_modeling = self.time_modeling(
            src_node_ids, dst_node_ids, node_interact_times,
            src_nodes_neighbor_ids_list, src_nodes_neighbor_times_list
        )
        # Accumulate regularization losses
        for loss_name, loss_value in reg_losses_time_modeling.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


        # pad the sequences of first-hop neighbors for source and destination nodes
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, src_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_reg_losses_get_feat = \
            self.get_features(node_interact_times=node_interact_times, # Absolute current interaction time
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times) # Absolute neighbor times
        # Accumulate regularization losses
        for loss_name, loss_value in src_reg_losses_get_feat.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_reg_losses_get_feat = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times)
        # Accumulate regularization losses
        for loss_name, loss_value in dst_reg_losses_get_feat.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


        # get the patches for source and destination nodes
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)


        # align the patch encoding dimension
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)


        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        src_patches_data = [src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features,
                            src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features]
        src_patches_data = torch.stack(src_patches_data, dim=2)
        src_patches_data = src_patches_data.reshape(batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim)

        dst_patches_data = [dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features,
                            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features]
        dst_patches_data = torch.stack(dst_patches_data, dim=2)
        dst_patches_data = dst_patches_data.reshape(batch_size, dst_num_patches,
                                                self.num_channels * self.channel_embedding_dim)

        # reduce to channel embsize
        src_patches_data = self.reduce_layer(src_patches_data)
        dst_patches_data = self.reduce_layer(dst_patches_data)


        for mamba in self.mamba:
            src_patches_data = mamba(src_patches_data) + src_patches_data
            dst_patches_data = mamba(dst_patches_data) + dst_patches_data
            src_patches_data = self.channel_norm(src_patches_data)
            dst_patches_data = self.channel_norm(dst_patches_data)
            src_patches_data = self.channel_feedforward(src_patches_data) + src_patches_data
            dst_patches_data = self.channel_feedforward(dst_patches_data) + dst_patches_data

        padded_time_diff_emb = self.projection_layer_t_diff(padded_time_diff_emb)
        for mamba_t in self.mamba_t_diff:
            padded_time_diff_emb = mamba_t(padded_time_diff_emb) + padded_time_diff_emb

        src_weight = self.weightagg(src_patches_data).transpose(1, 2)
        dst_weight = self.weightagg(dst_patches_data).transpose(1, 2)

        src_patches_data_ = src_weight.matmul(src_patches_data).squeeze(dim=1)
        dst_patches_data_ = dst_weight.matmul(dst_patches_data).squeeze(dim=1)

        time_diff_emb = torch.mean(padded_time_diff_emb, dim=1)
        time_diff_emb_ = self.projection_layer_t_diff_up(time_diff_emb)

        # Tensor, shape (batch_size, 1, channel_embedding_dim)
        src_selection_param = (self.neighbor_selection_layer(dst_patches_data_) * time_diff_emb_).unsqueeze(1)
        dst_selection_param = (self.neighbor_selection_layer(src_patches_data_) * time_diff_emb_).unsqueeze(1)


        src_patches_data = torch.sum(src_patches_data * torch.nn.functional.softmax(torch.sum(src_selection_param * src_patches_data, dim=2), dim=1).unsqueeze(2), dim=1)
        dst_patches_data = torch.sum(dst_patches_data * torch.nn.functional.softmax(torch.sum(dst_selection_param * dst_patches_data, dim=2), dim=1).unsqueeze(2), dim=1)


        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        time_diff_emb = self.output_layer_t_diff(time_diff_emb)
        return src_node_embeddings, dst_node_embeddings, time_diff_emb, total_time_encoder_reg_losses # ADDED regularization losses
    
    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_input_sequence_length: int):
        """
        pad the sequences for nodes in node_ids.
        This method places the current node at the *last* index of the sequence,
        and left-pads historical neighbors.
        """
        max_seq_length = max_input_sequence_length

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
                # Fill from `-(num_historical + 1)` up to `-1` (excluding the last element which is the current node)
                padded_nodes_neighbor_ids[idx, -(num_historical + 1):-1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -(num_historical + 1):-1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -(num_historical + 1):-1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def find_previous_interaction(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_node_interact_times: np.ndarray, dst_node_interact_times: np.ndarray,
                                  src_nodes_neighbor_ids_list: list, dst_nodes_neighbor_ids_list: list, src_nodes_edge_ids_list: list, dst_nodes_edge_ids_list: list,
                                  src_nodes_neighbor_times_list: list, dst_nodes_neighbor_times_list: list):
        # This function does not seem to be called anywhere in the original DyGMamba, so it remains unchanged.
        # If it were called, its time encoder usage would need similar updates.
        src_latest_time_interaction, dst_latest_time_interaction = [], []
        gamma = 100
        shrink_ratio = 1e8
        shrink_coeff = torch.tensor(1/(gamma * shrink_ratio), device=self.device)

        for idx in range(len(src_node_ids)):

            find_interact = np.where(src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx], src_nodes_neighbor_ids_list[idx], 0)

            find_interact_index = np.nonzero(find_interact)

            if find_interact_index[0].shape[0] == 0: # previous interaction not found
                src_latest_time_interaction.append(0.0)
            else:

                src_latest_time_interaction.append(src_node_interact_times[idx].item() - src_nodes_neighbor_times_list[idx][find_interact_index[0][-1]].item())

        pair_latest_time_interaction = torch.exp(-torch.from_numpy(np.array(src_latest_time_interaction)).to(self.device) * shrink_coeff)

        return pair_latest_time_interaction

    def time_modeling(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_node_interact_times: np.ndarray,
                      src_nodes_neighbor_ids_list: list, src_nodes_neighbor_times_list: list): # REMOVED time_encoder from arguments
        """
        Generates time features for interactions between source and destination nodes based on their shared history.
        """
        # Initialize losses for this computation path
        current_time_encoder_reg_losses = {}
        device = self.device

        max_interaction_times = self.max_interaction_times

        # padded_time will store time differences for historical interactions of the specific (src, dst) pair.
        padded_time = np.ones((len(src_node_ids), max_interaction_times)).astype(np.float32) * 1e10 # Changed to float32 for consistency

        for idx in range(len(src_node_ids)):
            # Find common interactions in source's neighbor history where the neighbor is the current destination
            find_interact = np.where(src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx], src_nodes_neighbor_ids_list[idx], 0)
            find_interact_index = np.nonzero(find_interact) # Indices where match occurs

            if find_interact_index[0].shape[0] == 0: # previous interaction not found for this pair
                continue
            else:
                # Get unique timestamps of these shared interactions
                unique_ts = np.unique(src_nodes_neighbor_times_list[idx][find_interact_index[0]])
                # Create a sequence of times starting from 0, then unique historical times, then current time
                find_idx_back = np.concatenate((unique_ts, [src_node_interact_times[idx].item()]))
                find_idx_front = np.concatenate(([0.0], unique_ts))
                
                # Calculate time differences for this sequence
                time_diff = find_idx_back - find_idx_front # This array contains delta_t values

                # Pad these time differences to `max_interaction_times`
                if time_diff.shape[0] - 1 < max_interaction_times:
                    # If few interactions, left-pad with 1e10 (or 0.0 depending on desired semantics)
                    # using values from index 1 to end (excluding initial 0.0)
                    padded_time[idx][-time_diff.shape[0]+1:] = time_diff[1:]
                else:
                    # If many interactions, take the most recent `max_interaction_times` differences
                    padded_time[idx][:] = time_diff[-max_interaction_times:]

        # Call the new TimeEncoder wrapper with the time differences.
        # Here, `timestamps` (input to original TimeEncoder) is `delta_t`.
        # So we pass `delta_t` as `current_times` and a zero tensor as `neighbor_times`
        # for our new unified TimeEncoder interface.
        padded_time_diff_emb, reg_losses_time_modeling_call = self.time_encoder(
            current_times=torch.from_numpy(padded_time).float().to(device),
            neighbor_times=torch.zeros_like(torch.from_numpy(padded_time)).float().to(device) # Reference point is zero
        )
        # Accumulate regularization losses
        for loss_name, loss_value in reg_losses_time_modeling_call.items():
            current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


        return padded_time_diff_emb, current_time_encoder_reg_losses # Return output and accumulated losses

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray): # REMOVED time_encoder from arguments
        """
        get node, edge and time features
        """
        # Initialize losses for this computation path
        current_time_encoder_reg_losses = {}
        device = self.device

        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        
        # Calls the new TimeEncoder wrapper, passing absolute times
        # current_times: absolute time of the main node repeated across the sequence elements
        current_times_for_encoder = torch.from_numpy(node_interact_times[:, np.newaxis].repeat(padded_nodes_neighbor_ids.shape[1], axis=1)).float().to(device)
        # neighbor_times: absolute times of the sequence elements themselves
        neighbor_times_for_encoder = torch.from_numpy(padded_nodes_neighbor_times).float().to(device)

        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        nodes_neighbor_time_features, reg_losses_time_feat = self.time_encoder(
            current_times=current_times_for_encoder,
            neighbor_times=neighbor_times_for_encoder
        )
        # Accumulate regularization losses
        for loss_name, loss_value in reg_losses_time_feat.items():
            current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, nodes_neighbor_time_features, current_time_encoder_reg_losses

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size


        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):

        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):

        # two lists to store the appearances of source and destination nodes
        src_nodes_appearances, dst_nodes_appearances = [], []
        # src_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            # Calculate unique keys and counts for source and destination
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True,
                                                                         return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True,
                                                                         return_counts=True)

            # Create mappings from node IDs to their counts
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Adjust counts specifically for the cases where src_node_id appears in dst's neighbors and vice versa
            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

            # Calculate appearances in each other's lists
            src_node_neighbor_counts_in_dst = torch.tensor(
                [dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor(
                [src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            # Stack counts to get a two-column tensor for each node list
            src_nodes_appearances.append(torch.stack(
                [torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device),
                 src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src,
                                                      torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(
                                                          self.device)], dim=1))

        # Stack to form batch tensors
        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray,
                dst_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_nodes_neighbor_ids and dst_nodes_neighbor_ids
        :param src_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,
                                                                                    dst_node_ids=dst_node_ids,
                                                                                    src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                                                    dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        # Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        # Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)

        src_nodes_nif_features = (src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = (dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        return src_nodes_nif_features, dst_nodes_nif_features


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)
