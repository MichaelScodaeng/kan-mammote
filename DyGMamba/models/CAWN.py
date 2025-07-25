import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from DyGMamba.models.modules import TimeEncoder, TransformerEncoder
from DyGMamba.utils.utils import NeighborSampler


import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Import the new TimeEncoder wrapper from modules
from DyGMamba.models.modules import TimeEncoder, TransformerEncoder 


class CAWN(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, position_feat_dim: int, walk_length: int = 2, num_walk_heads: int = 8, dropout: float = 0.1, device: str = 'cpu',
                 time_encoder_type: str = 'FixedSinusoidal', time_encoder_config: dict = None): # ADDED time_encoder_type and time_encoder_config
        """
        Causal anonymous walks network.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param position_feat_dim: int, dimension of position features (encodings)
        :param walk_length: int, length of each random walk
        :param num_walk_heads: int, number of attention heads to aggregate random walks
        :param dropout: float, dropout rate
        :param device: str, device
        :param time_encoder_type: str, type of time encoder to use (e.g., 'KANMAMMOTE', 'LeTE', 'SPE', 'LPE', 'NoTime', 'FixedSinusoidal')
        :param time_encoder_config: dict, configuration dictionary for the time encoder
        """
        super(CAWN, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.position_feat_dim = position_feat_dim
        self.walk_length = walk_length
        self.num_walk_heads = num_walk_heads
        self.dropout = dropout
        self.device = device

        # Instantiate the new TimeEncoder wrapper
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, time_encoder_type=time_encoder_type, time_encoder_config=time_encoder_config) # UPDATED

        self.position_encoder = PositionEncoder(position_feat_dim=self.position_feat_dim, walk_length=self.walk_length, device=device)

        self.walk_encoder = WalkEncoder(input_dim=self.node_feat_dim + self.edge_feat_dim + self.time_feat_dim + self.position_feat_dim,
                                        position_feat_dim=self.position_feat_dim, output_dim=self.node_feat_dim, num_walk_heads=self.num_walk_heads, dropout=dropout)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, ) (Absolute current interaction time)
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
            src_node_embeddings (Tensor): Temporal embeddings for source nodes. Shape (batch_size, node_feat_dim).
            dst_node_embeddings (Tensor): Temporal embeddings for destination nodes. Shape (batch_size, node_feat_dim).
            time_encoder_reg_losses (dict): Dictionary of regularization losses from the time encoder.
        """
        # Collect regularization losses from time encoder (if any)
        total_time_encoder_reg_losses = {}

        # get the multi-hop graph for each node
        # tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        src_node_multi_hop_graphs = self.neighbor_sampler.get_multi_hop_neighbors(num_hops=self.walk_length, node_ids=src_node_ids,
                                                                                  node_interact_times=node_interact_times, num_neighbors=num_neighbors)
        # tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        dst_node_multi_hop_graphs = self.neighbor_sampler.get_multi_hop_neighbors(num_hops=self.walk_length, node_ids=dst_node_ids,
                                                                                  node_interact_times=node_interact_times, num_neighbors=num_neighbors)

        # count the appearances appearances of nodes in the multi-hop graphs that are generated by random walks that
        # start from src node in src_node_ids and dst node in dst_node_ids
        self.position_encoder.count_nodes_appearances(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                                                      node_interact_times=node_interact_times,
                                                      src_node_multi_hop_graphs=src_node_multi_hop_graphs,
                                                      dst_node_multi_hop_graphs=dst_node_multi_hop_graphs)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings, src_reg_losses = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    node_multi_hop_graphs=src_node_multi_hop_graphs, num_neighbors=num_neighbors)
        # Update total_time_encoder_reg_losses
        for loss_name, loss_value in src_reg_losses.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings, dst_reg_losses = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    node_multi_hop_graphs=dst_node_multi_hop_graphs, num_neighbors=num_neighbors)
        # Update total_time_encoder_reg_losses
        for loss_name, loss_value in dst_reg_losses.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        return src_node_embeddings, dst_node_embeddings, total_time_encoder_reg_losses # ADDED regularization losses

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray, node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        given node interaction time node_interact_times and node multi-hop graphs node_multi_hop_graphs,
        return the temporal embeddings of nodes
        :param node_interact_times: ndarray, shape (batch_size, ) (Absolute interaction time of current node/query node)
        :param node_multi_hop_graphs: tuple of three ndarrays, each array with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        :return: Tuple[torch.Tensor, dict]
            torch.Tensor: Node embeddings
            dict: Regularization losses from time encoder
        """
        # Initialize losses for this computation path
        current_time_encoder_reg_losses = {}
        device = self.device

        # three ndarrays, each array with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = \
            self.convert_format_from_tree_to_array(node_ids=node_ids, node_interact_times=node_interact_times,
                                                   node_multi_hop_graphs=node_multi_hop_graphs, num_neighbors=num_neighbors)

        # get raw features of nodes in the multi-hop graphs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, node_feat_dim)
        neighbor_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]

        # ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        walks_valid_lengths = (nodes_neighbor_ids != 0).sum(axis=-1)

        # get time features of nodes in the multi-hop graphs using the new interface
        # The current_times for each element in the walks sequence is the start time of that walk.
        # The neighbor_times for each element are the absolute times along that walk.
        # Reshape to (batch_size * num_walks, walk_length + 1) for the time encoder input.
        current_times_for_encoder_reshaped = torch.from_numpy(nodes_neighbor_times[:, :, 0]).float().to(self.device).unsqueeze(dim=-1).repeat(1, 1, self.walk_length + 1)
        current_times_for_encoder_reshaped = current_times_for_encoder_reshaped.view(-1, self.walk_length + 1)

        neighbor_times_for_encoder_reshaped = torch.from_numpy(nodes_neighbor_times).float().to(self.device)
        neighbor_times_for_encoder_reshaped = neighbor_times_for_encoder_reshaped.view(-1, self.walk_length + 1)
        
        # Now pass to self.time_encoder
        # It returns (time_embeddings, reg_losses_dict)
        # time_embeddings will have shape (batch_size * num_walks, walk_length + 1, time_feat_dim)
        neighbor_time_features, reg_losses_time_encoder = self.time_encoder(
            current_times=current_times_for_encoder_reshaped,
            neighbor_times=neighbor_times_for_encoder_reshaped
        )
        # Accumulate regularization losses
        for loss_name, loss_value in reg_losses_time_encoder.items():
            current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # Reshape neighbor_time_features back to (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, time_feat_dim)
        neighbor_time_features = neighbor_time_features.view(
            nodes_neighbor_ids.shape[0], nodes_neighbor_ids.shape[1], nodes_neighbor_ids.shape[2], self.time_feat_dim
        )
        
        # get edge features of nodes in the multi-hop graphs
        # ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        # check that the edge ids of the target node is denoted by zeros
        assert (nodes_edge_ids[:, :, 0] == 0).all()
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]

        # get position features of nodes in the multi-hop graphs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        neighbor_position_features = self.position_encoder(nodes_neighbor_ids=nodes_neighbor_ids)

        # encode the random walks by walk encoder
        # Tensor, shape (batch_size, self.output_dim)
        final_node_embeddings = self.walk_encoder(neighbor_raw_features=neighbor_raw_features, neighbor_time_features=neighbor_time_features,
                                                  edge_features=edge_features, neighbor_position_features=neighbor_position_features,
                                                  walks_valid_lengths=walks_valid_lengths)
        return final_node_embeddings, current_time_encoder_reg_losses # Return output and accumulated losses

    def convert_format_from_tree_to_array(self, node_ids: np.ndarray, node_interact_times: np.ndarray, node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        convert the multi-hop graphs from tree-like data format to aligned array-like format
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = node_multi_hop_graphs

        # add the target node to the list to generate random walks in array-like format
        nodes_neighbor_ids = [node_ids[:, np.newaxis]] + nodes_neighbor_ids
        # follow the CAWN official implementation, the edge ids of the target node is denoted by zeros
        nodes_edge_ids = [np.zeros((len(node_ids), 1)).astype(np.longlong)] + nodes_edge_ids
        nodes_neighbor_times = [node_interact_times[:, np.newaxis]] + nodes_neighbor_times

        array_format_data_list = []
        for tree_format_data in [nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times]:
            # num_last_hop_neighbors equals to num_neighbors ** self.walk_length
            batch_size, num_last_hop_neighbors, walk_length_plus_1, dtype = \
                tree_format_data[0].shape[0], tree_format_data[-1].shape[-1], len(tree_format_data), tree_format_data[0].dtype
            assert batch_size == len(node_ids) and num_last_hop_neighbors == num_neighbors ** self.walk_length and walk_length_plus_1 == self.walk_length + 1
            # record the information of random walks with num_last_hop_neighbors paths, where each path has length walk_length_plus_1 (include the target node)
            # ndarray, shape (batch_size, num_last_hop_neighbors, walk_length_plus_1)
            array_format_data = np.empty((batch_size, num_last_hop_neighbors, walk_length_plus_1), dtype=dtype)
            for hop_idx, hop_data in enumerate(tree_format_data):
                assert (num_last_hop_neighbors % hop_data.shape[-1] == 0)
                # pad the data at each hop to be the same shape with the last hop data (which has the most number of neighbors)
                # repeat the traversed nodes in tree_format_data to get the aligned array-like format
                array_format_data[:, :, hop_idx] = np.repeat(hop_data, repeats=num_last_hop_neighbors // hop_data.shape[-1], axis=1)
            array_format_data_list.append(array_format_data)
        # three ndarrays with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        return array_format_data_list[0], array_format_data_list[1], array_format_data_list[2]

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


class PositionEncoder(nn.Module):
    # ... (PositionEncoder class remains the same) ...
    def __init__(self, position_feat_dim: int, walk_length: int, device: str = 'cpu'):
        super(PositionEncoder, self).__init__()
        self.position_feat_dim = position_feat_dim
        self.walk_length = walk_length
        self.device = device

        self.position_encode_layer = nn.Sequential(nn.Linear(in_features=self.walk_length + 1, out_features=self.position_feat_dim),
                                                   nn.ReLU(),
                                                   nn.Linear(in_features=self.position_feat_dim, out_features=self.position_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                src_node_multi_hop_graphs: tuple, dst_node_multi_hop_graphs: tuple):
        src_nodes_neighbor_ids, _, src_nodes_neighbor_times = src_node_multi_hop_graphs
        dst_nodes_neighbor_ids, _, dst_nodes_neighbor_times = dst_node_multi_hop_graphs

        self.nodes_appearances = {}
        for idx, (src_node_id, dst_node_id, node_interact_time) in enumerate(zip(src_node_ids, dst_node_ids, node_interact_times)):
            src_node_neighbor_ids = [src_nodes_single_hop_neighbor_ids[idx] for src_nodes_single_hop_neighbor_ids in src_nodes_neighbor_ids]
            src_node_neighbor_times = [src_nodes_single_hop_neighbor_times[idx] for src_nodes_single_hop_neighbor_times in src_nodes_neighbor_times]
            dst_node_neighbor_ids = [dst_nodes_single_hop_neighbor_ids[idx] for dst_nodes_single_hop_neighbor_ids in dst_nodes_neighbor_ids]
            dst_node_neighbor_times = [dst_nodes_single_hop_neighbor_times[idx] for dst_nodes_single_hop_neighbor_times in dst_nodes_neighbor_times]

            tmp_nodes_appearances = {}
            src_node_neighbor_ids, src_node_neighbor_times = [[src_node_id]] + src_node_neighbor_ids, [[node_interact_time]] + src_node_neighbor_times
            dst_node_neighbor_ids, dst_node_neighbor_times = [[dst_node_id]] + dst_node_neighbor_ids, [[node_interact_time]] + dst_node_neighbor_times
            for current_hop in range(self.walk_length + 1):
                for src_node_neighbor_id, src_node_neighbor_time, dst_node_neighbor_id, dst_node_neighbor_time in \
                        zip(src_node_neighbor_ids[current_hop], src_node_neighbor_times[current_hop], dst_node_neighbor_ids[current_hop], dst_node_neighbor_times[current_hop]):

                    src_node_key = '-'.join([str(idx), str(src_node_neighbor_id)])
                    dst_node_key = '-'.join([str(idx), str(dst_node_neighbor_id)])

                    if src_node_key not in tmp_nodes_appearances:
                        tmp_nodes_appearances[src_node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
                    if dst_node_key not in tmp_nodes_appearances:
                        tmp_nodes_appearances[dst_node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)

                    num_current_hop_neighbors = len(src_node_neighbor_ids[current_hop])
                    tmp_nodes_appearances[src_node_key][0, current_hop] += 1 / num_current_hop_neighbors
                    tmp_nodes_appearances[dst_node_key][1, current_hop] += 1 / num_current_hop_neighbors
            tmp_nodes_appearances['-'.join([str(idx), str(0)])] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
            self.nodes_appearances.update(tmp_nodes_appearances)

    def forward(self, nodes_neighbor_ids: np.ndarray):
        batch_indices = np.arange(nodes_neighbor_ids.shape[0]).repeat(nodes_neighbor_ids.shape[1] * nodes_neighbor_ids.shape[2]).reshape(nodes_neighbor_ids.shape)
        batch_keys = ['-'.join([str(batch_indices[i][j][k]), str(nodes_neighbor_ids[i][j][k])])
                      for i in range(batch_indices.shape[0]) for j in range(batch_indices.shape[1]) for k in range(batch_indices.shape[2])]
        unique_keys, inverse_indices = np.unique(batch_keys, return_inverse=True)
        unique_node_appearances = np.array([self.nodes_appearances[unique_key] for unique_key in unique_keys])
        node_appearances = unique_node_appearances[inverse_indices, :].reshape(nodes_neighbor_ids.shape[0], nodes_neighbor_ids.shape[1],
                                                                               nodes_neighbor_ids.shape[2], 2, self.walk_length + 1)
        position_features = self.position_encode_layer(torch.Tensor(node_appearances).float().to(self.device))
        position_features = position_features.sum(dim=-2)
        return position_features


class WalkEncoder(nn.Module):

    def __init__(self, input_dim: int, position_feat_dim: int, output_dim: int, num_walk_heads: int, dropout: float = 0.1):
        """
        Walk encoder that first encodes each random walk by BiLSTM and then aggregates all the walks by the self-attention in Transformer
        :param input_dim: int, dimension of the input
        :param position_feat_dim: int, dimension of position features (encodings)
        :param output_dim: int, dimension of the output
        :param num_walk_heads: int, number of attention heads to aggregate random walks
        :param dropout: float, dropout rate
        """
        super(WalkEncoder, self).__init__()
        self.input_dim = input_dim
        self.position_feat_dim = position_feat_dim
        # follow the CAWN official implementation, take half of the model dimension to save computation cost for attention
        self.attention_dim = self.input_dim // 2
        self.output_dim = output_dim
        self.num_walk_heads = num_walk_heads
        self.dropout = dropout
        # make sure that the attention dimension can be divided by number of walk heads
        if self.attention_dim % self.num_walk_heads != 0:
            self.attention_dim += (self.num_walk_heads - self.attention_dim % self.num_walk_heads)

        # BiLSTM Encoders, encode the node features along each random walk
        self.feature_encoder = BiLSTMEncoder(input_dim=self.input_dim, hidden_dim=self.input_dim)
        # encode position features along each temporal walk
        self.position_encoder = BiLSTMEncoder(input_dim=self.position_feat_dim, hidden_dim=self.position_feat_dim)

        self.transformer_encoder = TransformerEncoder(attention_dim=self.attention_dim, num_heads=self.num_walk_heads, dropout=self.dropout)

        # due to the usage of BiLSTM, self.feature_encoder.model_dim may not be equal to self.input_dim, since self.input_dim may not be an even number
        # also, self.position_encoder.model_dim may not be equal to self.input_dim, since self.input_dim may not be an even number
        # projection layers for 1) combination of outputs from self.feature_encoder and self.position_encoder; and 2) final output
        self.projection_layers = nn.ModuleList([
            nn.Linear(in_features=self.feature_encoder.model_dim + self.position_encoder.model_dim, out_features=self.attention_dim),
            nn.Linear(in_features=self.attention_dim, out_features=self.output_dim)
        ])

    def forward(self, neighbor_raw_features: torch.Tensor, neighbor_time_features: torch.Tensor, edge_features: torch.Tensor,
                neighbor_position_features: torch.Tensor, walks_valid_lengths: np.ndarray):
        """
        first encode each random walk by BiLSTM and then aggregate all the walks by the self-attention in Transformer
        :param neighbor_raw_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, node_feat_dim)
        :param neighbor_time_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, time_feat_dim)
        :param edge_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, edge_feat_dim)
        :param neighbor_position_features: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        :param walks_valid_lengths: ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        :return:
        """
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, node_feat_dim + time_feat_dim + edge_feat_dim + position_feat_dim)
        combined_features = torch.cat([neighbor_raw_features, neighbor_time_features, edge_features, neighbor_position_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.feature_encoder.model_dim), feed the combined features to BiLSTM
        combined_features = self.feature_encoder(inputs=combined_features, lengths=walks_valid_lengths)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.position_encoder.model_dim), feed the position features to BiLSTM
        neighbor_position_features = self.position_encoder(inputs=neighbor_position_features, lengths=walks_valid_lengths)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.feature_encoder.model_dim + self.position_encoder.model_dim)
        combined_features = torch.cat([combined_features, neighbor_position_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.attention_dim)
        combined_features = self.projection_layers[0](combined_features)
        # Tensor, shape (batch_size, self.attention_dim), feed into Transformer and then perform mean pooling over multiple random walks
        combined_features = self.transformer_encoder(inputs_query=combined_features).mean(dim=-2)
        # Tensor, shape (batch_size, self.output_dim)
        outputs = self.projection_layers[1](combined_features)
        return outputs


class BiLSTMEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        BiLSTM encoder.
        :param input_dim: int, dimension of the input
        :param hidden_dim: int, dimension of the hidden state
        """
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim_one_direction = hidden_dim // 2
        self.model_dim = self.hidden_dim_one_direction * 2
        self.bilstm_encoder = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_one_direction, batch_first=True, bidirectional=True)

    def forward(self, inputs: torch.Tensor, lengths: np.ndarray):
        """
        encode the inputs by BiLSTM encoder based on lengths
        :param inputs: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, input_dim)
        :param lengths: ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        :return:
        """
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length), self.walk_length + 1, input_dim), which corresponds to the LSTM input (batch_size, seq_len, input_dim)
        inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # a PackedSequence object, pack the padded sequence for efficient computation and avoid the errors of computing padded value, set enforce_sorted to False
        inputs = pack_padded_sequence(inputs, lengths.flatten(), batch_first=True, enforce_sorted=False)
        # the outputs of LSTM are output, (h_n, c_n), and we only use the output and do not use hidden states
        encoded_features, _ = self.bilstm_encoder(inputs)
        # encoded_features, Tensor, shape (batch_size * (num_neighbors ** self.walk_length), self.walk_length + 1, self.model_dim), pad the packed sequence
        # seq_lengths, Tensor, shape (batch_size * (num_neighbors ** self.walk_length), )
        encoded_features, seq_lengths = pad_packed_sequence(encoded_features, batch_first=True)
        assert (seq_lengths.numpy() == lengths.flatten()).all()
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length), ), the shifted sequence lengths
        shifted_seq_lengths = seq_lengths + torch.tensor([i * encoded_features.shape[1] for i in range(encoded_features.shape[0])])
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1), self.model_dim)
        encoded_features = encoded_features.reshape(encoded_features.shape[0] * encoded_features.shape[1], encoded_features.shape[2])
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.model_dim), get the encodings of each walk at the last position
        # note that we need to use shifted_seq_lengths - 1 to get the shifted indices
        encoded_features = encoded_features[shifted_seq_lengths - 1].reshape(lengths.shape[0], lengths.shape[1], self.model_dim)

        return encoded_features
