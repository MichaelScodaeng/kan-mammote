import numpy as np
import torch
import torch.nn as nn

# Import the new TimeEncoder wrapper from modules
from DyGMamba.models.modules import TimeEncoder, TransformerEncoder
from DyGMamba.utils.utils import NeighborSampler

class TCL(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, num_depths: int = 20, dropout: float = 0.1, device: str = 'cpu',
                 time_encoder_type: str = 'FixedSinusoidal', time_encoder_config: dict = None): # ADDED time_encoder_type and time_encoder_config
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param num_depths: int, number of depths, identical to the number of sampled neighbors plus 1 (involving the target node)
        :param dropout: float, dropout rate
        :param device: str, device
        :param time_encoder_type: str, type of time encoder to use (e.g., 'KANMAMMOTE', 'LeTE', 'SPE', 'LPE', 'NoTime', 'FixedSinusoidal')
        :param time_encoder_config: dict, configuration dictionary for the time encoder
        """
        super(TCL, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_depths = num_depths
        self.dropout = dropout
        self.device = device

        # Instantiate the new TimeEncoder wrapper
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, time_encoder_type=time_encoder_type, time_encoder_config=time_encoder_config) # UPDATED
        
        self.depth_embedding = nn.Embedding(num_embeddings=num_depths, embedding_dim=self.node_feat_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.node_feat_dim, bias=True),
            # Original TCL used `time` as a key in `projection_layer` but then passed `node_time_features` directly
            # to attention. Let's ensure this mapping is also used.
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.node_feat_dim, bias=True)
        })

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.node_feat_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, ) (Absolute current interaction time)
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
            src_node_embeddings (Tensor): Temporal embeddings for source nodes. Shape (batch_size, node_feat_dim).
            dst_node_embeddings (Tensor): Temporal embeddings for destination nodes. Shape (batch_size, node_feat_dim).
            time_encoder_reg_losses (dict): Dictionary of regularization losses from the time encoder.
        """
        total_time_encoder_reg_losses = {}

        # get temporal neighbors of source nodes, including neighbor ids, edge ids and time information
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # get temporal neighbors of destination nodes, including neighbor ids, edge ids and time information
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1) (Absolute time for each element in the sequence)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_node_ids = np.concatenate((dst_node_ids[:, np.newaxis], dst_neighbor_node_ids), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_edge_ids = np.concatenate((np.zeros((len(dst_node_ids), 1)).astype(np.longlong), dst_neighbor_edge_ids), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], dst_neighbor_times), axis=1)

        # pad the features of the sequence of source and destination nodes
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # src_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_depth_features, src_reg_losses_get_feat = \
            self.get_features(node_interact_times=node_interact_times, # Absolute time of the current query node
                              nodes_neighbor_ids=src_neighbor_node_ids,
                              nodes_edge_ids=src_neighbor_edge_ids,
                              nodes_neighbor_times=src_neighbor_times) # Absolute times of all nodes in sequence
        # Update total_time_encoder_reg_losses
        for loss_name, loss_value in src_reg_losses_get_feat.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


        # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # dst_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # dst_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features, dst_reg_losses_get_feat = \
            self.get_features(node_interact_times=node_interact_times,
                              nodes_neighbor_ids=dst_neighbor_node_ids,
                              nodes_edge_ids=dst_neighbor_edge_ids,
                              nodes_neighbor_times=dst_neighbor_times)
        # Update total_time_encoder_reg_losses
        for loss_name, loss_value in dst_reg_losses_get_feat.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_node_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features + src_nodes_neighbor_depth_features
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_node_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features + dst_nodes_neighbor_depth_features

        for transformer in self.transformers:
            # self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_features = transformer(inputs_query=src_node_features, inputs_key=src_node_features,
                                            inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_features = transformer(inputs_query=dst_node_features, inputs_key=dst_node_features,
                                            inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # cross-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_embeddings = transformer(inputs_query=src_node_features, inputs_key=dst_node_features,
                                              inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_embeddings = transformer(inputs_query=dst_node_features, inputs_key=src_node_features,
                                              inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

            src_node_features, dst_node_features = src_node_embeddings, dst_node_embeddings

        # retrieve the embedding of the corresponding target node, which is at the first position of the sequence
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_node_embeddings[:, 0, :])
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_node_embeddings[:, 0, :])

        return src_node_embeddings, dst_node_embeddings, total_time_encoder_reg_losses # ADDED regularization losses

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray): # REMOVED time_encoder as an argument, use self.time_encoder
        """
        get node, edge, time and depth features
        :param node_interact_times: ndarray, shape (batch_size, ) (Absolute time of the query node)
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_edge_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_neighbor_times: ndarray, shape (batch_size, num_neighbors + 1) (Absolute times of sequence elements)
        :return: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]
            node_raw_features, edge_raw_features, time_features, depth_features, regularization_losses
        """
        # Initialize losses for this helper call
        current_time_encoder_reg_losses = {}
        device = self.device

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        
        # Calls the new TimeEncoder wrapper, passing absolute times
        # current_times: absolute time of the main node repeated across the sequence elements
        current_times_for_encoder = torch.from_numpy(node_interact_times[:, np.newaxis].repeat(nodes_neighbor_ids.shape[1], axis=1)).float().to(device)
        # neighbor_times: absolute times of the sequence elements themselves
        neighbor_times_for_encoder = torch.from_numpy(nodes_neighbor_times).float().to(device)

        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        nodes_neighbor_time_features, reg_losses_time_feat = self.time_encoder(
            current_times=current_times_for_encoder,
            neighbor_times=neighbor_times_for_encoder
        )
        # Update losses
        for loss_name, loss_value in reg_losses_time_feat.items():
            current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        assert nodes_neighbor_ids.shape[1] == self.depth_embedding.weight.shape[0]
        # Tensor, shape (num_neighbors + 1, node_feat_dim)
        nodes_neighbor_depth_features = self.depth_embedding(torch.tensor(range(nodes_neighbor_ids.shape[1])).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_depth_features, current_time_encoder_reg_losses

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