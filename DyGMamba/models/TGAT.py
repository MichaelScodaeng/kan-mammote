import numpy as np
import torch
import torch.nn as nn

# Import the new TimeEncoder wrapper from modules
from DyGMamba.models.modules import TimeEncoder, MergeLayer, MultiHeadAttention 
# No longer need TimeEncoder from modules directly, as it's wrapped.
from DyGMamba.utils.utils import NeighborSampler

class TGAT(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu',
                 time_encoder_type: str = 'FixedSinusoidal', time_encoder_config: dict = None): # ADDED time_encoder_type and time_encoder_config
        """
        TGAT model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        :param time_encoder_type: str, type of time encoder to use (e.g., 'KANMAMMOTE', 'LeTE', 'SPE', 'LPE', 'NoTime', 'FixedSinusoidal')
        :param time_encoder_config: dict, configuration dictionary for the time encoder
        """
        super(TGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        # Instantiate the new TimeEncoder wrapper
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, time_encoder_type=time_encoder_type, time_encoder_config=time_encoder_config) # UPDATED

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        # follow the TGAT paper, use merge layer to combine the attention results and node original feature
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

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
        # Collect regularization losses from time encoder (if any)
        total_time_encoder_reg_losses = {}

        # The initial `node_interact_times` is the absolute time of the current edge.
        # When calling `compute_node_temporal_embeddings`, this `node_interact_times`
        # acts as the `current_times` for the target node at each layer.
        
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings, src_reg_losses = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        # Update total_time_encoder_reg_losses
        for loss_name, loss_value in src_reg_losses.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings, dst_reg_losses = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        # Update total_time_encoder_reg_losses
        for loss_name, loss_value in dst_reg_losses.items():
            total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        return src_node_embeddings, dst_node_embeddings, total_time_encoder_reg_losses # ADDED regularization losses

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray, # node_interact_times here is the 'current_times' for this node at this layer/hop
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times (absolute time of node_ids)
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        :return: Tuple[torch.Tensor, dict]
            torch.Tensor: Node embeddings
            dict: Regularization losses from time encoder
        """
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        # Initialize losses for this computation path
        current_time_encoder_reg_losses = {}

        # The query (source) node's own time feature. Here, current_times = node_interact_times, neighbor_times = node_interact_times (delta_t = 0)
        # Shape: (batch_size, 1, time_feat_dim)
        node_time_features, reg_losses_node = self.time_encoder(
            current_times=torch.from_numpy(node_interact_times).unsqueeze(dim=1).to(device),
            neighbor_times=torch.from_numpy(node_interact_times).unsqueeze(dim=1).to(device) # t_k - t_k = 0
        )
        # Update losses
        for loss_name, loss_value in reg_losses_node.items():
            current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # Tensor, shape (batch_size, node_feat_dim)
        node_raw_features = self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_raw_features, current_time_encoder_reg_losses # Return raw features at layer 0

        else:
            # get source node representations by aggregating embeddings from the previous (current_layer_num - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features, reg_losses_conv_node = self.compute_node_temporal_embeddings(node_ids=node_ids,
                                                                       node_interact_times=node_interact_times, # This is the current_times for the recursive call
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)
            # Update losses
            for loss_name, loss_value in reg_losses_conv_node.items():
                current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_times, ndarray, shape (batch_size, num_neighbors) (Absolute interaction times of neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features, reg_losses_conv_neighbor = self.compute_node_temporal_embeddings(node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(), # This is the current_times for the neighbor's recursive call
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)
            # Update losses
            for loss_name, loss_value in reg_losses_conv_neighbor.items():
                current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


            # Compute time features for neighbors using the new interface:
            # current_times = node_interact_times repeated for each neighbor in the sequence
            # neighbor_times = neighbor_times (absolute times of neighbors)
            # Shape: (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features, reg_losses_neighbor_time_feat = self.time_encoder(
                current_times=torch.from_numpy(node_interact_times[:, np.newaxis].repeat(num_neighbors, axis=1)).float().to(device),
                neighbor_times=torch.from_numpy(neighbor_times).float().to(device)
            )
            # Update losses
            for loss_name, loss_value in reg_losses_neighbor_time_feat.items():
                current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features, # This is time features for the query node
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features, # This is time features for the neighbor nodes
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids) # Pass neighbor_node_ids as numpy for mask

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_raw_features)

            return output, current_time_encoder_reg_losses # Return output and accumulated losses

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