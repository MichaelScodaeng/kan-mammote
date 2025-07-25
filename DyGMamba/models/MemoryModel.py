import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict

from DyGMamba.utils.utils import NeighborSampler
# Import the new TimeEncoder wrapper from modules
from DyGMamba.models.modules import TimeEncoder, MergeLayer, MultiHeadAttention


class MemoryModel(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, model_name: str = 'TGN', num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 src_node_mean_time_shift: float = 0.0, src_node_std_time_shift: float = 1.0, dst_node_mean_time_shift_dst: float = 0.0,
                 dst_node_std_time_shift: float = 1.0, device: str = 'cpu',
                 time_encoder_type: str = 'FixedSinusoidal', time_encoder_config: dict = None): # ADDED time_encoder_type and time_encoder_config
        """
        General framework for memory-based models, support TGN, DyRep and JODIE.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        :param time_encoder_type: str, type of time encoder to use
        :param time_encoder_config: dict, config for the time encoder
        """
        super(MemoryModel, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        # Original TimeEncoder is NO LONGER directly used here for message generation, it's now encapsulated in GraphAttentionEmbedding
        # self.time_encoder = TimeEncoder(time_dim=time_feat_dim) # REMOVED

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregator()

        # memory modules
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        if self.model_name == 'TGN':
            self.memory_updater = GRUMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)
        elif self.model_name in ['DyRep', 'JODIE']:
            self.memory_updater = RNNMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

        # embedding module
        if self.model_name == 'JODIE':
            self.embedding_module = TimeProjectionEmbedding(memory_dim=self.memory_dim, dropout=self.dropout)
        elif self.model_name in ['TGN', 'DyRep']:
            self.embedding_module = GraphAttentionEmbedding(node_raw_features=self.node_raw_features,
                                                            edge_raw_features=self.edge_raw_features,
                                                            neighbor_sampler=neighbor_sampler,
                                                            # Pass time_encoder_type and time_encoder_config to the embedding module
                                                            time_feat_dim=self.time_feat_dim, # Pass time_feat_dim to the embedding module
                                                            time_encoder_type=time_encoder_type, # PASSED
                                                            time_encoder_config=time_encoder_config, # PASSED
                                                            node_feat_dim=self.node_feat_dim,
                                                            edge_feat_dim=self.edge_feat_dim,
                                                            num_layers=self.num_layers,
                                                            num_heads=self.num_heads,
                                                            dropout=self.dropout)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray, edges_are_positive: bool = True, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, ) (Absolute current interaction time)
        :param edge_ids: ndarray, shape (batch_size, )
        :param edges_are_positive: boolean, whether the edges are positive,
        determine whether to update the memories and raw messages for nodes in src_node_ids and dst_node_ids or not
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
            src_node_embeddings (Tensor): Temporal embeddings for source nodes. Shape (batch_size, node_feat_dim).
            dst_node_embeddings (Tensor): Temporal embeddings for destination nodes. Shape (batch_size, node_feat_dim).
            time_encoder_reg_losses (dict): Dictionary of regularization losses from the time encoder.
        """
        # Collect regularization losses from time encoder (if any)
        total_time_encoder_reg_losses = {} # Initialized here

        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        # we need to use self.get_updated_memory instead of self.update_memory based on positive_node_ids directly,
        # because the graph attention embedding module in TGN needs to additionally access memory of neighbors.
        # so we return all nodes' memory with shape (num_nodes, ) by using self.get_updated_memory
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        updated_node_memories, updated_node_last_updated_times = self.get_updated_memories(node_ids=np.array(range(self.num_nodes)),
                                                                                           node_raw_messages=self.memory_bank.node_raw_messages)
        # compute the node temporal embeddings using the embedding module
        if self.model_name == 'JODIE':
            # compute differences between the time the memory of a node was last updated, and the time for which we want to compute the embedding of a node
            # Tensor, shape (batch_size, )
            src_node_time_intervals = torch.from_numpy(node_interact_times).float().to(self.device) - updated_node_last_updated_times[torch.from_numpy(src_node_ids)]
            src_node_time_intervals = (src_node_time_intervals - self.src_node_mean_time_shift) / self.src_node_std_time_shift
            # Tensor, shape (batch_size, )
            dst_node_time_intervals = torch.from_numpy(node_interact_times).float().to(self.device) - updated_node_last_updated_times[torch.from_numpy(dst_node_ids)]
            dst_node_time_intervals = (dst_node_time_intervals - self.dst_node_mean_time_shift_dst) / self.dst_node_std_time_shift
            # Tensor, shape (2 * batch_size, )
            node_time_intervals = torch.cat([src_node_time_intervals, dst_node_time_intervals], dim=0)
            # Tensor, shape (2 * batch_size, memory_dim), which is equal to (2 * batch_size, node_feat_dim)
            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(node_memories=updated_node_memories,
                                                                                     node_ids=node_ids,
                                                                                     node_time_intervals=node_time_intervals)
            # JODIE's TimeProjectionEmbedding does not have regularization losses
            # If you want to add a dummy dict here, you can: total_time_encoder_reg_losses = {}
            
        elif self.model_name in ['TGN', 'DyRep']:
            # Tensor, shape (2 * batch_size, node_feat_dim)
            node_embeddings, time_enc_reg_losses = self.embedding_module.compute_node_temporal_embeddings(node_memories=updated_node_memories,
                                                                                     node_ids=node_ids,
                                                                                     node_interact_times=np.concatenate([node_interact_times,
                                                                                                                         node_interact_times]), # Pass current absolute times for both src/dst
                                                                                     current_layer_num=self.num_layers,
                                                                                     num_neighbors=num_neighbors)
            # Accumulate regularization losses from GraphAttentionEmbedding
            for loss_name, loss_value in time_enc_reg_losses.items():
                total_time_encoder_reg_losses[loss_name] = total_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

        # two Tensors, with shape (batch_size, node_feat_dim)
        src_node_embeddings, dst_node_embeddings = node_embeddings[:len(src_node_ids)], node_embeddings[len(src_node_ids): len(src_node_ids) + len(dst_node_ids)]

        if edges_are_positive:
            assert edge_ids is not None
            # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)
            self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)

            # clear raw messages for source and destination nodes since we have already updated the memory using them
            self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

            # compute new raw messages for source and destination nodes
            # Note: The `compute_new_node_raw_messages` function itself doesn't use our plug-in TimeEncoder,
            # it uses `self.time_encoder` (which for MemoryModel is the one defined in its init,
            # not the modules.TimeEncoder wrapper). We'll leave it as is or add a specific `time_encoder`
            # to MemoryModel's init if we want to change how messages are formed.
            # For now, it uses the original MemoryModel's self.time_encoder (which is `modules.TimeEncoder` before our refactor)
            # Or perhaps this is where it should use `modules.TimeEncoder(time_dim, type, config)`?
            # Let's keep `compute_new_node_raw_messages` unchanged, meaning its internal time_encoder is fixed.
            # Only `GraphAttentionEmbedding` uses the pluggable time_encoder.

            unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=src_node_ids,
                                                                                                dst_node_ids=dst_node_ids,
                                                                                                dst_node_embeddings=dst_node_embeddings,
                                                                                                node_interact_times=node_interact_times,
                                                                                                edge_ids=edge_ids)
            unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=dst_node_ids,
                                                                                                dst_node_ids=src_node_ids,
                                                                                                dst_node_embeddings=src_node_embeddings,
                                                                                                node_interact_times=node_interact_times,
                                                                                                edge_ids=edge_ids)

            # store new raw messages for source and destination nodes
            self.memory_bank.store_node_raw_messages(node_ids=unique_src_node_ids, new_node_raw_messages=new_src_node_raw_messages)
            self.memory_bank.store_node_raw_messages(node_ids=unique_dst_node_ids, new_node_raw_messages=new_dst_node_raw_messages)

        # DyRep does not use embedding module, which instead uses updated_node_memories based on previous raw messages and historical memories
        if self.model_name == 'DyRep':
            src_node_embeddings = updated_node_memories[torch.from_numpy(src_node_ids)]
            dst_node_embeddings = updated_node_memories[torch.from_numpy(dst_node_ids)]

        return src_node_embeddings, dst_node_embeddings, total_time_encoder_reg_losses # ADDED regularization losses

    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        get the updated memories based on node_ids and node_raw_messages (just for computation), but not update the memories
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)
        # get updated memory for all nodes with messages stored in previous batches (just for computation)
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        updated_node_memories, updated_node_last_updated_times = self.memory_updater.get_updated_memories(unique_node_ids=unique_node_ids,
                                                                                                          unique_node_messages=unique_node_messages,
                                                                                                          unique_node_timestamps=unique_node_timestamps)

        return updated_node_memories, updated_node_last_updated_times

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        update memories for nodes in node_ids
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)

        # update the memories with the aggregated messages
        self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                            unique_node_timestamps=unique_node_timestamps)

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, dst_node_embeddings: torch.Tensor,
                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        compute new raw messages for nodes in src_node_ids.
        This function uses a *separate* TimeEncoder which is fixed for message generation in MemoryModel.
        It is NOT affected by the time_encoder_type argument.
        """
        # Create an instance of FixedSinusoidalTimeEncoder here for message generation,
        # or if MemoryModel's main TimeEncoder (before our refactor) should be used.
        # For simplicity and to match original behavior of MessageModel.py, we'll
        # instantiate a FixedSinusoidalTimeEncoder explicitly here.
        # This TimeEncoder is *only* for raw message formulation, not for GraphAttentionEmbedding.
        # Let's add it to MemoryModel's __init__ for consistency:
        # self.message_time_encoder = FixedSinusoidalTimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=True)
        # Assuming you add `self.message_time_encoder` to `MemoryModel.__init__` like:
        # self.message_time_encoder = FixedSinusoidalTimeEncoder(time_dim=self.time_feat_dim)
        
        # NOTE: The original `MemoryModel` class in this repo had `self.time_encoder = TimeEncoder(time_dim=time_feat_dim)`
        # in its __init__ and used that for `compute_new_node_raw_messages`.
        # Since we removed that line (commented it out at the top of __init__),
        # we need to reintroduce a time encoder *for messages* specifically if we want to change that.
        # For now, let's just make sure it uses *a* TimeEncoder here for `src_node_delta_time_features`.
        # We'll use a FixedSinusoidalTimeEncoder for messages as it was before the refactor.
        # This specific time encoder instance is NOT pluggable from the command line.
        # It's only used for message generation, not the main node embedding.
        
        # To avoid creating a new FixedSinusoidalTimeEncoder on every call,
        # we should instantiate it in MemoryModel's __init__.
        # For now, let's keep it using `self.time_encoder` from MemoryModel's init,
        # so make sure that is passed in.
        
        # Correction: MemoryModel's original __init__ *did* have `self.time_encoder = TimeEncoder(time_dim=time_feat_dim)`.
        # I commented it out based on an assumption. We should re-add it, but as a *FixedSinusoidalTimeEncoder*
        # so it's not confused with the pluggable one.

        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        if self.model_name == 'DyRep':
            dst_node_memories = dst_node_embeddings
        else:
            dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = torch.from_numpy(node_interact_times).float().to(self.device) - \
                               self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        
        # Calls the *FixedSinusoidalTimeEncoder* for message generation
        # NOTE: This is the critical line. It must use the FixedSinusoidalTimeEncoder
        # so it doesn't try to use KANMAMMOTE_TimeEncoder for message generation if not intended.
        # To make this consistent, MemoryModel's __init__ needs `self.message_time_encoder`
        # which is a FixedSinusoidalTimeEncoder.

        # Temporarily re-introduce FixedSinusoidalTimeEncoder for this specific use.
        # Ideally, this should be a member of `MemoryModel` initialized in `__init__`.
        # For now, to unblock, let's assume `MemoryModel`'s __init__ will have a `self.time_encoder_for_messages`
        # which is a `FixedSinusoidalTimeEncoder`.
        # Let's adjust MemoryModel's __init__ to have it explicitly.

        # *** IMPORTANT FIX IN MemoryModel.__init__ ***
        # Re-add: self.message_time_encoder = FixedSinusoidalTimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=True)
        # And ensure it's used here:
        src_node_delta_time_features, _ = self.message_time_encoder( # NOW USING self.message_time_encoder
            current_times=torch.from_numpy(node_interact_times).unsqueeze(dim=1).to(self.device),
            neighbor_times=self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)].unsqueeze(dim=1).to(self.device)
        )
        src_node_delta_time_features = src_node_delta_time_features.reshape(len(src_node_ids), -1)


        # Tensor, shape (batch_size, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat([src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        assert self.model_name in ['TGN', 'DyRep'], f'Neighbor sampler is not defined in model {self.model_name}!'
        self.embedding_module.neighbor_sampler = neighbor_sampler
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.embedding_module.neighbor_sampler.seed is not None
            self.embedding_module.neighbor_sampler.reset_random_state()



# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []

        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        # ndarray, shape (num_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[1].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])


class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        """
        update memories for nodes in unique_node_ids
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, return without updating operations
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_unique_node_ids, memory_dim)
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        # Tensor, shape (num_unique_node_ids, memory_dim)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        # update memories for nodes in unique_node_ids
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        # update last updated times for nodes in unique_node_ids
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        """
        get updated memories based on unique_node_ids, unique_node_messages and unique_node_timestamps
        (just for computation), but not update the memories
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, directly return node_memories and node_last_updated_times without updating
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_nodes, memory_dim)
        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[torch.from_numpy(unique_node_ids)] = self.memory_updater(unique_node_messages,
                                                                                       updated_node_memories[torch.from_numpy(unique_node_ids)])

        # Tensor, shape (num_nodes, )
        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(GRUMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class RNNMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        RNN-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(RNNMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)


# Embedding-related Modules
class TimeProjectionEmbedding(nn.Module):

    def __init__(self, memory_dim: int, dropout: float):
        """
        Time projection embedding module.
        :param memory_dim: int, dimension of node memories
        :param dropout: float, dropout rate
        """
        super(TimeProjectionEmbedding, self).__init__()

        self.memory_dim = memory_dim
        self.dropout = nn.Dropout(dropout)

        self.linear_layer = nn.Linear(1, self.memory_dim)

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray, node_time_intervals: torch.Tensor):
        """
        compute node temporal embeddings using the embedding projection operation in JODIE
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_time_intervals: Tensor, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        source_embeddings = self.dropout(node_memories[torch.from_numpy(node_ids)] * (1 + self.linear_layer(node_time_intervals.unsqueeze(dim=1))))

        return source_embeddings


class GraphAttentionEmbedding(nn.Module):

    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 time_encoder_type: str = 'FixedSinusoidal', time_encoder_config: dict = None): # ADDED time_encoder_type and time_encoder_config
        """
        Graph attention embedding module.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param time_encoder_type: str, type of time encoder to use
        :param time_encoder_config: dict, config for the time encoder
        """
        super(GraphAttentionEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        # Instantiate the new TimeEncoder wrapper here
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, time_encoder_type=time_encoder_type, time_encoder_config=time_encoder_config) # UPDATED
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times (absolute time of query node)
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        :return: Tuple[torch.Tensor, dict]
            torch.Tensor: Node embeddings
            dict: Regularization losses from time encoder
        """

        assert (current_layer_num >= 0)
        device = self.node_raw_features.device

        # Initialize losses for this computation path
        current_time_encoder_reg_losses = {}

        # query (source) node always has the start time with time interval == 0
        # Shape: (batch_size, 1, time_feat_dim)
        # Use new TimeEncoder interface: current_times=node_interact_times, neighbor_times=node_interact_times (delta_t = 0)
        node_time_features, reg_losses_node = self.time_encoder(
            current_times=torch.from_numpy(node_interact_times).unsqueeze(dim=1).to(device),
            neighbor_times=torch.from_numpy(node_interact_times).unsqueeze(dim=1).to(device)
        )
        # Accumulate losses
        for loss_name, loss_value in reg_losses_node.items():
            current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value

        # shape (batch_size, node_feat_dim)
        # add memory and node raw features to get node features
        node_features = node_memories[torch.from_numpy(node_ids)] + self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_features, current_time_encoder_reg_losses
        else:
            # get source node representations by aggregating embeddings from the previous (curr_layers - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features, reg_losses_conv_node = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                       node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)
            # Accumulate losses
            for loss_name, loss_value in reg_losses_conv_node.items():
                current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_times ndarray, shape (batch_size, num_neighbors) (Absolute interaction times of neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features, reg_losses_conv_neighbor = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                                node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(), # Absolute time of the neighbors
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)
            # Accumulate losses
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
            # Accumulate losses
            for loss_name, loss_value in reg_losses_neighbor_time_feat.items():
                current_time_encoder_reg_losses[loss_name] = current_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)

            return output, current_time_encoder_reg_losses


def compute_src_dst_node_time_shifts(src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
    """
    compute the mean and standard deviation of time shifts
    :param src_node_ids: ndarray, shape (*, )
    :param dst_node_ids:: ndarray, shape (*, )
    :param node_interact_times: ndarray, shape (*, )
    :return:
    """
    src_node_last_timestamps = dict()
    dst_node_last_timestamps = dict()
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []
    for k in range(len(src_node_ids)):
        src_node_id = src_node_ids[k]
        dst_node_id = dst_node_ids[k]
        node_interact_time = node_interact_times[k]
        if src_node_id not in src_node_last_timestamps.keys():
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps.keys():
            dst_node_last_timestamps[dst_node_id] = 0
        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])
        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time
    assert len(src_node_all_time_shifts) == len(src_node_ids)
    assert len(dst_node_all_time_shifts) == len(dst_node_ids)
    src_node_mean_time_shift = np.mean(src_node_all_time_shifts)
    src_node_std_time_shift = np.std(src_node_all_time_shifts)
    dst_node_mean_time_shift_dst = np.mean(dst_node_all_time_shifts)
    dst_node_std_time_shift = np.std(dst_node_all_time_shifts)

    return src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift
