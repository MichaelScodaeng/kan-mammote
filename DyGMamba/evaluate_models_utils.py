import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json


from DyGMamba.utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from DyGMamba.utils.utils import set_random_seed
from DyGMamba.utils.utils import NegativeEdgeSampler, NeighborSampler
from DyGMamba.utils.DataLoader import Data


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
        evaluate_losses (list): List of loss values.
        evaluate_metrics (list): List of metric dictionaries.
        evaluate_time_encoder_reg_losses (dict): Aggregated regularization losses from time encoder.
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        # Store aggregated regularization losses from time encoders
        evaluate_time_encoder_reg_losses = {} # NEW

        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # We need to compute for positive and negative edges respectively.
            # All models now return (src_emb, dst_emb, reg_losses_dict).
            # DyGMamba returns (src_emb, dst_emb, time_diff_emb, reg_losses_dict).

            # Handle DyGMamba's unique 4-output signature
            if model_name == 'DyGMamba':
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_emb, positive_reg_losses = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_time_diff_emb, negative_reg_losses = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times) # Note: node_interact_times is for positives, keep consistent for negatives

            # Handle all other models that return (src_emb, dst_emb, reg_losses_dict)
            elif model_name in ['TGAT', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']: # Removed DyGMamba from here
                batch_src_node_embeddings, batch_dst_node_embeddings, positive_reg_losses = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, negative_reg_losses = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
                # For models that don't output a specific time_diff_emb, create a dummy one for MergeLayerTD
                batch_time_diff_emb = torch.zeros_like(batch_src_node_embeddings).to(batch_src_node_embeddings.device)
                batch_neg_time_diff_emb = torch.zeros_like(batch_neg_src_node_embeddings).to(batch_neg_src_node_embeddings.device)

            # Handle MemoryModel specific call (returns 3 values, reg_losses is the 3rd)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # Memory models need careful handling of state updates (positive vs negative)
                # First, compute for negative edges (no memory update)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, negative_reg_losses = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)
                # Then, compute for positive edges (memory updates happen)
                batch_src_node_embeddings, batch_dst_node_embeddings, positive_reg_losses = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
                # For models that don't output a specific time_diff_emb, create a dummy one for MergeLayerTD
                batch_time_diff_emb = torch.zeros_like(batch_src_node_embeddings).to(batch_src_node_embeddings.device)
                batch_neg_time_diff_emb = torch.zeros_like(batch_neg_src_node_embeddings).to(batch_neg_src_node_embeddings.device)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            
            # Aggregate regularization losses for the current batch
            for loss_name, loss_value in positive_reg_losses.items():
                evaluate_time_encoder_reg_losses[loss_name] = evaluate_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value
            for loss_name, loss_value in negative_reg_losses.items():
                evaluate_time_encoder_reg_losses[loss_name] = evaluate_time_encoder_reg_losses.get(loss_name, 0.0) + loss_value


            # Now, the link predictor call, which consistently uses 3 inputs (src_emb, dst_emb, time_diff_emb)
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, input_3=batch_time_diff_emb).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, input_3=batch_neg_time_diff_emb).squeeze(dim=-1).sigmoid()
            
            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics, evaluate_time_encoder_reg_losses # ADDED regularization losses