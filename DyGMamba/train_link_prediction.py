import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from DyGMamba.models.TGAT import TGAT
from DyGMamba.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from DyGMamba.models.CAWN import CAWN
from DyGMamba.models.TCL import TCL
from DyGMamba.models.GraphMixer import GraphMixer
from DyGMamba.models.DyGFormer import DyGFormer
from DyGMamba.models.DyGMamba import DyGMamba
# ADDED NEW IMPORTS
from DyGMamba.models.modules import MergeLayer, MergeLayerTD, TimeEncoder # TimeEncoder is now our wrapper
from DyGMamba.utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from DyGMamba.utils.utils import get_neighbor_sampler, NegativeEdgeSampler
# Updated evaluate_model_link_prediction to return regularization losses
from DyGMamba.evaluate_models_utils import evaluate_model_link_prediction 
from DyGMamba.utils.metrics import get_link_prediction_metrics
from DyGMamba.utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from DyGMamba.utils.EarlyStopping import EarlyStopping
from DyGMamba.utils.load_configs import get_link_prediction_args
import warnings
from collections import defaultdict
import sys
import os

if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                             sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                             time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data,
                                                            sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                            time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # Prepare time_encoder_config based on selected type
        time_encoder_config = {}
        if args.time_encoder_type == 'KANMAMMOTE':
            time_encoder_config = {
                'D_time': args.D_time, 'num_experts': args.num_experts, 'K_top': args.K_top,
                'use_aux_features_router': args.use_aux_features_router, 'raw_event_feature_dim': args.raw_event_feature_dim,
                'use_load_balancing': args.use_load_balancing, 'balance_coefficient': args.balance_coefficient,
                'router_noise_scale': args.router_noise_scale, 'kan_noise_scale': args.kan_noise_scale,
                'kan_scale_base_mu': args.kan_scale_base_mu, 'kan_scale_base_sigma': args.kan_scale_base_sigma,
                'kan_grid_eps': args.kan_grid_eps, 'kan_grid_range': args.kan_grid_range,
                'kan_sp_trainable': args.kan_sp_trainable, 'kan_sb_trainable': args.kan_sb_trainable,
                'fourier_k_prime': args.fourier_k_prime, 'fourier_learnable_params': args.fourier_learnable_params,
                'rkhs_num_mixture_components': args.rkhs_num_mixture_components, 'wavelet_num_wavelets': args.wavelet_num_wavelets,
                'wavelet_mother_type': args.wavelet_mother_type, 'wavelet_learnable_params': args.wavelet_learnable_params,
                'spline_grid_size': args.spline_grid_size, 'spline_degree': args.spline_degree,
                'lambda_sobolev_l2': args.lambda_sobolev_l2, 'lambda_total_variation': args.lambda_total_variation,
                'device': args.device, 'dtype': torch.float32 # Pass device and dtype for KANMAMMOTEConfig
            }
        elif args.time_encoder_type == 'LeTE':
            time_encoder_config = {
                'p': args.lete_p, 'layer_norm': args.lete_layer_norm, 'scale': args.lete_scale,
                'parameter_requires_grad': args.lete_param_requires_grad
            }
        elif args.time_encoder_type == 'LPE':
            time_encoder_config = {
                'num_time_bins': args.lpe_num_time_bins, 'max_time_diff': args.lpe_max_time_diff
            }
        # SPE and NoTime encoders don't need specific configs, they're handled internally.

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device,
                                    time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                                           time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device,
                                    time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device,
                                   time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device,
                                          time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS

        elif args.model_name == 'DyGMamba':
            dynamic_backbone = DyGMamba(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,gamma=args.gamma,
                                         max_input_sequence_length=args.max_input_sequence_length, max_interaction_times=args.max_interaction_times,device=args.device,
                                         time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS

        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device,
                                         time_encoder_type=args.time_encoder_type, time_encoder_config=time_encoder_config) # PASSED TIME ENCODER ARGS
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        # link_predictor
        # All models will now use MergeLayerTD, which expects 3 inputs
        link_predictor = MergeLayerTD(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1], input_dim3=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()
        best_acc = 0
        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_time_encoder_reg_losses = defaultdict(float) # NEW: To accumulate regularization losses from time encoder

            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):

                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # All models now return (src_emb, dst_emb, reg_losses_dict). DyGMamba also has time_diff_emb.
                # All link_predictor calls will use 3 inputs.

                if args.model_name == 'DyGMamba':
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_emb, positive_reg_losses = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_time_diff_emb, negative_reg_losses = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times) # node_interact_times is for positives, keep consistent for negatives

                elif args.model_name in ['TGAT', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings, positive_reg_losses = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, negative_reg_losses = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                    # Create dummy third input for MergeLayerTD
                    batch_time_diff_emb = torch.zeros_like(batch_src_node_embeddings).to(args.device)
                    batch_neg_time_diff_emb = torch.zeros_like(batch_neg_src_node_embeddings).to(args.device)

                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # First, compute for negative edges (no memory update)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, negative_reg_losses = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)
                    # Then, compute for positive edges (memory updates happen)
                    batch_src_node_embeddings, batch_dst_node_embeddings, positive_reg_losses = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                    # Create dummy third input for MergeLayerTD
                    batch_time_diff_emb = torch.zeros_like(batch_src_node_embeddings).to(args.device)
                    batch_neg_time_diff_emb = torch.zeros_like(batch_neg_src_node_embeddings).to(args.device)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                
                # Accumulate regularization losses for the current batch
                for loss_name, loss_value in positive_reg_losses.items():
                    train_time_encoder_reg_losses[loss_name] += loss_value
                for loss_name, loss_value in negative_reg_losses.items():
                    train_time_encoder_reg_losses[loss_name] += loss_value


                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, input_3=batch_time_diff_emb).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, input_3=batch_neg_time_diff_emb).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                # ADD REGULARIZATION LOSS TO TOTAL LOSS
                total_loss = loss
                for loss_name, reg_loss_value in train_time_encoder_reg_losses.items():
                    # Only add if the loss is non-zero
                    if reg_loss_value.item() != 0.0:
                        total_loss += reg_loss_value # Summing the scalar loss with the accumulated regularization loss
                        # We also want to log these individual regularization losses
                        # For now, just sum them. Logging will happen at epoch end.

                optimizer.zero_grad()
                total_loss.backward() # Use total_loss for backward pass
                optimizer.step()
                
                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item():.4f}, total loss: {total_loss.item():.4f}') # Log total_loss


                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.detach_memory_bank()


            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            # Evaluate on validation sets
            val_losses, val_metrics, val_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
            


            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

            new_node_val_losses, new_node_val_metrics, new_node_val_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)



            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            # Log time encoder regularization losses for training
            for loss_name, loss_value in train_time_encoder_reg_losses.items():
                logger.info(f'train {loss_name}: {loss_value.item() / len(train_idx_data_loader):.6f}') # Average per batch
            
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            # Log time encoder regularization losses for validation
            for loss_name, loss_value in val_time_encoder_reg_losses.items():
                logger.info(f'validate {loss_name}: {loss_value.item() / len(val_idx_data_loader):.6f}') # Average per batch

            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')
            # Log time encoder regularization losses for new node validation
            for loss_name, loss_value in new_node_val_time_encoder_reg_losses.items():
                logger.info(f'new node validate {loss_name}: {loss_value.item() / len(new_node_val_idx_data_loader):.6f}') # Average per batch

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics, test_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)


                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                new_node_test_losses, new_node_test_metrics, new_node_test_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)


                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                    if metric_name == 'average_precision':
                        if np.mean([test_metric[metric_name] for test_metric in test_metrics]) > best_acc:
                            best_acc = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                        logger.info(
                            f'best test average_precision: {best_acc:.4f}')
                # Log time encoder regularization losses for test
                for loss_name, loss_value in test_time_encoder_reg_losses.items():
                    logger.info(f'test {loss_name}: {loss_value.item() / len(test_idx_data_loader):.6f}') # Average per batch
                
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')
                # Log time encoder regularization losses for new node test
                for loss_name, loss_value in new_node_test_time_encoder_reg_losses.items():
                    logger.info(f'new node test {loss_name}: {loss_value.item() / len(new_node_test_idx_data_loader):.6f}') # Average per batch


            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics, val_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
        
            new_node_val_losses, new_node_val_metrics, new_node_val_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics, test_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        new_node_test_losses, new_node_test_metrics, new_node_test_time_encoder_reg_losses = evaluate_model_link_prediction(model_name=args.model_name, # UPDATED CALL
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}
        
        # Also store regularization losses in results json
        val_reg_loss_dict, new_node_val_reg_loss_dict = {}, {}
        test_reg_loss_dict, new_node_test_reg_loss_dict = {}, {}


        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric
            for loss_name, loss_value in val_time_encoder_reg_losses.items(): # LOG AND STORE REG LOSSES
                logger.info(f'validate {loss_name}: {loss_value.item() / len(val_idx_data_loader):.6f}')
                val_reg_loss_dict[loss_name] = loss_value.item() / len(val_idx_data_loader)
            
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric
            for loss_name, loss_value in new_node_val_time_encoder_reg_losses.items(): # LOG AND STORE REG LOSSES
                logger.info(f'new node validate {loss_name}: {loss_value.item() / len(new_node_val_idx_data_loader):.6f}')
                new_node_val_reg_loss_dict[loss_name] = loss_value.item() / len(new_node_val_idx_data_loader)

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric
        for loss_name, loss_value in test_time_encoder_reg_losses.items(): # LOG AND STORE REG LOSSES
            logger.info(f'test {loss_name}: {loss_value.item() / len(test_idx_data_loader):.6f}')
            test_reg_loss_dict[loss_name] = loss_value.item() / len(test_idx_data_loader)

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric
        for loss_name, loss_value in new_node_test_time_encoder_reg_losses.items(): # LOG AND STORE REG LOSSES
            logger.info(f'new node test {loss_name}: {loss_value.item() / len(new_node_test_idx_data_loader):.6f}')
            new_node_test_reg_loss_dict[loss_name] = loss_value.item() / len(new_node_test_idx_data_loader)

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict},
                "validate regularization losses": val_reg_loss_dict, # ADDED REG LOSSES TO JSON
                "new node validate regularization losses": new_node_val_reg_loss_dict, # ADDED REG LOSSES TO JSON
                "test regularization losses": test_reg_loss_dict, # ADDED REG LOSSES TO JSON
                "new node test regularization losses": new_node_test_reg_loss_dict # ADDED REG LOSSES TO JSON
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict},
                "test regularization losses": test_reg_loss_dict, # ADDED REG LOSSES TO JSON
                "new node test regularization losses": new_node_test_reg_loss_dict # ADDED REG LOSSES TO JSON
            }

        result_json = json.dumps(result_json, indent=4)


        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)

        timestamp = str(time.time())
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_{timestamp}.json")


        while os.path.exists(save_result_path):
            timestamp = str(time.time())
            save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_{timestamp}.json")
    
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')
    
        for metric_name in new_node_val_metric_all_runs[0].keys():
            logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
            logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()