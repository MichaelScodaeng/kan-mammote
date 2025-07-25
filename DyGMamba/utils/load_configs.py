import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGMamba', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba',
                                 'KANMAMMOTE', 'LeTEModel', 'NoTimeEmbeddingModel', 'SPETimeEmbeddingModel', 'LPETimeEmbeddingModel']) # ADDED YOUR MODELS
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='uniform', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=1, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--max_interaction_times', type=int, default=10,
                        help='max interactions for src and dst to consider')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    # ------------------ KANMAMMOTE Specific Arguments ------------------
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of KAN-MAMMOTE internal states and Mamba states')
    parser.add_argument('--D_time', type=int, default=64, help='Dimension of time embedding from K-MOTE/FasterKAN in KAN-MAMMOTE')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in K-MOTE')
    parser.add_argument('--K_top', type=int, default=4, help='Number of top-K experts to select in K-MOTE')
    parser.add_argument('--use_aux_features_router', action='store_true', default=False, help='Whether K-MOTE router uses auxiliary features')
    parser.add_argument('--raw_event_feature_dim', type=int, default=0, help='Dimension of raw auxiliary event features for router')
    parser.add_argument('--router_noise_scale', type=float, default=1e-2, help='Noise scale for K-MOTE router logits')
    parser.add_argument('--use_load_balancing', action='store_true', default=True, help='Whether to apply load balancing loss for K-MOTE MoE')
    parser.add_argument('--balance_coefficient', type=float, default=0.01, help='Coefficient for K-MOTE MoE load balancing loss')
    parser.add_argument('--kan_noise_scale', type=float, default=0.1, help='Noise scale for MatrixKANLayer')
    parser.add_argument('--kan_scale_base_mu', type=float, default=0.0, help='Mean for base scale initialization in MatrixKANLayer')
    parser.add_argument('--kan_scale_base_sigma', type=float, default=1.0, help='Std for base scale initialization in MatrixKANLayer')
    parser.add_argument('--kan_grid_eps', type=float, default=0.02, help='Epsilon for adaptive grid update in MatrixKANLayer')
    parser.add_argument('--kan_grid_range_min', type=float, default=-1.0, help='Min value for initial grid range in MatrixKANLayer')
    parser.add_argument('--kan_grid_range_max', type=float, default=1.0, help='Max value for initial grid range in MatrixKANLayer')
    parser.add_argument('--kan_sp_trainable', action='store_true', default=True, help='Scale spline trainable in MatrixKANLayer')
    parser.add_argument('--kan_sb_trainable', action='store_true', default=True, help='Scale base trainable in MatrixKANLayer')
    parser.add_argument('--fourier_k_prime', type=int, default=5, help='Number of harmonics for FourierBasis in KANLayer')
    parser.add_argument('--fourier_learnable_params', action='store_true', default=True, help='Whether Fourier params (A, omega, phi) are learnable')
    parser.add_argument('--rkhs_num_mixture_components', type=int, default=8, help='Number of Gaussian mixture components in RKHSBasis')
    # rkhs_learnable_params is default to True for nn.Parameters
    parser.add_argument('--wavelet_num_wavelets', type=int, default=8, help='Number of wavelets for WaveletBasis')
    parser.add_argument('--wavelet_mother_type', type=str, default='mexican_hat', choices=['mexican_hat', 'morlet'], help='Type of mother wavelet for WaveletBasis')
    # wavelet_learnable_params is default to True for nn.Parameters
    parser.add_argument('--spline_grid_size', type=int, default=5, help='Number of grid intervals (G) for B-splines in MatrixKANLayer')
    parser.add_argument('--spline_degree', type=int, default=3, help='Piecewise polynomial order (k) for splines in MatrixKANLayer')
    parser.add_argument('--mamba_d_state', type=int, default=16, help='SSM state expansion factor for Mamba in KAN-MAMMOTE')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Local convolution width for Mamba in KAN-MAMMOTE')
    parser.add_argument('--mamba_expand', type=int, default=1, help='Block expansion factor for Mamba in KAN-MAMMOTE')
    parser.add_argument('--mamba_headdim', type=int, default=64, help='Head dimension for Mamba d_ssm / nheads in KAN-MAMMOTE')
    parser.add_argument('--mamba_dt_min', type=float, default=0.001, help='Minimum initial value for dt in Mamba')
    parser.add_argument('--mamba_dt_max', type=float, default=0.1, help='Maximum initial value for dt in Mamba')
    parser.add_argument('--mamba_dt_init_floor', type=float, default=1e-4, help='Floor for initial dt values in Mamba')
    parser.add_argument('--mamba_bias', action='store_true', default=False, help='Whether to use bias in Mamba linear projections')
    parser.add_argument('--mamba_conv_bias', action='store_true', default=True, help='Whether to use bias in Mamba conv1d layer')
    parser.add_argument('--mamba_chunk_size', type=int, default=256, help='Chunk size for combined SSM kernel in Mamba')
    parser.add_argument('--mamba_use_mem_eff_path', action='store_true', default=True, help='Use memory-efficient path for Mamba (should be True)')
    parser.add_argument('--mamba_d_ssm', type=int, default=None, help='If not None, Mamba SSM applies only to this many dimensions. Default: None (d_inner)') # Ensure default None type is handled
    parser.add_argument('--lambda_sobolev_l2', type=float, default=0.0, help='Coefficient for Sobolev L2 regularization loss (set to 0 for now)') # Explicitly set to 0.0
    parser.add_argument('--lambda_total_variation', type=float, default=0.0, help='Coefficient for Total Variation regularization loss (set to 0 for now)') # Explicitly set to 0.0

    # ------------------ LeTEModel Specific Arguments ------------------
    parser.add_argument('--lete_p', type=float, default=0.5, help='Fraction of dimensions for Fourier vs Spline in LeTE')
    parser.add_argument('--lete_fourier_grid_size', type=int, default=5, help='Grid size for Fourier part in LeTE')
    parser.add_argument('--lete_spline_grid_size', type=int, default=5, help='Grid size for Spline part in LeTE')
    parser.add_argument('--lete_spline_order', type=int, default=3, help='Spline order for Spline part in LeTE')
    parser.add_argument('--lete_layer_norm', action='store_true', default=True, help='Whether to apply layer norm in LeTE')
    parser.add_argument('--lete_scale', action='store_true', default=True, help='Whether to apply learnable scale in LeTE')
    parser.add_argument('--lete_param_requires_grad', action='store_true', default=True, help='Whether LeTE params require grad')

    # ------------------ LPETimeEmbeddingModel Specific Arguments ------------------
    parser.add_argument('--lpe_num_time_bins', type=int, default=1000, help='Number of bins for LPE time discretization')
    parser.add_argument('--lpe_max_time_diff', type=float, default=2.6e7, help='Maximum time difference for LPE binning (e.g., max timestamp in dataset)')

    parser.add_argument('--time_encoder_type', type=str, default='FixedSinusoidal', 
                        choices=['KANMAMMOTE', 'LeTE', 'SPE', 'LPE', 'NoTime', 'FixedSinusoidal'],
                        help='Type of time encoder to use internally in graph models') # THIS IS THE NEW LINE
    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        # Combine kan_grid_range_min/max into a list for the config object
        args.kan_grid_range = [args.kan_grid_range_min, args.kan_grid_range_max]
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['enron']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == 'JODIE':
            if args.dataset_name in ['mooc']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['uci']:
                args.dropout = 0.4
            else:
                args.dropout = 0.1
        elif args.model_name == 'DyRep':
            if args.dataset_name in ['mooc', 'lastfm', 'enron', 'uci']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == 'TGN'
            if args.dataset_name in ['mooc']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['enron', 'SocialEvo']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ['TGN', 'DyRep']:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        if args.dataset_name in ['mooc', 'SocialEvo', 'uci']:
            args.num_neighbors = 64
        elif args.dataset_name in ['lastfm']:
            args.num_neighbors = 128
        else:
            args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['SocialEvo', 'uci']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['wikipedia']:
            args.num_neighbors = 30
        elif args.dataset_name in ['reddit', 'lastfm']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ['wikipedia', 'reddit', 'enron']:
            args.dropout = 0.5
        elif args.dataset_name in ['mooc', 'uci']:
            args.dropout = 0.4
        elif args.dataset_name in ['lastfm']:
            args.dropout = 0.0
        elif args.dataset_name in ['SocialEvo']:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc', 'enron']:
            args.max_input_sequence_length = 256
            args.patch_size = 4
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        else:
            args.max_input_sequence_length = 32 
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    elif args.model_name == 'DyGMamba':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 1
        elif args.dataset_name in ['mooc', 'enron']:
            args.max_input_sequence_length = 256
            args.patch_size = 1
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        if args.dataset_name in ['enron']:
            args.max_interaction_times = 30
        elif args.dataset_name in ['mooc','lastfm']:
            args.max_interaction_times = 10
        else:
            args.max_interaction_times = 5
    # ------------------ Your New Models' Best Configs (Initial Draft) ------------------
    elif args.model_name == 'KANMAMMOTE':
        # These are initial suggestions, you will need to tune them for best performance
        args.d_model = 128
        args.D_time = 64
        args.num_layers = 2
        args.max_input_sequence_length = 32 # Default from DyGMamba/DyGFormer
        args.patch_size = 1 # KAN-MAMMOTE_Core does not directly use patch_size for feature aggregation, but for helper consistency

        args.num_experts = 4
        args.K_top = 4 # Use all experts initially
        args.use_load_balancing = True
        args.balance_coefficient = 0.01
        args.router_noise_scale = 1e-2

        args.fourier_k_prime = 5
        args.rkhs_num_mixture_components = 8
        args.wavelet_num_wavelets = 8
        
        args.spline_grid_size = 5
        args.spline_degree = 3
        args.kan_grid_range_min = -1.0
        args.kan_grid_range_max = 1.0
        
        args.mamba_d_state = 16
        args.mamba_d_conv = 4
        args.mamba_expand = 1
        args.mamba_headdim = 64
        args.mamba_dt_min = 0.001
        args.mamba_dt_max = 0.1
        args.mamba_dt_init_floor = 1e-4

        # Dropout and other general params
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        args.learning_rate = 0.0001 # A common good starting point

    elif args.model_name == 'LeTEModel':
        args.time_feat_dim = 100 # Standard embedding dimension
        args.num_layers = 2 # Just for interface compatibility if a base GNN uses it
        args.num_heads = 2 # Just for interface compatibility
        args.dropout = 0.1 # General dropout for any internal MLPs

        args.lete_p = 0.5 # Equal split between Fourier and Spline
        args.lete_fourier_grid_size = 5
        args.lete_spline_grid_size = 5
        args.lete_spline_order = 3
        args.lete_layer_norm = True
        args.lete_scale = True
        args.lete_param_requires_grad = True
        
        args.learning_rate = 0.0001

    elif args.model_name == 'NoTimeEmbeddingModel':
        args.time_feat_dim = 100 # Dummy value, not used
        args.num_layers = 2 # For interface compatibility
        args.num_heads = 2 # For interface compatibility
        args.dropout = 0.1 # General dropout for internal MLPs
        args.learning_rate = 0.0001

    elif args.model_name == 'SPETimeEmbeddingModel':
        args.time_feat_dim = 100 # Standard embedding dimension for SPE
        args.num_layers = 2 # For interface compatibility
        args.num_heads = 2 # For interface compatibility
        args.dropout = 0.1 # General dropout for internal MLPs
        args.learning_rate = 0.0001

    elif args.model_name == 'LPETimeEmbeddingModel':
        args.time_feat_dim = 100 # Standard embedding dimension for LPE
        args.num_layers = 2 # For interface compatibility
        args.num_heads = 2 # For interface compatibility
        args.dropout = 0.1 # General dropout for internal MLPs

        args.lpe_num_time_bins = 1000 # A reasonable number of bins
        # max_time_diff needs to be dataset specific for optimal performance.
        # For a general default, choose a large value that covers most common datasets.
        # Max timestamp in Wikipedia is ~1.7e7, Reddit ~2.6e7. So 2.6e7 is a good general max.
        args.lpe_max_time_diff = 2.6e7 
        
        args.learning_rate = 0.0001

    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")