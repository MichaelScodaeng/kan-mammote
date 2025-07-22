# kan_mamote/src/utils/config.py

import torch

class KANMAMMOTEConfig:
    """
    Configuration class for the KAN-MAMMOTE model and its components.
    Supports overriding defaults via kwargs.
    """
    def __init__(self, **kwargs):
        # Global Model Parameters
        self.d_model = kwargs.get('d_model', 128)
        self.D_time = kwargs.get('D_time', 64)
        self.num_layers = kwargs.get('num_layers', 2)
        self.input_feature_dim = kwargs.get('input_feature_dim', 10)
        self.output_dim_for_task = kwargs.get('output_dim_for_task', 1)

        # K-MOTE Parameters
        self.K_top = kwargs.get('K_top', 4)
        self.use_aux_features_router = kwargs.get('use_aux_features_router', False)
        self.raw_event_feature_dim = kwargs.get('raw_event_feature_dim', 0)

        # Router Parameters (MoERouter)
        self.router_noise_scale = kwargs.get('router_noise_scale', 1e-2)
        self.use_load_balancing = kwargs.get('use_load_balancing', True)
        self.balance_coefficient = kwargs.get('balance_coefficient', 0.01)

        # KANLayer and Basis Function Parameters
        self.kan_noise_scale = kwargs.get('kan_noise_scale', 0.1)
        self.kan_scale_base_mu = kwargs.get('kan_scale_base_mu', 0.0)
        self.kan_scale_base_sigma = kwargs.get('kan_scale_base_sigma', 1.0)
        self.kan_grid_eps = kwargs.get('kan_grid_eps', 0.02)
        self.kan_grid_range = kwargs.get('kan_grid_range', [-1, 1])
        self.kan_sp_trainable = kwargs.get('kan_sp_trainable', True)
        self.kan_sb_trainable = kwargs.get('kan_sb_trainable', True)

        # Fourier Basis
        self.fourier_k_prime = kwargs.get('fourier_k_prime', 10)
        self.fourier_learnable_params = kwargs.get('fourier_learnable_params', True)

        # RKHS / Gaussian Kernel Basis
        self.rkhs_num_mixture_components = kwargs.get('rkhs_num_mixture_components', 10)
        self.rkhs_learnable_params = kwargs.get('rkhs_learnable_params', True)

        # Wavelet Basis
        self.wavelet_num_wavelets = kwargs.get('wavelet_num_wavelets', 10)
        self.wavelet_mother_type = kwargs.get('wavelet_mother_type', 'mexican_hat')
        self.wavelet_learnable_params = kwargs.get('wavelet_learnable_params', True)

        # Spline Basis (MatrixKANLayer)
        self.spline_grid_size = kwargs.get('spline_grid_size', 5)
        self.spline_degree = kwargs.get('spline_degree', 3)

        # Mamba2 Parameters
        self.mamba_d_state = kwargs.get('mamba_d_state', 128)
        self.mamba_d_conv = kwargs.get('mamba_d_conv', 4)
        self.mamba_expand = kwargs.get('mamba_expand', 2)
        self.mamba_headdim = kwargs.get('mamba_headdim', 32)
        self.mamba_dt_min = kwargs.get('mamba_dt_min', 0.001)
        self.mamba_dt_max = kwargs.get('mamba_dt_max', 0.1)
        self.mamba_dt_init_floor = kwargs.get('mamba_dt_init_floor', 1e-4)
        self.mamba_bias = kwargs.get('mamba_bias', False)
        self.mamba_conv_bias = kwargs.get('mamba_conv_bias', True)
        self.mamba_chunk_size = kwargs.get('mamba_chunk_size', 256)
        self.mamba_use_mem_eff_path = kwargs.get('mamba_use_mem_eff_path', True)
        self.mamba_layer_idx = kwargs.get('mamba_layer_idx', None)
        self.mamba_d_ssm = kwargs.get('mamba_d_ssm', None) # ADDED THIS LINE
        
        # Regularization
        self.lambda_sobolev_l2 = kwargs.get('lambda_sobolev_l2', 0.01)
        self.lambda_total_variation = kwargs.get('lambda_total_variation', 0.01)

        # Training Parameters
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.num_epochs = kwargs.get('num_epochs', 5)
        self.batch_size = kwargs.get('batch_size', 32)
        self.sequence_length = kwargs.get('sequence_length', 100)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)

    def to_dict(self):
        return self.__dict__