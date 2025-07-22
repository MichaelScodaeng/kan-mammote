# kan_mamote/src/layers/dynamic_mamba_ssm.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from einops import rearrange, repeat

# Import necessary Mamba-SSM utilities, handling potential ImportError for Triton
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    # Import these if you are using distributed training (otherwise can be commented out or use dummy)
    # from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    # from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    
    # We will replicate Mamba2's structure, so we don't directly inherit its forward
    # but we will use its __init__ logic and components.
    # We'll import it just for constants and parameter names reference
    from mamba_ssm import Mamba2 as OriginalMamba2_Reference 
    TRITON_KERNELS_AVAILABLE = True
except ImportError:
    print("Warning: Triton Mamba kernels (causal_conv1d, mamba_ssm.ops.triton) not found.")
    print("Using dummy fallbacks (which might not be fully functional or performant).")
    causal_conv1d_fn, causal_conv1d_update = None, None
    causal_conv1d_varlen_states = None
    selective_state_update = None
    RMSNormGated = nn.LayerNorm # Fallback to standard LayerNorm
    OriginalMamba2_Reference = nn.Module # Dummy reference
    # ColumnParallelLinear = nn.Linear; RowParallelLinear = nn.Linear
    # all_reduce = lambda x, *args: x; reduce_scatter = lambda x, *args: x
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None
    TRITON_KERNELS_AVAILABLE = False
finally:
    print(f"DEBUG: Triton Kernels Available: {TRITON_KERNELS_AVAILABLE}")

class DynamicMambaSSM(nn.Module): # Inherit from nn.Module, not OriginalMamba2_Reference
    """
    A Mamba2 block adapted to incorporate dynamic parameters influenced by
    the delta_t_embedding from K-MOTE. This class re-implements Mamba2's
    forward pass to allow explicit injection of `dt_modulation_sequence`.
    """
    def __init__(self,
                 d_model: int,
                 k_mote_delta_t_embedding_dim: int, # D_time from KANMAMMOTEConfig
                 d_state=128,
                 d_conv=4,
                 conv_init=None,
                 expand=2,
                 headdim=64,
                 d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
                 ngroups=1,
                 A_init_range=(1, 16),
                 D_has_hdim=False,
                 rmsnorm=True,
                 norm_before_gate=False,
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init_floor=1e-4,
                 dt_limit=(0.0, float("inf")),
                 bias=False,
                 conv_bias=True,
                 chunk_size=256,
                 use_mem_eff_path=True, # For vectorized path, this should be True
                 layer_idx=None, # For inference cache management
                 process_group=None, # For distributed (assuming single device for now)
                 sequence_parallel=False, # For distributed (assuming single device for now)
                 device=None,
                 dtype=None,
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Store Mamba2 specific configuration
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group # Kept for generality, assume None for single GPU
        self.sequence_parallel = sequence_parallel # Kept for generality, assume False for single GPU
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu" # Mamba's default
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx # Used for inference cache management

        # Store KAN-MOTE specific dimension
        self.k_mote_delta_t_embedding_dim = k_mote_delta_t_embedding_dim

        # Mamba's input projection (u -> z, x, B, C, dt_raw)
        # Order: [z, x, B, C, dt]
        d_in_proj_total = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        # Use ColumnParallelLinear if distributed, else nn.Linear
        self.in_proj = nn.Linear(self.d_model, d_in_proj_total, bias=bias, **factory_kwargs)

        # Mamba's conv1d layer
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim, # Grouped convolution for efficiency
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU() # Activation function after conv

        # Learnable parameters A, D, dt_bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # Inverse of softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # RMSNorm for SSM output
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        # Output projection
        # Use RowParallelLinear if distributed, else nn.Linear
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # KAN-MOTE specific delta_t modulation projection
        # This projects `k_mote_delta_t_embedding_dim` (D_time) to `nheads` for adding to dt
        self.dt_modulation_proj = nn.Linear(k_mote_delta_t_embedding_dim, self.nheads, bias=True)
        #print(f"DEBUG_DMS_INIT: dt_modulation_proj in_features={self.dt_modulation_proj.in_features}, out_features={self.dt_modulation_proj.out_features} (expected {k_mote_delta_t_embedding_dim} -> {self.nheads})") # ADD THIS LINE
        #print(f"DEBUG_DMS_INIT: Calculated self.nheads={self.nheads}") # ADD THIS LINE
        nn.init.zeros_(self.dt_modulation_proj.weight)
        nn.init.zeros_(self.dt_modulation_proj.bias)


    # Re-implemented forward method to allow dt_modulation_sequence injection
    def forward(self, u: torch.Tensor, # (batch_size, seq_len, d_model)
                dt_modulation_sequence: Optional[torch.Tensor] = None, # (batch_size, seq_len, nheads) - NEW ARG
                seqlen: Optional[int] = None, seq_idx: Optional[torch.Tensor] = None, 
                cu_seqlens: Optional[torch.Tensor] = None, inference_params: Optional[dict] = None):
        """
        u: (batch, seqlen, hidden_dim) or (batch * seqlen, hidden_dim)
        dt_modulation_sequence: (batch, seqlen, nheads) to be added to Mamba's internal dt.
        Returns: same shape as u
        """
        # --- Handle seqlen logic from OriginalMamba2.forward ---
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen
            u = rearrange(u, "(b l) d -> b l d", l=seqlen) # Reshape to (B, L, D) if flattened

        # --- Inference path (single token decoding) ---
        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                raise NotImplementedError("Single-step inference with dynamic dt_modulation requires separate implementation via self.step()")

        # --- Training path (vectorized processing) ---
        # Replicates Mamba2's vectorized forward for efficiency
        
        # This path uses mamba_chunk_scan_combined which allows dt injection.
        # It replicates the non-memory efficient path from Mamba2 for full control.

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj_total)
        
        # A_log is typically negative. A = -torch.exp(self.A_log.float())
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # Split zxbcdt into its components
        # Order in Mamba2's forward: [z0, x0, z, xBC, dt_raw_from_proj]
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        
        # These split sizes are for the *full* `zxbcdt` tensor
        split_sizes = [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads]
        
        # Extract components. Handle d_mlp potentially being 0 (no MLP part).
        # We need to explicitly check if d_mlp is positive for slicing `z0` and `x0`
        # and adjust the start index for `z`, `xBC`, `dt_raw_from_proj` accordingly.
        
        # Mamba2's source (mamba2.py -> forward) splits are dynamic:
        # z0, x0, z, xBC, dt = torch.split(zxbcdt, split_sizes, dim=-1)
        # This split is safe for d_mlp=0, just z0/x0 will be empty tensors.

        z0, x0, z, xBC, dt_raw_from_proj = torch.split(zxbcdt, split_sizes, dim=-1)
        
        # 1. Conv1d
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            # Fallback to standard PyTorch conv if Triton causal_conv1d is not available
            xBC_conv_out = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
            )
        else:
            xBC_transposed_contiguous = xBC.transpose(1, 2).contiguous()
            xBC_conv_out = causal_conv1d_fn(
                xBC_transposed_contiguous,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                # seq_idx=seq_idx, # Pass seq_idx if you are dealing with variable length sequences with cu_seqlens
            ).transpose(1, 2)
        
        # 2. Split conv output into x, B, C
        x_ssm_in, B_ssm_in, C_ssm_in = torch.split(xBC_conv_out, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        # 3. Process dt: Inject dt_modulation_sequence
        # dt_raw_from_proj is (B, L, nheads)
        # dt_modulation_sequence is (B, L, nheads) from KAN-MOTE
        #print(f"DEBUG_DMS_FWD: dt_raw_from_proj shape: {dt_raw_from_proj.shape}") # ADD THIS
        #print(f"DEBUG_DMS_FWD: dt_modulation_sequence shape: {dt_modulation_sequence.shape}") # ADD THIS
        dt_combined = dt_raw_from_proj + dt_modulation_sequence.to(dt_raw_from_proj.dtype)
        
        # 4. SSM (Selective Scan) via mamba_chunk_scan_combined
        if mamba_chunk_scan_combined is None:
            raise RuntimeError("mamba_chunk_scan_combined not available for vectorized path without Triton.")

        # Prepare inputs for mamba_chunk_scan_combined as per mamba2.py
        y_ssm_out = mamba_chunk_scan_combined(
            rearrange(x_ssm_in, "b l (h p) -> b l h p", p=self.headdim), # x input to SSM
            dt_combined, # Modified dt! (B, L, nheads)
            A, # A_log (already converted to A)
            rearrange(B_ssm_in, "b l (g n) -> b l g n", g=self.ngroups), # B
            rearrange(C_ssm_in, "b l (g n) -> b l g n", g=self.ngroups), # C
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None, # z for gating if not RMSNorm
            dt_bias=self.dt_bias, # Still apply the fixed dt_bias
            dt_softplus=True, # Apply softplus to (dt_combined + dt_bias)
            seq_idx=seq_idx, # Pass if available (for varlen support within the kernel)
            cu_seqlens=cu_seqlens, # Pass if available (for varlen support within the kernel)
            **dt_limit_kwargs,
            # return_final_states=ssm_state is not None, # These are for inference only
            # return_varlen_states=cu_seqlens is not None and inference_params is not None, # These are for inference only
        )
        y_ssm_out = rearrange(y_ssm_out, "b l h p -> b l (h p)") # Reshape SSM output

        # 5. RMSNorm and Output Projection
        if self.rmsnorm:
            y_post_norm = self.norm(y_ssm_out, z) # z is also passed here
        else:
            y_post_norm = y_ssm_out
        
        # 6. MLP part (if d_mlp > 0)
        if d_mlp > 0: 
            # The structure for d_mlp > 0 is: F.silu(z0) * x0 + y_post_norm
            # This is different from `torch.cat` used previously, which would increase dimension.
            # It's an element-wise gating and addition, similar to Mamba's internal structure.
            y_mlp_gated = F.silu(z0) * x0 # (B, L, d_mlp)
            y_combined = torch.cat([y_mlp_gated, y_post_norm], dim=-1) # Concatenate MLP gated part with SSM output
        else:
            y_combined = y_post_norm # If no MLP part, just use SSM output

        # 7. Output Projection
        out = self.out_proj(y_combined) # (B, L, D_model)
        
        # Ensure output matches input shape if it was flattened at the start
        if seqlen_og is not None:
             out = rearrange(out, "b l d -> (b l) d")
        
        # If using distributed training, handle reduce_scatter (not active in your current config)
        # if self.process_group is not None:
        #     reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        #     out = reduce_fn(out, self.process_group)

        return out

    # This `step` method is exclusively for single-token inference/decoding.
    # It remains distinct from the vectorized `forward` method above.
    def step(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor,
             dt_modulation_step: Optional[torch.Tensor] = None):
        """
        Step function for inference (single token decoding), adapted to inject dt_modulation_step.
        This method is primarily called by KANMAMMOTE for recurrent processing (which is now replaced).
        Keep for explicit single-step inference/decoding only.
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "DynamicMambaSSM.step expects single token input (seqlen=1)."

        x_in = hidden_states.squeeze(1) # (batch_size, d_model)

        zxbcdt = self.in_proj(x_in) # (B, d_in_proj_total)

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        
        # FIX: Directly assign from torch.split results as in mamba2.py
        # This is more robust for cases where d_mlp can be 0.
        # The variables z0, x0, z, xBC, dt_raw_from_proj will be directly assigned
        # and if d_mlp is 0, z0 and x0 will be empty tensors, which is fine.
        z0, x0, z, xBC_for_conv, dt_raw_from_proj = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        
        # Add prints here to confirm shapes before the error line (these were from previous step)
        #print(f"DEBUG_DMS_STEP_SPLIT: z0 shape: {z0.shape}")
        #print(f"DEBUG_DMS_STEP_SPLIT: x0 shape: {x0.shape}")
        #print(f"DEBUG_DMS_STEP_SPLIT: z shape: {z.shape}")
        #print(f"DEBUG_DMS_STEP_SPLIT: xBC_for_conv shape: {xBC_for_conv.shape}")
        #print(f"DEBUG_DMS_STEP_SPLIT: dt_raw_from_proj shape: {dt_raw_from_proj.shape}")
        

        # --- Inject dt_modulation_step ---
        dt_for_kernel = (dt_raw_from_proj + dt_modulation_step.to(dt_raw_from_proj.dtype))
        dt_for_kernel = repeat(dt_for_kernel, "b h -> b h p", p=self.headdim).to(dtype=torch.float32).contiguous()

        # Conv step
        xBC_for_conv = xBC_for_conv.contiguous()
        if causal_conv1d_update is None:
            raise NotImplementedError("CPU fallback for Mamba2 conv update is not robustly implemented here. Ensure causal_conv1d is installed.")
        else:
            x_conv_out = causal_conv1d_update(
                xBC_for_conv,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w").contiguous(),
                self.conv1d.bias.contiguous() if self.conv1d.bias is not None else None,
                self.activation,
            )
        x_ssm_in, B_ssm_in, C_ssm_in = torch.split(x_conv_out, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        # SSM step
        A_kernel_format = repeat(self.A_log.float().exp().neg(), "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32).contiguous()
        
        B_kernel_format = rearrange(B_ssm_in, "b (g n) -> b g n", g=self.ngroups).contiguous()
        C_kernel_format = rearrange(C_ssm_in, "b (g n) -> b g n", g=self.ngroups).contiguous()
        D_kernel_format = repeat(self.D, "h -> h p", p=self.headdim).contiguous()
        x_ssm_input_kernel_format = rearrange(x_ssm_in, "b (h p) -> b h p", p=self.headdim).contiguous()

        z_kernel_format = None
        if not self.rmsnorm: # If RMSNorm is False, z is used for gating INSIDE selective_state_update
            z_kernel_format = rearrange(z, "b (h p) -> b h p", p=self.headdim).contiguous()

        dt_bias_arg_for_kernel = repeat(self.dt_bias.contiguous(), "h -> h p", p=self.headdim).to(dtype=torch.float32).contiguous()

        y_ssm_out_kernel_format = selective_state_update(
            ssm_state,
            x_ssm_input_kernel_format,
            dt_for_kernel, # This is raw_dt + modulation.
            A_kernel_format,
            B_kernel_format,
            C_kernel_format,
            D_kernel_format,
            z=z_kernel_format,
            dt_bias=dt_bias_arg_for_kernel, # Pass the original dt_bias
            dt_softplus=True # Kernel will apply softplus(dt_for_kernel + dt_bias_arg_for_kernel)
        )
        y_ssm_out = rearrange(y_ssm_out_kernel_format, "b h p -> b (h p)") # (B, d_ssm)

        # Post-SSM Processing
        if self.rmsnorm:
            y_post_norm = self.norm(y_ssm_out, z.contiguous())
        else:
            y_post_norm = y_ssm_out

        final_output_before_proj = y_post_norm
        if d_mlp > 0:
            if z0 is None or x0 is None:
                raise RuntimeError("z0/x0 are None but d_mlp > 0. Split logic error.")
            final_output_before_proj = y_post_norm + F.silu(z0).contiguous() * x0.contiguous()
        
        out = self.out_proj(final_output_before_proj).unsqueeze(1)

        return out, conv_state, ssm_state

    # Allocate inference cache (needed if you still use inference_params or _get_states_from_cache)
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        # This method is directly from Mamba2.py and is used by its `forward` for inference.
        # It's retained if you plan to use `inference_params` with this DynamicMambaSSM.
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state