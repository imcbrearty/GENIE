import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import gamma, beta
import time
from torch_cluster import knn
from torch.nn import functional as F
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.utils import add_self_loops, subgraph
from torch_geometric.nn.pool import radius
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import softmax
from torch.autograd import Variable
from torch_scatter import scatter
from numpy.matlib import repmat
import pathlib
# from torch_geometric.pool import radius
import itertools
import pdb
import pathlib
import yaml

from utils import hash_rows

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

with open('process_config.yaml', 'r') as file:
    process_config = yaml.safe_load(file)

path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

# use_updated_model_definition = config['use_updated_model_definition']
name_of_project = config['name_of_project']
scale_rel = config['scale_rel'] # 30e3
k_sta_edges = config['k_sta_edges']
k_spc_edges = config['k_spc_edges']
template_ver = process_config['template_ver']


scale_t = train_config['kernel_sig_t']*3.0
eps = train_config['kernel_sig_t']*3.0
kernel_sig_t = train_config['kernel_sig_t']

z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
scale_time = z['scale_time']/1000.0
z.close()

# use_updated_model_definition = True
use_phase_types = config['use_phase_types']
use_absolute_pos = config['use_absolute_pos']
use_neighbor_assoc_edges = config.get('use_neighbor_assoc_edges', False)
use_expanded = config['use_expanded']
use_gradient_loss = train_config['use_gradient_loss']
use_embedding = config['use_embedding']
use_sigmoid = config['use_sigmoid']
attach_time = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  ## or use cpu


# =====================================================================
# 1. BASE BUILDING BLOCK: Single Data Aggregation Layer
# =====================================================================
class DataAggregationLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, n_dim_mask=4, embed_dim=10, 
                 use_offsets=True, has_expander=True, ndim_proj_sta=6, ndim_proj_src=9):
        super(DataAggregationLayer, self).__init__('mean')

        self.use_offsets = use_offsets
        self.has_expander = has_expander
        self.out_channels = out_channels

        # Local Branch Transforms
        self.l_t1_1 = nn.Linear(in_channels, out_channels)
        self.l_t1_2 = nn.Linear(in_channels + out_channels + n_dim_mask, out_channels)
        self.l_t2_1 = nn.Linear(in_channels, out_channels)
        self.l_t2_2 = nn.Linear(in_channels + out_channels + n_dim_mask, out_channels)

        self.act11, self.act12, self.act_local = nn.PReLU(), nn.PReLU(), nn.PReLU()
        self.film_local = FiLM(embed_dim, 2 * out_channels)

        # Expander Branch Transforms (Optional)
        if self.has_expander:
            self.l_t1_1c = nn.Linear(in_channels, out_channels)
            self.l_t1_2c = nn.Linear(in_channels + out_channels + n_dim_mask, out_channels)
            self.l_t2_1c = nn.Linear(in_channels, out_channels)
            self.l_t2_2c = nn.Linear(in_channels + out_channels + n_dim_mask, out_channels)

            self.act11c, self.act12c, self.act_exp = nn.PReLU(), nn.PReLU(), nn.PReLU()
            self.gate = nn.Linear(4 * out_channels + embed_dim, 2 * out_channels)
            nn.init.constant_(self.gate.bias, -2.0)

        # Multi-Scale Gamma Generators
        if self.use_offsets:
            self.merge_edges_sta = nn.Sequential(nn.Linear(out_channels + ndim_proj_sta, out_channels), nn.PReLU())
            self.merge_edges_src = nn.Sequential(nn.Linear(out_channels + ndim_proj_src, out_channels), nn.PReLU())

            init_sp_gammas = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32).reshape(1, 3)
            init_src_gammas = torch.tensor([0.1, 1.0, 5.0, 0.5, 10.0], dtype=torch.float32).reshape(1, 5)

            self.log_gamma_sta_base = nn.Parameter(torch.log(init_sp_gammas))
            self.log_gamma_src_base = nn.Parameter(torch.log(init_src_gammas))

            self.f_gamma_sta = nn.Linear(embed_dim, 1 + 3)
            self.f_gamma_src = nn.Linear(embed_dim, 1 + 5)
            nn.init.zeros_(self.f_gamma_sta.weight); nn.init.zeros_(self.f_gamma_sta.bias)
            nn.init.zeros_(self.f_gamma_src.weight); nn.init.zeros_(self.f_gamma_src.bias)

    def _compute_edge_attrs(self, pos_rel_sta, pos_rel_src, embed_context):
        if not self.use_offsets or pos_rel_sta is None or pos_rel_src is None:
            return None, None

        # Station Edges (6D: 3D Direction + 3 Spatial RBFs)
        sta_sp = pos_rel_sta[:, 0:3]
        sta_norm_sp = torch.sqrt(torch.sum(sta_sp**2, dim=1, keepdim=True) + 1e-8)
        d_sta = self.f_gamma_sta(embed_context)
        gammas_sta = torch.exp(self.log_gamma_sta_base + d_sta[:, :1] + 0.2 * torch.tanh(d_sta[:, 1:]))
        edge_sta = torch.cat((sta_sp / sta_norm_sp, torch.exp(-1.0 * sta_norm_sp * gammas_sta)), dim=1)

        # Source Edges (9D: 3D Direction + 3 Spatial RBFs + 2 Temporal RBFs + 1 dt)
        src_sp, src_tm = pos_rel_src[:, 0:3], pos_rel_src[:, 3:4]
        src_norm_sp = torch.sqrt(torch.sum(src_sp**2, dim=1, keepdim=True) + 1e-8)
        src_norm_tm = torch.abs(src_tm)
        d_src = self.f_gamma_src(embed_context)
        gammas_src = torch.exp(self.log_gamma_src_base + d_src[:, :1] + 0.2 * torch.tanh(d_src[:, 1:]))
        
        sp_decay = torch.exp(-1.0 * src_norm_sp * gammas_src[:, 0:3])
        tm_decay = torch.exp(-1.0 * src_norm_tm * gammas_src[:, 3:5])
        edge_src = torch.cat((src_sp / src_norm_sp, sp_decay, tm_decay, src_tm), dim=1)

        return edge_sta, edge_src

    def forward(self, x, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta=None, pos_rel_src=None):
        edge_sta, edge_src = self._compute_edge_attrs(pos_rel_sta, pos_rel_src, embed_context)

        # Local Path
        x1 = self.l_t1_2(torch.cat((x, self.propagate(A_in_sta, x=self.act11(self.l_t1_1(x)), edge_attr=edge_sta, edge_type=1), mask), dim=1))
        x2 = self.l_t2_2(torch.cat((x, self.propagate(A_in_src[0] if isinstance(A_in_src, (tuple, list)) else A_in_src, x=self.act12(self.l_t2_1(x)), edge_attr=edge_src, edge_type=2), mask), dim=1))
        x_local = self.act_local(torch.cat((x1, x2), dim=1))
        x_local = self.film_local(x_local, embed_context)

        if not self.has_expander:
            return x_local

        # Expander Path
        x1_c = self.l_t1_2c(torch.cat((x, self.propagate(A_in_sta, x=self.act11c(self.l_t1_1c(x)), edge_attr=edge_sta, edge_type=1), mask), dim=1))
        x2_c = self.l_t2_2c(torch.cat((x, self.propagate(A_in_src[1], x=self.act12c(self.l_t2_1c(x)), edge_attr=None, edge_type=2), mask), dim=1))
        x_exp = self.act_exp(torch.cat((x1_c, x2_c), dim=1))

        # Gated Fusion
        embed_expand = embed_context.expand(len(x), -1)
        g = torch.sigmoid(self.gate(torch.cat((x_local, x_exp, embed_expand), dim=1)))
        return x_local + g * x_exp

    def message(self, x_j, edge_attr, edge_type):
        if edge_attr is not None:
            return self.merge_edges_sta(torch.cat((x_j, edge_attr), dim=1)) if edge_type == 1 else self.merge_edges_src(torch.cat((x_j, edge_attr), dim=1))
        return x_j


# =====================================================================
# 2. MAIN STACK MODULE: Observation Network with Optional Preconditioner
# =====================================================================
class DataAggregationExpanded(nn.Module):
    def __init__(self, in_channels, out_channels, n_hidden=30, n_dim_mask=4, 
                 use_absolute_pos=True, use_offsets=True, embed_dim=10, use_embedding=True):
        super().__init__()

        self.use_embedding = use_embedding
        if use_absolute_pos:
            in_channels += 6

        # --- OPTIONAL GEOMETRIC PRECONDITIONER (Pre-GNN) ---
        if self.use_embedding:
            geom_in_dim = 1 + 7  # Bias (1D) + Relative Position Features (7D)
            self.init_geom = nn.Linear(geom_in_dim, n_hidden)
            self.film_geom_init = FiLM(embed_dim, n_hidden)
            self.act_geom_init = nn.PReLU()

            self.geom_layer1 = DataAggregationLayer(
                in_channels=n_hidden, out_channels=n_hidden, n_dim_mask=n_dim_mask, 
                embed_dim=embed_dim, use_offsets=use_offsets, has_expander=False
            )
            self.geom_layer2 = DataAggregationLayer(
                in_channels=2 * n_hidden, out_channels=n_hidden, n_dim_mask=n_dim_mask, 
                embed_dim=embed_dim, use_offsets=use_offsets, has_expander=False
            )
            in_channels += 2 * n_hidden  # Concatenate structural embedding to main input

        # --- MAIN OBSERVATION GNN STACK ---
        self.init_trns = nn.Linear(in_channels + n_dim_mask, n_hidden)
        self.film_init = FiLM(embed_dim, n_hidden)
        self.act_init = nn.PReLU()

        self.layer1 = DataAggregationLayer(
            in_channels=n_hidden, out_channels=n_hidden, n_dim_mask=n_dim_mask, 
            embed_dim=embed_dim, use_offsets=use_offsets, has_expander=True
        )
        self.layer2 = DataAggregationLayer(
            in_channels=2 * n_hidden, out_channels=n_hidden, n_dim_mask=n_dim_mask, 
            embed_dim=embed_dim, use_offsets=use_offsets, has_expander=True
        )
        self.layer3 = DataAggregationLayer(
            in_channels=2 * n_hidden, out_channels=out_channels, n_dim_mask=n_dim_mask, 
            embed_dim=embed_dim, use_offsets=use_offsets, has_expander=False
        )

    def forward(self, tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta=None, pos_rel_src=None):
        # 1. Run Preconditioner if Enabled
        if self.use_embedding:
            ndim_slice = -7
            struct_input = torch.cat(
                (torch.ones(len(tr), 1, dtype=tr.dtype, device=tr.device), tr[:, ndim_slice:]), 
                dim=1
            )
            g_emb = self.act_geom_init(self.film_geom_init(self.init_geom(struct_input), embed_context))
            g_emb = self.geom_layer1(g_emb, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
            g_emb = self.geom_layer2(g_emb, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)

            # Concatenate structural embedding with original observation slice
            tr = torch.cat((tr, g_emb), dim=-1)

        # 2. Main Observation Processing Stack
        tr = torch.cat((tr, mask), dim=-1)
        tr = self.act_init(self.film_init(self.init_trns(tr), embed_context))

        tr = self.layer1(tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
        tr = self.layer2(tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
        tr = self.layer3(tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)

        return tr


class BipartiteGraphOperator(MessagePassing):
    def __init__(self, ndim_in, ndim_out, ndim_edges=8, ndim_mask=4, embed_dim=10):
        super(BipartiteGraphOperator, self).__init__(aggr="add")

        # 1. Edge MLP: Evaluates travel-time misfit + 4D geometry
        # 3 (unit dir) + 4 (RBF gammas) + 1 (time) = 8 spatial-temporal features
        self.fc1 = nn.Sequential(
            nn.Linear(ndim_in + 8, ndim_in),
            nn.PReLU(),
            nn.Linear(ndim_in, ndim_in),
            nn.PReLU(),
        )

        # 2. Low-Rank Gating (Prevents synthetic memorization)
        self.mask_gate = nn.Sequential(
            nn.Linear(ndim_mask, 4),
            nn.PReLU(),
            nn.Linear(4, ndim_in),
            nn.Sigmoid(),
        )

        # 3. Dynamic Bandwidth Predictor driven by domain context embedding
        # Outputs 4 dynamic bandwidth offsets
        self.f_gamma = nn.Linear(embed_dim, 4)

        # Learnable baseline log-gammas (initialized near ~[0.05, 0.3, 0.8, 2.0])
        self.log_gamma_base = nn.Parameter(torch.tensor([-3.0, -1.2, -0.2, 0.7]).reshape(1, -1))

        # 4. LayerNorm standardizes feature distribution across channels
        self.norm = nn.LayerNorm(ndim_in)

        # 5. Final Projection
        self.fc2 = nn.Linear(ndim_in, ndim_out)
        self.activate_out = nn.PReLU()

    def forward(self, inpt, A_src_in_edges, mask, embed_context, n_sta=None, n_temp=None, num_target_nodes=None):
        """
        Args:
            inpt: [E, ndim_in]
            A_src_in_edges: PyG Data object containing edge_index and x
            mask: [E, ndim_mask]
            embed_context: [E, embed_dim] or [1, embed_dim] context embedding per edge or graph
        """
        # Target nodes (M) represent the factor graph space being stacked onto
        N = inpt.shape[0]
        if num_target_nodes is not None:
            M = num_target_nodes
        else:
            M = A_src_in_edges.edge_index[1].max().item() + 1 if A_src_in_edges.edge_index.numel() > 0 else 0

        # Step 1: Existential gate for active mask edges
        absolute_gate = mask.max(1, keepdims=True)[0]

        # Step 2: Dynamic Scale-Conditioned Gammas
        # Bounded offset in [-1.2, 1.2] ensures dynamic multipliers stay roughly in [0.3x, 3.3x]
        gamma_offset = 1.2 * torch.tanh(self.f_gamma(embed_context))
        gammas = torch.exp(self.log_gamma_base + gamma_offset)  # [E, 4] or [1, 4]

        # Step 3: Compute non-linear geometric features & phase routing
        norm_pos = torch.sqrt(torch.sum(A_src_in_edges.x[:, 0:3] ** 2, dim=1, keepdim=True) + 1e-8)
        
        # Exponential RBF decay conditioned on context
        rbf_decay = torch.exp(-1.0 * norm_pos * gammas)  # [E, 4]

        rel_pos = torch.cat(
            (A_src_in_edges.x[:, 0:3] / norm_pos, rbf_decay, A_src_in_edges.x[:, 3:4]),
            dim=1,
        )

        geo_features = self.fc1(torch.cat((inpt, rel_pos), dim=-1))
        phase_routing_vectors = self.mask_gate(mask)

        # Step 4: Gated message composition
        msg = absolute_gate * (phase_routing_vectors * geo_features)

        # Step 5: Perform raw physical stacking (Constructive Summing)
        stacked = self.propagate(A_src_in_edges.edge_index, size=(N, M), x=msg)

        # Step 6: Compute active degree E_i per target node (only counting active mask edges)
        target_indices = A_src_in_edges.edge_index[1]
        deg = torch.zeros((M, 1), device=stacked.device, dtype=stacked.dtype)
        deg.index_add_(0, target_indices, absolute_gate)

        # Step 7: Scale by 1 / sqrt(E_i) -> Keeps noise floor equal between core hubs & sparse nodes
        stacked_normalized = stacked / torch.sqrt(deg.clamp(min=1.0))

        # Step 8: Channel-wise standardization & final projection
        return self.activate_out(self.fc2(self.norm(stacked_normalized)))


# class SpatialAggregation(MessagePassing):
#     def __init__(self, in_channels, out_channels, embed_dim=10, scale_rel=1.0, n_global=5, n_hidden=30, zero_offsets=False):
#         super(SpatialAggregation, self).__init__(aggr='mean')

#         self.zero_offsets = zero_offsets
#         self.scale_rel = scale_rel

#         if not self.zero_offsets:
#             # Predict 1 global scale (alpha) + 5 per-frequency residuals (3 spatial + 2 temporal)
#             self.f_gamma = nn.Linear(embed_dim, 1 + 5)
            
#             # Base gammas: 3 spatial (0.1, 1.0, 5.0) + 2 temporal (0.5 [broad], 10.0 [sharp])
#             init_gammas = torch.tensor([0.1, 1.0, 5.0, 0.5, 10.0]).reshape(1, -1)
#             # init_gammas = torch.tensor([0.2, 1.0, 5.0, 0.5, 15.0]).reshape(1, -1)
#             self.log_gamma_base = nn.Parameter(torch.log(init_gammas))

#             # Edge dim: 3D dir (3) + Spatial RBF (3) + Temporal RBF (2) + Normalized dt (1) = 9
#             edge_dim = 9
#         else:
#             edge_dim = 0

#         # Feature transformations
#         self.fc1 = nn.Linear(in_channels + edge_dim + n_global, n_hidden)
#         self.fc2 = nn.Linear(n_hidden + in_channels, out_channels)
#         self.fglobal = nn.Linear(in_channels, n_global)

#         # FiLM Conditioning parameters
#         self.film_gamma = nn.Linear(embed_dim, n_hidden)
#         self.film_beta = nn.Linear(embed_dim, n_hidden)

#         self.activate1 = nn.PReLU()
#         self.activate2 = nn.PReLU()
#         self.activate3 = nn.PReLU()

#         # # FiLM identity initialization
#         # nn.init.ones_(self.film_gamma.weight)
#         # nn.init.zeros_(self.film_gamma.bias)
#         # nn.init.zeros_(self.film_beta.weight)
#         # nn.init.zeros_(self.film_beta.bias)

# 		# In __init__:
# 		nn.init.zeros_(self.film_gamma.weight)
# 		nn.init.zeros_(self.film_gamma.bias)
# 		nn.init.zeros_(self.film_beta.weight)
# 		nn.init.zeros_(self.film_beta.bias)

#     def forward(self, tr, embed_context, A_src, pos):
#         # 1. OPTIMIZATION: Compute context projections ONCE per forward pass
#         # Handle single graph (1D/2D single row) or batched contexts safely
#         ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)
        
#         film_g = self.film_gamma(ctx)
#         film_b = self.film_beta(ctx)

#         if not self.zero_offsets:
#             # Unified 4D relative position normalized by scale_rel
#             pos_rel = (pos[A_src[1]] - pos[A_src[0]]) / self.scale_rel
            
#             pos_rel_sp = pos_rel[:, 0:3]
#             pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp ** 2, dim=1, keepdim=True) + 1e-8)

#             pos_rel_tm = pos_rel[:, 3:4]
#             pos_norm_tm = torch.abs(pos_rel_tm)

#             # Decomposed Gammas: Global Alpha + Bounded Residuals
#             delta = self.f_gamma(ctx)
#             alpha = delta[:, :1]                           # Global zoom/density factor
#             residuals = 0.2 * torch.tanh(delta[:, 1:])     # Bounded shape adjustment [-0.2, +0.2]

#             gammas = torch.exp(self.log_gamma_base + alpha + residuals)
#             edge_gammas = gammas[A_src[0]] if gammas.shape[0] > 1 else gammas

#             # Anisotropic Spatial and Temporal Decays
#             spatial_decay = torch.exp(-1.0 * pos_norm_sp * edge_gammas[:, 0:3])
#             temporal_decay = torch.exp(-1.0 * pos_norm_tm * edge_gammas[:, 3:5])

#             # Construct 9D Edge Features
#             edge_attr = torch.cat((pos_rel_sp / pos_norm_sp, spatial_decay, temporal_decay, pos_rel_tm), dim=1)
#         else:
#             edge_attr = torch.zeros((A_src.shape[1], 0), dtype=tr.dtype, device=tr.device)

#         # Global feature pooling
#         global_feat = self.activate3(self.fglobal(tr)).mean(dim=0, keepdim=True)

#         # Expand FiLM vectors to match edges if multi-graph batch
#         film_g_edge = film_g[A_src[0]] if film_g.shape[0] > 1 else film_g
#         film_b_edge = film_b[A_src[0]] if film_b.shape[0] > 1 else film_b

#         aggr_out = self.propagate(
#             A_src, 
#             x=tr, 
#             edge_attr=edge_attr, 
#             global_feat=global_feat, 
#             film_g=film_g_edge, 
#             film_b=film_b_edge
#         )
        
#         out = torch.cat((tr, aggr_out), dim=-1)
#         return self.activate2(self.fc2(out))

#     def message(self, x_j, edge_attr, global_feat, film_g, film_b):
#         if not self.zero_offsets:
#             inputs = torch.cat((x_j, edge_attr, global_feat.expand(len(x_j), -1)), dim=-1)
#         else:
#             inputs = torch.cat((x_j, global_feat.expand(len(x_j), -1)), dim=-1)

#         h = self.fc1(inputs)

#         # Apply pre-projected FiLM Feature Modulation
#         # return self.activate1(h * film_g + film_b)
# 		return self.activate1(h * (1.0 + film_g) + film_b)



class SpatialAggregation(MessagePassing):
    def __init__(self, in_channels, out_channels, embed_dim=10, scale_rel=1.0, 
                 n_global=5, n_hidden=30, zero_offsets=False):
        super(SpatialAggregation, self).__init__(aggr='mean')

        self.zero_offsets = zero_offsets
        self.scale_rel = scale_rel

        if not self.zero_offsets:
            # Predict 1 global scale (alpha) + 5 per-frequency residuals (3 spatial + 2 temporal)
            self.f_gamma = nn.Linear(embed_dim, 1 + 5)
            nn.init.zeros_(self.f_gamma.weight)
            nn.init.zeros_(self.f_gamma.bias)
            
            # Base gammas: 3 spatial (0.1, 1.0, 5.0) + 2 temporal (0.5 [broad], 10.0 [sharp])
            init_gammas = torch.tensor([0.1, 1.0, 5.0, 0.5, 10.0]).reshape(1, -1)
            self.log_gamma_base = nn.Parameter(torch.log(init_gammas))

            # Edge dim: 3D dir (3) + Spatial RBF (3) + Temporal RBF (2) + Normalized dt (1) = 9
            edge_dim = 9
        else:
            edge_dim = 0

        # Feature transformations
        self.fc1 = nn.Linear(in_channels + edge_dim + n_global, n_hidden)
        self.fc2 = nn.Linear(n_hidden + in_channels, out_channels)
        self.fglobal = nn.Linear(in_channels, n_global)

        # FiLM Conditioning Block
        self.film = FiLM(embed_dim, n_hidden)

        self.activate1 = nn.PReLU()
        self.activate2 = nn.PReLU()
        self.activate3 = nn.PReLU()

    def forward(self, tr, embed_context, A_src, pos):
        # Ensure context is at least 2D [1, embed_dim]
        ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

        if not self.zero_offsets:
            # Unified 4D relative position normalized by scale_rel
            pos_rel = (pos[A_src[1]] - pos[A_src[0]]) / self.scale_rel
            
            pos_rel_sp = pos_rel[:, 0:3]
            pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp ** 2, dim=1, keepdim=True) + 1e-8)

            pos_rel_tm = pos_rel[:, 3:4]
            pos_norm_tm = torch.abs(pos_rel_tm)

            # Decomposed Gammas: Global Alpha + Bounded Residuals
            delta = self.f_gamma(ctx)
            alpha = delta[:, :1]                           # Global zoom/density factor
            residuals = 0.2 * torch.tanh(delta[:, 1:])     # Bounded shape adjustment [-0.2, +0.2]

			# Optional: Cap alpha shift to a max 3x scale factor change (~ exp(1.1))
			# alpha = 1.1 * torch.tanh(delta[:, :1])
			# residuals = 0.2 * torch.tanh(delta[:, 1:])

            gammas = torch.exp(self.log_gamma_base + alpha + residuals)
            edge_gammas = gammas[A_src[0]] if gammas.shape[0] > 1 else gammas

            # Anisotropic Spatial and Temporal Decays
            spatial_decay = torch.exp(-1.0 * pos_norm_sp * edge_gammas[:, 0:3])
            temporal_decay = torch.exp(-1.0 * pos_norm_tm * edge_gammas[:, 3:5])

            # Construct 9D Edge Features
            edge_attr = torch.cat((pos_rel_sp / pos_norm_sp, spatial_decay, temporal_decay, pos_rel_tm), dim=1)
        else:
            edge_attr = torch.zeros((A_src.shape[1], 0), dtype=tr.dtype, device=tr.device)

        # Global feature pooling
        global_feat = self.activate3(self.fglobal(tr)).mean(dim=0, keepdim=True)

        # Message Passing execution
        aggr_out = self.propagate(
            A_src, 
            x=tr, 
            edge_attr=edge_attr, 
            global_feat=global_feat, 
            embed_context=ctx
        )
        
        out = torch.cat((tr, aggr_out), dim=-1)
        return self.activate2(self.fc2(out))

    def message(self, x_j, edge_attr, global_feat, embed_context):
        if not self.zero_offsets:
            inputs = torch.cat((x_j, edge_attr, global_feat.expand(len(x_j), -1)), dim=-1)
        else:
            inputs = torch.cat((x_j, global_feat.expand(len(x_j), -1)), dim=-1)

        h = self.fc1(inputs)

        # Apply unified FiLM modulation and activation
        return self.activate1(self.film(h, embed_context))


class SpaceTimeDirect(nn.Module):
	def __init__(self, inpt_dim, out_channels):
		super(SpaceTimeDirect, self).__init__() #  "Max" aggregation.

		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.activate = nn.PReLU()

	def forward(self, inpts):

		return self.activate(self.f_direct(inpts))



# class SpaceTimeAttention(MessagePassing):
#     """Multi-Resolution Continuous Space-Time Interpolator.

#     Directly conditioned on physical domain scale embeddings for exact
#     multi-scale Gaussian rendering across a unified continuous spacetime manifold.
#     """

#     def __init__(
#         self,
#         inpt_dim,
#         out_channels,
#         n_dim,
#         n_latent,
#         embed_dim=10,
#         n_hidden=30,
#         n_heads=5,
#         scale_rel=1.0,
#         scale_time=1.0,
#         device="cuda",
#     ):
#         super(SpaceTimeAttention, self).__init__(node_dim=0, aggr="add")
#         self.n_heads = n_heads
#         self.n_latent = n_latent
#         self.out_channels = out_channels
#         self.scale_rel = scale_rel
#         self.scale_time = scale_time

#         # 1. Feature Value Transformation (fuses context features and scale embedding)
#         self.f_values = nn.Sequential(
#             nn.Linear(inpt_dim + embed_dim, n_heads * n_latent),
#             nn.PReLU(),
#         )

#         # 2. Dynamic Bandwidth Predictor
#         # Predicts spatial & temporal gamma offsets per head strictly from domain scale embedding
#         self.f_gamma = nn.Linear(embed_dim, n_heads * 2)

#         # Learnable baseline gammas initialized to log(1.0) = 0.0
#         self.log_gamma_base = nn.Parameter(torch.zeros(1, n_heads, 2))

#         # 3. Context Feature Score Modulation & Super-Resolution Gain Gate
#         self.f_feature_score = nn.Linear(inpt_dim + embed_dim, n_heads)
#         self.f_gain = nn.Sequential(
#             nn.Linear(inpt_dim + embed_dim, n_heads * n_latent),
#             nn.Sigmoid(),
#         )

#         # 4. Final Readout Projection
#         self.proj = nn.Linear(n_heads * n_latent, out_channels)
#         self.activate2 = nn.PReLU()

#         # Graph storage for fixed evaluation setups
#         self.fixed_edges = None
#         self.edge_features = None
#         self.use_fixed_edges = False

#     def _build_edge_attr(self, x_query, x_context, x_query_t, x_context_t, k=12):
#         # 4D Spacetime k-NN graph construction
#         edge_index = knn(
#             torch.cat((x_context / 1000.0, self.scale_time * x_context_t.reshape(-1, 1)), dim=1),
#             torch.cat((x_query / 1000.0, self.scale_time * x_query_t.reshape(-1, 1)), dim=1),
#             k=k,
#         ).flip(0)

#         # Relative spatial distance squared (normalized by isotropic scale_rel)
#         r_spatial_sq = torch.sum(
#             ((x_query[edge_index[1], 0:3] - x_context[edge_index[0], 0:3]) / self.scale_rel) ** 2,
#             dim=1,
#             keepdim=True,
#         )

#         # Relative temporal distance squared
#         # scale_time projects time into equivalent spatial distance, scale_rel normalizes the 4D metric
#         r_time_sq = (
#             (1000.0 * self.scale_time * (x_query_t[edge_index[1]].reshape(-1, 1) - x_context_t[edge_index[0]].reshape(-1, 1)))
#             / self.scale_rel
#         ) ** 2

#         edge_attr = torch.cat((r_spatial_sq, r_time_sq), dim=1)
#         return edge_index, edge_attr

#     def forward(self, inpts, x_query, x_context, x_query_t, x_context_t, embed_context, k=12, fixed_type=0):
#         if not self.use_fixed_edges:
#             edge_index, edge_attr = self._build_edge_attr(x_query, x_context, x_query_t, x_context_t, k=k)
#         else:
#             edge_index, edge_attr = self.fixed_edges, self.edge_features

#         # Ensure embed_context has shape [N_context, embed_dim] for PyG propagation mapping
#         if embed_context.dim() == 1:
#             embed_context = embed_context.unsqueeze(0).expand(x_context.shape[0], -1)
#         elif embed_context.shape[0] == 1:
#             embed_context = embed_context.expand(x_context.shape[0], -1)

#         # Message passing over bipartite graph (context nodes -> query nodes)
#         interpolated = self.propagate(
#             edge_index,
#             x=inpts,
#             embed=embed_context,
#             edge_attr=edge_attr,
#             size=(x_context.shape[0], x_query.shape[0]),
#         )

#         flattened = interpolated.view(x_query.shape[0], -1)
#         out = self.proj(flattened)
#         return self.activate2(out)

#     def message(self, x_j, embed_j, index, edge_attr):
#         # Concatenate context feature values with domain scale embedding
#         node_and_scale = torch.cat((x_j, embed_j), dim=-1)

#         # 1. Transform context features into Values & Scale-Conditioned Gains
#         value_embed = self.f_values(node_and_scale).view(-1, self.n_heads, self.n_latent)
#         gain = 0.5 + 2.0 * self.f_gain(node_and_scale).view(-1, self.n_heads, self.n_latent)
#         value_embed = value_embed * gain

#         # 2. Dynamic Gaussian Bandwidth (Driven strictly by scale/context embedding)
#         # Bounded offset in [-1.6, 1.6] ensures smooth multiplicative range ~[0.2x, 5.0x] base
#         gamma_offset = 1.6 * torch.tanh(self.f_gamma(embed_j).view(-1, self.n_heads, 2))
#         gammas = torch.exp(self.log_gamma_base + gamma_offset)

#         g_sp, g_tm = gammas[:, :, 0:1], gammas[:, :, 1:2]
#         r_sp_sq, r_tm_sq = edge_attr[:, 0:1].unsqueeze(1), edge_attr[:, 1:2].unsqueeze(1)

#         distance_logits = -1.0 * (g_sp * r_sp_sq + g_tm * r_tm_sq).squeeze(-1)

#         # 3. Context Feature Score Modulation & Softmax Attention
#         logits = distance_logits + self.f_feature_score(node_and_scale)
#         alpha = softmax(logits, index)

#         return alpha.unsqueeze(-1) * value_embed

#     def set_edges(self, x_query, x_context, x_query_t, x_context_t, k=12):
#         """Fixes graph structure and edge attributes for static/cached evaluation."""
#         edge_index, edge_attr = self._build_edge_attr(x_query, x_context, x_query_t, x_context_t, k=k)
#         self.fixed_edges = edge_index
#         self.edge_features = edge_attr
#         self.use_fixed_edges = True


# class SpaceTimeAttention(MessagePassing):
#     """Multi-Resolution Continuous Space-Time Interpolator.

#     Directly conditioned on physical domain scale embeddings for exact
#     multi-scale Gaussian rendering across an anisotropic 4D spacetime manifold.
#     """

#     def __init__(
#         self,
#         inpt_dim,
#         out_channels,
#         n_dim=4,
#         n_latent=30,
#         embed_dim=10,
#         n_heads=5,
#         scale_rel=1.0,
#         scale_time=1.0,
#     ):
#         super(SpaceTimeAttention, self).__init__(node_dim=0, aggr="add")
#         self.n_heads = n_heads
#         self.n_latent = n_latent
#         self.out_channels = out_channels
#         self.scale_rel = scale_rel
#         self.scale_time = scale_time

#         # 1. Feature Value Transformation (Modulated via FiLM)
#         self.f_values = nn.Linear(inpt_dim, n_heads * n_latent)
#         self.film_values = FiLM(embed_dim, n_heads * n_latent)
#         self.act_values = nn.PReLU()

#         # 2. Anisotropic Dynamic Bandwidth Predictor (3 Spatial + 1 Temporal = 4 dimensions)
#         # Predicts 1 Global Alpha + (4 * n_heads) per-head frequency residuals
#         self.f_gamma = nn.Linear(embed_dim, 1 + n_heads * 4)
#         nn.init.zeros_(self.f_gamma.weight)
#         nn.init.zeros_(self.f_gamma.bias)

#         # Base gammas initialized per head: shape [1, n_heads, 4]
#         # Heads start with log-spaced bandwidth sensitivities
#         init_gammas = torch.tensor([0.1, 1.0, 5.0, 10.0]).repeat(n_heads, 1).unsqueeze(0)
#         self.log_gamma_base = nn.Parameter(torch.log(init_gammas))

#         # 3. Context Feature Score Modulation & Super-Resolution Gain Gate
#         self.f_feature_score = nn.Linear(inpt_dim, n_heads)
#         self.film_score = FiLM(embed_dim, n_heads)

#         self.f_gain = nn.Linear(inpt_dim, n_heads * n_latent)
#         self.film_gain = FiLM(embed_dim, n_heads * n_latent)

#         # 4. Readout Projection
#         self.proj = nn.Linear(n_heads * n_latent, out_channels)
#         self.activate2 = nn.PReLU()

#         # Graph storage for fixed evaluation setups
#         self.fixed_edges = None
#         self.edge_features = None
#         self.use_fixed_edges = False

#     def _build_edge_attr(self, x_query, x_context, x_query_t, x_context_t, k=12):
#         # Scale time into the spatial coordinate space
#         ctx_4d = torch.cat((x_context / self.scale_rel, (self.scale_time * x_context_t).reshape(-1, 1) / self.scale_rel), dim=1)
#         qry_4d = torch.cat((x_query / self.scale_rel, (self.scale_time * x_query_t).reshape(-1, 1) / self.scale_rel), dim=1)

#         # Bipartite k-NN in normalized metric space
#         edge_index = knn(ctx_4d, qry_4d, k=k).flip(0)

#         # Anisotropic squared distances along x, y, z (3D spatial) and t (1D temporal)
#         diff_sp = (x_query[edge_index[1], 0:3] - x_context[edge_index[0], 0:3]) / self.scale_rel
#         diff_tm = (self.scale_time * (x_query_t[edge_index[1]] - x_context_t[edge_index[0]])).reshape(-1, 1) / self.scale_rel

#         # Edge feature shape: [Num_Edges, 4] -> (dx^2, dy^2, dz^2, dt^2)
#         edge_attr = torch.cat((diff_sp ** 2, diff_tm ** 2), dim=1)
#         return edge_index, edge_attr

#     def forward(self, inpts, x_query, x_context, x_query_t, x_context_t, embed_context, k=12):
#         if not self.use_fixed_edges:
#             edge_index, edge_attr = self._build_edge_attr(x_query, x_context, x_query_t, x_context_t, k=k)
#         else:
#             edge_index, edge_attr = self.fixed_edges, self.edge_features

#         ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

#         # Message passing over bipartite graph (context nodes -> query nodes)
#         interpolated = self.propagate(
#             edge_index,
#             x=inpts,
#             embed_context=ctx,
#             edge_attr=edge_attr,
#             size=(x_context.shape[0], x_query.shape[0]),
#         )

#         flattened = interpolated.view(x_query.shape[0], -1)
#         out = self.proj(flattened)
#         return self.activate2(out)

#     def message(self, x_j, embed_context_j, index, edge_attr):
#         # 1. Transform context features into Values & Scale-Conditioned Gains via FiLM
#         value_embed = self.act_values(self.film_values(self.f_values(x_j), embed_context_j))
#         value_embed = value_embed.view(-1, self.n_heads, self.n_latent)

#         gain = 0.5 + 2.0 * torch.sigmoid(self.film_gain(self.f_gain(x_j), embed_context_j))
#         gain = gain.view(-1, self.n_heads, self.n_latent)
#         value_embed = value_embed * gain

#         # 2. Anisotropic Dynamic Gaussian Bandwidth Gammas
#         delta = self.f_gamma(embed_context_j)
#         alpha = delta[:, :1].unsqueeze(-1)                           # Global zoom/scale factor [E, 1, 1]
#         residuals = 0.2 * torch.tanh(delta[:, 1:].view(-1, self.n_heads, 4)) # Bounded head shifts

#         # Gammas per edge, per head, per dimension [E, n_heads, 4]
#         gammas = torch.exp(self.log_gamma_base + alpha + residuals)

#         # Anisotropic Gaussian distance logits: - sum_d (gamma_d * dr_d^2)
#         # edge_attr: [E, 4] -> [E, 1, 4]
#         r_sq = edge_attr.unsqueeze(1)
#         distance_logits = -1.0 * torch.sum(gammas * r_sq, dim=-1)     # Shape: [E, n_heads]

#         # 3. Context Feature Score Modulation & Softmax Attention
#         score = self.film_score(self.f_feature_score(x_j), embed_context_j)
#         logits = distance_logits + score
#         alpha_attn = softmax(logits, index)                           # Shape: [E, n_heads]

#         return alpha_attn.unsqueeze(-1) * value_embed

#     def set_edges(self, x_query, x_context, x_query_t, x_context_t, k=12):
#         """Fixes graph structure and edge attributes for static/cached evaluation."""
#         edge_index, edge_attr = self._build_edge_attr(x_query, x_context, x_query_t, x_context_t, k=k)
#         self.fixed_edges = edge_index
#         self.edge_features = edge_attr
#         self.use_fixed_edges = True



class SpaceTimeAttention(MessagePassing):
    """Multi-Resolution Continuous Space-Time Gaussian Super-Resolution Interpolator.

    Predicts continuous space-time fields from sparse reference graphs, capable
    of super-resolving Gaussian peaks between reference nodes while maintaining 
    smooth manifold continuity.
    """

    def __init__(
        self,
        inpt_dim,
        out_channels,
        n_dim=4,
        n_latent=30,
        embed_dim=10,
        n_heads=5,
        scale_rel=1.0,
        scale_time=1.0,
    ):
        super(SpaceTimeAttention, self).__init__(node_dim=0, aggr="add")
        self.n_heads = n_heads
        self.n_latent = n_latent
        self.out_channels = out_channels
        self.scale_rel = scale_rel
        self.scale_time = scale_time

        # 1. Feature Value Transformation into Latent Space
        self.f_values = nn.Linear(inpt_dim, n_heads * n_latent)
        self.film_values = FiLM(embed_dim, n_heads * n_latent)
        self.act_values = nn.PReLU()

        # Super-resolution Gain Gate: Bounded range [0.5, 2.5]
        # Allows constructive amplitude recovery (up to 2.5x local node values) 
        # when query falls between sparse reference nodes.
        self.f_gain = nn.Linear(inpt_dim, n_heads * n_latent)
        self.film_gain = FiLM(embed_dim, n_heads * n_latent)

        # 2. Anisotropic Dynamic Gaussian Bandwidth Predictor (3 Spatial + 1 Temporal = 4D)
        self.f_gamma = nn.Linear(embed_dim, 1 + n_heads * 4)
        nn.init.zeros_(self.f_gamma.weight)
        nn.init.zeros_(self.f_gamma.bias)

        # Multi-frequency head initialization (Broad to Sharp)
        init_spatial = torch.logspace(-1, 0.7, steps=n_heads).unsqueeze(1).repeat(1, 3)  # [n_heads, 3]
        init_temporal = torch.logspace(-0.3, 1.0, steps=n_heads).unsqueeze(1)            # [n_heads, 1]
        init_gammas = torch.cat((init_spatial, init_temporal), dim=1).unsqueeze(0)      # [1, n_heads, 4]
        self.log_gamma_base = nn.Parameter(torch.log(init_gammas))

        # 3. Context Feature Score Modulation
        self.f_feature_score = nn.Linear(inpt_dim, n_heads)
        self.film_score = FiLM(embed_dim, n_heads)

        # 4. Final Readout Projection (Combines multi-head latent space into smooth output)
        self.proj = nn.Linear(n_heads * n_latent, out_channels)
        self.activate2 = nn.PReLU()

        # Graph storage for fixed evaluation setups
        self.fixed_edges = None
        self.edge_features = None
        self.use_fixed_edges = False

    def _build_edge_attr(self, x_query, x_context, x_query_t, x_context_t, k=12):
        ctx_4d = torch.cat((x_context / self.scale_rel, (self.scale_time * x_context_t).reshape(-1, 1) / self.scale_rel), dim=1)
        qry_4d = torch.cat((x_query / self.scale_rel, (self.scale_time * x_query_t).reshape(-1, 1) / self.scale_rel), dim=1)

        edge_index = knn(ctx_4d, qry_4d, k=k).flip(0)

        diff_sp = (x_query[edge_index[1], 0:3] - x_context[edge_index[0], 0:3]) / self.scale_rel
        diff_tm = (self.scale_time * (x_query_t[edge_index[1]] - x_context_t[edge_index[0]])).reshape(-1, 1) / self.scale_rel

        # Edge feature shape: [E, 4] -> (dx^2, dy^2, dz^2, dt^2)
        edge_attr = torch.cat((diff_sp ** 2, diff_tm ** 2), dim=1)
        return edge_index, edge_attr

    def forward(self, inpts, x_query, x_context, x_query_t, x_context_t, embed_context, k=12):
        if not self.use_fixed_edges:
            edge_index, edge_attr = self._build_edge_attr(x_query, x_context, x_query_t, x_context_t, k=k)
        else:
            edge_index, edge_attr = self.fixed_edges, self.edge_features

        ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

        # Message passing over bipartite graph (context -> query)
        interpolated = self.propagate(
            edge_index,
            x=inpts,
            embed_context=ctx,
            edge_attr=edge_attr,
            size=(x_context.shape[0], x_query.shape[0]),
        )

        # Reshape concatenated latent heads: [N_query, n_heads * n_latent]
        flattened = interpolated.view(x_query.shape[0], -1)
        
        # Readout projection into target channels with smooth activation
        out = self.proj(flattened)
        return self.activate2(out)

    def message(self, x_j, embed_context_j, index, edge_attr):
        # 1. Transform context features into Values & Scale-Conditioned Gains via FiLM
        value_embed = self.act_values(self.film_values(self.f_values(x_j), embed_context_j))
        value_embed = value_embed.view(-1, self.n_heads, self.n_latent)

        # Super-Resolution Gain Gate: [0.5, 2.5] via Sigmoid
        # Allows local amplitude amplification to reconstruct peaks between sparse nodes
        gain = 0.5 + 2.0 * torch.sigmoid(self.film_gain(self.f_gain(x_j), embed_context_j))
        gain = gain.view(-1, self.n_heads, self.n_latent)
        value_embed = value_embed * gain

        # 2. Anisotropic Dynamic Gaussian Bandwidth Gammas
        delta = self.f_gamma(embed_context_j)
        alpha = delta[:, :1].unsqueeze(-1)                                     # Global zoom/scale factor
        residuals = 0.2 * torch.tanh(delta[:, 1:].view(-1, self.n_heads, 4))   # Bounded shape adjustment

        gammas = torch.exp(self.log_gamma_base + alpha + residuals)            # [E, n_heads, 4]

        # Anisotropic Gaussian distance logits: - sum_d (gamma_d * dr_d^2)
        r_sq = edge_attr.unsqueeze(1)                                          # [E, 1, 4]
        distance_logits = -1.0 * torch.sum(gammas * r_sq, dim=-1)              # [E, n_heads]

        # 3. Score Modulation & Normalized Softmax Attention
        score = self.film_score(self.f_feature_score(x_j), embed_context_j)
        logits = distance_logits + score
        alpha_attn = softmax(logits, index)                                    # [E, n_heads]

        # Shape: [E, n_heads, n_latent]
        return alpha_attn.unsqueeze(-1) * value_embed

    def set_edges(self, x_query, x_context, x_query_t, x_context_t, k=12):
        edge_index, edge_attr = self._build_edge_attr(x_query, x_context, x_query_t, x_context_t, k=k)
        self.fixed_edges = edge_index
        self.edge_features = edge_attr
        self.use_fixed_edges = True



class SpaceTimeAttentionQuery(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, n_hidden = 30, n_heads = 5, kernel_sig_t = kernel_sig_t, locs_use = None, trv = None, ftrns2 = None, scale_rel = scale_rel, scale_time = scale_time):
		super(SpaceTimeAttentionQuery, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		self.f_queries = nn.Linear(n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_context = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.scale_time = scale_time ## 1 Second is 10 km
		self.ftrns2 = ftrns2
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		
		self.locs_use = locs_use # torch.Tensor(locs_use).to()
		self.trv = trv
		self.trv_out_fixed = None
		self.fixed_edges = None
		self.edge_features = None
		self.use_fixed_edges = False
		self.kernel_sig_t = kernel_sig_t
		# self.embed_misfit = lambda tval, ind, t, p:  torch.exp(-0.5*(tval[:,ind,p] - torch.Tensor(tpick[i1]).to(self.device))**2/(self.embed_t**2))
		# self.activate3 = nn.PReLU()

	def forward(self, inpts, x_query, x_context, x_query_t, x_context_t, locs_use, tpick, ipick, phase_label, k = 30): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		## First use all source points to determine which subset of stations we need travel times for (?). E.g., use Lipchitz constraints
		## (min and max bounds) that specify which stations have times within fraction of source times for each query.
		## Or just query travel times for the subset of unique stations
		if self.use_fixed_edges == True:
			trv_out = self.trv_out_fixed
		else:
			trv_out = self.trv(torch.Tensor(locs_use).to(tpick.device), self.ftrns2(x_query)) + x_query_t.unsqueeze(2) ## Use full travel times, as we check for stations from the full product

		# ipick_unique = torch.unique(ipick).long()
		i1 = torch.where(phase_label == 0)[0]
		i2 = torch.where(phase_label == 1)[0]

		misfit_time = torch.zeros((len(x_query), len(tpick), 4)).to(self.device)
		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/(self.kernel_sig_t**2))
		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/(self.kernel_sig_t**2))
		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/(self.kernel_sig_t**2))
		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/(self.kernel_sig_t**2))
		## Compute misfit times between all source and pick pairs
		## For each station and query pair, find nearest matching arrival in tpick.
		## Can either use relative time embedding on linear scale, or use message passing
		## layer to aggregate over all statons for each pick. E.g., we can readily measure all
		## misfits with trv_out[:,ipick_perm,0] - tpick[ipick_perm]
	
		## Determine unique station indices
		ipick_unique = np.unique(ipick.cpu().detach().numpy())
		tree_stations = cKDTree(ipick.cpu().detach().numpy().reshape(-1,1))
		len_ipick_unique = len(ipick_unique)
		edges_read_in = tree_stations.query_ball_point(ipick_unique.reshape(-1,1), r = 0)

		edges_source = np.hstack([np.array(list(edges_read_in[i])) for i in range(len_ipick_unique)])
		edges_trgt = np.hstack([ipick_unique[i]*np.ones(len(edges_read_in[i])) for i in range(len_ipick_unique)])
		edges_read_in = torch.Tensor(np.concatenate((edges_source.reshape(1,-1), edges_trgt.reshape(1,-1)), axis = 0)).long().to(self.device)
		embed_picks = scatter(misfit_time[edges_read_in[0]], edges_read_in[1], dim = 1, dim_size = len(locs_use_cart), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations

		if self.use_fixed_edges == False:
			edge_index = knn(torch.cat((x_context/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = k).flip(0)
			edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x
			# return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).reshape(len(x_query), -1))) # mean over different heads
			return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

		else:
			# edge_index = self.fixed_edges
			# edge_attr = self.edge_features
			# return self.activate2(self.proj(self.propagate(self.fixed_edges, x = inpts, edge_attr = self.edge_features, size = (x_context.shape[0], x_query.shape[0])).reshape(len(x_query), -1))) # mean over different heads
			return self.activate2(self.proj(self.propagate(self.fixed_edges, x = inpts, edge_attr = self.edge_features, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads
			
		# edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x
		# return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

	def message(self, x_j, index, edge_attr):

		## Why are there no queries in this layer
		query_embed = self.f_queries(edge_attr).view(-1, self.n_heads, self.n_latent)
		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((query_embed*context_embed).sum(-1)/self.scale)
		alpha = softmax(alpha, index)
		
		return alpha.unsqueeze(-1)*value_embed

	def set_edges(self, x_query, x_context, x_query_t, x_context_t, k = 30):
		edge_index = knn(torch.cat((x_context/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = k).flip(0)
		edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x
		self.fixed_edges = edge_index
		self.edge_features = edge_attr
		self.use_fixed_edges = True



class BipartiteGraphReadOutOperator(MessagePassing):
    """Bipartite Readout Operator (Source Factor Graph -> Product Graph).

    Transforms source factor features using scale-conditioned RBF kernel decay
    and reads out messages onto target product graph nodes.
    """

    def __init__(self, ndim_in, ndim_out, ndim_edges=8, embed_dim=10, n_gammas=4):
        # Aggregating from Source Factor (edge_index[0]) to Product Graph (edge_index[1])
        super(BipartiteGraphReadOutOperator, self).__init__(aggr="add")

        self.n_gammas = n_gammas
        self.embed_dim = embed_dim

        # Geometry features = 3 (unit vector) + n_gammas (RBF) + 1 (time) = 8 total
        ndim_geo_pos = 3 + n_gammas + 1

        # 2-layer MLP for non-linear association evaluation
        self.fc1 = nn.Sequential(
            nn.Linear(ndim_in + ndim_geo_pos, ndim_in),
            nn.PReLU(),
            nn.Linear(ndim_in, ndim_in),
            nn.PReLU(),
        )

        # Dynamic Bandwidth Predictor driven by context embedding
        self.f_gamma = nn.Linear(embed_dim, n_gammas)

        # Learnable baseline log-gammas (initialized near ~[0.05, 0.3, 0.8, 2.0])
        self.log_gamma_base = nn.Parameter(
            torch.tensor([-3.0, -1.2, -0.2, 0.7]).reshape(1, -1)
        )

        # Readout Projection
        self.fc2 = nn.Linear(ndim_in, ndim_out)
        self.activate_out = nn.PReLU()

    def forward(
        self,
        inpt,
        A_Lg_in_srcs,
        mask,
        embed_context,
        n_sta=None,
        n_temp=None,
        num_target_nodes=None,
    ):
        """Args:

        inpt: [N_src, ndim_in] Source node features
        A_Lg_in_srcs: PyG Data object containing edge_index [2, E] and x [E, 4]
        mask: [N_src, ndim_mask] or [E, ndim_mask] Source/edge prediction mask
        embed_context: [E, embed_dim] or [1, embed_dim] Context/scale embedding
        num_target_nodes: Optional explicit scalar M for target product graph size
        """
        # Safe determination of N (source nodes) and M (target product graph nodes)
        N = inpt.shape[0]
        if num_target_nodes is not None:
            M = num_target_nodes
        else:
            M = (
                A_Lg_in_srcs.edge_index[1].max().item() + 1
                if A_Lg_in_srcs.edge_index.numel() > 0
                else 0
            )

        # 1. Dynamic Scale-Conditioned Gammas
        # Bounded offset in [-1.2, 1.2] keeps multipliers safely in ~[0.3x, 3.3x] range
        gamma_offset = 1.2 * torch.tanh(self.f_gamma(embed_context))
        gammas = torch.exp(self.log_gamma_base + gamma_offset)  # [E, n_gammas] or [1, n_gammas]

        # 2. Compute spatial unit direction and RBF distance decay
        norm_pos = torch.sqrt(
            torch.sum(A_Lg_in_srcs.x[:, 0:3] ** 2, dim=1, keepdim=True) + 1e-8
        )
        rbf_decay = torch.exp(-1.0 * norm_pos * gammas)

        rel_pos = torch.cat(
            (
                A_Lg_in_srcs.x[:, 0:3] / norm_pos,
                rbf_decay,
                A_Lg_in_srcs.x[:, 3:4],
            ),
            dim=1,
        )

        # 3. Propagate message from Source Factor (edge_index[0]) to Product Graph (edge_index[1])
        out = self.propagate(
            A_Lg_in_srcs.edge_index,
            size=(N, M),
            x=inpt,
            edge_attr=rel_pos,
            mask=mask,
        )

        # 4. Mask mapped to sending source nodes (edge_index[0])
        # Handles mask whether passed as per-node [N, dim] or per-edge [E, dim]
        if mask.dim() > 1 and mask.shape[0] == N:
            out_mask = mask[A_Lg_in_srcs.edge_index[0]]
        else:
            out_mask = mask

        return self.activate_out(self.fc2(out)), out_mask

    def message(self, x_j, mask_j, edge_attr):
        # x_j: Source node features mapped to edges [E, ndim_in]
        # mask_j: Source mask mapped to edges [E, ndim_mask]
        # edge_attr: Dynamic 8D spatio-temporal geometric feature vector [E, 8]
        return (mask_j + 0.01) * self.fc1(torch.cat((x_j, edge_attr), dim=-1))


# class DataAggregationAssociation(nn.Module):
#     """
#     Association Phase (Decoder) Block.
#     Processes the unpooled source likelihoods alongside encoder latents and masks
#     to perform final source-receiver association predictions.
#     """
#     def __init__(self, in_channels, out_channels, n_hidden=30, n_dim_latent=30, 
#                  n_dim_mask=5, embed_dim=10, use_offsets=True):
#         super().__init__()

#         # Input vector: Unpooled Source Features (in_channels) + Encoder Latents (n_dim_latent) + Masks (n_dim_mask + 1)
#         total_in_dim = in_channels + n_dim_latent + (n_dim_mask + 1)
        
#         self.init_trns = nn.Linear(total_in_dim, n_hidden)
#         self.film_init = FiLM(embed_dim, n_hidden)
#         self.act_init = nn.PReLU()

#         # Re-use core DataAggregationLayer without Expander edges
#         self.layer1 = DataAggregationLayer(
#             in_channels=n_hidden,
#             out_channels=n_hidden,
#             n_dim_mask=n_dim_mask + 1,  # Includes mask_out_1
#             embed_dim=embed_dim,
#             use_offsets=use_offsets,
#             has_expander=False  # Expander edges strictly omitted for association
#         )

#         self.layer2 = DataAggregationLayer(
#             in_channels=2 * n_hidden,
#             out_channels=out_channels,
#             n_dim_mask=n_dim_mask + 1,
#             embed_dim=embed_dim,
#             use_offsets=use_offsets,
#             has_expander=False
#         )

#     def forward(self, s, x_latent, mask_out_1, mask, A_in_sta, A_in_src, embed_context, 
#                 pos_rel_sta=None, pos_rel_src=None):
#         # 1. Combine Masks and Latents
#         combined_mask = torch.cat((mask, mask_out_1), dim=-1)
#         x = torch.cat((s, x_latent, combined_mask), dim=-1)

#         # 2. Project and Condition
#         x = self.act_init(self.film_init(self.init_trns(x), embed_context))

#         # 3. Association Graph Convolutions
#         x = self.layer1(x, combined_mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
#         x = self.layer2(x, combined_mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)

#         return x


class DataAggregationAssociation(nn.Module):
    """
    Association Phase (Decoder) Block built using standard DataAggregationLayers.
    Replaces DataAggregationAssociationPhase with modular, per-layer gamma learning.
    """
    def __init__(self, in_channels, out_channels, n_hidden=30, n_dim_latent=30, 
                 n_dim_mask=4, embed_dim=10, use_offsets=True):
        super().__init__()

        # Input: Unpooled Features (s) + Encoder Latents (x_latent) + Mask + Source Mask (mask_out_1)
        total_mask_dim = n_dim_mask + 1  # Original observation mask + source likelihood mask
        total_in_dim = in_channels + n_dim_latent + total_mask_dim

        self.init_trns = nn.Linear(total_in_dim, n_hidden)
        self.film_init = FiLM(embed_dim, n_hidden)
        self.act_init = nn.PReLU()

        # Association Layer 1 (Independent per-layer gammas, no expander edges needed)
        self.layer1 = DataAggregationLayer(
            in_channels=n_hidden,
            out_channels=n_hidden,
            n_dim_mask=total_mask_dim,
            embed_dim=embed_dim,
            use_offsets=use_offsets,
            has_expander=False
        )

        # Association Layer 2 (Independent per-layer gammas, no expander edges needed)
        self.layer2 = DataAggregationLayer(
            in_channels=2 * n_hidden,
            out_channels=out_channels,
            n_dim_mask=total_mask_dim,
            embed_dim=embed_dim,
            use_offsets=use_offsets,
            has_expander=False
        )

    def forward(self, s, x_latent, mask_out_1, mask, A_in_sta, A_in_src, embed_context, 
                pos_rel_sta=None, pos_rel_src=None):
        # 1. Combine Masks and Latents
        combined_mask = torch.cat((mask, mask_out_1), dim=-1)
        x = torch.cat((s, x_latent, combined_mask), dim=-1)

        # 2. Project and FiLM condition
        x = self.act_init(self.film_init(self.init_trns(x), embed_context))

        # 3. Association Graph Convolutions with Per-Layer Gammas
        x = self.layer1(x, combined_mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
        x = self.layer2(x, combined_mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)

        return x

## Note: can maybe reduce dilate scale and scale_misfit, as the default kernel_sig_t is likely larger
## Can also maybe reduce the scaling of eps

class ArrivalEmbedding(MessagePassing):
	def __init__(self, ndim_arv_in, ndim_out, n_hidden = 20, n_dim_embed = 30, n_phase_embed = 5, embed_vector_dim = 10, ndim_out_src = 1, scale_rel = scale_rel, k_spc_edges = k_spc_edges, kernel_sig_t = kernel_sig_t, use_phase_types = use_phase_types, scale_time = scale_time, min_thresh = 0.01, trv = None, ftrns2 = None, device = 'cuda'):
		# super(SourceArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean
		super(ArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean

		## Goal of this module is just to implement Bipartite aggregation of each source query - pick pair, of their misfits,
		## and while aggregating over the relevant nodes of the (subgraph) Cartesian product
		self.ftrns2 = ftrns2
		self.trv = trv
		self.use_phase_types = use_phase_types
		self.kernel_sig_t = kernel_sig_t
		self.min_thresh = min_thresh
		self.scale_time = scale_time
		self.scale_rel = scale_rel
		self.k_spc_edges = k_spc_edges
		self.device = device
		self.dilate_scale = 2.0 # 3.0
		self.scale_misfit = 2.0 # 3.0
		# self.null_embed = nn.Parameter(torch.randn(1, 1, n_hidden).to(device) * 0.01) # .to(device)
		# self.null_embed = nn.Parameter(torch.zeros(1, 1, n_hidden).to(device)) # .to(device)
		self.null_embed = nn.Parameter(torch.zeros(1, 1, n_hidden)) # .to(device)

		n_phase_types = 2
		n_phase_embed = 5
		# self.phase_embed = nn.Parameter(torch.randn(n_phase_types, n_phase_embed) * 0.01).to(device)
		self.phase_embed = nn.Embedding(n_phase_types, n_phase_embed)
		self.fc1 = nn.Sequential(nn.Linear(ndim_arv_in + 12 + 10 + 11 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.fc2 = nn.Sequential(nn.Linear(ndim_arv_in + 6 + 3 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.fc3 = nn.Sequential(nn.Linear(ndim_arv_in + 6 + 3 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.ioffset = torch.tensor([-1, 0], dtype=torch.long, device=self.device)

		# self.w_gamma1 = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
		# self.w_gamma2 = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
		# self.w_gamma3 = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
		# self.w_gamma4 = nn.Parameter(torch.tensor([0.1, 0.5, 2.0]).reshape(1, -1))

		self.w_gamma1_time = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
		self.w_gamma2_time = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))



		# Predict scale-conditioned gamma offsets from domain context
		self.f_gamma1 = nn.Linear(embed_vector_dim, 4)
		self.f_gamma2 = nn.Linear(embed_vector_dim, 4)
		self.f_gamma3 = nn.Linear(embed_vector_dim, 4)
		self.f_gamma4 = nn.Linear(embed_vector_dim, 4)
		self.log_gamma_base1 = nn.Parameter(torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1)))
		self.log_gamma_base2 = nn.Parameter(torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1)))
		self.log_gamma_base3 = nn.Parameter(torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1)))
		self.log_gamma_base4 = nn.Parameter(torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1)))

		self.f_gamma_time1 = nn.Linear(embed_vector_dim, 3)
		self.f_gamma_time2 = nn.Linear(embed_vector_dim, 4)
		self.f_gamma_time3 = nn.Linear(embed_vector_dim, 4)
		self.log_gamma_base_time1 = nn.Parameter(torch.log(torch.tensor([0.1, 0.5, 2.0]).reshape(1, -1)))
		self.log_gamma_base_time2 = nn.Parameter(torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1)))
		self.log_gamma_base_time3 = nn.Parameter(torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1)))

		# self.w_gamma3_time = nn.Parameter(torch.tensor([[0.1, 0.5, 2.0]).reshape(1, -1))
		# self.w_gamma4_time = nn.Parameter(torch.tensor([[0.1, 0.5, 2.0]).reshape(1, -1))
		
		# self.w_gamma3_time = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
	
		# self.fc1_src = nn.Sequential(nn.Linear(ndim_arv_in + 12 + 10 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		# self.fc2_src = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		# self.fc3_src = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features

		self.fc_merge = nn.Sequential(nn.Linear(3*n_hidden, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, ndim_out))
		# self.fc_merge_src = nn.Sequential(nn.Linear(3*n_hidden, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, ndim_out_src))

	def forward(self, x, x_context_cart, x_context_t, x_query_cart, x_query_t, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, embed_context, trv_out = None): # reference k nearest spatial points

		if trv_out is None:
			trv_out = self.trv(self.ftrns2(locs_use_cart), self.ftrns2(x_query_cart)) + x_query_t.reshape(-1, 1, 1) ## Use full travel times, as we check for stations from the full product
		else: 
			trv_out = trv_out + x_query_t.reshape(-1, 1, 1) ## Is this being applied outside this layer?

		if self.use_phase_types == False:
			phase_label = phase_label*0.0

		# ipick_unique = torch.unique(ipick).long()
		i1 = torch.where(phase_label == 0)[0]
		i2 = torch.where(phase_label == 1)[0]

		## Note: computing misfit times but not even using them other than for mask
		misfit_time = torch.zeros((len(x_query_cart), len(tpick), 4)).to(self.device) ## Question: is it necessary to produce these pairwise misfits? Can we focus on the pairs that "likely" have arrival times within threshold (e.g., bound min and max times based on distances between src reciever first, before computing travel times)
		# misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		# misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		# misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		# misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))

		tpick = tpick if isinstance(tpick, torch.Tensor) else torch.as_tensor(tpick, device=self.device)
		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - tpick[i1])**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - tpick[i2])**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - tpick)**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - tpick)**2/((self.dilate_scale*self.kernel_sig_t)**2))
				
		## Can compute these degree vectors outside of loop
		degree_srcs = degree(A_src_in_sta[1], num_nodes = len(x_context_cart), dtype = torch.long)
		cum_degree_srcs = torch.cat((torch.zeros(1).to(self.device), torch.cumsum(degree_srcs, dim = 0)[0:-1]), dim = 0).long()
		## Should check if minimal degree srcs really are accessing nearest stations
		mask_misfit_time = misfit_time.max(2).values > self.min_thresh ## Save this, so can use as mask in the attention layer
		isrc, iarv = torch.where(mask_misfit_time == 1)

		## Build src-src indices (may or may not use the edge feature of source query to source node offsets)
		edge_index = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query_cart/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = self.k_spc_edges).flip(0).contiguous()

		# Build a single flattened arange from size = sum(idx)
		deg_slice = degree_srcs[edge_index[0]]
		assert(deg_slice.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
		inc_inds = torch.arange(deg_slice.sum()).long().to(self.device)
		inc_inds = inc_inds - torch.repeat_interleave(torch.cumsum(deg_slice, dim = 0) - deg_slice, deg_slice)
		nodes_of_product = cum_degree_srcs[edge_index[0]].repeat_interleave(degree_srcs[edge_index[0]]) + inc_inds
		ind_query = torch.arange(len(x_query_cart)).long().to(self.device).repeat_interleave(scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)
		sta_src_pairs = A_src_in_sta[:, nodes_of_product]
		## Query_vals is shaped based on nodes_of_product. So when we aggregate or want to extract Cartesian product node features, we can use these.

		# k_matches = knn(sta_src_pairs.T, torch.cat((ipick[iarv].reshape(-1,1), ))
		query_vals = torch.cat((sta_src_pairs[0].reshape(-1,1), ind_query.reshape(-1,1)), dim = 1).long() # .float()
		pick_vals = torch.cat((ipick[iarv].reshape(-1,1), isrc.reshape(-1,1)), dim = 1).long() # .float()

		## Note: query_vals represents the pairs of station and query inds
		## pick_vals represents the pairs of station and query inds
		hash_picks, hash_queries = hash_rows(pick_vals), hash_rows(query_vals) ## Do not define directly if only using one mask below
		mask_picks = torch.isin(hash_picks, hash_queries) # set(map(tuple, l1))
		mask_queries = torch.isin(hash_queries, hash_picks) # set(map(tuple, l1))
		iwhere_picks = torch.where(mask_picks == 1)[0] ## Not used
		iwhere_query = torch.where(mask_queries == 1)[0]
		# assert(torch.abs(query_vals[iwhere_query] - pick_vals[knn(pick_vals, query_vals[iwhere_query], k = 1)[1]]).max() == 0)
		# assert(torch.abs(pick_vals[iwhere_picks] - query_vals[knn(query_vals, pick_vals[iwhere_picks], k = 1)[1]]).max() == 0)
		## The point of query vals is these are the nodes on the Cartesian product we are accessing and aggregating across.
		## How can we "read into" these nodes, or match to these nodes, for all possible (> min thresh) pick vals.
		## Can we use degrees or cumulative degrees of query vals to directly read in? Can we catch cases where the pick
		## has no match (e.g., read in, but then find mis-match of values and remove?)
		# print('Time %0.4f'%(time.time() - st))

		# print('Time %0.4f'%(time.time() - st))
		sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
		ind_extract = torch.searchsorted(sorted_hash_picks, hash_queries[iwhere_query])
		valid_ind = (ind_extract < len(sorted_hash_picks)) & (sorted_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)] == hash_queries[iwhere_query])
		inds_queries_to_picks = order_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)][valid_ind]
		assert(valid_ind.sum() == len(valid_ind))

		# use_checks = True
		# if use_checks == True:
		# 	## For a random set of queries, check if have correct edges
		# 	n_check = 30
		# 	for n in range(n_check):
		# 		i0 = np.random.choice(len(x_query))
		# 		e1 = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query_cart/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1)[i0,:].reshape(1,-1), k = self.k_spc_edges).flip(0).contiguous()

		## Compute features
		misfit_rel_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - tlatent[nodes_of_product[iwhere_query]]
		misfit_query_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - trv_out[query_vals[iwhere_query,1], ipick[iarv[inds_queries_to_picks]], :]
		# misfit_rel_time = torch.cat((torch.exp(-0.5*(misfit_rel_time**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_rel_time)), dim = 1)
		# misfit_query_time = torch.cat((torch.exp(-0.5*(misfit_query_time**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_query_time)), dim = 1)

		misfit_rel_time = torch.cat((torch.exp(-1.0*torch.abs(misfit_rel_time)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_rel_time)), dim = 1)
		misfit_query_time = torch.cat((torch.exp(-1.0*torch.abs(misfit_query_time)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_query_time)), dim = 1)

		offset_src_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_query_cart[query_vals[iwhere_query,1]])/(10.0*self.scale_rel)
		offset_ref_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]],:])/(10.0*self.scale_rel)

		## Distances between reference nodes and query (including time offsets)
		offset_ref_src = (x_query_cart[query_vals[iwhere_query,1]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]]])/(1.0*self.scale_rel)
		# offset_ref_src_t = (x_query_t[query_vals[iwhere_query,1]].reshape(-1,1) - x_context_t[A_src_in_sta[1,nodes_of_product[iwhere_query]]].reshape(-1,1))/(1.0*self.scale_time)
		offset_ref_src_t = 1000.0*self.scale_time*(x_query_t[query_vals[iwhere_query,1]].reshape(-1,1) - x_context_t[A_src_in_sta[1,nodes_of_product[iwhere_query]]].reshape(-1,1))/(3.0*self.scale_rel)

		eps_time = 1e-8
		offset_src_sta_norm = torch.norm(offset_src_sta, dim = 1, keepdim = True).clamp(min=eps_time)
		offset_ref_sta_norm = torch.norm(offset_ref_sta, dim = 1, keepdim = True).clamp(min=eps_time)
		offset_ref_src_norm = torch.norm(offset_ref_src, dim = 1, keepdim = True).clamp(min=eps_time)

		## Src to ref are not usually large distances so use one kernel radius
		# offset_src_sta_norm_kernel = torch.exp(-1.0*torch.abs(offset_src_sta_norm)/(3.0))
		# offset_ref_src_norm_kernel = torch.exp(-1.0*torch.abs(offset_ref_src_norm)/(1.0))
		# offset_ref_sta_norm_kernel = torch.exp(-1.0*torch.abs(offset_ref_sta_norm)/(3.0))

		# offset_src_sta_norm_kernel = torch.cat((offset_src_sta/offset_src_sta_norm, torch.tanh(-1.0*offset_src_sta_norm*F.softplus(self.w_gamma1))), dim = 1)
		# offset_ref_sta_norm_kernel = torch.cat((offset_ref_sta/offset_ref_sta_norm, torch.tanh(-1.0*offset_ref_sta_norm*F.softplus(self.w_gamma2))), dim = 1)
		# offset_ref_src_norm_kernel = torch.cat((offset_ref_src/offset_ref_src_norm, torch.tanh(-1.0*offset_ref_src_norm*F.softplus(self.w_gamma3))), dim = 1)
	
		gammas1 = torch.exp(self.log_gamma_base1 + 1.6 * torch.tanh(self.f_gamma1(embed_context)))	
		gammas2 = torch.exp(self.log_gamma_base2 + 1.6 * torch.tanh(self.f_gamma2(embed_context)))	
		gammas3 = torch.exp(self.log_gamma_base3 + 1.6 * torch.tanh(self.f_gamma3(embed_context)))	

		# 3. Compute RBF distance decay banks
		rbf_src_sta = torch.exp(-1.0 * offset_src_sta_norm * gammas1)
		rbf_ref_sta = torch.exp(-1.0 * offset_ref_sta_norm * gammas2)
		rbf_ref_src = torch.exp(-1.0 * offset_ref_src_norm * gammas3)

		# 4. Construct rich 8D spatio-temporal features: [unit_dir (3D), RBF bank (4D), dt (1D)]
		feat_src_sta = torch.cat((offset_src_sta/offset_src_sta_norm, rbf_src_sta), dim=-1)
		feat_ref_sta = torch.cat((offset_ref_sta/offset_ref_sta_norm, rbf_ref_sta), dim=-1)
		feat_ref_src = torch.cat((offset_ref_src/offset_ref_src_norm, rbf_ref_src), dim=-1)


		# offset_ref_src_norm_kernel_t = torch.cat((torch.exp(-1.0*torch.abs(offset_ref_src_t)/(1.0)).reshape(-1,1), torch.sign(offset_ref_src_t).reshape(-1,1)), dim = 1)
		# offset_ref_src_norm_kernel_t = torch.cat((torch.exp(-1.0*torch.abs(offset_ref_src_t)/(1.0)).reshape(-1,1), torch.sign(offset_ref_src_t).reshape(-1,1)), dim = 1)
	
		# offset_ref_src_norm_kernel_t = torch.cat((offset_ref_src_t, torch.exp(-1.0*torch.abs(offset_ref_src_t).reshape(-1,1)*F.softplus(self.w_gamma4))), dim = 1)
		gammas1_time = torch.exp(self.log_gamma_base_time1 + 1.6 * torch.tanh(self.f_gamma_time1(embed_context)))	
		rbf_time = torch.exp(-1.0 * offset_ref_src_t * gammas1_time)
		feat_time = torch.cat((offset_ref_src_t, rbf_time), dim=-1)

		# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, self.phase_embed(phase_label[iarv[inds_queries_to_picks]].reshape(-1).long())), dim = 1)
		inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, feat_src_sta, feat_ref_sta, feat_ref_src, feat_time, self.phase_embed(phase_label[iarv[inds_queries_to_picks]].reshape(-1).long())), dim = 1)
		
		aggregate_product = scatter(self.fc1(inpt_aggregate), inds_queries_to_picks, dim = 0, dim_size = len(iarv), reduce = 'mean') ## Can consider
		# print('T2 %0.4f'%(time.time() - t1))
		# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], x_embed_trns[query_vals[iwhere_query,1]], misfit_rel_time, misfit_query_time, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
		# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
		## Note: could first transfrom the features: misfit_rel_time, misfit_query_time, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel seperately from embed
		## For increased stability of merging with the embeddings
		
		use_time_based_embedding = True
		if use_time_based_embedding == True:

			min_time_shift = tlatent.amin()
			max_time_offset = (tlatent.amax() - min_time_shift)*2.5
			query_time = ((tpick - min_time_shift) + max_time_offset*ipick).reshape(-1,1)
			val_sort_p, ind_sort_p = torch.sort((tlatent[:,0] - min_time_shift) + max_time_offset*A_src_in_sta[0]) ## Could do these steps outside the training loop
			val_sort_s, ind_sort_s = torch.sort((tlatent[:,1] - min_time_shift) + max_time_offset*A_src_in_sta[0])
			ind_extract_p = torch.searchsorted(val_sort_p, (tpick - min_time_shift) + max_time_offset*ipick)
			ind_extract_s = torch.searchsorted(val_sort_s, (tpick - min_time_shift) + max_time_offset*ipick)

			iarg_p = torch.argmin(torch.abs(torch.cat((val_sort_p[torch.clamp(ind_extract_p - 1, min = 0)].reshape(-1,1), val_sort_p[torch.clamp(ind_extract_p, max = len(val_sort_p) - 1)].reshape(-1,1)), dim = 1) - query_time), dim = 1)
			iarg_s = torch.argmin(torch.abs(torch.cat((val_sort_s[torch.clamp(ind_extract_s - 1, min = 0)].reshape(-1,1), val_sort_s[torch.clamp(ind_extract_s, max = len(val_sort_s) - 1)].reshape(-1,1)), dim = 1) - query_time), dim = 1)
			# ioffset = torch.Tensor([-1, 0]).long().to(self.device)
			ind_grab_p = ind_sort_p[ind_extract_p + self.ioffset[iarg_p]] ## For each pick, the nearest arrival time of the nodes of the product
			ind_grab_s = ind_sort_s[ind_extract_s + self.ioffset[iarg_s]] ## (Must confirm station indices are identical and mask if not)
			sta_match_p = (A_src_in_sta[0,ind_grab_p] == ipick)
			sta_match_s = (A_src_in_sta[0,ind_grab_s] == ipick)

			# print('T4 %0.4f'%(time.time() - t1))
			edge_index_p = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1)[A_src_in_sta[1, ind_grab_p]], k = self.k_spc_edges).flip(0).contiguous()
			edge_index_s = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1)[A_src_in_sta[1, ind_grab_s]], k = self.k_spc_edges).flip(0).contiguous()

			# Build a single flattened arange from size = sum(idx)
			deg_slice_p = degree_srcs[edge_index_p[0]]
			deg_slice_s = degree_srcs[edge_index_s[0]]
			assert(deg_slice_p.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
			assert(deg_slice_s.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
			inc_inds_p = torch.arange(deg_slice_p.sum()).long().to(self.device)
			inc_inds_p = inc_inds_p - torch.repeat_interleave(torch.cumsum(deg_slice_p, dim = 0) - deg_slice_p, deg_slice_p)

			inc_inds_s = torch.arange(deg_slice_s.sum()).long().to(self.device)
			inc_inds_s = inc_inds_s - torch.repeat_interleave(torch.cumsum(deg_slice_s, dim = 0) - deg_slice_s, deg_slice_s)

			nodes_of_product_p = cum_degree_srcs[edge_index_p[0]].repeat_interleave(degree_srcs[edge_index_p[0]]) + inc_inds_p
			nodes_of_product_s = cum_degree_srcs[edge_index_s[0]].repeat_interleave(degree_srcs[edge_index_s[0]]) + inc_inds_s

			ind_query_p = torch.arange(len(tpick)).long().to(self.device).repeat_interleave(scatter(deg_slice_p, edge_index_p[1], dim = 0, dim_size = len(tpick), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)
			ind_query_s = torch.arange(len(tpick)).long().to(self.device).repeat_interleave(scatter(deg_slice_s, edge_index_s[1], dim = 0, dim_size = len(tpick), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)

			sta_src_pairs_p = A_src_in_sta[:, nodes_of_product_p]
			sta_src_pairs_s = A_src_in_sta[:, nodes_of_product_s]

			## Note: do we use all the pick_vals or just the pick_vals with positive entries, like above. We have actually created these queries based on "all" the picks
			# k_matches = knn(sta_src_pairs.T, torch.cat((ipick[iarv].reshape(-1,1), ))
			query_vals_p = torch.cat((sta_src_pairs_p[0].reshape(-1,1), ind_query_p.reshape(-1,1)), dim = 1).long() # .float()
			query_vals_s = torch.cat((sta_src_pairs_s[0].reshape(-1,1), ind_query_s.reshape(-1,1)), dim = 1).long() # .float()

			pick_vals_time = torch.cat((ipick.reshape(-1,1), torch.arange(len(ipick)).reshape(-1,1).to(self.device)), dim = 1).long() # .float()
			hash_picks_time = hash_rows(pick_vals_time)
			hash_queries_p, hash_queries_s = hash_rows(query_vals_p), hash_rows(query_vals_s)
			mask_queries_p = torch.isin(hash_queries_p, hash_picks_time) # set(map(tuple, l1))
			mask_queries_s = torch.isin(hash_queries_s, hash_picks_time) # set(map(tuple, l1))
			iwhere_query_p = torch.where(mask_queries_p == 1)[0]
			iwhere_query_s = torch.where(mask_queries_s == 1)[0]
			# print('T5 %0.4f'%(time.time() - t1))
			# assert(torch.abs(ipick[ind_query_p] - A_src_in_sta[0, nodes_of_product_p]).max() == 0)
			# assert(torch.abs(ipick[ind_query_s] - A_src_in_sta[0, nodes_of_product_s]).max() == 0)
			## Now for each pick and subset of nodes of product need to find matched station
			
			# print('Time %0.4f'%(time.time() - st))
			sorted_hash_picks_time, order_hash_picks_time = torch.sort(hash_picks_time)
			ind_extract_p = torch.searchsorted(sorted_hash_picks_time, hash_queries_p[iwhere_query_p])
			ind_extract_s = torch.searchsorted(sorted_hash_picks_time, hash_queries_s[iwhere_query_s])

			valid_ind_p = (ind_extract_p < len(sorted_hash_picks_time)) & (sorted_hash_picks_time[ind_extract_p.clamp(max = len(sorted_hash_picks_time) - 1)] == hash_queries_p[iwhere_query_p])
			valid_ind_s = (ind_extract_s < len(sorted_hash_picks_time)) & (sorted_hash_picks_time[ind_extract_s.clamp(max = len(sorted_hash_picks_time) - 1)] == hash_queries_s[iwhere_query_s])

			inds_queries_to_picks_p = order_hash_picks_time[ind_extract_p.clamp(max = len(sorted_hash_picks_time) - 1)][valid_ind_p]
			inds_queries_to_picks_s = order_hash_picks_time[ind_extract_s.clamp(max = len(sorted_hash_picks_time) - 1)][valid_ind_s]
			# assert(valid_ind_p.sum() == len(valid_ind_p))
			# assert(valid_ind_s.sum() == len(valid_ind_s))
			# assert(torch.abs(pick_vals_time[inds_queries_to_picks_p,0] - query_vals_p[iwhere_query_p,0]).amax() == 0)
			# assert(torch.abs(pick_vals_time[inds_queries_to_picks_s,0] - query_vals_s[iwhere_query_s,0]).amax() == 0)

			misfit_rel_time_p = tpick[inds_queries_to_picks_p].reshape(-1,1) - tlatent[nodes_of_product_p[iwhere_query_p],0].reshape(-1,1)
			misfit_rel_time_s = tpick[inds_queries_to_picks_s].reshape(-1,1) - tlatent[nodes_of_product_s[iwhere_query_s],1].reshape(-1,1)
			# assert(degree(inds_queries_to_picks_p).amax() <= self.k_spc_edges)
			# assert(degree(inds_queries_to_picks_s).amax() <= self.k_spc_edges)

			misfit_rel_time_p = torch.cat((torch.exp(-1.0*torch.abs(misfit_rel_time_p)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_rel_time_p)), dim = 1)
			misfit_rel_time_s = torch.cat((torch.exp(-1.0*torch.abs(misfit_rel_time_s)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_rel_time_s)), dim = 1)

			offset_ref_sta_p = (locs_use_cart[ipick[inds_queries_to_picks_p]] - x_context_cart[A_src_in_sta[1,nodes_of_product_p[iwhere_query_p]],:])/(10.0*self.scale_rel)
			offset_ref_sta_s = (locs_use_cart[ipick[inds_queries_to_picks_s]] - x_context_cart[A_src_in_sta[1,nodes_of_product_s[iwhere_query_s]],:])/(10.0*self.scale_rel)

			offset_ref_sta_norm_p = torch.norm(offset_ref_sta_p, dim = 1, keepdim = True).clamp(min = eps_time)
			offset_ref_sta_norm_s = torch.norm(offset_ref_sta_s, dim = 1, keepdim = True).clamp(min = eps_time)

			gammas_time2 = torch.exp(self.log_gamma_base_time2 + 1.6 * torch.tanh(self.f_gamma_time2(embed_context)))	
			gammas_time3 = torch.exp(self.log_gamma_base_time3 + 1.6 * torch.tanh(self.f_gamma_time3(embed_context)))	

			rbf_ref_sta_p = torch.exp(-1.0 * offset_ref_sta_norm_p * gammas_time2)
			rbf_ref_sta_s = torch.exp(-1.0 * offset_ref_sta_norm_s * gammas_time3)

			offset_ref_sta_norm_kernel_p = torch.cat((offset_ref_sta_p/offset_ref_sta_norm_p, rbf_ref_sta_p), dim = 1)
			offset_ref_sta_norm_kernel_s = torch.cat((offset_ref_sta_s/offset_ref_sta_norm_s, rbf_ref_sta_s), dim = 1)
			
			# offset_ref_sta_norm_kernel_p = torch.exp(-1.0*torch.abs(offset_ref_sta_norm_p)/(3.0))
			# offset_ref_sta_norm_kernel_s = torch.exp(-1.0*torch.abs(offset_ref_sta_norm_s)/(3.0))

			# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
			# inpt_aggregate_p = torch.cat((x[nodes_of_product_p[iwhere_query_p]], misfit_rel_time_p, offset_ref_sta_p, offset_ref_sta_norm_kernel_p, self.phase_embed(phase_label[inds_queries_to_picks_p].reshape(-1).long())), dim = 1)
			# inpt_aggregate_s = torch.cat((x[nodes_of_product_s[iwhere_query_s]], misfit_rel_time_s, offset_ref_sta_s, offset_ref_sta_norm_kernel_s, self.phase_embed(phase_label[inds_queries_to_picks_s].reshape(-1).long())), dim = 1)
			inpt_aggregate_p = torch.cat((x[nodes_of_product_p[iwhere_query_p]], misfit_rel_time_p, offset_ref_sta_norm_kernel_p, self.phase_embed(phase_label[inds_queries_to_picks_p].reshape(-1).long())), dim = 1)
			inpt_aggregate_s = torch.cat((x[nodes_of_product_s[iwhere_query_s]], misfit_rel_time_s, offset_ref_sta_norm_kernel_s, self.phase_embed(phase_label[inds_queries_to_picks_s].reshape(-1).long())), dim = 1)
			
			aggregate_product_p = scatter(self.fc2(inpt_aggregate_p), inds_queries_to_picks_p, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider
			aggregate_product_s = scatter(self.fc3(inpt_aggregate_s), inds_queries_to_picks_s, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider


		# arv_embed = self.null_embed.clone().expand(len(x_query_cart), len(tpick), -1).clone() # torch.zeros((len(x_query_cart), len(tpick), aggregate_picks.shape[1])).to(device)
		# arv_embed[pick_vals[:,1], iarv, :] = aggregate_product
		# arv_embed = self.fc_merge((torch.cat((arv_embed, aggregate_product_p.unsqueeze(0).expand(len(x_query_cart), -1, -1), aggregate_product_s.unsqueeze(0).expand(len(x_query_cart), -1, -1)), dim = 2)))


		# # Create base null embedding expanded across (N_queries, N_picks, n_hidden)
		# base_embed = self.null_embed.expand(len(x_query_cart), len(tpick), -1)
		
		# # Build a zero tensor for active aggregate features
		# dense_picks = torch.zeros_like(base_embed)
		# dense_picks[pick_vals[:, 1], iarv, :] = aggregate_product
		
		# # Build a binary mask for valid picks
		# pick_mask = torch.zeros((len(x_query_cart), len(tpick), 1), device=self.device, dtype=torch.bool)
		# pick_mask[pick_vals[:, 1], iarv] = True
		
		# # Out-of-place selection (100% autograd safe & clean!)
		# arv_embed = torch.where(pick_mask, dense_picks, base_embed)

		arv_embed = self.null_embed.expand(len(x_query_cart), len(tpick), -1).clone()
		arv_embed[pick_vals[:,1], iarv, :] = aggregate_product
		arv_embed = self.fc_merge(torch.cat((arv_embed, aggregate_product_p.unsqueeze(0).expand(len(x_query_cart), -1, -1), aggregate_product_s.unsqueeze(0).expand(len(x_query_cart), -1, -1)), dim = 2))

		return arv_embed, mask_misfit_time ## Make sure this is correct reshape (not transposed)


class SourceStationAttention(MessagePassing):

	def __init__(self, ndim_src_in, ndim_arv_in, ndim_out, n_latent, ndim_extra = 1, n_dim_out_src = 1, n_heads = 5, n_hidden = 30, eps = eps, use_src_pred = False, use_dual_attention = True, use_phase_types = use_phase_types, device = device):
		super(SourceStationAttention, self).__init__(node_dim = 0, aggr = 'add') # check node dim.

		self.f_pick_query = nn.Sequential(nn.Linear(ndim_arv_in + 9, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))
		self.f_pick_context = nn.Sequential(nn.Linear(ndim_arv_in + 9, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))
		self.f_pick_values = nn.Sequential(nn.Linear(ndim_arv_in + 9, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))

		if use_dual_attention == True:
			self.f_source_query = nn.Sequential(nn.Linear(ndim_arv_in + n_heads*n_latent + 9, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))
			self.f_source_context = nn.Sequential(nn.Linear(ndim_arv_in + n_heads*n_latent + 9, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))
			self.f_source_values = nn.Sequential(nn.Linear(ndim_arv_in + n_heads*n_latent + 9, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))
			self.merge_attn = nn.Sequential(nn.Linear(2*n_latent, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_latent))
			# self.alpha_source = nn.Parameter(torch.Tensor([np.log(0.5 / (1 - 0.5))]).to(device)) ## Initilizes as 0.5
			# self.alpha_src = nn.Parameter(torch.Tensor([0.5]).to(device)) ## Initilizes as 0.5
			self.alpha_src = nn.Parameter(torch.Tensor([0.5])) ## Initilizes as 0.5

			self.self_dummy_src = nn.Parameter(torch.zeros(1, n_heads))
			self.dummy_keys_src = nn.Parameter(torch.zeros(1, n_heads, n_latent)) # .to(device)
			self.dummy_queries_src = nn.Parameter(torch.randn(1, n_heads, n_latent) * 0.01) # .to(device)
			self.dummy_values_src = nn.Parameter(torch.randn(1, n_heads, n_latent) * 0.01) # .to(device)


		# self.f_values_1 = nn.Linear(ndim_arv_in + 5, n_hidden) # add second layer transformation.
		# self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.
		# self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_1 = nn.Linear(n_latent*n_heads, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, ndim_out) # can remove this layer possibly.
		if use_src_pred == True:
			self.proj_src_1 = nn.Linear(n_latent*n_heads, n_hidden) # can remove this layer possibly.
			self.proj_src_2 = nn.Linear(n_hidden, n_hidden) # can remove this layer possibly.
			self.proj_src_3 = nn.Linear(n_hidden, n_dim_out_src)
			self.proj_attn = nn.Linear(n_hidden, 1)
			self.activate_src = nn.PReLU()			
			self.activate_src1 = nn.PReLU()			
			self.use_src_pred = True
			self.n_dim_out_src = n_dim_out_src
			self.log_tau = nn.Parameter(torch.tensor([np.log(0.1)], dtype = torch.float32, device = device))
			
		else:
			self.use_src_pred = False

		# self.embed_trns = nn.Sequential(nn.Linear(ndim_src_in, ndim_src_in), nn.PReLU())
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.eps = eps
		self.t_kernel_sq = torch.Tensor([eps]).to(device)**2

		self.self_bias = nn.Parameter(torch.zeros(1, n_heads)) # .to(device) # zeros
		self.self_dummy = nn.Parameter(torch.zeros(1, n_heads)) # .to(device) # zeros
		self.dummy_keys = nn.Parameter(torch.randn(1, n_heads, n_latent) * 0.01) # .to(device)
		self.dummy_values = nn.Parameter(torch.randn(1, n_heads, n_latent) * 0.01) # .to(device)

		n_dim_phase = 5
		self.embed_phase = nn.Embedding(2 + 1, n_dim_phase)

		# self.alpha = nn.Parameter(torch.Tensor([np.log(0.5 / (1 - 0.5))]).to(device)) ## Initilizes as 0.5
		# self.alpha = nn.Parameter(torch.Tensor([0.5]).to(device)) ## Initilizes as 0.5 # self.log_temp = nn.Parameter(torch.Tensor([0.5])).to(device)
		self.alpha = nn.Parameter(torch.Tensor([0.5])) ## Initilizes as 0.5 # self.log_temp = nn.Parameter(torch.Tensor([0.5])).to(device)

		self.use_dual_attention = use_dual_attention
		
		self.ndim_feat = ndim_arv_in + ndim_extra
		self.use_phase_types = use_phase_types
		self.ndim_arv_in = ndim_arv_in
		self.n_phases = ndim_out

		self.use_src_context = False
		if self.use_src_context == True:
			self.embed_src = nn.Sequential(nn.Linear(ndim_src_in, n_hidden), nn.PReLU())
			self.gate_src = nn.Sequential(nn.Linear(ndim_src_in + n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))
			self.downscale = torch.Tensor([0.1]).to(device)

		self.activate4 = nn.PReLU()
		# self.activate5 = nn.PReLU()
		self.device = device


	def forward(self, stime, trv_src, locs_cart, arrival, mask_arv, tpick, ipick, phase_label): # reference k nearest spatial points

		# src isn't used. Only trv_src is needed.
		n_src, n_sta, n_arv = len(stime), trv_src.shape[1], len(tpick) # + 1 ## Note: adding 1 to size of arrivals!
		if self.use_phase_types == False:
			phase_label = phase_label*0.0

		# edges = remove_self_loops(radius(ipick.reshape(-1,1).float(), ipick.reshape(-1,1).float(), max_num_neighbors = len(ipick), r = 0.5))[0]
		edges = add_self_loops(remove_self_loops(radius(ipick.reshape(-1,1).float(), ipick.reshape(-1,1).float(), max_num_neighbors = len(ipick), r = 0.2))[0])[0].flip(0).contiguous()
		n_edge = edges.shape[1]

		## Now must duplicate edges, for each unique source. (different accumulation points)
		edges = (edges.repeat(1, n_src) + torch.cat(((torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(self.device), (torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(self.device)), dim = 0)).long().contiguous()
		src_index = torch.arange(n_src).repeat_interleave(n_edge).contiguous().long().to(self.device)
		self_link = (edges[0] == edges[1]).reshape(-1,1).detach() # Each accumulation index (an entry from src cross arrivals). The number of arrivals is edge_index.max() exactly (since tensor is composed of number arrivals + 1)

		use_sparse = True
		if use_sparse == True:

			## Note: let's add one more level of sparsity : only include pick pairs within a radius? Because e.g., some high pick rate stations
			## will have many useless picks to attent too.. (however this is problematic to base it on time offsets, as either phase type)
			## might be viable (.e.g, comparing between P and S can be useful). So could in theory use "time adjacenecy" allowing swaps of phase type
			## to create these neighborhoods. This might help prevent explosions in memory during this layer for high pick rates or noisy stations.
			ikeep = torch.where((mask_arv[src_index, torch.remainder(edges[0], n_arv).long()] > 0) + (edges[0] == edges[1]))[0]
			edges = edges[:,ikeep].contiguous()
			# edges = torch.cat((edges[0][ikeep].reshape(1,-1), edges[1][ikeep].reshape(1,-1)), dim = 0).contiguous()
			src_index = src_index[ikeep]
			self_link = self_link[ikeep]	

		if len(src_index) == 0:
			if self.use_src_pred == True:
				return torch.zeros(n_src, n_arv, self.n_phases).to(self.device), torch.zeros(n_src, self.n_dim_out_src).to(self.device)
			else:
				return torch.zeros(n_src, n_arv, self.n_phases).to(self.device)

		edge_dummy = torch.cat(((n_arv*n_src)*torch.ones(1,n_arv*n_src), torch.arange(n_arv*n_src).reshape(1,-1)), dim = 0).long().to(self.device)

		## Create n_src dummy "arrivals" to link to each source.
		if self.use_dual_attention == True: ## Is this arrival reshape correct?
			## Should add phase embedding
			arrival_inpt = torch.cat((arrival.reshape(n_arv*n_src,-1), torch.zeros(1 + n_src, self.ndim_arv_in, device = self.device)), dim = 0)
			phase_inpt = torch.cat((torch.tile(phase_label, (n_src, 1)), 2.0*torch.ones(1 + n_src,1).to(self.device)), dim = 0)
			# phase_inpt = torch.cat((phase_label.expand(n_src, -1), -1.0*torch.ones(1 + n_src,1).to(self.device)), dim = 0)
			## The dummy source indices should be the "correct" ones for those specific source-arrival pairs
			# src_index = torch.cat((src_index, n_src*torch.ones(n_arv*n_src).to(device), torch.arange(n_src).to(device)), dim = 0).long().contiguous()
			src_index = torch.cat((src_index, torch.arange(n_src).repeat_interleave(n_arv, dim = 0).to(device), torch.arange(n_src).to(device)), dim = 0).long().contiguous()
			# src_index = torch.cat((src_index, n_src*torch.ones(n_arv*n_src).to(device), torch.arange(n_src).to(device)), dim = 0).long().contiguous()
			self_link = torch.cat((self_link, torch.zeros(n_arv*n_src + n_src,1).to(device)), dim = 0).float()
			edge_dummy_src = torch.cat(( (torch.arange(n_src).reshape(1,-1) + n_src*n_arv + 1), torch.arange(n_src).reshape(1,-1) ), dim = 0).long().to(device) ## Reciever nodes can be arbitrarily listed here (the features aren't used at torch.arange(n_src).reshape(1,-1))
			edges = torch.cat((edges, edge_dummy, edge_dummy_src), dim = 1).contiguous()

			N = n_arv*n_src + 1 + n_src # still correct?
			M = n_arv*n_src

		else:

			arrival_inpt = torch.cat((arrival.reshape(n_arv*n_src,-1), torch.zeros(1, self.ndim_arv_in, device = self.device)), dim = 0)
			phase_inpt = torch.cat((torch.tile(phase_label, (n_src, 1)), torch.Tensor([2.0]).reshape(1,1).to(self.device)), dim = 0)
			# src_index = torch.cat((src_index, n_src*torch.ones(n_arv*n_src).to(device)), dim = 0).long().contiguous() ## The dummy "source index"
			src_index = torch.cat((src_index, torch.arange(n_src).repeat_interleave(n_arv, dim = 0).to(device)), dim = 0).long().contiguous() ## The dummy "source index"
			self_link = torch.cat((self_link, torch.zeros(n_arv*n_src,1).to(device)), dim = 0).float()
			edges = torch.cat((edges, edge_dummy), dim = 1).contiguous()

			N = n_arv*n_src + 1 # still correct?
			M = n_arv*n_src

		
		# src_embed_trns = self.embed_trns(src_embed)
		src_ind_repeat = torch.arange(n_src).repeat_interleave(n_arv).contiguous().long().to(self.device)
		# out = self.proj_2(self.embed_src(src_embed[src_ind_repeat]) + self.activate4(self.proj_1(self.propagate(edges, x = arrival.reshape(n_arv*n_src,-1), sembed = src_embed, stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = ipick.repeat(n_src), atime = tpick.repeat(n_src), phase = phase_label.repeat(n_src, 1), self_link = self_link, size = (N, M)).view(-1, self.n_latent*self.n_heads)))) # M is output. Taking mean over heads

		if self.use_src_pred == True:
			# out_embed = self.propagate(edges, x = (arrival_inpt, arrival_inpt[0:(n_arv*n_src)]), stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = torch.tile(ipick, (n_src,)), atime = torch.tile(tpick, (n_src,)), phase = (phase_inpt, phase_inpt[0:(n_arv*n_src)]), self_link = self_link, num_queries = torch.Tensor([n_arv*n_src]).to(self.device), size = (N, M)).view(-1, self.n_latent*self.n_heads) # M is output. Taking mean over heads
			# out_src = self.proj_src_3(self.activate_src1(self.proj_src_2(self.activate_src(self.proj_src_1(out_embed))).view(n_src, n_arv, -1).sum(1)))
			# out = self.proj_2(self.activate4(self.proj_1(out_embed)))
			# return out.view(n_src, n_arv, -1), out_src ## Make sure this is correct reshape (not transposed)

			out_embed = self.propagate(edges, x = (arrival_inpt, arrival_inpt[0:(n_arv*n_src)]), stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = torch.tile(ipick, (n_src,)), atime = torch.tile(tpick, (n_src,)), phase = (phase_inpt, phase_inpt[0:(n_arv*n_src)]), self_link = self_link, num_queries = torch.Tensor([n_arv*n_src]).to(self.device), size = (N, M)).view(-1, self.n_latent*self.n_heads) # M is output. Taking mean over heads
			# out_src = self.proj_src_3(self.activate_src1(self.proj_src_2(self.activate_src(self.proj_src_1(out_embed))).view(n_src, n_arv, -1).sum(1)))
			tau_base = torch.exp(self.log_tau) 
			tau_deg = tau_base * (n_arv ** 0.5)
			out_src = self.activate_src(self.proj_src_1(out_embed)).view(n_src, n_arv, -1)
			# alpha_score = torch.softmax(self.proj_attn(out_src) / tau, dim = 1)
			alpha_score = torch.softmax(self.proj_attn(out_src) / tau_deg, dim = 1)
			out_src = self.proj_src_3(self.activate_src1(self.proj_src_2((alpha_score*out_src).sum(1))))
			out = self.proj_2(self.activate4(self.proj_1(out_embed)))
			return out.view(n_src, n_arv, -1), out_src ## Make sure this is correct reshape (not transposed)
		
		else:

			out = self.proj_2(self.activate4(self.proj_1(self.propagate(edges, x = (arrival_inpt, arrival_inpt[0:(n_arv*n_src)]), stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = torch.tile(ipick, (n_src,)), atime = torch.tile(tpick, (n_src,)), phase = (phase_inpt, phase_inpt[0:(n_arv*n_src)]), self_link = self_link, num_queries = torch.Tensor([n_arv*n_src]).to(self.device), size = (N, M)).view(-1, self.n_latent*self.n_heads)))) # M is output. Taking mean over heads
			## Could do concatenation and summation of the source embedding
			# out = self.proj_2(torch.cat((src_embed, self.embed_src(src_embed) + self.activate4(self.proj_1(self.propagate(edges, x = arrival.reshape(n_arv*n_src,-1), sembed = src_embed, stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = ipick.repeat(n_src), atime = tpick.repeat(n_src), phase = phase_label.repeat(n_src, 1), self_link = self_link, size = (N, M)).view(-1, self.n_latent*self.n_heads)))))) # M is output. Taking mean over heads

		return out.view(n_src, n_arv, -1) ## Make sure this is correct reshape (not transposed)


	def message(self, x_j, x_i, edge_index, index, tsrc_p, tsrc_s, sindex, stindex, stime, atime, self_link, num_queries, phase_j, phase_i): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.

		
		## Does this converge on standard behavior if not using dual_attention
		ifake_edge_src = (edge_index[0] > num_queries)
		inot_fake_src = ~ifake_edge_src ## Can only compute the travel time misfits for these (to avoid source overload)

		ifake_edge = (edge_index[0] == num_queries)*(inot_fake_src == 1) ## Null node
		inot_fake = ~ifake_edge

		real_edge = (~ifake_edge)*(inot_fake_src == 1) ## Real edges for pick queries are not fake edges of both types

		rel_t_p = (atime[edge_index[0][real_edge]] - (tsrc_p[sindex[real_edge], stindex[edge_index[0][real_edge]]] + stime[sindex[real_edge]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_p = torch.cat((torch.exp(-0.5*(rel_t_p**2)/self.t_kernel_sq), torch.sign(rel_t_p).detach()), dim = 1) # phase[edge_index[0]]
		rel_t_s = (atime[edge_index[0][real_edge]] - (tsrc_s[sindex[real_edge], stindex[edge_index[0][real_edge]]] + stime[sindex[real_edge]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s = torch.cat((torch.exp(-0.5*(rel_t_s**2)/self.t_kernel_sq), torch.sign(rel_t_s).detach()), dim = 1) # phase[edge_index[0]]
		rel_t = torch.cat((rel_t_p, rel_t_s, self.embed_phase(phase_j[real_edge].long().reshape(-1))), dim = 1) ## only indexed for not fake source

		rel_t_p1 = (atime[edge_index[1][inot_fake_src]] - (tsrc_p[sindex[inot_fake_src], stindex[edge_index[1][inot_fake_src]]] + stime[sindex[inot_fake_src]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_p1 = torch.cat((torch.exp(-0.5*(rel_t_p1**2)/self.t_kernel_sq), torch.sign(rel_t_p1).detach()), dim = 1) # phase[edge_index[0]]
		rel_t_s1 = (atime[edge_index[1][inot_fake_src]] - (tsrc_s[sindex[inot_fake_src], stindex[edge_index[1][inot_fake_src]]] + stime[sindex[inot_fake_src]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s1 = torch.cat((torch.exp(-0.5*(rel_t_s1**2)/self.t_kernel_sq), torch.sign(rel_t_s1).detach()), dim = 1) # phase[edge_index[0]]
		rel_t1 = torch.cat((rel_t_p1, rel_t_s1, self.embed_phase(phase_i[inot_fake_src].long().reshape(-1))), dim = 1)

		## Queries using reciever nodes (i) because each reciever is trying to decide which of neighboring picks is "relevant", and it also uses source embedding because this is dependant on the source
		## Contexts (actually keys) and values use the sender nodes as these are the ones the queries are attending over ## Note: I did used to include the source origin time..
		# queries_real_and_null = self.f_pick_query(torch.cat((x_i[inot_fake_src], rel_t1, sembed[sindex[inot_fake_src]], self_link[inot_fake_src]), dim = 1)).view(-1, self.n_heads, self.n_latent)

		queries_real_and_null = self.f_pick_query(torch.cat((x_i[inot_fake_src], rel_t1), dim = 1)).view(-1, self.n_heads, self.n_latent)

		contexts_real = self.f_pick_context(torch.cat((x_j[real_edge], rel_t), dim = 1)).view(-1, self.n_heads, self.n_latent) ## Do not include self link in context to avoid short cut of information		
		values_real = self.f_pick_values(torch.cat((x_j[real_edge], rel_t), dim = 1)).view(-1, self.n_heads, self.n_latent) ## Note self_link optional here


		queries = torch.zeros(len(index), self.n_heads, self.n_latent, device = self.device)
		contexts = torch.zeros(len(index), self.n_heads, self.n_latent, device = self.device)
		values = torch.zeros(len(index), self.n_heads, self.n_latent, device = self.device)

		queries[inot_fake_src,:,:] = queries_real_and_null
		contexts[real_edge,:,:] = contexts_real
		values[real_edge,:,:] = values_real

		n_fake = int(ifake_edge.sum())
		# contexts[ifake_edge,:,:] = self.dummy_keys.repeat(n_fake, 1, 1)
		# values[ifake_edge,:,:] = self.dummy_values.repeat(n_fake, 1, 1)

		contexts[ifake_edge,:,:] = self.dummy_keys # .repeat(n_fake, 1, 1)
		values[ifake_edge,:,:] = self.dummy_values # .repeat(n_fake, 1, 1)
		## Compute attention
		scores = (queries*contexts).sum(-1)/self.scale
		
		## Clip degrees
		deg = torch.clamp(degree(edge_index[1][inot_fake_src], num_nodes = len(atime)).detach(), min = 1)
		temp = torch.log1p(deg).pow(torch.clamp(self.alpha, min = 0.25, max = 2.0))[edge_index[1]].reshape(-1,1) # [edge_index[1]].reshape(-1,1)
		temp[deg[edge_index[1]] <= 2] = 1.0 ## Stabalize temperature for low degree cases
		## Add bias terms
		scores[self_link[:,0] == 1] = scores[self_link[:,0] == 1] + self.self_bias
		scores[ifake_edge] = scores[ifake_edge] + self.self_dummy

		scores = scores / temp.sqrt()

		## Add dual attention aggregation
		# alpha = softmax(scores, index, num_nodes = ) # 
		alpha = softmax(scores, index) # 

		if self.use_dual_attention == False:

			return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))

		else:

			## Note: as two seperate steps can implement with aggregation of the obtained features from previous step
			# attn_picks = alpha.unsqueeze(-1)*values

			attn_picks = alpha.unsqueeze(-1)*values

			rel_t_p2 = (atime[edge_index[1][real_edge]] - (tsrc_p[sindex[real_edge], stindex[edge_index[1][real_edge]]] + stime[sindex[real_edge]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
			rel_t_p2 = torch.cat((torch.exp(-0.5*(rel_t_p2**2)/self.t_kernel_sq), torch.sign(rel_t_p2).detach()), dim = 1) # phase[edge_index[0]]
			rel_t_s2 = (atime[edge_index[1][real_edge]] - (tsrc_s[sindex[real_edge], stindex[edge_index[1][real_edge]]] + stime[sindex[real_edge]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
			rel_t_s2 = torch.cat((torch.exp(-0.5*(rel_t_s2**2)/self.t_kernel_sq), torch.sign(rel_t_s2).detach()), dim = 1) # phase[edge_index[0]]
			rel_t2 = torch.cat((rel_t_p2, rel_t_s2, self.embed_phase(phase_i[real_edge].long().reshape(-1))), dim = 1)


			attn_slice = attn_picks.view(-1, self.n_heads*self.n_latent)[real_edge]

			
			queries_src_real = self.f_source_query(torch.cat((x_i[real_edge], attn_slice, rel_t2), dim = 1)).view(-1, self.n_heads, self.n_latent)
			contexts_src_real = self.f_source_context(torch.cat((x_j[real_edge], attn_slice, rel_t), dim = 1)).view(-1, self.n_heads, self.n_latent) ## Do not include self link in context to avoid short cut of information
			values_src_real = self.f_source_values(torch.cat((x_j[real_edge], attn_slice, rel_t), dim = 1)).view(-1, self.n_heads, self.n_latent) ## Note self_link optional here
			# values_src = self.f_source_values(torch.cat((x_j, attn_picks, rel_t), dim = 1)).view(-1, self.n_heads, self.n_latent) ## Note self_link optional here

			queries_src = torch.zeros(len(index), self.n_heads, self.n_latent, device = self.device)
			contexts_src = torch.zeros(len(index), self.n_heads, self.n_latent, device = self.device)
			values_src = torch.zeros(len(index), self.n_heads, self.n_latent, device = self.device)


			queries_src[real_edge,:,:] = queries_src_real
			contexts_src[real_edge,:,:] = contexts_src_real
			values_src[real_edge,:,:] = values_src_real

			n_fake_src = int(ifake_edge_src.sum())
			queries_src[ifake_edge_src,:,:] = self.dummy_queries_src # .repeat(n_fake_src, 1, 1)
			contexts_src[ifake_edge_src,:,:] = self.dummy_keys_src # .repeat(n_fake_src, 1, 1)
			values_src[ifake_edge_src,:,:] = self.dummy_values_src # .repeat(n_fake_src, 1, 1)


			scores_src = (queries_src*contexts_src).sum(-1)/self.scale
			deg = torch.clamp(degree(sindex, num_nodes = len(stime)).detach(), min = 1)

			# temp_src = torch.clamp(degree(sindex, num_nodes = len(sembed)).detach(), min = 1).pow(torch.clamp(torch.sigmoid(self.alpha_src), min = 0.25))[edge_index[1]].reshape(-1,1)
			temp_src = torch.log1p(deg).pow(torch.clamp(self.alpha_src, min = 0.25, max = 2.0))[sindex].reshape(-1,1) # [edge_index[1]].reshape(-1,1) # [edge_index[1]].reshape(-1,1)
			temp_src[deg[sindex] <= 2.0] = 1.0

			# scores_src[self_link[:,0] == 1] = scores_src[self_link[:,0] == 1] + self.self_bias
			scores_src[ifake_edge_src] = scores_src[ifake_edge_src] + self.self_dummy_src

			scores_src = scores_src / temp_src.sqrt()
			alpha_src = softmax(scores_src, sindex)
			attn_src = alpha_src.unsqueeze(-1)*values_src

			## Now merge with the messages of the previous attention layer and aggregate
			merge_attn = self.merge_attn(torch.cat((attn_picks, attn_src), dim = 2))

			return merge_attn
			

# ## FiLM class to merge (modulate) the embed_context information into feature space
# class FiLM(nn.Module):
#     """Feature-wise Linear Modulation based on embed_context."""
#     def __init__(self, embed_dim, feature_dim):
#         super().__init__()
#         self.fc = nn.Linear(embed_dim, 2 * feature_dim)
#         nn.init.zeros_(self.fc.weight)
#         nn.init.zeros_(self.fc.bias)

#     def forward(self, x, embed_context):
#         film_params = self.fc(embed_context)
#         gamma, beta = film_params.chunk(2, dim=-1)
#         return x * (1.0 + gamma) + beta


class FiLM(nn.Module):
    """Feature-wise Linear Modulation with zero-initialized identity defaults."""
    def __init__(self, embed_dim, feature_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2 * feature_dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, embed_context):
        film_params = self.fc(embed_context)
        gamma, beta = film_params.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class GCN_Detection_Network_extended(nn.Module):
	def __init__(self, ftrns1, ftrns2, scale_rel = scale_rel, scale_time = scale_time, use_absolute_pos = use_absolute_pos, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded, use_embedding = use_embedding, use_src_pred = False, use_sigmoid = use_sigmoid, attach_time = attach_time, use_absolute_offset = True, trv = None, device = 'cuda'):
		super(GCN_Detection_Network_extended, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
		n_dim_extra_inpt = 0 if attach_time == False else 1
		n_dim_extra_feat = 0 if use_embedding == False else 20
		if use_absolute_offset == True: n_dim_extra_inpt = n_dim_extra_inpt + 7 # concatenate the spatial offsets between source nodes and recievers into input feature
		
		embed_vector_dim = 10 ## Note can add normalization to output
		self.embed_vector = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, embed_vector_dim))

		# if use_expanded == False:
		# 	# self.DataAggregation = DataAggregation(4 + n_dim_extra_inpt + n_dim_extra_feat + embed_vector_dim, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
		# 	self.DataAggregation = DataAggregation(4 + n_dim_extra_inpt + n_dim_extra_feat, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
		# else:
		# 	self.DataAggregation = DataAggregationExpanded(4 + n_dim_extra_inpt + n_dim_extra_feat, 15, device = device).to(device) # output size is latent size for (half of) bipartite code # , 15				

		# Main Encoder Stack
		self.DataAggregation = DataAggregationExpanded(
		    in_channels= 4 + n_dim_extra_inpt + n_dim_extra_feat + embed_vector_dim,
		    out_channels=15,
		    # n_hidden=n_hidden,
		    # embed_dim=embed_dim,
		    use_embedding=use_embedding
		)

		## Maybe add expander convolution on SpatialAggregation
		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 8).to(device) # 30, 15
		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpaceTimeDirect = SpaceTimeDirect(30, 30).to(device) # 15, 30
		self.SpaceTimeAttention = SpaceTimeAttention(30, 30, 8, 15, device = device).to(device)
		# self.SpaceTimeAttention = SpaceTimeAttention(30, 30, 4, 15, device = device).to(device)

		if use_expanded == True:
			# self.SpatialAggregation1_expanded = SpatialAggregation(30, 30).to(device) # 15, 30
			self.SpatialAggregation2_expanded = SpatialAggregation(30, 30, zero_offsets = True).to(device) # 15, 30
			self.gate_expanded = nn.Linear(2*30 + embed_vector_dim, 30)
			nn.init.constant_(self.gate_expanded.bias, -2.0)

		self.proj_soln1 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))
		self.proj_soln2 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))

		self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15).to(device)

		## For now, don't use expanded on the downstream DataAggregationAssociationPhase (may be slightly unnecessary)
		# if use_expanded == False:
		# self.DataAggregationAssociation = DataAggregationAssociation(15, 15).to(device) # need to add concatenation

		self.DataAggregationAssociation = DataAggregationAssociation(
		    in_channels=15,  # Dimension of unpooled feature 's'
		    out_channels=15,
		    # n_hidden=n_hidden,
		    # n_dim_latent=n_hidden,
		    # n_dim_mask=Mask.shape[-1],
		    # embed_dim=embed_dim,
		    use_offsets=True
		)

		## Make association module layers (note, previous arrival embeddings used to be smaller)
		self.ArrivalEmbedding = ArrivalEmbedding(30, 30, trv = trv, device = device, ftrns2 = ftrns2) ## [note: merging the embeddings for P and S into one (oveloaded) layer rather than keeping as seperate layers?]
		self.Arrivals = SourceStationAttention(30, 30, 2, 15, n_heads = 3, use_src_pred = use_src_pred, device = device).to(device)
		if use_src_pred == True:
			self.alpha = nn.Parameter(torch.tensor([0.1], device = device))

		# if use_embedding == True:
		# 	# self.DataAggregationEmbedding = DataAggregationEmbedding(1 + n_dim_extra_inpt + embed_vector_dim, int(n_dim_extra_feat/2))
		# 	self.DataAggregationEmbedding = DataAggregationEmbedding(1 + n_dim_extra_inpt, int(n_dim_extra_feat/2))

		self.use_absolute_pos = use_absolute_pos
		self.scale_rel = scale_rel
		self.scale_time = scale_time
		self.use_expanded = use_expanded
		self.use_gradient_loss = use_gradient_loss
		self.activate_gradient_loss = False
		self.attach_time = attach_time
		self.use_embedding = use_embedding
		self.use_direct_output = True
		self.use_sigmoid = use_sigmoid
		self.use_src_pred = use_src_pred
		self.use_absolute_offset = use_absolute_offset

		# if use_absolute_offset == True:
		# 	# Predict scale-conditioned gamma offsets from domain context
		# 	self.f_gamma = nn.Linear(embed_vector_dim, 4)
		# 	# Base log-gammas (e.g. multi-scale physical bounds)
		# 	self.log_gamma_base = nn.Parameter(
		# 		torch.log(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
		# 	)
		
        
        # embed_vector_dim = 10
        # self.embed_vector = nn.Sequential(
        #     nn.Linear(6, 30), 
        #     nn.PReLU(), 
        #     nn.Linear(30, embed_vector_dim)
        # )

        # ---------------------------------------------------------------------
        # 1. Initialize RBF Gammas for the Product Graph Offset Features
        # ---------------------------------------------------------------------
        if self.use_absolute_offset:
            # Baseline spatial gammas: fine (5.0), medium (1.0), broad (0.1)
            init_gammas_sp = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32).reshape(1, 3)
            self.log_gamma_base = nn.Parameter(torch.log(init_gammas_sp))  # Shape: [1, 3]

            # Linear projection from embed_context (dim 10) to 4 values:
            # Index 0: Global alpha zoom factor
            # Index 1-3: Residual tweaks for each spatial gamma
            self.f_gamma = nn.Linear(embed_vector_dim, 1 + 3)

            # Zero-initialize the projection weight and bias so the model starts
            # strictly at baseline log_gamma_base during early training steps.
            nn.init.zeros_(self.f_gamma.weight)
            nn.init.zeros_(self.f_gamma.bias)


		# self.w_gamma = nn.Parameter(torch.tensor([0.05, 0.3, 0.8, 2.0]).reshape(1, -1))
		# self.use_src_pred = self.Arrivals.src_pred
		# self.scale_output = torch.Tensor([1.0/10.0]).to(device)
		# self.use_sigmoid = use_sigmoid
		self.device = device

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q, save_state = False):

		n_line_nodes = Slice.shape[0]
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		assert(x_temp_cuda_cart.shape[1] == 3)
		
		embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)		

		A_in_src_slice = A_in_src[0] if self.use_expanded else A_in_src
		pos_rel_sta, pos_rel_src = None, None


		# 1. Compute relative edge vectors ONLY if offsets are enabled
		if self.use_absolute_offset: # (or self.use_offsets)
		    pos_rel_sta = torch.cat((
		        (locs_use_cart[A_src_in_sta[0][A_in_sta[1]]] - locs_use_cart[A_src_in_sta[0][A_in_sta[0]]]), 
		        1000.0 * self.scale_time * (x_temp_cuda_t[A_src_in_sta[1][A_in_sta[1]]] - x_temp_cuda_t[A_src_in_sta[1][A_in_sta[0]]]).view(-1, 1)
		    ), dim=1) / self.scale_rel

		    pos_rel_src = torch.cat((
		        (x_temp_cuda_cart[A_src_in_sta[1][A_in_src_slice[1]]] - x_temp_cuda_cart[A_src_in_sta[1][A_in_src_slice[0]]]), 
		        1000.0 * self.scale_time * (x_temp_cuda_t[A_src_in_sta[1][A_in_src_slice[1]]] - x_temp_cuda_t[A_src_in_sta[1][A_in_src_slice[0]]]).view(-1, 1)
		    ), dim=1) / self.scale_rel


		# 2. Append 7D features ONLY if the Geometric Preconditioner (use_embedding) is active
		if self.use_embedding:
		    pos_rel_sp = A_src_in_edges.x[:, 0:3]
		    pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp**2, dim=1, keepdim=True) + 1e-8)
		    
		    delta = self.f_gamma(embed_context)
		    alpha = delta[:, :1]
		    residuals = 0.2 * torch.tanh(delta[:, 1:])
		    gammas = torch.exp(self.log_gamma_base[:, :3] + alpha + residuals)
		    spatial_decay = torch.exp(-1.0 * pos_norm_sp * gammas)
		    
		    pos_rel_tm = A_src_in_edges.x[:, 3:4]

		    rel_pos_feat = torch.cat((pos_rel_sp / pos_norm_sp, spatial_decay, pos_rel_tm), dim=-1) # 7D
		    Slice = torch.cat((Slice, rel_pos_feat), dim=1)

		
		# Runs both Optional Preconditioner (if self.use_embedding=True) AND Main GNN Stack
		x_latent = self.DataAggregation(
		    tr=Slice, 
		    mask=Mask, 
		    A_in_sta=A_in_sta, 
		    A_in_src=A_in_src, 
		    embed_context=embed_context, 
		    pos_rel_sta=pos_rel_sta,  # Raw 3D + dt coordinates
		    pos_rel_src=pos_rel_src   # Raw 3D + dt coordinates
		)

		x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, embed_context, n_sta, n_temp)
		x = self.SpatialAggregation1(x, embed_context, A_src if self.use_expanded == False else A_src[0], x_temp_cuda) # x_temp_cuda_cart
		x_local = self.SpatialAggregation2(x, embed_context, A_src if self.use_expanded == False else A_src[0], x_temp_cuda)
		if self.use_expanded == True:
			x_expand = self.SpatialAggregation2_expanded(x, embed_context, A_src[1], x_temp_cuda) # x_temp_cuda_cart
			gate = torch.sigmoid(self.gate_expanded(torch.cat((x_local, x_expand, embed_context.expand(x_local.shape[0], -1)), dim = 1)))
			x = x_local + gate*x_expand
		else:
			x = x_local
		x_spatial = self.SpatialAggregation3(x, embed_context, A_src if self.use_expanded == False else A_src[0], x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		
		if self.use_direct_output == True:
			y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		else:
			y_latent = self.SpaceTimeAttention(x_spatial, x_temp_cuda_cart, x_temp_cuda_cart, x_temp_cuda_t, x_temp_cuda_t, embed_context) # contains data on spatial and temporal solution at fixed nodes

		y = self.proj_soln1(y_latent)
		
		if save_state == True:
			self.set_internal_state(x_spatial, x_temp_cuda_cart, x_temp_cuda_t)
			
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t, embed_context) # second slowest module (could use this embedding to seed source source attention vector).

		x_src = []
		x = self.proj_soln2(x)
		
		slope_width = 0.1
		mask_p_thresh = 0.1
		mask_out = torch.relu(y - mask_p_thresh)
		
		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, embed_context, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		
		# ## Maybe re-concatenate the initial Cartesian product input misfit features back into s here
		# if self.use_expanded == False:
		# 	s = self.DataAggregationAssociation(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, A_in_sta, A_in_src, embed_context, pos_rel_sta = pos_rel_sta, pos_rel_src = pos_rel_src) # detach x_latent. Just a "reference"

		# else: ## This assumes that DataAggregationAssociationPhase does not use expanded version
		# 	s = self.DataAggregationAssociation(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, A_in_sta, A_in_src[0], embed_context, pos_rel_sta = pos_rel_sta, pos_rel_src = pos_rel_src) # detach x_latent. Just a "reference"

		latent_ref = x_latent if self.use_src_pred else x_latent.detach()

		# Run standardized association phase
		s = self.DataAggregationAssociationPhase(
		    s=s,
		    x_latent=latent_ref,
		    mask_out_1=mask_out_1,
		    mask=Mask,
		    A_in_sta=A_in_sta,
		    A_in_src=A_in_src_slice,
		    embed_context=embed_context,
		    pos_rel_sta=pos_rel_sta,  # Direct raw offset reuse
		    pos_rel_src=pos_rel_src   # Direct raw offset reuse
		)

		arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, embed_context, trv_out = trv_out_q)

		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return y, x, arv_p, arv_s, src

		else:
			
			arv = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)			
			return y, x, arv_p, arv_s

	def set_scale_coefficients(self, scale_rel, scale_time, kernel_sig_t, eps, src_x_kernel, src_t_kernel, time_shift_range):

		self.scale_rel = scale_rel
		self.scale_time = scale_time

		if self.use_embedding == True:
			self.DataAggregationEmbedding.scale_rel = scale_rel
			self.DataAggregationEmbedding.scale_time = scale_time

		self.SpatialAggregation1.scale_rel = scale_rel
		self.SpatialAggregation1.scale_time = scale_time
		self.SpatialAggregation2.scale_rel = scale_rel
		self.SpatialAggregation2.scale_time = scale_time
		self.SpatialAggregation3.scale_rel = scale_rel
		self.SpatialAggregation3.scale_time = scale_time

		if self.use_expanded == True:
			# self.SpatialAggregation1_expanded.scale_rel = 10.0*scale_rel
			# self.SpatialAggregation1_expanded.scale_time = 10.0*scale_time
			self.SpatialAggregation2_expanded.scale_rel = 10.0*scale_rel
			self.SpatialAggregation2_expanded.scale_time = 10.0*scale_time

		self.SpaceTimeAttention.scale_rel = scale_rel
		self.SpaceTimeAttention.scale_time = scale_time
		
		self.ArrivalEmbedding.scale_rel = scale_rel
		self.ArrivalEmbedding.scale_time = scale_time
		self.ArrivalEmbedding.kernel_sig_t = kernel_sig_t

		# self.SpaceTimeAttentionQuery.scale_rel = scale_rel
		# self.SpaceTimeAttentionQuery.scale_time = scale_time
		# self.SpaceTimeAttentionQuery.kernel_sig_t = kernel_sig_t
		
		self.Arrivals.eps = eps
		self.embedding_vector = torch.tensor([np.log(scale_rel)/5.0, np.log(scale_time), np.log(kernel_sig_t), np.log(src_x_kernel)/3.0, np.log(src_t_kernel), np.log(time_shift_range)/2.0], device = self.device).reshape(1,-1).float()
		self.embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1)
		
	def set_adjacencies(self, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, pos_loc, pos_src):

		# pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]] - pos_loc[A_src_in_sta[0][A_in_sta[1]]])/self.DataAggregation.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		# pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]] - pos_src[A_src_in_sta[1][A_in_src[1]]])/self.DataAggregation.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		# dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
		# dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
		# pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
		# pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)
		
		self.A_in_sta = A_in_sta
		self.A_in_src = A_in_src
		self.A_src_in_edges = A_src_in_edges
		self.A_Lg_in_src = A_Lg_in_src
		self.A_src_in_sta = A_src_in_sta

		if self.use_expanded == False:
			self.A_src = A_src # [0] # if self.use_expanded == True else A_src
		else:
			self.A_src = A_src[0]
			self.Ac = A_src[1]

		self.A_edges_p = A_edges_p
		self.A_edges_s = A_edges_s
		self.dt_partition = dt_partition
		self.tlatent = tlatent
		# self.pos_rel_sta = pos_rel_sta
		# self.pos_rel_src = pos_rel_src

	def set_internal_state(self, x_spatial, x_temp_cuda_cart, x_temp_cuda_t): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)
		## Use this to set state for rapid queries of attention layer
		self.x_spatial = x_spatial
		self.x_temp_cuda_cart = x_temp_cuda_cart
		self.x_temp_cuda_t = x_temp_cuda_t

	def set_internal_state_queries(self, s, x_spatial, x_temp_cuda_cart, x_temp_cuda_t, locs_use_cart, tlatent): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)
		## Use this to set state for rapid queries of attention layer
		
		self.s = s
		self.x_spatial = x_spatial
		self.x_temp_cuda_cart = x_temp_cuda_cart
		self.x_temp_cuda_t = x_temp_cuda_t
		self.locs_use_cart = locs_use_cart
		self.tlatent = tlatent

	def forward_queries(self, x_query_cart, t_query, train = False): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)

		## Use this to obtain query predictions. Note, can modify to also return the spatial embeddings (prior to proj_soln)
		return self.proj_soln2(self.SpaceTimeAttention(self.x_spatial, x_query_cart, self.x_temp_cuda_cart, t_query, self.x_temp_cuda_t, self.embed_context))

	def forward_src_queries(self, x_query_src_cart, tq_sample, tpick, ipick, phase_label, trv_out_q): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)

		arv_embed, mask_arv = self.ArrivalEmbedding(self.s, self.x_temp_cuda_cart, self.x_temp_cuda_t, x_query_src_cart, tq_sample, self.A_src_in_sta, tpick, ipick, phase_label, self.locs_use_cart, self.tlatent, trv_out = trv_out_q)
		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, self.locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return arv_p, arv_s, src

		else:
			arv = self.Arrivals(tq_sample, trv_out_q, self.locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return arv_p, arv_s


	def forward_fixed(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

		# embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		
		n_line_nodes = Slice.shape[0]
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)
		pos_rel_sta, pos_rel_src = None, None
		# embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		assert(x_temp_cuda_cart.shape[1] == 3)

		
		# if self.use_absolute_offset == True:
		# 	norm_pos = torch.sqrt(torch.sum(self.A_src_in_edges.x[:,0:3]**2, dim = 1, keepdim = True) + 1e-8)
		# 	gamma_offset = 1.6 * torch.tanh(self.f_gamma(self.embed_context))
		# 	gammas = torch.exp(self.log_gamma_base + gamma_offset)
		# 	spatial_decay = torch.exp(-1.0 * norm_pos * gammas)  # [N_product, 4]
		# 	rel_pos_feat = torch.cat((self.A_src_in_edges.x[:,0:3]/norm_pos, spatial_decay, self.A_src_in_edges.x[:,3:4]), dim=-1)
		# 	Slice = torch.cat((Slice, rel_pos_feat), dim = 1)

		if self.use_absolute_offset:
		    # Build 7D relative position feature vector
		    pos_rel_sp = self.A_src_in_edges.x[:, 0:3]
		    pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp**2, dim=1, keepdim=True) + 1e-8)
		    
		    delta = self.f_gamma(self.embed_context)
		    alpha = delta[:, :1]
		    residuals = 0.2 * torch.tanh(delta[:, 1:])
		    gammas = torch.exp(self.log_gamma_base[:, :3] + alpha + residuals)
		    spatial_decay = torch.exp(-1.0 * pos_norm_sp * gammas)
		    
		    pos_rel_tm = self.A_src_in_edges.x[:, 3:4]

		    rel_pos_feat = torch.cat((pos_rel_sp / pos_norm_sp, spatial_decay, pos_rel_tm), dim=-1) # 7D
		    Slice = torch.cat((Slice, rel_pos_feat), dim=1)	


		if self.use_embedding == True:
			ndim_slice = -1 if (self.attach_time == True)*(self.use_absolute_offset == False) else -8
			inpt_embedding = torch.cat((torch.ones(len(Slice),1, dtype = Slice.dtype, device = Slice.device),  Slice[:, ndim_slice::]), dim = 1) if ((self.attach_time == True) or (self.use_absolute_offset == True)) else torch.ones(len(Slice),1, dtype = Slice.dtype, device = Slice.device) # .to(Slice.device)
			# inpt_embedding = torch.cat((inpt_embedding, self.embed_context.expand(n_line_nodes, -1)), dim = 1)

			embedding, pos_rel_sta, pos_rel_src = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src[0], self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, self.embed_context) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, self.embed_context)
			Slice = torch.cat((Slice, embedding), dim = 1)

		
		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src, self.embed_context) if self.DataAggregation.use_offsets == False else self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src, self.embed_context, pos_rel_sta = pos_rel_sta, pos_rel_src = pos_rel_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, self.embed_context, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.embed_context, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		x_local = self.SpatialAggregation2(x, self.embed_context, self.A_src, x_temp_cuda)
		if self.use_expanded == True:
			x_expand = self.SpatialAggregation2_expanded(x, self.embed_context, self.Ac, x_temp_cuda) # x_temp_cuda_cart
			gate = torch.sigmoid(self.gate_expanded(torch.cat((x_local, x_expand, self.embed_context.expand(x_local.shape[0], -1)), dim = 1)))
			x = x_local + gate*x_expand
		else:
			x = x_local
		x_spatial = self.SpatialAggregation3(x, self.embed_context, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		

		# use_direct_output = False
		if self.use_direct_output == True:
			y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes

		else:
			y_latent = self.SpaceTimeAttention(x_spatial, x_temp_cuda_cart, x_temp_cuda_cart, x_temp_cuda_t, x_temp_cuda_t, self.embed_context) # contains data on spatial and temporal solution at fixed nodes

		y = self.proj_soln1(y_latent)
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t, self.embed_context) # second slowest module (could use this embedding to seed source source attention vector).
		
		x_src = []
		x = self.proj_soln2(x)

		slope_width = 0.1
		mask_p_thresh = 0.1
		mask_out = torch.relu(y - mask_p_thresh)
		

		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, self.embed_context, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer

		if self.use_expanded == False:
			s = self.DataAggregationAssociation(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, self.A_in_sta, self.A_in_src, self.embed_context, pos_rel_sta = pos_rel_sta, pos_rel_src = pos_rel_src) # detach x_latent. Just a "reference"

		else: ## This assumes that DataAggregationAssociationPhase does not use expanded version
			s = self.DataAggregationAssociation(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, self.A_in_sta, self.A_in_src[0], self.embed_context, pos_rel_sta = pos_rel_sta, pos_rel_src = pos_rel_src) # detach x_latent. Just a "reference"

		## Arrival embedding
		arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, self.A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, self.tlatent, self.embed_context, trv_out = trv_out_q)
		
		## x_query_src_cart
		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)

		else:
			arv = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)

		
		if self.use_src_pred == True:
			return y, x, arv_p, arv_s, src

		else:
			return y, x, arv_p, arv_s

		
	## Maye need to add new module that maps to the association - source locations
	def forward_fixed_source(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, t_query, n_reshape = 1, save_state = False):
	
		# embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)

		n_line_nodes = Slice.shape[0]
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)
		pos_rel_sta, pos_rel_src = None, None
		assert(x_temp_cuda_cart.shape[1] == 3)
		

		# if self.use_absolute_offset == True:
		# 	norm_pos = torch.sqrt(torch.sum(self.A_src_in_edges.x[:,0:3]**2, dim = 1, keepdim = True) + 1e-8)
		# 	gamma_offset = 1.6 * torch.tanh(self.f_gamma(self.embed_context))
		# 	gammas = torch.exp(self.log_gamma_base + gamma_offset)
		# 	spatial_decay = torch.exp(-1.0 * norm_pos * gammas)  # [N_product, 4]
		# 	rel_pos_feat = torch.cat((self.A_src_in_edges.x[:,0:3]/norm_pos, spatial_decay, self.A_src_in_edges.x[:,3:4]), dim=-1)
		# 	Slice = torch.cat((Slice, rel_pos_feat), dim = 1)
		
		if self.use_absolute_offset:
		    # Build 7D relative position feature vector
		    pos_rel_sp = self.A_src_in_edges.x[:, 0:3]
		    pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp**2, dim=1, keepdim=True) + 1e-8)
		    
		    delta = self.f_gamma(self.embed_context)
		    alpha = delta[:, :1]
		    residuals = 0.2 * torch.tanh(delta[:, 1:])
		    gammas = torch.exp(self.log_gamma_base[:, :3] + alpha + residuals)
		    spatial_decay = torch.exp(-1.0 * pos_norm_sp * gammas)
		    
		    pos_rel_tm = self.A_src_in_edges.x[:, 3:4]

		    rel_pos_feat = torch.cat((pos_rel_sp / pos_norm_sp, spatial_decay, pos_rel_tm), dim=-1) # 7D
		    Slice = torch.cat((Slice, rel_pos_feat), dim=1)		


		if self.use_embedding == True:
			ndim_slice = -1 if (self.attach_time == True)*(self.use_absolute_offset == False) else -8
			inpt_embedding = torch.cat((torch.ones(len(Slice),1, dtype = Slice.dtype, device = Slice.device),  Slice[:, ndim_slice::]), dim = 1) if ((self.attach_time == True) or (self.use_absolute_offset == True)) else torch.ones(len(Slice),1, dtype = Slice.dtype, device = Slice.device) # .to(Slice.device)
			# inpt_embedding = torch.cat((inpt_embedding, self.embed_context.expand(n_line_nodes, -1)), dim = 1)

			embedding, pos_rel_sta, pos_rel_src = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src[0], self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, self.embed_context) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, self.embed_context)
			Slice = torch.cat((Slice, embedding), dim = 1)


		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src, self.embed_context) if self.DataAggregation.use_offsets == False else self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src, self.embed_context, pos_rel_sta = pos_rel_sta, pos_rel_src = pos_rel_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, self.embed_context, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.embed_context, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		x_local = self.SpatialAggregation2(x, self.embed_context, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		if self.use_expanded == True:
			x_expand = self.SpatialAggregation2_expanded(x, self.embed_context, self.Ac, x_temp_cuda) # x_temp_cuda_cart
			gate = torch.sigmoid(self.gate_expanded(torch.cat((x_local, x_expand, self.embed_context.expand(x_local.shape[0], -1)), dim = 1)))
			x = x_local + gate*x_expand
		else:
			x = x_local
			
		x_spatial = self.SpatialAggregation3(x, self.embed_context, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)

		if save_state == True:
			self.set_internal_state(x_spatial, x_temp_cuda_cart, x_temp_cuda_t)
		
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t, self.embed_context) # second slowest module (could use this embedding to seed source source attention vector).
		x = self.proj_soln2(x)

		if n_reshape > 1: ## Use this to map (n_reshape) repeated spatial queries (x_temp_cuda_cart) at different origin times, to predictions for fixed coordinates and across time
			x = x.reshape(-1,n_reshape,1)

		return [], x


#### EXTRA


class VModel(nn.Module):

	def __init__(self, n_phases = 2, n_hidden = 50, n_embed = 10, device = 'cuda'): # v_mean = np.array([6500.0, 3400.0]), norm_pos = None, inorm_pos = None, inorm_time = None, norm_vel = None, conversion_factor = None, 
		super(VModel, self).__init__()

		## Relative offset prediction [2]
		self.fc1_1 = nn.Linear(3 + n_embed, n_hidden)
		self.fc1_2 = nn.Linear(n_hidden, n_hidden)
		self.fc1_3 = nn.Linear(n_hidden, n_hidden)
		self.fc1_4 = nn.ModuleList()
		for j in range(n_phases):
			self.fc1_4.append(nn.Linear(n_hidden, 1))
			# self.fc1_41 = nn.Linear(n_hidden, 1)
			# self.fc1_42 = nn.Linear(n_hidden, 1)
		self.activate1_1 = lambda x: torch.sin(x)
		self.activate1_2 = lambda x: torch.sin(x)
		self.activate1_3 = lambda x: torch.sin(x)
		self.activate = nn.Softplus()
		self.mask = torch.zeros((1, 3)).to(device) # + n_embed)).to(device)
		self.mask[0,2] = 1.0
		self.n_phases = n_phases

	def fc1_block(self, x):

		# x = x*torch.Tensor([0.0, 0.0, 1.0]).reshape(1,-1).to(x.device)
		x1 = self.activate1_1(self.fc1_1(x))
		x = self.activate1_2(self.fc1_2(x1)) + x1
		x1 = self.activate1_3(self.fc1_3(x)) + x
		# out = [self.activate(self.fc1_4[j](x1)) for j in range(self.n_phases)]

		return [self.activate(self.fc1_4[j](x1)) for j in range(self.n_phases)]

	def forward(self, src, embed):

		out = self.fc1_block(torch.cat((src, embed), dim = 1))
		lout = [out[0]]
		for j in range(1, self.n_phases):
			lout.append(out[0]*out[j])
		# out[:,1] = out[:,0]*out[:,1] ## Vs is a fraction of Vp

		return torch.cat(lout, dim = 1)


class TravelTimesPN1(nn.Module):

        def __init__(self, ftrns1, ftrns2, n_phases = 1, n_srcs = 0, n_hidden = 50, n_embed = 10, v_mean = np.array([6500.0, 3400.0]), norm_pos = None, inorm_pos = None, inorm_time = None, norm_vel = None, conversion_factor = None, corrs = None, locs_corr = None, device = 'cuda'):
                super(TravelTimesPN1, self).__init__()

                ## Relative offset prediction [2]
                self.fc1_1 = nn.Linear(4 + n_phases + n_embed, n_hidden)
                self.fc1_2 = nn.Linear(n_hidden, n_hidden)
                self.fc1_3 = nn.Linear(n_hidden, n_hidden)
                # self.fc1_4 = nn.Linear(n_hidden, n_phases)
                self.activate1_1 = lambda x: torch.sin(x)
                self.activate1_2 = lambda x: torch.sin(x)
                self.activate1_3 = lambda x: torch.sin(x)

                ## Absolute position prediction [3]
                self.fc2_1 = nn.Linear(7 + n_phases + n_embed, n_hidden)
                self.fc2_2 = nn.Linear(n_hidden, n_hidden)
                self.fc2_3 = nn.Linear(n_hidden, n_hidden)
                # self.fc2_4 = nn.Linear(n_hidden, n_phases)
                self.activate2_1 = lambda x: torch.sin(x)
                self.activate2_2 = lambda x: torch.sin(x)
                self.activate2_3 = lambda x: torch.sin(x)

                self.merge = nn.Sequential(nn.Linear(2*n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_phases))

                ## Embed source [3]
                # self.fc3_1 = nn.Linear(3 + 2 + 1, n_hidden)
                self.fc3_1 = nn.Linear(4, n_hidden)
                self.fc3_2 = nn.Linear(n_hidden, n_hidden)
                self.fc3_3 = nn.Linear(n_hidden, n_hidden)
                self.fc3_4 = nn.Linear(n_hidden, n_embed)
                self.activate3_1 = lambda x: torch.sin(x)
                self.activate3_2 = lambda x: torch.sin(x)
                self.activate3_3 = lambda x: torch.sin(x)

                ## Projection functions
                self.ftrns1 = ftrns1
                self.ftrns2 = ftrns2
                # self.scale = torch.Tensor([scale_val]).to(device) ## Might want to scale inputs before converting to Tensor
                # self.tscale = torch.Tensor([trav_val]).to(device)
                self.v_mean = torch.Tensor(v_mean).to(device)
                self.v_mean_norm = torch.Tensor(norm_vel(v_mean)).to(device)
                self.device = device
                self.norm_pos = norm_pos
                self.inorm_pos = inorm_pos
                self.inorm_time = inorm_time
                self.norm_vel = norm_vel
                self.conversion_factor = conversion_factor
                self.vmodel = VModel(n_phases = n_phases, n_embed = n_embed, device = device).to(device)
                self.mask = torch.Tensor([0.0, 0.0, 1.0]).reshape(1,-1).to(device)
                self.scale_angles = torch.Tensor([180.0, 180.0]).reshape(1,-1).to(device) ## Make these adaptive
                self.scale_depths = torch.Tensor([300e3]).reshape(1,-1).to(device)
                if locs_corr is not None:
                        self.tree_corr = cKDTree(ftrns1(torch.Tensor(locs_corr).to(device)).cpu().detach().numpy())
                        self.corrs = torch.Tensor(corrs).to(device)
                        self.use_corr = True
                else:
                        self.use_corr = False

                if n_srcs > 0:
                        self.reloc_x = nn.Parameter(torch.zeros((n_srcs, 3))) # .to(device)
                        self.reloc_t = nn.Parameter(torch.zeros((n_srcs, 1))) # .to(device)

                # self.Tp_average

        def fc1_block(self, x):

                x1 = self.activate1_1(self.fc1_1(x))
                x = self.activate1_2(self.fc1_2(x1)) + x1
                x1 = self.activate1_3(self.fc1_3(x)) + x

                return x1 # self.fc1_4(x1)

        def fc2_block(self, x):

                x1 = self.activate2_1(self.fc2_1(x))
                x = self.activate2_2(self.fc2_2(x1)) + x1
                x1 = self.activate2_3(self.fc2_3(x)) + x

                return x1 # self.fc2_4(x1)

        def fc3_block(self, x):

                x1 = self.activate3_1(self.fc3_1(x))
                x = self.activate3_2(self.fc3_2(x1)) + x1
                x1 = self.activate3_3(self.fc3_3(x)) + x

                return self.fc3_4(x1)

        def embed_src(self, src):

                return self.fc3_block(torch.cat((self.norm_pos(self.ftrns1(src)), self.norm_pos(src[:,2].reshape(-1,1))), dim = 1))

        # def embed_src(self, src):

        #       return self.fc3_block(torch.cat((self.norm_pos(self.ftrns1(src)), src[:,0:2]/self.scale_angles, src[:,[2]]/self.scale_depths), dim = 1))

        def src_proj(self, src):

                return self.norm_pos(self.ftrns1(src))

        def forward(self, sta, src, method = 'pairs', train = False):

                # embed_src = self.fc3_block(self.norm_pos(self.ftrns1(src)))
                # embed_src = self.embed_src(src*self.mask)
                embed_src = self.embed_src(src)

                if method == 'direct':

                        sta_proj = self.norm_pos(self.ftrns1(sta))
                        src_proj = self.norm_pos(self.ftrns1(src))

                        if train == True:
                                src_proj = Variable(src_proj, requires_grad = True)

                        base_val = self.conversion_factor*torch.norm(sta_proj - src_proj, dim = 1, keepdim = True)/self.v_mean_norm.reshape(1,-1)

                        pred1 = self.fc1_block( torch.cat((sta_proj - src_proj, self.norm_pos(src[:,2].reshape(-1,1)), base_val, embed_src), dim = 1) )
                        pred2 = self.fc2_block( torch.cat((sta_proj, src_proj, self.norm_pos(src[:,2]).reshape(-1,1), base_val, embed_src), dim = 1) )
                        pred = self.merge(torch.cat((pred1, pred2), dim = 1))

                        if train == True:
                                return base_val, pred, src_proj, embed_src

                        else:
                                if self.use_corr == True:
                                        imatch = self.tree_corr.query(self.ftrns1(sta).cpu().detach().numpy())[1]
                                        return torch.relu(self.inorm_time(base_val + pred) + self.corrs[imatch,:])

                                else:
                                        return torch.relu(self.inorm_time(base_val + pred))


                elif method == 'pairs':

                        ## First, create all pairs of srcs and recievers
                        src_repeat = self.norm_pos(self.ftrns1(src)).repeat_interleave(len(sta), dim = 0) # /self.scale
                        sta_repeat = self.norm_pos(self.ftrns1(sta)).repeat(len(src), 1) # /self.scale
                        src_embed_repeat = embed_src.repeat_interleave(len(sta), dim = 0)

                        if train == True:
                                src_repeat = Variable(src_repeat, requires_grad = True)

                        base_val = self.conversion_factor*(torch.norm(sta_repeat - src_repeat, dim = 1, keepdim = True)/self.v_mean_norm.reshape(1,-1)) # .reshape(len(src), len(sta), -1)

                        pred1 = self.fc1_block(torch.cat((sta_repeat - src_repeat, self.norm_pos(src[:,2].reshape(-1,1)).repeat_interleave(len(sta), dim = 0), base_val, src_embed_repeat), dim = 1)) # .reshape(len(src), len(sta), -1)
                        pred2 = self.fc2_block(torch.cat((sta_repeat, src_repeat, self.norm_pos(src[:,2].reshape(-1,1)).repeat_interleave(len(sta), dim = 0), base_val, src_embed_repeat), dim = 1)) # .reshape(len(src), len(sta), -1)
                        pred = self.merge(torch.cat((pred1, pred2), dim = 1)).reshape(len(src), len(sta), -1)

                        if train == True:
                                return base_val.reshape(len(src), len(sta), -1), pred, src_repeat.reshape(len(src), len(sta), -1), src_embed_repeat.reshape(len(src), len(sta), -1)

                        else:

                                if self.use_corr == True:
                                        imatch = self.tree_corr.query(self.ftrns1(sta).cpu().detach().numpy())[1]
                                        return torch.relu(self.inorm_time(base_val.reshape(len(src), len(sta), -1) + pred) + self.corrs[imatch,:].unsqueeze(0))

                                return torch.relu(self.inorm_time(base_val.reshape(len(src), len(sta), -1) + pred))
                                # return torch.relu(self.inorm_time(base_val.reshape(len(src), len(sta), -1) + pred))



class TravelTimesPN(nn.Module):

	def __init__(self, ftrns1, ftrns2, n_phases = 1, n_srcs = 0, n_hidden = 50, n_embed = 10, v_mean = np.array([6500.0, 3400.0]), norm_pos = None, inorm_pos = None, inorm_time = None, norm_vel = None, conversion_factor = None, corrs = None, locs_corr = None, device = 'cuda'):
		super(TravelTimesPN, self).__init__()

		## Relative offset prediction [2]
		self.fc1_1 = nn.Linear(3 + n_phases + n_embed, n_hidden)
		self.fc1_2 = nn.Linear(n_hidden, n_hidden)
		self.fc1_3 = nn.Linear(n_hidden, n_hidden)
		# self.fc1_4 = nn.Linear(n_hidden, n_phases)
		self.activate1_1 = lambda x: torch.sin(x)
		self.activate1_2 = lambda x: torch.sin(x)
		self.activate1_3 = lambda x: torch.sin(x)

		## Absolute position prediction [3]
		self.fc2_1 = nn.Linear(6 + n_phases + n_embed, n_hidden)
		self.fc2_2 = nn.Linear(n_hidden, n_hidden)
		self.fc2_3 = nn.Linear(n_hidden, n_hidden)
		# self.fc2_4 = nn.Linear(n_hidden, n_phases)
		self.activate2_1 = lambda x: torch.sin(x)
		self.activate2_2 = lambda x: torch.sin(x)
		self.activate2_3 = lambda x: torch.sin(x)

		self.merge = nn.Sequential(nn.Linear(2*n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_phases))

		## Embed source [3]
		# self.fc3_1 = nn.Linear(3 + 2 + 1, n_hidden)
		self.fc3_1 = nn.Linear(3, n_hidden)
		self.fc3_2 = nn.Linear(n_hidden, n_hidden)
		self.fc3_3 = nn.Linear(n_hidden, n_hidden)
		self.fc3_4 = nn.Linear(n_hidden, n_embed)
		self.activate3_1 = lambda x: torch.sin(x)
		self.activate3_2 = lambda x: torch.sin(x)
		self.activate3_3 = lambda x: torch.sin(x)

		## Projection functions
		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2
		# self.scale = torch.Tensor([scale_val]).to(device) ## Might want to scale inputs before converting to Tensor
		# self.tscale = torch.Tensor([trav_val]).to(device)
		self.v_mean = torch.Tensor(v_mean).to(device)
		self.v_mean_norm = torch.Tensor(norm_vel(v_mean)).to(device)
		self.device = device
		self.norm_pos = norm_pos
		self.inorm_pos = inorm_pos
		self.inorm_time = inorm_time
		self.norm_vel = norm_vel
		self.conversion_factor = conversion_factor
		self.vmodel = VModel(n_phases = n_phases, n_embed = n_embed, device = device).to(device)
		self.mask = torch.Tensor([0.0, 0.0, 1.0]).reshape(1,-1).to(device)
		self.scale_angles = torch.Tensor([180.0, 180.0]).reshape(1,-1).to(device) ## Make these adaptive
		self.scale_depths = torch.Tensor([300e3]).reshape(1,-1).to(device)
		if locs_corr is not None:
			self.tree_corr = cKDTree(ftrns1(torch.Tensor(locs_corr).to(device)).cpu().detach().numpy())
			self.corrs = torch.Tensor(corrs).to(device)
			self.use_corr = True
		else:
			self.use_corr = False
		
		if n_srcs > 0:
			self.reloc_x = nn.Parameter(torch.zeros((n_srcs, 3))) # .to(device)
			self.reloc_t = nn.Parameter(torch.zeros((n_srcs, 1))) # .to(device)

		# self.Tp_average

	def fc1_block(self, x):

		x1 = self.activate1_1(self.fc1_1(x))
		x = self.activate1_2(self.fc1_2(x1)) + x1
		x1 = self.activate1_3(self.fc1_3(x)) + x

		return x1 # self.fc1_4(x1)

	def fc2_block(self, x):

		x1 = self.activate2_1(self.fc2_1(x))
		x = self.activate2_2(self.fc2_2(x1)) + x1
		x1 = self.activate2_3(self.fc2_3(x)) + x

		return x1 # self.fc2_4(x1)

	def fc3_block(self, x):

		x1 = self.activate3_1(self.fc3_1(x))
		x = self.activate3_2(self.fc3_2(x1)) + x1
		x1 = self.activate3_3(self.fc3_3(x)) + x

		return self.fc3_4(x1)

	def embed_src(self, src):

		return self.fc3_block(self.norm_pos(self.ftrns1(src)))

	# def embed_src(self, src):

	# 	return self.fc3_block(torch.cat((self.norm_pos(self.ftrns1(src)), src[:,0:2]/self.scale_angles, src[:,[2]]/self.scale_depths), dim = 1))

	def src_proj(self, src):

		return self.norm_pos(self.ftrns1(src))

	def forward(self, sta, src, method = 'pairs', train = False):

		# embed_src = self.fc3_block(self.norm_pos(self.ftrns1(src)))
		# embed_src = self.embed_src(src*self.mask)
		embed_src = self.embed_src(src)

		if method == 'direct':

			sta_proj = self.norm_pos(self.ftrns1(sta))
			src_proj = self.norm_pos(self.ftrns1(src))

			if train == True:
				src_proj = Variable(src_proj, requires_grad = True)

			base_val = self.conversion_factor*torch.norm(sta_proj - src_proj, dim = 1, keepdim = True)/self.v_mean_norm.reshape(1,-1)

			pred1 = self.fc1_block( torch.cat((sta_proj - src_proj, base_val, embed_src), dim = 1) )
			pred2 = self.fc2_block( torch.cat((sta_proj, src_proj, base_val, embed_src), dim = 1) )
			pred = self.merge(torch.cat((pred1, pred2), dim = 1))

			if train == True:
				return base_val, pred, src_proj, embed_src

			else:
				if self.use_corr == True:
					imatch = self.tree_corr.query(self.ftrns1(sta).cpu().detach().numpy())[1]
					return torch.relu(self.inorm_time(base_val + pred) + self.corrs[imatch,:])

				else:
					return torch.relu(self.inorm_time(base_val + pred))

		
		elif method == 'pairs':

			## First, create all pairs of srcs and recievers
			src_repeat = self.norm_pos(self.ftrns1(src)).repeat_interleave(len(sta), dim = 0) # /self.scale
			sta_repeat = self.norm_pos(self.ftrns1(sta)).repeat(len(src), 1) # /self.scale
			src_embed_repeat = embed_src.repeat_interleave(len(sta), dim = 0)

			if train == True:
				src_repeat = Variable(src_repeat, requires_grad = True)

			base_val = self.conversion_factor*(torch.norm(sta_repeat - src_repeat, dim = 1, keepdim = True)/self.v_mean_norm.reshape(1,-1)) # .reshape(len(src), len(sta), -1)

			pred1 = self.fc1_block(torch.cat((sta_repeat - src_repeat, base_val, src_embed_repeat), dim = 1)) # .reshape(len(src), len(sta), -1)
			pred2 = self.fc2_block(torch.cat((sta_repeat, src_repeat, base_val, src_embed_repeat), dim = 1)) # .reshape(len(src), len(sta), -1)
			pred = self.merge(torch.cat((pred1, pred2), dim = 1)).reshape(len(src), len(sta), -1)

			if train == True:
				return base_val.reshape(len(src), len(sta), -1), pred, src_repeat.reshape(len(src), len(sta), -1), src_embed_repeat.reshape(len(src), len(sta), -1)

			else:

				if self.use_corr == True:
					imatch = self.tree_corr.query(self.ftrns1(sta).cpu().detach().numpy())[1]
					return torch.relu(self.inorm_time(base_val.reshape(len(src), len(sta), -1) + pred) + self.corrs[imatch,:].unsqueeze(0))		

				return torch.relu(self.inorm_time(base_val.reshape(len(src), len(sta), -1) + pred))
				# return torch.relu(self.inorm_time(base_val.reshape(len(src), len(sta), -1) + pred))


## Magnitude class
class Magnitude(nn.Module):
	def __init__(self, locs, grid, ftrns1_diff, ftrns2_diff, k = 1, device = 'cuda'):
		# super(Magnitude, self).__init__(aggr = 'max') # node dim
		super(Magnitude, self).__init__() # node dim
		## Predict magnitudes with trainable coefficients,
		## and spatial-reciver biases (with knn interp k)
		# In elliptical coordinates
		self.locs = locs
		self.grid = grid
		self.grid_cart = ftrns1_diff(grid)
		self.ftrns1 = ftrns1_diff
		self.ftrns2 = ftrns2_diff
		self.k = k
		self.device = device

		## Setup like regular log_amp = C1 * Mag + C2 * log_dist_depths_0 + C3 * log_dist_depths + Bias (for each phase type)
		self.mag_coef = nn.Parameter(torch.ones(2))
		self.epicenter_spatial_coef = nn.Parameter(torch.ones(2))
		self.depth_spatial_coef = nn.Parameter(torch.zeros(2))
		self.bias = nn.Parameter(torch.zeros(grid.shape[0], locs.shape[0], 2))
		self.activate = nn.Softplus()
		self.grid_save = nn.Parameter(grid, requires_grad = False)
		self.zvec = torch.Tensor([1.0,1.0,0.0]).reshape(1,-1).to(device)
		# self.bias = nn.Parameter(torch.zeros(locs.shape[0], grid.shape[0], 2), requires_grad = True).to(device)
	
	## Need to double check these routines
	def log_amplitudes(self, ind, src, mag, phase):
		## Input src: n_srcs x 3;
		## ind: indices into absolute locs array (can repeat, for phase types)
		## log_amp (base 10), for each ind
		## phase type for each ind 

		# Compute pairwise distances;
		fudge = 1.0 # add before log10, to avoid log10(0)
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2) + fudge)
		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)) + fudge)
		inds = knn(self.grid_cart/1000.0, self.ftrns1(src)/1000.0, k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)
		log_amp = mag*torch.maximum(self.activate(self.mag_coef[phase]), torch.Tensor([1e-12]).to(self.device)) - self.activate(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias
		# log_amp = mag*torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device)) + self.epicenter_spatial_coef[phase]*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias
		## Can directly use torch_scatter to coalesce the data
		
		return log_amp

	def train(self, ind, src, mag, phase):
		## Input src: n_srcs x 3;
		## ind: indices into absolute locs array (can repeat, for phase types)
		## log_amp (base 10), for each ind
		## phase type for each ind 

		# Compute pairwise distances;
		fudge = 1.0 # add before log10, to avoid log10(0)
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec) - self.ftrns1(self.locs[ind]*self.zvec), dim = 1) + fudge)
		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1) - self.locs[ind,2].view(-1)) + fudge)
		sta_ind = ind.repeat_interleave(self.k)
		inds = knn(self.grid_cart/1000.0, self.ftrns1(src)/1000.0, k = self.k) # [1] # .reshape(-1,self.k) ## for each of the second one, find indices in the first

		bias = self.bias[inds[1], sta_ind, :] # .mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)
		bias = scatter(bias, inds[0], dim = 0, reduce = 'mean')[torch.arange(len(src)).long().to(self.device),phase]
		log_amp = mag*torch.maximum(self.activate(self.mag_coef[phase]), torch.Tensor([1e-12]).to(self.device)) - self.activate(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

		return log_amp
	
	## Note, closer between amplitudes and forward
	def forward(self, ind, src, log_amp, phase):
		## Input src: n_srcs x 3;
		## ind: indices into absolute locs array (can repeat, for phase types)
		## log_amp (base 10), for each ind
		## phase type for each ind

		# Compute pairwise distances;
		fudge = 1.0 # add before log10, to avoid log10(0)
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2) + fudge)
		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)) + fudge)
		inds = knn(self.grid_cart/1000.0, self.ftrns1(src)/1000.0, k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)
		mag = (log_amp + self.activate(self.epicenter_spatial_coef[phase])*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/torch.maximum(self.activate(self.mag_coef[phase]), torch.Tensor([1e-12]).to(self.device))

		return mag

		## Can directly use torch_scatter to coalesce the data
		# bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)
		# log_amp = mag*torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device)) + self.epicenter_spatial_coef[phase]*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias


		## Can directly use torch_scatter to coalesce the data?
		# mag = (log_amp - self.epicenter_spatial_coef[phase]*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device))
