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


# scale_t = train_config['kernel_sig_t']*3.0
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
				 use_offsets=True, use_expanded=use_expanded, ndim_proj_sta=6, ndim_proj_src=9):
		super(DataAggregationLayer, self).__init__('mean')

		self.use_offsets = use_offsets
		self.use_expanded = use_expanded
		self.out_channels = out_channels

		# Local Branch Transforms
		self.l_t1_1 = nn.Linear(in_channels, out_channels)
		self.l_t1_2 = nn.Linear(in_channels + out_channels + n_dim_mask, out_channels)
		self.l_t2_1 = nn.Linear(in_channels, out_channels)
		self.l_t2_2 = nn.Linear(in_channels + out_channels + n_dim_mask, out_channels)

		self.act11, self.act12, self.act_local = nn.PReLU(), nn.PReLU(), nn.PReLU()
		self.film_local = FiLM(embed_dim, 2 * out_channels)

		# Expander Branch Transforms (Optional)
		if self.use_expanded:
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
			self.f_gamma_src = nn.Linear(embed_dim, 3 + 5)
			nn.init.normal_(self.f_gamma_sta.weight, std = 0.01); nn.init.zeros_(self.f_gamma_sta.bias)
			nn.init.normal_(self.f_gamma_src.weight, std = 0.01); nn.init.zeros_(self.f_gamma_src.bias)

	def _compute_edge_attrs(self, pos_rel_sta, pos_rel_src, embed_context):
		if not self.use_offsets or pos_rel_sta is None or pos_rel_src is None:
			return None, None

		# Station Edges (6D: 3D Direction + 3 Spatial RBFs)
		sta_sp = pos_rel_sta[:, 0:3]
		# sta_norm_sp = torch.sqrt(torch.sum(sta_sp**2, dim=1, keepdim=True) + 1e-6)
		sta_norm_sp = torch.linalg.vector_norm(sta_sp, dim = 1, keepdim = True)

		d_sta = self.f_gamma_sta(embed_context)
		gammas_sta = torch.exp(self.log_gamma_sta_base + 0.5 * torch.tanh(d_sta[:, :1]) + 0.2 * torch.tanh(d_sta[:, 1:]))
		# edge_sta = torch.cat((sta_sp / sta_norm_sp, torch.exp(-1.0 * sta_norm_sp * gammas_sta)), dim=1)
		edge_sta = torch.cat((sta_sp / sta_norm_sp.clamp(min = 1e-6), torch.exp(-1.0 * sta_norm_sp * gammas_sta)), dim=1)

		# Source Edges (9D: 3D Direction + 3 Spatial RBFs + 2 Temporal RBFs + 1 dt)
		src_sp, src_tm = pos_rel_src[:, 0:3], pos_rel_src[:, 3:4]
		# src_norm_sp = torch.sqrt(torch.sum(src_sp**2, dim=1, keepdim=True) + 1e-6)
		src_norm_sp = torch.linalg.vector_norm(src_sp, dim = 1, keepdim = True)

		src_norm_tm = torch.abs(src_tm)
		d_src = self.f_gamma_src(embed_context)
		alpha_src = 0.5*torch.tanh(d_src[:, 0:1])
		alpha_src_space = 0.25*torch.tanh(d_src[:, 1:2])
		alpha_src_time = 0.25*torch.tanh(d_src[:, 2:3])
		alpha_src = torch.cat([(alpha_src + alpha_src_space).expand(-1,3), (alpha_src + alpha_src_time).expand(-1,2)], dim = 1)
		gammas_src = torch.exp(self.log_gamma_src_base + alpha_src + 0.2 * torch.tanh(d_src[:, 3:]))
		
		sp_decay = torch.exp(-1.0 * src_norm_sp * gammas_src[:, 0:3])
		tm_decay = torch.exp(-1.0 * src_norm_tm * gammas_src[:, 3:5])
		# edge_src = torch.cat((src_sp / src_norm_sp, sp_decay, tm_decay, src_tm), dim=1)
		edge_src = torch.cat((src_sp / src_norm_sp.clamp(min = 1e-6), sp_decay, tm_decay, src_tm), dim=1)

		return edge_sta, edge_src

	def forward(self, x, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta=None, pos_rel_src=None):
		edge_sta, edge_src = self._compute_edge_attrs(pos_rel_sta, pos_rel_src, embed_context)

		# Local Path
		x1 = self.l_t1_2(torch.cat((x, self.propagate(A_in_sta, x=self.act11(self.l_t1_1(x)), edge_attr=edge_sta, edge_type=1), mask), dim=1))
		x2 = self.l_t2_2(torch.cat((x, self.propagate(A_in_src[0] if isinstance(A_in_src, (tuple, list)) else A_in_src, x=self.act12(self.l_t2_1(x)), edge_attr=edge_src, edge_type=2), mask), dim=1))
		x_local = self.act_local(torch.cat((x1, x2), dim=1))
		x_local = self.film_local(x_local, embed_context)

		if not self.use_expanded:
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
				 use_absolute_pos=True, use_offsets=True, embed_dim=10, n_embedding = 10, use_expanded = use_expanded, use_embedding=True):
		super(DataAggregationExpanded, self).__init__()

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
				embed_dim=embed_dim, use_offsets=use_offsets, use_expanded=False
			)
			self.geom_layer2 = DataAggregationLayer(
				in_channels=2 * n_hidden, out_channels=n_embedding, n_dim_mask=n_dim_mask, 
				embed_dim=embed_dim, use_offsets=use_offsets, use_expanded=False
			)
			in_channels += 2 * n_embedding  # Concatenate structural embedding to main input

		# --- MAIN OBSERVATION GNN STACK ---
		self.init_trns = nn.Linear(in_channels + n_dim_mask - 37, n_hidden)
		self.film_init = FiLM(embed_dim, n_hidden)
		self.act_init = nn.PReLU()
					 
		# # Calculate feature dimensions dynamically
		# main_in_dim = in_channels + (2 * n_embedding if self.use_embedding else 0) + n_dim_mask
		# self.init_trns = nn.Linear(main_in_dim, n_hidden)
					 
		self.layer1 = DataAggregationLayer(
			in_channels=n_hidden, out_channels=n_hidden, n_dim_mask=n_dim_mask, 
			embed_dim=embed_dim, use_offsets=use_offsets, use_expanded = use_expanded
		)
		self.layer2 = DataAggregationLayer(
			in_channels=2 * n_hidden, out_channels=n_hidden, n_dim_mask=n_dim_mask, 
			embed_dim=embed_dim, use_offsets=use_offsets, use_expanded = use_expanded
		)
		self.layer3 = DataAggregationLayer(
			in_channels=2 * n_hidden, out_channels=out_channels, n_dim_mask=n_dim_mask, 
			embed_dim=embed_dim, use_offsets=use_offsets, use_expanded = False
		)

	def forward(self, tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta=None, pos_rel_src=None):
		# 1. Run Preconditioner if Enabled

		g_emb = None
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
			# print('Use embedding')
			# print(tr.shape)

		# 2. Main Observation Processing Stack
		tr = torch.cat((tr, mask), dim=-1)
		# print(tr.shape)
		# print(self.init_trns)
		# print(self.film_init)
		tr = self.act_init(self.film_init(self.init_trns(tr), embed_context))
		# print(tr.shape)

		tr = self.layer1(tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
		tr = self.layer2(tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
		tr = self.layer3(tr, mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
		# print(tr.shape)

		return tr, g_emb.detach()


class BipartiteGraphOperator(MessagePassing):
	"""Product Graph to Source Graph Bipartite Projection Operator.

	Maps product-space features (Source-Station pairs) onto target source nodes.
	Uses multi-scale spatial RBFs, FiLM conditioning, mask gating, and
	coverage/evidence-aware backprojection.
	"""

	def __init__(
		self,
		ndim_in,
		ndim_out,
		ndim_mask=4,
		embed_dim=10,
		n_gammas=3, # 4
		scale_rel=scale_rel,
		scale_time=scale_time,
	):
		super(BipartiteGraphOperator, self).__init__(aggr="add")

		self.n_gammas = n_gammas
		self.scale_rel = scale_rel
		self.scale_time = scale_time

		# 1. Edge MLP
		self.fc_edge = nn.Linear(ndim_in + 3 + n_gammas, ndim_in)
		self.film_edge = FiLM(embed_dim, ndim_in)
		self.act_edge = nn.PReLU()

		# 2. Channel-wise Mask Gate
		self.mask_gate = nn.Sequential(
			nn.Linear(ndim_mask, 8), nn.PReLU(),
			nn.Linear(8, ndim_in), nn.Sigmoid(),
		)

		# 2b. Source-level Support Gate: [log_coverage, log_evidence, match_fraction]
		self.support_gate = nn.Sequential(
			nn.Linear(3, 8), nn.PReLU(), nn.Linear(8, 1), nn.Sigmoid()
		)
		nn.init.constant_(self.support_gate[-2].bias, -1.0)

		# 3. Dynamic Bandwidth Predictor
		self.f_gamma = nn.Linear(embed_dim, 1 + n_gammas)
		nn.init.normal_(self.f_gamma.weight, std=0.01)
		nn.init.zeros_(self.f_gamma.bias)

		init_spatial = torch.logspace(-2, 0.5, steps=n_gammas).reshape(1, -1)
		self.log_gamma_base = nn.Parameter(torch.log(init_spatial))

		# 4. Pattern Normalization and Readout
		# self.norm = nn.LayerNorm(ndim_in)
		self.norm = RMSNorm(ndim_in)
		self.fc_out = nn.Linear(ndim_in + 3, ndim_out)
		self.act_out = nn.PReLU()

	def forward(self, inpt, A_src_in_edges, mask, embed_context, num_target_nodes=None):
		"""
		Args:
			inpt: [E_edges, ndim_in]
			A_src_in_edges: PyG Data with edge_index and x [E_edges, 4]
			mask: [E_edges, ndim_mask]
			embed_context: [E_edges, embed_dim] or [1, embed_dim]
		"""
		N = inpt.shape[0]
		if num_target_nodes is not None:
			M = num_target_nodes
		else:
			M = A_src_in_edges.edge_index[1].max().item() + 1 if A_src_in_edges.edge_index.numel() > 0 else 0

		ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

		# Step 1: Spatial geometry
		diff_sp = A_src_in_edges.x[:, 0:3]
		norm_pos = torch.linalg.vector_norm(diff_sp, dim=1, keepdim=True)
		unit_dir = diff_sp / norm_pos.clamp(min=1e-6)

		# Step 2: Scale-conditioned RBF bandwidths
		delta = self.f_gamma(ctx)
		alpha = 0.5 * torch.tanh(delta[:, 0:1])
		residuals = 0.2 * torch.tanh(delta[:, 1:])
		gammas = torch.exp(self.log_gamma_base + alpha + residuals)

		# Step 3: Multi-scale spatial RBFs
		r_sp_sq = norm_pos ** 2
		r_aniso = torch.sqrt(gammas * r_sp_sq + 1e-5)
		rbf_decay = torch.exp(-r_aniso)

		# Step 4: Edge feature fusion
		rel_pos = torch.cat((unit_dir, rbf_decay), dim=-1)
		edge_inpt = torch.cat((inpt, rel_pos), dim=-1)
		geo_features = self.act_edge(self.film_edge(self.fc_edge(edge_inpt), ctx))

		# Step 5: Edge gating
		absolute_gate = mask.max(1, keepdims=True)[0]  # Forced per-edge support gate
		phase_routing = self.mask_gate(mask)
		msg = absolute_gate * phase_routing * geo_features

		# Step 6: Backprojection
		target_indices = A_src_in_edges.edge_index[1]
		stacked = scatter(msg, target_indices, dim=0, dim_size=M, reduce="sum")

		# Step 7: Coverage and evidence
		coverage = scatter(torch.ones_like(absolute_gate), target_indices, dim=0, dim_size=M, reduce="sum")
		evidence = scatter(absolute_gate, target_indices, dim=0, dim_size=M, reduce="sum")
		# has_evidence = (evidence > 0)
		hard_support = (evidence > 0).to(inpt.dtype)
		
		# Step 8: Coverage-normalized pattern + LayerNorm
		stacked_normalized = stacked / torch.sqrt(coverage.clamp(min=1.0))
		pattern = hard_support * self.norm(stacked_normalized)

		# Step 9: Explicit support features
		log_coverage = torch.log1p(coverage)
		log_evidence = torch.log1p(evidence)
		match_fraction = evidence / coverage.clamp(min=1.0)

		# Step 10: Merge pattern + support
		features = torch.cat((pattern, log_coverage, log_evidence, match_fraction), dim=-1)

		# Step 11: Readout
		out = self.act_out(self.fc_out(features))

		# Step 12: Source-level support gate
		support_features = torch.cat((log_coverage, log_evidence, match_fraction), dim=-1)
		learned_support = self.support_gate(support_features)

		# No observational evidence => no source activation
		out = out * learned_support # * hard_support

		return out, torch.cat((support_features, hard_support*learned_support), dim = 1).detach()



class BipartiteGraphOperator1(MessagePassing):
	"""
	Product Graph to Source Graph Bipartite Projection Operator.
	Features:
	- Multi-scale dynamic spatial kernels (scaled by scale_rel).
	- Continuous learned edge gating & distance-aware missing-station penalty.
	- Multi-scale + Per-Phase coverage/evidence backprojection.
	"""

	def __init__(self, ndim_in, ndim_out, ndim_mask = 4, embed_dim = 10, n_gammas = 3,
				 n_kernels = 3, scale_rel = scale_rel, scale_time = scale_time,
				 n_phase_channels=2):  # e.g., 2 for [P, S]
		super().__init__(aggr="add")

		self.n_gammas = n_gammas
		self.scale_rel = scale_rel
		self.scale_time = scale_time
		self.n_phase_channels = n_phase_channels  # [P, S]


		self.n_dist_kernels = n_kernels
		init_radii = torch.logspace(-2, 0.5, steps=n_kernels)   # (K,)
		self.log_kernel_radii = nn.Parameter(init_radii.log())
		self.f_radii = nn.Linear(embed_dim, 1 + n_kernels)
		nn.init.normal_(self.f_radii.weight, std=0.01)
		nn.init.zeros_(self.f_radii.bias)

		# 1. Edge MLP
		self.fc_edge = nn.Linear(ndim_in + 3 + n_gammas, ndim_in)
		self.film_edge = FiLM(embed_dim, ndim_in)
		self.act_edge = nn.PReLU()

		# 2. Continuous Edge Support & Penalty Predictor
		# Takes inpt + mask features + distance norm
		self.fc_edge_gates = nn.Sequential(
			nn.Linear(ndim_in + ndim_mask + 1, 32), nn.PReLU(),
			nn.Linear(32, 2), nn.Sigmoid()  # [pos_support_gate, neg_penalty_gate]
		)

		# 2b. Channel-wise Feature Routing
		self.mask_gate = nn.Sequential(
			nn.Linear(ndim_mask, 16), nn.PReLU(),
			nn.Linear(16, ndim_in), nn.Sigmoid()
		)

		# 3. Dynamic Bandwidth Predictor (RBF)
		self.f_gamma = nn.Linear(embed_dim, 1 + n_gammas)
		nn.init.normal_(self.f_gamma.weight, std=0.01)
		nn.init.zeros_(self.f_gamma.bias)

		init_spatial = torch.logspace(-2, 0.5, steps=n_gammas).reshape(1, -1)
		self.log_gamma_base = nn.Parameter(torch.log(init_spatial))

		# 4. Source-level Multi-Scale & Multi-Phase Support Gate
		# Support channels: (N_phases + 1 [Joint]) * N_dist_kernels * 3 [log_cov, log_ev, match_frac]
		self.n_support_groups = (self.n_phase_channels + 1) * self.n_dist_kernels
		support_feat_dim = 3 * self.n_support_groups

		self.support_gate = nn.Sequential(
			nn.Linear(support_feat_dim, 32), nn.PReLU(),
			nn.Linear(32, 16), nn.PReLU(),
			nn.Linear(16, 1), nn.Sigmoid()
		)
		nn.init.constant_(self.support_gate[-2].bias, -1.0)

		# 5. Readout
		self.norm = RMSNorm(ndim_in)
		self.fc_out = nn.Linear(ndim_in + support_feat_dim, ndim_out)
		self.act_out = nn.PReLU()

	def forward(self, inpt, A_src_in_edges, mask, embed_context, num_target_nodes=None):

		## Could re-use N instead of E_edges
		E_edges = A_src_in_edges.edge_index.shape[1] if A_src_in_edges.edge_index.numel() > 0 else 0
		
		if num_target_nodes is not None:
			M = num_target_nodes
		else:
			M = A_src_in_edges.edge_index[1].max().item() + 1 if E_edges > 0 else 0

		ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

		# Dynamic physical radii scaled relative to region/domain
		# effective_radii = torch.exp(self.log_kernel_radii) * self.scale_rel  # [N_kernels]

		# delta_r = self.f_radii(ctx)
		# alpha_r = 0.5 * torch.tanh(delta_r[:, :1])		  # shared log-shift
		# resid_r = 0.2 * torch.tanh(delta_r[:, 1:])		  # per-scale tweak
		# log_r = self.log_kernel_radii + alpha_r + resid_r
		# effective_radii = torch.exp(log_r)				  # (1, K) or (K,)

		delta_r = self.f_radii(ctx)
		alpha_r = 0.5 * torch.tanh(delta_r[:, :1])
		resid_r = 0.2 * torch.tanh(delta_r[:, 1:])
		effective_radii = torch.exp(self.log_kernel_radii + alpha_r + resid_r).reshape(-1)  # (K,)

		# near_field_bias = torch.exp(-norm_pos / effective_radii[0].clamp(min=1e-6))
		# dist_weights = torch.exp(-norm_pos / effective_radii.clamp(min=1e-6))  # (E, K)

		# Step 1: Spatial geometry
		diff_sp = A_src_in_edges.x[:, 0:3]
		norm_pos = torch.linalg.vector_norm(diff_sp, dim=1, keepdim=True)  # [E_edges, 1]
		unit_dir = diff_sp / norm_pos.clamp(min=1e-6)

		# Step 2: Scale-conditioned RBF bandwidths
		delta = self.f_gamma(ctx)
		alpha = 0.5 * torch.tanh(delta[:, 0:1])
		residuals = 0.2 * torch.tanh(delta[:, 1:])
		gammas = torch.exp(self.log_gamma_base + alpha + residuals)

		# Step 3: Multi-scale spatial RBFs
		r_sp_sq = norm_pos ** 2
		r_aniso = torch.sqrt(gammas * r_sp_sq + 1e-5)
		rbf_decay = torch.exp(-r_aniso)

		# Step 4: Edge feature fusion
		rel_pos = torch.cat((unit_dir, rbf_decay), dim=-1)
		edge_inpt = torch.cat((inpt, rel_pos), dim=-1)
		geo_features = self.act_edge(self.film_edge(self.fc_edge(edge_inpt), ctx))

		# Step 5: Continuous Learned Edge Support & Distance-Aware Penalty
		edge_gate_in = torch.cat((inpt, mask, norm_pos), dim=-1)
		gates = self.fc_edge_gates(edge_gate_in)
		pos_gate = gates[:, 0:1]  # [E_edges, 1]
		neg_gate = gates[:, 1:2]  # [E_edges, 1]

		phase_routing = self.mask_gate(mask)

		near_field_bias = torch.exp(-norm_pos / effective_radii[0].clamp(min=1e-6))
		msg = (pos_gate * phase_routing * geo_features) - (neg_gate * (1.0 - pos_gate) * near_field_bias * geo_features)

		# Step 6: Backprojection to Source Graph
		target_indices = A_src_in_edges.edge_index[1]
		stacked = scatter(msg, target_indices, dim=0, dim_size=M, reduce="sum")

		# Step 7: Per-Phase + Joint Multi-Scale Support (Fully Vectorized)
		dist_weights = torch.exp(-norm_pos / effective_radii.clamp(min=1e-6).unsqueeze(0))  # [E_edges, K_kernels]

		phase_masks = mask[:, :self.n_phase_channels]  # [E_edges, N_phases]
		joint_mask = torch.ones_like(pos_gate)		 # [E_edges, 1]
		all_masks = torch.cat([phase_masks, joint_mask], dim=-1)  # [E_edges, P_phases]

		# Expand dims for 3D broadcasting: [E_edges, P_phases, K_kernels]
		all_masks_3d = all_masks.unsqueeze(-1)
		pos_gate_3d = pos_gate.unsqueeze(-1)
		dist_weights_3d = dist_weights.unsqueeze(1)

		evidence_3d = (pos_gate_3d * all_masks_3d) * dist_weights_3d
		coverage_3d = all_masks_3d * dist_weights_3d

		P_phases = self.n_phase_channels + 1
		K_kernels = self.n_dist_kernels

		# Flatten both to 2D using -1 (safe for any E_edges)
		evidence_flat = evidence_3d.view(-1, P_phases * K_kernels)
		coverage_flat = coverage_3d.view(-1, P_phases * K_kernels)

		# Stack into single tensor [E_edges, 2 * P_phases * K_kernels] for 1 single scatter call
		combined_flat = torch.cat([evidence_flat, coverage_flat], dim=-1)
		combined_src = scatter(combined_flat, target_indices, dim=0, dim_size=M, reduce="sum")

		# Split back into evidence and coverage [M_sources, P_phases * K_kernels]
		evidence_src, coverage_src = combined_src.chunk(2, dim=-1)

		# Parallel metric computation
		match_fraction_src = evidence_src / coverage_src.clamp(min=1e-5)
		log_coverage_src = torch.log1p(coverage_src)
		log_evidence_src = torch.log1p(evidence_src)

		# Reshape to [M_sources, P_phases, K_kernels, 1] and interleave metrics
		log_coverage_3d = log_coverage_src.view(M, P_phases, K_kernels, 1)
		log_evidence_3d = log_evidence_src.view(M, P_phases, K_kernels, 1)
		match_fraction_3d = match_fraction_src.view(M, P_phases, K_kernels, 1) ## 2 * 

		multi_scale_support = torch.cat([log_coverage_3d, log_evidence_3d, match_fraction_3d], dim=-1).view(M, -1)

		# Step 8: Pattern Normalization
		# Extract global joint coverage from the last phase group (Joint) and unweighted kernel
		global_coverage = scatter(torch.ones_like(pos_gate), target_indices, dim=0, dim_size=M, reduce="sum")
		hard_support = (global_coverage > 0).to(inpt.dtype) ## Note global coverage is already computed
		# # Optional: Replace Step 8 scatter with direct indexing if you add a unit kernel,
		# OR keep it as-is if E_edges is small. The speedup from Step 7 vectorization is already ~10-15x.

		stacked_normalized = stacked / torch.sqrt(global_coverage.clamp(min=1.0))
		pattern = hard_support * self.norm(stacked_normalized)

		# Step 9: Readout & Gating
		features = torch.cat((pattern, multi_scale_support), dim=-1)
		out = self.act_out(self.fc_out(features))

		learned_support = self.support_gate(multi_scale_support)
		out = out * learned_support

		return out, torch.cat((multi_scale_support, hard_support * learned_support), dim=1).detach()





use_anisotropic_spatial_aggregation = False
if use_anisotropic_spatial_aggregation == True:

	class StableAnisotropicSpatialAggregation(MessagePassing):
		def __init__(self, in_channels, out_channels, embed_dim=10, scale_sp=1.0, scale_tm=1.0, 
					 n_gammas=3, n_global=5, n_hidden=30, zero_offsets=False):
			super().__init__(aggr='mean')
			self.zero_offsets = zero_offsets
			self.scale_sp = scale_sp
			self.scale_tm = scale_tm
			self.n_gammas = n_gammas

			if not self.zero_offsets:
				# Predict 1 global scale + (n_gammas * 4) directional residuals
				self.f_gamma = nn.Linear(embed_dim, 1 + n_gammas * 4)
				nn.init.normal_(self.f_gamma.weight, std=0.001) # Small init for stability
				nn.init.zeros_(self.f_gamma.bias)

				# Log-spaced initial bandwidths
				init_spatial = torch.logspace(-2, 0.5, steps=n_gammas).unsqueeze(1).repeat(1, 3)
				init_temporal = torch.logspace(-1, 0.7, steps=n_gammas).unsqueeze(1)
				init_gammas = torch.cat((init_spatial, init_temporal), dim=1).unsqueeze(0)
				self.log_gamma_base = nn.Parameter(torch.log(init_gammas))

				edge_dim = 3 + n_gammas + 1  # dir(3) + RBF(n_gammas) + dt(1)
			else:
				edge_dim = 0

			self.fc1 = nn.Linear(in_channels + edge_dim + n_global, n_hidden)
			self.fc2 = nn.Linear(n_hidden + in_channels, out_channels)
			self.fglobal = nn.Linear(in_channels, n_global)
			self.film = FiLM(embed_dim, n_hidden)

			self.activate1 = nn.PReLU()
			self.activate2 = nn.PReLU()
			self.activate3 = nn.PReLU()

		def forward(self, tr, embed_context, A_src, pos):
			ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

			if not self.zero_offsets:
				# Explicit physical scale separation
				diff_sp = (pos[A_src[1]] - pos[A_src[0]]) / self.scale_sp
				diff_tm = diff_sp[:,3:4] # (pos[A_src[1], 3:4] - pos[A_src[0], 3:4]) # / self.scale_tm

				norm_sp = torch.linalg.vector_norm(diff_sp, dim=1, keepdim=True)
				unit_dir = diff_sp / norm_sp.clamp(min=1e-6)

				# Predict gammas with tight bounding on anisotropic variations
				delta = self.f_gamma(ctx)
				alpha = delta[:, :1].unsqueeze(-1)
				residuals = 0.1 * torch.tanh(delta[:, 1:].view(-1, self.n_gammas, 4)) # Tighter clamp (0.1)
				gammas = torch.exp(self.log_gamma_base + alpha + residuals)

				# Quadratic Mahalanobis Distance (No sqrt = Smooth Gradients!)
				r_sq = torch.cat((diff_sp ** 2, diff_tm ** 2), dim=1).unsqueeze(1) # [E, 1, 4]
				dist_mahalanobis_sq = torch.sum(gammas * r_sq, dim=-1)			 # [E, n_gammas]
				
				# Smooth Gaussian RBF decay
				rbf_decay = torch.exp(-0.5 * dist_mahalanobis_sq)

				edge_attr = torch.cat((unit_dir, rbf_decay, diff_tm), dim=-1)
			else:
				edge_attr = torch.zeros((A_src.shape[1], 0), dtype=tr.dtype, device=tr.device)

			global_feat = self.activate3(self.fglobal(tr)).mean(dim=0, keepdim=True)
			aggr_out = self.propagate(A_src, x=tr, edge_attr=edge_attr, global_feat=global_feat, embed_context=ctx)
			
			out = torch.cat((tr, aggr_out), dim=-1)
			return self.activate2(self.fc2(out))

else:

	
	class SpatialAggregation(MessagePassing):
		def __init__(self, in_channels, out_channels, embed_dim=10, scale_rel=scale_rel,
					 n_global=5, n_hidden=30, zero_offsets=False, support_dim=4):
			super(SpatialAggregation, self).__init__(aggr='mean')
	
			self.zero_offsets = zero_offsets
			self.scale_rel = scale_rel
			self.support_dim = support_dim
	
			if not self.zero_offsets:
				self.f_gamma = nn.Linear(embed_dim, 3 + 5)
				nn.init.normal_(self.f_gamma.weight, std=0.01)
				nn.init.zeros_(self.f_gamma.bias)
	
				init_gammas = torch.tensor([0.1, 1.0, 5.0, 0.5, 10.0]).reshape(1, -1)
				self.log_gamma_base = nn.Parameter(torch.log(init_gammas))
				edge_dim = 9
			else:
				edge_dim = 0
	
			# Protects against unbounded log_coverage/log_evidence values
			self.f_support = nn.Sequential(
				nn.Linear(support_dim, 8),
				nn.PReLU(),
				nn.Linear(8, 8) ## Maybe add layer norm
			)
	
			self.fc1 = nn.Linear(in_channels + 8 + edge_dim + n_global, n_hidden)
			self.fc2 = nn.Linear(n_hidden + in_channels + 8, out_channels)
			self.fglobal = nn.Linear(in_channels, n_global)
	
			self.film = FiLM(embed_dim, n_hidden)
	
			self.activate1 = nn.PReLU()
			self.activate2 = nn.PReLU()
			self.activate3 = nn.PReLU()
	
		def forward(self, tr, embed_context, A_src, pos, support=None):
			ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)
	
			if support is None:
				support = torch.zeros((tr.shape[0], self.support_dim), dtype=tr.dtype, device=tr.device)
	
			if not self.zero_offsets:
				pos_rel = (pos[A_src[1]] - pos[A_src[0]]) / self.scale_rel
				pos_rel_sp = pos_rel[:, 0:3]
				pos_norm_sp = torch.linalg.vector_norm(pos_rel_sp, dim=1, keepdim=True)
				pos_rel_tm = pos_rel[:, 3:4]
				pos_norm_tm = torch.abs(pos_rel_tm)
	
				delta = self.f_gamma(ctx)
				alpha_global = 0.5 * torch.tanh(delta[:, 0:1])
				alpha_space = 0.25 * torch.tanh(delta[:, 1:2])
				alpha_time = 0.25 * torch.tanh(delta[:, 2:3])
				residuals = 0.2 * torch.tanh(delta[:, 3:])
	
				alpha = torch.cat([
					(alpha_global + alpha_space).expand(-1, 3),
					(alpha_global + alpha_time).expand(-1, 2)
				], dim=1)
	
				gammas = torch.exp(self.log_gamma_base + alpha + residuals)
				edge_gammas = gammas[A_src[0]] if gammas.shape[0] > 1 else gammas
	
				# Preserved: Continuous Exponential (Laplacian) decay for robust multi-scale tails
				spatial_decay = torch.exp(-pos_norm_sp * edge_gammas[:, 0:3])
				temporal_decay = torch.exp(-pos_norm_tm * edge_gammas[:, 3:5])
	
				edge_attr = torch.cat((
					pos_rel_sp / pos_norm_sp.clamp(min=1e-6),
					spatial_decay,
					temporal_decay,
					pos_rel_tm
				), dim=1)
			else:
				edge_attr = torch.zeros((A_src.shape[1], 0), dtype=tr.dtype, device=tr.device)
	
			global_feat = self.activate3(self.fglobal(tr)).mean(dim=0, keepdim=True)
			support_embed = self.f_support(support)
	
			aggr_out = self.propagate(
				A_src,
				x=tr,
				support_embed=support_embed,
				edge_attr=edge_attr,
				global_feat=global_feat,
				embed_context=ctx,
			)
	
			out = torch.cat((tr, aggr_out, support_embed), dim=-1)
			return self.activate2(self.fc2(out))
	
		def message(self, x_j, support_embed_j, edge_attr, global_feat, embed_context):
			if not self.zero_offsets:
				inputs = torch.cat((
					x_j,
					support_embed_j,
					edge_attr,
					global_feat.expand(len(x_j), -1)
				), dim=-1)
			else:
				inputs = torch.cat((
					x_j,
					support_embed_j,
					global_feat.expand(len(x_j), -1)
				), dim=-1)
	
			h = self.fc1(inputs)
			return self.activate1(self.film(h, embed_context))




class SpaceTimeDirect(nn.Module):
	def __init__(self, inpt_dim, out_channels):
		super(SpaceTimeDirect, self).__init__() #  "Max" aggregation.

		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.activate = nn.PReLU()

	def forward(self, inpts):

		return self.activate(self.f_direct(inpts))



class SpaceTimeAttention(MessagePassing):
	"""Continuous 4D renderer from sparse source hypotheses.

	Geometry strictly dictates spatial attention weights. Source support acts as an 
	additive value channel without imposing sharp zero-cliffs on Gaussian tails.
	"""

	def __init__(self, inpt_dim, out_channels, n_dim=4, n_latent=16, embed_dim=10,
				 n_heads=5, support_dim=4, scale_rel=scale_rel, scale_time=scale_time):
		super(SpaceTimeAttention, self).__init__(node_dim=0, aggr="add")

		self.n_heads = n_heads
		self.n_latent = n_latent
		self.support_dim = support_dim
		self.scale_rel = scale_rel
		self.scale_time = scale_time

		# Source value + support embedding
		self.f_values = nn.Linear(inpt_dim, n_latent)
		self.film_values = FiLM(embed_dim, n_latent)
		self.act_values = nn.PReLU()

		self.f_support = nn.Sequential(
			nn.Linear(support_dim, 8), nn.PReLU(), nn.Linear(8, n_latent)
		)

		# Source-feature attention correction
		self.f_feature_score = nn.Linear(inpt_dim, n_heads)
		self.film_score = FiLM(embed_dim, n_heads)

		# Dynamic space-time bandwidths
		self.f_gamma = nn.Linear(embed_dim, 3 + 2 * n_heads)
		nn.init.normal_(self.f_gamma.weight, std=0.01)
		nn.init.zeros_(self.f_gamma.bias)

		init_spatial = torch.logspace(-1, 0.7, steps=n_heads).unsqueeze(1)
		init_temporal = torch.logspace(-0.3, 1.0, steps=n_heads).unsqueeze(1)
		init_gammas = torch.cat([init_spatial, init_temporal], dim=1).unsqueeze(0)
		self.log_gamma_base = nn.Parameter(torch.log(init_gammas))

		# Geometry -> latent edge embedding
		rbf_edge_dim = 3 + 2 * n_heads + 1
		self.edge_proj = nn.Sequential(
			nn.Linear(rbf_edge_dim, n_latent), nn.PReLU(), nn.Linear(n_latent, n_latent)
		)

		# Bounded continuous-peak recovery
		self.f_max_gain_cap = nn.Sequential(
			nn.Linear(embed_dim, 16), nn.PReLU(), nn.Linear(16, 1), nn.Sigmoid()
		)

		# Query confidence gate
		self.spatial_gate = nn.Sequential(
			nn.Linear(n_latent * n_heads, 1), nn.Sigmoid()
		)

		# Pure feature readout + FiLM context modulation
		self.proj = nn.Linear(n_latent * n_heads, out_channels)
		self.film_readout = FiLM(embed_dim, out_channels)
		self.activate2 = nn.PReLU()

		self.use_fixed_edges = False
		self.fixed_edges = None
		self.edge_features = None

	def _build_edge_attr(self, x_query, x_context, x_query_t, x_context_t, k=16):
		ctx_4d = torch.cat((
			x_context / self.scale_rel,
			(1000.0 * self.scale_time * x_context_t).reshape(-1, 1) / self.scale_rel
		), dim=1)
		qry_4d = torch.cat((
			x_query / self.scale_rel,
			(1000.0 * self.scale_time * x_query_t).reshape(-1, 1) / self.scale_rel
		), dim=1)

		edge_index = knn(ctx_4d, qry_4d, k=k).flip(0).to(x_query.device)

		diff_sp = (
			x_query[edge_index[1], :3] - x_context[edge_index[0], :3]
		) / self.scale_rel
		diff_tm = (
			1000.0 * self.scale_time *
			(x_query_t[edge_index[1]].view(-1) - x_context_t[edge_index[0]].view(-1))
		).reshape(-1, 1) / self.scale_rel

		return edge_index, torch.cat((diff_sp, diff_tm), dim=1)

	def set_edges(self, x_query, x_context, x_query_t, x_context_t, k=16):
		self.fixed_edges, self.edge_features = self._build_edge_attr(
			x_query, x_context, x_query_t, x_context_t, k=k
		)
		self.use_fixed_edges = True

	def message(self, x_j, support_j, embed_context, index, edge_attr):
		pos_rel_sp = edge_attr[:, :3]
		pos_rel_tm = edge_attr[:, 3:4]

		pos_norm_sp = torch.linalg.vector_norm(pos_rel_sp, dim=1, keepdim=True)
		pos_norm_tm = torch.abs(pos_rel_tm)
		spatial_sq = pos_norm_sp.square()
		temporal_sq = pos_norm_tm.square()

		# Dynamic multi-scale bandwidths
		delta = self.f_gamma(embed_context)
		alpha_global = 0.5 * torch.tanh(delta[:, 0:1])
		alpha_space = 0.25 * torch.tanh(delta[:, 1:2])
		alpha_time = 0.25 * torch.tanh(delta[:, 2:3])

		alpha = (
			alpha_global.unsqueeze(1) +
			torch.cat([alpha_space, alpha_time], dim=1).unsqueeze(1)
		)
		residuals = 0.1 * torch.tanh(delta[:, 3:].view(-1, self.n_heads, 2))

		gammas = torch.exp(self.log_gamma_base + alpha + residuals)
		gammas_sp = torch.clamp(gammas[:, :, 0], min=1e-3, max=50.0)
		gammas_tm = torch.clamp(gammas[:, :, 1], min=1e-3, max=50.0)

		# Multi-scale geometric RBF features
		rbf_spatial = torch.exp(-gammas_sp * spatial_sq)
		rbf_temporal = torch.exp(-gammas_tm * temporal_sq)
		unit_dir_sp = pos_rel_sp / torch.sqrt(spatial_sq + 1e-4)
		
		rbf_edge_attr = torch.cat((
			unit_dir_sp, rbf_spatial, rbf_temporal, pos_rel_tm
		), dim=1)
		edge_embed = self.edge_proj(rbf_edge_attr)

		# Source content + geometry + support (Additive incorporation)
		value_embed = self.act_values(
			self.film_values(self.f_values(x_j), embed_context)
		)
		value_embed = value_embed + edge_embed + self.f_support(support_j)

		# Pure Geometric distance attention logits
		distance_logits = (-gammas_sp * spatial_sq - gammas_tm * temporal_sq)

		# Bounded content correction
		source_score = 0.2 * torch.tanh(
			self.film_score(self.f_feature_score(x_j), embed_context)
		)

		logits = distance_logits + source_score

		# KNN BOUNDARY FADING (Safe -1e9 mask)
		dist_4d_sq = spatial_sq + temporal_sq
		max_dist_sq = scatter(dist_4d_sq, index, dim=0, reduce="max")[index] + 1e-6
		fade_factor = (1.0 - (dist_4d_sq / max_dist_sq)).clamp(min=0.0)

		log_fade = torch.where(
			fade_factor < 1e-5,
			torch.full_like(fade_factor, -1e9),
			torch.log(fade_factor.clamp(min=1e-5))
		)
		logits = logits + log_fade

		alpha_attn = softmax(logits, index)

		head_values = (
			alpha_attn.unsqueeze(-1) * value_embed.unsqueeze(1)
		).reshape(-1, self.n_heads * self.n_latent)

		return torch.cat((
			head_values,
			alpha_attn.square()
		), dim=1)

	def update(self, aggr_out):
		n_value = self.n_latent * self.n_heads
		n_head = self.n_heads

		agg_values = aggr_out[:, :n_value]
		agg_alpha_sq = aggr_out[:, n_value:n_value + n_head]

		# Spatial concentration and recovery
		concentration = agg_alpha_sq.mean(dim=1, keepdim=True)
		
		# Peak recovery scales purely with attention dispersion (smooth continuous recovery)
		recovery = (1.0 - concentration).clamp(0.0, 1.0)

		return agg_values, recovery

	def forward(self, inpts, x_query, x_context, x_query_t, x_context_t,
				embed_context, support, k=16):
		if self.use_fixed_edges and self.fixed_edges is not None:
			edge_index, edge_attr = self.fixed_edges, self.edge_features
		else:
			edge_index, edge_attr = self._build_edge_attr(
				x_query, x_context, x_query_t, x_context_t, k=k
			)

		ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)

		interpolated, recovery = self.propagate(
			edge_index,
			x=inpts,
			support=support,
			embed_context=ctx,
			edge_attr=edge_attr,
			size=(x_context.shape[0], x_query.shape[0])
		)

		# Bounded correction for continuous Gaussian peaks between samples
		max_gain = 1.5 * self.f_max_gain_cap(ctx)
		local_gain = 1.0 + max_gain * recovery
		interpolated = interpolated * local_gain

		# Readout: Pure Spatial Aggregation
		out = self.proj(interpolated)

		# FiLM Context Modulation
		out = self.film_readout(out, ctx)

		# Non-linear activation BEFORE query gating
		out = self.activate2(out)

		# Continuous learned query gate (prevents tail truncation on wide Gaussians)
		gate = self.spatial_gate(interpolated)
		out = out * gate

		return out



class BipartiteGraphReadOutOperator(nn.Module):
	"""Source Space -> Product Graph Space readout.

	Propagates source hypotheses onto source-station product edges using
	multi-scale spatial RBFs, FiLM conditioning, and soft source-support gating.
	Each product edge has exactly one incoming source, so no aggregation is needed.
	"""

	def __init__(
		self,
		ndim_in,
		ndim_out,
		ndim_mask=1,
		embed_dim=10,
		n_gammas=3, # 4
		support_dim=4,
		baseline_gate=0.001, # 0.01
	):
		super(BipartiteGraphReadOutOperator, self).__init__()

		self.n_gammas = n_gammas
		self.baseline_gate = baseline_gate

		# 1. Edge Feature Evaluator
		self.fc_edge = nn.Linear(ndim_in + 3 + n_gammas + 8, ndim_in)
		self.film_edge = FiLM(embed_dim, ndim_in)
		self.act_edge = nn.PReLU()

		# 2. Phase / Mask Gate Router
		self.mask_gate = nn.Sequential(
			nn.Linear(ndim_mask, 8), nn.PReLU(),
			nn.Linear(8, ndim_in), nn.Sigmoid(),
		)

		# Protects against unbounded log_coverage/log_evidence values
		self.f_support = nn.Sequential(
			nn.Linear(support_dim, 8),
			nn.PReLU(),
			nn.Linear(8, 8)
		)

		# 3. Dynamic Bandwidth Predictor
		self.f_gamma = nn.Linear(embed_dim, 1 + n_gammas)
		nn.init.normal_(self.f_gamma.weight, std=0.01)
		nn.init.zeros_(self.f_gamma.bias)

		init_spatial = torch.logspace(-2, 0.5, steps=n_gammas).reshape(1, -1)
		self.log_gamma_base = nn.Parameter(torch.log(init_spatial))

		# 4. Readout Normalization and Projection
		# self.norm = nn.LayerNorm(ndim_in)
		self.fc_out = nn.Linear(ndim_in, ndim_out)
		self.act_out = nn.PReLU()

	def forward(self, inpt, A_Lg_in_srcs, mask, embed_context, support, num_target_nodes=None):
		"""Args:
			inpt: [N_src, ndim_in] source node features
			A_Lg_in_srcs: edge_index [2, E], x [E, 4]
			mask: [N_src, ndim_mask] or [E, ndim_mask] source support mask
			embed_context: [E, embed_dim] or [1, embed_dim]
		"""
		N = inpt.shape[0]
		# M = num_target_nodes if num_target_nodes is not None else (
		# 	A_Lg_in_srcs.edge_index[1].max().item() + 1
		# 	if A_Lg_in_srcs.edge_index.numel() > 0 else 0
		# )

		ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)
		source_idx = A_Lg_in_srcs.edge_index[0]

		# Step 1: Spatial geometry
		diff_sp = A_Lg_in_srcs.x[:, 0:3]
		norm_pos = torch.linalg.vector_norm(diff_sp, dim=1, keepdim=True)
		unit_dir = diff_sp / norm_pos.clamp(min=1e-6)

		# Step 2: Scale-conditioned RBF bandwidths
		delta = self.f_gamma(ctx)
		alpha = 0.5 * torch.tanh(delta[:, 0:1])
		residuals = 0.2 * torch.tanh(delta[:, 1:])
		gammas = torch.exp(self.log_gamma_base + alpha + residuals)

		# Step 3: Multi-scale spatial RBFs
		r_sp_sq = norm_pos ** 2
		r_aniso = torch.sqrt(gammas * r_sp_sq + 1e-5)
		rbf_decay = torch.exp(-r_aniso)

		rel_pos = torch.cat((unit_dir, rbf_decay), dim=-1)

		# Step 4: Map source mask to edge indexing
		if mask.dim() > 1 and mask.shape[0] == N:
			edge_mask = mask[source_idx]
		else:
			edge_mask = mask

		# Step 5: Direct source -> product-edge mapping
		support_proj = self.f_support(support)
		edge_inpt = torch.cat((inpt[source_idx], rel_pos, support_proj), dim=-1)
		geo_features = self.act_edge(self.film_edge(self.fc_edge(edge_inpt), ctx))

		# Step 6: Phase routing
		phase_routing = self.mask_gate(edge_mask)
		pattern = phase_routing * geo_features

		# Step 7: Normalize pattern only
		# pattern = self.norm(pattern)

		# Step 8: Project to product-space features
		out = self.act_out(self.fc_out(pattern))

		# Step 9: Soft source-support / baseline gate
		absolute_gate = edge_mask.max(1, keepdims=True)[0]
		gate = absolute_gate + self.baseline_gate
		out = gate * out

		return out, edge_mask




class DataAggregationAssociation(nn.Module):
	"""
	Association Phase (Decoder) Block built using standard DataAggregationLayers.
	Replaces DataAggregationAssociationPhase with modular, per-layer gamma learning.
	"""
	def __init__(self, in_channels, out_channels, n_hidden=30, n_dim_latent=30, use_absolute_pos = True,
				 n_dim_mask=4, embed_dim=10, n_embedding = 10, support_dim = 4, use_embedding = True, use_offsets = True):
		super().__init__()

		# Input: Unpooled Features (s) + Encoder Latents (x_latent) + Mask + Source Mask (mask_out_1)
		total_mask_dim = n_dim_mask + 1  # Original observation mask + source likelihood mask
		total_in_dim = in_channels + n_dim_latent + total_mask_dim
		extra_dim = 8 + 2*n_embedding if use_embedding else 8
		extra_dim += 7 if use_absolute_pos else 0
					 
		self.init_trns = nn.Linear(2 * n_hidden, n_hidden)
		self.film_init = FiLM(embed_dim, n_hidden)
		self.act_init = nn.PReLU()

		self.dynamic_proj = nn.Sequential(
    		nn.Linear(total_in_dim, n_hidden),
    		nn.PReLU()
		)
					 
		self.spatial_proj = nn.Sequential(
		    nn.Linear(extra_dim, n_hidden),
		    nn.PReLU(),
		    nn.Linear(n_hidden, n_hidden),
			nn.PReLU()
		)

		# Protects against unbounded log_coverage/log_evidence values
		self.f_support = nn.Sequential(
			nn.Linear(support_dim, 8),
			nn.PReLU(),
			nn.Linear(8, 8) ## Maybe add layer norm
		)
					 
		# Association Layer 1 (Independent per-layer gammas, no expander edges needed)
		self.layer1 = DataAggregationLayer(
			in_channels=n_hidden,
			out_channels=n_hidden,
			n_dim_mask=total_mask_dim,
			embed_dim=embed_dim,
			use_offsets=use_offsets,
			use_expanded=False
		)

		# Association Layer 2 (Independent per-layer gammas, no expander edges needed)
		self.layer2 = DataAggregationLayer(
			in_channels=2 * n_hidden,
			out_channels=out_channels,
			n_dim_mask=total_mask_dim,
			embed_dim=embed_dim,
			use_offsets=use_offsets,
			use_expanded=False
		)

	## Add support, geometric embed (detached), absolute offset
	def forward(self, s, x_latent, mask_out_1, mask, A_in_sta, A_in_src, embed_context, g_embed = None, 
				relative_feat = None, support = None, pos_rel_sta = None, pos_rel_src = None):
		# 1. Combine Masks and Latents
		combined_mask = torch.cat((mask, mask_out_1), dim=-1)

		support_proj = self.f_support(support)
		embed_spatial = self.spatial_proj(torch.cat((g_embed, relative_feat, support_proj), dim = 1))
		x = self.dynamic_proj(torch.cat((s, x_latent, combined_mask), dim=-1))

		# 2. Project and FiLM condition
		x = self.act_init(self.film_init(self.init_trns(torch.cat((x, embed_spatial), dim = 1)), embed_context))

		# 3. Association Graph Convolutions with Per-Layer Gammas
		x = self.layer1(x, combined_mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)
		x = self.layer2(x, combined_mask, A_in_sta, A_in_src, embed_context, pos_rel_sta, pos_rel_src)

		return x


class ArrivalEmbedding(nn.Module):
	def __init__(
		self,
		ndim_arv_in,
		ndim_out,
		n_hidden=20,
		n_dim_embed=30,
		n_phase_embed=5,
		embed_vector_dim=10,
		scale_rel=scale_rel,
		k_spc_edges=k_spc_edges,
		kernel_sig_t=kernel_sig_t,
		use_phase_types=use_phase_types,
		scale_time=scale_time,
		min_thresh=0.01,
		trv=None,
		ftrns2=None,
		debug_asserts=True,
		device = device
	):
		super().__init__()
		self.ftrns2 = ftrns2
		self.trv = trv
		self.debug_asserts = debug_asserts
		self.use_phase_types = use_phase_types
		self.kernel_sig_t = kernel_sig_t
		self.min_thresh = min_thresh
		self.scale_time = scale_time
		self.scale_rel = scale_rel
		self.k_spc_edges = k_spc_edges
		self.dilate_scale = 4.0
		self.scale_misfit = 3.0
		self.scale_s_window = 1.5

		self.null_embed = nn.Parameter(torch.zeros(1, 1, n_hidden))
		self.phase_embed = nn.Embedding(2, n_phase_embed)
		self.log_sig_t_scale = nn.Parameter(torch.zeros(2))
		self.null_out = nn.Parameter(torch.randn(ndim_out))

		self.fc1 = nn.Sequential(
			nn.Linear(ndim_arv_in + 25 + n_phase_embed, 2 * n_hidden),
			nn.PReLU(),
			nn.Linear(2 * n_hidden, n_hidden),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(ndim_arv_in + 8 + n_phase_embed, 2 * n_hidden),
			nn.PReLU(),
			nn.Linear(2 * n_hidden, n_hidden),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(ndim_arv_in + 8 + n_phase_embed, 2 * n_hidden),
			nn.PReLU(),
			nn.Linear(2 * n_hidden, n_hidden),
		)

		self.f_gamma1, self.log_gamma_base1 = self._init_gamma_bank(embed_vector_dim, [0.1, 1.0, 5.0])
		self.f_gamma2, self.log_gamma_base2 = self._init_gamma_bank(embed_vector_dim, [0.1, 1.0, 5.0])
		self.f_gamma3, self.log_gamma_base3 = self._init_gamma_bank(embed_vector_dim, [0.1, 1.0, 5.0, 0.5, 10.0])
		self.f_gamma_t2, self.log_gamma_base_t2 = self._init_gamma_bank(embed_vector_dim, [0.1, 1.0, 5.0])
		self.f_gamma_t3, self.log_gamma_base_t3 = self._init_gamma_bank(embed_vector_dim, [0.1, 1.0, 5.0])

		self.film1 = FiLM(embed_vector_dim, n_hidden)
		self.film2 = FiLM(embed_vector_dim, n_hidden)
		self.film3 = FiLM(embed_vector_dim, n_hidden)

		self.norm1 = RMSNorm(n_hidden)
		self.norm2 = RMSNorm(n_hidden)
		self.norm3 = RMSNorm(n_hidden)

		self.gate = nn.Sequential(nn.Linear(3 * n_hidden, 3 * n_hidden), nn.Sigmoid())
		self.fc_merge = nn.Sequential(
			nn.Linear(3 * n_hidden, 2 * n_hidden),
			nn.PReLU(),
			nn.Linear(2 * n_hidden, ndim_out),
		)

	def _init_gamma_bank(self, embed_dim, init_gammas):
		f_g = nn.Linear(embed_dim, 2 * len(init_gammas))
		nn.init.normal_(f_g.weight, std=0.01)
		nn.init.zeros_(f_g.bias)
		log_base = nn.Parameter(
			torch.log(torch.tensor(init_gammas, dtype=torch.float32).reshape(1, -1))
		)
		return f_g, log_base

	def _get_gammas(self, f_g, log_base, ctx):
		n_g = log_base.shape[1]
		delta = f_g(ctx.reshape(-1, ctx.shape[-1]))
		alpha = 1.1 * torch.tanh(delta[..., :n_g])
		res = 0.2 * torch.tanh(delta[..., n_g:])
		# return torch.exp(log_base.to(ctx.device) + alpha + res).mean(dim=0, keepdim=True)
		return torch.exp(log_base + alpha + res).mean(dim=0, keepdim=True)

	def _process_phase_branch(
		self,
		tpick,
		tlatent,			 # [N_prod, 2] -> columns 0: P, 1: S
		ipick,
		A_src_in_sta,
		x_context_cart,
		x_context_t,
		locs_use_cart,
		x,				   # [N_prod, D_x]
		phase_label,
		ctx,
		deg_srcs,
		cum_deg_srcs,
	):
		device = x.device
		N_picks = len(tpick)
		N_prod = A_src_in_sta.size(1)

		if N_picks == 0 or N_prod == 0:
			out_dim = self.fc2[-1].out_features
			return (
				torch.zeros((N_picks, out_dim), device=device),
				torch.zeros((N_picks, out_dim), device=device),
			)

		# sig_scale = torch.exp(self.log_sig_t_scale)
		# sig_p = (self.scale_misfit * self.kernel_sig_t * sig_scale[0]) ** 2
		# sig_s = (self.scale_misfit * self.kernel_sig_t * sig_scale[1]) ** 2

		# 1. Fully Canonical Station Matching (Shared across both P and S)
		prod_sta = A_src_in_sta[0]
		prod_ctx = A_src_in_sta[1]

		prod_key = prod_sta.to(torch.int64) * (len(x_context_cart) + 1) + prod_ctx.to(torch.int64)
		sta_order = torch.argsort(prod_key, stable=True)
		sorted_sta = prod_sta[sta_order]

		p_left = torch.searchsorted(sorted_sta, ipick, side="left")
		p_right = torch.searchsorted(sorted_sta, ipick, side="right")

		has_match = p_right > p_left
		if not has_match.any():
			out_dim = self.fc2[-1].out_features
			return (
				torch.zeros((N_picks, out_dim), device=device),
				torch.zeros((N_picks, out_dim), device=device),
			)

		valid_picks = torch.where(has_match)[0]
		counts = p_right[valid_picks] - p_left[valid_picks]
		starts = p_left[valid_picks]

		offsets = torch.arange(counts.sum(), device=device) - torch.repeat_interleave(
			torch.cumsum(counts, 0) - counts, counts
		)
		matched_prod_indices = sta_order[starts.repeat_interleave(counts) + offsets]
		matched_pick_indices = valid_picks.repeat_interleave(counts)

		# 2. Canonical Multi-Key Tie Breaking (Run per phase_idx in parallel [2, N_matched])
		# Compute dt for both P (idx 0) and S (idx 1) simultaneously -> Shape: [2, N_matched]
		dt_both = torch.abs(
			tpick[matched_pick_indices].unsqueeze(0) - tlatent[matched_prod_indices, :2].T
		)

		# We do top-match selection per phase
		ind_grab_sorted_list = []
		uniq_picks_sorted_list = []

		for phase_idx in range(2):
			dt = dt_both[phase_idx]
			dt_order = torch.argsort(dt, stable=True)
			dt_rank = torch.zeros_like(dt_order)
			dt_rank[dt_order] = torch.arange(len(dt), device=device)

			key = (
				matched_pick_indices.to(torch.int64) * (len(dt) * N_prod)
				+ dt_rank.to(torch.int64) * N_prod
				+ matched_prod_indices.to(torch.int64)
			)
			order = torch.argsort(key, stable=True)
			m_pick = matched_pick_indices[order]
			m_prod = matched_prod_indices[order]

			mask_first = torch.cat([
				torch.tensor([True], device=device),
				m_pick[1:] != m_pick[:-1],
			])
			ind_grab = m_prod[mask_first]
			uniq_picks = m_pick[mask_first]

			uniq_picks_sorted, sort_order = torch.sort(uniq_picks, stable=True)
			ind_grab_sorted = ind_grab[sort_order]

			ind_grab_sorted_list.append(ind_grab_sorted)
			uniq_picks_sorted_list.append(uniq_picks_sorted)

		# 3. KNN Expansion over context source per pick
		x_ctx_combined = torch.cat(
			(x_context_cart / 1000.0, self.scale_time * x_context_t.reshape(-1, 1)), dim=1
		)

		# Stack both P and S context nearest neighbors -> Shape [2 * N_uniq]
		ctx_nn_p = A_src_in_sta[1, ind_grab_sorted_list[0]]
		ctx_nn_s = A_src_in_sta[1, ind_grab_sorted_list[1]]
		ctx_nn_both = torch.cat([ctx_nn_p, ctx_nn_s], dim=0)

		edge_idx = knn(
			x_ctx_combined,
			x_ctx_combined[ctx_nn_both],
			k=self.k_spc_edges,
		)

		nbr_sources = edge_idx[1]
		
		# Map edge sources back to P or S pick index
		uniq_picks_both = torch.cat([uniq_picks_sorted_list[0], uniq_picks_sorted_list[1]], dim=0)
		phase_owner_both = torch.cat([
			torch.zeros(len(uniq_picks_sorted_list[0]), dtype=torch.long, device=device),
			torch.ones(len(uniq_picks_sorted_list[1]), dtype=torch.long, device=device),
		], dim=0)

		pick_of_nbr = uniq_picks_both[edge_idx[0]]
		phase_of_nbr = phase_owner_both[edge_idx[0]]

		deg_nbr = deg_srcs[nbr_sources]
		start = cum_deg_srcs[nbr_sources]

		inc_inds = torch.arange(deg_nbr.sum(), device=device) - torch.repeat_interleave(
			torch.cumsum(deg_nbr, 0) - deg_nbr, deg_nbr
		)
		nodes_all = start.repeat_interleave(deg_nbr) + inc_inds
		pick_all = pick_of_nbr.repeat_interleave(deg_nbr)
		phase_all = phase_of_nbr.repeat_interleave(deg_nbr)

		station_ok = A_src_in_sta[0, nodes_all] == ipick[pick_all]
		if not station_ok.any():
			out_dim = self.fc2[-1].out_features
			return (
				torch.zeros((N_picks, out_dim), device=device),
				torch.zeros((N_picks, out_dim), device=device),
			)

		nodes = nodes_all[station_ok]
		inds = pick_all[station_ok]
		phases = phase_all[station_ok]

		# 4. Process P (phase=0) and S (phase=1) Edges
		results = []
		fc_layers = [self.fc2, self.fc3]
		film_layers = [self.film2, self.film3]
		f_gammas = [self.f_gamma_t2, self.f_gamma_t3]
		log_bases = [self.log_gamma_base_t2, self.log_gamma_base_t3]

		sig_scale = torch.exp(self.log_sig_t_scale)
		sig_p = (self.scale_misfit * self.kernel_sig_t * sig_scale[0]).clamp(min = 1e-3) # ) ** 2
		sig_s = (self.scale_misfit * self.kernel_sig_t * sig_scale[1] * self.scale_s_window).clamp(min = 1e-3) # ) ** 2
		sig_phases = [sig_p, sig_s]
		# sig_phase = torch.where(p_idx == 0, sig_p, sig_s).unsqueeze(1).clamp(min = 1e-3)

		for p_idx in range(2):
			p_mask = phases == p_idx
			if not p_mask.any():
				out_dim = fc_layers[p_idx][-1].out_features
				results.append(torch.zeros((N_picks, out_dim), device=device))
				continue

			p_nodes = nodes[p_mask]
			p_inds = inds[p_mask]

			# Canonical Sort per phase branch
			canon_order = torch.argsort(
				ipick[p_inds].to(torch.int64) * (N_picks * N_prod)
				+ p_inds.to(torch.int64) * N_prod
				+ p_nodes.to(torch.int64),
				stable=True,
			)
			p_nodes = p_nodes[canon_order]
			p_inds = p_inds[canon_order]

			# Feature Assembly
			diff_t = tpick[p_inds].reshape(-1, 1) - tlatent[p_nodes, p_idx].reshape(-1, 1)
			misfit_rel = torch.cat(
				[
					torch.exp(-1.0 * torch.abs(diff_t) / sig_phases[p_idx]),
					torch.tanh(diff_t / sig_phases[p_idx])
				],
				dim=1,
			)

			# torch.exp(-1.0 * torch.abs(diff_t) / (self.scale_misfit * self.kernel_sig_t)),

			off_sta = (locs_use_cart[ipick[p_inds]] - x_context_cart[A_src_in_sta[1, p_nodes]]) / (10.0 * self.scale_rel)
			norm_s = torch.linalg.vector_norm(off_sta, dim=1, keepdim=True).clamp(min=1e-6)
			gammas = self._get_gammas(f_gammas[p_idx], log_bases[p_idx], ctx)
			feat = torch.cat((off_sta / norm_s, torch.exp(-1.0 * norm_s * gammas[:, :3])), dim=1)

			inpt = torch.cat(
				(
					x[p_nodes],
					misfit_rel,
					feat,
					self.phase_embed(phase_label[p_inds].long()),
				),
				dim=1,
			)

			agg = scatter(
				film_layers[p_idx](fc_layers[p_idx](inpt), ctx.expand(len(inpt), -1)),
				p_inds,
				dim=0,
				dim_size=N_picks,
				reduce="mean",
			)
			results.append(agg)

		return self.norm2(results[0]), self.norm3(results[1])


	def forward(
		self,
		x,
		x_context_cart,
		x_context_t,
		x_query_cart,
		x_query_t,
		A_src_in_sta,
		tpick,
		ipick,
		phase_label,
		locs_use_cart,
		tlatent,
		embed_context,
		trv_out=None,
	):
		# start_time = time.time()
		device = x.device
		tpick = torch.as_tensor(tpick, device=device)

		tpick = tpick.view(-1)
		ipick = ipick.view(-1)
		phase_label = phase_label.view(-1)
		
		N_q = len(x_query_cart)
		N_p = len(tpick)
		N_prod = A_src_in_sta.size(1)

		if trv_out is None:
			trv_out = self.trv(self.ftrns2(locs_use_cart), self.ftrns2(x_query_cart))
		trv_out = trv_out + x_query_t.reshape(-1, 1, 1)

		if not self.use_phase_types:
			phase_label = torch.zeros_like(phase_label)

		# 1. Misfit Filtering (Filter active pairs FIRST)
		# sig_scale = torch.exp(self.log_sig_t_scale)
		# sig_p = self.dilate_scale * self.kernel_sig_t # * sig_scale[0] # ) ** 2
		# sig_s = self.dilate_scale * self.kernel_sig_t # * sig_scale[1] # ) ** 2

		var_p = max(self.dilate_scale * self.kernel_sig_t, 1e-3)**2
		var_s = max(self.dilate_scale * self.kernel_sig_t * self.scale_s_window, 1e-3)**2

		trv_p = trv_out[:, ipick, 0]
		trv_s = trv_out[:, ipick, 1]

		misfit_p = torch.exp(-0.5 * (trv_p - tpick.reshape(1,-1)) ** 2 / var_p )
		misfit_s = torch.exp(-0.5 * (trv_s - tpick.reshape(1,-1)) ** 2 / var_s )
		# misfit_p_ = torch.exp(-0.5 * (trv_p - tpick) ** 2 / sig_p)
		# misfit_s_ = torch.exp(-0.5 * (trv_s - tpick) ** 2 / sig_s)		

		phase_is_p = (phase_label == 0).unsqueeze(0)
		phase_misfit = torch.where(phase_is_p, misfit_p, misfit_s)
		max_misfit = torch.maximum(phase_misfit, torch.maximum(misfit_p, misfit_s))
		mask_misfit_time = max_misfit > self.min_thresh
		isrc, iarv = torch.where(mask_misfit_time)

		# Pre-compute graph degrees
		deg_srcs = scatter(
			torch.ones(A_src_in_sta.size(1), device=device, dtype=torch.long),
			A_src_in_sta[1],
			dim=0,
			dim_size=len(x_context_cart),
			reduce="sum",
		)
		cum_deg_srcs = torch.cat(
			(torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(deg_srcs, dim=0)[:-1]), dim=0
		)

		ctx = embed_context if embed_context.dim() == 2 else embed_context.unsqueeze(0)
		D_h = self.fc1[-1].out_features

		# Direct sparse initialization without full grid expansion
		# Flatten self.null_embed to [D_h] or [1, D_h] before expanding
		arv_flat = self.null_embed.view(-1, D_h).expand(N_q * N_p, D_h).clone()

		if len(iarv) > 0:
			# 2. KNN only over active queries

			## Old sort
			# active_q_unique, active_q_inv = torch.unique(isrc, return_inverse=True)
			
			# Fast unique for sorted 1D tensors on GPU
			# mask_unique = torch.cat([torch.tensor([True], device=device), isrc[1:] != isrc[:-1]])
			first_elem = torch.ones(1, dtype=torch.bool, device=isrc.device)
			mask_unique = torch.cat([first_elem, isrc[1:] != isrc[:-1]])
			active_q_unique = isrc[mask_unique]
			# torch.unique_consecutive(isrc)


			edge_idx = knn(
				torch.cat((x_context_cart / 1000.0, self.scale_time * x_context_t.reshape(-1, 1)), dim=1),
				torch.cat((x_query_cart[active_q_unique] / 1000.0, self.scale_time * x_query_t[active_q_unique].reshape(-1, 1)), dim=1),
				k=self.k_spc_edges,
			)
			
			# Remap edge_idx[0] back to full N_q query indices
			q_knn_indices = active_q_unique[edge_idx[0]]
			ref_context_nodes = edge_idx[1]
			
			# Map query KNN neighbors directly to pair active set
			deg_sl = deg_srcs[ref_context_nodes]
			inc_inds = (
				torch.arange(deg_sl.sum(), device=device, dtype=torch.long)
				- torch.repeat_interleave(torch.cumsum(deg_sl, dim=0) - deg_sl, deg_sl)
			)
			nodes_prod = cum_deg_srcs[ref_context_nodes].repeat_interleave(deg_sl) + inc_inds
			ind_query = q_knn_indices.repeat_interleave(deg_sl)

			# Match active pairs (isrc, iarv) against expanded product nodes
			sta_src_pairs = A_src_in_sta[:, nodes_prod]


			# Composite key: (Query ID * N_stations) + Station ID
			key_prod = ind_query.to(torch.int64) * len(locs_use_cart) + sta_src_pairs[0].to(torch.int64)
			order_kp = torch.argsort(key_prod, stable=True)
			key_prod_sorted = key_prod[order_kp]

			# Query-Pick target composite key
			key_active = isrc.to(torch.int64) * len(locs_use_cart) + ipick[iarv].to(torch.int64)

			p_left = torch.searchsorted(key_prod_sorted, key_active, side="left")
			p_right = torch.searchsorted(key_prod_sorted, key_active, side="right")

			has_match = p_right > p_left
			if has_match.any():
				valid_indices = torch.where(has_match)[0]
				counts = p_right[valid_indices] - p_left[valid_indices]
				starts = p_left[valid_indices]

				offsets = torch.arange(counts.sum(), device=device) - torch.repeat_interleave(
					torch.cumsum(counts, 0) - counts, counts
				)
				matched_prod_indices = order_kp[starts.repeat_interleave(counts) + offsets]
				i_pick_match = valid_indices.repeat_interleave(counts)

				match_mask = ind_query[matched_prod_indices] == isrc[i_pick_match]
				matched_prod_indices = matched_prod_indices[match_mask]
				i_pick_match = i_pick_match[match_mask]

				q_idx = isrc[i_pick_match]
				p_arr_idx = iarv[i_pick_match]
				node_idx = nodes_prod[matched_prod_indices]
				p_idx = phase_label[p_arr_idx].long()

				# Canonical ordering independent of input pick perm
				canon_order = torch.argsort(
					q_idx.to(torch.int64) * (N_p * N_prod)
					+ p_arr_idx.to(torch.int64) * N_prod
					+ node_idx.to(torch.int64),
					stable=True,
				)

				q_idx = q_idx[canon_order]
				p_arr_idx = p_arr_idx[canon_order]
				node_idx = node_idx[canon_order]
				p_idx = p_idx[canon_order]

				tlat = tlatent[node_idx, p_idx].reshape(-1, 1)
				m_rel = tpick[p_arr_idx].reshape(-1, 1) - tlat
				trv_ph = trv_out[q_idx, ipick[p_arr_idx], p_idx].reshape(-1, 1)
				m_query = tpick[p_arr_idx].reshape(-1, 1) - trv_ph

				sig_scale = torch.exp(self.log_sig_t_scale)
				sig_p = self.scale_misfit * self.kernel_sig_t * sig_scale[0] # ) ** 2
				sig_s = self.scale_misfit * self.kernel_sig_t * sig_scale[1] * self.scale_s_window # ) ** 2
				sig_phase = torch.where(p_idx == 0, sig_p, sig_s).unsqueeze(1).clamp(min = 1e-3)

				m_rel_feat = torch.cat(
					[torch.exp(-1.0 * torch.abs(m_rel) / sig_phase),
					 torch.tanh(m_rel / sig_phase)], dim = 1 #  # torch.sign(m_rel)), dim=1
				)
				m_query_feat = torch.cat(
					[torch.exp(-1.0 * torch.abs(m_query) / sig_phase),
					 torch.tanh(m_query / sig_phase)], dim = 1 # # torch.sign(m_query)), dim=1
				)

				# (torch.exp(-1.0 * torch.abs(m_rel) / (self.scale_misfit * self.kernel_sig_t)),
				# (torch.exp(-1.0 * torch.abs(m_rel) / (self.scale_misfit * self.kernel_sig_t)),

				off_src_sta = (locs_use_cart[ipick[p_arr_idx]] - x_query_cart[q_idx]) / (10.0 * self.scale_rel)
				off_ref_sta = (locs_use_cart[ipick[p_arr_idx]] - x_context_cart[A_src_in_sta[1, node_idx]]) / (10.0 * self.scale_rel)
				off_ref_src = (x_query_cart[q_idx] - x_context_cart[A_src_in_sta[1, node_idx]]) / (1.0 * self.scale_rel)
				off_ref_src_t = (
					1000.0 * self.scale_time
					* (x_query_t[q_idx].reshape(-1, 1) - x_context_t[A_src_in_sta[1, node_idx]].reshape(-1, 1))
					/ (3.0 * self.scale_rel)
				)

				n_src_sta = torch.linalg.vector_norm(off_src_sta, dim=1, keepdim=True).clamp(min=1e-6)
				n_ref_sta = torch.linalg.vector_norm(off_ref_sta, dim=1, keepdim=True).clamp(min=1e-6)
				n_ref_src = torch.linalg.vector_norm(off_ref_src, dim=1, keepdim=True).clamp(min=1e-6)

				g1 = self._get_gammas(self.f_gamma1, self.log_gamma_base1, ctx)
				g2 = self._get_gammas(self.f_gamma2, self.log_gamma_base2, ctx)
				g3 = self._get_gammas(self.f_gamma3, self.log_gamma_base3, ctx)

				feat_src_sta = torch.cat((off_src_sta / n_src_sta, torch.exp(-1.0 * n_src_sta * g1[:, :3])), dim=-1)
				feat_ref_sta = torch.cat((off_ref_sta / n_ref_sta, torch.exp(-1.0 * n_ref_sta * g2[:, :3])), dim=-1)
				feat_ref_src = torch.cat((off_ref_src / n_ref_src, torch.exp(-1.0 * n_ref_src * g3[:, :3])), dim=-1)
				feat_time = torch.cat((off_ref_src_t, torch.exp(-1.0 * torch.abs(off_ref_src_t) * g3[:, 3:5])), dim=-1)

				inpt = torch.cat(
					(
						x[node_idx],
						m_rel_feat,
						m_query_feat,
						feat_src_sta,
						feat_ref_sta,
						feat_ref_src,
						feat_time,
						self.phase_embed(p_idx),
					),
					dim=1,
				)
				emb = self.film1(self.fc1(inpt), ctx.expand(len(inpt), -1))
				flat_target = q_idx * N_p + p_arr_idx

				# FIXED
				counts = scatter(torch.ones(len(flat_target), 1, device=device), flat_target, dim=0, dim_size=N_q * N_p, reduce="sum")
				mask_active_scat = (counts > 0).squeeze(-1)

				raw_scatter = scatter(emb, flat_target, dim=0, dim_size=N_q * N_p, reduce="mean")
				if mask_active_scat.any():
					arv_flat[mask_active_scat] = self.norm1(raw_scatter[mask_active_scat])


		arv_embed = arv_flat.view(N_q, N_p, D_h)

		# print('Time [1] : %0.4f'%(time.time() - start_time))
		# start_time = time.time()

		agg_p, agg_s = self._process_phase_branch(
			tpick, tlatent, ipick, A_src_in_sta,
			x_context_cart, x_context_t, locs_use_cart, x, phase_label, ctx,
			deg_srcs, cum_deg_srcs,
		)


		# 1. Active pair index calculation (isrc, iarv derived from mask_misfit_time earlier)
		flat_active = isrc * N_p + iarv

		# 2. Extract features only for active pairs
		arv_active = arv_flat[flat_active]   # [N_active, D_h]
		exp_p_active = agg_p[iarv]		   # [N_active, D_h] (Direct 1D index, skips N_q expansion)
		exp_s_active = agg_s[iarv]		   # [N_active, D_h] (Direct 1D index, skips N_q expansion)

		concat_active = torch.cat((arv_active, exp_p_active, exp_s_active), dim=-1) # [N_active, 3 * D_h]

		# 3. Evaluate FC networks ONLY on active pairs (~30% of total workload)
		gated_active = concat_active * self.gate(concat_active)
		out_active = self.fc_merge(gated_active)

		# 2. Scatter results into prefilled buffer initialized with trainable null_out
		out_flat = self.null_out.expand(N_q * N_p, -1).clone()
		out_flat[flat_active] = out_active
		out = out_flat.view(N_q, N_p, -1)

		# print('Time [3] : %0.4f' % (time.time() - start_time))

		return out, mask_misfit_time


class VerificationSuite(ArrivalEmbedding):
	"""Subclass containing edge-case stress tests, graph sensitivity checks, and autograd validation."""

	@classmethod
	def test_run(cls, device='cpu'):
		print(f"--- Launching Advanced Edge-Case Stress Suite [{device.upper()}] ---")

		# ---------------------------------------------------------
		# 1. Setup Dimensions & Partial Graph Indexing
		# ---------------------------------------------------------
		n_queries = 2		  # Heavy query-to-pick asymmetry (1 : 4250)
		n_picks = 8500		 
		n_context = 50
		n_stations = 120
		n_edges = 1000		 # Low graph density -> unindexed stations

		ndim_arv_in = 16
		ndim_out = 32
		n_hidden = 64
		embed_dim = 16

		model = cls(
			ndim_arv_in=ndim_arv_in, ndim_out=ndim_out, n_hidden=n_hidden, 
			embed_vector_dim=embed_dim, device=device
		).to(device)

		# Physical Coordinates (~100 km geodetic scale)
		x_context_cart = torch.randn(n_context, 3, device=device) * 1e5
		x_context_t = torch.randn(n_context, device=device) * 100.0
		x_query_cart = torch.randn(n_queries, 3, device=device) * 1e5
		x_query_t = torch.tensor([12.0, -840.0], device=device)

		x = torch.randn(n_edges, ndim_arv_in, device=device, requires_grad=True)

		# Unconnected/Isolated Stations (Graph edges strictly link stations 0..59)
		A_src_in_sta = torch.stack([
			torch.randint(0, 60, (n_edges,), device=device, dtype=torch.long),
			torch.randint(0, n_context, (n_edges,), device=device, dtype=torch.long)
		], dim=0)


		sort_order = torch.argsort(A_src_in_sta[1])
		A_src_in_sta = A_src_in_sta[:, sort_order]

		target_sta = 5
		target_phase = 0

		# Guarantee station 5 has at least one edge regardless of random seed
		edges_sta = (A_src_in_sta[0] == target_sta).nonzero(as_tuple=True)[0]
		if len(edges_sta) == 0:
			A_src_in_sta[0, 0] = target_sta
			A_src_in_sta[1, 0] = 0
			edges_sta = torch.tensor([0], device=device)

		# Synthetic Pick Inputs across full station index range (0..119)
		tpick = torch.rand(n_picks, device=device) * 500.0
		ipick = torch.randint(0, n_stations, (n_picks,), device=device, dtype=torch.long)
		phase_label = torch.randint(0, 2, (n_picks,), device=device, dtype=torch.long)

		locs_use_cart = torch.randn(n_stations, 3, device=device) * 1e5
		tlatent = torch.randn(n_edges, 2, device=device) * 50.0
		embed_context = torch.randn(1, embed_dim, device=device)
		trv_out = torch.rand(n_queries, n_stations, 2, device=device) * 100.0

		# ---------------------------------------------------------
		# 2. Inject Deterministic Boundary, Isolated Node, & Collision Cases
		# ---------------------------------------------------------
		thresh = getattr(model, 'min_thresh', 1.0)
		fixed_trv = 10.0

		# Isolate target station travel times across ALL phases for Query 0
		trv_out[0, target_sta, 0] = fixed_trv
		trv_out[0, target_sta, 1] = fixed_trv + 1000.0  # Far offset to prevent phase collision

		# Target expected arrival = origin_time + travel_time
		expected_arrival_q0 = x_query_t[0].item() + fixed_trv

		# Pick 100: Exact match with query 0 travel time -> Must evaluate True
		ipick[100], phase_label[100] = target_sta, target_phase
		tpick[100] = expected_arrival_q0

		# Pick 101: Outside threshold window -> Must evaluate False
		ipick[101], phase_label[101] = target_sta, target_phase
		tpick[101] = expected_arrival_q0 + thresh + 50.0

		# Pick 150: Force to Isolated Station 80 (omitted from A_src_in_sta)
		isolated_sta = 80
		ipick[150], phase_label[150] = isolated_sta, 0
		trv_out[0, isolated_sta, 0] = fixed_trv
		tpick[150] = expected_arrival_q0

		# Test Case B: Identical Duplicate Arrivals for Query 1
		expected_arrival_q1 = x_query_t[1].item() + 88.0
		ipick[200:205] = 12
		phase_label[200:205] = 1
		trv_out[1, 12, 1] = 88.0
		trv_out[1, 12, 0] = 88.0 + 1000.0
		tpick[200:205] = expected_arrival_q1

		# Test Case C: Boundary Out-of-bounds Extreme Timestamps
		tpick[500] = -99999.0
		tpick[501] = 99999.0

		# ---------------------------------------------------------
		# 3. Forward Pass & Strict Unmasked Tensor Audits
		# ---------------------------------------------------------
		model.eval()  # Disable dropout for deterministic evaluation
		with torch.no_grad():
			out, mask = model(
				x, x_context_cart, x_context_t, x_query_cart, x_query_t,
				A_src_in_sta, tpick, ipick, phase_label, locs_use_cart,
				tlatent, embed_context, trv_out=trv_out
			)

		# Output Diagnostic Verification
		res_100 = torch.abs(tpick[100] - (x_query_t[0] + trv_out[0, target_sta, target_phase])).item()
		res_101 = torch.abs(tpick[101] - (x_query_t[0] + trv_out[0, target_sta, target_phase])).item()
		print(f"[Diag] Min Threshold = {thresh}")
		print(f"[Diag] Pick 100 Misfit = {res_100:.2f}s | Mask = {mask[0, 100].item()}")
		print(f"[Diag] Pick 101 Misfit = {res_101:.2f}s | Mask = {mask[0, 101].item()}")

		# Structural Tensor Integrity
		assert out.shape == (n_queries, n_picks, ndim_out), f"Output shape mismatch: {out.shape}"
		assert mask.shape == (n_queries, n_picks), f"Mask shape mismatch: {mask.shape}"
		assert not torch.isnan(out).any(), "Raw unmasked output contains NaNs"
		assert not torch.isinf(out).any(), "Raw unmasked output contains Infs"

		# Boundary Mask Assertions
		assert mask[0, 100].item() == True, f"Boundary test failed: Inside threshold (misfit={res_100:.2f}s) evaluated False"
		assert mask[0, 101].item() == False, f"Boundary test failed: Outside threshold (misfit={res_101:.2f}s) evaluated True"
		assert mask[1, 200:205].all().item() == True, "Duplicate arrival masking failed"

		# Isolated Station Assertion
		assert not torch.isnan(out[:, 150, :]).any(), "Isolated station output produced NaNs"

		print("[✔] Unmasked activation tensors verified clean (No NaNs/Infs).")
		print("[✔] Origin-time relative boundary and isolated station checks passed.")

		# ---------------------------------------------------------
		# 4. Universal Seed-Agnostic Graph Sensitivity Check
		# ---------------------------------------------------------
		model.eval()

		with torch.no_grad():
			out_orig, mask_orig = model(
				x, x_context_cart, x_context_t, x_query_cart, x_query_t,
				A_src_in_sta, tpick, ipick, phase_label, locs_use_cart,
				tlatent, embed_context, trv_out=trv_out
			)

		# Perturb ALL edges connected to target_sta
		x_perturbed = x.clone()
		x_perturbed[edges_sta] += 50.0

		with torch.no_grad():
			out_perturbed, _ = model(
				x_perturbed, x_context_cart, x_context_t, x_query_cart, x_query_t,
				A_src_in_sta, tpick, ipick, phase_label, locs_use_cart,
				tlatent, embed_context, trv_out=trv_out
			)

		# Find picks belonging to target_sta that are actively UNMASKED for Query 0
		target_sta_picks = (ipick == target_sta).nonzero(as_tuple=True)[0]
		unmasked_target_picks = target_sta_picks[mask_orig[0, target_sta_picks]]

		if len(unmasked_target_picks) > 0:
			# Check sensitivity on an active (unmasked) pick embedding
			pick_idx = unmasked_target_picks[0].item()
			diff = torch.max(torch.abs(out_orig[0, pick_idx] - out_perturbed[0, pick_idx])).item()
			
			assert diff > 1e-4, \
				f"Graph Sensitivity Failure: Modifying all {len(edges_sta)} edge features for station {target_sta} " \
				f"had no effect on unmasked pick {pick_idx} embedding (Max Diff: {diff:.6e})!"
		else:
			# Fallback: Check overall graph-level tensor perturbation across Query 0 if all target picks got masked
			diff = torch.max(torch.abs(out_orig[0] - out_perturbed[0])).item()
			assert diff > 1e-4, \
				f"Graph Sensitivity Failure: Modifying edges for station {target_sta} had zero global output effect!"

		print(f"[✔] Graph Sensitivity verified: Modifying {len(edges_sta)} edges for station {target_sta} "
			  f"correctly propagated to embeddings (Max Diff: {diff:.4f}).")

		# ---------------------------------------------------------
		# 5. Backward Pass & Autograd Gradient Flow
		# ---------------------------------------------------------
		model.train()  # Enable training mode for autograd gradient checks
		out_tr, mask_tr = model(
			x, x_context_cart, x_context_t, x_query_cart, x_query_t,
			A_src_in_sta, tpick, ipick, phase_label, locs_use_cart,
			tlatent, embed_context, trv_out=trv_out
		)

		loss = (out_tr * mask_tr.unsqueeze(-1)).sum()
		loss.backward()

		grad_failures = []
		for name, param in model.named_parameters():
			if param.requires_grad:
				if param.grad is None:
					grad_failures.append(f"{name}: No gradient computed")
				elif torch.isnan(param.grad).any():
					grad_failures.append(f"{name}: Gradient contains NaNs")
				elif torch.isinf(param.grad).any():
					grad_failures.append(f"{name}: Gradient contains Infs")

		assert len(grad_failures) == 0, f"Autograd issues found:\n" + "\n".join(grad_failures)
		assert x.grad is not None and not torch.isnan(x.grad).any(), "Input graph features (x) gradient failed"

		print("[✔] Backward pass successful. Non-zero, finite gradients verified across all layers.")
		print("[✔] Complete Verification Suite Finished Successfully.")



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
		self.t_kernel_sq = eps ** 2 # torch.Tensor([eps]).to(device)**2
		# self.t_kernel = 

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
		rel_t_p = torch.cat((torch.exp(-0.5*(rel_t_p**2)/self.t_kernel_sq), torch.tanh(rel_t_p / self.t_kernel_sq)), dim = 1) # phase[edge_index[0]]
		rel_t_s = (atime[edge_index[0][real_edge]] - (tsrc_s[sindex[real_edge], stindex[edge_index[0][real_edge]]] + stime[sindex[real_edge]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s = torch.cat((torch.exp(-0.5*(rel_t_s**2)/self.t_kernel_sq), torch.tanh(rel_t_s / self.t_kernel_sq)), dim = 1) # phase[edge_index[0]]
		rel_t = torch.cat((rel_t_p, rel_t_s, self.embed_phase(phase_j[real_edge].long().reshape(-1))), dim = 1) ## only indexed for not fake source

		rel_t_p1 = (atime[edge_index[1][inot_fake_src]] - (tsrc_p[sindex[inot_fake_src], stindex[edge_index[1][inot_fake_src]]] + stime[sindex[inot_fake_src]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_p1 = torch.cat((torch.exp(-0.5*(rel_t_p1**2)/self.t_kernel_sq), torch.tanh(rel_t_p1 / self.t_kernel_sq)), dim = 1) # phase[edge_index[0]]
		rel_t_s1 = (atime[edge_index[1][inot_fake_src]] - (tsrc_s[sindex[inot_fake_src], stindex[edge_index[1][inot_fake_src]]] + stime[sindex[inot_fake_src]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s1 = torch.cat((torch.exp(-0.5*(rel_t_s1**2)/self.t_kernel_sq), torch.tanh(rel_t_s1 / self.t_kernel_sq)), dim = 1) # phase[edge_index[0]]
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
			rel_t_p2 = torch.cat((torch.exp(-0.5*(rel_t_p2**2)/self.t_kernel_sq), torch.tanh(rel_t_p2 / self.t_kernel_sq)), dim = 1) # phase[edge_index[0]]
			rel_t_s2 = (atime[edge_index[1][real_edge]] - (tsrc_s[sindex[real_edge], stindex[edge_index[1][real_edge]]] + stime[sindex[real_edge]])).reshape(-1,1) # .detach() # correct? (edges[0] point to input data, we access the augemted data time)
			rel_t_s2 = torch.cat((torch.exp(-0.5*(rel_t_s2**2)/self.t_kernel_sq), torch.tanh(rel_t_s2 / self.t_kernel_sq)), dim = 1) # phase[edge_index[0]]
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


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x):
		# Calculate the root mean square along the last dimension
		variance = x.pow(2).mean(-1, keepdim=True)
		# Normalize and scale with the learnable weight parameter
		return x * torch.rsqrt(variance + self.eps) * self.weight


class GCN_Detection_Network_extended(nn.Module):
	def __init__(self, ftrns1, ftrns2, scale_rel = scale_rel, scale_time = scale_time, use_absolute_pos = use_absolute_pos, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded, use_embedding = use_embedding, use_src_pred = False, use_sigmoid = use_sigmoid, attach_time = attach_time, use_absolute_offset = True, trv = None, device = 'cuda'):
		super(GCN_Detection_Network_extended, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
		n_dim_extra_inpt = 0 if attach_time == False else 1
		n_dim_extra_feat = 0 if use_embedding == False else 20
		if use_absolute_offset == True: n_dim_extra_inpt = n_dim_extra_inpt + 7 # concatenate the spatial offsets between source nodes and recievers into input feature
		
		embed_vector_dim = 10 ## Note can add normalization to output
		# self.embed_vector = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, embed_vector_dim))
		# self.embed_vector = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, embed_vector_dim), nn.LayerNorm(embed_vector_dim))
		# Un-bounded, un-normalized scale context projection
		self.embed_vector = nn.Sequential(
			nn.Linear(6, 30),
			nn.SiLU(),  # SiLU (Swish) or GELU extrapolates better than PReLU
			nn.Linear(30, embed_vector_dim)
		)		
		
		# Main Encoder Stack
		self.DataAggregation = DataAggregationExpanded(
			in_channels= 4 + n_dim_extra_inpt + n_dim_extra_feat + embed_vector_dim,
			out_channels=15,
			# n_hidden=n_hidden,
			# embed_dim=embed_dim,
			use_embedding=use_embedding
		)

		## Maybe add expander convolution on SpatialAggregation
		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15).to(device) # ndim_edges = 8 # 30, 15
		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpaceTimeDirect = SpaceTimeDirect(30, 30).to(device) # 15, 30
		self.SpaceTimeAttention = SpaceTimeAttention(30, 30, 8, 15).to(device)
		# self.SpaceTimeAttention = SpaceTimeAttention(30, 30, 4, 15, device = device).to(device)

		if use_expanded == True:
			# self.SpatialAggregation1_expanded = SpatialAggregation(30, 30).to(device) # 15, 30
			self.SpatialAggregation2_expanded = SpatialAggregation(30, 30, zero_offsets = True).to(device) # 15, 30
			self.gate_expanded = nn.Linear(2*30 + embed_vector_dim, 30)
			nn.init.constant_(self.gate_expanded.bias, -2.0)

		self.proj_soln1 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))
		self.proj_soln2 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))

		self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15).to(device)

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
		self.ArrivalEmbedding = ArrivalEmbedding(30, 30, trv = trv, k_spc_edges = k_spc_edges, device = device, ftrns2 = ftrns2) ## [note: merging the embeddings for P and S into one (oveloaded) layer rather than keeping as seperate layers?]
		self.Arrivals = SourceStationAttention(30, 30, 2, 15, n_heads = 3, use_src_pred = use_src_pred, device = device).to(device)
		if use_src_pred == True:
			self.alpha = nn.Parameter(torch.tensor([0.1], device = device))

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

		# ---------------------------------------------------------------------
		# 1. Initialize RBF Gammas for the Product Graph Offset Features
		# ---------------------------------------------------------------------
		if self.use_absolute_offset:
			init_gammas_sp = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32).reshape(1, 3)
			self.log_gamma_base = nn.Parameter(torch.log(init_gammas_sp))  # Shape: [1, 3]
			self.f_gamma = nn.Linear(embed_vector_dim, 1 + 3)
			nn.init.normal_(self.f_gamma.weight, std = 0.01)
			nn.init.zeros_(self.f_gamma.bias)


		self.device = device

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

		use_activation = False
		if use_activation == True:
			# self.activate = lambda x: F.leaky_relu(x, negative_slope=0.01) + 1.0
			self.activate = lambda x: F.leaky_relu(x + 0.1, negative_slope=0.01)

		else:
			self.activate = lambda x: x


		# 1. Initialize SpaceTimeDirect
		nn.init.kaiming_normal_(self.SpaceTimeDirect.f_direct.weight, nonlinearity='leaky_relu')
		if self.SpaceTimeDirect.f_direct.bias is not None:
			nn.init.zeros_(self.SpaceTimeDirect.f_direct.bias)
		
		# 2. Layer 1 (Hidden Projection): Kaiming Normal for PReLU
		for proj in (self.proj_soln1, self.proj_soln2):
			nn.init.kaiming_normal_(proj[0].weight, nonlinearity='leaky_relu')
			if proj[0].bias is not None:
				nn.init.zeros_(proj[0].bias)
		
		# 3. Layer 2 (Final Readout): Small weights + Zero bias
		for proj in (self.proj_soln1, self.proj_soln2):
			nn.init.normal_(proj[2].weight, std=0.01)
			if proj[2].bias is not None:
				nn.init.zeros_(proj[2].bias)


	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q, save_state = False):

		# start_time = time.time()

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

		# print('Rel')
		# print(pos_rel_sta.amin(0))
		# print(pos_rel_sta.amax(0))
		# print(pos_rel_src.amin(0))
		# print(pos_rel_src.amax(0))

		# 2. Append 7D features ONLY if the Geometric Preconditioner (use_embedding) is active
		if self.use_embedding:
			pos_rel_sp = A_src_in_edges.x[:, 0:3]
			# pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp**2, dim=1, keepdim=True)).clamp(min = 1e-6)
			pos_norm_sp = torch.linalg.vector_norm(pos_rel_sp, dim = 1, keepdim = True) # ).clamp(min = 1e-6)

			delta = self.f_gamma(embed_context)
			alpha = 0.5*torch.tanh(delta[:, :1])
			residuals = 0.2 * torch.tanh(delta[:, 1:])
			gammas = torch.exp(self.log_gamma_base[:, :3] + alpha + residuals)
			spatial_decay = torch.exp(-1.0 * pos_norm_sp * gammas)
			
			pos_rel_tm = A_src_in_edges.x[:, 3:4]

			rel_pos_feat = torch.cat((pos_rel_sp / pos_norm_sp.clamp(min = 1e-6), spatial_decay, pos_rel_tm), dim=-1) # 7D
			Slice = torch.cat((Slice, rel_pos_feat), dim=1)

		
		# Runs both Optional Preconditioner (if self.use_embedding=True) AND Main GNN Stack
		x_latent, g_embed = self.DataAggregation(
			tr=Slice, 
			mask=Mask, 
			A_in_sta=A_in_sta, 
			A_in_src=A_in_src, 
			embed_context=embed_context, 
			pos_rel_sta=pos_rel_sta,  # Raw 3D + dt coordinates
			pos_rel_src=pos_rel_src   # Raw 3D + dt coordinates
		)

		x, support = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, embed_context, num_target_nodes = n_temp)
		x = self.SpatialAggregation1(x, embed_context, A_src if self.use_expanded == False else A_src[0], x_temp_cuda, support = support) # x_temp_cuda_cart
		x_local = self.SpatialAggregation2(x, embed_context, A_src if self.use_expanded == False else A_src[0], x_temp_cuda, support = support)
		if self.use_expanded == True:
			x_expand = self.SpatialAggregation2_expanded(x, embed_context, A_src[1], x_temp_cuda, support = support) # x_temp_cuda_cart
			gate = torch.sigmoid(self.gate_expanded(torch.cat((x_local, x_expand, embed_context.expand(x_local.shape[0], -1)), dim = 1)))
			x = x_local + gate*x_expand
		else:
			x = x_local
		x_spatial = self.SpatialAggregation3(x, embed_context, A_src if self.use_expanded == False else A_src[0], x_temp_cuda, support = support) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		
		if self.use_direct_output == True:
			y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		else:
			y_latent = self.SpaceTimeAttention(x_spatial, x_temp_cuda_cart, x_temp_cuda_cart, x_temp_cuda_t, x_temp_cuda_t, embed_context, support) # contains data on spatial and temporal solution at fixed nodes

		y = self.proj_soln1(y_latent)
		
		if save_state == True:
			self.set_internal_state(x_spatial, x_temp_cuda_cart, x_temp_cuda_t, support)
			
		# print('Shapes')
		# print(x_spatial.shape)
		# print(x_query_cart.shape)
		# print(x_temp_cuda_cart.shape)
		# print(t_query.shape)
		# print(x_temp_cuda_t.shape)
		# print(embed_context.shape)
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t, embed_context, support) # second slowest module (could use this embedding to seed source source attention vector).

		x_src = []
		x = self.proj_soln2(x)
		
		# slope_width = 0.1
		# mask_p_thresh = 0.1
		# mask_out = torch.relu(y - mask_p_thresh)
		# Replaces: mask_out = torch.relu(y - mask_p_thresh)
		# Smooth, differentiable soft gate (centered at 0.05, temperature 0.025)
		mask_out = y * torch.sigmoid((y - 0.05) / 0.025)
		support_prod = support[A_src_in_sta[1]]
		
		if mask_out.dim() == 1:
			mask_out = mask_out.unsqueeze(-1)

		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, embed_context, support_prod, num_target_nodes = n_line_nodes) # could we concatenate masks and pass through a single one into next layer
		

		latent_ref = x_latent if self.use_src_pred else x_latent.detach()

		# Run standardized association phase
		s = self.DataAggregationAssociation(
			s=s,
			x_latent=latent_ref,
			mask_out_1=mask_out_1,
			mask=Mask,
			A_in_sta=A_in_sta,
			A_in_src=A_in_src_slice,
			embed_context=embed_context,
			g_embed = g_embed,
			relative_feat = rel_pos_feat,
			support = support_prod,
			pos_rel_sta=pos_rel_sta,  # Direct raw offset reuse
			pos_rel_src=pos_rel_src   # Direct raw offset reuse
		)

		arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, embed_context, trv_out = trv_out_q)

		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return self.activate(y), self.activate(x), self.activate(arv_p), self.activate(arv_s), self.activate(src)

		else:
			
			arv = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)			
			return self.activate(y), self.activate(x), self.activate(arv_p), self.activate(arv_s)

	def set_scale_coefficients(self, scale_rel, scale_time, kernel_sig_t, eps, src_x_kernel, src_t_kernel, time_shift_range):

		self.scale_rel = scale_rel
		self.scale_time = scale_time

		# if self.use_embedding == True:
		# 	self.DataAggregationEmbedding.scale_rel = scale_rel
		# 	self.DataAggregationEmbedding.scale_time = scale_time

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
		self.Arrivals.t_kernel_sq = eps ** 2
		self.embedding_vector = torch.tensor([np.log(scale_rel)/5.0, np.log(scale_time), np.log(kernel_sig_t), np.log(src_x_kernel)/3.0, np.log(src_t_kernel), np.log(time_shift_range)/2.0], device = self.device).reshape(1,-1).float()
		self.embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1)
		
	def set_adjacencies(self, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, pos_loc, pos_src):

		
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

	def set_internal_state(self, x_spatial, x_temp_cuda_cart, x_temp_cuda_t, support): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)
		## Use this to set state for rapid queries of attention layer
		self.x_spatial = x_spatial
		self.x_temp_cuda_cart = x_temp_cuda_cart
		self.x_temp_cuda_t = x_temp_cuda_t
		self.support = support

	def set_internal_state_queries(self, s, x_spatial, x_temp_cuda_cart, x_temp_cuda_t, locs_use_cart, tlatent, support): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)
		## Use this to set state for rapid queries of attention layer
		
		self.s = s
		self.x_spatial = x_spatial
		self.x_temp_cuda_cart = x_temp_cuda_cart
		self.x_temp_cuda_t = x_temp_cuda_t
		self.locs_use_cart = locs_use_cart
		self.tlatent = tlatent
		self.support = support

	def forward_queries(self, x_query_cart, t_query, train = False): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)

		embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		## Use this to obtain query predictions. Note, can modify to also return the spatial embeddings (prior to proj_soln)
		return self.activate(self.proj_soln2(self.SpaceTimeAttention(self.x_spatial, x_query_cart, self.x_temp_cuda_cart, t_query, self.x_temp_cuda_t, embed_context, self.support)))

	def forward_src_queries(self, x_query_src_cart, tq_sample, tpick, ipick, phase_label, trv_out_q): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)

		embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)

		arv_embed, mask_arv = self.ArrivalEmbedding(self.s, self.x_temp_cuda_cart, self.x_temp_cuda_t, x_query_src_cart, tq_sample, self.A_src_in_sta, tpick, ipick, phase_label, self.locs_use_cart, self.tlatent, embed_context, trv_out = trv_out_q)
		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, self.locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return self.activate(arv_p), self.activate(arv_s), self.activate(src)

		else:
			arv = self.Arrivals(tq_sample, trv_out_q, self.locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return self.activate(arv_p), self.activate(arv_s)


	def forward_fixed(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

		# start_time = time.time()

		n_line_nodes = Slice.shape[0]
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		assert(x_temp_cuda_cart.shape[1] == 3)
		
		# embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)		

		A_in_src_slice = self.A_in_src[0] if self.use_expanded else self.A_in_src
		pos_rel_sta, pos_rel_src = None, None


		# 1. Compute relative edge vectors ONLY if offsets are enabled
		if self.use_absolute_offset: # (or self.use_offsets)
			pos_rel_sta = torch.cat((
				(locs_use_cart[self.A_src_in_sta[0][self.A_in_sta[1]]] - locs_use_cart[self.A_src_in_sta[0][self.A_in_sta[0]]]), 
				1000.0 * self.scale_time * (x_temp_cuda_t[self.A_src_in_sta[1][self.A_in_sta[1]]] - x_temp_cuda_t[self.A_src_in_sta[1][self.A_in_sta[0]]]).view(-1, 1)
			), dim=1) / self.scale_rel

			pos_rel_src = torch.cat((
				(x_temp_cuda_cart[self.A_src_in_sta[1][A_in_src_slice[1]]] - x_temp_cuda_cart[self.A_src_in_sta[1][A_in_src_slice[0]]]), 
				1000.0 * self.scale_time * (x_temp_cuda_t[self.A_src_in_sta[1][A_in_src_slice[1]]] - x_temp_cuda_t[self.A_src_in_sta[1][A_in_src_slice[0]]]).view(-1, 1)
			), dim=1) / self.scale_rel


		# 2. Append 7D features ONLY if the Geometric Preconditioner (use_embedding) is active
		if self.use_embedding:
			pos_rel_sp = self.A_src_in_edges.x[:, 0:3]
			# pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp**2, dim=1, keepdim=True)).clamp(min = 1e-6)
			pos_norm_sp = torch.linalg.vector_norm(pos_rel_sp, dim = 1, keepdim = True) # ).clamp(min = 1e-6)

			delta = self.f_gamma(self.embed_context)
			alpha = 0.5*torch.tanh(delta[:, :1])
			residuals = 0.2 * torch.tanh(delta[:, 1:])
			gammas = torch.exp(self.log_gamma_base[:, :3] + alpha + residuals)
			spatial_decay = torch.exp(-1.0 * pos_norm_sp * gammas)
			
			pos_rel_tm = self.A_src_in_edges.x[:, 3:4]

			rel_pos_feat = torch.cat((pos_rel_sp / pos_norm_sp.clamp(min = 1e-6), spatial_decay, pos_rel_tm), dim=-1) # 7D
			Slice = torch.cat((Slice, rel_pos_feat), dim=1)

		
		# Runs both Optional Preconditioner (if self.use_embedding=True) AND Main GNN Stack
		x_latent, g_embed = self.DataAggregation(
			tr=Slice, 
			mask=Mask, 
			A_in_sta=self.A_in_sta, 
			A_in_src=self.A_in_src, 
			embed_context=self.embed_context, 
			pos_rel_sta=pos_rel_sta,  # Raw 3D + dt coordinates
			pos_rel_src=pos_rel_src   # Raw 3D + dt coordinates
		)

		x, support = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, self.embed_context, num_target_nodes = n_temp)
		x = self.SpatialAggregation1(x, self.embed_context, self.A_src, x_temp_cuda, support = support) # x_temp_cuda_cart
		x_local = self.SpatialAggregation2(x, self.embed_context, self.A_src, x_temp_cuda, support = support)
		if self.use_expanded == True:
			x_expand = self.SpatialAggregation2_expanded(x, self.embed_context, self.Ac, x_temp_cuda, support = support) # x_temp_cuda_cart
			gate = torch.sigmoid(self.gate_expanded(torch.cat((x_local, x_expand, self.embed_context.expand(x_local.shape[0], -1)), dim = 1)))
			x = x_local + gate*x_expand
		else:
			x = x_local
		x_spatial = self.SpatialAggregation3(x, self.embed_context, self.A_src, x_temp_cuda, support = support) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		
		if self.use_direct_output == True:
			y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		else:
			y_latent = self.SpaceTimeAttention(x_spatial, x_temp_cuda_cart, x_temp_cuda_cart, x_temp_cuda_t, x_temp_cuda_t, self.embed_context, support) # contains data on spatial and temporal solution at fixed nodes

		y = self.proj_soln1(y_latent)
		
		# if save_state == True:
		# 	self.set_internal_state(x_spatial, x_temp_cuda_cart, x_temp_cuda_t)
			
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t, self.embed_context, support) # second slowest module (could use this embedding to seed source source attention vector).

		x_src = []
		x = self.proj_soln2(x)
		
		# slope_width = 0.1
		# mask_p_thresh = 0.1
		# mask_out = torch.relu(y - mask_p_thresh)
		mask_out = y * torch.sigmoid((y - 0.05) / 0.025)
		support_prod = support[self.A_src_in_sta[1]]
		
		if mask_out.dim() == 1:
			mask_out = mask_out.unsqueeze(-1)

		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, self.embed_context, support_prod, num_target_nodes = n_line_nodes) # could we concatenate masks and pass through a single one into next layer
		

		latent_ref = x_latent if self.use_src_pred else x_latent.detach()

		# Run standardized association phase
		s = self.DataAggregationAssociation(
			s=s,
			x_latent=latent_ref,
			mask_out_1=mask_out_1,
			mask=Mask,
			A_in_sta=self.A_in_sta,
			A_in_src=A_in_src_slice,
			embed_context=self.embed_context,
			g_embed = g_embed,
			relative_feat = rel_pos_feat,
			support = support_prod,
			pos_rel_sta=pos_rel_sta,  # Direct raw offset reuse
			pos_rel_src=pos_rel_src   # Direct raw offset reuse
		)

		arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, self.A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, self.tlatent, self.embed_context, trv_out = trv_out_q)

		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			return self.activate(y), self.activate(x), self.activate(arv_p), self.activate(arv_s), self.activate(src)

		else:
			
			arv = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)			
			return self.activate(y), self.activate(x), self.activate(arv_p), self.activate(arv_s)


	def forward_fixed_source(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, t_query, n_reshape = 1, save_state = False):

		# start_time = time.time()

		n_line_nodes = Slice.shape[0]
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		assert(x_temp_cuda_cart.shape[1] == 3)
		
		# embed_context = self.embed_vector(self.embedding_vector) # .expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)		

		A_in_src_slice = self.A_in_src[0] if self.use_expanded else self.A_in_src
		pos_rel_sta, pos_rel_src = None, None


		# 1. Compute relative edge vectors ONLY if offsets are enabled
		if self.use_absolute_offset: # (or self.use_offsets)
			pos_rel_sta = torch.cat((
				(locs_use_cart[self.A_src_in_sta[0][self.A_in_sta[1]]] - locs_use_cart[self.A_src_in_sta[0][self.A_in_sta[0]]]), 
				1000.0 * self.scale_time * (x_temp_cuda_t[self.A_src_in_sta[1][self.A_in_sta[1]]] - x_temp_cuda_t[self.A_src_in_sta[1][self.A_in_sta[0]]]).view(-1, 1)
			), dim=1) / self.scale_rel

			pos_rel_src = torch.cat((
				(x_temp_cuda_cart[self.A_src_in_sta[1][A_in_src_slice[1]]] - x_temp_cuda_cart[self.A_src_in_sta[1][A_in_src_slice[0]]]), 
				1000.0 * self.scale_time * (x_temp_cuda_t[self.A_src_in_sta[1][A_in_src_slice[1]]] - x_temp_cuda_t[self.A_src_in_sta[1][A_in_src_slice[0]]]).view(-1, 1)
			), dim=1) / self.scale_rel

		# 2. Append 7D features ONLY if the Geometric Preconditioner (use_embedding) is active
		if self.use_embedding:
			pos_rel_sp = self.A_src_in_edges.x[:, 0:3]
			# pos_norm_sp = torch.sqrt(torch.sum(pos_rel_sp**2, dim=1, keepdim=True)).clamp(min = 1e-6)
			pos_norm_sp = torch.linalg.vector_norm(pos_rel_sp, dim = 1, keepdim = True) # ).clamp(min = 1e-6)

			delta = self.f_gamma(self.embed_context)
			alpha = 0.5*torch.tanh(delta[:, :1])
			residuals = 0.2 * torch.tanh(delta[:, 1:])
			gammas = torch.exp(self.log_gamma_base[:, :3] + alpha + residuals)
			spatial_decay = torch.exp(-1.0 * pos_norm_sp * gammas)
			
			pos_rel_tm = self.A_src_in_edges.x[:, 3:4]

			rel_pos_feat = torch.cat((pos_rel_sp / pos_norm_sp.clamp(min = 1e-6), spatial_decay, pos_rel_tm), dim=-1) # 7D
			Slice = torch.cat((Slice, rel_pos_feat), dim=1)

		
		# Runs both Optional Preconditioner (if self.use_embedding=True) AND Main GNN Stack
		x_latent, _ = self.DataAggregation(
			tr=Slice, 
			mask=Mask, 
			A_in_sta=self.A_in_sta, 
			A_in_src=self.A_in_src, 
			embed_context=self.embed_context, 
			pos_rel_sta=pos_rel_sta,  # Raw 3D + dt coordinates
			pos_rel_src=pos_rel_src   # Raw 3D + dt coordinates
		)

		x, support = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, self.embed_context, num_target_nodes = n_temp)
		x = self.SpatialAggregation1(x, self.embed_context, self.A_src, x_temp_cuda, support = support) # x_temp_cuda_cart
		x_local = self.SpatialAggregation2(x, self.embed_context, self.A_src, x_temp_cuda, support = support)
		if self.use_expanded == True:
			x_expand = self.SpatialAggregation2_expanded(x, self.embed_context, self.Ac, x_temp_cuda, support = support) # x_temp_cuda_cart
			gate = torch.sigmoid(self.gate_expanded(torch.cat((x_local, x_expand, self.embed_context.expand(x_local.shape[0], -1)), dim = 1)))
			x = x_local + gate*x_expand
		else:
			x = x_local
		x_spatial = self.SpatialAggregation3(x, self.embed_context, self.A_src, x_temp_cuda, support = support) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		

		if save_state == True:
			self.set_internal_state(x_spatial, x_temp_cuda_cart, x_temp_cuda_t, support)
			
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t, self.embed_context, support) # second slowest module (could use this embedding to seed source source attention vector).

		x_src = []
		x = self.proj_soln2(x)
		

		if n_reshape > 1: ## Use this to map (n_reshape) repeated spatial queries (x_temp_cuda_cart) at different origin times, to predictions for fixed coordinates and across time
			x = x.reshape(-1,n_reshape,1)

		return [], self.activate(x)
		

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

		#	   return self.fc3_block(torch.cat((self.norm_pos(self.ftrns1(src)), src[:,0:2]/self.scale_angles, src[:,[2]]/self.scale_depths), dim = 1))

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


