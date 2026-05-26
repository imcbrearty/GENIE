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


class DataAggregation(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_mask = 4, use_absolute_pos = use_absolute_pos):
		super(DataAggregation, self).__init__('mean') # node dim

		if use_absolute_pos == True:
			in_channels = in_channels + 3*2
		
		## Use two layers of SageConv.
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_mask, n_hidden)

		# self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		# self.l1_t2_1 = nn.Linear(in_channels, n_hidden)
		self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)
		self.activate11 = nn.PReLU() # can extend to each channel
		self.activate12 = nn.PReLU() # can extend to each channel
		self.activate1 = nn.PReLU() # can extend to each channel

		self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)

		self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)
		self.activate21 = nn.PReLU() # can extend to each channel
		self.activate22 = nn.PReLU() # can extend to each channel
		self.activate2 = nn.PReLU() # can extend to each channel

	def forward(self, tr, mask, A_in_sta, A_in_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr)), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr)), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.


class DataAggregationExpanded(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_mask = 4, use_absolute_pos = use_absolute_pos, device = device):
		super(DataAggregationExpanded, self).__init__('mean') # node dim

		if use_absolute_pos == True:
			in_channels = in_channels + 3*2
		
		## Use two layers of SageConv.
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_mask, n_hidden)

		# self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		# self.l1_t2_1 = nn.Linear(in_channels, n_hidden)
		self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)
		self.activate11 = nn.PReLU() # can extend to each channel
		self.activate12 = nn.PReLU() # can extend to each channel
		self.activate1 = nn.PReLU() # can extend to each channel

		self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)

		self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		self.activate21 = nn.PReLU() # can extend to each channel
		self.activate22 = nn.PReLU() # can extend to each channel
		self.activate2 = nn.PReLU() # can extend to each channel

		## Add third layer
		self.l3_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l3_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)

		self.l3_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l3_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)
		self.activate31 = nn.PReLU() # can extend to each channel
		self.activate32 = nn.PReLU() # can extend to each channel
		self.activate3 = nn.PReLU() # can extend to each channel

		## Expanded layers

		self.l1_t1_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l1_t1_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		# self.l1_t1_2c = nn.Linear(2*n_hidden, n_hidden)

		self.l1_t2_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l1_t2_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		self.activate11c = nn.PReLU() # can extend to each channel
		self.activate12c = nn.PReLU() # can extend to each channel
		self.activate1c = nn.PReLU() # can extend to each channel

		self.l2_t1_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)

		self.l2_t2_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		self.activate21c = nn.PReLU() # can extend to each channel
		self.activate22c = nn.PReLU() # can extend to each channel
		self.activate2c = nn.PReLU() # can extend to each channel

		self.alpha_expand1 = nn.Parameter(torch.tensor([0.1], device = device)) # device = device
		self.alpha_expand2 = nn.Parameter(torch.tensor([0.1], device = device)) # device = device



	def forward(self, tr, mask, A_in_sta, A_in_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr)), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate12(tr)), mask), dim = 1))
		tr_local = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l1_t1_2c(torch.cat((tr_local, self.propagate(A_in_sta, x = self.activate11c(self.l1_t1_1c(tr_local))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2c(torch.cat((tr_local, self.propagate(A_in_src[1], x = self.activate12c(self.l1_t2_1c(tr_local))), mask), dim = 1))
		tr_expanded = self.activate1c(torch.cat((tr1, tr2), dim = 1))
		tr = tr_local + self.alpha_expand1*tr_expanded

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr_local = self.activate2(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2c(torch.cat((tr_local, self.propagate(A_in_sta, x = self.activate21c(self.l2_t1_1c(tr_local))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2c(torch.cat((tr_local, self.propagate(A_in_src[1], x = self.activate22c(self.l2_t2_1c(tr_local))), mask), dim = 1))
		tr_expanded = self.activate2c(torch.cat((tr1, tr2), dim = 1))
		tr = tr_local + self.alpha_expand2*tr_expanded

		tr1 = self.l3_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate31(self.l3_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l3_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate32(self.l3_t2_1(tr))), mask), dim = 1))
		tr = self.activate3(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

class DataAggregationEmbedding(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, scale_rel = scale_rel, scale_time = scale_time, ndim_proj = 4):
		super(DataAggregationEmbedding, self).__init__('mean') # node dim
		## Use two layers of SageConv.
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels, n_hidden)

		# self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden, n_hidden)

		# self.l1_t2_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t2_2 = nn.Linear(2*n_hidden, n_hidden)
		self.activate11 = nn.PReLU() # can extend to each channel
		self.activate12 = nn.PReLU() # can extend to each channel
		self.activate1 = nn.PReLU() # can extend to each channel

		self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2 = nn.Linear(3*n_hidden, out_channels)

		self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2 = nn.Linear(3*n_hidden, out_channels)
		self.activate21 = nn.PReLU() # can extend to each channel
		self.activate22 = nn.PReLU() # can extend to each channel
		self.activate2 = nn.PReLU() # can extend to each channel

		self.scale_rel = scale_rel
		self.scale_time = scale_time
		self.merge_edges = nn.Sequential(nn.Linear(n_hidden + ndim_proj, n_hidden), nn.PReLU())

	def forward(self, tr, A_in_sta, A_in_src, A_src_in_sta, pos_loc, pos_src, pos_src_t):

		# tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		# embed_sta_edges = self.fproj_edges_sta(pos_loc/1e6)

		pos_rel_sta = torch.cat(((pos_loc[A_src_in_sta[0][A_in_sta[0]]]/1000.0 - pos_loc[A_src_in_sta[0][A_in_sta[1]]]/1000.0)/(self.scale_rel/1000.0), (pos_src_t[A_src_in_sta[1][A_in_sta[0]]] - pos_src_t[A_src_in_sta[1][A_in_sta[1]]]).reshape(-1,1)/self.scale_time), dim = 1)   # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		pos_rel_src = torch.cat(((pos_src[A_src_in_sta[1][A_in_src[0]]]/1000.0 - pos_src[A_src_in_sta[1][A_in_src[1]]]/1000.0)/(self.scale_rel/1000.0), (pos_src_t[A_src_in_sta[1][A_in_src[0]]] - pos_src_t[A_src_in_sta[1][A_in_src[1]]]).reshape(-1,1)/self.scale_time), dim = 1)  # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)

		## Could add binary edge type information to indicate data type
		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr), edge_attr = pos_rel_sta)), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr), edge_attr = pos_rel_src)), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), edge_attr = pos_rel_sta)), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), edge_attr = pos_rel_src)), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

	def message(self, x_j, edge_attr):

		return self.merge_edges(torch.cat((x_j, edge_attr), dim = 1)) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

# class BipartiteGraphOperator(MessagePassing):
# 	def __init__(self, ndim_in, ndim_out, ndim_edges = 4):
# 		super(BipartiteGraphOperator, self).__init__('add')
# 		# include a single projection map
# 		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
# 		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

# 		self.activate1 = nn.PReLU() # added activation.
# 		self.activate2 = nn.PReLU() # added activation.

# 	def forward(self, inpt, A_src_in_edges, mask, n_sta, n_temp):

# 		N = A_src_in_edges.edge_index[0].max().item() + 1
# 		M = A_src_in_edges.edge_index[1].max().item() + 1

# 		return self.activate2(self.fc2(self.propagate(A_src_in_edges.edge_index, size = (N, M), x = mask.max(1, keepdims = True)[0]*self.activate1(self.fc1(torch.cat((inpt, A_src_in_edges.x), dim = -1))))))

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_edges = 4, ndim_mask = 4):
		super(BipartiteGraphOperator, self).__init__('add')
		# 1. Standard projection for the geometric features
		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
		self.fc2 = nn.Linear(ndim_in, ndim_out) 
		# 2. A tiny 2-layer router that transforms the 4-channel mask into 
		# a 0.0 to 1.0 gating multiplier for EVERY hidden channel in ndim_in
		self.mask_gate = nn.Sequential(
			nn.Linear(ndim_mask, ndim_in),
			nn.Sigmoid()
		)
		self.activate1 = nn.PReLU() 
		self.activate2 = nn.PReLU() 

	def forward(self, inpt, A_src_in_edges, mask, n_sta, n_temp):

		N = A_src_in_edges.edge_index[0].max().item() + 1
		M = A_src_in_edges.edge_index[1].max().item() + 1
		# Step 1: Strict outer existential kill-switch
		absolute_gate = mask.max(1, keepdims = True)[0]

		# Step 2: Compute your standard local geometric/arrival representation
		geo_features = self.activate1(self.fc1(torch.cat((inpt, A_src_in_edges.x), dim = -1)))

		# Step 3: Compute semantic routing gates (returns a 0.0-1.0 vector of size ndim_in)
		phase_routing_vectors = self.mask_gate(mask)

		# Step 4: Multiply them element-wise! (Hard zero for dead pairs, soft zero for wrong phases)
		msg = absolute_gate * (phase_routing_vectors * geo_features)

		return self.activate2(self.fc2(self.propagate(A_src_in_edges.edge_index, size = (N, M), x = msg)))


# class BipartiteGraphOperator(MessagePassing):
# 	def __init__(self, ndim_in, ndim_out, ndim_edges = 4, ndim_mask = 4):
# 		super(BipartiteGraphOperator, self).__init__('add')
		
# 		# 1. Project raw features and edge coordinates into a unified latent geometric space
# 		self.geo_projector = nn.Sequential(
# 			nn.Linear(ndim_in + ndim_edges, ndim_in),
# 			nn.PReLU(),
# 			nn.Linear(ndim_in, ndim_in)
# 		)

# 		# 2. Map the 4-channel phase mask to the exact same hidden feature dimension
# 		self.mask_router = nn.Sequential(
# 			nn.Linear(ndim_mask, ndim_in),
# 			nn.Sigmoid() # Forces values between 0.0 and 1.0 to act as continuous gating switches
# 		)

# 		# 3. Final mixing and output layers
# 		self.fc1 = nn.Linear(ndim_in, ndim_in)
# 		self.fc2 = nn.Linear(ndim_in, ndim_out)

# 		self.activate1 = nn.PReLU()
# 		self.activate2 = nn.PReLU()

# 	def forward(self, inpt, A_src_in_edges, mask, n_sta, n_temp):

# 		N = A_src_in_edges.edge_index[0].max().item() + 1
# 		M = A_src_in_edges.edge_index[1].max().item() + 1

# 		# Step A: Existential hard kill-switch (Absolute protection against non-observations)
# 		existential_gate = mask.max(1, keepdims = True)[0]

# 		# Step B: Compute pure geometric/arrival features
# 		geo_feat = torch.cat((inpt, A_src_in_edges.x), dim = -1)
# 		latent_geo = self.geo_projector(geo_feat)

# 		# Step C: Compute the continuous multi-channel semantic mask routing weights
# 		phase_gates = self.mask_router(mask)

# 		# Step D: Apply semantic phase gating AND the existential hard shutoff
# 		# Element-wise multiplication forces specific latent channels to zero if their phase fails
# 		gated_messages = existential_gate * (phase_gates * latent_geo)

# 		# Step E: Mix the perfectly isolated phase representations and propagate
# 		msg = self.activate1(self.fc1(gated_messages))
		
# 		return self.activate2(self.fc2(self.propagate(A_src_in_edges.edge_index, size = (N, M), x = msg)))


class SpatialAggregation(MessagePassing):
	def __init__(self, in_channels, out_channels, scale_rel = scale_rel, scale_time = scale_time, n_dim = 4, n_global = 5, n_hidden = 30):
		super(SpatialAggregation, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.fc1 = nn.Linear(in_channels + n_dim + n_global, n_hidden)
		self.fc2 = nn.Linear(n_hidden + in_channels, out_channels)
		self.fglobal = nn.Linear(in_channels, n_global)
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.scale_rel = scale_rel

	def forward(self, tr, A_src, pos):

		return self.activate2(self.fc2(torch.cat((tr, self.propagate(A_src, x = tr, pos = pos/self.scale_rel)), dim = -1)))

	def message(self, x_j, pos_i, pos_j):
		
		return self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j, self.activate3(self.fglobal(x_j)).mean(0, keepdims = True).repeat(x_j.shape[0], 1)), dim = -1))) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

class SpaceTimeDirect(nn.Module):
	def __init__(self, inpt_dim, out_channels):
		super(SpaceTimeDirect, self).__init__() #  "Max" aggregation.

		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.activate = nn.PReLU()

	def forward(self, inpts):

		return self.activate(self.f_direct(inpts))

class SpaceTimeAttention(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, n_hidden = 30, n_heads = 5, scale_rel = scale_rel, scale_time = scale_time, device = device):
		super(SpaceTimeAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		# self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.f_queries = nn.Linear(n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_context = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate2 = nn.PReLU()
		self.log_temp = nn.Parameter(torch.Tensor([0.5]))
		self.fixed_degree = torch.Tensor([30.0]).to(device)
		# self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		# self.activate1 = nn.PReLU()
		# self.alpha = nn.Parameter(torch.Tensor([np.log(0.5 / (1 - 0.5))]).to(device)) ## Initilizes as 0.5
		# self.log_temp = nn.Parameter(torch.Tensor([0.5]).to(device))
		
		self.scale_time = scale_time ## 1 Second is 10 km
		self.fixed_edges = None
		self.edge_features = None
		self.use_fixed_edges = False
		# self.activate3 = nn.PReLU()

	def forward(self, inpts, x_query, x_context, x_query_t, x_context_t, k = 30, fixed_type = 0): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		if self.use_fixed_edges == False:

			edge_index = knn(torch.cat((x_context/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = k).flip(0)
			edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x
			# return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).reshape(len(x_query), -1))) # mean over different heads
			return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

		else:

			return self.activate2(self.proj(self.propagate(self.fixed_edges, x = inpts, edge_attr = self.edge_features, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

			# edge_index = self.fixed_edges
			# edge_attr = self.edge_features
			# return self.activate2(self.proj(self.propagate(self.fixed_edges, x = inpts, edge_attr = self.edge_features, size = (x_context.shape[0], x_query.shape[0])).reshape(len(x_query), -1))) # mean over different heads
			# return self.activate2(self.proj(self.propagate(self.fixed_edges[fixed_type], x = inpts, edge_attr = self.edge_features[fixed_type], size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

	def message(self, x_j, index, edge_attr):

		## Why are there no queries in this layer
		query_embed = self.f_queries(edge_attr).view(-1, self.n_heads, self.n_latent)
		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = (query_embed*context_embed).sum(-1)/self.scale ## Removing actication function here
		alpha = alpha / torch.exp(self.log_temp) # / self.fixed_degree.pow(torch.sigmoid(self.alpha)).sqrt()
		alpha = softmax(alpha, index)
		# alpha = self.activate1((query_embed*context_embed).sum(-1)/self.scale)
		## Temperature scale the scores
		# alpha = alpha / self.fixed_degree.pow(torch.sigmoid(self.alpha)).sqrt()
		return alpha.unsqueeze(-1)*value_embed


	def set_edges(self, x_query, x_context, x_query_t, x_context_t, k = 30):
		edge_index = knn(torch.cat((x_context/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = k).flip(0)
		edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x
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
	def __init__(self, ndim_in, ndim_out, ndim_edges = 4):
		super(BipartiteGraphReadOutOperator, self).__init__('add')
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, A_Lg_in_srcs, mask, n_sta, n_temp):

		N = A_Lg_in_srcs.edge_index[0].max().item() + 1
		M = A_Lg_in_srcs.edge_index[1].max().item() + 1

		return self.activate2(self.fc2(self.propagate(A_Lg_in_srcs.edge_index, size = (N, M), x = inpt, edge_attr = A_Lg_in_srcs.x, mask = mask))), mask[A_Lg_in_srcs.edge_index[0]] # note: outputting multiple outputs

	def message(self, x_j, mask_j, edge_attr):

		return mask_j*self.activate1(self.fc1(torch.cat((x_j, edge_attr), dim = -1)))	

class DataAggregationAssociationPhase(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_latent = 30, n_dim_mask = 5, use_absolute_pos = use_absolute_pos):
		super(DataAggregationAssociationPhase, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?

		if use_absolute_pos == True:
			in_channels = in_channels + 2*3
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_latent + n_dim_mask, n_hidden)

		self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		self.l1_t2_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)
		self.activate11 = nn.PReLU() # can extend to each channel
		self.activate12 = nn.PReLU() # can extend to each channel
		self.activate1 = nn.PReLU() # can extend to each channel

		self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)

		self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)
		self.activate21 = nn.PReLU() # can extend to each channel
		self.activate22 = nn.PReLU() # can extend to each channel
		self.activate2 = nn.PReLU() # can extend to each channel

	def forward(self, tr, latent, mask1, mask2, A_in_sta, A_in_src):

		mask = torch.cat((mask1, mask2), dim = - 1)
		tr = torch.cat((tr, latent, mask), dim = -1)
		tr = self.activate(self.init_trns(tr)) # should tlatent appear here too? Not on first go..

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(self.l1_t1_1(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(self.l1_t2_1(tr))), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

class DataAggregationAssociationPhaseExpanded(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_latent = 30, n_dim_mask = 5, use_absolute_pos = use_absolute_pos, device = device):
		super(DataAggregationAssociationPhaseExpanded, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?

		if use_absolute_pos == True:
			in_channels = in_channels + 2*3
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_latent + n_dim_mask, n_hidden)

		self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		self.l1_t2_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)
		self.activate11 = nn.PReLU() # can extend to each channel
		self.activate12 = nn.PReLU() # can extend to each channel
		self.activate1 = nn.PReLU() # can extend to each channel

		self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)

		self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		self.activate21 = nn.PReLU() # can extend to each channel
		self.activate22 = nn.PReLU() # can extend to each channel
		self.activate2 = nn.PReLU() # can extend to each channel

		## Add third layer
		self.l3_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l3_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)

		self.l3_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l3_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)
		self.activate31 = nn.PReLU() # can extend to each channel
		self.activate32 = nn.PReLU() # can extend to each channel
		self.activate3 = nn.PReLU() # can extend to each channel

		## Make expanded layers

		self.l1_t1_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l1_t1_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)

		# self.l1_t1_2c = nn.Linear(2*n_hidden, n_hidden)

		self.l1_t2_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l1_t2_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		self.activate11c = nn.PReLU() # can extend to each channel
		self.activate12c = nn.PReLU() # can extend to each channel
		self.activate1c = nn.PReLU() # can extend to each channel

		self.l2_t1_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)

		self.l2_t2_1c = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2c = nn.Linear(3*n_hidden + n_dim_mask, n_hidden)
		self.activate21c = nn.PReLU() # can extend to each channel
		self.activate22c = nn.PReLU() # can extend to each channel
		self.activate2c = nn.PReLU() # can extend to each channel

		self.alpha_expand1 = nn.Parameter(torch.tensor([0.1], device = device)) # device = device
		self.alpha_expand2 = nn.Parameter(torch.tensor([0.1], device = device)) # device = device


	def forward(self, tr, latent, mask1, mask2, A_in_sta, A_in_src):

		mask = torch.cat((mask1, mask2), dim = - 1)
		tr = torch.cat((tr, latent, mask), dim = -1)
		tr = self.activate(self.init_trns(tr)) # should tlatent appear here too? Not on first go..

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(self.l1_t1_1(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate12(self.l1_t2_1(tr))), mask), dim = 1))
		tr_local = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l1_t1_2c(torch.cat((tr_local, self.propagate(A_in_sta, x = self.activate11c(self.l1_t1_1c(tr_local))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l1_t2_2c(torch.cat((tr_local, self.propagate(A_in_src[1], x = self.activate12c(self.l1_t2_1c(tr_local))), mask), dim = 1))
		tr_expanded = self.activate1c(torch.cat((tr1, tr2), dim = 1))
		tr = tr_local + self.alpha_expand1*tr_expanded

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr_local = self.activate2(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2c(torch.cat((tr_local, self.propagate(A_in_sta, x = self.activate21c(self.l2_t1_1c(tr_local))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2c(torch.cat((tr_local, self.propagate(A_in_src[1], x = self.activate22c(self.l2_t2_1c(tr_local))), mask), dim = 1))
		tr_expanded = self.activate2c(torch.cat((tr1, tr2), dim = 1))
		tr = tr_local + self.alpha_expand2*tr_expanded

		tr1 = self.l3_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate31(self.l3_t1_1(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l3_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate32(self.l3_t2_1(tr))), mask), dim = 1))
		tr = self.activate3(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.


## Note: can maybe reduce dilate scale and scale_misfit, as the default kernel_sig_t is likely larger
## Can also maybe reduce the scaling of eps

class ArrivalEmbedding(MessagePassing):
	def __init__(self, ndim_arv_in, ndim_out, n_hidden = 20, n_dim_embed = 30, n_phase_embed = 5, ndim_out_src = 1, scale_rel = scale_rel, k_spc_edges = k_spc_edges, kernel_sig_t = kernel_sig_t, use_phase_types = use_phase_types, scale_time = scale_time, min_thresh = 0.01, trv = None, ftrns2 = None, device = 'cuda'):
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
		self.fc1 = nn.Sequential(nn.Linear(ndim_arv_in + 12 + 10 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.fc2 = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.fc3 = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features

		# self.fc1_src = nn.Sequential(nn.Linear(ndim_arv_in + 12 + 10 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		# self.fc2_src = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		# self.fc3_src = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features

		self.fc_merge = nn.Sequential(nn.Linear(3*n_hidden, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, ndim_out))
		# self.fc_merge_src = nn.Sequential(nn.Linear(3*n_hidden, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, ndim_out_src))

	def forward(self, x, x_context_cart, x_context_t, x_query_cart, x_query_t, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = None): # reference k nearest spatial points

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
		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		
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

		offset_src_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_query_cart[query_vals[iwhere_query,1]])/(5.0*self.scale_rel)
		offset_ref_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]],:])/(5.0*self.scale_rel)

		## Distances between reference nodes and query (including time offsets)
		offset_ref_src = (x_query_cart[query_vals[iwhere_query,1]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]]])/(1.0*self.scale_rel)
		offset_ref_src_t = (x_query_t[query_vals[iwhere_query,1]].reshape(-1,1) - x_context_t[A_src_in_sta[1,nodes_of_product[iwhere_query]]].reshape(-1,1))/(1.0*self.scale_time)

		offset_src_sta_norm = torch.norm(offset_src_sta, dim = 1, keepdim = True)
		offset_ref_sta_norm = torch.norm(offset_ref_sta, dim = 1, keepdim = True)
		offset_ref_src_norm = torch.norm(offset_ref_src, dim = 1, keepdim = True)

		## Src to ref are not usually large distances so use one kernel radius
		offset_src_sta_norm_kernel = torch.exp(-1.0*torch.abs(offset_src_sta_norm)/(3.0))
		offset_ref_src_norm_kernel = torch.exp(-1.0*torch.abs(offset_ref_src_norm)/(1.0))
		offset_ref_sta_norm_kernel = torch.exp(-1.0*torch.abs(offset_ref_sta_norm)/(3.0))

		offset_ref_src_norm_kernel_t = torch.cat((torch.exp(-1.0*torch.abs(offset_ref_src_t)/(1.0)).reshape(-1,1), torch.sign(offset_ref_src_t).reshape(-1,1)), dim = 1)

		inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, self.phase_embed(phase_label[iarv[inds_queries_to_picks]].reshape(-1).long())), dim = 1)
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
			ioffset = torch.Tensor([-1, 0]).long().to(self.device)
			ind_grab_p = ind_sort_p[ind_extract_p + ioffset[iarg_p]] ## For each pick, the nearest arrival time of the nodes of the product
			ind_grab_s = ind_sort_s[ind_extract_s + ioffset[iarg_s]] ## (Must confirm station indices are identical and mask if not)
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

			offset_ref_sta_p = (locs_use_cart[ipick[inds_queries_to_picks_p]] - x_context_cart[A_src_in_sta[1,nodes_of_product_p[iwhere_query_p]],:])/(5.0*self.scale_rel)
			offset_ref_sta_s = (locs_use_cart[ipick[inds_queries_to_picks_s]] - x_context_cart[A_src_in_sta[1,nodes_of_product_s[iwhere_query_s]],:])/(5.0*self.scale_rel)

			offset_ref_sta_norm_p = torch.norm(offset_ref_sta_p, dim = 1, keepdim = True)
			offset_ref_sta_norm_s = torch.norm(offset_ref_sta_s, dim = 1, keepdim = True)

			offset_ref_sta_norm_kernel_p = torch.exp(-1.0*torch.abs(offset_ref_sta_norm_p)/(3.0))
			offset_ref_sta_norm_kernel_s = torch.exp(-1.0*torch.abs(offset_ref_sta_norm_s)/(3.0))

			# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
			inpt_aggregate_p = torch.cat((x[nodes_of_product_p[iwhere_query_p]], misfit_rel_time_p, offset_ref_sta_p, offset_ref_sta_norm_kernel_p, self.phase_embed(phase_label[inds_queries_to_picks_p].reshape(-1).long())), dim = 1)
			inpt_aggregate_s = torch.cat((x[nodes_of_product_s[iwhere_query_s]], misfit_rel_time_s, offset_ref_sta_s, offset_ref_sta_norm_kernel_s, self.phase_embed(phase_label[inds_queries_to_picks_s].reshape(-1).long())), dim = 1)

			aggregate_product_p = scatter(self.fc2(inpt_aggregate_p), inds_queries_to_picks_p, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider
			aggregate_product_s = scatter(self.fc3(inpt_aggregate_s), inds_queries_to_picks_s, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider


		arv_embed = self.null_embed.clone().expand(len(x_query_cart), len(tpick), -1).clone() # torch.zeros((len(x_query_cart), len(tpick), aggregate_picks.shape[1])).to(device)
		arv_embed[pick_vals[:,1], iarv, :] = aggregate_product
		arv_embed = self.fc_merge((torch.cat((arv_embed, aggregate_product_p.unsqueeze(0).expand(len(x_query_cart), -1, -1), aggregate_product_s.unsqueeze(0).expand(len(x_query_cart), -1, -1)), dim = 2)))

		return arv_embed, mask_misfit_time ## Make sure this is correct reshape (not transposed)


# class SourceArrivalEmbedding(MessagePassing):

# 	def __init__(self, ndim_arv_in, ndim_out, n_hidden = 20, n_dim_embed = 30, n_phase_embed = 5, scale_rel = scale_rel, k_spc_edges = k_spc_edges, kernel_sig_t = kernel_sig_t, use_phase_types = use_phase_types, scale_time = scale_time, min_thresh = 0.01, trv = None, ftrns2 = None, device = 'cuda'):
# 		# super(SourceArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean
# 		super(SourceArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean

# 		## Goal of this module is just to implement Bipartite aggregation of each source query - pick pair, of their misfits,
# 		## and while aggregating over the relevant nodes of the (subgraph) Cartesian product
# 		self.ftrns2 = ftrns2
# 		self.trv = trv
# 		self.use_phase_types = use_phase_types
# 		self.kernel_sig_t = kernel_sig_t
# 		self.min_thresh = min_thresh
# 		self.scale_time = scale_time
# 		self.scale_rel = scale_rel
# 		self.k_spc_edges = k_spc_edges
# 		self.device = device
# 		self.dilate_scale = 2.0 # 3.0
# 		self.scale_misfit = 2.0 # 3.0
# 		# self.null_embed = nn.Parameter(torch.randn(1, 1, n_hidden).to(device) * 0.01) # .to(device)
# 		# self.null_embed = nn.Parameter(torch.zeros(1, 1, n_hidden).to(device)) # .to(device)
# 		self.null_embed = nn.Parameter(torch.zeros(1, 1, n_hidden)) # .to(device)

# 		n_phase_types = 2
# 		n_phase_embed = 5
# 		# self.phase_embed = nn.Parameter(torch.randn(n_phase_types, n_phase_embed) * 0.01).to(device)
# 		self.phase_embed = nn.Embedding(n_phase_types, n_phase_embed)
# 		self.fc1 = nn.Sequential(nn.Linear(ndim_arv_in + 12 + 10 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
# 		self.fc2 = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
# 		self.fc3 = nn.Sequential(nn.Linear(ndim_arv_in + 6 + n_phase_embed, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features


# 		self.fc_merge = nn.Sequential(nn.Linear(3*n_hidden, 2*n_hidden), nn.PReLU(), nn.Linear(2*n_hidden, ndim_out))

# 	def forward(self, x, x_context_cart, x_context_t, x_query_cart, x_query_t, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = None): # reference k nearest spatial points

# 		if trv_out is None:
# 			trv_out = self.trv(self.ftrns2(locs_use_cart), self.ftrns2(x_query_cart)) + x_query_t.reshape(-1, 1, 1) ## Use full travel times, as we check for stations from the full product
# 		else: 
# 			trv_out = trv_out + x_query_t.reshape(-1, 1, 1) ## Is this being applied outside this layer?

# 		if self.use_phase_types == False:
# 			phase_label = phase_label*0.0

# 		# ipick_unique = torch.unique(ipick).long()
# 		i1 = torch.where(phase_label == 0)[0]
# 		i2 = torch.where(phase_label == 1)[0]

# 		## Note: computing misfit times but not even using them other than for mask
# 		misfit_time = torch.zeros((len(x_query_cart), len(tpick), 4)).to(self.device) ## Question: is it necessary to produce these pairwise misfits? Can we focus on the pairs that "likely" have arrival times within threshold (e.g., bound min and max times based on distances between src reciever first, before computing travel times)
# 		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
# 		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
# 		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
# 		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		
# 		## Can compute these degree vectors outside of loop
# 		degree_srcs = degree(A_src_in_sta[1], num_nodes = len(x_context_cart), dtype = torch.long)
# 		cum_degree_srcs = torch.cat((torch.zeros(1).to(self.device), torch.cumsum(degree_srcs, dim = 0)[0:-1]), dim = 0).long()
# 		## Should check if minimal degree srcs really are accessing nearest stations
# 		mask_misfit_time = misfit_time.max(2).values > self.min_thresh ## Save this, so can use as mask in the attention layer
# 		isrc, iarv = torch.where(mask_misfit_time == 1)

# 		## Build src-src indices (may or may not use the edge feature of source query to source node offsets)
# 		edge_index = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query_cart/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = self.k_spc_edges).flip(0).contiguous()

# 		# Build a single flattened arange from size = sum(idx)
# 		deg_slice = degree_srcs[edge_index[0]]
# 		assert(deg_slice.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
# 		inc_inds = torch.arange(deg_slice.sum()).long().to(self.device)
# 		inc_inds = inc_inds - torch.repeat_interleave(torch.cumsum(deg_slice, dim = 0) - deg_slice, deg_slice)
# 		nodes_of_product = cum_degree_srcs[edge_index[0]].repeat_interleave(degree_srcs[edge_index[0]]) + inc_inds
# 		ind_query = torch.arange(len(x_query_cart)).long().to(self.device).repeat_interleave(scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)
# 		sta_src_pairs = A_src_in_sta[:, nodes_of_product]
# 		## Query_vals is shaped based on nodes_of_product. So when we aggregate or want to extract Cartesian product node features, we can use these.

# 		# k_matches = knn(sta_src_pairs.T, torch.cat((ipick[iarv].reshape(-1,1), ))
# 		query_vals = torch.cat((sta_src_pairs[0].reshape(-1,1), ind_query.reshape(-1,1)), dim = 1).long() # .float()
# 		pick_vals = torch.cat((ipick[iarv].reshape(-1,1), isrc.reshape(-1,1)), dim = 1).long() # .float()

# 		## Note: query_vals represents the pairs of station and query inds
# 		## pick_vals represents the pairs of station and query inds
# 		hash_picks, hash_queries = hash_rows(pick_vals), hash_rows(query_vals) ## Do not define directly if only using one mask below
# 		mask_picks = torch.isin(hash_picks, hash_queries) # set(map(tuple, l1))
# 		mask_queries = torch.isin(hash_queries, hash_picks) # set(map(tuple, l1))
# 		iwhere_picks = torch.where(mask_picks == 1)[0] ## Not used
# 		iwhere_query = torch.where(mask_queries == 1)[0]
# 		# assert(torch.abs(query_vals[iwhere_query] - pick_vals[knn(pick_vals, query_vals[iwhere_query], k = 1)[1]]).max() == 0)
# 		# assert(torch.abs(pick_vals[iwhere_picks] - query_vals[knn(query_vals, pick_vals[iwhere_picks], k = 1)[1]]).max() == 0)
# 		## The point of query vals is these are the nodes on the Cartesian product we are accessing and aggregating across.
# 		## How can we "read into" these nodes, or match to these nodes, for all possible (> min thresh) pick vals.
# 		## Can we use degrees or cumulative degrees of query vals to directly read in? Can we catch cases where the pick
# 		## has no match (e.g., read in, but then find mis-match of values and remove?)
# 		# print('Time %0.4f'%(time.time() - st))

# 		# print('Time %0.4f'%(time.time() - st))
# 		sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
# 		ind_extract = torch.searchsorted(sorted_hash_picks, hash_queries[iwhere_query])
# 		valid_ind = (ind_extract < len(sorted_hash_picks)) & (sorted_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)] == hash_queries[iwhere_query])
# 		inds_queries_to_picks = order_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)][valid_ind]
# 		assert(valid_ind.sum() == len(valid_ind))

# 		# use_checks = True
# 		# if use_checks == True:
# 		# 	## For a random set of queries, check if have correct edges
# 		# 	n_check = 30
# 		# 	for n in range(n_check):
# 		# 		i0 = np.random.choice(len(x_query))
# 		# 		e1 = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query_cart/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1)[i0,:].reshape(1,-1), k = self.k_spc_edges).flip(0).contiguous()

# 		## Compute features
# 		misfit_rel_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - tlatent[nodes_of_product[iwhere_query]]
# 		misfit_query_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - trv_out[query_vals[iwhere_query,1], ipick[iarv[inds_queries_to_picks]], :]
# 		# misfit_rel_time = torch.cat((torch.exp(-0.5*(misfit_rel_time**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_rel_time)), dim = 1)
# 		# misfit_query_time = torch.cat((torch.exp(-0.5*(misfit_query_time**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_query_time)), dim = 1)

# 		misfit_rel_time = torch.cat((torch.exp(-1.0*torch.abs(misfit_rel_time)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_rel_time)), dim = 1)
# 		misfit_query_time = torch.cat((torch.exp(-1.0*torch.abs(misfit_query_time)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_query_time)), dim = 1)

# 		offset_src_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_query_cart[query_vals[iwhere_query,1]])/(5.0*self.scale_rel)
# 		offset_ref_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]],:])/(5.0*self.scale_rel)

# 		## Distances between reference nodes and query (including time offsets)
# 		offset_ref_src = (x_query_cart[query_vals[iwhere_query,1]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]]])/(1.0*self.scale_rel)
# 		offset_ref_src_t = (x_query_t[query_vals[iwhere_query,1]].reshape(-1,1) - x_context_t[A_src_in_sta[1,nodes_of_product[iwhere_query]]].reshape(-1,1))/(1.0*self.scale_time)

# 		offset_src_sta_norm = torch.norm(offset_src_sta, dim = 1, keepdim = True)
# 		offset_ref_sta_norm = torch.norm(offset_ref_sta, dim = 1, keepdim = True)
# 		offset_ref_src_norm = torch.norm(offset_ref_src, dim = 1, keepdim = True)

# 		## Src to ref are not usually large distances so use one kernel radius
# 		offset_src_sta_norm_kernel = torch.exp(-1.0*torch.abs(offset_src_sta_norm)/(3.0))
# 		offset_ref_src_norm_kernel = torch.exp(-1.0*torch.abs(offset_ref_src_norm)/(1.0))
# 		offset_ref_sta_norm_kernel = torch.exp(-1.0*torch.abs(offset_ref_sta_norm)/(3.0))

# 		offset_ref_src_norm_kernel_t = torch.cat((torch.exp(-1.0*torch.abs(offset_ref_src_t)/(1.0)).reshape(-1,1), torch.sign(offset_ref_src_t).reshape(-1,1)), dim = 1)

# 		inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, self.phase_embed(phase_label[iarv[inds_queries_to_picks]].reshape(-1).long())), dim = 1)
# 		aggregate_product = scatter(self.fc1(inpt_aggregate), inds_queries_to_picks, dim = 0, dim_size = len(iarv), reduce = 'mean') ## Can consider
# 		aggregate_source = scatter(self.fc1_src(inpt_aggregate), inds_queries_to_srcs, dim = 0, dim_size = len(iarv), reduce = 'mean')

# 		# print('T2 %0.4f'%(time.time() - t1))
# 		# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], x_embed_trns[query_vals[iwhere_query,1]], misfit_rel_time, misfit_query_time, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
# 		# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
# 		## Note: could first transfrom the features: misfit_rel_time, misfit_query_time, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel seperately from embed
# 		## For increased stability of merging with the embeddings
		
# 		use_time_based_embedding = True
# 		if use_time_based_embedding == True:

# 			min_time_shift = tlatent.amin()
# 			max_time_offset = (tlatent.amax() - min_time_shift)*2.5
# 			query_time = ((tpick - min_time_shift) + max_time_offset*ipick).reshape(-1,1)
# 			val_sort_p, ind_sort_p = torch.sort((tlatent[:,0] - min_time_shift) + max_time_offset*A_src_in_sta[0]) ## Could do these steps outside the training loop
# 			val_sort_s, ind_sort_s = torch.sort((tlatent[:,1] - min_time_shift) + max_time_offset*A_src_in_sta[0])
# 			ind_extract_p = torch.searchsorted(val_sort_p, (tpick - min_time_shift) + max_time_offset*ipick)
# 			ind_extract_s = torch.searchsorted(val_sort_s, (tpick - min_time_shift) + max_time_offset*ipick)

# 			iarg_p = torch.argmin(torch.abs(torch.cat((val_sort_p[torch.clamp(ind_extract_p - 1, min = 0)].reshape(-1,1), val_sort_p[torch.clamp(ind_extract_p, max = len(val_sort_p) - 1)].reshape(-1,1)), dim = 1) - query_time), dim = 1)
# 			iarg_s = torch.argmin(torch.abs(torch.cat((val_sort_s[torch.clamp(ind_extract_s - 1, min = 0)].reshape(-1,1), val_sort_s[torch.clamp(ind_extract_s, max = len(val_sort_s) - 1)].reshape(-1,1)), dim = 1) - query_time), dim = 1)
# 			ioffset = torch.Tensor([-1, 0]).long().to(self.device)
# 			ind_grab_p = ind_sort_p[ind_extract_p + ioffset[iarg_p]] ## For each pick, the nearest arrival time of the nodes of the product
# 			ind_grab_s = ind_sort_s[ind_extract_s + ioffset[iarg_s]] ## (Must confirm station indices are identical and mask if not)
# 			sta_match_p = (A_src_in_sta[0,ind_grab_p] == ipick)
# 			sta_match_s = (A_src_in_sta[0,ind_grab_s] == ipick)

# 			# print('T4 %0.4f'%(time.time() - t1))
# 			edge_index_p = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1)[A_src_in_sta[1, ind_grab_p]], k = self.k_spc_edges).flip(0).contiguous()
# 			edge_index_s = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1)[A_src_in_sta[1, ind_grab_s]], k = self.k_spc_edges).flip(0).contiguous()

# 			# Build a single flattened arange from size = sum(idx)
# 			deg_slice_p = degree_srcs[edge_index_p[0]]
# 			deg_slice_s = degree_srcs[edge_index_s[0]]
# 			assert(deg_slice_p.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
# 			assert(deg_slice_s.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
# 			inc_inds_p = torch.arange(deg_slice_p.sum()).long().to(self.device)
# 			inc_inds_p = inc_inds_p - torch.repeat_interleave(torch.cumsum(deg_slice_p, dim = 0) - deg_slice_p, deg_slice_p)

# 			inc_inds_s = torch.arange(deg_slice_s.sum()).long().to(self.device)
# 			inc_inds_s = inc_inds_s - torch.repeat_interleave(torch.cumsum(deg_slice_s, dim = 0) - deg_slice_s, deg_slice_s)

# 			nodes_of_product_p = cum_degree_srcs[edge_index_p[0]].repeat_interleave(degree_srcs[edge_index_p[0]]) + inc_inds_p
# 			nodes_of_product_s = cum_degree_srcs[edge_index_s[0]].repeat_interleave(degree_srcs[edge_index_s[0]]) + inc_inds_s

# 			ind_query_p = torch.arange(len(tpick)).long().to(self.device).repeat_interleave(scatter(deg_slice_p, edge_index_p[1], dim = 0, dim_size = len(tpick), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)
# 			ind_query_s = torch.arange(len(tpick)).long().to(self.device).repeat_interleave(scatter(deg_slice_s, edge_index_s[1], dim = 0, dim_size = len(tpick), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)

# 			sta_src_pairs_p = A_src_in_sta[:, nodes_of_product_p]
# 			sta_src_pairs_s = A_src_in_sta[:, nodes_of_product_s]

# 			## Note: do we use all the pick_vals or just the pick_vals with positive entries, like above. We have actually created these queries based on "all" the picks
# 			# k_matches = knn(sta_src_pairs.T, torch.cat((ipick[iarv].reshape(-1,1), ))
# 			query_vals_p = torch.cat((sta_src_pairs_p[0].reshape(-1,1), ind_query_p.reshape(-1,1)), dim = 1).long() # .float()
# 			query_vals_s = torch.cat((sta_src_pairs_s[0].reshape(-1,1), ind_query_s.reshape(-1,1)), dim = 1).long() # .float()

# 			pick_vals_time = torch.cat((ipick.reshape(-1,1), torch.arange(len(ipick)).reshape(-1,1).to(self.device)), dim = 1).long() # .float()
# 			hash_picks_time = hash_rows(pick_vals_time)
# 			hash_queries_p, hash_queries_s = hash_rows(query_vals_p), hash_rows(query_vals_s)
# 			mask_queries_p = torch.isin(hash_queries_p, hash_picks_time) # set(map(tuple, l1))
# 			mask_queries_s = torch.isin(hash_queries_s, hash_picks_time) # set(map(tuple, l1))
# 			iwhere_query_p = torch.where(mask_queries_p == 1)[0]
# 			iwhere_query_s = torch.where(mask_queries_s == 1)[0]
# 			# print('T5 %0.4f'%(time.time() - t1))
# 			# assert(torch.abs(ipick[ind_query_p] - A_src_in_sta[0, nodes_of_product_p]).max() == 0)
# 			# assert(torch.abs(ipick[ind_query_s] - A_src_in_sta[0, nodes_of_product_s]).max() == 0)
# 			## Now for each pick and subset of nodes of product need to find matched station
			
# 			# print('Time %0.4f'%(time.time() - st))
# 			sorted_hash_picks_time, order_hash_picks_time = torch.sort(hash_picks_time)
# 			ind_extract_p = torch.searchsorted(sorted_hash_picks_time, hash_queries_p[iwhere_query_p])
# 			ind_extract_s = torch.searchsorted(sorted_hash_picks_time, hash_queries_s[iwhere_query_s])

# 			valid_ind_p = (ind_extract_p < len(sorted_hash_picks_time)) & (sorted_hash_picks_time[ind_extract_p.clamp(max = len(sorted_hash_picks_time) - 1)] == hash_queries_p[iwhere_query_p])
# 			valid_ind_s = (ind_extract_s < len(sorted_hash_picks_time)) & (sorted_hash_picks_time[ind_extract_s.clamp(max = len(sorted_hash_picks_time) - 1)] == hash_queries_s[iwhere_query_s])

# 			inds_queries_to_picks_p = order_hash_picks_time[ind_extract_p.clamp(max = len(sorted_hash_picks_time) - 1)][valid_ind_p]
# 			inds_queries_to_picks_s = order_hash_picks_time[ind_extract_s.clamp(max = len(sorted_hash_picks_time) - 1)][valid_ind_s]
# 			# assert(valid_ind_p.sum() == len(valid_ind_p))
# 			# assert(valid_ind_s.sum() == len(valid_ind_s))
# 			# assert(torch.abs(pick_vals_time[inds_queries_to_picks_p,0] - query_vals_p[iwhere_query_p,0]).amax() == 0)
# 			# assert(torch.abs(pick_vals_time[inds_queries_to_picks_s,0] - query_vals_s[iwhere_query_s,0]).amax() == 0)

# 			misfit_rel_time_p = tpick[inds_queries_to_picks_p].reshape(-1,1) - tlatent[nodes_of_product_p[iwhere_query_p],0].reshape(-1,1)
# 			misfit_rel_time_s = tpick[inds_queries_to_picks_s].reshape(-1,1) - tlatent[nodes_of_product_s[iwhere_query_s],1].reshape(-1,1)
# 			# assert(degree(inds_queries_to_picks_p).amax() <= self.k_spc_edges)
# 			# assert(degree(inds_queries_to_picks_s).amax() <= self.k_spc_edges)

# 			misfit_rel_time_p = torch.cat((torch.exp(-1.0*torch.abs(misfit_rel_time_p)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_rel_time_p)), dim = 1)
# 			misfit_rel_time_s = torch.cat((torch.exp(-1.0*torch.abs(misfit_rel_time_s)/(((self.scale_misfit*self.kernel_sig_t)**1))), torch.sign(misfit_rel_time_s)), dim = 1)

# 			offset_ref_sta_p = (locs_use_cart[ipick[inds_queries_to_picks_p]] - x_context_cart[A_src_in_sta[1,nodes_of_product_p[iwhere_query_p]],:])/(5.0*self.scale_rel)
# 			offset_ref_sta_s = (locs_use_cart[ipick[inds_queries_to_picks_s]] - x_context_cart[A_src_in_sta[1,nodes_of_product_s[iwhere_query_s]],:])/(5.0*self.scale_rel)

# 			offset_ref_sta_norm_p = torch.norm(offset_ref_sta_p, dim = 1, keepdim = True)
# 			offset_ref_sta_norm_s = torch.norm(offset_ref_sta_s, dim = 1, keepdim = True)

# 			offset_ref_sta_norm_kernel_p = torch.exp(-1.0*torch.abs(offset_ref_sta_norm_p)/(3.0))
# 			offset_ref_sta_norm_kernel_s = torch.exp(-1.0*torch.abs(offset_ref_sta_norm_s)/(3.0))

# 			# inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_ref_src, offset_src_sta_norm_kernel, offset_ref_src_norm_kernel, offset_ref_sta_norm_kernel, offset_ref_src_norm_kernel_t, phase_label[iarv[inds_queries_to_picks]].reshape(-1,1)), dim = 1)
# 			inpt_aggregate_p = torch.cat((x[nodes_of_product_p[iwhere_query_p]], misfit_rel_time_p, offset_ref_sta_p, offset_ref_sta_norm_kernel_p, self.phase_embed(phase_label[inds_queries_to_picks_p].reshape(-1).long())), dim = 1)
# 			inpt_aggregate_s = torch.cat((x[nodes_of_product_s[iwhere_query_s]], misfit_rel_time_s, offset_ref_sta_s, offset_ref_sta_norm_kernel_s, self.phase_embed(phase_label[inds_queries_to_picks_s].reshape(-1).long())), dim = 1)

# 			aggregate_product_p = scatter(self.fc2(inpt_aggregate_p), inds_queries_to_picks_p, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider
# 			aggregate_product_s = scatter(self.fc3(inpt_aggregate_s), inds_queries_to_picks_s, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider

# 			aggregate_src_p = scatter(self.fc2_src(inpt_aggregate_p), inds_queries_to_srcs_p, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider
# 			aggregate_src_s = scatter(self.fc3_src(inpt_aggregate_s), inds_queries_to_srcs_s, dim = 0, dim_size = len(tpick), reduce = 'mean') ## Can consider


# 		arv_embed = self.null_embed.clone().expand(len(x_query_cart), len(tpick), -1).clone() # torch.zeros((len(x_query_cart), len(tpick), aggregate_picks.shape[1])).to(device)
# 		arv_embed[pick_vals[:,1], iarv, :] = aggregate_product
# 		arv_embed = self.fc_merge((torch.cat((arv_embed, aggregate_product_p.unsqueeze(0).expand(len(x_query_cart), -1, -1), aggregate_product_s.unsqueeze(0).expand(len(x_query_cart), -1, -1)), dim = 2)))
# 		src_embed = self.fc_merge_src(torch.cat((aggregate_source, aggregate_src_p, aggregate_src_s), dim = 1))

# 		return arv_embed, src_embed, mask_misfit_time ## Make sure this is correct reshape (not transposed)



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
			


class GCN_Detection_Network_extended(nn.Module):
	def __init__(self, ftrns1, ftrns2, scale_rel = scale_rel, scale_time = scale_time, use_absolute_pos = use_absolute_pos, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded, use_embedding = use_embedding, use_src_pred = True, use_sigmoid = use_sigmoid, attach_time = attach_time, trv = None, device = 'cuda'):
		super(GCN_Detection_Network_extended, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
		n_dim_extra_inpt = 0 if attach_time == False else 1
		n_dim_extra_feat = 0 if use_embedding == False else 20

		
		embed_vector_dim = 10 ## Note can add normalization to output
		self.embed_vector = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, embed_vector_dim))

		if use_expanded == False:
			self.DataAggregation = DataAggregation(4 + n_dim_extra_inpt + n_dim_extra_feat + embed_vector_dim, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
		else:
			self.DataAggregation = DataAggregationExpanded(4 + n_dim_extra_inpt + n_dim_extra_feat + embed_vector_dim, 15, device = device).to(device) # output size is latent size for (half of) bipartite code # , 15				

		## Maybe add expander convolution on SpatialAggregation
		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 4).to(device) # 30, 15
		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpaceTimeDirect = SpaceTimeDirect(30, 30).to(device) # 15, 30
		self.SpaceTimeAttention = SpaceTimeAttention(30, 30, 4, 15, device = device).to(device)

		if use_expanded == True:
			self.SpatialAggregation1_expanded = SpatialAggregation(30, 30).to(device) # 15, 30
			self.SpatialAggregation2_expanded = SpatialAggregation(30, 30).to(device) # 15, 30
			self.alpha_expand1 = nn.Parameter(torch.tensor([0.1], device = device))
			self.alpha_expand2 = nn.Parameter(torch.tensor([0.1], device = device))

		if use_sigmoid == False:
			self.proj_soln1 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))
			self.proj_soln2 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))
		else:
			self.proj_soln1 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 2))
			self.proj_soln2 = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 2))

		self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15).to(device)

		## For now, don't use expanded on the downstream DataAggregationAssociationPhase (may be slightly unnecessary)
		# if use_expanded == False:
		# 	self.DataAggregationAssociationPhase = DataAggregationAssociationPhase(15, 15).to(device) # need to add concatenation
		# else:
		# 	self.DataAggregationAssociationPhase = DataAggregationAssociationPhaseExpanded(15, 15, device = device).to(device) # need to add concatenation

		## For now, don't use expanded on the downstream DataAggregationAssociationPhase (may be slightly unnecessary)
		# if use_expanded == False:
		self.DataAggregationAssociationPhase = DataAggregationAssociationPhase(15, 15).to(device) # need to add concatenation
		# else:
		# self.DataAggregationAssociationPhase = DataAggregationAssociationPhaseExpanded(15, 15, device = device).to(device) # need to add concatenation

		## Make association module layers (note, previous arrival embeddings used to be smaller)
		self.ArrivalEmbedding = ArrivalEmbedding(30, 30, trv = trv, device = device, ftrns2 = ftrns2) ## [note: merging the embeddings for P and S into one (oveloaded) layer rather than keeping as seperate layers?]
		self.Arrivals = SourceStationAttention(30, 30, 2, 15, n_heads = 3, use_src_pred = use_src_pred, device = device).to(device)
		if use_src_pred == True:
			self.alpha = nn.Parameter(torch.tensor([0.1], device = device))

		if use_embedding == True:
			self.DataAggregationEmbedding = DataAggregationEmbedding(1 + n_dim_extra_inpt + embed_vector_dim, int(n_dim_extra_feat/2))

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
		# self.use_src_pred = self.Arrivals.src_pred
		# self.scale_output = torch.Tensor([1.0/10.0]).to(device)
		# self.use_sigmoid = use_sigmoid
		self.device = device

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q, save_state = False):

		n_line_nodes = Slice.shape[0]
		# mask_p_thresh = 0.025
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]

		embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)

		Slice = torch.cat((Slice, embed_context), dim = 1)

		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			inpt_embedding = torch.cat((inpt_embedding, embed_context), dim = 1)

			embedding = self.DataAggregationEmbedding(inpt_embedding, A_in_sta, A_in_src[0], A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, A_in_sta, A_in_src, A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t)
			Slice = torch.cat((Slice, embedding), dim = 1)
			

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)

		
		if (self.use_gradient_loss == True)*(self.activate_gradient_loss == True):
			x_temp_cuda = Variable(x_temp_cuda, requires_grad = True)
			x_query_cart = Variable(x_query_cart, requires_grad = True)
			t_query = Variable(t_query, requires_grad = True)

		x_latent = self.DataAggregation(Slice, Mask, A_in_sta, A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, A_src if self.use_expanded == False else A_src[0], x_temp_cuda) # x_temp_cuda_cart
		if self.use_expanded == True:
			x = x + self.alpha_expand1*self.SpatialAggregation1_expanded(x, A_src[1], x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, A_src if self.use_expanded == False else A_src[0], x_temp_cuda)
		if self.use_expanded == True:
			x = x + self.alpha_expand2*self.SpatialAggregation2_expanded(x, A_src[1], x_temp_cuda) # x_temp_cuda_cart
		x_spatial = self.SpatialAggregation3(x, A_src if self.use_expanded == False else A_src[0], x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		
		if self.use_direct_output == True:
			y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		else:
			y_latent = self.SpaceTimeAttention(x_spatial, x_temp_cuda_cart, x_temp_cuda_cart, x_temp_cuda_t, x_temp_cuda_t) # contains data on spatial and temporal solution at fixed nodes

		y = self.proj_soln1(y_latent)
		
		if save_state == True:
			self.set_internal_state(x_spatial, x_temp_cuda_cart, x_temp_cuda_t)
			
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).

		x_src = []
		x = self.proj_soln2(x)

		grad_grid_src, grad_grid_t, grad_query_src, grad_query_t = [], [], [], []
		if (self.use_gradient_loss == True)*(self.activate_gradient_loss == True):
			torch_one_vec = torch.ones(len(x_temp_cuda_cart),1).to(x_temp_cuda_cart.device)
			grad_grid = torch.autograd.grad(inputs = x_temp_cuda, outputs = y, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			grad_grid_src, grad_grid_t = grad_grid[:,0:3], (1000.0*self.scale_time)*grad_grid[:,3]
			torch_one_vec = torch.ones(len(x_query_cart),1).to(x_query_cart.device)
			grad_query_src = torch.autograd.grad(inputs = x_query_cart, outputs = x, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			grad_query_t = torch.autograd.grad(inputs = t_query, outputs = x, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]


		slope_width = 0.1
		mask_p_thresh = 0.1
		if self.use_sigmoid == False:
			# mask_out = 1.0*(y.detach() > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
			mask_out = torch.clamp((y - (mask_p_thresh - slope_width/2)) / slope_width, min=0.0, max=1.0)

		else:
			# mask_out = 1.0*(torch.round(torch.sigmoid(y[:,1].reshape(-1,1))).detach()).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
			mask_out = torch.clamp((torch.sigmoid(y[:,1].reshape(-1,1)) - (mask_p_thresh - slope_width/2)) / slope_width, min=0.0, max=1.0)


		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		if self.use_absolute_pos == True:
			s = torch.cat((s, locs_use_cart[A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		## Maybe re-concatenate the initial Cartesian product input misfit features back into s here
		if self.use_expanded == False:
			s = self.DataAggregationAssociationPhase(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"

		else: ## This assumes that DataAggregationAssociationPhase does not use expanded version
			s = self.DataAggregationAssociationPhase(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, A_in_sta, A_in_src[0]) # detach x_latent. Just a "reference"

		arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = trv_out_q)

		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
			if self.use_gradient_loss == False:
				return y, x, arv_p, arv_s, src
			else:
				return [y, x, arv_p, arv_s], [grad_grid_src, grad_grid_t, grad_query_src, grad_query_t], src

		else:
			arv = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)			
			if self.use_gradient_loss == False:
				return y, x, arv_p, arv_s
			else:
				return [y, x, arv_p, arv_s], [grad_grid_src, grad_grid_t, grad_query_src, grad_query_t]


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
			self.SpatialAggregation1_expanded.scale_rel = 10.0*scale_rel
			self.SpatialAggregation1_expanded.scale_time = 10.0*scale_time
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

		# arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = trv_out_q)

		# if self.use_src_pred == True:
		# 	arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)

	def forward_queries(self, x_query_cart, t_query, train = False): # x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t)

		## Use this to obtain query predictions. Note, can modify to also return the spatial embeddings (prior to proj_soln)
		if train == True:

			if self.use_sigmoid == False:
				return self.proj_soln2(self.SpaceTimeAttention(self.x_spatial, x_query_cart, self.x_temp_cuda_cart, t_query, self.x_temp_cuda_t))

			else:
				out = self.proj_soln2(self.SpaceTimeAttention(self.x_spatial, x_query_cart, self.x_temp_cuda_cart, t_query, self.x_temp_cuda_t))
				return out # (torch.round(torch.sigmoid(out[:,1]))*out[:,0]).reshape(-1,1)

		else:
			
			if self.use_sigmoid == False:
				return self.proj_soln2(self.SpaceTimeAttention(self.x_spatial, x_query_cart, self.x_temp_cuda_cart, t_query, self.x_temp_cuda_t))
				
			else:
				
				out = self.proj_soln2(self.SpaceTimeAttention(self.x_spatial, x_query_cart, self.x_temp_cuda_cart, t_query, self.x_temp_cuda_t))
				return (torch.round(torch.sigmoid(out[:,1]))*out[:,0]).reshape(-1,1)

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

		embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)
		
		n_line_nodes = Slice.shape[0]
		# mask_p_thresh = 0.025
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)
		
		Slice = torch.cat((Slice, embed_context), dim = 1)
		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			inpt_embedding = torch.cat((inpt_embedding, embed_context), dim = 1)

			embedding = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src[0], self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t)
			Slice = torch.cat((Slice, embedding), dim = 1)

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)


		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		if self.use_expanded == True:
			x = x + self.alpha_expand1*self.SpatialAggregation1_expanded(x, self.Ac, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		# x = self.SpatialAggregation2(x, A_src, x_temp_cuda)
		if self.use_expanded == True:
			x = x + self.alpha_expand2*self.SpatialAggregation2_expanded(x, self.Ac, x_temp_cuda) # x_temp_cuda_cart
		# x = self.SpatialAggregation2(x, A_src, x_temp_cuda)
		# x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		

		# use_direct_output = False
		if self.use_direct_output == True:
			y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes

		else:
			y_latent = self.SpaceTimeAttention(x_spatial, x_temp_cuda_cart, x_temp_cuda_cart, x_temp_cuda_t, x_temp_cuda_t) # contains data on spatial and temporal solution at fixed nodes

		y = self.proj_soln1(y_latent)
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		
		x_src = []
		x = self.proj_soln2(x)

		## Note below: why detach x_latent?
		# if self.use_sigmoid == False:
		# 	mask_out = 1.0*(y.detach() > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
		# else:
		# 	mask_out = 1.0*(torch.round(torch.sigmoid(y[:,1].reshape(-1,1))).detach()).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?

		slope_width = 0.1
		mask_p_thresh = 0.1
		if self.use_sigmoid == False:
			# mask_out = 1.0*(y.detach() > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
			mask_out = torch.clamp((y - (mask_p_thresh - slope_width/2)) / slope_width, min=0.0, max=1.0)

		else:
			# mask_out = 1.0*(torch.round(torch.sigmoid(y[:,1].reshape(-1,1))).detach()).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
			mask_out = torch.clamp((torch.sigmoid(y[:,1].reshape(-1,1)) - (mask_p_thresh - slope_width/2)) / slope_width, min=0.0, max=1.0)


		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		if self.use_absolute_pos == True:
			s = torch.cat((s, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		if self.use_expanded == False:
			s = self.DataAggregationAssociationPhase(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, self.A_in_sta, self.A_in_src) # detach x_latent. Just a "reference"

		else: ## This assumes that DataAggregationAssociationPhase does not use expanded version
			s = self.DataAggregationAssociationPhase(s, x_latent.detach() if self.use_src_pred == False else self.alpha*x_latent, mask_out_1, Mask, self.A_in_sta, self.A_in_src[0]) # detach x_latent. Just a "reference"

		## Arrival embedding
		arv_embed, mask_arv = self.ArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, self.A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, self.tlatent, trv_out = trv_out_q)
		
		## x_query_src_cart
		if self.use_src_pred == True:
			arv, src = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)

		else:
			arv = self.Arrivals(tq_sample, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)


		if self.use_sigmoid == True:
			y = (torch.round(torch.sigmoid(y[:,1]))*y[:,0]).reshape(-1,1)
			x = (torch.round(torch.sigmoid(x[:,1]))*x[:,0]).reshape(-1,1)

		if self.use_src_pred == True:
			return y, x, arv_p, arv_s, src

		else:
			return y, x, arv_p, arv_s

		
	## Maye need to add new module that maps to the association - source locations
	def forward_fixed_source(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, t_query, n_reshape = 1):
	
		embed_context = self.embed_vector(self.embedding_vector).expand(Slice.shape[0], -1) # .expand(Slice.shape[0], dim = 0)

		n_line_nodes = Slice.shape[0]
		# mask_p_thresh = 0.025
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)

		Slice = torch.cat((Slice, embed_context), dim = 1)

		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			inpt_embedding = torch.cat((inpt_embedding, embed_context), dim = 1)

			embedding = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src[0], self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t)
			Slice = torch.cat((Slice, embedding), dim = 1)

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)


		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		# x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		if self.use_expanded == True:
			x = x + self.alpha_expand1*self.SpatialAggregation1_expanded(x, self.Ac, x_temp_cuda) # x_temp_cuda_cart
		# x = self.SpatialAggregation2(x, A_src, x_temp_cuda)
		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		if self.use_expanded == True:
			x = x + self.alpha_expand2*self.SpatialAggregation2_expanded(x, self.Ac, x_temp_cuda) # x_temp_cuda_cart

		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		x = self.proj_soln2(x)

		if self.use_sigmoid == True:
			x = (torch.round(torch.sigmoid(x[:,1]))*x[:,0]).reshape(-1,1)

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


		# self.f_arrival_query_1 = nn.Linear(ndim_src_in + ndim_arv_in + 5 + 1, n_hidden) # add edge data (observed arrival - theoretical arrival)
		# self.f_arrival_query_2 = nn.Linear(n_hidden, n_heads*n_latent) # Could use nn.Sequential to combine these.
		# self.f_src_context_1 = nn.Linear(ndim_arv_in + 5, n_hidden) # only use single tranform layer for source embdding (which already has sufficient information)
		# self.f_src_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # only use single tranform layer for source embdding (which already has sufficient information)

		# self.f_pick_query = nn.Sequential(nn.Linear(ndim_src_in + ndim_arv_in + 5 + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent))
		
		# + 1


		# self.activate1 = nn.PReLU()
		# self.activate2 = nn.PReLU()
		# self.activate3 = nn.PReLU()

		## Append dummy edge
		# edges = torch.cat((edges, edge_dummy), dim = 1)
		# src_index = torch.cat((src_index, n_src*torch.ones(n_arv*n_src,1)), dim = 0) ## The dummy "source index"
		# src_index = torch.cat((src_index, n_src*torch.ones(n_arv*n_src,1)).to(device), dim = 0) ## The dummy "source index"
		# self_link = torch.cat((self_link, torch.zeros(n_arv*n_src,1)).to(device), dim = 0)
		## Could use search sorted to insert into edge list (sorted)
		## Append dummy edges (new sending nodes from index n_arv*n_src (e.g., > all real picks) to all real nodes)



		## Use gate to merge the two branches
		# gate = torch.sigmoid(self.gate_linear(src_embed_trns[src_ind_repeat]))
		# out = self.proj_2(
		#     gate * self.embed_src(src_embed_trns[src_ind_repeat]) +
		#     (1 - gate) * self.activate4(self.proj_1(...))
		# )


		# if self.use_src_context == True:

		# 	aggregate = self.activate4(self.proj_1(self.propagate(edges, x = (arrival_inpt, arrival_inpt[0:(n_arv*n_src)]), sembed = src_embed_trns, stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = ipick.repeat(n_src), atime = tpick.repeat(n_src), phase = (phase_inpt, phase_inpt[0:(n_arv*n_src)]), self_link = self_link, num_queries = torch.Tensor([n_arv*n_src]).to(self.device), size = (N, M)).view(-1, self.n_latent*self.n_heads)))
		# 	# gate = torch.sigmoid(self.gate_src(torch.cat((src_embed_trns[src_ind_repeat], aggregate), dim = 1)))
		# 	gate = torch.sigmoid(self.gate_src(torch.cat((F.layer_norm(src_embed_trns[src_ind_repeat], src_embed_trns[src_ind_repeat].shape[-1:]), F.layer_norm(aggregate, aggregate.shape[-1:])), dim = 1)))
		# 	out = self.proj_2(self.downscale*gate*self.embed_src(src_embed_trns[src_ind_repeat]) + aggregate)
		# 	# out = self.proj_2(self.embed_src(src_embed_trns[src_ind_repeat]) + ) # M is output. Taking mean over heads

		# else:


