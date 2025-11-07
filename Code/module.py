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
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.autograd import Variable
from torch_scatter import scatter
from numpy.matlib import repmat
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

use_updated_model_definition = config['use_updated_model_definition']
scale_rel = config['scale_rel'] # 30e3
k_sta_edges = config['k_sta_edges']
k_spc_edges = config['k_spc_edges']

## Removing scale_t and eps as free variables. Instead set proportionally to kernel_sig_t
# scale_t = config['scale_t'] # 10.0
# eps = config['eps'] # 15.0
scale_t = train_config['kernel_sig_t']*3.0
eps = train_config['kernel_sig_t']*5.0
scale_time = train_config['scale_time']
kernel_sig_t = train_config['kernel_sig_t']

# use_updated_model_definition = True
use_phase_types = config['use_phase_types']
use_absolute_pos = config['use_absolute_pos']
use_neighbor_assoc_edges = config.get('use_neighbor_assoc_edges', False)
use_expanded = config['use_expanded']
use_gradient_loss = train_config['use_gradient_loss']
use_embedding = config['use_embedding']
attach_time = True

device = torch.device('cuda') ## or use cpu


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

		self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		self.l1_t2_1 = nn.Linear(in_channels, n_hidden)
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
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_mask = 4, use_absolute_pos = use_absolute_pos):
		super(DataAggregationExpanded, self).__init__('mean') # node dim

		if use_absolute_pos == True:
			in_channels = in_channels + 3*2
		
		## Use two layers of SageConv.
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_mask, n_hidden)

		self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		self.l1_t2_1 = nn.Linear(in_channels, n_hidden)
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


	def forward(self, tr, mask, A_in_sta, A_in_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr)), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate12(tr)), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l1_t1_2c(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11c(self.l1_t1_1c(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2c(torch.cat((tr, self.propagate(A_in_src[1], x = self.activate12c(self.l1_t2_1c(tr))), mask), dim = 1))
		tr = self.activate1c(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2c(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21c(self.l2_t1_1c(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2c(torch.cat((tr, self.propagate(A_in_src[1], x = self.activate22c(self.l2_t2_1c(tr))), mask), dim = 1))
		tr = self.activate2c(torch.cat((tr1, tr2), dim = 1))

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

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_edges = 4):
		super(BipartiteGraphOperator, self).__init__('add')
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, A_src_in_edges, mask, n_sta, n_temp):

		N = A_src_in_edges.edge_index[0].max().item() + 1
		M = A_src_in_edges.edge_index[1].max().item() + 1

		return self.activate2(self.fc2(self.propagate(A_src_in_edges.edge_index, size = (N, M), x = mask.max(1, keepdims = True)[0]*self.activate1(self.fc1(torch.cat((inpt, A_src_in_edges.x), dim = -1))))))

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
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, n_hidden = 30, n_heads = 5, scale_rel = scale_rel, scale_time = scale_time):
		super(SpaceTimeAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		# self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.f_queries = nn.Linear(n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_context = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.scale_time = scale_time ## 1 Second is 10 km
		self.fixed_edges = None
		self.edge_features = None
		self.use_fixed_edges = False
		# self.activate3 = nn.PReLU()

	def forward(self, inpts, x_query, x_context, x_query_t, x_context_t, k = 30): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

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


# class SpatialAttention_with_MisfitAggregation(MessagePassing):
# 	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, scale_rel = 30e3, n_picks = 15, n_hidden = 30, n_heads = 5, embed_t = 15.0, device = 'cuda'):
# 		super(SpatialAttention_with_MisfitAggregation, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
# 		# notice node_dim = 0.
# 		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
# 		self.f_context = nn.Linear(inpt_dim + n_dim + n_picks, n_heads*n_latent) # add second layer transformation.
# 		self.f_values = nn.Linear(inpt_dim + n_dim + n_picks, n_heads*n_latent) # add second layer transformation.
# 		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
# 		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
# 		self.scale = np.sqrt(n_latent)
# 		self.n_heads = n_heads
# 		self.n_latent = n_latent
# 		self.scale_rel = scale_rel
# 		self.activate1 = nn.PReLU()
# 		self.activate2 = nn.PReLU()

# 		self.max_aggregate = MaximizeBipartiteAggregation()

# 		self.StationAggregation1 = StationAggregation(4, 30).to(device) # 15, 30
# 		self.StationAggregation2 = StationAggregation(30, 30).to(device) # 15, 30
# 		self.StationAggregation3 = StationAggregation(30, n_picks).to(device) # 15, 30

# 		self.embed_t = embed_t
# 		self.device = device
# 		# self.activate3 = nn.PReLU()

# 	def forward(self, trv, locs, locs_cart, A_sta, inpts, x_query, x_context, tpick, ipick, phase_label, k = 10): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

# 		## Make sure ipick indexes into locs (e.g., uses correct combination of absolute and relative indices)

# 		## Assuming query has origin time of zero?

# 		trv_out = trv(torch.Tensor(locs).to(self.device), x_query)
# 		## For each station and query pair, find nearest matching arrival in tpick.
# 		## Can either use relative time embedding on linear scale, or use message passing
# 		## layer to aggregate over all statons for each pick. E.g., we can readily measure all
# 		## misfits with trv_out[:,ipick_perm,0] - tpick[ipick_perm]
# 		i1 = np.where(phase_label == 0)[0]
# 		i2 = np.where(phase_label == 1)[0]

# 		misfit_time = torch.zeros((len(x_query), len(tpick), 4)).to(self.device)
# 		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/(self.embed_t**2))
# 		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/(self.embed_t**2))
# 		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/(self.embed_t**2))
# 		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/(self.embed_t**2))
		
# 		## Determine unique station indices
# 		ipick_unique = np.unique(ipick.cpu().detach().numpy())
# 		tree_stations = cKDTree(ipick.cpu().detach().numpy().reshape(-1,1))
# 		len_ipick_unique = len(ipick_unique)

# 		edges_read_in = tree_stations.query_ball_point(ipick_unique.reshape(-1,1), r = 0)
# 		edges_source = np.hstack([np.array(list(edges_read_in[i])) for i in range(len_ipick_unique)])
# 		edges_trgt = np.hstack([ipick_unique[i]*np.ones(len(edges_read_in[i])) for i in range(len_ipick_unique)])
# 		edges_read_in = torch.Tensor(np.concatenate((edges_source.reshape(1,-1), edges_trgt.reshape(1,-1)), axis = 0)).long().to(self.device)
# 		locs_cart_unsqueeze = locs_cart.unsqueeze(0).repeat(len(x_query),1,1)

# 		embed_picks = self.max_aggregate(misfit_time, edges_read_in, len(ipick), len(locs))
# 		embed_picks = self.StationAggregation1(embed_picks, A_sta, locs_cart_unsqueeze)
# 		embed_picks = self.StationAggregation2(embed_picks, A_sta, locs_cart_unsqueeze)
# 		embed_picks = self.StationAggregation3(embed_picks, A_sta, locs_cart_unsqueeze)
# 		embed_picks = embed_picks.mean(1)

# 		## Apply regualar spatial attention with extra features from station aggregation
# 		edge_index = knn(x_context/1000.0, x_query/1000.0, k = k).flip(0)
# 		edge_attr = torch.cat(((x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel, embed_picks[edge_index[1]]), dim = 1) # /scale_x

# 		return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

# 	def message(self, x_j, index, edge_attr):

# 		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
# 		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
# 		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

# 		alpha = softmax(alpha, index)

# 		return alpha.unsqueeze(-1)*value_embed

# class MaximizeBipartiteAggregation(MessagePassing):

# 	def __init__(self):
# 		super(MaximizeBipartiteAggregation, self).__init__(node_dim = 1, aggr = 'max')

# 	def forward(self, inpt, edges, N, M):

# 		return self.propagate(edges, size = (N, M), x = inpt)

class SpaceTimeAttentionQuery(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, n_hidden = 30, n_heads = 5, kernel_sig_t = kernel_sig_t, locs_use = None, trv = None, ftrns2 = None, scale_rel = scale_rel, scale_time = scale_time):
		super(SpaceTimeAttentionQuery, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		# self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.f_queries = nn.Linear(n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_context = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		# self.proj = nn.Linear(n_latent*n_heads, out_channels) # can remove this layer possibly.
		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.scale_time = scale_time ## 1 Second is 10 km
		self.ftrns2 = ftrns2

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

		## Compute misfit times between all source and pick pairs
		## For each station and query pair, find nearest matching arrival in tpick.
		## Can either use relative time embedding on linear scale, or use message passing
		## layer to aggregate over all statons for each pick. E.g., we can readily measure all
		## misfits with trv_out[:,ipick_perm,0] - tpick[ipick_perm]

		misfit_time = torch.zeros((len(x_query), len(tpick), 4)).to(self.device)
		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/(self.kernel_sig_t**2))
		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/(self.kernel_sig_t**2))
		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/(self.kernel_sig_t**2))
		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/(self.kernel_sig_t**2))
		
		## Determine unique station indices
		ipick_unique = np.unique(ipick.cpu().detach().numpy())
		tree_stations = cKDTree(ipick.cpu().detach().numpy().reshape(-1,1))
		len_ipick_unique = len(ipick_unique)
		edges_read_in = tree_stations.query_ball_point(ipick_unique.reshape(-1,1), r = 0)

		edges_source = np.hstack([np.array(list(edges_read_in[i])) for i in range(len_ipick_unique)])
		edges_trgt = np.hstack([ipick_unique[i]*np.ones(len(edges_read_in[i])) for i in range(len_ipick_unique)])
		edges_read_in = torch.Tensor(np.concatenate((edges_source.reshape(1,-1), edges_trgt.reshape(1,-1)), axis = 0)).long().to(self.device)
		# locs_cart_unsqueeze = locs_cart.unsqueeze(0).repeat(len(x_query),1,1)

		## Shouldnt the output here be the size of ipick and the number of sources
		## This may be why we are using broadcasting to duplicate the source - station features?
		## So when we aggregate we are only aggregating over the stations

		# embed_picks = self.max_aggregate(misfit_time, edges_read_in, len(ipick), len(locs_use))
		# embed_picks = scatter(misfit_time[edges_read_in[0]], edges_read_in[1], dim = 0, dim_size = len(ipick), reduce = 'max')
		embed_picks = scatter(misfit_time[edges_read_in[0]], edges_read_in[1], dim = 1, dim_size = len(locs_use_cart), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations

		## Note: can also include the misfits between the theoretical arrivals of the reference nodes (e.g., Xt) used in the aggregation over the Cartesian product.
		## That is, the differences between tlatent and observed times. (Though these are given by the initial input features).
		## This gives an additional reference of the misfits with respect to the reference node, which should help weight those nodes
		## contributions in the aggregation

		pdb.set_trace()

		## Be careful as the aggregation over the input (recall the graph is bipartite) is over the pick indices or over station indices?
		## The output is supposed to be pick indices. Apparently the input is also pick indices (not station indices), as this is necessary
		## due to having multiple picks per station, and also misfit_time is shaped this way.


		## May in theory be shape: source query x picks x phase type, based on max fits over any set of source query and stations and picks.
		## If so, it should be usable / equivalent in the source arrival attention query layers? Is it per-pick or per-station?
		## Indeed but perhaps we now need to "pair" these pick misfits with the induced nodes of the Cartwsian product.
		## Can graph these by using the cumulative degrees of the source nodes, and the degree of source nodes, to extract the
		## subset of nodes of the product which contain the "possible" relevant stations for all sources.
		## We now need to pair any of the picks (which have a station index) to the subset of these they match with within these groups.
		## Ideally we can find where the pick index == the station index in these subsets of nodes of the product.
		## Or if not, we can aggregate over all of them (for each station), and use a masking operation to zero out
		## the components that dont have equal station indices. This seems expensive as it duplicates the number of edges of each station
		## to these subsets of nodes by the degree of each subset, and just zeros out the irrelevant components during aggregation.

		## More generally, we know the station indices of each of these subsets (per query). We know the pick indices we have, and we already know
		## which of those picks (for a given query) has a non-negligable misfit (question: are the already obtained pick misfit embeddings unreaonably)
		## expensive to have currently obtained? Could we reduce their cost by using sparisty on the bounds of travel times before computing all pairs of travel times?
		## I.e., why are we computing the dense travel time pairs (these are necessary for getting pairs beyond the subgraph).

		## Based on the already obtained minimum misfit, we know for any query and pick if fitness >> threshold.
		## For this sparser set, we can either do the "overcomplete" aggregation with masking, or can afford to identify matched indices.
		## Again, we know for each query the subset of nodes of product relevant, and these have a station index (and source index).
		## Now for each non-zero misfit pair (source and station), we need to find if they exist in these subsets (e.g., a tree.query_ball_point)
		## type operation. This should likely be broadcastable for all the pairs sources-stations using a cKDTree and query_ball_point.
		## The non-empty sets of the query ball point will identify the stations that do exists in these subsets, and then we can directly
		## create this set of edges. We should then be able to aggregate for each query and pick (> min misfit), over the subset of relevant entries in the
		## Cartesian product (which are nearby source nodes to the query source and the subset of nodes of matched stations to the pick). We can concatenate
		## the misfit itself into the aggregation, so hence the embedded feature accounts for the misfit and geometric features.

		## Then this biparite aggregate (over the product to the source-pick query) can be globally summed or meaned to obtain the per-source embedding
		## and it can also be used in the Source-Arrival attention layer (as it will represent source-arrival pick enriched features).
		## Hence, this replaces the arrival embedding layers. Note, how do we currently use the absolute misfit between pick query and the theoretical arrival
		## in the current source-arrival attention layers


		## Could in theory, "access" each pick feature as a carrier along with Cartesian product graph, and for any
		## pick without a node on the Cartesian product (for that source), aggregate as a global sum of the misfits, or an
		## aggregation on the induced (or created) full station graph.

		## To pair with the cartesian product, for each source query must find k-nn edges of source nodes, then the cumulative degrees
		## of these nodes to access the station nodes of those nodes of the product, then use aggregation to find picks that actually occur
		## on these nodes (and if so, transmit those matched features of the cartesian product). Then can take a global sum over the transformed
		## misfits and Cartesian product vectors of this per-query set of per-pick (station) features. Then add the global sum of the "remainder"
		## from the missed part of the Cartesian product (e.g., misfits, but no features).

		## Together this gives a reasonable query feature (in space) that's "enriched" by the misfit features.
		## Note: re-using the Catesian product features are necessary so that the misfit values themselves don't
		## totally drive the prediction (e.g., this introduct the geometry component).
		

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
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_latent = 30, n_dim_mask = 5, use_absolute_pos = use_absolute_pos):
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


	def forward(self, tr, latent, mask1, mask2, A_in_sta, A_in_src):

		mask = torch.cat((mask1, mask2), dim = - 1)
		tr = torch.cat((tr, latent, mask), dim = -1)
		tr = self.activate(self.init_trns(tr)) # should tlatent appear here too? Not on first go..

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(self.l1_t1_1(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate12(self.l1_t2_1(tr))), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l1_t1_2c(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11c(self.l1_t1_1c(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l1_t2_2c(torch.cat((tr, self.propagate(A_in_src[1], x = self.activate12c(self.l1_t2_1c(tr))), mask), dim = 1))
		tr = self.activate1c(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2c(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21c(self.l2_t1_1c(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2c(torch.cat((tr, self.propagate(A_in_src[1], x = self.activate22c(self.l2_t2_1c(tr))), mask), dim = 1))
		tr = self.activate2c(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l3_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate31(self.l3_t1_1(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l3_t2_2(torch.cat((tr, self.propagate(A_in_src[0], x = self.activate32(self.l3_t2_1(tr))), mask), dim = 1))
		tr = self.activate3(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.
			

class SourceArrivalEmbedding(MessagePassing):
	def __init__(self, ndim_src_in, ndim_out, n_hidden = 30, scale_rel = scale_rel, k_spc_edges = k_spc_edges, kernel_sig_t = kernel_sig_t, use_phase_types = use_phase_types, scale_time = scale_time, min_thresh = 0.01, trv = None, ftrns2 = None, device = device):
		# super(SourceArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean
		super(SourceArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean

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
		self.dilate_scale = 1.0
		self.scale_misfit = 2.0

		self.fc1 = nn.Sequential(nn.Linear(n_hidden + 16, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.fc2 = nn.Sequential(nn.Linear(8, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden)) ## Inputs: 4 x misfit features, query and reference, 6 offset features, query and reference, 2 norm features
		self.fc3 = nn.Sequential(nn.Linear(2*n_hidden, n_hidden), nn.PReLU()) ## Can consider changing this merging layer

		# self.fixed_edges

	def forward(self, x, x_context_cart, x_context_t, x_query_cart, x_query_t, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = None): # reference k nearest spatial points

		## Can add fixed edge option for use in SpaceTimeAttentionQuery
		# if self.use_fixed_edges == True:
		# 	trv_out = self.trv_out_fixed

		if trv_out is None:
			trv_out = self.trv(self.ftrns2(locs_use_cart), self.ftrns2(x_query_cart)) + x_query_t.reshape(-1, 1, 1) ## Use full travel times, as we check for stations from the full product
		else: 
			trv_out = trv_out + x_query_t.reshape(-1, 1, 1)

		## degree_srcs, cum_degree_srcs

		## Note: should also consider using source reciever offset positions.
		## Note, can use this feature even for the isolated query node - reciever message (e.g., irrespective of incoming Cartesian product nodes)

		st = time.time()

		if self.use_phase_types == False:
			phase_label = phase_label*0.0

		## degree_srcs on cartesian product
		## cum_degree_srcs on cartesian product
		## tlatent are travel times to the reference nodes of Cartesian product (note: could these bound the pairs that are relevent for a given query?)

		# ipick_unique = torch.unique(ipick).long()
		i1 = torch.where(phase_label == 0)[0]
		i2 = torch.where(phase_label == 1)[0]

		misfit_time = torch.zeros((len(x_query_cart), len(tpick), 4)).to(self.device) ## Question: is it necessary to produce these pairwise misfits? Can we focus on the pairs that "likely" have arrival times within threshold (e.g., bound min and max times based on distances between src reciever first, before computing travel times)
		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		

		use_pick_embedding = False
		if use_pick_embedding == True:

			## Note this is not used

			## Determine unique station indices
			ipick_unique = np.unique(ipick.cpu().detach().numpy())
			tree_stations = cKDTree(ipick.cpu().detach().numpy().reshape(-1,1))
			len_ipick_unique = len(ipick_unique)
			edges_read_in = tree_stations.query_ball_point(ipick_unique.reshape(-1,1), r = 0)

			edges_source = np.hstack([np.array(list(edges_read_in[i])) for i in range(len_ipick_unique)])
			edges_trgt = np.hstack([ipick_unique[i]*np.ones(len(edges_read_in[i])) for i in range(len_ipick_unique)])
			edges_read_in = torch.Tensor(np.concatenate((edges_source.reshape(1,-1), edges_trgt.reshape(1,-1)), axis = 0)).long().to(self.device)
			
			# embed_picks = scatter(misfit_time[edges_read_in[0]], edges_read_in[1], dim = 1, dim_size = len(locs_use_cart), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations
			embed_picks = scatter(misfit_time[:,edges_read_in[0],:], edges_read_in[1], dim = 1, dim_size = len(locs_use_cart), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations

		# embed_picks = scatter(misfit_time[:,edges_read_in[0],:], edges_read_in[1], dim = 1, dim_size = len(ipick), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations

		# print('Time %0.4f'%(time.time() - st))

		## Can compute these degree vectors outside of loop
		degree_srcs = degree(A_src_in_sta[1], num_nodes = len(x_context_cart), dtype = torch.long)
		cum_degree_srcs = torch.cat((torch.zeros(1).to(self.device), torch.cumsum(degree_srcs, dim = 0)[0:-1]), dim = 0).long()
		## Should check if minimal degree srcs really are accessing nearest stations

		# print('Time %0.4f'%(time.time() - st))

		## Find active source - arrival queries (base it on exact P and S fits, rather than max over the set; is it very different?)
		# i1p, i1s = torch.where(misfit_time)
		mask_misfit_time = misfit_time.max(2).values > self.min_thresh ## Save this, so can use as mask in the attention layer
		isrc, iarv = torch.where(mask_misfit_time == 1)
		## For this subset of source - arrivals, now must find the "matches" to entries of the subset of extracted indices from the subgraph Cartesian product (based on queries)

		## Build src-src indices (may or may not use the edge feature of source query to source node offsets)
		edge_index = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query_cart/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = self.k_spc_edges).flip(0)
		# edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x

		# Build a single flattened arange from size = sum(idx)
		deg_slice = degree_srcs[edge_index[0]]
		assert(deg_slice.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
		inc_inds = torch.arange(deg_slice.sum()).long().to(self.device)
		inc_inds = inc_inds - torch.repeat_interleave(torch.cumsum(deg_slice, dim = 0) - deg_slice, deg_slice)
		nodes_of_product = cum_degree_srcs[edge_index[0]].repeat_interleave(degree_srcs[edge_index[0]]) + inc_inds
		ind_query = torch.arange(len(x_query_cart)).long().to(device).repeat_interleave(scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)

		sta_src_pairs = A_src_in_sta[:, nodes_of_product]
		## Query_vals is shaped based on nodes_of_product. So when we aggregate or want to extract Cartesian product node features, we can use these.

		# k_matches = knn(sta_src_pairs.T, torch.cat((ipick[iarv].reshape(-1,1), ))
		query_vals = torch.cat((sta_src_pairs[0].reshape(-1,1), ind_query.reshape(-1,1)), dim = 1).long() # .float()
		pick_vals = torch.cat((ipick[iarv].reshape(-1,1), isrc.reshape(-1,1)), dim = 1).long() # .float()

		## Note: query_vals represents the pairs of station and query inds
		## pick_vals represents the pairs of station and query inds

		# print('Time %0.4f'%(time.time() - st))

		hash_picks, hash_queries = hash_rows(pick_vals), hash_rows(query_vals) ## Do not define directly if only using one mask below
		mask_picks = torch.isin(hash_picks, hash_queries) # set(map(tuple, l1))
		mask_queries = torch.isin(hash_queries, hash_picks) # set(map(tuple, l1))
		iwhere_picks = torch.where(mask_picks == 1)[0]
		iwhere_query = torch.where(mask_queries == 1)[0]
		# assert(torch.abs(query_vals[iwhere_query] - pick_vals[knn(pick_vals, query_vals[iwhere_query], k = 1)[1]]).max() == 0)
		# assert(torch.abs(pick_vals[iwhere_picks] - query_vals[knn(query_vals, pick_vals[iwhere_picks], k = 1)[1]]).max() == 0)
		## The point of query vals is these are the nodes on the Cartesian product we are accessing and aggregating across.
		## How can we "read into" these nodes, or match to these nodes, for all possible (> min thresh) pick vals.
		## Can we use degrees or cumulative degrees of query vals to directly read in? Can we catch cases where the pick
		## has no match (e.g., read in, but then find mis-match of values and remove?)

		# print('Time %0.4f'%(time.time() - st))

		sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
		ind_extract = torch.searchsorted(sorted_hash_picks, hash_queries[iwhere_query])
		valid_ind = (ind_extract < len(sorted_hash_picks)) & (sorted_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)] == hash_queries[iwhere_query])
		inds_queries_to_picks = order_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)][valid_ind]

		# print('Time %0.4f'%(time.time() - st))

		## Compute features
		misfit_rel_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - tlatent[nodes_of_product[iwhere_query]]
		misfit_query_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - trv_out[query_vals[iwhere_query,1], ipick[iarv[inds_queries_to_picks]], :]
		misfit_rel_time = torch.cat((torch.exp(-0.5*(misfit_rel_time**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_rel_time)), dim = 1)
		misfit_query_time = torch.cat((torch.exp(-0.5*(misfit_query_time**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_query_time)), dim = 1)
		offset_src_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_query_cart[query_vals[iwhere_query,1]])/(5.0*self.scale_rel)
		offset_ref_sta = (locs_use_cart[ipick[iarv[inds_queries_to_picks]]] - x_context_cart[A_src_in_sta[1,nodes_of_product[iwhere_query]],:])/(5.0*self.scale_rel)
		offset_src_sta_norm = torch.norm(offset_src_sta, dim = 1, keepdim = True)
		offset_ref_sta_norm = torch.norm(offset_ref_sta, dim = 1, keepdim = True)
		inpt_aggregate = torch.cat((x[nodes_of_product[iwhere_query]], misfit_rel_time, misfit_query_time, offset_src_sta, offset_ref_sta, offset_src_sta_norm, offset_ref_sta_norm), dim = 1)
		aggregate_product = scatter(self.fc1(inpt_aggregate), inds_queries_to_picks, dim = 0, dim_size = len(iarv), reduce = 'mean')

		# print('Time %0.4f'%(time.time() - st))

		## Make direct pick feature embedding vector (just based on pick_vals; or station index, arrival index, and picks)
		misfit_query_time_direct = tpick[iarv].reshape(-1,1) - trv_out[pick_vals[:,1], ipick[iarv], :] ## Can check if these embeddings match the approach with embed_picks
		misfit_query_time_direct = torch.cat((torch.exp(-0.5*(misfit_query_time_direct**2)/(((self.scale_misfit*self.kernel_sig_t)**2))), torch.sign(misfit_query_time_direct)), dim = 1)
		offset_src_sta_direct = (locs_use_cart[ipick[iarv]] - x_query_cart[pick_vals[:,1]])/(5.0*self.scale_rel)
		offset_src_sta_norm_direct = torch.norm(offset_src_sta_direct, dim = 1, keepdim = True)
		inpt_direct = torch.cat((misfit_query_time_direct, offset_src_sta_direct/offset_src_sta_norm_direct, offset_src_sta_norm_direct), dim = 1)
		aggregate_direct = self.fc2(inpt_direct)

		# print('Time %0.4f'%(time.time() - st))

		## Make merged embedding
		aggregate_picks = self.fc3(torch.cat((aggregate_product, aggregate_direct), dim = 1))

		## Map to full array (for consistency with StationSourceArrivalAttention; should change to 
		## only use the sparse set; could implement the attention layer here, inside this module)
		arv_embed = torch.zeros((len(x_query_cart), len(tpick), aggregate_picks.shape[1])).to(device)
		arv_embed[pick_vals[:,1], iarv, :] = aggregate_picks

		## Now map this into the full query vs. pick vs feature dimension

		## Want to aggregate for fixed pick and query pairs (note: degrees should be > 1 because each query induces a neighborhood of source nodes on the product)

		# accum_inds = torch.cat(())

		## Per pick and query embeddings

		# print('Time %0.4f'%(time.time() - st))

		# pdb.set_trace()

		return arv_embed, mask_misfit_time ## Make sure this is correct reshape (not transposed)

		# return aggregate_picks, pick_vals ## Make sure this is correct reshape (not transposed)

	# def message(self, x_j, edge_index, index, tsrc_p, tsrc_s, sembed, sindex, stindex, stime, atime, phase_j): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.


	# 	return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))



class StationSourceAttention(MessagePassing):
	def __init__(self, ndim_src_in, ndim_arv_in, ndim_out, n_latent, ndim_extra = 1, n_heads = 5, n_hidden = 30, eps = eps, use_phase_types = use_phase_types, device = device):
		super(StationSourceAttention, self).__init__(node_dim = 0, aggr = 'add') # check node dim.

		self.f_arrival_query_1 = nn.Linear(2*ndim_arv_in + 6, n_hidden) # add edge data (observed arrival - theoretical arrival)
		self.f_arrival_query_2 = nn.Linear(n_hidden, n_heads*n_latent) # Could use nn.Sequential to combine these.
		self.f_src_context_1 = nn.Linear(ndim_src_in + ndim_extra + 1, n_hidden) # only use single tranform layer for source embdding (which already has sufficient information)
		self.f_src_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # only use single tranform layer for source embdding (which already has sufficient information)

		self.f_values_1 = nn.Linear(2*ndim_arv_in + ndim_extra + 6, n_hidden) # add second layer transformation.
		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, ndim_out) # can remove this layer possibly.

		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.eps = eps
		self.t_kernel_sq = torch.Tensor([eps]).to(device)**2
		self.ndim_feat = ndim_arv_in + ndim_extra
		self.use_phase_types = use_phase_types
		self.n_phases = ndim_out

		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		# self.activate5 = nn.PReLU()
		self.device = device

	def forward(self, src, stime, src_embed, trv_src, locs_cart, arrival, mask_arv, tpick, ipick, phase_label): # reference k nearest spatial points

		# src isn't used. Only trv_src is needed.
		n_src, n_sta, n_arv = src.shape[0], trv_src.shape[1], len(tpick) # + 1 ## Note: adding 1 to size of arrivals!
		# n_arv = len(tpick)
		ip_unique = torch.unique(ipick).float().cpu().detach().numpy() # unique stations
		tree_indices = cKDTree(ipick.float().cpu().detach().numpy().reshape(-1,1))
		unique_sta_lists = tree_indices.query_ball_point(ip_unique.reshape(-1,1), r = 0)
		if self.use_phase_types == False:
			phase_label = phase_label*0.0

		edges = torch.Tensor(np.copy(np.flip(np.hstack([np.ascontiguousarray(np.array(list(zip(itertools.product(unique_sta_lists[j], unique_sta_lists[j]))))[:,0,:].T) for j in range(len(unique_sta_lists))]), axis = 0))).long().to(self.device) # note: preferably could remove loop here.
		n_edge = edges.shape[1]

		## Now must duplicate edges, for each unique source. (different accumulation points)
		edges = (edges.repeat(1, n_src) + torch.cat(((torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(self.device), (torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(self.device)), dim = 0)).long().contiguous()
		src_index = torch.arange(n_src).repeat_interleave(n_edge).contiguous().long().to(self.device)

		use_sparse = True
		if use_sparse == True:
			ikeep = torch.where((mask_arv.reshape(-1)[edges[1]] > 0) + (mask_arv.reshape(-1)[edges[0]] > 0) + (edges[0] == edges[1]))[0] ## Either query is within the threshold amount of time
			edges = torch.cat((edges[0][ikeep].reshape(1,-1), edges[1][ikeep].reshape(1,-1)), dim = 0).contiguous()
			src_index = src_index[ikeep]		
		
		N = n_arv*n_src # still correct?
		M = n_arv*n_src

		if len(src_index) == 0:
			return torch.zeros(n_src, n_arv, self.n_phases).to(self.device)

		# pdb.set_trace()
		
		out = self.proj_2(self.activate4(self.proj_1(self.propagate(edges, x = arrival.reshape(n_arv*n_src,-1), sembed = src_embed, stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = ipick.repeat(n_src), atime = tpick.repeat(n_src), phase = phase_label.repeat(n_src, 1), size = (N, M)).mean(1)))) # M is output. Taking mean over heads

		return out.view(n_src, n_arv, -1) ## Make sure this is correct reshape (not transposed)

	def message(self, x_j, edge_index, index, tsrc_p, tsrc_s, sembed, sindex, stindex, stime, atime, phase_j): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.

		assert(abs(edge_index[1] - index).max().item() == 0)

		ifind = torch.where(edge_index[0] == edge_index[0].max())[0]

		rel_t_p = (atime[edge_index[0]] - (tsrc_p[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_p = torch.cat((torch.exp(-0.5*(rel_t_p**2)/self.t_kernel_sq), torch.sign(rel_t_p), phase_j), dim = 1) # phase[edge_index[0]]

		rel_t_s = (atime[edge_index[0]] - (tsrc_s[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s = torch.cat((torch.exp(-0.5*(rel_t_s**2)/self.t_kernel_sq), torch.sign(rel_t_s), phase_j), dim = 1) # phase[edge_index[0]]

		# Denote self-links by a feature.
		self_link = (edge_index[0] == torch.remainder(edge_index[1], edge_index[0].max().item() + 1)).reshape(-1,1).detach().float() # Each accumulation index (an entry from src cross arrivals). The number of arrivals is edge_index.max() exactly (since tensor is composed of number arrivals + 1)
		contexts = self.f_src_context_2(self.activate1(self.f_src_context_1(torch.cat((sembed[sindex], stime[sindex].reshape(-1,1).detach(), self_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)
		queries = self.f_arrival_query_2(self.activate2(self.f_arrival_query_1(torch.cat((x_j, rel_t_p, rel_t_s), dim = 1)))).view(-1, self.n_heads, self.n_latent)
		values = self.f_values_2(self.activate3(self.f_values_1(torch.cat((x_j, rel_t_p, rel_t_s, self_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)

		# When using sparse, this assert is not true
		# assert(self_link.sum() == (len(atime) - 1)*tsrc_p.shape[0])

		## Do computation
		scores = (queries*contexts).sum(-1)/self.scale
		alpha = softmax(scores, index)

		return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))




class StationSourceAttentionMergedPhases(MessagePassing):
	def __init__(self, ndim_src_in, ndim_arv_in, ndim_out, n_latent, ndim_extra = 1, n_heads = 5, n_hidden = 30, scale_rel = scale_rel, k_sta_edges = k_sta_edges, eps = eps, use_neighbor_assoc_edges = use_neighbor_assoc_edges, use_phase_types = use_phase_types, device = device):
		super(StationSourceAttentionMergedPhases, self).__init__(node_dim = 0, aggr = 'add') # check node dim.

		# if use_neighbor_assoc_edges == True: ndim_extra = ndim_extra + 1 + 3 ## Add one bimary feature to indicate if edge is for a common station, and the relative offset positions
		self.f_arrival_query_1 = nn.Linear(2*ndim_arv_in + 6, n_hidden) # add edge data (observed arrival - theoretical arrival)
		self.f_arrival_query_2 = nn.Linear(n_hidden, n_heads*n_latent) # Could use nn.Sequential to combine these.
		self.f_src_context_1 = nn.Linear(ndim_src_in + ndim_extra + 2, n_hidden) # only use single tranform layer for source embdding (which already has sufficient information)
		self.f_src_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # only use single tranform layer for source embdding (which already has sufficient information)

		self.f_values_1 = nn.Linear(2*ndim_arv_in + ndim_extra + 7, n_hidden) # add second layer transformation.
		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, ndim_out) # can remove this layer possibly.

		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.eps = eps
		self.t_kernel_sq = torch.Tensor([eps]).to(device)**2
		self.ndim_feat = ndim_arv_in + ndim_extra
		self.use_phase_types = use_phase_types
		self.use_neighbor_assoc_edges = use_neighbor_assoc_edges

		frac_sta_edges = 1.0
		self.k_sta_edges = int(np.ceil(frac_sta_edges*k_sta_edges))

		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		self.scale_rel = scale_rel
		# self.activate5 = nn.PReLU()
		self.device = device

	def forward(self, src, stime, src_embed, trv_src, locs_cart, arrival_p, arrival_s, tpick, ipick, phase_label): # reference k nearest spatial points

		# src isn't used. Only trv_src is needed.
		n_src, n_sta, n_arv = src.shape[0], trv_src.shape[1], len(tpick) # + 1 ## Note: adding 1 to size of arrivals!
		# n_arv = len(tpick)
		ip_unique = torch.unique(ipick).float().cpu().detach().numpy() # unique stations
		tree_indices = cKDTree(ipick.float().cpu().detach().numpy().reshape(-1,1))
		unique_sta_lists = tree_indices.query_ball_point(ip_unique.reshape(-1,1), r = 0)
		if self.use_phase_types == False:
			phase_label = phase_label*0.0

		arrival_p = torch.cat((arrival_p, torch.zeros(1,arrival_p.shape[1]).to(self.device)), dim = 0) # add null arrival, that all arrivals link too. This acts as a "stabalizer" in the inner-product space, and allows softmax to not blow up for arrivals with only self loops. May not be necessary.
		arrival_s = torch.cat((arrival_s, torch.zeros(1,arrival_s.shape[1]).to(self.device)), dim = 0) # add null arrival, that all arrivals link too. This acts as a "stabalizer" in the inner-product space, and allows softmax to not blow up for arrivals with only self loops. May not be necessary.
		arrival = torch.cat((arrival_p, arrival_s), dim = 1) # Concatenate across feature axis

		edges = torch.Tensor(np.copy(np.flip(np.hstack([np.ascontiguousarray(np.array(list(zip(itertools.product(unique_sta_lists[j], np.concatenate((unique_sta_lists[j], np.array([n_arv])), axis = 0)))))[:,0,:].T) for j in range(len(unique_sta_lists))]), axis = 0))).long().to(self.device) # note: preferably could remove loop here.
		n_edge = edges.shape[1]

		## Now must duplicate edges, for each unique source. (different accumulation points)
		edges = (edges.repeat(1, n_src) + torch.cat((torch.zeros(1, n_src*n_edge).to(self.device), (torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(self.device)), dim = 0)).long().contiguous()
		src_index = torch.arange(n_src).repeat_interleave(n_edge).contiguous().long().to(self.device)

		use_sparse = True
		if use_sparse == True:
			# pdb.set_trace()
			## Find which values have offset times that exceed max time, and ignore these edges (does this work?)
			rel_t_p = (torch.cat((tpick, torch.Tensor([-self.eps]).to(self.device)), dim = 0)[edges[0]] - (torch.cat((trv_src[:,:,0], -self.eps*torch.ones(n_src,1).to(self.device)), dim = 1).detach()[src_index, torch.cat((ipick, torch.Tensor([n_sta]).long().to(self.device)), dim = 0)[edges[0]]] + stime[src_index])).reshape(-1,1).detach()
			rel_t_s = (torch.cat((tpick, torch.Tensor([-self.eps]).to(self.device)), dim = 0)[edges[0]] - (torch.cat((trv_src[:,:,1], -self.eps*torch.ones(n_src,1).to(self.device)), dim = 1).detach()[src_index, torch.cat((ipick, torch.Tensor([n_sta]).long().to(self.device)), dim = 0)[edges[0]]] + stime[src_index])).reshape(-1,1).detach()
			ikeep = torch.where(((torch.abs(rel_t_p) < 2.0*torch.sqrt(self.t_kernel_sq)) + (torch.abs(rel_t_s) < 2.0*torch.sqrt(self.t_kernel_sq))).reshape(-1) > 0)[0].cpu().detach().numpy() ## Either query is within the threshold amount of time
			# edges = edges[:,ikeep]
			edges = torch.cat((edges[0][ikeep].reshape(1,-1), edges[1][ikeep].reshape(1,-1)), dim = 0).contiguous()
			src_index = src_index[ikeep]

		
		## Add neighbor edges
		if self.use_neighbor_assoc_edges == True:
			## Can only run k nearest neighbors for one 
			# k_edges = knn(locs_cart/1000.0, locs_cart/1000.0, k = self.k_sta_edges + 1).flip(0)
			# iactive_sta = (-1*torch.ones(len(locs_cart)).to(device)).long()
			# iactive_sta[ipick[edges[1]]] = 1 ## At least one pick for a station
			pass
			
			
		
		N = n_arv + 1 # still correct?
		M = n_arv*n_src

		out = self.proj_2(self.activate4(self.proj_1(self.propagate(edges, x = arrival, sembed = src_embed, stime = stime, tsrc_p = torch.cat((trv_src[:,:,0], -self.eps*torch.ones(n_src,1).to(self.device)), dim = 1).detach(), tsrc_s = torch.cat((trv_src[:,:,1], -self.eps*torch.ones(n_src,1).to(self.device)), dim = 1).detach(), sindex = src_index, stindex = torch.cat((ipick, torch.Tensor([n_sta]).long().to(self.device)), dim = 0), atime = torch.cat((tpick, torch.Tensor([-self.eps]).to(self.device)), dim = 0), phase = torch.cat((phase_label, torch.Tensor([-1.0]).reshape(1,1).to(self.device)), dim = 0), size = (N, M)).mean(1)))) # M is output. Taking mean over heads

		return out.view(n_src, n_arv, -1) ## Make sure this is correct reshape (not transposed)

	def message(self, x_j, edge_index, index, tsrc_p, tsrc_s, sembed, sindex, stindex, stime, atime, phase_j): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.

		assert(abs(edge_index[1] - index).max().item() == 0)

		ifind = torch.where(edge_index[0] == edge_index[0].max())[0]

		rel_t_p = (atime[edge_index[0]] - (tsrc_p[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_p = torch.cat((torch.exp(-0.5*(rel_t_p**2)/self.t_kernel_sq), torch.sign(rel_t_p), phase_j), dim = 1) # phase[edge_index[0]]

		rel_t_s = (atime[edge_index[0]] - (tsrc_s[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s = torch.cat((torch.exp(-0.5*(rel_t_s**2)/self.t_kernel_sq), torch.sign(rel_t_s), phase_j), dim = 1) # phase[edge_index[0]]

		# Denote self-links by a feature.
		self_link = (edge_index[0] == torch.remainder(edge_index[1], edge_index[0].max().item())).reshape(-1,1).detach().float() # Each accumulation index (an entry from src cross arrivals). The number of arrivals is edge_index.max() exactly (since tensor is composed of number arrivals + 1)
		null_link = (edge_index[0] == edge_index[0].max().item()).reshape(-1,1).detach().float()
		contexts = self.f_src_context_2(self.activate1(self.f_src_context_1(torch.cat((sembed[sindex], stime[sindex].reshape(-1,1).detach(), self_link, null_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)
		queries = self.f_arrival_query_2(self.activate2(self.f_arrival_query_1(torch.cat((x_j, rel_t_p, rel_t_s), dim = 1)))).view(-1, self.n_heads, self.n_latent)
		values = self.f_values_2(self.activate3(self.f_values_1(torch.cat((x_j, rel_t_p, rel_t_s, self_link, null_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)

		# When using sparse, this assert is not true
		# assert(self_link.sum() == (len(atime) - 1)*tsrc_p.shape[0])

		## Do computation
		scores = (queries*contexts).sum(-1)/self.scale
		alpha = softmax(scores, index)

		return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))


# class StationSourceAttentionMergedPhases(MessagePassing):
# 	def __init__(self, ndim_src_in, ndim_arv_in, ndim_out, n_latent, ndim_extra = 1, n_heads = 5, n_hidden = 30, eps = eps, use_phase_types = use_phase_types, device = device):
# 		super(StationSourceAttentionMergedPhases, self).__init__(node_dim = 0, aggr = 'add') # check node dim.

# 		self.f_arrival_query_1 = nn.Linear(2*ndim_arv_in + 6, n_hidden) # add edge data (observed arrival - theoretical arrival)
# 		self.f_arrival_query_2 = nn.Linear(n_hidden, n_heads*n_latent) # Could use nn.Sequential to combine these.
# 		self.f_src_context_1 = nn.Linear(ndim_src_in + ndim_extra + 1, n_hidden) # only use single tranform layer for source embdding (which already has sufficient information)
# 		self.f_src_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # only use single tranform layer for source embdding (which already has sufficient information)

# 		self.f_values_1 = nn.Linear(2*ndim_arv_in + ndim_extra + 6, n_hidden) # add second layer transformation.
# 		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

# 		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
# 		self.proj_2 = nn.Linear(n_hidden, ndim_out) # can remove this layer possibly.

# 		self.scale = np.sqrt(n_latent)
# 		self.n_heads = n_heads
# 		self.n_latent = n_latent
# 		self.eps = eps
# 		self.t_kernel_sq = torch.Tensor([eps]).to(device)**2
# 		self.ndim_feat = ndim_arv_in + ndim_extra
# 		self.use_phase_types = use_phase_types
# 		self.n_phases = ndim_out

# 		self.activate1 = nn.PReLU()
# 		self.activate2 = nn.PReLU()
# 		self.activate3 = nn.PReLU()
# 		self.activate4 = nn.PReLU()
# 		# self.activate5 = nn.PReLU()
# 		self.device = device

# 	def forward(self, src, stime, src_embed, trv_src, arrival_p, arrival_s, tpick, ipick, phase_label): # reference k nearest spatial points

# 		# src isn't used. Only trv_src is needed.
# 		n_src, n_sta, n_arv = src.shape[0], trv_src.shape[1], len(tpick) # + 1 ## Note: adding 1 to size of arrivals!
# 		# n_arv = len(tpick)
# 		ip_unique = torch.unique(ipick).float().cpu().detach().numpy() # unique stations
# 		tree_indices = cKDTree(ipick.float().cpu().detach().numpy().reshape(-1,1))
# 		unique_sta_lists = tree_indices.query_ball_point(ip_unique.reshape(-1,1), r = 0)
# 		if self.use_phase_types == False:
# 			phase_label = phase_label*0.0

# 		# arrival_p = torch.cat((arrival_p, torch.zeros(1,arrival_p.shape[1]).to(self.device)), dim = 0) # add null arrival, that all arrivals link too. This acts as a "stabalizer" in the inner-product space, and allows softmax to not blow up for arrivals with only self loops. May not be necessary.
# 		# arrival_s = torch.cat((arrival_s, torch.zeros(1,arrival_s.shape[1]).to(self.device)), dim = 0) # add null arrival, that all arrivals link too. This acts as a "stabalizer" in the inner-product space, and allows softmax to not blow up for arrivals with only self loops. May not be necessary.
# 		arrival = torch.cat((arrival_p, arrival_s), dim = 1) # Concatenate across feature axis

# 		edges = torch.Tensor(np.copy(np.flip(np.hstack([np.ascontiguousarray(np.array(list(zip(itertools.product(unique_sta_lists[j], unique_sta_lists[j]))))[:,0,:].T) for j in range(len(unique_sta_lists))]), axis = 0))).long().to(self.device) # note: preferably could remove loop here.
# 		n_edge = edges.shape[1]

# 		## Now must duplicate edges, for each unique source. (different accumulation points)
# 		edges = (edges.repeat(1, n_src) + torch.cat((torch.zeros(1, n_src*n_edge).to(self.device), (torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(self.device)), dim = 0)).long().contiguous()
# 		src_index = torch.arange(n_src).repeat_interleave(n_edge).contiguous().long().to(self.device)

# 		use_sparse = True
# 		if use_sparse == True:
# 			# pdb.set_trace()
# 			## Find which values have offset times that exceed max time, and ignore these edges (does this work?)
# 			rel_t_p = (tpick[edges[0]] - (trv_src[:,:,0][src_index, ipick[edges[0]]] + stime[src_index])).reshape(-1,1).detach()
# 			rel_t_s = (tpick[edges[0]] - (trv_src[:,:,1][src_index, ipick[edges[0]]] + stime[src_index])).reshape(-1,1).detach()
# 			ikeep = torch.where(((torch.abs(rel_t_p) < 2.5*torch.sqrt(self.t_kernel_sq)) + (torch.abs(rel_t_s) < 2.5*torch.sqrt(self.t_kernel_sq))).reshape(-1) > 0)[0].cpu().detach().numpy() ## Either query is within the threshold amount of time
# 			# edges = edges[:,ikeep]
# 			edges = torch.cat((edges[0][ikeep].reshape(1,-1), edges[1][ikeep].reshape(1,-1)), dim = 0).contiguous()
# 			src_index = src_index[ikeep]		
		
# 		N = n_arv # still correct?
# 		M = n_arv*n_src

# 		if len(src_index) == 0:
# 			return torch.zeros(n_src, n_arv, self.n_phases).to(self.device)
		
# 		out = self.proj_2(self.activate4(self.proj_1(self.propagate(edges, x = arrival, sembed = src_embed, stime = stime, tsrc_p = trv_src[:,:,0], tsrc_s = trv_src[:,:,1], sindex = src_index, stindex = ipick, atime = tpick, phase = phase_label, size = (N, M)).mean(1)))) # M is output. Taking mean over heads

# 		return out.view(n_src, n_arv, -1) ## Make sure this is correct reshape (not transposed)

# 	def message(self, x_j, edge_index, index, tsrc_p, tsrc_s, sembed, sindex, stindex, stime, atime, phase_j): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.

# 		assert(abs(edge_index[1] - index).max().item() == 0)

# 		ifind = torch.where(edge_index[0] == edge_index[0].max())[0]

# 		rel_t_p = (atime[edge_index[0]] - (tsrc_p[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
# 		rel_t_p = torch.cat((torch.exp(-0.5*(rel_t_p**2)/self.t_kernel_sq), torch.sign(rel_t_p), phase_j), dim = 1) # phase[edge_index[0]]

# 		rel_t_s = (atime[edge_index[0]] - (tsrc_s[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
# 		rel_t_s = torch.cat((torch.exp(-0.5*(rel_t_s**2)/self.t_kernel_sq), torch.sign(rel_t_s), phase_j), dim = 1) # phase[edge_index[0]]

# 		# Denote self-links by a feature.
# 		self_link = (edge_index[0] == torch.remainder(edge_index[1], edge_index[0].max().item() + 1)).reshape(-1,1).detach().float() # Each accumulation index (an entry from src cross arrivals). The number of arrivals is edge_index.max() exactly (since tensor is composed of number arrivals + 1)
# 		contexts = self.f_src_context_2(self.activate1(self.f_src_context_1(torch.cat((sembed[sindex], stime[sindex].reshape(-1,1).detach(), self_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)
# 		queries = self.f_arrival_query_2(self.activate2(self.f_arrival_query_1(torch.cat((x_j, rel_t_p, rel_t_s), dim = 1)))).view(-1, self.n_heads, self.n_latent)
# 		values = self.f_values_2(self.activate3(self.f_values_1(torch.cat((x_j, rel_t_p, rel_t_s, self_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)

# 		# When using sparse, this assert is not true
# 		# assert(self_link.sum() == (len(atime) - 1)*tsrc_p.shape[0])

# 		## Do computation
# 		scores = (queries*contexts).sum(-1)/self.scale
# 		alpha = softmax(scores, index)

# 		return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))


# if use_updated_model_definition == False:

class GCN_Detection_Network_extended(nn.Module):
	def __init__(self, ftrns1, ftrns2, scale_rel = scale_rel, scale_time = scale_time, use_absolute_pos = use_absolute_pos, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded, use_embedding = use_embedding, attach_time = attach_time, trv = None, device = 'cuda'):
		super(GCN_Detection_Network_extended, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
		n_dim_extra_inpt = 0 if attach_time == False else 1
		n_dim_extra_feat = 0 if use_embedding == False else 20

		if use_expanded == False:
			self.DataAggregation = DataAggregation(4 + n_dim_extra_inpt + n_dim_extra_feat, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
		else:
			self.DataAggregation = DataAggregationExpanded(4 + n_dim_extra_inpt + n_dim_extra_feat, 15).to(device) # output size is latent size for (half of) bipartite code # , 15				
		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 4).to(device) # 30, 15
		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpaceTimeDirect = SpaceTimeDirect(30, 30).to(device) # 15, 30
		self.SpaceTimeAttention = SpaceTimeAttention(30, 30, 4, 15).to(device)
		self.proj_soln = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, 1))
		# self.TemporalAttention = TemporalAttention(30, 1, 15).to(device)

		self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15).to(device)
		if use_expanded == False:
			self.DataAggregationAssociationPhase = DataAggregationAssociationPhase(15, 15).to(device) # need to add concatenation
		else:
			self.DataAggregationAssociationPhase = DataAggregationAssociationPhaseExpanded(15, 15).to(device) # need to add concatenation

		## Make association module layers
		self.SourceArrivalEmbedding = SourceArrivalEmbedding(30, 30, trv = trv, ftrns2 = ftrns2) ## [note: merging the embeddings for P and S into one (oveloaded) layer rather than keeping as seperate layers?]

		# self.LocalSliceLgCollapseP = LocalSliceLgCollapse(30, 15, device = device).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
		# self.LocalSliceLgCollapseS = LocalSliceLgCollapse(30, 15, device = device).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
		# self.Arrivals = StationSourceAttentionMergedPhases(30, 15, 2, 15, n_heads = 3, device = device).to(device)
		# self.ArrivalS = StationSourceAttention(30, 15, 1, 15, n_heads = 3, device = device).to(device)
		self.Arrivals = StationSourceAttention(30, 15, 2, 15, n_heads = 3, device = device).to(device)

		if use_embedding == True:
			self.DataAggregationEmbedding = DataAggregationEmbedding(1 + n_dim_extra_inpt, int(n_dim_extra_feat/2))

		self.use_absolute_pos = use_absolute_pos
		self.scale_rel = scale_rel
		self.scale_time = scale_time
		self.use_expanded = use_expanded
		self.use_gradient_loss = use_gradient_loss
		self.activate_gradient_loss = False
		self.attach_time = attach_time
		self.use_embedding = use_embedding
		self.device = device

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

		t1 = time.time()

		n_line_nodes = Slice.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)

		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			embedding = self.DataAggregationEmbedding(inpt_embedding, A_in_sta, A_in_src[0], A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, A_in_sta, A_in_src, A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t)
			Slice = torch.cat((Slice, embedding), dim = 1)

		## Now, t_query are the pointwise query times of all x_query_cart queries
		## And there's a new input of the template node times as well, x_temp_cuda_t

		## Should adapt Bipartite Read in to use space-time informtion
		## Should add time information to node features of Cartesian product
		## Or implement as relative time information on edges

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)

		if (self.use_gradient_loss == True)*(self.activate_gradient_loss == True):
			x_temp_cuda = Variable(x_temp_cuda, requires_grad = True)
			x_query_cart = Variable(x_query_cart, requires_grad = True)
			t_query = Variable(t_query, requires_grad = True)

		# print('Time [1] %0.4f'%(time.time() - t1))

		x_latent = self.DataAggregation(Slice, Mask, A_in_sta, A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, A_src, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		

		# print('Time [2] %0.4f'%(time.time() - t1))

		y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		y = self.proj_soln(y_latent)


		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		x_src = self.SpaceTimeAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart, tq_sample, x_temp_cuda_t) # obtain spatial embeddings, source want to query associations for.
		x = self.proj_soln(x)

		# print('Time [3] %0.4f'%(time.time() - t1))

		grad_grid_src, grad_grid_t, grad_query_src, grad_query_t = [], [], [], []
		if (self.use_gradient_loss == True)*(self.activate_gradient_loss == True):
			torch_one_vec = torch.ones(len(x_temp_cuda_cart),1).to(x_temp_cuda_cart.device)
			grad_grid = torch.autograd.grad(inputs = x_temp_cuda, outputs = y, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			grad_grid_src, grad_grid_t = grad_grid[:,0:3], (1000.0*self.scale_time)*grad_grid[:,3]
			torch_one_vec = torch.ones(len(x_query_cart),1).to(x_query_cart.device)
			grad_query_src = torch.autograd.grad(inputs = x_query_cart, outputs = x, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			grad_query_t = torch.autograd.grad(inputs = t_query, outputs = x, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		# x = self.TemporalAttention(x, t_query) # on random queries
		## In LocalSliceLg Collapse should use relative node time information between arrivals and moveouts
		## (it may already be included in relative travel time vectors (e.g., tlatent?))

		## Note below: why detach x_latent?
		mask_out = 1.0*(y.detach() > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		if self.use_absolute_pos == True:
			s = torch.cat((s, locs_use_cart[A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
		# s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"
		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"
		## Remove the detach command?

		# print('Time [4] %0.4f'%(time.time() - t1))

		## Arrival embeddings
		# arv_embed = self.SourceArrivalEmbedding(x, x_context_cart, x_context_t, x_query_cart, x_query_t, A_src_in_sta, degree_srcs, cum_degree_srcs, tpick, ipick, phase_label, tlatent)
		
		## Can compute these degree vectors outside the model
		# degree_srcs = degree(A_src_in_sta[1], num_nodes = len(x_temp_cuda_cart), dtype = torch.long)
		# cum_degree_srcs = torch.cat((torch.zeros(1), torch.cumsum(degree_srcs, dim = 0)[0:-1]), dim = 0)
		## degree_srcs, cum_degree_srcs
		arv_embed, mask_arv = self.SourceArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = trv_out_q)

		# print('Time [5] %0.4f'%(time.time() - t1))

		arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)

		# arv_p = self.LocalSliceLgCollapseP(A_edges_p, dt_partition, tpick, ipick, phase_label, s, tlatent[:,0].reshape(-1,1), n_temp, n_sta) ## arv_p and arv_s will be same size # locs_use_cart, x_temp_cuda_cart, A_src_in_sta
		# arv_s = self.LocalSliceLgCollapseS(A_edges_s, dt_partition, tpick, ipick, phase_label, s, tlatent[:,1].reshape(-1,1), n_temp, n_sta)
		# arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
		
		arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)


		# print('Time [6] %0.4f'%(time.time() - t1))

		# pdb.set_trace()

		if self.use_gradient_loss == False:

			return y, x, arv_p, arv_s

		else:

			return [y, x, arv_p, arv_s], [grad_grid_src, grad_grid_t, grad_query_src, grad_query_t]

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
		self.A_src = A_src[0] if self.use_expanded == True else A_src
		self.A_edges_p = A_edges_p
		self.A_edges_s = A_edges_s
		self.dt_partition = dt_partition
		self.tlatent = tlatent
		# self.pos_rel_sta = pos_rel_sta
		# self.pos_rel_src = pos_rel_src
	
	def forward_fixed(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

		# t1 = time.time()

		n_line_nodes = Slice.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)

		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			embedding = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src[0], self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t)
			Slice = torch.cat((Slice, embedding), dim = 1)

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)

		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		

		y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		y = self.proj_soln(y_latent)


		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		x_src = self.SpaceTimeAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart, tq_sample, x_temp_cuda_t) # obtain spatial embeddings, source want to query associations for.
		x = self.proj_soln(x)


		## Note below: why detach x_latent?
		mask_out = 1.0*(y.detach() > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		if self.use_absolute_pos == True:
			s = torch.cat((s, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
		# s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"
		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, self.A_in_sta, self.A_in_src) # detach x_latent. Just a "reference"


		arv_embed, mask_arv = self.SourceArrivalEmbedding(s, x_temp_cuda_cart, x_temp_cuda_t, x_query_src_cart, tq_sample, self.A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, self.tlatent, trv_out = trv_out_q)


		arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_embed, mask_arv, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)


		arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)


		return y, x, arv_p, arv_s


	def forward_fixed_source(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, t_query, n_reshape = 1):
	
		# t1 = time.time()

		n_line_nodes = Slice.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)

		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			embedding = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src[0], self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) if self.use_expanded == True else self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t)
			Slice = torch.cat((Slice, embedding), dim = 1)

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)

		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		

		y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		y = self.proj_soln(y_latent)


		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		# x_src = self.SpaceTimeAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart, tq_sample, x_temp_cuda_t) # obtain spatial embeddings, source want to query associations for.
		x = self.proj_soln(x)


		if n_reshape > 1: ## Use this to map (n_reshape) repeated spatial queries (x_temp_cuda_cart) at different origin times, to predictions for fixed coordinates and across time
			# y = y.reshape(-1,n_reshape,1) ## Assumed feature dimension output is 1
			x = x.reshape(-1,n_reshape,1)

		return y, x

  
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
			self.reloc_x = nn.Parameter(torch.zeros((n_srcs, 3))).to(device)
			self.reloc_t = nn.Parameter(torch.zeros((n_srcs, 1))).to(device)

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

	# def embed_src(self, src):

	# 	return self.fc3_block(src)

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
				
				# return torch.relu(self.inorm_time(base_val + pred))

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
		# self.bias = nn.Parameter(torch.zeros(locs.shape[0], grid.shape[0], 2), requires_grad = True).to(device)
		self.bias = nn.Parameter(torch.zeros(grid.shape[0], locs.shape[0], 2))
		self.activate = nn.Softplus()
		
		self.grid_save = nn.Parameter(grid, requires_grad = False)

		self.zvec = torch.Tensor([1.0,1.0,0.0]).reshape(1,-1).to(device)

	## Need to double check these routines
	def log_amplitudes(self, ind, src, mag, phase):

		## Input src: n_srcs x 3;
		## ind: indices into absolute locs array (can repeat, for phase types)
		## log_amp (base 10), for each ind
		## phase type for each ind 

		fudge = 1.0 # add before log10, to avoid log10(0)

		# Compute pairwise distances;
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2) + fudge)
		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)) + fudge)

		inds = knn(self.grid_cart/1000.0, self.ftrns1(src)/1000.0, k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
		## Can directly use torch_scatter to coalesce the data

		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)

		# log_amp = mag*torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device)) + self.epicenter_spatial_coef[phase]*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias
		log_amp = mag*torch.maximum(self.activate(self.mag_coef[phase]), torch.Tensor([1e-12]).to(self.device)) - self.activate(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

		return log_amp

	def train(self, ind, src, mag, phase):

		## Input src: n_srcs x 3;
		## ind: indices into absolute locs array (can repeat, for phase types)
		## log_amp (base 10), for each ind
		## phase type for each ind 

		fudge = 1.0 # add before log10, to avoid log10(0)

		# Compute pairwise distances;
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec) - self.ftrns1(self.locs[ind]*self.zvec), dim = 1) + fudge)
		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1) - self.locs[ind,2].view(-1)) + fudge)

		sta_ind = ind.repeat_interleave(self.k)
		inds = knn(self.grid_cart/1000.0, self.ftrns1(src)/1000.0, k = self.k) # [1] # .reshape(-1,self.k) ## for each of the second one, find indices in the first
		## Can directly use torch_scatter to coalesce the data

		# bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)
		bias = self.bias[inds[1], sta_ind, :] # .mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)

		bias = scatter(bias, inds[0], dim = 0, reduce = 'mean')[torch.arange(len(src)).long().to(self.device),phase]

		# log_amp = mag*torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device)) + self.epicenter_spatial_coef[phase]*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias
		log_amp = mag*torch.maximum(self.activate(self.mag_coef[phase]), torch.Tensor([1e-12]).to(self.device)) - self.activate(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

		return log_amp
	
	## Note, closer between amplitudes and forward
	def forward(self, ind, src, log_amp, phase):

		## Input src: n_srcs x 3;
		## ind: indices into absolute locs array (can repeat, for phase types)
		## log_amp (base 10), for each ind
		## phase type for each ind

		fudge = 1.0 # add before log10, to avoid log10(0)

		# Compute pairwise distances;
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2) + fudge)
		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)) + fudge)

		inds = knn(self.grid_cart/1000.0, self.ftrns1(src)/1000.0, k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
		## Can directly use torch_scatter to coalesce the data?

		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)

		# mag = (log_amp - self.epicenter_spatial_coef[phase]*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device))
		mag = (log_amp + self.activate(self.epicenter_spatial_coef[phase])*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/torch.maximum(self.activate(self.mag_coef[phase]), torch.Tensor([1e-12]).to(self.device))

		return mag










# class SourceArrivalEmbedding(MessagePassing):
# 	def __init__(self, ndim_src_in, ndim_out, n_hidden = 30, scale_rel = scale_rel, k_spc_edges = k_spc_edges, kernel_sig_t = kernel_sig_t, use_phase_types = use_phase_types, scale_time = scale_time, min_thresh = 0.01, trv = None, ftrns2 = None, device = device):
# 		super(SourceArrivalEmbedding, self).__init__(node_dim = 0, aggr = 'add') # check node dim. ## Use sum or mean

# 		## Goal of this module is just to implement Bipartite aggregation of each source query - pick pair, of their misfits,
# 		## and while aggregating over the relevant nodes of the (subgraph) Cartesian product
# 		self.ftrns2 = ftrns2
# 		self.trv = trv
# 		self.use_phase_types = use_phase_types
# 		self.kernel_sig_t = kernel_sig_t
# 		self.min_thresh = min_thresh
# 		self.scale_time = scale_time
# 		self.k_spc_edges = k_spc_edges
# 		self.device = device
# 		self.dilate_scale = 1.0
# 		# self.fixed_edges

# 		## ndim_src_in is dimension of features on Cartesian product
# 		## ndim_arv_in is dimension of features on Cartesian product (ndim_arv_in not used currently)


# 	def forward(self, x, x_context_cart, x_context_t, x_query_cart, x_query_t, A_src_in_sta, tpick, ipick, phase_label, locs_use_cart, tlatent, trv_out = None): # reference k nearest spatial points

# 		## Can add fixed edge option for use in SpaceTimeAttentionQuery
# 		# if self.use_fixed_edges == True:
# 		# 	trv_out = self.trv_out_fixed

# 		if trv_out is None:
# 			trv_out = self.trv(self.ftrns2(locs_use_cart), self.ftrns2(x_query_cart)) + x_query_t.reshape(-1, 1, 1) ## Use full travel times, as we check for stations from the full product
# 		else: 
# 			trv_out = trv_out + x_query_t.reshape(-1, 1, 1)

# 		## degree_srcs, cum_degree_srcs

# 		## Note: should also consider using source reciever offset positions.
# 		## Note, can use this feature even for the isolated query node - reciever message (e.g., irrespective of incoming Cartesian product nodes)

# 		st = time.time()

# 		if self.use_phase_types == False:
# 			phase_label = phase_label*0.0

# 		## degree_srcs on cartesian product
# 		## cum_degree_srcs on cartesian product
# 		## tlatent are travel times to the reference nodes of Cartesian product (note: could these bound the pairs that are relevent for a given query?)

# 		# ipick_unique = torch.unique(ipick).long()
# 		i1 = torch.where(phase_label == 0)[0]
# 		i2 = torch.where(phase_label == 1)[0]

# 		misfit_time = torch.zeros((len(x_query_cart), len(tpick), 4)).to(self.device) ## Question: is it necessary to produce these pairwise misfits? Can we focus on the pairs that "likely" have arrival times within threshold (e.g., bound min and max times based on distances between src reciever first, before computing travel times)
# 		misfit_time[:,i1,0] = torch.exp(-0.5*(trv_out[:,ipick[i1],0] - torch.Tensor(tpick[i1]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
# 		misfit_time[:,i2,1] = torch.exp(-0.5*(trv_out[:,ipick[i2],1] - torch.Tensor(tpick[i2]).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
# 		misfit_time[:,:,2] = torch.exp(-0.5*(trv_out[:,ipick,0] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
# 		misfit_time[:,:,3] = torch.exp(-0.5*(trv_out[:,ipick,1] - torch.Tensor(tpick).to(self.device))**2/((self.dilate_scale*self.kernel_sig_t)**2))
		

# 		use_pick_embedding = False
# 		if use_pick_embedding == True:

# 			## Note this is not used

# 			## Determine unique station indices
# 			ipick_unique = np.unique(ipick.cpu().detach().numpy())
# 			tree_stations = cKDTree(ipick.cpu().detach().numpy().reshape(-1,1))
# 			len_ipick_unique = len(ipick_unique)
# 			edges_read_in = tree_stations.query_ball_point(ipick_unique.reshape(-1,1), r = 0)

# 			edges_source = np.hstack([np.array(list(edges_read_in[i])) for i in range(len_ipick_unique)])
# 			edges_trgt = np.hstack([ipick_unique[i]*np.ones(len(edges_read_in[i])) for i in range(len_ipick_unique)])
# 			edges_read_in = torch.Tensor(np.concatenate((edges_source.reshape(1,-1), edges_trgt.reshape(1,-1)), axis = 0)).long().to(self.device)
			

# 			## Note: the point of embedding embed_picks is not clear: since we ultimately want per pick embeddings not per station.
# 			## Though we could concievably obtain per station and query embeddings and then map to the per pick embeddings seperately, if this was more convieient?
# 			## E.g., these embed picks are created here, and then the mask is taken over embed picks, and this may help as the Cartesian product graph
# 			## Entries more easily allow merging over fixed stations.. (e.g., right now we cant easilt idenfity the residual between the pick
# 			## and these possible incoming edges).

# 			# embed_picks = scatter(misfit_time[edges_read_in[0]], edges_read_in[1], dim = 1, dim_size = len(locs_use_cart), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations
# 			embed_picks = scatter(misfit_time[:,edges_read_in[0],:], edges_read_in[1], dim = 1, dim_size = len(locs_use_cart), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations

# 		# embed_picks = scatter(misfit_time[:,edges_read_in[0],:], edges_read_in[1], dim = 1, dim_size = len(ipick), reduce = 'max') ## Note: using broadcasting to duplicate sources over the stations and only aggregation over stations

# 		print('Time %0.4f'%(time.time() - st))

# 		## Can compute these degree vectors outside of loop
# 		degree_srcs = degree(A_src_in_sta[1], num_nodes = len(x_context_cart), dtype = torch.long)
# 		cum_degree_srcs = torch.cat((torch.zeros(1).to(self.device), torch.cumsum(degree_srcs, dim = 0)[0:-1]), dim = 0).long()
# 		## Should check if minimal degree srcs really are accessing nearest stations

# 		print('Time %0.4f'%(time.time() - st))

# 		## Find active source - arrival queries (base it on exact P and S fits, rather than max over the set; is it very different?)
# 		# i1p, i1s = torch.where(misfit_time)
# 		mask_misfit_time = misfit_time.max(2).values > self.min_thresh ## Save this, so can use as mask in the attention layer
# 		isrc, iarv = torch.where(mask_misfit_time == 1)
# 		## For this subset of source - arrivals, now must find the "matches" to entries of the subset of extracted indices from the subgraph Cartesian product (based on queries)

# 		## Build src-src indices (may or may not use the edge feature of source query to source node offsets)
# 		edge_index = knn(torch.cat((x_context_cart/1000.0, self.scale_time*x_context_t.reshape(-1,1)), dim = 1), torch.cat((x_query_cart/1000.0, self.scale_time*x_query_t.reshape(-1,1)), dim = 1), k = self.k_spc_edges).flip(0)
# 		# edge_attr = torch.cat(((x_query[edge_index[1],0:3] - x_context[edge_index[0],0:3])/self.scale_rel, x_query_t[edge_index[1]].reshape(-1,1)/self.scale_time - x_context_t[edge_index[0]].reshape(-1,1)/self.scale_time), dim = 1) # /scale_x

# 		# Build a single flattened arange from size = sum(idx)
# 		deg_slice = degree_srcs[edge_index[0]]
# 		assert(deg_slice.min() > 0) ## This may not work for degree zero nodes (which shouldn't exist on the subgraph? E.g., all source nodes have some connected stations)
# 		inc_inds = torch.arange(deg_slice.sum()).long().to(self.device)
# 		inc_inds = inc_inds - torch.repeat_interleave(torch.cumsum(deg_slice, dim = 0) - deg_slice, deg_slice)
# 		nodes_of_product = cum_degree_srcs[edge_index[0]].repeat_interleave(degree_srcs[edge_index[0]]) + inc_inds
# 		ind_query = torch.arange(len(x_query_cart)).long().to(device).repeat_interleave(scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum'), dim = 0) ## The indices of a fixed query source (is this correct?)

# 		sta_src_pairs = A_src_in_sta[:, nodes_of_product]
# 		## Query_vals is shaped based on nodes_of_product. So when we aggregate or want to extract Cartesian product node features, we can use these.

# 		# k_matches = knn(sta_src_pairs.T, torch.cat((ipick[iarv].reshape(-1,1), ))
# 		query_vals = torch.cat((sta_src_pairs[0].reshape(-1,1), ind_query.reshape(-1,1)), dim = 1).float()
# 		pick_vals = torch.cat((ipick[iarv].reshape(-1,1), isrc.reshape(-1,1)), dim = 1).float()
# 		## The point of query vals is these are the nodes on the Cartesian product we are accessing and aggregating across.
# 		## How can we "read into" these nodes, or match to these nodes, for all possible (> min thresh) pick vals.
# 		## Can we use degrees or cumulative degrees of query vals to directly read in? Can we catch cases where the pick
# 		## has no match (e.g., read in, but then find mis-match of values and remove?)

# 		## Alternatively we could just aggregate the picks and the nodes of the Cartesian product sperately

# 		## Or put another way, we aggregate for all picks using fixed stations, and zero out the incoming messages from
# 		## Cartesian product that do not have matched picks.. 


# 		print('Time %0.4f'%(time.time() - st))

# 		## Possibly k-nn nearest neighbor search isn't necessary - maybe we can manipulate the degree_stas (or subgraph of degree_stas)
# 		## vector to directly "access" the station indices wanted for each pick? Or we can pre-compute these and save (similar to A_edges_p and A_edges_s)
# 		# def hash_rows(val):
# 		# 	## Hash unstable if val[:,0] > 2**32
# 		# 	return val[:,0].to(torch.int64) << 32 | val[:,1].to(torch.int64) ## We actually want to find all of query_vals that have matches to picks
# 		# mask_ind = torch.isin(hash_rows(query_vals), hash_rows(pick_vals)) # set(map(tuple, l1))
# 		hash_picks, hash_queries = hash_rows(pick_vals), hash_rows(query_vals) ## Do not define directly if only using one mask below
# 		mask_picks = torch.isin(hash_picks, hash_queries) # set(map(tuple, l1))
# 		mask_queries = torch.isin(hash_queries, hash_picks) # set(map(tuple, l1))
# 		iwhere_picks = torch.where(mask_picks == 1)[0]
# 		iwhere_query = torch.where(mask_queries == 1)[0]
# 		# assert(torch.abs(query_vals[iwhere_query] - pick_vals[knn(pick_vals, query_vals[iwhere_query], k = 1)[1]]).max() == 0)
# 		# assert(torch.abs(pick_vals[iwhere_picks] - query_vals[knn(query_vals, pick_vals[iwhere_picks], k = 1)[1]]).max() == 0)
# 		## Mask_queries identifies which queries also have a pick.
# 		## Hence, to aggregate for each pick, we want that picks incoming data for the source (exists for all picks > threshold)
# 		## And any of the incoming Cartesian product nodes.

# 		## Based on the set of possible messages: A_src_in_sta[:, nodes_of_product[iwhere_query]] now must create all
# 		## incoming edges for each pick. Unfortunently since there can be multiple picks of the same station not clear
# 		## how to obtain these directly.

# 		## We want to pair these incoming messages with the relavent information of:
# 		## Is it a Cartesian product node or just the base "source" incoming edge
# 		## The relative misfit time between the query itself (for all nodes? Or just for the base node? Probably all nodes)
# 		## And the relative misfit time for any of the Cartesian product nodes (for the base node, for this value, use the exact relative offset time)
# 		## The source - station offset distance (or scaled or unscaled vector). For the base node we have the misfit features already.
# 		## For the Cartesian product nodes we can compute this using tlatent.

# 		## Recall the pick_vals are: (station index x source query index)

# 		## Rather than treat the base node as a fixed incoming message, we can add it as a learned "base" vector
# 		## e.g., a mapping with same output dimension of the scatter over the Cartesian product nodes, and base this
# 		## mapping only on the exact source - reciever data (e.g., misfit or relative offsets). Then an additional
# 		## mapping over this summed (or concatenated) output to obtain the pick embeddings (which should be stable
# 		## even for no incoming messages from the Cartesian product due to the merging layer).

# 		print('Time %0.4f'%(time.time() - st))


# 		## Find matches of the incoming messages of Cartesian product to the picks
# 		# sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
# 		# ind_extract = torch.searchsorted(sorted_hash_picks, hash_queries)
# 		# valid_ind = (ind_extract < len(sorted_hash_picks)) & (sorted_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)] == hash_queries)
# 		# queries_to_pick_inds = order_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)][mask_queries & valid_ind]

# 		sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
# 		ind_extract = torch.searchsorted(sorted_hash_picks, hash_queries[iwhere_query])
# 		valid_ind = (ind_extract < len(sorted_hash_picks)) & (sorted_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)] == hash_queries[iwhere_query])
# 		inds_queries_to_picks = order_hash_picks[ind_extract.clamp(max = len(sorted_hash_picks) - 1)][valid_ind]


# 		pdb.set_trace()


# 		## Compute features
# 		misfit_rel_time = tpick[iarv[inds_queries_to_picks]].reshape(-1,1) - tlatent[nodes_of_product[iwhere_query]]

# 		# pick_embedding = fc_embed()
# 		## We may actually need to use the iwhere_pick to specifically idenfity which 

# 		# trel_misfit = 

# 		# inpt_messages = torch.cat((x[nodes_of_product[iwhere_query]], ))


# 		print('Time %0.4f'%(time.time() - st))

# 		## We want to aggregate data for each source - arrival query pair above threshold (e.g., all pick_vals).
# 		## For some of these, we have "matches" in query_vals (given by iwhere_queries)


# 		pdb.set_trace()

# 		# k_matches = knn(query_vals, pick_vals, k = 1) ## Might be faster with cKDTree
# 		# iwhere = torch.where(torch.abs(query_vals[k_matches[1]] - pick_vals[k_matches[0]]).max(1).values == 0)[0]


# 		# print('Time %0.4f'%(time.time() - st))



# 		## Recall must aggregate over all pick_vals to some extent (> min_thresh) matches between queries and picks.
# 		## The subset satisfying iwhere are those with matched entries in the subgraph Cartesian product (subsets)
# 		## of the queries.


# 		## Now for these subsets of matched pairs we want to match the values of the Cartesian product to relative features between
# 		## these nodes, queries, and the picks.



# 		## The indices of a fixed query source (is this correct?)
# 		# ind_query = torch.arange(len(x_query_cart)).long().to(device).repeat_interleave(scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum'), dim = 0) # .to(device)

# 		# deg_query = scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum')
# 		# ind_query = torch.arange(len(x_query_cart)).long().to(device).repeat_interleave(scatter(deg_slice, edge_index[1], dim = 0, dim_size = len(x_query_cart), reduce = 'sum'), dim = 0) # .to(device)
# 		## src_query_ind = torch.arange()

# 		## Note: for these pairs, they represent fixed source and reciever pairs, for degrees set by each source, and also,
# 		## groups of k-nn sources for each continuous query source. May need to create edge indices based on the query source
# 		## nodes and k-nn neighborhoods.



# 		## Note that nodes_of_product can be used to directly aggregate features from the Cartesian product subgraph.
# 		## This is what we want to do, but we want to concatenate some station, pick, and relative source-reciever information
# 		## into these features before aggregating (and transforming). Note, we still need to account for the zero incoming edge
# 		## case for a given source pick query (with > min_thresh fit). Note that we are not yet using isrc and iarv (the sparse set
# 		## of source reiever queries we are interested in).

# 		## To match the > min_thresh source - pick misfits (embed_picks) to these messages of the Cartesian product,
# 		## must use the sta_src_pairs information and nearest neighbor matching (with distance == 0)



# 		# print('Time %0.4f'%(time.time() - st))

# 		## Now map these incoming edges to the subsets (and repeat interleave based on the degrees of these source nodes, and add increment sequences)
# 		## to identify the nodes of the Cartesian product


# 		# out = self.proj_2(self.activate4(self.proj_1(self.propagate(edges, x = arrival, sembed = src_embed, stime = stime, tsrc_p = torch.cat((trv_src[:,:,0], -self.eps*torch.ones(n_src,1).to(self.device)), dim = 1).detach(), tsrc_s = torch.cat((trv_src[:,:,1], -self.eps*torch.ones(n_src,1).to(self.device)), dim = 1).detach(), sindex = src_index, stindex = torch.cat((ipick, torch.Tensor([n_sta]).long().to(self.device)), dim = 0), atime = torch.cat((tpick, torch.Tensor([-self.eps]).to(self.device)), dim = 0), phase = torch.cat((phase_label, torch.Tensor([-1.0]).reshape(1,1).to(self.device)), dim = 0), size = (N, M)).mean(1)))) # M is output. Taking mean over heads

# 		return out.view(n_src, n_arv, -1) ## Make sure this is correct reshape (not transposed)

# 	def message(self, x_j, edge_index, index, tsrc_p, tsrc_s, sembed, sindex, stindex, stime, atime, phase_j): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.


# 		return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))





