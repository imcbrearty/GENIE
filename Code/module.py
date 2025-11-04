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
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.autograd import Variable
from torch_scatter import scatter
from numpy.matlib import repmat
import itertools
import pathlib
import yaml

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

use_updated_model_definition = config['use_updated_model_definition']
scale_rel = config['scale_rel'] # 30e3
k_sta_edges = config['k_sta_edges']

## Removing scale_t and eps as free variables. Instead set proportionally to kernel_sig_t
# scale_t = config['scale_t'] # 10.0
# eps = config['eps'] # 15.0
scale_t = train_config['kernel_sig_t']*3.0
eps = train_config['kernel_sig_t']*5.0
scale_time = train_config['scale_time']

# use_updated_model_definition = True
use_phase_types = config['use_phase_types']
use_absolute_pos = config['use_absolute_pos']
use_neighbor_assoc_edges = config.get('use_neighbor_assoc_edges', False)
use_expanded = config['use_expanded']
use_gradient_loss = train_config['use_gradient_loss']
use_embedding = config['use_embedding']
attach_time = True

device = torch.device('cuda') ## or use cpu

if use_updated_model_definition == False:

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

else:

	class DataAggregationEdges(MessagePassing): # make equivelent version with sum operations.
		def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_mask = 4, ndim_proj = 3, scale_rel = scale_rel, use_absolute_pos = use_absolute_pos):
			super(DataAggregationEdges, self).__init__('mean') # node dim
			## Use two layers of SageConv.
			if use_absolute_pos == True:
				in_channels = in_channels + 3*2
				
			self.in_channels = in_channels
			self.out_channels = out_channels
			self.n_hidden = n_hidden

	
			self.activate = nn.PReLU() # can extend to each channel
			self.init_trns = nn.Linear(in_channels + n_dim_mask, n_hidden)
	
			self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
			self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask + ndim_proj + 1, n_hidden)
	
			self.l1_t2_1 = nn.Linear(in_channels, n_hidden)
			self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask + ndim_proj + 1, n_hidden)
			self.activate11 = nn.PReLU() # can extend to each channel
			self.activate12 = nn.PReLU() # can extend to each channel
			self.activate1 = nn.PReLU() # can extend to each channel
	
			self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
			self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask + ndim_proj + 1, out_channels)
	
			self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
			self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask + ndim_proj + 1, out_channels)
			self.activate21 = nn.PReLU() # can extend to each channel
			self.activate22 = nn.PReLU() # can extend to each channel
			self.activate2 = nn.PReLU() # can extend to each channel
	
			self.scale_rel = scale_rel
			# self.merge_edges = nn.Sequential(nn.Linear(n_hidden + ndim_proj + 1, n_hidden), nn.PReLU())
			self.pos_rel_sta = None
			self.pos_rel_src = None
	
		def forward(self, tr, mask, A_in_sta, A_in_src): # A_src_in_sta, pos_loc, pos_src
	
			tr = torch.cat((tr, mask), dim = -1)
			tr = self.activate(self.init_trns(tr))
			# mask_in = mask.max(1, keepdims = True)[0]
			
			# embed_sta_edges = self.fproj_edges_sta(pos_loc/1e6)
	
			# pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]] - pos_loc[A_src_in_sta[0][A_in_sta[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
			# pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]] - pos_src[A_src_in_sta[1][A_in_src[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
			# dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
			# dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
			# pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
			# pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)
			
			## Could add binary edge type information to indicate data type
			tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr), message_type = 1), mask), dim = 1)) # could concatenate edge features here, and before.
			tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr), message_type = 2), mask), dim = 1))
			tr = self.activate1(torch.cat((tr1, tr2), dim = 1))
	
			tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), message_type = 1), mask), dim = 1)) # could concatenate edge features here, and before.
			tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), message_type = 2), mask), dim = 1))
			tr = self.activate2(torch.cat((tr1, tr2), dim = 1))
	
			return tr # the new embedding.
	
		def message(self, x_j, message_type = 1):

			if message_type == 1:
			
				return torch.cat((x_j, self.pos_rel_sta), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

			elif message_type == 2:

				return torch.cat((x_j, self.pos_rel_src), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

		# def forward(self, tr, mask, A_in_sta, A_in_src): # A_src_in_sta, pos_loc, pos_src
	
		# 	tr = torch.cat((tr, mask), dim = -1)
		# 	tr = self.activate(self.init_trns(tr))
		# 	mask_in = mask.max(1, keepdims = True)[0]
			
		# 	# embed_sta_edges = self.fproj_edges_sta(pos_loc/1e6)
	
		# 	# pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]] - pos_loc[A_src_in_sta[0][A_in_sta[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		# 	# pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]] - pos_src[A_src_in_sta[1][A_in_src[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		# 	# dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
		# 	# dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
		# 	# pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
		# 	# pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)
			
		# 	## Could add binary edge type information to indicate data type
		# 	tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr), message_type = 1, mask = mask_in), mask), dim = 1)) # could concatenate edge features here, and before.
		# 	tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr), message_type = 2, mask = mask_in), mask), dim = 1))
		# 	tr = self.activate1(torch.cat((tr1, tr2), dim = 1))
	
		# 	tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), message_type = 1, mask = mask_in), mask), dim = 1)) # could concatenate edge features here, and before.
		# 	tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), message_type = 2, mask = mask_in), mask), dim = 1))
		# 	tr = self.activate2(torch.cat((tr1, tr2), dim = 1))
	
		# 	return tr # the new embedding.
	
		# def message(self, x_j, mask_j, message_type = 1):

		# 	if message_type == 1:
			
		# 		return torch.cat((x_j, mask_j*self.pos_rel_sta), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

		# 	elif message_type == 2:

		# 		return torch.cat((x_j, mask_j*self.pos_rel_src), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.
				

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

class SpatialDirect(nn.Module):
	def __init__(self, inpt_dim, out_channels):
		super(SpatialDirect, self).__init__() #  "Max" aggregation.

		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.activate = nn.PReLU()

	def forward(self, inpts):

		return self.activate(self.f_direct(inpts))

class SpaceTimeDirect(nn.Module):
	def __init__(self, inpt_dim, out_channels):
		super(SpaceTimeDirect, self).__init__() #  "Max" aggregation.

		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.activate = nn.PReLU()

	def forward(self, inpts):

		return self.activate(self.f_direct(inpts))

class SpatialAttention(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, n_hidden = 30, n_heads = 5, scale_rel = scale_rel):
		super(SpatialAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
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
		# self.activate3 = nn.PReLU()

	def forward(self, inpts, x_query, x_context, k = 10): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		edge_index = knn(x_context/1000.0, x_query/1000.0, k = k).flip(0)
		edge_attr = (x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel # /scale_x

		return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

	def message(self, x_j, index, edge_attr):

		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

		alpha = softmax(alpha, index)

		return alpha.unsqueeze(-1)*value_embed

class TemporalAttention(MessagePassing): ## Hopefully replace this.
	def __init__(self, inpt_dim, out_channels, n_latent, n_hidden = 30, n_heads = 5, scale_t = scale_t):
		super(TemporalAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.

		self.temporal_query_1 = nn.Linear(1, n_hidden)
		self.temporal_query_2 = nn.Linear(n_hidden, n_heads*n_latent)
		self.f_context_1 = nn.Linear(inpt_dim, n_hidden) # add second layer transformation.
		self.f_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.f_values_1 = nn.Linear(inpt_dim, n_hidden) # add second layer transformation.
		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, out_channels) # can remove this layer possibly.

		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_t = scale_t

		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		self.activate5 = nn.PReLU()

	def forward(self, inpts, t_query):

		context = self.f_context_2(self.activate1(self.f_context_1(inpts))).view(-1, self.n_heads, self.n_latent) # add more non-linear transform here?
		values = self.f_values_2(self.activate2(self.f_values_1(inpts))).view(-1, self.n_heads, self.n_latent) # add more non-linear transform here?
		query = self.temporal_query_2(self.activate3(self.temporal_query_1(t_query/self.scale_t))).view(-1, self.n_heads, self.n_latent) # must repeat t output for all spatial coordinates.

		return self.proj_2(self.activate5(self.proj_1(self.activate4((((context.unsqueeze(1)*query.unsqueeze(0)).sum(-1, keepdims = True)/self.scale)*values.unsqueeze(1)).mean(2))))) # linear.

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

if use_updated_model_definition == False:

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

else:

	class DataAggregationAssociationPhaseEdges(MessagePassing): # make equivelent version with sum operations.
		def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_latent = 30, n_dim_mask = 5, ndim_proj = 3, scale_rel = scale_rel, use_absolute_pos = use_absolute_pos):
			super(DataAggregationAssociationPhaseEdges, self).__init__('mean') # node dim
			## Use two layers of SageConv. Explictly or implicitly?
			if use_absolute_pos == True:
				in_channels = in_channels + 2*3
			
			self.in_channels = in_channels
			self.out_channels = out_channels
			self.n_hidden = n_hidden
	
			self.activate = nn.PReLU() # can extend to each channel
			self.init_trns = nn.Linear(in_channels + n_dim_latent + n_dim_mask, n_hidden)
	
			self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
			self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask + ndim_proj + 1, n_hidden)
	
			self.l1_t2_1 = nn.Linear(n_hidden, n_hidden)
			self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask + ndim_proj + 1, n_hidden)
			self.activate11 = nn.PReLU() # can extend to each channel
			self.activate12 = nn.PReLU() # can extend to each channel
			self.activate1 = nn.PReLU() # can extend to each channel
	
			self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
			self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask + ndim_proj + 1, out_channels)
	
			self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
			self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask + ndim_proj + 1, out_channels)
			self.activate21 = nn.PReLU() # can extend to each channel
			self.activate22 = nn.PReLU() # can extend to each channel
			self.activate2 = nn.PReLU() # can extend to each channel
	
			self.scale_rel = scale_rel
			# self.merge_edges = nn.Sequential(nn.Linear(n_hidden + ndim_proj + 1, n_hidden), nn.PReLU())
			self.pos_rel_sta = None
			self.pos_rel_src = None
	
		def forward(self, tr, latent, mask1, mask2, A_in_sta, A_in_src): # A_src_in_sta, pos_loc, pos_src
	
			mask = torch.cat((mask1, mask2), dim = - 1)
			tr = torch.cat((tr, latent, mask), dim = -1)
			tr = self.activate(self.init_trns(tr)) # should tlatent appear here too? Not on first go..
			# mask_in = mask.max(1, keepdims = True)[0]
			
			# if pos_rel_sta is None:
			# 	pos_rel_sta = self.pos_rel_sta
			# 	pos_rel_src = self.pos_rel_src			
			
			# pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]] - pos_loc[A_src_in_sta[0][A_in_sta[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
			# pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]] - pos_src[A_src_in_sta[1][A_in_src[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
			# dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
			# dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
			# pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
			# pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)	
	
			tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(self.l1_t1_1(tr)), message_type = 1), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
			tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(self.l1_t2_1(tr)), message_type = 2), mask), dim = 1))
			tr = self.activate1(torch.cat((tr1, tr2), dim = 1))
	
			tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), message_type = 1), mask), dim = 1)) # could concatenate edge features here, and before.
			tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), message_type = 2), mask), dim = 1))
			tr = self.activate2(torch.cat((tr1, tr2), dim = 1))
	
			return tr # the new embedding.
	
		def message(self, x_j, message_type = 1):

			if message_type == 1:
			
				return torch.cat((x_j, self.pos_rel_sta), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.
		
			elif message_type == 2:

				return torch.cat((x_j, self.pos_rel_src), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

		# def forward(self, tr, latent, mask1, mask2, A_in_sta, A_in_src): # A_src_in_sta, pos_loc, pos_src
	
		# 	mask = torch.cat((mask1, mask2), dim = - 1)
		# 	tr = torch.cat((tr, latent, mask), dim = -1)
		# 	tr = self.activate(self.init_trns(tr)) # should tlatent appear here too? Not on first go..
		# 	mask_in = mask.max(1, keepdims = True)[0]
			
		# 	# if pos_rel_sta is None:
		# 	# 	pos_rel_sta = self.pos_rel_sta
		# 	# 	pos_rel_src = self.pos_rel_src			
			
		# 	# pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]] - pos_loc[A_src_in_sta[0][A_in_sta[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		# 	# pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]] - pos_src[A_src_in_sta[1][A_in_src[1]]])/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		# 	# dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
		# 	# dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
		# 	# pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
		# 	# pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)	
	
		# 	tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(self.l1_t1_1(tr)), message_type = 1, mask = mask_in), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		# 	tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(self.l1_t2_1(tr)), message_type = 2, mask = mask_in), mask), dim = 1))
		# 	tr = self.activate1(torch.cat((tr1, tr2), dim = 1))
	
		# 	tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), message_type = 1, mask = mask_in), mask), dim = 1)) # could concatenate edge features here, and before.
		# 	tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), message_type = 2, mask = mask_in), mask), dim = 1))
		# 	tr = self.activate2(torch.cat((tr1, tr2), dim = 1))
	
		# 	return tr # the new embedding.
	
		# def message(self, x_j, mask_j, message_type = 1):

		# 	if message_type == 1:
			
		# 		return torch.cat((x_j, mask_j*self.pos_rel_sta), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.
		
		# 	elif message_type == 2:

		# 		return torch.cat((x_j, mask_j*self.pos_rel_src), dim = 1) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

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
			

# class LocalSliceLgCollapse(MessagePassing):
# 	def __init__(self, ndim_in, ndim_out, n_edge = 2, n_hidden = 30, eps = 15.0, device = 'cuda'):
# 		super(LocalSliceLgCollapse, self).__init__('mean') # NOTE: mean here? Or add is more expressive for individual arrivals?
# 		self.fc1 = nn.Linear(ndim_in + n_edge, n_hidden) # non-multi-edge type. Since just collapse on fixed stations, with fixed slice of Xq. (how to find nodes?)
# 		self.fc2 = nn.Linear(n_hidden, ndim_out)
# 		self.activate1 = nn.PReLU()
# 		self.activate2 = nn.PReLU()
# 		self.eps = eps
# 		self.device = device

# 	def forward(self, A_edges, dt_partition, tpick, ipick, phase_label, inpt, tlatent, n_temp, n_sta): # reference k nearest spatial points

# 		## Assert is problem?
# 		k_infer = int(len(A_edges)/(n_sta*len(dt_partition)))
# 		assert(k_infer == 10)
# 		n_arvs, l_dt = len(tpick), len(dt_partition)
# 		N = n_sta*n_temp # Lg graph
# 		M = n_arvs# M is target
# 		dt = dt_partition[1] - dt_partition[0]

# 		t_index = torch.floor((tpick - dt_partition[0])/torch.Tensor([dt]).to(self.device)).long() # index into A_edges, which is each station, each dt_point, each k.
# 		t_index = ((ipick*l_dt*k_infer + t_index*k_infer).view(-1,1) + torch.arange(k_infer).view(1,-1).to(self.device)).reshape(-1).long() # check this

# 		src_index = torch.arange(n_arvs).view(-1,1).repeat(1,k_infer).view(1,-1).to(self.device)
# 		sliced_edges = torch.cat((A_edges[t_index].view(1,-1), src_index), dim = 0).long()
# 		t_rel = tpick[sliced_edges[1]] - tlatent[sliced_edges[0],0] # Breaks here?

# 		ikeep = torch.where(abs(t_rel) < self.eps)[0]
# 		sliced_edges = sliced_edges[:,ikeep] # only use times within range. (need to specify target node cardinality)
# 		out = self.activate2(self.fc2(self.propagate(sliced_edges, x = inpt, pos = (tlatent, tpick.view(-1,1)), phase = phase_label, size = (N, M))))

# 		return out

# 	def message(self, x_j, pos_i, pos_j, phase_i):

# 		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.eps, phase_i), dim = -1))) # note scaling of relative time

## This version uses spatial coordianates during arrival embedding
# class LocalSliceLgCollapse(MessagePassing):
# 	def __init__(self, ndim_in, ndim_out, n_edge = 2, n_hidden = 30, n_dim_edges = 10, eps = eps, scale_rel = scale_rel, use_phase_types = use_phase_types, device = 'cuda'):
# 		super(LocalSliceLgCollapse, self).__init__('mean') # NOTE: mean here? Or add is more expressive for individual arrivals?
# 		self.fc1 = nn.Linear(ndim_in + n_edge + n_dim_edges, n_hidden) # non-multi-edge type. Since just collapse on fixed stations, with fixed slice of Xq. (how to find nodes?)
# 		self.fc2 = nn.Linear(n_hidden, ndim_out)
# 		self.embed_edges = nn.Sequential(nn.Linear(3, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_dim_edges))
# 		self.activate1 = nn.PReLU()
# 		self.activate2 = nn.PReLU()
# 		self.eps = eps
# 		self.scale_rel = scale_rel
# 		self.device = device
# 		self.use_phase_types = use_phase_types

# 	def forward(self, A_edges, dt_partition, tpick, ipick, phase_label, inpt, tlatent, locs_cart, x_temp_cart, A_src_in_sta, n_temp, n_sta, k_infer = 10): # reference k nearest spatial points

# 		## Assert is problem?
# 		# k_infer = int(len(A_edges)/(n_sta*len(dt_partition)))
# 		assert(k_infer == 10)
# 		n_arvs, l_dt = len(tpick), len(dt_partition)
# 		N = inpt.shape[0] # Lg graph
# 		M = n_arvs # M is target
# 		dt = dt_partition[1] - dt_partition[0]
# 		if self.use_phase_types == False:
# 			phase_label = phase_label*0.0

# 		t_index = torch.floor((tpick - dt_partition[0])/torch.Tensor([dt]).to(self.device)).long() # index into A_edges, which is each station, each dt_point, each k.
# 		t_index = ((ipick*l_dt*k_infer + t_index*k_infer).view(-1,1) + torch.arange(k_infer).view(1,-1).to(self.device)).reshape(-1).long() # check this

# 		src_index = torch.arange(n_arvs).view(-1,1).repeat(1,k_infer).view(1,-1).to(self.device)

# 		sliced_edges = torch.cat((A_edges[t_index].reshape(1,-1), src_index), dim = 0).contiguous().long()

# 		t_rel = tpick[sliced_edges[1]] - tlatent[sliced_edges[0],0] # Breaks here?

# 		if len(t_rel) > 0:
# 			ikeep = torch.where(abs(t_rel) < 2.0*self.eps)[0] ## Add a larger window for this time embedding
# 		else:
# 			ikeep = torch.Tensor([]).long().to(device)

# 		sliced_edges = sliced_edges[:,ikeep] # only use times within range. (need to specify target node cardinality)
# 		edge_offset = self.embed_edges((locs_cart[A_src_in_sta[0][sliced_edges[0]]] - x_temp_cart[A_src_in_sta[0][sliced_edges[0]]])/(5.0*self.scale_rel))
		
# 		out = self.activate2(self.fc2(self.propagate(sliced_edges, x = inpt, pos = (tlatent, tpick.view(-1,1)), edge_attr = edge_offset, phase = phase_label, size = (N, M))))

# 		return out

# 	## Adding back after removed
# 	def message(self, x_j, pos_i, pos_j, phase_i, edge_attr):

# 		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.eps, edge_attr, phase_i), dim = -1))) # note scaling of relative time
		
class LocalSliceLgCollapse(MessagePassing):
	def __init__(self, ndim_in, ndim_out, n_edge = 2, n_hidden = 30, eps = eps, use_phase_types = use_phase_types, device = 'cuda'): # n_dim_edges = 10
		super(LocalSliceLgCollapse, self).__init__('mean') # NOTE: mean here? Or add is more expressive for individual arrivals?
		self.fc1 = nn.Linear(ndim_in + n_edge, n_hidden) # non-multi-edge type. Since just collapse on fixed stations, with fixed slice of Xq. (how to find nodes?)
		self.fc2 = nn.Linear(n_hidden, ndim_out)
		# self.embed_edges = nn.Sequential(nn.Linear(3, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_dim_edges))
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.eps = eps
		# self.scale_rel = scale_rel
		self.device = device
		self.use_phase_types = use_phase_types

	def forward(self, A_edges, dt_partition, tpick, ipick, phase_label, inpt, tlatent, n_temp, n_sta, k_infer = 10): # reference k nearest spatial points

		## Assert is problem?
		# k_infer = int(len(A_edges)/(n_sta*len(dt_partition)))
		assert(k_infer == 10)
		n_arvs, l_dt = len(tpick), len(dt_partition)
		N = inpt.shape[0] # Lg graph
		M = n_arvs # M is target
		dt = dt_partition[1] - dt_partition[0]
		if self.use_phase_types == False:
			phase_label = phase_label*0.0

		t_index = torch.floor((tpick - dt_partition[0])/torch.Tensor([dt]).to(self.device)).long() # index into A_edges, which is each station, each dt_point, each k.
		t_index = ((ipick*l_dt*k_infer + t_index*k_infer).view(-1,1) + torch.arange(k_infer).view(1,-1).to(self.device)).reshape(-1).long() # check this

		src_index = torch.arange(n_arvs).view(-1,1).repeat(1,k_infer).view(1,-1).to(self.device)

		sliced_edges = torch.cat((A_edges[t_index].reshape(1,-1), src_index), dim = 0).contiguous().long()

		t_rel = tpick[sliced_edges[1]] - tlatent[sliced_edges[0],0] # Breaks here?

		if len(t_rel) > 0:
			ikeep = torch.where(abs(t_rel) < 2.0*self.eps)[0] ## Add a larger window for this time embedding
		else:
			ikeep = torch.Tensor([]).long().to(device)

		sliced_edges = sliced_edges[:,ikeep] # only use times within range. (need to specify target node cardinality)
		# edge_offset = self.embed_edges((locs_cart[A_src_in_sta[0][sliced_edges[0]]] - x_temp_cart[A_src_in_sta[0][sliced_edges[0]]])/(5.0*self.scale_rel))
		
		out = self.activate2(self.fc2(self.propagate(sliced_edges, x = inpt, pos = (tlatent, tpick.view(-1,1)), phase = phase_label, size = (N, M))))

		return out

	## Adding back after removed
	def message(self, x_j, pos_i, pos_j, phase_i):

		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.eps, phase_i), dim = -1))) # note scaling of relative time
		

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
	def __init__(self, ftrns1, ftrns2, scale_rel = scale_rel, scale_time = scale_time, use_absolute_pos = use_absolute_pos, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded, use_embedding = use_embedding, attach_time = attach_time, device = 'cuda'):
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
		self.LocalSliceLgCollapseP = LocalSliceLgCollapse(30, 15, device = device).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
		self.LocalSliceLgCollapseS = LocalSliceLgCollapse(30, 15, device = device).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
		self.Arrivals = StationSourceAttentionMergedPhases(30, 15, 2, 15, n_heads = 3, device = device).to(device)
		# self.ArrivalS = StationSourceAttention(30, 15, 1, 15, n_heads = 3).to(device)

		if use_embedding == True:
			self.DataAggregationEmbedding = DataAggregationEmbedding(1 + n_dim_extra_inpt, int(n_dim_extra_feat/2))

		self.use_absolute_pos = use_absolute_pos
		self.scale_rel = scale_rel
		self.scale_time = scale_time
		self.use_expanded = use_expanded
		self.use_gradient_loss = use_gradient_loss
		self.attach_time = attach_time
		self.use_embedding = use_embedding
		self.device = device

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

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

		if self.use_gradient_loss == True:
			x_temp_cuda = Variable(x_temp_cuda, requires_grad = True)
			x_query_cart = Variable(x_query_cart, requires_grad = True)
			t_query = Variable(t_query, requires_grad = True)

		x_latent = self.DataAggregation(Slice, Mask, A_in_sta, A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, A_src, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		
		# if self.use_gradient_loss == True:
		#	x_spatial = Variable(x_spatial, requires_grad = True)

		y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		y = self.proj_soln(y_latent)

		# y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid

		# if self

		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		x_src = self.SpaceTimeAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart, tq_sample, x_temp_cuda_t) # obtain spatial embeddings, source want to query associations for.
		x = self.proj_soln(x)


		if self.use_gradient_loss == True:

			torch_one_vec = torch.ones(len(x_temp_cuda_cart),1).to(x_temp_cuda_cart.device)
			grad_grid = torch.autograd.grad(inputs = x_temp_cuda, outputs = y, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			# grad_grid_t = torch.autograd.grad(inputs = t_query, outputs = x, grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			# grad_grid_src, grad_grid_t = grad_grid[:,0:3], grad_grid[:,3]/(1000.0*self.scale_time)
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
		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"
		arv_p = self.LocalSliceLgCollapseP(A_edges_p, dt_partition, tpick, ipick, phase_label, s, tlatent[:,0].reshape(-1,1), n_temp, n_sta) ## arv_p and arv_s will be same size # locs_use_cart, x_temp_cuda_cart, A_src_in_sta
		arv_s = self.LocalSliceLgCollapseS(A_edges_s, dt_partition, tpick, ipick, phase_label, s, tlatent[:,1].reshape(-1,1), n_temp, n_sta)
		arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
		
		arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)

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
		self.A_src = A_src
		self.A_edges_p = A_edges_p
		self.A_edges_s = A_edges_s
		self.dt_partition = dt_partition
		self.tlatent = tlatent
		# self.pos_rel_sta = pos_rel_sta
		# self.pos_rel_src = pos_rel_src
	
	def forward_fixed(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

		n_line_nodes = Slice.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)


		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)


		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			embedding = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) # tr, A_in_sta, A_in_src, A_src_in_sta, pos_loc, pos_src, pos_src_t
			Slice = torch.cat((Slice, embedding), dim = 1)


		## Now, t_query are the pointwise query times of all x_query_cart queries
		## And there's a new input of the template node times as well, x_temp_cuda_t

		## Should adapt Bipartite Read in to use space-time informtion
		## Should add time information to node features of Cartesian product
		## Or implement as relative time information on edges


		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)

		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		y = self.proj_soln(y_latent)

		# y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		x_src = self.SpaceTimeAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart, tq_sample, x_temp_cuda_t) # obtain spatial embeddings, source want to query associations for.
		x = self.proj_soln(x)

		# x = self.TemporalAttention(x, t_query) # on random queries
		## In LocalSliceLg Collapse should use relative node time information between arrivals and moveouts
		## (it may already be included in relative travel time vectors (e.g., tlatent?))

		## Note below: why detach x_latent?
		mask_out = 1.0*(y.detach() > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		if self.use_absolute_pos == True:
			s = torch.cat((s, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, self.A_in_sta, self.A_in_src) # detach x_latent. Just a "reference"
		arv_p = self.LocalSliceLgCollapseP(self.A_edges_p, self.dt_partition, tpick, ipick, phase_label, s, self.tlatent[:,0].reshape(-1,1), n_temp, n_sta) ## arv_p and arv_s will be same size # locs_use_cart, x_temp_cuda_cart, A_src_in_sta
		arv_s = self.LocalSliceLgCollapseS(self.A_edges_s, self.dt_partition, tpick, ipick, phase_label, s, self.tlatent[:,1].reshape(-1,1), n_temp, n_sta)
		arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
		
		arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)


		return y, x, arv_p, arv_s

	def forward_fixed_source(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t, x_query_cart, t_query, n_reshape = 1):
	
		n_line_nodes = Slice.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
		if self.use_absolute_pos == True:
			Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)


		if self.attach_time == True:
			Slice = torch.cat((Slice, x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1)


		if self.use_embedding == True:
			inpt_embedding = torch.cat((torch.ones(len(Slice),1).to(Slice.device),  x_temp_cuda_t[self.A_src_in_sta[1]].reshape(-1,1)/self.scale_time), dim = 1) if self.attach_time == True else torch.ones(len(Slice),1).to(Slice.device)
			embedding = self.DataAggregationEmbedding(inpt_embedding, self.A_in_sta, self.A_in_src, self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t) # tr, A_in_sta, A_in_src, A_src_in_sta, pos_loc, pos_src, pos_src_t
			Slice = torch.cat((Slice, embedding), dim = 1)


		## Now, t_query are the pointwise query times of all x_query_cart queries
		## And there's a new input of the template node times as well, x_temp_cuda_t

		## Should adapt Bipartite Read in to use space-time informtion
		## Should add time information to node features of Cartesian product
		## Or implement as relative time information on edges

		x_temp_cuda = torch.cat((x_temp_cuda_cart, 1000.0*self.scale_time*x_temp_cuda_t.reshape(-1,1)), dim = 1)

		x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda) # x_temp_cuda_cart
		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda)
		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		y_latent = self.SpaceTimeDirect(x_spatial) # contains data on spatial and temporal solution at fixed nodes
		y = self.proj_soln(y_latent)

		# y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
		x = self.SpaceTimeAttention(x_spatial, x_query_cart, x_temp_cuda_cart, t_query, x_temp_cuda_t) # second slowest module (could use this embedding to seed source source attention vector).
		# x_src = self.SpaceTimeAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart, tq_sample, x_temp_cuda_t) # obtain spatial embeddings, source want to query associations for.
		x = self.proj_soln(x)

		if n_reshape > 1: ## Use this to map (n_reshape) repeated spatial queries (x_temp_cuda_cart) at different origin times, to predictions for fixed coordinates and across time
			# y = y.reshape(-1,n_reshape,1) ## Assumed feature dimension output is 1
			x = x.reshape(-1,n_reshape,1)

		return y, x

# elif use_updated_model_definition == True:

# 	class GCN_Detection_Network_extended(nn.Module):
# 		def __init__(self, ftrns1, ftrns2, scale_rel = scale_rel, use_absolute_pos = use_absolute_pos, device = 'cuda'):
# 			super(GCN_Detection_Network_extended, self).__init__()
# 			# Define modules and other relavent fixed objects (scaling coefficients.)
# 			# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
# 			self.DataAggregation = DataAggregationEdges(4, 15, use_absolute_pos = use_absolute_pos).to(device) # output size is latent size for (half of) bipartite code # , 15
# 			self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 3).to(device) # 30, 15
# 			self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
# 			self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
# 			self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
# 			self.SpatialDirect = SpatialDirect(30, 30).to(device) # 15, 30
# 			self.SpatialAttention = SpatialAttention(30, 30, 3, 15).to(device)
# 			self.TemporalAttention = TemporalAttention(30, 1, 15).to(device)
	
# 			self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15).to(device)
# 			self.DataAggregationAssociationPhase = DataAggregationAssociationPhaseEdges(15, 15, use_absolute_pos = use_absolute_pos).to(device) # need to add concatenation
# 			self.LocalSliceLgCollapseP = LocalSliceLgCollapse(30, 15, device = device).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
# 			self.LocalSliceLgCollapseS = LocalSliceLgCollapse(30, 15, device = device).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
# 			self.Arrivals = StationSourceAttentionMergedPhases(30, 15, 2, 15, n_heads = 3, device = device).to(device)
# 			self.use_absolute_pos = use_absolute_pos
# 			self.scale_rel = scale_rel # self.DataAggregation.scale_rel
# 			# self.ArrivalS = StationSourceAttention(30, 15, 1, 15, n_heads = 3).to(device)
	
# 			self.ftrns1 = ftrns1
# 			self.ftrns2 = ftrns2
	
# 		def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):
	
# 			n_line_nodes = Slice.shape[0]
# 			mask_p_thresh = 0.01
# 			n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]
			
# 			if self.use_absolute_pos == True:
# 				Slice = torch.cat((Slice, locs_use_cart[A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)

# 			pos_rel_sta = (locs_use_cart[A_src_in_sta[0][A_in_sta[0]]] - locs_use_cart[A_src_in_sta[0][A_in_sta[1]]]) # /self.scale_rel # self.DataAggregation.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
# 			pos_rel_src = (x_temp_cuda_cart[A_src_in_sta[1][A_in_src[0]]] - x_temp_cuda_cart[A_src_in_sta[1][A_in_src[1]]]) # /self.scale_rel # self.DataAggregation.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
# 			dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
# 			dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
# 			pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
# 			pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)
# 			## Embed edge features
# 			pos_rel_sta = torch.sign(pos_rel_sta)*torch.exp(-0.5*(pos_rel_sta**2)/(self.scale_rel**2))
# 			pos_rel_src = torch.sign(pos_rel_src)*torch.exp(-0.5*(pos_rel_src**2)/(self.scale_rel**2))

# 			self.DataAggregation.pos_rel_sta = pos_rel_sta
# 			self.DataAggregation.pos_rel_src = pos_rel_src
# 			self.DataAggregationAssociationPhase.pos_rel_sta = pos_rel_sta
# 			self.DataAggregationAssociationPhase.pos_rel_src = pos_rel_src
			
# 			x_latent = self.DataAggregation(Slice, Mask, A_in_sta, A_in_src) # A_src_in_sta, locs_use_cart, x_temp_cuda_cart # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
# 			x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, n_sta, n_temp)
# 			x = self.SpatialAggregation1(x, A_src, x_temp_cuda_cart)
# 			x = self.SpatialAggregation2(x, A_src, x_temp_cuda_cart)
# 			x_spatial = self.SpatialAggregation3(x, A_src, x_temp_cuda_cart) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
# 			y_latent = self.SpatialDirect(x_spatial) # contains data on spatial solution.
# 			y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
# 			x = self.SpatialAttention(x_spatial, x_query_cart, x_temp_cuda_cart) # second slowest module (could use this embedding to seed source source attention vector).
# 			x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
# 			x = self.TemporalAttention(x, t_query) # on random queries
	
# 			## Note below: why detach x_latent?
# 			mask_out = 1.0*(y[:,:,0].detach().max(1, keepdims = True)[0] > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
# 			s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
# 			if self.use_absolute_pos == True:
# 				s = torch.cat((s, locs_use_cart[A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
			
# 			s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # A_src_in_sta, locs_use_cart, x_temp_cuda_cart # detach x_latent. Just a "reference"
# 			arv_p = self.LocalSliceLgCollapseP(A_edges_p, dt_partition, tpick, ipick, phase_label, s, tlatent[:,0].reshape(-1,1), n_temp, n_sta) ## arv_p and arv_s will be same size
# 			arv_s = self.LocalSliceLgCollapseS(A_edges_s, dt_partition, tpick, ipick, phase_label, s, tlatent[:,1].reshape(-1,1), n_temp, n_sta)
# 			arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			
# 			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
	
# 			return y, x, arv_p, arv_s
	
# 		def set_adjacencies(self, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src_in_sta, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, pos_loc, pos_src):

# 			pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]] - pos_loc[A_src_in_sta[0][A_in_sta[1]]]) # /self.scale_rel # self.DataAggregation.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
# 			pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]] - pos_src[A_src_in_sta[1][A_in_src[1]]]) # /self.scale_rel # self.DataAggregation.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
# 			dist_rel_sta = torch.norm(pos_rel_sta, dim = 1, keepdim = True)
# 			dist_rel_src = torch.norm(pos_rel_src, dim = 1, keepdim = True)
# 			pos_rel_sta = torch.cat((pos_rel_sta, dist_rel_sta), dim = 1)
# 			pos_rel_src = torch.cat((pos_rel_src, dist_rel_src), dim = 1)

# 			## Embed edge features
# 			pos_rel_sta = torch.sign(pos_rel_sta)*torch.exp(-0.5*(pos_rel_sta**2)/(self.scale_rel**2))
# 			pos_rel_src = torch.sign(pos_rel_src)*torch.exp(-0.5*(pos_rel_src**2)/(self.scale_rel**2))
			
# 			self.A_in_sta = A_in_sta
# 			self.A_in_src = A_in_src
# 			self.A_src_in_edges = A_src_in_edges
# 			self.A_Lg_in_src = A_Lg_in_src
# 			self.A_src_in_sta = A_src_in_sta
# 			self.A_src = A_src
# 			self.A_edges_p = A_edges_p
# 			self.A_edges_s = A_edges_s
# 			self.dt_partition = dt_partition
# 			self.tlatent = tlatent
# 			self.DataAggregation.pos_rel_sta = pos_rel_sta
# 			self.DataAggregation.pos_rel_src = pos_rel_src
# 			self.DataAggregationAssociationPhase.pos_rel_sta = pos_rel_sta
# 			self.DataAggregationAssociationPhase.pos_rel_src = pos_rel_src
		
# 		def forward_fixed(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):
	
# 			n_line_nodes = Slice.shape[0]
# 			mask_p_thresh = 0.01
# 			n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]

# 			if self.use_absolute_pos == True:
# 				Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)		
			
# 			# x_temp_cuda_cart = self.ftrns1(x_temp_cuda)
# 			# x = self.TemporalConvolve(Slice).view(n_line_nodes,-1) # slowest module
# 			x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
# 			x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
# 			x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda_cart)
# 			x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda_cart)
# 			x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda_cart) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
# 			y_latent = self.SpatialDirect(x_spatial) # contains data on spatial solution.
# 			y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
# 			x = self.SpatialAttention(x_spatial, x_query_cart, x_temp_cuda_cart) # second slowest module (could use this embedding to seed source source attention vector).
# 			x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
# 			x = self.TemporalAttention(x, t_query) # on random queries
	
# 			## Note below: why detach x_latent?
# 			mask_out = 1.0*(y[:,:,0].detach().max(1, keepdims = True)[0] > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
# 			s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
# 			if self.use_absolute_pos == True:
# 				s = torch.cat((s, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
			
# 			s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, self.A_in_sta, self.A_in_src) # self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart # detach x_latent. Just a "reference"
# 			arv_p = self.LocalSliceLgCollapseP(self.A_edges_p, self.dt_partition, tpick, ipick, phase_label, s, self.tlatent[:,0].reshape(-1,1), n_temp, n_sta)
# 			arv_s = self.LocalSliceLgCollapseS(self.A_edges_s, self.dt_partition, tpick, ipick, phase_label, s, self.tlatent[:,1].reshape(-1,1), n_temp, n_sta)
# 			arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, locs_use_cart, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
			
# 			arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)
	
# 			return y, x, arv_p, arv_s

# 		def forward_fixed_source(self, Slice, Mask, tpick, ipick, phase_label, locs_use_cart, x_temp_cuda_cart, x_query_cart, t_query):
	
# 			n_line_nodes = Slice.shape[0]
# 			mask_p_thresh = 0.01
# 			n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use_cart.shape[0]

# 			if self.use_absolute_pos == True:
# 				Slice = torch.cat((Slice, locs_use_cart[self.A_src_in_sta[0]]/(3.0*self.scale_rel), x_temp_cuda_cart[self.A_src_in_sta[1]]/(3.0*self.scale_rel)), dim = 1)
			
# 			# x_temp_cuda_cart = self.ftrns1(x_temp_cuda)
# 			# x = self.TemporalConvolve(Slice).view(n_line_nodes,-1) # slowest module
# 			x_latent = self.DataAggregation(Slice, Mask, self.A_in_sta, self.A_in_src) # self.A_src_in_sta, locs_use_cart, x_temp_cuda_cart # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
# 			x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, n_sta, n_temp)
# 			x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda_cart)
# 			x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda_cart)
# 			x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda_cart) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
# 			y_latent = self.SpatialDirect(x_spatial) # contains data on spatial solution.
# 			y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
# 			x = self.SpatialAttention(x_spatial, x_query_cart, x_temp_cuda_cart) # second slowest module (could use this embedding to seed source source attention vector).
# 			x = self.TemporalAttention(x, t_query) # on random queries
	
# 			return y, x

  
#### EXTRA
class TravelTimes(nn.Module):

	def __init__(self, ftrns1, ftrns2, n_phases = 1, scale_val = 1e6, trav_val = 200.0, device = 'cuda'):
		super(TravelTimes, self).__init__()

		## Relative offset prediction [2]
		self.fc1 = nn.Sequential(nn.Linear(3, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Absolute position prediction [3]
		self.fc2 = nn.Sequential(nn.Linear(6, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Relative offset prediction [2]
		self.fc3 = nn.Sequential(nn.Linear(3, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Absolute position prediction [3]
		self.fc4 = nn.Sequential(nn.Linear(6, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Projection functions
		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2
		self.scale = torch.Tensor([scale_val]).to(device) ## Might want to scale inputs before converting to Tensor
		self.tscale = torch.Tensor([trav_val]).to(device)
		self.device = device
		# self.Tp_average

	def forward(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return self.tscale*(self.fc1(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + self.fc2(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return self.tscale*(self.fc1(sta_repeat - src_repeat) + self.fc2(torch.cat((sta_repeat, src_repeat)), dim = 1)).reshape(len(src), len(sta), -1)

	def forward_train(self, sta, src, p = 0.5, method = 'pairs'):

		## In training mode, drop p percent of the `local' corrections, so that the default source-reciever distance
		## model is trained to be accurate enough in a stand-alone fashion. This way, travel time estimates can still
		## be obtained for non-sampled source regions, which can help with the global application and simulated data
		## in non-sampeled regions.

		if method == 'direct':

			rand_mask = (torch.rand(sta.shape[0],1).to(self.device) > p).float()

			return self.tscale*(self.fc1(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + rand_mask*self.fc2(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			rand_mask = (torch.rand(sta_repeat.shape[0],1).to(self.device) > p).float()

			return self.tscale*(self.fc1(sta_repeat - src_repeat) + rand_mask*self.fc2(torch.cat((sta_repeat, src_repeat), dim = 1))).reshape(len(src), len(sta), -1)

	def forward_relative(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return self.tscale*(self.fc1(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return self.tscale*(self.fc1(sta_repeat - src_repeat)).reshape(len(src), len(sta), -1)

	def forward_mask(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return torch.sigmoid(self.fc3(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + self.fc4(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return torch.sigmoid(self.fc3(sta_repeat - src_repeat) + self.fc4(torch.cat((sta_repeat, src_repeat)), dim = 1)).reshape(len(src), len(sta), -1)

	def forward_mask_train(self, sta, src, p = 0.5, method = 'pairs'):

		## In training mode, drop p percent of the `local' corrections, so that the default source-reciever distance
		## model is trained to be accurate enough in a stand-alone fashion. This way, travel time estimates can still
		## be obtained for non-sampled source regions, which can help with the global application and simulated data
		## in non-sampeled regions.

		if method == 'direct':

			rand_mask = (torch.rand(sta.shape[0],1).to(self.device) > p).float()

			return torch.sigmoid(self.fc3(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + rand_mask*self.fc4(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			rand_mask = (torch.rand(sta_repeat.shape[0],1).to(self.device) > p).float()

			return torch.sigmoid(self.fc3(sta_repeat - src_repeat) + rand_mask*self.fc4(torch.cat((sta_repeat, src_repeat), dim = 1))).reshape(len(src), len(sta), -1)

	def forward_mask_relative(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return torch.sigmoid(self.fc3(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return torch.sigmoid(self.fc3(sta_repeat - src_repeat)).reshape(len(src), len(sta), -1)


# class VModel(nn.Module):

# 	def __init__(self, n_phases = 2, n_hidden = 50, n_embed = 10, device = 'cuda'): # v_mean = np.array([6500.0, 3400.0]), norm_pos = None, inorm_pos = None, inorm_time = None, norm_vel = None, conversion_factor = None, 
# 		super(VModel, self).__init__()

# 		## Relative offset prediction [2]
# 		self.fc1_1 = nn.Linear(3 + n_embed, n_hidden)
# 		self.fc1_2 = nn.Linear(n_hidden, n_hidden)
# 		self.fc1_3 = nn.Linear(n_hidden, n_hidden)
# 		self.fc1_41 = nn.Linear(n_hidden, 1)
# 		self.fc1_42 = nn.Linear(n_hidden, 1)
# 		self.activate1_1 = lambda x: torch.sin(x)
# 		self.activate1_2 = lambda x: torch.sin(x)
# 		self.activate1_3 = lambda x: torch.sin(x)
# 		self.activate = nn.Softplus()
# 		self.mask = torch.zeros((1, 3)).to(device) # + n_embed)).to(device)
# 		self.mask[0,2] = 1.0

# 		# ## Projection functions
# 		# self.ftrns1 = ftrns1
# 		# self.ftrns2 = ftrns2
# 		# # self.scale = torch.Tensor([scale_val]).to(device) ## Might want to scale inputs before converting to Tensor
# 		# # self.tscale = torch.Tensor([trav_val]).to(device)
# 		# self.v_mean = torch.Tensor(v_mean).to(device)
# 		# self.v_mean_norm = torch.Tensor(norm_vel(v_mean)).to(device)
# 		# self.device = device
# 		# self.norm_pos = norm_pos
# 		# self.inorm_pos = inorm_pos
# 		# self.inorm_time = inorm_time
# 		# self.norm_vel = norm_vel
# 		# self.conversion_factor = conversion_factor
# 		# self.Tp_average

# 	def fc1_block(self, x):

# 		# x = x*torch.Tensor([0.0, 0.0, 1.0]).reshape(1,-1).to(x.device)
# 		x1 = self.activate1_1(self.fc1_1(x))
# 		x = self.activate1_2(self.fc1_2(x1)) + x1
# 		x1 = self.activate1_3(self.fc1_3(x)) + x

# 		return self.activate(self.fc1_41(x1)), self.activate(self.fc1_42(x1))

# 	def forward(self, src, embed):

# 		out1, out2 = self.fc1_block(torch.cat((src, embed), dim = 1))
# 		out2 = out1*out2
# 		# out[:,1] = out[:,0]*out[:,1] ## Vs is a fraction of Vp

# 		return torch.cat((out1, out2), dim = 1)

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














