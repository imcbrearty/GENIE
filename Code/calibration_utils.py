import numpy as np
import torch
from scipy.spatial import cKDTree
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
import scipy
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import gamma, beta
import time
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from numpy.matlib import repmat
import networkx as nx
import cvxpy as cp
import itertools
import pathlib
import yaml

def create_laplacian(x_grid, ftrns1, k = 25):

	A_src_src = knn(torch.Tensor(ftrns1(x_grid)/1000.0), torch.Tensor(ftrns1(x_grid)/1000.0), k = k).flip(0).long().contiguous() # )[0]
	lap = get_laplacian(A_src_src, normalization = 'rw')

	return lap

class Laplacian(MessagePassing):

	def __init__(self, lap, lap_w):
		super(Laplacian, self).__init__('sum', node_dim = 0) # consider mean
		self.lap = lap # edges
		self.lap_w = lap_w.reshape(-1,1) # edge weights

	def forward(self, x):

		## Assumes x is a square matrix
		## (what about batch?, e.g. multiple stations)

		return self.propagate(self.lap, x = x, edge_attr = self.lap_w)

	def message(self, x_j, edge_attr):

		return edge_attr*x_j
		
def global_sum_pool(x, batch, size=None):
	## From: "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html"
	"""
	Globally pool node embeddings into graph embeddings, via elementwise mean.
	Pooling function takes in node embedding [num_nodes x emb_dim] and
	batch (indices) and outputs graph embedding [num_graphs x emb_dim].

	Args:
		x (torch.tensor): Input node embeddings
		batch (torch.tensor): Batch tensor that indicates which node
		belongs to which graph
		size (optional): Total number of graphs. Can be auto-inferred.

	Returns: Pooled graph embeddings

	"""
	size = batch.max().item() + 1 if size is None else size
	return scatter(x, batch, dim=0, dim_size=size, reduce='sum')

class Interpolate(MessagePassing):

	def __init__(self, ftrns1_diff, k = 15, sig = None, device = 'cpu'):
		super(Interpolate, self).__init__('mean', node_dim = 0) # consider mean
		self.ftrns1_diff = ftrns1_diff
		self.device = device
		self.k = k

	def forward(self, x_context, x_query, x, ker_x):

		# edges = knn(torch.Tensor(ftrns1(x_context)/1000.0).to(device), torch.Tensor(ftrns1(x_query)/1000.0).to(device), k = self.k).flip(0).long().contiguous().to(device)
		edges = knn(self.ftrns1_diff(x_context)/1000.0, self.ftrns1_diff(x_query)/1000.0, k = self.k).flip(0).long().contiguous() # .to(device)

		return self.propagate(edges, x = x, size = (x_context.shape[0], x_query.shape[0]))

	def message(self, x_j):
		
		return x_j

class InterpolateWeighted(MessagePassing):

	def __init__(self, ftrns1_diff, k = 15, sig = 10.0, device = 'cpu'):
		super(Interpolate, self).__init__('sum', node_dim = 0) # consider mean
		self.ftrns1_diff = ftrns1_diff
		self.device = device
		self.k = k
		self.sig = sig

	def forward(self, x_context, x_query, x, ker_x):

		# edges = knn(torch.Tensor(ftrns1(x_context)/1000.0).to(device), torch.Tensor(ftrns1(x_query)/1000.0).to(device), k = self.k).flip(0).long().contiguous().to(device)
		pos1 = self.ftrns1_diff(x_context)/1000.0
		pos2 = self.ftrns1_diff(x_query)/1000.0
		edges = knn(pos1, pos2, k = self.k).flip(0).long().contiguous() # .to(device)
		weight = torch.exp(-0.5*torch.norm(pos1[edges[0]] - pos2[edges[1]], dim = 1)**2/(self.sig**2)).reshape(-1,1)
		weight_sum = global_sum_pool(weight, edges[1]).repeat_interleave(self.k, dim = 0)
		weight_sum[weight_sum == 0] = 1.0
		weight = weight/weight_sum

		return self.propagate(edges, x = x, edge_attr = weight, size = (x_context.shape[0], x_query.shape[0]))

	def message(self, x_j, edge_attr):

		return edge_attr.unsqueeze(1)*x_j

class InterpolateAnisotropic(MessagePassing):

	def __init__(self, ftrns1_diff, k = 15, sig = 10.0, device = 'cpu'):
		super(InterpolateAnisotropic, self).__init__('sum', node_dim = 0) # consider mean
		self.ftrns1_diff = ftrns1_diff
		self.device = device
		self.k = k
		self.sig = sig
		self.Softplus = nn.Softplus()

	def forward(self, x_context, x_query, x, ker_x):

		# edges = knn(torch.Tensor(ftrns1(x_context)/1000.0).to(device), torch.Tensor(ftrns1(x_query)/1000.0).to(device), k = self.k).flip(0).long().contiguous().to(device)
		pos1 = self.ftrns1_diff(x_context)/1000.0
		pos2 = self.ftrns1_diff(x_query)/1000.0
		edges = knn(pos1, pos2, k = self.k).flip(0).long().contiguous() # .to(device)
		# weight = torch.exp(-0.5*torch.norm(pos1[edges[0]] - pos2[edges[1]], dim = 1)**2/(self.sig**2)).reshape(-1,1)
		weight = torch.exp(-0.5*(((pos1[edges[0]] - pos2[edges[1]])/self.Softplus(ker_x[edges[0], :]))**2).sum(1)).reshape(-1,1)
		## kernel is not station adaptive, sice this is called independent of station, and broadcast.

		weight_sum = global_sum_pool(weight, edges[1]).repeat_interleave(self.k, dim = 0)
		weight_sum[weight_sum == 0] = 1.0
		weight = weight/weight_sum

		return self.propagate(edges, x = x, edge_attr = weight, size = (x_context.shape[0], x_query.shape[0]))

	def message(self, x_j, edge_attr):

		return edge_attr.unsqueeze(1)*x_j

## Attach corrections to travel times
class TrvTimesCorrection(nn.Module):

	def __init__(self, trv, x_grid, locs_ref, coefs, ftrns1_diff, coefs_ker = None, interp_type = 'anisotropic', k = 15, sig = 10.0, device = 'cpu'):
		super(TrvTimesCorrection, self).__init__() # consider mean
		self.trv = trv
		self.x_grid = x_grid
		self.coefs = coefs
		self.coefs_ker = coefs_ker
		self.interp_type = interp_type
		self.locs_ref = locs_ref
		self.device = device
		self.sig = sig
		self.k = k
		self.ftrns1_diff = ftrns1_diff
		self.locs_ref_cart = ftrns1_diff(torch.Tensor(self.locs_ref).to(device))/1000.0

		if interp_type == 'mean':
			self.Interp = Interpolate(ftrns1_diff, k = k, device = device)

		elif interp_type == 'weighted':
			self.Interp = InterpolateWeighted(ftrns1_diff, k = k, sig = sig, device = device)

		elif interp_type == 'anisotropic':
			self.Interp = InterpolateAnisotropic(ftrns1_diff, k = k, sig = sig, device = device)

		else:
			error('no interp type')


	def forward(self, sta, src):

		sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[0].long().contiguous()
		# sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1).flip(0)[1].long().contiguous()

		return self.trv(sta, src) + self.correction(sta_ind, src) # [:,knn_nearest,:]

	def correction(self, sta_ind, src):

		return self.Interp(self.x_grid, src, self.coefs[:,sta_ind,:], self.coefs_ker)
