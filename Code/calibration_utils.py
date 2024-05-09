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
		super(InterpolateWeighted, self).__init__('sum', node_dim = 0) # consider mean
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

		pos1 = self.ftrns1_diff(x_context)/1000.0
		pos2 = self.ftrns1_diff(x_query)/1000.0
		edges = knn(pos1, pos2, k = self.k).flip(0).long().contiguous() # .to(device)

		# weight = torch.exp(-0.5*torch.norm(pos1[edges[0]] - pos2[edges[1]], dim = 1)**2/(self.sig**2)).reshape(-1,1)
		weight = torch.exp(-0.5*(((pos1[edges[0]] - pos2[edges[1]]).unsqueeze(1)/self.Softplus(ker_x[edges[0], :, :]))**2).sum(2)) # .reshape(-1,1)

		weight_sum = global_sum_pool(weight, edges[1]).repeat_interleave(self.k, dim = 0)
		weight_sum[weight_sum == 0] = 1.0
		weight = weight/weight_sum

		return self.propagate(edges, x = x, edge_attr = weight, size = (x_context.shape[0], x_query.shape[0]))

	def message(self, x_j, edge_attr):

		return edge_attr.unsqueeze(2)*x_j

## Attach corrections to travel times
# class TrvTimesCorrection(nn.Module):

# 	def __init__(self, trv, x_grid, locs_ref, coefs, ftrns1_diff, coefs_ker = None, interp_type = 'anisotropic', k = 15, sig = 10.0, device = 'cpu'):
# 		super(TrvTimesCorrection, self).__init__() # consider mean
# 		self.trv = trv
# 		self.x_grid = x_grid
# 		self.coefs = coefs
# 		self.coefs_ker = coefs_ker
# 		self.interp_type = interp_type
# 		self.locs_ref = locs_ref
# 		self.device = device
# 		self.sig = sig
# 		self.k = k
# 		self.ftrns1_diff = ftrns1_diff
# 		self.locs_ref_cart = ftrns1_diff(torch.Tensor(self.locs_ref).to(device))/1000.0

# 		if interp_type == 'mean':
# 			self.Interp = Interpolate(ftrns1_diff, k = k, device = device)

# 		elif interp_type == 'weighted':
# 			self.Interp = InterpolateWeighted(ftrns1_diff, k = k, sig = sig, device = device)

# 		elif interp_type == 'anisotropic':
# 			self.Interp = InterpolateAnisotropic(ftrns1_diff, k = k, sig = sig, device = device)

# 		else:
# 			error('no interp type')


# 	def forward(self, sta, src):

# 		sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[1].long().contiguous()
# 		# sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1).flip(0)[1].long().contiguous()

# 		return self.trv(sta, src) + self.correction(sta_ind, src) # [:,knn_nearest,:]

# 	def correction(self, sta_ind, src):

# 		return self.Interp(self.x_grid, src, self.coefs[:,sta_ind,:], self.coefs_ker)

class TrvTimesCorrection(nn.Module):

	def __init__(self, trv, x_grid, locs_ref, coefs, ftrns1_diff, coefs_ker = None, interp_type = 'anisotropic', k = 15, sig = 10.0, trv_direct = None, device = 'cpu'):
		super(TrvTimesCorrection, self).__init__() # consider mean
		self.trv = trv
		self.trv_direct = trv_direct
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


	def forward(self, sta, src, method = 'pairs'):

		if method == 'direct':

			sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[1].long().contiguous()

			return self.trv_direct(sta, src) + self.correction(sta_ind, src, method = 'direct') # [:,knn_nearest,:]


		elif method == 'pairs':

			sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[1].long().contiguous()
			# sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1).flip(0)[1].long().contiguous()

			return self.trv(sta, src) + self.correction(sta_ind, src) # [:,knn_nearest,:]


	def correction(self, sta_ind, src, method = 'pairs'):

		if method == 'direct':

			scale_vec = np.array([1.0, 1.0, 1000.0]).reshape(1,-1) # .to(self.device)
			src_unique = np.unique(src.cpu().detach().numpy()/scale_vec, axis = 0)
			tree_src = cKDTree(src_unique)
			ip_match_src = torch.Tensor(tree_src.query(src.cpu().detach().numpy()/scale_vec)[1]).long().to(self.device)

			sta_ind_unique = np.sort(np.unique(sta_ind.cpu().detach().numpy()))
			tree_sta = cKDTree(sta_ind_unique.reshape(-1,1))
			ip_match_sta = torch.Tensor(tree_sta.query(sta_ind.cpu().detach().numpy().reshape(-1,1))[1]).long().to(self.device)

			return self.Interp(self.x_grid, torch.Tensor(src_unique*scale_vec).to(self.device), self.coefs[:,sta_ind_unique,:], self.coefs_ker[:,sta_ind_unique,:])[ip_match_src, ip_match_sta, :]

		elif method == 'pairs':

			return self.Interp(self.x_grid, src, self.coefs[:,sta_ind,:], self.coefs_ker[:,sta_ind,:])

## Magnitude class
class Magnitude(nn.Module):
	def __init__(self, x_grid, locs_ref, ftrns1_diff, interp_type = 'anisotropic', k = 15, sig = 10.0, device = 'cpu'):
		# super(MagPred, self).__init__(aggr = 'max') # node dim
		super(Magnitude, self).__init__() # node dim

		# In elliptical coordinates
		self.x_grid = x_grid
		self.locs_ref = torch.Tensor(locs_ref).to(device)
		self.ftrns1_diff = ftrns1_diff
		self.k = k
		self.sig = sig
		self.device = device
		self.x_grid_cart = ftrns1_diff(x_grid)
		self.locs_ref_cart = ftrns1_diff(torch.Tensor(self.locs_ref).to(device))/1000.0
		self.lref = len(locs_ref)

		self.mag_coef = nn.Parameter(torch.ones(2))
		self.epicenter_spatial_coef = nn.Parameter(torch.ones(2))
		self.depth_spatial_coef = nn.Parameter(torch.zeros(2))
		self.coefs = nn.Parameter(torch.zeros(x_grid.shape[0], locs_ref.shape[0], 2))
		self.coefs_ker = nn.Parameter(sig*torch.ones(x_grid.shape[0], locs_ref.shape[0], 3))
		self.x_grid_save = nn.Parameter(x_grid, requires_grad = False)
		self.zvec = torch.Tensor([1.0,1.0,0.0]).reshape(1,-1).to(device)
		self.Softplus = nn.Softplus()

		if interp_type == 'mean':
			self.Interp = Interpolate(ftrns1_diff, k = k, device = device)

		elif interp_type == 'weighted':
			self.Interp = InterpolateWeighted(ftrns1_diff, k = k, sig = sig, device = device)

		elif interp_type == 'anisotropic':
			self.Interp = InterpolateAnisotropic(ftrns1_diff, k = k, sig = sig, device = device)

		else:
			error('no interp type')

	## Need to double check these routines
	def forward(self, sta, src, mag, phase, method = 'pairs'):

		## Assume inputs are any number of picks (specified by sta positions and phase), and any number of sources; will populate all pairs
		sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[1].long().contiguous()

		if method == 'pairs':

			fudge = 1.0 # add before log10, to avoid log10(0)
			pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1_diff(src*self.zvec).unsqueeze(1) - self.ftrns1_diff(sta*self.zvec).unsqueeze(0), dim = 2) + fudge)
			pw_log_dist_depths = torch.log10(torch.abs(src[:,2].view(-1,1) - sta[:,2].view(1,-1)) + fudge)
			
			if len(sta_ind) < self.lref:
				bias_l = self.Interp(self.x_grid, src, self.coefs[:,sta_ind,:], self.coefs_ker[:,sta_ind,:])
			else:
				bias_l = self.Interp(self.x_grid, src, self.coefs, self.coefs_ker)[:,sta_ind,:]

			iwhere_p = torch.where(phase == 0)[0]
			iwhere_s = torch.where(phase == 1)[0]
			bias = torch.zeros((bias_l.shape[0], bias_l.shape[1])).to(self.device)
			bias[:,iwhere_p] = bias_l[:,iwhere_p,0]
			bias[:,iwhere_s] = bias_l[:,iwhere_s,1]
			phase = phase.reshape(1,-1)

			log_amp = mag.reshape(-1,1)*self.Softplus(self.mag_coef[phase]) - self.Softplus(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

		elif method == 'direct':

			fudge = 1.0 # add before log10, to avoid log10(0)
			pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1_diff(src*self.zvec) - self.ftrns1_diff(sta*self.zvec), dim = 1) + fudge)
			pw_log_dist_depths = torch.log10(torch.abs(src[:,2] - sta[:,2]) + fudge)
			
			bias_l = self.Interp(self.x_grid, src, self.coefs, self.coefs_ker) # [:,sta_ind,:]
			iwhere_p = torch.where(phase == 0)[0]
			iwhere_s = torch.where(phase == 1)[0]
			bias = torch.zeros(len(phase)).to(self.device)
			bias[iwhere_p] = bias_l[iwhere_p, sta_ind[iwhere_p], 0]
			bias[iwhere_s] = bias_l[iwhere_s, sta_ind[iwhere_s], 1]
			# phase = phase.reshape(1,-1)

			log_amp = mag*self.Softplus(self.mag_coef[phase]) - self.Softplus(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias
			log_amp = log_amp.reshape(-1,1)

		return log_amp

	def magnitude(self, sta, src, log_amp, phase, method = 'pairs'):

		## Assume inputs are any number of picks (specified by sta positions and phase), and any number of sources; will populate all pairs
		sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[1].long().contiguous()

		if method == 'pairs':

			fudge = 1.0 # add before log10, to avoid log10(0)
			pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1_diff(src*self.zvec).unsqueeze(1) - self.ftrns1_diff(sta*self.zvec).unsqueeze(0), dim = 2) + fudge)
			pw_log_dist_depths = torch.log10(torch.abs(src[:,2].view(-1,1) - sta[:,2].view(1,-1)) + fudge)
			
			if len(sta_ind) < self.lref:
				bias_l = self.Interp(self.x_grid, src, self.coefs[:,sta_ind,:], self.coefs_ker[:,sta_ind,:])
			else:
				bias_l = self.Interp(self.x_grid, src, self.coefs, self.coefs_ker)[:,sta_ind,:]

			iwhere_p = torch.where(phase == 0)[0]
			iwhere_s = torch.where(phase == 1)[0]
			bias = torch.zeros((bias_l.shape[0], bias_l.shape[1])).to(self.device)
			bias[:,iwhere_p] = bias_l[:,iwhere_p,0]
			bias[:,iwhere_s] = bias_l[:,iwhere_s,1]
			phase = phase.reshape(1,-1)

			mag = (log_amp.reshape(1,-1) + self.Softplus(self.epicenter_spatial_coef[phase])*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/self.Softplus(self.mag_coef[phase])
			# log_amp = mag.reshape(-1,1)*self.Softplus(self.mag_coef[phase]) - self.Softplus(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

		elif method == 'direct':

			fudge = 1.0 # add before log10, to avoid log10(0)
			pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1_diff(src*self.zvec) - self.ftrns1_diff(sta*self.zvec), dim = 1) + fudge)
			pw_log_dist_depths = torch.log10(torch.abs(src[:,2] - sta[:,2]) + fudge)
			
			bias_l = self.Interp(self.x_grid, src, self.coefs, self.coefs_ker) # [:,sta_ind,:]
			iwhere_p = torch.where(phase == 0)[0]
			iwhere_s = torch.where(phase == 1)[0]
			bias = torch.zeros(len(phase)).to(self.device)
			bias[iwhere_p] = bias_l[iwhere_p, sta_ind[iwhere_p], 0]
			bias[iwhere_s] = bias_l[iwhere_s, sta_ind[iwhere_s], 1]
			# phase = phase.reshape(1,-1)
			# log_amp = mag*self.Softplus(self.mag_coef[phase]) - self.Softplus(self.epicenter_spatial_coef[phase])*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias
			mag = (log_amp + self.Softplus(self.epicenter_spatial_coef[phase])*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/self.Softplus(self.mag_coef[phase])
			mag = mag.reshape(-1,1)

		return mag

	def train(self, sta, src, mag, num, phase_type):

		## Assume inputs are any number of picks (specified by sta positions and phase), and any number of sources; will populate all pairs
		sta_ind = knn(self.locs_ref_cart, self.ftrns1_diff(sta)/1000.0, k = 1)[1].long().contiguous()

		bias_l = self.Interp(self.x_grid, src, self.coefs, self.coefs_ker) # [:,sta_ind,:]
		bias = bias_l.repeat_interleave(torch.Tensor(num).to(self.device).long(), dim = 0)[torch.arange(num.sum()), sta_ind, phase_type].reshape(-1,1)
		# bias_s = bias_l.repeat_interleave(torch.Tensor(num_s).to(device).long(), dim = 0)[torch.arange(num_s.sum()),ind_s,1]
		src_slice = src.repeat_interleave(torch.Tensor(num).to(self.device).long(), dim = 0)


		fudge = 1.0 # add before log10, to avoid log10(0)
		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1_diff(src_slice*self.zvec) - self.ftrns1_diff(sta*self.zvec), dim = 1, keepdim = True) + fudge)
		pw_log_dist_depths = torch.log10(torch.abs(src_slice[:,2] - sta[:,2]) + fudge).reshape(-1,1)


		log_amp = mag.reshape(-1,1)*self.Softplus(self.mag_coef[phase_type]) - self.Softplus(self.epicenter_spatial_coef[phase_type])*pw_log_dist_zero + self.depth_spatial_coef[phase_type]*pw_log_dist_depths + bias

		return log_amp
