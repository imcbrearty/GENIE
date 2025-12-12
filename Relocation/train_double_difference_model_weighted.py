
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import glob
from obspy.geodetics.base import calc_vincenty_inverse

## Make this file self-contained.
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
import os
import obspy
from obspy.core import UTCDateTime
from obspy.clients.fdsn.client import Client
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
import time
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.utils import get_laplacian
from sklearn.neighbors import KernelDensity
from torch_geometric.data import Data, Batch
from torch_geometric.data import HeteroData
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.cluster import SpectralClustering
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from numpy.matlib import repmat
from torch_geometric.utils import degree
from functools import partial
from torch import Tensor
import pdb
import pathlib
import itertools
import gc
import sys

from scipy.signal import find_peaks
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from obspy.geodetics.base import calc_vincenty_inverse
import matplotlib.gridspec as gridspec
import networkx as nx
import cvxpy as cp
import glob


from utils import *
from module import *
from process_utils import *
from calibration_utils import *


path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_file += seperator


argvs = sys.argv
if len(argvs) < 2: 
	argvs.append(0) 

if len(argvs) < 3:
	argvs.append(0)


print('name of program is %s'%argvs[0])
print('day is %s'%argvs[1])


## Load device
device = 'cuda' # calibration_config['device']


# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

name_of_project = config['name_of_project']
use_physics_informed = config['use_physics_informed']

# Load region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load stations
z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
z.close()

## Create path to write files
# write_training_file = path_to_file + 'GNN_TrainedModels/' + name_of_project + '_'

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

rbest_cuda = torch.Tensor(rbest).to(device)
mn_cuda = torch.Tensor(mn).to(device)

def lla2ecef_diff(p, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	# p = p.detach().clone().float().to(device) # why include detach here?
	pi = torch.Tensor([np.pi]).to(device)
	corr_val = torch.Tensor([pi/180.0, pi/180.0]).view(1,-1).to(device)
	# p[:,0:2] = p[:,0:2]*
	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0]*corr_val[0,0])**2)
	# results:
	x = (N + p[:,2])*torch.cos(p[:,0]*corr_val[0,0])*torch.cos(p[:,1]*corr_val[0,1])
	y = (N + p[:,2])*torch.cos(p[:,0]*corr_val[0,0])*torch.sin(p[:,1]*corr_val[0,1])
	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0]*corr_val[0,0])

	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

def ecef2lla_diff(x, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	pi = torch.Tensor([np.pi]).to(device)
	b = torch.sqrt((a**2)*(1 - e**2))
	ep = torch.sqrt((a**2 - b**2)/(b**2))
	p = torch.sqrt(x[:,0]**2 + x[:,1]**2)
	th = torch.atan2(a*x[:,2], b*p)
	lon = torch.atan2(x[:,1], x[:,0])
	lat = torch.atan2((x[:,2] + (ep**2)*b*(torch.sin(th)**3)), (p - (e**2)*a*(torch.cos(th)**3)))
	N = a/torch.sqrt(1 - (e**2)*(torch.sin(lat)**2))
	alt = p/torch.cos(lat) - N
	# lon = np.mod(lon, 2.0*np.pi) # don't use!
	k = (torch.abs(x[:,0]) < 1) & (torch.abs(x[:,1]) < 1)
	alt[k] = torch.abs(x[k,2]) - b
	
	return torch.cat((180.0*lat[:,None]/pi, 180.0*lon[:,None]/pi, alt[:,None]), axis = 1)

def global_mean_pool(x, batch, size=None):
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
	return scatter(x, batch, dim=0, dim_size=size, reduce='mean')

def global_max_pool(x, batch, size=None):
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
	return scatter(x, batch, dim=0, dim_size=size, reduce='max')

def hash_rows(val):
	
	return val[:,0].to(torch.int64) << 32 | val[:,1].to(torch.int64)

# use_spherical = False
if config['use_spherical'] == True:

	earth_radius = 6371e3
	ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T # just subtract mean
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device) # just subtract mean

else:

	earth_radius = 6378137.0
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T # just subtract mean
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device) # just subtract mean

if config['train_travel_time_neural_network'] == False:

	## Load travel times
	z = np.load(path_to_file + '1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))
	
	Tp = z['Tp_interp']
	Ts = z['Ts_interp']
	
	locs_ref = z['locs_ref']
	X = z['X']
	z.close()
	
	x1 = np.unique(X[:,0])
	x2 = np.unique(X[:,1])
	x3 = np.unique(X[:,2])
	assert(len(x1)*len(x2)*len(x3) == X.shape[0])
	
	
	## Load fixed grid for velocity models
	Xmin = X.min(0)
	Dx = [np.diff(x1[0:2]),np.diff(x2[0:2]),np.diff(x3[0:2])]
	Mn = np.array([len(x3), len(x1)*len(x3), 1]) ## Is this off by one index? E.g., np.where(np.diff(xx[:,0]) != 0)[0] isn't exactly len(x3)
	N = np.array([len(x1), len(x2), len(x3)])
	X0 = np.array([locs_ref[0,0], locs_ref[0,1], 0.0]).reshape(1,-1)
	
	trv = interp_1D_velocity_model_to_3D_travel_times(X, locs_ref, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2, device = device) # .to(device)

elif config['train_travel_time_neural_network'] == True:

	n_ver_trv_time_model_load = 1
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', use_physics_informed = use_physics_informed, device = device)
	# trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, use_physics_informed = use_physics_informed, device = device)


class DataAggregation(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, scale_rel = 10.0, n_dim = 3, n_dim_mask = 2, ndim_proj = 3):
		super(DataAggregation, self).__init__('mean') # node dim
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

		self.scale_rel = scale_rel
		self.merge_edges = nn.Sequential(nn.Linear(n_hidden + ndim_proj, n_hidden), nn.PReLU())

	def forward(self, tr, mask, A_in_sta, A_in_src, A_src_in_sta, pos_loc, pos_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		# embed_sta_edges = self.fproj_edges_sta(pos_loc/1e6)

		pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]]/1000.0 - pos_loc[A_src_in_sta[0][A_in_sta[1]]]/1000.0)/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]]/1000.0 - pos_src[A_src_in_sta[1][A_in_src[1]]]/1000.0)/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)

		## Could add binary edge type information to indicate data type
		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr), edge_attr = pos_rel_sta), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr), edge_attr = pos_rel_src), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), edge_attr = pos_rel_sta), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), edge_attr = pos_rel_src), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

	def message(self, x_j, edge_attr):

		return self.merge_edges(torch.cat((x_j, edge_attr), dim = 1)) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_mask = 11, ndim_edges = 3, scale_rel = 10e3):
		super(BipartiteGraphOperator, self).__init__('mean') # add
		# include a single projection map
		self.fc1 = nn.Sequential(nn.Linear(ndim_in + ndim_edges + ndim_mask, ndim_in), nn.PReLU(), nn.Linear(ndim_in, ndim_in))
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.
		self.scale_rel = scale_rel

	def forward(self, x, mask, A_src_in_edges, A_src_in_sta, locs_cart, src_cart):

		N = x.shape[0]
		M = src_cart.shape[0]

		# print('Bipartite Aggregation')

		return self.activate2(self.fc2(self.propagate(A_src_in_edges, x = torch.cat((x, mask), dim = 1), pos = (locs_cart[A_src_in_sta[0]], src_cart), size = (N, M))))

	def message(self, x_j, pos_i, pos_j):

		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.scale_rel), dim = 1)))

class BipartiteGraphOperatorSta(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_mask = 11, ndim_edges = 3, scale_rel = 10e3):
		super(BipartiteGraphOperatorSta, self).__init__('mean') # add
		# include a single projection map
		self.fc1 = nn.Sequential(nn.Linear(ndim_in + ndim_edges + ndim_mask, ndim_in), nn.PReLU(), nn.Linear(ndim_in, ndim_in))
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.
		self.scale_rel = scale_rel

	def forward(self, x, mask, A_src_in_edges, A_src_in_sta, locs_cart, src_cart):

		N = x.shape[0]
		M = locs_cart.shape[0]

		# print('Bipartite Aggregation')

		return self.activate2(self.fc2(self.propagate(A_src_in_edges, x = torch.cat((x, mask), dim = 1), pos = (src_cart[A_src_in_sta[1]], locs_cart), size = (N, M))))

	def message(self, x_j, pos_i, pos_j):

		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.scale_rel), dim = 1)))

class GNN_Location(nn.Module):

	def __init__(self, ftrns1, ftrns2, inpt_sources = True, use_sta_corr = True, use_memory = False, use_mask = False, use_aggregation = True, use_attention = False, use_smooth_corrections = False, use_proj_embed = False, use_pick_embed = True, use_src_embed = True, use_sta_embed = True, use_uncertainty_weighting = True, n_inpt = 15, n_mask = 15, n_hidden = 20, n_embed = 10, scale_fixed = 1000.0, num_picks = 10000, num_srcs = 10000, num_stas = 100, n_embed_picks = 5, n_globe = 5, n_phases = 2, init_weights = [-3.5, -3.5], init_weights_diff = [-4.0, -4.0], init_weights_sta = [-4.5,-4.5], min_sta_sigma = 0.01, use_cpu_embeddings = False, device = 'cuda'):

		super(GNN_Location, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim

		if inpt_sources == True:
			n_inpt = n_inpt + 3
			n_mask = n_mask + 3

		if use_memory == True:
			n_read_out = 30
			self.proj_memory = nn.Sequential(nn.Linear(4, 30), nn.PReLU(), nn.Linear(30, 15))
			self.merge_data = nn.Sequential(nn.Linear(30, 30), nn.PReLU(), nn.Linear(30, n_read_out))
			n_inpt = n_inpt + 4
			n_mask = n_mask + 4
		else:
			n_read_out = 15

		self.DataAggregation1 = DataAggregation(n_inpt, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation2 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation3 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation4 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation5 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15

		## Could make attention layer for read out (if limit the number of neighbors)
		self.BipartiteReadOut1 = BipartiteGraphOperator(30, 15, ndim_mask = n_embed)
		self.BipartiteReadOut2 = BipartiteGraphOperatorSta(30, 15, ndim_mask = n_embed)

		self.embed_inpt = nn.Sequential(nn.Linear(n_mask, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_embed))
		self.proj = nn.Sequential(nn.Linear(n_read_out, 30), nn.PReLU(), nn.Linear(30, 3))
		self.proj_t = nn.Sequential(nn.Linear(n_read_out, 15), nn.PReLU(), nn.Linear(15, 1))

		# if use_sta_corr == True:
		if use_smooth_corrections == True:
			self.proj_c = nn.Sequential(nn.Linear(15, 15), nn.PReLU(), nn.Linear(15, 4))
		else:
			self.proj_c = nn.Sequential(nn.Linear(15, 15), nn.PReLU(), nn.Linear(15, 2))
		self.use_smooth_corrections = use_smooth_corrections ## If True, use radially distant dependent corrections

		if use_mask == True:
			self.proj_mask = nn.Sequential(nn.Linear(30, 15), nn.PReLU(), nn.Linear(15, 2))
		self.use_memory = use_memory


		assert(n_phases) == 2
		emb_device = device if use_cpu_embeddings == False else 'cpu' ## Allow embeddings to exist on cpu for low ram cost
		if use_pick_embed == True:
			use_sparse = True if num_picks > 5e6 else False
			self.pick_embed = nn.Embedding(num_picks*n_phases, n_embed_picks, device = emb_device, sparse = use_sparse) 

		if use_src_embed == True:
			use_sparse = True if num_srcs > 5e6 else False
			self.src_embed = nn.Embedding(num_srcs, n_embed_picks, device = emb_device, sparse = use_sparse)

		if use_sta_embed == True:
			use_sparse = True if num_stas > 5e6 else False
			self.sta_embed = nn.Embedding(num_stas, n_embed_picks, device = emb_device, sparse = use_sparse) 

		self.globe = nn.Parameter(torch.zeros(1, n_globe)) ## Share globe in absolute and diff branch
		self.globe_sta = nn.Parameter(torch.zeros(1, n_globe))
		self.type_logvar_bias_abs = nn.Parameter(torch.tensor([init_weights[0], init_weights[1]]).unsqueeze(1)) # Could initilize
		self.type_logvar_bias_diff = nn.Parameter(torch.tensor([init_weights_diff[0], init_weights_diff[1]]).unsqueeze(1)) # Could initilize
		self.type_logvar_bias_sta = nn.Parameter(torch.tensor([init_weights_sta[0], init_weights_sta[1]]).unsqueeze(1)) # Could initilize

		# ## Add separate bias for absolute times
		# if use_abs_bias == True:
		# 	# self.type_logvar_bias_abs = nn.Parameter(torch.tensor([init_weights[0], init_weights[1]])) # Could initilize
		# 	self.type_logvar_bias_abs = nn.Parameter(torch.tensor([0.0, 0.0]).unsqueeze(1)) # Could initilize
		# else:
		# 	# self.type_logvar_bias_abs = torch.Tensor([0.0, 0.0])
		# 	self.register_buffer("type_logvar_bias_abs", torch.zeros(2,1))
		# self.use_abs_bias = use_abs_bias
		self.num_picks = num_picks
		self.num_srcs = num_srcs
		self.num_stas = num_stas


		## Base it on absolute distances (and source depths)
		self.proj_distance = nn.Sequential(nn.Linear(1 + 1, 16), nn.PReLU(), nn.Linear(16, n_embed_picks))
		self.proj_distance_diff = nn.Sequential(nn.Linear(3 + 1 + 2, 32), nn.PReLU(), nn.Linear(32, 2*n_embed_picks))

		# # n_total_dim = n_globe
		# n_total_dim_abs = n_globe + n_embed_picks + n_embed_picks + n_embed_picks ## globe, pick embedding, source embedding, distance_embedding
		# n_total_dim_diff = n_globe + 2*n_embed_picks + 2*n_embed_picks + 2*n_embed_picks ## globe, 2*pick embedding, 2*source embedding, 2*distance_embedding
		# n_total_dim_sta = n_globe + n_embed_picks ## globe and station embedding

		## Dont use source specific embeddings yet
		# n_total_dim = n_globe
		n_total_dim_abs = n_globe + n_embed_picks + n_embed_picks ## globe, pick embedding, source embedding, distance_embedding
		n_total_dim_diff = n_globe + 2*n_embed_picks + 2*n_embed_picks ## globe, 2*pick embedding, 2*source embedding, 2*distance_embedding
		n_total_dim_sta = n_globe + n_embed_picks ## globe and station embedding

		# if use_src_embed == True: n_total_dim = n_total_dim + n_embed_picks
		# if use_sta_embed == True: n_total_dim = n_total_dim + n_embed_picks
		# if use_proj_embed == True: n_total_dim = n_total_dim + 2*n_embed_picks
		self.f_embed_abs = nn.Sequential(nn.LayerNorm(n_total_dim_abs), nn.Linear(n_total_dim_abs, 1)) ## LayerNorm
		# self.f_embed_diff = nn.Sequential(nn.LayerNorm(n_total_dim_diff), nn.Linear(n_total_dim_diff, 1)) ## LayerNorm
		self.f_embed_diff = nn.Sequential(nn.LayerNorm(n_total_dim_diff), nn.Linear(n_total_dim_diff, n_total_dim_diff), nn.PReLU(), nn.Linear(n_total_dim_diff, 1)) ## LayerNorm
		self.f_embed_sta = nn.Sequential(nn.LayerNorm(n_total_dim_sta), nn.Linear(n_total_dim_sta, 2))

		## Initilize forward maps
		nn.init.normal_(self.f_embed_abs[1].weight,   mean=0.0, std=0.01)
		nn.init.constant_(self.f_embed_abs[1].bias,   0.0)

		nn.init.normal_(self.f_embed_diff[1].weight,   mean=0.0, std=0.01)
		nn.init.constant_(self.f_embed_diff[1].bias,   0.0)

		nn.init.normal_(self.f_embed_sta[1].weight,   mean=0.0, std=0.01)
		nn.init.constant_(self.f_embed_sta[1].bias,   0.0)

		## Initilize Embeddings
		nn.init.normal_(self.pick_embed.weight,      mean=0.0, std=0.02)
		nn.init.normal_(self.src_embed.weight,      mean=0.0, std=0.02)
		nn.init.normal_(self.sta_embed.weight,      mean=0.0, std=0.02)

		if use_uncertainty_weighting == True:
			self.UncertaintyWeighting = UncertaintyWeighting()

		self.subset_params = ['pick_embed', 'src_embed', 'sta_embed', 
		'globe', 'globe_sta', 'type_logvar_bias_abs', 'type_logvar_bias_diff', 
		'type_logvar_bias_sta', 'proj_distance', 'proj_distance_diff', 
		'f_embed_abs', 'f_embed_diff', 'f_embed_sta']

		self.use_sta_corr = use_sta_corr
		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2
		self.scale = torch.Tensor([scale_fixed]).to(device)
		self.min_sta_sigma = min_sta_sigma
		# self.scale_proj = torch.Tensor([1.0/5.0]).to(device)
		self.device = device

	def forward(self, x, mask, A_in_pick, A_in_src, A_src_in_product, A_sta_in_product, A_src_in_sta, locs_cart, srcs_cart, memory = False):

		if self.use_memory == True:
			mask = self.embed_inpt(torch.cat((mask, memory[A_src_in_sta[1]]), dim = 1))
			x = torch.cat((x, memory[A_src_in_sta[1]]), dim = 1)
		else:
			mask = self.embed_inpt(mask)

		x = self.DataAggregation1(x, mask, A_in_pick, A_in_src, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation2(x, mask, A_in_pick, A_in_src, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation3(x, mask, A_in_pick, A_in_src, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation4(x, mask, A_in_pick, A_in_src, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation5(x, mask, A_in_pick, A_in_src, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers

		x1 = self.BipartiteReadOut1(x, mask, A_src_in_product, A_src_in_sta, locs_cart, srcs_cart)
		x2 = self.BipartiteReadOut2(x, mask, A_sta_in_product, A_src_in_sta, locs_cart, srcs_cart)

		if self.use_memory == True:
			proj_memory = self.proj_memory(memory)
			x1 = self.merge_data(torch.cat((x1, proj_memory), dim = 1)) ## Can add memory for station corrections as well

		return self.scale*self.proj(x1), self.proj_t(x1), self.proj_c(x2), x # self.proj_mask(x)


	def forward_picks_abs(self, pick_id, src_id, sta_id, dist_coord, src_depth, phase_type):

		pck = self.pick_embed(pick_id) # ; print(pck.shape) ## Overload phase type as well
		# src = self.src_embed(src_id) # ; print(src.shape)
		glb = self.globe.expand(len(pick_id), -1)
		dst = self.proj_distance(torch.cat((dist_coord, src_depth), dim = 1))

		log_variance = (self.f_embed_abs(torch.cat((pck, dst, glb), dim = 1)) + self.type_logvar_bias_abs[phase_type]).clamp(min = -20.0, max = 80.0) # *emb.shape[:-1]

		return log_variance

	def forward_picks_diff(self, pick_id1, pick_id2, src_id1, src_id2, sta_id1, sta_id2, dist_coord1, dist_coord2, dist_coord3, src_depth1, src_depth2, dot_prod, phase_type):

		pck1 = self.pick_embed(pick_id1)
		pck2 = self.pick_embed(pick_id2)
		dst = self.proj_distance_diff(torch.cat((dist_coord1, dist_coord2, dist_coord3, src_depth1, src_depth2, dot_prod), dim = 1))
		glb = self.globe.expand(len(pick_id1), -1)

		log_variance = (self.f_embed_diff(torch.cat((pck1, pck2, dst, glb), dim = 1)) + self.type_logvar_bias_diff[phase_type]).clamp(min = -20.0, max = 80.0) # *emb.shape[:-1]

		return log_variance

	# def forward_picks_sta(self, pick_id, src_id, sta_id, dist_coord, phase_type):
	def forward_picks_sta(self, sta_id, phase_type):

		sta = self.sta_embed(sta_id)
		# pck = self.pick_embed(pick_id) # ; print(pck.shape) ## Overload phase type as well
		# src = self.src_embed(src_id) # ; print(src.shape)
		glb = self.globe_sta.expand(len(sta_id), -1)

		log_variance = (self.f_embed_sta(torch.cat((sta, glb), dim = 1)) + self.type_logvar_bias_sta[phase_type]).clamp(min = -20.0, max = 80.0) # *emb.shape[:-1]

		return log_variance

	def forward_fixed(self, x, mask, memory = False):

		if self.use_memory == True:
			mask = self.embed_inpt(torch.cat((mask, memory[self.A_src_in_sta[1]]), dim = 1))
			x = torch.cat((x, memory[self.A_src_in_sta[1]]), dim = 1)
		else:
			mask = self.embed_inpt(mask)

		x = self.DataAggregation1(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation2(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation3(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation4(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation5(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers

		x1 = self.BipartiteReadOut1(x, mask, self.A_src_in_product, self.A_src_in_sta, self.locs_cart, self.srcs_cart)
		x2 = self.BipartiteReadOut2(x, mask, self.A_sta_in_product, self.A_src_in_sta, self.locs_cart, self.srcs_cart)

		if self.use_memory == True:
			proj_memory = self.proj_memory(memory)
			x1 = self.merge_data(torch.cat((x1, proj_memory), dim = 1)) ## Can add memory for station corrections as well

		return self.scale*self.proj(x1), self.proj_t(x1), self.proj_c(x2), x # self.proj_mask(x)

	def set_adjacencies(self, A_in_pick, A_in_src, A_src_in_product, A_sta_in_product, A_src_in_sta, locs_cart, srcs_cart): # phase_type = 'P'

		self.A_in_pick = A_in_pick
		self.A_in_src = A_in_src
		self.A_src_in_product = A_src_in_product
		self.A_sta_in_product = A_sta_in_product
		self.A_src_in_sta = A_src_in_sta
		self.locs_cart = locs_cart
		self.srcs_cart = srcs_cart

#### Loss functions ######

def compute_abs_loss(m, trv_out_perturbed, Arrivals, Arrivals_hash, sorted_hash_picks, srcs_cart, locs_cart, anneal_val = 1.0):

	ip1 = torch.where(Arrivals[:,3] == 1)[0]
	is1 = torch.where(Arrivals[:,4] == 1)[0]

	imatch1 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[ip1])
	imatch2 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[is1])

	scale_dist1 = torch.norm(srcs_cart[unique_picks[imatch1,1],0:3]/(1000.0*10.0) - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0), dim = 1, keepdim = True)
	scale_dist2 = torch.norm(srcs_cart[unique_picks[imatch2,1],0:3]/(1000.0*10.0) - locs_cart[unique_picks[imatch2,0]]/(1000.0*10.0), dim = 1, keepdim = True)

	log_var1 = m.forward_picks_abs(imatch1, unique_picks[imatch1,1], unique_picks[imatch1,0], scale_dist1, ftrns2_diff(srcs_cart[unique_picks[imatch1,1],0:3])[:,2].reshape(-1,1)/(1000.0*10.0), torch.zeros(len(imatch1)).long().to(device))[:,0]
	log_var2 = m.forward_picks_abs(imatch2 + m.num_picks, unique_picks[imatch2,1], unique_picks[imatch2,0], scale_dist2, ftrns2_diff(srcs_cart[unique_picks[imatch2,1],0:3])[:,2].reshape(-1,1)/(1000.0*10.0), torch.ones(len(imatch2)).long().to(device))[:,0]

	b_scale1 = torch.exp(0.5 * log_var1)
	b_scale2 = torch.exp(0.5 * log_var2)

	Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
	Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]

	res_loss_p_abs = torch.median(loss_func(Res_p)).item() # /n_batch
	res_loss_s_abs = torch.median(loss_func(Res_s)).item() # /n_batch

	loss_p_abs = (loss_func(Res_p)/b_scale1 + anneal_val*0.5*log_var1).mean() # /n_batch
	loss_s_abs = (loss_func(Res_s)/b_scale2 + anneal_val*0.5*log_var2).mean() # /n_batch
	# loss_abs_val += (0.5*(loss_func(Res_p)/b_scale1).detach().mean() + 0.5*(loss_func(Res_s)/b_scale2).detach().mean()).item()/n_batch ## Expensive for logging

	loss_abs = (0.5*loss_p_abs + 0.5*loss_s_abs) # /n_batch

	return loss_abs, res_loss_p_abs, res_loss_s_abs

def compute_sta_loss(m, trv_out_perturbed, Arrivals, Arrivals_hash, sorted_hash_picks, A_src_in_sta, anneal_val = 1.0):

	ip1 = torch.where(Arrivals[:,3] == 1)[0]
	is1 = torch.where(Arrivals[:,4] == 1)[0]

	imatch1 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[ip1])
	imatch2 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[is1])

	log_var1 = m.forward_picks_sta(torch.arange(m.num_stas).long().to(device), torch.zeros(m.num_stas).long().to(device))[:,0]
	log_var2 = m.forward_picks_sta(torch.arange(m.num_stas).long().to(device), torch.ones(m.num_stas).long().to(device))[:,1] ## Using second channel output

	b_scale1 = torch.exp(0.5 * log_var1).clamp(min = m.min_sta_sigma)
	b_scale2 = torch.exp(0.5 * log_var2).clamp(min = m.min_sta_sigma)

	# b_scale1 = m.min_sta_sigma + torch.softplus(torch.exp(0.5 * log_var1) - m.min_sta_sigma)
	# b_scale2 = m.min_sta_sigma + torch.softplus(torch.exp(0.5 * log_var2) - m.min_sta_sigma)


	Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
	Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]

	Res_sta_p = global_mean_pool(Res_p, A_src_in_sta[0,ip1]) # + global_mean_pool(0.5*log_var1, A_src_in_sta[0,ip1])
	Res_sta_s = global_mean_pool(Res_s, A_src_in_sta[0,is1]) # + global_mean_pool(0.5*log_var2, A_src_in_sta[0,is1])

	res_loss_p_sta = torch.median(loss_func(Res_sta_p)).item() # /n_batch
	res_loss_s_sta = torch.median(loss_func(Res_sta_s)).item() # /n_batch

	# loss_p_sta = (loss_func(Res_sta_p)/b_scale1 + anneal_val*0.5*log_var1).mean()
	# loss_s_sta = (loss_func(Res_sta_s)/b_scale2 + anneal_val*0.5*log_var2).mean()
	loss_p_sta = (loss_func(Res_sta_p)/b_scale1 + anneal_val*torch.log(b_scale1)).mean()
	loss_s_sta = (loss_func(Res_sta_s)/b_scale2 + anneal_val*torch.log(b_scale2)).mean()

	# loss_sta_val += (0.5*loss_func(Res_sta_p).detach().mean() + 0.5*loss_func(Res_sta_s).detach().mean()).item()/n_batch

	loss_sta = (0.5*loss_p_sta + 0.5*loss_s_sta) # /n_batch

	return loss_sta, res_loss_p_sta, res_loss_s_sta


def compute_diff_loss(m, srcs_perturbed, source_indices, merged_values, imatch_srcs, sorted_hash_picks, srcs_cart, locs_cart, unique_picks, anneal_val = 1.0):

	islice_edges, values_slice = subgraph(torch.Tensor(imatch_srcs).to(device).long(), source_indices, merged_values)
	n_edges_slice = islice_edges.shape[1]

	imatch1 = torch.searchsorted(sorted_hash_picks, hash_rows(torch.cat((values_slice[:,1].reshape(-1,1), islice_edges[0].reshape(-1,1)), dim = 1)))
	imatch2 = torch.searchsorted(sorted_hash_picks, hash_rows(torch.cat((values_slice[:,1].reshape(-1,1), islice_edges[1].reshape(-1,1)), dim = 1)))

	v1 = srcs_cart[unique_picks[imatch1,1]]/(1000.0*10.0) - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0)
	v2 = srcs_cart[unique_picks[imatch2,1]]/(1000.0*10.0) - locs_cart[unique_picks[imatch2,0]]/(1000.0*10.0)
	scale_dist1 = torch.norm(v1, dim = 1, keepdim = True)
	scale_dist2 = torch.norm(v2, dim = 1, keepdim = True)
	scale_dist3 = torch.norm(srcs_cart[unique_picks[imatch1,1]]/(1000.0*10.0) - srcs_cart[unique_picks[imatch2,1]]/(1000.0*10.0), dim = 1, keepdim = True)
	src_depth1 = ftrns2_diff(srcs_cart[unique_picks[imatch1,1]])[:,2].reshape(-1,1)/(1000.0*10.0) # - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0)
	src_depth2 = ftrns2_diff(srcs_cart[unique_picks[imatch2,1]])[:,2].reshape(-1,1)/(1000.0*10.0) # - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0)
	assert(torch.abs(locs_cart[unique_picks[imatch1,0]] - locs_cart[unique_picks[imatch2,0]]).amax() == 0)

	cos_sim = nn.functional.cosine_similarity(v1, v2, dim = 1).unsqueeze(1)
	log_var_merged = m.forward_picks_diff(imatch1 + m.num_picks*values_slice[:,0].long(), imatch2 + m.num_picks*values_slice[:,0].long(), unique_picks[imatch1,1], unique_picks[imatch2,1], unique_picks[imatch1,0], unique_picks[imatch2,0], scale_dist1, scale_dist2, scale_dist3, src_depth1, src_depth2, cos_sim, values_slice[:,0].long())[:,0]

	b_scale = torch.exp(0.5 * log_var_merged)


	perm_vec = -1*torch.ones(len(srcs_cart)).long().to(device)
	perm_vec[imatch_srcs] = torch.arange(len(imatch_srcs)).to(device)
	src_slice_edge1 = perm_vec[islice_edges[0]]
	src_slice_edge2 = perm_vec[islice_edges[1]]

	assert(torch.min(torch.Tensor([src_slice_edge1.min(), src_slice_edge2.min()])) > -1) # , src_s_slice_edge1.min(), src_s_slice_edge2.min()])) > -1)
	trv_out_perturbed_i = trv_pairwise(locs_cuda[values_slice[:,1].long()], srcs_perturbed[src_slice_edge1])[torch.arange(n_edges_slice), values_slice[:,0].long()] + pred_t[src_slice_edge1,0] + pred_c[values_slice[:,1].long(),values_slice[:,0].long()]
	trv_out_perturbed_j = trv_pairwise(locs_cuda[values_slice[:,1].long()], srcs_perturbed[src_slice_edge2])[torch.arange(n_edges_slice), values_slice[:,0].long()] + pred_t[src_slice_edge2,0] + pred_c[values_slice[:,1].long(),values_slice[:,0].long()]

	trgt_diff = values_slice[:,3]
	pred_diff = trv_out_perturbed_i - trv_out_perturbed_j


	# if i < n_iter_initilize:
	# res_aggregate = scatter(trgt_diff - pred_diff, values_slice[:,0].long(), reduce = 'mean', dim = 0, dim_size = 2)
	res_aggregate = global_mean_pool(loss_func(trgt_diff - pred_diff), values_slice[:,0].long())
	res_loss_p_diff = res_aggregate[0].item()
	res_loss_s_diff = res_aggregate[1].item()

	abs_error = (values_slice[:,2].sqrt())*loss_func(trgt_diff - pred_diff) # Weighted
	loss_diff = ((abs_error / b_scale) + anneal_val*0.5*log_var_merged).mean() # /n_batch


	return loss_diff, res_loss_p_diff, res_loss_s_diff



######### Extra dataset load functions #########


def move_to(obj, device, non_blocking=True):
    """
    Recursively move tensors, lists, dicts, PyG Data/Batch/HeteroData to device.
    Works with any nesting depth.
    """
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=non_blocking)

    # PyTorch Geometric support
    if Data is not None:
        if isinstance(obj, Data):
            return obj.to(device, non_blocking=non_blocking)
        if isinstance(obj, Batch):
            return obj.to(device, non_blocking=non_blocking)
        if isinstance(obj, HeteroData):
            return obj.to(device, non_blocking=non_blocking)

    # Containers
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(x, device, non_blocking) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to(v, device, non_blocking) for k, v in obj.items()}

    # Scalars, strings, None, etc.
    return obj



def move_to_inplace(obj, device, non_blocking=True):
    """
    In-place version — modifies the object directly.
    Useful when you want zero allocation overhead.
    """
    if isinstance(obj, Tensor):
        # In-place move (only works if tensor is not a view)
        obj.data = obj.to(device, non_blocking=non_blocking)
        if obj.grad is not None:
            obj.grad.data = obj.grad.to(device, non_blocking=non_blocking)
        return obj

    if Data is not None:
        if isinstance(obj, (Data, Batch, HeteroData)):
            obj.to(device, non_blocking=non_blocking)  # PyG objects have .to() that does this in-place
            return obj

    if isinstance(obj, (list, tuple)):
        for i in range(len(obj)):
            obj[i] = move_to_inplace(obj[i], device, non_blocking)
        return obj

    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = move_to_inplace(obj[k], device, non_blocking)
        return obj

    return obj

def to_float32(x):
    if isinstance(x, torch.Tensor):
        # Only convert float64 → float32
        if x.dtype == torch.float64:
            return x.float()
        else:
            return x   # leave longs, ints, etc.
    elif isinstance(x, dict):
        return {k: to_float32(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_float32(v) for v in x)
    else:
        return x


def collate_no_batch(batch):
    # assert batch_size=1 for safety
    # assert len(batch) == 1
    return to_float32(batch) ## Map to float (could also put this in the data loader if prefered)


# At top of file — define once
to_gpu = partial(move_to, device='cuda', non_blocking=True)
to_cpu = partial(move_to, device='cpu', non_blocking=False)
to_gpu_inplace = partial(move_to_inplace, device='cuda', non_blocking=True)



################# ######### Dataset Loader ######## #################


class TrainingDataset(Dataset):

	def __init__(self, list_of_hdf5_paths, n_batch = 1): ## This n_batch is not used (instead do batching with loader - though could change to save larger .hdf5 files with multiple samples)
		self.files = list_of_hdf5_paths   # e.g. 1_000_000 files
		self.n_batch = n_batch

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		path = self.files[idx]

		with h5py.File(path, 'r') as z:

			# srcs = torch.from_numpy(z['srcs'][:])
			# srcs_cart = torch.from_numpy(ftrns1(z['srcs'][:]))
			nodes_all = torch.from_numpy(z['nodes_all'][:])
			node_types = torch.from_numpy(z['node_types'][:])
			srcs_slice = torch.from_numpy(z['srcs_slice'][:])
			imatch_srcs = torch.from_numpy(z['imatch_srcs'][:])

			Input = torch.from_numpy(z['Input'][:]) # .to(device)
			Arrivals = torch.from_numpy(z['Arrivals'][:]) # .to(device)
			A_prod_sta_sta = torch.from_numpy(z['A_prod_sta_sta'][:]).long()
			A_prod_src_src = torch.from_numpy(z['A_prod_src_src'][:]).long()
			A_src_in_prod = torch.from_numpy(z['A_src_in_prod'][:]).long()
			A_sta_in_prod = torch.from_numpy(z['A_sta_in_prod'][:]).long()
			A_src_in_sta = torch.from_numpy(z['A_src_in_sta'][:]).long()
			locs_cart = torch.from_numpy(z['locs_cart'][:])
			srcs_slice_cart = torch.from_numpy(z['srcs_slice_cart'][:])

			ifind_p_edges = torch.from_numpy(z['ifind_p_edges'][:]).long()
			ifind_s_edges = torch.from_numpy(z['ifind_s_edges'][:]).long()

			## Hash pick indices
			Arrivals_hash = hash_rows(Arrivals).long()

		# return srcs, srcs_cart, nodes_all, node_types, srcs_slice, imatch_srcs, Input, Arrivals, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, ifind_p_edges, ifind_s_edges, Arrivals_hash
		return nodes_all, node_types, srcs_slice, imatch_srcs, Input, Arrivals, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, ifind_p_edges, ifind_s_edges, Arrivals_hash


################# Define Balancer #################

## Note: could use the uncertainity weighting approach

class LossAccumulationBalancer:
    def __init__(
        self,
        anchor: str = 'loss_diff',
        group_targets: dict = None,
        alpha: float = 0.98,
        primary_ext: str = 'loss_diff',
        device: str = 'cuda'
    ):
        self.anchor = anchor
        self.alpha = alpha
        self.primary_ext = primary_ext
        self.device = device

        # === Group targets ===
        if group_targets is None:
            group_targets = {'primary': 1.0, 'aux': 0.2}
        self.group_targets = group_targets

        # === Persistent state ===
        self.primary_ema = {}
        self.aux_ema = defaultdict(dict)           # aux_ema[group][name] = ema
        self._anchor_ema_current = None

        # === Accumulation buffers (reset every full batch) ===
        self._accum_prim = {}
        self._accum_aux = defaultdict(dict)        # _accum_aux[group][name] = sum
        self._participation = {}                   # primary: name → count
        self._participation_aux = defaultdict(dict)  # aux: group → name → count

        self._step_count = 0
        self.accum_steps = None

    def _get_group(self, name: str) -> str:
        if name.startswith(self.primary_ext):
            return 'primary'
        for group in sorted(self.group_targets.keys(), key=len, reverse=True):
            if group != 'primary' and name.startswith(group):
                return group
        return 'aux'  # fallback

    def __call__(self, losses_dict: dict, accum_steps: int = None, is_last_accum_step: bool = False):
        if accum_steps is not None:
            self.accum_steps = accum_steps

        total_loss = 0.0

        # 1. Accumulate values + participation counters
        for name, loss in losses_dict.items():
            val = loss.detach().mean().item()
            group = self._get_group(name)

            if group == 'primary':
                self._participation[name] = self._participation.get(name, 0) + 1
                self._accum_prim[name] = self._accum_prim.get(name, 0.0) + val
            else:
                self._participation_aux[group][name] = self._participation_aux[group].get(name, 0) + 1
                self._accum_aux[group][name] = self._accum_aux[group].get(name, 0.0) + val

        self._step_count += 1

        # 2. Final microbatch → update EMAs with correct per-loss counts
        if is_last_accum_step or (self.accum_steps and self._step_count >= self.accum_steps):
            # Primary losses
            anchor_ema_new = None
            for name, accum_val in self._accum_prim.items():
                n = self._participation.get(name, 1)
                batch_val = accum_val / n
                if name not in self.primary_ema:
                    self.primary_ema[name] = batch_val
                else:
                    self.primary_ema[name] = self.alpha * self.primary_ema[name] + (1 - self.alpha) * batch_val
                if name == self.anchor:
                    anchor_ema_new = self.primary_ema[name]

            # Auxiliary losses
            for group, accum_dict in self._accum_aux.items():
                for name, accum_val in accum_dict.items():
                    n = self._participation_aux[group].get(name, 1)
                    batch_val = accum_val / n
                    ema_dict = self.aux_ema[group]
                    if name not in ema_dict:
                        ema_dict[name] = batch_val
                    else:
                        ema_dict[name] = self.alpha * ema_dict[name] + (1 - self.alpha) * batch_val

            if anchor_ema_new is not None:
                self._anchor_ema_current = anchor_ema_new
            elif self._anchor_ema_current is None and self.primary_ema:
                self._anchor_ema_current = max(self.primary_ema.values())

            # Reset all accumulators
            self._accum_prim.clear()
            self._accum_aux.clear()
            self._participation.clear()
            self._participation_aux.clear()
            self._step_count = 0

        # 3. Scale current microbatch losses
        anchor_val = self._anchor_ema_current if self._anchor_ema_current is not None else 0.5

        for name, loss in losses_dict.items():
            group = self._get_group(name)
            if group == 'primary':
                ema = self.primary_ema.get(name, 1.0)
                scale = anchor_val / (ema + 1e-8)
            else:
                ema = self.aux_ema[group].get(name, 1.0)
                n_losses = max(len(self.aux_ema[group]), 1)
                target = self.group_targets[group]
                scale = target * anchor_val / (n_losses * (ema + 1e-8))

            scale = torch.clamp(torch.tensor(scale, device=self.device), 
                               min=0.01 if group != 'primary' else 0.1,
                               max=100.0 if group != 'primary' else 300.0)
            total_loss += scale * loss

        return total_loss

    # Optional: make it checkpoint-safe
    def state_dict(self):
        return {
            'anchor': self.anchor,
            'group_targets': self.group_targets,
            'alpha': self.alpha,
            'primary_ext': self.primary_ext,
            'accum_steps': self.accum_steps,
            'primary_ema': self.primary_ema,
            'aux_ema': dict(self.aux_ema),
            '_anchor_ema_current': self._anchor_ema_current,
        }

    def load_state_dict(self, sd):
        self.anchor = sd['anchor']
        self.group_targets = sd.get('group_targets', {'primary': 1.0, 'aux': 0.2})
        self.alpha = sd['alpha']
        self.primary_ext = sd['primary_ext']
        self.accum_steps = sd.get('accum_steps', None)
        self.primary_ema = sd['primary_ema']
        self.aux_ema = defaultdict(dict, sd.get('aux_ema', {}))
        self._anchor_ema_current = sd.get('_anchor_ema_current', None)

################# Uncertainity Balancer #################



# class UncertaintyWeighting(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # diffs are the soft anchor → fixed precision = 1.0
#         self.log_weight_abs    = nn.Parameter(torch.tensor(0.0))
#         self.log_weight_sta = nn.Parameter(torch.tensor(0.0))
#         # diff weight fixed → no parameter

#     def forward(self, loss_abs, loss_diff, loss_sta):
#         w_abs    = torch.exp(-self.log_weight_abs)
#         w_sta = torch.exp(-self.log_weight_sta)

#         total = w_abs * loss_abs    \
#               + 1.0 * loss_diff   \
#               + w_sta * loss_sta

#         return total


class UncertaintyWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_weight_diff = nn.Parameter(torch.tensor(0.0))
        self.log_weight_sta = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_abs, loss_diff, loss_sta):
        w_diff = torch.exp(-self.log_weight_diff)
        w_sta = torch.exp(-self.log_weight_sta)
        
        total = 1.0 * loss_abs \
              + w_diff * loss_diff \
              + w_sta * loss_sta \
              + self.log_weight_diff \
              + self.log_weight_sta
        
        return total



n_batch = 5
n_epochs = 50001
verbose = False
n_ver_save = 1


inpt_sources = True
use_double_diff = False
use_absolute = True
use_sta_corr = True
use_calibration = False
use_mask = False
use_diff = True
use_memory = False

## Specify type of loss function
use_huber = True



if use_huber == False:

	loss_func = nn.L1Loss()

else:

	def huber_loss(residual, delta = 0.025): # 0.05

		huber_term = torch.where(
		    torch.abs(residual) <= delta,
		    0.5 * (residual ** 2),  # L2 for small errors
		    delta * (torch.abs(residual) - 0.5 * delta)  # L1 for large
		)
		# weighted_huber = edge_weight * huber_term

		return huber_term/delta

	loss_func = lambda r: huber_loss(r)


n_ver_load_files = 1
path_save = path_to_file + 'DoubleDifferenceModels/'
path_data = '/scratch/users/imcbrear/LocationTest/DoubleDifferenceData/'
assert((use_double_diff + use_absolute + use_sta_corr + use_calibration + use_diff) > 0)

st_files = glob.glob(path_data + '*ver_%d.hdf5'%n_ver_load_files)

#########  Load unique picks ######### 
unique_picks = []
inc_cnt, n_flush = 0, 30
for s in st_files:
	z = h5py.File(s, 'r')
	unique_picks.append(z['Arrivals'][:, np.array([2,5])].astype('int')) ## Is is 2 or 3?
	if inc_cnt == 0: 
		cur_len = len(np.unique(np.vstack(unique_picks), axis = 0))
	if (np.mod(inc_cnt, n_flush) == 0)*(inc_cnt > 0):
		if (len(np.unique(np.vstack(unique_picks), axis = 0)) == cur_len)*(inc_cnt > n_flush):
			print('Finished loading unique picks %d/%d'%(inc_cnt, len(st_files)))
			z.close()
			break
		unique_picks = list([np.unique(np.vstack(unique_picks), axis = 0)])
		cur_len = len(unique_picks[0])
	z.close()
	inc_cnt += 1
unique_picks = torch.Tensor(np.unique(np.vstack(unique_picks), axis = 0)).long().to(device) ## Can skip device if too expensive


#########  Check for calibration files ######### 
check_calibration = True
srcs_ref = np.zeros((0,4))
Matches = np.zeros((0,2))
if (check_calibration == True)*(use_calibration == True):
	z = h5py.File(st_files[0], 'r')
	srcs_ref = z['srcs_ref'][:]
	Matches = z['Matches'][:]
	Arrivals = z['Arrivals'][:]
	z.close()

	if len(Matches) == 0:
		if use_calibration == True:
			print('Note, no matches loaded, so not using calibration loss')
			use_calibration = False

	print('Add check for no calibration')


#########  Load initial set of reference sources (should add by referencing the actual catalog file) ######### 
z = h5py.File(st_files[0], 'r')
srcs = z['srcs'][:]
srcs_fixed = np.copy(srcs)
tree_srcs = cKDTree(ftrns1(srcs[:,0:3]))
z.close()

if inpt_sources == True:
	srcs_scale_mean = torch.Tensor(srcs[:,0:3].min(0, keepdims = True)).to(device)
	srcs_scale_std = torch.Tensor(srcs[:,0:3].max(0, keepdims = True) - srcs[:,0:3].min(0, keepdims = True)).to(device)


######### Load Diff data #############

if use_diff == True:

	## If using HypoDD format for loss function
	f = open(path_to_file + 'dt.cc', 'r')
	lines = f.readlines()
	f.close()

	diff_time = []
	station_index = []
	source_indices = []
	phase_type = []
	weights = []
	for i in range(len(lines)):
		line = list(filter(lambda x: len(x) > 0, lines[i].strip().split(' ')))
		if line[0] == '#':
			src_ind1, src_ind2 = int(line[1]) - 1, int(line[2]) - 1
			continue
		i1 = np.where(stas == line[0])[0]
		assert(len(i1) == 1)
		station_index.append(i1[0])
		diff_time.append(float(line[1]))
		weights.append(float(line[2]))
		source_indices.append(np.array([src_ind1, src_ind2]))
		if line[3] == 'P':
			phase_type.append(0)
		elif line[3] == 'S':
			phase_type.append(1)
		else:
			error('No correct phase type')

	diff_time = torch.Tensor(np.hstack(diff_time)).reshape(-1,1).to(device)
	station_index = torch.Tensor(np.hstack(station_index)).long().reshape(-1,1).to(device)
	source_indices = torch.Tensor(np.vstack(source_indices).T).long().to(device)
	phase_type = torch.Tensor(np.hstack(phase_type)).reshape(-1,1).to(device)
	weights = torch.Tensor(np.hstack(weights)).reshape(-1,1).to(device)
	merged_values = torch.cat((phase_type, station_index, weights, diff_time), dim = 1)


	unique_picks_diff = torch.cat((station_index, source_indices[0].reshape(-1,1)), dim = 1)
	unique_picks_diff = torch.cat((unique_picks_diff, torch.cat((station_index, source_indices[1].reshape(-1,1)), dim = 1)), dim = 0)
	unique_picks_diff = torch.unique(unique_picks_diff, dim = 0)
	# tree_picks_diff = cKDTree(unique_picks_diff.cpu().detach().numpy())

## Could use the diff pick weights to get a unique "per pick" weight, and add these to the absolute loss
## Merge all unique picks
## Use cuda hash table for picks
unique_picks = torch.unique(torch.cat((unique_picks, unique_picks_diff), dim = 0), dim = 0)
hash_picks = hash_rows(unique_picks) ## Create unique hash (then use torch search sorted)
sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
unique_picks = unique_picks[order_hash_picks]
hash_picks = hash_rows(unique_picks) ## Create unique hash (then use torch search sorted)
sorted_hash_picks, order_hash_picks = torch.sort(hash_picks)
assert(torch.abs(torch.diff(order_hash_picks).amax()) == 1)
num_srcs = unique_picks[:,1].amax() + 1


######### Create dataset and loader #########

dataset = TrainingDataset(np.random.permutation(st_files)) ## Dont use n_batch here

loader = DataLoader(
    dataset,
    batch_size=n_batch,              # ← 64 events at a time → 64 × 10 = 640 sub-samples
    shuffle=True,
    num_workers=3,             # ← THIS is what makes it fast
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=3,          # PyTorch 2.0+: loads 4×batch_size ahead
    collate_fn=collate_no_batch
)

## ################## ## ############### ##


locs_cuda = torch.Tensor(locs).to(device)
srcs_cuda = torch.Tensor(srcs).to(device)
locs_cart = torch.Tensor(ftrns1(locs)).to(device)
srcs_cart = torch.Tensor(ftrns1(srcs)).to(device)

## Setup buffer variables
buffer_weight = 0.98 ## This fraction old data, other fraction new data
# buffer_window = 10



corrections_c = np.zeros((len(locs),2))

srcs_pred_perturb = np.nan*np.zeros((len(srcs),4))
srcs_pred = np.nan*np.zeros((len(srcs),4))
# srcs_pred_std = np.nan*np.zeros((buffer_window, len(srcs), 4))
use_buffer = True

scale_memory = torch.Tensor([5000.0, 5000.0, 5000.0, 5.0]).reshape(1,-1).to(device) ## Can set scale dependent parameters for this (and for convolutions inside GNN layers)


losses = []
losses_abs = []
losses_sta = []
losses_cal = []
losses_cal_abs = []
losses_double_diff = [] # GNN_Location
losses_diff = []

min_sta_sigma = 0.01
use_uncertainty_weighting = True

# tree_picks = cKDTree(uniuqe_picks.cpu().detach().numpy())
m = GNN_Location(ftrns1, ftrns2, inpt_sources = inpt_sources, use_sta_corr = use_sta_corr, use_memory = use_memory, use_mask = use_mask, use_aggregation = False, use_attention = False, num_picks = len(unique_picks), num_srcs = num_srcs, num_stas = len(locs), use_uncertainty_weighting = use_uncertainty_weighting, min_sta_sigma = min_sta_sigma, device = device).to(device)

# n_iter_initilize = 10 ## Initilize the bias terms over 10 samples
res_loss_p_abs = []
res_loss_s_abs = []
res_loss_p_diff = []
res_loss_s_diff = []
res_loss_p_sta = []
res_loss_s_sta = []



init_bias = True
if init_bias == True:

	n_iter_initilize = 10
	with torch.no_grad():

		for batch_idx, inputs in enumerate(loader):	

			# n_iter = 10 ## Compute baseline loss from initilized model
			# res_loss_p, res_loss_s = 0.0, 0.0
			# for i in range(n_iter):

			print('Running initilize %d'%batch_idx)
			inputs = dataset.__getitem__(np.random.choice(len(st_files)))
			nodes_all, node_types, srcs_slice, imatch_srcs, Input, Arrivals, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, ifind_p_edges, ifind_s_edges, Arrivals_hash = to_gpu(to_float32(inputs))		
			imatch_srcs = imatch_srcs.cpu().detach().numpy()
			if inpt_sources == True: Input = torch.cat((Input, (torch.Tensor(srcs_slice[:,0:3]).to(device)[A_src_in_sta[1]] - srcs_scale_mean)/srcs_scale_std), dim = 1)
			if use_memory == True:
				inpt_memory_slice = torch.Tensor(srcs_pred_perturb[imatch_srcs]).to(device)/scale_memory
				inpt_memory_slice[torch.isnan(inpt_memory_slice)] = 0.0
			else:
				inpt_memory_slice = False
			pred, pred_t, pred_c, pred_mask = m(Input, Input, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, memory = inpt_memory_slice)
			srcs_perturbed = ftrns2_diff(srcs_slice_cart + pred) # .cpu().detach().numpy()
			trv_out_perturbed = trv_pairwise(locs_cuda[A_src_in_sta[0]], srcs_perturbed[A_src_in_sta[1]]) + pred_c[A_src_in_sta[0]] + pred_t[A_src_in_sta[1]] # [:, 0]
			# ip1 = torch.where(Arrivals[:,3] == 1)[0]
			# is1 = torch.where(Arrivals[:,4] == 1)[0]
			# Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
			# Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]
			# res_loss_p += torch.median(loss_func(Res_p)).item()/n_iter
			# res_loss_s += torch.median(loss_func(Res_s)).item()/n_iter

			_, res_loss_p_abs_val, res_loss_s_abs_val = compute_abs_loss(m, trv_out_perturbed, Arrivals, Arrivals_hash, sorted_hash_picks, srcs_cart, locs_cart)
			res_loss_p_abs.append(res_loss_p_abs_val)
			res_loss_s_abs.append(res_loss_s_abs_val)

			_, res_loss_p_sta_val, res_loss_s_sta_val = compute_sta_loss(m, trv_out_perturbed, Arrivals, Arrivals_hash, sorted_hash_picks, A_src_in_sta)
			res_loss_p_sta.append(res_loss_p_sta_val)
			res_loss_s_sta.append(res_loss_s_sta_val)

			_, res_loss_p_diff_val, res_loss_s_diff_val = compute_diff_loss(m, srcs_perturbed, source_indices, merged_values, imatch_srcs, sorted_hash_picks, srcs_cart, locs_cart, unique_picks)
			res_loss_p_diff.append(res_loss_p_diff_val)
			res_loss_s_diff.append(res_loss_s_diff_val)

			if batch_idx == n_iter_initilize:
				break

		res_loss_p_abs = np.median(np.hstack(res_loss_p_abs))
		res_loss_s_abs = np.median(np.hstack(res_loss_s_abs))
		res_loss_p_sta = np.median(np.hstack(res_loss_p_sta))
		res_loss_s_sta = np.median(np.hstack(res_loss_s_sta))
		res_loss_p_diff = np.median(np.hstack(res_loss_p_diff))
		res_loss_s_diff = np.median(np.hstack(res_loss_s_diff))

		m.type_logvar_bias_abs.data = torch.Tensor([torch.log(torch.tensor(res_loss_p_abs)), torch.log(torch.tensor(res_loss_s_abs))]).reshape(-1,1).to(m.type_logvar_bias_abs.device)
		m.type_logvar_bias_diff.data = torch.Tensor([torch.log(torch.tensor(res_loss_p_diff)), torch.log(torch.tensor(res_loss_s_diff))]).reshape(-1,1).to(m.type_logvar_bias_diff.device)
		m.type_logvar_bias_sta.data = torch.Tensor([torch.log(torch.tensor(res_loss_p_sta)), torch.log(torch.tensor(res_loss_s_sta))]).reshape(-1,1).to(m.type_logvar_bias_sta.device)

		print('Initilize scales:')
		print('Abs: %0.8f, %0.8f'%(m.type_logvar_bias_abs[0].item(), m.type_logvar_bias_abs[1].item()))
		print('Sta: %0.8f, %0.8f'%(m.type_logvar_bias_sta[0].item(), m.type_logvar_bias_sta[1].item()))
		print('Diff: %0.8f, %0.8f'%(m.type_logvar_bias_diff[0].item(), m.type_logvar_bias_diff[1].item()))



def initilize_optimizer(m):

	uw_params = set(m.UncertaintyWeighting.parameters())
	decay_params = []
	no_decay_params = []
	slow_params = []
	for name, param in m.named_parameters():
		if not param.requires_grad:
			continue
			# Skip uncertainty params here — handled separately
		if param in uw_params:
			continue
		# Embeddings → weight decay
		if any(k in name for k in ["pick_embed", "src_embed", "sta_embed"]):
			decay_params.append(param)
		elif any(k in name for k in m.subset_params):
			slow_params.append(param)
		else:
			no_decay_params.append(param)

		for param in m.UncertaintyWeighting.parameters():
			slow_params.append(param)

	optimizer = optim.AdamW([
			# Main model (no decay)
			{
				"params": no_decay_params,
				"lr": 1e-3,
				"weight_decay": 0.0,
			},
			# Embeddings (with decay)
			{
	            "params": decay_params,
	            "lr": 1e-4,
	            "weight_decay": 1e-4,  # embeddings like decay
			},
			# Uncertainty weighting (slower, usually no decay)
			{
	            "params": slow_params, # m.UncertaintyWeighting.parameters()
	            "lr": 1e-4,
	            "weight_decay": 0.0,
			}])

	# all_params = set(no_decay_params) | set(decay_params) | set(m.UncertaintyWeighting.parameters())
	# assert len(all_params) == sum(p.numel() for p in m.parameters() if p.requires_grad)
	all_params = (
	    set(no_decay_params)
	    | set(decay_params)
	    | set(slow_params) # m.UncertaintyWeighting.parameters()
	)
	model_params = set(p for p in m.parameters() if p.requires_grad)
	assert all_params == model_params

	return optimizer



optimizer = initilize_optimizer(m)

# use_huber = True
n_flush = 10
log_buffer = []

use_anneal_bias = True
if use_anneal_bias == True:
	n_anneal_steps = 1500 # 1000
	def anneal_schedule(n, n_anneal_steps = n_anneal_steps, n_iter_initilize = 0):
		angle = np.pi * (n - n_iter_initilize) / float(n_anneal_steps)
		anneal_value = 0.5 * (1.0 - np.cos(angle))
		return (n >= n_iter_initilize)*(anneal_value*((n - n_iter_initilize) <= n_anneal_steps) + 1.0*((n - n_iter_initilize) > n_anneal_steps))
else:
	anneal_schedule = lambda x: 1.0


## Extra fixed lambda weights on log bias
use_detach = True ## If True, detach sigma from the loss terms
lam_weight_abs = 0.01
lam_weight_diff = 0.01
lam_weight_sta = 0.01


n_restart = False
n_restart_step = 0



# for i in range(n_restart_step, n_epochs):
for batch_idx, inputs in enumerate(loader):


	## Effective step size
	i = int(n_restart_step*n_restart) + batch_idx
	if i > n_epochs:
		print('Finished training')
		sys.exit()


	# load_model = False
	if n_restart == True:

		m.load_state_dict(torch.load(path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d.h5'%(n_restart_step, n_ver_load_files, n_ver_save), map_location = device))
		optimizer.load_state_dict(torch.load(path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_optimizer.h5'%(n_restart_step, n_ver_load_files, n_ver_save), map_location = device))
		zlosses = np.load(path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_losses.npz'%(n_restart_step, n_ver_load_files, n_ver_save))
		losses[0:n_restart_step] = zlosses['losses'][0:n_restart_step]
		print('loaded model for restart on step %d ver %d \n'%(n_restart_step, n_ver_save))

		## Check this
		corrections_c = zlosses['corrections_c']
		srcs_perturbed = zlosses['srcs_perturbed']
		srcs_pred_perturb = ftrns1(srcs_perturbed[:,0:3]) - ftrns1(srcs[:,0:3])
		srcs_pred_perturb = np.concatenate((srcs_pred_perturb, srcs_perturbed[:,[3]]), axis = 1)
		# srcs_perturbed = ftrns2_diff(srcs_cart + torch.Tensor(srcs_pred_perturb[:,0:3]).to(device)).cpu().detach().numpy()
		# srcs_perturbed = np.concatenate((srcs_perturbed, srcs_pred_perturb[:,3].reshape(-1,1)), axis = 1)

		zlosses.close()	

	else:
		n_restart_step = 0


	if (np.mod(i, 1000) == 0)*(i > n_restart_step):

		srcs_slice_perturbed = ftrns2_diff(srcs_slice_cart + pred).cpu().detach().numpy()

		srcs_perturbed = ftrns2_diff(srcs_cart + torch.Tensor(srcs_pred_perturb[:,0:3]).to(device)).cpu().detach().numpy()
		srcs_perturbed = np.concatenate((srcs_perturbed, srcs_pred_perturb[:,3].reshape(-1,1)), axis = 1)
		imask = np.where(np.isnan(srcs_pred_perturb[:,0]) == 1)[0]
		srcs_perturbed[imask,:] = np.nan

		torch.save(m.state_dict(), path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d.h5'%(i, n_ver_load_files, n_ver_save))
		torch.save(optimizer.state_dict(), path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_optimizer.h5'%(i, n_ver_load_files, n_ver_save))
		np.savez_compressed(path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_losses.npz'%(i, n_ver_load_files, n_ver_save), losses = losses, srcs_slice = srcs_slice.cpu().detach().numpy(), srcs_slice_perturbed = srcs_slice_perturbed, srcs = srcs, srcs_perturbed = srcs_perturbed, srcs_pred_perturb = srcs_pred_perturb, pred_t = pred_t.cpu().detach().numpy(), corrections_c = corrections_c, srcs_ref = srcs_ref, Matches = Matches, locs = locs, losses_abs = losses_abs, losses_sta = losses_sta, losses_cal = losses_cal, losses_cal_abs = losses_cal_abs, losses_double_diff = losses_double_diff, losses_diff = losses_diff)
		print('saved model %s data %d step %d'%(n_ver_load_files, n_ver_save, i))
		assert(np.abs(srcs - srcs_fixed).max() < 1e-2)



	optimizer.zero_grad()

	loss_val = 0.0
	loss_val_abs = 0.0
	loss_val_sta = 0.0
	loss_val_cal = 0.0
	loss_val_cal_abs = 0.0
	loss_val_double_diff = 0.0
	loss_val_diff1 = 0.0
	loss_val_diff2 = 0.0

	loss_abs_val = 0.0
	loss_sta_val = 0.0

	loss_abs = torch.Tensor([0.0]).to(device)
	loss_sta = torch.Tensor([0.0]).to(device)
	loss_cal = torch.Tensor([0.0]).to(device)
	loss_cal_abs = torch.Tensor([0.0]).to(device)
	loss_double_difference = torch.Tensor([0.0]).to(device)
	loss_diff = torch.Tensor([0.0]).to(device)
	anneal_val = anneal_schedule(i)

	for j in range(n_batch):

		# srcs, srcs_cart, nodes_all, node_types, srcs_slice, imatch_srcs, Input, Arrivals, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, ifind_p_edges, ifind_s_edges, Arrivals_hash = to_gpu(inputs[j])
		nodes_all, node_types, srcs_slice, imatch_srcs, Input, Arrivals, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, ifind_p_edges, ifind_s_edges, Arrivals_hash = to_gpu(inputs[j])		

		imatch_srcs = imatch_srcs.cpu().detach().numpy()

		## Find subset of matches
		if use_calibration == True:
			tree_matches = cKDTree(imatch_srcs.reshape(-1,1))
			ip_where = np.where(tree_matches.query(Matches[:,1][:,None])[0] == 0)[0]
			Matches_slice = Matches[ip_where]
			perm_matches = -1*np.ones(len(srcs)).astype('int')
			perm_matches[imatch_srcs] = np.arange(len(imatch_srcs))
			Matches_slice[:,1] = perm_matches[Matches_slice[:,1]]
			Matches_slice = torch.Tensor(Matches_slice).long().to(device)

		if verbose == True:
			print('Number of nodes %d'%A_src_in_prod.shape[1])
			print('Number of prod src src edges %0.2f million'%(A_prod_src_src.shape[1]/1e6))
			print('Number of prod sta sta edges %0.2f million'%(A_prod_sta_sta.shape[1]/1e6))
			# print('Number src edges %d'%A_src_src.shape[1])
			# print('Number sta edges %d'%A_sta_sta.shape[1])

		if inpt_sources == True:

			Input = torch.cat((Input, (torch.Tensor(srcs_slice[:,0:3]).to(device)[A_src_in_sta[1]] - srcs_scale_mean)/srcs_scale_std), dim = 1)

		if use_memory == True:
			inpt_memory_slice = torch.Tensor(srcs_pred_perturb[imatch_srcs]).to(device)/scale_memory
			inpt_memory_slice[torch.isnan(inpt_memory_slice)] = 0.0

		else:
			inpt_memory_slice = False

		# pred, pred_t = m(Input, Input, A_sta_fixed_src, A_src_fixed_sta, A_src_in_product, A_src_in_sta, locs_cart, srcs_slice_cart)
		pred, pred_t, pred_c, pred_mask = m(Input, Input, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta, locs_cart, srcs_slice_cart, memory = inpt_memory_slice)

		if use_sta_corr == False:
			pred_c = torch.zeros(pred_c.shape).to(device)

		if use_buffer == True:

			## Only update buffers for sources with degree > min_degree, in this sub-graph?

			## Update the buffers of sources
			im1, im2 = np.where(np.isnan(srcs_pred_perturb[imatch_srcs,0]) == 0)[0], np.where(np.isnan(srcs_pred_perturb[imatch_srcs,0]) == 1)[0]
			
			## For non-nan entries, take weighted mean value
			srcs_pred_perturb[imatch_srcs[im1],:] = np.sum(np.array([buffer_weight, 1.0 - buffer_weight]).reshape(-1,1,1)*np.concatenate((np.expand_dims(srcs_pred_perturb[imatch_srcs[im1]], axis = 0), np.expand_dims(np.concatenate((pred[im1].cpu().detach().numpy(), pred_t[im1].cpu().detach().numpy().reshape(-1,1)), axis = 1), axis = 0)), axis = 0), axis = 0)

			## For nan-entires, write value
			srcs_pred_perturb[imatch_srcs[im2],:] = np.concatenate((pred[im2].cpu().detach().numpy(), pred_t[im2].cpu().detach().numpy().reshape(-1,1)), axis = 1)


			## Update station corrections

			min_src_per_sta = 5
			pred_c_cpu = pred_c.cpu().detach().numpy()
			iwhere_degree_sta = np.where(degree(A_src_in_sta[0]).cpu().detach().numpy() > min_src_per_sta)[0]
			iwhere1_degree_sta = np.where(corrections_c[iwhere_degree_sta,0] == 0)[0]
			iwhere2_degree_sta = np.where(corrections_c[iwhere_degree_sta,1] == 0)[0]
			iwhere3_degree_sta = np.where(corrections_c[iwhere_degree_sta,0] != 0)[0]
			iwhere4_degree_sta = np.where(corrections_c[iwhere_degree_sta,1] != 0)[0]

			corrections_c[iwhere_degree_sta[iwhere1_degree_sta],0] = pred_c_cpu[iwhere_degree_sta[iwhere1_degree_sta],0]
			corrections_c[iwhere_degree_sta[iwhere2_degree_sta],1] = pred_c_cpu[iwhere_degree_sta[iwhere2_degree_sta],1]
			corrections_c[iwhere_degree_sta[iwhere3_degree_sta],0] = np.sum(np.array([buffer_weight, 1.0 - buffer_weight]).reshape(-1,1)*np.concatenate((np.expand_dims(corrections_c[iwhere_degree_sta[iwhere3_degree_sta],0], axis = 0), np.expand_dims(pred_c_cpu[iwhere_degree_sta[iwhere3_degree_sta],0], axis = 0)), axis = 0), axis = 0)
			corrections_c[iwhere_degree_sta[iwhere4_degree_sta],1] = np.sum(np.array([buffer_weight, 1.0 - buffer_weight]).reshape(-1,1)*np.concatenate((np.expand_dims(corrections_c[iwhere_degree_sta[iwhere4_degree_sta],1], axis = 0), np.expand_dims(pred_c_cpu[iwhere_degree_sta[iwhere4_degree_sta],1], axis = 0)), axis = 0), axis = 0)


		srcs_init = ftrns2_diff(srcs_slice_cart) # .cpu().detach().numpy()
		trv_out_initial = trv_pairwise(locs_cuda[A_src_in_sta[0]], srcs_init[A_src_in_sta[1]]) # [:, 0]

		srcs_perturbed = ftrns2_diff(srcs_slice_cart + pred) # .cpu().detach().numpy()
		trv_out_perturbed = trv_pairwise(locs_cuda[A_src_in_sta[0]], srcs_perturbed[A_src_in_sta[1]]) + pred_c[A_src_in_sta[0]] + pred_t[A_src_in_sta[1]] # [:, 0]
		# trv_out_perturbed_s = trv_pairwise(locs_cuda[A_src_in_sta[0]], ftrns2_diff(srcs_slice_cart[src_s_slice_edge1]))[:, 1]

		if use_calibration == True:

			ip_where_query = tree_matches.query(A_src_in_sta[1].cpu().detach().numpy()[:,None])
			ip_where = torch.Tensor(np.where(ip_where_query[0] == 0)[0]).long().to(device)

			## For each matches source, find where source appears in Matches_slice (and if it appears in Matches_slice)
			tree_matches_find = cKDTree(Matches_slice[:,1].cpu().detach().numpy()[:,None])
			tree_find_matches = tree_matches_find.query(A_src_in_sta[1][ip_where].cpu().detach().numpy().reshape(-1,1))
			ifind_match = np.where(tree_find_matches[0] == 0)[0]

			## Select a subset of ifind_match
			src_ref_ind = Matches_slice[tree_find_matches[1][ifind_match],0]

			trv_out_calibration_pred = trv_pairwise(locs_cuda[A_src_in_sta[0][ip_where[ifind_match]]], srcs_perturbed[A_src_in_sta[1][ip_where[ifind_match]]]) + pred_c[A_src_in_sta[0][ip_where[ifind_match]]] + pred_t[A_src_in_sta[1][ip_where[ifind_match]]] # [:, 0]
			trv_out_calibration_target = trv_pairwise(locs_cuda[A_src_in_sta[0][ip_where[ifind_match]]], torch.Tensor(srcs_ref).to(device)[src_ref_ind]) + pred_c[A_src_in_sta[0][ip_where[ifind_match]]] + (torch.Tensor(srcs_ref).to(device)[src_ref_ind,3].reshape(-1,1) - torch.Tensor(srcs_slice).to(device)[A_src_in_sta[1][ip_where[ifind_match]],3].reshape(-1,1)) # [:, 0]
			loss_cal_abs = loss_func(trv_out_calibration_pred, trv_out_calibration_target)/n_batch

			ip1_cal = torch.where(Arrivals[ip_where[ifind_match],3] == 1)[0]
			is1_cal = torch.where(Arrivals[ip_where[ifind_match],4] == 1)[0]

			Res_cal_p = Arrivals[ip_where[torch.Tensor(ifind_match).long().to(device)[ip1_cal]],0] - trv_out_calibration_target[ip1_cal,0]
			Res_cal_s = Arrivals[ip_where[torch.Tensor(ifind_match).long().to(device)[is1_cal]],1] - trv_out_calibration_target[is1_cal,1]
			loss_p_cal = loss_func(Res_cal_p, torch.zeros(Res_cal_p.shape).to(device))
			# loss_s_cal = weight_s_loss*loss_func(Res_cal_s, torch.zeros(Res_cal_s.shape).to(device))
			loss_s_cal = loss_func(Res_cal_s, torch.zeros(Res_cal_s.shape).to(device))
			loss_cal = (0.5*loss_p_cal + 0.5*loss_s_cal)/n_batch

			## Produces loss_cal and loss_cal_abs


		if use_absolute == True:

			ip1 = torch.where(Arrivals[:,3] == 1)[0]
			is1 = torch.where(Arrivals[:,4] == 1)[0]

			assert(abs(Arrivals[:,2] - A_src_in_sta[0]).max().item() == 0)
			assert(abs((Arrivals[ip1,0] - trv_out_initial[ip1,0]) - Input[ip1,0]).max() < 1e-1) ## Residuals match the input residual for unpertubed data
			assert(abs((Arrivals[is1,1] - trv_out_initial[is1,1]) - Input[is1,1]).max() < 1e-1)
			# assert(abs(Arrivals[:,5] - A_src_in_sta_abs[1]).max().item() == 0)

			imatch1 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[ip1])
			imatch2 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[is1])

			scale_dist1 = torch.norm(srcs_cart[unique_picks[imatch1,1],0:3]/(1000.0*10.0) - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0), dim = 1, keepdim = True)
			scale_dist2 = torch.norm(srcs_cart[unique_picks[imatch2,1],0:3]/(1000.0*10.0) - locs_cart[unique_picks[imatch2,0]]/(1000.0*10.0), dim = 1, keepdim = True)

			log_var1 = m.forward_picks_abs(imatch1, unique_picks[imatch1,1], unique_picks[imatch1,0], scale_dist1, ftrns2_diff(srcs_cart[unique_picks[imatch1,1],0:3])[:,2].reshape(-1,1)/(1000.0*10.0), torch.zeros(len(imatch1)).long().to(device))[:,0]
			log_var2 = m.forward_picks_abs(imatch2 + m.num_picks, unique_picks[imatch2,1], unique_picks[imatch2,0], scale_dist2, ftrns2_diff(srcs_cart[unique_picks[imatch2,1],0:3])[:,2].reshape(-1,1)/(1000.0*10.0), torch.ones(len(imatch2)).long().to(device))[:,0]


			b_scale1 = torch.exp(0.5 * log_var1)
			b_scale2 = torch.exp(0.5 * log_var2)

			Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
			Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]




			loss_res_p = loss_func(Res_p)
			loss_res_s = loss_func(Res_s)

			loss_p_abs = (loss_res_p/b_scale1.detach() + loss_res_p.detach()/b_scale1 + (anneal_val + lam_weight_abs)*0.5*log_var1).mean() # /n_batch
			loss_s_abs = (loss_res_s/b_scale2.detach() + loss_res_s.detach()/b_scale2 + (anneal_val + lam_weight_abs)*0.5*log_var2).mean() # /n_batch


			# loss_abs_val += (0.5*(loss_func(Res_p)/b_scale1).detach().mean() + 0.5*(loss_func(Res_s)/b_scale2).detach().mean()).item()/n_batch ## Expensive for logging
			loss_abs_val += (0.5*loss_func(Res_p).detach().mean() + 0.5*loss_func(Res_s).detach().mean()).item()/n_batch ## Expensive for logging

			loss_abs = (0.5*loss_p_abs + 0.5*loss_s_abs) # /n_batch


		if use_sta_corr == True:

			if use_absolute == False:

				ip1 = torch.where(Arrivals[:,3] == 1)[0]
				is1 = torch.where(Arrivals[:,4] == 1)[0]

				# imatch1 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[ip1])
				# imatch2 = torch.searchsorted(sorted_hash_picks, Arrivals_hash[is1])

				assert(abs(Arrivals[:,2] - A_src_in_sta[0]).max().item() == 0)
				assert(abs((Arrivals[ip1,0] - trv_out_initial[ip1,0]) - Input[ip1,0]).max() < 1e-1) ## Residuals match the input residual for unpertubed data
				assert(abs((Arrivals[is1,1] - trv_out_initial[is1,1]) - Input[is1,1]).max() < 1e-1)
				# assert(abs(Arrivals[:,5] - A_src_in_sta_abs[1]).max().item() == 0)

				Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
				Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]



			log_var1 = m.forward_picks_sta(torch.arange(m.num_stas).long().to(device), torch.zeros(m.num_stas).long().to(device))[:,0]
			log_var2 = m.forward_picks_sta(torch.arange(m.num_stas).long().to(device), torch.ones(m.num_stas).long().to(device))[:,1] ## Using second channel output

			# b_scale1 = torch.exp(0.5 * log_var1)
			# b_scale2 = torch.exp(0.5 * log_var2)

			b_scale1 = torch.exp(0.5 * log_var1).clamp(min = m.min_sta_sigma)
			b_scale2 = torch.exp(0.5 * log_var2).clamp(min = m.min_sta_sigma)


			Res_sta_p = global_mean_pool(Res_p, A_src_in_sta[0,ip1]) # + global_mean_pool(0.5*log_var1, A_src_in_sta[0,ip1])
			Res_sta_s = global_mean_pool(Res_s, A_src_in_sta[0,is1]) # + global_mean_pool(0.5*log_var2, A_src_in_sta[0,is1])


			loss_res_p = loss_func(Res_sta_p)
			loss_res_s = loss_func(Res_sta_s)

			loss_p_sta = (loss_res_p/b_scale1.detach() + loss_res_p.detach()/b_scale1 + (anneal_val + lam_weight_sta)*torch.log(b_scale1)).mean()
			loss_s_sta = (loss_res_s/b_scale2.detach() + loss_res_s.detach()/b_scale2 + (anneal_val + lam_weight_sta)*torch.log(b_scale2)).mean()

			loss_sta_val += (0.5*loss_func(Res_sta_p).detach().mean() + 0.5*loss_func(Res_sta_s).detach().mean()).item()/n_batch


			# sigma_sta = min_sigma_sta + torch.softplus(torch.exp(log_sigma_sta) - min_sigma_sta)


			loss_sta = (0.5*loss_p_sta + 0.5*loss_s_sta) # /n_batch

			## Produces loss_sta

		#### Compute double difference residual ######

		# assert(node_types[A_src_in_sta[1][A_prod_src_src[:,ifind_edges]].cpu().detach().numpy()].max() == 1) ## Level two nodes are not compared against
		# assert(Residuals[node_])

		if use_double_diff == True:

			p_slice_edge1 = A_prod_src_src[0,ifind_p_edges]
			p_slice_edge2 = A_prod_src_src[1,ifind_p_edges]
			s_slice_edge1 = A_prod_src_src[0,ifind_s_edges]
			s_slice_edge2 = A_prod_src_src[1,ifind_s_edges]

			sta_p_slice_edge1 = A_src_in_sta[0,p_slice_edge1]
			sta_p_slice_edge2 = A_src_in_sta[0,p_slice_edge2]
			sta_s_slice_edge1 = A_src_in_sta[0,s_slice_edge1]
			sta_s_slice_edge2 = A_src_in_sta[0,s_slice_edge2]

			src_p_slice_edge1 = A_src_in_sta[1,p_slice_edge1]
			src_p_slice_edge2 = A_src_in_sta[1,p_slice_edge2]
			src_s_slice_edge1 = A_src_in_sta[1,s_slice_edge1]
			src_s_slice_edge2 = A_src_in_sta[1,s_slice_edge2]

			Trgts_p = Input[p_slice_edge1,0] - Input[p_slice_edge2,0]
			Trgts_s = Input[s_slice_edge1,1] - Input[s_slice_edge2,1]
			
			## Use all samples for residual (or remove some samples)
			i1 = np.arange(len(Trgts_p))
			i2 = np.arange(len(Trgts_s))

			## Initial (or pre-compute these)
			trv_out_perturbed_i_p_initial = trv_pairwise(locs_cuda[sta_p_slice_edge1], ftrns2_diff(srcs_slice_cart[src_p_slice_edge1]))[:, 0]
			trv_out_perturbed_j_p_initial = trv_pairwise(locs_cuda[sta_p_slice_edge2], ftrns2_diff(srcs_slice_cart[src_p_slice_edge2]))[:, 0]

			trv_out_perturbed_i_s_initial = trv_pairwise(locs_cuda[sta_s_slice_edge1], ftrns2_diff(srcs_slice_cart[src_s_slice_edge1]))[:, 1]
			trv_out_perturbed_j_s_initial = trv_pairwise(locs_cuda[sta_s_slice_edge2], ftrns2_diff(srcs_slice_cart[src_s_slice_edge2]))[:, 1]

			trv_out_perturbed_i_p = trv_pairwise(locs_cuda[sta_p_slice_edge1], ftrns2_diff(srcs_slice_cart[src_p_slice_edge1] + pred[src_p_slice_edge1]))[:, 0] + pred_t[src_p_slice_edge1,0] + pred_c[sta_p_slice_edge1,0]
			trv_out_perturbed_j_p = trv_pairwise(locs_cuda[sta_p_slice_edge2], ftrns2_diff(srcs_slice_cart[src_p_slice_edge2] + pred[src_p_slice_edge2]))[:, 0] + pred_t[src_p_slice_edge2,0] + pred_c[sta_p_slice_edge2,0]

			trv_out_perturbed_i_s = trv_pairwise(locs_cuda[sta_s_slice_edge1], ftrns2_diff(srcs_slice_cart[src_s_slice_edge1] + pred[src_s_slice_edge1]))[:, 1] + pred_t[src_s_slice_edge1,0] + pred_c[sta_s_slice_edge1,1]
			trv_out_perturbed_j_s = trv_pairwise(locs_cuda[sta_s_slice_edge2], ftrns2_diff(srcs_slice_cart[src_s_slice_edge2] + pred[src_s_slice_edge2]))[:, 1] + pred_t[src_s_slice_edge2,0] + pred_c[sta_s_slice_edge2,1]

			Res_i_p = (trv_out_perturbed_i_p - trv_out_perturbed_i_p_initial)
			Res_j_p = (trv_out_perturbed_j_p - trv_out_perturbed_j_p_initial)

			Res_i_s = (trv_out_perturbed_i_s - trv_out_perturbed_i_s_initial)
			Res_j_s = (trv_out_perturbed_j_s - trv_out_perturbed_j_s_initial)

			Pred_p = Res_i_p - Res_j_p
			Pred_s = Res_i_s - Res_j_s

			loss_p_double_difference = loss_func(Pred_p[i1], Trgts_p[i1])
			# loss_s_double_difference = weight_s_loss*loss_func(Pred_s[i2], Trgts_s[i2])
			loss_s_double_difference = loss_func(Pred_s[i2], Trgts_s[i2])
			loss_double_difference = (0.5*loss_p_double_difference + 0.5*loss_s_double_difference)/n_batch


		## Add loss related to the (cross-correlation) differential times themselves
		if use_diff == True:

			islice_edges, values_slice = subgraph(torch.Tensor(imatch_srcs).to(device).long(), source_indices, merged_values)
			n_edges_slice = islice_edges.shape[1]
			## Inside values_slice: (first column phase type, second column station index, third column weight)
			## Note: only compare constant phase types

			imatch1 = torch.searchsorted(sorted_hash_picks, hash_rows(torch.cat((values_slice[:,1].reshape(-1,1), islice_edges[0].reshape(-1,1)), dim = 1)))
			imatch2 = torch.searchsorted(sorted_hash_picks, hash_rows(torch.cat((values_slice[:,1].reshape(-1,1), islice_edges[1].reshape(-1,1)), dim = 1)))

			# forward_picks(self, pick_id, src_coord, sta_coord, src_id, phase_type)

			# log_var1 = m.forward_picks(imatch1 + values_slice[:,0].long()*m.num_picks, torch.zeros(len(imatch1)).long().to(device))[:,0]
			# log_var2 = m.forward_picks(imatch2 + values_slice[:,0].long()*m.num_picks, torch.ones(len(imatch2)).long().to(device))[:,0]

			# pick_id1, picks_id2, src_id1, src_id2, sta_id1, sta_id2, dist_coord1, dist_coord2, dist_coord3, phase_type
			v1 = srcs_cart[unique_picks[imatch1,1]]/(1000.0*10.0) - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0)
			v2 = srcs_cart[unique_picks[imatch2,1]]/(1000.0*10.0) - locs_cart[unique_picks[imatch2,0]]/(1000.0*10.0)
			scale_dist1 = torch.norm(v1, dim = 1, keepdim = True)
			scale_dist2 = torch.norm(v2, dim = 1, keepdim = True)
			scale_dist3 = torch.norm(srcs_cart[unique_picks[imatch1,1]]/(1000.0*10.0) - srcs_cart[unique_picks[imatch2,1]]/(1000.0*10.0), dim = 1, keepdim = True)
			src_depth1 = ftrns2_diff(srcs_cart[unique_picks[imatch1,1]])[:,2].reshape(-1,1)/(1000.0*10.0) # - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0)
			src_depth2 = ftrns2_diff(srcs_cart[unique_picks[imatch2,1]])[:,2].reshape(-1,1)/(1000.0*10.0) # - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0)
			cos_sim = nn.functional.cosine_similarity(v1, v2, dim = 1).unsqueeze(1)

			# scale_dist1 = torch.norm(srcs_cart[unique_picks[imatch1,1]]/(1000.0*10.0) - locs_cart[unique_picks[imatch1,0]]/(1000.0*10.0), dim = 1, keepdim = True)
			# scale_dist2 = torch.norm(srcs_cart[unique_picks[imatch2,1]]/(1000.0*10.0) - locs_cart[unique_picks[imatch2,0]]/(1000.0*10.0), dim = 1, keepdim = True)
			## Note: not using the source or station ids

			## Can use weight as well if prefered
			log_var_merged = m.forward_picks_diff(imatch1 + m.num_picks*values_slice[:,0].long(), imatch2 + m.num_picks*values_slice[:,0].long(), unique_picks[imatch1,1], unique_picks[imatch2,1], unique_picks[imatch1,0], unique_picks[imatch2,0], scale_dist1, scale_dist2, scale_dist3, src_depth1, src_depth2, cos_sim, values_slice[:,0].long())[:,0]
			# log_var2 = m.forward_picks_diff(imatch2 + m.num_picks*values_slice[:,0].long(), unique_picks[imatch2,1], unique_picks[imatch2,0], srcs_cart[unique_picks[imatch2,1]]/(1000.0*5.0), locs_cart[unique_picks[imatch2,0]]/(1000.0*5.0), torch.ones(len(imatch2)).long().to(device))[:,0]

			## Can add regularization term (e.g., deviation of difference std. from added variances)
			## Add a regularization term: λ * |log(σ_diff_ij) - 0.5 * log(σ_i² + σ_j²)| to encourage proximity to the independent sum, while allowing deviation for cancellation.

			# log_var1 = m.forward_picks(imatch1 + values_slice[:,0].long()*m.num_picks, torch.zeros(len(imatch1)).long().to(device))[:,0]
			# log_var2 = m.forward_picks(imatch2 + values_slice[:,0].long()*m.num_picks, torch.ones(len(imatch2)).long().to(device))[:,0]
			# log_var_merged = torch.logaddexp(log_var1, log_var2)

			b_scale = torch.exp(0.5 * log_var_merged)

			# var_merged = torch.exp(log_var_merged)

			# harm_mean_sigma = 2.0*sigma1*sigma2/(sigma1 + sigma2)

			perm_vec = -1*torch.ones(len(srcs)).long().to(device)
			perm_vec[imatch_srcs] = torch.arange(len(imatch_srcs)).to(device)
			src_slice_edge1 = perm_vec[islice_edges[0]]
			src_slice_edge2 = perm_vec[islice_edges[1]]

			assert(torch.min(torch.Tensor([src_slice_edge1.min(), src_slice_edge2.min()])) > -1) # , src_s_slice_edge1.min(), src_s_slice_edge2.min()])) > -1)


			trv_out_perturbed_i = trv_pairwise(locs_cuda[values_slice[:,1].long()], srcs_perturbed[src_slice_edge1])[torch.arange(n_edges_slice), values_slice[:,0].long()] + pred_t[src_slice_edge1,0] + pred_c[values_slice[:,1].long(),values_slice[:,0].long()]
			trv_out_perturbed_j = trv_pairwise(locs_cuda[values_slice[:,1].long()], srcs_perturbed[src_slice_edge2])[torch.arange(n_edges_slice), values_slice[:,0].long()] + pred_t[src_slice_edge2,0] + pred_c[values_slice[:,1].long(),values_slice[:,0].long()]

			trgt_diff = values_slice[:,3]
			pred_diff = trv_out_perturbed_i - trv_out_perturbed_j


			# if i < n_iter_initilize:
			# 	# res_aggregate = scatter(trgt_diff - pred_diff, values_slice[:,0].long(), reduce = 'mean', dim = 0, dim_size = 2)
			# 	res_aggregate = global_mean_pool(loss_func(trgt_diff - pred_diff), values_slice[:,0].long())
			# 	res_loss_p_diff.append(res_aggregate[0].item())
			# 	res_loss_s_diff.append(res_aggregate[1].item())


			# if use_huber == False:

			# 	abs_error = values_slice[:,2]*torch.abs(trgt_diff - pred_diff) # Weighted
			# 	# loss_diff = ((abs_error / b_scale) + 0.5*log_var_merged).mean()/n_batch

			# else:

			abs_error = (values_slice[:,2].sqrt())*loss_func(trgt_diff - pred_diff) # Weighted
			# loss_diff = ((abs_error / b_scale) + 0.5*log_var_merged).mean()/n_batch

			## Differential loss
			loss_diff = ((abs_error / b_scale.detach()) + (abs_error.detach() / b_scale) + (anneal_val + lam_weight_diff)*0.5*log_var_merged).mean() # /n_batch

			## Merge loss values
			loss_diff_val1 = loss_diff.item()/n_batch
			loss_diff_val2 = abs_error.detach().mean().item()/n_batch

			# loss_diff = (weight_vec_phase*values_slice[:,2]*torch.abs(trgt_diff - pred_diff)).mean()/n_batch # (0.5*loss_p_diff + 0.5*loss_s_diff)/n_batch			


		## Just direct weighting and merge
		
		# loss = loss_diff + loss_abs + loss_sta

		use_loss_balancer = False
		if use_uncertainty_weighting == False:

			loss = (loss_diff + 0.25*loss_abs + 0.15*loss_sta)/n_batch

		elif use_loss_balancer == True:

			## Compute base losses
			loss_dict = {
			'loss_diff': loss_diff,
			'loss_abs': loss_abs,
			'loss_sta': loss_sta,
			}

			loss = LossBalancer(loss_dict, accum_steps = n_batch, is_last_accum_step = (inc == (n_batch - 1))) # losses_dict: dict, accum_steps: int = None, is_last_accum_step: bool = False
			loss = loss/n_batch

		elif use_uncertainty_weighting == True:

			loss = m.UncertaintyWeighting(loss_abs, loss_diff, loss_sta)/n_batch

			if (np.mod(i, 100) == 0)*(j == 0):
				# print('\n Weights (abs, diff, sta): %0.8f, %0.8f, %0.8f \n'%(m.UncertaintyWeighting.log_weight_abs.item(), 0.0, m.UncertaintyWeighting.log_weight_sta.item()))
				print('\n Weights (abs, diff, sta): %0.8f, %0.8f, %0.8f \n'%(1.0, m.UncertaintyWeighting.log_weight_diff.item(), m.UncertaintyWeighting.log_weight_sta.item()))


		loss_val += loss.item()
		loss_val_abs += loss_abs.item()
		loss_val_sta += loss_sta.item()
		loss_val_cal += loss_cal.item()
		loss_val_cal_abs += loss_cal_abs.item()
		loss_val_double_diff += loss_double_difference.item()
		# loss_val_diff += loss_diff.item()
		loss_val_diff1 += loss_diff_val1
		loss_val_diff2 += loss_diff_val2

		# loss = 0.5*loss + 0.5*loss_sta + 0.25*loss_cal + 0.25*loss_cal_abs


		# if i >= n_iter_initilize:

		if j != (n_batch - 1):
			loss.backward(retain_graph = True) ## Why must this be done?
		else:
			loss.backward(retain_graph = False) ## Why must this be done?

		# cnt_file += 1

	## Update model

	losses.append(loss_val)
	losses_abs.append(loss_val_abs)
	losses_sta.append(loss_val_sta)
	losses_cal.append(loss_val_cal)
	losses_cal_abs.append(loss_val_cal_abs)
	losses_double_diff.append(loss_val_double_diff)
	losses_diff.append(loss_val_diff1)
	# losses_diff2.append(loss_val_diff2)


	# if i >= n_iter_initilize:
	optimizer.step()

	loss_print = '%d %0.8f, %0.8f, %0.8f, Abs: %0.8f, %0.8f \n'%(i, loss_val, loss_val_diff1, loss_val_diff2, loss_abs_val, loss_sta_val)

	print(loss_print)
	log_buffer.append(loss_print)

	write_training_file = path_save + 'output_ver_%d.txt'%n_ver_save
	if np.mod(i, n_flush) == 0:
		with open(write_training_file, 'a') as text_file:
			for log in log_buffer:
				# text_file.write('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f [%0.5f, %0.5f, %0.5f, %0.5f, %0.5f] (reg %0.8f) \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4, loss_src_val, loss_asc_val, loss_negative_val, loss_cap_val, loss_consistency_val, (10e4)*loss_regularize_val))
				text_file.write(log)
		log_buffer.clear()


	remove_samples = False

	if remove_samples == True:

		error('Not implemented')
		## Change this so it's not based on single predictions, but a "buffer" saved over picks or sources,
		## accumulated over several batches

		if (np.mod(i, 500) == 0)*(i > 0):
			i1p = np.where(np.abs(Pred_p[i1].cpu().detach().numpy() - Trgts_p[i1].cpu().detach().numpy()) < np.quantile(np.abs(Pred_p[i1].cpu().detach().numpy() - Trgts_p[i1].cpu().detach().numpy()), 0.998))[0]
			i2p = np.where(np.abs(Pred_s[i2].cpu().detach().numpy() - Trgts_s[i2].cpu().detach().numpy()) < np.quantile(np.abs(Pred_s[i2].cpu().detach().numpy() - Trgts_s[i2].cpu().detach().numpy()), 0.998))[0]
			i1 = i1[i1p]
			i2 = i2[i2p]


