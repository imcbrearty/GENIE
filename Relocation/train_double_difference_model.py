
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Note: there is already a way to build the subgraphs in automatic differentation and direct training script, but those ways are
## somewhat complex. This is to simplify it and build the "full" dense graph (not k-nn sampled), and then sample from it.

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
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.cluster import SpectralClustering
from torch.autograd import Variable
from numpy.matlib import repmat
from torch_geometric.utils import degree
import pathlib
import itertools
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


day_select = int(argvs[1])
offset_select = int(argvs[2])

print('name of program is %s'%argvs[0])
print('day is %s'%argvs[1])

### Settings: ###

with open('calibration_config.yaml', 'r') as file:
    calibration_config = yaml.safe_load(file)

with open('process_config.yaml', 'r') as file:
    process_config = yaml.safe_load(file)

## Load device
device = calibration_config['device']

## Load Processing settings
n_ver_load = process_config['n_ver_load']
n_step_load = process_config['n_step_load']
n_save_ver = process_config['n_save_ver']
n_ver_picks = process_config['n_ver_picks']

template_ver = process_config['template_ver']
vel_model_ver = process_config['vel_model_ver']
process_days_ver = process_config['process_days_ver']

offset_increment = process_config['offset_increment']
n_rand_query = process_config['n_rand_query']
n_query_grid = process_config['n_query_grid']

thresh = process_config['thresh'] # Threshold to declare detection
thresh_assoc = process_config['thresh_assoc'] # Threshold to declare src-arrival association
spr = process_config['spr'] # Sampling rate to save temporal predictions
tc_win = process_config['tc_win'] # Temporal window (s) to link events in Local Marching
sp_win = process_config['sp_win'] # Distance (m) to link events in Local Marching
break_win = process_config['break_win'] # Temporal window to find disjoint groups of sources, 
## so can run Local Marching without memory issues.
spr_picks = process_config['spr_picks'] # Assumed sampling rate of picks 
## (can be 1 if absolute times are used for pick time values)

d_win = process_config['d_win'] ## Lat and lon window to re-locate initial source detetections with refined sampling over
d_win_depth = process_config['d_win_depth'] ## Depth window to re-locate initial source detetections with refined sampling over
dx_depth = process_config['dx_depth'] ## Depth resolution to locate events with travel time based re-location

step = process_config['step']
step_abs = process_config['step_abs']

cost_value = process_config['cost_value'] # If use expanded competitve assignment, then this is the fixed cost applied per source


compute_magnitudes = process_config['compute_magnitudes']
min_log_amplitude_val = process_config['min_log_amplitude_val']
process_known_events = process_config['process_known_events']
use_expanded_competitive_assignment = process_config['use_expanded_competitive_assignment']
use_differential_evolution_location = process_config['use_differential_evolution_location']

print('Beginning calibration')
### Begin automated processing ###

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

name_of_project = config['name_of_project']

use_physics_informed = config['use_physics_informed']
use_subgraph = config['use_subgraph']
if use_subgraph == True:
	max_deg_offset = config['max_deg_offset']
	k_nearest_pairs = config['k_nearest_pairs']

# Load region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load templates
z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
x_grids = z['x_grids']
z.close()

# Load stations
z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
z.close()

## Create path to write files
write_training_file = path_to_file + 'GNN_TrainedModels/' + name_of_project + '_'

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

if (use_differential_evolution_location == False)*(config['train_travel_time_neural_network'] == False):
	hull = ConvexHull(X)
	hull = hull.points[hull.vertices]
else:
	hull = []

## Load calibration config parameters

interp_type = calibration_config['interp_type'] ## Type of spatial interpolation (mean, weighted, anisotropic)
k_spc_lap = calibration_config['k_spc_lap'] ## k-nn value for laplacian smoothing
k_spc_interp = calibration_config['k_spc_interp'] ## k-nn value for interpolation
grid_index = calibration_config['grid_index'] ## grid index choice (of x_grids)

n_ver_events = calibration_config['n_ver_events']
n_ver_reference = calibration_config['n_ver_reference']
n_ver_save = calibration_config['n_ver_save']
interp_type = calibration_config['interp_type'] ## Type of spatial interpolation (mean, weighted, anisotropic)
k_spc_lap = calibration_config['k_spc_lap'] ## k-nn value for laplacian smoothing
k_spc_interp = calibration_config['k_spc_interp'] ## k-nn value for interpolation
sig_ker = calibration_config['sig_ker'] ## spatial kernel (in km) of the weighting kernel (fixed for weighted, starting value for anisotropic) 
grid_index = calibration_config['grid_index'] ## grid index choice (of x_grids)
n_batch = calibration_config['n_batch'] ## 1000 earthquakes per batch
n_updates = calibration_config['n_updates'] ## Find automatic convergence criteria

temporal_match =  calibration_config['temporal_match'] ## window for matched events (seconds)
spatial_match = calibration_config['spatial_match'] ## spatial distance for matched events (m)
min_picks = calibration_config['min_picks'] ## minimum number of total picks to use event in calibration
min_threshold = calibration_config['min_threshold'] ## minimum detection threshold value to use event in calibration
compute_relocations = calibration_config['compute_relocations'] ## Compute example event relocations with travel time corrections
n_relocations = calibration_config['n_relocations'] ## Number of example event relocations with travel time corrections
save_with_data = calibration_config['save_with_data'] ## Flag whether to save data with the calibration file

## Apply input

class DataAggregation(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, scale_rel = 30.0, n_dim = 3, n_dim_mask = 2, ndim_proj = 3):
		super(DataAggregation, self).__init__('mean') # node dim
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
	def __init__(self, ndim_in, ndim_out, ndim_mask = 11, ndim_edges = 3, scale_rel = 30e3):
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
	def __init__(self, ndim_in, ndim_out, ndim_mask = 11, ndim_edges = 3, scale_rel = 30e3):
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

	def __init__(self, ftrns1, ftrns2, inpt_sources = True, use_sta_corr = True, use_memory = False, use_mask = False, use_aggregation = True, use_attention = False, n_inpt = 15, n_mask = 15, n_hidden = 20, n_embed = 10, scale_fixed = 5000.0, device = 'cuda'):

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
		self.proj_c = nn.Sequential(nn.Linear(15, 15), nn.PReLU(), nn.Linear(15, 2))

		if use_mask == True:
			self.proj_mask = nn.Sequential(nn.Linear(30, 15), nn.PReLU(), nn.Linear(15, 2))

		self.use_memory = use_memory
		self.use_sta_corr = use_sta_corr
		self.scale = torch.Tensor([scale_fixed]).to(device)
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


n_batch = 3
n_epochs = 50001
verbose = False
n_ver_save = 1


inpt_sources = True
use_double_diff = True
use_absolute = True
use_sta_corr = True
use_calibration = False
use_mask = False
use_diff = False
use_memory = True


n_ver_load_files = 1
path_save = path_to_file + 'DoubleDifferenceModels/'
path_data = path_to_file + 'DoubleDifferenceData/'
assert((use_double_diff + use_absolute + use_sta_corr + use_calibration + use_diff) > 0)

m = GNN_Location(ftrns1, ftrns2, inpt_sources = inpt_sources, use_sta_corr = use_sta_corr, use_memory = use_memory, use_mask = use_mask, use_aggregation = False, use_attention = False, device = device).to(device)

optimizer = optim.Adam(m.parameters(), lr = 0.001)

loss_func = nn.L1Loss()

losses = []
losses_abs = []
losses_sta = []
losses_cal = []
losses_cal_abs = []
losses_double_diff = []
losses_diff = []


st_files = glob.glob(path_data + '*ver_%d.hdf5'%n_ver_load_files)


n_repeats = int(np.floor(n_epochs*n_batch/len(st_files))) + 1
ind_samples = np.hstack([np.random.permutation(len(st_files)) for j in range(n_repeats)])
cnt_file = 0


z = h5py.File(st_files[ind_samples[cnt_file]], 'r')
srcs_ref = z['srcs_ref'][:]
Matches = z['Matches'][:]
Arrivals = z['Arrivals'][:]
z.close()

if len(Matches) == 0:
	if use_calibration == True:
		print('Note, no matches loaded, so not using calibration loss')
		use_calibration = False

z = h5py.File(st_files[ind_samples[cnt_file]], 'r')
srcs = z['srcs'][:]
srcs_fixed = np.copy(srcs)
tree_srcs = cKDTree(ftrns1(srcs[:,0:3]))

if inpt_sources == True:
	srcs_scale_mean = torch.Tensor(srcs[:,0:3].min(0, keepdims = True)).to(device)
	srcs_scale_std = torch.Tensor(srcs[:,0:3].max(0, keepdims = True) - srcs[:,0:3].min(0, keepdims = True)).to(device)

locs_cuda = torch.Tensor(locs).to(device)
srcs_cuda = torch.Tensor(srcs).to(device)
locs_cart = torch.Tensor(ftrns1(locs)).to(device)
srcs_cart = torch.Tensor(ftrns1(srcs)).to(device)

## Setup buffer variables
buffer_weight = 0.98 ## This fraction old data, other fraction new data
buffer_window = 10

weight_ratio = 0.8 ## 0.8 double difference, remaining all others
weight_s_loss = 0.5 ## Relative weight of P versus S


corrections_c = np.zeros((len(locs),2))

srcs_pred_perturb = np.nan*np.zeros((len(srcs),4))
srcs_pred = np.nan*np.zeros((len(srcs),4))
srcs_pred_std = np.nan*np.zeros((buffer_window, len(srcs), 4))
use_buffer = True

scale_memory = torch.Tensor([5000.0, 5000.0, 5000.0, 5.0]).reshape(1,-1).to(device) ## Can set scale dependent parameters for this (and for convolutions inside GNN layers)



## Also measure moving residual of each source and pick

load_model = True
if load_model == True:

	n_restart_step = 16000
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


for i in range(n_restart_step, n_epochs):

	## Should sub-select the active sources with non-zero predictions

	## Must map output predictions through pairwise travel time neural network, to compute loss on pairwise residuals
	## Should map subset of sources to permuted indices (so Bipartite read-in does not populate all values)

	## Need to find high (double difference) residual samples. Possibly predict them.

	## Should sub-select which set of predictions (edges beteen A_src_fixed_sta)

	## Add absolute information to graphs, make bipartite layer an attention mechanism

	optimizer.zero_grad()

	loss_val = 0.0
	loss_val_abs = 0.0
	loss_val_sta = 0.0
	loss_val_cal = 0.0
	loss_val_cal_abs = 0.0
	loss_val_double_diff = 0.0
	loss_val_diff = 0.0

	loss_abs = torch.Tensor([0.0]).to(device)
	loss_sta = torch.Tensor([0.0]).to(device)
	loss_cal = torch.Tensor([0.0]).to(device)
	loss_cal_abs = torch.Tensor([0.0]).to(device)
	loss_double_difference = torch.Tensor([0.0]).to(device)
	loss_diff = torch.Tensor([0.0]).to(device)

	for j in range(n_batch):

		## Load sample

		if cnt_file != 0:

			z = h5py.File(st_files[ind_samples[cnt_file]], 'r')

		srcs = z['srcs'][:]
		srcs_cart = torch.Tensor(ftrns1(srcs)).to(device)
		nodes_all = z['nodes_all'][:] # nodes_all, node_types, srcs_slice, A_src_src, A_src_src_abs, ifind_edges
		node_types = z['node_types'][:]
		srcs_slice = z['srcs_slice'][:]
		imatch_srcs = z['imatch_srcs'][:]

		# srcs_slice_cart = torch.Tensor(z['srcs_slice_cart'][:]).to(device)
		# locs_cart = torch.Tensor(z['locs_cart'][:]).to(device)
		# z['Data_extract'] = Data_extract
		# count_sta = z['count_sta'][:]

		Input = torch.Tensor(z['Input'][:]).to(device)
		Arrivals = torch.Tensor(z['Arrivals'][:]).to(device)
		# A_sta_sta = torch.Tensor(z['A_sta_sta']).long().to(device)
		# A_src_src = torch.Tensor(z['A_src_src']).long().to(device)
		A_prod_sta_sta = torch.Tensor(z['A_prod_sta_sta'][:]).long().to(device)
		A_prod_src_src = torch.Tensor(z['A_prod_src_src'][:]).long().to(device)
		A_src_in_prod = torch.Tensor(z['A_src_in_prod'][:]).long().to(device)
		A_sta_in_prod = torch.Tensor(z['A_sta_in_prod'][:]).long().to(device)
		# A_src_in_prod_x = z['A_src_in_prod_x'][:]
		A_src_in_sta = torch.Tensor(z['A_src_in_sta'][:]).long().to(device)
		locs_cart = torch.Tensor(z['locs_cart'][:]).to(device)
		srcs_slice_cart = torch.Tensor(z['srcs_slice_cart'][:]).to(device)

		ifind_p_edges = torch.Tensor(z['ifind_p_edges'][:]).long().to(device)
		ifind_s_edges = torch.Tensor(z['ifind_s_edges'][:]).long().to(device)
		z.close()

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

		# moi

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
			loss_s_cal = weight_s_loss*loss_func(Res_cal_s, torch.zeros(Res_cal_s.shape).to(device))
			loss_cal = (0.5*loss_p_cal + 0.5*loss_s_cal)/n_batch

			## Produces loss_cal and loss_cal_abs


		if use_absolute == True:

			ip1 = torch.where(Arrivals[:,3] == 1)[0]
			is1 = torch.where(Arrivals[:,4] == 1)[0]

			assert(abs(Arrivals[:,2] - A_src_in_sta[0]).max().item() == 0)
			assert(abs((Arrivals[ip1,0] - trv_out_initial[ip1,0]) - Input[ip1,0]).max() < 1e-1) ## Residuals match the input residual for unpertubed data
			assert(abs((Arrivals[is1,1] - trv_out_initial[is1,1]) - Input[is1,1]).max() < 1e-1)
			# assert(abs(Arrivals[:,5] - A_src_in_sta_abs[1]).max().item() == 0)


			Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
			Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]

			## Absolute residual (can normalize each event)
			loss_p_abs = loss_func(Res_p, torch.zeros(Res_p.shape).to(device))
			loss_s_abs = weight_s_loss*loss_func(Res_s, torch.zeros(Res_s.shape).to(device))
			loss_abs = (0.5*loss_p_abs + 0.5*loss_s_abs)/n_batch

			## Produces loss_abs

		if use_sta_corr == True:

			if use_absolute == False:

				ip1 = torch.where(Arrivals[:,3] == 1)[0]
				is1 = torch.where(Arrivals[:,4] == 1)[0]

				assert(abs(Arrivals[:,2] - A_src_in_sta[0]).max().item() == 0)
				assert(abs((Arrivals[ip1,0] - trv_out_initial[ip1,0]) - Input[ip1,0]).max() < 1e-1) ## Residuals match the input residual for unpertubed data
				assert(abs((Arrivals[is1,1] - trv_out_initial[is1,1]) - Input[is1,1]).max() < 1e-1)
				# assert(abs(Arrivals[:,5] - A_src_in_sta_abs[1]).max().item() == 0)

				Res_p = Arrivals[ip1,0] - trv_out_perturbed[ip1,0]
				Res_s = Arrivals[is1,1] - trv_out_perturbed[is1,1]

			## Station corrections
			Res_sta_p = global_mean_pool(Res_p, A_src_in_sta[0,ip1])
			Res_sta_s = global_mean_pool(Res_s, A_src_in_sta[0,is1])
			loss_p_sta = loss_func(Res_sta_p, torch.zeros(Res_sta_p.shape).to(device))
			loss_s_sta = weight_s_loss*loss_func(Res_sta_s, torch.zeros(Res_sta_s.shape).to(device))
			loss_sta = (0.5*loss_p_sta + 0.5*loss_s_sta)/n_batch

			## Produces loss_sta

		#### Compute double difference residual ######

		# moi

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
			loss_s_double_difference = weight_s_loss*loss_func(Pred_s[i2], Trgts_s[i2])
			loss_double_difference = (0.5*loss_p_double_difference + 0.5*loss_s_double_difference)/n_batch

		# if use_double_diff == True:

		# 	p_slice_edge1 = A_prod_src_src[0,ifind_p_edges]
		# 	p_slice_edge2 = A_prod_src_src[1,ifind_p_edges]
		# 	s_slice_edge1 = A_prod_src_src[0,ifind_s_edges]
		# 	s_slice_edge2 = A_prod_src_src[1,ifind_s_edges]

		# 	sta_p_slice_edge1 = A_src_in_sta[0,p_slice_edge1]
		# 	sta_p_slice_edge2 = A_src_in_sta[0,p_slice_edge2]
		# 	sta_s_slice_edge1 = A_src_in_sta[0,s_slice_edge1]
		# 	sta_s_slice_edge2 = A_src_in_sta[0,s_slice_edge2]

		# 	src_p_slice_edge1 = A_src_in_sta[1,p_slice_edge1]
		# 	src_p_slice_edge2 = A_src_in_sta[1,p_slice_edge2]
		# 	src_s_slice_edge1 = A_src_in_sta[1,s_slice_edge1]
		# 	src_s_slice_edge2 = A_src_in_sta[1,s_slice_edge2]

		# 	Trgts_p = Input[p_slice_edge1,0] - Input[p_slice_edge2,0]
		# 	Trgts_s = Input[s_slice_edge1,1] - Input[s_slice_edge2,1]
			
		# 	## Use all samples for residual (or remove some samples)
		# 	i1 = np.arange(len(Trgts_p))
		# 	i2 = np.arange(len(Trgts_s))

		# 	## Initial (or pre-compute these)
		# 	trv_out_perturbed_i_p_initial = trv_pairwise(locs_cuda[sta_p_slice_edge1], ftrns2_diff(srcs_slice_cart[src_p_slice_edge1]))[:, 0]
		# 	trv_out_perturbed_j_p_initial = trv_pairwise(locs_cuda[sta_p_slice_edge2], ftrns2_diff(srcs_slice_cart[src_p_slice_edge2]))[:, 0]

		# 	trv_out_perturbed_i_s_initial = trv_pairwise(locs_cuda[sta_s_slice_edge1], ftrns2_diff(srcs_slice_cart[src_s_slice_edge1]))[:, 1]
		# 	trv_out_perturbed_j_s_initial = trv_pairwise(locs_cuda[sta_s_slice_edge2], ftrns2_diff(srcs_slice_cart[src_s_slice_edge2]))[:, 1]

		# 	trv_out_perturbed_i_p = trv_pairwise(locs_cuda[sta_p_slice_edge1], ftrns2_diff(srcs_slice_cart[src_p_slice_edge1] + pred[src_p_slice_edge1]))[:, 0] + pred_t[src_p_slice_edge1,0] + pred_c[sta_p_slice_edge1,0]
		# 	trv_out_perturbed_j_p = trv_pairwise(locs_cuda[sta_p_slice_edge2], ftrns2_diff(srcs_slice_cart[src_p_slice_edge2] + pred[src_p_slice_edge2]))[:, 0] + pred_t[src_p_slice_edge2,0] + pred_c[sta_p_slice_edge2,0]

		# 	trv_out_perturbed_i_s = trv_pairwise(locs_cuda[sta_s_slice_edge1], ftrns2_diff(srcs_slice_cart[src_s_slice_edge1] + pred[src_s_slice_edge1]))[:, 1] + pred_t[src_s_slice_edge1,0] + pred_c[sta_s_slice_edge1,1]
		# 	trv_out_perturbed_j_s = trv_pairwise(locs_cuda[sta_s_slice_edge2], ftrns2_diff(srcs_slice_cart[src_s_slice_edge2] + pred[src_s_slice_edge2]))[:, 1] + pred_t[src_s_slice_edge2,0] + pred_c[sta_s_slice_edge2,1]

		# 	Res_i_p = (trv_out_perturbed_i_p - trv_out_perturbed_i_p_initial)
		# 	Res_j_p = (trv_out_perturbed_j_p - trv_out_perturbed_j_p_initial)

		# 	Res_i_s = (trv_out_perturbed_i_s - trv_out_perturbed_i_s_initial)
		# 	Res_j_s = (trv_out_perturbed_j_s - trv_out_perturbed_j_s_initial)

		# 	Pred_p = Res_i_p - Res_j_p
		# 	Pred_s = Res_i_s - Res_j_s

		# 	loss_p_double_difference = loss_func(Pred_p[i1], Trgts_p[i1])
		# 	loss_s_double_difference = weight_s_loss*loss_func(Pred_s[i2], Trgts_s[i2])
		# 	loss_double_difference = (0.5*loss_p_double_difference + 0.5*loss_s_double_difference)/n_batch

		## Add loss related to the (cross-correlation) differential times themselves
		if use_diff == True:

			islice_edges, values_slice = subgraph(torch.Tensor(imatch_srcs).to(device).long(), source_indices, merged_values)
			n_edges_slice = islice_edges.shape[1]
			## Inside values_slice: (first column phase type, second column station index, third column weight)

			perm_vec = -1*torch.ones(len(srcs)).long().to(device)
			perm_vec[imatch_srcs] = torch.arange(len(imatch_srcs)).to(device)
			src_slice_edge1 = perm_vec[islice_edges[0]]
			src_slice_edge2 = perm_vec[islice_edges[1]]

			assert(torch.min(torch.Tensor([src_slice_edge1.min(), src_slice_edge2.min()])) > -1) # , src_s_slice_edge1.min(), src_s_slice_edge2.min()])) > -1)


			trv_out_perturbed_i = trv_pairwise(locs_cuda[values_slice[:,1].long()], srcs_perturbed[src_slice_edge1])[torch.arange(n_edges_slice), values_slice[:,0].long()] + pred_t[src_slice_edge1,0] + pred_c[values_slice[:,1].long(),values_slice[:,0].long()]
			trv_out_perturbed_j = trv_pairwise(locs_cuda[values_slice[:,1].long()], srcs_perturbed[src_slice_edge2])[torch.arange(n_edges_slice), values_slice[:,0].long()] + pred_t[src_slice_edge2,0] + pred_c[values_slice[:,1].long(),values_slice[:,0].long()]

			trgt_diff = values_slice[:,3]
			pred_diff = trv_out_perturbed_i - trv_out_perturbed_j

			weight_vec_phase = torch.ones(len(trgt_diff)).to(device)
			weight_vec_phase[values_slice[:,0] == 1] = 0.5

			# moi

			loss_diff = (weight_vec_phase*values_slice[:,2]*torch.abs(trgt_diff - pred_diff)).mean()/n_batch # (0.5*loss_p_diff + 0.5*loss_s_diff)/n_batch			


		loss = 0.0
		if use_double_diff == True:
			loss = loss + 0.8*loss_double_difference
		if use_diff == True:
			loss = loss + 0.8*loss_diff
		if use_absolute == True:
			loss = loss + 0.2*0.5*loss_abs
		if use_sta_corr == True:
			loss = loss + 0.2*0.5*loss_sta
		if use_calibration == True:
			loss = loss + 0.2*(0.25*loss_cal + 0.25*loss_cal_abs)


		# loss = 0.2*(0.5*loss_abs + 0.5*loss_sta + 0.25*loss_cal + 0.25*loss_cal_abs) + 0.8*loss_double_difference

		loss_val += loss.item()
		loss_val_abs += loss_abs.item()
		loss_val_sta += loss_sta.item()
		loss_val_cal += loss_cal.item()
		loss_val_cal_abs += loss_cal_abs.item()
		loss_val_double_diff += loss_double_difference.item()
		loss_val_diff += loss_diff.item()

		# loss = 0.5*loss + 0.5*loss_sta + 0.25*loss_cal + 0.25*loss_cal_abs

		loss.backward(retain_graph = True) ## Why must this be done?

		cnt_file += 1


	## Update model

	losses.append(loss_val)
	losses_abs.append(loss_val_abs)
	losses_sta.append(loss_val_sta)
	losses_cal.append(loss_val_cal)
	losses_cal_abs.append(loss_val_cal_abs)
	losses_double_diff.append(loss_val_double_diff)
	losses_diff.append(loss_val_diff)


	optimizer.step()

	# del pred, pred_t

	print('%d %0.8f'%(i, loss_val))


	if np.mod(i, 1000) == 0:

		srcs_slice_perturbed = ftrns2_diff(srcs_slice_cart + pred).cpu().detach().numpy()

		srcs_perturbed = ftrns2_diff(srcs_cart + torch.Tensor(srcs_pred_perturb[:,0:3]).to(device)).cpu().detach().numpy()
		srcs_perturbed = np.concatenate((srcs_perturbed, srcs_pred_perturb[:,3].reshape(-1,1)), axis = 1)
		imask = np.where(np.isnan(srcs_pred_perturb[:,0]) == 1)[0]
		srcs_perturbed[imask,:] = np.nan

		## Save model
		# srcs_cart = torch.Tensor(ftrns1(srcs)).to(device)
		# srcs_perturbed = ftrns2_diff(srcs_slice_cart + pred).cpu().detach().numpy()
		torch.save(m.state_dict(), path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d.h5'%(i, n_ver_load_files, n_ver_save))
		torch.save(optimizer.state_dict(), path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_optimizer.h5'%(i, n_ver_load_files, n_ver_save))
		np.savez_compressed(path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_losses.npz'%(i, n_ver_load_files, n_ver_save), losses = losses, srcs_slice = srcs_slice, srcs_slice_perturbed = srcs_slice_perturbed, srcs = srcs, srcs_perturbed = srcs_perturbed, srcs_pred_perturb = srcs_pred_perturb, pred_t = pred_t.cpu().detach().numpy(), corrections_c = corrections_c, srcs_ref = srcs_ref, Matches = Matches, locs = locs, losses_abs = losses_abs, losses_sta = losses_sta, losses_cal = losses_cal, losses_cal_abs = losses_cal_abs, losses_double_diff = losses_double_diff, losses_diff = losses_diff)
		print('saved model %s data %d step %d'%(n_ver_load_files, n_ver_save, i))
		assert(np.abs(srcs - srcs_fixed).max() < 1e-2)


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

