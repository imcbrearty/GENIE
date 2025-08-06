import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import yaml
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import gamma, beta
import time
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from sklearn.neighbors import KernelDensity
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
from torch_scatter import scatter
from numpy.matlib import repmat
from obspy.core import UTCDateTime
from sklearn.cluster import KMeans
from skopt import gp_minimize
import pathlib
import glob
import pdb
import sys

from utils import *
from module import *
from process_utils import *
# from generate_synthetic_data import generate_synthetic_data 
## For now not using the seperate files definition of generate_synthetic_data

## Note: you should try changing the synthetic data parameters and visualizing the 
## results some, some values are better than others depending on region and stations

use_wandb_logging = False
if use_wandb_logging == True:

	import wandb
	# Initialize wandb run 
	wandb.init(project="GENIE")


# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load training configuration from YAML
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

name_of_project = config['name_of_project']

path_to_file = str(pathlib.Path().absolute())
path_to_file += '\\' if '\\' in path_to_file else '/'
seperator = '\\' if '\\' in path_to_file else '/'

## Graph params
k_sta_edges = config['k_sta_edges']
k_spc_edges = config['k_spc_edges']
k_time_edges = config['k_time_edges']
use_physics_informed = config['use_physics_informed']
use_phase_types = config['use_phase_types']
use_subgraph = config['use_subgraph']
use_topography = config['use_topography']
if use_subgraph == True:
    max_deg_offset = config['max_deg_offset']
    k_nearest_pairs = config['k_nearest_pairs']	

graph_params = [k_sta_edges, k_spc_edges, k_time_edges]

# File versions
template_ver = train_config['template_ver'] # spatial grid version
vel_model_ver = train_config['vel_model_ver'] # velocity model version
n_ver = train_config['n_ver'] # GNN save version

## Training params
n_batch = train_config['n_batch']
n_epochs = train_config['n_epochs'] # add 1, so it saves on last iteration (since it saves every 100 steps)
n_spc_query = train_config['n_spc_query'] # Number of src queries per sample
n_src_query = train_config['n_src_query'] # Number of src-arrival queries per sample
training_params = [n_spc_query, n_src_query]

## Reference catalog parameters
use_reference_spatial_density = train_config['use_reference_spatial_density'] # False ## If True, must store reference sources in "Calibration/"yr"/"project_name"_reference_2023_1_3_ver_1.npz with field "srcs_ref"
spatial_sigma = train_config['spatial_sigma'] # 20000.0 ## The amount of spatial smoothing to the reference catalog
min_magnitude_ref = train_config['min_magnitude_ref'] # 1.5
percentile_threshold_ref = train_config['percentile_threshold_ref'] # 0.1
n_reference_clusters = train_config['n_reference_clusters'] # 10000 ## The amount of "quasi-uniiform" nodes to obtain for representing the source catalog distribution
n_frac_reference_catalog = train_config['n_frac_reference_catalog'] # 0.8 ## The amount of sources to simulate from the reference catalog coordinates compared to background


## Prediction params
kernel_sig_t = train_config['kernel_sig_t'] # Kernel to embed arrival time - theoretical time misfit (s)
src_t_kernel = train_config['src_t_kernel'] # Kernel or origin time label (s)
src_t_arv_kernel = train_config['src_t_arv_kernel'] # Kernel for arrival association time label (s)
src_x_kernel = train_config['src_x_kernel'] # Kernel for source label, horizontal distance (m)
src_x_arv_kernel = train_config['src_x_arv_kernel'] # Kernel for arrival-source association label, horizontal distance (m)
src_depth_kernel = train_config['src_depth_kernel'] # Kernel of Cartesian projection, vertical distance (m)
# t_win = config['t_win'] ## This is the time window over which predictions are made. Shouldn't be changed for now.

use_adaptive_window = True
if use_adaptive_window == True:
	n_resolution = 9 ## The discretization of the source time function output
	t_win = np.round(np.copy(np.array([2*src_t_kernel]))[0], 2) ## Set window size to the source kernel width (i.e., prediction window is of length +/- src_t_kernel, or [-src_t_kernel + t0, t0 + src_t_kernel])
	dt_win = np.diff(np.linspace(-t_win/2.0, t_win/2.0, n_resolution))[0]
else:
	dt_win = 1.0 ## Default version
	t_win = 10.0

## Dataset parameters
load_training_data = train_config['load_training_data']
build_training_data = train_config['build_training_data'] ## If try, will use system argument to build a set of data
optimize_training_data = train_config['optimize_training_data']


max_number_pick_association_labels_per_sample = config['max_number_pick_association_labels_per_sample']
make_visualize_predictions = config['make_visualize_predictions']

## Note that right now, this shouldn't change, as the GNN definitions also assume this is 10 s.

## Will update to be adaptive soon. The step size of temporal prediction is fixed at 1 s right now.

## Should add src_x_arv_kernel and src_t_arv_kerne to pred_params, but need to check usage of this variable in this and later scripts
pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]

device = torch.device(config['device']) ## or use cpu

if torch.cuda.is_available() == False:
	print('No GPU available')
	device = torch.device('cpu')
	if config['device'] == 'cuda':
		print('Overwritting cuda to cpu since no gpu available')

## Setup training folder parameters
if (load_training_data == True) or (build_training_data == True):
	path_to_data = train_config['path_to_data'] ## Path to training data files
	n_ver_training_data = train_config['n_ver_training_data'] ## Version of training files
	if (path_to_data[-1] != '/')*(path_to_data[-1] != '\\'):
		path_to_data = path_to_data + seperator

if use_topography == True:
	surface_profile = np.load(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project)['surface_profile'] # (os.path.isfile(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project) == True)
	tree_surface = cKDTree(surface_profile[:,0:2])

## Load specific subsets of stations to train on in addition to random
## subnetworks from the total set of possible stations
load_subnetworks = train_config['fixed_subnetworks']
if load_subnetworks == True:
	
	min_sta_per_graph = int(k_sta_edges + 1)

	if (os.path.exists(path_to_file + '%s_subnetworks.hdf5'%name_of_project) == True)*(train_config['refresh_subnetworks'] == False):
		
		h_subnetworks = h5py.File(path_to_file + '%s_subnetworks.hdf5'%name_of_project, 'r')
		key_names = list(h_subnetworks.keys())
		Ind_subnetworks = []
		for s in key_names:
			if len(h_subnetworks[s][:]) > min_sta_per_graph:
				Ind_subnetworks.append(h_subnetworks[s][:].astype('int'))				
		h_subnetworks.close()
		
	else:
		print('Building subnetworks from pick data')
		save_subnetwork_file = False
		Ind_subnetworks = [] ## Check for subnetworks available in pick data
		st1 = glob.glob(path_to_file + 'Picks/19*') ## Assuming years are 1900 and 2000's
		st2 = glob.glob(path_to_file + 'Picks/20*')
		st = np.concatenate((st1, st2), axis = 0)
		for i in range(len(st)):
			st1 = glob.glob(st[i] + '/*.npz')
			for j in range(len(st1)):
				z = np.load(st1[j])
				ind_use = np.unique(z['P'][:,1]).astype('int')
				z.close()
				if len(ind_use) > min_sta_per_graph:
					Ind_subnetworks.append(ind_use)
					
		if save_subnetwork_file == True:
			h_subnetworks = h5py.File(path_to_file + '%s_subnetworks.hdf5'%name_of_project, 'w')
			for j in range(len(Ind_subnetworks)):
				h_subnetworks['subnetwork_%d'%j] = Ind_subnetworks[j]
			h_subnetworks.close()

	if len(Ind_subnetworks) == 0:
		print('Did not find any subnetwork configurations')
		Ind_subnetworks = False
		load_subnetworks = False

else:
	Ind_subnetworks = False


class LocalMarching(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, device = 'cpu'):
		super(LocalMarching, self).__init__(aggr = 'max') # node dim
		self.device = device

	## Changed dt to 5 s
	def forward(self, val, srcs_t, tc_win = 5, n_steps_max = 100, tol = 1e-12, scale_depth = 1.0, use_directed = True):

		tree_t = cKDTree(srcs_t.reshape(-1,1))
		lp_t = tree_t.query_ball_point(srcs_t.reshape(-1,1), r = tc_win)
		cedges = [np.array(lp_t[i]) for i in range(len(lp_t))]
		cedges1 = np.hstack([i*np.ones(len(cedges[i])) for i in range(len(cedges))])
		edges = torch.Tensor(np.concatenate((np.hstack(cedges).reshape(1,-1), cedges1.reshape(1,-1)), axis = 0)).long().to(self.device)

		Data_obj = Data(edge_index = to_undirected(edges), num_nodes = edges.max().item() + 1) # undirected
		nx_g = to_networkx(Data_obj).to_undirected()
		lp = list(nx.connected_components(nx_g))
		clen = [len(list(lp[i])) for i in range(len(lp))]
		## Remove disconnected points with only 1 maxima.

		if use_directed == True:

			max_val = torch.where(val[edges[1]] <= val[edges[0]])[0]
			edges = edges[:,max_val]

		srcs_keep = []
		for i in range(len(lp)):
			nodes = np.sort(np.array(list(lp[i])))

			if (len(nodes) == 1):
				srcs_keep.append(nodes.reshape(-1))

			else:

				edges_need = subgraph(torch.Tensor(nodes).long().to(self.device), edges, relabel_nodes = True)[0]

				vals = torch.Tensor(val[nodes]).view(-1,1).to(self.device)
				vals_initial = torch.Tensor(val[nodes]).view(-1,1).to(self.device)
				vtol = 1e9
				nt = 0

				while (vtol > tol) and (nt < n_steps_max):
					vals0 = 1.0*vals + 0.0 # copy vals
					vals = self.propagate(edges_need, x = vals)
					vtol = abs(vals - vals0).max().item()
					nt += 1

				ip_slice = torch.where(torch.isclose(vals_initial[:,0], vals[:,0], rtol = tol) == 1)[0] ## Keep nodes with final values similar to starting values
				srcs_keep.append(nodes[ip_slice.cpu().detach().numpy()])

		if len(srcs_keep) > 0:
			srcs_keep = srcs_t[np.hstack(srcs_keep)]

		return srcs_keep
	

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
seperator = '\\' if '\\' in path_to_file else '/'
write_training_file = path_to_file + 'GNN_TrainedModels' + seperator + name_of_project + '_'

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

rbest_cuda = torch.Tensor(rbest).to(device)
mn_cuda = torch.Tensor(mn).to(device)

# use_spherical = False
if config['use_spherical'] == True:

	earth_radius = 6371e3
	ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius)

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device)

else:

	earth_radius = 6378137.0
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device)
	

## Check for reference catalog
if use_reference_spatial_density == True:

	n_reference_ver = 1
	load_reference_density = True
	if (os.path.isfile(path_to_file + 'Grids' + seperator + 'reference_source_density_ver_1.npz') == 1)*(load_reference_density == True):
		srcs_ref = np.load(path_to_file + 'Grids' + seperator + 'reference_source_density_ver_%d.npz'%n_reference_ver)['srcs_ref']
	else:
		st1 = glob.glob(path_to_file + 'Calibration/19*') ## Assuming years are 1900 and 2000's
		st2 = glob.glob(path_to_file + 'Calibration/20*')
		st = np.concatenate((st1, st2), axis = 0)
		st = np.hstack([glob.glob(s + '/*ver_%d.npz'%n_reference_ver) for s in st])
		srcs_ref = []
		for s in st:
			srcs_ref.append(np.load(s)['srcs_ref'])
			print('Read %s'%s)
		srcs_ref = np.vstack(srcs_ref)
		scale_x_ = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
		offset_x_ = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
		if (min_magnitude_ref is not False)*(np.isnan(srcs_ref[:,4]).sum() == 0):
			srcs_ref = srcs_ref[srcs_ref[:,4] >= min_magnitude_ref,:]
		
		srcs_ref = kmeans_packing_fit_sources(srcs_ref, scale_x_, offset_x_, 3, n_reference_clusters, ftrns1, ftrns2, n_batch = 5000, n_steps = 5000, blur_sigma = spatial_sigma)[0]
		m_ref = KernelDensity(kernel = 'gaussian', bandwidth = spatial_sigma).fit(ftrns1(srcs_ref))
		## Make uniform grid, query if prob > percentile prob of the kernel density; keep these points and repeat kmeans_packing_fit_sources
		dlen1, dlen2 = np.diff(lat_range_extend), np.diff(lon_range_extend)
		dscale = np.sqrt(dlen1*dlen2)[0]/50
		dscale_depth = np.minimum(spatial_sigma, 110e3*dscale)
		x1_lat, x2_lon, x3_depth = np.arange(lat_range_extend[0], lat_range_extend[1], dscale), np.arange(lon_range_extend[0], lon_range_extend[1], dscale), np.arange(depth_range[0], depth_range[1] + dscale_depth, dscale_depth)
		x11, x12, x13 = np.meshgrid(x1_lat, x2_lon, x3_depth)
		xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
		prob = m_ref.score_samples(ftrns1(xx))
		prob = np.exp(prob - prob.max())
		itrue = np.where(prob > percentile_threshold_ref)[0] # np.where(prob/prob.max() > np.quantile(prob/prob.max(), percentile_threshold_ref))[0]
		srcs_ref = kmeans_packing_fit_sources(xx[itrue], scale_x_, offset_x_, 3, n_reference_clusters, ftrns1, ftrns2, n_batch = 5000, n_steps = 5000, blur_sigma = spatial_sigma)[0]
		
		if load_reference_density == True:
			np.savez_compressed(path_to_file + 'Grids' + seperator + 'reference_source_density_ver_%d.npz'%n_reference_ver, srcs_ref = srcs_ref)
					


## Training synthic data parameters

# ## Training params list 2
# spc_random = train_config['spc_random']
# sig_t = train_config['sig_t'] # 3 percent of travel time error on pick times
# spc_thresh_rand = train_config['spc_thresh_rand']
# min_sta_arrival = train_config['min_sta_arrival']
# coda_rate = train_config['coda_rate'] # 5 percent arrival have code. Probably more than this? Increased from 0.035.
# coda_win = np.array(train_config['coda_win']) # coda occurs within 0 to 25 s after arrival (should be less?) # Increased to 25, from 20.0
# max_num_spikes = train_config['max_num_spikes']
# spike_time_spread = train_config['spike_time_spread']
# s_extra = train_config['s_extra'] ## If this is non-zero, it can increase (or decrease) the total rate of missed s waves compared to p waves
# use_stable_association_labels = train_config['use_stable_association_labels']
# thresh_noise_max = train_config['thresh_noise_max'] # ratio of sig_t*travel time considered excess noise
# min_misfit_allowed = train_config['min_misfit_allowed'] ## The minimum error on theoretical vs. observed travel times that beneath which, picks have positive associaton labels (the upper limit is set by a percentage of the travel time)
# total_bias = train_config['total_bias'] ## The total (uniform across stations) bias on travel times for each synthetic earthquake (helps add robustness to uncertainity on assumed and true velocity models)
# training_params_2 = [spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max, min_misfit_allowed, total_bias]

# ## Training params list 3
# # n_batch = train_config['n_batch']
# dist_range = train_config['dist_range'] # Should be chosen proportional to physical domain size
# max_rate_events = train_config['max_rate_events']
# max_miss_events = train_config['max_miss_events']
# max_false_events = train_config['max_rate_events']*train_config['max_false_events'] # Make max_false_events an absolute value, but it's based on the ratio of the value in the config file times the event rate
# miss_pick_fraction = train_config['miss_pick_fraction']
# T = train_config['T']
# dt = train_config['dt']
# tscale = train_config['tscale']
# n_sta_range = train_config['n_sta_range'] # n_sta_range[0]*locs.shape[0] must be >= the number of station edges chosen (k_sta_edges)
# use_sources = train_config['use_sources']
# use_full_network = train_config['use_full_network']
# if Ind_subnetworks is not False:
# 	fixed_subnetworks = Ind_subnetworks
# else:
# 	fixed_subnetworks = False # train_config['fixed_subnetworks']
# use_preferential_sampling = train_config['use_preferential_sampling']
# use_shallow_sources = train_config['use_shallow_sources']
# use_extra_nearby_moveouts = train_config['use_extra_nearby_moveouts']
# training_params_3 = [n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, miss_pick_fraction, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources, use_extra_nearby_moveouts]


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

	z.close()

elif config['train_travel_time_neural_network'] == True:

	n_ver_trv_time_model_load = vel_model_ver # 1
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', use_physics_informed = use_physics_informed, device = device)

use_only_active_stations = False
if use_only_active_stations == True:
	unique_inds = np.unique(np.hstack(Ind_subnetworks))
	perm_vec = -1*np.ones(locs.shape[0]).astype('int')
	perm_vec[unique_inds] = np.arange(len(unique_inds))

	for i in range(len(Ind_subnetworks)):
		Ind_subnetworks[i] = perm_vec[Ind_subnetworks[i]]
		assert(Ind_subnetworks[-1].min() > -1)

	locs = locs[unique_inds]
	stas = stas[unique_inds]

	min_sta = 10
	ifind = np.where([len(Ind_subnetworks[i]) >= min_sta for i in range(len(Ind_subnetworks))])[0]
	Ind_subnetworks = [Ind_subnetworks[i] for i in ifind]


load_distance_model = True
if load_distance_model == True:
	n_mag_ver = 1
	use_softplus = True
	dist_supp = np.load(path_to_file + 'Grids' + seperator + 'distance_magnitude_model_ver_%d.npz'%(n_mag_ver))
	if use_softplus == False:
		poly_dist_p, poly_dist_s, min_dist = dist_supp['dist_p'], dist_supp['dist_s'], dist_supp['min_dist']
		pdist_p = lambda mag: np.maximum(min_dist[0], np.polyval(poly_dist_p, mag))
		pdist_s = lambda mag: np.maximum(min_dist[1], np.polyval(poly_dist_s, mag))
	else:
		dist_params = dist_supp['params']
		pdist_p = lambda mag: dist_params[4]*(1.0/dist_params[1])*np.log(1.0 + np.exp(dist_params[1]*mag)) + dist_params[0]
		pdist_s = lambda mag: dist_params[4]*(1.0/dist_params[3])*np.log(1.0 + np.exp(dist_params[3]*mag)) + dist_params[2]


## Check if knn is working on cuda
if device.type == 'cuda' or device.type == 'cpu':
	check_len = knn(torch.rand(10,3).to(device), torch.rand(10,3).to(device), k = 5).numel()
	if check_len != 100: # If it's less than 2 * 10 * 5, there's an issue
		raise SystemError('Issue with knn on cuda for some versions of pytorch geometric and cuda')

	check_len = knn(10.0*torch.rand(200,3).to(device), 10.0*torch.rand(100,3).to(device), k = 15).numel()
	if check_len != 3000: # If it's less than 2 * 10 * 5, there's an issue
		raise SystemError('Issue with knn on cuda for some versions of pytorch geometric and cuda')
	## Note: can update train script to still use cuda except use cpu for all knn operations,
	## (need to convert inputs to knn to .cpu(), and then outputs of knn back to .cuda())
	## or, just update cuda to the latest version (e.g., >= 12.1)
	## See these issues: https://github.com/rusty1s/pytorch_cluster/issues/181,
	## https://github.com/pyg-team/pytorch_geometric/issues/7475





## Load dataset ##
n_ver_events = 1
st_load = glob.glob(path_to_file + 'Catalog/19*') # Load years 1900's
st_load.extend(glob.glob(path_to_file + 'Catalog/20*')) # Load years 2000's
iarg = np.argsort([int(st_load[i].split(seperator)[-1]) for i in range(len(st_load))])
st_load = [st_load[i] for i in iarg]
st_load_l = []
Srcs = []
Mags = []
Inds = []
Picks_P_lists = []
Picks_S_lists = []
Srcs1_remove = []
Srcs2_remove = []
Mags_remove = []
Inds_remove = []
Picks_P_lists_remove = []
Picks_S_lists_remove = []

max_deviation = 35e3 ## Remove sources greater than this deviation between srcs_trv and srcs
max_outlier_quant = 0.9 ## Based on 90% percentile distance between source - receiever
max_outlier_multiplier = 1.5 ## Based on the percentile distance, remove picks greater than this distance fraction times quantile distance
max_rms_residual = 3.0 ## Max source station residual to allow
min_sta_required = 6 ## min number unique stations (after removing outliers; and re-locating?)
min_magnitude = -1.0 ## Minimum magnitude to use

check_for_glitches = True ## Check for network wide glitches
spatial_clusters = 5 ## Number of regions, to check for network wide variability
win_count = 2.0 ## Window to compute counts
quantile_val = 1.0001 - 8*0.0001
min_time_offset = 8.0 ## Min origin time offset from spike time (note: this doesn't work perfectly, since origin times can be offset)

cnt_outlier_picks_p, cnt_outlier_picks_s = [], []
cnt_outlier_picks_p_remove, cnt_outlier_picks_s_remove = [], []
# t_times = [UTCDatTime()]

st_list = []
for i in range(len(st_load)):
	st = glob.glob(st_load[i] + seperator + '*continuous*ver_%d.hdf5'%(n_ver_events))
	if len(st) > 0:
		st_list.extend(st)

st_list = [st_list[j] for j in np.argsort(np.array([UTCDateTime(int(s.split('/')[-1].split('_')[4]), int(s.split('/')[-1].split('_')[5]), int(s.split('/')[-1].split('_')[6])) - UTCDateTime(2000, 1, 1) for s in st_list]))]

cnts_remove = np.zeros((len(st_list),5))

for i in range(len(st_list)):

	z = h5py.File(st_list[i], 'r')
	P, P_perm = z['P'][:], z['P_perm'][:]
	srcs = z['srcs'][:]
	srcs_trv = z['srcs_trv'][:]
	mag_trv = z['mag_trv'][:]
	locs_use = z['locs_use'][:]
	ind_use = z['ind_use'][:]
	date = z['date'][:]
	Picks_P_perm = [z['Picks/%d_Picks_P_perm'%j][:] for j in range(len(srcs_trv))]
	Picks_S_perm = [z['Picks/%d_Picks_S_perm'%j][:] for j in range(len(srcs_trv))]
	Picks_P_perm_init = [z['Picks/%d_Picks_P_perm'%j][:] for j in range(len(srcs_trv))]
	Picks_S_perm_init = [z['Picks/%d_Picks_S_perm'%j][:] for j in range(len(srcs_trv))]
	iwith_p = np.where([len(Picks_P_perm[j]) > 0 for j in range(len(Picks_P_perm))])[0]
	iwith_s = np.where([len(Picks_S_perm[j]) > 0 for j in range(len(Picks_S_perm))])[0]
	z.close()

	## [1] Compute source pairwise distance (of either source type)
	dist_src = np.linalg.norm(ftrns1(srcs) - ftrns1(srcs_trv), axis = 1)
	iwithin_dist = np.where(dist_src < max_deviation)[0]

	## [2] Find outlier picks
	dist_src_to_reciever = np.linalg.norm(np.expand_dims(ftrns1(srcs), axis = 1) - np.expand_dims(ftrns1(locs_use), axis = 0), axis = 2)
	imax_p, imax_s = np.zeros((2, len(srcs)))
	imax_p[iwith_p] = np.array([np.quantile(dist_src_to_reciever[j,Picks_P_perm[j][:,1].astype('int')], max_outlier_quant)*max_outlier_multiplier for j in iwith_p])
	imax_s[iwith_s] = np.array([np.quantile(dist_src_to_reciever[j,Picks_S_perm[j][:,1].astype('int')], max_outlier_quant)*max_outlier_multiplier for j in iwith_s])
	# p_dist_val = [[] for j in range(len(srcs))]
	# s_dist_val = [[] for j in range(len(srcs))]
	p_dist_val = [dist_src_to_reciever[j,Picks_P_perm[j][:,1].astype('int')] for j in iwith_p]
	s_dist_val = [dist_src_to_reciever[j,Picks_S_perm[j][:,1].astype('int')] for j in iwith_s]

	# min_sta_per_phase = 3
	cnt_remove_p = np.zeros(len(srcs))
	for inc, j in enumerate(iwith_p):
		idel_p = np.where(p_dist_val[inc] > imax_p[j])[0]
		if len(idel_p) > 0:
			Picks_P_perm[j] = np.delete(Picks_P_perm[j], idel_p, axis = 0)
			cnt_remove_p[j] = len(idel_p)

	cnt_remove_s = np.zeros(len(srcs))
	for inc, j in enumerate(iwith_s):
		idel_s = np.where(s_dist_val[inc] > imax_s[j])[0]
		if len(idel_s) > 0:
			Picks_S_perm[j] = np.delete(Picks_S_perm[j], idel_s, axis = 0)
			cnt_remove_s[j] = len(idel_s)

	## [3] Min number of unique stations
	iwith_sta = np.where([len(np.unique(np.concatenate((Picks_P_perm[j][:,1], Picks_S_perm[j][:,1]), axis = 0))) for j in range(len(srcs))])[0]

	## [4] Check travel time residual
	trv_out = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_trv).to(device)).cpu().detach().numpy() + srcs_trv[:,3].reshape(-1,1,1)
	merged_picks = [np.concatenate((Picks_P_perm[j]))]
	res = [np.concatenate((trv_out[j,Picks_P_perm[j][:,1].astype('int'),0] - Picks_P_perm[j][:,0], trv_out[j,Picks_S_perm[j][:,1].astype('int'),1] - Picks_S_perm[j][:,0]), axis = 0) for j in range(len(srcs))]
	rms_residual = np.array([np.linalg.norm(res[j])/np.sqrt(len(res[j])) for j in range(len(srcs))])
	iwithin_rms = np.where(rms_residual < max_rms_residual)[0]

	## [5] Minimum magnitude
	iwithin_mag = np.where(mag_trv > min_magnitude)[0]

	## [6] Check for glitches
	inot_spike = np.arange(len(srcs)) ## Include all indices, if not checking for spikes
	if check_for_glitches == True:
		core_clusters = KMeans(n_clusters = spatial_clusters).fit(ftrns1(locs_use))
		cluster_assignments = core_clusters.labels_
		perm_cluster = (-1.0*np.ones(len(locs_use))).astype('int')
		for j in range(spatial_clusters): perm_cluster[np.where(cluster_assignments == j)[0]] = j
		perm_inds = perm_cluster[P_perm[:,1].astype('int')]
		tsteps = np.arange(P_perm[:,0].min() - win_count*3.0, P_perm[:,0].max() + 3.0*win_count, win_count)

		Counts = []
		for j in range(spatial_clusters):
			ifind_ind = np.where(perm_inds == j)[0] ## Picks that occur in this cluster
			tree = cKDTree(P_perm[ifind_ind,0].reshape(-1,1))
			ip = tree.query_ball_point(tsteps.reshape(-1,1) + win_count/2.0, r = win_count/2.0)
			counts = np.array([len(ip[j]) for j in range(len(ip))])
			# thresh = np.quantile(counts, np.arange(0, 1.0001, 0.0001))[-8] # 2.0
			thresh = np.quantile(counts, quantile_val)
			# ifind = np.where(counts >= thresh)[0]
			Counts.append(counts.reshape(1,-1) >= thresh)
		Counts = np.vstack(Counts).sum(0) ## Sum across different regions
		# ispike = np.where(Counts >= int(np.ceil(spatial_clusters/2)))[0]
		ispike = np.where(Counts >= int(np.floor(spatial_clusters/2)))[0]
		if len(ispike) > 0:
			mp = LocalMarching(device = device)
			tpicks = mp(torch.Tensor(Counts[ispike]).to(device), tsteps[ispike] + win_count/2.0, tc_win = win_count*1.25)
			# if len(tpicks) > 0:
			inot_spike = np.where(np.abs(srcs_trv[:,3].reshape(-1,1) - tpicks.reshape(1,-1)).min(1) > min_time_offset)[0]

		# Counts = Counts.sum(0) >= int(np.ceil(spatial_clusters/2)) ## Possible spikes



	## [7] Choose events
	ikeep = set(iwithin_dist).intersection(iwith_sta) #  + set(iwithin_rms) + set(iwithin_mag)))
	ikeep = ikeep.intersection(iwithin_rms)
	ikeep = ikeep.intersection(iwithin_mag)
	ikeep = np.array(list(ikeep.intersection(inot_spike))).astype('int')
	iremove = np.delete(np.arange(len(srcs)), ikeep, axis = 0)
	print('Keeping %d of %d events on %d/%d/%d [%d, %d, %d, %d, %d]'%(len(ikeep), len(srcs), date[0], date[1], date[2], len(iwithin_dist), len(iwith_sta), len(iwithin_rms), len(iwithin_mag), len(inot_spike)))
	cnts_remove[i,0] = len(srcs) - len(iwithin_dist)
	cnts_remove[i,1] = len(srcs) - len(iwith_sta)
	cnts_remove[i,2] = len(srcs) - len(iwithin_rms)
	cnts_remove[i,3] = len(srcs) - len(iwithin_mag)
	cnts_remove[i,4] = len(srcs) - len(inot_spike)

	## Subset the events
	if len(ikeep) == 0:
		print('Skipping day, no events')
		continue

	if len(iremove) > 0:
		Srcs1_remove.append(srcs_trv[iremove])
		Srcs2_remove.append(srcs[iremove])
		Mags_remove.append(mag_trv[iremove])
		cnt_outlier_picks_p_remove.append(cnt_remove_p[iremove])
		cnt_outlier_picks_s_remove.append(cnt_remove_s[iremove])
		for j in iremove:
			Picks_P_lists_remove.append(Picks_P_perm_init[j])
			Picks_S_lists_remove.append(Picks_S_perm_init[j])
			Inds_remove.append(ind_use)


	srcs = srcs[ikeep]
	srcs_trv = srcs_trv[ikeep]
	mag_trv = mag_trv[ikeep]
	Picks_P_perm = [Picks_P_perm[j] for j in ikeep]
	Picks_S_perm = [Picks_S_perm[j] for j in ikeep]
	cnt_remove_p = cnt_remove_p[ikeep]
	cnt_remove_s = cnt_remove_s[ikeep]

	## Append to files (saving srcs_trv instead of srcs)
	Srcs.append(srcs_trv)
	Mags.append(mag_trv)
	cnt_outlier_picks_p.append(cnt_remove_p)
	cnt_outlier_picks_s.append(cnt_remove_s)
	for j in range(len(srcs)):
		Picks_P_lists.append(Picks_P_perm[j])
		Picks_S_lists.append(Picks_S_perm[j])
		Inds.append(ind_use)
	# print('%d events on %d/%d/%d'%(len(srcs), date[0], date[1], date[2]))

Srcs = np.vstack(Srcs)
Mags = np.hstack(Mags)
Srcs1_remove = np.vstack(Srcs1_remove)
Srcs2_remove = np.vstack(Srcs2_remove)
Mags_remove = np.hstack(Mags_remove)
cnt_outlier_picks_p = np.hstack(cnt_outlier_picks_p)
cnt_outlier_picks_s = np.hstack(cnt_outlier_picks_s)
cnt_outlier_picks_p_remove = np.hstack(cnt_outlier_picks_p_remove)
cnt_outlier_picks_s_remove = np.hstack(cnt_outlier_picks_s_remove)
assert(len(Srcs) == len(Mags))
assert(len(Srcs) == len(Inds))
assert(len(Srcs) == len(Picks_P_lists))
assert(len(Srcs) == len(Picks_S_lists))
assert(len(Srcs1_remove) == len(Srcs2_remove))
assert(len(Srcs1_remove) == len(Mags_remove))
assert(len(Srcs1_remove) == len(Picks_P_lists_remove))
assert(len(Srcs1_remove) == len(Picks_S_lists_remove))
# assert(np.sign(cnts_remove).max(1).sum() == len(Srcs1_remove))
cnt_p = np.array([len(Picks_P_lists[j]) for j in range(len(Srcs))])
cnt_s = np.array([len(Picks_S_lists[j]) for j in range(len(Srcs))])
print('\nKept %d sources of %d total sources (%0.2f)'%(len(Srcs), len(Srcs) + len(Srcs1_remove), len(Srcs)/(len(Srcs) + len(Srcs1_remove))))
cnts_remove_sum = cnts_remove.sum(0)
print('\nRemoved: %d [dist offset], %d [to few stations], %d [large rms], %d [to small mag], %d [possible spikes]  (of %d)'%(cnts_remove_sum[0], cnts_remove_sum[1], cnts_remove_sum[2], cnts_remove_sum[3], cnts_remove_sum[4], len(Srcs1_remove)))
print('\nRemoved picks: %d of %d (%0.4f) P, %d of %d (%0.4f) S'%(cnt_outlier_picks_p.sum(), cnt_p.sum(), cnt_outlier_picks_p.sum()/cnt_p.sum(), cnt_outlier_picks_s.sum(), cnt_s.sum(), cnt_outlier_picks_s.sum()/cnt_s.sum()))
print('\nCreated variables Srcs, Mags, Inds, Picks_P_list, Picks_S_list, which can be used to easily access event data')
print('Srcs: (lat, lon, depth) source locations; Mags: local magnitudes; Inds: Subset of station indices used for each event (i.e., locs_use = locs[Ind[j]] for each jth event)')
print('Picks_P_list and Picks_S_list: associated station picks for each event, using perm_indices')

####################### Finished loading data  ####################### 
from generator_utils import sample_synthetic_moveout_pattern_generator, compute_morans_I_metric, calculate_inertia


####################### Apply generator and optimize parameters  ####################### 

def compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = False):

	## Assumes values are > 1e-12
	def rel_error(x, y, tol = 1e-12): ## x is pred, y is target
		val = (np.abs(y) > tol)*np.abs(x - y)/np.maximum(1e-12, np.abs(y))
		return val

	## Assumes features contains the: Morans, Inertia, Cnt, and Intersection variables
	## Each contains the (pred_p, pred_s, trgt_p, trgt_s), and Intersection (intersection_p, intersection_s)
	## These metrics are all computed as a "vector" over the batch
	Morans, Inertia, Cnt, Intersection = features ## Computing relative misfit features
	misfit_morans_p = rel_error(Morans[0], Morans[2])
	misfit_morans_s = rel_error(Morans[1], Morans[3])
	misfit_inertia_p = rel_error(Inertia[0], Inertia[2])
	misfit_inertia_s = rel_error(Inertia[1], Inertia[3])
	misfit_cnt_p = rel_error(Cnt[0], Cnt[2])
	misfit_cnt_s = rel_error(Cnt[1], Cnt[3])
	misfit_intersection_p = rel_error(Intersection[0], 1.0) ## Target of intersection is "1.0"
	misfit_intersection_s = rel_error(Intersection[1], 1.0) ## Target of intersection is "1.0"

	## We now bin the errors into magnitude bins
	mag_bins = np.linspace(np.quantile(mags_sample, 0.02), np.quantile(mags_sample, 0.98), n_mag_bins)
	mag_bin_width = np.diff(mag_bins[0:2])
	ip = cKDTree(mags_sample.reshape(-1,1)).query_ball_point(mag_bins.reshape(-1,1) + mag_bin_width/2.0, r = mag_bin_width/2.0)
	inot_nan_p = np.where(np.isnan(Intersection[0]) == 0)[0]
	inot_nan_s = np.where(np.isnan(Intersection[1]) == 0)[0]
	ip_p, ip_s = [], [] ## Create sets of sampling indices based on magnitude bins and whether trgts has "any" picks for that phase type
	for j in range(len(ip)):
		ip_p.append(np.array(list(set(ip[j]).intersection(inot_nan_p))))
		ip_s.append(np.array(list(set(ip[j]).intersection(inot_nan_s))))

	## Compute median relative error over batch for each magnitude bin (or could output other quantiles/statistics)
	q_val = np.array([0.25, 0.5, 0.75]) ## Compute weighted mean residual over the different quantile bins
	weight_bins = np.array([0.25, 1.0, 0.25]).reshape(-1,1) ## Upper quantile can be unstable
	weight_bins = weight_bins/weight_bins.sum()
	if len(q_val) == 1: weight_bins = np.array([1.0]).reshape(-1,1)
	assert(len(q_val) == len(weight_bins))
	# pdb.set_trace()
	zero_vec = np.zeros((len(q_val),1))
	res_morans_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_morans_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_morans_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_morans_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_inertia_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_inertia_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_inertia_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_inertia_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_cnt_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_cnt_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_cnt_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_cnt_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_intersection_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_intersection_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	res_intersection_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_intersection_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin

	# pdb.set_trace()
	weights = np.array([1.0, 1.0, 1.0, 1.0])
	weights = weights/weights.sum()

	median_loss_p = weights[0]*res_morans_p_per_mag_bin + weights[1]*res_inertia_p_per_mag_bin + weights[2]*res_cnt_p_per_mag_bin + weights[3]*res_intersection_p_per_mag_bin
	median_loss_s = weights[0]*res_morans_s_per_mag_bin + weights[1]*res_inertia_s_per_mag_bin + weights[2]*res_cnt_s_per_mag_bin + weights[3]*res_intersection_s_per_mag_bin

	## RMS residual over the magnitude bins (and averaged over phase types) ## Note: doing uniform weighting over magnitudes (but sampling has already been done to balance samples from different magnoitude bins)
	median_loss = np.linalg.norm((0.5*median_loss_p + 0.5*median_loss_s))/np.sqrt(n_mag_bins)

	if return_diagnostics == False:

		return median_loss

	else:

		median_res_vals_p = [res_morans_p_per_mag_bin, res_inertia_p_per_mag_bin, res_cnt_p_per_mag_bin, res_intersection_p_per_mag_bin]
		median_res_vals_s = [res_morans_s_per_mag_bin, res_inertia_s_per_mag_bin, res_cnt_s_per_mag_bin, res_intersection_s_per_mag_bin]

		res_vals_p = [misfit_morans_p, misfit_inertia_p, misfit_cnt_p, misfit_intersection_p]
		res_vals_s = [misfit_morans_s, misfit_inertia_s, misfit_cnt_s, misfit_intersection_s]

		return median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_s]



## Create probability sampling vector based on magnitudes

mag_bin = 0.5 ## Set magnitude bin
mag_vals = np.copy(Mags) ## Magnitudes of catalog (Copy the magnitudes here)
## Note: no assumption that Mags is sorted

mag_range = np.minimum((mag_vals.max() - mag_vals.min())/4.0, mag_bin)
mag_bins = np.arange(mag_vals.min(), mag_vals.max() + mag_bin, mag_bin)
ip = cKDTree(mag_vals.reshape(-1,1)).query_ball_point(mag_bins.reshape(-1,1) + mag_bin/2.0, r = mag_bin/2.0)
prob_vec = (-1.0*np.ones(len(mag_vals)))
for j in ip: prob_vec[j] = (len(j) > 0)*(1.0/np.maximum(1.0, len(j)))
prob_vec = prob_vec/prob_vec.sum()
assert(prob_vec.min() > 0)


## Set Cholesky parameters
chol_params = {}
chol_params['p_exp'] = 1.5                   # Radial function exponent (fixed integer)
chol_params['miss_pick_rate'] = 0.0              # Miss pick rate (scale_factor = 1 - miss_pick_rate)
chol_params['sigma_noise'] = 100000               # Sigma noise for cluster spreading (in meters)
chol_params['lambda_noise'] = 0.01                # Correlation between radial function and noise
chol_params['radial_factor_p'] = 1.04     # P-wave detection radius factor (before division)
chol_params['radial_factor_s'] = 0.8      # S-wave detection radius factor (before division)
chol_params['perturb_factor'] = 0.0    # Perturbation factor for ellipse parameters

## Choose batch size
n_batch = 750

######################## Test Zone #######################
test = True
if test == True:
	n_batch = 100
	srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(Srcs, Mags, Inds, locs, prob_vec, chol_params, ftrns1, pdist_p, pdist_s, n_samples = n_batch, return_features = True, Picks_P_lists = Picks_P_lists, Picks_S_lists = Picks_S_lists, debug=True)
	median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_s] = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = True)
	labels = ['Morans', 'Inertia', 'Cnts', 'Intersection']

	print(f"Loss: {median_loss}")
	## Print updated residuals
	print('\nUpdated')
	for inc, r in enumerate(median_res_vals_p):
		# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
		print('Res on %s, %s (P waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))
	print('\n')
	for inc, r in enumerate(median_res_vals_s): 
		# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
		print('Res on %s, %s (S waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))

	
	raise StopError("Stop Error")



## Sample a generation
st_time = time.time()
srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(Srcs, Mags, Inds, locs, prob_vec, chol_params, ftrns1, pdist_p, pdist_s, n_samples = n_batch, return_features = False, Picks_P_lists = Picks_P_lists, Picks_S_lists = Picks_S_lists)
print('\nData generation time %0.4f for %d samples (without features)'%(time.time() - st_time, n_batch))

## Sample a generation
st_time = time.time()
srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(Srcs, Mags, Inds, locs, prob_vec, chol_params, ftrns1, pdist_p, pdist_s, n_samples = n_batch, Picks_P_lists = Picks_P_lists, Picks_S_lists = Picks_S_lists)
print('\nData generation time %0.4f for %d samples (with features)'%(time.time() - st_time, n_batch))

## Compute residuals
st_time = time.time()
median_loss = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = False)
median_loss_initial = np.copy(median_loss)
print('\nResidual computation time %0.4f for %d samples (median loss: %0.4f)'%(time.time() - st_time, n_batch, median_loss))

## Compute residuals (with diagnostics)
st_time = time.time()
median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_s] = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = True)
median_res_vals_p_init = np.copy(median_res_vals_p)
median_res_vals_s_init = np.copy(median_res_vals_s)
print('\nResidual computation time %0.4f for %d samples (median loss: %0.4f; with diagnostics) \n'%(time.time() - st_time, n_batch, median_loss))
labels = ['Morans', 'Inertia', 'Cnts', 'Intersection']
for inc, r in enumerate(median_res_vals_p):
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (P waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))
print('\n')
for inc, r in enumerate(median_res_vals_s): 
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (S waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))


plot_on = True
if plot_on == True:

	for i in range(len(srcs_sample)):

		fig, ax = plt.subplots(2,2, sharex = True, sharey = True)
		ax[0,0].set_title('%0.3f'%Mags[ind_sample[i]])
		ax[0,1].set_title('%0.3f'%Mags[ind_sample[i]])
		locs_use = locs[Inds[ind_sample[i]]]
		for j in [[0,0], [0,1], [1,0], [1,1]]:
			ax[j[0], j[1]].scatter(locs_use[:,1], locs_use[:,0], c = 'grey', marker = '^')
			ax[j[0], j[1]].set_aspect(1.0/np.cos(np.pi*locs_use[:,0].mean()/180.0))
			ax[j[0], j[1]].scatter(srcs_sample[i,1], srcs_sample[i,0], c = 'm', marker = 's')
		## Real event (P and S)
		ax[0,0].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		ax[1,0].scatter(locs_use[Picks_S_lists[ind_sample[i]][:,1].astype('int'),1], locs_use[Picks_S_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		# ax[0,1].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		# ax[1,0].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		## Synthetic event (P and S)
		ind_found_p = ikeep_p2[np.where(ikeep_p1 == i)[0]]
		ind_found_s = ikeep_s2[np.where(ikeep_s1 == i)[0]]
		ax[0,1].scatter(locs_use[ind_found_p,1], locs_use[ind_found_p,0], c = 'red', marker = '^')
		ax[1,1].scatter(locs_use[ind_found_s,1], locs_use[ind_found_s,0], c = 'red', marker = '^')
		#ax[1,0].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		fig.set_size_inches([12,8])
		fig.savefig(path_to_file + 'Plots' + seperator + 'example_synthetic_data_%d.png'%i)
		plt.close('all')

		print('Counts %d %d; %d %d'%(len(Picks_P_lists[ind_sample[i]]), len(Picks_S_lists[ind_sample[i]]), len(ind_found_p), len(ind_found_s)))
	# irand = np.random.choice()

########################### End initilization ##########################


########################### Run optimization ###########################

def evaluate_bayesian_objective_evaluate(x, n_batch = n_batch, n_mag_bins = 5, return_config = False):
	## Set Cholesky parameters
	chol_params = {}
	chol_params['p_exp'] = x[0] # 4                   # Radial function exponent (fixed integer)
	chol_params['miss_pick_rate'] = x[1] # 0.4              # Miss pick rate (scale_factor = 1 - miss_pick_rate)
	chol_params['sigma_noise'] = x[2] # 60000               # Sigma noise for cluster spreading (in meters)
	chol_params['lambda_noise'] = x[3] # 0.005                 # Correlation between radial function and noise
	chol_params['radial_factor_p'] = x[4] # 1.6      # P-wave detection radius factor (before division)
	chol_params['radial_factor_s'] = x[5] # 1.4172001463561372      # S-wave detection radius factor (before division)
	chol_params['perturb_factor'] = x[6] # 0.4      # Perturbation factor for ellipse parameters

	## Sample a generation
	# st_time = time.time()
	srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(Srcs, Mags, Inds, locs, prob_vec, chol_params, ftrns1, pdist_p, pdist_s, n_samples = n_batch, Picks_P_lists = Picks_P_lists, Picks_S_lists = Picks_S_lists)
	# print('\nData generation time %0.4f for %d samples (with features)'%(time.time() - st_time, n_batch))

	## Compute residuals
	# st_time = time.time()
	median_loss = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = n_mag_bins, return_diagnostics = False)
	# print('\nResidual computation time %0.4f for %d samples (median loss: %0.4f)'%(time.time() - st_time, n_batch, median_loss))
	print('Loss %0.4f'%median_loss)

	if return_config == True:

		return median_loss, chol_params

	else:

		return median_loss

# from sklearn.optimize import gp_minimize

bounds = [(1.5, 5.0), # p_exp
		(0.0, 0.5), # miss_pick_rate
		(1e3, 100e3), # sigma_noise
		(0.01, 0.5), # lambda_noise
		(0.1, 5.0), # radial_factor_p
		(0.1, 5.0), # radial_factor_s
		(0.0, 0.4)] # perturb_factor

# strings = ['p_exponent', 'scale_factor']


n_repeat = 3 ## Set to 1 for no repeat
zoom_factor = (1/3.0) ## Repeat the optimization and "zoom" the bounds in proportionally closer to the optimal point
for n in range(n_repeat):

	print('\nStarting optimization iteration %d of %d'%(n + 1, n_repeat))

	if n > 0:
		x_optimal = optimize.x
		left_diff = np.array([x_optimal[j] - b[0] for j, b in enumerate(bounds)]) ## How much more point is away from left boundary
		right_diff = np.array([b[1] - x_optimal[j] for j, b in enumerate(bounds)]) ## How much more right boundary is away from optimal point
		assert(left_diff.min() >= 0)
		assert(right_diff.min() >= 0)
		bounds_copy = [(b[0] + zoom_factor*left_diff[j], b[1] - zoom_factor*right_diff[j]) for j, b in enumerate(bounds)]
		for j, b in enumerate(bounds_copy): assert((x_optimal[j] >= b[0])*(x_optimal[j] <= b[1]))
		bounds = [l for l in bounds_copy] ## Copy

	optimize = gp_minimize(evaluate_bayesian_objective_evaluate,                  # the function to minimize
	                  bounds,      # the bounds on each dimension of x
	                  acq_func="EI",      # the acquisition function
	                  n_calls=250,         # the number of evaluations of f
	                  n_random_starts=100,  # the number of random initialization points
	                  noise='gaussian',       # the noise level (optional)
	                  random_state=None, # the random seed
	                  initial_point_generator = 'lhs',
	                  model_queue_size = 150)


## Plot optimized data

print('\nFinished optimizing')
x_optimal = optimize.x

median_loss, chol_params = evaluate_bayesian_objective_evaluate(optimize.x, return_config = True)
np.savez_compressed(path_to_file + 'Grids' + seperator + 'optimized_hyperparameters_ver_1.npz', x = x_optimal)
print('Optimal parameters loss: %0.4f (initially %0.4f)'%(median_loss, median_loss_initial))

## Sample a generation
srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(Srcs, Mags, Inds, locs, prob_vec, chol_params, ftrns1, pdist_p, pdist_s, Picks_P_lists = Picks_P_lists, Picks_S_lists = Picks_S_lists, n_samples = n_batch)
## Compute residuals (with diagnostics)
st_time = time.time()
median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_s] = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = True)
labels = ['Morans', 'Inertia', 'Cnts', 'Intersection']

## Print updated residuals
print('\nUpdated')
for inc, r in enumerate(median_res_vals_p):
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (P waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))
print('\n')
for inc, r in enumerate(median_res_vals_s): 
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (S waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))

## Print previous residuals
print('\nInitial')
for inc, r in enumerate(median_res_vals_p_init):
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (P waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))
print('\n')
for inc, r in enumerate(median_res_vals_s_init): 
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (S waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))

# Find the latest version number for the output file and increment it ==========
latest_ver = 0
pattern = path_to_file + f'Grids/{name_of_project}_optimized_hyperparameters_ver_*.npz'
files = glob.glob(pattern)
for f in files:
	m = re.search(r'_ver_(\d+)', f)
	if m:
		v = int(m.group(1))
		if v > latest_ver:
			latest_ver = v
n_ver_optimize_out = latest_ver + 1
# ==========

np.savez_compressed(path_to_file + 'Grids/%s_optimized_hyperparameters_ver_%d.npz' % (name_of_project, n_ver_optimize_out), x = np.array(optimize.x))

print('Finished optimized hyperparameters')

plot_on = True
if plot_on == True:

	for i in range(len(srcs_sample)):

		fig, ax = plt.subplots(2,2, sharex = True, sharey = True)
		ax[0,0].set_title('%0.3f'%Mags[ind_sample[i]])
		ax[0,1].set_title('%0.3f'%Mags[ind_sample[i]])
		locs_use = locs[Inds[ind_sample[i]]]
		for j in [[0,0], [0,1], [1,0], [1,1]]:
			ax[j[0], j[1]].scatter(locs_use[:,1], locs_use[:,0], c = 'grey', marker = '^')
			ax[j[0], j[1]].set_aspect(1.0/np.cos(np.pi*locs_use[:,0].mean()/180.0))
			ax[j[0], j[1]].scatter(srcs_sample[i,1], srcs_sample[i,0], c = 'm', marker = 's')
		## Real event (P and S)
		ax[0,0].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		ax[1,0].scatter(locs_use[Picks_S_lists[ind_sample[i]][:,1].astype('int'),1], locs_use[Picks_S_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		# ax[0,1].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		# ax[1,0].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		## Synthetic event (P and S)
		ind_found_p = ikeep_p2[np.where(ikeep_p1 == i)[0]]
		ind_found_s = ikeep_s2[np.where(ikeep_s1 == i)[0]]
		ax[0,1].scatter(locs_use[ind_found_p,1], locs_use[ind_found_p,0], c = 'red', marker = '^')
		ax[1,1].scatter(locs_use[ind_found_s,1], locs_use[ind_found_s,0], c = 'red', marker = '^')
		#ax[1,0].scatter(locs_use[Picks_P_lists[ind_sample[i]][:,1].astype('int'),1], locs[Picks_P_lists[ind_sample[i]][:,1].astype('int'),0], c = 'red', marker = '^')
		fig.set_size_inches([12,8])
		fig.savefig(path_to_file + 'Plots' + seperator + 'example_synthetic_data_optimized_%d.png'%i)
		plt.close('all')

		print('Counts %d %d; %d %d'%(len(Picks_P_lists[ind_sample[i]]), len(Picks_S_lists[ind_sample[i]]), len(ind_found_p), len(ind_found_s)))
	# irand = np.random.choice()



# res, Trgts, arrivals = evaluate_bayesian_objective(optimize.x, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win, return_vals = True)

# strings = ['spc_random', 'spc_thresh_rand', 'coda_rate', 'coda_win', 'dist_range[0]', 'dist_range[1]', 'max_rate_events', 'max_miss_events', 'max_false_events', 'miss_pick_fraction[0]', 'miss_pick_fraction[0]']
    