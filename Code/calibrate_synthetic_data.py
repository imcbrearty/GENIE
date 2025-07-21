import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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




####################### Apply generator and optimize parameters  ####################### 

def inv_2x2(matrix):
	"""Compute inverse of a 2x2 matrix using direct formula."""
	a, b = matrix[0, 0], matrix[0, 1]
	c, d = matrix[1, 0], matrix[1, 1]
	det = a * d - b * c
	return np.array([[d, -b], [-c, a]]) / det

def radial_function(points, center, inv_cov, sigma_radial, p_exp, scale_factor):
	"""Compute Mahalanobis distances and PDF values."""
	diff = points - center
	mahalanobis2 = np.sqrt(np.sum((diff[:,0:2]/sigma_radial) @ inv_cov * (diff[:,0:2]/sigma_radial), axis=1))
	radial_pdf = scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis2, 2.0*p_exp))

	# mahalanobis3 = np.sqrt(np.sum(diff[:,0:2] @ inv_cov * diff[:,0:2], axis=1))
	# radial_pdf1= scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis3/sigma_radial, 2.0*p_exp))
	# assert(np.abs(radial_pdf - radial_pdf1).max() < 1e-2)

	return radial_pdf

def sample_correlated_noise(covariance, ind_use, sigma_noise, n_repeat = 1, cholesky_matrix = None):
	"""Generate spatially correlated noise using Cholesky decomposition.
	TO DO: use pre-computed coefficients.
	Args:
		points (np.ndarray): Array of points
		sigma_noise (float): Covariance scale parameter
		cholesky_matrix (np.ndarray): Pre-computed Cholesky matrix
	Returns:
	np.ndarray: Spatially correlated noise
	"""
	# covariance = compute_covariance(distance, sigma_noise=sigma_noise)
	if cholesky_matrix == None:
		L = np.linalg.cholesky(covariance[ind_use.reshape(-1,1), ind_use.reshape(1,-1)])
	else:
		L = np.copy(cholesky_matrix)

	z = np.random.randn(len(ind_use),n_repeat)
	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
	noise = (L @ z).squeeze()

	return noise

def logistic(x, sigma_logistic=1.0):
	"""Logistic function with scaling parameter sigma_radial."""
	# print(f'  [DEBUG] x: {x.min()}, {x.max()}')
	# print(f'  [DEBUG] sigma_logistic: {sigma_logistic}')
	# print(f'  [DEBUG] 1 / (1 + np.exp(-x/sigma_logistic)): {1 / (1 + np.exp(-x/sigma_logistic)).min()}, {1 / (1 + np.exp(-x/sigma_logistic)).max()}')
	return 1 / (1 + np.exp(-x/sigma_logistic))

def invert_probabilities(radial_pdf, noise, lambda_corr=0.5, sigma_logistic=1.0, decaying_factor=300):
    """Compute final selection probabilities."""
    # =========== Option 1: adding dumbly =============
    # Add the noise to the pdf values
    # alpha = 0.125
    # noise_pdf = alpha * (logistic(noise) - 0.5)
    # pdf_final = np.clip(radial_pdf + noise_pdf, 0, 1)

    # ========== Option 2: adding a logistic function with convex sum =============
    # noise_pdf = logistic(noise, sigma_logistic)
    # # Mixing function
    # def g(p, lam):
    #     return lam * p * (1 - p)
    # # Compute pdf_final (q)
    # g_vals = g(radial_pdf, lambda_corr)
    # pdf_final = (1 - g_vals) * radial_pdf + g_vals * noise_pdf

    # ========== Option 3: adding a logistic function with "proportional" sum =============
    def g_prop(radial_pdf):
        return radial_pdf
    
    def g_step(radial_pdf, decaying_factor=25):  # decaying_factor = 25 => decays from 0.25
        return 1 - np.power((1 - radial_pdf), decaying_factor)

    noise_pdf = 2 * (logistic(noise, sigma_logistic) - 0.5) # between -1 and 1

    pdf_final = np.clip(radial_pdf + lambda_corr * g_step(radial_pdf, decaying_factor) * noise_pdf, 0, 1)

    return pdf_final

def generate_ellipse_parameters(angle=None, length1=None, length2=None):
	"""Generate random ellipse parameters."""
	if angle is None:
		angle = np.random.uniform(0, 2*np.pi)
	if length1 is None: 
		length1 = np.random.uniform(0.9, 1.1)
	if length2 is None:
		length2 = np.random.uniform(0.9, 1.1)
    
	R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
	D = np.array([[length1, 0],
                  [0, length2]])
    
	cov_ellipse = R @ D @ D @ R.T
	inv_cov = inv_2x2(cov_ellipse)
    
	return angle, length1, length2, cov_ellipse, inv_cov

def calculate_inertia(points, indices, scale_km = 1000.0):
	"""Calculate inertia for a subset of points."""
	if len(indices) == 0:
		return 0.0
	center = points[indices].mean(axis=0, keepdims = True)
	return np.sum(((points[indices] - center)/scale_km)**2)/len(indices) ## Shuld we actually take mean?

def compute_morans_I_metric(W, ind_use, locs_use_cart, src_cart, selected_list, return_matrix = False):
	"""Calculate Moran's I for a subset of points being inside the circle of radius maximum distance between the centroid and one of the selected points.
	This method first filters the points to create a new selected vector and new W then returns morans_I_binary(new_selected, new_W)
	"""
	# Find the maximum distance between the centroid and one of the selected points
	# centroid = np.copy(src_cart) # points[selected].mean(axis=0)
	assert(len(selected_list) <= 2)
	offset_dist = np.linalg.norm(locs_use_cart - src_cart, axis = 1)
	if len(selected_list[0]) > 0:
		filtering_distance = 1.25*np.quantile(offset_dist[selected_list[0]], 0.98) ## Base the distance threshold on the first index set in selected (e.g., the targets)
	elif len(selected_list) == 1: ## If only one input, no distance
		filtering_distance = 0.0
	elif len(selected_list[1]) > 0:
		filtering_distance = 1.25*np.quantile(offset_dist[selected_list[1]], 0.98)
	else:
		filtering_distance = 0.0

	ipoints_within_radius = np.where(offset_dist < filtering_distance)[0] ## Want footprint of network within a proportion of radius of max association
	ind_use_subset = ind_use[ipoints_within_radius]

	
	W_select = W[ind_use_subset.reshape(-1,1), ind_use_subset.reshape(1,-1)]
	perm_vec = (-1*np.ones(len(ind_use))).astype('int')
	perm_vec[ipoints_within_radius] = np.arange(len(ipoints_within_radius))

	morans_vals = []
	for selected in selected_list:
		new_selected = perm_vec[selected] ## Map selectred indices to the new index ordering
		morans_vals.append(morans_I_binary(W_select, new_selected))

	if return_matrix == True:

		return np.array(morans_vals), W_select, ind_use_subset

	else:

		return np.array(morans_vals)

def morans_I_binary(W, selected):
	"""Calculate Moran's I for binary selection vector using sparse W."""
	N = W.shape[0]
	y = np.zeros(len(W)) # selected.astype(float)
	y[selected] = 1.0
	k = y.sum()
	if k == 0 or k == len(y):  # Return 0 if no points or all points are selected
		return 0.0
	y_bar = k / len(y)
	row_sum = W.sum()
	num = (W*(y[np.arange(N).reshape(-1,1)] - y_bar)*(y[np.arange(N).reshape(1,-1)] - y_bar)).sum()
	denom = np.sum((y - y_bar) ** 2)
	I = (len(y) / row_sum) * (num / denom)
	return I

## Note: this assumes "Srcs", "Mags", "Inds" are global variables that wont be changed
## Also that Picks_P_lists and Picks_S_lists are available (perm indices), and can index into the observed associated picks
def sample_synthetic_moveout_pattern_generator(prob_vec, chol_params, ftrns1, n_samples = 100, use_l1 = False, mask_noise = False, return_features = True): # n_repeat : can repeatedly draw either from the covariance matrices, or the binomial distribution

	## Sample sources
	ichoose = np.random.choice(len(Srcs), p = prob_vec, size = n_samples)
	locs_use_list = [locs[Inds[j]] for j in ichoose]
	locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
	srcs_sample = Srcs[ichoose]
	mags_sample = Mags[ichoose]
	srcs_samples_cart = ftrns1(srcs_sample)

	## Extract parameters
	p_exp = chol_params['p_exponent']  # Use optimized value
	sigma_radial_p_factor = chol_params['sigma_radial_p_factor']
	sigma_radial_s_factor = chol_params['sigma_radial_s_factor']
	sigma_radial_divider = chol_params['sigma_radial_divider']
	scale_factor = chol_params['scale_factor'] # scaling factor for the radial function - use optimized value
	sigma_noise = chol_params['sigma_noise'] # Covariance matrix/kernel distances sigma_radial, controls the spreading of the cluster - use optimized value directly
	threshold_logistic = chol_params['threshold_logistic'] 	# Logistic function sigma_radial, controls the roughness of cluster border - use optimized value
	max_value_logistic = 0.99 # < 1, the maximum value of the logistic function for the threshold, don't tune this.
	sigma_logistic = - threshold_logistic / np.log(1/max_value_logistic - 1) 
	lambda_corr = chol_params['lambda_corr'] 	# Mixing function lambda, controls the correlation between radial function and correlated noise - use optimized value
	k_neighbours = chol_params['k_neighbours']  # Use from params (though this one isn't optimized)
	angle_perturbation = chol_params['angle_perturbation']
	length_perturbation = chol_params['length_perturbation']
	miss_pick_rate = chol_params['miss_pick_rate']
	random_scale_factor_phase = chol_params['random_scale_factor_phase'] # random_scale_factor_phase = 0.35

	## Note: could likely remove scale_factor, and only use these per phase type and event scale factors
	random_scale_factor_range = [1.0 - random_scale_factor_phase, 1.0 + random_scale_factor_phase]
	random_scale_factor_phase_range = [1.0 - random_scale_factor_phase, 1.0 + random_scale_factor_phase]
	scale_factor_sample = np.random.rand(n_samples)*(random_scale_factor_range[1] - random_scale_factor_range[0]) + random_scale_factor_range[0]
	scale_factor_p, scale_factor_s = np.random.rand(2,n_samples)*(random_scale_factor_phase_range[1] - random_scale_factor_phase_range[0]) + random_scale_factor_phase_range[0]

	## Noise correlation range
	phase_noise_corr_range = [0.1, 0.4] ## Range to "weight" the independent S wave noise probabilities compared to P waves
	phase_noise_corr_sample = np.random.rand(n_samples)*(phase_noise_corr_range[1] - phase_noise_corr_range[0]) + phase_noise_corr_range[0]

	## Radius values per sources (need to add per-source random perturbation, not fixed scaling)
	sigma_radial_p = np.array([sigma_radial_p_factor * pdist_p(magnitude) / sigma_radial_divider for magnitude in mags_sample])  # P-wave detection radius
	sigma_radial_s = np.array([sigma_radial_s_factor * pdist_s(magnitude) / sigma_radial_divider for magnitude in mags_sample])  # S-wave detection radius

	## Setup absolute network parameters
	tol = 1e-8
	distance_abs = pd(ftrns1(locs), ftrns1(locs)) ## Absolute stations
	if use_l1 == False:
		covariance_abs = np.exp(-0.5*(distance_abs**2) / (sigma_noise**2)) + tol*np.eye(distance_abs.shape[0])
	else:
		covariance_abs = np.exp(-1.0*np.abs(distance_abs) / (sigma_noise**1)) + tol*np.eye(distance_abs.shape[0])

	length_scale = sigma_noise ## Is this reasonable? ## Instead, can base it on a multiple of nearest neighbors
	W_abs = np.exp(-(distance_abs**2) / (2 * (length_scale**2))) # Compute weights using a Gaussian kernel: W_ij = exp(-d^2 / (2*length_scale^2))
	np.fill_diagonal(W_abs, 0.0)  # Remove self-loops

	## Could cach the repeated set indices, and save cholesky factors for each
	# # pdb.set_trace()
	# pre_check_unique_sets = True
	# if pre_check_unique_sets == True:
	# 	set_equal = np.vstack([np.array([set(Inds[ichoose[i]]) == set(Inds[ichoose[j]]) for j in range(len(ichoose))]).reshape(1,-1) for i in range(len(ichoose))])
	# 	iset1, iset2 = np.where(set_equal > 0)
	# 	graph_components = nx.connected_components(to_networkx(Data(edge_index = torch.Tensor(np.concatenate((iset1.reshape(1,-1), iset2.reshape(1,-1)), axis = 0)).long())).to_undirected()) # .connected_components()

	ikeep_p1, ikeep_p2 = [], []
	ikeep_s1, ikeep_s2 = [], []

	for i in range(n_samples):

		# angle_p, length1_p, length2_p = experiment_result_p['parameters']['angle'], experiment_result_p['parameters']['length1'], experiment_result_p['parameters']['length2']
		angle_p, length1_p, length2_p, _, inv_cov_p = generate_ellipse_parameters(angle = None, length1 = None, length2 = None)
		angle_s = (angle_p + np.random.uniform(-angle_perturbation, angle_perturbation)) % (2.0*np.pi)
		length1_s = length1_p * np.random.uniform(length_perturbation[0], length_perturbation[1])
		length2_s = length2_p * np.random.uniform(length_perturbation[0], length_perturbation[1])
		angle_s, length1_s, length2_s, _, inv_cov_s = generate_ellipse_parameters(angle = angle_s, length1 = length1_s, length2 = length2_s)

		# points, center, inv_cov, sigma_radial
		## Radial component (rather than fixed scale_factor, need to be a perturbation per event within tolerance)
		radial_pdf_p = radial_function(locs_use_cart_list[i], srcs_samples_cart[i], inv_cov_p, sigma_radial_p[i], p_exp, scale_factor_p[i]*scale_factor_sample[i]) ## Note: p_exp is the exponent term
		radial_pdf_s = radial_function(locs_use_cart_list[i], srcs_samples_cart[i], inv_cov_s, sigma_radial_s[i], p_exp, scale_factor_s[i]*scale_factor_sample[i])

		## Sample P and S wave noise (do not use pre-built Cholesky matrix for S wave sampling)
		## Could "cache" the cholesky factor for each unique (repeated) set of Inds[ichoose[i]]. 
		## In cases where Inds[ichoose[i]] is repeated, then the cholesky factor is fixed. (should occur in practice during sampling)
		if np.random.rand() < 0.5: ## Note: can make sigma_noise a random sample over a range
			noise_p = sample_correlated_noise(covariance_abs, Inds[ichoose[i]], sigma_noise, n_repeat = 1)
			noise_s = noise_p + phase_noise_corr_sample[i]*sample_correlated_noise(covariance_abs, Inds[ichoose[i]], sigma_noise, n_repeat = 1)
		else:
			noise_s = sample_correlated_noise(covariance_abs, Inds[ichoose[i]], sigma_noise, n_repeat = 1)
			noise_p = noise_s + phase_noise_corr_sample[i]*sample_correlated_noise(covariance_abs, Inds[ichoose[i]], sigma_noise, n_repeat = 1)			
    
		# else:

		# Updated selection
		if mask_noise == True:
			updated_pdf_p = np.copy(radial_pdf_p)
			updated_pdf_s = np.copy(radial_pdf_s)
		else:
			updated_pdf_p = invert_probabilities(radial_pdf_p, noise_p, lambda_corr, sigma_logistic) # radial_pdf, noise, lambda_corr, sigma_logistic   
			updated_pdf_s = invert_probabilities(radial_pdf_s, noise_s, lambda_corr, sigma_logistic) # radial_pdf, noise, lambda_corr, sigma_logistic   

		rand_miss_mask_p = np.ones(len(updated_pdf_p))
		rand_miss_mask_s = np.ones(len(updated_pdf_s))
		if miss_pick_rate[1] > 0:
			miss_pick_rate_phase_val = [0.0, 0.2] ## The proportion that different phase types mix their noise level
			miss_pick_rate_phase_corr = np.random.rand()*(miss_pick_rate_phase_val[1] - miss_pick_rate_phase_val[0]) + miss_pick_rate_phase_val[0]
			if np.random.rand() > 0.5:
				miss_pick_rate_sample_p = np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]
				miss_pick_rate_sample_s = miss_pick_rate_sample_p + miss_pick_rate_phase_corr*np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]
			else:
				miss_pick_rate_sample_s = np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]
				miss_pick_rate_sample_p = miss_pick_rate_sample_s + miss_pick_rate_phase_corr*np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]

			rand_miss_mask_p = 1.0*(np.random.rand(len(radial_pdf_p)) > miss_pick_rate_sample_p) ## If less than miss rate, then definitely miss pick
			rand_miss_mask_s = 1.0*(np.random.rand(len(radial_pdf_s)) > miss_pick_rate_sample_s)

		updated_mask_p = np.random.binomial(1, updated_pdf_p*rand_miss_mask_p)
		updated_idx_p = np.where(updated_mask_p)[0]
		updated_mask_s = np.random.binomial(1, updated_pdf_s*rand_miss_mask_s)
		updated_idx_s = np.where(updated_mask_s)[0]

		ikeep_p1.append(i*np.ones(len(updated_idx_p)))
		ikeep_p2.append(updated_idx_p)
		ikeep_s1.append(i*np.ones(len(updated_idx_s)))
		ikeep_s2.append(updated_idx_s)

	## Merge results
	ikeep_p1 = np.hstack(ikeep_p1)
	ikeep_p2 = np.hstack(ikeep_p2)
	ikeep_s1 = np.hstack(ikeep_s1)
	ikeep_s2 = np.hstack(ikeep_s2)

	## Add computation of features
	Features = []
	if return_features == True:

		Morans_pred_p = []
		Morans_pred_s = []
		Inertia_pred_p = []
		Inertia_pred_s = []
		Cnt_pred_p, Cnt_pred_s = [], []

		Morans_trgt_p = []
		Morans_trgt_s = []
		Inertia_trgt_p = []
		Inertia_trgt_s = []
		Cnt_trgt_p, Cnt_trgt_s = [], []

		Intersection_p_ratio = []
		Intersection_s_ratio = []

		for i in range(n_samples):
			ind_pred_p = ikeep_p2[np.where(ikeep_p1 == i)[0]]
			ind_pred_s = ikeep_s2[np.where(ikeep_s1 == i)[0]]		
			ind_trgt_p = Picks_P_lists[ichoose[i]][:,1].astype('int')
			ind_trgt_s = Picks_S_lists[ichoose[i]][:,1].astype('int')	

			## [1]. Morans metric
			morans_metric_p = compute_morans_I_metric(W_abs, Inds[ichoose[i]], locs_use_cart_list[i], srcs_samples_cart[i], [ind_trgt_p, ind_pred_p]) ## Should pass in trgt first, as filtering scale is set by target data
			morans_metric_s = compute_morans_I_metric(W_abs, Inds[ichoose[i]], locs_use_cart_list[i], srcs_samples_cart[i], [ind_trgt_s, ind_pred_s])
			Morans_trgt_p.append(morans_metric_p[0])
			Morans_trgt_s.append(morans_metric_s[0])
			Morans_pred_p.append(morans_metric_p[1])
			Morans_pred_s.append(morans_metric_s[1])

			# morans_metric_p_trgt, morans_metric_p_pred = morans_metric_p
			# morans_metric_s_trgt, morans_metric_s_pred = morans_metric_s

			## [2]. Inertia
			Inertia_trgt_p.append(calculate_inertia(locs_use_cart_list[i], ind_trgt_p))
			Inertia_trgt_s.append(calculate_inertia(locs_use_cart_list[i], ind_trgt_s))
			Inertia_pred_p.append(calculate_inertia(locs_use_cart_list[i], ind_pred_p))
			Inertia_pred_s.append(calculate_inertia(locs_use_cart_list[i], ind_pred_s))

			## [3]. Counts
			Cnt_trgt_p.append(len(ind_trgt_p))
			Cnt_trgt_s.append(len(ind_trgt_s))
			Cnt_pred_p.append(len(ind_pred_p))
			Cnt_pred_s.append(len(ind_pred_s))

			## [4]. Intersection
			Intersection_p_ratio.append((1.0 if (len(ind_trgt_p) > 0) else np.nan)*len(set(ind_trgt_p).intersection(ind_pred_p))/np.maximum(len(ind_trgt_p), 1.0))
			Intersection_s_ratio.append((1.0 if (len(ind_trgt_s) > 0) else np.nan)*len(set(ind_trgt_s).intersection(ind_pred_s))/np.maximum(len(ind_trgt_s), 1.0))

		## Merge outputs
		Morans_pred_p = np.hstack(Morans_pred_p)
		Morans_pred_s = np.hstack(Morans_pred_s)
		Morans_trgt_p = np.hstack(Morans_trgt_p)
		Morans_trgt_s = np.hstack(Morans_trgt_s)

		Inertia_pred_p = np.hstack(Inertia_pred_p)
		Inertia_pred_s = np.hstack(Inertia_pred_s)
		Inertia_trgt_p = np.hstack(Inertia_trgt_p)
		Inertia_trgt_s = np.hstack(Inertia_trgt_s)

		Cnt_pred_p = np.hstack(Cnt_pred_p)
		Cnt_pred_s = np.hstack(Cnt_pred_s)
		Cnt_trgt_p = np.hstack(Cnt_trgt_p)
		Cnt_trgt_s = np.hstack(Cnt_trgt_s)

		Intersection_p_ratio = np.hstack(Intersection_p_ratio)
		Intersection_s_ratio = np.hstack(Intersection_s_ratio)

		Features.append([Morans_pred_p, Morans_pred_s, Morans_trgt_p, Morans_trgt_s])
		Features.append([Inertia_pred_p, Inertia_pred_s, Inertia_trgt_p, Inertia_trgt_s])
		Features.append([Cnt_pred_p, Cnt_pred_s, Cnt_trgt_p, Cnt_trgt_s])
		Features.append([Intersection_p_ratio, Intersection_s_ratio])
		# Features = [[Morans_trgt_p, Morans_trgt_s, Morans_pred_p, Morans_pred_p], [Inertia_trgt_p, Inertia_trgt_s, Inertia_pred_p, Inertia_pred_p], ]

		# error('Not yet implemented')
		# pass

	return srcs_sample, mags_sample, Features, ichoose, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2]


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

		return median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_p]



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
chol_params['p_exponent'] = 4                   # Radial function exponent (fixed integer)
chol_params['scale_factor'] = 0.5995930570290529              # Scaling factor for radial function
chol_params['sigma_noise'] = 60000               # Sigma noise for cluster spreading (in meters)
chol_params['sigma_radial_divider'] = 10.0         # Divisor for detection radius calculation
chol_params['threshold_logistic'] = 8.0           # Logistic function threshold
chol_params['lambda_corr'] = 0.005                 # Correlation between radial function and noise
chol_params['k_neighbours'] = 8                  # Number of neighbors for analysis
chol_params['sigma_radial_p_factor'] = 1.6      # P-wave detection radius factor (before division)
chol_params['sigma_radial_s_factor'] = 1.4172001463561372      # S-wave detection radius factor (before division)
chol_params['angle_perturbation'] = 0.5689419133182898      # S-wave angle perturbation range
chol_params['length_perturbation'] = [0.8730268537646498, 1.3] # S-wave ellipse length perturbation range
chol_params['miss_pick_rate'] = [0.0, 0.15]
chol_params['random_scale_factor_phase'] = 0.35 ## should be between [0.0,1.0] (smaller means less random pertubation to distance threshold)


## Sample a generation
st_time = time.time()
n_batch = 1000
srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(prob_vec, chol_params, ftrns1, n_samples = n_batch, return_features = False)
print('\nData generation time %0.4f for %d samples (without features)'%(time.time() - st_time, n_batch))

## Sample a generation
st_time = time.time()
srcs_sample, mags_sample, features, ind_sample, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2] = sample_synthetic_moveout_pattern_generator(prob_vec, chol_params, ftrns1, n_samples = n_batch)
print('\nData generation time %0.4f for %d samples (with features)'%(time.time() - st_time, n_batch))

## Compute residuals
st_time = time.time()
median_loss = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = False)
print('\nResidual computation time %0.4f for %d samples (median loss: %0.4f)'%(time.time() - st_time, n_batch, median_loss))

## Compute residuals (with diagnostics)
st_time = time.time()
median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_p] = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = True)
print('\nResidual computation time %0.4f for %d samples (median loss: %0.4f; with diagnostics) \n'%(time.time() - st_time, n_batch, median_loss))
labels = ['Morans', 'Inertia', 'Cnts', 'Intersection']
for inc, r in enumerate(median_res_vals_p):
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (P waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))
print('\n')
for inc, r in enumerate(median_res_vals_p): 
	# strings_p = [str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))]
	print('Res on %s, %s (S waves)'%(labels[inc], ' '.join([str(np.round(r[j],3)) + ',' if j < (len(r) - 1) else str(np.round(r[j],3)) for j in range(len(r))])))


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





# def radial_function(points, center, inv_cov, sigma_radial, p_exp=4, scale_factor=1.0):
# 	"""Compute Mahalanobis distances and PDF values."""
# 	diff = points - center
# 	mahalanobis2 = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
# 	radial_pdf = scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis2/sigma_radial, 2.0*p_exp))
# 	return radial_pdf


# def compute_covariance(distance, sigma_noise=1.0): ## This function is basically "one line of code", so not worth adding
# 	"""Compute covariance matrix from distances."""
# 	N = distance.shape[0]
# 	covariance = np.exp(-(distance**2) / (2 * (sigma_noise**2)))
# 	covariance += 1e-8 * np.eye(N)
# 	return covariance

# def sample_correlated_noise(locs_cart, sigma_noise, tol = 1e-8, n_repeat = 1, use_l1 = False, cholesky_matrix = None):
# 	"""Generate spatially correlated noise using Cholesky decomposition.
# 	TO DO: use pre-computed coefficients.
# 	Args:
# 		points (np.ndarray): Array of points
# 		sigma_noise (float): Covariance scale parameter
# 		cholesky_matrix (np.ndarray): Pre-computed Cholesky matrix
# 	Returns:
# 	np.ndarray: Spatially correlated noise
# 	"""

# 	if cholesky_matrix is None:
# 		distance = pd(locs_cart)
# 		if use_l1 == False:
# 			covariance = np.exp(-0.5*(distance**2) / (sigma_noise**2)) + tol*np.eye(distance.shape[0]) ## Can use a different L1 norm to establish correlation length
# 		else:
# 			covariance = np.exp(-1.0*(distance**1) / (sigma_noise**1.0)) + tol*np.eye(distance.shape[0]) ## Can use a different L1 norm to establish correlation length			

# 		# covariance = compute_covariance(distance, sigma_noise=sigma_noise)
# 		L = np.linalg.cholesky(covariance)
# 	else:
# 		L = cholesky_matrix
    

# 	z = np.random.randn(len(locs_cart),n_repeat)
# 	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
# 	noise = (L @ z).squeeze()

# 	return noise


		# if mask_noise == True:

		# 	rand_miss_mask_p = np.ones(len(radial_pdf_p))
		# 	rand_miss_mask_s = np.ones(len(radial_pdf_s))
		# 	if miss_pick_rate[1] > 0:
		# 		miss_pick_rate_phase_val = [0.0, 0.2]
		# 		miss_pick_rate_phase_corr = np.random.rand()*(miss_pick_rate_phase_val[1] - miss_pick_rate_phase_val[0]) + miss_pick_rate_phase_val[0]
		# 		if np.random.rand() > 0.5:
		# 			miss_pick_rate_sample_p = np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]
		# 			miss_pick_rate_sample_s = miss_pick_rate_sample_p + miss_pick_rate_phase_corr*np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]
		# 		else:
		# 			miss_pick_rate_sample_s = np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]
		# 			miss_pick_rate_sample_p = miss_pick_rate_sample_s + miss_pick_rate_phase_corr*np.random.rand()*(miss_pick_rate[1] - miss_pick_rate[0]) + miss_pick_rate[0]

		# 		rand_miss_mask_p = 1.0*(np.random.rand(len(radial_pdf_p)) > miss_pick_rate_sample_p) ## If less than miss rate, then mask out
		# 		rand_miss_mask_s = 1.0*(np.random.rand(len(radial_pdf_s)) > miss_pick_rate_sample_s)

		# 	# Phase A: Initial selection
		# 	initial_mask_p = np.random.binomial(1, radial_pdf_p*rand_miss_mask_p)
		# 	initial_idx_p = np.where(initial_mask_p)[0]
		# 	initial_mask_s = np.random.binomial(1, radial_pdf_s*rand_miss_mask_s)
		# 	initial_idx_s = np.where(initial_mask_s)[0]
		# 	updated_idx_p = np.copy(initial_idx_p)
		# 	updated_idx_s = np.copy(initial_idx_s)



# def compute_W(points, length_scale=None):
#     """Compute weights matrix W for the given points.
    
#     Args:
#         points (np.ndarray): Array of shape (N, 2) containing point coordinates
#         W (np.ndarray, optional): Precomputed weights matrix. If None, it will be computed.
#         length_scale (float, optional): Characteristic length scale for Gaussian kernel. If None,
#                                         it will be set to a fraction of the median distance.
        
#     Returns:
#         np.ndarray: Weights matrix W of shape (N, N)
#     """
#         # Compute pairwise distances
#     dists = distances(points)
#     if length_scale is None:
#         # Set a characteristic length scale (e.g., median distance or a fraction of max distance)
#         length_scale = np.clip((1/4)*np.median(dists[dists > 0]), 0, 25000)  # TO CHANGE
#     # Compute weights using a Gaussian kernel: W_ij = exp(-d^2 / (2*length_scale^2))
#     W = np.exp(-dists**2 / (2 * length_scale**2))
#     np.fill_diagonal(W, 0.0)  # Remove self-loops
#     return W



