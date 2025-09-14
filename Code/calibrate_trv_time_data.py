import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
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




####################### Apply generator and optimize parameters  ####################### 

def inv_2x2(matrix):
	"""Compute inverse of a 2x2 matrix using direct formula."""
	a, b = matrix[0, 0], matrix[0, 1]
	c, d = matrix[1, 0], matrix[1, 1]
	det = a * d - b * c
	return np.array([[d, -b], [-c, a]]) / det

# def radial_function(points, center, inv_cov, sigma_radial, p_exp, scale_factor):
# 	"""Compute Mahalanobis distances and PDF values."""
# 	diff = points - center
# 	mahalanobis2 = np.sqrt(np.sum((diff[:,0:2]/sigma_radial) @ inv_cov * (diff[:,0:2]/sigma_radial), axis=1))
# 	radial_pdf = scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis2, 2.0*p_exp))

# 	# mahalanobis3 = np.sqrt(np.sum(diff[:,0:2] @ inv_cov * diff[:,0:2], axis=1))
# 	# radial_pdf1= scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis3/sigma_radial, 2.0*p_exp))
# 	# assert(np.abs(radial_pdf - radial_pdf1).max() < 1e-2)

# 	return radial_pdf

# def sample_correlated_noise(covariance, ind_use, sigma_noise, n_repeat = 1, cholesky_matrix = None):
# 	"""Generate spatially correlated noise using Cholesky decomposition.
# 	TO DO: use pre-computed coefficients.
# 	Args:
# 		points (np.ndarray): Array of points
# 		sigma_noise (float): Covariance scale parameter
# 		cholesky_matrix (np.ndarray): Pre-computed Cholesky matrix
# 	Returns:
# 	np.ndarray: Spatially correlated noise
# 	"""
# 	# covariance = compute_covariance(distance, sigma_noise=sigma_noise)
# 	if cholesky_matrix == None:
# 		L = np.linalg.cholesky(covariance[ind_use.reshape(-1,1), ind_use.reshape(1,-1)])
# 	else:
# 		L = np.copy(cholesky_matrix)

# 	z = np.random.randn(len(ind_use),n_repeat)
# 	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
# 	noise = (L @ z).squeeze()

# 	return noise

def sample_correlated_noise_pre_computed(cholesky_matrix, ind_use, n_repeat = 1):
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
	# if cholesky_matrix == None:
	# 	L = np.linalg.cholesky(covariance[ind_use.reshape(-1,1), ind_use.reshape(1,-1)])
	# else:
	# 	L = np.copy(cholesky_matrix)

	z = np.random.randn(len(cholesky_matrix),n_repeat)
	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
	noise = (cholesky_matrix @ z).squeeze()[ind_use]

	return noise

def sample_correlated_travel_time_noise_pre_computed(cholesky_matrix_trv, mean_vec, bias_factors, std_factor, ind_use, compute_log_likelihood = False, observed_indices = None, observed_times = None, min_tol = 0.005, n_repeat = 1):
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
	# if cholesky_matrix == None:
	# 	L = np.linalg.cholesky(covariance[ind_use.reshape(-1,1), ind_use.reshape(1,-1)])
	# else:
	# 	L = np.copy(cholesky_matrix)

	## Scale absolute "mean" travel times by bias factor
	if len(bias_factors) > 1:
		bias_val = np.random.uniform(1.0 - bias_factors[0], 1.0 + bias_factors[1])
	else:
		bias_val = np.random.uniform(1.0 - bias_factors[0], 1.0 + bias_factors[0])		

	## Set the standard deviation as proportional to travel time
	if len(std_factor) > 1:
		std_val = np.random.uniform(std_factor[0], std_factor[0] + std_factor[1])
	else:
		std_val = std_factor[0]

	standard_deviation = np.diag(mean_vec*std_val)

	# std_val = np.random.uniform(min_tol, std_factor)
	# standard_deviation = np.diag(mean_vec*std_factor)

	z = np.random.randn(len(cholesky_matrix_trv), n_repeat)

	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
	scaled_chol_matrix = (standard_deviation @ cholesky_matrix_trv)
	scaled_mean_vec = mean_vec*bias_val

	# Compute simulated times
	simulated_times = ((scaled_chol_matrix @ z) + (scaled_mean_vec).reshape(-1,1))[ind_use].squeeze() # [ind_use]

	if compute_log_likelihood == False:

		return simulated_times

	else: ## In this case, compute the log likelihood of the observations (and simulations) given the model

		# pdb.set_trace()
		# inv_cov_subset = np.linalg.pinv((cholesky_matrix_trv @ cholesky_matrix_trv.T)[ind_use[observed_indices].reshape(-1,1), ind_use[observed_indices].reshape(1,-1)])
		cov_subset = (scaled_chol_matrix @ scaled_chol_matrix.T)[ind_use[observed_indices].reshape(-1,1), ind_use[observed_indices].reshape(1,-1)]
		res_vec_obs = observed_times - scaled_mean_vec[ind_use[observed_indices]]
		res_vec_sim = simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]]
		inv_cov_prod_res = np.linalg.solve(cov_subset, res_vec_obs.reshape(-1,1))
		inv_cov_prod_sim = np.linalg.solve(cov_subset, res_vec_sim.reshape(-1,1))
		# log_likelihood_obs = -(len(observed_indices)/2.0)*np.log(2.0*np.pi) - 1.0*(np.log(np.diag(scaled_chol_matrix)).sum()) - 0.5*((observed_times - scaled_mean_vec[ind_use[observed_indices]])*(inv_cov_subset @ (observed_times - scaled_mean_vec[ind_use[observed_indices]]).reshape(-1,1))).sum()
		# log_likelihood_sim = -(len(observed_indices)/2.0)*np.log(2.0*np.pi) - 1.0*(np.log(np.diag(scaled_chol_matrix)).sum()) - 0.5*((simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]])*(inv_cov_subset @ (simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]]).reshape(-1,1))).sum()
		log_likelihood_obs = -(len(observed_indices)/2.0)*np.log(2.0*np.pi) - 1.0*(np.log(np.diag(scaled_chol_matrix)).sum()) - 0.5*(res_vec_obs*inv_cov_prod_res).sum() # ((observed_times - scaled_mean_vec[ind_use[observed_indices]])*(inv_cov_subset @ (observed_times - scaled_mean_vec[ind_use[observed_indices]]).reshape(-1,1))).sum()
		log_likelihood_sim = -(len(observed_indices)/2.0)*np.log(2.0*np.pi) - 1.0*(np.log(np.diag(scaled_chol_matrix)).sum()) - 0.5*(res_vec_sim*inv_cov_prod_sim).sum() # ((simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]])*(inv_cov_subset @ (simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]]).reshape(-1,1))).sum()

		return simulated_times, log_likelihood_obs, log_likelihood_sim

# def sample_correlated_travel_time_noise_pre_computed_backup(cholesky_matrix_trv, mean_vec, bias_factor, std_factor, ind_use, compute_log_likelihood = False, observed_indices = None, observed_times = None, min_tol = 1e-8, n_repeat = 1):
# 	"""Generate spatially correlated noise using Cholesky decomposition.
# 	TO DO: use pre-computed coefficients.
# 	Args:
# 		points (np.ndarray): Array of points
# 		sigma_noise (float): Covariance scale parameter
# 		cholesky_matrix (np.ndarray): Pre-computed Cholesky matrix
# 	Returns:
# 	np.ndarray: Spatially correlated noise
# 	"""
# 	# covariance = compute_covariance(distance, sigma_noise=sigma_noise)
# 	# if cholesky_matrix == None:
# 	# 	L = np.linalg.cholesky(covariance[ind_use.reshape(-1,1), ind_use.reshape(1,-1)])
# 	# else:
# 	# 	L = np.copy(cholesky_matrix)

# 	## Scale absolute "mean" travel times by bias factor
# 	bias_val = np.random.uniform(1.0 - bias_factor, 1.0 + bias_factor)

# 	## Set the standard deviation as proportional to travel time
# 	std_val = np.random.uniform(min_tol, std_factor)

# 	standard_deviation = np.diag(mean_vec*std_val)

# 	z = np.random.randn(len(cholesky_matrix_trv), n_repeat)

# 	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
# 	scaled_chol_matrix = (standard_deviation @ cholesky_matrix_trv)
# 	scaled_mean_vec = mean_vec*bias_val

# 	# Compute simulated times
# 	simulated_times = ((scaled_chol_matrix @ z) + (scaled_mean_vec).reshape(-1,1))[ind_use].squeeze() # [ind_use]

# 	if compute_log_likelihood == False:

# 		return simulated_times

# 	else: ## In this case, compute the log likelihood of the observations (and simulations) given the model

# 		# pdb.set_trace()
# 		# inv_cov_subset = np.linalg.pinv((cholesky_matrix_trv @ cholesky_matrix_trv.T)[ind_use[observed_indices].reshape(-1,1), ind_use[observed_indices].reshape(1,-1)])
# 		inv_cov_subset = np.linalg.pinv((scaled_chol_matrix @ scaled_chol_matrix.T)[ind_use[observed_indices].reshape(-1,1), ind_use[observed_indices].reshape(1,-1)])
# 		log_likelihood_obs = -(len(observed_indices)/2.0)*np.log(2.0*np.pi) - 1.0*(np.log(np.diag(scaled_chol_matrix)).sum()) - 0.5*((observed_times - scaled_mean_vec[ind_use[observed_indices]])*(inv_cov_subset @ (observed_times - scaled_mean_vec[ind_use[observed_indices]]).reshape(-1,1))).sum()
# 		log_likelihood_sim = -(len(observed_indices)/2.0)*np.log(2.0*np.pi) - 1.0*(np.log(np.diag(scaled_chol_matrix)).sum()) - 0.5*((simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]])*(inv_cov_subset @ (simulated_times[observed_indices] - scaled_mean_vec[ind_use[observed_indices]]).reshape(-1,1))).sum()

# 		return simulated_times, log_likelihood_obs, log_likelihood_sim

# def logistic(x, sigma_logistic=1.0):
# 	"""Logistic function with scaling parameter sigma_radial."""
# 	# print(f'  [DEBUG] x: {x.min()}, {x.max()}')
# 	# print(f'  [DEBUG] sigma_logistic: {sigma_logistic}')
# 	# print(f'  [DEBUG] 1 / (1 + np.exp(-x/sigma_logistic)): {1 / (1 + np.exp(-x/sigma_logistic)).min()}, {1 / (1 + np.exp(-x/sigma_logistic)).max()}')
# 	return 1 / (1 + np.exp(-x/sigma_logistic))

# def invert_probabilities(radial_pdf, noise, lambda_corr=0.5, sigma_logistic=1.0, decaying_factor=300):
#     """Compute final selection probabilities."""
#     # =========== Option 1: adding dumbly =============
#     # Add the noise to the pdf values
#     # alpha = 0.125
#     # noise_pdf = alpha * (logistic(noise) - 0.5)
#     # pdf_final = np.clip(radial_pdf + noise_pdf, 0, 1)

#     # ========== Option 2: adding a logistic function with convex sum =============
#     # noise_pdf = logistic(noise, sigma_logistic)
#     # # Mixing function
#     # def g(p, lam):
#     #     return lam * p * (1 - p)
#     # # Compute pdf_final (q)
#     # g_vals = g(radial_pdf, lambda_corr)
#     # pdf_final = (1 - g_vals) * radial_pdf + g_vals * noise_pdf

#     # ========== Option 3: adding a logistic function with "proportional" sum =============
#     def g_prop(radial_pdf):
#         return radial_pdf
    
#     def g_step(radial_pdf, decaying_factor=25):  # decaying_factor = 25 => decays from 0.25
#         return 1 - np.power((1 - radial_pdf), decaying_factor)

#     noise_pdf = 2 * (logistic(noise, sigma_logistic) - 0.5) # between -1 and 1

#     pdf_final = np.clip(radial_pdf + lambda_corr * g_step(radial_pdf, decaying_factor) * noise_pdf, 0, 1)

#     return pdf_final

# def generate_ellipse_parameters(angle=None, length1=None, length2=None):
# 	"""Generate random ellipse parameters."""
# 	if angle is None:
# 		angle = np.random.uniform(0, 2*np.pi)
# 	if length1 is None: 
# 		length1 = np.random.uniform(0.9, 1.1)
# 	if length2 is None:
# 		length2 = np.random.uniform(0.9, 1.1)
    
# 	R = np.array([[np.cos(angle), -np.sin(angle)],
#                   [np.sin(angle), np.cos(angle)]])
# 	D = np.array([[length1, 0],
#                   [0, length2]])
    
# 	cov_ellipse = R @ D @ D @ R.T
# 	inv_cov = inv_2x2(cov_ellipse)
    
# 	return angle, length1, length2, cov_ellipse, inv_cov

# def calculate_inertia(points, indices, scale_km = 1000.0):
# 	"""Calculate inertia for a subset of points."""
# 	if len(indices) == 0:
# 		return 0.0
# 	center = points[indices].mean(axis=0, keepdims = True)
# 	return np.sum(((points[indices] - center)/scale_km)**2)/len(indices) ## Shuld we actually take mean?

# def compute_morans_I_metric(W, ind_use, locs_use_cart, src_cart, selected_list, return_matrix = False):
# 	"""Calculate Moran's I for a subset of points being inside the circle of radius maximum distance between the centroid and one of the selected points.
# 	This method first filters the points to create a new selected vector and new W then returns morans_I_binary(new_selected, new_W)
# 	"""
# 	# Find the maximum distance between the centroid and one of the selected points
# 	# centroid = np.copy(src_cart) # points[selected].mean(axis=0)
# 	assert(len(selected_list) <= 2)
# 	offset_dist = np.linalg.norm(locs_use_cart - src_cart, axis = 1)
# 	if len(selected_list[0]) > 0:
# 		filtering_distance = 1.25*np.quantile(offset_dist[selected_list[0]], 0.98) ## Base the distance threshold on the first index set in selected (e.g., the targets)
# 	elif len(selected_list) == 1: ## If only one input, no distance
# 		filtering_distance = 0.0
# 	elif len(selected_list[1]) > 0:
# 		filtering_distance = 1.25*np.quantile(offset_dist[selected_list[1]], 0.98)
# 	else:
# 		filtering_distance = 0.0

# 	ipoints_within_radius = np.where(offset_dist < filtering_distance)[0] ## Want footprint of network within a proportion of radius of max association
# 	ind_use_subset = ind_use[ipoints_within_radius]

	
# 	W_select = W[ind_use_subset.reshape(-1,1), ind_use_subset.reshape(1,-1)]
# 	perm_vec = (-1*np.ones(len(ind_use))).astype('int')
# 	perm_vec[ipoints_within_radius] = np.arange(len(ipoints_within_radius))

# 	morans_vals = []
# 	for selected in selected_list:
# 		new_selected = perm_vec[selected] ## Map selectred indices to the new index ordering
# 		morans_vals.append(morans_I_binary(W_select, new_selected))

# 	if return_matrix == True:

# 		return np.array(morans_vals), W_select, ind_use_subset

# 	else:

# 		return np.array(morans_vals)

# def morans_I_binary(W, selected):
# 	"""Calculate Moran's I for binary selection vector using sparse W."""
# 	N = W.shape[0]
# 	y = np.zeros(len(W)) # selected.astype(float)
# 	y[selected] = 1.0
# 	k = y.sum()
# 	if k == 0 or k == len(y):  # Return 0 if no points or all points are selected
# 		return 0.0
# 	y_bar = k / len(y)
# 	row_sum = W.sum()
# 	num = (W*(y[np.arange(N).reshape(-1,1)] - y_bar)*(y[np.arange(N).reshape(1,-1)] - y_bar)).sum()
# 	denom = np.sum((y - y_bar) ** 2)
# 	I = (len(y) / row_sum) * (num / denom)
# 	return I
	
## Note: this assumes "Srcs", "Mags", "Inds" are global variables that wont be changed
## Also that Picks_P_lists and Picks_S_lists are available (perm indices), and can index into the observed associated picks
def simulate_travel_times(prob_vec, chol_params, ftrns1, n_samples = 100, use_l1 = False, srcs = None, mags = None, ichoose = None, locs_use_list = None, ind_use_slice = None, mask_noise = False, return_features = True): # n_repeat : can repeatedly draw either from the covariance matrices, or the binomial distribution

	if srcs is None:
		## Sample sources
		if ichoose is None: ichoose = np.random.choice(len(Srcs), p = prob_vec, size = n_samples)
		locs_use_list = [locs[Inds[j]] for j in ichoose]
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
		srcs_sample = Srcs[ichoose]
		mags_sample = Mags[ichoose]
		srcs_samples_cart = ftrns1(srcs_sample)
		ind_use_slice = [Inds[ichoose[i]] for i in range(len(ichoose))]
		sample_fixed = False

	else:
		ichoose = np.arange(len(srcs))
		n_samples = len(srcs)
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
		srcs_sample = np.copy(srcs)
		mags_sample = np.copy(mags_sample)
		srcs_samples_cart = ftrns1(srcs_sample)
		# ind_use_slice = [np.arange(len(locs)) for i in range(len(ichoose))]
		sample_fixed = True


	rel_trv_factor1 = chol_params['relative_travel_time_factor1'] # random_scale_factor_phase = 0.35
	rel_trv_factor2 = chol_params['relative_travel_time_factor2'] # random_scale_factor_phase = 0.35
	travel_time_bias_scale_factor1 = chol_params['travel_time_bias_scale_factor1']
	travel_time_bias_scale_factor2 = chol_params['travel_time_bias_scale_factor2']
	correlation_scale_distance = chol_params['correlation_scale_distance']

	## Setup absolute network parameters
	tol = 1e-8
	distance_abs = pd(ftrns1(locs), ftrns1(locs)) ## Absolute stations
	if use_l1 == False:
		# covariance_abs = np.exp(-0.5*(distance_abs**2) / (sigma_noise**2)) + tol*np.eye(distance_abs.shape[0])
		covariance_trv = np.exp(-0.5*(distance_abs**2) / (correlation_scale_distance**2)) + tol*np.eye(distance_abs.shape[0])
	else:
		# covariance_abs = np.exp(-1.0*np.abs(distance_abs) / (sigma_noise**1)) + tol*np.eye(distance_abs.shape[0])
		covariance_trv = np.exp(-1.0*np.abs(distance_abs) / (correlation_scale_distance**1)) + tol*np.eye(distance_abs.shape[0])


	chol_trv_matrix = np.linalg.cholesky(covariance_trv)


	Log_prob_p = []
	Log_prob_s = []
	Simulated_p = []
	Simulated_s = []
	scale_log_prob = 100.0

	for i in range(n_samples):
		## Sample correlated travel time noise
		trv_out_vals = trv(torch.Tensor(locs).to(device), torch.Tensor(srcs_sample[i].reshape(1,-1)).to(device)).cpu().detach().numpy()
		if sample_fixed == False:
			time_trgt_p, time_trgt_s = Picks_P_lists[ichoose[i]][:,0].astype('int') - srcs_sample[i,3], Picks_S_lists[ichoose[i]][:,0].astype('int') - srcs_sample[i,3]
			ind_trgt_p, ind_trgt_s = Picks_P_lists[ichoose[i]][:,1].astype('int'), Picks_S_lists[ichoose[i]][:,1].astype('int')
			simulated_trv_p, log_likelihood_obs_p, log_likelihood_sim_p = sample_correlated_travel_time_noise_pre_computed(chol_trv_matrix, trv_out_vals[0,:,0], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], ind_use_slice[i], observed_times = time_trgt_p, observed_indices = ind_trgt_p, compute_log_likelihood = True)
			simulated_trv_s, log_likelihood_obs_s, log_likelihood_sim_s = sample_correlated_travel_time_noise_pre_computed(chol_trv_matrix, trv_out_vals[0,:,1], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], ind_use_slice[i], observed_times = time_trgt_s, observed_indices = ind_trgt_s, compute_log_likelihood = True)
		else:
			simulated_trv_p, _, _ = sample_correlated_travel_time_noise_pre_computed(chol_trv_matrix, trv_out_vals[0,:,0], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], ind_use_slice[i])
			simulated_trv_s, _, _ = sample_correlated_travel_time_noise_pre_computed(chol_trv_matrix, trv_out_vals[0,:,1], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], ind_use_slice[i])

		Log_prob_p.append(log_likelihood_sim_p/np.maximum(1.0, len(time_trgt_p))) ## Check normalization
		Log_prob_s.append(log_likelihood_sim_s/np.maximum(1.0, len(time_trgt_s))) ## Check normalization
		Simulated_p.append(simulated_trv_p)
		Simulated_s.append(simulated_trv_s)

	return srcs_sample, mags_sample, ichoose, Simulated_p, Simulated_s, np.array(Log_prob_p)/scale_log_prob, np.array(Log_prob_s)/scale_log_prob



# def compute_data_misfit_loss_backup(srcs_sample, mags_sample, features, n_mag_bins = 5, return_diagnostics = False):

# 	## Assumes values are > 1e-12
# 	def rel_error(x, y, tol = 1e-12): ## x is pred, y is target
# 		val = (np.abs(y) > tol)*np.abs(x - y)/np.maximum(1e-12, np.abs(y))
# 		return val

# 	## Assumes features contains the: Morans, Inertia, Cnt, and Intersection variables
# 	## Each contains the (pred_p, pred_s, trgt_p, trgt_s), and Intersection (intersection_p, intersection_s)
# 	## These metrics are all computed as a "vector" over the batch
# 	Morans, Inertia, Cnt, Intersection = features ## Computing relative misfit features
# 	misfit_morans_p = rel_error(Morans[0], Morans[2])
# 	misfit_morans_s = rel_error(Morans[1], Morans[3])
# 	misfit_inertia_p = rel_error(Inertia[0], Inertia[2])
# 	misfit_inertia_s = rel_error(Inertia[1], Inertia[3])
# 	misfit_cnt_p = rel_error(Cnt[0], Cnt[2])
# 	misfit_cnt_s = rel_error(Cnt[1], Cnt[3])
# 	misfit_intersection_p = rel_error(Intersection[0], 1.0) ## Target of intersection is "1.0"
# 	misfit_intersection_s = rel_error(Intersection[1], 1.0) ## Target of intersection is "1.0"

# 	## We now bin the errors into magnitude bins
# 	mag_bins = np.linspace(np.quantile(mags_sample, 0.02), np.quantile(mags_sample, 0.98), n_mag_bins)
# 	mag_bin_width = np.diff(mag_bins[0:2])
# 	ip = cKDTree(mags_sample.reshape(-1,1)).query_ball_point(mag_bins.reshape(-1,1) + mag_bin_width/2.0, r = mag_bin_width/2.0)
# 	inot_nan_p = np.where(np.isnan(Intersection[0]) == 0)[0]
# 	inot_nan_s = np.where(np.isnan(Intersection[1]) == 0)[0]
# 	ip_p, ip_s = [], [] ## Create sets of sampling indices based on magnitude bins and whether trgts has "any" picks for that phase type
# 	for j in range(len(ip)):
# 		ip_p.append(np.array(list(set(ip[j]).intersection(inot_nan_p))))
# 		ip_s.append(np.array(list(set(ip[j]).intersection(inot_nan_s))))

# 	## Compute median relative error over batch for each magnitude bin (or could output other quantiles/statistics)
# 	q_val = np.array([0.25, 0.5, 0.75]) ## Compute weighted mean residual over the different quantile bins
# 	weight_bins = np.array([0.25, 1.0, 0.25]).reshape(-1,1) ## Upper quantile can be unstable
# 	weight_bins = weight_bins/weight_bins.sum()
# 	if len(q_val) == 1: weight_bins = np.array([1.0]).reshape(-1,1)
# 	assert(len(q_val) == len(weight_bins))
# 	# pdb.set_trace()
# 	zero_vec = np.zeros((len(q_val),1))
# 	res_morans_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_morans_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_morans_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_morans_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_inertia_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_inertia_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_inertia_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_inertia_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_cnt_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_cnt_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_cnt_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_cnt_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_intersection_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_intersection_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
# 	res_intersection_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_intersection_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin

# 	# pdb.set_trace()
# 	weights = np.array([1.0, 1.0, 1.0, 1.0])
# 	weights = weights/weights.sum()

# 	median_loss_p = weights[0]*res_morans_p_per_mag_bin + weights[1]*res_inertia_p_per_mag_bin + weights[2]*res_cnt_p_per_mag_bin + weights[3]*res_intersection_p_per_mag_bin
# 	median_loss_s = weights[0]*res_morans_s_per_mag_bin + weights[1]*res_inertia_s_per_mag_bin + weights[2]*res_cnt_s_per_mag_bin + weights[3]*res_intersection_s_per_mag_bin

# 	## RMS residual over the magnitude bins (and averaged over phase types) ## Note: doing uniform weighting over magnitudes (but sampling has already been done to balance samples from different magnoitude bins)
# 	median_loss = np.linalg.norm((0.5*median_loss_p + 0.5*median_loss_s))/np.sqrt(n_mag_bins)

# 	if return_diagnostics == False:

# 		return median_loss

# 	else:

# 		median_res_vals_p = [res_morans_p_per_mag_bin, res_inertia_p_per_mag_bin, res_cnt_p_per_mag_bin, res_intersection_p_per_mag_bin]
# 		median_res_vals_s = [res_morans_s_per_mag_bin, res_inertia_s_per_mag_bin, res_cnt_s_per_mag_bin, res_intersection_s_per_mag_bin]

# 		res_vals_p = [misfit_morans_p, misfit_inertia_p, misfit_cnt_p, misfit_intersection_p]
# 		res_vals_s = [misfit_morans_s, misfit_inertia_s, misfit_cnt_s, misfit_intersection_s]

# 		return median_loss, [res_vals_p, res_vals_s], [median_res_vals_p, median_res_vals_s]


def compute_data_misfit_loss(mags_sample, features, n_mag_bins = 5, return_diagnostics = False):

	## Assumes values are > 1e-12
	def rel_error(x, y, tol = 1e-12): ## x is pred, y is target
		val = (np.abs(y) > tol)*np.abs(x - y)/np.maximum(1e-12, np.abs(y))
		return val

	## Assumes features contains the: Morans, Inertia, Cnt, and Intersection variables
	## Each contains the (pred_p, pred_s, trgt_p, trgt_s), and Intersection (intersection_p, intersection_s)
	## These metrics are all computed as a "vector" over the batch
	Log_prob_p, Log_prob_s = features ## Computing relative misfit features

	## We now bin the errors into magnitude bins
	mag_bins = np.linspace(np.quantile(mags_sample, 0.02), np.quantile(mags_sample, 0.98), n_mag_bins)
	mag_bin_width = np.diff(mag_bins[0:2])
	ip = cKDTree(mags_sample.reshape(-1,1)).query_ball_point(mag_bins.reshape(-1,1) + mag_bin_width/2.0, r = mag_bin_width/2.0)
	# inot_nan_p = np.where(np.isnan(Intersection[0]) == 0)[0]
	# inot_nan_s = np.where(np.isnan(Intersection[1]) == 0)[0]
	inot_nan_p = np.arange(len(mags_sample)) ## Overwritting not nan
	inot_nan_s = np.arange(len(mags_sample))
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
	log_prob_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(-1.0*Log_prob_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	log_prob_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(-1.0*Log_prob_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	# res_inertia_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_inertia_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	# res_inertia_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_inertia_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	# res_cnt_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_cnt_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	# res_cnt_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_cnt_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	# res_intersection_p_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_intersection_p[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_p])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin
	# res_intersection_s_per_mag_bin = (weight_bins*np.hstack([np.quantile(misfit_intersection_s[j], q_val).reshape(-1,1) if len(j) > 0 else zero_vec for j in ip_s])).mean(0) # .squeeze() ## Note "j" is a vector of indices for all events within that magnitude bin

	# pdb.set_trace()
	weights = np.array([1.0, 1.0, 1.0, 1.0])
	weights = weights/weights.sum()

	median_loss_p = log_prob_p_per_mag_bin # weights[0]*res_morans_p_per_mag_bin + weights[1]*res_inertia_p_per_mag_bin + weights[2]*res_cnt_p_per_mag_bin + weights[3]*res_intersection_p_per_mag_bin
	median_loss_s = log_prob_s_per_mag_bin # weights[0]*res_morans_s_per_mag_bin + weights[1]*res_inertia_s_per_mag_bin + weights[2]*res_cnt_s_per_mag_bin + weights[3]*res_intersection_s_per_mag_bin

	## RMS residual over the magnitude bins (and averaged over phase types) ## Note: doing uniform weighting over magnitudes (but sampling has already been done to balance samples from different magnoitude bins)
	median_loss = (0.5*log_prob_p_per_mag_bin + 0.5*log_prob_s_per_mag_bin).mean() # np.linalg.norm((0.5*median_loss_p + 0.5*median_loss_s))/np.sqrt(n_mag_bins)

	if return_diagnostics == False:

		return median_loss

	else:

		# median_res_vals_p = [res_morans_p_per_mag_bin, res_inertia_p_per_mag_bin, res_cnt_p_per_mag_bin, res_intersection_p_per_mag_bin]
		# median_res_vals_s = [res_morans_s_per_mag_bin, res_inertia_s_per_mag_bin, res_cnt_s_per_mag_bin, res_intersection_s_per_mag_bin]

		# res_vals_p = [misfit_morans_p, misfit_inertia_p, misfit_cnt_p, misfit_intersection_p]
		# res_vals_s = [misfit_morans_s, misfit_inertia_s, misfit_cnt_s, misfit_intersection_s]

		return median_loss, [median_loss_p, median_loss_s]


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
chol_params['relative_travel_time_factor1'] = 0.01 # random_scale_factor_phase = 0.35
chol_params['relative_travel_time_factor2'] = 0.04 # random_scale_factor_phase = 0.35
chol_params['travel_time_bias_scale_factor1'] = 0.03
chol_params['travel_time_bias_scale_factor2'] = 0.03
chol_params['correlation_scale_distance'] = 30e3


########################### End initilization ##########################




########################### Run optimization ###########################

n_batch = 300

def evaluate_bayesian_objective_evaluate(x, n_batch = n_batch, n_mag_bins = 5, return_config = False):

	## Set Cholesky parameters
	chol_params = {}
	chol_params['relative_travel_time_factor1'] = x[0] # random_scale_factor_phase = 0.35
	chol_params['relative_travel_time_factor2'] = x[1] # random_scale_factor_phase = 0.35
	chol_params['travel_time_bias_scale_factor1'] = x[2]
	chol_params['travel_time_bias_scale_factor2'] = x[3]
	chol_params['correlation_scale_distance'] = x[4]

	## Sample a generation
	# st_time = time.time()

	check_val = 0
	inc_val = 0
	while (check_val == 0)*(inc_val < 10):
		try:
			srcs_sample, mags_sample, ichoose, Simulated_p, Simulated_s, Log_prob_p, Log_prob_s = simulate_travel_times(prob_vec, chol_params, ftrns1, n_samples = n_batch)
			# print('\nData generation time %0.4f for %d samples (with features)'%(time.time() - st_time, n_batch))
			check_val = 1
		except:
			inc_val += 1
	if check_val == 0:
		print('Error')
		error('Error')

	## Instead can bin using the misfit loss function
	median_loss = compute_data_misfit_loss(mags_sample, [Log_prob_p, Log_prob_s])
	# median_loss = -(0.5*np.median(Log_prob_p) + 0.5*np.median(Log_prob_s))/100.0

	## Compute residuals
	# st_time = time.time()
	# median_loss = compute_data_misfit_loss(srcs_sample, mags_sample, features, n_mag_bins = n_mag_bins, return_diagnostics = False)
	# print('\nResidual computation time %0.4f for %d samples (median loss: %0.4f)'%(time.time() - st_time, n_batch, median_loss))
	print('Loss %0.4f'%median_loss)

	if return_config == True:

		return median_loss, chol_params

	else:

		return median_loss

# from sklearn.optimize import gp_minimize

bounds = [(0.005, 0.075), # travel time noise level
          (0.005, 0.075), # travel time noise level
          (0.005, 0.08), # bias level [1]
          (0.005, 0.08), # bias level [2]
          (1e3, 150e3)] # correlation scale distance ## Check solution when using no correlation

# strings = ['p_exponent', 'scale_factor']


## Initial residual (using mean of the bounds)
n_average = 10
x = [np.mean(b) for b in bounds]
initial_loss = np.median([evaluate_bayesian_objective_evaluate(x) for i in range(n_average)])
print('Initial loss: %0.4f'%(initial_loss))

## Simulate travel times for several source coordinates
x1 = np.linspace(lat_range[0], lat_range[-1], 5)
x2 = np.linspace(lon_range[0], lon_range[-1], 5)
x3 = np.zeros(1)
x11, x12, x13 = np.meshgrid(x1, x2, x3)
xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)


x_list = []
n_repeat = 2 ## Set to 1 for no repeat
zoom_factor = (1/3.0) ## Repeat the optimization and "zoom" the bounds in proportionally closer to the optimal point
loss_vals = [initial_loss]
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
	                  n_calls=100,         # the number of evaluations of f
	                  n_random_starts=100,  # the number of random initialization points
	                  noise='gaussian',       # the noise level (optional)
	                  random_state=None, # the random seed
	                  initial_point_generator = 'lhs',
	                  model_queue_size = 150)

	x = optimize.x # np.mean(b) for b in bouns]
	loss_val = np.median([evaluate_bayesian_objective_evaluate(x) for i in range(n_average)])
	print('\nValues:')
	print(list(x))
	print('Intermediate loss (%d): %0.4f \n'%(n, loss_val))
	loss_vals.append(loss_val)
	x_list.append(x)


## Simulate some travel times
chol_params1 = {}
chol_params1['relative_travel_time_factor1'] = x[0]
chol_params1['relative_travel_time_factor2'] = x[1]
chol_params1['travel_time_bias_scale_factor1'] = x[2]
chol_params1['travel_time_bias_scale_factor2'] = x[3]
chol_params1['correlation_scale_distance'] = x[4]


n_batch = 30
n_repeat = 1
srcs_sample, mags_sample, ichoose, Simulated_p, Simulated_s, Log_prob_p, Log_prob_s = simulate_travel_times(prob_vec, chol_params1, ftrns1, n_samples = n_batch)
ind_use_slice = [Inds[ichoose[i]] for i in range(len(ichoose))]
observed_time_p = [Picks_P_lists[ichoose[i]][:,0] - srcs_sample[i,3] for i in range(len(ichoose))]
observed_time_s = [Picks_S_lists[ichoose[i]][:,0] - srcs_sample[i,3] for i in range(len(ichoose))]
observed_ind_p = [Picks_P_lists[ichoose[i]][:,1].astype('int') for i in range(len(ichoose))]
observed_ind_s = [Picks_S_lists[ichoose[i]][:,1].astype('int') for i in range(len(ichoose))]

Sim_p = [np.zeros((n_repeat, len(ind_use_slice[i]))) for i in range(n_batch)]
Sim_s = [np.zeros((n_repeat, len(ind_use_slice[i]))) for i in range(n_batch)]
for i in range(n_repeat):
	srcs_sample_, mags_sample_, ichoose, Simulated_p, Simulated_s, Log_prob_p, Log_prob_s = simulate_travel_times(prob_vec, chol_params1, ftrns1, ichoose = ichoose, n_samples = n_batch)
	assert(np.abs(srcs_sample_ - srcs_sample).max() == 0)
	assert(np.abs(mags_sample_ - mags_sample).max() == 0)
	for j in range(n_batch): Sim_p[j][i,:] = Simulated_p[j]
	for j in range(n_batch): Sim_s[j][i,:] = Simulated_s[j]
# Sim_p = np.vstack(Sim_p)
# Sim_s = np.vstack(Sim_s)


# np.savez_compressed(path_to_file + 'Grids' + seperator + 'optimized_travel_time_parameters_ver_1.npz', x = x, x_list = np.vstack(x_list), loss_vals = np.array(loss_vals))

median_loss = np.mean(loss_vals)

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
np.savez_compressed(path_to_file + 'Grids/%s_optimized_trv_parameters_%s_loss_%0.4f.npz' % (name_of_project, date, median_loss), x = np.array(optimize.x), median_loss=median_loss)

z = h5py.File(path_to_file + 'Grids' + seperator + f'optimized_travel_time_simulations_ver_1_loss_{median_loss}.hdf5', 'w')

z['srcs_samples'] = srcs_sample
z['mags_sample'] = mags_sample
for i in range(n_batch):
	z['Sim_p_%d'%i] = Sim_p[i]
	z['Sim_s_%d'%i] = Sim_s[i]
	z['observed_time_p_%d'%i] = observed_time_p[i]
	z['observed_time_s_%d'%i] = observed_time_s[i]
	z['observed_ind_p_%d'%i] = observed_ind_p[i]
	z['observed_ind_s_%d'%i] = observed_ind_s[i]
	z['ind_use_%d'%i] = ind_use_slice[i]
# srcs_sample = srcs_sample, mags_sample = mags_sample, Sim_p = Sim_p, Sim_s = Sim_s, ind_use_slice = ind_use_slice
z.close()


# for i in range(30):
# 	plt.figure()
# 	iarg = np.argsort(z['observed_time_p_%d'%i][:])
# 	plt.scatter(z['observed_time_p_%d'%i][:][iarg] - srcs[i,3], np.arange(len(iarg)), c = 'C1')
# 	plt.plot(z['Sim_p_%d'%i][:][:,np.array(z['observed_ind_p_%d'%i][:][iarg])].T, np.arange(len(iarg)), alpha = 0.5, c = 'C0')
# 	fig = plt.gcf()
# 	fig.set_size_inches(12, 8)

## Plot optimized data

