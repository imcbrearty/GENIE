
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.neighbors import KernelDensity
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from torch_geometric.data import Data, Batch
from torch_geometric.data import HeteroData  # if you use heterogeneous graphs
from torch.nn import Softplus
from torch_scatter import scatter
from numpy.matlib import repmat
from scipy.stats import gamma
from functools import partial
from scipy.stats import chi2
from itertools import cycle
from torch import Tensor
import pdb
import pathlib
import glob
import sys


# # Optional: for PyG
# try:
#     from torch_geometric.data import Data, Batch
#     from torch_geometric.data import HeteroData  # if you use heterogeneous graphs
# except ImportError:
#     Data = Batch = HeteroData = None   # graceful fallback if PyG not installed

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
use_sign_input = config.get('use_sign_input', False)
use_topography = config['use_topography']
use_station_corrections = config.get('use_station_corrections', False)
if use_subgraph == True:
    max_deg_offset = config['max_deg_offset']
    k_nearest_pairs = config['k_nearest_pairs']	

graph_params = [k_sta_edges, k_spc_edges, k_time_edges]

## Load time shift variables
use_time_shift = config['use_time_shift']
use_expanded = config['use_expanded']
use_sigmoid = config['use_sigmoid']


# time_shift_range = config['time_shift_range'] # 30.0
# time_shift_scale = config['time_shift_scale'] # 8.0
# time_shift_scale = 0 if use_time_shift == False else time_shift_scale


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
src_kernel_mean = np.mean([src_x_kernel, src_x_kernel, src_depth_kernel])
src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

scale_time = train_config['scale_time']

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
else:
	torch.cuda.set_device(0)
	torch.ones(1).cuda()  # warms up GPU

## Setup training folder parameters
if (load_training_data == True) or (build_training_data == True):
	path_to_data = train_config['path_to_data'] ## Path to training data files
	n_ver_training_data = train_config['n_ver_training_data'] ## Version of training files
	if (path_to_data[-1] != '/')*(path_to_data[-1] != '\\'):
		path_to_data = path_to_data + seperator

if use_topography == True:
	surface_profile = np.load(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project)['surface_profile'] # (os.path.isfile(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project) == True)
	tree_surface = cKDTree(surface_profile[:,0:2])


use_consistency_loss = True
use_gradient_loss = train_config['use_gradient_loss']
init_gradient_loss = False
use_negative_loss = True ## If True, up-sample the false positive predictions 
use_negative_loss_step = 1


use_teleseisim_noise = True
if use_teleseisim_noise == True:
	z = np.load(path_to_file + 'Grids' + seperator + 'teleseismic_travel_time_grid_ver_1.npz')
	xx_teleseism, trv_teleseism, phase_types = z['xx_teleseism'], z['trv_teleseism'], z['phase_types']
	z.close()
	ipos = np.where(xx_teleseism[:,0] > 0)[0]
	# inot_nan1, inot_nan2 = np.where(np.isnan(trv_teleseism) == 0)
	xx_teleseism = xx_teleseism[ipos]
	trv_teleseism = trv_teleseism[ipos]

	# unique_depths = np.unique(xx_teleseism[:,1])
	# ip_depths = [np.array(v) for v in cKDTree(xx_teleseism[:,1].reshape(-1,1)).query_ball_point(unique_depths.reshape(-1,1), r = 0)]
	# iarg = [np.argsort(xx_teleseism[ip_depths[j],0]) for j in range(len(ip_depths))]
	# ip_depths = [np.ascontiguousarray(ip_depths[j][iarg[j]]) for j in range(len(ip_depths))]

	# deg_vals = [np.ascontiguousarray(xx_teleseism[ip_depths[j],0]) for j in range(len(ip_depths))]
	# trv_vals = [np.ascontiguousarray(trv_teleseism[ip_depths[j],:]) for j in range(len(ip_depths))]
	# inot_nan = [[np.where(np.isnan(trv_vals[j][:,k]) == 0)[0] for j in range(len(ip_depths))] for k in range(len(phase_types))]
	# f_teleseisims = [[lambda x: np.interp(x, deg_vals[j], trv_vals[j][inot_nan,k], left = np.nan, right = np.nan) for j in range(len(ip_depths))] for k in range(len(phase_types))]



if use_negative_loss == True:
	assert(use_gradient_loss == False) ## Right now might not be compatible with gradient loss due to the query layer not re-computing gradients



## Load specific subsets of stations to train on in addition to random
## subnetworks from the total set of possible stations
load_subnetworks = train_config['fixed_subnetworks']
if (load_subnetworks == True)*(load_training_data == False): ## Only load subnetworks if not loading the data
	
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

if use_expanded == True:
	Ac = np.load(path_to_file + 'Grids/%s_seismic_network_expanders_ver_%d.npz'%(name_of_project, template_ver))['Ac']
else:
	Ac = False


if use_time_shift == True:
	z = np.load(path_to_file + 'Grids' + seperator + 'grid_time_shift_ver_1.npz')
	time_shifts = z['time_shifts'] ## Shape (n_grids, n_nodes, n_times)
	z.close()
else:
	time_shifts = None # np.zeros((x_grids.shape[0], x_grids.shape[1]))


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

## Training params list 2
spc_random = train_config['spc_random']
sig_t = train_config['sig_t'] # 3 percent of travel time error on pick times
spc_thresh_rand = train_config['spc_thresh_rand']
min_sta_arrival = train_config['min_sta_arrival']
min_pick_arrival = train_config.get('min_pick_arrival', min_sta_arrival)
coda_rate = train_config['coda_rate'] # 5 percent arrival have code. Probably more than this? Increased from 0.035.
coda_win = np.array(train_config['coda_win']) # coda occurs within 0 to 25 s after arrival (should be less?) # Increased to 25, from 20.0
max_num_spikes = train_config['max_num_spikes']
spike_time_spread = train_config['spike_time_spread']
s_extra = train_config['s_extra'] ## If this is non-zero, it can increase (or decrease) the total rate of missed s waves compared to p waves
use_stable_association_labels = train_config['use_stable_association_labels']
thresh_noise_max = train_config['thresh_noise_max'] # ratio of sig_t*travel time considered excess noise
min_misfit_allowed = train_config['min_misfit_allowed'] ## The minimum error on theoretical vs. observed travel times that beneath which, picks have positive associaton labels (the upper limit is set by a percentage of the travel time)
total_bias = train_config['total_bias'] ## The total (uniform across stations) bias on travel times for each synthetic earthquake (helps add robustness to uncertainity on assumed and true velocity models)
training_params_2 = [spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max, min_misfit_allowed, total_bias]

## Training params list 3
# n_batch = train_config['n_batch']
dist_range = train_config['dist_range'] # Should be chosen proportional to physical domain size
max_rate_events = train_config['max_rate_events']
max_miss_events = train_config['max_miss_events']
max_false_events = train_config['max_rate_events']*train_config['max_false_events'] # Make max_false_events an absolute value, but it's based on the ratio of the value in the config file times the event rate
miss_pick_fraction = train_config['miss_pick_fraction']
T = train_config['T']
dt = train_config['dt']
tscale = train_config['tscale']
n_sta_range = train_config['n_sta_range'] # n_sta_range[0]*locs.shape[0] must be >= the number of station edges chosen (k_sta_edges)
use_sources = train_config['use_sources']
use_full_network = train_config['use_full_network']
if Ind_subnetworks is not False:
	fixed_subnetworks = Ind_subnetworks
else:
	fixed_subnetworks = False # train_config['fixed_subnetworks']
use_preferential_sampling = train_config['use_preferential_sampling']
use_shallow_sources = train_config['use_shallow_sources']
use_extra_nearby_moveouts = train_config['use_extra_nearby_moveouts']
training_params_3 = [n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, miss_pick_fraction, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources, use_extra_nearby_moveouts]

def simulate_travel_times(prob_vec, chol_params, ftrns1, n_samples = 100, use_l1 = False, srcs = None, mags = None, ichoose = None, locs_use_list = None, ind_use_slice = None, return_features = True): # n_repeat : can repeatedly draw either from the covariance matrices, or the binomial distribution

	if srcs is None:
		## Sample sources
		if ichoose is None: ichoose = np.random.choice(len(Srcs), p = prob_vec, size = n_samples)
		locs_use_list = [locs[Inds[j]] for j in ichoose]
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
		srcs_sample = Srcs[ichoose]
		# mags_sample = Mags[ichoose]
		srcs_samples_cart = ftrns1(srcs_sample)
		ind_use_slice = [Inds[ichoose[i]] for i in range(len(ichoose))]
		sample_fixed = False

	else:
		ichoose = np.arange(len(srcs))
		n_samples = len(srcs)
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
		srcs_sample = np.copy(srcs)
		# mags_sample = np.copy(mags_sample)
		srcs_samples_cart = ftrns1(srcs_sample)
		# ind_use_slice = [np.arange(len(locs)) for i in range(len(ichoose))]
		sample_fixed = True

	## Removing the use of mags from the function

	rel_trv_factor1 = chol_params['relative_travel_time_factor1'] # random_scale_factor_phase = 0.35
	rel_trv_factor2 = chol_params['relative_travel_time_factor2'] # random_scale_factor_phase = 0.35
	travel_time_bias_scale_factor1 = chol_params['travel_time_bias_scale_factor1']
	travel_time_bias_scale_factor2 = chol_params['travel_time_bias_scale_factor2']
	correlation_scale_distance = chol_params['correlation_scale_distance']
	softplus_beta = chol_params['softplus_beta']
	softplus_shift = chol_params['softplus_shift']

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
	Mean_trv_p = []
	Mean_trv_s = []
	Std_val_p = []
	Std_val_s = []
	scale_log_prob = 100.0

	locs_cuda = torch.Tensor(locs).to(device)
	srcs_cuda = torch.Tensor(srcs_sample).to(device)
	for i in range(n_samples):
		## Sample correlated travel time noise
		trv_out_vals = trv(locs_cuda, srcs_cuda[i].reshape(1,-1)).cpu().detach().numpy()
		if sample_fixed == False:
			time_trgt_p, time_trgt_s = Picks_P_lists[ichoose[i]][:,0].astype('int') - srcs_sample[i,3], Picks_S_lists[ichoose[i]][:,0].astype('int') - srcs_sample[i,3]
			ind_trgt_p, ind_trgt_s = Picks_P_lists[ichoose[i]][:,1].astype('int'), Picks_S_lists[ichoose[i]][:,1].astype('int')
			simulated_trv_p, scaled_mean_vec_p, std_val_p, log_likelihood_obs_p, log_likelihood_sim_p = sample_correlated_travel_time_noise(chol_trv_matrix, trv_out_vals[0,:,0], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], softplus_beta, softplus_shift, ind_use_slice[i], observed_times = time_trgt_p, observed_indices = ind_trgt_p, compute_log_likelihood = True)
			simulated_trv_s, scaled_mean_vec_s, std_val_s, log_likelihood_obs_s, log_likelihood_sim_s = sample_correlated_travel_time_noise(chol_trv_matrix, trv_out_vals[0,:,1], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], softplus_beta, softplus_shift, ind_use_slice[i], observed_times = time_trgt_s, observed_indices = ind_trgt_s, compute_log_likelihood = True)
			Log_prob_p.append(log_likelihood_sim_p/np.maximum(1.0, len(time_trgt_p))) ## Check normalization
			Log_prob_s.append(log_likelihood_sim_s/np.maximum(1.0, len(time_trgt_s))) ## Check normalization

		else:
			simulated_trv_p, scaled_mean_vec_p, std_val_p = sample_correlated_travel_time_noise(chol_trv_matrix, trv_out_vals[0,:,0], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], softplus_beta, softplus_shift, ind_use_slice[i])
			simulated_trv_s, scaled_mean_vec_s, std_val_s = sample_correlated_travel_time_noise(chol_trv_matrix, trv_out_vals[0,:,1], [travel_time_bias_scale_factor1, travel_time_bias_scale_factor2], [rel_trv_factor1, rel_trv_factor2], softplus_beta, softplus_shift, ind_use_slice[i])


		Simulated_p.append(simulated_trv_p)
		Simulated_s.append(simulated_trv_s)
		Mean_trv_p.append(scaled_mean_vec_p)
		Mean_trv_s.append(scaled_mean_vec_s)
		Std_val_p.append(std_val_p)
		Std_val_s.append(std_val_s)


	return srcs_sample, [], ichoose, Simulated_p, Simulated_s, Mean_trv_p, Mean_trv_s, np.vstack(Std_val_p), np.vstack(Std_val_s), np.array(Log_prob_p)/scale_log_prob, np.array(Log_prob_s)/scale_log_prob
	# _, _, _, Simulated_p, Simulated_s, Mean_trv_p, Mean_trv_s, _, _

def sample_correlated_travel_time_noise(cholesky_matrix_trv, mean_vec, bias_factors, std_factor, softplus_beta, softplus_shift, ind_use, compute_log_likelihood = False, observed_indices = None, observed_times = None, min_tol = 0.005, n_repeat = 1):
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

	softplus = nn.Softplus(beta = np.pow(10.0, softplus_beta))
	scale_val = softplus(torch.Tensor(bias_val*mean_vec*std_val + softplus_shift)).cpu().detach().numpy()
	standard_deviation = np.diag(scale_val)
	
	# standard_deviation = np.diag(mean_vec*std_val)

	# std_val = np.random.uniform(min_tol, std_factor)
	# standard_deviation = np.diag(mean_vec*std_factor)

	z = np.random.randn(len(cholesky_matrix_trv), n_repeat)

	# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
	scaled_chol_matrix = (standard_deviation @ cholesky_matrix_trv)
	scaled_mean_vec = mean_vec*bias_val

	# Compute simulated times
	simulated_times = ((scaled_chol_matrix @ z) + (scaled_mean_vec).reshape(-1,1))[ind_use].squeeze() # [ind_use]

	if compute_log_likelihood == False:

		return simulated_times, scaled_mean_vec, std_val

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

		return simulated_times, scaled_mean_vec, scale_val, log_likelihood_obs, log_likelihood_sim

def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, plot_on = False, verbose = False, skip_graphs = False, use_sign_input = use_sign_input, use_time_shift = use_time_shift, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded, Ac = Ac, return_only_data = False):

	if verbose == True:
		st = time.time()

	k_sta_edges, k_spc_edges, k_time_edges = graph_params
	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	n_spc_query, n_src_query = training_params
	spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max, min_misfit_allowed, total_bias = training_params_2
	n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, miss_pick_fraction, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources, use_extra_nearby_moveouts = training_params_3

	assert(np.floor(n_sta_range[0]*locs.shape[0]) > k_sta_edges)

	## Note: this uses a different definition of scale_x and offset_x than the rest of the script (the should really be called scale_x_extend and offset_x_extend to be consistent)
	## Should update these names and use the correct name throught the rest of this function
	scale_x = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)
	n_sta = locs.shape[0]

	t_slice = np.arange(-t_win/2.0, t_win/2.0 + dt_win, dt_win)

	tsteps = np.arange(0, T + dt, dt)
	tvec = np.arange(-tscale*4, tscale*4 + dt, dt)
	tvec_kernel = np.exp(-(tvec**2)/(2.0*(tscale**2)))

	p_rate_events = fftconvolve(np.random.randn(2*locs.shape[0] + 3, len(tsteps)), tvec_kernel.reshape(1,-1).repeat(2*locs.shape[0] + 3,0), 'same', axes = 1)
	global_event_rate, global_miss_rate, global_false_rate = p_rate_events[0:3,:]

	# Process global event rate, to physical units.
	global_event_rate = (global_event_rate - global_event_rate.min())/(global_event_rate.max() - global_event_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_rate_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_rate_events - min_add) + 0.5*max_rate_events
	global_event_rate = global_event_rate*scale + min_add

	global_miss_rate = (global_miss_rate - global_miss_rate.min())/(global_miss_rate.max() - global_miss_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
	global_miss_rate = global_miss_rate*scale + min_add

	global_false_rate = (global_false_rate - global_false_rate.min())/(global_false_rate.max() - global_false_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_false_events - min_add) + 0.5*max_false_events
	global_false_rate = global_false_rate*scale + min_add

	station_miss_rate = p_rate_events[3 + np.arange(n_sta),:]
	station_miss_rate = (station_miss_rate - station_miss_rate.min(1, keepdims = True))/(station_miss_rate.max(1, keepdims = True) - station_miss_rate.min(1, keepdims = True)) # [0,1] scale
	min_add = np.random.rand(n_sta,1)*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand(n_sta,1)*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
	station_miss_rate = station_miss_rate*scale + min_add

	station_false_rate = p_rate_events[3 + n_sta + np.arange(n_sta),:]
	station_false_rate = (station_false_rate - station_false_rate.min(1, keepdims = True))/(station_false_rate.max(1, keepdims = True) - station_false_rate.min(1, keepdims = True))
	min_add = np.random.rand(n_sta,1)*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand(n_sta,1)*(0.5*max_false_events - min_add) + 0.5*max_false_events
	station_false_rate = station_false_rate*scale + min_add

	## Sample events.
	vals = np.random.poisson(dt*global_event_rate/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average
	src_times = np.sort(np.hstack([np.random.rand(vals[j])*dt + tsteps[j] for j in range(len(vals))]))

	if len(src_times) == 0:
		src_times = np.array([np.random.rand()*T])
	
	n_src = len(src_times)
	src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
	src_magnitude = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)

	if use_reference_spatial_density == True:
		n_rand_sample = int(len(src_positions)*n_frac_reference_catalog)
		if n_rand_sample > 0:
			rand_sample = ftrns2(ftrns1(srcs_ref[np.random.choice(len(srcs_ref), size = n_rand_sample),0:3]) + spatial_sigma*np.random.randn(n_rand_sample,3))
			ioutside = np.where(((rand_sample[:,2] < depth_range[0]) + (rand_sample[:,2] > depth_range[1])) > 0)[0]
			rand_sample[ioutside,2] = np.random.rand(len(ioutside))*(depth_range[1] - depth_range[0]) + depth_range[0]		
			src_positions[np.random.choice(len(src_positions), size = n_rand_sample, replace = False)] = rand_sample
	
	if use_shallow_sources == True:
		sample_random_depths = gamma(1.75, 0.0).rvs(n_src)
		sample_random_grab = np.where(sample_random_depths > 5)[0] # Clip the long tails, and place in uniform, [0,5].
		sample_random_depths[sample_random_grab] = 5.0*np.random.rand(len(sample_random_grab))
		sample_random_depths = sample_random_depths/sample_random_depths.max() # Scale to range
		sample_random_depths = -sample_random_depths*(scale_x[0,2] - 2e3) + (offset_x[0,2] + scale_x[0,2] - 2e3) # Project along axis, going negative direction. Removing 2e3 on edges.
		src_positions[:,2] = sample_random_depths

	use_aftershocks = True
	if (use_aftershocks == True)*(len(src_positions) > 1):
			n_iterations = 1
			for i in range(n_iterations):
				aftershock_rate, aftershock_scale_x, aftershock_scale_t = 0.1, float(src_x_kernel/0.35), float(src_t_kernel/0.35)
				if (int(np.ceil(aftershock_rate*len(src_positions))) > 0):
					ichoose = np.random.choice(np.arange(1, len(src_positions)), size = int(np.ceil(aftershock_rate*len(src_positions))), replace = False)
					rand_vec = np.random.randn(len(ichoose),3)
					rand_vec = rand_vec/np.linalg.norm(rand_vec, axis = 1, keepdims = True)
					samp_spc = gamma.rvs(0.5, 1.0, size = len(rand_vec))*aftershock_scale_x
					rand_vec = rand_vec*samp_spc.reshape(-1,1)
					src_positions[ichoose] = ftrns2(ftrns1(src_positions[ichoose - 1]) + rand_vec)
					src_positions = np.clip(src_positions, np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1), np.array([lat_range_extend[1], lon_range_extend[1], depth_range[1]]).reshape(1,-1))
					src_times[ichoose] = src_times[ichoose - 1] + aftershock_scale_t*gamma.rvs(0.5, 1.0, size = len(rand_vec))
		
	if use_topography == True: ## Don't simulate any sources in the air
		imatch = tree_surface.query(src_positions[:,0:2])[1]
		ifind_match = np.where(src_positions[:,2] > surface_profile[imatch,2])[0]
		src_positions[ifind_match,2] = np.random.rand(len(ifind_match))*(surface_profile[imatch[ifind_match],2] - depth_range[0]) + depth_range[0]
		
	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

	use_uniform_distance_threshold = False
	## This previously sampled a uniform distribution by default, now it samples a skewed
	## distribution of the maximum source-reciever distances allowed for each event.
	if use_uniform_distance_threshold == True:
		dist_thresh = np.random.rand(n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
	else:
		## Use beta distribution to generate more samples with smaller moveouts
	
		if use_extra_nearby_moveouts == True:
		
			## For half of samples, use only half of range supplied
			## (this is to increase training for sources that span only small range of network)

			n1 = int(n_src*0.3)
			n2 = int(n_src*0.3)
			n3 = n_src - n1 - n2

			dist_thresh1 = beta(2,5).rvs(size = n1).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
			ireplace = np.random.choice(len(dist_thresh1), size = int(0.15*len(dist_thresh1)), replace = False)
			dist_thresh1[ireplace] = beta(1,5).rvs(size = len(ireplace)).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]

			dist_thresh2 = beta(2,5).rvs(size = n2).reshape(-1,1)*(dist_range[1] - dist_range[0])/2.0 + dist_range[0]
			ireplace = np.random.choice(len(dist_thresh2), size = int(0.15*len(dist_thresh2)), replace = False)
			dist_thresh2[ireplace] = beta(1,5).rvs(size = len(ireplace)).reshape(-1,1)*(dist_range[1] - dist_range[0])/2.0 + dist_range[0]

			dist_thresh3 = beta(2,5).rvs(size = n3).reshape(-1,1)*(dist_range[1] - dist_range[0])/3.0 + dist_range[0]
			ireplace = np.random.choice(len(dist_thresh3), size = int(0.15*len(dist_thresh3)), replace = False)
			dist_thresh3[ireplace] = beta(1,5).rvs(size = len(ireplace)).reshape(-1,1)*(dist_range[1] - dist_range[0])/3.0 + dist_range[0]

			dist_thresh = np.concatenate((dist_thresh1, dist_thresh2, dist_thresh3), axis = 0)
		
		else:
	
			dist_thresh = beta(2,5).rvs(size = n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
			ireplace = np.random.choice(len(dist_thresh), size = int(0.15*len(dist_thresh)), replace = False)
			dist_thresh[ireplace] = beta(1,5).rvs(size = len(ireplace)).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]


	
	use_large_distances = True
	if use_large_distances == True:
		ireplace = np.random.choice(len(dist_thresh), size = int(0.2*len(dist_thresh)), replace = False)
		dist_thresh[ireplace] = 5.0*beta(1,5).rvs(size = len(ireplace)).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]	

	
	# create different distance dependent thresholds.
	dist_thresh_p = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
	dist_thresh_s = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None]

	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

	# arrivals_theoretical = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()


	use_correlated_travel_time_noise = False
	if use_correlated_travel_time_noise == True:
		# trv_time_noise_params = np.array([0.0417, 0.0309, 0.0319, 0.0585, 126677.6764])
		trv_time_noise_params = np.array([0.019731435811040067, 0.04961629822710047, 0.006929868148854273, 0.03715930048600429, 224205.70749207088, 0.5310707796290268, -24.559947281657784])
		chol_params_trv = {}
		chol_params_trv['relative_travel_time_factor1'] = trv_time_noise_params[0] 
		chol_params_trv['relative_travel_time_factor2'] = trv_time_noise_params[1]
		chol_params_trv['travel_time_bias_scale_factor1'] = trv_time_noise_params[2]
		chol_params_trv['travel_time_bias_scale_factor2'] = trv_time_noise_params[3]
		chol_params_trv['correlation_scale_distance'] = trv_time_noise_params[4]
		chol_params_trv['softplus_beta'] = trv_time_noise_params[5]
		chol_params_trv['softplus_shift'] = trv_time_noise_params[6]
		ind_use_slice = [np.arange(len(locs)) for j in range(len(src_positions))] ## Note the dependency on which ind_use_slice and locs_use_list depend on eachother
		locs_use_list = [locs[ind_use_slice[j]] for j in range(len(src_positions))]
		## Need to add correlation between P and S waves
		_, _, _, Simulated_p, Simulated_s, Mean_trv_p, Mean_trv_s, Std_val_p, Std_val_s, _, _ = simulate_travel_times([], chol_params_trv, ftrns1, srcs = src_positions, locs_use_list = locs_use_list, ind_use_slice = ind_use_slice, return_features = False)
		## Can use difference between Simulated_p, Simulatred_s, and Mean_trv_P, Mean_trv_s, to define the "remove outliers" re-labelling approach
		## Can assume there's always at least one source, and all moveout vectors are the same size
		Simulated_p = np.vstack(Simulated_p)
		Simulated_s = np.vstack(Simulated_s)
		Mean_trv_p = np.vstack(Mean_trv_p)
		Mean_trv_s = np.vstack(Mean_trv_s)
		Res_p = Simulated_p - Mean_trv_p ## Res with respect to the biased travel time vector
		Res_s = Simulated_s - Mean_trv_s
		iexcess_noise_p1, iexcess_noise_p2 = np.where(np.abs(Res_p) > np.maximum(min_misfit_allowed, thresh_noise_max*Std_val_p)) # Std_val_p.reshape(-1,1)*Simulated_p
		iexcess_noise_s1, iexcess_noise_s2 = np.where(np.abs(Res_s) > np.maximum(min_misfit_allowed, thresh_noise_max*Std_val_s)) # Std_val_s.reshape(-1,1)*Simulated_s
		arrivals_theoretical = np.concatenate((np.expand_dims(Simulated_p, axis = 2), np.expand_dims(Simulated_s, axis = 2)), axis = 2)
		mask_excess_noise = np.zeros(arrivals_theoretical.shape)
		mask_excess_noise[iexcess_noise_p1, iexcess_noise_p2, 0] = 1
		mask_excess_noise[iexcess_noise_s1, iexcess_noise_s2, 1] = 1
		# pdb.set_trace()

	else:

		arrivals_theoretical = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()
		mask_excess_noise = np.expand_dims(src_times.reshape(-1,1).repeat(len(locs), axis = 1), axis = 2).repeat(2, axis = 2)
	
	if use_correlated_travel_time_noise == False:
		add_bias_scaled_travel_time_noise = True ## This way, some "true moveouts" will have travel time 
		## errors that are from a velocity model different than used for sampling, training, and application, etc.
		## Uses a different bias for both p and s waves, but constant for all stations, for each event
		if add_bias_scaled_travel_time_noise == True:
			# total_bias = 0.03 # up to 3% scaled (uniform across station) travel time error (now specified in train_config.yaml)
			# scale_bias = np.random.rand(len(src_positions),1,2)*total_bias - total_bias/2.0
			# avg_p_vel = (sr_distances/arrivals_theoretical[:,:,0]).mean()
			# avg_s_vel = (sr_distances/arrivals_theoretical[:,:,1]).mean()
			# mean_ps_ratio = avg_p_vel/avg_s_vel
			## Note, it would be better to implement the biases in terms of velocity, rather than time, to more accurately reflect the perturbation
			frac_bias_s_ratio = 0.3
			scale_bias_p = np.random.rand(len(src_positions),1,1)*total_bias - total_bias/2.0
			scale_bias_s_ratio = (np.random.rand(len(src_positions),1,1)*total_bias - total_bias/2.0)*frac_bias_s_ratio
			scale_bias = np.concatenate((scale_bias_p, scale_bias_p + scale_bias_s_ratio), axis = 2)
			# scale_bias_ps_ratio = np.random.rand(len(src_positions),1,1)*total_bias - total_bias/2.0
			# scale_bias_s = 
			scale_bias = scale_bias + 1.0
			arrivals_theoretical = arrivals_theoretical*scale_bias
	
	arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
	arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
	src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

	## Save the excess noise mask in the fourth column instead of the origin time; after using this mask, can overwrite back to the origin time
	# arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
	# arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), mask_excess_noise[ikeep_p1, ikeep_p2, 0].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), mask_excess_noise[ikeep_s1, ikeep_s2, 1].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
	arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

	if len(arrivals) == 0:
		arrivals = -1*np.zeros((1,5))
		arrivals[0,0] = np.random.rand()*T
		arrivals[0,1] = int(np.floor(np.random.rand()*(locs.shape[0] - 1)))
	
	# s_extra = 0.0 ## If this is non-zero, it can increase (or decrease) the total rate of missed s waves compared to p waves
	t_inc = np.floor(arrivals[:,3]/dt).astype('int')
	p_miss_rate = 0.5*station_miss_rate[arrivals[:,1].astype('int'), t_inc] + 0.5*global_miss_rate[t_inc]

	if miss_pick_fraction is not False: ## Scale random delete rates to min and max values (times inflate)
		inflate = 1.5
		p_miss_rate1 = np.copy(p_miss_rate)
		low_val, high_val = np.quantile(p_miss_rate, 0.1), np.quantile(p_miss_rate, 0.9)
		p_miss_rate1 = (p_miss_rate - low_val)/(high_val - low_val) # approximate min-max normalization with quantiles
		p_miss_rate1 = inflate*p_miss_rate1*(miss_pick_fraction[1] - miss_pick_fraction[0]) + miss_pick_fraction[0]
		p_miss_rate1 = p_miss_rate1 + 0.5*(np.random.rand() - 0.5)*(miss_pick_fraction[1] - miss_pick_fraction[0]) ## Random shift of 25% of range
		idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < p_miss_rate1)[0]
		print('Deleting %d of %d (%0.2f) picks \n'%(len(idel), len(arrivals), len(idel)/len(arrivals)))
	else:
		## Previous delete random pick version
		idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < dt*p_miss_rate/T)[0]
		print('Deleting %d of %d (%0.2f) picks \n'%(len(idel), len(arrivals), len(idel)/len(arrivals)))

	arrivals = np.delete(arrivals, idel, axis = 0)
	n_events = len(src_times)

	icoda = np.where(np.random.rand(arrivals.shape[0]) < coda_rate)[0]
	if len(icoda) > 0:
		false_coda_arrivals = np.random.rand(len(icoda))*(coda_win[1] - coda_win[0]) + coda_win[0] + arrivals[icoda,0] + arrivals[icoda,3]
		false_coda_arrivals = np.concatenate((false_coda_arrivals.reshape(-1,1), arrivals[icoda,1].reshape(-1,1), -1.0*np.ones((len(icoda),1)), np.zeros((len(icoda),1)), -1.0*np.ones((len(icoda),1))), axis = 1)
		arrivals = np.concatenate((arrivals, false_coda_arrivals), axis = 0)

	## Base false events
	station_false_rate_eval = 0.5*station_false_rate + 0.5*global_false_rate

	# if use_false_ratio_value == True: ## If true, use the ratio of real events to guide false picks
	# 	station_false_rate_eval = max_false_events*np.mean(miss_pick_fraction)*station_false_rate_eval*(global_event_rate.mean()/station_false_rate_eval.mean()) # station_false_rate_eval

	use_clean_data_interval = True
	if use_clean_data_interval == True:
		## Remove a section of false picks completely, so very clean events are also shown in training (to stabalize the single input pick per station per phase type case with the attention mechanism)
		frac_interval = [0.1, 0.3] ## Between 0.1 and 0.3 of the full time window
		frac_length_sample = np.random.rand()*(frac_interval[1] - frac_interval[0]) + frac_interval[0]
		interval_length = int(np.floor(station_false_rate_eval.shape[1]*frac_length_sample))
		ichoose_start = np.random.choice(station_false_rate_eval.shape[1] - interval_length)
		station_false_rate_eval[:,ichoose_start:(ichoose_start + interval_length)] = 0.0 ## Set false pick rate to zero during this interval, for all stations
	
	
	vals = np.random.poisson(dt*station_false_rate_eval/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average

	# How to speed up this part?
	i1, i2 = np.where(vals > 0)
	v_val, t_val = vals[i1,i2], tsteps[i2]
	false_times = np.repeat(t_val, v_val) + np.random.rand(vals.sum())*dt
	false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])
	n_false = len(false_times)
	false_arrivals = np.concatenate((false_times.reshape(-1,1), false_indices.reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
	arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)

	# n_spikes = np.random.randint(0, high = int(max_num_spikes*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
	if int(max_num_spikes*T/(3600*24)) > 0:
		n_spikes = np.random.randint(0, high = int(max_num_spikes*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
		n_spikes_extent = np.random.randint(int(np.floor(n_sta*0.35)), high = n_sta, size = n_spikes) ## This many stations per spike
		time_spikes = np.random.rand(n_spikes)*T
		sta_ind_spikes = [np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)]
		if len(sta_ind_spikes) > 0: ## Add this catch, to avoid error of np.hstack if len(sta_ind_spikes) == 0
			sta_ind_spikes = np.hstack(sta_ind_spikes)
			sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*spike_time_spread for j in range(n_spikes)])
			false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
			arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes


	if use_teleseisim_noise == True:
		max_num_teleseisms = 10

		n_teleseisms = np.random.randint(0, high = int(max_num_spikes*T/(3600*24)))
		n_teleseisms_extent = np.random.randint(int(np.floor(n_sta*0.35)), high = n_sta, size = n_teleseisms)
		sta_ind_teleseisms = [np.random.choice(n_sta, size = n_teleseisms_extent[j], replace = False) for j in range(n_teleseisms)]

		picks_teleseism = []
		if n_teleseisms > 0:

			n_trial_point = int(100*n_teleseisms)
			ichoose = np.random.choice(len(xx_teleseism), size = n_trial_point)
			x_base = np.hstack([np.random.uniform(lat_range_extend[0], lat_range_extend[1], size = n_trial_point).reshape(-1,1), np.random.uniform(lon_range_extend[0], lon_range_extend[1], size = n_trial_point).reshape(-1,1)]) 
			rand_vec = np.random.randn(n_trial_point, 2)
			deg_dist = 0.75*np.random.uniform(xx_teleseism[:,0].min(), xx_teleseism[:,0].max(), size = n_trial_point)

			x_base = x_base + deg_dist.reshape(-1,1)*(rand_vec/np.linalg.norm(rand_vec, axis = 1, keepdims = True))
			ifind = np.where((x_base[:,0] > lat_range_extend[0])*(x_base[:,0] < lat_range_extend[1])*(x_base[:,1] > lon_range_extend[0])*(x_base[:,1] < lon_range_extend[1]))[0]
			x_base = x_base[np.random.choice(np.delete(np.arange(len(x_base)), ifind, axis = 0), size = n_teleseisms)]
			x_depth = np.random.choice(np.unique(xx_teleseism[:,1]), size = len(x_base))
			x_time = np.random.rand(len(x_base))*T

			for j in range(n_teleseisms):
				k_use = np.random.choice(trv_teleseism.shape[1])
				isubset = np.where(xx_teleseism[:,1] == x_depth[j])[0]
				dist_deg = np.linalg.norm(locs[sta_ind_teleseisms[j],0:2] - x_base[j,0:2].reshape(1,-1), axis = 1)
				inearest = cKDTree(xx_teleseism[isubset,0].reshape(-1,1)).query(dist_deg.reshape(-1,1))[1]
				trv_estimate = trv_teleseism[isubset[inearest],k_use]
				inot_nan = np.where(np.isnan(trv_estimate) == 0)[0]
				trv_noise = np.random.randn(len(inot_nan))*sig_t*trv_estimate[inot_nan]

				if len(inot_nan) > 0:
					picks_teleseism.append(np.concatenate((x_time[j] + trv_estimate[inot_nan].reshape(-1,1) + trv_noise.reshape(-1,1), sta_ind_teleseisms[j][inot_nan].reshape(-1,1), np.array([-1, 0, -1]).reshape(1,-1).repeat(len(inot_nan), axis = 0)), axis = 1))

				# print(trv_estimate[inot_nan])

		if len(picks_teleseism) > 0:
			picks_teleseism = np.vstack(picks_teleseism)
			arrivals = np.concatenate((arrivals, picks_teleseism), axis = 0)

		# pdb.set_trace()


		# ip_nearest = []



		# pdb.set_trace()

		# deg_vals = 

		# trv_times = [trv_teleseism[cKDTree(np.array([100e3, 1.0]).reshape(1,-1)*(xx_teleseism)]]

		## From these source coordinates must estimate travel times to stations (should use interpolation)



		# src_teleseism = x_base[ifind]
		# times_teleseism = 





	# use_stable_association_labels = True
	## Check which true picks have so much noise, they should be marked as `false picks' for the association labels
	iexcess_noise = []
	assert(use_stable_association_labels == True)
	if use_stable_association_labels == True: ## It turns out association results are fairly sensitive to this choice
		# thresh_noise_max = 2.5 # ratio of sig_t*travel time considered excess noise
		# min_misfit_allowed = 1.0 # min misfit time for establishing excess noise (now set in train_config.yaml)
		iz = np.where(arrivals[:,4] >= 0)[0]


		if use_correlated_travel_time_noise == True:
		
			iexcess_noise = np.where(arrivals[iz,3] > 0)[0]
			# noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
			# iexcess_noise = np.where(np.abs(noise_values) > np.maximum(min_misfit_allowed, thresh_noise_max*sig_t*arrivals[iz,0]))[0]
			arrivals[iz,0] = arrivals[iz,0] + src_times[arrivals[iz,2].astype('int')] # + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise
			arrivals[iz,3] = src_times[arrivals[iz,2].astype('int')] ## Write real picks fourth column back to origin times, for consistency with previous approach
		
		else:
			noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
			iexcess_noise = np.where(np.abs(noise_values) > np.maximum(min_misfit_allowed, thresh_noise_max*sig_t*arrivals[iz,0]))[0]
			arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise
		
		if len(iexcess_noise) > 0: ## Set these arrivals to "false arrivals", since noise is so high
			init_phase_type = arrivals[iz[iexcess_noise],4]
			arrivals[iz[iexcess_noise],2] = -1
			arrivals[iz[iexcess_noise],3] = 0 ## Could choose to leave this equal to the original source time, as it was originally connected to those sources (must check if this column is ever used later based on noise class)
			arrivals[iz[iexcess_noise],4] = -1

	else: ## This was the original version
		iz = np.where(arrivals[:,4] >= 0)[0]
		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]


	
	# else: ## This was the original version
	# 	iz = np.where(arrivals[:,4] >= 0)[0]
	# 	arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]

	
	## Check which sources are active
	source_tree_indices = cKDTree(arrivals[:,2].reshape(-1,1))
	lp = source_tree_indices.query_ball_point(np.arange(n_events).reshape(-1,1), r = 0)
	lp_backup = [lp[j] for j in range(len(lp))]
	n_unique_station_counts = np.array([len(np.unique(arrivals[lp[j],1])) for j in range(n_events)])
	cnt_p_srcs = np.array([len(np.where(arrivals[lp[j],4] == 0)[0]) for j in range(n_events)])
	cnt_s_srcs = np.array([len(np.where(arrivals[lp[j],4] == 1)[0]) for j in range(n_events)])
	## Compute density of counts based on stations
	# active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
	active_sources = np.where(((n_unique_station_counts >= min_sta_arrival)*((cnt_p_srcs + cnt_s_srcs) >= min_pick_arrival)))[0] # subset of sources

	
	src_times_active = src_times[active_sources]

	## If true, return only the synthetic arrivals
	if return_only_data == True:
		srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
		data = [arrivals, srcs, active_sources]	## Note: active sources within region are only active_sources[np.where(inside_interior[active_sources] > 0)[0]]
		return data
	
	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))

	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
	phase_observed = np.copy(arrivals[:,-1]).astype('int')

	if len(iwhere_false) > 0: # For false picks, assign a random phase type
		phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))
	if len(iexcess_noise) > 0:
		phase_observed[iz[iexcess_noise]] = init_phase_type ## These "false" picks are only false because they have unusually high travel time error, but the phase type should not be randomly chosen 

	perturb_phases = True # For true picks, randomly flip a fraction of phases
	if (len(phase_observed) > 0)*(perturb_phases == True):
		frac_perturb_interval = [0.1, 0.3]
		frac_perturb_sample = np.random.rand()*(frac_perturb_interval[1] - frac_perturb_interval[0]) + frac_perturb_interval[0]
		if len(iexcess_noise) > 0:
			iwhere_real = np.sort(np.array(list(set(iwhere_real).union(iz[iexcess_noise])))).astype('int')
		n_switch = int(np.random.rand()*(frac_perturb_sample*len(iwhere_real))) # switch up to 20% phases
		iflip = np.random.choice(iwhere_real, size = n_switch, replace = False)
		phase_observed[iflip] = np.mod(phase_observed[iflip] + 1, 2)
	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.
	

	if use_sources == False:
		time_samples = np.sort(np.random.rand(n_batch)*T) ## Uniform

	elif use_sources == True:
		time_samples = src_times_active[np.sort(np.random.choice(len(src_times_active), size = n_batch))]

	l_src_times_active = len(src_times_active)
	if (use_preferential_sampling == True)*(len(src_times_active) > 1): # Should the second condition just be (len(src_times_active) > 0) ?
		for j in range(n_batch):
			if np.random.rand() > 0.5: # 30% of samples, re-focus time. # 0.7
				# time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()
				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*(time_shift_range/2.0)*np.random.laplace()

	time_samples = np.sort(time_samples)

	ilen = 0
	if use_consistency_loss == True: # (i > int(n_epochs/5))
		ilen = int(np.floor(len(time_samples)/2/2))
		# ind_sample_consistency 

		ichoose_sample = np.sort(np.random.choice(len(time_samples), size = ilen, replace = False)).astype('int')
		inot_sample = np.sort(np.random.choice(np.delete(np.arange(len(time_samples)), ichoose_sample, axis = 0).astype('int'), size = n_batch - 2*ilen, replace = False))
		irandt_shift = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = ilen)/2.0
		irandt_shift_repeat = (irandt_shift.repeat(2))*np.tile(np.array([0,1]), len(irandt_shift))

		time_samples = np.concatenate((time_samples[inot_sample], time_samples[ichoose_sample].repeat(2) + irandt_shift_repeat), axis = 0)

		# time_samples[-ilen::] = time_samples[ichoose_sample]


	max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))
	min_t = float(np.floor(min([x_grids_trv[j].min() for j in range(len(x_grids_trv))]))) if use_time_shift == True else 0.0

	tree_src_times_all = cKDTree(src_times[:,np.newaxis])
	tree_src_times = cKDTree(src_times_active[:,np.newaxis])
	# lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)
	lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 1.0*(time_shift_range/2.0))

	st = time.time()
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + (max_t - min_t)/2.0 + min_t, r = t_win + (max_t - min_t)/2.0) 

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
	if len(lp_concat) == 0:
		lp_concat = np.array([0]) # So it doesnt fail?
	arrivals_select = arrivals[lp_concat]
	phase_observed_select = phase_observed[lp_concat]

	Trv_subset_p = []
	Trv_subset_s = []
	Station_indices = []
	Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	if (fixed_subnetworks != False):
		fixed_subnetworks_flag = 1
	else:
		fixed_subnetworks_flag = 0		

	active_sources_per_slice_l = []

	for i in range(n_batch):

		# if (use_consistency_loss == True)*(np.mod(i, 2) == 1)*(i >= (n_batch - 2*ilen)):
		# 	i0 = Grid_indices[i - 1] ## Use repeated grid if use_consistency_loss = True
		# else:
		# 	i0 = np.random.randint(0, high = len(x_grids))

		# if (use_consistency_loss == True)*(np.mod(i, 2) == 1)*(i >= (n_batch - 2*ilen)):
		# 	i0 = Grid_indices[i - 1] ## Use repeated grid if use_consistency_loss = True
		# else:
		
		i0 = np.random.randint(0, high = len(x_grids))

		# i0 = np.random.randint(0, high = len(x_grids))
		n_spc = x_grids[i0].shape[0]
		if use_full_network == True:
			n_sta_select = n_sta
			ind_sta_select = np.arange(n_sta)

		else:
			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
				n_sta_select = len(fixed_subnetworks[isub_network])
				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.

			else:
				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

			min_spc_allowed = None
			if min_spc_allowed is not None:
				mp = LocalMarching(device = device)
				locs_out = mp(np.concatenate((locs[ind_sta_select], np.zeros((len(ind_sta_select),1)), np.ones((len(ind_sta_select),1)) + 0.1*np.random.rand(len(ind_sta_select),1)), axis = 1), ftrns1, tc_win = 1.0, sp_win = min_spc_allowed)[:,0:3]					
				tree_locs = cKDTree(ftrns1(locs[ind_sta_select]))
				ip_retained = tree_locs.query(ftrns1(locs_out))[1]
				ind_sta_select = ind_sta_select[ip_retained]
				n_sta_select = len(ind_sta_select)
				
		# if use_time_shift == False:

		Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication

		# else:

		# Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		# Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication

		# Trv_subset_p[-1] = Trv_subset_p[-1]


		Station_indices.append(ind_sta_select) # record subsets used
		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
		Grid_indices.append(i0)
		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
		sc += len(Sample_indices[-1])

		tree_subset = cKDTree(ind_sta_select.reshape(-1,1))
		active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]
		cnt_per_slice_p = np.array([len(np.where((arrivals[lp_backup[j],4] == 0)*(tree_subset.query(arrivals[lp_backup[j],1].reshape(-1,1))[0] == 0))[0]) for j in lp_src_times_all[i]])
		cnt_per_slice_s = np.array([len(np.where((arrivals[lp_backup[j],4] == 1)*(tree_subset.query(arrivals[lp_backup[j],1].reshape(-1,1))[0] == 0))[0]) for j in lp_src_times_all[i]])
		active_sources_per_slice = np.array(list(set(active_sources_per_slice).intersection(np.where((cnt_per_slice_p + cnt_per_slice_s) >= min_pick_arrival)[0]))).astype('int')
		
		active_sources_per_slice_l.append(active_sources_per_slice)

	Trv_subset_p = np.vstack(Trv_subset_p)
	Trv_subset_s = np.vstack(Trv_subset_s)
	Batch_indices = np.hstack(Batch_indices)


	offset_per_batch = 1.5*(np.abs(max_t - min_t))
	offset_per_station = 1.5*n_batch*offset_per_batch


	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
	n_arvs = arrivals_select.shape[0]

	# Rather slow!
	iargsort = np.argsort(arrivals_select[:,0])
	arrivals_select = arrivals_select[iargsort]
	phase_observed_select = phase_observed_select[iargsort]

	iwhere_p = np.where(phase_observed_select == 0)[0]
	iwhere_s = np.where(phase_observed_select == 1)[0]
	n_arvs_p = len(iwhere_p)
	n_arvs_s = len(iwhere_s)

	query_time_p = Trv_subset_p[:,0] + Batch_indices*offset_per_batch + Trv_subset_p[:,1]*offset_per_station
	query_time_s = Trv_subset_s[:,0] + Batch_indices*offset_per_batch + Trv_subset_s[:,1]*offset_per_station

	## No phase type information
	ip_p = np.searchsorted(arrivals_select[:,0], query_time_p)
	ip_s = np.searchsorted(arrivals_select[:,0], query_time_s)

	ip_p_pad = ip_p.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s_pad = ip_s.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
	ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

	if use_sign_input == False:
		rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)
	else:
		rel_t_p = query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_s = query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]
		rel_t_p_ind = np.argmin(np.abs(rel_t_p), axis = 1)
		rel_t_s_ind = np.argmin(np.abs(rel_t_s), axis = 1)
		rel_t_p_slice = rel_t_p[np.arange(len(rel_t_p)),rel_t_p_ind]
		rel_t_s_slice = rel_t_s[np.arange(len(rel_t_s)),rel_t_s_ind]
		rel_t_p = np.sign(rel_t_p_slice)*np.abs(rel_t_p_slice) ## Preserve sign information
		rel_t_s = np.sign(rel_t_s_slice)*np.abs(rel_t_s_slice)

	## With phase type information
	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

	if use_sign_input == False:
	
		if len(iwhere_p) > 0:
			rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		else:
			# rel_t_p1 = np.zeros(rel_t_p.shape)
			rel_t_p1 = np.random.choice([-1.0, 1.0], size = rel_t_p.shape)*np.ones(rel_t_p.shape)*kernel_sig_t*10.0 ## Need to place null values as large offset, so they map to zero
	
		if len(iwhere_s) > 0:
			rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)
		else:
			# rel_t_s1 = np.zeros(rel_t_s.shape)
			rel_t_s1 = np.random.choice([-1.0, 1.0], size = rel_t_s.shape)*np.ones(rel_t_s.shape)*kernel_sig_t*10.0 ## Need to place null values as large offset, so they map to zero

	else:

		if len(iwhere_p) > 0:
			rel_t_p1 = query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_p1_ind = np.argmin(np.abs(rel_t_p1), axis = 1)
			rel_t_p1_slice = rel_t_p1[np.arange(len(rel_t_p1)),rel_t_p1_ind]
			rel_t_p1 = np.sign(rel_t_p1_slice)*np.abs(rel_t_p1_slice) ## Preserve sign information
		else:
			# rel_t_p1 = np.zeros(rel_t_p.shape)
			rel_t_p1 = np.random.choice([-1.0, 1.0], size = rel_t_p.shape)*np.ones(rel_t_p.shape)*kernel_sig_t*10.0 ## Need to place null values as large offset, so they map to zero
	
		if len(iwhere_s) > 0:
			rel_t_s1 = query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_s1_ind = np.argmin(np.abs(rel_t_s1), axis = 1)
			rel_t_s1_slice = rel_t_s1[np.arange(len(rel_t_s1)),rel_t_s1_ind]
			rel_t_s1 = np.sign(rel_t_s1_slice)*np.abs(rel_t_s1_slice) ## Preserve sign information
		else:
			# rel_t_s1 = np.zeros(rel_t_s.shape)
			rel_t_s1 = np.random.choice([-1.0, 1.0], size = rel_t_s.shape)*np.ones(rel_t_s.shape)*kernel_sig_t*10.0 ## Need to place null values as large offset, so they map to zero
			

	Inpts = []
	Masks = []
	Lbls = []
	Lbls_query = []
	X_fixed = []
	X_query = []
	Locs = []
	Trv_out = []

	A_sta_sta_l = []
	A_src_src_l = []
	A_prod_sta_sta_l = []
	A_prod_src_src_l = []
	A_src_in_prod_l = []
	A_edges_time_p_l = []
	A_edges_time_s_l = []
	A_edges_ref_l = []

	# if use_expanded_graphs == True:
	Ac_src_src_l = []
	Ac_prod_src_src_l = []

	lp_times = []
	lp_stations = []
	lp_phases = []
	lp_meta = []
	lp_srcs = []

	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grids[grid_select].shape[0]
		n_sta_slice = len(sta_select)

		inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
		if use_sign_input == False:
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))
		else:
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.sign(rel_t_p[ind_select])*np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.sign(rel_t_s[ind_select])*np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.sign(rel_t_p1[ind_select])*np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.sign(rel_t_s1[ind_select])*np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))
		
		trv_out = x_grids_trv[grid_select][:,sta_select,:] ## Subsetting, into sliced indices.
		Inpts.append(inpt[:,sta_select,:]) # sub-select, subset of stations.
		Masks.append(1.0*(np.abs(inpt[:,sta_select,:]) > thresh_mask))
		Trv_out.append(trv_out)
		Locs.append(locs[sta_select])
		X_fixed.append(x_grids[grid_select])

		## Assemble pick datasets
		perm_vec = -1*np.ones(n_sta)
		perm_vec[sta_select] = np.arange(len(sta_select))
		meta = arrivals[lp[i],:]
		phase_vals = phase_observed[lp[i]]
		times = meta[:,0]
		indices = perm_vec[meta[:,1].astype('int')]
		ineed = np.where(indices > -1)[0]
		times = times[ineed] ## Overwrite, now. Double check if this is ok.
		indices = indices[ineed]
		phase_vals = phase_vals[ineed]
		meta = meta[ineed]

		# pdb.set_trace()

		active_sources_per_slice = np.array(lp_src_times_all[i])[np.array(active_sources_per_slice_l[i])]
		ind_inside = np.where(inside_interior[active_sources_per_slice.astype('int')] > 0)[0]
		active_sources_per_slice = active_sources_per_slice[ind_inside]

		ind_src_unique = np.unique(meta[meta[:,2] > -1.0,2]).astype('int') # ignore -1.0 entries.

		if len(ind_src_unique) > 0:
			ind_src_unique = np.sort(np.array(list(set(ind_src_unique).intersection(active_sources_per_slice)))).astype('int')

		src_subset = np.concatenate((src_positions[ind_src_unique], src_times[ind_src_unique].reshape(-1,1) - time_samples[i]), axis = 1)
		if len(ind_src_unique) > 0:
			perm_vec_meta = np.arange(ind_src_unique.max() + 1)
			perm_vec_meta[ind_src_unique] = np.arange(len(ind_src_unique))
			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)
			ifind = np.where([meta[j,2] in ind_src_unique for j in range(meta.shape[0])])[0]
			meta[ifind,-1] = perm_vec_meta[meta[ifind,2].astype('int')] # save pointer to active source, for these picks (in new, local index, of subset of sources)
		else:
			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)

		# Do these really need to be on cuda?
		lex_sort = np.lexsort((times, indices)) ## Make sure lexsort doesn't cause any problems
		lp_times.append(times[lex_sort] - time_samples[i])
		lp_stations.append(indices[lex_sort])
		lp_phases.append(phase_vals[lex_sort])
		lp_meta.append(meta[lex_sort]) # final index of meta points into 
		lp_srcs.append(src_subset)

		if skip_graphs == False:
		
			A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), k = k_sta_edges + 1).flip(0).contiguous())[0]
			# A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), k = k_spc_edges + 1).flip(0).contiguous())[0]

			A_src_src = remove_self_loops(knn(torch.cat((torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), scale_time*torch.Tensor(x_grids[grid_select][:,3].reshape(-1,1)).to(device)), dim = 1), torch.cat((torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), scale_time*torch.Tensor(x_grids[grid_select][:,3].reshape(-1,1)).to(device)), dim = 1), k = k_spc_edges + 1).flip(0).contiguous())[0]
			## Cross-product graph is: source node x station node. Order as, for each source node, all station nodes.
	
			# Cross-product graph, nodes connected by: same source node, connected stations
			A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).to(device)).contiguous()
			A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).to(device)).contiguous()	
	

			if use_expanded == True:

				use_perm_expand = True
				if use_perm_expand == True:
					perm_vec_expand = np.random.permutation(np.arange(x_grids[grid_select].shape[0])).astype('int')
					Ac_src_src = torch.Tensor(perm_vec_expand[Ac]).long().to(device)
				else:
					perm_vec_expand = np.arange(x_grids[grid_select].shape[0]).astype('int')
					Ac_src_src = torch.Tensor(perm_vec_expand[Ac]).long().to(device)

				A_src_in_sta = torch.Tensor(np.concatenate((np.tile(np.arange(locs[sta_select].shape[0]), len(x_grids[grid_select])).reshape(1,-1), np.arange(len(x_grids[grid_select])).repeat(len(locs[sta_select]), axis = 0).reshape(1,-1)), axis = 0)).long().to(device)

				Ac_prod_src_src = build_src_src_product(Ac_src_src, A_src_in_sta, locs[sta_select], x_grids[grid_select], device = device)

				Ac_src_src_l.append(Ac_src_src.cpu().detach().numpy())
				Ac_prod_src_src_l.append(Ac_prod_src_src.cpu().detach().numpy())

			# else:

			# 	Ac_src_src_l.append([])
			# 	Ac_prod_src_src_l.append([])				


			# For each unique spatial point, sum in all edges.
			A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1)), dim = 0).to(device).contiguous()
	
			## Sub-selecting from the time-arrays, is easy, since the time-arrays are indexed by station (triplet indexing; )
			len_dt = len(x_grids_trv_refs[grid_select])
	
			### Note: A_edges_time_p needs to be augmented: by removing stations, we need to re-label indices for subsequent nodes,
			### To the "correct" number of stations. Since, not n_sta shows up in definition of edges. "assemble_pointers.."
			A_edges_time_p = x_grids_trv_pointers_p[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
			A_edges_time_s = x_grids_trv_pointers_s[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
			## Need to convert these edges again. Convention is:
			## subtract i (station index absolute list), divide by n_sta, mutiply by N stations, plus ith station (in permutted indices)
			# shape is len_dt*k_time_edges*len(sta_select)
			one_vec = np.repeat(sta_select*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere
			A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
			A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
	
			assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
			assert(A_edges_time_s.max() < n_spc*n_sta_slice)
	
			A_sta_sta_l.append(A_sta_sta.cpu().detach().numpy())
			A_src_src_l.append(A_src_src.cpu().detach().numpy())
			A_prod_sta_sta_l.append(A_prod_sta_sta.cpu().detach().numpy())
			A_prod_src_src_l.append(A_prod_src_src.cpu().detach().numpy())
			A_src_in_prod_l.append(A_src_in_prod.cpu().detach().numpy())
			A_edges_time_p_l.append(A_edges_time_p)
			A_edges_time_s_l.append(A_edges_time_s)
			A_edges_ref_l.append(x_grids_trv_refs[grid_select])

		else:
			
			A_sta_sta_l.append([])
			A_src_src_l.append([])
			A_prod_sta_sta_l.append([])
			A_prod_src_src_l.append([])
			A_src_in_prod_l.append([])
			A_edges_time_p_l.append([])
			A_edges_time_s_l.append([])
			A_edges_ref_l.append([])

			Ac_src_src_l.append([])
			Ac_prod_src_src_l.append([])

		
		x_query = np.random.rand(n_spc_query, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.
		x_query_t = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = len(x_query))

		if len(lp_srcs[-1]) > 0:
			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]
			x_query_t[0:len(lp_srcs[-1])] = lp_srcs[-1][:,3]

		n_frac_focused_queries = 0.2
		n_concentration_focused_queries = 0.05 # 5% of scale of domain
		if (len(lp_srcs[-1]) > 0)*(n_frac_focused_queries > 0):
			n_focused_queries = int(n_frac_focused_queries*n_spc_query)
			ind_overwrite_focused_queries = np.sort(np.random.choice(n_spc_query, size = n_focused_queries, replace = False))
			ind_source_focused = np.random.choice(len(lp_srcs[-1]), size = n_focused_queries)

			# x_query_focused = np.random.randn(n_focused_queries, 3)*scale_x*n_concentration_focused_queries
			# x_query_focused = x_query_focused + lp_srcs[-1][ind_source_focused,0:3]
			x_query_focused = 2.0*np.random.randn(n_focused_queries, 3)*np.mean([src_x_kernel, src_depth_kernel])			
			x_query_focused = ftrns2(x_query_focused + ftrns1(lp_srcs[-1][ind_source_focused,0:3]))

			ioutside = np.where(((x_query_focused[:,2] < depth_range[0]) + (x_query_focused[:,2] > depth_range[1])) > 0)[0]
			x_query_focused[ioutside,2] = np.random.rand(len(ioutside))*(depth_range[1] - depth_range[0]) + depth_range[0]			
			x_query_focused = np.maximum(np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1), x_query_focused)
			x_query_focused = np.minimum(np.array([lat_range_extend[1], lon_range_extend[1], depth_range[1]]).reshape(1,-1), x_query_focused)
			x_query[ind_overwrite_focused_queries] = x_query_focused

			x_query_focused_t = 2.0*np.random.randn(n_focused_queries)*src_t_kernel			
			x_query_focused_t = lp_srcs[-1][ind_source_focused,3] + x_query_focused_t
			# ioutside = np.where(((x_query_focused_t < min_t) + (x_query_focused_t > max_t)) > 0)[0]
			ioutside = np.where(((x_query_focused_t < (-time_shift_range/2.0)) + (x_query_focused_t > (time_shift_range/2.0))) > 0)[0]
			x_query_focused_t[ioutside] = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = len(ioutside))
			x_query_t[ind_overwrite_focused_queries] = x_query_focused_t


		if (use_consistency_loss == True)*(np.mod(i, 2) == 1)*(i >= (n_batch - 2*ilen)):
			ind_consistency = int(np.floor(len(X_query[i - 1])/2))
			x_query[ind_consistency::] = X_query[i - 1][ind_consistency::,0:3]
			x_query_t[ind_consistency::] = X_query[i - 1][ind_consistency::,3] - irandt_shift[int((i - (n_batch - 2*ilen))/2)]
			# ioutside = np.where(((x_query_t < min_t) + (x_query_t > max_t)) > 0)[0]
			ioutside = np.where(((x_query_t < (-time_shift_range/2.0)) + (x_query_t > (time_shift_range/2.0))) > 0)[0]
			x_query_t[ioutside] = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = len(ioutside))



		# print('Len [3] %d'%len(active_sources_per_slice))

		if len(active_sources_per_slice) == 0:
			lbls_grid = np.zeros((x_grids[grid_select].shape[0], 1))
			lbls_query = np.zeros((n_spc_query, 1))

			if use_gradient_loss == True:
				lbls_grid = [lbls_grid, np.zeros(((len(lbls_grid)),3)), np.zeros(len(lbls_grid))]
				lbls_query = [lbls_query, np.zeros(((len(lbls_query)),3)), np.zeros(len(lbls_query))]

		else:
			active_sources_per_slice = active_sources_per_slice.astype('int')
			# print('Len [1] %d'%len(active_sources_per_slice))

			# Combine components
			# lbls_grid = (np.expand_dims(spatial_exp_term.sum(2), axis=1) * temporal_exp_term).max(2)
			# lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

			## Note for consistency with above, should be using ind_src_unique for the indices of sources, though it looks like ind_src_unique == active_sources_per_slice

			if use_gradient_loss == False:

				# lbls_grid = (np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*(((time_samples[i] + x_grids[grid_select][:,3]).reshape(-1,1) - src_times[active_sources_per_slice].reshape(1,-1))**2)/(src_t_kernel**2))).max(1).reshape(-1,1)
				# lbls_query = (np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*(((time_samples[i] + x_query_t).reshape(-1,1) - src_times[active_sources_per_slice].reshape(1,-1))**2)/(src_t_kernel**2))).max(1).reshape(-1,1)
				lbls_grid = (np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*((x_grids[grid_select][:,3].reshape(-1,1) - (src_times[active_sources_per_slice].reshape(1,-1) - time_samples[i]))**2)/(src_t_kernel**2))).max(1).reshape(-1,1)
				lbls_query = (np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*((x_query_t.reshape(-1,1) - (src_times[active_sources_per_slice].reshape(1,-1) - time_samples[i]))**2)/(src_t_kernel**2))).max(1).reshape(-1,1)

				# if len(active_sources_per_slice) > 1:
				# 	pass
					# print('Pdb')
					#  pdb.set_trace()

			else:

				# x_inpt = Variable(torch.Tensor(np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1)))

				lbls_grid_arg = np.argmax((np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*(((time_samples[i] + x_grids[grid_select][:,3]).reshape(-1,1) - src_times[active_sources_per_slice].reshape(1,-1))**2)/(src_t_kernel**2))), axis = 1) # .max(1) # .reshape(-1,1)
				lbls_query_arg = np.argmax((np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*(((time_samples[i] + x_query_t).reshape(-1,1) - src_times[active_sources_per_slice].reshape(1,-1))**2)/(src_t_kernel**2))), axis = 1) # .max(1).reshape(-1,1)

				# lbls_grid = (np.exp(-0.5*(((ftrns1(x_grids[grid_select]) - ftrns1(src_positions[active_sources_per_slice[lbls_grid_arg]].reshape(1,-1)))**2)/(src_spatial_kernel**2)).sum(1))*np.exp(-0.5*(((time_samples[i] + x_grids[grid_select][:,3]).reshape(-1) - src_times[active_sources_per_slice[lbls_grid_arg]].reshape(-1))**2)/(src_t_kernel**2))) # .max(1).reshape(-1,1)
				# lbls_query = (np.exp(-0.5*(((ftrns1(x_query) - ftrns1(src_positions[active_sources_per_slice[lbls_query_arg]].reshape(1,-1)))**2)/(src_spatial_kernel**2)).sum(1))*np.exp(-0.5*(((time_samples[i] + x_query_t).reshape(-1) - src_times[active_sources_per_slice[lbls_query_arg]].reshape(-1))**2)/(src_t_kernel**2))) # .max(1).reshape(-1,1)

				src_t_kernel_cuda = torch.Tensor([src_t_kernel]).to(device)
				src_spatial_kernel_cuda = torch.Tensor(src_spatial_kernel[0]).to(device)
				inpt_grid = Variable(ftrns1_diff(torch.Tensor(x_grids[grid_select]).to(device)), requires_grad = True)
				inpt_query = Variable(ftrns1_diff(torch.Tensor(x_query).to(device)), requires_grad = True)
				inpt_grid_t = Variable(torch.Tensor(x_grids[grid_select][:,3]).to(device), requires_grad = True)
				inpt_query_t = Variable(torch.Tensor(x_query_t).to(device), requires_grad = True)

				# lbls_grid = (torch.exp(-0.5*(((ftrns1_diff(x_grids[grid_select]) - ftrns1_diff(torch.Tensor(src_positions[active_sources_per_slice[lbls_grid_arg]].reshape(1,-1)).to(device)))**2)/(src_spatial_kernel_cuda**2)).sum(1))*torch.exp(-0.5*(((torch.Tensor(time_samples[i]).to(device) + x_grids[grid_select][:,3]).reshape(-1) - torch.Tensor(src_times[active_sources_per_slice[lbls_grid_arg]].reshape(-1))**2)/(src_t_kernel_cuda**2)))) # .max(1).reshape(-1,1)
				# lbls_query = (torch.exp(-0.5*(((ftrns1_diff(x_query) - ftrns1_diff(torch.Tensor(src_positions[active_sources_per_slice[lbls_query_arg]].reshape(1,-1)).to(device)))**2)/(src_spatial_kernel_cuda**2)).sum(1))*torch.exp(-0.5*(((torch.Tensor(time_samples[i]).to(device) + x_query_t).reshape(-1) - torch.Tensor(src_times[active_sources_per_slice[lbls_query_arg]].reshape(-1))**2)/(src_t_kernel_cuda**2)))) # .max(1).reshape(-1,1)
				lbls_grid = (torch.exp(-0.5*(((inpt_grid - ftrns1_diff(torch.Tensor(src_positions[active_sources_per_slice[lbls_grid_arg]].reshape(1,-1)).to(device)))**2)/(src_spatial_kernel_cuda**2)).sum(1))*torch.exp(-0.5*(((torch.Tensor([time_samples[i]]).to(device) + inpt_grid_t).reshape(-1) - torch.Tensor(src_times[active_sources_per_slice[lbls_grid_arg]]).to(device).reshape(-1))**2)/(src_t_kernel_cuda**2))) # .max(1).reshape(-1,1)
				lbls_query = (torch.exp(-0.5*(((inpt_query - ftrns1_diff(torch.Tensor(src_positions[active_sources_per_slice[lbls_query_arg]].reshape(1,-1)).to(device)))**2)/(src_spatial_kernel_cuda**2)).sum(1))*torch.exp(-0.5*(((torch.Tensor([time_samples[i]]).to(device) + inpt_query_t).reshape(-1) - torch.Tensor(src_times[active_sources_per_slice[lbls_query_arg]]).to(device).reshape(-1))**2)/(src_t_kernel_cuda**2))) # .max(1).reshape(-1,1)

				torch_grid_vec = torch.ones(len(inpt_grid)).to(device)
				torch_query_vec = torch.ones(len(inpt_query)).to(device)

				grad_grid_spc = src_spatial_kernel_cuda.mean()*torch.autograd.grad(inputs = inpt_grid, outputs = lbls_grid, grad_outputs = torch_grid_vec, retain_graph = True, create_graph = True)[0]
				grad_grid_t = src_t_kernel_cuda*torch.autograd.grad(inputs = inpt_grid_t, outputs = lbls_grid, grad_outputs = torch_grid_vec, retain_graph = True, create_graph = True)[0]

				grad_query_spc = src_spatial_kernel_cuda.mean()*torch.autograd.grad(inputs = inpt_query, outputs = lbls_query, grad_outputs = torch_query_vec, retain_graph = True, create_graph = True)[0]
				grad_query_t = src_t_kernel_cuda*torch.autograd.grad(inputs = inpt_query_t, outputs = lbls_query, grad_outputs = torch_query_vec, retain_graph = True, create_graph = True)[0]

				lbls_grid = [lbls_grid.cpu().detach().numpy().reshape(-1,1), grad_grid_spc.cpu().detach().numpy(), grad_grid_t.cpu().detach().numpy()]
				lbls_query = [lbls_query.cpu().detach().numpy().reshape(-1,1), grad_query_spc.cpu().detach().numpy(), grad_query_t.cpu().detach().numpy()]

				# d2 = torch.autograd.grad(inputs = inpt_grad, outputs = pred, grad_outputs = torch_two_vec, retain_graph = True, create_graph = True)[0]
				# d3 = torch.autograd.grad(inputs = inpt_grad, outputs = pred, grad_outputs = torch_three_vec, retain_graph = True, create_graph = True)[0]


			# lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			# lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

			# lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			# lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
		
		# print('Grad')
		# print(lbls_grid[-1].shape)
		# print(lbls_query[-1].shape)

		X_query.append(np.concatenate((x_query, x_query_t.reshape(-1,1)), axis = 1))
		Lbls.append(lbls_grid)
		Lbls_query.append(lbls_query)

	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
	data = [arrivals, srcs, active_sources]	## Note: active sources within region are only active_sources[np.where(inside_interior[active_sources] > 0)[0]]

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	if (use_expanded == False) or (skip_graphs == True):

		return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

	else:

		return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, [A_src_src_l, Ac_src_src_l], A_prod_sta_sta_l, A_prod_src_src_l, [A_src_in_prod_l, Ac_src_in_prod_l], A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)



def pick_labels_extract_interior_region_flattened(xq_src_cart, xq_src_t, source_pick, src_slice, lat_range_interior, lon_range_interior, ftrns1, radius_frac = 0.5, mix_ratio = 0.3, sig_x = 15e3, sig_t = 6.5, use_flattening = False): # can expand kernel widths to other size if prefered

	iz = np.where(source_pick[:,1] > -1.0)[0]
	lbl_trgt = torch.zeros((xq_src_cart.shape[0], source_pick.shape[0], 2)).to(device)
	src_pick_indices = source_pick[iz,1].astype('int')

	inside_interior = ((src_slice[src_pick_indices,0] <= lat_range_interior[1])*(src_slice[src_pick_indices,0] >= lat_range_interior[0])*(src_slice[src_pick_indices,1] <= lon_range_interior[1])*(src_slice[src_pick_indices,1] >= lon_range_interior[0]))

	if len(iz) > 0:

		if use_flattening == False: ## Use Gaussian labels
			d = torch.Tensor(inside_interior.reshape(1,-1)*np.exp(-0.5*(pd(xq_src_cart, ftrns1(src_slice[src_pick_indices,0:3]))**2)/(sig_x**2))*np.exp(-0.5*(pd(xq_src_t.reshape(-1,1), src_slice[src_pick_indices,3].reshape(-1,1))**2)/(sig_t**2))).to(device)

		else: ## Use Box-cars embedded in Gaussians

			dist_val = pd(xq_src_cart, ftrns1(src_slice[src_pick_indices,0:3]))
			mask_dist = dist_val > radius_frac*sig_x
			val_dist = mask_dist*(dist_val - mask_dist*(radius_frac*sig_x)) ## If within radius, this maps to zero; else

			dist_t = pd(xq_src_t.reshape(-1,1), src_slice[src_pick_indices,3].reshape(-1,1))
			mask_t = dist_t > radius_frac*sig_t
			val_t = mask_t*(dist_t - mask_t*(radius_frac*sig_t))

			d = torch.Tensor(inside_interior.reshape(1,-1)*np.exp(-0.5*(val_dist**2)/(sig_x**2))*np.exp(-0.5*(val_t**2)/(sig_t**2))).to(device)

			# if mix_ratio > 0:
			# 	## Mix the Gaussian and box car inside the region
			# 	# d[(1 - mask_dist)*(1 - mask_t)] = ((1.0 - mix_ratio)*d + mix_ratio*torch.Tensor(inside_interior.reshape(1,-1)*np.exp(-0.5*(pd(xq_src_cart, ftrns1(src_slice[src_pick_indices,0:3]))**2)/(sig_x**2))*np.exp(-0.5*(pd(xq_src_t.reshape(-1,1), src_slice[src_pick_indices,3].reshape(-1,1))**2)/(sig_t**2))).to(device))[(1 - mask_dist)*(1 - mask_t)]
			# 	d[(1 - mask_dist)*(1 - mask_t)] = d[(1 - mask_dist)*(1 - mask_t)] + mix_ratio*torch.Tensor(inside_interior.reshape(1,-1)*np.exp(-0.5*(pd(xq_src_cart, ftrns1(src_slice[src_pick_indices,0:3]))**2)/(sig_x**2))*np.exp(-0.5*(pd(xq_src_t.reshape(-1,1), src_slice[src_pick_indices,3].reshape(-1,1))**2)/(sig_t**2))).to(device)[(1 - mask_dist)*(1 - mask_t)]
			# 	d = d/(1.0 + mix_ratio)
			# 	assert(d.amax() <= 1.0)

		lbl_trgt[:,iz,0] = d*torch.Tensor((source_pick[iz,0] == 0)).to(device).float()
		lbl_trgt[:,iz,1] = d*torch.Tensor((source_pick[iz,0] == 1)).to(device).float()

	return lbl_trgt

def compute_source_labels(x_query, x_query_t, src_x, src_t, src_spatial_kernel, src_t_kernel, ftrns1):

	# lbls_grid = (np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*(((time_samples[i] + x_grids[grid_select][:,3]).reshape(-1,1) - src_times[active_sources_per_slice].reshape(1,-1))**2)/(src_t_kernel**2))).max(1).reshape(-1,1)
	
	if len(src_x) == 0:

		return np.zeros((len(x_query),1))

	else:

		return (np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_x), axis = 0))**2)/(src_spatial_kernel**2)).sum(2))*np.exp(-0.5*((x_query_t.reshape(-1,1) - src_t.reshape(1,-1))**2)/(src_t_kernel**2))).max(1).reshape(-1,1)


# def sample_dense_queries(x_query, x_query_t, prob, lat_range_extend, lon_range_extend, depth_range, src_x_kernel, src_depth_kernel, src_t_kernel, time_shift_range, ftrns1, ftrns2, n_frac_focused_queries = 0.2, replace = True, randomize = True, baseline = 0.2):
def sample_dense_queries(x_query, x_query_t, prob, lat_range_extend, lon_range_extend, depth_range, src_x_kernel, src_depth_kernel, src_t_kernel, time_shift_range, ftrns1, ftrns2, n_frac_focused_queries = 0.2, replace = False, randomize = False, baseline = 0.2):

	# n_frac_focused_queries = 0.2
	# n_concentration_focused_queries = 0.05 # 5% of scale of domain

	x_query_sample = np.copy(x_query)
	x_query_sample_t = np.copy(x_query_t)
	n_spc_query = len(x_query)

	if (baseline is not None)*(prob.sum() > 0):
		prob1 = np.copy(prob)
		prob1[(prob <= np.quantile(prob[prob > 0], baseline))*(prob > 0)] = np.quantile(prob[prob > 0], baseline)
		prob1[(prob >= np.quantile(prob[prob > 0], 1.0 - baseline))*(prob > 0)] = np.quantile(prob[prob > 0], 1.0 - baseline)
		prob = np.copy(prob)

	if (len(prob) > 0)*(n_frac_focused_queries > 0):

		n_focused_queries = int(n_frac_focused_queries*n_spc_query)
		ind_overwrite_focused_queries = np.sort(np.random.choice(n_spc_query, size = n_focused_queries, replace = False))
		ind_source_focused = np.random.choice(n_spc_query, p = prob, size = n_focused_queries, replace = True)

		# x_query_focused = np.random.randn(n_focused_queries, 3)*scale_x*n_concentration_focused_queries
		# x_query_focused = x_query_focused + lp_srcs[-1][ind_source_focused,0:3]
		x_query_focused = 2.0*np.random.randn(n_focused_queries, 3)*np.mean([src_x_kernel, src_depth_kernel])			
		x_query_focused = ftrns2(x_query_focused + ftrns1(x_query[ind_source_focused,0:3]))

		ioutside = np.where(((x_query_focused[:,2] < depth_range[0]) + (x_query_focused[:,2] > depth_range[1])) > 0)[0]
		x_query_focused[ioutside,2] = np.random.rand(len(ioutside))*(depth_range[1] - depth_range[0]) + depth_range[0]			
		x_query_focused = np.maximum(np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1), x_query_focused)
		x_query_focused = np.minimum(np.array([lat_range_extend[1], lon_range_extend[1], depth_range[1]]).reshape(1,-1), x_query_focused)
		x_query_sample[ind_overwrite_focused_queries] = x_query_focused

		x_query_focused_t = 2.0*np.random.randn(n_focused_queries)*src_t_kernel			
		x_query_focused_t = x_query_t[ind_source_focused] + x_query_focused_t
		# ioutside = np.where(((x_query_focused_t < min_t) + (x_query_focused_t > max_t)) > 0)[0]
		ioutside = np.where(((x_query_focused_t < (-time_shift_range/2.0)) + (x_query_focused_t > (time_shift_range/2.0))) > 0)[0]
		x_query_focused_t[ioutside] = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = len(ioutside))
		x_query_sample_t[ind_overwrite_focused_queries] = x_query_focused_t

		if randomize == True:
			ind_fixed = np.delete(np.arange(n_spc_query), ind_overwrite_focused_queries, axis = 0)
			x_rand_uniform = np.hstack([np.random.uniform(u[0], u[1], size = len(ind_fixed)).reshape(-1,1) for u in [lat_range_extend, lon_range_extend, depth_range]])
			x_rand_t = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = len(ind_fixed))
			x_query_sample[ind_fixed] = x_rand_uniform
			x_query_sample_t[ind_fixed] = x_rand_t


	if replace == True:

		return x_query_sample, x_query_sample_t

	else:

		return x_query_sample[ind_overwrite_focused_queries], x_query_sample_t[ind_overwrite_focused_queries]


## Alpha : upweight positive samples (alpha near 1)
## Gamma : downweight negatives (gamma large)

class SoftFocalLoss(nn.Module):
	# def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
	def __init__(self, alpha=0.25, gamma=1.0, reduction='mean'):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, logits, targets, mask = None):
		"""
		logits: (N, ...) raw model outputs (no sigmoid)
		targets: same shape, floats in [0,1]
		"""

		# targets_clamp = targets.clamp(min = 1e-4, max = 1 - 1e-4)
		targets_clamp = targets.clamp(min = 1e-3, max = 1 - 1e-4)


		probs = torch.sigmoid(logits)
		eps = 1e-8

		# Clip for numerical safety
		probs = torch.clamp(probs, eps, 1. - eps)

		# Focal weighting
		pt = probs * targets_clamp + (1 - probs) * (1 - targets_clamp)
		focal_weight = (self.alpha * targets_clamp + (1 - self.alpha) * (1 - targets_clamp)) \
						* (1 - pt) ** self.gamma

		# Standard BCE using probs
		bce = -(targets_clamp * torch.log(probs) + (1 - targets_clamp) * torch.log(1 - probs))

		loss = focal_weight * bce

		if mask is not None:
			# mask = torch.ones(loss.shape).to(logits.device)
			loss = loss*mask

		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		return loss

# class GaussianDiceLoss(nn.Module):

# 	# Start with bg_weight = 1.0
# 	# If too many false positives → increase bg_weight to 1.5–3.0
# 	# If missing weak Gaussians → decrease bg_weight to 0.5–0.8

# 	def __init__(self, smooth=1e-5, bg_weight=1.0):
# 		super().__init__()
# 		self.smooth = smooth
# 		self.bg_weight = bg_weight   # usually 1.0, sometimes 0.5–2.0

# 	def forward(self, pred, target):
# 		# No sigmoid! pred is raw linear output
# 		pred = pred.float()
# 		target = target.float()

# 		# intersection = (pred * target).sum(dim=(-2,-1))  # sum over spatial + channel if multi-channel
# 		# pred_sum = (pred ** 2).sum(dim=(-2,-1))
# 		# target_sum = (target ** 2).sum(dim=(-2,-1))

# 		intersection = (pred * target).sum(dim=(-1))  # sum over spatial + channel if multi-channel
# 		pred_sum = (pred ** 2).sum(dim=(-1))
# 		target_sum = (target ** 2).sum(dim=(-1))

# 		dice = 1 - ((2.0 * intersection + self.smooth) /
#                     (pred_sum + self.bg_weight * target_sum + self.smooth))

# 		return dice.mean()

## Replacing row wise sum with mean
# class GaussianDiceLoss(nn.Module):

# 	# Start with bg_weight = 1.0
# 	# If too many false positives → increase bg_weight to 1.5–3.0
# 	# If missing weak Gaussians → decrease bg_weight to 0.5–0.8

# 	def __init__(self, smooth=1e-5, bg_weight=1.0):
# 		super().__init__()
# 		self.smooth = smooth
# 		self.bg_weight = bg_weight   # usually 1.0, sometimes 0.5–2.0

# 	def forward(self, pred, target):
# 		# No sigmoid! pred is raw linear output
# 		pred = pred.float()
# 		target = target.float()

# 		# intersection = (pred * target).sum(dim=(-2,-1))  # sum over spatial + channel if multi-channel
# 		# pred_sum = (pred ** 2).sum(dim=(-2,-1))
# 		# target_sum = (target ** 2).sum(dim=(-2,-1))

# 		## Assume shape is n_srcs x n_associations (avg. over association axis)
# 		if pred.shape[1] > 1:

# 			intersection = (pred * target).mean(dim=(-1))  # sum over spatial + channel if multi-channel
# 			pred_sum = (pred ** 2).mean(dim=(-1))
# 			target_sum = (target ** 2).mean(dim=(-1))

# 			dice = 1 - ((2.0 * intersection + self.smooth) /
# 	                    (pred_sum + self.bg_weight * target_sum + self.smooth))

# 			return dice.mean()

# 		## Assume shape is n_src_queries
# 		else:

# 			intersection = (pred * target).sum()  # sum over spatial + channel if multi-channel
# 			pred_sum = (pred ** 2).sum()
# 			target_sum = (target ** 2).sum()

# 			dice = 1 - ((2.0 * intersection + self.smooth) /
# 	                    (pred_sum + self.bg_weight * target_sum + self.smooth))

# 			return dice # .mean()			

## Replacing row wise sum with mean
class GaussianDiceLoss(nn.Module):

	# Start with bg_weight = 1.0
	# If too many false positives → increase bg_weight to 1.5–3.0
	# If missing weak Gaussians → decrease bg_weight to 0.5–0.8

	def __init__(self, smooth=1e-5, bg_weight=1.0):
		super().__init__()
		self.smooth = smooth
		self.bg_weight = bg_weight   # usually 1.0, sometimes 0.5–2.0

	def forward(self, pred, target):
		# No sigmoid! pred is raw linear output
		pred = pred.float()
		target = target.float()

		# intersection = (pred * target).sum(dim=(-2,-1))  # sum over spatial + channel if multi-channel
		# pred_sum = (pred ** 2).sum(dim=(-2,-1))
		# target_sum = (target ** 2).sum(dim=(-2,-1))

		# ## Assume shape is n_srcs x n_associations (avg. over association axis)
		# if pred.shape[1] > 1:

		# 	intersection = (pred * target).mean(dim=(-1))  # sum over spatial + channel if multi-channel
		# 	pred_sum = (pred ** 2).mean(dim=(-1))
		# 	target_sum = (target ** 2).mean(dim=(-1))

		# 	dice = 1 - ((2.0 * intersection + self.smooth) /
	    #                 (pred_sum + self.bg_weight * target_sum + self.smooth))

		# 	return dice.mean()

		## Assume shape is n_src_queries
		# else:

		intersection = (pred * target).sum()/pred.shape[1]  # sum over spatial + channel if multi-channel
		pred_sum = (pred ** 2).sum()/pred.shape[1]
		target_sum = (target ** 2).sum()/pred.shape[1]

		dice = 1 - ((2.0 * intersection + self.smooth) /
                    (pred_sum + self.bg_weight * target_sum + self.smooth))

		return dice # .mean()


class GaussianDiceLoss1(nn.Module):
	def __init__(self, smooth=1e-5, bg_weight=1.0):
		super().__init__()
		self.smooth = smooth
		self.bg_weight = bg_weight

	def forward(self, pred, target):
		# pred, target: (L, K) or (L, 1)
		# Squeeze the dummy channel if present
		if pred.ndim == 3 and pred.shape[-1] == 1:
			pred = pred.squeeze(-1)      # (L,)
			target = target.squeeze(-1)  # (L,)

		# Now shape is either (L,) → treat as (L,1) or (L,K)
		if pred.ndim == 1:
			pred = pred.unsqueeze(-1)    # (L,1)
			target = target.unsqueeze(-1)

		# Critical: mean/sum over spatial dimension only (dim=0)
		# This makes it invariant to different grid resolutions L
		intersection = (pred * target).mean(dim=0)   # (K,)
		pred_sum     = (pred ** 2).mean(dim=0)       # (K,)
		target_sum   = (target ** 2).mean(dim=0)     # (K,)

		numerator   = 2.0 * intersection + self.smooth
		denominator = pred_sum + self.bg_weight * target_sum + self.smooth

		per_station_dice = 1.0 - numerator / denominator    # (K,)
		return per_station_dice.mean()


class GradientNormBalancer:
	def __init__(self, losses_names, initial_loss=None, target_norm=1.0):
		self.names = losses_names
		self.target_norm = target_norm
		self.scales = {name: 1.0 for name in losses_names}       # will auto-update
		self.initial_loss = initial_loss or {}                   # optional warm-up values
        	
	def update_scales(self, loss_dict):
		for name in self.names:
			loss = loss_dict[name]
			if loss.requires_grad:
				grad_norm = torch.norm(torch.mean(torch.abs(loss.grad)), p=2) if loss.grad is not None else 0
				# Or simpler and more common:
				grad_norm = torch.norm(loss * torch.ones_like(loss), p=2).detach()

				target = self.target_norm
				if name in self.initial_loss:
					target *= self.initial_loss[name] / loss.item()   # optional homoscedastic boost
                
				self.scales[name] = self.scales[name] * (target / (grad_norm + 1e-8)).detach()
				self.scales[name] = torch.clamp(self.scales[name], 0.01, 100.0)  # stability

	def __call__(self, loss_dict):
		balanced = 0.0
		for name, loss in loss_dict.items():
			balanced += loss * self.scales[name]
		return balanced


class LossMagnitudeBalancer:
	def __init__(self, anchor='dice', alpha = 0.98):
		self.anchor = anchor
		self.values = {}
		self.scales = {}
		self.alpha = alpha

	def update(self, losses_dict):
		for k, v in losses_dict.items():
			v = v.detach().mean()
			if k not in self.values:
				self.values[k] = v
				self.scales[k] = 1.0
			else:
				# EMA of loss magnitude
				self.values[k] = self.alpha * self.values[k] + (1.0 - self.alpha) * v

		anchor_val = self.values[self.anchor]
		for k in losses_dict:
			self.scales[k] = (anchor_val / (self.values[k] + 1e-8)).clamp(0.1, 10.0)

	def __call__(self, losses_dict):
		self.update(losses_dict)
		total = sum(losses_dict[k] * self.scales[k] for k in losses_dict)
		return total


# class LossAccumulationBalancer:

#     def __init__(self, accum_steps, anchor_head = 'dice', alpha=0.98):
#         self.accum_steps = accum_steps
#         self.alpha = alpha
#         self.values = {}
#         self.anchor = anchor_head

#     def __call__(self, losses_dict):

#         total = 0.0
#         anchor_val = None

#         for name, loss in losses_dict.items():
#             val = loss.detach().mean().item()
#             if name not in self.values:
#                 self.values[name] = val * self.accum_steps   # bootstrap
#             else:
#                 self.values[name] = (self.alpha * self.values[name] + 
#                                    (1 - self.alpha) * val * self.accum_steps)
#             if name == self.anchor:
#                 anchor_val = self.values[name]

#         if anchor_val is None:
#             anchor_val = next(iter(self.values.values()))

#         for name, loss in losses_dict.items():
#             scale = anchor_val / (self.values[name] + 1e-8)
#             total += loss * scale.clamp(0.1, 100.0)

#         return total

# class LossAccumulationTwoTierBalancer:
# 	def __init__(self, accum_steps=8, alpha=0.99):
# 		self.accum_steps = accum_steps
# 		self.alpha = alpha
        
# 		self.primary_ema = {}    # only the 4 dice_* losses
# 		self.aux_ema     = {}    # consistency, hardneg, temporal, etc.
        
# 		self.primary_target = 0.15   # what one primary head should contribute
# 		self.aux_target     = 0.018  # ≈ 0.12× primary (tune this once: 0.01–0.03 range)

# 	def __call__(self, losses_dict):
# 		total = 0.0

# 		# === 1. Update EMAs ===
# 		for name, loss in losses_dict.items():
# 			val = loss.detach().mean().item() * self.accum_steps   # full-batch equivalent

# 			if name.startswith('dice_'):
# 				if name not in self.primary_ema:
# 					self.primary_ema[name] = val
# 				else:
# 					self.primary_ema[name] = self.alpha * self.primary_ema[name] + (1-self.alpha) * val

# 			else:  # auxiliary
# 				if name not in self.aux_ema:
# 					self.aux_ema[name] = val
# 				else:
# 					self.aux_ema[name] = self.alpha * self.aux_ema[name] + (1-self.alpha) * val

# 		# === 2. Compute scales ===
# 		# Primary: balance against each other → rare head gets high scale
# 		primary_anchor = max(self.primary_ema.values())  # or your known hardest head
# 		for name, loss in losses_dict.items():
# 			if name.startswith('dice_'):
# 				ema_val = self.primary_ema[name]
# 				scale = self.primary_target * len(self.primary_ema) / (ema_val + 1e-8)
# 				# equivalent to: primary_anchor / ema_val * (primary_target/primary_anchor)*4
# 			else:
# 				ema_val = self.aux_ema[name]
# 				scale = self.aux_target / (ema_val + 1e-8)

# 			scale = scale.clamp(0.05, 500.0)
# 			total += loss * scale

# 		return total

class LossAccumulationBalancer1: # TwoTier

	def __init__(self, anchor = 'loss_dice2', accum_steps = 10, aux_target = 0.018, alpha = 0.98, primary_ext = 'loss_dice', device = device):

		self.accum_steps = accum_steps
		self.alpha = alpha

		self.primary_ema = {}   # only dice_0, dice_1, dice_2, dice_3
		self.aux_ema     = {}   # consistency, hardneg_*, etc.
		self.primary_ext = primary_ext

		self.device = device
		self.anchor_head = anchor         # ← your rarest / hardest head
		self.aux_target  = aux_target             # total aux contribution ≈ 0.018
		                                     # (0.10–0.15× one primary head)

	def __call__(self, losses_dict):

		total = 0.0
		anchor_val = None

		# 1. Update EMAs (full-batch equivalent)
		for name, loss in losses_dict.items():
			val = loss.detach().mean().item() * self.accum_steps

			if name.startswith(self.primary_ext):
				if name not in self.primary_ema:
					self.primary_ema[name] = val
				else:
					self.primary_ema[name] = self.alpha * self.primary_ema[name] + (1-self.alpha) * val
				if name == self.anchor_head:
					anchor_val = self.primary_ema[name]
				    
			else:  # auxiliary
				if name not in self.aux_ema:
					self.aux_ema[name] = val
				else:
					self.aux_ema[name] = self.alpha * self.aux_ema[name] + (1-self.alpha) * val

		# If anchor not seen yet, fall back
		if anchor_val is None:
			anchor_val = max(self.primary_ema.values(), default=0.1)

		# 2. Apply scales
		for name, loss in losses_dict.items():

			if name.startswith(self.primary_ext):
				# Anchor-based: rarest head gets scale ≈ 1.0
				ema_val = self.primary_ema[name]
				scale = torch.tensor(anchor_val / (ema_val + 1e-8), device = self.device)
				scale = scale.clamp(0.1, 300.0)

			else:

				# Target-based: all aux together contribute ~aux_target
				ema_val = self.aux_ema[name]
				scale = torch.tensor(self.aux_target / (len(self.aux_ema) * (ema_val + 1e-8)), device = self.device)
				# or: scale = self.aux_target / (ema_val + 1e-8) if you have one global aux
				scale = scale.clamp(0.01, 50.0) # scale = 0.06 * anchor_val / (ema_val + 1e-8)   # aux ≈ 6% of anchor head

			total += loss * scale

		return total
	
	def state_dict(self):
		return {
			'accum_steps': self.accum_steps,
			'alpha': self.alpha,
			'primary_ema': self.primary_ema,
			'aux_ema': self.aux_ema,
			'primary_ext': self.primary_ext,
			'anchor_head': self.anchor_head,
			'aux_target': self.aux_target,
			# device is not saved – will be set on load
		}

	def load_state_dict(self, state_dict, device='cpu'):
		self.accum_steps = state_dict['accum_steps']
		self.alpha = state_dict['alpha']
		self.primary_ema = state_dict['primary_ema']
		self.aux_ema = state_dict['aux_ema']
		self.primary_ext = state_dict['primary_ext']
		self.anchor_head = state_dict['anchor_head']
		self.aux_target = state_dict['aux_target']
		self.device = device  # update device on load



# class LossAccumulationBalancer:
#     def __init__(
#         self,
#         anchor: str = 'loss_dice4',
#         aux_target: float = 0.018,
#         alpha: float = 0.98,
#         primary_ext: str = 'loss_dice',
#         device: str = 'cuda'
#     ):
#         self.anchor = anchor
#         self.aux_target = aux_target
#         self.alpha = alpha
#         self.primary_ext = primary_ext
#         self.device = device

#         self.primary_ema = {}
#         self.aux_ema = {}

#         # === Accumulation state ===
#         self._accum_prim = {}   # temporary sum over accum_steps
#         self._accum_aux  = {}
#         self._step_count = 0
#         self.accum_steps = None   # will be set on first call

#     def __call__(self, losses_dict: dict, accum_steps: int = None, is_last_accum_step: bool = False):
#         """
#         Call this on every microbatch.
#         Set is_last_accum_step=True on the final accum step (before optimizer.step()).
#         """
#         if accum_steps is not None:
#             self.accum_steps = accum_steps

#         total_loss = 0.0

#         # --------------------------------------------------
#         # 1. Accumulate raw loss values (microbatch → full batch)
#         # --------------------------------------------------
#         for name, loss in losses_dict.items():
#             val = loss.detach().mean()   # keep as tensor for now

#             if name.startswith(self.primary_ext):
#                 if name not in self._accum_prim:
#                     self._accum_prim[name] = val
#                 else:
#                     self._accum_prim[name] = self._accum_prim[name] + val
#             else:
#                 if name not in self._accum_aux:
#                     self._accum_aux[name] = val
#                 else:
#                     self._accum_aux[name] = self._accum_aux[name] + val

#         self._step_count += 1

#         # --------------------------------------------------
#         # 2. On the last accum step → update EMA with full-batch stats
#         # --------------------------------------------------
#         if is_last_accum_step or (self._step_count == self.accum_steps):
#             # Average over accum_steps to get true batch magnitude
#             anchor_ema = None

#             for name, accum_val in self._accum_prim.items():
#                 batch_val = (accum_val / self.accum_steps).item()

#                 if name not in self.primary_ema:
#                     self.primary_ema[name] = batch_val
#                 else:
#                     self.primary_ema[name] = (
#                         self.alpha * self.primary_ema[name] +
#                         (1 - self.alpha) * batch_val
#                     )
#                 if name == self.anchor:
#                     anchor_ema = self.primary_ema[name]

#             for name, accum_val in self._accum_aux.items():
#                 batch_val = (accum_val / self.accum_steps).item()
#                 if name not in self.aux_ema:
#                     self.aux_ema[name] = batch_val
#                 else:
#                     self.aux_ema[name] = (
#                         self.alpha * self.aux_ema[name] +
#                         (1 - self.alpha) * batch_val
#                     )

#             # Reset accumulators
#             self._accum_prim.clear()
#             self._accum_aux.clear()
#             self._step_count = 0

#             # Fallback anchor
#             if anchor_ema is None:
#                 anchor_ema = max(self.primary_ema.values(), default=0.1)

#         # --------------------------------------------------
#         # 3. Apply scaling (using latest EMA, even mid-accumulation)
#         # --------------------------------------------------
#         # We still scale every microbatch — that's fine and necessary
#         for name, loss in losses_dict.items():
#             if name.startswith(self.primary_ext):
#                 ema = self.primary_ema.get(name, 1.0)  # fallback early
#                 scale = anchor_ema / (ema + 1e-8)
#                 scale = scale.clamp(0.1, 300.0)
#             else:
#                 ema = self.aux_ema.get(name, 1.0)
#                 scale = self.aux_target / (len(self.aux_ema) * (ema + 1e-8) if self.aux_ema else 1.0)
#                 scale = scale.clamp(0.01, 50.0)

#             total_loss += scale * loss

#         return total_loss


class LossAccumulationBalancer:
    def __init__(
        self,
        anchor: str = 'loss_dice4',
        aux_target: float = 0.05,
        alpha: float = 0.98,
        primary_ext: str = 'loss_dice',
        device: str = 'cuda'
    ):
        # aux_target: float = 0.018,

        self.anchor = anchor
        self.aux_target = aux_target
        self.alpha = alpha
        self.primary_ext = primary_ext
        self.device = device

        self.primary_ema = {}
        self.aux_ema = {}

        # self.register_buffer('_anchor_ema_current', torch.tensor(0.0))

        # === Persistent state ===
        self._anchor_ema_current = None      # ← this holds the latest known value
        self._accum_prim = {}
        self._accum_aux  = {}
        self._step_count = 0
        self.accum_steps = None

    def __call__(self, losses_dict: dict, accum_steps: int = None, is_last_accum_step: bool = False):
        if accum_steps is not None:
            self.accum_steps = accum_steps

        total_loss = 0.0

        # --------------------------------------------------
        # 1. Accumulate microbatch statistics
        # --------------------------------------------------
        for name, loss in losses_dict.items():
            val = loss.detach().mean()

            if name.startswith(self.primary_ext):
                self._accum_prim[name] = self._accum_prim.get(name, 0.0) + val
            else:
                self._accum_aux[name] = self._accum_aux.get(name, 0.0) + val

        self._step_count += 1

        # --------------------------------------------------
        # 2. On last accum step → update EMA with full-batch stats
        # --------------------------------------------------
        updated = False
        if is_last_accum_step or (self.accum_steps is not None and self._step_count >= self.accum_steps):
            updated = True

            # Compute full-batch averages
            denom = self.accum_steps or self._step_count
            anchor_ema_new = None

            for name, accum_val in self._accum_prim.items():
                batch_val = (accum_val / denom).item()

                if name not in self.primary_ema:
                    self.primary_ema[name] = batch_val
                else:
                    self.primary_ema[name] = (
                        self.alpha * self.primary_ema[name] +
                        (1 - self.alpha) * batch_val
                    )
                if name == self.anchor:
                    anchor_ema_new = self.primary_ema[name]

            for name, accum_val in self._accum_aux.items():
                batch_val = (accum_val / denom).item()
                if name not in self.aux_ema:
                    self.aux_ema[name] = batch_val
                else:
                    self.aux_ema[name] = (
                        self.alpha * self.aux_ema[name] +
                        (1 - self.alpha) * batch_val
                    )

            # Update the persistent anchor value
            if anchor_ema_new is not None:
                self._anchor_ema_current = anchor_ema_new
            elif self._anchor_ema_current is None and self.primary_ema:
                # Fallback: use max of existing primary EMAs
                self._anchor_ema_current = max(self.primary_ema.values())

            # Reset accumulators
            self._accum_prim.clear()
            self._accum_aux.clear()
            self._step_count = 0

        # --------------------------------------------------
        # 3. Scale losses using the latest known anchor_ema
        # --------------------------------------------------
        # Safe fallback chain:
        if self._anchor_ema_current is None:
            # First forward pass ever → assume reasonable default
            anchor_val = 0.5
        else:
            anchor_val = self._anchor_ema_current

        for name, loss in losses_dict.items():
            if name.startswith(self.primary_ext):
                ema = self.primary_ema.get(name, 1.0)
                scale = torch.tensor(anchor_val / (ema + 1e-8), device = self.device)
                scale = scale.clamp(0.1, 300.0)
            else:
                ema = self.aux_ema.get(name, 1.0)
                n_aux = max(len(self.aux_ema), 1)
                scale = torch.tensor(self.aux_target / (n_aux * (ema + 1e-8)), device = self.device)
                scale = scale.clamp(0.01, 50.0)

            total_loss += scale * loss

        return total_loss

    def state_dict(self):
        return {
            'anchor': self.anchor,
            'aux_target': self.aux_target,
            'alpha': self.alpha,
            'primary_ext': self.primary_ext,
            'accum_steps': self.accum_steps,
            'primary_ema': self.primary_ema,
            'aux_ema': self.aux_ema,
            '_anchor_ema_current': self._anchor_ema_current,
        }

    def load_state_dict(self, state_dict, device=None):
        self.anchor = state_dict['anchor']
        self.aux_target = state_dict['aux_target']
        self.alpha = state_dict['alpha']
        self.primary_ext = state_dict['primary_ext']
        self.accum_steps = state_dict.get('accum_steps', None)

        self.primary_ema = state_dict['primary_ema']
        self.aux_ema = state_dict['aux_ema']
        self._anchor_ema_current = state_dict.get('_anchor_ema_current', None)

        if device is not None:
            self.device = device

        # Reset transient accumulators (always!)
        self._accum_prim = {}
        self._accum_aux = {}
        self._step_count = 0

        # self.anchor = anchor
        # self.aux_target = aux_target
        # self.alpha = alpha
        # self.primary_ext = primary_ext
        # self.device = device

        # self.primary_ema = {}
        # self.aux_ema = {}

        # # === Persistent state ===
        # self._anchor_ema_current = None      # ← this holds the latest known value
        # self._accum_prim = {}
        # self._accum_aux  = {}
        # self._step_count = 0
        # self.accum_steps = None


class UncertaintyBalancer(nn.Module):
	def __init__(self, num_tasks):
		super().__init__()
		self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # log σ_i²

	def forward(self, losses):  # losses: dict of task losses
		total = 0
		for i, (k, loss) in enumerate(losses.items()):
			precision = torch.exp(-self.log_vars[i])
			total += precision * loss + self.log_vars[i]
		return total / len(losses)
# Usage: balancer = UncertaintyBalancer(4); total = balancer(losses)


if use_station_corrections == True:
	n_ver_corrections = 1
	path_station_corrections = path_to_file + 'Grids' + seperator + 'station_corrections_ver_%d.npz'%n_ver_corrections
	if os.path.isfile(path_station_corrections) == False:
		print('No station corrections available')
		locs_corr, corrs = None, None
	else:
		z = np.load(path_station_corrections)
		locs_corr, corrs = z['locs_corr'], z['corrs']
		z.close()
else:
	locs_corr, corrs = None, None


## Replacing mse loss with huber loss (delta 0.3), moving loss_dice1 to auxilary loss
## masking out all zeros in negative loss. Up weighting auxilary losses to 0.05.
## Decreasing up scaling factor for negative loss to 1000. Increase pre_scale_weights1
## to 30 (e.g., why is negative up scaled more than base loss?)
## Adding cap loss


	
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
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise1 = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)

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
	
## Make supplemental information for grids
x_grids_trv = []
x_grids_trv_pointers_p = []
x_grids_trv_pointers_s = []
x_grids_trv_refs = []
# x_grids_edges = []

if config['train_travel_time_neural_network'] == False:
	ts_max_val = Ts.max()

x_grids_trv = compute_travel_times(trv, locs, x_grids, device = device)

if use_time_shift == True:
	for i in range(len(x_grids_trv)):
		x_grids_trv[i] = x_grids_trv[i] + time_shifts[i].reshape(-1,1,1)
	print('Appending time shifts')


time_shift_range = np.max([time_shifts[j].max() - time_shifts[j].min() for j in range(len(time_shifts))])

max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))])))
min_t = float(np.floor(min([x_grids_trv[i].min() for i in range(len(x_grids_trv))]))) if use_time_shift == True else 0.0

# for i in range(len(x_grids)):

# 	if locs.shape[0]*x_grids[i].shape[0] > 150e3:
# 		trv_out_l = []
# 		for j in range(locs.shape[0]):
# 			trv_out = trv(torch.Tensor(locs[j,:].reshape(1,-1)).to(device), torch.Tensor(x_grids[i]).to(device))
# 			trv_out_l.append(trv_out.cpu().detach().numpy())
# 		trv_out = torch.Tensor(np.concatenate(trv_out_l, axis = 1)).to(device)
# 	else:
# 		trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
	
# 	# trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
# 	x_grids_trv.append(trv_out.cpu().detach().numpy())

for i in range(len(x_grids)):
	
	## Note, this definition of dt and win must match the definition used in process_continous_days
	A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(x_grids_trv[i], k = k_time_edges, max_t = max_t, min_t = min_t, dt = kernel_sig_t/5.0, win = kernel_sig_t*2.0)

	if config['train_travel_time_neural_network'] == False:
		assert(x_grids_trv[i].min() > 0.0)
		assert(x_grids_trv[i].max() < (ts_max_val + 3.0))

	x_grids_trv_pointers_p.append(A_edges_time_p)
	x_grids_trv_pointers_s.append(A_edges_time_s)
	x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?

	# edge_index = knn(torch.Tensor(ftrns1(x_grids[i])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[i])/1000.0).to(device), k = k_spc_edges).flip(0).contiguous()
	# edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
	# x_grids_edges.append(edge_index)

## Check if this can cause an issue (can increase max_t to a bit larger than needed value)
# max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # + 10.0
# min_t = float(np.floor(min([x_grids_trv[i].min() for i in range(len(x_grids_trv))]))) if use_time_shift == True else 0.0 # + 10.0

## Implement training.
mz = GCN_Detection_Network_extended(ftrns1_diff, ftrns2_diff, trv = trv, device = device).to(device)
optimizer = optim.Adam(mz.parameters(), lr = 0.001)


# bce_loss = BCEWithLogitsLoss()
# focal_loss = SoftFocalLoss() ## Try using this on main lmse_lossoss targets
mse_loss = torch.nn.MSELoss()
# loss_func_mse = torch.nn.MSELoss()
huber_loss = torch.nn.HuberLoss(delta = 0.5, reduction = 'mean') ## Beneath delta, L2 loss is applied
l1_loss = torch.nn.L1Loss()
DiceLoss = GaussianDiceLoss(bg_weight = 1.0) ## Can change the bg_weight


# loss_names = ['loss_dice1', 'loss_dice2', 'loss_dice3', 'loss_dice4', 'loss_negative', 'loss_consistency', 'loss_gradient', 'loss_global']


np.random.seed() ## randomize seed

losses = np.zeros(n_epochs)
mx_trgt_1, mx_trgt_2, mx_trgt_3, mx_trgt_4 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
mx_pred_1, mx_pred_2, mx_pred_3, mx_pred_4 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)

weights = torch.Tensor([0.1, 0.4, 0.25, 0.25]).to(device)

lat_range_interior = [lat_range[0], lat_range[1]]
lon_range_interior = [lon_range[0], lon_range[1]]

cnt_plot = 0

n_restart = train_config['restart_training']
n_restart_step = train_config['n_restart_step']
if n_restart == False:
	n_restart_step = 0 # overwrite to 0, if restart is off

if load_training_data == True:

	files_load = glob.glob(path_to_data + '*ver_%d.hdf5'%n_ver_training_data)
	print('Number of found training files %d'%len(files_load))
	if build_training_data == False:
		assert(len(files_load) > 0)


use_dice_loss = True
use_mse_loss = False
use_negative_loss = True
# use_global_loss = False
use_consistency_loss = True
use_gradient_loss = False
use_cap_loss = True
use_huber_loss = True


# n_burn_in = int(0.5*n_epochs/5)

n_burn_in = int(1*n_epochs/5)
loss_names = ['loss_dice1', 'loss_dice2', 'loss_dice3', 'loss_dice4', 'loss_negative', 'loss_consistency']

# LossBalancer = LossMagnitudeBalancer(anchor = 'loss_dice4') ## Use sparsest target as the anchor (could also use loss_dice2)
# LossBalancer = LossMagnitudeBalancer(anchor = 'loss_dice2') ## Use sparsest target as the anchor (could also use loss_dice2)
# LossBalancer = LossAccumulationBalancer(anchor = 'loss_dice2', accum_steps = n_batch, primary_ext = 'loss_dice', device = device) ## Use sparsest target as the anchor (could also use loss_dice2)
# balancer = LossAccumulationBalancer(anchor = 'loss_dice2', accum_steps = n_batch, primary_ext = 'loss_dice', device = device) ## Use sparsest target as the anchor (could also use loss_dice2)
# LossBalancer = LossAccumulationBalancer(anchor = 'loss_dice4', primary_ext = 'loss_dice', device = device) ## Use sparsest target as the anchor (could also use loss_dice2)
LossBalancer = LossAccumulationBalancer(anchor = 'loss_dice3', primary_ext = 'loss_dice', device = device) ## Use sparsest target as the anchor (could also use loss_dice2)

## Need to add hard negatives to association outputs

## May need to up-sample positive associations more (or broaden the label)
## and can add hard negatives to association labels



def create_training_inputs(trv, Inpts, Masks, Locs, X_fixed, A_src_in_sta_l, A_src_in_prod_l, A_prod_sta_sta_l, A_prod_src_src_l, lp_srcs, lp_times, lp_stations, lp_phases, lp_meta, device = device):

	## Should add increased samples in x_src_query around places of coherency
	## and true labels
	x_src_query = np.random.rand(n_src_query,3)*scale_x_extend + offset_x_extend


	## lp_srcs[i0]
	## Locs[i0]
	## X_fixed[i0]
	## A_src_in_sta_l[i0]
	## A_src_in_prod_l[i0]
	## Inpts[i0]
	## Masks[i0]
	## A_prod_sta_sta_l[i0]
	## A_prod_src_src_l[i0]
	## lp_times[i0]
	## lp_stations[i0]
	## lp_phases[i0]
	## lp_meta[i0]


	tq_sample = torch.rand(n_src_query).to(device)*t_win - t_win/2.0
	if use_time_shift == True:
		tq_sample = torch.Tensor(np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = n_src_query)).to(device)


	n_frac_focused_association_queries = 0.2 # concentrate 10% of association queries around true sources
	n_concentration_focused_association_queries = 0.03 # 3% of scale of domain
	if (len(lp_srcs) > 0)*(n_frac_focused_association_queries > 0):

		n_focused_queries = int(n_frac_focused_association_queries*n_src_query)
		ind_overwrite_focused_queries = np.sort(np.random.choice(n_src_query, size = n_focused_queries, replace = False))
		ind_source_focused = np.random.choice(len(lp_srcs), size = n_focused_queries)

		# x_query_focused = np.random.randn(n_focused_queries, 3)*scale_x_extend*n_concentration_focused_association_queries
		x_query_focused = 2.0*np.random.randn(n_focused_queries, 3)*np.mean([src_x_kernel, src_depth_kernel])
		x_query_focused = ftrns2(x_query_focused + ftrns1(lp_srcs[ind_source_focused,0:3]))
		ioutside = np.where(((x_query_focused[:,2] < depth_range[0]) + (x_query_focused[:,2] > depth_range[1])) > 0)[0]
		x_query_focused[ioutside,2] = np.random.rand(len(ioutside))*(depth_range[1] - depth_range[0]) + depth_range[0]
		
		x_query_focused = np.maximum(np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1), x_query_focused)
		x_query_focused = np.minimum(np.array([lat_range_extend[1], lon_range_extend[1], depth_range[1]]).reshape(1,-1), x_query_focused)
		x_src_query[ind_overwrite_focused_queries] = x_query_focused

		x_query_focused_t = 2.0*np.random.randn(n_focused_queries)*src_t_kernel			
		x_query_focused_t = lp_srcs[ind_source_focused,3] + x_query_focused_t
		# ioutside = np.where(((x_query_focused_t < min_t) + (x_query_focused_t > max_t)) > 0)[0]
		ioutside = np.where(((x_query_focused_t < (-time_shift_range/2.0)) + (x_query_focused_t > (time_shift_range/2.0))) > 0)[0]

		x_query_focused_t[ioutside] = np.random.uniform(-time_shift_range/2.0, time_shift_range/2.0, size = len(ioutside))
		tq_sample[ind_overwrite_focused_queries] = torch.Tensor(x_query_focused_t).to(device)


	if len(lp_srcs) > 0:
		x_src_query[0:len(lp_srcs),0:3] = lp_srcs[:,0:3]
	
	if len(lp_srcs) > 0:
		ifind_src = np.where(np.abs(lp_srcs[:,3]) <= t_win/2.0)[0]
		tq_sample[ifind_src] = torch.Tensor(lp_srcs[ifind_src,3]).to(device)

	x_src_query_cart = ftrns1(x_src_query)
	
	trv_out_src = trv(torch.Tensor(Locs).to(device), torch.Tensor(x_src_query).to(device)).detach()

	if use_time_shift == True: 
		grid_match = np.argmin([np.linalg.norm(X_fixed - x_grids[j]) for j in range(len(x_grids))])
	
	if use_subgraph == True:
		if use_time_shift == False:
			# A_src_in_prod_l = Data(torch.Tensor(A_src_in_prod_x_l).to(device), edge_index = torch.Tensor(A_src_in_prod_edges_l).long().to(device))
			trv_out = trv_pairwise(torch.Tensor(Locs[A_src_in_sta_l[0].cpu().detach().numpy()]).to(device), torch.Tensor(X_fixed[A_src_in_sta_l[1].cpu().detach().numpy()]).to(device))
		else:
			trv_out = trv_pairwise(torch.Tensor(Locs[A_src_in_sta_l[0].cpu().detach().numpy()]).to(device), torch.Tensor(X_fixed[A_src_in_sta_l[1].cpu().detach().numpy()]).to(device)) + torch.Tensor(time_shifts[grid_match, A_src_in_sta_l[1].cpu().detach().numpy()]).reshape(-1,1).to(device)

		spatial_vals = torch.cat((torch.Tensor((X_fixed[A_src_in_prod_l[1].cpu().detach().numpy()][:,0:3] - Locs[A_src_in_sta_l[0][A_src_in_prod_l[0]].cpu().detach().numpy()])/scale_x_extend).to(device), torch.Tensor(X_fixed[A_src_in_prod_l[1].cpu().detach().numpy()][:,[3]]).to(device)/time_shift_range), dim = 1)

	else:

		if use_time_shift == False:
			trv_out = trv(torch.Tensor(Locs).to(device), torch.Tensor(X_fixed).to(device)).detach().reshape(-1,2) ## Note: could also just take this from x_grids_trv
		else:
			trv_out = (trv(torch.Tensor(Locs).to(device), torch.Tensor(X_fixed).to(device)).detach() + torch.Tensor(np.expand_dims(time_shift[[grid_match],:], axis = 0)).to(device)).reshape(-1,2) ## Note: could also just take this from x_grids_trv


		spatial_vals = torch.cat((torch.Tensor(((np.repeat(np.expand_dims(X_fixed[:,0:3], axis = 1), Locs.shape[0], axis = 1) - np.repeat(np.expand_dims(Locs, axis = 0), X_fixed.shape[0], axis = 0)).reshape(-1,3))/scale_x_extend).to(device), torch.Tensor(X_fixed[:,[3]]).to(device)/time_shift_range), dim = 1)



	tq = torch.arange(-t_win/2.0, t_win/2.0 + dt_win, dt_win).reshape(-1,1).float().to(device)
	if use_time_shift == True:
		tq = torch.arange(-time_shift_range/2.0, time_shift_range/2.0 + dt_win, dt_win).reshape(-1,1).float().to(device)


	if use_phase_types == False:
		Inpts[:,2::] = 0.0 ## Phase type informed features zeroed out
		Masks[:,2::] = 0.0


	# Pre-process tensors for Inpts and Masks
	input_tensor_1 = torch.Tensor(Inpts).to(device) # .reshape(-1, 4)
	input_tensor_2 = torch.Tensor(Masks).to(device) # .reshape(-1, 4)

	# Process tensors for A_prod and A_src arrays
	A_prod_sta_tensor = A_prod_sta_sta_l
	A_prod_src_tensor = A_prod_src_src_l

	# Process edge index data
	edge_index_1 = A_src_in_prod_l
	flipped_edge = np.ascontiguousarray(np.flip(A_src_in_prod_l.cpu().detach().numpy(), axis = 0))
	edge_index_2 = torch.Tensor(flipped_edge).long().to(device)

	data_1 = Data(x=spatial_vals, edge_index=edge_index_1)
	data_2 = Data(x=spatial_vals, edge_index=edge_index_2)

	use_updated_pick_max_associations = True # Changing to True
	if (len(lp_times) > max_number_pick_association_labels_per_sample)*(use_updated_pick_max_associations == True):

		## Cnt number of picks per station
		## Optimally choose n stations to compute association labels for
		## so that sum cnt_i < max_mumber of picks used. 
		## Permute station indices to not bias ordering.
		## Keep all picks for this set of stations.
		## (note: this does not effect output values, only the ones we compute losses on)

		tree_sta_slice = cKDTree(lp_stations.reshape(-1,1))
		lp_cnt_sta = tree_sta_slice.query_ball_point(np.arange(Locs.shape[0]).reshape(-1,1), r = 0)
		cnt_lp_sta = np.array([len(lp_cnt_sta[j]) for j in range(len(lp_cnt_sta))])

		# Maximize the number of associations. Permute
		sta_grab = optimize_station_selection(cnt_lp_sta, max_number_pick_association_labels_per_sample)
		isample_picks = np.hstack([lp_cnt_sta[j] for j in sta_grab])

		lp_times = lp_times[isample_picks]
		lp_stations = lp_stations[isample_picks]
		lp_phases = lp_phases[isample_picks]
		lp_meta = lp_meta[isample_picks]

		assert(len(lp_times) <= max_number_pick_association_labels_per_sample)
	
	elif len(lp_times) > max_number_pick_association_labels_per_sample:

		## Randomly choose max number of picks to compute association labels for.
		isample_picks = np.sort(np.random.choice(len(lp_times), size = max_number_pick_association_labels_per_sample, replace = False))
		lp_times = lp_times[isample_picks]
		lp_stations = lp_stations[isample_picks]
		lp_phases = lp_phases[isample_picks]
		lp_meta = lp_meta[isample_picks]


	return x_src_query, tq_sample, x_src_query_cart, trv_out, trv_out_src, spatial_vals, tq, input_tensor_1, input_tensor_2, A_prod_sta_tensor, A_prod_src_tensor, data_1, data_2, lp_times, lp_stations, lp_phases, lp_meta




if build_training_data == True:

	## If true, use this script to build the training data.
	## For efficiency, each instance of this script (e.g., python train_GENIE_model.py $i$ for different integer $i$ calls)
	## will build train_config['n_batches_per_job_training_data'] batches of training data and save them to train_config['path_to_data'].
	## Call this script ~100's of times to build a large training dataset. Then set flag "build_training_data : False" in train_config.yaml
	## and call this script with "load_training_data : True" to train with the available pre-built training data.

	## If false, this script begins training the model, and builds a batch of data on the fly between each update step.

	n_repeat = train_config['n_batches_per_job'] ## Number of batches to make per job

	argvs = sys.argv
	if len(argvs) < 2:
		argvs.append(0)

	job_number = int(argvs[1]) ## Choose job index

	print('Build and save training data on job index %d'%job_number)

	for n in range(n_repeat):

		file_index = n_repeat*job_number + n ## Unique file index

		if use_subgraph == True:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = True)
			if use_expanded == True:
				Ac_src_src_l = [[] for j in range(len(A_src_src_l))]
				Ac_prod_src_src_l = [[] for j in range(len(A_prod_src_src_l))]

		else:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = False)
			if use_expanded == True:
				A_src_src_l, Ac_src_src_l = A_src_src_l
				A_prod_src_src_l, Ac_prod_src_src_l = A_prod_src_src_l

			Inpts = [Inpts[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Inpts))]
			Masks = [Masks[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Masks))]
			A_sta_sta_l = [torch.Tensor(A_sta_sta_l[j]).long().to(device) for j in range(len(A_sta_sta_l))]
			A_src_src_l = [torch.Tensor(A_src_src_l[j]).long().to(device) for j in range(len(A_src_src_l))]
			A_prod_sta_sta_l = [torch.Tensor(A_prod_sta_sta_l[j]).long().to(device) for j in range(len(A_prod_sta_sta_l))]
			A_prod_src_src_l = [torch.Tensor(A_prod_src_src_l[j]).long().to(device) for j in range(len(A_prod_src_src_l))]
			A_src_in_prod_l = [torch.Tensor(A_src_in_prod_l[j]).long().to(device) for j in range(len(A_src_in_prod_l))]
			if use_expanded == True:
				Ac_src_src_l = [torch.Tensor(Ac_src_src_l[j]).long().to(device) for j in range(len(Ac_src_src_l))]
				Ac_prod_src_src_l = [torch.Tensor(Ac_prod_src_src_l[j]).long().to(device) for j in range(len(Ac_prod_src_src_l))]


		h = h5py.File(path_to_data + 'training_data_slice_%d_ver_%d.hdf5'%(file_index, n_ver_training_data), 'w')
		h['data'] = data[0]
		h['srcs'] = data[1]
		h['srcs_active'] = data[2]
		A_src_in_sta_l = [[] for i in range(n_batch)]

		for i in range(n_batch):

			if use_subgraph == True:
				A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[i], X_fixed[i], ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, Ac = Ac, device = device)
				A_edges_time_p, A_edges_time_s, dt_partition = compute_time_embedding_vectors(trv_pairwise, Locs[i], X_fixed[i], A_src_in_sta, max_t, min_t = min_t, time_shift = X_fixed[i][:,3], dt_res = kernel_sig_t/5.0, t_win = kernel_sig_t*2.0, device = device)
				if use_expanded == True:
					A_src_src, Ac_src_src = A_src_src
					A_prod_src_src, Ac_prod_src_src = A_prod_src_src

				A_sta_sta_l[i] = A_sta_sta ## These should be equal
				A_src_src_l[i] = A_src_src ## These should be equal
				A_prod_sta_sta_l[i] = A_prod_sta_sta
				A_prod_src_src_l[i] = A_prod_src_src
				A_src_in_prod_l[i] = A_src_in_prod
				A_edges_time_p_l[i] = A_edges_time_p
				A_edges_time_s_l[i] = A_edges_time_s
				A_edges_ref_l[i] = dt_partition
				A_src_in_sta_l[i] = A_src_in_sta
				Inpts[i] = np.copy(np.ascontiguousarray(Inpts[i][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))
				Masks[i] = np.copy(np.ascontiguousarray(Masks[i][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))			
				if use_expanded == True:
					Ac_src_src_l[i] = Ac_src_src
					Ac_prod_src_src_l[i] = Ac_prod_src_src

			if use_subgraph == False:
				# A_src_in_sta = torch.Tensor(np.concatenate((np.tile(np.arange(Locs[i].shape[0]), len(X_fixed[i])).reshape(1,-1), np.arange(len(X_fixed[i])).repeat(len(Locs[i]), axis = 0).reshape(1,-1)), axis = 0)).to(device).long()
				A_src_in_sta_l[i] = torch.Tensor(np.concatenate((np.tile(np.arange(Locs[i].shape[0]), len(X_fixed[i])).reshape(1,-1), np.arange(len(X_fixed[i])).repeat(len(Locs[i]), axis = 0).reshape(1,-1)), axis = 0)).to(device).long() # A_src_in_sta

			## Call extra inputs
			x_src_query, tq_sample, x_src_query_cart, trv_out, trv_out_src, spatial_vals, tq, input_tensor_1, input_tensor_2, A_prod_sta_tensor, A_prod_src_tensor, data_1, data_2, lp_times_slice, lp_stations_slice, lp_phases_slice, lp_meta_slice = create_training_inputs(trv, Inpts[i], Masks[i], Locs[i], X_fixed[i], A_src_in_sta_l[i], A_src_in_prod_l[i], A_prod_sta_sta_l[i], A_prod_src_src_l[i], lp_srcs[i], lp_times[i], lp_stations[i], lp_phases[i], lp_meta[i], device = device)

			# pdb.set_trace()

			pick_lbls = pick_labels_extract_interior_region_flattened(x_src_query_cart, tq_sample.cpu().detach().numpy(), lp_meta_slice[:,-2::], lp_srcs[i], lat_range_interior, lon_range_interior, ftrns1, sig_t = src_t_arv_kernel, sig_x = src_x_arv_kernel)

			## New inputs, or modified inputs
			# Inpts_slice
			# Masks_slice
			# x_src_query
			# tq_sample
			# x_src_query_cart
			# trv_out
			# trv_out_src
			# spatial_vals
			# tq
			# input_tensor_1
			# input_tensor_2
			# A_prod_sta_tensor
			# A_prod_src_tensor
			# data_1
			# data_2
			# lp_time_slice
			# lp_station_slice
			# lp_phases_slice
			# lp_met_slice
			# pick_lbls

			h['Inpts_%d'%i] = Inpts[i] # Inpts[i]
			h['Masks_%d'%i] = Masks[i] # Masks[i]
			h['X_fixed_%d'%i] = X_fixed[i]
			h['X_fixed_cart_%d'%i] = ftrns1(X_fixed[i])
			h['X_query_%d'%i] = X_query[i]
			h['X_query_cart_%d'%i] = ftrns1(X_query[i])
			h['Locs_%d'%i] = Locs[i]
			h['Locs_cart_%d'%i] = ftrns1(Locs[i])
			h['Trv_out_%d'%i] = Trv_out[i]

			if use_gradient_loss == False:
				h['Lbls_%d'%i] = Lbls[i]
				h['Lbls_query_%d'%i] = Lbls_query[i]
			else:
				h['Lbls_%d'%i] = Lbls[i][0]
				h['Lbls_query_%d'%i] = Lbls_query[i][0]
				h['Lbls_grad_spc_%d'%i] = Lbls[i][1]
				h['Lbls_query_grad_spc_%d'%i] = Lbls_query[i][1]
				h['Lbls_grad_t_%d'%i] = Lbls[i][2]
				h['Lbls_query_grad_t_%d'%i] = Lbls_query[i][2]

			h['lp_times_%d'%i] = lp_times_slice # [i]
			h['lp_stations_%d'%i] = lp_stations_slice # [i]
			h['lp_phases_%d'%i] = lp_phases_slice # [i]
			h['lp_meta_%d'%i] = lp_meta_slice # [i]
			h['lp_srcs_%d'%i] = lp_srcs[i]

			h['A_sta_sta_%d'%i] = A_sta_sta_l[i].cpu().detach().numpy()
			h['A_src_src_%d'%i] = A_src_src_l[i].cpu().detach().numpy()
			h['A_prod_sta_sta_%d'%i] = A_prod_sta_sta_l[i].cpu().detach().numpy()
			h['A_prod_src_src_%d'%i] = A_prod_src_src_l[i].cpu().detach().numpy()
			h['A_src_in_prod_%d'%i] = A_src_in_prod_l[i].cpu().detach().numpy()
			# h['A_src_in_prod_x_%d'%i] = A_src_in_prod_l[i].x
			# h['A_src_in_prod_edges_%d'%i] = A_src_in_prod_l[i].edge_index
			if use_subgraph == True:
				h['A_src_in_sta_%d'%i] = A_src_in_sta.cpu().detach().numpy()
			else:
				h['A_src_in_sta_%d'%i] = A_src_in_sta.cpu().detach().numpy() # np.concatenate((np.tile(np.arange(Locs[i].shape[0]), len(X_fixed[i])).reshape(1,-1), np.arange(len(X_fixed[i])).repeat(len(Locs[i]), axis = 0).reshape(1,-1)), axis = 0)

			h['A_edges_time_p_%d'%i] = A_edges_time_p_l[i]
			h['A_edges_time_s_%d'%i] = A_edges_time_s_l[i]
			h['A_edges_ref_%d'%i] = A_edges_ref_l[i]
			h['dt_partition_%d'%i] = dt_partition # _l[i]

			if use_expanded == True:
				h['Ac_src_src_%d'%i] = Ac_src_src_l[i].cpu().detach().numpy()
				h['Ac_prod_src_src_%d'%i] = Ac_prod_src_src_l[i].cpu().detach().numpy()


			## Save extra input parameters:
			h['x_src_query_%d'%i] = x_src_query
			h['x_src_query_cart_%d'%i] = x_src_query_cart
			h['tq_sample_%d'%i] = tq_sample.cpu().detach().numpy()
			h['trv_out_%d'%i] = trv_out.cpu().detach().numpy()
			h['trv_out_src_%d'%i] = trv_out_src.cpu().detach().numpy()
			h['spatial_vals_%d'%i] = spatial_vals.cpu().detach().numpy()
			h['tq_%d'%i] = tq.cpu().detach().numpy()
			h['input_tensor_1_%d'%i] = input_tensor_1.cpu().detach().numpy()
			h['input_tensor_2_%d'%i] = input_tensor_2.cpu().detach().numpy()
			h['A_prod_sta_tensor_%d'%i] = A_prod_sta_tensor.cpu().detach().numpy()
			h['A_prod_src_tensor_%d'%i] = A_prod_src_tensor.cpu().detach().numpy()
			h['data_1_edges_%d'%i] = data_1.edge_index.cpu().detach().numpy()
			h['data_2_edges_%d'%i] = data_2.edge_index.cpu().detach().numpy()
			h['pick_lbls_%d'%i] = pick_lbls.cpu().detach().numpy()

			# Inpts_slice
			# Masks_slice
			# x_src_query
			# tq_sample
			# x_src_query_cart
			# trv_out
			# trv_out_src
			# spatial_vals
			# tq
			# input_tensor_1
			# input_tensor_2
			# A_prod_sta_tensor
			# A_prod_src_tensor
			# data_1
			# data_2
			# lp_time_slice
			# lp_station_slice
			# lp_phases_slice
			# lp_met_slice

		h.close()

	print('Finished building training data for job %d'%job_number)

	print('Data set built; call the training script again once all data has been built')
	sys.exit()


class TrainingDataset(Dataset):

	def __init__(self, list_of_hdf5_paths, n_batch, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded):
		self.files = list_of_hdf5_paths   # e.g. 1_000_000 files
		self.n_batch = n_batch
		self.use_gradient_loss = use_gradient_loss
		self.use_expanded = use_expanded

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		path = self.files[idx]

		with h5py.File(path, 'r') as f:

			lp_srcs = []
			lp_times = []
			lp_stations = []
			lp_phases = []
			X_query = []
			Lbls = []
			Locs = []
			Lbls_query = []
			pick_lbls_l = []
			spatial_vals_l = []
			x_src_query_cart_l = []
			input_tensors_l = []

			if self.use_gradient_loss == True:
				Lbls_grad_spc = []
				Lbls_query_grad_spc = []
				Lbls_grad_t = []
				Lbls_query_grad_t = []

			for i in range(self.n_batch):  # or however many you have

				## Could send to device here if have enough ram (or can send to device using map across the list?)

				spatial_vals = torch.from_numpy(f['spatial_vals_%d'%i][:])

				lp_srcs.append(torch.from_numpy(f['lp_srcs_%d'%i][:]))
				lp_times.append(torch.from_numpy(f['lp_times_%d'%i][:]))
				lp_stations.append(torch.from_numpy(f['lp_stations_%d'%i][:]).long())
				lp_phases.append(torch.from_numpy(f['lp_phases_%d'%i][:]))
				spatial_vals_l.append(spatial_vals)

				X_query.append(torch.from_numpy(f['X_query_%d'%i][:]))
				x_src_query_cart_l.append(torch.from_numpy(f['x_src_query_cart_%d'%i][:]))

				Lbls.append(torch.from_numpy(f['Lbls_%d'%i][:]))
				Lbls_query.append(torch.from_numpy(f['Lbls_query_%d'%i][:]))
				pick_lbls_l.append(torch.from_numpy(f['pick_lbls_%d'%i][:]))
				Locs.append(torch.from_numpy(f['Locs_%d'%i][:]))
				if self.use_gradient_loss == True:
					Lbls_grad_spc.append(torch.from_numpy(f['Lbls_grad_spc_%d'%i][:]))
					Lbls_query_grad_spc.append(torch.from_numpy(f['Lbls_query_grad_spc_%d'%i][:]))
					Lbls_grad_t.append(torch.from_numpy(f['Lbls_grad_t_%d'%i][:]))
					Lbls_query_grad_t.append(torch.from_numpy(f['Lbls_query_grad_t_%d'%i][:]))

				## Make input list
				if self.use_expanded == False:
					A_prod_src_tensor = torch.from_numpy(f['A_prod_src_tensor_%d'%i][:]).long()
				else:
					A_prod_src_tensor = [torch.from_numpy(f['A_prod_src_tensor_%d'%i][:]).long(), torch.from_numpy(f['Ac_prod_src_src_%d'%i][:]).long()]

				# Continue processing the rest of the inputs
				input_tensors = [
					torch.from_numpy(f['input_tensor_1_%d'%i][:]), 
					torch.from_numpy(f['input_tensor_2_%d'%i][:]), 
					torch.from_numpy(f['A_prod_sta_tensor_%d'%i][:]).long(), 
					A_prod_src_tensor, # torch.from_numpy(f['A_prod_src_tensor_%d'%i]).long()
					torch.from_numpy(f['data_1_edges_%d'%i][:]).long(), 
					torch.from_numpy(f['data_2_edges_%d'%i][:]).long(),
					# Data(x = spatial_vals, edge_index = torch.from_numpy(f['data_1_edges_%d'%i][:]).long()), 
					# Data(x = spatial_vals, edge_index = torch.from_numpy(f['data_2_edges_%d'%i][:]).long()),
					torch.from_numpy(f['A_src_in_sta_%d'%i][:]).long(),
					torch.from_numpy(f['A_src_src_%d'%i][:]).long(),
					torch.zeros(1), # torch.Tensor(A_edges_time_p_l[i0]).long().to(device)
					torch.zeros(1), # torch.Tensor(A_edges_time_s_l[i0]).long().to(device)
					torch.zeros(1), # torch.Tensor(A_edges_ref_l[i0]).to(device)
					torch.from_numpy(f['trv_out_%d'%i][:]),
					torch.from_numpy(f['lp_times_%d'%i][:]),
					torch.from_numpy(f['lp_stations_%d'%i][:]).long(),
					torch.from_numpy(f['lp_phases_%d'%i][:].reshape(-1,1)).float(),
					torch.from_numpy(f['Locs_cart_%d'%i][:]),
					torch.from_numpy(f['X_fixed_cart_%d'%i][:]),
					torch.from_numpy(f['X_fixed_%d'%i][:,3][:]),
					torch.from_numpy(f['X_query_cart_%d'%i][:]),
					torch.from_numpy(f['x_src_query_cart_%d'%i][:]),
					torch.from_numpy(f['X_query_%d'%i][:,3]), 
					torch.from_numpy(f['tq_sample_%d'%i][:]), 
					torch.from_numpy(f['trv_out_src_%d'%i][:])
				]

				## Could possibly map to cuda here (note: possible ragged list for A_prod_src_tensor)
				input_tensors_l.append(input_tensors)


		if self.use_gradient_loss == False:

			return input_tensors_l, [lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l]

		else:

			return input_tensors_l, [lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l, Lbls_grad_spc, Lbls_query_grad_spc, Lbls_grad_t, Lbls_query_grad_t]



# class HDF5FileDataset(Dataset):
#     def __init__(self, hdf5_file_path, device='cpu'):
#         self.file_path = hdf5_file_path
#         self.device = device
        
#         print(f"Loading entire file into RAM: {os.path.basename(hdf5_file_path)}")
#         with h5py.File(hdf5_file_path, 'r') as f:
#             # Adjust these keys to your actual HDF5 layout
#             self.labels = torch.from_numpy(f['labels'][:])                    # (N,)
            
#             # Example: 4 graphs per sample
#             self.graphs = [
#                 torch.from_numpy(f['graph_0'][:]),   # (N, ...)
#                 torch.from_numpy(f['graph_1'][:]),
#                 torch.from_numpy(f['graph_2'][:]),
#                 torch.from_numpy(f['graph_3'][:]),
#             ]
#             # Optional: pre-move to GPU right now if you're brave and have VRAM
#             # if device != 'cpu':
#             #     self.graphs = [g.to(device, non_blocking=True) for g in self.graphs]
#             #     self.labels = self.labels.to(device, non_blocking=True)
        
#         self.n_samples = len(self.labels)
#         print(f"Loaded {self.n_samples} samples → ready for blazing-fast training")

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         # All data already in RAM → this is basically free
#         return tuple(g[idx] for g in self.graphs), self.labels[idx]

# lp_srcs_l = []
# lp_times_l = []
# lp_stations_l = []
# lp_phases_l = []
# X_query = []
# Lbls = []
# Lbls_query = []
# pick_lbls_l = []
# x_src_query_cart_l = []
# input_tensors_l = []


# h['Lbls_grad_spc_%d'%i] = Lbls[i][1]
# h['Lbls_query_grad_spc_%d'%i] = Lbls_query[i][1]
# h['Lbls_grad_t_%d'%i] = Lbls[i][2]
# h['Lbls_query_grad_t_%d'%i] = Lbls_query[i][2]


# adjust key names to your actual layout
# input_tensor_1_l.append(torch.from_numpy(f['input_tensor_1_%d'%i][:]))
# input_tensor_2_l.append(torch.from_numpy(f['input_tensor_2_%d'%i][:]))
# A_prod_sta_tensor_l.append(torch.from_numpy(f['A_prod_sta_tensor_%d'%i]).long())
# A_prod_src_tensor_l.append(torch.from_numpy(f['A_prod_src_tensor_%d'%i]).long())
# data_1_l.append(Data(x = spatial_vals, edge_index = torch.from_numpy(f['data_1_edges_%d'%i]).long()))
# data_2_l.append(Data(x = spatial_vals, edge_index = torch.from_numpy(f['data_2_edges_%d'%i]).long()))
# A_src_in_sta_l.append(torch.from_numpy(f['A_src_in_sta_%d'%i]).long())
# A_src_src_l.append(torch.from_numpy(f['A_src_src_%d'%i]).long())
# A_edges_time_p_l.append(torch.from_numpy(f['A_edges_time_p_%d'%i]).long())
# A_edges_time_s_l.append(torch.from_numpy(f['A_edges_time_s_%d'%i]).long())
# A_edges_ref_l.append(torch.from_numpy(f['A_edges_ref_%d'%i]))
# trv_out_l.append(torch.from_numpy(f['trv_out_%d'%i]))

# Locs_cart_l.append(torch.from_numpy(f['Locs_cart_%d'%i]))
# X_fixed_cart_l.append(torch.from_numpy(f['X_fixed_cart_%d'%i]))
# X_fixed_t_l.append(torch.from_numpy(f['X_fixed_%d'%i][:,3]))

# X_query_cart_l.append(torch.from_numpy(f['X_query_cart_%d'%i]))
# X_query_t_l.append(torch.from_numpy(f['X_query_%d'%i][:,3]))
# tq_sample_l.append(torch.from_numpy(f['tq_sample_%d'%i]))
# trv_out_src_l.append(torch.from_numpy(f['trv_out_src_%d'%i]))
# if self.use_expanded == True:
# 	Ac_prod_src_src_l.append(torch.from_numpy(f['Ac_prod_src_src_%d'%i]).long())
# else:
# 	Ac_prod_src_src_l.append([])



# Load all ~10 graphs for this event at once
# input_tensor_1_l = []
# input_tensor_2_l = []			
# A_prod_sta_tensor_l = []
# A_prod_src_tensor_l = []
# data_1_l = []
# data_2_l = []
# A_src_in_sta_l = []
# A_src_src_l = []
# A_edges_time_p_l = [] ## No longer used
# A_edges_time_s_l = [] ## No longer used
# A_edges_ref_l = [] ## No longer used
# trv_out_l = []
# Locs_cart_l = [] ## Transform
# X_fixed_cart_l = [] ## Transform
# X_fixed_t_l = [] ## Slice
# X_query_cart_l = [] ## Transform
# x_src_query_cart_l = []
# X_query_t_l = [] ## Slice
# tq_sample_l = []
# trv_out_src_l = []
# Ac_prod_src_src_l = [] # if self.use_expanded == True: 

# h['Lbls_grad_spc_%d'%i] = Lbls[i][1]
# h['Lbls_query_grad_spc_%d'%i] = Lbls_query[i][1]
# h['Lbls_grad_t_%d'%i] = Lbls[i][2]
# h['Lbls_query_grad_t_%d'%i] = Lbls_query[i][2]	


# with h5py.File(path, 'r') as f:
# 	# Load all ~10 graphs for this event at once
# 	# input_tensor_1_l = []
# 	# input_tensor_2_l = []			
# 	# A_prod_sta_tensor_l = []
# 	# A_prod_src_tensor_l = []
# 	# data_1_l = []
# 	# data_2_l = []
# 	# A_src_in_sta_l = []
# 	# A_src_src_l = []
# 	# A_edges_time_p_l = [] ## No longer used
# 	# A_edges_time_s_l = [] ## No longer used
# 	# A_edges_ref_l = [] ## No longer used
# 	# trv_out_l = []
# 	lp_srcs_l = []
# 	lp_times_l = []
# 	lp_stations_l = []
# 	lp_phases_l = []
# 	# Locs_cart_l = [] ## Transform
# 	# X_fixed_cart_l = [] ## Transform
# 	# X_fixed_t_l = [] ## Slice
# 	X_query = []

# 	# X_query_cart_l = [] ## Transform
# 	# x_src_query_cart_l = []
# 	# X_query_t_l = [] ## Slice
# 	# tq_sample_l = []
# 	# trv_out_src_l = []
# 	# Ac_prod_src_src_l = [] # if self.use_expanded == True: 

# 	Lbls_l = []
# 	Lbls_query_l = []
# 	pick_lbls_l = []

# 	if self.use_gradient_loss == True:
# 		Lbls_grad_spc_l = []
# 		Lbls_query_grad_spc_l = []
# 		Lbls_grad_t_l = []
# 		Lbls_query_grad_t_l = []
		
# 		# h['Lbls_grad_spc_%d'%i] = Lbls[i][1]
# 		# h['Lbls_query_grad_spc_%d'%i] = Lbls_query[i][1]
# 		# h['Lbls_grad_t_%d'%i] = Lbls[i][2]
# 		# h['Lbls_query_grad_t_%d'%i] = Lbls_query[i][2]	



		# data = [h['data'][:], h['srcs'][:], h['srcs_active'][:]]

## These are necessary inputs
# # Continue processing the rest of the inputs
# input_tensors = [
# 	input_tensor_1, input_tensor_2, A_prod_sta_tensor, [A_prod_src_tensor, Ac_prod_src_src_l[i0]],
# 	data_1, data_2,
# 	A_src_in_sta_l[i0],
# 	A_src_src_l[i0],
# 	torch.Tensor(A_edges_time_p_l[i0]).long().to(device),
# 	torch.Tensor(A_edges_time_s_l[i0]).long().to(device),
# 	torch.Tensor(A_edges_ref_l[i0]).to(device),
# 	trv_out,
# 	torch.Tensor(lp_times[i0]).to(device),
# 	torch.Tensor(lp_stations[i0]).long().to(device),
# 	torch.Tensor(lp_phases[i0]).reshape(-1, 1).float().to(device),
# 	torch.Tensor(ftrns1(Locs[i0])).to(device),
# 	torch.Tensor(ftrns1(X_fixed[i0])).to(device),
# 	torch.Tensor(X_fixed[i0][:,3]).to(device),
# 	torch.Tensor(ftrns1(X_query[i0])).to(device),
# 	torch.Tensor(x_src_query_cart).to(device),
# 	torch.Tensor(X_query[i0][:,3]).to(device), tq_sample, trv_out_src
# ]	



# def move_to(obj, device):
# 	"""Move tensor, list, tuple, dict, or nested combination thereof to device"""
# 	if isinstance(obj, torch.Tensor):
# 		return obj.to(device, non_blocking=True)
# 	if isinstance(obj, (list, tuple)):
# 		return type(obj)(move_to(x, device) for x in obj)
# 	if isinstance(obj, dict):
# 		return {k: move_to(v, device) for k, v in obj.items()}
# 	return obj  # scalar, etc.


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



# At top of file — define once
to_gpu = partial(move_to, device='cuda', non_blocking=True)
to_cpu = partial(move_to, device='cpu', non_blocking=False)
to_gpu_inplace = partial(move_to_inplace, device='cuda', non_blocking=True)

# graphs = list(map(lambda g: g.to(device, non_blocking=True), graphs))

# def to_device_inplace(obj, device):
# 	if isinstance(obj, torch.Tensor):
# 		obj.copy_(obj.to(device, non_blocking=True))   # if you really need in-place
# 	elif isinstance(obj, (list, tuple)):
# 		for i, x in enumerate(obj):
# 			obj[i] = to_device_inplace(x, device)
# 	elif isinstance(obj, dict):
# 		for k in obj:
# 			obj[k] = to_device_inplace(obj[k], device)
# 	return obj


## Load Dataset
if load_training_data == True:
	dataset = TrainingDataset(np.random.permutation(files_load), n_batch, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded)


# loader = DataLoader(
#     dataset,
#     batch_size=8,              # ← 64 events at a time → 64 × 10 = 640 sub-samples
#     shuffle=True,
#     num_workers=12,             # ← THIS is what makes it fast
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=4,          # PyTorch 2.0+: loads 4×batch_size ahead
# )

# def to_float32(batch):
#     if isinstance(batch, torch.Tensor):
#         return batch.float()
#     elif isinstance(batch, dict):
#         return {k: to_float32(v) for k, v in batch.items()}
#     elif isinstance(batch, (list, tuple)):
#         return type(batch)(to_float32(v) for v in batch)
#     else:
#         return batch

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
    assert len(batch) == 1
    return to_float32(batch[0]) ## Map to float (could also put this in the data loader if prefered)


loader = DataLoader(
    dataset,
    batch_size=1,              # ← 64 events at a time → 64 × 10 = 640 sub-samples
    shuffle=True,
    num_workers=3,             # ← THIS is what makes it fast
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=3,          # PyTorch 2.0+: loads 4×batch_size ahead
    collate_fn=collate_no_batch
)



## Can potentially batch the inputs (batch_size > 1) using a custom collate function 

# def cust_collate(batch_dict):
#     '''
#     Collate function that concatenates the data for each individual using axis 0
#     Args:
#         batch_dict (dict): dictionary with the data for each individual
#     Returns:
#         batch_dict (dict): dictionary with the concatenated (axis = 0 ) data for each individual
#     '''
#     # concatenate the data for each individual using axis 0
#     for i in range(len(batch_dict)):
#         if i == 0:
#             X = batch_dict[i]['X']
#             Y = batch_dict[i]['Y']
#             Z = batch_dict[i]['Z']
#             id = batch_dict[i]['id']
#         else:
#             X = torch.cat((X, batch_dict[i]['X']), axis=0)
#             Y = torch.cat((Y, batch_dict[i]['Y']), axis=0)
#             Z = torch.cat((Z, batch_dict[i]['Z']), axis=0)
#             id = torch.cat((id, batch_dict[i]['id']), axis=0)
#     return {'X': X, 'Z': Z, 'Y': Y, 'id': id}

# choice_data = ChoiceDataset_all(data, args, id_variable="id_ind")
# data_loader = DataLoader(choice_data, batch_size=2, 
#                          shuffle=False, num_workers=0, 
#                          drop_last=False, 
#                          collate_fn=cust_collate)


# moi

# st_time = time.time()

# for batch_idx, vals in enumerate(loader):

# 	if batch_idx > 10:
# 		break

# 	print(len(vals))

# print('Time %0.4f'%(time.time() - st_time))


## Set initial counter
# i = n_restart_step
log_buffer = [] ## Append write operations to here and flush every 10 steps


# for i in range(n_restart_step, n_epochs):
for batch_idx, inputs in enumerate(loader):

	## Effective step size
	i = n_restart_step + batch_idx
	if i == n_epochs:
		print('Finished training')
		sys.exit()


	if (i == n_restart_step)*(n_restart == True):
		## Load model and optimizer.
		mz.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(n_restart_step, n_ver), map_location = device))
		optimizer.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_optimizer.h5'%(n_restart_step, n_ver), map_location = device))
		LossBalancer.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_balancer.h5'%(n_restart_step, n_ver), map_location = device))
		zlosses = np.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(n_restart_step, n_ver))
		losses[0:n_restart_step] = zlosses['losses'][0:n_restart_step]
		mx_trgt_1[0:n_restart_step] = zlosses['mx_trgt_1'][0:n_restart_step]; mx_trgt_2[0:n_restart_step] = zlosses['mx_trgt_2'][0:n_restart_step]
		mx_trgt_3[0:n_restart_step] = zlosses['mx_trgt_3'][0:n_restart_step]; mx_trgt_4[0:n_restart_step] = zlosses['mx_trgt_4'][0:n_restart_step]
		mx_pred_1[0:n_restart_step] = zlosses['mx_pred_1'][0:n_restart_step]; mx_pred_2[0:n_restart_step] = zlosses['mx_pred_2'][0:n_restart_step]
		mx_pred_3[0:n_restart_step] = zlosses['mx_pred_3'][0:n_restart_step]; mx_pred_4[0:n_restart_step] = zlosses['mx_pred_4'][0:n_restart_step]
		print('loaded model for restart on step %d ver %d \n'%(n_restart_step, n_ver))
		zlosses.close()

	
	if ((np.mod(i, 1000) == 0) or (i == (n_epochs - 1)))*(i != n_restart_step):

		## Add save state of loss balancer so can re load
		torch.save(mz.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(i, n_ver))
		torch.save(optimizer.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d_optimizer.h5'%(i, n_ver))
		torch.save(LossBalancer.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d_balancer.h5'%(i, n_ver))
		# torch.save(balancer.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d_balancer.h5'%(i, n_ver))
		np.savez_compressed(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(i, n_ver), losses = losses, mx_trgt_1 = mx_trgt_1, mx_trgt_2 = mx_trgt_2, mx_trgt_3 = mx_trgt_3, mx_trgt_4 = mx_trgt_4, mx_pred_1 = mx_pred_1, mx_pred_2 = mx_pred_2, mx_pred_3 = mx_pred_3, mx_pred_4 = mx_pred_4, scale_x = scale_x, offset_x = offset_x, scale_x_extend = scale_x_extend, offset_x_extend = offset_x_extend, training_params = training_params, graph_params = graph_params, pred_params = pred_params)
		print('saved model %s %d'%(n_ver, i))
		print('saved model at step %d'%i)


	optimizer.zero_grad()


	## Need to overwrite the data entries in input_tensors
	if use_gradient_loss == False:
		lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l = inputs[1]
	else:
		lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l, Lbls_grad_spc, Lbls_query_grad_spc, Lbls_grad_t, Lbls_query_grad_t = inputs[1]		


	input_tensors_l = inputs[0]
	for j in range(n_batch): input_tensors_l[j][4] = Data(x = spatial_vals_l[j], edge_index = input_tensors_l[j][4]) ## Ideally remove these
	for j in range(n_batch): input_tensors_l[j][5] = Data(x = spatial_vals_l[j], edge_index = input_tensors_l[j][5])
	input_tensors_l = to_gpu(inputs[0]) if device.type == 'cuda' else inputs[0]
	if device.type == 'cuda': pick_lbls_l = to_gpu(pick_lbls_l)


	# ## Generate batch of synthetic inputs. Note, if this is too slow to interleave with model updates, 
	# ## you can  build these synthetic training data offline and then just load during training. The 
	# ## dataset would likely have a large memory footprint if doing so (e.g. > 1 Tb)

	# if load_training_data == False:

	# 	## Build a training batch on the fly
	# 	if use_subgraph == False:
	# 		[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True)
	# 		if use_expanded == True:
	# 			A_src_src_l, Ac_src_src_l = A_src_src_l
	# 			A_src_in_prod_l, Ac_src_in_prod_l = A_src_in_prod_l			
	# 		Inpts = [Inpts[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Inpts))]
	# 		Masks = [Masks[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Masks))]
	# 		A_sta_sta_l = [torch.Tensor(A_sta_sta_l[j]).long().to(device) for j in range(len(A_sta_sta_l))]
	# 		A_src_src_l = [torch.Tensor(A_src_src_l[j]).long().to(device) for j in range(len(A_src_src_l))]
	# 		A_prod_sta_sta_l = [torch.Tensor(A_prod_sta_sta_l[j]).long().to(device) for j in range(len(A_prod_sta_sta_l))]
	# 		A_prod_src_src_l = [torch.Tensor(A_prod_src_src_l[j]).long().to(device) for j in range(len(A_prod_src_src_l))]
	# 		A_src_in_prod_l = [torch.Tensor(A_src_in_prod_l[j]).long().to(device) for j in range(len(A_src_in_prod_l))]
	# 		A_src_in_sta_l = [torch.Tensor(np.concatenate((np.tile(np.arange(Locs[j].shape[0]), len(X_fixed[j])).reshape(1,-1), np.arange(len(X_fixed[j])).repeat(len(Locs[j]), axis = 0).reshape(1,-1)), axis = 0)).long().to(device) for j in range(len(Inpts))]
	# 		if use_expanded == True:
	# 			Ac_src_src_l = [torch.Tensor(Ac_src_src_l[j]).long().to(device) for j in range(len(Ac_src_src_l))]
	# 			Ac_prod_src_src_l = [torch.Tensor(Ac_prod_src_src_l[j]).long().to(device) for j in range(len(Ac_prod_src_src_l))]


	# 	if use_subgraph == True:
	# 		[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = True)
	# 		A_src_in_sta_l = [[] for j in range(len(Inpts))]
	# 		if use_expanded == True:
	# 			Ac_src_src_l = [[] for j in range(len(A_src_src_l))]
	# 			Ac_prod_src_src_l = [[] for j in range(len(A_prod_src_src_l))]				
			
	# 		for n in range(len(Inpts)):
	# 			A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[n], X_fixed[n], ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, Ac = Ac, device = device)
	# 			A_edges_time_p, A_edges_time_s, dt_partition = compute_time_embedding_vectors(trv_pairwise, Locs[n], X_fixed[n], A_src_in_sta, max_t, min_t = min_t, time_shift = X_fixed[n][:,3], dt_res = kernel_sig_t/5.0, t_win = kernel_sig_t*2.0, device = device)
	# 			if use_expanded == True:
	# 				A_src_src, Ac_src_src = A_src_src
	# 				A_prod_src_src, Ac_prod_src_src = A_prod_src_src
	# 			A_sta_sta_l[n] = A_sta_sta ## These should be equal
	# 			A_src_src_l[n] = A_src_src ## These should be equal
	# 			A_prod_sta_sta_l[n] = A_prod_sta_sta
	# 			A_prod_src_src_l[n] = A_prod_src_src
	# 			A_src_in_prod_l[n] = A_src_in_prod
	# 			A_src_in_sta_l[n] = A_src_in_sta
	# 			A_edges_time_p_l[n] = A_edges_time_p
	# 			A_edges_time_s_l[n] = A_edges_time_s
	# 			A_edges_ref_l[n] = dt_partition
	# 			Inpts[n] = np.copy(np.ascontiguousarray(Inpts[n][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))
	# 			Masks[n] = np.copy(np.ascontiguousarray(Masks[n][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))
	# 			if use_expanded == True:
	# 				Ac_src_src_l[n] = Ac_src_src ## These should be equal
	# 				Ac_prod_src_src_l[n] = Ac_prod_src_src


	# 	if use_gradient_loss == True:
	# 		Lbls_grad_t = [Lbls[n][2] for n in range(len(Lbls))]
	# 		Lbls_grad_spc = [Lbls[n][1] for n in range(len(Lbls))]
	# 		Lbls_query_grad_t = [Lbls_query[n][2] for n in range(len(Lbls_query))]
	# 		Lbls_query_grad_spc = [Lbls_query[n][1] for n in range(len(Lbls_query))]
	# 		Lbls = [Lbls[n][0] for n in range(len(Lbls))]	
	# 		Lbls_query = [Lbls_query[n][0] for n in range(len(Lbls_query))]

	# # else:


	# 	# file_choice = np.random.choice(files_load)

	# 	# h = h5py.File(file_choice, 'r')
		
	# 	# data = [h['data'][:], h['srcs'][:], h['srcs_active'][:]]


	# else:

	# 	pass


	loss_val = 0
	loss_regularize_val, loss_regularize_cnt = 0, 0
	loss_negative_val, loss_global_val = 0, 0
	mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4 = 0.0, 0.0, 0.0, 0.0
	mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4 = 0.0, 0.0, 0.0, 0.0

	loss_src_val = 0.0
	loss_asc_val = 0.0
	loss_grad_val = 0.0
	loss_dice_val = 0.0
	loss_consistency_val = 0.0
	loss_negative_val = 0.0
	loss_global_val = 0.0
	loss_bce_val = 0.0
	loss_cap_val = 0.0


	for inc, i0 in enumerate(range(n_batch)):

		# if load_training_data == True:

		# 	## Overwrite i0 and create length-1 lists for the training samples loaded from .hdf5 file
		# 	Inpts = []
		# 	Masks = []
		# 	X_fixed = []
		# 	X_query = []
		# 	Locs = []
		# 	Trv_out = []
		# 	Lbls = []
		# 	Lbls_query = []
		# 	lp_times = []
		# 	lp_stations = []
		# 	lp_phases = []
		# 	lp_meta = []
		# 	lp_srcs = []
		# 	A_sta_sta_l = []
		# 	A_src_src_l = []
		# 	A_prod_sta_sta_l = []
		# 	A_prod_src_src_l = []
		# 	A_src_in_prod_l = []
		# 	# A_src_in_prod_x_l = []
		# 	# A_src_in_prod_edges_l = []
		# 	A_edges_time_p_l = []
		# 	A_edges_time_s_l = []
		# 	A_edges_ref_l = []
		# 	A_src_in_sta_l = []

		# 	if use_expanded == True:
		# 		Ac_src_src_l = []
		# 		Ac_prod_src_src_l = []

		# 	if use_gradient_loss == True:
		# 		Lbls_grad_t = []
		# 		Lbls_grad_spc = []
		# 		Lbls_query_grad_t = []
		# 		Lbls_query_grad_spc = []

		# 	## Note: it would be more efficient (speed and memory) to pass 
		# 	## in each sample one at time, rather than appending batch to a list
			
		# 	Inpts.append(h['Inpts_%d'%i0][:])
		# 	Masks.append(h['Masks_%d'%i0][:])
		# 	X_fixed.append(h['X_fixed_%d'%i0][:])
		# 	X_query.append(h['X_query_%d'%i0][:])
		# 	Locs.append(h['Locs_%d'%i0][:])
		# 	Trv_out.append(h['Trv_out_%d'%i0][:])
		# 	Lbls.append(h['Lbls_%d'%i0][:])
		# 	Lbls_query.append(h['Lbls_query_%d'%i0][:])
		# 	lp_times.append(h['lp_times_%d'%i0][:])
		# 	lp_stations.append(h['lp_stations_%d'%i0][:])
		# 	lp_phases.append(h['lp_phases_%d'%i0][:])
		# 	lp_meta.append(h['lp_meta_%d'%i0][:])
		# 	lp_srcs.append(h['lp_srcs_%d'%i0][:])
		# 	A_sta_sta_l.append(torch.Tensor(h['A_sta_sta_%d'%i0][:]).long().to(device))
		# 	A_src_src_l.append(torch.Tensor(h['A_src_src_%d'%i0][:]).long().to(device))
		# 	A_prod_sta_sta_l.append(torch.Tensor(h['A_prod_sta_sta_%d'%i0][:]).long().to(device))
		# 	A_prod_src_src_l.append(torch.Tensor(h['A_prod_src_src_%d'%i0][:]).long().to(device))
		# 	A_src_in_prod_l.append(torch.Tensor(h['A_src_in_prod_%d'%i0][:]).long().to(device))

		# 	# A_src_in_prod_l.append(h['A_src_in_prod_%d'%i0][:])
		# 	# A_src_in_prod_x_l.append(h['A_src_in_prod_x_%d'%i0][:])
		# 	# A_src_in_prod_edges_l.append(h['A_src_in_prod_edges_%d'%i0][:])
		# 	# if use_subgraph == True:
		# 	A_src_in_sta_l.append(torch.Tensor(h['A_src_in_sta_%d'%i0][:]).long().to(device))
			
		# 	A_edges_time_p_l.append(h['A_edges_time_p_%d'%i0][:])
		# 	A_edges_time_s_l.append(h['A_edges_time_s_%d'%i0][:])
		# 	A_edges_ref_l.append(h['A_edges_ref_%d'%i0][:])

		# 	if use_expanded == True:
		# 		Ac_src_src_l.append(torch.Tensor(h['Ac_src_src_%d'%i0][:]).long().to(device))
		# 		Ac_prod_src_src_l.append(torch.Tensor(h['Ac_prod_src_src_%d'%i0][:]).long().to(device))

		# 	if use_gradient_loss == True:
		# 		Lbls_grad_spc.append(h['Lbls_grad_spc_%d'%i0][:])
		# 		Lbls_grad_t.append(h['Lbls_grad_t_%d'%i0][:])
		# 		Lbls_query_grad_spc.append(h['Lbls_query_grad_spc_%d'%i0][:])
		# 		Lbls_query_grad_t.append(h['Lbls_query_grad_t_%d'%i0][:])

		# 	i0 = 0 ## Over-write, so below indexing 


		## Now i0 is not set to 0
		## Adding skip... to skip samples with zero input picks
		if len(lp_times[i0]) == 0:
			print('skip a sample!') ## If this skips, and yet i0 == (n_batch - 1), is it a problem?
			continue ## Skip this!

		
		# if use_expanded == False:

		# 	# Continue processing the rest of the inputs
		# 	input_tensors = [
		# 		input_tensor_1, input_tensor_2, A_prod_sta_tensor, A_prod_src_tensor,
		# 		data_1, data_2,
		# 		A_src_in_sta_l[i0],
		# 		A_src_src_l[i0],
		# 		torch.Tensor(A_edges_time_p_l[i0]).long().to(device),
		# 		torch.Tensor(A_edges_time_s_l[i0]).long().to(device),
		# 		torch.Tensor(A_edges_ref_l[i0]).to(device),
		# 		trv_out,
		# 		torch.Tensor(lp_times[i0]).to(device),
		# 		torch.Tensor(lp_stations[i0]).long().to(device),
		# 		torch.Tensor(lp_phases[i0]).reshape(-1, 1).float().to(device),
		# 		torch.Tensor(ftrns1(Locs[i0])).to(device),
		# 		torch.Tensor(ftrns1(X_fixed[i0])).to(device),
		# 		torch.Tensor(X_fixed[i0][:,3]).to(device),
		# 		torch.Tensor(ftrns1(X_query[i0])).to(device),
		# 		torch.Tensor(x_src_query_cart).to(device),
		# 		torch.Tensor(X_query[i0][:,3]).to(device), tq_sample, trv_out_src
		# 	]

		# else:

		# 	# Continue processing the rest of the inputs
		# 	input_tensors = [
		# 		input_tensor_1, input_tensor_2, A_prod_sta_tensor, [A_prod_src_tensor, Ac_prod_src_src_l[i0]],
		# 		data_1, data_2,
		# 		A_src_in_sta_l[i0],
		# 		A_src_src_l[i0],
		# 		torch.Tensor(A_edges_time_p_l[i0]).long().to(device),
		# 		torch.Tensor(A_edges_time_s_l[i0]).long().to(device),
		# 		torch.Tensor(A_edges_ref_l[i0]).to(device),
		# 		trv_out,
		# 		torch.Tensor(lp_times[i0]).to(device),
		# 		torch.Tensor(lp_stations[i0]).long().to(device),
		# 		torch.Tensor(lp_phases[i0]).reshape(-1, 1).float().to(device),
		# 		torch.Tensor(ftrns1(Locs[i0])).to(device),
		# 		torch.Tensor(ftrns1(X_fixed[i0])).to(device),
		# 		torch.Tensor(X_fixed[i0][:,3]).to(device),
		# 		torch.Tensor(ftrns1(X_query[i0])).to(device),
		# 		torch.Tensor(x_src_query_cart).to(device),
		# 		torch.Tensor(X_query[i0][:,3]).to(device), tq_sample, trv_out_src
		# 	]			

		# locs_use_cart, x_temp_cuda_cart, x_temp_cuda_t

		# Call the model with pre-processed tensors
		## To integrate negative_loss, must replace out[1] (which can't be overwritten, or can it?)
		## If not, can replace with a variable name that is overwritten

		## Extract inputs from inputs
		# input_tensors = to_gpu(inputs[0]) if device == 'cuda' else inputs[0]


		if use_gradient_loss == False:

			out = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])

		else:

			# out, grads = mz(*input_tensors)
			out, grads = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])
			grad_grid_src, grad_grid_t, grad_query_src, grad_query_t = grads


		# moi


		## Note, negative loss could be used on x_src_query_cart, but it is more essential for constraining the false positives in out[1]
		# pick_lbls = pick_labels_extract_interior_region_flattened(x_src_query_cart, tq_sample.cpu().detach().numpy(), lp_meta[i0][:,-2::], lp_srcs[i0], lat_range_interior, lon_range_interior, ftrns1, sig_t = src_t_arv_kernel, sig_x = src_x_arv_kernel)


		## Select the specific pick labels (or can re-create above)
		pick_lbls = pick_lbls_l[i0]



		## Make plots
		make_plot = False
		if make_plot == True:
			fig, ax = plt.subplots(4, 1, sharex = True)
			for j in range(2):
				i1 = np.where(Lbls_query[i0][:,0].cpu().detach().numpy() > 0.1)[0]
				i2 = np.where(out[1][:,0].cpu().detach().numpy() > 0.1)[0]
				ax[2*j].scatter(X_query[i0][i1,3].cpu().detach().numpy(), X_query[0][i1,j].cpu().detach().numpy(), c = Lbls_query[i0][i1,0].cpu().detach().numpy())
				ax[2*j + 1].scatter(X_query[i0][i2,3].cpu().detach().numpy(), X_query[i0][i2,j].cpu().detach().numpy(), c = out[1][i2,0].cpu().detach().numpy())
				ax[2*j].set_xlim(X_query[i0][:,3].amin(), X_query[i0][:,3].amax())
				ax[2*j + 1].set_xlim(X_query[i0][:,3].amin(), X_query[i0][:,3].amax())
				ax[2*j].set_ylim(X_query[i0][:,j].amin(), X_query[i0][:,j].amax())
				ax[2*j + 1].set_ylim(X_query[i0][:,j].amin(), X_query[i0][:,j].amax())

			fig.set_size_inches(10,8)
			fig.savefig(path_to_file + 'Plots/example_sources_%d.png'%cnt_plot)

			fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
			ax[0].scatter(X_query[i0][:,3].cpu().detach().numpy(), Lbls_query[i0][:,0].cpu().detach().numpy(), c = X_query[i0][:,0].cpu().detach().numpy())
			ax[1].scatter(X_query[i0][:,3].cpu().detach().numpy(), out[1][:,0].cpu().detach().numpy(), c = X_query[i0][:,0].cpu().detach().numpy())
			fig.savefig(path_to_file + 'Plots/example_sources_in_time_%d.png'%cnt_plot)



			fig, ax = plt.subplots(2,2, figsize = [12,8])
			iarg = np.argmax((pick_lbls[:,:,0] + pick_lbls[:,:,1]).sum(1).cpu().detach().numpy())
			min_thresh_val = 0.15


			ifindp = np.where(pick_lbls[iarg,:,0].cpu().detach().numpy() > min_thresh_val)[0] # ].astype('int')
			ifinds = np.where(pick_lbls[iarg,:,1].cpu().detach().numpy() > min_thresh_val)[0] # ].astype('int')
			ax[0,0].scatter(Locs[i0][:,1], Locs[i0][:,0], c = 'grey', marker = '^')
			ax[0,0].scatter(Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifindp],1], Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifindp],0], c = pick_lbls[iarg,ifindp,0].cpu().detach().numpy(), marker = '^')
			ax[0,1].scatter(Locs[i0][:,1], Locs[i0][:,0], c = 'grey', marker = '^')
			ax[0,1].scatter(Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifinds],1], Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifinds],0], c = pick_lbls[iarg,ifinds,1].cpu().detach().numpy(), marker = '^')
			ax[0,0].set_aspect(1.0/np.cos(locs[:,0].mean()*np.pi/180.0))
			ax[0,1].set_aspect(1.0/np.cos(locs[:,0].mean()*np.pi/180.0))
			src_plot = ftrns2(x_src_query_cart_l[i0].cpu().detach().numpy()[iarg].reshape(1,-1))
			ax[0,0].scatter(src_plot[:,1], src_plot[:,0], c = 'm')
			ax[0,1].scatter(src_plot[:,1], src_plot[:,0], c = 'm')


			ifindp = np.where(out[2][iarg,:,0].cpu().detach().numpy() > min_thresh_val)[0] # ].astype('int')
			ifinds = np.where(out[3][iarg,:,0].cpu().detach().numpy() > min_thresh_val)[0] # ].astype('int')
			ax[1,0].scatter(Locs[i0][:,1], Locs[i0][:,0], c = 'grey', marker = '^')
			ax[1,0].scatter(Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifindp],1], Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifindp],0], c = out[2][iarg,ifindp,0].cpu().detach().numpy(), marker = '^')
			ax[1,1].scatter(Locs[i0][:,1], Locs[i0][:,0], c = 'grey', marker = '^')
			ax[1,1].scatter(Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifinds],1], Locs[i0][lp_stations[i0].cpu().detach().numpy().astype('int')[ifinds],0], c = out[3][iarg,ifinds,0].cpu().detach().numpy(), marker = '^')
			ax[1,0].set_aspect(1.0/np.cos(locs[:,0].mean()*np.pi/180.0))
			ax[1,1].set_aspect(1.0/np.cos(locs[:,0].mean()*np.pi/180.0))
			src_plot = ftrns2(x_src_query_cart_l[i0].cpu().detach().numpy()[iarg].reshape(1,-1))
			ax[1,0].scatter(src_plot[:,1], src_plot[:,0], c = 'm')
			ax[1,1].scatter(src_plot[:,1], src_plot[:,0], c = 'm')
			fig.savefig(path_to_file + 'Plots/example_stations_%d.png'%cnt_plot)


			print('Saved figures %d'%cnt_plot)
			cnt_plot += 1
			plt.close('all')



		# use_dice_loss = True
		if use_dice_loss == True:


			loss_base1 = DiceLoss(out[0], torch.Tensor(Lbls[i0]).to(device))
			loss_dice2 = DiceLoss(out[1], torch.Tensor(Lbls_query[i0]).to(device))
			loss_dice3 = DiceLoss(out[2][:,:,0], pick_lbls[:,:,0])
			loss_dice4 = DiceLoss(out[3][:,:,0], pick_lbls[:,:,1])
			# loss_dice = (loss_dice1 + loss_dice2 + loss_dice3 + loss_dice4)/4.0

			# loss_src_val += (loss_dice1.item() + loss_dice2.item())/n_batch
			loss_src_val += (loss_base1.item() + loss_dice2.item())/n_batch
			loss_asc_val += (loss_dice3.item() + loss_dice4.item())/n_batch

			# loss = 0.8*loss + 0.2*loss_dice
			# print('Dice %0.5f %0.5f'%(n_batch*loss_src_val, n_batch*loss_asc_val)) # loss.item(), 


		# use_mse_loss = False ## Switching to Huber loss
		if use_mse_loss == True:

			if use_sigmoid == False:

				# loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1]))/n_batch
				# loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
				loss_mse1 = weights[0]*mse_loss(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*mse_loss(out[1], torch.Tensor(Lbls_query[i0]).to(device))
				loss_mse2 = (weights[2]*mse_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*mse_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
				# loss = loss_mse1 + 2.0*loss_mse2

			else:

				min_val_lbl = 0.05
				ifind_mask1 = torch.Tensor(Lbls[i0][:,0] > min_val_lbl).long().to(device)
				ifind_mask2 = torch.Tensor(Lbls_query[i0][:,0] > min_val_lbl).long().to(device)

				loss_bce = weights[0]*bce_loss(out[0][:,1], ifind_mask1.float()) + weights[1]*bce_loss(out[1][:,1], ifind_mask2.float())
				loss_mse1 = weights[0]*mse_loss(out[0][ifind_mask1,0].reshape(-1,1), torch.Tensor(Lbls[i0]).to(device)[ifind_mask1]) + weights[1]*mse_loss(out[1][ifind_mask2,0].reshape(-1,1), torch.Tensor(Lbls_query[i0]).to(device)[ifind_mask2])
				loss_mse2 = (weights[2]*mse_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*mse_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch

				# loss = loss_mse1 + loss_mse2 + (1/100.0)*loss_bce
				# loss_bce_val += 0.1*loss_bce.item()


		# use_huber_loss = True
		if use_huber_loss == True:

			if use_sigmoid == False:

				# loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1]))/n_batch
				# loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
				loss_huber1 = weights[0]*huber_loss(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*huber_loss(out[1], torch.Tensor(Lbls_query[i0]).to(device))
				loss_huber2 = (weights[2]*huber_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*huber_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
				# loss = loss_mse1 + 2.0*loss_mse2

			else:

				min_val_lbl = 0.05
				ifind_mask1 = torch.Tensor(Lbls[i0][:,0] > min_val_lbl).long().to(device)
				ifind_mask2 = torch.Tensor(Lbls_query[i0][:,0] > min_val_lbl).long().to(device)

				loss_bce = weights[0]*bce_loss(out[0][:,1], ifind_mask1.float()) + weights[1]*bce_loss(out[1][:,1], ifind_mask2.float())
				loss_huber1 = weights[0]*mse_loss(out[0][ifind_mask1,0].reshape(-1,1), torch.Tensor(Lbls[i0]).to(device)[ifind_mask1]) + weights[1]*mse_loss(out[1][ifind_mask2,0].reshape(-1,1), torch.Tensor(Lbls_query[i0]).to(device)[ifind_mask2])
				loss_huber2 = (weights[2]*mse_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*mse_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch

				# loss = loss_mse1 + loss_mse2 + (1/100.0)*loss_bce
				# loss_bce_val += 0.1*loss_bce.item()


			# loss_src_val += loss_mse1.item()/n_batch
			# loss_asc_val += loss_mse2.item()/n_batch


		# use_cap_loss = True
		if (use_cap_loss == True)*(i > n_burn_in):

			scale_cap = 1.0
			cap_limit = 0.7

			ifind_cap1 = np.where(Lbls[i0] > cap_limit)[0]
			ifind_cap2 = np.where(Lbls_query[i0] > cap_limit)[0]
			# ifind_cap11, ifind_cap12 = np.where(pick_lbls[:,:,0].cpu().detach().numpy() > cap_limit) # [0]
			# ifind_cap21, ifind_cap22 = np.where(pick_lbls[:,:,1].cpu().detach().numpy() > cap_limit) # [0]

			loss_cap1 = torch.tensor(0.0).to(device)
			loss_cap2 = torch.tensor(0.0).to(device)
			if len(ifind_cap1) > 0: loss_cap1 += scale_cap*(weights[0]*huber_loss(out[0][ifind_cap1], torch.Tensor(Lbls[i0][ifind_cap1]).to(device)))
			if len(ifind_cap2) > 0: loss_cap1 += scale_cap*(weights[1]*huber_loss(out[1][ifind_cap2], torch.Tensor(Lbls_query[i0][ifind_cap2]).to(device)))

			# if len(ifind_cap11) > 0: loss_cap2 += scale_cap*(weights[2]*huber_loss(out[2][ifind_cap11,ifind_cap12,0], pick_lbls[ifind_cap11,ifind_cap12,0]))
			# if len(ifind_cap21) > 0: loss_cap2 += scale_cap*(weights[3]*huber_loss(out[3][ifind_cap21,ifind_cap22,0], pick_lbls[ifind_cap21,ifind_cap22,1]))

			loss_cap_val += (loss_cap1 + loss_cap2)/n_batch

			# else:
			# 	loss_cap2 = torch.tensor(0).to(device)


		if (use_negative_loss == True)*(i > n_burn_in):


			min_up_sample = 0.1
			## Up-sample queries for regions of high prediction but low labels. Or alternatively, essentially run a peak finder on the output.
			## Do not include points that are < thresh for both labels and predictions
			prob_up_sample = np.maximum(out[1][:,0].detach().cpu().detach().numpy()*(out[1][:,0].detach().cpu().detach().numpy() > min_up_sample)*(Lbls_query[i0][:,0].cpu().detach().numpy() < min_up_sample), 0.0)
			# prob_up_sample = 
			if prob_up_sample.sum() == 0: prob_up_sample = np.ones(len(prob_up_sample))
			prob_up_sample = prob_up_sample/prob_up_sample.sum() ## Can transform these probabilities or clip them
			x_query_sample, x_query_sample_t = sample_dense_queries(X_query[i0][:,0:3].cpu().detach().numpy(), X_query[i0][:,3].cpu().detach().numpy(), prob_up_sample, lat_range_extend, lon_range_extend, depth_range, src_x_kernel, src_depth_kernel, src_t_kernel, time_shift_range, ftrns1, ftrns2, replace = False, randomize = False) # replace = False
			out_query = mz.forward_queries(torch.Tensor(ftrns1(x_query_sample)).to(device), torch.Tensor(x_query_sample_t).to(device), train = True) # x_query_cart, t_query
			lbls_query = compute_source_labels(x_query_sample, x_query_sample_t, lp_srcs[i0][:,0:3].cpu().detach().numpy(), lp_srcs[i0][:,3].cpu().detach().numpy(), src_spatial_kernel, src_t_kernel, ftrns1) ## Compute updated labels


			# mask_query = (out_query > min_up_sample) + (torch.Tensor(lbls_query).to(device) > min_up_sample) ## At least one field > min_up_sample
			# mask_query = (out_query > min_up_sample) + (torch.Tensor(lbls_query).to(device) > min_up_sample) ## At least one field > min_up_sample


			if use_sigmoid == False:

				# loss_negative = mse_loss(out_query[mask_query], torch.Tensor(lbls_query).to(device)[mask_query]) # weights[1]*

				# if mask_query.sum() > 0:
				# loss_negative = mse_loss(out_query[mask_query], torch.Tensor(lbls_query).to(device)[mask_query]) # weights[1]*
				# loss_negative = mse_loss(out_query, torch.Tensor(lbls_query).to(device)) # weights[1]*
				loss_negative = huber_loss(out_query, torch.Tensor(lbls_query).to(device)) # weights[1]*
				# else:
				# loss_negative = torch.tensor(0.0).to(device)


				# loss2 = (weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
				# loss = 0.975*loss + 0.025*loss_negative

			else:

				## Not completely implemented
				ifind_mask = torch.Tensor(lbls_query[:,0] > min_val_lbl).long().to(device)
				loss_negative = (1/100.0)*bce_loss(out_query[:,1], ifind_mask.float()) + mse_loss(out_query[ifind_mask,0], torch.Tensor(lbls_query).to(device)[ifind_mask,0]) # weights[1]*
				# loss2 = (weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
				# loss = 0.9*loss + 0.1*loss_negative				

			loss_negative_val += loss_negative.item() # /n_batch
			# print('loss negative %0.8f'%loss_negative)


		loss_consistency_flag = False
		if (use_consistency_loss == True)*(i > n_burn_in):

			ilen = int(np.floor(n_batch/2/2))

			## For consistency loss, compute seperately for positive and negative classes

			if (np.mod(inc, 2) == 1)*(inc >= (n_batch - 2*ilen)):
				if (i == iter_loss[0])*(inc == (iter_loss[1] + 1)):
					ind_consistency = int(np.floor(len(Lbls_save[0])/2))
					# pdb.set_trace()

					if use_sigmoid == False:

						# weight1 = ((Lbls[i0] - Lbls_save[0]).max() == 0) # .float()
						mask_loss = torch.Tensor((np.abs(Lbls_query[i0][ind_consistency::].cpu().detach().numpy() - Lbls_save[0][ind_consistency::]) < 0.01)).to(device).float()  # .float()
						loss_consistency = mse_loss(out[1][ind_consistency::][mask_loss.long()], out_save[0][ind_consistency::][mask_loss.long()]) # )/torch.maximum(torch.Tensor([1.0]).to(device), (weight1*weights[0] + weight2*weights[1]))
						# loss = 0.9*loss + 0.1*loss_consistency ## Need to check relative scaling of this compared to focal loss
						# loss_consistency_val += 0.1*loss_consistency.item()/n_batch

					else:

						mask_loss = torch.Tensor((np.abs(Lbls_query[i0][ind_consistency::].cpu().detach().numpy() - Lbls_save[0][ind_consistency::]) < 0.01)).to(device).float()  # .float()
						loss_consistency = mse_loss(out[1][ind_consistency::][mask_loss.long()][:,0], out_save[0][ind_consistency::][mask_loss.long()][:,0]) # )/torch.maximum(torch.Tensor([1.0]).to(device), (weight1*weights[0] + weight2*weights[1]))
						# loss = 0.9*loss + 0.1*loss_consistency ## Need to check relative scaling of this compared to focal loss
						# loss_consistency_val += 0.1*loss_consistency.item()/n_batch					

					loss_consistency_val += loss_consistency.item() # /n_batch
					loss_consistency_flag = True


			out_save = [out[1]]
			Lbls_save = [Lbls_query[i0].cpu().detach().numpy()]
			iter_loss = [i, inc]
			X_query_save = [X_query[i0].cpu().detach().numpy()]


		# if (use_gradient_loss == True)*(i > int(n_epochs/5)):
		if (use_gradient_loss == True)*(i > n_burn_in):

			# if init_gradient_loss == False:
			# 	init_gradient_loss, mz.activate_gradient_loss = True, True
			# else:
			# 	loss_grad1 = 0.5*weights[0]*mse_loss(torch.Tensor([src_kernel_mean]).to(device)*grad_grid_src, torch.Tensor(Lbls_grad_spc[i0]).to(device)) + 0.5*weights[0]*mse_loss(torch.Tensor([src_t_kernel]).to(device)*grad_grid_t, torch.Tensor(Lbls_grad_t[i0]).to(device))
			# 	loss_grad2 = 0.5*weights[1]*mse_loss(torch.Tensor([src_kernel_mean]).to(device)*grad_query_src, torch.Tensor(Lbls_query_grad_spc[i0]).to(device)) + 0.5*weights[1]*mse_loss(torch.Tensor([src_t_kernel]).to(device)*grad_query_t.reshape(-1), torch.Tensor(Lbls_query_grad_t[i0]).to(device))
			# 	loss_grad = (loss_grad1 + loss_grad2)/(weights[0] + weights[1])

			# 	# loss = 0.5*loss + 0.5*loss_grad
			# 	loss_grad_val += 0.5*loss_grad.item()/n_batch

			if init_gradient_loss == False:
				init_gradient_loss, mz.activate_gradient_loss = True, True
			else:

				loss_grad_magnitude_spc1 = l1_loss(torch.norm(torch.Tensor([src_kernel_mean]).to(device)*grad_grid_src, dim = 1), torch.norm(torch.Tensor(Lbls_grad_spc[i0]).to(device), dim = 1))
				loss_grad_magnitude_time1 = l1_loss(torch.norm(torch.Tensor([src_t_kernel]).to(device)*grad_grid_t, dim = 1), torch.norm(torch.Tensor(Lbls_grad_t[i0]).to(device), dim = 1))

				loss_grad_magnitude_spc2 = l1_loss(torch.norm(torch.Tensor([src_kernel_mean]).to(device)*grad_query_src, dim = 1), torch.norm(torch.Tensor(Lbls_query_grad_spc[i0]).to(device), dim = 1))
				loss_grad_magnitude_time2 = l1_loss(torch.norm(torch.Tensor([src_t_kernel]).to(device)*grad_query_t.reshape(-1), dim = 1), torch.norm(torch.Tensor(Lbls_query_grad_t[i0]).to(device), dim = 1))

				## Add (non normalized) cosine distance between vectors

				# def gradient_loss(pred, gt):
				#     # pred, gt: [B,1,H,W] or [B,3,H,W] etc.
				#     gy_pred, gx_pred = torch.gradient(pred, spacing=1.0)  # or sobel for isotropic
				#     gy_gt, gx_gt   = torch.gradient(gt)
				#     grad_mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)
				#     grad_mag_gt   = torch.sqrt(gx_gt**2 + gy_gt**2 + 1e-6)
				    
				#     # L1 on magnitude + cosine similarity on direction
				#     loss_mag = F.l1_loss(grad_mag_pred, grad_gt_mag)
				#     loss_dir = 1 - F.cosine_similarity(gx_pred*gt + gy_gt, gx_pred*gt + gy_pred + 1e-6).mean() ## Not normalizing
				#     return loss_mag + loss_dir


				# loss_grad1 = 0.5*weights[0]*mse_loss(torch.Tensor([src_kernel_mean]).to(device)*grad_grid_src, torch.Tensor(Lbls_grad_spc[i0]).to(device)) + 0.5*weights[0]*mse_loss(torch.Tensor([src_t_kernel]).to(device)*grad_grid_t, torch.Tensor(Lbls_grad_t[i0]).to(device))
				# loss_grad2 = 0.5*weights[1]*mse_loss(torch.Tensor([src_kernel_mean]).to(device)*grad_query_src, torch.Tensor(Lbls_query_grad_spc[i0]).to(device)) + 0.5*weights[1]*mse_loss(torch.Tensor([src_t_kernel]).to(device)*grad_query_t.reshape(-1), torch.Tensor(Lbls_query_grad_t[i0]).to(device))
				# loss_grad = (loss_grad1 + loss_grad2)/(weights[0] + weights[1])

				# loss = 0.5*loss + 0.5*loss_grad
				loss_grad_val += 0.5*loss_grad.item()/n_batch



		## Merge loss values in the Scalar (normalize with respect to batch size)

		# pre_scale_weights1 = [10.0, 10.0] ## May have to decrease these as training goes on (as MSE converged much closer to zero)
		pre_scale_weights1 = [30.0, 30.0] ## May have to decrease these as training goes on (as MSE converged much closer to zero)
		pre_scale_weights2 = [1e3, 1e4]
		# pre_scale_weights2 = [1e4, 1e4]


		# ## Compute base losses
		# loss_dict = {
		# # 'loss_dice1': loss_base1, # loss_dice1
		# 'loss_base1': loss_base1, # loss_dice1
		# 'loss_dice2': loss_dice2,
		# 'loss_dice3': loss_dice3,
		# 'loss_dice4': loss_dice4,
		# 'loss_huber1': loss_huber1*pre_scale_weights1[0],
		# 'loss_huber2': loss_huber2*pre_scale_weights1[1],
		# 'loss_cap1': loss_cap1*pre_scale_weights1[0],
		# 'loss_cap2': loss_cap2*pre_scale_weights1[1],
		# }

		## Compute base losses
		loss_dict = {
		# 'loss_dice1': loss_base1, # loss_dice1
		'loss_dice1': loss_base1, # loss_dice1
		'loss_dice2': loss_dice2,
		'loss_dice3': 0.5*loss_dice3 + 0.5*loss_dice4,
		# 'loss_dice4': loss_dice4,
		'loss_huber1': loss_huber1*pre_scale_weights1[0],
		'loss_huber2': loss_huber2*pre_scale_weights1[1],
		# 'loss_cap1': loss_cap1*pre_scale_weights1[0],
		# 'loss_cap2': loss_cap2*pre_scale_weights1[1],
		}

		# 'loss_mse1': loss_mse1*pre_scale_weights1[0],
		# 'loss_mse2': loss_mse2*pre_scale_weights1[1]


		if i > n_burn_in:
			loss_dict.update({'loss_negative': loss_negative*pre_scale_weights2[0]})

			if loss_consistency_flag == True:
				loss_dict.update({'loss_consistency': loss_consistency*pre_scale_weights2[1]})
			
			loss_dict.update({'loss_cap1': loss_cap1*pre_scale_weights1[0]})
			# loss_dict.update({'loss_cap2': loss_cap2*pre_scale_weights1[1]})


		# moi
		loss = LossBalancer(loss_dict, accum_steps = n_batch, is_last_accum_step = (inc == (n_batch - 1))) # losses_dict: dict, accum_steps: int = None, is_last_accum_step: bool = False
		loss = loss/n_batch

		# loss = loss/n_batch


		n_visualize_step = 1000
		n_visualize_fraction = 0.2
		make_visualize_predictions = False
		if (make_visualize_predictions == True)*(np.mod(i, n_visualize_step) == 0)*(inc == 0): # (i0 < n_visualize_fraction*n_batch)
			save_plots_path = path_to_file + seperator + 'Plots' + seperator
			out_plot = [out[0], out[1], out[2], out[3]]
			visualize_predictions(out_plot, Lbls_query[i0], pick_lbls, X_query[i0], lp_times[i0], lp_stations[i0], Locs[i0], data, i0, save_plots_path, n_step = i, n_ver = n_ver)


		if inc != (n_batch - 1):
			loss.backward(retain_graph = True)
		else:
			loss.backward(retain_graph = False)


		loss_val += loss.item()
		mx_trgt_val_1 += Lbls[i0].max()
		mx_trgt_val_2 += Lbls_query[i0].max()
		mx_trgt_val_3 += pick_lbls[:,:,0].max().item()
		mx_trgt_val_4 += pick_lbls[:,:,1].max().item()
		mx_pred_val_1 += out[0].max().item()
		mx_pred_val_2 += out[1].max().item()
		mx_pred_val_3 += out[2].max().item()
		mx_pred_val_4 += out[3].max().item()


	# if load_training_data == True:
	# 	h.close() ## Close training file


	optimizer.step()
	losses[i] = loss_val
	mx_trgt_1[i] = mx_trgt_val_1/n_batch
	mx_trgt_2[i] = mx_trgt_val_2/n_batch
	mx_trgt_3[i] = mx_trgt_val_3/n_batch
	mx_trgt_4[i] = mx_trgt_val_4/n_batch

	mx_pred_1[i] = mx_pred_val_1/n_batch
	mx_pred_2[i] = mx_pred_val_2/n_batch
	mx_pred_3[i] = mx_pred_val_3/n_batch
	mx_pred_4[i] = mx_pred_val_4/n_batch
	loss_regularize_val = loss_regularize_val/np.maximum(1.0, loss_regularize_cnt)

	print('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f [%0.5f, %0.5f, %0.5f, %0.5f, %0.5f] (reg %0.8f) \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4, loss_src_val, loss_asc_val, loss_negative_val, loss_cap_val, loss_consistency_val, (10e4)*loss_regularize_val))

	# Log losses
	if use_wandb_logging == True:
		wandb.log({"loss": loss_val})

	log_buffer.append('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f [%0.5f, %0.5f, %0.5f, %0.5f, %0.5f] (reg %0.8f) \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4, loss_src_val, loss_asc_val, loss_negative_val, loss_cap_val, loss_consistency_val, (10e4)*loss_regularize_val))

	if np.mod(i, 10) == 0:
		with open(write_training_file + 'output_%d.txt'%n_ver, 'a') as text_file:
			for log in log_buffer:
				# text_file.write('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f [%0.5f, %0.5f, %0.5f, %0.5f, %0.5f] (reg %0.8f) \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4, loss_src_val, loss_asc_val, loss_negative_val, loss_cap_val, loss_consistency_val, (10e4)*loss_regularize_val))
				text_file.write(log)
		log_buffer.clear()








## Apply offline threshold estimator


# def detect_discrete_events(val):

# 	# iz1, iz2 = np.where(val > 0.01) # Zeros out all values less than this
# 	# Out_2_sparse = np.concatenate((iz1.reshape(-1,1), iz2.reshape(-1,1), Out_2[iz1,iz2].reshape(-1,1)), axis = 1)

# 	print('Continuous processing time %0.4f'%(time.time() - st_process))

# 	xq = np.copy(X_query)
# 	ts = np.copy(tsteps_abs)
# 	assert(np.allclose(np.diff(ts)[0] - dt_win, 0.0))

# 	if use_fixed_edges == True:
# 		assert(len(x_grid_ind_list) == 1)
# 		mz_list[x_grid_ind_list[0]].SpaceTimeAttention.use_fixed_edges = False
# 		# x_grid_ind = x_grid_ind_list[0]

# 	print('Begin peak finding')
# 	use_sparse_peak_finding = False
# 	if use_sparse_peak_finding == True:

# 		srcs_init = []
# 		for i in range(xq.shape[0]):

# 			ifind_x = np.where(iz1 == i)[0]
# 			if len(ifind_x) > 0:

# 				trace = np.zeros(len(ts))
# 				trace[iz2[ifind_x]] = Out_2_sparse[ifind_x,2]
				
# 				# ip = np.where(Out[:,i] > thresh)[0]
# 				ip = find_peaks(trace, height = thresh, distance = int(1.5*src_t_kernel/dt_win)) ## Note: should add prominence as thresh/2.0, which might help detect nearby events. Also, why is min time spacing set as 2 seconds?
# 				if len(ip[0]) > 0: # why use xq here?
# 					val = np.concatenate((xq[i,:].reshape(1,-1)*np.ones((len(ip[0]),3)), ts[ip[0]].reshape(-1,1), ip[1]['peak_heights'].reshape(-1,1)), axis = 1)
# 					srcs_init.append(val)		
	
# 	else:
	
# 		Out = np.zeros((X_query.shape[0], len(tsteps_abs))) ## Use dense out array
# 		Out[Out_2_sparse[:,0].astype('int'), Out_2_sparse[:,1].astype('int')] = Out_2_sparse[:,2]
	
# 		srcs_init = []
# 		for i in range(Out.shape[0]):
# 			# ip = np.where(Out[:,i] > thresh)[0]
# 			ip = find_peaks(Out[i,:], height = thresh, distance = int(1.5*src_t_kernel/dt_win)) ## Note: should add prominence as thresh/2.0, which might help detect nearby events. Also, why is min time spacing set as 2 seconds?
# 			if len(ip[0]) > 0: # why use xq here?
# 				val = np.concatenate((xq[i,:].reshape(1,-1)*np.ones((len(ip[0]),3)), ts[ip[0]].reshape(-1,1), ip[1]['peak_heights'].reshape(-1,1)), axis = 1)
# 				srcs_init.append(val)

# 	if len(srcs_init) == 0:
# 		continue ## No sources, continue

# 	srcs_init = np.vstack(srcs_init) # Could this have memory issues?

# 	srcs_init = srcs_init[np.argsort(srcs_init[:,3]),:]
# 	tdiff = np.diff(srcs_init[:,3])
# 	ibreak = np.where(tdiff >= break_win)[0]
# 	srcs_groups_l = []
# 	ind_inc = 0

# 	if len(ibreak) > 0:
# 		for i in range(len(ibreak)):
# 			srcs_groups_l.append(srcs_init[np.arange(ind_inc, ibreak[i] + 1)])
# 			ind_inc = ibreak[i] + 1
# 		if len(np.vstack(srcs_groups_l)) < srcs_init.shape[0]:
# 			srcs_groups_l.append(srcs_init[(ibreak[-1] + 1)::])
# 	else:
# 		srcs_groups_l.append(srcs_init)

# 	print('Begin local marching')
# 	srcs_l = []
# 	scale_depth_clustering = 0.2
# 	for i in range(len(srcs_groups_l)):
# 		if len(srcs_groups_l[i]) == 1:
# 			srcs_l.append(srcs_groups_l[i])
# 		else:
# 			mp = LocalMarching(device = device)
# 			srcs_out = mp(srcs_groups_l[i], ftrns1, tc_win = tc_win, sp_win = sp_win, scale_depth = scale_depth_clustering, n_steps_max = 2, use_directed = False)
# 			# srcs_out = mp(srcs_groups_l[i], ftrns1, tc_win = tc_win, sp_win = sp_win, scale_depth = scale_depth_clustering)
# 			# srcs_out = mp(srcs_groups_l[i], ftrns1, tc_win = 2.5*dt_win, sp_win = 2.5*dist_grid, scale_depth = scale_depth_clustering, use_directed = False, n_steps_max = 5) # tc_win = 2*dt_win, sp_win = 2*dist_offset, scale_depth = scale_depth_clustering, n_steps_max = 5, use_directed = False
# 			if len(srcs_out) > 0:
# 				srcs_l.append(srcs_out)
# 	srcs = np.vstack(srcs_l)

# 	if len(srcs) == 0:
# 		print('No sources detected, finishing script')
# 		continue ## No sources, continue

# 	print('Detected %d number of initial local maxima'%srcs.shape[0])

# 	srcs = srcs[np.argsort(srcs[:,3])]



def compute_loss(x, n_repeat = 5, return_metrics = False):

	## Source threshold x[0]
	## Association threshold x[1]
	## Could also estimate thresholds based on a function


	n_found = 0
	n_target = 0
	n_match = 0
	Srcs = []
	Srcs_trgt = []
	Matches = []
	Ind, Ind1 = [], []
	c1, c2 = 0, 0

	inc = 0
	for n in n_repeat:


		inputs = dataset.getitem__(np.random.choice(len(files_load)))
		## Need to overwrite the data entries in input_tensors
		if use_gradient_loss == False:
			lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l = inputs[1]
		else:
			lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l, Lbls_grad_spc, Lbls_query_grad_spc, Lbls_grad_t, Lbls_query_grad_t = inputs[1]		
		input_tensors_l = inputs[0]
		for j in range(n_batch): input_tensors_l[j][4] = Data(x = spatial_vals_l[j], edge_index = input_tensors_l[j][4]) ## Ideally remove these
		for j in range(n_batch): input_tensors_l[j][5] = Data(x = spatial_vals_l[j], edge_index = input_tensors_l[j][5])
		input_tensors_l = to_gpu(inputs[0]) if device.type == 'cuda' else inputs[0]
		if device.type == 'cuda': pick_lbls_l = to_gpu(pick_lbls_l)

		for i0 in range(len(X_query)):


			if use_gradient_loss == False:
				out = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])
			else:
				# out, grads = mz(*input_tensors)
				out, grads = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])
				grad_grid_src, grad_grid_t, grad_query_src, grad_query_t = grads


			## First detect maxima
			ifind = torch.where(out[1][:,0] > x[0])[0].cpu().detach().numpy()
			mp = LocalMarching(device = device)
			tc_win = pred_params[2]*1.35 ## Note: could learn these clustering windows as well
			sp_win = pred_params[3]*1.35
			scale_depth_clustering = 0.2
			srcs_out = mp(X_query[i0][ifind], ftrns1, tc_win = tc_win, sp_win = sp_win, scale_depth = scale_depth_clustering, n_steps_max = 2, use_directed = False)



			## Apply bipartite matching
			temporal_win_match = src_t_kernel*2.0
			spatial_win_match = src_x_kernel*2.0
			if len(lp_srcs[i0]) > 0:
				matches = maximize_bipartite_assignment(lp_srcs[i0], srcs_out, ftrns1, ftrns2, temporal_win = temporal_win_match, spatial_win = spatial_win_match)[0]
			else:
				matches = np.zeros((0,2))


			n_found += len(srcs_out)
			n_target += len(lp_srcs[i0])
			n_match += len(matches)

			Srcs.append(srcs_out)
			Srcs_trgt.append(lp_srcs[i0])
			if len(matches) > 0:
				Matches.append(matches + np.array([c1, c2]).reshape(1,-1))
			c1 += len(lp_srcs[i0])
			c2 += len(srcs_out)
			Ind.append(inc*np.ones(len(srcs_out))) ## These track which sources are in independent groups (note: could also just offset in time)
			Ind1.append(inc*np.ones(len(lp_srcs[i0])))
			inc += 1


	## Compute metrics
	prec = n_match/n_found
	rec = n_match/(n_match + (n_target - n_match))
	f1 = 2.0*prec*rec/(prec + rec)
	if len(Srcs) > 0: Srcs = np.vstack(Srcs); Ind = np.hstack(Ind)
	if len(Srcs_trgt) > 0: Srcs_trgt = np.vstack(Srcs_trgt); Ind1 = np.hstack(Ind1)
	if len(Matches) > 0: Matches = np.vstack(Matches)

	if return_metrics == False:

		return -1.0*f1

	else:

		return f1, prec, rec, Srcs, Srcs_trgt, Matches, Ind, Ind1 ## Can include detected events


	## Could also compute location uncertainies and residuals











# 	loss_dict = {
# 	'loss_dice1': loss_base1, ## loss_dice1 ## Possibly move loss_dice1 to auxilary loss
# 	'loss_dice2': loss_dice2,
# 	'loss_dice3': loss_dice3,
# 	'loss_dice4': loss_dice4,
# 	'loss_negative': loss_negative*pre_scale_weights2[0],
# 	'loss_mse1': loss_mse1*pre_scale_weights1[0],
# 	'loss_mse2': loss_mse2*pre_scale_weights1[1]
# 	}		

# 	if loss_consistency_flag == True:

# 		loss_dict = {
# 		'loss_dice1': loss_base1, ## loss_dice1 ## Possibly move loss_dice1 to auxilary loss
# 		'loss_dice2': loss_dice2,
# 		'loss_dice3': loss_dice3,
# 		'loss_dice4': loss_dice4,
# 		'loss_negative': loss_negative*pre_scale_weights2[0],
# 		'loss_consistency': loss_consistency*pre_scale_weights2[1],
# 		'loss_mse1': loss_mse1*pre_scale_weights1[0],
# 		'loss_mse2': loss_mse2*pre_scale_weights1[1]
# 		}

# 	else:		

# else:


## Global loss
# # use_global_loss = False
# if (use_global_loss == True)*(i > n_burn_in):

# 	## Find all points far away from true sources
# 	if len(lp_srcs[i0]) > 0:
# 		imask_query = np.where(Lbls_query[i0][:,0] < 0.001)[0]

# 		if use_sigmoid == False:

# 			total_sum = out[1][imask_query].sum()/100.0 # .clamp(min = 0.0).mean()
# 			loss_sum = mse_loss(total_sum, torch.zeros(total_sum.shape).to(device))
# 			# loss = 0.9*loss + 0.1*loss_sum
# 			loss = 0.95*loss + 0.05*loss_sum

# 		else:

# 			total_sum = (torch.sigmoid(out[1][imask_query,1])*out[1][imask_query,0]).sum()/100.0 # .clamp(min = 0.0).mean()
# 			loss_sum = mse_loss(total_sum, torch.zeros(total_sum.shape).to(device))
# 			# loss = 0.9*loss + 0.1*loss_sum
# 			loss = 0.9*loss + 0.1*loss_sum

# 		loss_global_val += 0.1*loss_sum.item()/n_batch



# class RobustMagnitudeBalancer:
#     def __init__(self, anchor_head='dice_3', alpha=0.99):
#         self.alpha = alpha
#         self.values = {}      # EMA of each loss magnitude
#         self.scales = {}
#         self.anchor = anchor_head

#     def __call__(self, losses_dict):
#         total = 0.0
#         anchor_val = None

#         for name, loss in losses_dict.items():
#             val = loss.detach().mean().item()
            
#             if name not in self.values:
#                 self.values[name] = val
#             else:
#                 self.values[name] = (self.alpha * self.values[name] + 
#                                    (1 - self.alpha) * val)
            
#             if name == self.anchor:
#                 anchor_val = self.values[name]

#             scale = anchor_val / (self.values[name] + 1e-8)
#             scale = scale.clamp(0.1, 100.0)
#             self.scales[name] = scale
            
#             total += loss * scale

#         return total


# def pick_labels_extract_interior_region(xq_src_cart, xq_src_t, source_pick, src_slice, lat_range_interior, lon_range_interior, ftrns1, sig_x = 15e3, sig_t = 6.5): # can expand kernel widths to other size if prefered

# 	iz = np.where(source_pick[:,1] > -1.0)[0]
# 	lbl_trgt = torch.zeros((xq_src_cart.shape[0], source_pick.shape[0], 2)).to(device)
# 	src_pick_indices = source_pick[iz,1].astype('int')

# 	inside_interior = ((src_slice[src_pick_indices,0] <= lat_range_interior[1])*(src_slice[src_pick_indices,0] >= lat_range_interior[0])*(src_slice[src_pick_indices,1] <= lon_range_interior[1])*(src_slice[src_pick_indices,1] >= lon_range_interior[0]))

# 	if len(iz) > 0:
# 		d = torch.Tensor(inside_interior.reshape(1,-1)*np.exp(-0.5*(pd(xq_src_cart, ftrns1(src_slice[src_pick_indices,0:3]))**2)/(sig_x**2))*np.exp(-0.5*(pd(xq_src_t.reshape(-1,1), src_slice[src_pick_indices,3].reshape(-1,1))**2)/(sig_t**2))).to(device)
# 		lbl_trgt[:,iz,0] = d*torch.Tensor((source_pick[iz,0] == 0)).to(device).float()
# 		lbl_trgt[:,iz,1] = d*torch.Tensor((source_pick[iz,0] == 1)).to(device).float()

# 	return lbl_trgt


# def dice_loss(pred, trgts, thresh = [0.075], eps = 1e-5):

# 	dim, j = trgts.ndim, 0
# 	if dim == 1:
# 		# for j in range(len(thresh)):
# 		i1 = torch.where(trgts > thresh[j])[0]
# 		i2 = torch.where(trgts <= thresh[j])[0]

# 		dice_pos = 2.0*(pred[i1]*trgts[i1]).sum()/(pred[i1].sum() + trgts[i1].sum() + eps)
# 		dice_neg = 2.0*((1.0 - pred[i2])*(1 - trgts[i2])).sum()/((1.0 - pred[i2]).sum() + (1.0 - trgts[i2]).sum() + eps)
# 		dice_out = 0.5*dice_pos + 0.5*dice_neg

# 	elif dim == 2:

# 		i1, i2 = torch.where(trgts > thresh[j])
# 		i3, i4 = torch.where(trgts <= thresh[j])

# 		dice_pos = 2.0*(pred[i1,i2]*trgts[i1,i2]).sum()/(pred[i1,i2].sum() + trgts[i1,i2].sum() + eps)
# 		dice_neg = 2.0*((1.0 - pred[i3, i4])*(1 - trgts[i3, i4])).sum()/((1.0 - pred[i3, i4]).sum() + (1.0 - trgts[i3, i4]).sum() + eps)
# 		dice_out = 0.5*dice_pos + 0.5*dice_neg

# 	return 1.0 - dice_out ## Loss function

# def dice_loss(pred, trgts, thresh = [0.075], lambda_pos = 0.5, eps = 1e-5):

# 	dim, j = trgts.ndim, 0
# 	if dim == 1:
# 		# for j in range(len(thresh)):
# 		# i1 = torch.where(trgts > thresh[j])[0]
# 		# i2 = torch.where(trgts <= thresh[j])[0]

# 		dice_pos = 2.0*(pred*trgts).sum()/(pred.sum() + trgts.sum() + eps)
# 		dice_neg = 2.0*((1.0 - pred)*(1 - trgts)).sum()/((1.0 - pred).sum() + (1.0 - trgts).sum() + eps)
# 		dice_out = lambda_pos*dice_pos + (1.0 - lambda_pos)*dice_neg

# 	elif dim == 2:

# 		# i1, i2 = torch.where(trgts > thresh[j])
# 		# i3, i4 = torch.where(trgts <= thresh[j])

# 		dice_pos = 2.0*(pred*trgts).sum(1)/(pred.sum(1) + trgts.sum(1) + eps)
# 		dice_neg = 2.0*((1.0 - pred)*(1 - trgts)).sum(1)/((1.0 - pred).sum(1) + (1.0 - trgts).sum(1) + eps)
# 		dice_out = lambda_pos*dice_pos + (1.0 - lambda_pos)*dice_neg
# 		dice_out = dice_out.mean()

# 	return 1.0 - dice_out ## Loss function



# ifind = np.where((np.abs(X_query[i0][:,0:3] - x_query_sample).max(1) == 0)*(X_query[i0][:,3] == x_query_sample_t.reshape(-1)))[0]
# print('Max val %0.4f, %0.4f, %0.4f'%(np.abs(Lbls_query[i0][ifind,0] - lbls_query[ifind,0]).max(), Lbls_query[i0][ifind].max(), lbls_query[ifind].max()))
# print('Len [2] %d'%len(lp_srcs[i0]))
# if np.abs(Lbls_query[i0][ifind,0] - lbls_query[ifind,0]).max() > 0.01:
# 	# pass
# 	moi
# 	pass


# print('loss global %0.8f'%loss_sum)


# imask_query = np.where(Lbls_query[i0][:,0] > 0.01)[0]
# imask_query1 = np.delete(np.arange(len(Lbls_query[i0]), imask_query), axis = 0)
# idistant_from_sources = np.where(cKDTree(ftrns1(X_query[i0][imask_query])).query(ftrns1(X_query[i0][imask_query])) ) # np.where([len(ind_slice) for ind_slice in cKDTree(X_query[i0][imask_query]) ])


# inearest = knn(torch.Tensor(ftrns1(lp_srcs[i0]) ))



# loss_consistency = (weight1*weights[0]*loss_func(out_save[0], out[0]) + weight2*weights[1]*loss_func(out_save[1], out[1]))/torch.maximum(torch.Tensor([1.0]).to(device), (weight1*weights[0] + weight2*weights[1]))
# loss_consistency = loss_func(mask_loss*out_save[0][ind_consistency::], mask_loss*out[1][ind_consistency::]) # )/torch.maximum(torch.Tensor([1.0]).to(device), (weight1*weights[0] + weight2*weights[1]))
# loss_consistency = loss_func1(mask_loss*out_save[0][ind_consistency::], mask_loss*out[1][ind_consistency::]) # )/torch.maximum(torch.Tensor([1.0]).to(device), (weight1*weights[0] + weight2*weights[1]))


## Can check if prediction is consistent at un-changed points
# out[1] = out_query ## Can we overwrite; and also overwrite the query points?
# X_query[i0] = np.concatenate((x_query_sample, x_query_sample_t.reshape(-1,1)), axis = 1)
# Lbls_query[i0] = lbls_query

# pdb.set_trace()

## Also need to overwrite labels. Note: does this break for using gradient loss? Yes


# if pick_lbls.max() > 0.3:
	# np.savez_compressed('example_picks_%d.npz'%i, pick_lbls = pick_lbls.cpu().detach().numpy(), x_query = ftrns2(x_src_query_cart), x_cart = x_src_query_cart, srcs = lp_srcs[i0], srcs_cart = ftrns1(lp_srcs[i0]), times = lp_times[i0], inds = lp_stations[i0], meta = lp_meta[i0][:,-2::], tq_sample = tq_sample, sig_t = src_t_arv_kernel, sig_x = src_x_arv_kernel)

# n_burn_in = 5000
# if i < n_burn_in:

	# # loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1]))/n_batch
	# # loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
	# loss1 = weights[0]*bce_loss(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*bce_loss(out[1], torch.Tensor(Lbls_query[i0]).to(device))
	# loss2 = (weights[2]*bce_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*bce_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch
	# loss = loss1 + loss2



# if Lbls_query[i0][:,5].max() > 0.2: # Plot all true sources
# 	visualize_predictions(out, Lbls_query[i0], pick_lbls, X_query[i0], lp_times[i0], lp_stations[i0], Locs[i0], data, i0, save_plots_path, n_step = i, n_ver = n_ver)
# elif np.random.rand() > 0.8: # Plot a fraction of false sources
# 	visualize_predictions(out, Lbls_query[i0], pick_lbls, X_query[i0], lp_times[i0], lp_stations[i0], Locs[i0], data, i0, save_plots_path, n_step = i, n_ver = n_ver)


# use_dice_loss = False
# if (use_dice_loss == True)*(i > int(n_epochs/5)):


# 	loss_dice1 = dice_loss(out[0], torch.Tensor(Lbls[i0]).to(device))
# 	loss_dice2 = dice_loss(out[1], torch.Tensor(Lbls_query[i0]).to(device))
# 	loss_dice3 = dice_loss(out[2][:,:,0], pick_lbls[:,:,0])
# 	loss_dice4 = dice_loss(out[3][:,:,0], pick_lbls[:,:,1])
# 	loss_dice = (loss_dice1 + loss_dice2 + loss_dice3 + loss_dice4)/5.0

# 	loss = 0.8*loss + 0.2*loss_dice
# 	print('Dice %0.5f %0.5f'%(loss.item(), loss_dice.item()))


	## Can track the norms of the gradients of dice components
	## and then balance the positive contribution (can track over a moving window)
	# ratio = grad_pos / grad_neg
	# lambda_pos = ratio / (1 + ratio)
	# lambda_pos = clamp(lambda_pos, 0.6, 0.95)



	# def dice_loss(p, t, epsilon = 1e-6):
	# 	return (2.0*((p.clamp(min = 0.0, max = 1.0)*t.clamp(min = 0.0, max = 1.0)).sum()) + epsilon)/(p.clamp(min = 0.0, max = 1.0).pow(2).sum() + t.clamp(min = 0.0, max = 1.0).pow(2).sum() + epsilon)

	# def dice_loss1(p, t, epsilon = 1e-6):
	# 	return (2.0*((p.clamp(min = 0.0, max = 1.0)*t.clamp(min = 0.0, max = 1.0)).sum(1)) + epsilon)/(p.clamp(min = 0.0, max = 1.0).pow(2).sum(1) + t.clamp(min = 0.0, max = 1.0).pow(2).sum(1) + epsilon)

	# min_val_trgt = 0.1
	# z_vec = torch.Tensor([1.0]).to(device)
	# dice1 = z_vec if (Lbls[i0].max() < min_val_trgt)*(out[0].max().item() < min_val_trgt) else dice_loss(out[0], torch.Tensor(Lbls[i0]).to(device))
	# dice2 = z_vec if (Lbls_query[i0].max() < min_val_trgt)*(out[1].max().item() < min_val_trgt) else dice_loss(out[1], torch.Tensor(Lbls_query[i0]).to(device))
	# loss_dice1 = (weights[0]*dice1 + weights[1]*dice2)/(weights[0] + weights[1])

	# ## This may not be correct for picks since the max operation tends to mean many of the 
	# ## values will not satsify the mask. Might have to estimate the mask per source and pick
	# mask_dice3 = ((pick_lbls[:,:,0].max(1).values < min_val_trgt)*(out[2][:,:,0].max(1).values < min_val_trgt)).float()
	# mask_dice4 = ((pick_lbls[:,:,1].max(1).values < min_val_trgt)*(out[3][:,:,0].max(1).values < min_val_trgt)).float()
	# dice3 = mask_dice3 + (1.0 - mask_dice3)*dice_loss1(out[2][:,:,0], pick_lbls[:,:,0])
	# dice4 = mask_dice4 + (1.0 - mask_dice4)*dice_loss1(out[3][:,:,0], pick_lbls[:,:,1])
	# loss_dice2 = (weights[2]*dice3.mean() + weights[3]*dice4.mean())/(weights[2] + weights[3])

	# loss_dice = 1.0 - (0.5*loss_dice1 + 0.5*loss_dice2)

	# loss = 0.5*loss + 2.0*(0.5*loss_dice)/500.0 ## Why must the dice loss be scaled so small
	# loss_dice_val += 2*(0.5*loss_dice.item())/500.0/n_batch



# min_val_use = 0.1
# use_sensitivity_loss = False
# loss_regularize = torch.Tensor([0.0]).to(device)
# if (use_sensitivity_loss == True)*(((out[2].max().item() < min_val_use) + (out[3].max().item() < min_val_use)) == False):

# 	# if ((out[2].max().item() < min_val_use) + (out[3].max().item() < min_val_use)):
# 	# 	continue
	
# 	sig_d = 0.15 ## Assumed pick uncertainty (seconds)
# 	chi_pdf = chi2(df = 3).pdf(0.99)

# 	scale_val1 = 100.0*np.linalg.norm(ftrns1(x_src_query[:,0:3]) - ftrns1(x_src_query[:,0:3] + np.array([0.01, 0, 0]).reshape(1,-1)), axis = 1)[0]
# 	scale_val2 = 100.0*np.linalg.norm(ftrns1(x_src_query[:,0:3]) - ftrns1(x_src_query[:,0:3] + np.array([0.0, 0.01, 0]).reshape(1,-1)), axis = 1)[0]
# 	scale_val = 0.5*(scale_val1 + scale_val2)

# 	scale_partials = torch.Tensor((1/60.0)*np.array([1.0, 1.0, scale_val]).reshape(1,-1)).to(device)
# 	src_input_p = Variable(torch.Tensor(x_src_query).repeat_interleave(len(lp_stations[i0]), dim = 0).to(device), requires_grad = True)
# 	src_input_s = Variable(torch.Tensor(x_src_query).repeat_interleave(len(lp_stations[i0]), dim = 0).to(device), requires_grad = True)
# 	trv_out_p = trv_pairwise1(torch.Tensor(Locs[i0][lp_stations[i0].astype('int')]).repeat(len(x_src_query), 1).to(device), src_input_p, method = 'direct')[:,0]
# 	trv_out_s = trv_pairwise1(torch.Tensor(Locs[i0][lp_stations[i0].astype('int')]).repeat(len(x_src_query), 1).to(device), src_input_s, method = 'direct')[:,1]
# 	# trv_out = trv_out[np.arange(len(trv_out)), arrivals[n_inds_picks[i],4].astype('int')] # .cpu().detach().numpy() ## Select phase type
# 	d_p = scale_partials*torch.autograd.grad(inputs = src_input_p, outputs = trv_out_p, grad_outputs = torch.ones(len(trv_out_p)).to(device), retain_graph = True, create_graph = True, allow_unused = True)[0] # .cpu().detach().numpy()
# 	d_s = scale_partials*torch.autograd.grad(inputs = src_input_s, outputs = trv_out_s, grad_outputs = torch.ones(len(trv_out_s)).to(device), retain_graph = True, create_graph = True, allow_unused = True)[0] # .cpu().detach().numpy()
# 	d_p = d_p.reshape(-1, len(lp_stations[i0]), 3).detach() ## Do we detach this
# 	d_s = d_s.reshape(-1, len(lp_stations[i0]), 3).detach() ## Do we detach this
# 	d_grad = torch.Tensor([1000.0]).to(device)*(1.0/scale_partials)*torch.cat((torch.clip(out[2], min = 0.0)*d_p, torch.clip(out[3], min = 0.0)*d_s), dim = 0)/torch.Tensor([scale_val1, scale_val2, 1.0]).to(device).reshape(1,-1)
# 	var_cart = torch.bmm(d_grad.transpose(1,2), d_grad)
# 	# try:
# 	scale_loss = 10000.0
# 	tol_cond = 1000.0
# 	icond = torch.where(torch.linalg.cond(var_cart) < tol_cond)[0]
# 	if len(icond) == 3: icond = torch.cat((icond, torch.Tensor([icond[-1].item()]).to(device)), dim = 0).long()
# 	var_cart_inv = torch.linalg.solve(var_cart[icond], torch.eye(3).to(device))*torch.Tensor([(sig_d**2)*chi_pdf]).to(device)
# 	sigma_cart = torch.norm(var_cart_inv[:,torch.arange(3),torch.arange(3)]**(0.5), dim = 1)
# 	loss_regularize = (0.000002)*loss_func1(sigma_cart/scale_loss, torch.zeros(sigma_cart.shape).to(device)) # 0.001 # 0.0002
# 	if torch.isnan(loss_regularize) == False: 
# 		loss = loss + loss_regularize
# 	# losses_regularize[i] = loss_regularize.item()
# 	loss_regularize_val += loss_regularize.item()
# 	loss_regularize_cnt += 1




# Ac_prod_src_src = (n_sta_slice*Ac_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).to(device)).contiguous()	
## Must use "irregular" version of Cartesian product 

# def build_src_src_product(Ac_src_src, locs, x_grid):

# 	n_sta = len(locs)

# 	A_src_in_sta = torch.Tensor(np.concatenate((np.tile(np.arange(locs.shape[0]), len(x_grid)).reshape(1,-1), np.arange(len(x_grid)).repeat(len(locs), axis = 0).reshape(1,-1)), axis = 0)).long().to(device)
# 	tree_src_in_sta = cKDTree(A_src_in_sta[0].reshape(-1,1).cpu().detach().numpy())
# 	lp_fixed_stas = tree_src_in_sta.query_ball_point(np.arange(locs.shape[0]).reshape(-1,1), r = 0)

# 	degree_of_src_nodes = degree(A_src_in_sta[1])
# 	cum_count_degree_of_src_nodes = np.concatenate((np.array([0]), np.cumsum(degree_of_src_nodes.cpu().detach().numpy())), axis = 0).astype('int')

# 	sta_ind_lists = []
# 	for i in range(x_grid.shape[0]):
# 		ind_list = -1*np.ones(locs.shape[0])
# 		ind_list[A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]].cpu().detach().numpy()] = np.arange(degree_of_src_nodes[i].item())
# 		sta_ind_lists.append(ind_list)
# 	sta_ind_lists = np.hstack(sta_ind_lists).astype('int')

# 	Ac_prod_src_src = []
# 	for i in range(locs.shape[0]):
	
# 		slice_edges = subgraph(A_src_in_sta[1,np.array(lp_fixed_stas[i])], Ac_src_src, relabel_nodes = False)[0].cpu().detach().numpy()

# 		## This can happen when a station is only linked to one source
# 		if slice_edges.shape[1] == 0:
# 			continue

# 		shift_ind = sta_ind_lists[slice_edges*n_sta + i]
# 		assert(shift_ind.min() >= 0)
# 		## For each source, need to find where that station index is in the "order" of the subgraph Cartesian product
# 		Ac_prod_src_src.append(torch.Tensor(cum_count_degree_of_src_nodes[slice_edges] + shift_ind).to(device))

# 	Ac_prod_src_src = torch.Tensor(np.hstack(Ac_prod_src_src)).long().to(device)

# 	return Ac_prod_src_src


# if optimize_training_data == True:

# 	# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

# 	from skopt import gp_minimize

# 	# Load configuration from YAML
# 	with open('process_config.yaml', 'r') as file:
# 		process_config = yaml.safe_load(file)
# 		n_ver_picks = process_config['n_ver_picks']

# 	## If true, run Bayesian optimization to determine optimal training parameters
# 	st_load = glob.glob(path_to_file + 'Picks/19*') # Load years 1900's
# 	st_load.extend(glob.glob(path_to_file + 'Picks/20*')) # Load years 2000's
# 	iarg = np.argsort([int(st_load[i].split(seperator)[-1]) for i in range(len(st_load))])
# 	st_load = [st_load[i] for i in iarg]
# 	st_load_l = []
# 	for i in range(len(st_load)):
# 		st = glob.glob(st_load[i] + seperator + '*ver_%d.npz'%(n_ver_picks))
# 		if len(st) > 0:
# 			st_load_l.extend(st)
# 	print('Loading %d detected files for comparisons'%len(st_load_l))

# 	t_sample_win = 120.0 ## Bins to count picks in, and focus sampling around
# 	windows = [40e3, 150e3, 300e3]
# 	t_win_ball = [10.0, 15.0, 25.0]
# 	n_ver_optimize = 1
# 	n_max_files = 500

# 	if len(st_load_l) > n_max_files:
# 		ichoose = np.sort(np.random.choice(len(st_load_l), size = n_max_files, replace = False))
# 		st_load_l = [st_load_l[j] for j in ichoose]

# 	Trgts_list = []
# 	for n in range(len(st_load_l)):
# 		P = np.load(st_load_l[n])['P']
# 		Trgts_list.append(sample_picks(P, locs, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win))
# 		print('Finished file %d of %d'%(n, len(st_load_l)))

# 	evaluate_bayesian_objective_evaluate = lambda x: evaluate_bayesian_objective(x, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win)

# 	## Now apply Bayesian optimization to training parameters

# 	bounds = [(100.0, 300e3), # spc_random
# 	          (100.0, 300e3), # spc_thresh_rand
# 	          (0.001, 0.3), # coda_rate
# 	          (1.0, 180.0), # coda_win
# 	          (5000.0, 149e3), # dist_range[0]
# 	          (300e3, 800e3), # dist_range[1]
# 	          (5, 250), # max_rate_events
# 	          (5, 250), # max_miss_events
# 	          (0.2, 5.0), # max_false_events # (5, 350)
# 	          (0, 0.25), # miss_pick_fraction[0]
# 	          (0.25, 0.6)] # ] # miss_pick_fraction[0]

# 	optimize = gp_minimize(evaluate_bayesian_objective_evaluate,                  # the function to minimize
# 	                  bounds,      # the bounds on each dimension of x
# 	                  acq_func="EI",      # the acquisition function
# 	                  n_calls=150,         # the number of evaluations of f
# 	                  n_random_starts=100,  # the number of random initialization points
# 	                  noise='gaussian',       # the noise level (optional)
# 	                  random_state=1234, # the random seed
# 	                  initial_point_generator = 'lhs',
# 	                  model_queue_size = 150)

# 	res, Trgts, arrivals = evaluate_bayesian_objective(optimize.x, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win, return_vals = True)

# 	strings = ['spc_random', 'spc_thresh_rand', 'coda_rate', 'coda_win', 'dist_range[0]', 'dist_range[1]', 'max_rate_events', 'max_miss_events', 'max_false_events', 'miss_pick_fraction[0]', 'miss_pick_fraction[0]']
	
# 	np.savez_compressed(path_to_file + 'Grids/%s_optimized_training_data_parameters_ver_%d.npz'%(name_of_project, n_ver_optimize), res = res, x = np.array(optimize.x), arrivals = arrivals, strings = strings)

# 	print('Finished optimized training data')

# 	print('Data set optimized; call the training script again to build training data')
# 	sys.exit()



# def sample_picks(P, locs_abs, t_sample_win = 120.0, windows = [40e3, 150e3, 300e3], t_win_ball = [10.0, 15.0, 25.0]): # windows = [40e3, 150e3, 300e3]

# 	Trgts = []

# 	iunique = np.sort(np.unique(P[:,1]).astype('int'))
# 	lunique = len(iunique)

# 	locs_use = np.copy(locs_abs[iunique]) # Overwrite locs_use

# 	## An additional metric than can be added is the number of stations with a pick 
# 	## within a "convex hull" (or the ratio of stations) connecting the source and stations at different distances..
# 	## Hence, measures how much "filled in" versus "noisy" the foot print of associated stations is
# 	## (would need a reference catalog, and should only sample "large" sources, since reference
# 	## catalog would be biased to large sources)

# 	perm_vec = -1*np.ones(len(locs_abs))
# 	perm_vec[iunique] = np.arange(len(iunique))
# 	perm_vec = perm_vec.astype('int')
# 	P[:,1] = perm_vec[P[:,1].astype('int')] ## Overwrite pick indices
# 	iunique = np.sort(np.unique(P[:,1]).astype('int'))
# 	assert(len(iunique) == lunique)
# 	assert(iunique.min() == 0)
# 	assert(iunique.max() == (lunique - 1))

# 	pw_dist = pd(ftrns1(locs_use), ftrns1(locs_use))

# 	max_t_observed = P[:,0].max()
# 	counts_in_time, bins_in_time = np.histogram(P[:,0], bins = np.arange(0, max_t_observed + 3600, t_sample_win))
# 	upper_fifth_percentile = np.where(counts_in_time >= np.quantile(counts_in_time, 0.95))[0]


# 	tree_times = cKDTree(P[:,[0]])
# 	tree_indices = cKDTree(P[:,[1]])

# 	## [1] Average pick rates

# 	# ifind_per_sta = [np.where(P[:,1] == iunique[j])[0] for j in range(len(iunique))]
# 	# counts_per_sta = [len(ifind_per_sta[j]) for j in range(len(iunique))]
# 	ifind_per_sta = [np.where(P[:,1] == j)[0] for j in range(len(locs_use))]
# 	counts_per_sta = [len(ifind_per_sta[j]) for j in range(len(locs_use))]
# 	counts_per_hour = np.vstack([np.histogram(P[ifind_per_sta[j],0], bins = np.arange(0, max_t_observed + 3600, 3600.0))[0].reshape(1,-1) for j in range(len(ifind_per_sta))])
# 	upper_fifth_percentile_stas = iunique[np.where(counts_per_sta >= np.quantile(counts_per_sta, 0.95))[0]]

# 	Quants_counts = np.quantile(counts_per_hour, np.arange(0.1, 1.0, 0.2), axis = 0)
# 	Trgts.append(np.median(Quants_counts, axis = 1))

# 	## [2] Average "ratio" of picks within narrow spatial windows compared to outside, over max_t window (for random origin times)
# 	# windows = [40e3, 150e3, 300e3] # [0.029238671690285857, 0.07309667922571464, 0.14619335845142928] of pw_dist_max
# 	Ratio_bins = [[] for w in windows]
# 	num_iter = 150
# 	for j in range(num_iter):
# 		for inc, k in enumerate(range(len(windows))):

# 			ipick = np.random.choice(locs_use.shape[0]) ## Pick random station
# 			ifind = np.where(pw_dist[ipick,:] < windows[k])[0] ## Find other stations within window distance
# 			# ifind_outside = np.delete(np.arange(locs_use.shape[0]), ifind, axis = 0) ## Stations outside window distance

# 			## Choose random origin time
# 			t0 = np.random.rand()*3600*24

# 			## Find all picks within t0 + max_t*fraction

# 			fraction = 0.3
# 			ifind_time = np.array(tree_times.query_ball_point(np.array([t0]).reshape(1,1), r = max_t*fraction)[0]).astype('int')
# 			# tree_pick_indices = cKDTree(P[ifind_time,1].reshape(-1,1))
# 			tree_pick_indices = cKDTree(ifind.reshape(-1,1))

# 			## Of these picks, find subset that are nearby root station, and those that are not.
# 			ifind_picks_inside = np.where(tree_pick_indices.query(P[ifind_time,1].astype('int').reshape(-1,1))[0] == 0)[0]
# 			ifind_picks_outside = np.delete(np.arange(len(ifind_time)), ifind_picks_inside, axis = 0) ## Stations outside window distance

# 			Ratio_bins[inc].append(len(ifind_picks_inside)/np.maximum(len(ifind_picks_outside), 1.0))

# 	Ratio_bins = np.vstack([np.quantile(Ratio_bins[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Ratio_bins))])
# 	Trgts.append(Ratio_bins)


# 	## [3] Average "ratio" of picks within narrow spatial windows compared to outside, over max_t window (for "optimal" origin times and stations; e.g., near sources)
# 	# windows = [40e3, 150e3, 300e3] # [0.029238671690285857, 0.07309667922571464, 0.14619335845142928] of pw_dist_max
# 	Ratio_bins1 = [[] for w in windows]
# 	num_iter = 150
# 	prob_counts = 1.0/(1.0 + np.flip(np.argsort(Quants_counts.mean(0))))
# 	prob_counts = prob_counts/prob_counts.sum()
# 	for j in range(num_iter):
# 		for inc, k in enumerate(range(len(windows))):

# 			ipick = np.random.choice(upper_fifth_percentile_stas) ## Pick random station
# 			ifind = np.where(pw_dist[ipick,:] < windows[k])[0] ## Find other stations within window distance
# 			# ifind_outside = np.delete(np.arange(locs_use.shape[0]), ifind, axis = 0) ## Stations outside window distance

# 			## Choose origin time focused on the high pick count time intervals
# 			t0 = bins_in_time[np.random.choice(upper_fifth_percentile)] + np.random.rand()*t_sample_win

# 			## Find all picks within t0 + max_t*fraction

# 			fraction = 0.3
# 			ifind_time = np.array(tree_times.query_ball_point(np.array([t0]).reshape(1,1), r = max_t*fraction)[0]).astype('int')
# 			# tree_pick_indices = cKDTree(P[ifind_time,1].reshape(-1,1))
# 			tree_pick_indices = cKDTree(ifind.reshape(-1,1))

# 			## Of these picks, find subset that are nearby root station, and those that are not.
# 			ifind_picks_inside = np.where(tree_pick_indices.query(P[ifind_time,1].astype('int').reshape(-1,1))[0] == 0)[0]
# 			ifind_picks_outside = np.delete(np.arange(len(ifind_time)), ifind_picks_inside, axis = 0) ## Stations outside window distance

# 			Ratio_bins1[inc].append(len(ifind_picks_inside)/np.maximum(len(ifind_picks_outside), 1.0))

# 	Ratio_bins1 = np.vstack([np.quantile(Ratio_bins1[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Ratio_bins1))])
# 	Trgts.append(Ratio_bins1)

# 	## [4] Counts of station, for each neighboring station, if they have a pick at a similar time
# 	## Instead of random picks, could pick picks nearby times of high activity
# 	k_sta = 1*k_sta_edges + 0 # 10
# 	# locs_use_use = locs_use[iunique]
# 	edges = remove_self_loops(knn(torch.Tensor(ftrns1(locs_use)).to(device)/1000.0, torch.Tensor(ftrns1(locs_use)).to(device)/1000.0, k = k_sta))[0].flip(0).cpu().detach().numpy()

# 	tree_edges = cKDTree(edges[1].reshape(-1,1))

# 	# t_win_ball = [10.0, 15.0, 25.0]
# 	num_picks = 1000
# 	n_picks = len(P)
# 	Ratio_neighbors = [[] for t in t_win_ball]
# 	for j in range(num_picks):
# 		for k in range(len(t_win_ball)):
# 			ichoose = np.random.choice(n_picks)
# 			ifind_ball = np.array(tree_times.query_ball_point(np.array([P[ichoose,0]]).reshape(-1,1), r = t_win_ball[k])[0]).astype('int')
# 			ineighbors = edges[0][np.array(tree_edges.query_ball_point(np.array([P[ichoose,1]]).reshape(-1,1), r = 0)[0])]
# 			size_intersection = len(list(set(ineighbors).intersection(P[ifind_ball,1].astype('int'))))
# 			Ratio_neighbors[k].append(size_intersection/k_sta)

# 	Ratio_neighbors = np.vstack([np.quantile(Ratio_neighbors[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Ratio_neighbors))])
# 	Trgts.append(Ratio_neighbors)

# 	## [5] For each pick, number of times another pick occurs within ~15 seconds, 30 seconds, 45 seconds, etc.
# 	# t_win_ball = [5.0, 10.0, 15.0]
# 	num_picks = 1500
# 	Num_adjacent_picks = [[] for t in t_win_ball]
# 	for j in range(num_picks):
# 		for k in range(len(t_win_ball)):
# 			ichoose = np.random.choice(n_picks)
# 			sta_ind = P[ichoose,1]
# 			ifind_ball = np.array(tree_times.query_ball_point(np.array([P[ichoose,0]]).reshape(-1,1) + t_win_ball[k]/2.0 + 0.1, r = t_win_ball[k]/2.0)[0]).astype('int')
# 			min_sta_dist = (sta_ind == P[ifind_ball,1]).sum()
# 			Num_adjacent_picks[k].append(min_sta_dist)

# 	Num_adjacent_picks = np.vstack([np.quantile(Num_adjacent_picks[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Num_adjacent_picks))])
# 	Trgts.append(Num_adjacent_picks)

# 	## [6] Possibly correlation of pick traces between nearby stations

# 	return Trgts

# def evaluate_bayesian_objective(x, n_random = 30, t_sample_win = 120.0, windows = [40e3, 150e3, 300e3], t_win_ball = [10.0, 15.0, 25.0], return_vals = False): # 	windows = [40e3, 150e3, 300e3], 	


# 	training_params_2[0] = x[0] # spc_random
# 	training_params_2[2] = x[1] # spc_thresh_rand
# 	training_params_2[4] = x[2] # coda_rate
# 	training_params_2[5][1] = x[3] # coda_win

# 	training_params_3[1][0] = x[4] # dist_range[0]
# 	training_params_3[1][1] = x[5] # dist_range[1]
# 	training_params_3[2] = x[6] # max_rate_events
# 	training_params_3[3] = x[7] # max_miss_events
# 	training_params_3[4] = x[6]*x[8] # max_false_events (input of false rate to generate is an absolute number, not the ratio, which is x[8])
# 	training_params_3[5][0] = x[9] # miss_pick_fraction[0]
# 	training_params_3[5][1] = x[10] # miss_pick_fraction[0]

# 	arrivals = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, return_only_data = True)[0]

# 	P = np.copy(arrivals)

# 	Trgts = sample_picks(P, locs, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win)

# 	res1, res2, res3, res4, res5 = 0, 0, 0, 0, 0

# 	for n in range(n_random):

# 		ichoose = np.random.choice(len(Trgts_list))

# 		res1 += np.linalg.norm(Trgts[0] - Trgts_list[ichoose][0])/np.maximum(np.linalg.norm(Trgts_list[ichoose][0]), 1e-5)/n_random
# 		res2 += np.linalg.norm(Trgts[1] - Trgts_list[ichoose][1])/np.maximum(np.linalg.norm(Trgts_list[ichoose][1]), 1e-5)/n_random
# 		res3 += np.linalg.norm(Trgts[2] - Trgts_list[ichoose][2])/np.maximum(np.linalg.norm(Trgts_list[ichoose][2]), 1e-5)/n_random
# 		res4 += np.linalg.norm(Trgts[3] - Trgts_list[ichoose][3])/np.maximum(np.linalg.norm(Trgts_list[ichoose][3]), 1e-5)/n_random
# 		res5 += np.linalg.norm(Trgts[4] - Trgts_list[ichoose][4])/np.maximum(np.linalg.norm(Trgts_list[ichoose][4]), 1e-5)/n_random

# 	res = res1 + res2 + res3 + res4 + res5 ## Residual is average relative residual over all five objectives

# 	print(res)

# 	if return_vals == False:

# 		return res

# 	else:

# 		return res, Trgts, arrivals





## Backup of an old format for splitting up "generate_synthetic_data" into several sub functions


# def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, training_params_1, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, plot_on = False, verbose = False):

# 	if verbose == True:
# 		st = time.time()

# 	k_sta_edges, k_spc_edges, k_time_edges = graph_params
# 	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

# 	n_spc_query, n_src_query = training_params_1
# 	spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max = training_params_2
# 	n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources = training_params_3

# 	## Generate synthetic events
# 	arrivals, src_times, src_positions, src_magnitudes = generate_synthetic_events(locs)

# 	## Check active sources and compute sampling points
# 	arrivals_select, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources, active_sources_per_slice_l, inside_interior, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices = check_active_sources_and_compute_sampling_points(locs, arrivals, src_times, src_positions, src_magnitudes)

# 	## Compute inputs and labels
# 	[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l] = compute_inputs_and_labels(locs, arrivals, src_times, src_positions, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources_per_slice_l, inside_interior, t_win, src_t_kernel, src_x_kernel, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices)

# 	## Initilize source and data variables
# 	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitudes.reshape(-1,1)), axis = 1)
# 	data = [arrivals, srcs, active_sources]

# 	if verbose == True:
# 		print('batch gen time took %0.2f'%(time.time() - st))

# 	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

# def generate_synthetic_events(locs):

# 	assert(np.floor(n_sta_range[0]*locs.shape[0]) > k_sta_edges)

# 	scale_x = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
# 	offset_x = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)
# 	n_sta = locs.shape[0]
# 	locs_tensor = torch.Tensor(locs).to(device)

# 	# t_slice = np.arange(-t_win/2.0, t_win/2.0 + 1.0, 1.0)

# 	tsteps = np.arange(0, T + dt, dt)
# 	tvec = np.arange(-tscale*4, tscale*4 + dt, dt)
# 	tvec_kernel = np.exp(-(tvec**2)/(2.0*(tscale**2)))

# 	p_rate_events = fftconvolve(np.random.randn(2*locs.shape[0] + 3, len(tsteps)), tvec_kernel.reshape(1,-1).repeat(2*locs.shape[0] + 3,0), 'same', axes = 1)
# 	c_cor = (p_rate_events@p_rate_events.T) ## Not slow!
# 	global_event_rate, global_miss_rate, global_false_rate = p_rate_events[0:3,:]

# 	# Process global event rate, to physical units.
# 	global_event_rate = (global_event_rate - global_event_rate.min())/(global_event_rate.max() - global_event_rate.min()) # [0,1] scale
# 	min_add = np.random.rand()*0.25*max_rate_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand()*(0.5*max_rate_events - min_add) + 0.5*max_rate_events
# 	global_event_rate = global_event_rate*scale + min_add

# 	global_miss_rate = (global_miss_rate - global_miss_rate.min())/(global_miss_rate.max() - global_miss_rate.min()) # [0,1] scale
# 	min_add = np.random.rand()*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand()*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
# 	global_miss_rate = global_miss_rate*scale + min_add

# 	global_false_rate = (global_false_rate - global_false_rate.min())/(global_false_rate.max() - global_false_rate.min()) # [0,1] scale
# 	min_add = np.random.rand()*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand()*(0.5*max_false_events - min_add) + 0.5*max_false_events
# 	global_false_rate = global_false_rate*scale + min_add

# 	station_miss_rate = p_rate_events[3 + np.arange(n_sta),:]
# 	station_miss_rate = (station_miss_rate - station_miss_rate.min(1, keepdims = True))/(station_miss_rate.max(1, keepdims = True) - station_miss_rate.min(1, keepdims = True)) # [0,1] scale
# 	min_add = np.random.rand(n_sta,1)*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand(n_sta,1)*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
# 	station_miss_rate = station_miss_rate*scale + min_add

# 	station_false_rate = p_rate_events[3 + n_sta + np.arange(n_sta),:]
# 	station_false_rate = (station_false_rate - station_false_rate.min(1, keepdims = True))/(station_false_rate.max(1, keepdims = True) - station_false_rate.min(1, keepdims = True))
# 	min_add = np.random.rand(n_sta,1)*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand(n_sta,1)*(0.5*max_false_events - min_add) + 0.5*max_false_events
# 	station_false_rate = station_false_rate*scale + min_add

# 	## Sample events.
# 	vals = np.random.poisson(dt*global_event_rate/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average
# 	src_times = np.sort(np.hstack([np.random.rand(vals[j])*dt + tsteps[j] for j in range(len(vals))]))
# 	n_src = len(src_times)
# 	src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
# 	src_magnitudes = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)

# 	if use_shallow_sources == True:
# 		sample_random_depths = gamma(1.75, 0.0).rvs(n_src)
# 		sample_random_grab = np.where(sample_random_depths > 5)[0] # Clip the long tails, and place in uniform, [0,5].
# 		sample_random_depths[sample_random_grab] = 5.0*np.random.rand(len(sample_random_grab))
# 		sample_random_depths = sample_random_depths/sample_random_depths.max() # Scale to range
# 		sample_random_depths = -sample_random_depths*(scale_x[0,2] - 2e3) + (offset_x[0,2] + scale_x[0,2] - 2e3) # Project along axis, going negative direction. Removing 2e3 on edges.
# 		src_positions[:,2] = sample_random_depths

# 	m1 = [0.5761163, -0.21916288]
# 	m2 = 1.15

# 	amp_thresh = 1.0
# 	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

# 	use_uniform_distance_threshold = False
# 	## This previously sampled a skewed distribution by default, not it samples a uniform
# 	## distribution of the maximum source-reciever distances allowed for each event.
# 	if use_uniform_distance_threshold == True:
# 		dist_thresh = np.random.rand(n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
# 	else:
# 		## Use beta distribution to generate more samples with smaller moveouts
# 		# dist_thresh = -1.0*np.log(np.sqrt(np.random.rand(n_src))) ## Sort of strange dist threshold set!
# 		# dist_thresh = (dist_thresh*dist_range[1]/10.0 + dist_range[0]).reshape(-1,1)
# 		dist_thresh = beta(2,5).rvs(size = n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
		
# 	# create different distance dependent thresholds.
# 	dist_thresh_p = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
# 	dist_thresh_s = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None]

# 	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
# 	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

# 	arrivals_theoretical = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()
# 	arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
# 	arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
# 	src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

# 	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
# 	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
# 	arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

# 	t_inc = np.floor(arrivals[:,3]/dt).astype('int')
# 	p_miss_rate = 0.5*station_miss_rate[arrivals[:,1].astype('int'), t_inc] + 0.5*global_miss_rate[t_inc]
# 	idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < dt*p_miss_rate/T)[0]

# 	arrivals = np.delete(arrivals, idel, axis = 0)
# 	n_events = len(src_times)

# 	icoda = np.where(np.random.rand(arrivals.shape[0]) < coda_rate)[0]
# 	if len(icoda) > 0:
# 		false_coda_arrivals = np.random.rand(len(icoda))*(coda_win[1] - coda_win[0]) + coda_win[0] + arrivals[icoda,0] + arrivals[icoda,3]
# 		false_coda_arrivals = np.concatenate((false_coda_arrivals.reshape(-1,1), arrivals[icoda,1].reshape(-1,1), -1.0*np.ones((len(icoda),1)), np.zeros((len(icoda),1)), -1.0*np.ones((len(icoda),1))), axis = 1)
# 		arrivals = np.concatenate((arrivals, false_coda_arrivals), axis = 0)

# 	## Base false events
# 	station_false_rate_eval = 0.5*station_false_rate + 0.5*global_false_rate
# 	vals = np.random.poisson(dt*station_false_rate_eval/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average

# 	# How to speed up this part?
# 	i1, i2 = np.where(vals > 0)
# 	v_val, t_val = vals[i1,i2], tsteps[i2]
# 	false_times = np.repeat(t_val, v_val) + np.random.rand(vals.sum())*dt
# 	false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])
# 	n_false = len(false_times)
# 	false_arrivals = np.concatenate((false_times.reshape(-1,1), false_indices.reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
# 	arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)

# 	n_spikes = np.random.randint(0, high = int(max_num_spikes*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
# 	if n_spikes > 0:
# 		n_spikes_extent = np.random.randint(1, high = n_sta, size = n_spikes) ## This many stations per spike
# 		time_spikes = np.random.rand(n_spikes)*T
# 		sta_ind_spikes = np.hstack([np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)])
# 		sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*spike_time_spread for j in range(n_spikes)])
# 		false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
# 		arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes


# 	## Check which true picks have so much noise, they should be marked as `false picks' for the association labels
# 	if use_stable_association_labels == True:
# 		iz = np.where(arrivals[:,4] >= 0)[0]
# 		noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
# 		iexcess_noise = np.where(np.abs(noise_values) > thresh_noise_max*sig_t*arrivals[iz,0])[0]
# 		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise
# 		if len(iexcess_noise) > 0: ## Set these arrivals to "false arrivals", since noise is so high
# 			arrivals[iz[iexcess_noise],2] = -1
# 			arrivals[iz[iexcess_noise],3] = 0
# 			arrivals[iz[iexcess_noise],4] = -1
# 	else: ## This was the original version
# 		iz = np.where(arrivals[:,4] >= 0)[0]
# 		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]

# 	return arrivals, src_times, src_positions, src_magnitudes

# def check_active_sources_and_compute_sampling_points(locs, arrivals, src_times, src_positions, src_magnitudes):

# 	n_events = len(src_times)
# 	n_sta = locs.shape[0]

# 	## Check which sources are active
# 	source_tree_indices = cKDTree(arrivals[:,2].reshape(-1,1))
# 	lp = source_tree_indices.query_ball_point(np.arange(n_events).reshape(-1,1), r = 0)
# 	lp_backup = [lp[j] for j in range(len(lp))]
# 	n_unique_station_counts = np.array([len(np.unique(arrivals[lp[j],1])) for j in range(n_events)])
# 	active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
# 	non_active_sources = np.delete(np.arange(n_events), active_sources, axis = 0)
# 	src_positions_active = src_positions[active_sources]
# 	src_times_active = src_times[active_sources]
# 	src_magnitudes_active = src_magnitudes[active_sources] ## Not currently used

# 	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))
	
# 	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
# 	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
# 	phase_observed = np.copy(arrivals[:,-1]).astype('int')

# 	if len(iwhere_false) > 0: # For false picks, assign a random phase type
# 		phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))

# 	perturb_phases = True # For true picks, randomly flip a fraction of phases
# 	if (len(phase_observed) > 0)*(perturb_phases == True):
# 		n_switch = int(np.random.rand()*(0.2*len(iwhere_real))) # switch up to 20% phases
# 		iflip = np.random.choice(iwhere_real, size = n_switch, replace = False)
# 		phase_observed[iflip] = np.mod(phase_observed[iflip] + 1, 2)

# 	scale_vec = np.array([1,2*t_win]).reshape(1,-1)

# 	if use_sources == False:
# 		time_samples = np.sort(np.random.rand(n_batch)*T) ## Uniform

# 	elif use_sources == True:
# 		time_samples = src_times_active[np.sort(np.random.choice(len(src_times_active), size = n_batch))]

# 	l_src_times_active = len(src_times_active)
# 	if (use_preferential_sampling == True)*(len(src_times_active) > 1):
# 		for j in range(n_batch):
# 			if np.random.rand() > 0.5: # 30% of samples, re-focus time. # 0.7
# 				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()

# 	time_samples = np.sort(time_samples)

# 	max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))

# 	tree_src_times_all = cKDTree(src_times[:,np.newaxis])
# 	tree_src_times = cKDTree(src_times_active[:,np.newaxis])
# 	lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)
# 	lp_src_times = tree_src_times.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)

# 	st = time.time()
# 	tree = cKDTree(arrivals[:,0][:,None])
# 	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 

# 	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
# 	if len(lp_concat) == 0:
# 		lp_concat = np.array([0]) # So it doesnt fail?

# 	arrivals_select = arrivals[lp_concat]
# 	phase_observed_select = phase_observed[lp_concat]

# 	Trv_subset_p = []
# 	Trv_subset_s = []
# 	Station_indices = []
# 	Grid_indices = []
# 	Batch_indices = []
# 	Sample_indices = []
# 	sc = 0

# 	if (fixed_subnetworks is not None):
# 		fixed_subnetworks_flag = 1
# 	else:
# 		fixed_subnetworks_flag = 0		

# 	active_sources_per_slice_l = []
# 	src_positions_active_per_slice_l = []
# 	src_times_active_per_slice_l = []

# 	for i in range(n_batch):
# 		i0 = np.random.randint(0, high = len(x_grids))
# 		n_spc = x_grids[i0].shape[0]
# 		if use_full_network == True:
# 			n_sta_select = n_sta
# 			ind_sta_select = np.arange(n_sta)

# 		else:
# 			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
# 				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
# 				n_sta_select = len(fixed_subnetworks[isub_network])
# 				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.
			
# 			else:
# 				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
# 				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

# 		Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Station_indices.append(ind_sta_select) # record subsets used
# 		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
# 		Grid_indices.append(i0)
# 		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
# 		sc += len(Sample_indices[-1])

# 		active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

# 		active_sources_per_slice_l.append(active_sources_per_slice)

# 	Trv_subset_p = np.vstack(Trv_subset_p)
# 	Trv_subset_s = np.vstack(Trv_subset_s)
# 	Batch_indices = np.hstack(Batch_indices)

# 	return arrivals_select, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources, active_sources_per_slice_l, inside_interior, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices

# def compute_inputs_and_labels(locs, arrivals, src_times, src_positions, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources_per_slice_l, inside_interior, t_win, src_t_kernel, src_x_kernel, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices):

# 	n_sta = locs.shape[0]

# 	offset_per_batch = 1.5*max_t
# 	offset_per_station = 1.5*n_batch*offset_per_batch

# 	t_slice = np.arange(-t_win/2.0, t_win/2.0 + 1.0, 1.0)
# 	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

# 	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
# 	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
# 	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
# 	n_arvs = arrivals_select.shape[0]

# 	# Rather slow!
# 	iargsort = np.argsort(arrivals_select[:,0])
# 	arrivals_select = arrivals_select[iargsort]
# 	phase_observed_select = phase_observed_select[iargsort]

# 	iwhere_p = np.where(phase_observed_select == 0)[0]
# 	iwhere_s = np.where(phase_observed_select == 1)[0]
# 	n_arvs_p = len(iwhere_p)
# 	n_arvs_s = len(iwhere_s)

# 	query_time_p = Trv_subset_p[:,0] + Batch_indices*offset_per_batch + Trv_subset_p[:,1]*offset_per_station
# 	query_time_s = Trv_subset_s[:,0] + Batch_indices*offset_per_batch + Trv_subset_s[:,1]*offset_per_station

# 	## No phase type information
# 	ip_p = np.searchsorted(arrivals_select[:,0], query_time_p)
# 	ip_s = np.searchsorted(arrivals_select[:,0], query_time_s)

# 	ip_p_pad = ip_p.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_s_pad = ip_s.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
# 	ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
# 	ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

# 	rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)

# 	## With phase type information
# 	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
# 	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

# 	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
# 	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
# 	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

# 	rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)

# 	time_vec_slice = np.arange(k_time_edges)

# 	Inpts = []
# 	Masks = []
# 	Lbls = []
# 	Lbls_query = []
# 	X_fixed = []
# 	X_query = []
# 	Locs = []
# 	Trv_out = []

# 	A_sta_sta_l = []
# 	A_src_src_l = []
# 	A_prod_sta_sta_l = []
# 	A_prod_src_src_l = []
# 	A_src_in_prod_l = []
# 	A_edges_time_p_l = []
# 	A_edges_time_s_l = []
# 	A_edges_ref_l = []

# 	lp_times = []
# 	lp_stations = []
# 	lp_phases = []
# 	lp_meta = []
# 	lp_srcs = []
# 	lp_srcs_active = []

# 	thresh_mask = 0.01
# 	for i in range(n_batch):
# 		# Create inputs and mask
# 		grid_select = Grid_indices[i]
# 		ind_select = Sample_indices[i]
# 		sta_select = Station_indices[i]
# 		n_spc = x_grids[grid_select].shape[0]
# 		n_sta_slice = len(sta_select)

# 		inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
# 		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
# 		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
# 		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
# 		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))

# 		trv_out = x_grids_trv[grid_select][:,sta_select,:] ## Subsetting, into sliced indices.
# 		Inpts.append(inpt[:,sta_select,:]) # sub-select, subset of stations.
# 		Masks.append(1.0*(inpt[:,sta_select,:] > thresh_mask))
# 		Trv_out.append(trv_out)
# 		Locs.append(locs[sta_select])
# 		X_fixed.append(x_grids[grid_select])

# 		## Assemble pick datasets
# 		perm_vec = -1*np.ones(n_sta)
# 		perm_vec[sta_select] = np.arange(len(sta_select))
# 		meta = arrivals[lp[i],:]
# 		phase_vals = phase_observed[lp[i]]
# 		times = meta[:,0]
# 		indices = perm_vec[meta[:,1].astype('int')]
# 		ineed = np.where(indices > -1)[0]
# 		times = times[ineed] ## Overwrite, now. Double check if this is ok.
# 		indices = indices[ineed]
# 		phase_vals = phase_vals[ineed]
# 		meta = meta[ineed]

# 		active_sources_per_slice = np.array(lp_src_times_all[i])[np.array(active_sources_per_slice_l[i])]
# 		ind_inside = np.where(inside_interior[active_sources_per_slice.astype('int')] > 0)[0]
# 		active_sources_per_slice = active_sources_per_slice[ind_inside]

# 		ind_src_unique = np.unique(meta[meta[:,2] > -1.0,2]).astype('int') # ignore -1.0 entries.

# 		if len(ind_src_unique) > 0:
# 			ind_src_unique = np.sort(np.array(list(set(ind_src_unique).intersection(active_sources_per_slice)))).astype('int')

# 		src_subset = np.concatenate((src_positions[ind_src_unique], src_times[ind_src_unique].reshape(-1,1) - time_samples[i]), axis = 1)
# 		if len(ind_src_unique) > 0:
# 			perm_vec_meta = np.arange(ind_src_unique.max() + 1)
# 			perm_vec_meta[ind_src_unique] = np.arange(len(ind_src_unique))
# 			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)
# 			# ifind = np.where(meta[:,2] > -1.0)[0] ## Need to find picks with a source index inside the active_sources_per_slice
# 			ifind = np.where([meta[j,2] in ind_src_unique for j in range(meta.shape[0])])[0]
# 			meta[ifind,-1] = perm_vec_meta[meta[ifind,2].astype('int')] # save pointer to active source, for these picks (in new, local index, of subset of sources)
# 		else:
# 			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)

# 		# Do these really need to be on cuda?
# 		lex_sort = np.lexsort((times, indices)) ## Make sure lexsort doesn't cause any problems
# 		lp_times.append(times[lex_sort] - time_samples[i])
# 		lp_stations.append(indices[lex_sort])
# 		lp_phases.append(phase_vals[lex_sort])
# 		lp_meta.append(meta[lex_sort]) # final index of meta points into 
# 		lp_srcs.append(src_subset)

# 		A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), k = k_sta_edges + 1).flip(0).contiguous())[0]
# 		A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), k = k_spc_edges + 1).flip(0).contiguous())[0]
# 		## Cross-product graph is: source node x station node. Order as, for each source node, all station nodes.

# 		# Cross-product graph, nodes connected by: same source node, connected stations
# 		A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).to(device)).contiguous()
# 		A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).to(device)).contiguous()	

# 		# For each unique spatial point, sum in all edges.
# 		A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1)), dim = 0).to(device).contiguous()

# 		## Sub-selecting from the time-arrays, is easy, since the time-arrays are indexed by station (triplet indexing; )
# 		len_dt = len(x_grids_trv_refs[grid_select])

# 		### Note: A_edges_time_p needs to be augmented: by removing stations, we need to re-label indices for subsequent nodes,
# 		### To the "correct" number of stations. Since, not n_sta shows up in definition of edges. "assemble_pointers.."
# 		A_edges_time_p = x_grids_trv_pointers_p[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
# 		A_edges_time_s = x_grids_trv_pointers_s[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
# 		## Need to convert these edges again. Convention is:
# 		## subtract i (station index absolute list), divide by n_sta, mutiply by N stations, plus ith station (in permutted indices)
# 		# shape is len_dt*k_time_edges*len(sta_select)
# 		one_vec = np.repeat(sta_select*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere
# 		A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
# 		A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
# 		# print('permute indices 1')
# 		assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
# 		assert(A_edges_time_s.max() < n_spc*n_sta_slice)

# 		A_sta_sta_l.append(A_sta_sta.cpu().detach().numpy())
# 		A_src_src_l.append(A_src_src.cpu().detach().numpy())
# 		A_prod_sta_sta_l.append(A_prod_sta_sta.cpu().detach().numpy())
# 		A_prod_src_src_l.append(A_prod_src_src.cpu().detach().numpy())
# 		A_src_in_prod_l.append(A_src_in_prod.cpu().detach().numpy())
# 		A_edges_time_p_l.append(A_edges_time_p)
# 		A_edges_time_s_l.append(A_edges_time_s)
# 		A_edges_ref_l.append(x_grids_trv_refs[grid_select])

# 		x_query = np.random.rand(n_spc_query, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.

# 		if len(lp_srcs[-1]) > 0:
# 			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]

# 		if len(active_sources_per_slice) == 0:
# 			lbls_grid = np.zeros((x_grids[grid_select].shape[0], len(t_slice)))
# 			lbls_query = np.zeros((n_spc_query, len(t_slice)))
# 		else:
# 			active_sources_per_slice = active_sources_per_slice.astype('int')

# 			lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
# 			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

# 		X_query.append(x_query)
# 		Lbls.append(lbls_grid)
# 		Lbls_query.append(lbls_query)

# 	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l] # , data











































