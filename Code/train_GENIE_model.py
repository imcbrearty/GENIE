
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
from collections import defaultdict
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
# if use_variable_domain == FalseL
# kernel_sig_t = train_config['kernel_sig_t'] # Kernel to embed arrival time - theoretical time misfit (s)
# src_t_kernel = train_config['src_t_kernel'] # Kernel or origin time label (s)
# src_t_arv_kernel = train_config['src_t_arv_kernel'] # Kernel for arrival association time label (s)
# src_x_kernel = train_config['src_x_kernel'] # Kernel for source label, horizontal distance (m)
# src_x_arv_kernel = train_config['src_x_arv_kernel'] # Kernel for arrival-source association label, horizontal distance (m)
# src_depth_kernel = train_config['src_depth_kernel'] # Kernel of Cartesian projection, vertical distance (m)
# # t_win = config['t_win'] ## This is the time window over which predictions are made. Shouldn't be changed for now.
# src_kernel_mean = np.mean([src_x_kernel, src_x_kernel, src_depth_kernel])
# src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.


n_batches_per_job = train_config['n_batches_per_job']

use_real_data = train_config.get('use_real_data', True)
n_real_fraction = train_config.get('n_real_fraction', 0.3)

use_fixed_graphs = train_config.get('use_fixed_graphs', True) # True # True
use_variable_domain = train_config.get('use_variable_domain', True) # True
if use_variable_domain == True: assert(use_fixed_graphs == True)

if (use_variable_domain == False) or (1 == 1): ## Initilize so inputs exist

	kernel_sig_t = train_config['kernel_sig_t'] # Kernel to embed arrival time - theoretical time misfit (s)
	src_t_kernel = train_config['src_t_kernel'] # Kernel or origin time label (s)
	src_t_arv_kernel = train_config['src_t_arv_kernel'] # Kernel for arrival association time label (s)
	src_x_kernel = train_config['src_x_kernel'] # Kernel for source label, horizontal distance (m)
	src_x_arv_kernel = train_config['src_x_arv_kernel'] # Kernel for arrival-source association label, horizontal distance (m)
	src_depth_kernel = train_config['src_depth_kernel'] # Kernel of Cartesian projection, vertical distance (m)
	# t_win = config['t_win'] ## This is the time window over which predictions are made. Shouldn't be changed for now.
	src_kernel_mean = np.mean([src_x_kernel, src_x_kernel, src_depth_kernel])
	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

	## Else must load these per file



if use_real_data == True: ## If using real data, mask the labels near the source (don't center window on source, but just center mask; in time, and possibly in space)
	n_reference_ver = 1
	# n_real_fraction = 0.5
	st1 = glob.glob(path_to_file + 'Calibration/19*') ## Assuming years are 1900 and 2000's
	st2 = glob.glob(path_to_file + 'Calibration/20*')
	st = np.concatenate((st1, st2), axis = 0)
	st_calibration = np.hstack([glob.glob(s + '/*ver_%d.npz'%n_reference_ver) for s in st])
	if len(st_calibration) == 0: 
		use_real_data = False

	else:
		dates_calibration = np.vstack([[int(j) for j in s.strip().split('/')[-1].split('_')[2:5]] for s in st_calibration])
		iarg = np.lexsort((dates_calibration[:,0], dates_calibration[:,1], dates_calibration[:,2]))
		dates_calibration = dates_calibration[iarg]
		st_calibration = st_calibration[iarg]
		
		iperm_list = np.random.permutation(len(st_calibration))
		st_calibration = st_calibration[iperm_list]
		dates_calibration = dates_calibration[iperm_list]

	# st1_picks = glob.glob(path_to_file + 'Picks/19*') ## Assuming years are 1900 and 2000's
	# st2_picks = glob.glob(path_to_file + 'Picks/20*')
	# st_picks = np.concatenate((st1_picks, st2_picks), axis = 0)	
	# st_picks = np.hstack([glob.glob(s + '/*ver_%d.npz'%1) for s in st_picks]) # n_ver_picks = 1
	# dates_picks = np.vstack([[int(j) for j in s.strip().split('/')[-1].split('_')[0:3]] for s in st_picks])

	# tree_dates = cKDTree(dates_picks)
	# ifound = np.where(tree_dates.query(dates_calibration)[0] == 0)[0]

	# st_calibration = st_calibration[ifound]
	# dates_calibration = dates_calibration[ifound]


if use_fixed_graphs == True:
	# st_graphs = glob.glob('Domains/*graph*')
	st_graphs = glob.glob('Domains/*domain_file*')

# 	dates_allowed = np.vstack([[int(j) for j in s.strip().split('date_')[1].split('_')[0:3]] for s in st_graphs])
# 	tree_dates = cKDTree(dates_allowed)
# 	ifound = np.where(tree_dates.query(dates_calibration)[0] == 0)[0]
# 	st_calibration = st_calibration[ifound]
# 	dates_calibration = dates_calibration[ifound]


if (use_variable_domain == False) or (1 == 1):

	use_adaptive_window = True
	if use_adaptive_window == True:
		n_resolution = 9 ## The discretization of the source time function output
		t_win = np.round(np.copy(np.array([2*src_t_kernel]))[0], 2) ## Set window size to the source kernel width (i.e., prediction window is of length +/- src_t_kernel, or [-src_t_kernel + t0, t0 + src_t_kernel])
		dt_win = np.diff(np.linspace(-t_win/2.0, t_win/2.0, n_resolution))[0]
	else:
		dt_win = 1.0 ## Default version
		t_win = 10.0

	## Else; load variable per input


## Dataset parameters
load_training_data = train_config['load_training_data']
build_training_data = train_config['build_training_data'] ## If try, will use system argument to build a set of data
optimize_training_data = train_config['optimize_training_data']


max_number_pick_association_labels_per_sample = config['max_number_pick_association_labels_per_sample']
make_visualize_predictions = config['make_visualize_predictions']


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


use_consistency_loss = False
use_gradient_loss = train_config['use_gradient_loss']
init_gradient_loss = False
use_negative_loss = True ## If True, up-sample the false positive predictions 
use_negative_loss_step = 1


use_teleseisim_noise = False
if use_teleseisim_noise == True:
	z = np.load(path_to_file + 'Grids' + seperator + 'teleseismic_travel_time_grid_ver_1.npz')
	xx_teleseism, trv_teleseism, phase_types = z['xx_teleseism'], z['trv_teleseism'], z['phase_types']
	z.close()
	ipos = np.where(xx_teleseism[:,0] > 0)[0]
	# inot_nan1, inot_nan2 = np.where(np.isnan(trv_teleseism) == 0)
	xx_teleseism = xx_teleseism[ipos]
	trv_teleseism = trv_teleseism[ipos]



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
		
		cnt_files = 0
		for i in range(len(st)):
			cnt_files += len(glob.glob(st[i] + '/*.npz'))
	
		prob_use = 5.0*n_batches_per_job/cnt_files

		for i in range(len(st)):
			st1 = glob.glob(st[i] + '/*.npz')
			for j in range(len(st1)):

				if np.random.rand() > prob_use:
					continue

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

if (use_variable_domain == False) or (1 == 1):

	z = np.load(path_to_file + '%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	x_grids = z['x_grids']
	scale_time = z['scale_time']/1000.0
	time_shift_range = z['time_shift_range']
	Ac = z['Ac']
	z.close()

	# Load stations
	z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

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
		# z = np.load(path_to_file + 'Grids' + seperator + 'grid_time_shift_ver_1.npz')
		# time_shifts = z['time_shifts'] ## Shape (n_grids, n_nodes, n_times)
		time_shifts = x_grids[:,:,[3]]
		# z.close()
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



## Create path to write files
seperator = '\\' if '\\' in path_to_file else '/'
write_training_file = path_to_file + 'GNN_TrainedModels' + seperator + name_of_project + '_'


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

if (use_variable_domain == False) or (1 == 1):
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

def simulate_travel_times(locs, prob_vec, chol_params, ftrns1, n_samples = 100, use_l1 = False, srcs = None, mags = None, ichoose = None, locs_use_list = None, ind_use_slice = None, return_features = True): # n_repeat : can repeatedly draw either from the covariance matrices, or the binomial distribution

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

	global dt_win, time_shift_range
	n_spc_query, n_src_query = training_params
	spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max, min_misfit_allowed, total_bias = training_params_2
	n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, miss_pick_fraction, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources, use_extra_nearby_moveouts = training_params_3
	
	if use_variable_domain == True:

		ichoose_domain = np.random.choice(st_graphs)
		zfile = np.load(ichoose_domain)
		# ind_file = int(ichoose_domain.split('/')[-1].split('_')[2])

		ind_file, yr, mo, dy = [int(j) for j in ichoose_domain.strip().split('/')[-1].split('_')[2:6]]
		# ind_file1 = 

		z = np.load('Picks/%d/%d_%d_%d_ver_%d.npz'%(yr, yr, mo, dy, ind_file))
		# z = np.load('Domains/domain_slice_%d_ver_1.npz'%ind_file)
		P_ref = z['P']
		date = z['date']
		icorrupt = z['icorrupt']
		z.close()
		# date = np.array([int(j) for j in ichoose_domain.split('/')[-1].split('_')[1:4]])
		kernel_sig_t = zfile['sigma_input']
		src_x_kernel = zfile['source_label_width']
		src_t_kernel = zfile['source_label_width_t']
		src_depth_kernel = zfile['source_label_width']
		src_x_arv_kernel = zfile['association_label_width']
		src_t_arv_kernel = zfile['association_label_width_t']
		# kernel_sig_t = z['association_label_width_t']
		locs = zfile['locs_use']
		stas = zfile['stas_use']
		scale_time = zfile['scale_time']/1000.0
		time_shift_range = zfile['time_shift_range']
		lat_range = zfile['lat_range']
		lon_range = zfile['lon_range']
		lat_range_extend = zfile['lat_range_extend']
		lon_range_extend = zfile['lon_range_extend']
		depth_range = zfile['depth_range']
		x_grid = zfile['x_grid']
		depth_boost = zfile['depth_boost']
		rbest = zfile['rbest']
		mn = zfile['mn']
		if use_expanded == True:
			Ac = zfile['Ac']
		
		# zfile.close()

		# z.close()

		# params_extra = [locs, stas, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, rbest, mn, ftrns1, ftrns2] # = params_extra

		# params_extra = [lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, t_win, dt_win, time_shift_range, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, rbest, mn, ftrns1, ftrns2] # = params_extra


		# use_adaptive_window = True
		if use_adaptive_window == True:
			n_resolution = 9 ## The discretization of the source time function output
			t_win = np.round(np.copy(np.array([2*src_t_kernel]))[0], 2) ## Set window size to the source kernel width (i.e., prediction window is of length +/- src_t_kernel, or [-src_t_kernel + t0, t0 + src_t_kernel])
			dt_win = np.diff(np.linspace(-t_win/2.0, t_win/2.0, n_resolution))[0]
		else:
			dt_win = 1.0 ## Default version
			t_win = 10.0


		x_grids = np.expand_dims(x_grid, axis = 0)

		x_grids_trv = compute_travel_times(trv, locs, x_grids, device = device)


		scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
		offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)

		scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
		offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

		rbest_cuda = torch.Tensor(rbest).to(device)
		mn_cuda = torch.Tensor(mn).to(device)

		## Can load per sample Ac
		# if use_expanded == True:
		# 	Ac = np.load(path_to_file + 'Grids/%s_seismic_network_expanders_ver_%d.npz'%(name_of_project, template_ver))['Ac']
		# else:
		# 	Ac = False

		if use_time_shift == True:
			# z = np.load(path_to_file + 'Grids' + seperator + 'grid_time_shift_ver_1.npz')
			# time_shifts = z['time_shifts'] ## Shape (n_grids, n_nodes, n_times)
			time_shifts = x_grids[:,:,[3]]
			# z.close()
		else:
			time_shifts = None # np.zeros((x_grids.shape[0], x_grids.shape[1]))


		if use_time_shift == True:
			for i in range(len(x_grids_trv)):
				x_grids_trv[i] = x_grids_trv[i] + time_shifts[i].reshape(-1,1,1)
			# print('Appending time shifts')


		time_shift_range = np.max([time_shifts[j].max() - time_shifts[j].min() for j in range(len(time_shifts))])

		max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))])))
		min_t = float(np.floor(min([x_grids_trv[i].min() for i in range(len(x_grids_trv))]))) if use_time_shift == True else 0.0

		for i in range(len(x_grids)):
			
			## Note, this definition of dt and win must match the definition used in process_continous_days
			A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(x_grids_trv[i], k = k_time_edges, max_t = max_t, min_t = min_t, dt = kernel_sig_t/5.0, win = kernel_sig_t*2.0)

			if config['train_travel_time_neural_network'] == False:
				assert(x_grids_trv[i].min() > 0.0)
				assert(x_grids_trv[i].max() < (ts_max_val + 3.0))

			x_grids_trv_pointers_p.append(A_edges_time_p)
			x_grids_trv_pointers_s.append(A_edges_time_s)
			x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?


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





	n_sta_range[0] = np.maximum(n_sta_range[0], k_sta_edges/locs.shape[0])

	assert(np.floor(n_sta_range[0]*locs.shape[0]) >= k_sta_edges)

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

	use_real_data_sample = True if (use_real_data == True)*(np.random.rand() < n_real_fraction) else False
	if use_real_data_sample == True:
		if np.abs(dates_calibration - date.reshape(1,-1)).max(1).min() > 0:
			use_real_data_sample = False

		if icorrupt == True:
			# icorrupt
			use_real_data_sample = False
			assert(len(P_ref) < 10)


	if use_real_data_sample == True:
		# min_magnitude = 2.5

		if use_variable_domain == False:
			ichoose_data = np.random.choice(len(st_calibration))
			date_choose = dates_calibration[ichoose_data]
			sdata = st_calibration[ichoose_data]

		else:
			date_choose = np.copy(date)
			sdata = path_to_file + 'Calibration' + seperator + '%d'%date_choose[0] + seperator + '%s_reference_%d_%d_%d_ver_1.npz'%(name_of_project, date_choose[0], date_choose[1], date_choose[2])

			## Use ichoose_data consistent with the merged graph file

		srcs_known = np.load(sdata)['srcs_ref']

		if use_variable_domain == True:
			ikeep_srcs = np.where((srcs_known[:,0] < lat_range[1])*(srcs_known[:,0] > lat_range[0])*(srcs_known[:,1] < lon_range[1])*(srcs_known[:,1] > lon_range[0]))[0]
			srcs_known = srcs_known[ikeep_srcs]

		# P_ref = np.load('Picks/%d/%d_%d_%d_ver_1.npz'%(date_choose[0], date_choose[0], date_choose[1], date_choose[2]))['P']
		n_src = len(srcs_known)
		src_positions = srcs_known[:,0:3]
		src_magnitude = srcs_known[:,4]
		src_times = srcs_known[:,3]
		T = P_ref[:,0].max()
		# z.close()

	if len(src_positions) == 0:
		src_times = np.array([np.random.rand()*T])
		n_src = len(src_times)
		src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
		src_magnitude = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)

	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

	use_uniform_distance_threshold = False
	## This previously sampled a uniform distribution by default, now it samples a skewed
	## distribution of the maximum source-reciever distances allowed for each event.


	if use_variable_domain == True:
		fixed_dist_range = [0.05, 0.5]
		# min_dist_thresh = 15e3
		# max_dist_thresh =
		dist_range = pd(ftrns1(x_grid[:,0:3]), ftrns1(x_grid[:,0:3])) # [x, x1] ## Set dist_range proportional to domain
		# dist_range = [max(min_dist_thresh, fixed_dist_range[0]*dist_range.max()), fixed_dist_range[1]*dist_range.max()]
		dist_range = [fixed_dist_range[0]*dist_range.max(), fixed_dist_range[1]*dist_range.max()]

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

	## Use a maximum residual time on associated picks
	## Iterate the scatter filtering
	## Possibly use only largest disconnected component of subgraph for associated subset
	## Force uniqueness of picks between both P and S (e.g., only assigned to at most one P or S)


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
		_, _, _, Simulated_p, Simulated_s, Mean_trv_p, Mean_trv_s, Std_val_p, Std_val_s, _, _ = simulate_travel_times(locs, [], chol_params_trv, ftrns1, srcs = src_positions, locs_use_list = locs_use_list, ind_use_slice = ind_use_slice, return_features = False)
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

	if use_real_data_sample == False:

		arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
		arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
		src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

		## Save the excess noise mask in the fourth column instead of the origin time; after using this mask, can overwrite back to the origin time
		# arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
		# arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
		arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), mask_excess_noise[ikeep_p1, ikeep_p2, 0].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
		arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), mask_excess_noise[ikeep_s1, ikeep_s2, 1].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
		arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

	else:

		# min_time = 45.0 ## Both this time
		min_rel_error = 0.2 ## and this minimum relative error
		min_rel_time = 1.0
		min_ratio_neighbors = 0.2
		max_abs_error = 5.0

		trv_out_pred_base = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()
		# trv_out_pred_base[trv_out_pred_base > max_t] = np.nan
		trv_out_pred = trv_out_pred_base + src_times.reshape(-1,1,1)
		trv_time_p1 = np.concatenate((trv_out_pred[:,:,0].reshape(-1,1), np.arange(len(locs)).reshape(1,-1).repeat(len(src_positions), axis = 0).reshape(-1,1), np.arange(len(src_positions)).reshape(-1, 1).repeat(len(locs), axis = 1).reshape(-1,1)), axis = 1)
		trv_time_p2 = np.concatenate((trv_out_pred[:,:,1].reshape(-1,1), np.arange(len(locs)).reshape(1,-1).repeat(len(src_positions), axis = 0).reshape(-1,1), np.arange(len(src_positions)).reshape(-1, 1).repeat(len(locs), axis = 1).reshape(-1,1)), axis = 1)

		# trv_time_p1 = np.concatenate((trv_out_pred[:,:,0].reshape(-1,1), np.arange(len(locs)).reshape(1,-1).repeat(len(src_positions), axis = 0).reshape(-1,1), np.arange(len(locs)).reshape(1, -1).repeat(len(src_positions), axis = 0).reshape(-1,1)), axis = 1)
		# trv_time_p2 = np.concatenate((trv_out_pred[:,:,1].reshape(-1,1), np.arange(len(locs)).reshape(1,-1).repeat(len(src_positions), axis = 0).reshape(-1,1), np.arange(len(locs)).reshape(1, -1).repeat(len(src_positions), axis = 0).reshape(-1,1)), axis = 1)

		## For each simulated travel time, see if a pick exists in the pick dataset P_ref
		tree_picks = cKDTree(P_ref[:,0:2]*np.array([1.0, 3600.0*24.0*1.5]).reshape(1,-1))
		ip_query1 = tree_picks.query(np.nan_to_num(trv_time_p1[:,0:2], nan = -3600.0*24.0*1.5)*np.array([1.0, 3600.0*24.0*1.5]).reshape(1,-1))
		ip_query2 = tree_picks.query(np.nan_to_num(trv_time_p2[:,0:2], nan = -3600.0*24.0*1.5)*np.array([1.0, 3600.0*24.0*1.5]).reshape(1,-1))

		## Should add stable relative error
		# ifind1 = np.where((ip_query1[0] < min_time)*(np.abs(ip_query1[0]/np.maximum(1.0, trv_out_pred_base[:,:,0].reshape(-1))) < min_rel_error)*(np.isnan(trv_out_pred_base[:,:,0].reshape(-1)) == 0))[0]
		# ifind2 = np.where((ip_query2[0] < min_time)*(np.abs(ip_query2[0]/np.maximum(1.0, trv_out_pred_base[:,:,1].reshape(-1))) < min_rel_error)*(np.isnan(trv_out_pred_base[:,:,1].reshape(-1)) == 0))[0]
		# ifind1 = np.where((np.abs(ip_query1[0]/np.maximum(1.0, trv_out_pred_base[:,:,0].reshape(-1))) < min_rel_error)*(np.isnan(trv_out_pred_base[:,:,0].reshape(-1)) == 0))[0]
		# ifind2 = np.where((np.abs(ip_query2[0]/np.maximum(1.0, trv_out_pred_base[:,:,1].reshape(-1))) < min_rel_error)*(np.isnan(trv_out_pred_base[:,:,1].reshape(-1)) == 0))[0]
		ifind1 = np.where((np.abs(ip_query1[0]/np.maximum(1.0, trv_out_pred_base[:,:,0].reshape(-1))) < min_rel_error)*(np.isnan(trv_out_pred_base[:,:,0].reshape(-1)) == 0)*(ip_query1[0] < max_abs_error))[0]
		ifind2 = np.where((np.abs(ip_query2[0]/np.maximum(1.0, trv_out_pred_base[:,:,1].reshape(-1))) < min_rel_error)*(np.isnan(trv_out_pred_base[:,:,1].reshape(-1)) == 0)*(ip_query2[0] < max_abs_error))[0]


		ifind11 = np.where((ip_query1[0] < min_rel_time)*(np.isnan(trv_out_pred_base[:,:,0].reshape(-1)) == 0)*(ip_query1[0] < max_abs_error))[0]
		ifind21 = np.where((ip_query2[0] < min_rel_time)*(np.isnan(trv_out_pred_base[:,:,1].reshape(-1)) == 0)*(ip_query2[0] < max_abs_error))[0]
		ifind1 = np.unique(np.concatenate((ifind1, ifind11), axis = 0))
		ifind2 = np.unique(np.concatenate((ifind2, ifind21), axis = 0))

		## Filter the retained stations based on neighbor relationships
		## E.g., use scatter for each source, for each station, based on neighbors, and only retain those picks that > thresh proportion of neighbors
		## Can use batching for efficiency
		ind_unique = np.unique(P_ref[:,1].astype('int'))
		edges_sta = torch.Tensor(ind_unique).long().to(device)[remove_self_loops(knn(torch.Tensor(ftrns1(locs[ind_unique])/1000.0).to(device), torch.Tensor(ftrns1(locs[ind_unique])/1000.0).to(device), k = k_sta_edges + 1))[0].flip(0).contiguous()]
		edges_sta_repeat = torch.hstack([edges_sta + j*len(locs) for j in range(len(src_positions))]).to(device)

		converged_pooling, inc_pool = False, 0
		while (converged_pooling == False)*(inc_pool < 15):

			## Possibly do pooling between both P and S (e.g., mix the phase types)
			## Since sometimes, multiple adjacent stations will miss S picks, but clearly have P picks
			## Could also increase k_sta_edges

			n_int_total = len(ifind1) + len(ifind2)
			feature_vals_p = torch.zeros((len(src_positions)*len(locs)),1).to(device)
			feature_vals_s = torch.zeros((len(src_positions)*len(locs)),1).to(device)
			feature_vals_p[ifind1,0] = 1.0
			feature_vals_s[ifind2,0] = 1.0

			feature_pool_p = (feature_vals_p*scatter(feature_vals_p[edges_sta_repeat[0]], edges_sta_repeat[1], dim = 0, dim_size = len(src_positions)*len(locs), reduce = 'sum')).cpu().detach().numpy() # .reshape(len(src_positions), len(locs))
			feature_pool_s = (feature_vals_s*scatter(feature_vals_s[edges_sta_repeat[0]], edges_sta_repeat[1], dim = 0, dim_size = len(src_positions)*len(locs), reduce = 'sum')).cpu().detach().numpy() # .reshape(len(src_positions), len(locs))

			ifind1 = np.where(feature_pool_p >= max(1, np.round(k_sta_edges*min_ratio_neighbors)))[0]
			ifind2 = np.where(feature_pool_s >= max(1, np.round(k_sta_edges*min_ratio_neighbors)))[0]
			if len(ifind1) + len(ifind2) == n_int_total: converged_pooling = True

			inc_pool += 1

		print('Converged %d'%inc_pool)
		# print(inc_pool)


		ikeep_p1, ikeep_p2 = trv_time_p1[ifind1,2].astype('int'), trv_time_p1[ifind1,1].astype('int')
		ikeep_s1, ikeep_s2 = trv_time_p2[ifind2,2].astype('int'), trv_time_p2[ifind2,1].astype('int')

		## Use only one assignment for each pick (note, should make this an optimal selection)
		unique_ind1 = np.unique(ip_query1[1][ifind1])
		tree_ind1 = cKDTree(ip_query1[1][ifind1].reshape(-1,1))
		imatch_ind1 = tree_ind1.query(unique_ind1.reshape(-1,1))[1]
		ifind1 = ifind1[imatch_ind1]
		ikeep_p1, ikeep_p2 = ikeep_p1[imatch_ind1], ikeep_p2[imatch_ind1]

		unique_ind2 = np.unique(ip_query2[1][ifind2])
		tree_ind2 = cKDTree(ip_query2[1][ifind2].reshape(-1,1))
		imatch_ind2 = tree_ind2.query(unique_ind2.reshape(-1,1))[1]
		ifind2 = ifind2[imatch_ind2]
		ikeep_s1, ikeep_s2 = ikeep_s1[imatch_ind2], ikeep_s2[imatch_ind2]

		arrivals_theoretical = np.nan*np.ones((len(src_positions), len(locs), 2))
		arrivals_theoretical[ikeep_p1, ikeep_p2, 0] = P_ref[ip_query1[1][ifind1],0]
		arrivals_theoretical[ikeep_s1, ikeep_s2, 1] = P_ref[ip_query2[1][ifind2],0]

		## Need to add the "relative neighborhood" check to keep positive labels. E.g., within time buffer, within relative error, and with enough of nearest neighbor stations with positive associations
		## Note that the neighbrhood check might have to be iterative, or a integer linear programming problem must be solved (e.g., if removing some picks, might impy other picks also need to be removed, e.g., constraints)

		# arrivals_indices = np.nan*np.ones((len(src_positions), len(locs), 2))
		# arrivals_theoretical[ikeep_p1, ikeep_p2, 0] = P_ref[ip_query1[0][ifind1],0]
		# arrivals_theoretical[ikeep_s1, ikeep_s2, 1] = P_ref[ip_query2[0][ifind2],0]

		mask_excess_noise = np.expand_dims(src_times.reshape(-1,1).repeat(len(locs), axis = 1), axis = 2).repeat(2, axis = 2)

		## Note, removing origin times here, but they'll be added back
		arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1) - src_times[ikeep_p1].reshape(-1,1), ikeep_p2.reshape(-1,1), ikeep_p1.reshape(-1,1), mask_excess_noise[ikeep_p1, ikeep_p2, 0].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
		arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1) - src_times[ikeep_s1].reshape(-1,1), ikeep_s2.reshape(-1,1), ikeep_s1.reshape(-1,1), mask_excess_noise[ikeep_s1, ikeep_s2, 1].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
		# pdb.set_trace()
		arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)
		# false_arrivals = 
		inoise = np.delete(np.arange(len(P_ref)), np.unique(np.concatenate((ip_query1[1][ifind1], ip_query2[1][ifind2]), axis = 0)))

		print('Assigned %d real picks in %d sources (%d average) of all %d picks'%(len(arrivals), len(src_times), (len(arrivals)/len(src_times))/2.0, len(P_ref)))



	if len(arrivals) == 0:
		arrivals = -1*np.zeros((1,5))
		arrivals[0,0] = np.random.rand()*T
		arrivals[0,1] = int(np.floor(np.random.rand()*(locs.shape[0] - 1)))
	
	# n_events = len()

	n_events = len(src_times)

	if use_real_data_sample == False:

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


	else:

		n_false = len(inoise)
		init_false_phase = np.copy(P_ref[inoise,4])
		false_arrivals = np.concatenate((P_ref[inoise,0].reshape(-1,1), P_ref[inoise,1].reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
		ind_false_phase = np.arange(len(inoise)) + len(arrivals)
		arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)	




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


			if use_real_data_sample == False:
				noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
				iexcess_noise = np.where(np.abs(noise_values) > np.maximum(min_misfit_allowed, thresh_noise_max*sig_t*arrivals[iz,0]))[0]
				arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise

			else:
				noise_values = 0.0*np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
				iexcess_noise = np.where(np.abs(noise_values) > np.maximum(min_misfit_allowed, thresh_noise_max*sig_t*arrivals[iz,0]))[0]
				arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] # + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise				
				assert(len(iexcess_noise) == 0)

				# pdb.set_trace()


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
		data = [arrivals, srcs, active_sources, int(use_real_data_sample)]	## Note: active sources within region are only active_sources[np.where(inside_interior[active_sources] > 0)[0]]
		return data
	
	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))

	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
	phase_observed = np.copy(arrivals[:,-1]).astype('int')

	if len(iwhere_false) > 0: # For false picks, assign a random phase type

		if use_real_data_sample == True:
			phase_observed[ind_false_phase] = init_false_phase

		else:
			phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))
	

	if len(iexcess_noise) > 0:
		phase_observed[iz[iexcess_noise]] = init_phase_type ## These "false" picks are only false because they have unusually high travel time error, but the phase type should not be randomly chosen 

	perturb_phases = True # For true picks, randomly flip a fraction of phases
	if (len(phase_observed) > 0)*(perturb_phases == True)*(use_real_data_sample == False):
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
			if (np.random.rand() > 0.5) or (use_real_data_sample == True): # 30% of samples, re-focus time. # 0.7
				# time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()
				# time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*(time_shift_range/2.0)*np.random.laplace()
				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (3.0/3.0)*(src_t_kernel)*np.random.laplace()

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

	## Check use of t_win here - in default mode, this t_win is large?
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
	A_src_src_l = []
	Ac_src_src_l = []
	A_sta_sta_l = []
	A_prod_src_src_l = []
	A_prod_sta_sta_l = []
	A_src_in_sta_l = []
	A_src_in_prod_l = []
	A_prod_src_src_weights_l = []
	A_prod_sta_sta_weights_l = []
	sc = 0

	if (fixed_subnetworks != False):
		fixed_subnetworks_flag = 1
	else:
		fixed_subnetworks_flag = 0		

	active_sources_per_slice_l = []

	for i in range(n_batch):


		# i0 = np.random.randint(0, high = len(x_grids))
		# n_spc = x_grids[i0].shape[0]
		if use_full_network == True:
			n_sta_select = n_sta
			ind_sta_select = np.arange(n_sta)


		elif use_real_data_sample == True:

			n_sta_select = len(np.unique(P_ref[:,1]))
			ind_sta_select = np.unique(P_ref[:,1]).astype('int')
			if use_variable_domain == True: 

				assert(np.abs(ind_sta_select - np.arange(len(locs))).max() == 0)

				i0 = 0
				ind_sta_select = np.arange(len(locs))
				n_sta_select = len(ind_sta_select)
				# A_sta_sta_l.append(np.ascontiguousarray(np.flip(zfile['A_sta'][0:2].astype('int'), axis = 0)))
				A_sta_sta_l.append(zfile['A_sta'][0:2].astype('int'))
				A_src_in_sta_l.append(zfile['A_src_in_sta'][0:2].astype('int'))
				n_spc = x_grids[i0].shape[0]
				# A_src_src_l.append(np.ascontiguousarray(np.flip(zfile['A_src'][0:2].astype('int'), axis = 0)))
				A_src_src_l.append(zfile['A_src'][0:2].astype('int'))
				if use_expanded == True:
					Ac_src_src_l.append(zfile['Ac'][0:2].astype('int'))

			# if use_fixed_graphs

			elif (use_fixed_graphs == True)*(use_variable_domain == False):

				## Kind of slow for every iteration
				ifile = np.where(['%d_%d_%d'%(date_choose[0], date_choose[1], date_choose[2]) in s for s in st_graphs])[0]

				# ifile = np.random.choice(len(st_graphs))
				z = np.load(st_graphs[ifile[0]])
				ind_sta_select = z['ind_use']
				n_sta_select = len(ind_sta_select)
				A_sta_sta_l.append(np.ascontiguousarray(np.flip(z['A_sta'][0:2].astype('int'), axis = 0)))
				# z.close()

				i0 = z['ichoose_grid']
				A_src_in_sta_l.append(z['A_src_in_sta'][0:2].astype('int'))
				n_spc = x_grids[i0].shape[0]

				# z = np.load('Grids/Spatial_graph_%d.npz'%i0)
				A_src_src_l.append(np.ascontiguousarray(np.flip(z['A_src'][0:2].astype('int'), axis = 0)))
				if use_expanded == True:
					Ac_src_src_l.append(z['Ac'][0:2].astype('int'))

				z.close()

			else:

				i0 = np.random.randint(0, high = len(x_grids))
				n_spc = x_grids[i0].shape[0]


		else:


			if use_variable_domain == True:

				i0 = 0
				ind_sta_select = np.arange(len(locs))
				n_sta_select = len(ind_sta_select)
				A_sta_sta_l.append(zfile['A_sta'][0:2].astype('int'))
				# z.close()

				# z['ichoose_grid']
				A_src_in_sta_l.append(zfile['A_src_in_sta'][0:2].astype('int'))
				n_spc = x_grids[i0].shape[0]

				# z = np.load('Grids/Spatial_graph_%d.npz'%i0)
				A_src_src_l.append(zfile['A_src'][0:2].astype('int'))
				if use_expanded == True:
					Ac_src_src_l.append(zfile['Ac'][0:2].astype('int'))


			elif (use_fixed_graphs == True)*(use_variable_domain == False):

				ifile = np.random.choice(len(st_graphs))
				z = np.load(st_graphs[ifile])
				ind_sta_select = z['ind_use']
				n_sta_select = len(ind_sta_select)
				# A_sta_sta_l.append(np.ascontiguousarray(np.flip(z['A_sta'][0:2].astype('int'), axis = 0)))
				A_sta_sta_l.append(z['A_sta'][0:2].astype('int'), axis = 0) # ))

				# z.close()

				i0 = z['ichoose_grid']
				A_src_in_sta_l.append(z['A_src_in_sta'][0:2].astype('int'))
				n_spc = x_grids[i0].shape[0]

				# z = np.load('Grids/Spatial_graph_%d.npz'%i0)
				# A_src_src_l.append(np.ascontiguousarray(np.flip(z['A_src'][0:2].astype('int'), axis = 0)))
				A_src_src_l.append(z['A_src'][0:2].astype('int'), axis = 0)
				if use_expanded == True:
					Ac_src_src_l.append(z['Ac'][0:2].astype('int'))

				z.close()

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

				i0 = np.random.randint(0, high = len(x_grids))
				n_spc = x_grids[i0].shape[0]
				Ac_src_src_l.append(Ac)


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


	if use_variable_domain == True:
		zfile.close()


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

	# A_sta_sta_l = []
	# A_src_src_l = []
	# A_prod_sta_sta_l = []
	# A_prod_src_src_l = []
	# A_src_in_prod_l = []
	A_edges_time_p_l = []
	A_edges_time_s_l = []
	A_edges_ref_l = []

	# if use_expanded_graphs == True:
	# Ac_src_src_l = []
	Ac_prod_src_src_l = []

	lp_times = []
	lp_stations = []
	lp_phases = []
	lp_meta = []
	lp_srcs = []
	if skip_graphs == False:
		Ac_src_src_l = []

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

					# try:
					perm_vec_expand = np.random.permutation(np.arange(x_grids[grid_select].shape[0])).astype('int')
					Ac_src_src = torch.Tensor(perm_vec_expand[Ac]).long().to(device)
					# except:
					# pdb.set_trace()

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
			
			if use_fixed_graphs == False:
				A_sta_sta_l.append([])
				A_src_src_l.append([])
				A_src_in_sta_l.append([])				

			A_prod_sta_sta_l.append([])
			A_prod_src_src_l.append([])
			A_src_in_prod_l.append([])
			A_edges_time_p_l.append([])
			A_edges_time_s_l.append([])
			A_edges_ref_l.append([])

			# Ac_src_src_l.append([])
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

		# print('Grad')
		# print(lbls_grid[-1].shape)
		# print(lbls_query[-1].shape)

		X_query.append(np.concatenate((x_query, x_query_t.reshape(-1,1)), axis = 1))
		Lbls.append(lbls_grid)
		Lbls_query.append(lbls_query)

	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
	data = [arrivals, srcs, active_sources, int(use_real_data_sample)]	## Note: active sources within region are only active_sources[np.where(inside_interior[active_sources] > 0)[0]]


	if use_variable_domain == True:
		params_extra = [locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn, ftrns1, ftrns2, ftrns1_diff, ftrns2_diff] # = params_extra
	else:
		params_extra = None


	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	if (use_expanded == False) or (skip_graphs == True):

		Ac_src_in_prod_l = [[] for j in range(len(A_src_in_prod_l))]
		# return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_src_in_sta_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], params_extra, data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)
		return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, [A_src_src_l, Ac_src_src_l], A_src_in_sta_l, A_prod_sta_sta_l, A_prod_src_src_l, [A_src_in_prod_l, Ac_src_in_prod_l], A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], params_extra, data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

	else:

		return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, [A_src_src_l, Ac_src_src_l], A_src_in_sta_l, A_prod_sta_sta_l, A_prod_src_src_l, [A_src_in_prod_l, Ac_src_in_prod_l], A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], params_extra, data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)



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


		intersection = (pred * target).sum()/pred.shape[1]  # sum over spatial + channel if multi-channel
		pred_sum = (pred ** 2).sum()/pred.shape[1]
		target_sum = (target ** 2).sum()/pred.shape[1]

		dice = 1 - ((2.0 * intersection + self.smooth) /
                    (pred_sum + self.bg_weight * target_sum + self.smooth))

		return dice # .mean()


# class GaussianDiceLoss1(nn.Module):
# 	def __init__(self, smooth=1e-5, bg_weight=1.0):
# 		super().__init__()
# 		self.smooth = smooth
# 		self.bg_weight = bg_weight

# 	def forward(self, pred, target):
# 		# pred, target: (L, K) or (L, 1)
# 		# Squeeze the dummy channel if present
# 		if pred.ndim == 3 and pred.shape[-1] == 1:
# 			pred = pred.squeeze(-1)      # (L,)
# 			target = target.squeeze(-1)  # (L,)

# 		# Now shape is either (L,) → treat as (L,1) or (L,K)
# 		if pred.ndim == 1:
# 			pred = pred.unsqueeze(-1)    # (L,1)
# 			target = target.unsqueeze(-1)

# 		# Critical: mean/sum over spatial dimension only (dim=0)
# 		# This makes it invariant to different grid resolutions L
# 		intersection = (pred * target).mean(dim=0)   # (K,)
# 		pred_sum     = (pred ** 2).mean(dim=0)       # (K,)
# 		target_sum   = (target ** 2).mean(dim=0)     # (K,)

# 		numerator   = 2.0 * intersection + self.smooth
# 		denominator = pred_sum + self.bg_weight * target_sum + self.smooth

# 		per_station_dice = 1.0 - numerator / denominator    # (K,)
# 		return per_station_dice.mean()


# class GradientNormBalancer:
# 	def __init__(self, losses_names, initial_loss=None, target_norm=1.0):
# 		self.names = losses_names
# 		self.target_norm = target_norm
# 		self.scales = {name: 1.0 for name in losses_names}       # will auto-update
# 		self.initial_loss = initial_loss or {}                   # optional warm-up values
        	
# 	def update_scales(self, loss_dict):
# 		for name in self.names:
# 			loss = loss_dict[name]
# 			if loss.requires_grad:
# 				grad_norm = torch.norm(torch.mean(torch.abs(loss.grad)), p=2) if loss.grad is not None else 0
# 				# Or simpler and more common:
# 				grad_norm = torch.norm(loss * torch.ones_like(loss), p=2).detach()

# 				target = self.target_norm
# 				if name in self.initial_loss:
# 					target *= self.initial_loss[name] / loss.item()   # optional homoscedastic boost
                
# 				self.scales[name] = self.scales[name] * (target / (grad_norm + 1e-8)).detach()
# 				self.scales[name] = torch.clamp(self.scales[name], 0.01, 100.0)  # stability

# 	def __call__(self, loss_dict):
# 		balanced = 0.0
# 		for name, loss in loss_dict.items():
# 			balanced += loss * self.scales[name]
# 		return balanced


# class LossMagnitudeBalancer:
# 	def __init__(self, anchor='dice', alpha = 0.98):
# 		self.anchor = anchor
# 		self.values = {}
# 		self.scales = {}
# 		self.alpha = alpha

# 	def update(self, losses_dict):
# 		for k, v in losses_dict.items():
# 			v = v.detach().mean()
# 			if k not in self.values:
# 				self.values[k] = v
# 				self.scales[k] = 1.0
# 			else:
# 				# EMA of loss magnitude
# 				self.values[k] = self.alpha * self.values[k] + (1.0 - self.alpha) * v

# 		anchor_val = self.values[self.anchor]
# 		for k in losses_dict:
# 			self.scales[k] = (anchor_val / (self.values[k] + 1e-8)).clamp(0.1, 10.0)

# 	def __call__(self, losses_dict):
# 		self.update(losses_dict)
# 		total = sum(losses_dict[k] * self.scales[k] for k in losses_dict)
# 		return total



# def gaussian_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
def gaussian_heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred   = pred.squeeze(-1)
    # target = target.squeeze(-1)

    pos = target >= 0.01
    neg = ~pos

    loss = 0.0
    eps  = 1e-6                               # ← exact value used everywhere

    # === POSITIVES: log-space Charbonnier with clamp (official 4DGaussians version) ===
    if pos.any():
        # This exact line appears in every top repo:
        log_pred = torch.log(pred[pos].clamp(min=eps))
        log_tgt  = torch.log(target[pos] + eps)           # target can be exactly 0
        diff     = log_pred - log_tgt
        loss += torch.mean(torch.sqrt(diff*diff + eps))

    # # === NEGATIVES: focal-style background (γ=2.0) ===
    # if neg.any():
    #     residual = pred[neg]
    #     focal_weight = residual.abs().pow(2.0)
    #     loss += 15.0 * (focal_weight * residual.square()).mean()

    if neg.any():
        r = pred[neg]
        # This is the exact line in every single top model today
        loss += 25.0 * r.abs().mean()          # ← L1: kills any elevation instantly
        loss += 1.0 * r.square().mean()        # ← tiny L2: prevents jitter

    return loss


def gaussian_heatmap_loss_with_cap(
    pred: torch.Tensor,
    target: torch.Tensor,
    cap_threshold: float = 0.7,
    cap_huber_weight: float = 10.0,
    charb_downweight: float = 0.3,   # ← this is the magic number
    eps: float = 1e-6
) -> torch.Tensor:
    pos = target >= 0.01
    cap = target >= cap_threshold    # points where we want perfect amplitude
    neg = ~pos

    loss = 0.0

    # ==================== POSITIVES + CAP HANDLING ====================
    if pos.any():
        log_pred = torch.log(pred[pos].clamp(min=eps))
        log_tgt  = torch.log(target[pos] + eps)
        diff     = log_pred - log_tgt
        charb    = torch.sqrt(diff*diff + eps)

        # ↓↓↓ THIS IS THE CRUCIAL 3-LINE FIX ↓↓↓
        weight = torch.ones_like(charb)
        if cap.any():
            # Find which of the positive points are also in the cap region
            cap_in_pos = cap[pos]                 # boolean mask in the pos subspace
            weight[cap_in_pos] *= charb_downweight   # ← 0.2–0.4 works; 0.3 is consensus
        # ↑↑↑ END OF FIX ↑↑↑

        loss += (weight * charb).mean()

    # # ==================== BACKGROUND (unchanged) ====================
    # if neg.any():
    #     r = pred[neg]
    #     focal_weight = r.abs().pow(2.0)
    #     loss += 15.0 * (focal_weight * r.square()).mean()

    if neg.any():
        r = pred[neg]
        # This is the exact line in every single top model today
        loss += 25.0 * r.abs().mean()          # ← L1: kills any elevation instantly
        loss += 1.0 * r.square().mean()        # ← tiny L2: prevents jitter

    # ==================== CAP LOSS (strong absolute push) ====================
    if cap.any():
        # Huber with small delta → almost L1 on bright peaks
        loss += cap_huber_weight * F.smooth_l1_loss(
            pred[cap], target[cap], beta=0.5
        )

    return loss



# def consistency_loss1(pred1, pred2):
#     x1 = pred1.unsqueeze(0).unsqueeze(0)   # [1,1,N,1]
#     x2 = pred2.unsqueeze(0).unsqueeze(0)
#     return lpips_fn(x1, x2).mean()

# Option B — If LPIPS is too slow/heavy, use plain L1. It's honest and works great.
def consistency_loss(pred1, pred2):
    return F.l1_loss(pred1, pred2)


# Slightly stronger version — also penalize second differences (curvature)
# def tv_loss_2nd_order(pred, coords, w1=0.02, w2=0.005):
#     tv1 = 0.0
#     tv2 = 0.0
#     for dim in [0,1,2]:
#         idx = torch.argsort(coords[:, dim])
#         p = pred[idx].squeeze()
#         d1 = p[1:] - p[:-1]
#         tv1 += d1.abs().mean()
#         if len(d1) > 1:
#             tv2 += (d1[1:] - d1[:-1]).abs().mean()
#     return w1 * tv1 / 3.0 + w2 * tv2 / 3.0


# class LossAccumulationBalancer1: # TwoTier

# 	def __init__(self, anchor = 'loss_dice2', accum_steps = 10, aux_target = 0.018, alpha = 0.98, primary_ext = 'loss_dice', device = device):

# 		self.accum_steps = accum_steps
# 		self.alpha = alpha

# 		self.primary_ema = {}   # only dice_0, dice_1, dice_2, dice_3
# 		self.aux_ema     = {}   # consistency, hardneg_*, etc.
# 		self.primary_ext = primary_ext

# 		self.device = device
# 		self.anchor_head = anchor         # ← your rarest / hardest head
# 		self.aux_target  = aux_target             # total aux contribution ≈ 0.018
# 		                                     # (0.10–0.15× one primary head)

# 	def __call__(self, losses_dict):

# 		total = 0.0
# 		anchor_val = None

# 		# 1. Update EMAs (full-batch equivalent)
# 		for name, loss in losses_dict.items():
# 			val = loss.detach().mean().item() * self.accum_steps

# 			if name.startswith(self.primary_ext):
# 				if name not in self.primary_ema:
# 					self.primary_ema[name] = val
# 				else:
# 					self.primary_ema[name] = self.alpha * self.primary_ema[name] + (1-self.alpha) * val
# 				if name == self.anchor_head:
# 					anchor_val = self.primary_ema[name]
				    
# 			else:  # auxiliary
# 				if name not in self.aux_ema:
# 					self.aux_ema[name] = val
# 				else:
# 					self.aux_ema[name] = self.alpha * self.aux_ema[name] + (1-self.alpha) * val

# 		# If anchor not seen yet, fall back
# 		if anchor_val is None:
# 			anchor_val = max(self.primary_ema.values(), default=0.1)

# 		# 2. Apply scales
# 		for name, loss in losses_dict.items():

# 			if name.startswith(self.primary_ext):
# 				# Anchor-based: rarest head gets scale ≈ 1.0
# 				ema_val = self.primary_ema[name]
# 				scale = torch.tensor(anchor_val / (ema_val + 1e-8), device = self.device)
# 				scale = scale.clamp(0.1, 300.0)

# 			else:

# 				# Target-based: all aux together contribute ~aux_target
# 				ema_val = self.aux_ema[name]
# 				scale = torch.tensor(self.aux_target / (len(self.aux_ema) * (ema_val + 1e-8)), device = self.device)
# 				# or: scale = self.aux_target / (ema_val + 1e-8) if you have one global aux
# 				scale = scale.clamp(0.01, 50.0) # scale = 0.06 * anchor_val / (ema_val + 1e-8)   # aux ≈ 6% of anchor head

# 			total += loss * scale

# 		return total
	
# 	def state_dict(self):
# 		return {
# 			'accum_steps': self.accum_steps,
# 			'alpha': self.alpha,
# 			'primary_ema': self.primary_ema,
# 			'aux_ema': self.aux_ema,
# 			'primary_ext': self.primary_ext,
# 			'anchor_head': self.anchor_head,
# 			'aux_target': self.aux_target,
# 			# device is not saved – will be set on load
# 		}

# 	def load_state_dict(self, state_dict, device='cpu'):
# 		self.accum_steps = state_dict['accum_steps']
# 		self.alpha = state_dict['alpha']
# 		self.primary_ema = state_dict['primary_ema']
# 		self.aux_ema = state_dict['aux_ema']
# 		self.primary_ext = state_dict['primary_ext']
# 		self.anchor_head = state_dict['anchor_head']
# 		self.aux_target = state_dict['aux_target']
# 		self.device = device  # update device on load



class LossAccumulationBalancer:
    def __init__(
        self,
        anchor: str = 'loss_dice2',
        group_targets: dict = None,
        alpha: float = 0.98,
        primary_ext: str = 'loss_dice',
        device: str = 'cuda'
    ):
        self.anchor = anchor
        self.alpha = alpha
        self.primary_ext = primary_ext
        self.device = device

        # === Group targets ===
        if group_targets is None:
            group_targets = {'primary': 1.0, 'aux': 0.02}
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
        self.group_targets = sd.get('group_targets', {'primary': 1.0, 'aux': 0.02})
        self.alpha = sd['alpha']
        self.primary_ext = sd['primary_ext']
        self.accum_steps = sd.get('accum_steps', None)
        self.primary_ema = sd['primary_ema']
        self.aux_ema = defaultdict(dict, sd.get('aux_ema', {}))
        self._anchor_ema_current = sd.get('_anchor_ema_current', None)


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

## Make supplemental information for grids
x_grids_trv = []
x_grids_trv_pointers_p = []
x_grids_trv_pointers_s = []
x_grids_trv_refs = []
# x_grids_edges = []

if config['train_travel_time_neural_network'] == False:
	ts_max_val = Ts.max()



def compute_travel_times(trv, locs, x_grids, n_max_chunks = int(50e3), device = 'cpu'):

	x_grids_trv = []
	# locs_cuda = torch.Tensor(locs).to(device)
	for i in range(len(x_grids)):
		
		n_sta, n_temp = len(locs), len(x_grids[i])
		n_chunks = int(np.maximum(1, int((n_sta*n_temp)/n_max_chunks)))
		n_int = max(int(len(locs)/n_chunks), 1)
		n_chunks = np.minimum(n_chunks, len(locs))
		inds = [np.arange(n_int) + n_int*j for j in range(n_chunks)]

		# pdb.set_trace()
		if len(inds) == 0: inds = np.arange(len(locs))
		if (inds[-1][-1] < len(locs))*(len(inds) > 1): inds[-1] = np.arange(inds[-2][-1] + 1, len(locs))
		if (inds[-1][-1] < len(locs))*(len(inds) == 1): inds[-1] = np.arange(0, len(locs))
		if inds[-1][-1] > (len(locs) - 1): inds[-1] = np.arange(inds[-1][0], len(locs))
		assert(np.abs(np.hstack(inds) - np.arange(len(locs))).max() == 0)
	
		trv_out_l = []
		x_grid_cuda = torch.Tensor(x_grids[i]).to(device)
		for j in range(len(inds)):
			# trv_out_l.append(trv(locs_cuda[inds[j]], x_grid_cuda).cpu().detach().numpy())
			trv_out_l.append(trv(torch.Tensor(locs[inds[j]]).to(device), x_grid_cuda).cpu().detach().numpy())
		# trv_out = np.concatenate(trv_out_l, axis = 1)
		x_grids_trv.append(np.concatenate(trv_out_l, axis = 1))

	return x_grids_trv


## If an absolute station list is used, could use this x_grids_trv_base, and "slice" into it during batch generation
## Also, likely with use_variable_domain this base x_grids_trv, time_shift_range, max_t, min_t, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs
## all not needed



if use_variable_domain == False:

	x_grids_trv = compute_travel_times(trv, locs, x_grids, n_max_chunks = int(2e3), device = device)
	## Can also change this to use trv_pairwise

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
	## Also slow - if needed, could be sliced into during batch generation

	for i in range(len(x_grids)):
		
		## Note, this definition of dt and win must match the definition used in process_continous_days
		A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(x_grids_trv[i], k = k_time_edges, max_t = max_t, min_t = min_t, dt = kernel_sig_t/5.0, win = kernel_sig_t*2.0)

		if config['train_travel_time_neural_network'] == False:
			assert(x_grids_trv[i].min() > 0.0)
			assert(x_grids_trv[i].max() < (ts_max_val + 3.0))

		x_grids_trv_pointers_p.append(A_edges_time_p)
		x_grids_trv_pointers_s.append(A_edges_time_s)
		x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?

else:

	## Optionally can create x_grids_trv and then index it into if if "absolute" indices are inside the training files
	x_grids_trv = np.nan*np.ones((len(x_grids), len(x_grids[0]), len(locs), 2))
	time_shift_range = np.nan
	max_t, min_t = np.nan, np.nan

	x_grids_trv_pointers_p = [np.nan*np.zeros(1) for j in range(len(x_grids))] # .append(A_edges_time_p)
	x_grids_trv_pointers_s = [np.nan*np.zeros(1) for j in range(len(x_grids))] # .append(A_edges_time_s)
	x_grids_trv_refs = [np.nan*np.zeros(1) for j in range(len(x_grids))] # .append(dt_partition) # save as cuda tensor, or no?


## Note for each input of "GCN_Detection_Network_extended" need to fix ftrns1_diff, ftrns2_diff, scale_rel, eps, etc.

mz = GCN_Detection_Network_extended(ftrns1_diff, ftrns2_diff, trv = trv, device = device).to(device)
optimizer = optim.Adam(mz.parameters(), lr = 0.001)


# bce_loss = BCEWithLogitsLoss()
# focal_loss = SoftFocalLoss() ## Try using this on main lmse_lossoss targets
mse_loss = torch.nn.MSELoss()
# loss_func_mse = torch.nn.MSELoss()
huber_loss = torch.nn.HuberLoss(delta = 0.5, reduction = 'mean') ## Beneath delta, L2 loss is applied
l1_loss = torch.nn.L1Loss()
DiceLoss = GaussianDiceLoss(bg_weight = 1.0) ## Can change the bg_weight

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
use_consistency_loss = False
use_gradient_loss = False
use_cap_loss = True
use_huber_loss = False
use_l1_loss = True
# use_focal_loss = True


n_burn_in = int(1*n_epochs/5)
loss_names = ['loss_dice1', 'loss_dice2', 'loss_dice3', 'loss_dice4', 'loss_negative', 'loss_consistency']


LossBalancer = LossAccumulationBalancer(
    anchor='loss_dice2',
    group_targets={
        'primary':    1.0,       # everything starting with loss_dice
        'loss_regression': 0.02,      # smooth l1 loss
        'loss_consistency': 0.005,    # tiny regularizer
        'loss_negative':     0.02,      # loss_negative, loss_cap1, etc.
        'loss_cap':     0.01,      # loss_negative, loss_cap1, etc.
        'aux': 0.02, ## Base loss
        # add more whenever you want
    },
    primary_ext='loss_dice',
    alpha=0.98,
    device = device
)


def create_training_inputs(trv, Inpts, Masks, Locs, X_fixed, A_src_in_sta_l, A_src_in_prod_l, A_prod_sta_sta_l, A_prod_src_src_l, lp_srcs, lp_times, lp_stations, lp_phases, lp_meta, params_extra = None, device = device):

	## Should add increased samples in x_src_query around places of coherency
	## and true labels

	# if use_variable_domain == True:
	# 	lat_range_extend, lon_range_extend, lat_range_interior, lon_range_interior, depth_range, src_t_kernel, scale_x_extend, offset_x_extend, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, rbest, mn, ftrns1, ftrns2 = params_extra

	global ftrns1, ftrns2, locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn, ftrns1, ftrns2, ftrns1_diff, ftrns2_diff
	## Extract domain specific parameters
	if use_variable_domain == True:
		locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn, ftrns1, ftrns2, ftrns1_diff, ftrns2_diff = params_extra
		lat_range_interior = [lat_range[0], lat_range[1]]
		lon_range_interior = [lon_range[0], lon_range[1]]


	x_src_query = np.random.rand(n_src_query,3)*scale_x_extend + offset_x_extend



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
			# trv_out = trv_pairwise(torch.Tensor(Locs[A_src_in_sta_l[0].cpu().detach().numpy()]).to(device), torch.Tensor(X_fixed[A_src_in_sta_l[1].cpu().detach().numpy()]).to(device)) + torch.Tensor(time_shifts[grid_match, A_src_in_sta_l[1].cpu().detach().numpy()]).reshape(-1,1).to(device)
			trv_out = trv_pairwise(torch.Tensor(Locs[A_src_in_sta_l[0].cpu().detach().numpy()]).to(device), torch.Tensor(X_fixed[A_src_in_sta_l[1].cpu().detach().numpy()]).to(device)) + torch.Tensor(X_fixed[A_src_in_sta_l[1].cpu().detach().numpy(),3]).reshape(-1,1).to(device)

		## Consider changing spatial_vals to proportional to labels
		## Updating spatial vals to use scaled Cartesian coordinates
		# spatial_vals = torch.cat((torch.Tensor((X_fixed[A_src_in_prod_l[1].cpu().detach().numpy()][:,0:3] - Locs[A_src_in_sta_l[0][A_src_in_prod_l[0]].cpu().detach().numpy()])/scale_x_extend).to(device), torch.Tensor(X_fixed[A_src_in_prod_l[1].cpu().detach().numpy()][:,[3]]).to(device)/time_shift_range), dim = 1)
		spatial_vals = torch.cat((torch.Tensor((ftrns1(X_fixed[A_src_in_prod_l[1].cpu().detach().numpy()][:,0:3]) - ftrns1(Locs[A_src_in_sta_l[0][A_src_in_prod_l[0]].cpu().detach().numpy()]))/(30*src_x_kernel)).to(device), torch.Tensor(X_fixed[A_src_in_prod_l[1].cpu().detach().numpy()][:,[3]]).to(device)/time_shift_range), dim = 1)

	else:

		if use_time_shift == False:
			trv_out = trv(torch.Tensor(Locs).to(device), torch.Tensor(X_fixed).to(device)).detach().reshape(-1,2) ## Note: could also just take this from x_grids_trv
		else:
			# trv_out = (trv(torch.Tensor(Locs).to(device), torch.Tensor(X_fixed).to(device)).detach() + torch.Tensor(np.expand_dims(time_shift[[grid_match],:], axis = 0)).to(device)).reshape(-1,2) ## Note: could also just take this from x_grids_trv
			trv_out = (trv(torch.Tensor(Locs).to(device), torch.Tensor(X_fixed).to(device)).detach() + torch.Tensor(X_fixed[:,3].reshape(-1,1,1)).to(device)).reshape(-1,2) ## Note: could also just take this from x_grids_trv

		## Consider changing spatial_vals to proportional to labels
		## Updating spatial vals to use scaled Cartesian coordinates
		# spatial_vals = torch.cat((torch.Tensor(((np.repeat(np.expand_dims(X_fixed[:,0:3], axis = 1), Locs.shape[0], axis = 1) - np.repeat(np.expand_dims(Locs, axis = 0), X_fixed.shape[0], axis = 0)).reshape(-1,3))/scale_x_extend).to(device), torch.Tensor(X_fixed[:,[3]]).to(device)/time_shift_range), dim = 1)
		spatial_vals = torch.cat((torch.Tensor(((np.repeat(np.expand_dims(ftrns1(X_fixed[:,0:3]), axis = 1), Locs.shape[0], axis = 1) - np.repeat(np.expand_dims(ftrns1(Locs), axis = 0), X_fixed.shape[0], axis = 0)).reshape(-1,3))/(30*src_x_kernel)).to(device), torch.Tensor(X_fixed[:,[3]]).to(device)/time_shift_range), dim = 1)



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


	## Should start using the pairwise src-station x_grids_trv

	n_repeat = train_config['n_batches_per_job'] ## Number of batches to make per job

	argvs = sys.argv
	if len(argvs) < 2:
		argvs.append(0)

	job_number = int(argvs[1]) ## Choose job index

	print('Build and save training data on job index %d'%job_number)

	print('Note set t_win in input')

	for n in range(n_repeat):

		file_index = n_repeat*job_number + n ## Unique file index

		if os.path.isfile(path_to_data + 'training_data_slice_%d_ver_%d.hdf5'%(file_index, n_ver_training_data)) == True:
			continue

		if use_subgraph == True:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_src_in_sta_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], params_extra, data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = True)
			A_src_src_l, Ac_src_src_l = A_src_src_l
			A_src_in_prod_l, Ac_src_in_prod_l = A_src_in_prod_l
			# A_prod_src_src_l, Ac_prod_src_src_l = A_prod_src_src_l

			if use_expanded == True:
				# Ac_src_src_l = [[] for j in range(len(A_src_src_l))]
				Ac_prod_src_src_l = [[] for j in range(len(A_prod_src_src_l))]

		else:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_src_in_sta_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], params_extra, data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = False)
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


		## Extract domain specific parameters
		if use_variable_domain == True:
			locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn, ftrns1, ftrns2, ftrns1_diff, ftrns2_diff = params_extra
			lat_range_interior = [lat_range[0], lat_range[1]]
			lon_range_interior = [lon_range[0], lon_range[1]]
			x_grids_trv = np.vstack([np.expand_dims(x_grids_trv[0], axis = 0)])
			print('Stas: %d, %d'%(len(locs), len(stas)))


		## Need to use current files src_arv_kernels
		h = h5py.File(path_to_data + 'training_data_slice_%d_ver_%d.hdf5'%(file_index, n_ver_training_data), 'w')
		h['data'] = data[0]
		h['srcs'] = data[1]
		h['srcs_active'] = data[2]
		h['real_data'] = data[3]

		if use_fixed_graphs == False:
			A_src_in_sta_l = [[] for i in range(n_batch)]

		for i in range(n_batch):

			if use_subgraph == True:

				if use_fixed_graphs == False:
					A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[i], X_fixed[i], ftrns1, ftrns2, max_deg_offset = max_deg_offset, scale_time = scale_time, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, Ac = Ac_src_src_l[i], device = device)

				else:
					# A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[i], X_fixed[i], ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, Ac = Ac, A_sta_sta = torch.Tensor(A_sta_sta_l[i]).long().to(device), A_src_src = torch.Tensor(A_src_src_l[i]).long().to(device), device = device)
					A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[i], X_fixed[i], ftrns1, ftrns2, max_deg_offset = max_deg_offset, scale_time = scale_time, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, Ac = Ac_src_src_l[i], A_sta_sta = torch.Tensor(A_sta_sta_l[i]).long().to(device), A_src_src = torch.Tensor(A_src_src_l[i]).long().to(device), A_src_in_sta = torch.Tensor(A_src_in_sta_l[i]).long().to(device),  device = device)


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
			# print('Note set t_win in input')
			x_src_query, tq_sample, x_src_query_cart, trv_out, trv_out_src, spatial_vals, tq, input_tensor_1, input_tensor_2, A_prod_sta_tensor, A_prod_src_tensor, data_1, data_2, lp_times_slice, lp_stations_slice, lp_phases_slice, lp_meta_slice = create_training_inputs(trv, Inpts[i], Masks[i], Locs[i], X_fixed[i], A_src_in_sta_l[i], A_src_in_prod_l[i], A_prod_sta_sta_l[i], A_prod_src_src_l[i], lp_srcs[i], lp_times[i], lp_stations[i], lp_phases[i], lp_meta[i], params_extra = params_extra, device = device)

			# pdb.set_trace()

			pick_lbls = pick_labels_extract_interior_region_flattened(x_src_query_cart, tq_sample.cpu().detach().numpy(), lp_meta_slice[:,-2::], lp_srcs[i], lat_range_interior, lon_range_interior, ftrns1, sig_t = src_t_arv_kernel, sig_x = src_x_arv_kernel)


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
			# f_out.create_dataset(f'is_real_sample{suffix_dst}', data=is_real)
			h['is_real_sample_%d'%i] = data[3]

			# locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn, ftrns1, ftrns2, ftrns1_diff, ftrns2_diff = params_extra


			## Extra parameters
			if use_variable_domain == True:
				h['locs_%d'%i] = locs
				h['stas_%d'%i] = stas.astype('S')
				h['lat_range_extend_%d'%i] = lat_range_extend
				h['lon_range_extend_%d'%i] = lon_range_extend
				h['lat_range_%d'%i] = lat_range # _interior
				h['lon_range_%d'%i] = lon_range # _interior
				h['depth_range_%d'%i] = depth_range
				# h['src_t_kernel_%d'%i] = src_t_kernel
				h['scale_x_extend_%d'%i] = scale_x_extend
				h['offset_x_extend_%d'%i] = offset_x_extend
				h['scale_time_%d'%i] = scale_time
				h['t_win_%d'%i] = t_win
				h['dt_win_%d'%i] = dt_win
				h['time_shift_range_%d'%i] = time_shift_range
				h['kernel_sig_t_%d'%i] = kernel_sig_t
				h['src_x_kernel_%d'%i] = src_x_kernel
				h['src_t_kernel_%d'%i] = src_t_kernel
				h['src_depth_kernel_%d'%i] = src_depth_kernel
				h['src_x_arv_kernel_%d'%i] = src_x_arv_kernel
				h['src_t_arv_kernel_%d'%i] = src_t_arv_kernel
				# x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t
				h['x_grids_%d'%i] = x_grids
				h['x_grids_trv_%d'%i] = x_grids_trv ## May have to convert from list


				# for j in range(len(x_grids_trv_refs)):
				for j in range(len(x_grids_trv)):
					h['x_grids_trv_refs_%d_%d'%(i,j)] = x_grids_trv_refs[j]


				h['max_t_%d'%i] = max_t
				h['min_t_%d'%i] = min_t
				h['rbest_%d'%i] = rbest
				h['mn_%d'%i] = mn



		h.close()

	print('Finished building training data for job %d'%job_number)

	print('Data set built; call the training script again once all data has been built')
	sys.exit()




class TrainingDataset(Dataset):


	def __init__(self, list_of_hdf5_paths, n_batch, total_steps, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded): ## This n_batch is not used (instead do batching with loader - though could change to save larger .hdf5 files with multiple samples)
		self.files = list_of_hdf5_paths   # e.g. 1_000_000 files
		self.n_batch = n_batch
		self.use_gradient_loss = use_gradient_loss
		self.use_expanded = use_expanded

		# 1. Determine total length
		num_actual_files = len(self.files)
		if total_steps is None:
			self.total_steps = num_actual_files
		else:
			self.total_steps = total_steps

		# 2. Build a long, shuffled index map
		# If total_steps is 3000 and num_actual_files is 1000, 
		# this creates 3 unique permutations and joins them.
		repeats = (self.total_steps // num_actual_files) + 1
		master_indices = []
		for _ in range(repeats):
			master_indices.append(np.random.permutation(num_actual_files))
        
		self.index_map = np.concatenate(master_indices)[:self.total_steps]


	def __len__(self):
		return self.total_steps

	def __getitem__(self, idx):
		# path = self.files[idx]

		actual_file_idx = self.index_map[idx]
		path = self.files[actual_file_idx]

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
			# tq_sample = []
			input_tensors_l = []
			params_extra_l = []


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
					[torch.from_numpy(f['A_src_src_%d'%i][:]).long(), torch.from_numpy(f['Ac_src_src_%d'%i][:]).long()] if use_expanded == True else torch.from_numpy(f['A_src_src_%d'%i][:]).long(),
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

				params_extra = []
				# locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn, ftrns1, ftrns2, ftrns1_diff, ftrns2_diff = params_extra
				if use_variable_domain == True:
					params_extra.append(f['locs_%d'%i][:])
					params_extra.append(f['stas_%d'%i][:])
					params_extra.append(f['lat_range_extend_%d'%i][:])
					params_extra.append(f['lon_range_extend_%d'%i][:])
					params_extra.append(f['lat_range_%d'%i][:])
					params_extra.append(f['lon_range_%d'%i][:])
					params_extra.append(f['depth_range_%d'%i][:])
					params_extra.append(f['scale_x_extend_%d'%i][:])
					params_extra.append(f['offset_x_extend_%d'%i][:])
					params_extra.append(f['scale_time_%d'%i][()])
					params_extra.append(f['t_win_%d'%i][()])
					params_extra.append(f['dt_win_%d'%i][()])
					params_extra.append(f['time_shift_range_%d'%i][()])
					params_extra.append(f['kernel_sig_t_%d'%i][()])
					params_extra.append(f['src_x_kernel_%d'%i][()])
					params_extra.append(f['src_t_kernel_%d'%i][()])
					params_extra.append(f['src_depth_kernel_%d'%i][()])
					params_extra.append(f['src_x_arv_kernel_%d'%i][()])
					params_extra.append(f['src_t_arv_kernel_%d'%i][()])
					params_extra.append(f['x_grids_%d'%i][:])

					# params_extra.append(f['x_grids_trv_%d'%i][:])
					params_extra.append(np.expand_dims(f['x_grids_trv_%d'%i][:], axis = 0))

					## Need to check why x_grids_trv_refs_0_0 not always there

					x_grids_trv_refs = []
					# pdb.set_trace()
					# for j in range(len(params_extra[-1])):
					# 	x_grids_trv_refs.append(f['x_grids_trv_refs_%d_%d'%(i, j)][:])
					# params_extra.append(f['x_grids_trv_refs_%d'%i])
					params_extra.append(x_grids_trv_refs)

					params_extra.append(f['max_t_%d'%i][()])
					params_extra.append(f['min_t_%d'%i][()])
					params_extra.append(f['rbest_%d'%i][:])
					params_extra.append(f['mn_%d'%i][:])

					# locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn

				## Could possibly map to cuda here (note: possible ragged list for A_prod_src_tensor)
				input_tensors_l.append(input_tensors)
				params_extra_l.append(params_extra)

			real_data = f['real_data'][()]

			# is_real
			if 'is_real_sample_0' in f.keys():
				real_data_v = np.array([f['is_real_sample_%d'%i][()] for i in range(self.n_batch)])
				data = [f['srcs'][:][f['srcs_active'][:].astype('int')], real_data_v]

			else:
				data = [f['srcs'][:][f['srcs_active'][:].astype('int')], real_data]

		if self.use_gradient_loss == False:

			return input_tensors_l, [lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l], params_extra_l, data

		else:

			return input_tensors_l, [lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l, Lbls_grad_spc, Lbls_query_grad_spc, Lbls_grad_t, Lbls_query_grad_t], params_extra_l, data


import random
from tqdm import tqdm

# def reshuffle_with_flags(input_dir, output_dir, n_batch=10):
#     os.makedirs(output_dir, exist_ok=True)
    
#     all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.hdf5')]
#     random.shuffle(all_files)
    
#     # Process in chunks of n_batch files
#     for chunk_idx in tqdm(range(0, len(all_files), n_batch), desc="Re-batching"):
#         batch_files = all_files[chunk_idx : chunk_idx + n_batch]
#         if len(batch_files) < n_batch: break
            
#         handles = [h5py.File(f, 'r') for f in batch_files]
        
#         try:
#             for new_file_idx in range(n_batch):
#                 new_path = os.path.join(output_dir, f"mix_chunk_{chunk_idx}_{new_file_idx}.hdf5")
                
#                 with h5py.File(new_path, 'w') as f_out:
#                     for i_target in range(n_batch):
#                         # Pick a source file and a specific sample index from it
#                         source_h5 = handles[i_target]
#                         i_src = random.randint(0, n_batch - 1)
                        
#                         # 1. Copy all indexed data (spatial_vals_i, Lbls_i, etc.)
#                         suffix_src = f"_{i_src}"
#                         suffix_dst = f"_{i_target}"
                        
#                         for key in source_h5.keys():
#                             if key.endswith(suffix_src):
#                                 base_name = key.rsplit('_', 1)[0]
#                                 source_h5.copy(key, f_out, name=f"{base_name}{suffix_dst}")
                                
#                         # 2. Create the NEW per-sample flag
#                         # We pull the global 'real_data' value from the source file
#                         is_real = source_h5['real_data'][()] 
#                         f_out.create_dataset(f'is_real_sample{suffix_dst}', data=is_real)

#                     # 3. Copy global constants (if any)
#                     f_out.copy(handles[0]['srcs'], f_out, name='srcs')
#                     f_out.copy(handles[0]['srcs_active'], f_out, name='srcs_active')

#         finally:
#             for h in handles: h.close()



import os
import sys
import random
import h5py

def reshuffle_subset(input_dir, output_dir, job_idx, files_per_job, n_batch=15):
    """
    job_idx: The SLURM_ARRAY_TASK_ID
    files_per_job: How many HDF5 files this specific task should generate
    n_batch: Number of samples per output HDF5 file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get and Sort (Essential for consistency across parallel Slurm tasks)
    all_files = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.h5') or f.endswith('.hdf5')
    ])
    
    # 2. Global Shuffle (Use a fixed seed so all jobs see the same shuffled list)
    random.seed(42)
    random.shuffle(all_files)
    
    # 3. Consumption Math
    total_needed = files_per_job * n_batch
    total_available = len(all_files)
    
    # 4. Extract unique pool for this job using Modulo (Wrapping)
    source_pool = []
    for i in range(total_needed):
        global_idx = (job_idx * total_needed + i) % total_available
        source_pool.append(all_files[global_idx])

    # 5. Set seed once for this job for continuous internal randomness 
    random.seed(job_idx)

    # Grab version number from the first file in our shuffled list
    first_file_name = os.path.basename(all_files[0])
    n_ver = int(first_file_name.split('_')[-1].split('.')[0])
    start_naming_index = job_idx * files_per_job

    print(f"Job {job_idx}: Generating {files_per_job} files starting at ID {start_naming_index}")

    # Track an index for extra file draws if we hit an completely dead file
    fallback_pool_idx = 0

    for f_idx in range(files_per_job):
        unique_id = start_naming_index + f_idx
        new_filename = f"training_data_slice_{unique_id}_ver_{n_ver}.hdf5"
        new_path = os.path.join(output_dir, new_filename)
        
        # Slice the pool for this specific output file (n_batch files)
        current_sources = source_pool[f_idx * n_batch : (f_idx + 1) * n_batch]
        
        with h5py.File(new_path, 'w') as f_out:
            meta_copied = False
            i_target = 0
            
            while i_target < n_batch:
                # Get the next scheduled source file, or pull a fallback if we exceeded the slice
                if i_target < len(current_sources):
                    src_path = current_sources[i_target]
                else:
                    # Edge case fallback: Grab next file out of the global pool via wrapping
                    fallback_idx = (job_idx * total_needed + total_needed + fallback_pool_idx) % total_available
                    src_path = all_files[fallback_idx]
                    fallback_pool_idx += 1

                # Open files sequentially to prevent file descriptor leakage
                with h5py.File(src_path, 'r') as src_h5:
                    
                    # 1. Scan keys to find all available sample indices
                    source_keys = list(src_h5.keys())
                    sample_indices = set()
                    for key in source_keys:
                        if key.startswith("lp_times_"):
                            parts = key.rsplit('_', 1)
                            if len(parts) > 1 and parts[1].isdigit():
                                sample_indices.add(int(parts[1]))
                    
                    # 2. Filter out keys where lp_times_<i> is empty (length 0)
                    valid_indices = []
                    for idx in sample_indices:
                        lp_key = f"lp_times_{idx}"
                        # Check dataset size/shape cleanly without loading the full data array
                        if src_h5[lp_key].shape[0] > 0:
                            valid_indices.append(idx)
                    
                    # CRITICAL EDGE CASE: All indices in this file are bad
                    if not valid_indices:
                        print(f"Warning: {os.path.basename(src_path)} has 0 valid samples. Skipping file.")
                        # Do not increment i_target; let the loop retry with a fallback file
                        # If this was one of the original slice files, append a placeholder to keep index alignment
                        if i_target < len(current_sources):
                            current_sources.append(None) 
                        continue
                    
                    # Pick a valid sample index at random
                    i_src = random.choice(valid_indices)
                    
                    suffix_src = f"_{i_src}"
                    suffix_dst = f"_{i_target}"
                    
                    # Copy all datasets belonging to the chosen sample
                    for key in source_keys:
                        parts = key.rsplit('_', 1)
                        if len(parts) > 1 and parts[1] == str(i_src):
                            base_name = parts[0]
                            src_h5.copy(key, f_out, name=f"{base_name}{suffix_dst}")
                    
                    # Add the per-sample real data flag safely
                    if 'real_data' in src_h5:
                        is_real = src_h5['real_data'][()]
                        # f_out.create_dataset(f'is_real_sample{suffix_dst}', data=is_real)

                    # Copy global metadata from the first successful file in the mix
                    if not meta_copied:
                        for global_key in ['real_data', 'srcs', 'srcs_active']:
                            if global_key in src_h5:
                                f_out.create_dataset(global_key, data=src_h5[global_key][()])
                        meta_copied = True
                
                # Advance to next target sample inside the output file
                i_target += 1



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



## Load Dataset
if load_training_data == True:
	dataset = TrainingDataset(np.random.permutation(files_load), n_batch, n_epochs, use_gradient_loss = use_gradient_loss, use_expanded = use_expanded)



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



## Set initial counter
# i = n_restart_step
log_buffer = [] ## Append write operations to here and flush every 10 steps
len_loader = len(loader) ## Why not loop over data until n_epochs

# for i in range(n_restart_step, n_epochs):
for batch_idx, inputs in enumerate(loader):

	## Effective step size
	i = n_restart_step + batch_idx
	if i > n_epochs:
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

	
	if use_variable_domain == False:
		mz.set_scale_coefficients(src_x_kernel*2.0, scale_time, kernel_sig_t, kernel_sig_t*3.0, src_x_kernel, src_t_kernel, time_shift_range)
		
	
	if (((np.mod(i, 1000) == 0) or (i == (n_epochs - 1)))*(i != n_restart_step)) or (batch_idx == (len_loader - 1)):

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
	tq_sample = [inputs[0][j][-2] for j in range(n_batch)]

	# input_tensors = [
	# 	torch.from_numpy(f['input_tensor_1_%d'%i][:]), 
	# 	torch.from_numpy(f['input_tensor_2_%d'%i][:]), 
	# 	torch.from_numpy(f['A_prod_sta_tensor_%d'%i][:]).long(), 
	# 	A_prod_src_tensor, # torch.from_numpy(f['A_prod_src_tensor_%d'%i]).long()
	# 	torch.from_numpy(f['data_1_edges_%d'%i][:]).long(), 
	# 	torch.from_numpy(f['data_2_edges_%d'%i][:]).long(),
	# 	# Data(x = spatial_vals, edge_index = torch.from_numpy(f['data_1_edges_%d'%i][:]).long()), 
	# 	# Data(x = spatial_vals, edge_index = torch.from_numpy(f['data_2_edges_%d'%i][:]).long()),
	# 	torch.from_numpy(f['A_src_in_sta_%d'%i][:]).long(),
	# 	torch.from_numpy(f['A_src_src_%d'%i][:]).long(),
	# 	torch.zeros(1), # torch.Tensor(A_edges_time_p_l[i0]).long().to(device)
	# 	torch.zeros(1), # torch.Tensor(A_edges_time_s_l[i0]).long().to(device)
	# 	torch.zeros(1), # torch.Tensor(A_edges_ref_l[i0]).to(device)
	# 	torch.from_numpy(f['trv_out_%d'%i][:]),
	# 	torch.from_numpy(f['lp_times_%d'%i][:]),
	# 	torch.from_numpy(f['lp_stations_%d'%i][:]).long(),
	# 	torch.from_numpy(f['lp_phases_%d'%i][:].reshape(-1,1)).float(),
	# 	torch.from_numpy(f['Locs_cart_%d'%i][:]),
	# 	torch.from_numpy(f['X_fixed_cart_%d'%i][:]),
	# 	torch.from_numpy(f['X_fixed_%d'%i][:,3][:]),
	# 	torch.from_numpy(f['X_query_cart_%d'%i][:]),
	# 	torch.from_numpy(f['x_src_query_cart_%d'%i][:]),
	# 	torch.from_numpy(f['X_query_%d'%i][:,3]), 
	# 	torch.from_numpy(f['tq_sample_%d'%i][:]), 
	# 	torch.from_numpy(f['trv_out_src_%d'%i][:])
	# ]


	lbl_srcs = inputs[3][0]
	use_real_data_sample = inputs[3][1]
	if len(np.array([use_real_data_sample]).reshape(-1)) > 1:
		use_real_data_sample_v = np.array([use_real_data_sample]).reshape(-1) # np.copy(inputs[3][1]).reshape(-1)
	else:
		use_real_data_sample_v = use_real_data_sample*np.ones(n_batch)

	X_fixed_cart = [inputs[0][j][-7] for j in range(n_batch)]
	X_fixed_t = [inputs[0][j][-6] for j in range(n_batch)]
	num_fixed = [X_fixed_cart[j].shape[0] for j in range(n_batch)]

	mask_lbls_lv = []
	mask_lbls_query_lv = []
	mask_lbls_assoc_query_lv = []

	mask_lbls_l = [[] for j in range(n_batch)]
	mask_lbls_query_l = [[] for j in range(n_batch)]
	mask_lbls_assoc_query_l = [[] for j in range(n_batch)]


	weight_assoc_v = []
	for j in range(n_batch):


		if use_real_data_sample_v[j] == True:
			weight_assoc = 0.5 # else 1.0
			## Only use labels within ~3 std of the sources

			mask_lbls = torch.zeros(num_fixed[j],1).to(device)
			mask_lbls_query = torch.zeros(X_query[j].shape[0],1).to(device)
			mask_lbls_assoc_query = torch.zeros(x_src_query_cart_l[j].shape[0],1).to(device)

			if len(lp_srcs[j]) > 0:

				dist_srcs = ((np.linalg.norm(np.expand_dims(X_fixed_cart[j].cpu().detach().numpy()[:,0:3], axis = 0) - np.expand_dims(ftrns1(lp_srcs[j].cpu().detach().numpy()[:,0:3]), axis = 1), axis = 2) / src_x_kernel) < 5)
				dist_srcs_t = ((np.abs(np.expand_dims(X_fixed_t[j].cpu().detach().numpy().reshape(-1,1), axis = 0) - np.expand_dims(lp_srcs[j].cpu().detach().numpy()[:,[3]], axis = 1)) / src_t_kernel) < 3)[:,:,0]
				mask_lbls[np.where((dist_srcs*dist_srcs_t).max(0) > 0)[0]] = 1.0

				dist_srcs = ((np.linalg.norm(np.expand_dims(ftrns1(X_query[j].cpu().detach().numpy()[:,0:3]), axis = 0) - np.expand_dims(ftrns1(lp_srcs[j].cpu().detach().numpy()[:,0:3]), axis = 1), axis = 2) / src_x_kernel) < 5)
				dist_srcs_t = ((np.abs(np.expand_dims(X_query[j].cpu().detach().numpy()[:,[3]], axis = 0) - np.expand_dims(lp_srcs[j].cpu().detach().numpy()[:,[3]], axis = 1)) / src_t_kernel) < 3)[:,:,0]
				mask_lbls_query[np.where((dist_srcs*dist_srcs_t).max(0) > 0)[0]] = 1.0
			
				dist_srcs = ((np.linalg.norm(np.expand_dims(x_src_query_cart_l[j].cpu().detach().numpy()[:,0:3], axis = 0) - np.expand_dims(ftrns1(lp_srcs[j].cpu().detach().numpy()[:,0:3]), axis = 1), axis = 2) / src_x_kernel) < 5)
				dist_srcs_t = ((np.abs(np.expand_dims(tq_sample[j].reshape(-1,1), axis = 0) - np.expand_dims(lp_srcs[j].cpu().detach().numpy()[:,[3]], axis = 1)) / src_t_kernel) < 3)[:,:,0]
				mask_lbls_assoc_query[np.where((dist_srcs*dist_srcs_t).max(0) > 0)[0]] = 1.0

			mask_lbls_lv.append(mask_lbls)
			mask_lbls_query_lv.append(mask_lbls_query)
			mask_lbls_assoc_query_lv.append(mask_lbls_assoc_query)
			weight_assoc_v.append(weight_assoc)

		else:

			weight_assoc = 1.0
			# mask_lbls_lv = [torch.ones(num_fixed[j],1).to(device) for j in range(n_batch)]
			# mask_lbls_query_lv = [torch.ones(X_query[j].shape[0],1).to(device) for j in range(n_batch)]
			# mask_lbls_assoc_query_lv = [torch.ones(x_src_query_cart_l[j].shape[0],1).to(device) for j in range(n_batch)]

			mask_lbls_lv.append(torch.ones(num_fixed[j],1).to(device))
			mask_lbls_query_lv.append(torch.ones(X_query[j].shape[0],1).to(device))
			mask_lbls_assoc_query_lv.append(torch.ones(x_src_query_cart_l[j].shape[0],1).to(device))
			weight_assoc_v.append(weight_assoc)

	for j in range(n_batch):
		mask_lbls_l[j] = torch.where(mask_lbls_lv[j][:,0] > 0)[0]
		mask_lbls_query_l[j] = torch.where(mask_lbls_query_lv[j][:,0] > 0)[0]
		mask_lbls_assoc_query_l[j] = torch.where(mask_lbls_assoc_query_lv[j][:,0] > 0)[0]




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


		## Now i0 is not set to 0
		## Adding skip... to skip samples with zero input picks
		if len(lp_times[i0]) == 0:
			print('skip a sample!') ## If this skips, and yet i0 == (n_batch - 1), is it a problem?
			continue ## Skip this!



		if use_variable_domain == True:

			locs, stas, lat_range_extend, lon_range_extend, lat_range, lon_range, depth_range, scale_x_extend, offset_x_extend, scale_time, t_win, dt_win, time_shift_range, kernel_sig_t, src_x_kernel, src_t_kernel, src_depth_kernel, src_x_arv_kernel, src_t_arv_kernel, x_grids, x_grids_trv, x_grids_trv_refs, max_t, min_t, rbest, mn = inputs[2][i0] # ftrns1, ftrns2, ftrns1_diff, ftrns2_diff = # params_extra
			src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.
			lat_range_interior = [lat_range[0], lat_range[1]]
			lon_range_interior = [lon_range[0], lon_range[1]]


			## Set model hyper-parameters


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


			## Set model parameters
			# mz.set_scale_coefficients(scale_rel, scale_time, kernel_sig_t, eps)
			mz.set_scale_coefficients(src_x_kernel*2.0, scale_time, kernel_sig_t, kernel_sig_t*3.0, src_x_kernel, src_t_kernel, time_shift_range)
			# scale_rel, scale_time, kernel_sig_t, eps, src_x_kernel, src_t_kernel, time_shift_range


		if use_gradient_loss == False:

			out = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])

		else:

			# out, grads = mz(*input_tensors)
			out, grads = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])
			grad_grid_src, grad_grid_t, grad_query_src, grad_query_t = grads



		## Select the specific pick labels (or can re-create above)
		pick_lbls = pick_lbls_l[i0]



		## Make plots
		make_plot = False
		if make_plot == True:
			fig, ax = plt.subplots(4, 1, sharex = True)
			for j in range(2):
				i1 = np.where(Lbls_query[i0][:,0].cpu().detach().numpy() > 0.1)[0]
				i2 = np.where(out[1][:,0].cpu().detach().numpy() > 0.1)[0]
				ax[2*j].scatter(X_query[i0][i1,3].cpu().detach().numpy(), X_query[i0][i1,j].cpu().detach().numpy(), c = Lbls_query[i0][i1,0].cpu().detach().numpy())
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


			loss_base1 = DiceLoss(out[0][mask_lbls_l[i0]], torch.Tensor(Lbls[i0]).to(device)[mask_lbls_l[i0]])
			loss_dice2 = DiceLoss(out[1][mask_lbls_query_l[i0]], torch.Tensor(Lbls_query[i0]).to(device)[mask_lbls_query_l[i0]])


			loss_dice3 = DiceLoss(out[2][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,0])
			loss_dice4 = DiceLoss(out[3][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,1])


			# loss_src_val += (loss_dice1.item() + loss_dice2.item())/n_batch
			loss_src_val += (loss_base1.item() + loss_dice2.item())/n_batch
			loss_asc_val += (loss_dice3.item() + loss_dice4.item())/n_batch



		# use_mse_loss = False ## Switching to Huber loss
		if use_mse_loss == True:

			if use_sigmoid == False:

				loss_mse1 = weights[0]*mse_loss(out[0][mask_lbls_l[i0]], torch.Tensor(Lbls[i0][mask_lbls_l[i0]]).to(device)) + weights[1]*mse_loss(out[1][mask_lbls_query_l[i0]], torch.Tensor(Lbls_query[i0][mask_lbls_query_l[i0]]).to(device))
				loss_mse2 = weight_assoc_v[inc]*(weights[2]*mse_loss(out[2][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,0]) + weights[3]*mse_loss(out[3][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,1])) # /n_batch
				# loss = loss_mse1 + 2.0*loss_mse2
				# loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1]))/n_batch
				# loss = (weights[0]*loss_func(out[0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch

			else:

				min_val_lbl = 0.05
				ifind_mask1 = torch.Tensor(Lbls[i0][:,0] > min_val_lbl).long().to(device)
				ifind_mask2 = torch.Tensor(Lbls_query[i0][:,0] > min_val_lbl).long().to(device)

				loss_bce = weights[0]*bce_loss(out[0][:,1], ifind_mask1.float()) + weights[1]*bce_loss(out[1][:,1], ifind_mask2.float())
				loss_mse1 = weights[0]*mse_loss(out[0][ifind_mask1,0].reshape(-1,1), torch.Tensor(Lbls[i0]).to(device)[ifind_mask1]) + weights[1]*mse_loss(out[1][ifind_mask2,0].reshape(-1,1), torch.Tensor(Lbls_query[i0]).to(device)[ifind_mask2])
				loss_mse2 = weight_assoc_v[inc]*(weights[2]*mse_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*mse_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch



		# use_huber_loss = True
		if use_huber_loss == True:

			if use_sigmoid == False:

				loss_huber1 = weights[0]*huber_loss(out[0][mask_lbls_l[i0]], torch.Tensor(Lbls[i0][mask_lbls_l[i0]]).to(device)) + weights[1]*huber_loss(out[1][mask_lbls_query_l[i0]], torch.Tensor(Lbls_query[i0][mask_lbls_query_l[i0]]).to(device))
				loss_huber2 = weight_assoc_v[inc]*(weights[2]*huber_loss(out[2][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,0]) + weights[3]*huber_loss(out[3][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,1])) # /n_batch
				# loss = loss_mse1 + 2.0*loss_mse2

			else:

				min_val_lbl = 0.05
				ifind_mask1 = torch.Tensor(Lbls[i0][:,0] > min_val_lbl).long().to(device)
				ifind_mask2 = torch.Tensor(Lbls_query[i0][:,0] > min_val_lbl).long().to(device)

				loss_bce = weights[0]*bce_loss(out[0][:,1], ifind_mask1.float()) + weights[1]*bce_loss(out[1][:,1], ifind_mask2.float())
				loss_huber1 = weights[0]*mse_loss(out[0][ifind_mask1,0].reshape(-1,1), torch.Tensor(Lbls[i0]).to(device)[ifind_mask1]) + weights[1]*mse_loss(out[1][ifind_mask2,0].reshape(-1,1), torch.Tensor(Lbls_query[i0]).to(device)[ifind_mask2])
				loss_huber2 = weight_assoc_v[inc]*(weights[2]*mse_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*mse_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch


		# use_huber_loss = True
		if use_l1_loss == True:

			if use_sigmoid == False:

				if (use_cap_loss == False) or (i <= n_burn_in):

					loss_smooth_l1 = weights[0]*gaussian_heatmap_loss(out[0][mask_lbls_l[i0]], torch.Tensor(Lbls[i0]).to(device)[mask_lbls_l[i0]]) + weights[1]*gaussian_heatmap_loss(out[1][mask_lbls_query_l[i0]], torch.Tensor(Lbls_query[i0]).to(device)[mask_lbls_query_l[i0]])
					loss_smooth_l2 = weight_assoc_v[inc]*(weights[2]*gaussian_heatmap_loss(out[2][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,0]) + weights[3]*gaussian_heatmap_loss(out[3][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,1])) # /n_batch
					# loss = loss_mse1 + 2.0*loss_mse2

				else:

					loss_smooth_l1 = weights[0]*gaussian_heatmap_loss_with_cap(out[0][mask_lbls_l[i0]], torch.Tensor(Lbls[i0]).to(device)[mask_lbls_l[i0]]) + weights[1]*gaussian_heatmap_loss_with_cap(out[1][mask_lbls_query_l[i0]], torch.Tensor(Lbls_query[i0]).to(device)[mask_lbls_query_l[i0]])
					loss_smooth_l2 = weight_assoc_v[inc]*(weights[2]*gaussian_heatmap_loss_with_cap(out[2][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,0]) + weights[3]*gaussian_heatmap_loss_with_cap(out[3][mask_lbls_assoc_query_l[i0],:,0], pick_lbls[mask_lbls_assoc_query_l[i0],:,1])) # /n_batch
					# loss = loss_mse1 + 2.0*loss_mse2					


			else:

				min_val_lbl = 0.05
				ifind_mask1 = torch.Tensor(Lbls[i0][:,0] > min_val_lbl).long().to(device)
				ifind_mask2 = torch.Tensor(Lbls_query[i0][:,0] > min_val_lbl).long().to(device)

				loss_bce = weights[0]*bce_loss(out[0][:,1], ifind_mask1.float()) + weights[1]*bce_loss(out[1][:,1], ifind_mask2.float())
				loss_huber1 = weights[0]*mse_loss(out[0][ifind_mask1,0].reshape(-1,1), torch.Tensor(Lbls[i0]).to(device)[ifind_mask1]) + weights[1]*mse_loss(out[1][ifind_mask2,0].reshape(-1,1), torch.Tensor(Lbls_query[i0]).to(device)[ifind_mask2])
				loss_huber2 = weight_assoc_v[inc]*(weights[2]*mse_loss(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*mse_loss(out[3][:,:,0], pick_lbls[:,:,1])) # /n_batch



		# use_cap_loss = True
		if (use_cap_loss == True)*(i > n_burn_in):

			scale_cap = 1.0
			cap_limit = 0.7


			ifind_cap1 = np.where((Lbls[i0] > cap_limit)*(mask_lbls_lv[i0].cpu() > 0))[0]
			ifind_cap2 = np.where((Lbls_query[i0] > cap_limit)*(mask_lbls_query_lv[i0].cpu() > 0))[0]
			ifind_cap11, ifind_cap12 = torch.where((pick_lbls[:,:,0] > cap_limit)*(mask_lbls_assoc_query_lv[i0] > 0)) # [0]
			ifind_cap21, ifind_cap22 = torch.where((pick_lbls[:,:,1] > cap_limit)*(mask_lbls_assoc_query_lv[i0] > 0)) # [0]

			# 8.0 * F.smooth_l1_loss(pred[cap_mask], target[cap_mask], beta=0.5)

			loss_cap1 = torch.tensor(0.0).to(device)
			loss_cap2 = torch.tensor(0.0).to(device)

			if len(ifind_cap1) > 0: loss_cap1 += scale_cap*(weights[0]*F.smooth_l1_loss(out[0][ifind_cap1], torch.Tensor(Lbls[i0][ifind_cap1]).to(device), beta = 0.5))
			if len(ifind_cap2) > 0: loss_cap1 += scale_cap*(weights[1]*F.smooth_l1_loss(out[1][ifind_cap2], torch.Tensor(Lbls_query[i0][ifind_cap2]).to(device), beta = 0.5))


			if len(ifind_cap11) > 0: loss_cap2 += weight_assoc_v[inc]*scale_cap*(weights[2]*F.smooth_l1_loss(out[2][ifind_cap11,ifind_cap12,0], pick_lbls[ifind_cap11,ifind_cap12,0], beta = 0.4)) # 0.5
			if len(ifind_cap21) > 0: loss_cap2 += weight_assoc_v[inc]*scale_cap*(weights[3]*F.smooth_l1_loss(out[3][ifind_cap21,ifind_cap22,0], pick_lbls[ifind_cap21,ifind_cap22,1], beta = 0.4)) # 0.5

			loss_cap_val += (loss_cap1 + loss_cap2)/n_batch


		if (use_negative_loss == True)*(i > n_burn_in)*(use_real_data_sample_v[inc] == False):


			min_up_sample = 0.1
			# prob_up_sample = np.maximum(out[1][:,0].detach().cpu().detach().numpy()*(out[1][:,0].detach().cpu().detach().numpy() > min_up_sample)*(Lbls_query[i0][:,0].cpu().detach().numpy() < min_up_sample), 0.0)
			prob_up_sample = np.maximum((out[1][:,0].cpu().detach()*(out[1][:,0].cpu().detach() > min_up_sample)*(Lbls_query[i0][:,0].detach() < min_up_sample)).cpu().detach().numpy(), 0.0)
			# prob_up_sample = 
			if prob_up_sample.sum() == 0: prob_up_sample = np.ones(len(prob_up_sample))
			prob_up_sample = prob_up_sample/prob_up_sample.sum() ## Can transform these probabilities or clip them
			x_query_sample, x_query_sample_t = sample_dense_queries(X_query[i0][:,0:3].cpu().detach().numpy(), X_query[i0][:,3].cpu().detach().numpy(), prob_up_sample, lat_range_extend, lon_range_extend, depth_range, src_x_kernel, src_depth_kernel, src_t_kernel, time_shift_range, ftrns1, ftrns2, replace = False, randomize = False) # replace = False
			out_query = mz.forward_queries(torch.Tensor(ftrns1(x_query_sample)).to(device), torch.Tensor(x_query_sample_t).to(device), train = True) # x_query_cart, t_query
			lbls_query = compute_source_labels(x_query_sample, x_query_sample_t, lp_srcs[i0][:,0:3].cpu().detach().numpy(), lp_srcs[i0][:,3].cpu().detach().numpy(), src_spatial_kernel, src_t_kernel, ftrns1) ## Compute updated labels


			if use_sigmoid == False:


				loss_negative = gaussian_heatmap_loss(out_query, torch.Tensor(lbls_query).to(device)) # weights[1]*


			else:

				## Not completely implemented
				ifind_mask = torch.Tensor(lbls_query[:,0] > min_val_lbl).long().to(device)
				loss_negative = (1/100.0)*bce_loss(out_query[:,1], ifind_mask.float()) + mse_loss(out_query[ifind_mask,0], torch.Tensor(lbls_query).to(device)[ifind_mask,0]) # weights[1]*		

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
						mask_loss = torch.Tensor((np.abs(Lbls_query[i0][ind_consistency::].cpu().detach().numpy() - Lbls_save[0][ind_consistency::]) < 0.01)).to(device).float()*(mask_lbls_query_lv[i0][ind_consistency::] > 0)  # .float()

						loss_consistency = consistency_loss(out[1][ind_consistency::][mask_loss.long()], out_save[0][ind_consistency::][mask_loss.long()])


					else:

						mask_loss = torch.Tensor((np.abs(Lbls_query[i0][ind_consistency::].cpu().detach().numpy() - Lbls_save[0][ind_consistency::]) < 0.01)).to(device).float()  # .float()
						loss_consistency = mse_loss(out[1][ind_consistency::][mask_loss.long()][:,0], out_save[0][ind_consistency::][mask_loss.long()][:,0]) # )/torch.maximum(torch.Tensor([1.0]).to(device), (weight1*weights[0] + weight2*weights[1]))
	

					loss_consistency_val += loss_consistency.item() # /n_batch
					loss_consistency_flag = True


			out_save = [out[1].detach()]
			Lbls_save = [Lbls_query[i0].cpu().detach().numpy()]
			iter_loss = [i, inc]
			X_query_save = [X_query[i0].cpu().detach().numpy()]



		# if (use_gradient_loss == True)*(i > int(n_epochs/5)):
		if (use_gradient_loss == True)*(i > n_burn_in):


			if init_gradient_loss == False:
				init_gradient_loss, mz.activate_gradient_loss = True, True
			else:

				loss_grad_magnitude_spc1 = l1_loss(torch.norm(torch.Tensor([src_kernel_mean]).to(device)*grad_grid_src, dim = 1), torch.norm(torch.Tensor(Lbls_grad_spc[i0]).to(device), dim = 1))
				loss_grad_magnitude_time1 = l1_loss(torch.norm(torch.Tensor([src_t_kernel]).to(device)*grad_grid_t, dim = 1), torch.norm(torch.Tensor(Lbls_grad_t[i0]).to(device), dim = 1))

				loss_grad_magnitude_spc2 = l1_loss(torch.norm(torch.Tensor([src_kernel_mean]).to(device)*grad_query_src, dim = 1), torch.norm(torch.Tensor(Lbls_query_grad_spc[i0]).to(device), dim = 1))
				loss_grad_magnitude_time2 = l1_loss(torch.norm(torch.Tensor([src_t_kernel]).to(device)*grad_query_t.reshape(-1), dim = 1), torch.norm(torch.Tensor(Lbls_query_grad_t[i0]).to(device), dim = 1))

				# loss = 0.5*loss + 0.5*loss_grad
				loss_grad_val += 0.5*loss_grad.item()/n_batch


		# pre_scale_weights1 = [10.0, 10.0] ## May have to decrease these as training goes on (as MSE converged much closer to zero)
		pre_scale_weights1 = [2.0, 2.0] ## May have to decrease these as training goes on (as MSE converged much closer to zero)
		pre_scale_weights2 = [0.5e1, 0.5e2, 1e1, 12.0]

		# pre_scale_weights2 = [1e4, 1e4]


		## Compute base losses
		loss_dict = {
		# 'loss_dice1': loss_base1, # loss_dice1
		'loss_base1': 0.2*loss_base1, # loss_dice1
		'loss_dice2': loss_dice2,
		'loss_dice3': 0.5*loss_dice3 + 0.5*loss_dice4,
		# 'loss_dice4': loss_dice4,
		'loss_regression1': loss_smooth_l1*pre_scale_weights1[0],
		'loss_regression2': loss_smooth_l2*pre_scale_weights1[1],
		# 'loss_cap1': loss_cap1*pre_scale_weights1[0],
		# 'loss_cap2': loss_cap2*pre_scale_weights1[1],
		}



		if i > n_burn_in:

			if use_real_data_sample_v[inc] == False:
				loss_dict.update({'loss_negative': loss_negative*pre_scale_weights2[0]})

			if loss_consistency_flag == True:
				loss_dict.update({'loss_consistency': loss_consistency*pre_scale_weights2[1]})
			
			loss_dict.update({'loss_cap1': loss_cap1*pre_scale_weights2[2]})
			loss_dict.update({'loss_cap2': loss_cap2*pre_scale_weights2[3]})
			# loss_dict.update({'loss_cap2': loss_cap2*pre_scale_weights1[1]})


		loss = LossBalancer(loss_dict, accum_steps = n_batch, is_last_accum_step = (inc == (n_batch - 1))) # losses_dict: dict, accum_steps: int = None, is_last_accum_step: bool = False
		loss = loss/n_batch



		n_visualize_step = 1000
		n_visualize_fraction = 0.2
		make_visualize_predictions = False
		if (make_visualize_predictions == True)*(np.mod(i, n_visualize_step) == 0)*(inc == 0): # (i0 < n_visualize_fraction*n_batch)
			save_plots_path = path_to_file + seperator + 'Plots' + seperator
			out_plot = [out[0], out[1], out[2], out[3]]
			visualize_predictions(out_plot, Lbls_query[i0], pick_lbls, X_query[i0], lp_times[i0], lp_stations[i0], Locs[i0], data, i0, save_plots_path, n_step = i, n_ver = n_ver)


		if inc != (n_batch - 1):
			loss.backward(retain_graph = False)
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


	use_grad_norm = False
	if use_grad_norm == True:
		torch.nn.utils.clip_grad_norm_(mz.parameters(), max_norm = 5.0)


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




def compute_loss(x, n_repeat = 10, return_metrics = False):

	## Source threshold x[0]
	## Association threshold x[1]
	## Could also estimate thresholds based on a function

	with torch.no_grad():

		n_found = 0
		n_target = 0
		n_match = 0
		Srcs = []
		Srcs_trgt = []
		Matches = []
		Ind, Ind1 = [], []
		c1, c2 = 0, 0

		inc = 0
		for n in range(n_repeat):


			inputs = dataset.__getitem__(np.random.choice(len(files_load)))
			## Need to overwrite the data entries in input_tensors
			if use_gradient_loss == False:
				lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l = inputs[1]
			else:
				lp_srcs, lp_times, lp_stations, lp_phases, X_query, Lbls, Lbls_query, Locs, pick_lbls_l, x_src_query_cart_l, spatial_vals_l, Lbls_grad_spc, Lbls_query_grad_spc, Lbls_grad_t, Lbls_query_grad_t = inputs[1]		
			input_tensors_l = inputs[0]
			for j in range(n_batch): input_tensors_l[j][4] = Data(x = spatial_vals_l[j], edge_index = input_tensors_l[j][4]) ## Ideally remove these
			for j in range(n_batch): input_tensors_l[j][5] = Data(x = spatial_vals_l[j], edge_index = input_tensors_l[j][5])
			input_tensors_l = to_float32(to_gpu(inputs[0])) if device.type == 'cuda' else to_float32(inputs[0])
			if device.type == 'cuda': pick_lbls_l = to_gpu(pick_lbls_l)

			for i0 in range(len(X_query)):


				if len(lp_times[i0]) > 0:

					if use_gradient_loss == False:
						out = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])
					else:
						# out, grads = mz(*input_tensors)
						out, grads = mz(*input_tensors_l[i0], save_state = True) if (use_negative_loss == True)*(np.mod(i, use_negative_loss_step) == 0) else mz(*input_tensors_l[i0])
						grad_grid_src, grad_grid_t, grad_query_src, grad_query_t = grads


					## First detect maxima
					ifind = torch.where(out[1][:,0] > x[0])[0].cpu().detach().numpy()
				
				else:

					ifind = np.array([]).astype('int')


				if len(ifind) > 0:

					mp = LocalMarching(device = device)
					tc_win = pred_params[2]*1.35 ## Note: could learn these clustering windows as well
					sp_win = pred_params[3]*1.35
					scale_depth_clustering = 0.2
					srcs_out = mp(np.concatenate((X_query[i0][ifind].cpu().detach().numpy(),out[1][ifind,0].cpu().detach().numpy().reshape(-1,1)), axis = 1), ftrns1, tc_win = tc_win, sp_win = sp_win, scale_depth = scale_depth_clustering, n_steps_max = 2, use_directed = False)


					## Apply bipartite matching
					temporal_win_match = src_t_kernel*2.0
					spatial_win_match = src_x_kernel*2.0
					if len(lp_srcs[i0]) > 0:
						matches = maximize_bipartite_assignment(lp_srcs[i0].cpu().detach().numpy(), srcs_out, ftrns1, ftrns2, temporal_win = temporal_win_match, spatial_win = spatial_win_match)[0]
					else:
						matches = np.zeros((0,2))

				else:

					srcs_out = np.zeros((0,5))
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

		print('Obtained %0.4f precision, %0.4f recall, %0.4f f1'%(prec, rec, f1))

	if return_metrics == False:


		return -1.0*f1

	else:

		return f1, prec, rec, Srcs, Srcs_trgt, Matches, Ind, Ind1 ## Can include detected events








































