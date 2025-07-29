
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
import pathlib
import glob
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
use_amplitudes = config['use_amplitudes']
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

if use_amplitudes == True:
	n_mag_ver = 1
	mags_supp = np.load(path_to_file + 'Grids' + seperator + 'trained_magnitude_model_ver_%d_supplemental.npz'%n_mag_ver)
	mag_grid, k_grid = mags_supp['mag_grid'], int(mags_supp['k_grid'])
	Mag = Magnitude(torch.Tensor(locs).to(device), torch.Tensor(mag_grid).to(device), ftrns1_diff, ftrns2_diff, k = k_grid, device = device).to(device)
	Mag.load_state_dict(torch.load(path_to_file + 'Grids' + seperator + 'trained_magnitude_model_ver_%d.h5'%n_mag_ver, map_location = device))
	Mags = Magnitude(torch.Tensor(locs).to(device), torch.Tensor(mag_grid).to(device), ftrns1_diff, ftrns2_diff, k = k_grid, device = device).to(device)
	Mags.load_state_dict(torch.load(path_to_file + 'Grids' + seperator + 'trained_magnitude_model_ver_%d.h5'%n_mag_ver, map_location = device))
	# Mags.bias = torch.cat((Mags.bias, Mags.bias.mean(0, keepdims = True)), dim = 0) ## Append one entry for the "null" vector
	mags_supp.close()

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
	mag_vals = np.arange(-3.0, 10.0, 0.01)
	dist_vals_p = pdist_p(mag_vals)
	dist_vals_s = pdist_s(mag_vals)
	dist_supp.close()
	print('Will use amplitudes since a magnitude model was loaded')
	
	## Load emperical distribution of pick amplitudes
	n_rand_choose = 100
	st_load = glob.glob(path_to_file + 'Picks/19*') # Load years 1900's
	st_load.extend(glob.glob(path_to_file + 'Picks/20*')) # Load years 2000's
	iarg = np.argsort([int(st_load[i].split(seperator)[-1]) for i in range(len(st_load))])
	st_load = [st_load[i] for i in iarg]

	st_load_l = []
	for i in range(len(st_load)):
		# st = glob.glob(st_load[i] + seperator + '*ver_%d.npz'%(n_ver_picks))
		st = glob.glob(st_load[i] + seperator + '*ver_*.npz')
		if len(st) > 0:
			st_load_l.extend(st)
	iperm = np.random.permutation(len(st_load_l))
	iperm = iperm[0:np.minimum(n_rand_choose, len(st_load_l))]
	st_load = [st_load_l[i] for i in iperm]

	log_amp_emperical = []
	log_amp_ind_emperical = []
	q_range_emperical = np.linspace(0, 1, 30)
	n_range_emperical = len(q_range_emperical)
	log_amp_sta_distb = []
	for i in range(len(st_load)):
		z = np.load(st_load[i])
		P = z['P']
		z.close()
		log_amp_emperical.append(np.log10(P[:,2]))
		log_amp_ind_emperical.append(P[:,1].astype('int'))
	log_amp_emperical = np.hstack(log_amp_emperical)
	log_amp_ind_emperical = np.hstack(log_amp_ind_emperical)
	tree_ind = cKDTree(log_amp_ind_emperical.reshape(-1,1))
	log_amp_inds = tree_ind.query_ball_point(np.arange(len(locs)).reshape(-1,1), r = 0)
	for i in range(len(locs)):
		if len(log_amp_inds[i]) == 0:
			log_amp_sta_distb.append(np.quantile(log_amp_emperical, q_range_emperical).reshape(1,-1))
		else:
			log_amp_sta_distb.append(np.quantile(log_amp_emperical[log_amp_inds[i]], q_range_emperical).reshape(1,-1))
	log_amp_sta_distb = np.vstack(log_amp_sta_distb)
	log_amp_sta_distb_diff = np.diff(log_amp_sta_distb, axis = 1)

else:

	log_amp_random = np.random.rand(10000)*2.0 - 8.0 ## Arbitrary random amplitudes
	for i in range(len(locs)):
		log_amp_sta_distb.append(np.quantile(log_amp_random, q_range_emperical).reshape(1,-1))
	log_amp_sta_distb = np.vstack(log_amp_sta_distb)
	log_amp_sta_distb_diff = np.diff(log_amp_sta_distb, axis = 1)

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
training_params_3 = [n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, miss_pick_fraction, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources, use_extra_nearby_moveouts, use_amplitudes]

def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, plot_on = False, verbose = False, skip_graphs = False, return_only_data = False):

	if verbose == True:
		st = time.time()

	k_sta_edges, k_spc_edges, k_time_edges = graph_params
	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	n_spc_query, n_src_query = training_params
	spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max, min_misfit_allowed, total_bias = training_params_2
	n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, miss_pick_fraction, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources, use_extra_nearby_moveouts, use_amplitudes = training_params_3

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

	if use_amplitudes == True:
		mag_ratio = [0.75, 1.25]
		## Need to "match" these distances to the magnitude scale
		## One option: overwrite the distance sampling to sample realistic distances
		## for a given magnitude. And specify the magnitude sampling distribution
		## Or, use the current distance values, and find a corresponding magnitude (perhaps
		## with noise)
		# pdb.set_trace()
		## First approach, we estimate magnitude from the distance value, and add a pertubation
		imag1 = np.argmin(np.abs(dist_thresh - dist_vals_p.reshape(1,-1)), axis = 1)
		imag2 = np.argmin(np.abs(dist_thresh - dist_vals_s.reshape(1,-1)), axis = 1)
		mag_scale = np.random.rand(len(dist_thresh))*(mag_ratio[1] - mag_ratio[0]) + mag_ratio[0]
		src_magnitude = mag_scale*(0.5*mag_vals[imag1] + 0.5*mag_vals[imag2])
	
	# create different distance dependent thresholds.
	dist_thresh_p = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
	dist_thresh_s = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None]

	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

	arrivals_theoretical = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()

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

	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
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
		n_spikes_extent = np.random.randint(int(np.floor(n_sta*0.5)), high = n_sta, size = n_spikes) ## This many stations per spike
		time_spikes = np.random.rand(n_spikes)*T
		sta_ind_spikes = [np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)]
		if len(sta_ind_spikes) > 0: ## Add this catch, to avoid error of np.hstack if len(sta_ind_spikes) == 0
			sta_ind_spikes = np.hstack(sta_ind_spikes)
			sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*spike_time_spread for j in range(n_spikes)])
			false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
			arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes

	## Need to calibrate the magnitude-distance tradeoff
	if use_amplitudes == True:
		ratio_amplitudes = [0.5, 1.5] ## Per source (correlated noise)
		ratio_amplitudes_sta = [0.5, 1.5] ## Per station (iid noise)
		ratio_phases = [0.75, 1.25]
		## Compute amplitudes of picks
		itrue_p = np.where((arrivals[:,2] > -1)*(arrivals[:,4] == 0))[0] ## Real picks
		itrue_s = np.where((arrivals[:,2] > -1)*(arrivals[:,4] == 1))[0] ## Real picks
		ifalse = np.where(arrivals[:,2] == -1)[0] ## False picks

		log_amp_p = Mag.train(torch.Tensor(arrivals[itrue_p,1].astype('int')).long().to(device), torch.Tensor(src_positions[arrivals[itrue_p,2].astype('int'),0:3]).to(device), torch.Tensor(src_magnitude[arrivals[itrue_p,2].astype('int')]).to(device), torch.zeros(len(itrue_p)).long().to(device)).cpu().detach().numpy() # 1## 
		log_amp_s = Mag.train(torch.Tensor(arrivals[itrue_s,1].astype('int')).long().to(device), torch.Tensor(src_positions[arrivals[itrue_s,2].astype('int'),0:3]).to(device), torch.Tensor(src_magnitude[arrivals[itrue_s,2].astype('int')]).to(device), torch.ones(len(itrue_s)).long().to(device)).cpu().detach().numpy() # 1## 
		per_source_scale = np.random.rand(n_src)*(ratio_amplitudes[1] - ratio_amplitudes[0]) + ratio_amplitudes[0]
		per_source_scale_p = np.random.rand(n_src)*(ratio_phases[1] - ratio_phases[0]) + ratio_phases[0]
		per_source_scale_s = np.random.rand(n_src)*(ratio_phases[1] - ratio_phases[0]) + ratio_phases[0]
		per_pick_scale_p = np.random.rand(len(itrue_p))*(ratio_amplitudes_sta[1] - ratio_amplitudes_sta[0]) + ratio_amplitudes_sta[0]
		per_pick_scale_s = np.random.rand(len(itrue_s))*(ratio_amplitudes_sta[1] - ratio_amplitudes_sta[0]) + ratio_amplitudes_sta[0]

		## Add noise; note, b = log(a); y = log(ac) = log(a) + log(c) = b + log(c)
		log_amp_p = log_amp_p + np.log10(per_pick_scale_p) + np.log10(per_source_scale[arrivals[itrue_p,2].astype('int')]) + np.log10(per_source_scale_p[arrivals[itrue_p,2].astype('int')])
		log_amp_s = log_amp_s + np.log10(per_pick_scale_s) + np.log10(per_source_scale[arrivals[itrue_s,2].astype('int')]) + np.log10(per_source_scale_s[arrivals[itrue_s,2].astype('int')])
		irand_quantile = np.random.choice(n_range_emperical - 1, size = len(ifalse))
		log_amp_false = log_amp_sta_distb[arrivals[ifalse,1].astype('int'), irand_quantile] + np.random.rand(len(ifalse))*log_amp_sta_distb_diff[arrivals[ifalse,1].astype('int'), irand_quantile]
		log_amp = np.nan*np.ones(len(arrivals))
		log_amp[itrue_p] = log_amp_p
		log_amp[itrue_s] = log_amp_s
		log_amp[ifalse] = log_amp_false
		assert(np.isnan(log_amp).sum() == 0)

	else:
		ifalse = np.arange(len(arrivals))
		irand_quantile = np.random.choice(n_range_emperical - 1, size = len(ifalse))
		log_amp = log_amp_sta_distb[arrivals[ifalse,1].astype('int'), irand_quantile] + np.random.rand(len(ifalse))*log_amp_sta_distb_diff[arrivals[ifalse,1].astype('int'), irand_quantile]
		
	

	# use_stable_association_labels = True
	## Check which true picks have so much noise, they should be marked as `false picks' for the association labels
	iexcess_noise = []
	if use_stable_association_labels == True: ## It turns out association results are fairly sensitive to this choice
		# thresh_noise_max = 2.5 # ratio of sig_t*travel time considered excess noise
		# min_misfit_allowed = 1.0 # min misfit time for establishing excess noise (now set in train_config.yaml)
		iz = np.where(arrivals[:,4] >= 0)[0]
		noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
		iexcess_noise = np.where(np.abs(noise_values) > np.maximum(min_misfit_allowed, thresh_noise_max*sig_t*arrivals[iz,0]))[0]
		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise
		if len(iexcess_noise) > 0: ## Set these arrivals to "false arrivals", since noise is so high
			init_phase_type = arrivals[iz[iexcess_noise],4]
			arrivals[iz[iexcess_noise],2] = -1
			arrivals[iz[iexcess_noise],3] = 0
			arrivals[iz[iexcess_noise],4] = -1
	
	else: ## This was the original version
		iz = np.where(arrivals[:,4] >= 0)[0]
		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]

	
	## Check which sources are active
	source_tree_indices = cKDTree(arrivals[:,2].reshape(-1,1))
	lp = source_tree_indices.query_ball_point(np.arange(n_events).reshape(-1,1), r = 0)
	lp_backup = [lp[j] for j in range(len(lp))]
	n_unique_station_counts = np.array([len(np.unique(arrivals[lp[j],1])) for j in range(n_events)])
	cnt_p_srcs = np.array([len(np.where(arrivals[lp[j],4] == 0)[0]) for j in range(n_events)])
	cnt_s_srcs = np.array([len(np.where(arrivals[lp[j],4] == 1)[0]) for j in range(n_events)])
	# active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
	active_sources = np.where(((n_unique_station_counts >= min_sta_arrival)*((cnt_p_srcs + cnt_s_srcs) >= min_pick_arrival)))[0] # subset of sources

	
	src_times_active = src_times[active_sources]

	## If true, return only the synthetic arrivals
	if return_only_data == True:
		srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
		data = [arrivals, srcs, active_sources, log_amp] ## Note: active sources within region are only active_sources[np.where(inside_interior[active_sources] > 0)[0]]
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
				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()

	time_samples = np.sort(time_samples)

	max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))

	tree_src_times_all = cKDTree(src_times[:,np.newaxis])
	tree_src_times = cKDTree(src_times_active[:,np.newaxis])
	lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)

	st = time.time()
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
	if len(lp_concat) == 0:
		lp_concat = np.array([0]) # So it doesnt fail?
	arrivals_select = arrivals[lp_concat]
	log_amp_select = log_amp[lp_concat]
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
		i0 = np.random.randint(0, high = len(x_grids))
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

		Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
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


	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch


	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
	n_arvs = arrivals_select.shape[0]

	# Rather slow!
	iargsort = np.argsort(arrivals_select[:,0])
	arrivals_select = arrivals_select[iargsort]
	phase_observed_select = phase_observed_select[iargsort]
	log_amp_select = log_amp_select[iargsort]
	
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

#	rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)

	if use_amplitudes == True:
	
		## Can add sign information here
		diff_rel_t = np.abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0])
		iarg_rel_p = np.argmin(diff_rel_t, axis = 1)
		ivec = np.arange(diff_rel_t.shape[0])
		rel_t_p = diff_rel_t[ivec,iarg_rel_p] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		log_amp_p_val = log_amp_select[ip_p_pad[ivec,iarg_rel_p]]
	
		diff_rel_t = np.abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0])
		iarg_rel_s = np.argmin(diff_rel_t, axis = 1)
		ivec = np.arange(diff_rel_t.shape[0])
		rel_t_s = diff_rel_t[ivec,iarg_rel_s] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		log_amp_s_val = log_amp_select[ip_s_pad[ivec,iarg_rel_s]]

	else:

		rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)
	
	
	## With phase type information
	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

	if use_amplitudes == True:
	
		null_log_amp = -10.0
		if len(iwhere_p) > 0:
			# rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			diff_rel_t = np.abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0])
			iarg_rel_p = np.argmin(diff_rel_t, axis = 1)
			ivec = np.arange(diff_rel_t.shape[0])
			rel_t_p1 = diff_rel_t[ivec,iarg_rel_p] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			log_amp_p1_val = log_amp_select[iwhere_p[ip_p1_pad[ivec,iarg_rel_p]]]
		else:
			# rel_t_p1 = np.zeros(rel_t_p.shape)
			rel_t_p1 = np.random.choice([-1.0, 1.0], size = rel_t_p.shape)*np.ones(rel_t_p.shape)*kernel_sig_t*10.0 ## Need to place null values as large offset, so they map to zero
			log_amp_p1_val = null_log_amp*np.ones(rel_t_p.shape)
	
		if len(iwhere_s) > 0:
			# rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)
			diff_rel_t = np.abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0])
			iarg_rel_s = np.argmin(diff_rel_t, axis = 1)
			ivec = np.arange(diff_rel_t.shape[0])
			rel_t_s1 = diff_rel_t[ivec,iarg_rel_s] ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			log_amp_s1_val = log_amp_select[iwhere_s[ip_s1_pad[ivec,iarg_rel_s]]]
		else:
			# rel_t_s1 = np.zeros(rel_t_s.shape)
			rel_t_s1 = np.random.choice([-1.0, 1.0], size = rel_t_s.shape)*np.ones(rel_t_s.shape)*kernel_sig_t*10.0 ## Need to place null values as large offset, so they map to zero
			log_amp_s1_val = null_log_amp*np.ones(rel_t_s.shape)

	else:


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

	lp_times = []
	lp_stations = []
	lp_phases = []
	lp_meta = []
	lp_srcs = []
	lp_amp = []

	scale_mag = 5.0
	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grids[grid_select].shape[0]
		n_sta_slice = len(sta_select)

		if use_amplitudes == True:
			inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 8)) # Could make this smaller (on the subset of stations), to begin with.
		else:	
			inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.

		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))

		if use_amplitudes == True:
			## If this is subsetting source nodes (note, check which source graph is being used) and station indices (check, absolute station indices)
			## Then, can apply the Magnitude model here to reduce the amplitude measurements to magnitude features on the Cartesian product
			zero_vec = torch.zeros(len(ind_select)).long().to(device)
			one_vec = torch.ones(len(ind_select)).long().to(device)
			p_index = torch.Tensor(Trv_subset_p[ind_select,1].astype('int')).long().to(device)
			s_index = torch.Tensor(Trv_subset_s[ind_select,1].astype('int')).long().to(device)
			src_p = torch.Tensor(x_grids[grid_select][Trv_subset_p[ind_select,2].astype('int')]).to(device)
			src_s = torch.Tensor(x_grids[grid_select][Trv_subset_s[ind_select,2].astype('int')]).to(device)
			mag_pred_p = Mag.mag(p_index, src_p, torch.Tensor(log_amp_p_val[ind_select]).to(device), zero_vec).cpu().detach().numpy()
			mag_pred_s = Mag.mag(s_index, src_s, torch.Tensor(log_amp_s_val[ind_select]).to(device), one_vec).cpu().detach().numpy()
			mag_pred_p1 = Mag.mag(p_index, src_p, torch.Tensor(log_amp_p1_val[ind_select]).to(device), zero_vec).cpu().detach().numpy()
			mag_pred_s1 = Mag.mag(s_index, src_s, torch.Tensor(log_amp_s1_val[ind_select]).to(device), one_vec).cpu().detach().numpy()
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 4] = mag_pred_p/scale_mag
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 5] = mag_pred_s/scale_mag
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 6] = mag_pred_p1/scale_mag
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 7] = mag_pred_s1/scale_mag
			# pdb.set_trace()
		
		trv_out = x_grids_trv[grid_select][:,sta_select,:] ## Subsetting, into sliced indices.
		Inpts.append(inpt[:,sta_select,:]) # sub-select, subset of stations.
		Masks.append(1.0*(inpt[:,sta_select,:] > thresh_mask))
		Trv_out.append(trv_out)
		Locs.append(locs[sta_select])
		X_fixed.append(x_grids[grid_select])

		## Assemble pick datasets
		perm_vec = -1*np.ones(n_sta)
		perm_vec[sta_select] = np.arange(len(sta_select))
		meta = arrivals[lp[i],:]
		phase_vals = phase_observed[lp[i]]
		log_amp_vals = log_amp[lp[i]]
		times = meta[:,0]
		indices = perm_vec[meta[:,1].astype('int')]
		ineed = np.where(indices > -1)[0]
		times = times[ineed] ## Overwrite, now. Double check if this is ok.
		indices = indices[ineed]
		log_amp_vals = log_amp_vals[ineed]
		phase_vals = phase_vals[ineed]
		meta = meta[ineed]

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
		lp_amp.append(log_amp_vals[lex_sort])
		lp_srcs.append(src_subset)

		if skip_graphs == False:
		
			A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), k = k_sta_edges + 1).flip(0).contiguous())[0]
			A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), k = k_spc_edges + 1).flip(0).contiguous())[0]
			## Cross-product graph is: source node x station node. Order as, for each source node, all station nodes.
	
			# Cross-product graph, nodes connected by: same source node, connected stations
			A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).to(device)).contiguous()
			A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).to(device)).contiguous()	
	
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
		
		x_query = np.random.rand(n_spc_query, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.

		if len(lp_srcs[-1]) > 0:
			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]

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
		
		if len(active_sources_per_slice) == 0:
			lbls_grid = np.zeros((x_grids[grid_select].shape[0], len(t_slice)))
			lbls_query = np.zeros((n_spc_query, len(t_slice)))
		else:
			active_sources_per_slice = active_sources_per_slice.astype('int')
			
			# Combine components
			# lbls_grid = (np.expand_dims(spatial_exp_term.sum(2), axis=1) * temporal_exp_term).max(2)
			# lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

			lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

		
		X_query.append(x_query)
		Lbls.append(lbls_grid)
		Lbls_query.append(lbls_query)

	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
	data = [arrivals, srcs, active_sources, log_amp]	## Note: active sources within region are only active_sources[np.where(inside_interior[active_sources] > 0)[0]]

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_amp, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

def pick_labels_extract_interior_region(xq_src_cart, xq_src_t, source_pick, src_slice, lat_range_interior, lon_range_interior, ftrns1, sig_x = 15e3, sig_t = 6.5): # can expand kernel widths to other size if prefered

	iz = np.where(source_pick[:,1] > -1.0)[0]
	lbl_trgt = torch.zeros((xq_src_cart.shape[0], source_pick.shape[0], 2)).to(device)
	src_pick_indices = source_pick[iz,1].astype('int')

	inside_interior = ((src_slice[src_pick_indices,0] <= lat_range_interior[1])*(src_slice[src_pick_indices,0] >= lat_range_interior[0])*(src_slice[src_pick_indices,1] <= lon_range_interior[1])*(src_slice[src_pick_indices,1] >= lon_range_interior[0]))

	if len(iz) > 0:
		d = torch.Tensor(inside_interior.reshape(1,-1)*np.exp(-0.5*(pd(xq_src_cart, ftrns1(src_slice[src_pick_indices,0:3]))**2)/(sig_x**2))*np.exp(-0.5*(pd(xq_src_t.reshape(-1,1), src_slice[src_pick_indices,3].reshape(-1,1))**2)/(sig_t**2))).to(device)
		lbl_trgt[:,iz,0] = d*torch.Tensor((source_pick[iz,0] == 0)).to(device).float()
		lbl_trgt[:,iz,1] = d*torch.Tensor((source_pick[iz,0] == 1)).to(device).float()

	return lbl_trgt

def sample_picks(P, locs_abs, t_sample_win = 120.0, windows = [40e3, 150e3, 300e3], t_win_ball = [10.0, 15.0, 25.0]): # windows = [40e3, 150e3, 300e3]

	Trgts = []

	iunique = np.sort(np.unique(P[:,1]).astype('int'))
	lunique = len(iunique)

	locs_use = np.copy(locs_abs[iunique]) # Overwrite locs_use

	## An additional metric than can be added is the number of stations with a pick 
	## within a "convex hull" (or the ratio of stations) connecting the source and stations at different distances..
	## Hence, measures how much "filled in" versus "noisy" the foot print of associated stations is
	## (would need a reference catalog, and should only sample "large" sources, since reference
	## catalog would be biased to large sources)

	perm_vec = -1*np.ones(len(locs_abs))
	perm_vec[iunique] = np.arange(len(iunique))
	perm_vec = perm_vec.astype('int')
	P[:,1] = perm_vec[P[:,1].astype('int')] ## Overwrite pick indices
	iunique = np.sort(np.unique(P[:,1]).astype('int'))
	assert(len(iunique) == lunique)
	assert(iunique.min() == 0)
	assert(iunique.max() == (lunique - 1))

	pw_dist = pd(ftrns1(locs_use), ftrns1(locs_use))

	max_t_observed = P[:,0].max()
	counts_in_time, bins_in_time = np.histogram(P[:,0], bins = np.arange(0, max_t_observed + 3600, t_sample_win))
	upper_fifth_percentile = np.where(counts_in_time >= np.quantile(counts_in_time, 0.95))[0]


	tree_times = cKDTree(P[:,[0]])
	tree_indices = cKDTree(P[:,[1]])

	## [1] Average pick rates

	# ifind_per_sta = [np.where(P[:,1] == iunique[j])[0] for j in range(len(iunique))]
	# counts_per_sta = [len(ifind_per_sta[j]) for j in range(len(iunique))]
	ifind_per_sta = [np.where(P[:,1] == j)[0] for j in range(len(locs_use))]
	counts_per_sta = [len(ifind_per_sta[j]) for j in range(len(locs_use))]
	counts_per_hour = np.vstack([np.histogram(P[ifind_per_sta[j],0], bins = np.arange(0, max_t_observed + 3600, 3600.0))[0].reshape(1,-1) for j in range(len(ifind_per_sta))])
	upper_fifth_percentile_stas = iunique[np.where(counts_per_sta >= np.quantile(counts_per_sta, 0.95))[0]]

	Quants_counts = np.quantile(counts_per_hour, np.arange(0.1, 1.0, 0.2), axis = 0)
	Trgts.append(np.median(Quants_counts, axis = 1))

	## [2] Average "ratio" of picks within narrow spatial windows compared to outside, over max_t window (for random origin times)
	# windows = [40e3, 150e3, 300e3] # [0.029238671690285857, 0.07309667922571464, 0.14619335845142928] of pw_dist_max
	Ratio_bins = [[] for w in windows]
	num_iter = 150
	for j in range(num_iter):
		for inc, k in enumerate(range(len(windows))):

			ipick = np.random.choice(locs_use.shape[0]) ## Pick random station
			ifind = np.where(pw_dist[ipick,:] < windows[k])[0] ## Find other stations within window distance
			# ifind_outside = np.delete(np.arange(locs_use.shape[0]), ifind, axis = 0) ## Stations outside window distance

			## Choose random origin time
			t0 = np.random.rand()*3600*24

			## Find all picks within t0 + max_t*fraction

			fraction = 0.3
			ifind_time = np.array(tree_times.query_ball_point(np.array([t0]).reshape(1,1), r = max_t*fraction)[0]).astype('int')
			# tree_pick_indices = cKDTree(P[ifind_time,1].reshape(-1,1))
			tree_pick_indices = cKDTree(ifind.reshape(-1,1))

			## Of these picks, find subset that are nearby root station, and those that are not.
			ifind_picks_inside = np.where(tree_pick_indices.query(P[ifind_time,1].astype('int').reshape(-1,1))[0] == 0)[0]
			ifind_picks_outside = np.delete(np.arange(len(ifind_time)), ifind_picks_inside, axis = 0) ## Stations outside window distance

			Ratio_bins[inc].append(len(ifind_picks_inside)/np.maximum(len(ifind_picks_outside), 1.0))

	Ratio_bins = np.vstack([np.quantile(Ratio_bins[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Ratio_bins))])
	Trgts.append(Ratio_bins)


	## [3] Average "ratio" of picks within narrow spatial windows compared to outside, over max_t window (for "optimal" origin times and stations; e.g., near sources)
	# windows = [40e3, 150e3, 300e3] # [0.029238671690285857, 0.07309667922571464, 0.14619335845142928] of pw_dist_max
	Ratio_bins1 = [[] for w in windows]
	num_iter = 150
	prob_counts = 1.0/(1.0 + np.flip(np.argsort(Quants_counts.mean(0))))
	prob_counts = prob_counts/prob_counts.sum()
	for j in range(num_iter):
		for inc, k in enumerate(range(len(windows))):

			ipick = np.random.choice(upper_fifth_percentile_stas) ## Pick random station
			ifind = np.where(pw_dist[ipick,:] < windows[k])[0] ## Find other stations within window distance
			# ifind_outside = np.delete(np.arange(locs_use.shape[0]), ifind, axis = 0) ## Stations outside window distance

			## Choose origin time focused on the high pick count time intervals
			t0 = bins_in_time[np.random.choice(upper_fifth_percentile)] + np.random.rand()*t_sample_win

			## Find all picks within t0 + max_t*fraction

			fraction = 0.3
			ifind_time = np.array(tree_times.query_ball_point(np.array([t0]).reshape(1,1), r = max_t*fraction)[0]).astype('int')
			# tree_pick_indices = cKDTree(P[ifind_time,1].reshape(-1,1))
			tree_pick_indices = cKDTree(ifind.reshape(-1,1))

			## Of these picks, find subset that are nearby root station, and those that are not.
			ifind_picks_inside = np.where(tree_pick_indices.query(P[ifind_time,1].astype('int').reshape(-1,1))[0] == 0)[0]
			ifind_picks_outside = np.delete(np.arange(len(ifind_time)), ifind_picks_inside, axis = 0) ## Stations outside window distance

			Ratio_bins1[inc].append(len(ifind_picks_inside)/np.maximum(len(ifind_picks_outside), 1.0))

	Ratio_bins1 = np.vstack([np.quantile(Ratio_bins1[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Ratio_bins1))])
	Trgts.append(Ratio_bins1)

	## [4] Counts of station, for each neighboring station, if they have a pick at a similar time
	## Instead of random picks, could pick picks nearby times of high activity
	k_sta = 1*k_sta_edges + 0 # 10
	# locs_use_use = locs_use[iunique]
	edges = remove_self_loops(knn(torch.Tensor(ftrns1(locs_use)).to(device)/1000.0, torch.Tensor(ftrns1(locs_use)).to(device)/1000.0, k = k_sta))[0].flip(0).cpu().detach().numpy()

	tree_edges = cKDTree(edges[1].reshape(-1,1))

	# t_win_ball = [10.0, 15.0, 25.0]
	num_picks = 1000
	n_picks = len(P)
	Ratio_neighbors = [[] for t in t_win_ball]
	for j in range(num_picks):
		for k in range(len(t_win_ball)):
			ichoose = np.random.choice(n_picks)
			ifind_ball = np.array(tree_times.query_ball_point(np.array([P[ichoose,0]]).reshape(-1,1), r = t_win_ball[k])[0]).astype('int')
			ineighbors = edges[0][np.array(tree_edges.query_ball_point(np.array([P[ichoose,1]]).reshape(-1,1), r = 0)[0])]
			size_intersection = len(list(set(ineighbors).intersection(P[ifind_ball,1].astype('int'))))
			Ratio_neighbors[k].append(size_intersection/k_sta)

	Ratio_neighbors = np.vstack([np.quantile(Ratio_neighbors[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Ratio_neighbors))])
	Trgts.append(Ratio_neighbors)

	## [5] For each pick, number of times another pick occurs within ~15 seconds, 30 seconds, 45 seconds, etc.
	# t_win_ball = [5.0, 10.0, 15.0]
	num_picks = 1500
	Num_adjacent_picks = [[] for t in t_win_ball]
	for j in range(num_picks):
		for k in range(len(t_win_ball)):
			ichoose = np.random.choice(n_picks)
			sta_ind = P[ichoose,1]
			ifind_ball = np.array(tree_times.query_ball_point(np.array([P[ichoose,0]]).reshape(-1,1) + t_win_ball[k]/2.0 + 0.1, r = t_win_ball[k]/2.0)[0]).astype('int')
			min_sta_dist = (sta_ind == P[ifind_ball,1]).sum()
			Num_adjacent_picks[k].append(min_sta_dist)

	Num_adjacent_picks = np.vstack([np.quantile(Num_adjacent_picks[j], np.arange(0.1, 1.0, 0.2)).reshape(1,-1) for j in range(len(Num_adjacent_picks))])
	Trgts.append(Num_adjacent_picks)

	## [6] Possibly correlation of pick traces between nearby stations

	return Trgts

def evaluate_bayesian_objective(x, n_random = 30, t_sample_win = 120.0, windows = [40e3, 150e3, 300e3], t_win_ball = [10.0, 15.0, 25.0], return_vals = False): # 	windows = [40e3, 150e3, 300e3], 	


	training_params_2[0] = x[0] # spc_random
	training_params_2[2] = x[1] # spc_thresh_rand
	training_params_2[4] = x[2] # coda_rate
	training_params_2[5][1] = x[3] # coda_win

	training_params_3[1][0] = x[4] # dist_range[0]
	training_params_3[1][1] = x[5] # dist_range[1]
	training_params_3[2] = x[6] # max_rate_events
	training_params_3[3] = x[7] # max_miss_events
	training_params_3[4] = x[6]*x[8] # max_false_events (input of false rate to generate is an absolute number, not the ratio, which is x[8])
	training_params_3[5][0] = x[9] # miss_pick_fraction[0]
	training_params_3[5][1] = x[10] # miss_pick_fraction[0]

	arrivals = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, return_only_data = True)[0]

	P = np.copy(arrivals)

	Trgts = sample_picks(P, locs, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win)

	res1, res2, res3, res4, res5 = 0, 0, 0, 0, 0

	for n in range(n_random):

		ichoose = np.random.choice(len(Trgts_list))

		res1 += np.linalg.norm(Trgts[0] - Trgts_list[ichoose][0])/np.maximum(np.linalg.norm(Trgts_list[ichoose][0]), 1e-5)/n_random
		res2 += np.linalg.norm(Trgts[1] - Trgts_list[ichoose][1])/np.maximum(np.linalg.norm(Trgts_list[ichoose][1]), 1e-5)/n_random
		res3 += np.linalg.norm(Trgts[2] - Trgts_list[ichoose][2])/np.maximum(np.linalg.norm(Trgts_list[ichoose][2]), 1e-5)/n_random
		res4 += np.linalg.norm(Trgts[3] - Trgts_list[ichoose][3])/np.maximum(np.linalg.norm(Trgts_list[ichoose][3]), 1e-5)/n_random
		res5 += np.linalg.norm(Trgts[4] - Trgts_list[ichoose][4])/np.maximum(np.linalg.norm(Trgts_list[ichoose][4]), 1e-5)/n_random

	res = res1 + res2 + res3 + res4 + res5 ## Residual is average relative residual over all five objectives

	print(res)

	if return_vals == False:

		return res

	else:

		return res, Trgts, arrivals
	
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

for i in range(len(x_grids)):

	if locs.shape[0]*x_grids[i].shape[0] > 150e3:
		trv_out_l = []
		for j in range(locs.shape[0]):
			trv_out = trv(torch.Tensor(locs[j,:].reshape(1,-1)).to(device), torch.Tensor(x_grids[i]).to(device))
			trv_out_l.append(trv_out.cpu().detach().numpy())
		trv_out = torch.Tensor(np.concatenate(trv_out_l, axis = 1)).to(device)
	else:
		trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
	
	# trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
	x_grids_trv.append(trv_out.cpu().detach().numpy())
	A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(trv_out.cpu().detach().numpy(), k = k_time_edges)

	if config['train_travel_time_neural_network'] == False:
		assert(trv_out.min() > 0.0)
		assert(trv_out.max() < (ts_max_val + 3.0))

	x_grids_trv_pointers_p.append(A_edges_time_p)
	x_grids_trv_pointers_s.append(A_edges_time_s)
	x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?

	# edge_index = knn(torch.Tensor(ftrns1(x_grids[i])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[i])/1000.0).to(device), k = k_spc_edges).flip(0).contiguous()
	# edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
	# x_grids_edges.append(edge_index)

## Check if this can cause an issue (can increase max_t to a bit larger than needed value)
max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # + 10.0

## Implement training.
mz = GCN_Detection_Network_extended(ftrns1_diff, ftrns2_diff, device = device).to(device)
optimizer = optim.Adam(mz.parameters(), lr = 0.001)
loss_func = torch.nn.MSELoss()
np.random.seed() ## randomize seed

tree_stas = cKDTree(ftrns1(locs))
if use_amplitudes == True:
	mz.Mag = Mags
	mz.LocalSliceLgCollapseP.Mag = Mags
	mz.LocalSliceLgCollapseS.Mag = Mags
	mz.Arrivals.Mag = Mags

losses = np.zeros(n_epochs)
mx_trgt_1, mx_trgt_2, mx_trgt_3, mx_trgt_4 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
mx_pred_1, mx_pred_2, mx_pred_3, mx_pred_4 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)

weights = torch.Tensor([0.1, 0.4, 0.25, 0.25]).to(device)

lat_range_interior = [lat_range[0], lat_range[1]]
lon_range_interior = [lon_range[0], lon_range[1]]

n_restart = train_config['restart_training']
n_restart_step = train_config['n_restart_step']
if n_restart == False:
	n_restart_step = 0 # overwrite to 0, if restart is off

if load_training_data == True:

	files_load = glob.glob(path_to_data + '*ver_%d.hdf5'%n_ver_training_data)
	print('Number of found training files %d'%len(files_load))
	if build_training_data == False:
		assert(len(files_load) > 0)

if optimize_training_data == True:

	# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

	from skopt import gp_minimize

	# Load configuration from YAML
	with open('process_config.yaml', 'r') as file:
		process_config = yaml.safe_load(file)
		n_ver_picks = process_config['n_ver_picks']

	## If true, run Bayesian optimization to determine optimal training parameters
	st_load = glob.glob(path_to_file + 'Picks/19*') # Load years 1900's
	st_load.extend(glob.glob(path_to_file + 'Picks/20*')) # Load years 2000's
	iarg = np.argsort([int(st_load[i].split(seperator)[-1]) for i in range(len(st_load))])
	st_load = [st_load[i] for i in iarg]
	st_load_l = []
	for i in range(len(st_load)):
		st = glob.glob(st_load[i] + seperator + '*ver_%d.npz'%(n_ver_picks))
		if len(st) > 0:
			st_load_l.extend(st)
	print('Loading %d detected files for comparisons'%len(st_load_l))

	t_sample_win = 120.0 ## Bins to count picks in, and focus sampling around
	windows = [40e3, 150e3, 300e3]
	t_win_ball = [10.0, 15.0, 25.0]
	n_ver_optimize = 1
	n_max_files = 500

	if len(st_load_l) > n_max_files:
		ichoose = np.sort(np.random.choice(len(st_load_l), size = n_max_files, replace = False))
		st_load_l = [st_load_l[j] for j in ichoose]

	Trgts_list = []
	for n in range(len(st_load_l)):
		P = np.load(st_load_l[n])['P']
		Trgts_list.append(sample_picks(P, locs, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win))
		print('Finished file %d of %d'%(n, len(st_load_l)))

	evaluate_bayesian_objective_evaluate = lambda x: evaluate_bayesian_objective(x, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win)

	## Now apply Bayesian optimization to training parameters

	bounds = [(100.0, 300e3), # spc_random
	          (100.0, 300e3), # spc_thresh_rand
	          (0.001, 0.3), # coda_rate
	          (1.0, 180.0), # coda_win
	          (5000.0, 149e3), # dist_range[0]
	          (300e3, 800e3), # dist_range[1]
	          (5, 250), # max_rate_events
	          (5, 250), # max_miss_events
	          (0.2, 5.0), # max_false_events # (5, 350)
	          (0, 0.25), # miss_pick_fraction[0]
	          (0.25, 0.6)] # ] # miss_pick_fraction[0]

	optimize = gp_minimize(evaluate_bayesian_objective_evaluate,                  # the function to minimize
	                  bounds,      # the bounds on each dimension of x
	                  acq_func="EI",      # the acquisition function
	                  n_calls=150,         # the number of evaluations of f
	                  n_random_starts=100,  # the number of random initialization points
	                  noise='gaussian',       # the noise level (optional)
	                  random_state=1234, # the random seed
	                  initial_point_generator = 'lhs',
	                  model_queue_size = 150)

	res, Trgts, arrivals = evaluate_bayesian_objective(optimize.x, windows = windows, t_win_ball = t_win_ball, t_sample_win = t_sample_win, return_vals = True)

	strings = ['spc_random', 'spc_thresh_rand', 'coda_rate', 'coda_win', 'dist_range[0]', 'dist_range[1]', 'max_rate_events', 'max_miss_events', 'max_false_events', 'miss_pick_fraction[0]', 'miss_pick_fraction[0]']
	
	np.savez_compressed(path_to_file + 'Grids/%s_optimized_training_data_parameters_ver_%d.npz'%(name_of_project, n_ver_optimize), res = res, x = np.array(optimize.x), arrivals = arrivals, strings = strings)

	print('Finished optimized training data')

	error('Data set optimized; call the training script again to build training data')

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
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_amp, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = True)
		else:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_amp, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = False)
			Inpts = [Inpts[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Inpts))]
			Masks = [Masks[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Masks))]
			A_sta_sta_l = [torch.Tensor(A_sta_sta_l[j]).long().to(device) for j in range(len(A_sta_sta_l))]
			A_src_src_l = [torch.Tensor(A_src_src_l[j]).long().to(device) for j in range(len(A_src_src_l))]
			A_prod_sta_sta_l = [torch.Tensor(A_prod_sta_sta_l[j]).long().to(device) for j in range(len(A_prod_sta_sta_l))]
			A_prod_src_src_l = [torch.Tensor(A_prod_src_src_l[j]).long().to(device) for j in range(len(A_prod_src_src_l))]
			A_src_in_prod_l = [torch.Tensor(A_src_in_prod_l[j]).long().to(device) for j in range(len(A_src_in_prod_l))]

		h = h5py.File(path_to_data + 'training_data_slice_%d_ver_%d.hdf5'%(file_index, n_ver_training_data), 'w')
		h['data'] = data[0]
		h['srcs'] = data[1]
		h['srcs_active'] = data[2]

		for i in range(n_batch):

			if use_subgraph == True:
				A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[i], X_fixed[i], ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, device = device)
				A_edges_time_p, A_edges_time_s, dt_partition = compute_time_embedding_vectors(trv_pairwise, Locs[i], X_fixed[i], A_src_in_sta, max_t, device = device)
				A_sta_sta_l[i] = A_sta_sta ## These should be equal
				A_src_src_l[i] = A_src_src ## These should be equal
				A_prod_sta_sta_l[i] = A_prod_sta_sta
				A_prod_src_src_l[i] = A_prod_src_src
				A_src_in_prod_l[i] = A_src_in_prod
				A_edges_time_p_l[i] = A_edges_time_p
				A_edges_time_s_l[i] = A_edges_time_s
				A_edges_ref_l[i] = dt_partition
				Inpts[i] = np.copy(np.ascontiguousarray(Inpts[i][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))
				Masks[i] = np.copy(np.ascontiguousarray(Masks[i][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))			
				
			h['Inpts_%d'%i] = Inpts[i]
			h['Masks_%d'%i] = Masks[i]
			h['X_fixed_%d'%i] = X_fixed[i]
			h['X_query_%d'%i] = X_query[i]
			h['Locs_%d'%i] = Locs[i]
			h['Trv_out_%d'%i] = Trv_out[i]
			h['Lbls_%d'%i] = Lbls[i]
			h['Lbls_query_%d'%i] = Lbls_query[i]
			h['lp_times_%d'%i] = lp_times[i]
			h['lp_stations_%d'%i] = lp_stations[i]
			h['lp_phases_%d'%i] = lp_phases[i]
			h['lp_amp_%d'%i] = lp_amp[i]
			h['lp_meta_%d'%i] = lp_meta[i]
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
				h['A_src_in_sta_%d'%i] = np.concatenate((np.tile(np.arange(Locs[i].shape[0]), len(X_fixed[i])).reshape(1,-1), np.arange(len(X_fixed[i])).repeat(len(Locs[i]), axis = 0).reshape(1,-1)), axis = 0)

			h['A_edges_time_p_%d'%i] = A_edges_time_p_l[i]
			h['A_edges_time_s_%d'%i] = A_edges_time_s_l[i]
			h['A_edges_ref_%d'%i] = A_edges_ref_l[i]
			h['dt_partition_%d'%i] = dt_partition # _l[i]

		h.close()

	print('Finished building training data for job %d'%job_number)

	error('Data set built; call the training script again once all data has been built')

for i in range(n_restart_step, n_epochs):

	if (i == n_restart_step)*(n_restart == True):
		## Load model and optimizer.
		mz.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(n_restart_step, n_ver), map_location = device))
		optimizer.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_optimizer.h5'%(n_restart_step, n_ver), map_location = device))
		zlosses = np.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(n_restart_step, n_ver))
		losses[0:n_restart_step] = zlosses['losses'][0:n_restart_step]
		mx_trgt_1[0:n_restart_step] = zlosses['mx_trgt_1'][0:n_restart_step]; mx_trgt_2[0:n_restart_step] = zlosses['mx_trgt_2'][0:n_restart_step]
		mx_trgt_3[0:n_restart_step] = zlosses['mx_trgt_3'][0:n_restart_step]; mx_trgt_4[0:n_restart_step] = zlosses['mx_trgt_4'][0:n_restart_step]
		mx_pred_1[0:n_restart_step] = zlosses['mx_pred_1'][0:n_restart_step]; mx_pred_2[0:n_restart_step] = zlosses['mx_pred_2'][0:n_restart_step]
		mx_pred_3[0:n_restart_step] = zlosses['mx_pred_3'][0:n_restart_step]; mx_pred_4[0:n_restart_step] = zlosses['mx_pred_4'][0:n_restart_step]
		print('loaded model for restart on step %d ver %d \n'%(n_restart_step, n_ver))
		zlosses.close()
	
	optimizer.zero_grad()

	## Generate batch of synthetic inputs. Note, if this is too slow to interleave with model updates, 
	## you can  build these synthetic training data offline and then just load during training. The 
	## dataset would likely have a large memory footprint if doing so (e.g. > 1 Tb)

	if load_training_data == True:

		file_choice = np.random.choice(files_load)

		h = h5py.File(file_choice, 'r')
		
		data = [h['data'][:], h['srcs'][:], h['srcs_active'][:]]

		# h.close()

	else:

		## Build a training batch on the fly
		if use_subgraph == False:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_amp, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True)
			Inpts = [Inpts[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Inpts))]
			Masks = [Masks[j].reshape(len(X_fixed[j])*len(Locs[j]),-1) for j in range(len(Masks))]
			A_sta_sta_l = [torch.Tensor(A_sta_sta_l[j]).long().to(device) for j in range(len(A_sta_sta_l))]
			A_src_src_l = [torch.Tensor(A_src_src_l[j]).long().to(device) for j in range(len(A_src_src_l))]
			A_prod_sta_sta_l = [torch.Tensor(A_prod_sta_sta_l[j]).long().to(device) for j in range(len(A_prod_sta_sta_l))]
			A_prod_src_src_l = [torch.Tensor(A_prod_src_src_l[j]).long().to(device) for j in range(len(A_prod_src_src_l))]
			A_src_in_prod_l = [torch.Tensor(A_src_in_prod_l[j]).long().to(device) for j in range(len(A_src_in_prod_l))]
			A_src_in_sta_l = [torch.Tensor(np.concatenate((np.tile(np.arange(Locs[j].shape[0]), len(X_fixed[j])).reshape(1,-1), np.arange(len(X_fixed[j])).repeat(len(Locs[j]), axis = 0).reshape(1,-1)), axis = 0)).long().to(device) for j in range(len(Inpts))]

		if use_subgraph == True:
			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_amp, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, verbose = True, skip_graphs = True)
			A_src_in_sta_l = [[] for j in range(len(Inpts))]
			
			for n in range(len(Inpts)):
				A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(Locs[n], X_fixed[n], ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, device = device)
				A_edges_time_p, A_edges_time_s, dt_partition = compute_time_embedding_vectors(trv_pairwise, Locs[n], X_fixed[n], A_src_in_sta, max_t, device = device)
				A_sta_sta_l[n] = A_sta_sta ## These should be equal
				A_src_src_l[n] = A_src_src ## These should be equal
				A_prod_sta_sta_l[n] = A_prod_sta_sta
				A_prod_src_src_l[n] = A_prod_src_src
				A_src_in_prod_l[n] = A_src_in_prod
				A_src_in_sta_l[n] = A_src_in_sta
				A_edges_time_p_l[n] = A_edges_time_p
				A_edges_time_s_l[n] = A_edges_time_s
				A_edges_ref_l[n] = dt_partition
				Inpts[n] = np.copy(np.ascontiguousarray(Inpts[n][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))
				Masks[n] = np.copy(np.ascontiguousarray(Masks[n][A_src_in_sta[1].cpu().detach().numpy(), A_src_in_sta[0].cpu().detach().numpy()]))
	
	
	loss_val = 0
	mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4 = 0.0, 0.0, 0.0, 0.0
	mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4 = 0.0, 0.0, 0.0, 0.0

	if (np.mod(i, 1000) == 0) or (i == (n_epochs - 1)):
		torch.save(mz.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(i, n_ver))
		torch.save(optimizer.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d_optimizer.h5'%(i, n_ver))
		np.savez_compressed(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(i, n_ver), losses = losses, mx_trgt_1 = mx_trgt_1, mx_trgt_2 = mx_trgt_2, mx_trgt_3 = mx_trgt_3, mx_trgt_4 = mx_trgt_4, mx_pred_1 = mx_pred_1, mx_pred_2 = mx_pred_2, mx_pred_3 = mx_pred_3, mx_pred_4 = mx_pred_4, scale_x = scale_x, offset_x = offset_x, scale_x_extend = scale_x_extend, offset_x_extend = offset_x_extend, training_params = training_params, graph_params = graph_params, pred_params = pred_params)
		print('saved model %s %d'%(n_ver, i))
		print('saved model at step %d'%i)
	
	for inc, i0 in enumerate(range(n_batch)):

		if load_training_data == True:

			## Overwrite i0 and create length-1 lists for the training samples loaded from .hdf5 file
			Inpts = []
			Masks = []
			X_fixed = []
			X_query = []
			Locs = []
			Trv_out = []
			Lbls = []
			Lbls_query = []
			lp_times = []
			lp_stations = []
			lp_phases = []
			lp_amp = []
			lp_meta = []
			lp_srcs = []
			A_sta_sta_l = []
			A_src_src_l = []
			A_prod_sta_sta_l = []
			A_prod_src_src_l = []
			A_src_in_prod_l = []
			# A_src_in_prod_x_l = []
			# A_src_in_prod_edges_l = []
			A_edges_time_p_l = []
			A_edges_time_s_l = []
			A_edges_ref_l = []
			A_src_in_sta_l = []

			## Note: it would be more efficient (speed and memory) to pass 
			## in each sample one at time, rather than appending batch to a list
			
			Inpts.append(h['Inpts_%d'%i0][:])
			Masks.append(h['Masks_%d'%i0][:])
			X_fixed.append(h['X_fixed_%d'%i0][:])
			X_query.append(h['X_query_%d'%i0][:])
			Locs.append(h['Locs_%d'%i0][:])
			Trv_out.append(h['Trv_out_%d'%i0][:])
			Lbls.append(h['Lbls_%d'%i0][:])
			Lbls_query.append(h['Lbls_query_%d'%i0][:])
			lp_times.append(h['lp_times_%d'%i0][:])
			lp_stations.append(h['lp_stations_%d'%i0][:])
			lp_phases.append(h['lp_phases_%d'%i0][:])
			lp_amp.append(h['lp_amp_%d'%i0][:])
			lp_meta.append(h['lp_meta_%d'%i0][:])
			lp_srcs.append(h['lp_srcs_%d'%i0][:])
			A_sta_sta_l.append(torch.Tensor(h['A_sta_sta_%d'%i0][:]).long().to(device))
			A_src_src_l.append(torch.Tensor(h['A_src_src_%d'%i0][:]).long().to(device))
			A_prod_sta_sta_l.append(torch.Tensor(h['A_prod_sta_sta_%d'%i0][:]).long().to(device))
			A_prod_src_src_l.append(torch.Tensor(h['A_prod_src_src_%d'%i0][:]).long().to(device))
			A_src_in_prod_l.append(torch.Tensor(h['A_src_in_prod_%d'%i0][:]).long().to(device))

			# A_src_in_prod_l.append(h['A_src_in_prod_%d'%i0][:])
			# A_src_in_prod_x_l.append(h['A_src_in_prod_x_%d'%i0][:])
			# A_src_in_prod_edges_l.append(h['A_src_in_prod_edges_%d'%i0][:])
			# if use_subgraph == True:
			A_src_in_sta_l.append(torch.Tensor(h['A_src_in_sta_%d'%i0][:]).long().to(device))
			
			A_edges_time_p_l.append(h['A_edges_time_p_%d'%i0][:])
			A_edges_time_s_l.append(h['A_edges_time_s_%d'%i0][:])
			A_edges_ref_l.append(h['A_edges_ref_%d'%i0][:])

			i0 = 0 ## Over-write, so below indexing 

		## Adding skip... to skip samples with zero input picks
		if len(lp_times[i0]) == 0:
			print('skip a sample!') ## If this skips, and yet i0 == (n_batch - 1), is it a problem?
			continue ## Skip this!

		## Should add increased samples in x_src_query around places of coherency
		## and true labels
		x_src_query = np.random.rand(n_src_query,3)*scale_x_extend + offset_x_extend


		n_frac_focused_association_queries = 0.2 # concentrate 10% of association queries around true sources
		n_concentration_focused_association_queries = 0.03 # 3% of scale of domain
		if (len(lp_srcs[i0]) > 0)*(n_frac_focused_association_queries > 0):

			n_focused_queries = int(n_frac_focused_association_queries*n_src_query)
			ind_overwrite_focused_queries = np.sort(np.random.choice(n_src_query, size = n_focused_queries, replace = False))
			ind_source_focused = np.random.choice(len(lp_srcs[i0]), size = n_focused_queries)

			# x_query_focused = np.random.randn(n_focused_queries, 3)*scale_x_extend*n_concentration_focused_association_queries
			x_query_focused = 2.0*np.random.randn(n_focused_queries, 3)*np.mean([src_x_kernel, src_depth_kernel])
			x_query_focused = ftrns2(x_query_focused + ftrns1(lp_srcs[i0][ind_source_focused,0:3]))
			ioutside = np.where(((x_query_focused[:,2] < depth_range[0]) + (x_query_focused[:,2] > depth_range[1])) > 0)[0]
			x_query_focused[ioutside,2] = np.random.rand(len(ioutside))*(depth_range[1] - depth_range[0]) + depth_range[0]
			
			x_query_focused = np.maximum(np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1), x_query_focused)
			x_query_focused = np.minimum(np.array([lat_range_extend[1], lon_range_extend[1], depth_range[1]]).reshape(1,-1), x_query_focused)
			x_src_query[ind_overwrite_focused_queries] = x_query_focused

		if len(lp_srcs[i0]) > 0:
			x_src_query[0:len(lp_srcs[i0]),0:3] = lp_srcs[i0][:,0:3]
		
		x_src_query_cart = ftrns1(x_src_query)
		
		trv_out_src = trv(torch.Tensor(Locs[i0]).to(device), torch.Tensor(x_src_query).to(device)).detach()

		
		if use_subgraph == True:
			# A_src_in_prod_l[i0] = Data(torch.Tensor(A_src_in_prod_x_l[i0]).to(device), edge_index = torch.Tensor(A_src_in_prod_edges_l[i0]).long().to(device))
			trv_out = trv_pairwise(torch.Tensor(Locs[i0][A_src_in_sta_l[i0][0].cpu().detach().numpy()]).to(device), torch.Tensor(X_fixed[i0][A_src_in_sta_l[i0][1].cpu().detach().numpy()]).to(device))
			spatial_vals = torch.Tensor((X_fixed[i0][A_src_in_prod_l[i0][1].cpu().detach().numpy()] - Locs[i0][A_src_in_sta_l[i0][0][A_src_in_prod_l[i0][0]].cpu().detach().numpy()])/scale_x_extend).to(device)

		else:
			trv_out = trv(torch.Tensor(Locs[i0]).to(device), torch.Tensor(X_fixed[i0]).to(device)).detach().reshape(-1,2) ## Note: could also just take this from x_grids_trv
			spatial_vals = torch.Tensor(((np.repeat(np.expand_dims(X_fixed[i0], axis = 1), Locs[i0].shape[0], axis = 1) - np.repeat(np.expand_dims(Locs[i0], axis = 0), X_fixed[i0].shape[0], axis = 0)).reshape(-1,3))/scale_x_extend).to(device)
	
		
		tq_sample = torch.rand(n_src_query).to(device)*t_win - t_win/2.0
		tq = torch.arange(-t_win/2.0, t_win/2.0 + dt_win, dt_win).reshape(-1,1).float().to(device)

		if len(lp_srcs[i0]) > 0:
			ifind_src = np.where(np.abs(lp_srcs[i0][:,3]) <= t_win/2.0)[0]
			tq_sample[ifind_src] = torch.Tensor(lp_srcs[i0][ifind_src,3]).to(device)

		if use_phase_types == False:
			if use_amplitudes == False:
				Inpts[i0][:,2:4] = 0.0 ## Phase type informed features zeroed out
				Masks[i0][:,2:4] = 0.0
				# Inpts[i0][:,] = 0.0 ## Phase type informed features zeroed out
				# Masks[i0][:,2:4] = 0.0
			else:
				Inpts[i0][:,2:4] = 0.0 ## Phase type informed features zeroed out
				Masks[i0][:,2:4] = 0.0
				Inpts[i0][:,6::] = 0.0 ## Phase type informed features zeroed out
				Masks[i0][:,6::] = 0.0				

		# Pre-process tensors for Inpts and Masks
		input_tensor_1 = torch.Tensor(Inpts[i0]).to(device) # .reshape(-1, 4)
		input_tensor_2 = torch.Tensor(Masks[i0]).to(device) # .reshape(-1, 4)

		# Process tensors for A_prod and A_src arrays
		A_prod_sta_tensor = A_prod_sta_sta_l[i0]
		A_prod_src_tensor = A_prod_src_src_l[i0]

		# Process edge index data
		edge_index_1 = A_src_in_prod_l[i0]
		flipped_edge = np.ascontiguousarray(np.flip(A_src_in_prod_l[i0].cpu().detach().numpy(), axis = 0))
		edge_index_2 = torch.Tensor(flipped_edge).long().to(device)

		data_1 = Data(x=spatial_vals, edge_index=edge_index_1)
		data_2 = Data(x=spatial_vals, edge_index=edge_index_2)

		use_updated_pick_max_associations = True # Changing to True
		if (len(lp_times[i0]) > max_number_pick_association_labels_per_sample)*(use_updated_pick_max_associations == True):

			## Cnt number of picks per station
			## Optimally choose n stations to compute association labels for
			## so that sum cnt_i < max_mumber of picks used. 
			## Permute station indices to not bias ordering.
			## Keep all picks for this set of stations.
			## (note: this does not effect output values, only the ones we compute losses on)

			tree_sta_slice = cKDTree(lp_stations[i0].reshape(-1,1))
			lp_cnt_sta = tree_sta_slice.query_ball_point(np.arange(Locs[i0].shape[0]).reshape(-1,1), r = 0)
			cnt_lp_sta = np.array([len(lp_cnt_sta[j]) for j in range(len(lp_cnt_sta))])

			# Maximize the number of associations. Permute
			sta_grab = optimize_station_selection(cnt_lp_sta, max_number_pick_association_labels_per_sample)
			isample_picks = np.hstack([lp_cnt_sta[j] for j in sta_grab])

			lp_times[i0] = lp_times[i0][isample_picks]
			lp_stations[i0] = lp_stations[i0][isample_picks]
			lp_phases[i0] = lp_phases[i0][isample_picks]
			lp_amp[i0] = lp_amp[i0][isample_picks]
			lp_meta[i0] = lp_meta[i0][isample_picks]

			assert(len(lp_times[i0]) <= max_number_pick_association_labels_per_sample)
		
		elif len(lp_times[i0]) > max_number_pick_association_labels_per_sample:

			## Randomly choose max number of picks to compute association labels for.
			isample_picks = np.sort(np.random.choice(len(lp_times[i0]), size = max_number_pick_association_labels_per_sample, replace = False))
			lp_times[i0] = lp_times[i0][isample_picks]
			lp_stations[i0] = lp_stations[i0][isample_picks]
			lp_phases[i0] = lp_phases[i0][isample_picks]
			lp_amp[i0] = lp_amp[i0][isample_picks]
			lp_meta[i0] = lp_meta[i0][isample_picks]
		
		# Continue processing the rest of the inputs
		input_tensors = [
			input_tensor_1, input_tensor_2, A_prod_sta_tensor, A_prod_src_tensor,
			data_1, data_2,
			A_src_in_sta_l[i0],
			A_src_src_l[i0],
			torch.Tensor(A_edges_time_p_l[i0]).long().to(device),
			torch.Tensor(A_edges_time_s_l[i0]).long().to(device),
			torch.Tensor(A_edges_ref_l[i0]).to(device),
			trv_out,
			torch.Tensor(lp_times[i0]).to(device),
			torch.Tensor(lp_stations[i0]).long().to(device),
			torch.Tensor(lp_phases[i0]).reshape(-1, 1).float().to(device),
			torch.Tensor(lp_amp[i0]).reshape(-1, 1).float().to(device),
			torch.Tensor(ftrns1(Locs[i0])).to(device),
			torch.Tensor(ftrns1(X_fixed[i0])).to(device),
			torch.Tensor(ftrns1(X_query[i0])).to(device),
			torch.Tensor(x_src_query_cart).to(device),
			tq, tq_sample, trv_out_src
		]


		if use_amplitudes == True:
			imatch_vec = torch.Tensor(tree_stas.query(ftrns1(Locs[i0]))[1]).long().to(device)
			mz.Mag.ivec = imatch_vec
			mz.LocalSliceLgCollapseP.ivec = imatch_vec
			mz.LocalSliceLgCollapseS.ivec = imatch_vec
			mz.Arrivals.ivec = imatch_vec

		# Call the model with pre-processed tensors
		out = mz(*input_tensors)
		
		pick_lbls = pick_labels_extract_interior_region(x_src_query_cart, tq_sample.cpu().detach().numpy(), lp_meta[i0][:,-2::], lp_srcs[i0], lat_range_interior, lon_range_interior, ftrns1, sig_t = src_t_arv_kernel, sig_x = src_x_arv_kernel)
		loss = (weights[0]*loss_func(out[0][:,:,0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1][:,:,0], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1]))/n_batch

		n_visualize_step = 1000
		n_visualize_fraction = 0.2
		if (make_visualize_predictions == True)*(np.mod(i, n_visualize_step) == 0)*(i0 < n_visualize_fraction*n_batch):
			save_plots_path = path_to_file + seperator + 'Plots' + seperator

			if Lbls_query[i0][:,5].max() > 0.2: # Plot all true sources
				visualize_predictions(out, Lbls_query[i0], pick_lbls, X_query[i0], lp_times[i0], lp_stations[i0], Locs[i0], data, i0, save_plots_path, n_step = i, n_ver = n_ver)
			elif np.random.rand() > 0.8: # Plot a fraction of false sources
				visualize_predictions(out, Lbls_query[i0], pick_lbls, X_query[i0], lp_times[i0], lp_stations[i0], Locs[i0], data, i0, save_plots_path, n_step = i, n_ver = n_ver)

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

	if load_training_data == True:
		h.close() ## Close training file

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

	print('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4))

	# Log losses
	if use_wandb_logging == True:
		wandb.log({"loss": loss_val})

	with open(write_training_file + 'output_%d.txt'%n_ver, 'a') as text_file:
		text_file.write('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4))











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







