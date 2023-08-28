
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
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from numpy.matlib import repmat
import pathlib

from utils import *
from module import *
# from generate_synthetic_data import generate_synthetic_data 
## For now not using the seperate files definition of generate_synthetic_data

use_wandb_logging = False
if use_wandb_logging == True:

	import wandb
	# Initialize wandb run 
	wandb.init(project="GENIE")


# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

name_of_project = config['name_of_project']

path_to_file = str(pathlib.Path().absolute())
path_to_file += '\\' if '\\' in path_to_file else '/'

## Graph params
k_sta_edges = config['k_sta_edges']
k_spc_edges = config['k_spc_edges']
k_time_edges = config['k_time_edges']

graph_params = [k_sta_edges, k_spc_edges, k_time_edges]

## Training params
n_batch = config['n_batch']
n_epochs = config['n_epochs'] # add 1, so it saves on last iteration (since it saves every 100 steps)
n_spc_query = config['n_spc_query'] # Number of src queries per sample
n_src_query = config['n_src_query'] # Number of src-arrival queries per sample
training_params = [n_spc_query, n_src_query]

## Prediction params
kernel_sig_t = config['kernel_sig_t'] # Kernel to embed arrival time - theoretical time misfit (s)
src_t_kernel = config['src_t_kernel'] # Kernel or origin time label (s)
src_t_arv_kernel = config['src_t_arv_kernel'] # Kernel for arrival association time label (s)
src_x_kernel = config['src_x_kernel'] # Kernel for source label, horizontal distance (m)
src_x_arv_kernel = config['src_x_arv_kernel'] # Kernel for arrival-source association label, horizontal distance (m)
src_depth_kernel = config['src_depth_kernel'] # Kernel of Cartesian projection, vertical distance (m)
t_win = config['t_win'] ## This is the time window over which predictions are made. Shouldn't be changed for now.
## Note that right now, this shouldn't change, as the GNN definitions also assume this is 10 s.
dist_range = config['dist_range'] ## The spatial window over which to sample max distance of 
## source-station moveouts in m, per event. E.g., 15 - 500 km. Should set slightly lower if using small region.

# File versions
template_ver = 1 # spatial grid version
vel_model_ver = 1 # velocity model version
n_ver = 1 # GNN save version

## Will update to be adaptive soon. The step size of temporal prediction is fixed at 1 s right now.

## Should add src_x_arv_kernel and src_t_arv_kerne to pred_params, but need to check usage of this variable in this and later scripts
pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]

device = torch.device('cuda') ## or use cpu


## Extra train parameters

spc_random = 30e3
sig_t = 0.03 # 3 percent of travel time error on pick times
spc_thresh_rand = 20e3
min_sta_arrival = 4
coda_rate = 0.035 # 5 percent arrival have code. Probably more than this? Increased from 0.035.
coda_win = np.array([0, 25.0]) # coda occurs within 0 to 25 s after arrival (should be less?) # Increased to 25, from 20.0
max_num_spikes = 80
spike_time_spread = 0.15
s_extra = 0.0 ## If this is non-zero, it can increase (or decrease) the total rate of missed s waves compared to p waves
use_stable_association_labels = True
thresh_noise_max = 1.5
training_params_2 = [spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max]

## Training params list 3
n_batch = 75
dist_range = [15e3, 500e3] # Should be chosen proportional to physical domain size
max_rate_events = 5000/8
max_miss_events = 3000/8
max_false_events = 2000/8
T = 3600.0*3.0
dt = 30
tscale = 3600.0
n_sta_range = [0.35, 1.0] # n_sta_range[0]*locs.shape[0] must be >= the number of station edges chosen (k_sta_edges)
use_sources = False
use_full_network = False
fixed_subnetworks = None
use_preferential_sampling = False
use_shallow_sources = False
training_params_3 = [n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources]

def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, training_params, graph_params, pred_params, ftrns1, ftrns2, n_batch = 75, dist_range = [15e3, 500e3], max_rate_events = 6000/8, max_miss_events = 2500/8, max_false_events = 2500/8, T = 3600.0*3.0, dt = 30, tscale = 3600.0, n_sta_range = [0.35, 1.0], use_sources = False, use_full_network = False, fixed_subnetworks = None, use_preferential_sampling = False, use_shallow_sources = False, plot_on = False, verbose = False):

	if verbose == True:
		st = time.time()

	k_sta_edges, k_spc_edges, k_time_edges = graph_params
	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	n_spc_query, n_src_query = training_params
	spc_random = 30e3
	sig_t = 0.03 # 3 percent of travel time error on pick times
	spc_thresh_rand = 20e3
	min_sta_arrival = 4
	coda_rate = 0.035 # 5 percent arrival have code. Probably more than this? Increased from 0.035.
	coda_win = np.array([0, 25.0]) # coda occurs within 0 to 25 s after arrival (should be less?) # Increased to 25, from 20.0
	max_num_spikes = 80

	assert(np.floor(n_sta_range[0]*locs.shape[0]) > k_sta_edges)

	scale_x = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)
	n_sta = locs.shape[0]

	t_slice = np.arange(-t_win/2.0, t_win/2.0 + 1.0, 1.0)

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
	n_src = len(src_times)
	src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
	src_magnitude = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)

	if use_shallow_sources == True:
		sample_random_depths = gamma(1.75, 0.0).rvs(n_src)
		sample_random_grab = np.where(sample_random_depths > 5)[0] # Clip the long tails, and place in uniform, [0,5].
		sample_random_depths[sample_random_grab] = 5.0*np.random.rand(len(sample_random_grab))
		sample_random_depths = sample_random_depths/sample_random_depths.max() # Scale to range
		sample_random_depths = -sample_random_depths*(scale_x[0,2] - 2e3) + (offset_x[0,2] + scale_x[0,2] - 2e3) # Project along axis, going negative direction. Removing 2e3 on edges.
		src_positions[:,2] = sample_random_depths

	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

	use_uniform_distance_threshold = False
	## This previously sampled a skewed distribution by default, not it samples a uniform
	## distribution of the maximum source-reciever distances allowed for each event.
	if use_uniform_distance_threshold == True:
		dist_thresh = np.random.rand(n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
	else:
		## Use beta distribution to generate more samples with smaller moveouts
		# dist_thresh = -1.0*np.log(np.sqrt(np.random.rand(n_src))) ## Sort of strange dist threshold set!
		# dist_thresh = (dist_thresh*dist_range[1]/10.0 + dist_range[0]).reshape(-1,1)
		dist_thresh = beta(2,5).rvs(size = n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]

	# create different distance dependent thresholds.
	dist_thresh_p = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
	dist_thresh_s = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None]

	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

	arrivals_theoretical = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()
	arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
	arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
	src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
	arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

	s_extra = 0.0 ## If this is non-zero, it can increase (or decrease) the total rate of missed s waves compared to p waves
	t_inc = np.floor(arrivals[:,3]/dt).astype('int')
	p_miss_rate = 0.5*station_miss_rate[arrivals[:,1].astype('int'), t_inc] + 0.5*global_miss_rate[t_inc]
	idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < dt*p_miss_rate/T)[0]

	arrivals = np.delete(arrivals, idel, axis = 0)
	n_events = len(src_times)

	icoda = np.where(np.random.rand(arrivals.shape[0]) < coda_rate)[0]
	if len(icoda) > 0:
		false_coda_arrivals = np.random.rand(len(icoda))*(coda_win[1] - coda_win[0]) + coda_win[0] + arrivals[icoda,0] + arrivals[icoda,3]
		false_coda_arrivals = np.concatenate((false_coda_arrivals.reshape(-1,1), arrivals[icoda,1].reshape(-1,1), -1.0*np.ones((len(icoda),1)), np.zeros((len(icoda),1)), -1.0*np.ones((len(icoda),1))), axis = 1)
		arrivals = np.concatenate((arrivals, false_coda_arrivals), axis = 0)

	## Base false events
	station_false_rate_eval = 0.5*station_false_rate + 0.5*global_false_rate
	vals = np.random.poisson(dt*station_false_rate_eval/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average

	# How to speed up this part?
	i1, i2 = np.where(vals > 0)
	v_val, t_val = vals[i1,i2], tsteps[i2]
	false_times = np.repeat(t_val, v_val) + np.random.rand(vals.sum())*dt
	false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])
	n_false = len(false_times)
	false_arrivals = np.concatenate((false_times.reshape(-1,1), false_indices.reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
	arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)

	n_spikes = np.random.randint(0, high = int(max_num_spikes*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
	if n_spikes > 0:
		n_spikes_extent = np.random.randint(1, high = n_sta, size = n_spikes) ## This many stations per spike
		time_spikes = np.random.rand(n_spikes)*T
		sta_ind_spikes = np.hstack([np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)])
		sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*0.15 for j in range(n_spikes)])
		false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
		arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes


	use_stable_association_labels = True
	## Check which true picks have so much noise, they should be marked as `false picks' for the association labels
	if use_stable_association_labels == True:
		thresh_noise_max = 1.5
		iz = np.where(arrivals[:,4] >= 0)[0]
		noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
		iexcess_noise = np.where(np.abs(noise_values) > thresh_noise_max*sig_t*arrivals[iz,0])[0]
		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise
		if len(iexcess_noise) > 0: ## Set these arrivals to "false arrivals", since noise is so high
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
	active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
	src_times_active = src_times[active_sources]

	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))

	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
	phase_observed = np.copy(arrivals[:,-1]).astype('int')

	if len(iwhere_false) > 0: # For false picks, assign a random phase type
		phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))

	perturb_phases = True # For true picks, randomly flip a fraction of phases
	if (len(phase_observed) > 0)*(perturb_phases == True):
		n_switch = int(np.random.rand()*(0.2*len(iwhere_real))) # switch up to 20% phases
		iflip = np.random.choice(iwhere_real, size = n_switch, replace = False)
		phase_observed[iflip] = np.mod(phase_observed[iflip] + 1, 2)
	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.
	

	if use_sources == False:
		time_samples = np.sort(np.random.rand(n_batch)*T) ## Uniform

	elif use_sources == True:
		time_samples = src_times_active[np.sort(np.random.choice(len(src_times_active), size = n_batch))]

	l_src_times_active = len(src_times_active)
	if (use_preferential_sampling == True)*(len(src_times_active) > 1):
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
	phase_observed_select = phase_observed[lp_concat]

	Trv_subset_p = []
	Trv_subset_s = []
	Station_indices = []
	Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	if (fixed_subnetworks is not None):
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

		active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

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

	rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)

	## With phase type information
	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

	rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)


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

	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grids[grid_select].shape[0]
		n_sta_slice = len(sta_select)

		inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))

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
		times = meta[:,0]
		indices = perm_vec[meta[:,1].astype('int')]
		ineed = np.where(indices > -1)[0]
		times = times[ineed] ## Overwrite, now. Double check if this is ok.
		indices = indices[ineed]
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
		lp_srcs.append(src_subset)

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

		x_query = np.random.rand(n_spc_query, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.

		if len(lp_srcs[-1]) > 0:
			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]

		if len(active_sources_per_slice) == 0:
			lbls_grid = np.zeros((x_grids[grid_select].shape[0], len(t_slice)))
			lbls_query = np.zeros((n_spc_query, len(t_slice)))
		else:
			active_sources_per_slice = active_sources_per_slice.astype('int')

			## Must move the .sum(2) in the below def to use this expanded version of labels
			# lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			# Spatial component
			# spatial_term1 = np.expand_dims(ftrns1(x_grids[grid_select]), axis=1)
			# spatial_term2 = np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis=0)
			# spatial_diff = spatial_term1 - spatial_term2
			# spatial_exp_term = np.exp(-0.5 * (spatial_diff**2) / (src_spatial_kernel**2))

			# # Temporal component
			# temporal_term1 = (time_samples[i] + t_slice).reshape(1,-1,1)
			# temporal_term2 = src_times[active_sources_per_slice].reshape(1,1,-1)
			# temporal_diff = temporal_term1 - temporal_term2
			# temporal_exp_term = np.exp(-0.5 * (temporal_diff**2) / (src_t_kernel**2))

			# print('There is an error in these updated label definitions, since the first two targets should be of a similar value')
			
			# Combine components
			# lbls_grid = (np.expand_dims(spatial_exp_term.sum(2), axis=1) * temporal_exp_term).max(2)
			# lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

			lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

		
		X_query.append(x_query)
		Lbls.append(lbls_grid)
		Lbls_query.append(lbls_query)

	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
	data = [arrivals, srcs, active_sources]		

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)


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


# Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)

max_number_pick_association_labels_per_sample = config['max_number_pick_association_labels_per_sample']

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

# use_spherical = False
if config['use_spherical'] == True:

	earth_radius = 6371e3
	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T # just subtract mean
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device) # just subtract mean

else:

	earth_radius = 6378137.0
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
	
	
	trv = interp_1D_velocity_model_to_3D_travel_times(X, locs_ref, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2) # .to(device)

elif config['train_travel_time_neural_network'] == True:

	n_ver_trv_time_model_load = 1
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load)

load_subnetworks = False
if load_subnetworks == True:

	h_subnetworks = np.load(path_to_file + '%s_subnetworks.npz'%name_of_project, allow_pickle = True)
	Ind_subnetworks = h_subnetworks['Sta_inds']
	h_subnetworks.close()

else:

	Ind_subnetworks = None

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
if device.type == 'cuda':
	check_len = knn(torch.rand(10,3).to(device), torch.rand(10,3).to(device), k = 5).numel()
	if check_len < 100: # If it's less than 2 * 10 * 5, there's an issue
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
x_grids_edges = []

if config['train_travel_time_neural_network'] == False:
	ts_max_val = Ts.max()

for i in range(len(x_grids)):

	trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
	x_grids_trv.append(trv_out.cpu().detach().numpy())
	A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(trv_out.cpu().detach().numpy(), k = k_time_edges)

	if config['train_travel_time_neural_network'] == False:
		assert(trv_out.min() > 0.0)
		assert(trv_out.max() < (ts_max_val + 3.0))

	x_grids_trv_pointers_p.append(A_edges_time_p)
	x_grids_trv_pointers_s.append(A_edges_time_s)
	x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?

	edge_index = knn(torch.Tensor(ftrns1(x_grids[i])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[i])/1000.0).to(device), k = k_spc_edges).flip(0).contiguous()
	edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
	x_grids_edges.append(edge_index)

max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # + 10.0

## Implement training.
mz = GCN_Detection_Network_extended(ftrns1_diff, ftrns2_diff).to(device)
optimizer = optim.Adam(mz.parameters(), lr = 0.001)
loss_func = torch.nn.MSELoss()


losses = np.zeros(n_epochs)
mx_trgt_1, mx_trgt_2, mx_trgt_3, mx_trgt_4 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
mx_pred_1, mx_pred_2, mx_pred_3, mx_pred_4 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)

weights = torch.Tensor([0.2, 0.4, 0.2, 0.2]).to(device)

lat_range_interior = [lat_range[0], lat_range[1]]
lon_range_interior = [lon_range[0], lon_range[1]]

n_restart = False
n_restart_step = 0
if n_restart == False:
	n_restart_step = 0 # overwrite to 0, if restart is off

for i in range(n_restart_step, n_epochs):

	if (i == n_restart_step)*(n_restart == True):
		## Load model and optimizer.
		mz.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(n_restart_step, n_ver)))
		optimizer.load_state_dict(torch.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_optimizer.h5'%(n_restart_step, n_ver)))
		zlosses = np.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(n_restart_step, n_ver))
		losses[0:n_restart_step] = zlosses['losses'][0:n_restart_step]
		mx_trgt_1[0:n_restart_step] = zlosses['mx_trgt_1'][0:n_restart_step]; mx_trgt_2[0:n_restart_step] = zlosses['mx_trgt_2'][0:n_restart_step]
		mx_trgt_3[0:n_restart_step] = zlosses['mx_trgt_3'][0:n_restart_step]; mx_trgt_4[0:n_restart_step] = zlosses['mx_trgt_4'][0:n_restart_step]
		mx_pred_1[0:n_restart_step] = zlosses['mx_pred_1'][0:n_restart_step]; mx_pred_2[0:n_restart_step] = zlosses['mx_pred_2'][0:n_restart_step]
		mx_pred_3[0:n_restart_step] = zlosses['mx_pred_3'][0:n_restart_step]; mx_pred_4[0:n_restart_step] = zlosses['mx_pred_4'][0:n_restart_step]
		print('loaded model for restart on step %d ver %d \n'%(n_restart_step, n_ver))

	optimizer.zero_grad()

	cwork = 0
	inc_c = 0
	while (cwork == 0)*(inc_c < 10):
		# try: ## Does this actually ever through an exception? Probably not.

		[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, graph_params, pred_params, ftrns1, ftrns2, fixed_subnetworks = Ind_subnetworks, use_preferential_sampling = True, n_batch = n_batch, verbose = True, dist_range = dist_range)

		cwork = 1
		# except:
		# 	inc_c += 1
		# 	print('Failed data gen! %d'%inc_c)

		## To look at the synthetic data, do:
		## plt.scatter(data[0][:,0], data[0][:,1])
		## (plots time of pick against station index; will need to use an interactive plot to see details)

	loss_val = 0
	mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4 = 0.0, 0.0, 0.0, 0.0
	mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4 = 0.0, 0.0, 0.0, 0.0

	if np.mod(i, 100) == 0:
		torch.save(mz.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(i, n_ver))
		torch.save(optimizer.state_dict(), write_training_file + 'trained_gnn_model_step_%d_ver_%d_optimizer.h5'%(i, n_ver))
		np.savez_compressed(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(i, n_ver), losses = losses, mx_trgt_1 = mx_trgt_1, mx_trgt_2 = mx_trgt_2, mx_trgt_3 = mx_trgt_3, mx_trgt_4 = mx_trgt_4, mx_pred_1 = mx_pred_1, mx_pred_2 = mx_pred_2, mx_pred_3 = mx_pred_3, mx_pred_4 = mx_pred_4, scale_x = scale_x, offset_x = offset_x, scale_x_extend = scale_x_extend, offset_x_extend = offset_x_extend, training_params = training_params, graph_params = graph_params, pred_params = pred_params)
		print('saved model %s %d'%(n_ver, i))
		print('saved model at step %d'%i)		
	
	for i0 in range(n_batch):

		## Adding skip... to skip samples with zero input picks
		if len(lp_times[i0]) == 0:
			print('skip a sample!') ## If this skips, and yet i0 == (n_batch - 1), is it a problem?
			continue ## Skip this!

		## Should add increased samples in x_src_query around places of coherency
		## and true labels
		x_src_query = np.random.rand(n_src_query,3)*scale_x_extend + offset_x_extend

		if len(lp_srcs[i0]) > 0:
			x_src_query[0:len(lp_srcs[i0]),0:3] = lp_srcs[i0][:,0:3]

		x_src_query_cart = ftrns1(x_src_query)

		trv_out = trv(torch.Tensor(Locs[i0]).to(device), torch.Tensor(X_fixed[i0]).to(device)).detach().reshape(-1,2) ## Note: could also just take this from x_grids_trv
		trv_out_src = trv(torch.Tensor(Locs[i0]).to(device), torch.Tensor(x_src_query).to(device)).detach()
		tq_sample = torch.rand(n_src_query).to(device)*t_win - t_win/2.0
		tq = torch.arange(-t_win/2.0, t_win/2.0 + 1.0).reshape(-1,1).float().to(device)

		if len(lp_srcs[i0]) > 0:
			tq_sample[0:len(lp_srcs[i0])] = torch.Tensor(lp_srcs[i0][:,3]).to(device)

		spatial_vals = torch.Tensor(((np.repeat(np.expand_dims(X_fixed[i0], axis = 1), Locs[i0].shape[0], axis = 1) - np.repeat(np.expand_dims(Locs[i0], axis = 0), X_fixed[i0].shape[0], axis = 0)).reshape(-1,3))/scale_x_extend).to(device)

		# Pre-process tensors for Inpts and Masks
		input_tensor_1 = torch.Tensor(Inpts[i0]).to(device).reshape(-1, 4)
		input_tensor_2 = torch.Tensor(Masks[i0]).to(device).reshape(-1, 4)

		# Process tensors for A_prod and A_src arrays
		A_prod_sta_tensor = torch.Tensor(A_prod_sta_sta_l[i0]).long().to(device)
		A_prod_src_tensor = torch.Tensor(A_prod_src_src_l[i0]).long().to(device)

		# Process edge index data
		edge_index_1 = torch.Tensor(A_src_in_prod_l[i0]).long().to(device)
		flipped_edge = np.ascontiguousarray(np.flip(A_src_in_prod_l[i0], axis=0))
		edge_index_2 = torch.Tensor(flipped_edge).long().to(device)

		data_1 = Data(x=spatial_vals, edge_index=edge_index_1)
		data_2 = Data(x=spatial_vals, edge_index=edge_index_2)

		if len(lp_times[i0]) > max_number_pick_association_labels_per_sample:
			isample_picks = np.sort(np.random.choice(len(lp_times[i0]), size = max_number_pick_association_labels_per_sample, replace = False))
			lp_times[i0] = lp_times[i0][isample_picks]
			lp_stations[i0] = lp_stations[i0][isample_picks]
			lp_phases[i0] = lp_phases[i0][isample_picks]
			lp_meta[i0] = lp_meta[i0][isample_picks]
		
		# Continue processing the rest of the inputs
		input_tensors = [
			input_tensor_1, input_tensor_2, A_prod_sta_tensor, A_prod_src_tensor,
			data_1, data_2,
			torch.Tensor(A_src_src_l[i0]).long().to(device),
			torch.Tensor(A_edges_time_p_l[i0]).long().to(device),
			torch.Tensor(A_edges_time_s_l[i0]).long().to(device),
			torch.Tensor(A_edges_ref_l[i0]).to(device),
			trv_out,
			torch.Tensor(lp_times[i0]).to(device),
			torch.Tensor(lp_stations[i0]).long().to(device),
			torch.Tensor(lp_phases[i0]).reshape(-1, 1).float().to(device),
			torch.Tensor(ftrns1(Locs[i0])).to(device),
			torch.Tensor(ftrns1(X_fixed[i0])).to(device),
			torch.Tensor(ftrns1(X_query[i0])).to(device),
			torch.Tensor(x_src_query_cart).to(device),
			tq, tq_sample, trv_out_src
		]

		# Call the model with pre-processed tensors
		out = mz(*input_tensors)
		
		pick_lbls = pick_labels_extract_interior_region(x_src_query_cart, tq_sample.cpu().detach().numpy(), lp_meta[i0][:,-2::], lp_srcs[i0], lat_range_interior, lon_range_interior, ftrns1, sig_t = src_t_arv_kernel, sig_x = src_x_arv_kernel)
		loss = (weights[0]*loss_func(out[0][:,:,0], torch.Tensor(Lbls[i0]).to(device)) + weights[1]*loss_func(out[1][:,:,0], torch.Tensor(Lbls_query[i0]).to(device)) + weights[2]*loss_func(out[2][:,:,0], pick_lbls[:,:,0]) + weights[3]*loss_func(out[3][:,:,0], pick_lbls[:,:,1]))/n_batch

		if i0 != (n_batch - 1):
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

