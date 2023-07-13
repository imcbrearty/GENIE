
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
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
import itertools
import pathlib

## Graph params
k_sta_edges = 8
k_spc_edges = 15
k_time_edges = 10
graph_params = [k_sta_edges, k_spc_edges, k_time_edges]

## Training params
n_batch = 75
n_epochs = 20000 + 1 # add 1, so it saves on last iteration (since it saves every 100 steps)
n_spc_query = 4500 # Number of src queries per sample
n_src_query = 300 # Number of src-arrival queries per sample
training_params = [n_spc_query, n_src_query]

## Prediction params
kernel_sig_t = 5.0 # Kernel to embed arrival time - theoretical time misfit (s)
src_t_kernel = 6.5 # Kernel or origin time label (s)
src_t_arv_kernel = 6.5 # Kernel for arrival association time label (s)
src_x_kernel = 15e3 # Kernel for source label, horizontal distance (m)
src_x_arv_kernel = 15e3 # Kernel for arrival-source association label, horizontal distance (m)
src_depth_kernel = 15e3 # Kernel of Cartesian projection, vertical distance (m)
t_win = 10.0 ## This is the time window over which predictions are made. Shouldn't be changed for now.
## Note that right now, this shouldn't change, as the GNN definitions also assume this is 10 s.
dist_range = [15e3, 500e3] ## The spatial window over which to sample max distance of 
## source-station moveouts in m, per event. E.g., 15 - 500 km. Should set slightly lower if using small region.

# File versions
template_ver = 1 # spatial grid version
vel_model_ver = 1 # velocity model version
n_ver = 1 # GNN save version

## Will update to be adaptive soon. The step size of temporal prediction is fixed at 1 s right now.

## Should add src_x_arv_kernel and src_t_arv_kerne to pred_params, but need to check usage of this variable in this and later scripts
pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]

device = torch.device('cuda') ## or use cpu

def lla2ecef(p, a = 6378137.0, e = 8.18191908426215e-2): # 0.0818191908426215, previous 8.1819190842622e-2
	p = p.copy().astype('float')
	p[:,0:2] = p[:,0:2]*np.array([np.pi/180.0, np.pi/180.0]).reshape(1,-1)
	N = a/np.sqrt(1 - (e**2)*np.sin(p[:,0])**2)
    # results:
	x = (N + p[:,2])*np.cos(p[:,0])*np.cos(p[:,1])
	y = (N + p[:,2])*np.cos(p[:,0])*np.sin(p[:,1])
	z = ((1-e**2)*N + p[:,2])*np.sin(p[:,0])
	return np.concatenate((x[:,None],y[:,None],z[:,None]), axis = 1)

def ecef2lla(x, a = 6378137.0, e = 8.18191908426215e-2):
	x = x.copy().astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	b = np.sqrt((a**2)*(1 - e**2))
	ep = np.sqrt((a**2 - b**2)/(b**2))
	p = np.sqrt(x[:,0]**2 + x[:,1]**2)
	th = np.arctan2(a*x[:,2], b*p)
	lon = np.arctan2(x[:,1], x[:,0])
	lat = np.arctan2((x[:,2] + (ep**2)*b*(np.sin(th)**3)), (p - (e**2)*a*(np.cos(th)**3)))
	N = a/np.sqrt(1 - (e**2)*(np.sin(lat)**2))
	alt = p/np.cos(lat) - N
	# lon = np.mod(lon, 2.0*np.pi) # don't use!
	k = (np.abs(x[:,0]) < 1) & (np.abs(x[:,1]) < 1)
	alt[k] = np.abs(x[k,2]) - b
	return np.concatenate((180.0*lat[:,None]/np.pi, 180.0*lon[:,None]/np.pi, alt[:,None]), axis = 1)

def assemble_time_pointers_for_stations(trv_out, dt = 1.0, k = 10, win = 10.0, tbuffer = 10.0):

	n_temp, n_sta = trv_out.shape[0:2]
	dt_partition = np.arange(-win, win + trv_out.max() + dt + tbuffer, dt)

	edges_p = []
	edges_s = []
	for i in range(n_sta):
		tree_p = cKDTree(trv_out[:,i,0][:,np.newaxis])
		tree_s = cKDTree(trv_out[:,i,1][:,np.newaxis])
		q_p, ip_p = tree_p.query(dt_partition[:,np.newaxis], k = k)
		q_s, ip_s = tree_s.query(dt_partition[:,np.newaxis], k = k)
		# ip must project to Lg indices.
		edges_p.append((ip_p*n_sta + i).reshape(-1)) # Lg indices are each source x n_sta + sta_ind. The reshape places each subsequence of k, per time step, per station.
		edges_s.append((ip_s*n_sta + i).reshape(-1)) # Lg indices are each source x n_sta + sta_ind. The reshape places each subsequence of k, per time step, per station.
		# Overall, each station, each time step, each k sets of edges.
	edges_p = np.hstack(edges_p)
	edges_s = np.hstack(edges_s)

	return edges_p, edges_s, dt_partition

def assemble_time_pointers_for_stations_multiple_grids(trv_out, max_t, dt = 1.0, k = 10, win = 10.0):

	n_temp, n_sta = trv_out.shape[0:2]
	dt_partition = np.arange(-win, win + max_t + dt, dt)

	edges_p = []
	edges_s = []
	for i in range(n_sta):
		tree_p = cKDTree(trv_out[:,i,0][:,np.newaxis])
		tree_s = cKDTree(trv_out[:,i,1][:,np.newaxis])
		q_p, ip_p = tree_p.query(dt_partition[:,np.newaxis], k = k)
		q_s, ip_s = tree_s.query(dt_partition[:,np.newaxis], k = k)
		# ip must project to Lg indices.
		edges_p.append((ip_p*n_sta + i).reshape(-1))
		edges_s.append((ip_s*n_sta + i).reshape(-1))
		# Overall, each station, each time step, each k sets of edges.
	edges_p = np.hstack(edges_p)
	edges_s = np.hstack(edges_s)

	return edges_p, edges_s, dt_partition

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
	locs_tensor = torch.Tensor(locs).to(device)

	t_slice = np.arange(-t_win/2.0, t_win/2.0 + 1.0, 1.0)

	tsteps = np.arange(0, T + dt, dt)
	tvec = np.arange(-tscale*4, tscale*4 + dt, dt)
	tvec_kernel = np.exp(-(tvec**2)/(2.0*(tscale**2)))

	p_rate_events = fftconvolve(np.random.randn(2*locs.shape[0] + 3, len(tsteps)), tvec_kernel.reshape(1,-1).repeat(2*locs.shape[0] + 3,0), 'same', axes = 1)
	c_cor = (p_rate_events@p_rate_events.T) ## Not slow!
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

	m1 = [0.5761163, -0.21916288]
	m2 = 1.15

	amp_thresh = 1.0
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


	use_stable_association_labels = False ## There might be a possible bug when using this, when creating association labels
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
	non_active_sources = np.delete(np.arange(n_events), active_sources, axis = 0)
	src_positions_active = src_positions[active_sources]
	src_times_active = src_times[active_sources]
	src_magnitude_active = src_magnitude[active_sources] ## Not currently used

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

	scale_vec = np.array([1,2*t_win]).reshape(1,-1)

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
	lp_src_times = tree_src_times.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)

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
	src_positions_active_per_slice_l = []
	src_times_active_per_slice_l = []

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

	time_vec_slice = np.arange(k_time_edges)

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
	lp_srcs_active = []

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
			# ifind = np.where(meta[:,2] > -1.0)[0] ## Need to find picks with a source index inside the active_sources_per_slice
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
		# print('permute indices 1')
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

def interp_1D_velocity_model_to_3D_travel_times(X, locs, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2):

	i1 = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]])
	x10, x20, x30 = Xmin
	Xv = X - np.array([x10,x20,x30])[None,:]
	depth_grid = np.copy(locs[:,2]).reshape(1,-1)
	mask = np.array([1.0,1.0,0.0]).reshape(1,-1)

	print('Check way X0 is used')

	def evaluate_func(y, x):

		y, x = y.cpu().detach().numpy(), x.cpu().detach().numpy()

		ind_depth = np.tile(np.argmin(np.abs(y[:,2].reshape(-1,1) - depth_grid), axis = 1), x.shape[0])
		rel_pos = (np.expand_dims(x, axis = 1) - np.expand_dims(y*mask, axis = 0)).reshape(-1,3)
		rel_pos[:,0:2] = np.abs(rel_pos[:,0:2]) ## Postive relative pos, relative to X0.
		x_relative = X0 + rel_pos

		xv = x_relative - Xmin[None,:]
		nx = np.shape(rel_pos)[0] # nx is the number of query points in x
		nz_vals = np.array([np.rint(np.floor(xv[:,0]/Dx[0])),np.rint(np.floor(xv[:,1]/Dx[1])),np.rint(np.floor(xv[:,2]/Dx[2]))]).T
		nz_vals1 = np.minimum(nz_vals, N - 2)

		nz = (np.reshape(np.dot((np.repeat(nz_vals1, 8, axis = 0) + repmat(i1,nx,1)),Mn.T),(nx,8)).T).astype('int')

		val_p = Tp[nz,ind_depth]
		val_s = Ts[nz,ind_depth]

		x0 = np.reshape(xv,(1,nx,3)) - Xv[nz,:]
		x0 = (1 - abs(x0[:,:,0])/Dx[0])*(1 - abs(x0[:,:,1])/Dx[1])*(1 - abs(x0[:,:,2])/Dx[2])

		val_p = np.sum(val_p*x0, axis = 0).reshape(-1, y.shape[0])
		val_s = np.sum(val_s*x0, axis = 0).reshape(-1, y.shape[0])

		return torch.Tensor(np.concatenate((val_p[:,:,None], val_s[:,:,None]), axis = 2)).to(device)

	return lambda y, x: evaluate_func(y, x)

class DataAggregation(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_mask = 4):
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

	def forward(self, tr, mask, A_in_sta, A_in_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr)), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr)), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_edges = 3):
		super(BipartiteGraphOperator, self).__init__('add')
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, A_src_in_edges, mask, n_sta, n_temp):

		N = n_sta*n_temp
		M = n_temp

		return self.activate2(self.fc2(self.propagate(A_src_in_edges.edge_index, size = (N, M), x = mask.max(1, keepdims = True)[0]*self.activate1(self.fc1(torch.cat((inpt, A_src_in_edges.x), dim = -1))))))

class SpatialAggregation(MessagePassing):
	def __init__(self, in_channels, out_channels, scale_rel = 30e3, n_dim = 3, n_global = 5, n_hidden = 30):
		super(SpatialAggregation, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.fc1 = nn.Linear(in_channels + n_dim + n_global, n_hidden)
		self.fc2 = nn.Linear(n_hidden + in_channels, out_channels)
		self.fglobal = nn.Linear(in_channels, n_global)
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.scale_rel = scale_rel

	def forward(self, tr, A_src, pos):

		return self.activate2(self.fc2(torch.cat((tr, self.propagate(A_src, x = tr, pos = pos/self.scale_rel)), dim = -1)))

	def message(self, x_j, pos_i, pos_j):
		
		return self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j, self.activate3(self.fglobal(x_j)).mean(0, keepdims = True).repeat(x_j.shape[0], 1)), dim = -1))) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

class SpatialDirect(nn.Module):
	def __init__(self, inpt_dim, out_channels):
		super(SpatialDirect, self).__init__() #  "Max" aggregation.

		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.activate = nn.PReLU()

	def forward(self, inpts):

		return self.activate(self.f_direct(inpts))

class SpatialAttention(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, scale_rel = 30e3, n_hidden = 30, n_heads = 5):
		super(SpatialAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.f_context = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		# self.activate3 = nn.PReLU()

	def forward(self, inpts, x_query, x_context, k = 10): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		edge_index = knn(x_context/1000.0, x_query/1000.0, k = k).flip(0)
		edge_attr = (x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel # /scale_x

		return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

	def message(self, x_j, index, edge_attr):

		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

		alpha = softmax(alpha, index)

		return alpha.unsqueeze(-1)*value_embed

class TemporalAttention(MessagePassing): ## Hopefully replace this.
	def __init__(self, inpt_dim, out_channels, n_latent, scale_t = 10.0, n_hidden = 30, n_heads = 5):
		super(TemporalAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.

		self.temporal_query_1 = nn.Linear(1, n_hidden)
		self.temporal_query_2 = nn.Linear(n_hidden, n_heads*n_latent)
		self.f_context_1 = nn.Linear(inpt_dim, n_hidden) # add second layer transformation.
		self.f_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.f_values_1 = nn.Linear(inpt_dim, n_hidden) # add second layer transformation.
		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, out_channels) # can remove this layer possibly.

		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_t = scale_t

		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		self.activate5 = nn.PReLU()

	def forward(self, inpts, t_query):

		context = self.f_context_2(self.activate1(self.f_context_1(inpts))).view(-1, self.n_heads, self.n_latent) # add more non-linear transform here?
		values = self.f_values_2(self.activate2(self.f_values_1(inpts))).view(-1, self.n_heads, self.n_latent) # add more non-linear transform here?
		query = self.temporal_query_2(self.activate3(self.temporal_query_1(t_query/self.scale_t))).view(-1, self.n_heads, self.n_latent) # must repeat t output for all spatial coordinates.

		return self.proj_2(self.activate5(self.proj_1(self.activate4((((context.unsqueeze(1)*query.unsqueeze(0)).sum(-1, keepdims = True)/self.scale)*values.unsqueeze(1)).mean(2))))) # linear.

class BipartiteGraphReadOutOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_edges = 3):
		super(BipartiteGraphReadOutOperator, self).__init__('add')
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, A_Lg_in_srcs, mask, n_sta, n_temp):

		N = n_temp
		M = n_sta*n_temp

		return self.activate2(self.fc2(self.propagate(A_Lg_in_srcs.edge_index, size = (N, M), x = inpt, edge_attr = A_Lg_in_srcs.x, mask = mask))), mask[A_Lg_in_srcs.edge_index[0]] # note: outputting multiple outputs

	def message(self, x_j, mask_j, edge_attr):

		return mask_j*self.activate1(self.fc1(torch.cat((x_j, edge_attr), dim = -1)))	

class DataAggregationAssociationPhase(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_latent = 30, n_dim_mask = 5):
		super(DataAggregationAssociationPhase, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_latent + n_dim_mask, n_hidden)

		self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		self.l1_t2_1 = nn.Linear(n_hidden, n_hidden)
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

	def forward(self, tr, latent, mask1, mask2, A_in_sta, A_in_src):

		mask = torch.cat((mask1, mask2), dim = - 1)
		tr = torch.cat((tr, latent, mask), dim = -1)
		tr = self.activate(self.init_trns(tr)) # should tlatent appear here too? Not on first go..

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(self.l1_t1_1(tr))), mask), dim = 1)) # Supposed to use this layer. Now, using correct layer.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(self.l1_t2_1(tr))), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

class LocalSliceLgCollapse(MessagePassing):
	def __init__(self, ndim_in, ndim_out, n_edge = 2, n_hidden = 30, eps = 15.0):
		super(LocalSliceLgCollapse, self).__init__('mean') # NOTE: mean here? Or add is more expressive for individual arrivals?
		self.fc1 = nn.Linear(ndim_in + n_edge, n_hidden) # non-multi-edge type. Since just collapse on fixed stations, with fixed slice of Xq. (how to find nodes?)
		self.fc2 = nn.Linear(n_hidden, ndim_out)
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.eps = eps

	def forward(self, A_edges, dt_partition, tpick, ipick, phase_label, inpt, tlatent, n_temp, n_sta): # reference k nearest spatial points

		## Assert is problem?
		k_infer = int(len(A_edges)/(n_sta*len(dt_partition)))
		assert(k_infer == 10)
		n_arvs, l_dt = len(tpick), len(dt_partition)
		N = n_sta*n_temp # Lg graph
		M = n_arvs# M is target
		dt = dt_partition[1] - dt_partition[0]

		t_index = torch.floor((tpick - dt_partition[0])/torch.Tensor([dt]).to(device)).long() # index into A_edges, which is each station, each dt_point, each k.
		t_index = ((ipick*l_dt*k_infer + t_index*k_infer).view(-1,1) + torch.arange(k_infer).view(1,-1).to(device)).reshape(-1).long() # check this

		src_index = torch.arange(n_arvs).view(-1,1).repeat(1,k_infer).view(1,-1).to(device)
		sliced_edges = torch.cat((A_edges[t_index].view(1,-1), src_index), dim = 0).long()
		t_rel = tpick[sliced_edges[1]] - tlatent[sliced_edges[0],0] # Breaks here?

		ikeep = torch.where(abs(t_rel) < self.eps)[0]
		sliced_edges = sliced_edges[:,ikeep] # only use times within range. (need to specify target node cardinality)
		out = self.activate2(self.fc2(self.propagate(sliced_edges, x = inpt, pos = (tlatent, tpick.view(-1,1)), phase = phase_label, size = (N, M))))

		return out

	def message(self, x_j, pos_i, pos_j, phase_i):

		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.eps, phase_i), dim = -1))) # note scaling of relative time

class StationSourceAttentionMergedPhases(MessagePassing):
	def __init__(self, ndim_src_in, ndim_arv_in, ndim_out, n_latent, ndim_extra = 1, n_heads = 5, n_hidden = 30, eps = 15.0):
		super(StationSourceAttentionMergedPhases, self).__init__(node_dim = 0, aggr = 'add') # check node dim.

		self.f_arrival_query_1 = nn.Linear(2*ndim_arv_in + 6, n_hidden) # add edge data (observed arrival - theoretical arrival)
		self.f_arrival_query_2 = nn.Linear(n_hidden, n_heads*n_latent) # Could use nn.Sequential to combine these.
		self.f_src_context_1 = nn.Linear(ndim_src_in + ndim_extra + 2, n_hidden) # only use single tranform layer for source embdding (which already has sufficient information)
		self.f_src_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # only use single tranform layer for source embdding (which already has sufficient information)

		self.f_values_1 = nn.Linear(2*ndim_arv_in + ndim_extra + 7, n_hidden) # add second layer transformation.
		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, ndim_out) # can remove this layer possibly.

		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.eps = eps
		self.t_kernel_sq = torch.Tensor([10.0]).to(device)**2
		self.ndim_feat = ndim_arv_in + ndim_extra

		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		# self.activate5 = nn.PReLU()

	def forward(self, src, stime, src_embed, trv_src, arrival_p, arrival_s, tpick, ipick, phase_label): # reference k nearest spatial points

		# src isn't used. Only trv_src is needed.
		n_src, n_sta, n_arv = src.shape[0], trv_src.shape[1], len(tpick) # + 1 ## Note: adding 1 to size of arrivals!
		# n_arv = len(tpick)
		ip_unique = torch.unique(ipick).float().cpu().detach().numpy() # unique stations
		tree_indices = cKDTree(ipick.float().cpu().detach().numpy().reshape(-1,1))
		unique_sta_lists = tree_indices.query_ball_point(ip_unique.reshape(-1,1), r = 0)

		arrival_p = torch.cat((arrival_p, torch.zeros(1,arrival_p.shape[1]).to(device)), dim = 0) # add null arrival, that all arrivals link too. This acts as a "stabalizer" in the inner-product space, and allows softmax to not blow up for arrivals with only self loops. May not be necessary.
		arrival_s = torch.cat((arrival_s, torch.zeros(1,arrival_s.shape[1]).to(device)), dim = 0) # add null arrival, that all arrivals link too. This acts as a "stabalizer" in the inner-product space, and allows softmax to not blow up for arrivals with only self loops. May not be necessary.
		arrival = torch.cat((arrival_p, arrival_s), dim = 1) # Concatenate across feature axis

		edges = torch.Tensor(np.copy(np.flip(np.hstack([np.ascontiguousarray(np.array(list(zip(itertools.product(unique_sta_lists[j], np.concatenate((unique_sta_lists[j], np.array([n_arv])), axis = 0)))))[:,0,:].T) for j in range(len(unique_sta_lists))]), axis = 0))).long().to(device) # note: preferably could remove loop here.
		n_edge = edges.shape[1]

		## Now must duplicate edges, for each unique source. (different accumulation points)
		edges = (edges.repeat(1, n_src) + torch.cat((torch.zeros(1, n_src*n_edge).to(device), (torch.arange(n_src)*n_arv).repeat_interleave(n_edge).view(1,-1).to(device)), dim = 0)).long()
		src_index = torch.arange(n_src).repeat_interleave(n_edge).long()

		N = n_arv + 1 # still correct?
		M = n_arv*n_src

		out = self.proj_2(self.activate4(self.proj_1(self.propagate(edges, x = arrival, sembed = src_embed, stime = stime, tsrc_p = torch.cat((trv_src[:,:,0], -self.eps*torch.ones(n_src,1).to(device)), dim = 1).detach(), tsrc_s = torch.cat((trv_src[:,:,1], -self.eps*torch.ones(n_src,1).to(device)), dim = 1).detach(), sindex = src_index, stindex = torch.cat((ipick, torch.Tensor([n_sta]).long().to(device)), dim = 0), atime = torch.cat((tpick, torch.Tensor([-self.eps]).to(device)), dim = 0), phase = torch.cat((phase_label, torch.Tensor([-1.0]).reshape(1,1).to(device)), dim = 0), size = (N, M)).mean(1)))) # M is output. Taking mean over heads

		return out.view(n_src, n_arv, -1) ## Make sure this is correct reshape (not transposed)

	def message(self, x_j, edge_index, index, tsrc_p, tsrc_s, sembed, sindex, stindex, stime, atime, phase_j): # Can use phase_j, or directly call edge_index, like done for atime, stindex, etc.

		assert(abs(edge_index[1] - index).max().item() == 0)

		ifind = torch.where(edge_index[0] == edge_index[0].max())[0]

		rel_t_p = (atime[edge_index[0]] - (tsrc_p[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_p = torch.cat((torch.exp(-0.5*(rel_t_p**2)/self.t_kernel_sq), torch.sign(rel_t_p), phase_j), dim = 1) # phase[edge_index[0]]

		rel_t_s = (atime[edge_index[0]] - (tsrc_s[sindex, stindex[edge_index[0]]] + stime[sindex])).reshape(-1,1).detach() # correct? (edges[0] point to input data, we access the augemted data time)
		rel_t_s = torch.cat((torch.exp(-0.5*(rel_t_s**2)/self.t_kernel_sq), torch.sign(rel_t_s), phase_j), dim = 1) # phase[edge_index[0]]

		# Denote self-links by a feature.
		self_link = (edge_index[0] == torch.remainder(edge_index[1], edge_index[0].max().item())).reshape(-1,1).detach().float() # Each accumulation index (an entry from src cross arrivals). The number of arrivals is edge_index.max() exactly (since tensor is composed of number arrivals + 1)
		null_link = (edge_index[0] == edge_index[0].max().item()).reshape(-1,1).detach().float()
		contexts = self.f_src_context_2(self.activate1(self.f_src_context_1(torch.cat((sembed[sindex], stime[sindex].reshape(-1,1).detach(), self_link, null_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)
		queries = self.f_arrival_query_2(self.activate2(self.f_arrival_query_1(torch.cat((x_j, rel_t_p, rel_t_s), dim = 1)))).view(-1, self.n_heads, self.n_latent)
		values = self.f_values_2(self.activate3(self.f_values_1(torch.cat((x_j, rel_t_p, rel_t_s, self_link, null_link), dim = 1)))).view(-1, self.n_heads, self.n_latent)

		assert(self_link.sum() == (len(atime) - 1)*tsrc_p.shape[0])

		## Do computation
		scores = (queries*contexts).sum(-1)/self.scale
		alpha = softmax(scores, index)

		return alpha.unsqueeze(-1)*values # self.activate1(self.fc1(torch.cat((x_j, pos_i - pos_j), dim = -1)))

class GCN_Detection_Network_extended(nn.Module):
	def __init__(self, ftrns1, ftrns2, device = 'cuda'):
		super(GCN_Detection_Network_extended, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
		self.DataAggregation = DataAggregation(4, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 3).to(device) # 30, 15
		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialDirect = SpatialDirect(30, 30).to(device) # 15, 30
		self.SpatialAttention = SpatialAttention(30, 30, 3, 15).to(device)
		self.TemporalAttention = TemporalAttention(30, 1, 15).to(device)

		self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15).to(device)
		self.DataAggregationAssociationPhase = DataAggregationAssociationPhase(15, 15).to(device) # need to add concatenation
		self.LocalSliceLgCollapseP = LocalSliceLgCollapse(30, 15).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
		self.LocalSliceLgCollapseS = LocalSliceLgCollapse(30, 15).to(device) # need to add concatenation. Should it really shrink dimension? Probably not..
		self.Arrivals = StationSourceAttentionMergedPhases(30, 15, 2, 15, n_heads = 3).to(device)
		# self.ArrivalS = StationSourceAttention(30, 15, 1, 15, n_heads = 3).to(device)

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use, x_temp_cuda_cart, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

		n_line_nodes = Slice.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use.shape[0]

		x_latent = self.DataAggregation(Slice.view(n_line_nodes, -1), Mask, A_in_sta, A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, locs_use.shape[0], x_temp_cuda_cart.shape[0])
		x = self.SpatialAggregation1(x, A_src, x_temp_cuda_cart)
		x = self.SpatialAggregation2(x, A_src, x_temp_cuda_cart)
		x_spatial = self.SpatialAggregation3(x, A_src, x_temp_cuda_cart) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		y_latent = self.SpatialDirect(x) # contains data on spatial solution.
		y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
		x = self.SpatialAttention(x_spatial, x_query_cart, x_temp_cuda_cart) # second slowest module (could use this embedding to seed source source attention vector).
		x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
		x = self.TemporalAttention(x, t_query) # on random queries

		## Note below: why detach x_latent?
		mask_out = 1.0*(y[:,:,0].detach().max(1, keepdims = True)[0] > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"
		arv_p = self.LocalSliceLgCollapseP(A_edges_p, dt_partition, tpick, ipick, phase_label, s, tlatent[:,0].reshape(-1,1), n_temp, n_sta) ## arv_p and arv_s will be same size
		arv_s = self.LocalSliceLgCollapseS(A_edges_s, dt_partition, tpick, ipick, phase_label, s, tlatent[:,1].reshape(-1,1), n_temp, n_sta)
		arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
		
		arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)

		return y, x, arv_p, arv_s

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())


## Load Files
if '\\' in path_to_file: ## Windows

	# Load region
	name_of_project = path_to_file.split('\\')[-1] ## Windows
	z = np.load(path_to_file + '\\%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + '\\Grids\\%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	x_grids = z['x_grids']
	z.close()

	# Load stations
	z = np.load(path_to_file + '\\%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	## Load travel times
	z = np.load(path_to_file + '\\1D_Velocity_Models_Regional\\%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))

	## Create path to write files
	write_training_file = path_to_file + '\\GNN_TrainedModels\\' + name_of_project + '_'
	
else: ## Linux or Unix

	# Load region
	name_of_project = path_to_file.split('/')[-1] ## Windows
	z = np.load(path_to_file + '/%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + '/Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	x_grids = z['x_grids']
	z.close()

	# Load stations
	z = np.load(path_to_file + '/%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	## Load travel times
	z = np.load(path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))

	## Create path to write files
	write_training_file = path_to_file + '/GNN_TrainedModels/' + name_of_project + '_'
	
lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1
rbest_cuda = torch.Tensor(rbest).to(device)
mn_cuda = torch.Tensor(mn).to(device)
ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x) - mn_cuda).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda)

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

ts_max_val = Ts.max()

for i in range(len(x_grids)):

	trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
	x_grids_trv.append(trv_out.cpu().detach().numpy())
	A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(trv_out.cpu().detach().numpy(), k = k_time_edges)

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

weights = torch.Tensor([0.4, 0.2, 0.2, 0.2]).to(device)

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
	while cwork == 0:
		try: ## Does this actually ever through an exception? Probably not.

			[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, training_params, graph_params, pred_params, ftrns1, ftrns2, fixed_subnetworks = Ind_subnetworks, use_preferential_sampling = True, n_batch = n_batch, verbose = True, dist_range = dist_range)

			cwork = 1
		except:
			inc_c += 1
			print('Failed data gen! %d'%inc_c)

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

		out = mz(torch.Tensor(Inpts[i0]).to(device).reshape(-1,4), torch.Tensor(Masks[i0]).to(device).reshape(-1,4), torch.Tensor(A_prod_sta_sta_l[i0]).long().to(device), torch.Tensor(A_prod_src_src_l[i0]).long().to(device), Data(x = spatial_vals, edge_index = torch.Tensor(A_src_in_prod_l[i0]).long().to(device)), Data(x = spatial_vals, edge_index = torch.Tensor(np.ascontiguousarray(np.flip(A_src_in_prod_l[i0], axis = 0))).long().to(device)), torch.Tensor(A_src_src_l[i0]).long().to(device), torch.Tensor(A_edges_time_p_l[i0]).long().to(device), torch.Tensor(A_edges_time_s_l[i0]).long().to(device), torch.Tensor(A_edges_ref_l[i0]).to(device), trv_out, torch.Tensor(lp_times[i0]).to(device), torch.Tensor(lp_stations[i0]).long().to(device), torch.Tensor(lp_phases[i0].reshape(-1,1)).float().to(device), torch.Tensor(ftrns1(Locs[i0])).to(device), torch.Tensor(ftrns1(X_fixed[i0])).to(device), torch.Tensor(ftrns1(X_query[i0])).to(device), torch.Tensor(x_src_query_cart).to(device), tq, tq_sample, trv_out_src)

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

	print('%d %0.8f'%(i, loss_val))

	with open(write_training_file + 'output_%d.txt'%n_ver, 'a') as text_file:
		text_file.write('%d loss %0.9f, trgts: %0.5f, %0.5f, %0.5f, %0.5f, preds: %0.5f, %0.5f, %0.5f, %0.5f \n'%(i, loss_val, mx_trgt_val_1, mx_trgt_val_2, mx_trgt_val_3, mx_trgt_val_4, mx_pred_val_1, mx_pred_val_2, mx_pred_val_3, mx_pred_val_4))

