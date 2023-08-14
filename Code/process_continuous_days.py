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
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.cluster import SpectralClustering
from numpy.matlib import repmat
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

## This code cannot be run with cuda quite yet 
## (need to add .cuda()'s at appropriatte places)
## In general, it often makes sense to run this
## script in parallel for many days simulataneously (using argv[1]; 
## e.g., call "python process_continuous_days.py n" for many different n
## integers and each instance will run day t0_init + n.
## sbatch or a bash script can call this file for a parallel set of cpu threads
## (each for a different n, or, day).

## Need to update how extract_inputs_from_data_fixed_grids_with_phase_type uses a variable t_win parammeter, 
## and also adding inputs of training_params, graph_params, pred_params

t0_init = UTCDateTime(2019, 1, 1) ## Choose this day, add (day_select), or (day_select + offset_select*offset_increment)

argvs = sys.argv
if len(argvs) < 2:
	argvs.append(0) # choose day 0, if no other day chosen.

if len(argvs) < 3:
	argvs.append(0)

day_select = int(argvs[1])
offset_select = int(argvs[2])

print('name of program is %s'%argvs[0])
print('day is %s'%argvs[1])

template_ver = 1
vel_model_ver = 1

n_ver_load = 1
n_step_load = 20000
n_save_ver = 1
n_ver_picks = 1

offset_increment = 500
n_rand_query = 112000
n_query_grid = 5000

thresh = 0.125 # Threshold to declare detection
thresh_assoc = 0.125 # Threshold to declare src-arrival association
spr = 1 # Sampling rate to save temporal predictions
tc_win = 5.0 # Temporal window (s) to link events in Local Marching
sp_win = 15e3 # Distance (m) to link events in Local Marching
break_win = 15.0 # Temporal window to find disjoint groups of sources, 
## so can run Local Marching without memory issues.
spr_picks = 100 # Assumed sampling rate of picks 
## (can be 1 if absolute times are used for pick time values)

d_win = 0.25 ## Lat and lon window to re-locate initial source detetections with refined sampling over
d_win_depth = 10e3 ## Depth window to re-locate initial source detetections with refined sampling over
dx_depth = 50.0 ## Depth resolution to locate events with travel time based re-location

use_expanded_competitive_assignment = True
cost_value = 3.0 # If use expanded competitve assignment, then this is the fixed cost applied per source
## when optimizing joint source-arrival assignments between nearby sources. The value is in terms of the 
## `sum' over the predicted source-arrival assignment for each pick. Ideally could make this number more
## adpative, potentially with number of stations or number of possible observing picks for each event. 

device = torch.device('cpu') ## Right now, this isn't updated to work with cuda, since
## the necessary variables do not have .to(device) at the right places


def download_catalog(lat_range, lon_range, min_magnitude, startime, endtime, t0 = UTCDateTime(2000, 1, 1), client = 'NCEDC', include_arrivals = False):

	client = Client(client)
	cat_l = client.get_events(starttime = startime, endtime = endtime, minlatitude = lat_range[0], maxlatitude = lat_range[1], minlongitude = lon_range[0], maxlongitude = lon_range[1], minmagnitude = min_magnitude, includearrivals = include_arrivals, orderby = 'time-asc')

	# t0 = UTCDateTime(2021,4,1) ## zero time, for relative processing.
	time = np.array([cat_l[i].origins[0].time - t0 for i in np.arange(len(cat_l))])
	latitude = np.array([cat_l[i].origins[0].latitude for i in np.arange(len(cat_l))])
	longitude = np.array([cat_l[i].origins[0].longitude for i in np.arange(len(cat_l))])
	depth = np.array([cat_l[i].origins[0].depth for i in np.arange(len(cat_l))])
	mag = np.array([cat_l[i].magnitudes[0].mag for i in np.arange(len(cat_l))])
	event_type = np.array([cat_l[i].event_type for i in np.arange(len(cat_l))])

	cat = np.hstack([latitude.reshape(-1,1), longitude.reshape(-1,1), -1.0*depth.reshape(-1,1), time.reshape(-1,1), mag.reshape(-1,1)])

	return cat, cat_l, event_type

class LocalMarching(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, device = 'cpu'):
		super(LocalMarching, self).__init__(aggr = 'max') # node dim

	## Changed dt to 5 s
	def forward(self, srcs, tc_win = 5, sp_win = 35e3, n_steps_max = 100, tol = 1e-4, use_directed = True, device = 'cpu'):

		srcs_tensor = torch.Tensor(srcs).to(device)
		tree_t = cKDTree(srcs[:,3].reshape(-1,1))
		tree_x = cKDTree(ftrns1(srcs[:,0:3]))
		lp_t = tree_t.query_ball_point(srcs[:,3].reshape(-1,1), r = tc_win)
		lp_x = tree_x.query_ball_point(ftrns1(srcs[:,0:3]), r = sp_win)
		cedges = [np.array(list(set(lp_t[i]).intersection(lp_x[i]))) for i in range(len(lp_t))]
		cedges1 = np.hstack([i*np.ones(len(cedges[i])) for i in range(len(cedges))])
		edges = torch.Tensor(np.concatenate((np.hstack(cedges).reshape(1,-1), cedges1.reshape(1,-1)), axis = 0)).long().to(device)

		Data_obj = Data(edge_index = to_undirected(edges)) # undirected
		nx_g = to_networkx(Data_obj).to_undirected()
		lp = list(nx.connected_components(nx_g))
		clen = [len(list(lp[i])) for i in range(len(lp))]
		## Remove disconnected points with only 1 maxima.

		if use_directed == True:

			max_val = torch.where(srcs_tensor[edges[1],-1] <= srcs_tensor[edges[0],-1])[0]
			edges = edges[:,max_val]

		srcs_keep = []
		for i in range(len(lp)):
			nodes = np.sort(np.array(list(lp[i])))

			if (len(nodes) == 1):
				srcs_keep.append(nodes.reshape(-1))

			else:

				edges_need = subgraph(torch.Tensor(nodes).long().to(device), edges, relabel_nodes = True)[0]

				vals = torch.Tensor(srcs[nodes,4]).view(-1,1).to(device)
				vals_initial = torch.Tensor(srcs[nodes,4]).view(-1,1).to(device)
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
			srcs_keep = srcs[np.hstack(srcs_keep)]

		return srcs_keep	


def load_picks(path_to_file, date, locs, stas, lat_range, lon_range, thresh_cut = None, use_quantile = None, permute_indices = False, min_amplitude = None, n_ver = 1, spr_picks = 100):

	if '\\' in path_to_file:
		z = np.load(path_to_file + '\\Picks\\%d\\%d_%d_%d_ver_%d.npz'%(date[0], date[0], date[1], date[2], n_ver))
	elif '/' in path_to_file:
		z = np.load(path_to_file + '/Picks/%d/%d_%d_%d_ver_%d.npz'%(date[0], date[0], date[1], date[2], n_ver))

	yr, mn, dy = date[0], date[1], date[2]
	t0 = UTCDateTime(yr, mn, dy)
	P, sta_names_use, sta_ind_use = z['P'], z['sta_names_use'], z['sta_ind_use']
	
	if use_quantile is not None:
		iz = np.where(P[:,3] > np.quantile(P[:,3], use_quantile))[0]
		P = P[iz]

	if thresh_cut is not None:
		iz = np.where(P[:,3] > thresh_cut)[0]
		P = P[iz]

	if min_amplitude is not None:
		iz = np.where(P[:,2] < min_amplitude)[0]
		P = np.delete(P, iz, axis = 0) # remove picks with amplitude less than min possible amplitude

	P_l = []
	locs_use = []
	sta_use = []
	ind_use = []
	sc = 0
	for i in range(len(sta_names_use)):
		iz = np.where(sta_names_use[i] == stas)[0]
		if len(iz) == 0:
			# print('no match')
			continue
		iz1 = np.where(P[:,1] == sta_ind_use[i])[0]
		if len(iz1) == 0:
			# print('no picks')
			continue		
		p_select = P[iz1]
		if permute_indices == True:
			p_select[:,1] = sc ## New indices
		else:
			p_select[:,1] = iz ## Absolute indices
		P_l.append(p_select)
		locs_use.append(locs[iz])
		sta_use.append(stas[iz])
		ind_use.append(iz)
		sc += 1

	P_l = np.vstack(P_l)
	P_l[:,0] = P_l[:,0]/spr_picks ## Convert pick indices to time (note: if spr_picks = 1, then picks are already in absolute time)
	locs_use = np.vstack(locs_use)
	sta_use = np.hstack(sta_use)
	ind_use = np.hstack(ind_use)

	## Make sure ind_use is unique set. Then re-select others.
	ind_use = np.sort(np.unique(ind_use))
	locs_use = locs[ind_use]
	sta_use = stas[ind_use]

	if permute_indices == True:
		argsort = np.argsort(ind_use)
		P_l_1 = []
		sc = 0
		for i in range(len(argsort)):
			iz1  = np.where(P_l[:,1] == argsort[i])[0]
			p_slice = P_l[iz1]
			p_slice[:,1] = sc
			P_l_1.append(p_slice)
			sc += 1
		P_l = np.vstack(P_l)

	# julday = int((UTCDateTime(yr, mn, dy) - UTCDateTime(yr, 1, 1))/(3600*24.0) + 1)

	if download_catalog == True:
		cat, _, event_type = download_catalog(lat_range, lon_range, min_magnitude, t0, t0 + 3600*24.0, t0 = t0)
	else:
		cat, event_type = [], []

	z.close()

	return P_l, ind_use # Note: this permutation of locs_use.

def extract_inputs_from_data_fixed_grids_with_phase_type(trv, locs, ind_use, arrivals, phase_labels, time_samples, x_grid, x_grid_trv, lat_range, lon_range, depth_range, max_t, ftrns1, ftrns2, n_queries = 3000, n_batch = 75, max_rate_events = 5000, max_miss_events = 3500, max_false_events = 2000, T = 3600.0*24.0, dt = 30, tscale = 3600.0, n_sta_range = [0.25, 1.0], plot_on = False, verbose = False):

	if verbose == True:
		st = time.time()

	## Can simplify, and only focus on one network (could even re-create the correct x_grids_trv_pointers_p for specific network?)
	## This would speed up inference application

	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
	n_sta = locs.shape[0]
	n_spc = x_grid.shape[0]
	locs_tensor = torch.Tensor(locs)
	# grid_select = np.random.randint(0, high = len(x_grids))
	# time_samples = np.copy(srcs[:,3])

	## Removing this concatenate step, since now passing all information with arrivals
	# arrivals = np.concatenate((arrivals, 0*np.ones((arrivals.shape[0],3))), axis = 1) # why append 0's?

	# n_sta = locs.shape[0]
	# n_spc = x_grid.shape[0]
	# n_sta_slice = len(ind_use)

	# perm_vec = -1*np.ones(locs.shape[0])
	# perm_vec[ind_use] = np.arange(len(ind_use))

	t_win = 10.0
	scale_vec = np.array([1,2*t_win]).reshape(1,-1)
	n_batch = len(time_samples)

	## Max_t is dependent on x_grids_trv. Else, can pass max_t as an input.
	# max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))

	src_t_kernel = 8.0 # change these kernels
	src_x_kernel = 30e3
	src_depth_kernel = 5e3
	min_sta_arrival = 0

	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

	tree = cKDTree(arrivals[:,0][:,None]) ## It might be expensive to keep re-creating this every time step
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
	arrivals_select = arrivals[lp_concat]
	phase_labels_select = phase_labels[lp_concat]
	tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec)

	Trv_subset_p = []
	Trv_subset_s = []
	Station_indices = []
	# Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	## Note, this loop could be vectorized
	for i in range(n_batch):
		# i0 = np.random.randint(0, high = len(x_grids)) ## Will be fixed grid, if x_grids is length 1.
		# n_spc = x_grids[i0].shape[0]

		ind_sta_select = np.unique(ind_use) ## Subset of locations, from total set.
		n_sta_select = len(ind_sta_select)

		# Not, trv_subset_p and trv_subset_s only differ in the last entry, for all iterations of the loop.
		## In other wors, what's the point of this costly duplication of Trv_subset_p and s? Why not more
		## effectively use this data.

		Trv_subset_p.append(np.concatenate((x_grid_trv[:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_s.append(np.concatenate((x_grid_trv[:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Station_indices.append(ind_sta_select) # record subsets used
		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
		# Grid_indices.append(i0)
		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
		sc += len(Sample_indices[-1])

	# sc += len(Sample_indices[-1])
	Trv_subset_p = np.vstack(Trv_subset_p)
	Trv_subset_s = np.vstack(Trv_subset_s)
	Batch_indices = np.hstack(Batch_indices)

	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec
	n_arvs = arrivals_select.shape[0]

	iargsort = np.argsort(arrivals_select[:,0])
	arrivals_select = arrivals_select[iargsort]
	phase_labels_select = phase_labels_select[iargsort]

	## If len(arrivals_select) == 0, this breaks. However, if len(arrivals_select) == 0; then clearly there is no data here...

	query_time_p = Trv_subset_p[:,0] + Batch_indices*offset_per_batch + Trv_subset_p[:,1]*offset_per_station
	query_time_s = Trv_subset_s[:,0] + Batch_indices*offset_per_batch + Trv_subset_s[:,1]*offset_per_station

	ip_p = np.searchsorted(arrivals_select[:,0], query_time_p)
	ip_s = np.searchsorted(arrivals_select[:,0], query_time_s)

	ip_p_pad = ip_p.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s_pad = ip_s.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
	ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

	if len(arrivals_select) > 0:
		## See how much we can "skip", when len(arrivals_select) == 0. 
		rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)
	
	iwhere_p = np.where(phase_labels_select == 0)[0]
	iwhere_s = np.where(phase_labels_select == 1)[0]
	n_arvs_p = len(iwhere_p)
	n_arvs_s = len(iwhere_s)

	if len(iwhere_p) > 0:
		ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
		ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
		ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
		rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.

	if len(iwhere_s) > 0:
		ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)
		ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
		ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)
		rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)

	kernel_sig_t = 5.0 # Can speed up by only using matches.
	k_sta_edges = 8
	k_spc_edges = 15
	k_time_edges = 10 ## Make sure is same as in train_regional_GNN.py
	time_vec_slice = np.arange(k_time_edges)

	# X_fixed = [] # fixed
	# X_query = [] # fixed
	# Locs = [] # fixed
	# Trv_out = [] # fixed

	Inpts = [] # duplicate
	Masks = [] # duplicate
	lp_times = [] # duplicate
	lp_stations = [] # duplicate
	lp_phases = [] # duplicate
	lp_meta = [] # duplicate
	# lp_srcs = [] # duplicate

	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		# grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grid.shape[0]
		n_sta_slice = len(sta_select)

		inpt = np.zeros((x_grid.shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
		if len(arrivals_select) > 0:
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))

		if len(iwhere_p) > 0:
			inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
		
		if len(iwhere_s) > 0:  
			inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))

		trv_out = x_grid_trv[:,sta_select,:] ## Subsetting, into sliced indices.

		## Note adding reshape here, rather than down-stream.
		Inpts.append(inpt[:,sta_select,:].reshape(-1,4)) # sub-select, subset of stations.
		Masks.append((1.0*(inpt[:,sta_select,:] > thresh_mask)).reshape(-1,4))

		## Assemble pick datasets
		perm_vec = -1*np.ones(n_sta)
		perm_vec[sta_select] = np.arange(len(sta_select))
		meta = arrivals[lp[i],:]
		phase_vals = phase_labels[lp[i]]
		times = meta[:,0]
		indices = perm_vec[meta[:,1].astype('int')]
		ineed = np.where(indices > -1)[0]
		times = times[ineed] ## Overwrite, now. Double check if this is ok.
		indices = indices[ineed]
		phase_vals = phase_vals[ineed]
		meta = meta[ineed]	

		# ind_src_unique = np.unique(meta[meta[:,2] > -1.0,2]).astype('int') # ignore -1.0 entries.
		lex_sort = np.lexsort((times, indices)) ## Make sure lexsort doesn't cause any problems
		lp_times.append(times[lex_sort] - time_samples[i])
		lp_stations.append(indices[lex_sort])
		lp_phases.append(phase_vals[lex_sort])
		lp_meta.append(meta[lex_sort]) # final index of meta points into 
		# lp_srcs.append(src_subset)

	# Trv_out.append(trv_out)
	# Locs.append(locs[sta_select])
	# X_fixed.append(x_grid)

	# assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
	# assert(A_edges_time_s.max() < n_spc*n_sta_slice)

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)
## Return the adjacencies for a set of inputs.
def extract_inputs_adjacencies(trv, locs, ind_use, x_grid, x_grid_trv, x_grid_trv_ref, x_grid_trv_pointers_p, x_grid_trv_pointers_s, graph_params, verbose = False):

	if verbose == True:
		st = time.time()

	k_sta_edges, k_spc_edges, k_time_edges = graph_params

	n_sta = locs.shape[0]
	n_spc = x_grid.shape[0]
	n_sta_slice = len(ind_use)

	k_sta_edges = np.minimum(k_sta_edges, len(ind_use) - 2)

	perm_vec = -1*np.ones(locs.shape[0])
	perm_vec[ind_use] = np.arange(len(ind_use))

	A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[ind_use])), torch.Tensor(ftrns1(locs[ind_use])), k = k_sta_edges + 1).flip(0).contiguous())[0]
	A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grid)), torch.Tensor(ftrns1(x_grid)), k = k_spc_edges + 1).flip(0).contiguous())[0]
	A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1)).contiguous()
	A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1)).contiguous()	
	A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1)), dim = 0).contiguous()
	len_dt = len(x_grid_trv_ref)
	A_edges_time_p = x_grid_trv_pointers_p[np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*ind_use.repeat(k_time_edges*len_dt)]
	A_edges_time_s = x_grid_trv_pointers_s[np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*ind_use.repeat(k_time_edges*len_dt)]
	one_vec = np.repeat(ind_use*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere
	A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
	A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
	A_edges_ref = x_grid_trv_ref*1 + 0

	assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
	assert(A_edges_time_s.max() < n_spc*n_sta_slice)

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_edges_time_p, A_edges_time_s, A_edges_ref] ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

def load_templates_region(trv, x_grids, training_params, graph_params, pred_params, dt_embed = 1.0):

	k_sta_edges, k_spc_edges, k_time_edges = graph_params

	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	x_grids_trv = []
	x_grids_trv_pointers_p = []
	x_grids_trv_pointers_s = []
	x_grids_trv_refs = []
	x_grids_edges = []

	for i in range(len(x_grids)):

		trv_out = trv(torch.Tensor(locs), torch.Tensor(x_grids[i]))
		x_grids_trv.append(trv_out.cpu().detach().numpy())

		edge_index = knn(torch.Tensor(ftrns1(x_grids[i])).to(device), torch.Tensor(ftrns1(x_grids[i])).to(device), k = k_spc_edges).flip(0).contiguous()
		edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
		x_grids_edges.append(edge_index)

	max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # + 10.0

	for i in range(len(x_grids)):

		A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations_multiple_grids(x_grids_trv[i], max_t)
		x_grids_trv_pointers_p.append(A_edges_time_p)
		x_grids_trv_pointers_s.append(A_edges_time_s)
		x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?

	return x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs, max_t

def competitive_assignment(w, sta_inds, cost, min_val = 0.02, restrict = None, force_n_sources = None, verbose = False):

	# w is the edges, or weight matrix between sources and arrivals.
	# It's a list of 2d numpy arrays, w = [w1, w2, w3...], where each wi is the
	# weight matrix for phase type i. So usually, w = [wp, ws] (actually, algorithm will probably not work unless it is exactly this!)
	# and wp and ws are shape n_srcs, n_arvs = w[0].shape

	# sta_inds, is the list of station indices, for each of the arrivals, corresponding to entries in the w matrices.
	# E.g., each row of w[0] is the set of all arrivals edge weights to the source for that row of w[0] (using P-wave moveouts)
	# Hence, each row is always referencing the same set of picks (which have arbitrary ordering). But, we need to know which arrivals
	# of this list are associated with each station. So sta_inds just records the index for each arrival. (You could group all arrivals
	# of a given station to be sequential, or not, it doesn't make a difference as long as sta_inds is correct). 

	# The supplemental of Earthquake Arrival Association with Backprojection and Graph Theory explains the
	# pseudocode for making all of the constraint matrices.

	if verbose == True:
		start_time = time()

	n_unique_stations = len(np.unique(sta_inds))
	unique_stations = np.unique(sta_inds)
	n_phases = len(w)

	# Create cost vector c by concatenating all (arrival-source-phase) fitness weight matrices together

	np.random.seed(0)

	for i in range(n_phases):

		w[i][w[i] < min_val] = -1.0*min_val

	n_srcs, n_arvs = w[0].shape

	c = np.concatenate(((-1.0*np.concatenate(w, axis = 0).T).reshape(-1), cost*np.ones(n_srcs)), axis = 0)

	# Initilize constraints A1 - A3, b1 - b3
	# Constraint (1) force each arrival assigned to <= one phase
	# Constraint (2) force each station to have <= 1 phase to each source
	# Constraint (3) force source to be activates if has >= 1 assignment

	A1 = np.zeros((n_arvs, n_phases*n_srcs*n_arvs + n_srcs))
	A2 = np.zeros((n_unique_stations*n_phases*n_srcs, n_phases*n_srcs*n_arvs + n_srcs))
	A3 = np.zeros((n_srcs, n_phases*n_srcs*n_arvs + n_srcs))

	b1 = np.ones((n_arvs, 1))
	b2 = np.ones((n_unique_stations*n_phases*n_srcs, 1))
	b3 = np.zeros((n_srcs, 1))

	# Constraint A1

	for j in range(n_arvs):

		A1[j, j*n_phases*n_srcs:(j + 1)*n_phases*n_srcs] = 1

	# Constraint A2

	sc = 0
	step = n_phases*n_srcs

	for j in range(n_unique_stations):

		ip = np.where(sta_inds == unique_stations[j])[0]
		vec = ip*n_srcs*n_phases

		for k in range(n_srcs):

			for l in range(n_phases):

				for n in range(len(ip)):

					A2[sc, ip[n]*n_srcs*n_phases + k + l*n_srcs] = 1

				sc += 1

	# Constraint A3

	activation_term = -n_phases*n_unique_stations

	vec = np.arange(0, n_phases*n_srcs*n_arvs, n_srcs)

	for j in range(n_srcs):

		A3[j, vec + j] = 1
		A3[j, n_phases*n_srcs*n_arvs + j] = activation_term

	# Concatenate Constraints together

	A = np.concatenate((A1, A2, A3), axis = 0)
	b = np.concatenate((b1, b2, b3), axis = 0)

	# Optional constraints:
	# (1) Restrict activations between pairs of source to <= 1 (this can be used to force a spatio-temporal seperation between active sources)
	# (2) Force a certain number of sources to be active simultaneously

	# Optional Constraint 1

	if restrict is not None:

		O1 = []

		for j in range(len(restrict)):

			vec = np.zeros((1, n_phases*n_srcs*n_arvs + n_srcs))
			vec[0, restrict[j] + n_phases*n_srcs*n_arvs] = 1
			O1.append(vec)

		O1 = np.vstack(O1)
		b1 = np.ones((len(restrict),1))

		A = np.concatenate((A, O1), axis = 0)
		b = np.concatenate((b, b1), axis = 0)

	# Optional Constraint 2

	if force_n_sources is not None:

		O1 = np.zeros((1,n_phases*n_srcs*n_arvs + n_srcs))
		O1[0, n_phases*n_srcs*n_arvs::] = -1
		b1 = -force_n_sources*np.ones((1,1))

		A = np.concatenate((A, O1), axis = 0)
		b = np.concatenate((b, b1), axis = 0)

	# Solve ILP

	x = cp.Variable(n_phases*n_srcs*n_arvs + n_srcs, integer = True)

	prob = cp.Problem(cp.Minimize(c.T@x), constraints = [A@x <= b.reshape(-1), 0 <= x, x <= 1])

	prob.solve()

	assert prob.status == 'optimal', 'competitive assignment solution is not optimal'

	# read result
	# (1) find active sources
	# (2) find arrivals assigned to each source
	# (3) determine arrival phase types

	solution = np.round(x.value)
	
	# (1) find active sources
	
	sources_active = np.where(solution[n_phases*n_srcs*n_arvs::])[0]

	# (2,3) find arrival assignments and phase types

	assignments = solution[0:n_phases*n_srcs*n_arvs].reshape(n_arvs,-1)
	lists = [[] for j in range(len(sources_active))]

	for j in range(len(sources_active)):

		for k in range(n_phases):

			ip = np.where(assignments[:,sources_active[j] + n_srcs*k])[0]

			lists[j].append(ip)

	assignments = lists

	if verbose == True:
		print('competitive assignment took: %0.2f'%(time() - start_time))
		print('CA inferred number of sources: %d'%(len(sources_active)))
		print('CA took: %f seconds \n'%(time() - start_time))

	return assignments, sources_active

def competitive_assignment_split(w, sta_inds, cost, min_val = 0.02, restrict = None, force_n_sources = None, verbose = False):

	# w is the edges, or weight matrix between sources and arrivals.
	# It's a list of 2d numpy arrays, w = [w1, w2, w3...], where each wi is the
	# weight matrix for phase type i. So usually, w = [wp, ws] (actually, algorithm will probably not work unless it is exactly this!)
	# and wp and ws are shape n_srcs, n_arvs = w[0].shape

	# sta_inds, is the list of station indices, for each of the arrivals, corresponding to entries in the w matrices.
	# E.g., each row of w[0] is the set of all arrivals edge weights to the source for that row of w[0] (using P-wave moveouts)
	# Hence, each row is always referencing the same set of picks (which have arbitrary ordering). But, we need to know which arrivals
	# of this list are associated with each station. So sta_inds just records the index for each arrival. (You could group all arrivals
	# of a given station to be sequential, or not, it doesn't make a difference as long as sta_inds is correct). 

	# The supplemental of Earthquake Arrival Association with Backprojection and Graph Theory explains the
	# pseudocode for making all of the constraint matrices.

	if verbose == True:
		start_time = time()

	n_unique_stations = len(np.unique(sta_inds))
	unique_stations = np.unique(sta_inds)
	n_phases = len(w)

	# Create cost vector c by concatenating all (arrival-source-phase) fitness weight matrices together

	np.random.seed(0)

	for i in range(n_phases):

		w[i][w[i] < min_val] = -1.0*min_val

	n_srcs, n_arvs = w[0].shape

	c = np.concatenate(((-1.0*np.concatenate(w, axis = 0).T).reshape(-1), cost*np.ones(n_srcs)), axis = 0)

	# Initilize constraints A1 - A3, b1 - b3
	# Constraint (1) force each arrival assigned to <= one phase
	# Constraint (2) force each station to have <= 1 phase to each source
	# Constraint (3) force source to be activates if has >= 1 assignment

	A1 = np.zeros((n_arvs, n_phases*n_srcs*n_arvs + n_srcs))
	A2 = np.zeros((n_unique_stations*n_phases*n_srcs, n_phases*n_srcs*n_arvs + n_srcs))
	A3 = np.zeros((n_srcs, n_phases*n_srcs*n_arvs + n_srcs))

	b1 = np.ones((n_arvs, 1))
	b2 = np.inf*np.ones((n_unique_stations*n_phases*n_srcs, 1)) ## Allow muliple picks per station, in split phase
	b3 = np.zeros((n_srcs, 1))

	# Constraint A1

	for j in range(n_arvs):

		A1[j, j*n_phases*n_srcs:(j + 1)*n_phases*n_srcs] = 1

	# Constraint A2

	sc = 0
	step = n_phases*n_srcs

	for j in range(n_unique_stations):

		ip = np.where(sta_inds == unique_stations[j])[0]
		vec = ip*n_srcs*n_phases

		for k in range(n_srcs):

			for l in range(n_phases):

				for n in range(len(ip)):

					A2[sc, ip[n]*n_srcs*n_phases + k + l*n_srcs] = 1

				sc += 1

	# Constraint A3

	activation_term = -n_phases*n_unique_stations

	vec = np.arange(0, n_phases*n_srcs*n_arvs, n_srcs)

	for j in range(n_srcs):

		A3[j, vec + j] = 1
		A3[j, n_phases*n_srcs*n_arvs + j] = activation_term

	# Concatenate Constraints together

	A = np.concatenate((A1, A2, A3), axis = 0)
	b = np.concatenate((b1, b2, b3), axis = 0)

	# Optional constraints:
	# (1) Restrict activations between pairs of source to <= 1 (this can be used to force a spatio-temporal seperation between active sources)
	# (2) Force a certain number of sources to be active simultaneously

	# Optional Constraint 1

	if restrict is not None:

		O1 = []

		for j in range(len(restrict)):

			vec = np.zeros((1, n_phases*n_srcs*n_arvs + n_srcs))
			vec[0, restrict[j] + n_phases*n_srcs*n_arvs] = 1
			O1.append(vec)

		O1 = np.vstack(O1)
		b1 = np.ones((len(restrict),1))

		A = np.concatenate((A, O1), axis = 0)
		b = np.concatenate((b, b1), axis = 0)

	# Optional Constraint 2

	if force_n_sources is not None:

		O1 = np.zeros((1,n_phases*n_srcs*n_arvs + n_srcs))
		O1[0, n_phases*n_srcs*n_arvs::] = -1
		b1 = -force_n_sources*np.ones((1,1))

		A = np.concatenate((A, O1), axis = 0)
		b = np.concatenate((b, b1), axis = 0)

	# Solve ILP

	x = cp.Variable(n_phases*n_srcs*n_arvs + n_srcs, integer = True)

	prob = cp.Problem(cp.Minimize(c.T@x), constraints = [A@x <= b.reshape(-1), 0 <= x, x <= 1])

	prob.solve()

	assert prob.status == 'optimal', 'competitive assignment solution is not optimal'

	# read result
	# (1) find active sources
	# (2) find arrivals assigned to each source
	# (3) determine arrival phase types

	solution = np.round(x.value)
	
	# (1) find active sources
	
	sources_active = np.where(solution[n_phases*n_srcs*n_arvs::])[0]

	# (2,3) find arrival assignments and phase types

	assignments = solution[0:n_phases*n_srcs*n_arvs].reshape(n_arvs,-1)
	lists = [[] for j in range(len(sources_active))]

	for j in range(len(sources_active)):

		for k in range(n_phases):

			ip = np.where(assignments[:,sources_active[j] + n_srcs*k])[0]

			lists[j].append(ip)

	assignments = lists

	if verbose == True:
		print('competitive assignment took: %0.2f'%(time() - start_time))
		print('CA inferred number of sources: %d'%(len(sources_active)))
		print('CA took: %f seconds \n'%(time() - start_time))

	return assignments, sources_active

def MLE_particle_swarm_location_one_mean_stable_depth(trv, locs_use, arv_p, ind_p, arv_s, ind_s, lat_range, lon_range, depth_range, ftrns1, ftrns2, sig_t = 3.0, dx_depth = 50.0, n = 300, eps_thresh = 100, eps_steps = 5, init_vel = 1000, max_steps = 300, save_swarm = False, device = 'cpu'):

	if (len(arv_p) + len(arv_s)) == 0:
		return np.nan*np.ones((1,3)), np.nan, []

	def likelihood_estimate(x):

		## Assumes constant diagonal covariance matrix
		pred = trv(locs_use_cuda, torch.Tensor(x).to(device)).cpu().detach().numpy()
		sig_arv = np.concatenate((f_linear(pred[:,ind_p,0]), f_linear(pred[:,ind_s,1])), axis = 1)
		det_vals = np.prod(sig_arv, axis = 1) # *np.prod(sig_s, axis = 1)
		det_vals = 0.0 # np.prod(sig_p, axis = 1) # *np.prod(sig_s, axis = 1)
		pred_vals = remove_mean(np.concatenate((pred[:,ind_p,0], pred[:,ind_s,1]), axis = 1), 1)
		logprob = (-(len(ind_p) + len(ind_s))/2.0)*(2*np.pi) - 0.5*det_vals - 0.5*np.sum(((trgt - pred_vals)/sig_arv)**2, axis = 1)

		return logprob

	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
	x0 = np.random.rand(n, 3)*scale_x + offset_x
	locs_use_cuda = torch.Tensor(locs_use).to(device)

	trgt = remove_mean(np.concatenate((arv_p, arv_s), axis = 0).reshape(1, -1), 1)

	## Make sig_t adaptive to average distance of stations..
	f_linear = lambda t: sig_t*np.ones(t.shape)

	logprob_init = likelihood_estimate(x0)
	x0_argmax = np.argmax(logprob_init)
	x0_max = x0[x0_argmax].reshape(1,-1)
	x0_max_val = logprob_init.max()
	x0cart_max = ftrns1(x0_max)

	xcart = ftrns1(np.copy(x0)) ## Need to do momentum, in Cartesian domain.
	xcart_best = np.copy(xcart) ## List of best positions for each particle.
	xcart_best_val = np.copy(logprob_init)
	xvel = np.random.randn(n,3)
	xvel = init_vel*xvel/np.linalg.norm(xvel, axis = 1, keepdims = True)

	w, c1, c2 = 0.4, 0.4, 0.4 ## Check this parameters!
	## Run particle-swarm.
	diffs = np.inf*np.ones(eps_steps) ## Save update increments in here. (roll, and overwrite).
	## If triangle distance of diff vector is less than threshold, then can terminate updates.
	Swarm = []

	converged_val, ninc = np.inf, 0
	while (converged_val > eps_thresh)*(ninc < max_steps):

		# find new positions, and velocities
		xvel_new = w*xvel + c1*np.random.rand(n,1)*(xcart_best - xcart) + c2*np.random.rand(n,1)*(x0cart_max - xcart)
		xcart_new = xcart + xvel_new ## New X is updated with "new" velocity
		# comput new likelihoods.
		logprob_sample = likelihood_estimate(ftrns2(xcart_new))
		max_sample = logprob_sample.max()
		max_sample_argmax = np.argmax(logprob_sample)

		## Increment, and update differential position measure
		## of best coordinate.
		diffs = np.roll(diffs, 1) ## Roll array one increment.
		diffs[0] = np.linalg.norm(x0cart_max - xcart_new[max_sample_argmax].reshape(1,-1))

		if max_sample > x0_max_val: ## Update global maximum
			x0_argmax = np.argmax(logprob_sample)
			x0_max = ftrns2(xcart_new[x0_argmax].reshape(1,-1))
			x0_max_val = max_sample*1.0
			x0cart_max = xcart_new[x0_argmax].reshape(1,-1)
		ineed_update = np.where(logprob_sample > xcart_best_val)[0]
		if len(ineed_update) > 0: ## Update individual particle maxima
			xcart_best[ineed_update] = xcart_new[ineed_update]
			xcart_best_val[ineed_update] = logprob_sample[ineed_update]
		xcart = np.copy(xcart_new) ## Update positions and velocities
		xvel = np.copy(xvel_new)

		## Update converged val
		converged_val = np.sqrt(np.sum(diffs**2)) ## Triangle sum, to bound max update over a sequence.
		ninc += 1

		if save_swarm == True:
			Swarm.append(np.concatenate((ftrns2(xcart), logprob_sample.reshape(-1,1)), axis = 1))

	# dx_depth = 50.0
	max_elev = float(depth_range[1])
	xdepth_query = np.arange(depth_range[0]*1.0, max_elev, dx_depth)
	xdepth_query = np.sort(np.minimum(np.maximum(xdepth_query + dx_depth*np.random.randn(len(xdepth_query)), float(depth_range[0])), max_elev))

	x0_max_query = np.copy(x0_max).repeat(len(xdepth_query), axis = 0)
	x0_max_query[:,2] = xdepth_query
	logprob_depths = likelihood_estimate(x0_max_query)
	iargmax = np.argmax(logprob_depths)
	x0_max = x0_max_query[iargmax].reshape(1,-1)
	x0_max_val = logprob_depths[iargmax]

	return x0_max, x0_max_val, Swarm

def MLE_particle_swarm_location_one_mean_stable_depth_with_hull(trv, locs_use, arv_p, ind_p, arv_s, ind_s, lat_range, lon_range, depth_range, dx_depth, hull, ftrns1, ftrns2, sig_t = 3.0, n = 300, eps_thresh = 100, eps_steps = 5, init_vel = 1000, max_steps = 300, save_swarm = False, device = 'cpu'):

	if (len(arv_p) + len(arv_s)) == 0:
		return np.nan*np.ones((1,3)), np.nan, []

	def likelihood_estimate(x):

		## Assumes constant diagonal covariance matrix and 
		pred = trv(locs_use_cuda, torch.Tensor(x).to(device)).cpu().detach().numpy()
		sig_arv = np.concatenate((f_linear(pred[:,ind_p,0]), f_linear(pred[:,ind_s,1])), axis = 1)
		det_vals = np.prod(sig_arv, axis = 1) # *np.prod(sig_s, axis = 1)
		det_vals = 0.0 # np.prod(sig_p, axis = 1) # *np.prod(sig_s, axis = 1)
		pred_vals = remove_mean(np.concatenate((pred[:,ind_p,0], pred[:,ind_s,1]), axis = 1), 1)
		logprob = (-(len(ind_p) + len(ind_s))/2.0)*(2*np.pi) - 0.5*det_vals - 0.5*np.sum(((trgt - pred_vals)/sig_arv)**2, axis = 1)

		return logprob

	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
	x0 = np.random.rand(n, 3)*scale_x + offset_x
	locs_use_cuda = torch.Tensor(locs_use).to(device)

	trgt = remove_mean(np.concatenate((arv_p, arv_s), axis = 0).reshape(1, -1), 1)

	## Make sig_t adaptive to average distance of stations..
	f_linear = lambda t: sig_t*np.ones(t.shape)

	logprob_init = likelihood_estimate(x0)
	# logprob_sample = np.copy()
	x0_argmax = np.argmax(logprob_init)
	x0_max = x0[x0_argmax].reshape(1,-1)
	x0_max_val = logprob_init.max()
	x0cart_max = ftrns1(x0_max)
	# val_global_best = logprob_init.max()

	xcart = ftrns1(np.copy(x0)) ## Need to do momentum, in Cartesian domain.
	xcart_best = np.copy(xcart) ## List of best positions for each particle.
	xcart_best_val = np.copy(logprob_init)
	xvel = np.random.randn(n,3)
	xvel = init_vel*xvel/np.linalg.norm(xvel, axis = 1, keepdims = True)

	w, c1, c2 = 0.4, 0.4, 0.4 ## Check this parameters!
	## Run particle-swarm.
	diffs = np.inf*np.ones(eps_steps) ## Save update increments in here. (roll, and overwrite).
	## If triangle distance of diff vector is less than threshold, then can terminate updates.
	Swarm = []

	converged_val, ninc = np.inf, 0
	while (converged_val > eps_thresh)*(ninc < max_steps):

		# find new positions, and velocities
		xvel_new = w*xvel + c1*np.random.rand(n,1)*(xcart_best - xcart) + c2*np.random.rand(n,1)*(x0cart_max - xcart)
		xcart_new = xcart + xvel_new ## New X is updated with "new" velocity

		## Check if points outside hull; if so, re-initilize.
		in_hull_val = in_hull(ftrns2(xcart_new), hull) # Double check this
		ioutside = np.where(in_hull_val == False)[0]
		if len(ioutside) > 0:
			xcart_new[ioutside] = ftrns1(np.random.rand(len(ioutside), 3)*scale_x + offset_x)
			xvel_new[ioutside] = np.random.randn(len(ioutside),3)
			xvel_new[ioutside] = init_vel*xvel_new[ioutside]/np.linalg.norm(xvel_new[ioutside], axis = 1, keepdims = True)

		# comput new likelihoods.
		logprob_sample = likelihood_estimate(ftrns2(xcart_new))
		max_sample = logprob_sample.max()
		max_sample_argmax = np.argmax(logprob_sample)

		## Increment, and update differential position measure
		## of best coordinate.
		diffs = np.roll(diffs, 1) ## Roll array one increment.
		diffs[0] = np.linalg.norm(x0cart_max - xcart_new[max_sample_argmax].reshape(1,-1))

		if max_sample > x0_max_val: ## Update global maximum
			x0_argmax = np.argmax(logprob_sample)
			x0_max = ftrns2(xcart_new[x0_argmax].reshape(1,-1))
			x0_max_val = max_sample*1.0
			x0cart_max = xcart_new[x0_argmax].reshape(1,-1)
		ineed_update = np.where(logprob_sample > xcart_best_val)[0]
		if len(ineed_update) > 0: ## Update individual particle maxima
			xcart_best[ineed_update] = xcart_new[ineed_update]
			xcart_best_val[ineed_update] = logprob_sample[ineed_update]
		xcart = np.copy(xcart_new) ## Update positions and velocities
		xvel = np.copy(xvel_new)

		## Update converged val
		converged_val = np.sqrt(np.sum(diffs**2)) ## Triangle sum, to bound max update over a sequence.
		ninc += 1

		if save_swarm == True:
			Swarm.append(np.concatenate((ftrns2(xcart), logprob_sample.reshape(-1,1)), axis = 1))

	# dx_depth = 50.0
	max_elev = float(depth_range[1])
	xdepth_query = np.arange(depth_range[0]*1.0, max_elev, dx_depth)
	xdepth_query = np.sort(np.minimum(np.maximum(xdepth_query + dx_depth*np.random.randn(len(xdepth_query)), float(depth_range[0])), max_elev))

	x0_max_query = np.copy(x0_max).repeat(len(xdepth_query), axis = 0)
	x0_max_query[:,2] = xdepth_query
	logprob_depths = likelihood_estimate(x0_max_query)
	iargmax = np.argmax(logprob_depths)
	x0_max = x0_max_query[iargmax].reshape(1,-1)
	x0_max_val = logprob_depths[iargmax]

	return x0_max, x0_max_val, Swarm

def sample_random_spatial_query(lat_range, lon_range, depth_range, n):

	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
	X_query = np.random.rand(n, 3)*scale_x + offset_x
	X_query_cart = torch.Tensor(ftrns1(X_query))

	return X_query, X_query_cart

def maximize_bipartite_assignment(cat, srcs, ftrns1, ftrns2, temporal_win = 10.0, spatial_win = 75e3):

	tree_t = cKDTree(srcs[:,3].reshape(-1,1))
	tree_s = cKDTree(ftrns1(srcs[:,0:3]))

	lp_t = tree_t.query_ball_point(cat[:,3].reshape(-1,1), r = temporal_win)
	lp_s = tree_s.query_ball_point(ftrns1(cat[:,0:3]), r = spatial_win)

	edges_cat_to_srcs = [np.array(list(set(lp_t[j]).intersection(lp_s[j]))) for j in range(cat.shape[0])]
	edges_cat_non_zero = np.where(np.array([len(edges_cat_to_srcs[j]) for j in range(cat.shape[0])]) > 0)[0]

	if sum([len(edges_cat_to_srcs[j]) for j in range(cat.shape[0])]) == 0:
		return np.array([]), [], [], [], []

	edges_cat_to_srcs = np.hstack([np.concatenate((j*np.ones(len(edges_cat_to_srcs[j])).reshape(1,-1), edges_cat_to_srcs[j].reshape(1,-1)), axis = 0) for j in edges_cat_non_zero]).astype('int')

	unique_cat_ind = np.sort(np.unique(edges_cat_to_srcs[0,:]))
	unique_src_ind = np.sort(np.unique(edges_cat_to_srcs[1,:]))

	nunique_cat = len(unique_cat_ind)
	nunique_src = len(unique_src_ind)

	temporal_diffs = cat[unique_cat_ind,3].reshape(-1,1) - srcs[unique_src_ind,3].reshape(1,-1)
	spatial_diffs = np.linalg.norm(ftrns1(cat[unique_cat_ind,0:3]).reshape(-1,1,3) - ftrns1(srcs[unique_src_ind,0:3]).reshape(1,-1,3), axis = 2)

	temporal_diffs[abs(temporal_diffs) > temporal_win] = np.inf
	spatial_diffs[abs(spatial_diffs) > spatial_win] = np.inf

	weights = np.exp(-0.5*(temporal_diffs**2)/(temporal_win**2))*np.exp(-0.5*(spatial_diffs**2)/(spatial_win**2))

	weights_unravel = weights.reshape(-1)
	weights_unravel[weights_unravel < 0.01] = -0.01 # This way, non-matches, are left unassigned

	## Make constraint matrix.
	# Each cat, assignment vector to at most one other sources.
	A1 = np.zeros((nunique_cat, len(weights_unravel)))
	for i in range(nunique_cat):
		A1[i,np.arange(nunique_src) + nunique_src*i] = 1.0

	## Make constraint matrix.
	# Each src, assignment vector to at most one other cat.
	A2 = np.zeros((nunique_src, len(weights_unravel)))
	for i in range(nunique_src):
		A2[i,np.arange(nunique_cat)*nunique_src + i] = 1.0

	A = np.concatenate((A1, A2), axis = 0)
	b = np.ones((A.shape[0],1))

	x = cp.Variable(A.shape[1], integer = True)
	prob = cp.Problem(cp.Minimize(-weights_unravel.T@x), constraints = [A@x <= b.reshape(-1), 0 <= x, x <= 1])
	prob.solve()
	assert prob.status == 'optimal', 'competitive assignment solution is not optimal'
	solution = np.round(x.value)

	assignment_vectors = solution.reshape(nunique_cat, nunique_src)
	assert(abs(assignment_vectors.sum(1) <= 1).min() == 1)
	assert(abs(assignment_vectors.sum(0) <= 1).min() == 1)

	results = []
	res = []
	for i in range(nunique_cat):
		i1 = np.where(assignment_vectors[i,:] > 0)[0]

		# print('temporal diff %0.4f'%temporal_diffs[i, i1[0]])
		# print('spatial diff %0.4f'%spatial_diffs[i, i1[0]])

		if len(i1) > 0:
			results.append(np.array([unique_cat_ind[i], unique_src_ind[i1[0]]]).reshape(1,-1))
			res.append((cat[unique_cat_ind[i],0:4] - srcs[unique_src_ind[i1[0]],0:4]).reshape(1,-1))

			print('temporal diff %0.4f'%temporal_diffs[i, i1[0]])
			print('spatial diff %0.4f'%spatial_diffs[i, i1[0]])

	results = np.vstack(results)
	res = np.vstack(res)

	return results, res, assignment_vectors, unique_cat_ind, unique_src_ind

# class MagPred(nn.Module):
# 	def __init__(self, locs, grid, ftrns1_diff, ftrns2_diff, k = 1, device = 'cpu'):
# 		# super(MagPred, self).__init__(aggr = 'max') # node dim
# 		super(MagPred, self).__init__() # node dim

# 		## Predict magnitudes with trainable coefficients,
# 		## and spatial-reciver biases (with knn interp k)

# 		## If necessary, could make the bias parameters, and mag, epicentral, and depth parameters, all stochastic?
# 		## Hence, the entire forward model becomes stochastic, and we optimize the entire thing with the score function.

# 		# In elliptical coordinates
# 		self.locs = locs
# 		self.grid = grid
# 		self.grid_cart = ftrns1_diff(grid)
# 		self.ftrns1 = ftrns1_diff
# 		self.ftrns2 = ftrns2_diff
# 		self.k = k
# 		self.device = device

# 		## Setup like regular log_amp = C1 * Mag + C2 * log_dist_depths_0 + C3 * log_dist_depths + Bias (for each phase type)
# 		self.mag_coef = nn.Parameter(torch.ones(2))
# 		self.epicenter_spatial_coef = nn.Parameter(-torch.ones(2))
# 		self.depth_spatial_coef = nn.Parameter(torch.zeros(2))
# 		# self.bias = nn.Parameter(torch.zeros(locs.shape[0], grid.shape[0], 2), requires_grad = True).to(device)
# 		self.bias = nn.Parameter(torch.zeros(grid.shape[0], locs.shape[0], 2))

# 		self.grid_save = nn.Parameter(grid, requires_grad = False)

# 		self.zvec = torch.Tensor([1.0,1.0,0.0]).reshape(1,-1).to(device)

# 	def log_amplitudes(self, src, ind, mag, phase):

# 		## Input src: n_srcs x 3;
# 		## ind: indices into absolute locs array (can repeat, for phase types)
# 		## log_amp (base 10), for each ind
# 		## phase type for each ind 

# 		fudge = 1.0 # add before log10, to avoid log10(0)

# 		# Compute pairwise distances;
# 		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2) + fudge)
# 		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)) + fudge)

# 		inds = knn(self.grid_cart, self.ftrns1(src), k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
# 		## Can directly use torch_scatter to coalesce the data?

# 		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)

# 		log_amp = mag*torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device)) + self.epicenter_spatial_coef[phase]*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

# 		return log_amp

# 	def batch_log_amplitudes(self, ind, mag, log_dist, log_dist_d, phase):

# 		## Efficient version for training:
# 		## Inputs are just point-wise; indices of stations,
# 		## magnitudes of events, and phase types

# 		# Compute pairwise distances;
# 		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2))
# 		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)))

# 		inds = knn(self.grid_cart, self.ftrns1(src), k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
# 		## Can directly use torch_scatter to coalesce the data?

# 		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)

# 		log_amp = mag*torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device)) + self.epicenter_spatial_coef[phase]*pw_log_dist_zero + self.depth_spatial_coef[phase]*pw_log_dist_depths + bias

# 		return log_amp

# 	## Note, closer between amplitudes and forward
# 	def forward(self, src, ind, log_amp, phase):

# 		## Input src: n_srcs x 3;
# 		## ind: indices into absolute locs array (can repeat, for phase types)
# 		## log_amp (base 10), for each ind
# 		## phase type for each ind

# 		fudge = 1.0 # add before log10, to avoid log10(0)

# 		# Compute pairwise distances;
# 		pw_log_dist_zero = torch.log10(torch.norm(self.ftrns1(src*self.zvec).unsqueeze(1) - self.ftrns1(self.locs[ind]*self.zvec).unsqueeze(0), dim = 2) + fudge)
# 		pw_log_dist_depths = torch.log10(abs(src[:,2].view(-1,1) - self.locs[ind,2].view(1,-1)) + fudge)

# 		inds = knn(self.grid_cart, self.ftrns1(src), k = self.k)[1].reshape(-1,self.k) ## for each of the second one, find indices in the first
# 		## Can directly use torch_scatter to coalesce the data?

# 		bias = self.bias[inds][:,:,ind,phase].mean(1) ## Use knn to average coefficients (probably better to do interpolation or a denser grid + k value!)

# 		mag = (log_amp - self.epicenter_spatial_coef[phase]*pw_log_dist_zero - self.depth_spatial_coef[phase]*pw_log_dist_depths - bias)/torch.maximum(self.mag_coef[phase], torch.Tensor([1e-12]).to(self.device))

# 		return mag

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
	x_grids, corr1, corr2 = z['x_grids'], z['corr1'], z['corr2']
	z.close()

	# Load stations
	z = np.load(path_to_file + '\\%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	# Load trained model
	z = np.load(path_to_file + '\\GNN_TrainedModels\\%s_trained_gnn_model_step_%d_ver_%d_losses.npz'%(name_of_project, n_step_load, n_ver_load))
	training_params, graph_params, pred_params = z['training_params'], z['graph_params'], z['pred_params']
	t_win = pred_params[0]
	z.close()

	## Load travel times
	z = np.load(path_to_file + '\\1D_Velocity_Models_Regional\\%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))

else: ## Linux or Unix

	# Load region
	name_of_project = path_to_file.split('/')[-1] ## Windows
	z = np.load(path_to_file + '/%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + '/Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	x_grids, corr1, corr2 = z['x_grids'], z['corr1'], z['corr2']
	z.close()

	# Load stations
	z = np.load(path_to_file + '/%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	# Load trained model
	z = np.load(path_to_file + '/GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d_losses.npz'%(name_of_project, n_step_load, n_ver_load))
	training_params, graph_params, pred_params = z['training_params'], z['graph_params'], z['pred_params']
	t_win = pred_params[0]
	z.close()

	## Load travel times
	z = np.load(path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))

locs = locs - corr1 + corr2

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

hull = ConvexHull(X)
hull = hull.points[hull.vertices]

x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs, max_t = load_templates_region(trv, x_grids, training_params, graph_params, pred_params)
x_grids_cart_torch = [torch.Tensor(ftrns1(x_grids[i])) for i in range(len(x_grids))]

mz = GCN_Detection_Network_extended_fixed_adjacencies(ftrns1_diff, ftrns2_diff)

load_model = True
if load_model == True:

	mz_list = []
	for i in range(len(x_grids)):
		mz_slice = GCN_Detection_Network_extended_fixed_adjacencies(ftrns1_diff, ftrns2_diff)
		mz_slice.load_state_dict(torch.load(path_to_file + '/GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(name_of_project, n_step_load, n_ver_load), map_location = torch.device('cpu')))
		mz_slice.eval()
		mz_list.append(mz_slice)
		
failed = []
plot_on = False

print('Doing 1 s steps, to avoid issue of repeating time samples')

step = 2.0 # 10
step_abs = 1
day_len = 3600*24
tsteps = np.arange(0, day_len, step) ## Fixed solution grid.
tsteps_abs = np.arange(-t_win/2.0, day_len + t_win/2.0 + 1, step_abs) ## Fixed solution grid, assume 1 second
tree_tsteps = cKDTree(tsteps_abs.reshape(-1,1))

tsteps_abs_cat = cKDTree(tsteps.reshape(-1,1)) ## Make this tree, so can look up nearest time for all cat.

n_batch = 150
n_batches = int(np.floor(len(tsteps)/n_batch))
n_extra = len(tsteps) - n_batches*n_batch
n_overlap = int(t_win/step) # check this

n_samples = int(250e3)
plot_on = False
save_on = True

d_deg = 0.1 ## leads to 42 k grid?
print('Going to compute sources only in interior region')

x1 = np.arange(lat_range[0], lat_range[1] + d_deg, d_deg)
x2 = np.arange(lon_range[0], lon_range[1] + d_deg, d_deg)

use_irregular_reference_grid = True
if use_irregular_reference_grid == True:
	X_query = kmeans_packing_sampling_points(scale_x, offset_x, 3, n_query_grid, ftrns1, n_batch = 3000, n_steps = 5000, n_sim = 1)[0]
	X_query_cart = torch.Tensor(ftrns1(np.copy(X_query)))
else:
	x3 = np.arange(-45e3, 5e3 + 10e3, 20e3)
	x11, x12, x13 = np.meshgrid(x1, x2, x3)
	xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
	X_query = np.copy(xx)
	X_query_cart = torch.Tensor(ftrns1(np.copy(xx)))


# Window over which to "relocate" each 
# event with denser sampling from GNN output
d_deg = 0.018 ## Is this discretization being preserved?
x1 = np.arange(-d_win, d_win + d_deg, d_deg)
x2 = np.arange(-d_win, d_win + d_deg, d_deg)
x3 = np.arange(-d_win_depth, d_win_depth + d_win_depth/5.0, d_win_depth/5.0)
x11, x12, x13 = np.meshgrid(x1, x2, x3)
xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
X_offset = np.copy(xx)

check_if_finished = False

print('Should change this to use all grids, potentially')
x_grid_ind_list = np.sort(np.random.choice(len(x_grids), size = 1, replace = False)) # 15
x_grid_ind_list_1 = np.sort(np.random.choice(len(x_grids), size = len(x_grids), replace = False)) # 15

assert (max([abs(len(x_grids_trv_refs[0]) - len(x_grids_trv_refs[j])) for j in range(len(x_grids_trv_refs))]) == 0)

n_scale_x_grid = len(x_grid_ind_list)
n_scale_x_grid_1 = len(x_grid_ind_list_1)

fail_count = 0
success_count = 0

## Extra default parameters
n_src_query = 1
x_src_query = np.zeros((1,3)) # cat[i0,0:3].reshape(1,-1)
x_src_query_cart = torch.Tensor(ftrns1(x_src_query))
tq_sample = torch.rand(n_src_query)*t_win - t_win/2.0 # Note this part!
tq_sample = torch.zeros(1)
tq = torch.arange(-t_win/2.0, t_win/2.0 + 1.0).reshape(-1,1).float()

date = t0_init + day_select*day_len
yr, mo, dy = date.year, date.month, date.day
date = np.array([yr, mo, dy])

P, ind_use = load_picks(path_to_file, date, locs, stas, lat_range, lon_range, spr_picks = spr_picks, n_ver = n_ver_picks)
locs_use = locs[ind_use]

for cnt, strs in enumerate([0]):

	trv_out_src = trv(torch.Tensor(locs[ind_use]), torch.Tensor(x_src_query)).detach()
	locs_use_cart_torch = torch.Tensor(ftrns1(locs_use))

	for i in range(len(x_grids)):

		# x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs
		A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_edges_time_p, A_edges_time_s, A_edges_ref = extract_inputs_adjacencies(trv, locs, ind_use, x_grids[i], x_grids_trv[i], x_grids_trv_refs[i], x_grids_trv_pointers_p[i], x_grids_trv_pointers_s[i], graph_params)

		spatial_vals = torch.Tensor(((np.repeat(np.expand_dims(x_grids[i], axis = 1), len(ind_use), axis = 1) - np.repeat(np.expand_dims(locs[ind_use], axis = 0), x_grids[i].shape[0], axis = 0)).reshape(-1,3))/scale_x_extend)
		A_src_in_edges = Data(x = spatial_vals, edge_index = A_src_in_prod)
		A_Lg_in_src = Data(x = spatial_vals, edge_index = torch.Tensor(np.ascontiguousarray(np.flip(A_src_in_prod.cpu().detach().numpy(), axis = 0))).long())
		trv_out = trv(torch.Tensor(locs[ind_use]), torch.Tensor(x_grids[i])).detach().reshape(-1,2) ## Can replace trv_out with Trv_out
		mz_list[i].set_adjacencies(A_prod_sta_sta, A_prod_src_src, A_src_in_edges, A_Lg_in_src, A_src_src, torch.Tensor(A_edges_time_p).long(), torch.Tensor(A_edges_time_s).long(), torch.Tensor(A_edges_ref), trv_out)


	tree_picks = cKDTree(P[:,0:2]) # based on absolute indices


	P_perm = np.copy(P)
	perm_vec = -1*np.ones(locs.shape[0])
	perm_vec[ind_use] = np.arange(len(ind_use))
	P_perm[:,1] = perm_vec[P_perm[:,1].astype('int')]

	times_need_l = np.copy(tsteps) # 5 sec overlap, more or less

	## Double check this.
	n_batches = int(np.floor(len(times_need_l)/n_batch))
	times_need = [times_need_l[j*n_batch:(j + 1)*n_batch] for j in range(n_batches)]
	if n_batches*n_batch < len(times_need_l):
		times_need.append(times_need_l[n_batches*n_batch::]) ## Add last few samples

	assert(len(np.hstack(times_need)) == len(np.unique(np.hstack(times_need))))

	# Out_1 = np.zeros((x_grids[x_grid_ind_list[0]].shape[0], len(tsteps_abs))) # assumes all grids have same cardinality
	Out_2 = np.zeros((X_query_cart.shape[0], len(tsteps_abs)))

	for n in range(len(times_need)):

		tsteps_slice = times_need[n]
		tsteps_slice_indices = tree_tsteps.query(tsteps_slice.reshape(-1,1))[1]

		for x_grid_ind in x_grid_ind_list:

			## It might be more efficient if Inpts, Masks, lp_times, and lp_stations were already on Tensor
			[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = extract_inputs_from_data_fixed_grids_with_phase_type(trv, locs, ind_use, P, P[:,4], tsteps_slice, x_grids[x_grid_ind], x_grids_trv[x_grid_ind], lat_range_extend, lon_range_extend, depth_range, max_t, training_params, graph_params, pred_params, ftrns1, ftrns2)

			for i0 in range(len(tsteps_slice)):

				if len(lp_times[i0]) == 0:
					continue ## It will fail if len(lp_times[i0]) == 0!

				## Note: this is repeated, for each pass of the x_grid loop.
				ip_need = tree_tsteps.query(tsteps_abs[tsteps_slice_indices[i0]] + np.arange(-t_win/2.0, t_win/2.0).reshape(-1,1))

				## Need x_src_query_cart and trv_out_src
				with torch.no_grad():
					out = mz_list[x_grid_ind].forward_fixed(torch.Tensor(Inpts[i0]), torch.Tensor(Masks[i0]), torch.Tensor(lp_times[i0]), torch.Tensor(lp_stations[i0]).long(), torch.Tensor(lp_phases[i0].reshape(-1,1)).float(), locs_use, x_grids_cart_torch[x_grid_ind], X_query_cart, x_src_query_cart, tq, tq_sample, trv_out_src)

				# Out_1[:,ip_need[1]] += out[0][:,0:-1,0].cpu().detach().numpy()/n_overlap/n_scale_x_grid
				Out_2[:,ip_need[1]] += out[1][:,0:-1,0].cpu().detach().numpy()/n_overlap/n_scale_x_grid

				if np.mod(i0, 50) == 0:
					print('%d %d %0.2f'%(n, i0, out[1].max().item()))


	iz1, iz2 = np.where(Out_2 > 0.0025)
	Out_2_sparse = np.concatenate((iz1.reshape(-1,1), iz2.reshape(-1,1), Out_2[iz1,iz2].reshape(-1,1)), axis = 1)

	xq = np.copy(X_query)
	ts = np.copy(tsteps_abs)

	use_sparse_peak_finding = False
	if use_sparse_peak_finding == True:

		srcs_init = []
		for i in range(xq.shape[0]):

			ifind_x = np.where(iz1 == i)[0]
			if len(ifind_x) > 0:

				trace = np.zeros(len(ts))
				trace[iz2[ifind_x]] = Out_2_sparse[ifind_x,2]
				
				# ip = np.where(Out[:,i] > thresh)[0]
				ip = find_peaks(trace, height = thresh, distance = int(2*spr)) ## Note: should add prominence as thresh/2.0, which might help detect nearby events. Also, why is min time spacing set as 2 seconds?
				if len(ip[0]) > 0: # why use xq here?
					val = np.concatenate((xq[i,:].reshape(1,-1)*np.ones((len(ip[0]),3)), ts[ip[0]].reshape(-1,1), ip[1]['peak_heights'].reshape(-1,1)), axis = 1)
					srcs_init.append(val)		
	
	else:
	
		Out = np.zeros((X_query.shape[0], len(tsteps_abs))) ## Use dense out array
		Out[Out_2_sparse[:,0].astype('int'), Out_2_sparse[:,1].astype('int')] = Out_2_sparse[:,2]
	
		srcs_init = []
		for i in range(Out.shape[0]):
			# ip = np.where(Out[:,i] > thresh)[0]
			ip = find_peaks(Out[i,:], height = thresh, distance = int(2*spr)) ## Note: should add prominence as thresh/2.0, which might help detect nearby events. Also, why is min time spacing set as 2 seconds?
			if len(ip[0]) > 0: # why use xq here?
				val = np.concatenate((xq[i,:].reshape(1,-1)*np.ones((len(ip[0]),3)), ts[ip[0]].reshape(-1,1), ip[1]['peak_heights'].reshape(-1,1)), axis = 1)
				srcs_init.append(val)

	if len(srcs_init) == 0:
		continue ## No sources, continue

	srcs_init = np.vstack(srcs_init) # Could this have memory issues?

	srcs_init = srcs_init[np.argsort(srcs_init[:,3]),:]
	tdiff = np.diff(srcs_init[:,3])
	ibreak = np.where(tdiff >= break_win)[0]
	srcs_groups_l = []
	ind_inc = 0

	if len(ibreak) > 0:
		for i in range(len(ibreak)):
			srcs_groups_l.append(srcs_init[np.arange(ind_inc, ibreak[i] + 1)])
			ind_inc = ibreak[i] + 1
		if len(np.vstack(srcs_groups_l)) < srcs_init.shape[0]:
			srcs_groups_l.append(srcs_init[(ibreak[-1] + 1)::])
	else:
		srcs_groups_l.append(srcs_init)

	srcs_l = []
	for i in range(len(srcs_groups_l)):
		if len(srcs_groups_l[i]) == 1:
			srcs_l.append(srcs_groups_l[i])
		else:
			mp = LocalMarching()
			srcs_out = mp(srcs_groups_l[i], tc_win = tc_win, sp_win = sp_win)
			if len(srcs_out) > 0:
				srcs_l.append(srcs_out)
	srcs = np.vstack(srcs_l)

	if len(srcs) == 0:
		print('No sources detected, finishing script')
		continue ## No sources, continue

	print('Detected %d number of sources'%srcs.shape[0])

	srcs = srcs[np.argsort(srcs[:,3])]
	trv_out_srcs = trv(torch.Tensor(locs_use), torch.Tensor(srcs[:,0:3])).cpu().detach() # .cpu().detach().numpy() # + srcs[:,3].reshape(-1,1,1)

	## Run post processing detections.
	print('check the thresh assoc %f'%thresh_assoc)

	## Refine this

	n_segment = 100
	srcs_list = []
	n_intervals = int(np.floor(srcs.shape[0]/n_segment))

	for i in range(n_intervals):
		srcs_list.append(np.arange(n_segment) + i*n_segment)

	if len(srcs_list) == 0:
		srcs_list.append(np.arange(srcs.shape[0]))
	elif srcs_list[-1][-1] < (srcs.shape[0] - 1):
		srcs_list.append(np.arange(srcs_list[-1][-1] + 1, srcs.shape[0]))

	## This section is memory intensive if lots of sources are detected.
	## Can "loop" over segements of sources, to keep the cost for manegable.

	srcs_refined_l = []
	trv_out_srcs_l = []
	Out_p_save_l = []
	Out_s_save_l = []

	Save_picks = [] # save all picks..
	lp_meta_l = []

	for n in range(len(srcs_list)):

		Out_refined = []
		X_query_1_list = []
		X_query_1_cart_list = []

		srcs_slice = srcs[srcs_list[n]]
		trv_out_srcs_slice = trv_out_srcs[srcs_list[n]]

		for i in range(srcs_slice.shape[0]):
			# X_query = srcs[i,0:3] + X_offset
			X_query_1 = srcs_slice[i,0:3] + (np.random.rand(n_rand_query,3)*(X_offset.max(0, keepdims = True) - X_offset.min(0, keepdims = True)) + X_offset.min(0, keepdims = True))
			inside = np.where((X_query_1[:,0] > lat_range[0])*(X_query_1[:,0] < lat_range[1])*(X_query_1[:,1] > lon_range[0])*(X_query_1[:,1] < lon_range[1])*(X_query_1[:,2] > depth_range[0])*(X_query_1[:,2] < depth_range[1]))[0]
			X_query_1 = X_query_1[inside]
			X_query_1_cart = torch.Tensor(ftrns1(np.copy(X_query_1))) # 
			X_query_1_list.append(X_query_1)
			X_query_1_cart_list.append(X_query_1_cart)

			Out_refined.append(np.zeros((X_query_1.shape[0], len(tq))))

		for x_grid_ind in x_grid_ind_list_1:

			[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = extract_inputs_from_data_fixed_grids_with_phase_type(trv, locs, ind_use, P, P[:,4], srcs_slice[:,3], x_grids[x_grid_ind], x_grids_trv[x_grid_ind], lat_range_extend, lon_range_extend, depth_range, max_t, training_params, graph_params, pred_params, ftrns1, ftrns2)

			for i in range(srcs_slice.shape[0]):

				if len(lp_times[i]) == 0:
					continue ## It will fail if len(lp_times[i0]) == 0!

				ipick, tpick = lp_stations[i].astype('int'), lp_times[i] ## are these constant across different x_grid_ind?

				# note, trv_out_sources, is already on cuda, may cause memory issue with too many sources
				with torch.no_grad(): 
					out = mz_list[x_grid_ind].forward_fixed(torch.Tensor(Inpts[i]), torch.Tensor(Masks[i]), torch.Tensor(lp_times[i]), torch.Tensor(lp_stations[i]).long(), torch.Tensor(lp_phases[i].reshape(-1,1)).float(), locs_use, x_grids_cart_torch[x_grid_ind], X_query_1_cart_list[i], torch.Tensor(ftrns1(srcs_slice[i,0:3].reshape(1,-1))), tq, torch.zeros(1), trv_out_srcs_slice[[i],:,:])
					Out_refined[i] += out[1][:,:,0].cpu().detach().numpy()/n_scale_x_grid_1

		srcs_refined = []
		for i in range(srcs_slice.shape[0]):

			ip_argmax = np.argmax(Out_refined[i].max(1))
			ipt_argmax = np.argmax(Out_refined[i][ip_argmax,:])
			srcs_refined.append(np.concatenate((X_query_1_list[i][ip_argmax].reshape(1,-1), np.array([srcs_slice[i,3] + tq[ipt_argmax,0].item(), Out_refined[i].max()]).reshape(1,-1)), axis = 1)) 

		srcs_refined = np.vstack(srcs_refined)
		srcs_refined = srcs_refined[np.argsort(srcs_refined[:,3])] # note, this

		## Can do multiple grids simultaneously, for a single source? (by duplicating the source?)
		trv_out_srcs_slice = trv(torch.Tensor(locs_use), torch.Tensor(srcs_refined[:,0:3])).cpu().detach() # .cpu().detach().numpy() # + srcs[:,3].reshape(-1,1,1)		

		srcs_refined_l.append(srcs_refined)
		trv_out_srcs_l.append(trv_out_srcs_slice)

		## Dense, spatial view.
		d_deg = 0.1
		x1 = np.arange(lat_range[0], lat_range[1] + d_deg, d_deg)
		x2 = np.arange(lon_range[0], lon_range[1] + d_deg, d_deg)
		x3 = np.array([srcs_refined[i,2]])
		x11, x12, x13 = np.meshgrid(x1, x2, x3)
		xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
		X_save = np.copy(xx)
		X_save_cart = torch.Tensor(ftrns1(X_save))


		for inc, x_grid_ind in enumerate(x_grid_ind_list_1):

			[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = extract_inputs_from_data_fixed_grids_with_phase_type(trv, locs, ind_use, P, P[:,4], srcs_refined[:,3], x_grids[x_grid_ind], x_grids_trv[x_grid_ind], lat_range_extend, lon_range_extend, depth_range, max_t, training_params, graph_params, pred_params, ftrns1, ftrns2)

			if inc == 0:

				Out_p_save = [np.zeros(len(lp_times[j])) for j in range(srcs_refined.shape[0])]
				Out_s_save = [np.zeros(len(lp_times[j])) for j in range(srcs_refined.shape[0])]

			for i in range(srcs_refined.shape[0]):

				# Does this cause any issues? Could each ipick, tpick, not be constant, between grids?
				ipick, tpick = lp_stations[i].astype('int'), lp_times[i]

				if inc == 0:

					Save_picks.append(np.concatenate((tpick.reshape(-1,1), ipick.reshape(-1,1)), axis = 1))
					lp_meta_l.append(lp_meta[i])

				X_save[:,2] = srcs_refined[i,2]
				X_save_cart = torch.Tensor(ftrns1(X_save))

				with torch.no_grad():
					out = mz_list[x_grid_ind].forward_fixed(torch.Tensor(Inpts[i]), torch.Tensor(Masks[i]), torch.Tensor(lp_times[i]), torch.Tensor(lp_stations[i]).long(), torch.Tensor(lp_phases[i].reshape(-1,1)).long(), locs_use, x_grids_cart_torch[x_grid_ind], X_save_cart, torch.Tensor(ftrns1(srcs_refined[i,0:3].reshape(1,-1))), tq, torch.zeros(1), trv_out_srcs_slice[[i],:,:])
					# Out_save[i,:,:] += out[1][:,:,0].cpu().detach().numpy()/n_scale_x_grid_1
					Out_p_save[i] += out[2][0,:,0].cpu().detach().numpy()/n_scale_x_grid_1
					Out_s_save[i] += out[3][0,:,0].cpu().detach().numpy()/n_scale_x_grid_1

		for i in range(srcs_refined.shape[0]):
			Out_p_save_l.append(Out_p_save[i])
			Out_s_save_l.append(Out_s_save[i])


	srcs_refined = np.vstack(srcs_refined_l)
	trv_out_srcs = trv(torch.Tensor(locs_use), torch.Tensor(srcs_refined[:,0:3])).cpu().detach()
	Out_p_save = Out_p_save_l
	Out_s_save = Out_s_save_l

	iargsort = np.argsort(srcs_refined[:,3])
	srcs_refined = srcs_refined[iargsort]
	trv_out_srcs = trv_out_srcs[iargsort]
	Out_p_save = [Out_p_save[i] for i in iargsort]
	Out_s_save = [Out_s_save[i] for i in iargsort]
	Save_picks = [Save_picks[i] for i in iargsort]
	lp_meta = [lp_meta_l[i] for i in iargsort]

	if use_expanded_competitive_assignment == False:

		Assigned_picks = []
		Picks_P = []
		Picks_S = []
		Picks_P_perm = []
		Picks_S_perm = []
		# Out_save = []

		## Implement CA, so that is runs over disjoint sets of "nearby" sources.
		## Rather than individually, for each source.
		for i in range(srcs_refined.shape[0]):

			## Now do assignments, on the stacked association predictions (over grids)

			ipick, tpick = Save_picks[i][:,1].astype('int'), Save_picks[i][:,0]

			print(i)

			## Need to replace this with competitive assignment over "connected"
			## Sources. This will reduce duplicate events.
			wp = np.zeros((1,len(tpick))); wp[0,:] = Out_p_save[i]
			ws = np.zeros((1,len(tpick))); ws[0,:] = Out_s_save[i]
			wp[wp <= thresh_assoc] = 0.0
			ws[ws <= thresh_assoc] = 0.0
			assignments, srcs_active = competitive_assignment([wp, ws], ipick, 1.5, force_n_sources = 1) ## force 1 source?
			

			# Note, calling tree_picks
			ip_picks = tree_picks.query(lp_meta[i][:,0:2]) # meta uses absolute indices
			assert(abs(ip_picks[0]).max() == 0.0)
			ip_picks = ip_picks[1]

			# p_pred, s_pred = np.zeros(len(tpick)), np.zeros(len(tpick))
			assert(len(srcs_active) == 1)
			## Assumes 1 source

			ind_p = ipick[assignments[0][0]]
			ind_s = ipick[assignments[0][1]]
			arv_p = tpick[assignments[0][0]]
			arv_s = tpick[assignments[0][1]]

			p_assign = np.concatenate((P[ip_picks[assignments[0][0]],:], i*np.ones(len(assignments[0][0])).reshape(-1,1)), axis = 1) ## Note: could concatenate ip_picks, if desired here, so all picks in Picks_P lists know the index of the absolute pick index.
			s_assign = np.concatenate((P[ip_picks[assignments[0][1]],:], i*np.ones(len(assignments[0][1])).reshape(-1,1)), axis = 1)
			p_assign_perm = np.copy(p_assign)
			s_assign_perm = np.copy(s_assign)
			p_assign_perm[:,1] = perm_vec[p_assign_perm[:,1].astype('int')]
			s_assign_perm[:,1] = perm_vec[s_assign_perm[:,1].astype('int')]
			Picks_P.append(p_assign)
			Picks_S.append(s_assign)
			Picks_P_perm.append(p_assign_perm)
			Picks_S_perm.append(s_assign_perm)

			print('add relocation!')

			## Implemente CA, to deal with mixing events (nearby in time, with shared arrival association assignments)

	elif use_expanded_competitive_assignment == True:

		Assigned_picks = []
		Picks_P = []
		Picks_S = []
		Picks_P_perm = []
		Picks_S_perm = []
		# Out_save = []

		## Implement CA, so that is runs over disjoint sets of "nearby" sources.
		## Rather than individually, for each source.

		# ## Find overlapping events (events with shared pick assignments)
		all_picks = np.vstack(lp_meta) # [:,0:2] # np.vstack([Save_picks[i] for i in range(len(Save_picks))])
		# unique_picks = np.unique(all_picks[:,0:2], axis = 0)
		unique_picks = np.unique(all_picks, axis = 0)

		# ip_sort_unique = np.lexsort((unique_picks[:,0], unique_picks[:,1])) # sort by station
		ip_sort_unique = np.lexsort((unique_picks[:,1], unique_picks[:,0])) # sort by time
		unique_picks = unique_picks[ip_sort_unique]
		len_unique_picks = len(unique_picks)

		# tree_picks_select = cKDTree(all_picks[:,0:2])
		tree_picks_unique_select = cKDTree(unique_picks[:,0:2])
		# lp_tree_picks_select  = tree_picks_select.query_ball_point(unique_picks, r = 0)

		matched_src_arrival_indices = []
		matched_src_arrival_indices_p = []
		matched_src_arrival_indices_s = []

		min_picks = 4

		for i in range(len(lp_meta)):

			if len(lp_meta[i]) == 0:
				continue

			matched_arv_indices_val = tree_picks_unique_select.query(lp_meta[i][:,0:2])
			assert(matched_arv_indices_val[0].max() == 0)
			matched_arv_indices = matched_arv_indices_val[1]

			ifind_p = np.where(Out_p_save[i] > thresh_assoc)[0]
			ifind_s = np.where(Out_s_save[i] > thresh_assoc)[0]

			# Check for minimum number of picks, otherwise, skip source
			if (len(ifind_p) + len(ifind_s)) >= min_picks:

				ifind = np.unique(np.concatenate((ifind_p, ifind_s), axis = 0)) # Create combined set of indices

				## concatenate both p and s likelihoods and edges for all of ifind, so that the dense matrices extracted for each
				## disconnected component are the same size.

				## First row is arrival indices, second row are src indices
				# if len(ifind_p) > 0:
				# matched_src_arrival_indices_p.append(np.concatenate((matched_arv_indices[ifind_p].reshape(1,-1), i*np.ones(len(ifind_p)).reshape(1,-1), Out_p_save[i][ifind_p].reshape(1,-1)), axis = 0))
				matched_src_arrival_indices_p.append(np.concatenate((matched_arv_indices[ifind].reshape(1,-1), i*np.ones(len(ifind)).reshape(1,-1), Out_p_save[i][ifind].reshape(1,-1)), axis = 0))

				# if len(ifind_s) > 0:
				# matched_src_arrival_indices_s.append(np.concatenate((matched_arv_indices[ifind_s].reshape(1,-1), i*np.ones(len(ifind_s)).reshape(1,-1), Out_s_save[i][ifind_s].reshape(1,-1)), axis = 0))
				matched_src_arrival_indices_s.append(np.concatenate((matched_arv_indices[ifind].reshape(1,-1), i*np.ones(len(ifind)).reshape(1,-1), Out_s_save[i][ifind].reshape(1,-1)), axis = 0))

				matched_src_arrival_indices.append(np.concatenate((matched_arv_indices[ifind].reshape(1,-1), i*np.ones(len(ifind)).reshape(1,-1), np.concatenate((Out_p_save[i][ifind].reshape(1,-1), Out_s_save[i][ifind].reshape(1,-1)), axis = 0).max(0, keepdims = True)), axis = 0))


		## Are we adding all edges, not just edges above a threshold?

		## Then, remove sources with less than min number of pick assignments.

		## From this, we may not have memory issues with competitive assignment. If so,
		## can still reduce the size of disjoint groups.

		matched_src_arrival_indices = np.hstack(matched_src_arrival_indices)
		matched_src_arrival_indices_p = np.hstack(matched_src_arrival_indices_p)
		matched_src_arrival_indices_s = np.hstack(matched_src_arrival_indices_s)

		## Convert to linear graph, find disconected components, apply CA

		w_edges = np.concatenate((matched_src_arrival_indices[0,:][None,:], matched_src_arrival_indices[1,:][None,:] + len_unique_picks, matched_src_arrival_indices[2,:].reshape(1,-1)), axis = 0)
		wp_edges = np.concatenate((matched_src_arrival_indices_p[0,:][None,:], matched_src_arrival_indices_p[1,:][None,:] + len_unique_picks, matched_src_arrival_indices_p[2,:].reshape(1,-1)), axis = 0)
		ws_edges = np.concatenate((matched_src_arrival_indices_s[0,:][None,:], matched_src_arrival_indices_s[1,:][None,:] + len_unique_picks, matched_src_arrival_indices_s[2,:].reshape(1,-1)), axis = 0)
		assert(np.abs(wp_edges[0:2,:] - ws_edges[0:2,:]).max() == 0)
		# w_edges = np.copy(wp_edges)
		# w_edges[2,:] = np.maximum(wp_edges[2,:], ws_edges[2,:])

		## w_edges: first row are unique arrival indices
		## w_edges: second row are unique src indices (with index 0 being the len(unique_picks))

		## Need to combined wp and ws graphs
		G_nx = nx.Graph()
		G_nx.add_weighted_edges_from(w_edges.T)
		G_nx.add_weighted_edges_from(w_edges[np.array([1,0,2]),:].T)

		Gp_nx = nx.Graph()
		Gp_nx.add_weighted_edges_from(wp_edges.T)
		Gp_nx.add_weighted_edges_from(wp_edges[np.array([1,0,2]),:].T)

		Gs_nx = nx.Graph()
		Gs_nx.add_weighted_edges_from(ws_edges.T)
		Gs_nx.add_weighted_edges_from(ws_edges[np.array([1,0,2]),:].T)

		discon_components = list(nx.connected_components(G_nx))
		discon_components = [np.sort(np.array(list(discon_components[i])).astype('int')) for i in range(len(discon_components))]
		# tree_srcs = cKDTree(w_edges[0:2,:].T)


		finish_splits = False
		max_sources = 15 ## per competitive assignment run
		max_splits = 30
		num_splits = 0
		while finish_splits == False:

			# trgt_clusters = [] # Store the source indices and the target clusters
			# flag_disconnected = []
			# cnt_clusters = 0

			remove_edges_from = []

			discon_components = list(nx.connected_components(G_nx))
			discon_components = [np.sort(np.array(list(discon_components[i])).astype('int')) for i in range(len(discon_components))]

			len_discon = np.array([len(np.where(discon_components[j] > (len_unique_picks - 1))[0]) for j in range(len(discon_components))])
			print('Number discon components: %d \n'%(len(len_discon)))
			print('Number large discon components: %d \n'%(len(np.where(len_discon > max_sources)[0])))
			print('Largest discon component: %d \n'%(max(len_discon)))

			if (len(np.where(len_discon > max_sources)[0]) == 0) or (num_splits > max_splits):
				finish_splits = True
				continue

			print('Beginning split step %d'%num_splits)

			for i in range(len(discon_components)):

				subset_edges = G_nx.subgraph(discon_components[i])
				adj_matrix = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

				subset_edges = Gp_nx.subgraph(discon_components[i])
				adj_matrix_p = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

				subset_edges = Gs_nx.subgraph(discon_components[i])
				adj_matrix_s = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

				# ifind_matched_inds = tree_srcs.query(w_edges[0:2,discon_components[i]].T)[1]

				## Apply CA to the subset of sources/picks in a disconnected component
				ifind_src_inds = np.where(discon_components[i] > (len_unique_picks - 1))[0]
				ifind_arv_inds = np.delete(np.arange(len(discon_components[i])), ifind_src_inds, axis = 0)

				arv_ind_slice = np.sort(discon_components[i][ifind_arv_inds])
				arv_src_slice = np.sort(discon_components[i][ifind_src_inds]) - len_unique_picks
				len_arv_slice = len(arv_ind_slice)

				tpick = unique_picks[arv_ind_slice,0]
				ipick = unique_picks[arv_ind_slice,1].astype('int')

				if len(ifind_src_inds) <= max_sources:

					pass

					# trgt_clusters.append(np.concatenate((ifind_src_inds.reshape(1,-1), cnt_clusters*np.ones(len(ifind_src_inds)).reshape(1,-1)), axis = 0))
					# cnt_clusters = cnt_clusters + 1
					# flag_disconnected.append(np.ones(len(ifind_src_inds)))

				elif len(ifind_src_inds) > max_sources:

					## Create a source-source index graph, based on how much they "share" arrivals. Then find min-cut on this graph,
					## to seperate sources. Modify the discon_components so the sources are split.
					## See if either can directly modify the disconnected components, or use "remove edges" on the graph to
					## directly change disconnected components, and re-compute disconnected components (e.g., like was originally
					## tried). Need to decide how to partition picks between either source, since don't want to duplicate sources
					## once re-computing the solution for either seperately. Also don't want to miss events because of a bad split
					## of picks to either group. The duplicate events may be less problematic due to the remove duplicate event
					## script at the end.

					## If disconnected graphs doesn't work, can also try finding optimal split points by counting origin times or picks
					## in time, and finding min cuts based on minimum rates of origin times or picks

					# subset_edges = G_nx.subgraph(discon_components[i])
					# adj_matrix = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

					w_slice = adj_matrix[len_arv_slice::,0:len_arv_slice] # np.zeros((len(arv_src_slice), len(arv_ind_slice)))
					wp_slice = adj_matrix_p[len_arv_slice::,0:len_arv_slice] # np.zeros((len(arv_src_slice), len(arv_ind_slice)))
					ws_slice = adj_matrix_s[len_arv_slice::,0:len_arv_slice] # np.zeros((len(arv_src_slice), len(arv_ind_slice)))

					isource, iarv = np.where(w_slice > thresh_assoc)
					tree_src_ind = cKDTree(isource.reshape(-1,1)) ## all sources should appear here
					# lp_src_ind = tree_src_ind.query_ball_point(np.sort(np.unique(isource)).reshape(-1,1), r = 0)
					lp_src_ind = tree_src_ind.query_ball_point(np.arange(len(ifind_src_inds)).reshape(-1,1), r = 0)

					assert(len(np.sort(np.unique(isource))) == len(ifind_src_inds))

					## Note: could concievably use MCL on these graphs, just like in original association application.
					## May want to use MCL even on the original source-time graphs as well.
					## Why is memory overloading for recent runs?

					w_src_adj = np.zeros((len(ifind_src_inds), len(ifind_src_inds)))

					for j in range(len(ifind_src_inds)):
						for k in range(len(ifind_src_inds)):
							if j == k:
								continue
							if (len(lp_src_ind[j]) > 0)*(len(lp_src_ind[k]) > 0):
								w_src_adj[j,k] = len(list(set(iarv[lp_src_ind[j]]).intersection(iarv[lp_src_ind[k]])))

					## Try to implement mincut

					# inode1, inode2 = np.where(w_src_adj > 0)

					# _, partition = nx.minimum_cut(G_nx.subgraph(discon_components[i]), discon_components[i][ifind_src_inds[0]], discon_components[i][ifind_src_inds[-1]])
					# g_src = nx.Graph() ## Need to specify the number of sources
					# g_src.add_weighted_edges_from(np.concatenate((inode1.reshape(-1,1), inode2.reshape(-1,1), w_src_adj[inode1, inode2].reshape(-1,1)), axis = 1))
					# cutset = nx.minimum_edge_cut(g_src) # , s = 0, t = discon_components[i][ifind_src_inds[-1]]
					# cutset = nx.minimum_edge_cut(g_src, s = 0, t = len(ifind_src_inds) - 1)
					# cutset = list(cutset)

					## Simply split sources into groups of two (need to make sure this rarely cuts off indidual sources)
					clusters = SpectralClustering(n_clusters = 2, affinity = 'precomputed').fit_predict(w_src_adj)

					i1, i2 = np.where(clusters == 0)[0], np.where(clusters == 1)[0]

					# if len(i1) > 0:
					# 	trgt_clusters.append(np.concatenate((ifind_src_inds[i1].reshape(1,-1), cnt_clusters*np.ones(len(i1)).reshape(1,-1)), axis = 0))
					# 	cnt_clusters = cnt_clusters + 1

					# if len(i2) > 0:
					# 	trgt_clusters.append(np.concatenate((ifind_src_inds[i2].reshape(1,-1), cnt_clusters*np.ones(len(i2)).reshape(1,-1)), axis = 0))
					# 	cnt_clusters = cnt_clusters + 1

					## Optimize all (union) of picks between split sources, so can determine which edges (between arrivals and sources) to delete
					## This should `trim' the source-arrival graphs and increase amount of disconnected components.

					## Which srcs in clusters 1 need to be cut from sources in clusters 2 to disconnect graphs?
					## Can use min-cut with the last source in cluster 1 against the first source in cluster 2.
					# min_time1, min_time2 = srcs_refined[ifind_src_inds[i1],3].min(), srcs_refined[ifind_src_inds[i2],3].min()
					min_time1, min_time2 = srcs_refined[arv_src_slice[i1],3].min(), srcs_refined[arv_src_slice[i2],3].min()

					if min_time1 <= min_time2:
						# cutset = nx.minimum_edge_cut(g_src, s = max(i1), t = min(i2))
						pass
					else:
						i3 = np.copy(i1)
						i1 = np.copy(i2)
						i2 = np.copy(i3)


					## Instead of cut-set, find all sources that "link" across the two groups. Use these as reference sources.
					## In bad cases, could this set also be too big?
					cutset_left = []
					cutset_right = []
					for j in range(len(i1)):
						cutset_right.append(i2[np.where(w_src_adj[i1[j],i2] > 0)[0]])
					for j in range(len(i2)):
						cutset_left.append(i1[np.where(w_src_adj[i2[j],i1] > 0)[0]])

					cutset_left = np.unique(np.hstack(cutset_left))	
					cutset_right = np.unique(np.hstack(cutset_right))	
					cutset = np.unique(np.concatenate((cutset_left, cutset_right), axis = 0))

					# cutset = nx.minimum_edge_cut(g_src, s = max(i1), t = min(i2))

					## Extract the arrival-source weights from w_edges for these nodes
					## Then "take max value" of these picks across these sources
					## Then use CA to maximize assignment of picks to either "distinct"
					## cluster. Then remove those arrival attachements from the full graph
					## for the cluster the picks arn't assigned too. Then, do this for all
					## disconnected graphs, update the disconnected components, and iterate
					## until all graphs are less than or equal to maximum size.

					# cutset = np.array(list(cutset)).astype('int')
					unique_src_inds = np.sort(np.unique(cutset.reshape(-1,1))).astype('int')
					arv_indices_sliced = np.where(w_slice[unique_src_inds,:].max(0) > thresh_assoc)[0]

					arv_weights_p_cluster_1 = wp_slice[np.unique(cutset_left).astype('int').reshape(-1,1), arv_indices_sliced.reshape(1,-1)].max(0).reshape(1,-1)
					arv_weights_s_cluster_1 = ws_slice[np.unique(cutset_left).astype('int').reshape(-1,1), arv_indices_sliced.reshape(1,-1)].max(0).reshape(1,-1)

					arv_weights_p_cluster_2 = wp_slice[np.unique(cutset_right).astype('int').reshape(-1,1), arv_indices_sliced.reshape(1,-1)].max(0).reshape(1,-1)
					arv_weights_s_cluster_2 = ws_slice[np.unique(cutset_right).astype('int').reshape(-1,1), arv_indices_sliced.reshape(1,-1)].max(0).reshape(1,-1)

					arv_weights_p = np.concatenate((arv_weights_p_cluster_1, arv_weights_p_cluster_2), axis = 0)
					arv_weights_s = np.concatenate((arv_weights_s_cluster_1, arv_weights_s_cluster_2), axis = 0)

					## Now: use competitive assignment to optimize pick assignments to either cluster (use a cost on sources, or no?)
					# assignment_picks, srcs_active_picks = competitive_assignment_split([arv_weights_p, arv_weights_s], ipick[arv_indices_sliced], 1.0) ## force 1 source?
					assignment_picks, srcs_active_picks = competitive_assignment_split([arv_weights_p, arv_weights_s], ipick[arv_indices_sliced], 0.0) ## force 1 source?
					node_all_arrivals = arv_ind_slice[arv_indices_sliced]

					if len(assignment_picks) > 0:
						assign_picks_1 = np.unique(np.hstack(assignment_picks[0]))
						# node_arrival_1 = arv_ind_slice[arv_indices_sliced[assign_picks_1]]
					else:
						# node_arrival_1 = np.array([])
						assign_picks_1 = np.array([])

					## Cut these arrivals from sources in group 1
					node_src_1 = arv_src_slice[cutset_left] + len_unique_picks
					node_arrival_1_del = np.delete(node_all_arrivals, assign_picks_1, axis = 0)
					node_arrival_1_repeat = np.repeat(node_arrival_1_del, len(node_src_1), axis = 0)
					node_src_1_repeat = np.tile(node_src_1, len(node_arrival_1_del))
					remove_edges_from.append(np.concatenate((node_arrival_1_repeat.reshape(1,-1), node_src_1_repeat.reshape(1,-1)), axis = 0))

					if len(assignment_picks) > 1:
						assign_picks_2 = np.unique(np.hstack(assignment_picks[1]))
						# node_arrival_2 = arv_ind_slice[arv_indices_sliced[assign_picks_2]]
					else:
						# node_arrival_2 = np.array([])
						assign_picks_2 = np.array([])

					node_src_2 = arv_src_slice[cutset_right] + len_unique_picks
					node_arrival_2_del = np.delete(node_all_arrivals, assign_picks_2, axis = 0)
					node_arrival_2_repeat = np.repeat(node_arrival_2_del, len(node_src_2), axis = 0)
					node_src_2_repeat = np.tile(node_src_2, len(node_arrival_2_del))
					remove_edges_from.append(np.concatenate((node_arrival_2_repeat.reshape(1,-1), node_src_2_repeat.reshape(1,-1)), axis = 0))

					# for j in range(2):
						## Delete the opposite groups (not assigned) pick edges from all graphs, w, wp, ws.
						## Then update graphs and iterate

					# if len(node_arrival_2) > 0:
						## Remove from opposite set of sources

					# if len(node_arrival_1) > 0:
						## Remove from opposite set of sources

					# flag_disconnected.append(np.zeros(len(ifind_src_inds)))

					print('%d %d %d'%(len(arv_ind_slice), sum(clusters == 0), sum(clusters == 1)))


			if len(remove_edges_from) > 0:
				remove_edges_from = np.hstack(remove_edges_from)
				remove_edges_from = np.concatenate((remove_edges_from, np.flip(remove_edges_from, axis = 0)), axis = 1)

				G_nx.remove_edges_from(remove_edges_from.T)
				Gp_nx.remove_edges_from(remove_edges_from.T)
				Gs_nx.remove_edges_from(remove_edges_from.T)

			num_splits = num_splits + 1



		srcs_retained = []
		cnt_src = 0

		for i in range(len(discon_components)):

			## Need to check that each subgraph and sets of edges are for same combinations of source-arrivals,
			## for all three graphs.

			subset_edges = G_nx.subgraph(discon_components[i])
			adj_matrix = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

			subset_edges = Gp_nx.subgraph(discon_components[i])
			adj_matrix_p = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

			subset_edges = Gs_nx.subgraph(discon_components[i])
			adj_matrix_s = nx.adjacency_matrix(subset_edges, nodelist = discon_components[i]).toarray() # nodelist = np.arange(len(discon_components[i]))).toarray()

			# ifind_matched_inds = tree_srcs.query(w_edges[0:2,discon_components[i]].T)[1]

			## Apply CA to the subset of sources/picks in a disconnected component
			ifind_src_inds = np.where(discon_components[i] > (len_unique_picks - 1))[0]
			ifind_arv_inds = np.delete(np.arange(len(discon_components[i])), ifind_src_inds, axis = 0)

			arv_ind_slice = np.sort(discon_components[i][ifind_arv_inds])
			arv_src_slice = np.sort(discon_components[i][ifind_src_inds]) - len_unique_picks
			len_arv_slice = len(arv_ind_slice)

			wp_slice = adj_matrix_p[len_arv_slice::,0:len_arv_slice] # np.zeros((len(arv_src_slice), len(arv_ind_slice)))
			ws_slice = adj_matrix_s[len_arv_slice::,0:len_arv_slice] # np.zeros((len(arv_src_slice), len(arv_ind_slice)))
			
			tpick = unique_picks[arv_ind_slice,0]
			ipick = unique_picks[arv_ind_slice,1].astype('int')

			## Now do assignments, on the stacked association predictions (over grids)

			# ipick, tpick = Save_picks[i][:,1].astype('int'), Save_picks[i][:,0]

			# wp = np.zeros((1,len(tpick))); wp[0,:] = Out_p_save[i]
			# ws = np.zeros((1,len(tpick))); ws[0,:] = Out_s_save[i]

			if (len(ipick) == 0) or (len(arv_src_slice) == 0):
				continue

			# thresh_assoc = 0.125
			wp_slice[wp_slice <= thresh_assoc] = 0.0
			ws_slice[ws_slice <= thresh_assoc] = 0.0
			# assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, 1.5, force_n_sources = 1) ## force 1 source?
			assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, cost_value) ## force 1 source?

			if len(srcs_active) > 0:
				# srcs_retained.append(arv_src_slice[srcs_active])

				for j in range(len(srcs_active)):


					srcs_retained.append(srcs_refined[arv_src_slice[srcs_active[j]]].reshape(1,-1))


					# ind_p = ipick[assignments[0][0]]
					# ind_s = ipick[assignments[0][1]]
					# arv_p = tpick[assignments[0][0]]
					# arv_s = tpick[assignments[0][1]]

					# Assigned_picks.append(np.concatenate((arv_p.reshape(-1,1) + srcs_refined[i,3], ind_p.reshape(-1,1), np.zeros(len(assignments[0][0])).reshape(-1,1), i*np.ones(len(assignments[0][0])).reshape(-1,1)), axis = 1))
					# Assigned_picks.append(np.concatenate((arv_s.reshape(-1,1) + srcs_refined[i,3], ind_s.reshape(-1,1), np.ones(len(assignments[0][1])).reshape(-1,1), i*np.ones(len(assignments[0][1])).reshape(-1,1)), axis = 1))
					p_assign = np.concatenate((unique_picks[arv_ind_slice[assignments[j][0]],:], cnt_src*np.ones(len(assignments[j][0])).reshape(-1,1)), axis = 1) ## Note: could concatenate ip_picks, if desired here, so all picks in Picks_P lists know the index of the absolute pick index.
					s_assign = np.concatenate((unique_picks[arv_ind_slice[assignments[j][1]],:], cnt_src*np.ones(len(assignments[j][1])).reshape(-1,1)), axis = 1)
					p_assign_perm = np.copy(p_assign)
					s_assign_perm = np.copy(s_assign)
					p_assign_perm[:,1] = perm_vec[p_assign_perm[:,1].astype('int')]
					s_assign_perm[:,1] = perm_vec[s_assign_perm[:,1].astype('int')]
					Picks_P.append(p_assign)
					Picks_S.append(s_assign)
					Picks_P_perm.append(p_assign_perm)
					Picks_S_perm.append(s_assign_perm)

					cnt_src += 1


			print('%d : %d of %d'%(i, len(srcs_active), len(arv_src_slice)))

			## Find unique set of arrival indices, write to subset of matrix weights
			## for wp and ws.

			## Then solve CA. Need to scale weights so that: (i). Primarily, the cost is related to the number
			## of picks per event, and (ii). It still identifies "good" fit and "bad" fit source-arrival pairs,
			## based on the source-arrival weights.

			print('add relocation!')

			## Implemente CA, to deal with mixing events (nearby in time, with shared arrival association assignments)

		srcs_refined = np.vstack(srcs_retained)

	# Count number of P and S picks
	cnt_p, cnt_s = np.zeros(srcs_refined.shape[0]), np.zeros(srcs_refined.shape[0])
	for i in range(srcs_refined.shape[0]):
		cnt_p[i] = Picks_P[i].shape[0]
		cnt_s[i] = Picks_S[i].shape[0]


	srcs_trv = []
	for i in range(srcs_refined.shape[0]):
		xmle, logprob, Swarm = MLE_particle_swarm_location_one_mean_stable_depth_with_hull(trv, locs_use, Picks_P_perm[i][:,0], Picks_P_perm[i][:,1].astype('int'), Picks_S_perm[i][:,0], Picks_S_perm[i][:,1].astype('int'), lat_range_extend, lon_range_extend, depth_range, dx_depth, hull, ftrns1, ftrns2)
		if np.isnan(xmle).sum() > 0:
			srcs_trv.append(np.nan*np.ones((1, 4)))
			continue

		pred_out = trv(torch.Tensor(locs_use), torch.Tensor(xmle)).cpu().detach().numpy() + srcs_refined[i,3]

		arv_p, ind_p, arv_s, ind_s = Picks_P_perm[i][:,0], Picks_P_perm[i][:,1].astype('int'), Picks_S_perm[i][:,0], Picks_S_perm[i][:,1].astype('int')

		res_p = pred_out[0,ind_p,0] - arv_p
		res_s = pred_out[0,ind_s,1] - arv_s


		mean_shift = 0.0
		cnt_phases = 0
		if len(res_p) > 0:
			mean_shift += np.median(res_p)*(len(res_p)/(len(res_p) + len(res_s)))
			cnt_phases += 1

		if len(res_s) > 0:
			mean_shift += np.median(res_s)*(len(res_s)/(len(res_p) + len(res_s)))
			cnt_phases += 1

		srcs_trv.append(np.concatenate((xmle, np.array([srcs_refined[i,3] - mean_shift]).reshape(1,-1)), axis = 1))

	srcs_trv = np.vstack(srcs_trv)

	srcs_trv_times = np.nan*np.zeros((srcs_trv.shape[0], locs_use.shape[0], 2))
	ifind_not_nan = np.where(np.isnan(srcs_trv[:,0]) == 0)[0]
	srcs_trv_times[ifind_not_nan,:,:] = trv(torch.Tensor(locs_use), torch.Tensor(srcs_trv[ifind_not_nan,0:3])).cpu().detach().numpy() + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)

	## Compute magnitudes.

	mag_r = np.nan*np.ones(srcs_trv.shape[0])
	mag_trv = np.nan*np.ones(srcs_trv.shape[0])

	# mag_r = []
	# mag_trv = []
	# quant_range = [0.1, 0.9]

	# if load_magnitude_model == True:

	# 	for i in range(srcs_refined.shape[0]):

	# 		if (len(Picks_P[i]) + len(Picks_S[i])) > 0: # Does this fail on one pick?

	# 			ind_p = torch.Tensor(Picks_P[i][:,1]).long()
	# 			ind_s = torch.Tensor(Picks_S[i][:,1]).long()
	# 			log_amp_p = torch.Tensor(np.log10(Picks_P[i][:,2]))
	# 			log_amp_s = torch.Tensor(np.log10(Picks_S[i][:,2]))

	# 			src_r_val = torch.Tensor(srcs_refined[i,0:3].reshape(-1,3))
	# 			src_trv_val = torch.Tensor(srcs_trv[i,0:3].reshape(-1,3))

	# 			ind_val = torch.cat((ind_p, ind_s), dim = 0)
	# 			log_amp_val = torch.cat((log_amp_p, log_amp_s), dim = 0)

	# 			log_amp_val[log_amp_val < -2.0] = -torch.Tensor([np.inf]) # This measurments are artifacts

	# 			phase_val = torch.Tensor(np.concatenate((np.zeros(len(Picks_P[i])), np.ones(len(Picks_S[i]))), axis = 0)).long()

	# 			inot_zero = np.where(np.isinf(log_amp_val.cpu().detach().numpy()) == 0)[0]
	# 			if len(inot_zero) == 0:
	# 				mag_r.append(np.nan)
	# 				mag_trv.append(np.nan)
	# 				continue

	# 			pred_r_val = mags(torch.Tensor(srcs_refined[i,0:3]).reshape(1,-1), ind_val[inot_zero], log_amp_val[inot_zero], phase_val[inot_zero]).cpu().detach().numpy().reshape(-1)
	# 			pred_trv_val = mags(torch.Tensor(srcs_trv[i,0:3]).reshape(1,-1), ind_val[inot_zero], log_amp_val[inot_zero], phase_val[inot_zero]).cpu().detach().numpy().reshape(-1)


	# 			if len(ind_val) > 3:
	# 				qnt_vals = np.quantile(pred_r_val, [quant_range[0], quant_range[1]])
	# 				iwhere_val = np.where((pred_r_val > qnt_vals[0])*(pred_r_val < qnt_vals[1]))[0]
	# 				mag_r.append(np.median(pred_r_val[iwhere_val]))

	# 				qnt_vals = np.quantile(pred_trv_val, [quant_range[0], quant_range[1]])
	# 				iwhere_val = np.where((pred_trv_val > qnt_vals[0])*(pred_trv_val < qnt_vals[1]))[0]
	# 				mag_trv.append(np.median(pred_trv_val[iwhere_val]))

	# 			else:

	# 				mag_r.append(np.median(pred_r_val))
	# 				mag_trv.append(np.median(pred_trv_val))

	# 		else:

	# 			# No picks to estimate magnitude.
	# 			mag_r.append(np.nan)
	# 			mag_trv.append(np.nan)

	# 	mag_r = np.hstack(mag_r)
	# 	mag_trv = np.hstack(mag_trv)

	# else:

	# 	mag_r = np.nan*np.ones(srcs_refined.shape[0])
	# 	mag_trv = np.nan*np.ones(srcs_refined.shape[0])

	trv_out1 = trv(torch.Tensor(locs_use), torch.Tensor(srcs_refined[:,0:3])).cpu().detach().numpy() + srcs_refined[:,3].reshape(-1,1,1) 
	# trv_out2 = trv(torch.Tensor(locs_use), torch.Tensor(srcs_trv[:,0:3])).cpu().detach().numpy() + srcs_trv[:,3].reshape(-1,1,1) 

	trv_out2 = np.nan*np.zeros((srcs_trv.shape[0], locs_use.shape[0], 2))
	ifind_not_nan = np.where(np.isnan(srcs_trv[:,0]) == 0)[0]
	trv_out2[ifind_not_nan,:,:] = trv(torch.Tensor(locs_use), torch.Tensor(srcs_trv[ifind_not_nan,0:3])).cpu().detach().numpy() + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)

	srcs_refined[:,0:3] = srcs_refined[:,0:3] + corr1 - corr2
	srcs_trv[:,0:3] = srcs_trv[:,0:3] + corr1 - corr2

	extra_save = True
	save_on = True
	if save_on == True:

		if '\\' in path_to_file:
			ext_save = path_to_file + '\\Catalog\\%d\\%s_results_continuous_days_%d_%d_%d_ver_%d.hdf5'%(yr, name_of_project, date[0], date[1], date[2], n_save_ver)
		elif '/' in path_to_file:
			ext_save = path_to_file + '/Catalog/%d/%s_results_continuous_days_%d_%d_%d_ver_%d.hdf5'%(yr, name_of_project, date[0], date[1], date[2], n_save_ver)

		file_save = h5py.File(ext_save, 'w')

		julday = int((UTCDateTime(date[0], date[1], date[2]) - UTCDateTime(date[0], 1, 1))/(day_len)) + 1

		file_save['%d_%d_%d_%d_P'%(date[0], date[1], date[2], julday)] = P
		file_save['%d_%d_%d_%d_srcs'%(date[0], date[1], date[2], julday)] = srcs_refined
		file_save['%d_%d_%d_%d_srcs_trv'%(date[0], date[1], date[2], julday)] = srcs_trv
		file_save['%d_%d_%d_%d_locs_use'%(date[0], date[1], date[2], julday)] = locs_use
		file_save['%d_%d_%d_%d_ind_use'%(date[0], date[1], date[2], julday)] = ind_use
		file_save['%d_%d_%d_%d_date'%(date[0], date[1], date[2], julday)] = date
		# file_save['%d_%d_%d_%d_res1'%(date[0], date[1], date[2], julday)] = res1
		# file_save['%d_%d_%d_%d_res2'%(date[0], date[1], date[2], julday)] = res2
		# file_save['%d_%d_%d_%d_izmatch1'%(date[0], date[1], date[2], julday)] = matches1
		# file_save['%d_%d_%d_%d_izmatch2'%(date[0], date[1], date[2], julday)] = matches2
		file_save['%d_%d_%d_%d_cnt_p'%(date[0], date[1], date[2], julday)] = cnt_p
		file_save['%d_%d_%d_%d_cnt_s'%(date[0], date[1], date[2], julday)] = cnt_s
		file_save['%d_%d_%d_%d_tsteps_abs'%(date[0], date[1], date[2], julday)] = tsteps_abs
		file_save['%d_%d_%d_%d_X_query'%(date[0], date[1], date[2], julday)] = X_query
		file_save['%d_%d_%d_%d_mag_r'%(date[0], date[1], date[2], julday)] = mag_r
		file_save['%d_%d_%d_%d_mag_trv'%(date[0], date[1], date[2], julday)] = mag_trv
		file_save['%d_%d_%d_%d_x_grid_ind_list'%(date[0], date[1], date[2], julday)] = x_grid_ind_list
		file_save['%d_%d_%d_%d_x_grid_ind_list_1'%(date[0], date[1], date[2], julday)] = x_grid_ind_list_1
		file_save['%d_%d_%d_%d_trv_out1'%(date[0], date[1], date[2], julday)] = trv_out1
		file_save['%d_%d_%d_%d_trv_out2'%(date[0], date[1], date[2], julday)] = trv_out2

		if extra_save == False: # mem_save == True implies don't save these fields
			file_save['%d_%d_%d_%d_Out'%(date[0], date[1], date[2], julday)] = Out_2_sparse ## Is this heavy?

		for j in range(len(Picks_P)):

			file_save['%d_%d_%d_%d_Picks/%d_Picks_P'%(date[0], date[1], date[2], julday, j)] = Picks_P[j] ## Since these are lists, but they be appended seperatley?
			file_save['%d_%d_%d_%d_Picks/%d_Picks_S'%(date[0], date[1], date[2], julday, j)] = Picks_S[j]
			file_save['%d_%d_%d_%d_Picks/%d_Picks_P_perm'%(date[0], date[1], date[2], julday, j)] = Picks_P_perm[j]
			file_save['%d_%d_%d_%d_Picks/%d_Picks_S_perm'%(date[0], date[1], date[2], julday, j)] = Picks_S_perm[j]

		success_count = success_count + 1
		file_save.close()
		print('finished saving file %d %d %d'%(date[0], date[1], date[2]))
