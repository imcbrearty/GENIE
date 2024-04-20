
import numpy as np
import torch
from scipy.spatial import cKDTree
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
import scipy
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import gamma, beta
import time
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from numpy.matlib import repmat
import networkx as nx
import cvxpy as cp
import itertools
import pathlib
import yaml

from utils import remove_mean

class LocalMarching(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, device = 'cpu'):
		super(LocalMarching, self).__init__(aggr = 'max') # node dim

	## Changed dt to 5 s
	def forward(self, srcs, ftrns1, tc_win = 5, sp_win = 35e3, n_steps_max = 100, tol = 1e-4, scale_depth = 1.0, use_directed = True, device = 'cpu'):

		scale_vec = np.array([1.0, 1.0, scale_depth]).reshape(1,-1) ## Use this to down-weight importance of depths
		## which are sometimes highly seperated for nearby sources, since spatial graphs are usually sparsely populated
		## along depth axis compared to horizontal axis.
		
		srcs_tensor = torch.Tensor(srcs).to(device)
		tree_t = cKDTree(srcs[:,3].reshape(-1,1))
		tree_x = cKDTree(ftrns1(srcs[:,0:3])*scale_vec)
		lp_t = tree_t.query_ball_point(srcs[:,3].reshape(-1,1), r = tc_win)
		lp_x = tree_x.query_ball_point(ftrns1(srcs[:,0:3])*scale_vec, r = sp_win)
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

def extract_inputs_from_data_fixed_grids_with_phase_type(trv, locs, ind_use, arrivals, phase_labels, arrivals_tree, time_samples, x_grid, x_grid_trv, lat_range, lon_range, depth_range, max_t, training_params, graph_params, pred_params, ftrns1, ftrns2, verbose = False):

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

	k_sta_edges, k_spc_edges, k_time_edges = graph_params
	t_win, kernel_sig_t = pred_params[0], pred_params[1]
	
	scale_vec = np.array([1,2*t_win]).reshape(1,-1)
	n_batch = len(time_samples)

	## Max_t is dependent on x_grids_trv. Else, can pass max_t as an input.
	# max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))

	# arrivals_tree = cKDTree(arrivals[:,0][:,None]) ## It might be expensive to keep re-creating this every time step
	lp = arrivals_tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')

	## It is important that arrivals_select and phase_lebels_select remain "referencing the same picks" despite some transformations that occur to them (such as sorting)
	arrivals_select = arrivals[lp_concat]
	phase_labels_select = phase_labels[lp_concat]
	tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec)

	Station_indices = []
	# Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	ind_sta_select = np.unique(ind_use) ## Subset of locations, from total set.
	n_sta_select = len(ind_sta_select)
	
	ivec = np.vstack([i*np.ones((n_spc*len(ind_sta_select),1)) for i in range(n_batch)])
	tp1 = np.concatenate((x_grid_trv[:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1)), axis = 1)
	ts1 = np.concatenate((x_grid_trv[:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1)), axis = 1)
	Trv_subset_p = np.concatenate((np.tile(tp1, [n_batch, 1]), ivec), axis = 1)
	Trv_subset_s = np.concatenate((np.tile(ts1, [n_batch, 1]), ivec), axis = 1)
	
	## Note, this loop could be vectorized
	for i in range(n_batch):
		# i0 = np.random.randint(0, high = len(x_grids)) ## Will be fixed grid, if x_grids is length 1.
		# n_spc = x_grids[i0].shape[0]

		# Not, trv_subset_p and trv_subset_s only differ in the last entry, for all iterations of the loop.
		## In other wors, what's the point of this costly duplication of Trv_subset_p and s? Why not more
		## effectively use this data.

		# Trv_subset_p.append(np.concatenate((x_grid_trv[:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		# Trv_subset_s.append(np.concatenate((x_grid_trv[:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Station_indices.append(ind_sta_select) # record subsets used
		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
		# Grid_indices.append(i0)
		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
		sc += len(Sample_indices[-1])

	# sc += len(Sample_indices[-1])
	# Trv_subset_p = np.vstack(Trv_subset_p)
	# Trv_subset_s = np.vstack(Trv_subset_s)
	Batch_indices = np.hstack(Batch_indices)

	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec
	n_arvs = arrivals_select.shape[0]

	## It is important that arrivals_select and phase_lebels_select remain "referencing the same picks" despite some transformations that occur to them (such as sorting)
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

	# kernel_sig_t = 5.0 # Can speed up by only using matches.
	# k_sta_edges = 8
	# k_spc_edges = 15
	# k_time_edges = 10 ## Make sure is same as in train_regional_GNN.py
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

def extract_inputs_adjacencies(trv, locs, ind_use, x_grid, x_grid_trv, x_grid_trv_ref, x_grid_trv_pointers_p, x_grid_trv_pointers_s, ftrns1, graph_params, device = 'cpu', verbose = False):

	if verbose == True:
		st = time.time()

	k_sta_edges, k_spc_edges, k_time_edges = graph_params

	n_sta = locs.shape[0]
	n_spc = x_grid.shape[0]
	n_sta_slice = len(ind_use)

	k_sta_edges = np.minimum(k_sta_edges, len(ind_use) - 2)

	perm_vec = -1*np.ones(locs.shape[0])
	perm_vec[ind_use] = np.arange(len(ind_use))

	
	A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[ind_use])/1000.0).to(device), torch.Tensor(ftrns1(locs[ind_use])/1000.0).to(device), k = k_sta_edges + 1).flip(0).contiguous())[0]
	A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grid)/1000.0).to(device), torch.Tensor(ftrns1(x_grid)/1000.0).to(device), k = k_spc_edges + 1).flip(0).contiguous())[0]
	A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).to(device)).contiguous()
	A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).to(device)).contiguous()	
	A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1).to(device), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1).to(device)), dim = 0).contiguous()
	len_dt = len(x_grid_trv_ref)
	A_edges_time_p = x_grid_trv_pointers_p[np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*ind_use.repeat(k_time_edges*len_dt)]
	A_edges_time_s = x_grid_trv_pointers_s[np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*ind_use.repeat(k_time_edges*len_dt)]
	one_vec = np.repeat(ind_use*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere

	## Note: is there an issue in this code? Why is n_sta used here when only locs_use (based on locs[ind_use] is used to define A_edges_time_p and A_edges_time_s?)
	## is it because x_grid_trv_pointers_p and x_grid_trv_pointers_s is based on all of locs? Can check by seeing if for each arrival, A_edges_time_p references the k nearest
	## sources with moveouts nearby that arrival, for any choice of locs_use and picks.
	
	A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
	A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
	A_edges_ref = x_grid_trv_ref*1 + 0

	assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
	assert(A_edges_time_s.max() < n_spc*n_sta_slice)

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_edges_time_p, A_edges_time_s, A_edges_ref] ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

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
	b2 = 1e5*np.ones((n_unique_stations*n_phases*n_srcs, 1)) ## Allow muliple picks per station, in split phase
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

def differential_evolution_location(trv, locs_use, arv_p, ind_p, arv_s, ind_s, lat_range, lon_range, depth_range, sig_t = 1.5, weight = [1.0,1.0], popsize = 100, maxiter = 1000, device = 'cpu', disp = True, vectorized = True):

	if (len(arv_p) + len(arv_s)) == 0:
		return np.nan*np.ones((1,3)), np.nan

	def likelihood_estimate(x):

		# x = x.reshape(-1,3)
		if x.ndim == 1:
			x = x.reshape(-1,1)

		pred = trv(torch.Tensor(locs_use).to(device), torch.Tensor(x.T).to(device)).cpu().detach().numpy()
		# sig_arv = np.concatenate((f_linear(pred[:,ind_p,0]), f_linear(pred[:,ind_s,1])), axis = 1)
		# det_vals = np.prod(sig_arv, axis = 1) # *np.prod(sig_s, axis = 1)
		# det_vals = 0.0 # np.prod(sig_p, axis = 1) # *np.prod(sig_s, axis = 1)
		pred_vals = remove_mean(np.concatenate((pred[:,ind_p,0], pred[:,ind_s,1]), axis = 1), 1)
		logprob = -0.5*np.sum(((trgt - pred_vals)**2)*weight_vec/(f_linear(pred_vals)**2), axis = 1)/n_picks

		return -1.0*logprob

	n_loc = locs_use.shape[1]
	n_picks = len(arv_p) + len(arv_s)
	trgt = remove_mean(np.concatenate((arv_p, arv_s), axis = 0).reshape(1, -1), 1)
	if len(weight) == n_picks:
		weight_vec = np.copy(np.array(weight)).reshape(1,-1)
	else:
		weight_vec = np.concatenate((weight[0]*np.ones(len(ind_p)), weight[1]*np.ones(len(ind_s))), axis = 0).reshape(1,-1)

	## Make sig_t adaptive to average distance of stations..
	f_linear = lambda t: sig_t*np.ones(t.shape)

	bounds = [(lat_range[0], lat_range[1]), (lon_range[0], lon_range[1]), (depth_range[0], depth_range[1])]
	optim = scipy.optimize.differential_evolution(likelihood_estimate, bounds, popsize = popsize, maxiter = maxiter, disp = disp, vectorized = vectorized)

	return optim.x.reshape(1,-1), likelihood_estimate(optim.x.reshape(-1,1))

def MLE_particle_swarm_location_with_hull(trv, locs_use, arv_p, ind_p, arv_s, ind_s, lat_range, lon_range, depth_range, dx_depth, hull, ftrns1, ftrns2, sig_t = 3.0, n = 300, eps_thresh = 100, eps_steps = 5, init_vel = 1000, max_steps = 300, save_swarm = False, device = 'cpu'):

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

	return x0_max, x0_max_val

def maximize_bipartite_assignment(cat, srcs, ftrns1, ftrns2, temporal_win = 10.0, spatial_win = 75e3, verbose = True):

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

			if verbose == True:
				print('temporal diff %0.4f'%temporal_diffs[i, i1[0]])
				print('spatial diff %0.4f'%spatial_diffs[i, i1[0]])

	results = np.vstack(results)
	res = np.vstack(res)

	return results, res, assignment_vectors, unique_cat_ind, unique_src_ind
