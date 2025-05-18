
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Note: there is already a way to build the subgraphs in automatic differentation and direct training script, but those ways are
## somewhat complex. This is to simplify it and build the "full" dense graph (not k-nn sampled), and then sample from it.

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
from torch_geometric.utils import get_laplacian
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.cluster import SpectralClustering
from torch.autograd import Variable
from numpy.matlib import repmat
from torch_geometric.utils import degree
import pathlib
import itertools
import sys

from scipy.signal import find_peaks
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from obspy.geodetics.base import calc_vincenty_inverse
import matplotlib.gridspec as gridspec
import networkx as nx
import cvxpy as cp
import shutil
import glob

from utils import *
from module import *
from process_utils import *
from calibration_utils import *



path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

## Note, parameters d_deg (multiple re-uses), n_batch, n_segment, min_picks, 
## and max_sources (per connected graph), max_splits can be moved to config
## Also replace "iz1, iz2 = np.where(Out_2 > 0.0025)" with specified thresh

## Need to update how extract_inputs_from_data_fixed_grids_with_phase_type uses a variable t_win parammeter, 
## and also adding inputs of training_params, graph_params, pred_params

# The first system argument (after the file name; e.g., argvs[1]) is an integer used to select which
# day in the %s_process_days_list_ver_%d.txt file each call of this script will compute
argvs = sys.argv
if len(argvs) < 2: 
	argvs.append(0) 

if len(argvs) < 3:
	argvs.append(0)
# This index can also be incremented by the larger value: argvs[2]*offset_increment (defined in process_config)
# to help process very large pick lists with a combinations of using job arrays
# to increment argvs[1], and seperate sbatch scripts incrementing argvs[2]

day_select = int(argvs[1])
offset_select = int(argvs[2])

print('name of program is %s'%argvs[0])
print('day is %s'%argvs[1])

### Settings: ###

with open('calibration_config.yaml', 'r') as file:
    calibration_config = yaml.safe_load(file)

with open('process_config.yaml', 'r') as file:
    process_config = yaml.safe_load(file)

## Load device
device = calibration_config['device']
device = 'cpu'

## Load Processing settings
n_ver_load = process_config['n_ver_load']
n_step_load = process_config['n_step_load']
n_save_ver = process_config['n_save_ver']
n_ver_picks = process_config['n_ver_picks']

template_ver = process_config['template_ver']
vel_model_ver = process_config['vel_model_ver']
process_days_ver = process_config['process_days_ver']

offset_increment = process_config['offset_increment']
n_rand_query = process_config['n_rand_query']
n_query_grid = process_config['n_query_grid']

thresh = process_config['thresh'] # Threshold to declare detection
thresh_assoc = process_config['thresh_assoc'] # Threshold to declare src-arrival association
spr = process_config['spr'] # Sampling rate to save temporal predictions
tc_win = process_config['tc_win'] # Temporal window (s) to link events in Local Marching
sp_win = process_config['sp_win'] # Distance (m) to link events in Local Marching
break_win = process_config['break_win'] # Temporal window to find disjoint groups of sources, 
## so can run Local Marching without memory issues.
spr_picks = process_config['spr_picks'] # Assumed sampling rate of picks 
## (can be 1 if absolute times are used for pick time values)

d_win = process_config['d_win'] ## Lat and lon window to re-locate initial source detetections with refined sampling over
d_win_depth = process_config['d_win_depth'] ## Depth window to re-locate initial source detetections with refined sampling over
dx_depth = process_config['dx_depth'] ## Depth resolution to locate events with travel time based re-location

step = process_config['step']
step_abs = process_config['step_abs']

cost_value = process_config['cost_value'] # If use expanded competitve assignment, then this is the fixed cost applied per source
## when optimizing joint source-arrival assignments between nearby sources. The value is in terms of the 
## `sum' over the predicted source-arrival assignment for each pick. Ideally could make this number more
## adpative, potentially with number of stations or number of possible observing picks for each event. 

# device = torch.device(process_config['device']) ## Right now, this isn't updated to work with cuda, since
## the necessary variables do not have .to(device) at the right places

compute_magnitudes = process_config['compute_magnitudes']
min_log_amplitude_val = process_config['min_log_amplitude_val']
process_known_events = process_config['process_known_events']
use_expanded_competitive_assignment = process_config['use_expanded_competitive_assignment']
use_differential_evolution_location = process_config['use_differential_evolution_location']

print('Beginning calibration')
### Begin automated processing ###

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

name_of_project = config['name_of_project']

name_of_project = config['name_of_project']
use_physics_informed = config['use_physics_informed']
use_subgraph = config['use_subgraph']
if use_subgraph == True:
	max_deg_offset = config['max_deg_offset']
	k_nearest_pairs = config['k_nearest_pairs']

# # Load day to process
# z = open(path_to_file + '%s_process_days_list_ver_%d.txt'%(name_of_project, process_days_ver), 'r')
# lines = z.readlines()
# z.close()
# day_select_val = day_select + offset_select*offset_increment
# if '/' in lines[day_select_val]:
# 	date = lines[day_select_val].split('/')
# elif ',' in lines[day_select_val]:
# 	date = lines[day_select_val].split(',')
# else:
# 	date = lines[day_select_val].split(' ')	
# date = np.array([int(date[0]), int(date[1]), int(date[2])])

# Load region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load templates
z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
x_grids = z['x_grids']
z.close()

# load_absolute_sta_list = False
# if load_absolute_sta_list == True:
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

def lla2ecef_diff(p, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	# p = p.detach().clone().float().to(device) # why include detach here?
	pi = torch.Tensor([np.pi]).to(device)
	corr_val = torch.Tensor([pi/180.0, pi/180.0]).view(1,-1).to(device)
	# p[:,0:2] = p[:,0:2]*
	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0]*corr_val[0,0])**2)
	# results:
	x = (N + p[:,2])*torch.cos(p[:,0]*corr_val[0,0])*torch.cos(p[:,1]*corr_val[0,1])
	y = (N + p[:,2])*torch.cos(p[:,0]*corr_val[0,0])*torch.sin(p[:,1]*corr_val[0,1])
	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0]*corr_val[0,0])

	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

def ecef2lla_diff(x, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	pi = torch.Tensor([np.pi]).to(device)
	b = torch.sqrt((a**2)*(1 - e**2))
	ep = torch.sqrt((a**2 - b**2)/(b**2))
	p = torch.sqrt(x[:,0]**2 + x[:,1]**2)
	th = torch.atan2(a*x[:,2], b*p)
	lon = torch.atan2(x[:,1], x[:,0])
	lat = torch.atan2((x[:,2] + (ep**2)*b*(torch.sin(th)**3)), (p - (e**2)*a*(torch.cos(th)**3)))
	N = a/torch.sqrt(1 - (e**2)*(torch.sin(lat)**2))
	alt = p/torch.cos(lat) - N
	# lon = np.mod(lon, 2.0*np.pi) # don't use!
	k = (torch.abs(x[:,0]) < 1) & (torch.abs(x[:,1]) < 1)
	alt[k] = torch.abs(x[k,2]) - b
	
	return torch.cat((180.0*lat[:,None]/pi, 180.0*lon[:,None]/pi, alt[:,None]), axis = 1)

def global_max_pool(x, batch, size=None):
	## From: "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html"
	"""
	Globally pool node embeddings into graph embeddings, via elementwise mean.
	Pooling function takes in node embedding [num_nodes x emb_dim] and
	batch (indices) and outputs graph embedding [num_graphs x emb_dim].

	Args:
		x (torch.tensor): Input node embeddings
		batch (torch.tensor): Batch tensor that indicates which node
		belongs to which graph
		size (optional): Total number of graphs. Can be auto-inferred.

	Returns: Pooled graph embeddings

	"""
	size = batch.max().item() + 1 if size is None else size
	return scatter(x, batch, dim=0, dim_size=size, reduce='max')

def build_source_graph(srcs, tree_srcs, n_seed, n_neighbors, max_source_pair_distance, ftrns1, prob = False, n_neighbors_ratio = 1/3, weight_depth = 1.0, use_efficient_lookup = False, use_efficient_lookup_next = True, add_missing_edges = False):

	## Make graphs union with knn graphs, to improve coverage with isolated nodes

	if prob == False:
		isample = np.sort(np.random.choice(len(srcs), size = np.minimum(n_seed, len(srcs)), replace = False))
	else:
		isample = np.sort(np.random.choice(len(srcs), size = np.minimum(n_seed, len(srcs)), p = prob/prob.sum(), replace = False))

	weight_depth_vec = np.array([1.0, 1.0, weight_depth]).reshape(1,-1)
	nk_scale = 10 ## Scale the k-nearest neighbor set
	srcs_cart = ftrns1(srcs)

	## Must also remove self-loops (or leave in?)
	if use_efficient_lookup == True: ## If use efficient look-up, have to also delete source pairs of too large of distance
		irand_neighbors = tree_srcs.query(srcs_cart[isample]*weight_depth_vec, k = int(nk_scale*n_neighbors))[1] ## Random selection of k from within 3*k neighbors
	else:
		irand_neighbors = tree_srcs.query_ball_point(srcs_cart[isample]*weight_depth_vec, r = max_source_pair_distance)

	irand_neighbors = [np.random.choice(islice, size = np.minimum(n_neighbors, len(islice)), replace = False) for islice in irand_neighbors]

	## Create initial set of edges between sources and direct neighbors
	edges_srcs = np.hstack([np.concatenate((irand_neighbors[j].reshape(1,-1), isample[j]*np.ones(len(irand_neighbors[j])).reshape(1,-1)), axis = 0) for j in range(n_seed)]).astype('int')

	## Extract unique neighbors, and find neighborhoods for each
	unique_neighbors = np.sort(np.unique(edges_srcs[0]))

	## Must also remove self-loops (or leave in?)
	# use_efficient_lookup = True
	if use_efficient_lookup_next == True: ## If use efficient look-up, have to also delete source pairs of too large of distance
		# nk_scale = 3 ## Scale the k-nearest neighbor set
		irand_next_neighbors = tree_srcs.query(srcs_cart[unique_neighbors]*weight_depth_vec, k = int(nk_scale*n_neighbors))[1] ## Random selection of k from within 3*k neighbors
	else:
		irand_next_neighbors = tree_srcs.query_ball_point(srcs_cart[unique_neighbors]*weight_depth_vec, r = max_source_pair_distance)

	irand_next_neighbors = [np.random.choice(islice, size = np.minimum(n_neighbors, len(islice)), replace = False) for islice in irand_next_neighbors]

	## Create initial set of edges between sources and direct neighbors
	edges_srcs_next = np.hstack([np.concatenate((irand_next_neighbors[j].reshape(1,-1), unique_neighbors[j]*np.ones(len(irand_next_neighbors[j])).reshape(1,-1)), axis = 0) for j in range(len(unique_neighbors))]).astype('int')

	## Now add edges for all "new" nodes, back to all remaining nodes (or only type one and type two nodes)
	unique_new_neighbors = np.sort(np.unique(edges_srcs_next[0]))

	## Need to add edges allowed between tier 3 and other tier 3 nodes.
	# ind_reference = np.concatenate((isample, unique_neighbors), axis = 0) ## Put unique_neighbors first, so that
	ind_reference = np.concatenate((unique_neighbors, unique_new_neighbors, isample), axis = 0) ## Put unique_neighbors first, so that
	# ind_reference = np.concatenate((unique_neighbors, isample), axis = 0) ## Put unique_neighbors first, so that
	tree_reference_srcs = cKDTree(srcs_cart[ind_reference]*weight_depth_vec)

	## Must also remove self-loops (or leave in?)
	if use_efficient_lookup_next == True: ## If use efficient look-up, have to also delete source pairs of too large of distance
		# nk_scale = 3 ## Scale the k-nearest neighbor set
		irand_neighbors_close = tree_reference_srcs.query(srcs_cart[unique_new_neighbors]*weight_depth_vec, k = int(nk_scale*n_neighbors))[1] ## Random selection of k from within 3*k neighbors
	else:
		irand_neighbors_close = tree_reference_srcs.query_ball_point(srcs_cart[unique_new_neighbors]*weight_depth_vec, r = max_source_pair_distance)	

	n_neighbors_close = int(np.ceil(n_neighbors*n_neighbors_ratio))
	irand_neighbors_close = [np.random.choice(islice, size = np.minimum(n_neighbors_close, len(islice)), replace = False) for islice in irand_neighbors_close]

	## Create initial set of edges between sources and direct neighbors
	edges_srcs_close = np.hstack([np.concatenate((ind_reference[irand_neighbors_close[j]].reshape(1,-1), unique_new_neighbors[j]*np.ones(len(irand_neighbors_close[j])).reshape(1,-1)), axis = 0) for j in range(len(unique_new_neighbors))]).astype('int')

	## Concatenate all nodes, find node types, concatenate edges.
	## Note: after building subgraph of Cartesian product, remove source-station pairs with degree < min_degree (or else, since all sources are included, nearly all of picks will be included too)
	## Or will have to decrease seed nodes substantially.

	nodes_all = np.unique(np.concatenate((isample, unique_neighbors, unique_new_neighbors), axis = 0))
	edges_srcs_all = np.concatenate((edges_srcs, edges_srcs_next, edges_srcs_close), axis = 1)
	node_types = -1*np.ones(len(nodes_all)).astype('int')
	node_types[np.where(cKDTree(unique_new_neighbors.reshape(-1,1)).query(nodes_all.reshape(-1,1))[0] == 0)[0]] = 2
	node_types[np.where(cKDTree(unique_neighbors.reshape(-1,1)).query(nodes_all.reshape(-1,1))[0] == 0)[0]] = 1
	node_types[np.where(cKDTree(isample.reshape(-1,1)).query(nodes_all.reshape(-1,1))[0] == 0)[0]] = 0 ## Overwrite all nodes
	assert(node_types.min() > -1)
	assert(len(np.where(node_types == 0)[0]) == n_seed)
	# ifind_edges = np.where(cKDTree(isample.reshape(-1,1)).query(edges_srcs_all[1].reshape(-1,1))[0] == 0)[0]

	# assert(tree.query(edges_srcs_all.reshape(-1,1))[0].max() == 0)


	if add_missing_edges == True:

		use_efficient = True
		if use_efficient == False:

			tree_remaining = cKDTree(srcs_cart[nodes_all]*weight_depth_vec)
			irand_neighbors_remaining = tree_remaining.query_ball_point(srcs_cart[nodes_all], r = max_source_pair_distance)
			edges_srcs_remaining = np.hstack([np.concatenate((nodes_all[irand_neighbors_remaining[j]].reshape(1,-1), nodes_all[j]*np.ones(len(irand_neighbors_remaining[j])).reshape(1,-1)), axis = 0) for j in range(len(irand_neighbors_remaining))]).astype('int')
			edges_srcs_all = np.unique(np.concatenate((edges_srcs_all, edges_srcs_remaining), axis = 1), axis = 1)

		else:

			tree_remaining = cKDTree(srcs_cart[nodes_all]*weight_depth_vec)
			# irand_neighbors_remaining = tree_remaining.query(srcs_cart[nodes_all]*weight_depth_vec, k = int(nk_scale*n_neighbors))[1]
			irand_neighbors_remaining = tree_remaining.query(srcs_cart[nodes_all]*weight_depth_vec, k = int(nk_scale*n_neighbors))[1]
			irand_neighbors_remaining = [np.random.choice(islice, size = np.minimum(n_neighbors, len(islice)), replace = False) for islice in irand_neighbors_remaining]
			edges_srcs_remaining = np.hstack([np.concatenate((nodes_all[irand_neighbors_remaining[j]].reshape(1,-1), nodes_all[j]*np.ones(len(irand_neighbors_remaining[j])).reshape(1,-1)), axis = 0) for j in range(len(irand_neighbors_remaining))]).astype('int')
			edges_srcs_all = np.unique(np.concatenate((edges_srcs_all, edges_srcs_remaining), axis = 1), axis = 1)

	## Add missing edges between nodes in tier 1 and tier 2 levels (within radius).
	## Will increase the number of comparisons made
	add_missing_edges_upper_levels = True
	if add_missing_edges_upper_levels == True:

		n_up_scale_edges = 3
		ifind_upper = np.where(node_types <= 1)[0] ## Nodes of type 0 or 1
		nodes_slice = nodes_all[ifind_upper]
		tree_remaining_upper = cKDTree(srcs_cart[nodes_slice]*weight_depth_vec)
		irand_neighbors_remaining_upper = tree_remaining_upper.query_ball_point(srcs_cart[nodes_slice], r = max_source_pair_distance) ## To expensive to add all of them
		irand_neighbors_remaining_upper = [np.random.choice(islice, size = np.minimum(n_up_scale_edges*n_neighbors, len(islice)), replace = False) for islice in irand_neighbors_remaining_upper]
		edges_srcs_remaining_upper = np.hstack([np.concatenate((nodes_slice[irand_neighbors_remaining_upper[j]].reshape(1,-1), nodes_slice[j]*np.ones(len(irand_neighbors_remaining_upper[j])).reshape(1,-1)), axis = 0) for j in range(len(irand_neighbors_remaining_upper))]).astype('int')
		edges_srcs_all = np.unique(np.concatenate((edges_srcs_all, edges_srcs_remaining_upper), axis = 1), axis = 1)
		# print('Added missed edges %d'%edges_srcs_remaining_upper.shape[1])
		print(np.round(np.quantile(np.linalg.norm(srcs_cart[edges_srcs_remaining_upper[0]] - srcs_cart[edges_srcs_remaining_upper[1]], axis = 1), np.arange(0, 1.1, 0.1)), 3))

	## Remove self loops
	idel_edges = np.where(edges_srcs_all[0] == edges_srcs_all[1])[0]
	edges_srcs_all = np.delete(edges_srcs_all, idel_edges, axis = 1)


	## Remove edges too large; add edges within radius (or to k-nn nearest neighbors)
	remove_large_edges = True
	if remove_large_edges == True:
		dist_vals = np.linalg.norm(srcs_cart[edges_srcs_all[0]]*weight_depth_vec - srcs_cart[edges_srcs_all[1]]*weight_depth_vec, axis = 1)
		idel_edges = np.where(dist_vals > max_source_pair_distance)[0]
		edges_srcs_all = np.delete(edges_srcs_all, idel_edges, axis = 1)


	## Note: there are some isolated sources, following removing large edges (these are simply sources that should have been removed a priori)
	## Can remove them with "efficient" knn search (not radius look up)

	## Do this after permuting edges to subset of sources
	# ifind_edges = np.where((node_types[edges_srcs_all[1]] == 0)*(node_types[edges_srcs_all[0]] == 1))[0]

	perm_edges = -1*np.ones(len(srcs)).astype('int')
	perm_edges[nodes_all] = np.arange(len(nodes_all))
	edges_srcs_all = edges_srcs_all[:,np.lexsort(edges_srcs_all)]
	edges_srcs_abs = np.copy(edges_srcs_all)
	edges_srcs = perm_edges[np.copy(edges_srcs_all)]
	ifind_edges1 = np.where((node_types[edges_srcs[1]] == 0)*(node_types[edges_srcs[0]] == 1))[0] ## Pairs of sources between nodes of tier 1 and tier 2 type (of either direction)
	ifind_edges2 = np.where((node_types[edges_srcs[1]] == 1)*(node_types[edges_srcs[0]] == 0))[0]
	ifind_edges3 = np.where((node_types[edges_srcs[1]] == 0)*(node_types[edges_srcs[0]] == 0))[0]
	ifind_edges4 = np.where((node_types[edges_srcs[1]] == 1)*(node_types[edges_srcs[0]] == 1))[0]
	ifind_edges = np.unique(np.concatenate((ifind_edges1, ifind_edges2, ifind_edges3, ifind_edges4), axis = 0))
	srcs_slice = np.copy(srcs[nodes_all])
	assert(edges_srcs.max() < nodes_all.shape[0])

	# edges_srcs = edges_srcs[:,np.lexsort(edges_srcs)]

	return nodes_all, node_types, srcs_slice, edges_srcs, edges_srcs_abs, ifind_edges, isample

def extract_subgraph_of_cartesian_product(locs, srcs_slice, A_sta_sta, A_src_src, A_src_in_sta, ftrns1, ftrns2, verbose = False, scale_pairwise_sta_in_src_distances = 300e3, scale_deg = 110e3, device = device):

	## Connect all source-reciever pairs to their k_nearest_pairs, and those connections within max_deg_offset.
	## By using the K-nn neighbors as well as epsilon-pairs, this ensures all source nodes are at least
	## linked to some stations.

	## Note: can also make the src-src and sta-sta graphs as a combination of k-nn and epsilon-distance graphs

	if verbose == True:
		st = time.time()

	## Create "subgraph" Cartesian product graph edges
	## E.g., the "subgraph" Cartesian product is only "nodes" of pairs of sources-recievers in A_src_in_sta, rather than all pairs locs*x_grid.

	degree_of_src_nodes = degree(A_src_in_sta[1])
	cum_count_degree_of_src_nodes = np.concatenate((np.array([0]), np.cumsum(degree_of_src_nodes.cpu().detach().numpy())), axis = 0).astype('int')

	sta_ind_lists = []
	for i in range(srcs_slice.shape[0]):
		ind_list = -1*np.ones(locs.shape[0])
		ind_list[A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]].cpu().detach().numpy()] = np.arange(degree_of_src_nodes[i].item())
		sta_ind_lists.append(ind_list)
	sta_ind_lists = np.hstack(sta_ind_lists).astype('int')


	## A_src_in_prod : For each entry in A_src, need to determine which subset of A_src_in_sta is incoming.
	## E.g., which set of incoming stations for each pair of (source-station); so, all point source_1 == source_2
	## for source_1 in Source graph and source_2 in Cartesian product graph
	tree_srcs_in_prod = cKDTree(A_src_in_sta[1].cpu().detach().numpy()[:,None])
	lp_src_in_prod = tree_srcs_in_prod.query_ball_point(np.arange(srcs_slice.shape[0])[:,None], r = 0)
	A_src_in_prod = torch.Tensor(np.hstack([np.concatenate((np.array(lp_src_in_prod[j]).reshape(1,-1), j*np.ones(len(lp_src_in_prod[j])).reshape(1,-1)), axis = 0) for j in range(srcs_slice.shape[0])])).long().to(device)
	# spatial_vals = torch.Tensor(x_grid[A_src_in_sta[1]] - locs_use[A_src_in_sta[0]] ## This approach assumes all station indices are ordered
	# spatial_vals = torch.Tensor((ftrns1(x_grid[A_src_in_prod[1]]) - ftrns1(locs_use[A_src_in_sta[0][A_src_in_prod[0]]]))/110e3*scale_src_in_prod)
	spatial_vals = torch.Tensor((ftrns1(srcs_slice[A_src_in_prod[1].cpu().detach().numpy()]) - ftrns1(locs[A_src_in_sta[0][A_src_in_prod[0]].cpu().detach().numpy()]))/scale_pairwise_sta_in_src_distances).to(device)
	A_src_in_prod = Data(x = spatial_vals, edge_index = A_src_in_prod).to(device)

	tree_srcs_in_prod = cKDTree(A_src_in_sta[0].cpu().detach().numpy()[:,None])
	lp_sta_in_prod = tree_srcs_in_prod.query_ball_point(np.arange(locs.shape[0])[:,None], r = 0)
	A_sta_in_prod = torch.Tensor(np.hstack([np.concatenate((np.array(lp_sta_in_prod[j]).reshape(1,-1), j*np.ones(len(lp_sta_in_prod[j])).reshape(1,-1)), axis = 0) for j in range(locs.shape[0])])).long().to(device)

	## A_prod_sta_sta : connect any two nodes in A_src_in_sta (where these edges are distinct nodes in subgraph Cartesian product graph) 
	## if the stations are linked in A_sta_sta, and the sources are equal. 

	## Two approaches seem possible, but both have drawbacks: 
	## (i). Create the dense, full Cartesian product, with edges, then take subgraph based on retained nodes in the subgraph Cartesian product. [high memory cost]
	## (ii). Loop through each sub-segment of the subgraph Cartesian product (for fixed sources), and check which of the station pairs are linked.

	A_prod_sta_sta = []
	A_prod_src_src = []

	tree_src_in_sta = cKDTree(A_src_in_sta[0].reshape(-1,1).cpu().detach().numpy())
	lp_fixed_stas = tree_src_in_sta.query_ball_point(np.arange(locs.shape[0]).reshape(-1,1), r = 0)

	for i in range(srcs_slice.shape[0]):
		# slice_edges = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, relabel_nodes = False)[0]
		slice_edges = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, relabel_nodes = True)[0]
		A_prod_sta_sta.append(slice_edges + cum_count_degree_of_src_nodes[i])

	n_sta = len(locs)
	n_src = len(srcs_slice)
	for i in range(locs.shape[0]):
	
		if len(lp_fixed_stas[i]) == 0:
			continue

		slice_edges = subgraph(A_src_in_sta[1,np.array(lp_fixed_stas[i])], A_src_src, relabel_nodes = False, num_nodes = n_src)[0]
		# subgraph(A_src_in_sta[1,np.array(lp_fixed_stas[i])], A_src_src, relabel_nodes = False)[0]

		## This can happen when a station is only linked to one source
		if slice_edges.shape[1] == 0:
			continue

		shift_ind = sta_ind_lists[slice_edges.cpu().detach().numpy()*n_sta + i]
		assert(shift_ind.min() >= 0)
		## For each source, need to find where that station index is in the "order" of the subgraph Cartesian product
		A_prod_src_src.append(torch.Tensor(cum_count_degree_of_src_nodes[slice_edges.cpu().detach().numpy()] + shift_ind).to(device))

	## Make cartesian product graphs
	A_prod_sta_sta = torch.hstack(A_prod_sta_sta).long()
	A_prod_src_src = torch.hstack(A_prod_src_src).long()
	isort = np.lexsort((A_prod_src_src[0].cpu().detach().numpy(), A_prod_src_src[1].cpu().detach().numpy())) # Likely not actually necessary
	A_prod_src_src = A_prod_src_src[:,isort]

	return [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta] ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

def optimize_source_selection(cnt_per_source, n_total):

	## Randomize order of counts
	iperm = np.random.permutation(np.arange(len(cnt_per_source)))
	cnt_per_source_perm = cnt_per_source[iperm] ## Permuted cnts per station

	## Optimzation vector
	c = -np.copy(cnt_per_source_perm)
	A = np.ones((1,len(cnt_per_source_perm)))
	A[0,:] = cnt_per_source_perm
	b = n_total*np.ones((1,1))

	# Solve ILP
	x = cp.Variable(len(cnt_per_source_perm), integer = True)
	prob = cp.Problem(cp.Minimize(c.T@x), constraints = [A@x <= b.reshape(-1), 0 <= x, x <= 1])
	# prob.solve(scipy_options={'abstol': 100})
	prob.solve()
	soln = np.round(x.value)

	assert prob.status == 'optimal', 'competitive assignment solution is not optimal'

	src_grab = iperm[np.where(soln > 0)[0]]

	return src_grab

# use_spherical = False
if config['use_spherical'] == True:

	earth_radius = 6371e3
	ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T # just subtract mean
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device) # just subtract mean

else:

	earth_radius = 6378137.0
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean

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
	
	trv = interp_1D_velocity_model_to_3D_travel_times(X, locs_ref, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2, device = device) # .to(device)

elif config['train_travel_time_neural_network'] == True:

	n_ver_trv_time_model_load = 1
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, use_physics_informed = use_physics_informed, device = device)
	# trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', use_physics_informed = use_physics_informed, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, use_physics_informed = use_physics_informed, device = device)


if (use_differential_evolution_location == False)*(config['train_travel_time_neural_network'] == False):
	hull = ConvexHull(X)
	hull = hull.points[hull.vertices]
else:
	hull = []



interp_type = calibration_config['interp_type'] ## Type of spatial interpolation (mean, weighted, anisotropic)
k_spc_lap = calibration_config['k_spc_lap'] ## k-nn value for laplacian smoothing
k_spc_interp = calibration_config['k_spc_interp'] ## k-nn value for interpolation
grid_index = calibration_config['grid_index'] ## grid index choice (of x_grids)

n_ver_events = calibration_config['n_ver_events']
n_ver_reference = calibration_config['n_ver_reference']
n_ver_save = calibration_config['n_ver_save']
interp_type = calibration_config['interp_type'] ## Type of spatial interpolation (mean, weighted, anisotropic)
k_spc_lap = calibration_config['k_spc_lap'] ## k-nn value for laplacian smoothing
k_spc_interp = calibration_config['k_spc_interp'] ## k-nn value for interpolation
sig_ker = calibration_config['sig_ker'] ## spatial kernel (in km) of the weighting kernel (fixed for weighted, starting value for anisotropic) 
grid_index = calibration_config['grid_index'] ## grid index choice (of x_grids)
n_batch = calibration_config['n_batch'] ## 1000 earthquakes per batch
n_updates = calibration_config['n_updates'] ## Find automatic convergence criteria
# use_lap = calibration_config['use_lap'] ## laplacian penality on spatial coefficients
# use_norm = calibration_config['use_norm'] ## norm penality on spatial coefficients
# use_ker = calibration_config['use_ker'] ## laplacian penality on kernel of spatial coefficients (anisotropic case)
# lam = calibration_config['lam'] ## weighting of laplacian regularization loss
# lam1 = calibration_config['lam1'] ## weighting of norm loss
# lam2 = calibration_config['lam2'] ## weighting of laplacian loss on kernel
temporal_match =  calibration_config['temporal_match'] ## window for matched events (seconds)
spatial_match = calibration_config['spatial_match'] ## spatial distance for matched events (m)
min_picks = calibration_config['min_picks'] ## minimum number of total picks to use event in calibration
min_threshold = calibration_config['min_threshold'] ## minimum detection threshold value to use event in calibration
compute_relocations = calibration_config['compute_relocations'] ## Compute example event relocations with travel time corrections
n_relocations = calibration_config['n_relocations'] ## Number of example event relocations with travel time corrections
save_with_data = calibration_config['save_with_data'] ## Flag whether to save data with the calibration file



lat_lim = []
lon_lim = []

## Check for reference catalog for each of these days
srcs_l = []
srcs_ref_l = []
srcs_w_l = []
cnt_p = []
cnt_s = []
Matches = []
Picks_P = []
Picks_S = []
c1, c2 = 0, 0



use_radius_station_graph = False

n_seed = 30
n_neighbors = 30
max_source_pair_distance = 10e3 # 2.5e3
max_station_pair_distance = 30e3
scale_input_offsets = 100e3
k_stations = 15

# k_min_degree = 15 ## Require this many neighboring stations within max_source_pair_distance_check
# max_source_pair_distance_check = 10e3

n_ver_catalog = 1
ext_save = path_to_file
ext_save_dir = ext_save + 'DoubleDifferenceData/'
if os.path.isdir(ext_save_dir) == False:
    os.mkdir(ext_save_dir)


use_global_load = True
if use_global_load == True:

	shutil.copyfile(path_to_file + '%s_catalog_data_ver_%d.hdf5'%(name_of_project, n_ver_catalog), ext_save + '%s_catalog_data_ver_%d_copy_%d.hdf5'%(name_of_project, n_ver_catalog, int(argvs[1])))
	z = h5py.File(ext_save + '%s_catalog_data_ver_%d_copy_%d.hdf5'%(name_of_project, n_ver_catalog, int(argvs[1])), 'r')
	
	srcs = z['srcs'][:]
	Matches = z['Matches'][:]
	srcs_ref = z['srcs_ref'][:]
	Residuals = z['Residuals'][:]
	arrivals = z['arrivals'][:]
	TrvTimes_Initial = z['TrvTimes_Initial'][:]
	Phase_type = z['Phase_type'][:]
	Src_ind = z['Src_ind'][:]
	Partials = z['Partials'][:]
	Src_ind = z['Src_ind'][:]
	cnt_p = z['cnt_p'][:]
	cnt_s = z['cnt_s'][:]

	ind_pick_vec_p = z['ind_pick_vec_p'][:]
	ind_pick_vec_s = z['ind_pick_vec_s'][:]
	# z['ind_vec_p'] = ind_pick_vec_p
	# z['ind_vec_s'] = ind_pick_vec_s
	Picks_P_stack = z['Picks_P_stack'][:]
	Picks_S_stack = z['Picks_S_stack'][:]

	## Unravel indices
	lp_ind_arrival_srcs_vec = z['lp_ind_arrival_srcs_vec'][:] #  = np.hstack([j*np.ones(len(lp_ind_arrival_srcs[j])) for j in range(len(lp_ind_arrival_srcs))])
	lp_ind_arrival_srcs_stack = z['lp_ind_arrival_srcs_stack'][:] #  = np.hstack(lp_ind_arrival_srcs)	
	tree_indices = cKDTree(lp_ind_arrival_srcs_vec.reshape(-1,1))

	tree_p = cKDTree(ind_pick_vec_p.reshape(-1,1))
	tree_s = cKDTree(ind_pick_vec_s.reshape(-1,1))

	Picks_P = []
	Picks_S = []
	lp_ind_arrival_srcs = []

	n_batch_unravel = int(10e3)
	n_batches_unravel = int(np.floor(len(srcs)/n_batch_unravel))
	ind_batches_unravel = [np.arange(n_batch_unravel) + n_batch_unravel*i for i in range(np.maximum(1, n_batches_unravel))]
	if ind_batches_unravel[-1][-1] < (len(srcs) - 1):
		ind_batches_unravel.append(np.arange(ind_batches_unravel[-1][-1] + 1, len(srcs)))


	if len(ind_batches_unravel) == 1: ## If only one batch, use only all sources
		ind_batches_unravel[0] = np.arange(len(srcs))


	for n in range(len(ind_batches_unravel)):
		ip_query = tree_p.query_ball_point(ind_batches_unravel[n].reshape(-1,1), r = 0)
		is_query = tree_s.query_ball_point(ind_batches_unravel[n].reshape(-1,1), r = 0)
		ip_indices = tree_indices.query_ball_point(ind_batches_unravel[n].reshape(-1,1), r = 0)

		for j in range(len(ind_batches_unravel[n])):
			Picks_P.append(Picks_P_stack[ip_query[j]])
			Picks_S.append(Picks_S_stack[is_query[j]])
			lp_ind_arrival_srcs.append(np.array(ip_indices[j]))

		print('Unraveled %d of %d'%(n, len(ind_batches_unravel)))

	# lp_ind_arrival_srcs = cKDTree(arrivals[:,[5]]).query_ball_point(np.arange(len(srcs)).reshape(-1,1), r = 0)

	# Picks_P = [z['Picks_P_%d'%i][:] for i in range(len(srcs))]
	# Picks_S = [z['Picks_S_%d'%i][:] for i in range(len(srcs))]
	# lp_ind_arrival_srcs = [z['lp_ind_arrival_srcs_%d'%i][:] for i in range(len(srcs))]
	z.close()

	os.remove(ext_save + '%s_catalog_data_ver_%d_copy_%d.hdf5'%(name_of_project, n_ver_catalog, int(argvs[1])))


weight_depth = 1.0
weight_depth_vec = np.array([1.0, 1.0, weight_depth]).reshape(1,-1)
tree_srcs = cKDTree(ftrns1(srcs)*weight_depth_vec)

## Build fixed station radius graph

if use_radius_station_graph == True:

	tree_sta = cKDTree(ftrns1(locs)*weight_depth_vec)
	ineighbor_sta = tree_sta.query_ball_point(ftrns1(locs)*weight_depth_vec, r = max_station_pair_distance)
	A_sta_sta = np.hstack([np.concatenate((np.array(ineighbor_sta[i]).reshape(1,-1), i*np.ones((1,len(ineighbor_sta[i])))), axis = 0) for i in range(len(locs))]).astype('int')

else:

	# max_station_pair_distance
	A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs)/1000.0).to(device), torch.Tensor(ftrns1(locs)/1000.0).to(device), k = k_stations + 1).flip(0).contiguous())[0]

	remove_sta_edges_too_large = True
	if remove_sta_edges_too_large == True:
		idel = np.where(np.linalg.norm(ftrns1(locs[A_sta_sta.cpu().detach().numpy()[0]]) - ftrns1(locs[A_sta_sta.cpu().detach().numpy()[1]]), axis = 1) > max_station_pair_distance)[0]
		A_sta_sta = torch.Tensor(np.delete(A_sta_sta.cpu().detach().numpy(), idel, axis = 1)).long().to(device)


n_samples = 500
inds_save = np.arange(n_samples) + int(argvs[1])*n_samples

for n in range(n_samples):


	## Extract source graph
	nodes_all, node_types, srcs_slice, A_src_src, A_src_src_abs, ifind_edges, isample = build_source_graph(srcs, tree_srcs, n_seed, n_neighbors, max_source_pair_distance, ftrns1, prob = False, weight_depth = 1.0, use_efficient_lookup = True, add_missing_edges = False)
	imatch_srcs = tree_srcs.query(ftrns1(srcs_slice)*weight_depth_vec)[1]

	A_src_in_sta = []
	Data_extract = []
	Arrivals = []
	count_sta = []
	# Ind_arrivals = []
	for i in range(len(nodes_all)):
		# pick_p = Picks_P[nodes_all[i]]
		# pick_s = Picks_S[nodes_all[i]]
		ind_slice = np.array(lp_ind_arrival_srcs[nodes_all[i]])
		pick_slice = arrivals[ind_slice]
		ista_unique = np.unique(pick_slice[:,1]).astype('int')
		count_sta.append(len(ista_unique))
		A_src_in_sta.append(np.concatenate((ista_unique.reshape(1,-1), nodes_all[i]*np.ones((1, len(ista_unique)))), axis = 0))
		data_vec = np.zeros((len(ista_unique), 15)) ## Res_p, Res_s, Partial_p, Partial_s, src - reciever vec, ||src - reciever vec||, src_degree, mask_p, mask_s (could add absolute positions too)
		arrivals_vec = np.zeros((len(ista_unique), 6)) ## Arv_p, Arv_s, Ind_sta, Mask_p, Mask_s
		arrivals_vec[:,2] = ista_unique
		arrivals_vec[:,5] = nodes_all[i]
		data_vec[:,8:11] = (ftrns1(srcs[[nodes_all[i]]]) - ftrns1(locs[ista_unique]))/scale_input_offsets
		data_vec[:,11] = np.linalg.norm(ftrns1(srcs[[nodes_all[i]]]) - ftrns1(locs[ista_unique]), axis = 1)/scale_input_offsets
		data_vec[:,12] = np.log10(len(pick_slice))
		i1_p, i1_s = np.where(pick_slice[:,4] == 0)[0], np.where(pick_slice[:,4] == 1)[0]
		if len(i1_p) > 0:
			i1_p_match = np.searchsorted(ista_unique, pick_slice[i1_p,1].astype('int'))
			data_vec[i1_p_match,0] = Residuals[ind_slice[i1_p]]
			data_vec[i1_p_match,2:5] = Partials[ind_slice[i1_p]]
			data_vec[i1_p_match,13] = 1.0
			arrivals_vec[i1_p_match,0] = pick_slice[i1_p,0]
			arrivals_vec[i1_p_match,3] = 1.0
			# assert(pick_slice[i1_p,1] == ista_unique[i1_p_match])
		if len(i1_s) > 0:
			i1_s_match = np.searchsorted(ista_unique, pick_slice[i1_s,1].astype('int'))
			data_vec[i1_s_match,1] = Residuals[ind_slice[i1_s]]
			data_vec[i1_s_match,5:8] = Partials[ind_slice[i1_s]]
			data_vec[i1_s_match,14] = 1.0
			arrivals_vec[i1_s_match,1] = pick_slice[i1_s,0]
			arrivals_vec[i1_s_match,4] = 1.0
			# assert(pick_slice[i1_s,1] == ista_unique[i1_s_match])
		Data_extract.append(data_vec)
		Arrivals.append(arrivals_vec)

	A_src_in_sta = np.hstack(A_src_in_sta).astype('int')
	Data_extract = np.vstack(Data_extract)
	Arrivals = np.vstack(Arrivals)
	count_sta = np.hstack(count_sta)


	## Must project A_src_in_sta into the relabeled nodes
	perm_edges = -1*np.ones(len(srcs)).astype('int')
	perm_edges[nodes_all] = np.arange(len(nodes_all))
	A_src_in_sta_abs = np.copy(A_src_in_sta)
	A_src_in_sta[1] = perm_edges[A_src_in_sta[1]]

	# slice_edges = subgraph(A_src_in_sta[1,np.array(lp_fixed_stas[i])], A_src_src, relabel_nodes = False)[0]

	A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_sta_in_prod, A_src_in_sta = extract_subgraph_of_cartesian_product(locs, srcs_slice, torch.Tensor(A_sta_sta).long().to(device), torch.Tensor(A_src_src).long().to(device), torch.Tensor(A_src_in_sta).long().to(device), ftrns1, ftrns2)
	A_src_in_prod, A_src_in_prod_data = A_src_in_prod.edge_index, A_src_in_prod.x

	print('Number of nodes %d'%A_src_in_prod.shape[1])
	print('Number of prod src src edges %0.2f million'%(A_prod_src_src.shape[1]/1e6))
	print('Number of prod sta sta edges %0.2f million'%(A_prod_sta_sta.shape[1]/1e6))
	print('Number src edges %d'%A_src_src.shape[1])
	print('Number sta edges %d'%A_sta_sta.shape[1])
	print('Number of sources %d'%len(srcs_slice))


	A_src_in_sta = torch.Tensor(A_src_in_sta).to(device)


	locs_cuda = torch.Tensor(locs).to(device)
	srcs_cuda = torch.Tensor(srcs).to(device)
	locs_cart = torch.Tensor(ftrns1(locs)).to(device)
	srcs_cart = torch.Tensor(ftrns1(srcs)).to(device)
	srcs_slice_cart = torch.Tensor(ftrns1(srcs_slice)).to(device)

	Input = torch.Tensor(Data_extract).to(device)

	## Find edges that can compute losses on
	ifind_edges1 = np.where((node_types[A_src_in_sta[1][A_prod_src_src[1]].cpu().detach().numpy()] == 0)*(node_types[A_src_in_sta[1][A_prod_src_src[0]].cpu().detach().numpy()] == 1))[0] ## Pairs of sources between nodes of tier 1 and tier 2 type (of either direction)
	ifind_edges2 = np.where((node_types[A_src_in_sta[1][A_prod_src_src[1]].cpu().detach().numpy()] == 1)*(node_types[A_src_in_sta[1][A_prod_src_src[0]].cpu().detach().numpy()] == 0))[0]
	ifind_edges3 = np.where((node_types[A_src_in_sta[1][A_prod_src_src[1]].cpu().detach().numpy()] == 0)*(node_types[A_src_in_sta[1][A_prod_src_src[0]].cpu().detach().numpy()] == 0))[0]
	ifind_edges4 = np.where((node_types[A_src_in_sta[1][A_prod_src_src[1]].cpu().detach().numpy()] == 1)*(node_types[A_src_in_sta[1][A_prod_src_src[0]].cpu().detach().numpy()] == 1))[0]
	ifind_edges = np.unique(np.concatenate((ifind_edges1, ifind_edges2, ifind_edges3, ifind_edges4), axis = 0))

	n_total = int(2.5e5)
	degree_srcs = degree(A_src_in_sta[1][A_prod_src_src[1]])
	inot_zero = torch.where(degree_srcs > 0)[0]
	src_grab = optimize_source_selection(degree_srcs[inot_zero], n_total)
	tree_edges = cKDTree(inot_zero[src_grab].reshape(-1,1))
	iadd_edges = np.where(tree_edges.query(A_src_in_sta[1][A_prod_src_src[1]].cpu().detach().numpy().reshape(-1,1))[0] == 0)[0]
	assert(len(iadd_edges) == degree_srcs[inot_zero][src_grab].sum().item())

	iadd_edges1 = np.random.choice(A_prod_src_src.shape[1], size = np.minimum(int(1e5), A_prod_src_src.shape[1]), replace = False)
	# iadd_s_edges = np.random.randint(0, high = A_prod_src_src.shape[1], size = int(1e5), replace = False)

	ifind_edges = np.unique(np.concatenate((ifind_edges, iadd_edges, iadd_edges1), axis = 0))

	# ## Find subset of labels to make predictions on
	ifind_edges = torch.Tensor(ifind_edges).long().to(device)
	ifind_p_edges = torch.where((Input[A_prod_src_src[0,ifind_edges],-2] == 1)*(Input[A_prod_src_src[1,ifind_edges],-2] == 1))[0]
	ifind_s_edges = torch.where((Input[A_prod_src_src[0,ifind_edges],-1] == 1)*(Input[A_prod_src_src[1,ifind_edges],-1] == 1))[0]
	ifind_p_edges = ifind_edges[ifind_p_edges]
	ifind_s_edges = ifind_edges[ifind_s_edges]


	## Save output
	z = h5py.File(ext_save + seperator + ext_save_dir + 'location_training_sample_%d_ver_1.hdf5'%(inds_save[n]), 'w')
	z['srcs'] = srcs
	z['srcs_ref'] = srcs_ref
	z['Matches'] = Matches
	z['nodes_all'] = nodes_all # nodes_all, node_types, srcs_slice, A_src_src, A_src_src_abs, ifind_edges
	z['node_types'] = node_types
	z['srcs_slice'] = srcs_slice
	z['imatch_srcs'] = imatch_srcs
	# z['Data_extract'] = Data_extract
	z['count_sta'] = count_sta
	z['A_sta_sta'] = A_sta_sta.cpu().detach().numpy() # A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta
	z['A_src_src'] = A_src_src.cpu().detach().numpy()
	z['A_src_src_abs'] = A_src_src_abs
	z['A_prod_sta_sta'] = A_prod_sta_sta.cpu().detach().numpy()
	z['A_prod_src_src'] = A_prod_src_src.cpu().detach().numpy()
	z['A_src_in_prod'] = A_src_in_prod.cpu().detach().numpy()
	z['A_src_in_prod_data'] = A_src_in_prod_data.cpu().detach().numpy()
	z['A_sta_in_prod'] = A_sta_in_prod.cpu().detach().numpy()
	z['A_src_in_sta'] = A_src_in_sta.cpu().detach().numpy()
	z['locs_cart'] = locs_cart.cpu().detach().numpy()
	z['srcs_cart'] = srcs_cart.cpu().detach().numpy()
	z['srcs_slice_cart'] = srcs_slice_cart.cpu().detach().numpy()
	z['Input'] = Input.cpu().detach().numpy()
	z['Arrivals'] = Arrivals
	z['ifind_p_edges'] = ifind_p_edges.cpu().detach().numpy()
	z['ifind_s_edges'] = ifind_s_edges.cpu().detach().numpy()
	z['isample'] = isample
	## Should save Trgts and travel times as well
	z.close()

	print('Saved %d'%inds_save[n])
