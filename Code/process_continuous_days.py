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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.autograd import Variable
from numpy.matlib import repmat
from scipy.stats import chi2
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

from graph_utils import *
from utils import *
from module import *
from process_utils import *

## This code can be run on cuda, though
## in general, it often makes sense to run this script on seperate
## jobs/cpus for many days simulataneously (using argv[1]; 
## e.g., call "python process_continuous_days.py n" for many different n
## integers and each instance will run day t0_init + n
## sbatch or a bash script can call this file for a independent set of cpu threads
## (each for a different n, or, day).


path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

st_time = time.time()
argvs = sys.argv
if len(argvs) < 2: 
	argvs.append(0) 

if len(argvs) < 3:
	argvs.append(0)

day_select = int(argvs[1])
offset_select = int(argvs[2])

# The first system argument (after the file name; e.g., argvs[1]) is an integer used to select which
# day in the %s_process_days_list_ver_%d.txt file each call of this script will compute

# This index can also be incremented by the larger value: argvs[2]*offset_increment (defined in process_config)
# to help process very large pick lists with a combinations of using job arrays
# to increment argvs[1], and seperate sbatch scripts incrementing argvs[2]


## Add "identity" check for location - if in second pass of location, an identical set of picks is used for a given event
## don't relocate.

## Also shrink search radius for events (to within offset_ratio_quality_control*2.0 of kernel radius?); 
## though don't want this to limit ability to find outliers with this approach
## Also seed location with initial guess?
## Must evaluate location quality

## Add automated "check Calibration" or "download" reference events, and compute matched events
## Add disconnected components search to minimize cost of matched events


## Make more efficient compute_travel_times


### Settings: ###

with open('process_config.yaml', 'r') as file:
    process_config = yaml.safe_load(file)

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
spr_picks = process_config['spr_picks'] # Assumed sampling rate of picks 
use_restrict = process_config.get('use_restrict', False)

use_quality_check = process_config['use_quality_check'] ## If True, check all associated picks and set a maximum allowed relative error after obtaining initial location
max_relative_error = process_config['max_relative_error'] ## 0.15 corresponds to 15% maximum relative error allowed
min_time_buffer = process_config['min_time_buffer'] ## Uses this time (seconds) as a minimum residual time, beneath which, the relative error criterion is ignored (i.e., an associated pick is removed if both the relative error > max_relative_error and the residual > min_time_buffer)

device = torch.device(process_config['device']) ## Right now, this isn't updated to work with cuda, since
if (process_config['device'] == 'cuda')*(torch.cuda.is_available() == False):
	print('No cuda available, using cpu')
	device = torch.device('cpu')

## the necessary variables do not have .to(device) at the right places
torch.set_grad_enabled(False)

compute_magnitudes = process_config['compute_magnitudes']
min_log_amplitude_val = process_config['min_log_amplitude_val']
use_topography = process_config['use_topography']
process_known_events = process_config['process_known_events']
use_fixed_domain = process_config.get('use_fixed_domain', True)
use_offset_quality_control = process_config.get('use_offset_quality_control', True)
offset_ratio_quality_control = process_config.get('offset_ratio_quality_control', 2.0) # 3.0

## Minimum required picks and stations per event
min_required_picks = process_config['min_required_picks']
min_required_sta = process_config['min_required_sta']

use_time_shift = config['use_time_shift']

with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)


# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

k_sta_edges = config['k_sta_edges']
k_spc_edges = config['k_spc_edges']
k_time_edges = config['k_time_edges']

name_of_project = config['name_of_project']
use_physics_informed = config['use_physics_informed']
use_phase_types = config['use_phase_types']
use_subgraph = config['use_subgraph']
use_sign_input = config.get('use_sign_input', False)
use_station_corrections = config.get('use_station_corrections', False)
use_expanded = config['use_expanded']
assert(use_subgraph == True)

if use_subgraph == True:
    max_deg_offset = config['max_deg_offset']
    k_nearest_pairs = config['k_nearest_pairs']

# Load day to process
z = open(path_to_file + '%s_process_days_list_ver_%d.txt'%(name_of_project, process_days_ver), 'r')
lines = z.readlines()
z.close()
day_select_val = day_select + offset_select*offset_increment
if '/' in lines[day_select_val]:
	date = lines[day_select_val].split('/')
elif ',' in lines[day_select_val]:
	date = lines[day_select_val].split(',')
else:
	date = lines[day_select_val].split(' ')	
date = np.array([int(date[0]), int(date[1]), int(date[2])])
yr, mo, dy = date[0], date[1], date[2]

## Load primary domains spatial region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

## Check if using full Earth, set target sampling bounds ##
if (lat_range[0] <= -89.98)*(lat_range[1] >= 89.98)*(lon_range[0] <= -179.98)*(lon_range[1] >= 179.98):
	use_global = True
	lat_range_extend = [-90.0, 90.0]
	lon_range_extend = [-180.0, 180.0]
else:
	use_global = False

### Begin automated processing ###

print('\nName of program is %s'%argvs[0])
print('Beginning processing\n')

print('Date: %d %d %d'%(date[0], date[1], date[2]))
print('day is %s \n'%argvs[1])


# use_fixed_domain = True
# use_variable_domains = True
if use_fixed_domain == True:

	# Load region
	z = np.load(path_to_file + '%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	# x_grids = np.expand_dims(z['x_grid'], axis = 0)
	x_grids = z['x_grids'] # , axis = 0)
	x_grids_init = np.copy(x_grids)
	scale_time = z['scale_time']/1000.0
	z.close()

	## Need to add option for pre built graphs

	# Load stations
	z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	## Create path to write files
	write_training_file = path_to_file + 'GNN_TrainedModels/' + name_of_project + '_'

	z = np.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(n_step_load, n_ver_load))
	training_params = z['training_params']
	graph_params = z['graph_params']
	pred_params = z['pred_params']
	src_x_kernel = pred_params[3]
	src_t_kernel = pred_params[2]
	z.close()
	# pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]

	lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
	lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

	if use_expanded == True:
		Ac = np.load(path_to_file + 'Grids/%s_seismic_network_expanders_ver_%d.npz'%(name_of_project, template_ver))['Ac']
	else:
		Ac = False

	A_src_in_sta = None ## Need to define subgraph
	A_sta_sta, A_src_src = None, None


else:

	# Load stations
	z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()
		

	# Load region
	z = np.load(path_to_file + '%s_region.npz'%name_of_project)
	_, _, depth_range, _ = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	z = np.load(path_to_file + 'Picks/%d/%d_%d_%d_ver_1.npz'%(date[0], date[0], date[1], date[2]))
	# if 'date' in z.keys(): date = z['date']
	# P, locs, stas = z['P'], z['locs_use'], z['stas_use']

	P = z['P']
	if 'locs_use' in z.keys():
		locs, stas = z['locs_use'], z['stas_use']

	if 'rbest' in z.keys():
		mn, rbest = z['mn'], z['rbest']

	## Need to add option for pre built graphs
	use_subset_indices = True
	if use_subset_indices == True:
		ind_unique_ = np.unique(P[:,1]).astype('int')
		perm_vec_ = (-1*np.ones(len(locs))).astype('int')
		perm_vec_[ind_unique_] = np.arange(len(ind_unique_))
		P[:,1] = perm_vec_[P[:,1].astype('int')]
		assert(P[:,1].min() > -1)
		print('Using unique station indices %d of %d'%(len(ind_unique_), len(locs)))
		locs = locs[ind_unique_]
		stas = stas[ind_unique_]

	# if min_spc_allowed is not None: # *(use_updated_merge_stations == True)
	# 	print('Using minimum spacing stations: %0.4f'%min_spc_allowed)
	# 	P_keep, P_remove, ind_keep, locs_keep = merge_nearby_stations(P, locs, spatial_win = min_spc_allowed, merge_picks = False, merge_ratio = 0.5, use_depths = True, merge_window = 1.5, verbose = True)
	# 	P = np.copy(P_keep) ## Note: could consider removing non used stations, but would need to create graph

	# if use_phase_types == False:
	# 	P[:,4] = 0 ## No phase types

	# # ind_use = np.unique(P[:,1]).astype('int')
	# arrivals_tree = cKDTree(P[:,0][:,None])

	locs_use = np.copy(locs)
	stas_use = np.copy(stas)
	ind_use = np.arange(len(locs))
	assert((np.abs(ind_use) - np.arange(len(locs))).max() == 0)  # = np.arange(len(locs))
	z.close()
	
	# else:
	# 	# Load stations
	# 	z.close()
	# 	z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
	# 	locs, stas, _, _ = z['locs'], z['stas'], z['mn'], z['rbest']
	# 	z.close()

	earth_radius = 6378137.0
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T # just subtract mean
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device) # just subtract mean

	try:
		n_ver_domain = 1 ## or 1
		m_domain = load_model_domain(n_ver_domain, device = device)
		print('Domain parameters')
		print(m_domain)
	except:
		print('No model domain')
		m_domain = None


	## Can also apply this in travel time calculation
	apply_location_shift = True ## Add this to allow shifting the "Calibration" events as well
	if apply_location_shift == True:
		# buf_region = np.diff(lat_range)[0]*0.05
		if (locs[:,0].min() < (lat_range[0] - deg_pad)) or (locs[:,0].max() > (lat_range[1] + deg_pad)) or (locs[:,1].min() < (lon_range[0] - deg_pad)) or (locs[:,1].max() > (lon_range[1] + deg_pad)):
			print('Applying domain shift')
			locs_shifted, mn_shift, rbest_shift = generate_pseudo_lla_for_new_region(locs, mn, rbest, ftrns2)
			locs = np.copy(locs_shifted)
			locs_use = np.copy(locs_shifted)
		else:
			apply_location_shift = False
	
	
	## Add estimate of number of nodes / cartesian product size
	deg_padding = np.nan ## Use hueristic
	Vc = 3500.0 # Include this or not
	scale_domain = 1.1

	## Include this or not
	# depth_range = [-40e3, 2e3]

	# n_trgt_nodes = int(200e3)
	# number_of_spatial_nodes = 3000
	n_trgt_nodes = process_config.get('n_trgt_nodes', int(200e3))
	number_of_spatial_nodes = process_config.get('number_of_spatial_nodes', 3000)
	use_approximate_domain = process_config.get('use_approximate_domain', True)
	optimize_station_graphs = process_config.get('optimize_station_graphs', False) # False
	optimize_source_graphs = process_config.get('optimize_source_graphs', False) # False
	use_paths = process_config.get('use_paths', False) # False
	use_global = process_config.get('use_global', False) # False
	use_tuner = process_config.get('use_tuner', True) # False


	build_graphs_domain(m_domain, locs_use, stas_use, scale_domain, deg_padding, number_of_spatial_nodes, k_spc_edges, k_sta_edges, depth_range, ftrns1, ftrns2, use_global = use_global, assign_based_on_grid = False, max_nodes = number_of_spatial_nodes, n_trgt_nodes = n_trgt_nodes, Vc = Vc, rbest = rbest, mn = mn, file_index = day_select, date = date, use_paths = use_paths, optimize_station_graphs = optimize_station_graphs, optimize_source_graphs = optimize_source_graphs, use_domain_approximate = use_approximate_domain, use_tuner = use_tuner, device = device)
	z = np.load('Domains/domain_file_%d_%d_%d_%d_ver_1.npz'%(day_select, date[0], date[1], date[2]))
	# z = np.load('Domains/domain_file_%d_%d_%d_%d_ver_1.npz'%(day_select, date[0], date[1], date[2]))
	x_grids = np.expand_dims(z['x_grid'], axis = 0)

	scale_time = z['scale_time']/1000.0
	lat_range, lon_range, depth_range = z['lat_range'], z['lon_range'], z['depth_range']
	lat_range_extend, lon_range_extend, deg_pad = z['lat_range_extend'], z['lon_range_extend'], z['deg_padding']
	time_shift_range = z['time_shift_range']

	kernel_sig_t = z['sigma_input']
	src_x_kernel = z['source_label_width']
	src_depth_kernel = z['source_label_width']
	src_t_kernel = z['source_label_width_t']
	grid_choose = z['ichoose_grid']

	# A_sta_sta = torch.Tensor(np.ascontiguousarray(np.flip(z['A_sta'][0:2,:], axis = 0))).long().to(device)
	# A_src_src = torch.Tensor(np.ascontiguousarray(np.flip(z['A_src'][0:2,:], axis = 0))).long().to(device)

	A_sta_sta = torch.Tensor(z['A_sta'][0:2,:]).long().to(device)
	A_src_src = torch.Tensor(z['A_src'][0:2,:]).long().to(device)
	A_src_in_sta = torch.Tensor(z['A_src_in_sta'][0:2,:]).long().to(device)

	# A_src_in_prod = torch.Tensor(z['A_src_in_prod'][0:2,:]).long().to(device)
	if use_expanded == True:
		Ac = z['Ac'] # np.load(path_to_file + 'Grids/%s_seismic_network_expanders_ver_%d.npz'%(name_of_project, template_ver))['Ac']
	else:
		Ac = False
	z.close()

	n_resolution = 9 ## The discretization of the source time function output
	t_win = np.round(np.copy(np.array([2*src_t_kernel]))[0], 2) ## Set window size to the source kernel width (i.e., prediction window is of length +/- src_t_kernel, or [-src_t_kernel + t0, t0 + src_t_kernel])
	dt_win = np.diff(np.linspace(-t_win/2.0, t_win/2.0, n_resolution))[0]
	pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]
	write_training_file = path_to_file + 'GNN_TrainedModels/' + name_of_project + '_'

	## Note should not really use training_params or graph_params
	z = np.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(n_step_load, n_ver_load))
	training_params = z['training_params']
	graph_params = z['graph_params']
	# pred_params = z['pred_params']
	z.close()




scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)


rbest_cuda = torch.Tensor(rbest).to(device)
mn_cuda = torch.Tensor(mn).to(device)
time_shifts = x_grids[:,:,[3]]  ## Shape (n_grids, n_nodes, n_times)



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

	n_ver_trv_time_model_load = vel_model_ver
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise1 = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)


## Check if knn is working on cuda
if device.type == 'cuda' or device.type == 'cpu':
	check_len = knn(torch.rand(10,3).to(device), torch.rand(10,3).to(device), k = 5).numel()
	if check_len != 100: # If it's less than 2 * 10 * 5, there's an issue
		raise SystemError('Issue with knn on cuda for some versions of pytorch geometric and cuda')

	check_len = knn(10.0*torch.rand(200,3).to(device), 10.0*torch.rand(100,3).to(device), k = 15).numel()
	if check_len != 3000: # If it's less than 2 * 10 * 5, there's an issue
		raise SystemError('Issue with knn on cuda for some versions of pytorch geometric and cuda')

print('Compute travel times')
use_only_one_grid = process_config['use_only_one_grid']
x_grids_trv = compute_travel_times(trv, locs, x_grids, device = device)


print('Appending time shifts \n')
if use_time_shift == True:
	for i in range(len(x_grids_trv)):
		x_grids_trv[i] = x_grids_trv[i] + time_shifts[i].reshape(-1,1,1)
# print('Appending time shifts')


time_shift_range = np.max([time_shifts[j].max() - time_shifts[j].min() for j in range(len(time_shifts))])

max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))])))
min_t = float(np.floor(min([x_grids_trv[i].min() for i in range(len(x_grids_trv))]))) if use_time_shift == True else 0.0



# x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs, max_t_ = load_templates_region(trv, locs, x_grids, ftrns1, training_params, graph_params, pred_params, max_t = max_t, min_t = min_t, time_shifts = time_shifts, dt_embed = pred_params[1]/5.0, t_win = pred_params[1]*2.0, device = device) ## Note: setting time embedding vectors with respect to kernel_sig_t
## Check subsetting of grids was correct
if use_only_one_grid == True:
	assert(len(time_shifts) == 1)
	assert(len(x_grids) == 1)
	assert(len(x_grids_trv) == 1)
	if use_fixed_domain == True:
		for i in range(len(x_grids_init)):
			diff = np.linalg.norm(time_shifts[0,:] - x_grids_init[i,:,3])
			if diff == 0:
				assert(np.linalg.norm(x_grids_init[i,:,:] - x_grids[0]) == 0)
	# assert(np.abs(x_grids_trv[0] - (trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[0]).to(device)).cpu().detach().numpy() + x_grids[0][:,3].reshape(-1,1,1))).max() < 1e-2)

# assert(max_t_ == max_t)
x_grids_cart_torch = [torch.Tensor(ftrns1(x_grids[i])).to(device) for i in range(len(x_grids))]

# mz = GCN_Detection_Network_extended(ftrns1_diff, ftrns2_diff)
load_model = True
if load_model == True:

	mz_list = []
	for i in range(len(x_grids)):
		mz_slice = GCN_Detection_Network_extended(ftrns1_diff, ftrns2_diff, trv = trv, device = device).to(device)
		mz_slice.load_state_dict(torch.load(path_to_file + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(name_of_project, n_step_load, n_ver_load), map_location = device))

		if use_fixed_domain == False: # pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]
			mz_slice.set_scale_coefficients(pred_params[3]*2.0, scale_time, pred_params[1], pred_params[1]*3.0, pred_params[3], pred_params[2], time_shift_range)

		mz_slice.eval()
		mz_list.append(mz_slice)

failed = []
plot_on = False


day_len = 3600*24

use_adaptive_window = True
if use_adaptive_window == True:

	# n_resolution = 9
	frac_time_range = (2.0/3.0)
	t_win = np.round(frac_time_range*time_shift_range, 2) ## Set window size to the source kernel width (i.e., prediction window is of length +/- src_t_kernel, or [-src_t_kernel + t0, t0 + src_t_kernel])	
	n_resolution = int(5*(frac_time_range*time_shift_range)/(2*pred_params[2]))
	## Target: 5 points per +/- src_t_kernel

	# t_win = np.round(np.copy(np.array([2*pred_params[2]]))[0], 2) ## Set window size to the source kernel width (i.e., prediction window is of length +/- src_t_kernel, or [-src_t_kernel + t0, t0 + src_t_kernel])
	dt_win = np.diff(np.linspace(-t_win/2.0, t_win/2.0, n_resolution))[0]
	# assert(t_win == pred_params[0])
else:
	dt_win = 1.0 ## Default version
	t_win = 10.0


# step_size = process_config['step_size'] # 'full'
step_size = process_config['step_size']
if step_size == 'full':
	step = 1.0*t_win + 0.0
	n_overlap = 1.0
	assert(use_adaptive_window == True)
elif step_size == 'partial':
	step = (1/3.0)*t_win + 0.0
	n_overlap = 1.0 ## Check this
	assert(use_adaptive_window == True)
elif step_size == 'half':
	step = (1/2.0)*t_win + 0.0
	n_overlap = 1.0 ## Check this
	assert(use_adaptive_window == True)

# pred_params = [t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel]
tc_win = pred_params[2]*1.35 # 1.25 # process_config['tc_win'] # Temporal window (s) to link events in Local Marching
sp_win = pred_params[3]*1.35 # 1.25 #  process_config['sp_win'] # Distance (m) to link events in Local Marching
d_win = pred_params[3]*1.35/110e3 # 1.25 ## Converting km to degrees, roughly
d_win_depth = pred_params[4]*1.35 # 1.25 ## proportional to depth kernel
src_t_kernel = pred_params[2] ## temporal source kernel size

## Make topography surface
if (use_topography == True)*(os.path.isfile(path_to_file + 'Grids' + seperator + '%s_surface_elevation.npz'%name_of_project) == True):
	surface_profile = np.load(path_to_file + 'Grids' + seperator + '%s_surface_elevation.npz'%name_of_project)['surface_profile']
elif use_topography == True: ## If no surface profile saved, then interpolate a regular grid based on saved station elevations
	n_surface = 100 ## Default resolution of surface
	x1_surface, x2_surface = np.linspace(lat_range_extend[0], lat_range_extend[1], n_surface), np.linspace(lon_range_extend[0], lon_range_extend[1], n_surface)
	x11_surface, x12_surface = np.meshgrid(x1_surface, x2_surface)
	surface_profile = np.concatenate((x11_surface.reshape(-1,1), x12_surface.reshape(-1,1), np.zeros((len(x11_surface.reshape(-1)),1))), axis = 1)
	tree_sta = cKDTree(ftrns1(locs))
	surface_profile[:,2] = locs[tree_sta.query(ftrns1(surface_profile))[1],2]
	## Average the profile
	edges_surface = knn(torch.Tensor(ftrns1(surface_profile)), torch.Tensor(ftrns1(surface_profile)), k = 15).flip(0).contiguous()
	surface_profile[:,2] = scatter(torch.Tensor(surface_profile[edges_surface[0].cpu().detach().numpy(),2].reshape(-1,1)), edges_surface[1], dim = 0, reduce = 'mean').cpu().detach().numpy().reshape(-1)
else:
	surface_profile = None



# use_efficient_sampling_grid = True
# X_query = build_sampling_grid(lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, t_win/2.0, 1000.0*scale_time, 2*n_query_grid, ftrns1, ftrns2, depth_upscale_factor = 2.0, buffer_scale = 2.0)
X_query = build_sampling_grid(lat_range, lon_range, lat_range, lon_range, depth_range, t_win/2.0, 1000.0*scale_time, 3*n_query_grid, ftrns1, ftrns2, use_global = use_global, depth_upscale_factor = 2.0, buffer_scale = 2.0)
X_query_cart = torch.Tensor(ftrns1(X_query)).to(device)

## Estimate average grid spacing
tree_grid = cKDTree(np.concatenate((X_query_cart.cpu().detach().numpy(), 1000.0*scale_time*X_query[:,3].reshape(-1,1)), axis = 1))
irand_check = np.sort(np.random.choice(len(X_query_cart), size = int(0.05*len(X_query_cart)), replace = False))
dist_grid = np.median(tree_grid.query(np.concatenate((X_query_cart[irand_check].cpu().detach().numpy(), 1000.0*scale_time*X_query[irand_check,3].reshape(-1,1)), axis = 1), k = 5)[0][:,1::].mean(1))
print('Median offset in sampling grid: %0.4f'%(dist_grid))

loaded_mag_model = False
if compute_magnitudes == True:
	try:
		n_mag_ver = 1
		mags_supp = np.load(path_to_file + 'Grids' + seperator + 'trained_magnitude_model_ver_%d_supplemental.npz'%n_mag_ver)
		mag_grid, k_grid = mags_supp['mag_grid'], int(mags_supp['k_grid'])
		Mag = Magnitude(torch.Tensor(locs).to(device), torch.Tensor(mag_grid).to(device), ftrns1_diff, ftrns2_diff, k = k_grid, device = device).to(device)
		Mag.load_state_dict(torch.load(path_to_file + 'Grids' + seperator + 'trained_magnitude_model_ver_%d.h5'%n_mag_ver, map_location = device))
		loaded_mag_model = True
		print('Will compute magnitudes since a magnitude model was loaded')
	except:
		print('Will not compute magnitudes since no magnitude model was loaded')
		loaded_mag_model = False
else:
	print('Will not compute magnitudes since compute_magnitudes = False')	


check_if_finished = False
print('Should change this to use all grids, potentially')
x_grid_ind_list = np.sort(np.random.choice(len(x_grids), size = 1, replace = False)) # 15
x_grid_ind_list_1 = np.sort(np.random.choice(len(x_grids), size = len(x_grids), replace = False)) # 15

# use_only_one_grid = process_config['use_only_one_grid']
if use_only_one_grid == True:
	# x_grid_ind_list_1 = np.array([x_grid_ind_list_1[np.random.choice(len(x_grid_ind_list_l))]])
	x_grid_ind_list_1 = np.copy(x_grid_ind_list)

# assert (max([abs(len(x_grids_trv_refs[0]) - len(x_grids_trv_refs[j])) for j in range(len(x_grids_trv_refs))]) == 0)

n_scale_x_grid = len(x_grid_ind_list)
n_scale_x_grid_1 = len(x_grid_ind_list_1)
tq = torch.Tensor(np.copy(X_query[:,3])).reshape(-1,1).to(device)
# date = np.array([yr, mo, dy])



############### ############### ############### ###############
        ############### Load Picks ###############
############### ############### ############### ###############

if use_fixed_domain == True:
	P, ind_use = load_picks(path_to_file, date, spr_picks = spr_picks, n_ver = n_ver_picks)

min_spc_allowed = process_config.get('min_spc_allowed', None) ## Can remove nearby overlapping stations using min_spc_allowed
min_spc_allowed = 150.0

if min_spc_allowed is not None: # *(use_updated_merge_stations == True)
	print('Using minimum spacing stations: %0.4f'%min_spc_allowed)
	P_keep, P_remove, ind_keep, locs_keep = merge_nearby_stations(P, locs, ftrns1, spatial_win = min_spc_allowed, merge_picks = False, merge_ratio = 0.5, use_depths = True, merge_window = 1.5, verbose = True)
	P = np.copy(P_keep) ## Note: could consider removing non used stations, but would need to create graph

if use_phase_types == False:
	P[:,4] = 0 ## No phase types

arrivals_tree = cKDTree(P[:,0][:,None])	
# tree_picks = cKDTree(P[:,0:2]) # based on absolute indices

if use_fixed_domain == False: 
	# ind_use = np.arange(len(locs))
	assert(np.abs(ind_use - np.arange(len(locs))).max() == 0)

P_perm = np.copy(P)
perm_vec = -1*np.ones(locs.shape[0])
perm_vec[ind_use] = np.arange(len(ind_use))
P_perm[:,1] = perm_vec[P_perm[:,1].astype('int')]
locs_use = locs[ind_use]
stas_use = stas[ind_use]


# if use_fixed_domain == False:
# 	assert(np.abs(ind_use - np.arange(len(locs))).max() == 0)


print('\nPicks: %d, Sta: %d (%d Avg. per station)'%(len(P), len(locs_use), np.bincount(P_perm[P_perm[:,1] > -1,1].astype('int')).mean()))
print('Num %d P picks; %d S picks \n'%(len(np.where(P[:,4] == 0)[0]), len(np.where(P[:,4] == 1)[0])))



## Full time window to process
n_batch = 1 ## Rather than process full day, process window around available picks; ## Even more efficient, only around times of picks - max_moveout.
use_subset_window = True
if use_subset_window == True:
	# tsteps = np.arange(P[:,0].min(), P[:,0].max() + step, step) ## Make step any of 3 options for efficiency... (a full step, a hald step, and a fifth step?)
	tsteps = np.arange(np.maximum(0.0, P[:,0].min() - max_t), np.minimum(day_len, P[:,0].max()), step) ## Make step any of 3 options for efficiency... (a full step, a hald step, and a fifth step?)
	## Can process longer than 1 day with this approach
	# tsteps = np.arange(np.round(np.maximum(0.0, P[:,0].min() - max_t)), np.round(np.minimum(day_len, P[:,0].max())), step) ## Make step any of 3 options for efficiency... (a full step, a hald step, and a fifth step?)
else:
	tsteps = np.arange(0.0, day_len, step) ## Make step any of 3 options for efficiency... (a full step, a hald step, and a fifth step?)


## Quality control parameters

use_additional_quality_control = False
max_sigma = 1250.0 # 10e3 ## Remove events with uncertainity higher than this
tree_stas = cKDTree(ftrns1(locs))
max_perturb_offset = 50e3 ## 50 km

mag_thresh_check = 4.0
min_picks_check = 75

min_sta_count = 4
min_pick_count = 7

quantile_val = 0.9
quantile_scale_dist = 1.25
min_sta_neighbors = 15
min_neighbor_picks = 2

# n_ver_events = 8
cnt_inc = 0
cnt_remove = 0
cnt_remove_l = [0, 0, 0, 0]
cnt_isolated_picks = 0


## Input settings
use_updated_input = True
dt_embed_discretize = np.round(pred_params[1]/15.0, 2) # 0.05 ## Picks are discretized to this amount if using updated input to speed up input



############### ############### ############### ###############
       ############### Begin Processing ###############
############### ############### ############### ###############


if process_known_events == True: ## If true, only process around times of known events
	t0 = UTCDateTime(date[0], date[1], date[2])
	min_magnitude = 0.1
	srcs_known = download_catalog(lat_range, lon_range, min_magnitude, t0, t0 + 3600*24, t0 = t0, client = 'USGS')[0] # Choose client
	print('Processing %d known events'%len(srcs_known))


for cnt, strs in enumerate([0]):

	# trv_out_src = trv(torch.Tensor(locs[ind_use]).to(device), torch.Tensor(x_src_query).to(device)).detach() # .to(device)
	# locs_use_cart_torch = torch.Tensor(ftrns1(locs_use)).to(device)

	############### ############### ############### ###############
	     ############### Initial Data Checks ###############
	############### ############### ############### ###############

	A_src_in_sta_l = []
	
	for i in range(len(x_grids)):

		# x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs
		A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta = extract_inputs_adjacencies_subgraph(locs_use, x_grids[i], ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs, k_sta_edges = k_sta_edges, k_spc_edges = k_spc_edges, scale_time = scale_time, Ac = Ac, A_sta_sta = A_sta_sta, A_src_src = A_src_src, A_src_in_sta = A_src_in_sta, device = device)
		A_edges_time_p, A_edges_time_s, dt_partition = compute_time_embedding_vectors(trv_pairwise, locs_use, x_grids[i], A_src_in_sta, max_t, dt_res = pred_params[1]/5.0, t_win = pred_params[1]*2.0, min_t = min_t, time_shift = time_shifts[i], device = device)

		## Updating spatial vals to use scaled Cartesian offset distances
		# if use_time_shift == False:
		# 	spatial_vals = torch.Tensor((x_grids[i][A_src_in_prod[1].cpu().detach().numpy()][:,0:3] - locs_use[A_src_in_sta[0][A_src_in_prod[0]].cpu().detach().numpy()])/scale_x_extend).to(device)
		# else:
		# 	spatial_vals = torch.cat((torch.Tensor((x_grids[i][A_src_in_prod[1].cpu().detach().numpy()][:,0:3] - locs_use[A_src_in_sta[0][A_src_in_prod[0]].cpu().detach().numpy()])/scale_x_extend).to(device), torch.Tensor(x_grids[i][A_src_in_prod[1].cpu().detach().numpy(),3]).reshape(-1,1).to(device)/time_shift_range), dim = 1)

		if use_time_shift == False:
			spatial_vals = torch.Tensor((ftrns1(x_grids[i][A_src_in_prod[1].cpu().detach().numpy()][:,0:3]) - ftrns1(locs_use[A_src_in_sta[0][A_src_in_prod[0]].cpu().detach().numpy()]))/(30*src_x_kernel)).to(device)
		else:
			spatial_vals = torch.cat((torch.Tensor((ftrns1(x_grids[i][A_src_in_prod[1].cpu().detach().numpy()][:,0:3]) - ftrns1(locs_use[A_src_in_sta[0][A_src_in_prod[0]].cpu().detach().numpy()]))/(30*src_x_kernel)).to(device), torch.Tensor(x_grids[i][A_src_in_prod[1].cpu().detach().numpy(),3]).reshape(-1,1).to(device)/time_shift_range), dim = 1)
		
		A_src_in_prod = Data(x = spatial_vals, edge_index = A_src_in_prod)
		flipped_edge = torch.Tensor(np.ascontiguousarray(np.flip(A_src_in_prod.edge_index.cpu().detach().numpy(), axis = 0))).long().to(device)
		A_src_in_prod_flipped = Data(x = spatial_vals, edge_index = flipped_edge).to(device)
		trv_out = trv_pairwise(torch.Tensor(locs_use[A_src_in_sta[0].cpu().detach().numpy()]).to(device), torch.Tensor(x_grids[i][A_src_in_sta[1].cpu().detach().numpy()]).to(device))
		if use_time_shift == True: trv_out = trv_out + torch.Tensor(time_shifts[i]).to(device)[A_src_in_sta[1]].reshape(-1,1)

		mz_list[i].set_adjacencies(A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_prod_flipped, A_src_in_sta, A_src_src, torch.Tensor(A_edges_time_p).long().to(device), torch.Tensor(A_edges_time_s).long().to(device), torch.Tensor(dt_partition).to(device), trv_out, torch.Tensor(ftrns1(locs_use)).to(device), torch.Tensor(ftrns1(x_grids[i])).to(device))
		A_src_in_sta_l.append(A_src_in_sta.cpu().detach().numpy())


	check_overflow = True
	if (use_updated_input == True)*(check_overflow == True): ## Check if embedding correctly preserved all travel time indices (overflow can happen on GPU for very large spatial domains x number of stations when using scatter)
		## Note, must also add check that overflow doesn't happen during the second scatter operation in extract_input_from_data
		n_random_check = 5
		for i in range(n_random_check): ## n_random_check
			## Simulate picks
			src, src_origin = x_grids[0][np.random.choice(len(x_grids[0]))].reshape(1,-1), np.random.rand()*(np.nanmax(P[:,0]) - np.nanmin(P[:,0])) + np.nanmin(P[:,0])
			trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(src).to(device)).cpu().detach().numpy() + src_origin
			ikeep = np.sort(np.random.choice(len(ind_use), size = int(np.ceil(len(ind_use)*0.7)), replace = False))
			ikeep1 = np.sort(np.random.choice(len(ind_use), size = int(np.ceil(len(ind_use)*0.7)), replace = False))
			
			P1 = np.concatenate((trv_out[0,ind_use[ikeep],0].reshape(-1,1), ind_use[ikeep].reshape(-1,1), np.zeros((len(ikeep),3))), axis = 1)
			P1 = np.concatenate((P1, np.concatenate((trv_out[0,ind_use[ikeep1],1].reshape(-1,1), ind_use[ikeep1].reshape(-1,1), np.zeros((len(ikeep1),2)), np.ones((len(ikeep1),1))), axis = 1)), axis = 0)
			# if use_phase_types == False:
			# 	P1[:,4] = 0 ## No phase types
	
			x_grid_ind = x_grid_ind_list[0] ## Note: if this fails, essentially dt_embed_discretize is too small (resulting in too many time steps x number stations (combined with max moveout, max_t) leading to too large of graphs in the scatter operation for extracting inputs (e.g., ~ 100 million nodes))
			embed_p, embed_s, ind_unique_, abs_time_ref_, n_time_series_, n_sta_unique_ = extract_input_from_data(trv_pairwise, P1, np.array([src_origin]), ind_use, locs, x_grids[x_grid_ind], A_src_in_sta_l[x_grid_ind], trv_times = x_grids_trv[x_grid_ind], max_t = max_t, min_t = min_t, kernel_sig_t = pred_params[1], dt = dt_embed_discretize, use_sign_input = use_sign_input, return_embedding = True, device = device)
	
			## Check positive points
			vec_p_ = embed_p.reshape(n_sta_unique_, n_time_series_)
			vec_s_ = embed_s.reshape(n_sta_unique_, n_time_series_)
			tree_ = cKDTree(ind_unique_.reshape(-1,1))
			ip_ = tree_.query(P1[:,1].reshape(-1,1))[1] ## Matched index to unique indices
			ip1_, ip2_ = np.where(P1[:,4] == 0)[0], np.where(P1[:,4] == 1)[0]
			t_p_, t_s_ = ((P1[ip1_,0] - abs_time_ref_[0])/dt_embed_discretize).astype('int'), ((P1[ip2_,0] - abs_time_ref_[0])/dt_embed_discretize).astype('int')
			itp_, its_ = np.where((t_p_ >= 0)*(t_p_ < n_time_series_))[0], np.where((t_s_ >= 0)*(t_s_ < n_time_series_))[0]
			val_p_, val_s_ = vec_p_[ip_[ip1_[itp_]], t_p_[itp_]].cpu().detach().numpy(), vec_s_[ip_[ip2_[its_]], t_s_[its_]].cpu().detach().numpy()
			if len(val_p_) > 0: assert(val_p_.min() > 0.9)
			if len(val_s_) > 0: assert(val_s_.min() > 0.9)
			print('Min check val is %0.4f'%np.min(np.concatenate((val_p_, val_s_), axis = 0)))
	
			## Check zero points
			iselect_ = np.sort(np.random.choice(len(P1), size = 10000))
			iwhere_p_, iwhere_s_ = np.where(P1[iselect_,4] == 0)[0], np.where(P1[iselect_,4] == 1)[0]
			t_rand_p_ = P1[iselect_[iwhere_p_],0] + 4.0*pred_params[1]*np.random.choice([-1.0, 1.0], size = len(iwhere_p_))
			t_rand_s_ = P1[iselect_[iwhere_s_],0] + 4.0*pred_params[1]*np.random.choice([-1.0, 1.0], size = len(iwhere_s_))
	
			ip_1_ = tree_.query(P1[iselect_[iwhere_p_],1].reshape(-1,1))[1] ## Matched index to unique indices
			ip_2_ = tree_.query(P1[iselect_[iwhere_s_],1].reshape(-1,1))[1] ## Matched index to unique indices
			ip1_, ip2_ = np.where(P1[iselect_[iwhere_p_],4] == 0)[0], np.where(P1[iselect_[iwhere_s_],4] == 1)[0]
			t_p_, t_s_ = ((t_rand_p_[ip1_] - abs_time_ref_[0])/dt_embed_discretize).astype('int'), ((t_rand_s_[ip2_] - abs_time_ref_[0])/dt_embed_discretize).astype('int')
			itp_, its_ = np.where((t_p_ >= 0)*(t_p_ < n_time_series_))[0], np.where((t_s_ >= 0)*(t_s_ < n_time_series_))[0]
			val_p_, val_s_ = vec_p_[ip_1_[ip1_[itp_]], t_p_[itp_]].cpu().detach().numpy(), vec_s_[ip_2_[ip2_[its_]], t_s_[its_]].cpu().detach().numpy()
			if len(val_p_) > 0: assert(val_p_.max() < 0.1)
			if len(val_s_) > 0: assert(val_s_.max() < 0.1)
			print('Max check val is %0.4f \n'%np.max(np.concatenate((val_p_, val_s_), axis = 0)))
	

	if process_known_events == False: # If false, process continuous days
		times_need_l = np.copy(tsteps)
	else:  # If true, process around times of known events
		srcs_known_times = np.copy(srcs_known[:,3])
		tree_srcs_known_times = cKDTree(tsteps.reshape(-1,1))
		ip_nearest_srcs_known_times = tree_srcs_known_times.query(srcs_known_times.reshape(-1,1))[1]
		srcs_known_times = tsteps[ip_nearest_srcs_known_times]
		times_need_l = np.unique((srcs_known_times.reshape(-1,1) + np.arange(-pred_params[0]*3, pred_params[0]*3 + step, step).reshape(1,-1)).reshape(-1))
		
	## Double check this.
	n_batches = int(np.floor(len(times_need_l)/n_batch))
	times_need = [times_need_l[j*n_batch:(j + 1)*n_batch] for j in range(n_batches)]
	if n_batches*n_batch < len(times_need_l):
		times_need.append(times_need_l[n_batches*n_batch::]) ## Add last few samples

	assert(len(np.hstack(times_need)) == len(np.unique(np.hstack(times_need))))
	n_remove = 0

	skip_quiescent_intervals = True
	if skip_quiescent_intervals == True:
		min_pick_window = min_required_picks if min_required_picks != False else 1 ## Check windows with at least this many picks
		times_ind_need = []
		sc_inc = 0
		## Find time window where < min_pick_window occur on the input set, and do not process
		for i in range(len(times_need)):
			max_t_range = max_t - min_t
			lp = arrivals_tree.query_ball_point(times_need[i].reshape(-1,1) + max_t_range/2.0, r = t_win + max_t_range/2.0)
			for j in range(len(times_need[i])):
				if len(list(lp[j])) >= min_pick_window:
					times_ind_need.append(sc_inc)
				sc_inc += 1

		## Subselect times_need_l
		if len(times_ind_need) > 0:
			times_need_l = times_need_l[np.array(times_ind_need)]

			## Double check this.
			n_batches = int(np.floor(len(times_need_l)/n_batch))
			times_need = [times_need_l[j*n_batch:(j + 1)*n_batch] for j in range(n_batches)]
			if n_batches*n_batch < len(times_need_l):
				times_need.append(times_need_l[n_batches*n_batch::]) ## Add last few samples

			assert(len(np.hstack(times_need)) == len(np.unique(np.hstack(times_need))))
			print('Processing %d inputs in %d batches'%(len(times_need_l), len(times_need)))

		else:
			print('No windows with > %d picks (min_pick_window)'%min_pick_window)
			print('Stopping processing')
			continue


	     ########### ########## ########### ###########
	########### ############### ############### ###############

	############### Part 1 : Continuous Time Processing ############
	
	############### ############### ############### ############
	     ########### ########## ########### ###########


	tq_repeat = torch.clone(tq) # tq.repeat(len(X_query_cart),1)
	tq_repeat_cpu = tq_repeat.cpu().detach().numpy()
	X_query_repeat = np.copy(X_query)
	X_query_cart_repeat = torch.clone(X_query_cart) # X_query_cart.repeat_interleave(len(tq), dim = 0)		

	use_fixed_edges = True
	if use_fixed_edges == True:
		assert(len(x_grid_ind_list) == 1)
		x_grid_ind = x_grid_ind_list[0]
		mz_list[x_grid_ind].SpaceTimeAttention.set_edges(X_query_cart_repeat, torch.Tensor(ftrns1(x_grids[x_grid_ind])).to(device), tq_repeat, torch.Tensor(x_grids[x_grid_ind][:,3]).to(device))
		assert(mz_list[x_grid_ind].SpaceTimeAttention.use_fixed_edges == True)

	Out_2_sparse = []
	st_process = time.time()

	for n in range(len(times_need)):

		tsteps_slice = times_need[n]

		thresh_ratio = (2.5/3.0)
		for x_grid_ind in x_grid_ind_list:

			[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = extract_input_from_data(trv_pairwise, P, tsteps_slice, ind_use, locs, x_grids[x_grid_ind], A_src_in_sta_l[x_grid_ind], trv_times = x_grids_trv[x_grid_ind], max_t = max_t, min_t = min_t, kernel_sig_t = pred_params[1], dt = dt_embed_discretize, use_sign_input = use_sign_input, device = device)

			if use_phase_types == False:
				for i in range(len(Inpts)):
					Inpts[i][:,2::] = 0.0 ## Phase type informed features zeroed out
					Masks[i][:,2::] = 0.0
			
			for i0 in range(len(tsteps_slice)):
				if len(lp_times[i0]) == 0:
					continue ## It will fail if len(lp_times[i0]) == 0!

				out = mz_list[x_grid_ind].forward_fixed_source(Inpts[i0], Masks[i0], torch.Tensor(lp_times[i0]).to(device), torch.Tensor(lp_stations[i0]).long().to(device), torch.Tensor(lp_phases[i0].reshape(-1,1)).float().to(device), torch.Tensor(ftrns1(locs_use)).to(device), x_grids_cart_torch[x_grid_ind], torch.Tensor(x_grids[x_grid_ind][:,3].reshape(-1,1)).to(device), X_query_cart_repeat, tq_repeat)
				ifind_thresh = np.where(out[1][:,0].cpu().detach().numpy() > thresh*thresh_ratio)[0]
				if len(ifind_thresh):
					Out_2_sparse.append(np.concatenate((X_query_repeat[ifind_thresh,0:3], tq_repeat_cpu[ifind_thresh] + tsteps_slice[i0], out[1][ifind_thresh,0].reshape(-1,1).cpu().detach().numpy()), axis = 1))

				# out_cumulative_max += out[1].max().item() if (out[1].max().item() > 0.075) else 0
				if (np.mod(i0, 50) == 0) + ((np.mod(i0, 5) == 0)*(out[1].max().item() > 0.075)):
					print('%d %d %0.2f'%(n, i0, out[1].max().item()))
	
	print('Continuous processing time %0.4f'%(time.time() - st_process))

	############### ############### ############### ###############
	     ############### Local Peak Finding ###############
	############### ############### ############### ###############

	## Use disconnected components (using scaled kernel distances?) of Out_2_sparse, then iterate Local Marching on disconnected components
	Out_2_sparse = np.vstack(Out_2_sparse)
	isort = np.argsort(Out_2_sparse[:,3])
	Out_2_sparse = Out_2_sparse[isort]
	coords_norm = np.concatenate((ftrns1(Out_2_sparse[:,0:3])/sp_win, Out_2_sparse[:,[3]]/tc_win), axis = 1)

	gap_buffer = 1.1
	time_diffs = np.diff(coords_norm[:, 3])
	gap_indices = np.where(time_diffs > gap_buffer)[0] + 1
	segments = np.split(np.arange(len(coords_norm)), gap_indices)

	srcs_l = []
	cnt_marching = 0
	for segment_indices in segments:

		if len(segment_indices) == 0:
			continue
		    
		# If it's a lone point, it can't be a peak in this logic
		if len(segment_indices) == 1:
			# Optional: add logic here if you want to keep single detections
			continue

		# query_pairs finds all pairs within distance r=1.0
		# This replaces your old intersection of tree_t and tree_x
		# Build the 4D Tree
		tree = cKDTree(coords_norm[segment_indices])
		local_edges = tree.query_pairs(r=1.0, output_type='ndarray')

		## Now find disconnected components of edges
		if len(local_edges) > 0:
			# Find clusters within this temporal segment
			N_seg = len(segment_indices)
			adj = csr_matrix((np.ones(len(local_edges)), (local_edges[:, 0], local_edges[:, 1])), shape=(N_seg, N_seg))
			n_comps, labels = connected_components(adj, directed=False)
    
			# Iterate through the clusters found in this segment
			for i in range(n_comps):

				refined = []
				comp_mask = (labels == i)
				if np.sum(comp_mask) < 3: # Noise filter
					continue
            
				# Extract sub-cloud and march
				scale_depth_clustering = 0.2
				sub_srcs = Out_2_sparse[segment_indices[comp_mask]]
				mp = LocalMarching(device = device) ## Check n_steps_max = 2
				refined = mp(sub_srcs, ftrns1, tc_win = tc_win, sp_win = sp_win, scale_depth = scale_depth_clustering, n_steps_max = 2, use_directed = False)
				cnt_marching += 1

				if len(refined) > 0:
					srcs_l.append(refined)


	srcs = np.vstack(srcs_l)
	if len(srcs) == 0:
		print('No sources detected, finishing script')
		continue ## No sources, continue

	print('Detected %d number of initial local maxima (%d distinct Local Marchings)'%(srcs.shape[0], cnt_marching))
	srcs = srcs[np.argsort(srcs[:,3])]

	## Set fixed edges to false
	if use_fixed_edges == True:
		assert(len(x_grid_ind_list) == 1)
		mz_list[x_grid_ind_list[0]].SpaceTimeAttention.use_fixed_edges = False
		# x_grid_ind = x_grid_ind_list[0]


	############### ############### ############### ###############
       #########       #########       #########       #########
	############### ############### ############### ###############


	trv_out_srcs = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs[:,0:3]).to(device)).cpu().detach() # .cpu().detach().numpy() # + srcs[:,3].reshape(-1,1,1)
	trv_out_srcs_init1 = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs[:,0:3]).to(device)).cpu().detach().numpy() + srcs[:,3].reshape(-1,1,1) # .cpu().detach().numpy() # + srcs[:,3].reshape(-1,1,1)
	print('Number sources (after first local marching): %d'%len(srcs))
	

	     ########### ########## ########### ###########
	########### ############### ############### ###############

	############### Part 2 : Sources Refined Query ###############
	
	############### ############### ############### ############
	     ########### ########## ########### ###########


	## Find latitude range of events
	target_width = sp_win/2.0 # /2.0 # 2 * grid_win
	lat_range_events = np.arange(srcs[:,0].min(), srcs[:,0].max() + np.diff(lat_range_extend)/5.0, np.diff(lat_range_extend)/5.0)
	lat_deg_span = target_width / (np.deg2rad(1) * earth_radius)


	st_process = time.time()
	X_query_grid = []
	srcs_refined_l = []
	trv_out_srcs_l = []
	Out_p_save_l = []
	Out_s_save_l = []
	Save_picks = [] # save all picks..
	lp_meta_l = []



	## May need to adapt the scale_time for different density of nodes
	## Why print statements in build_sampling_grid
	for inc, lat_val in enumerate(lat_range_events):
		# 2. Calculate the lon degrees needed to cover target_width
		# We use the inverse of the longitudinal distance formula
		# Delta_Lon = Width / (deg_to_rad * R * cos(lat))
		## For each lat value, determine typical lon range to span the source label kernel width
		lat_rad = np.radians(lat_val)
		lon_deg_span = target_width / (np.deg2rad(1) * earth_radius * np.cos(lat_rad))
		lat_range_slice = np.array([lat_val - lat_deg_span/2.0, lat_val + lat_deg_span/2.0])
		lon_range_slice = np.array([np.mean(lon_range) - lon_deg_span/2.0, np.mean(lon_range) + lon_deg_span/2.0])
		X_query_slice = build_sampling_grid(lat_range_slice, lon_range_slice, lat_range_slice, lon_range_slice, [np.mean(depth_range) - target_width/2.0, np.mean(depth_range) + target_width/2.0], tc_win/2.0, 1000.0*scale_time, 2*n_query_grid, ftrns1, ftrns2, verbose = False if inc > 0 else True, use_global = use_global, depth_upscale_factor = 2.0, buffer_scale = 2.0)
		X_query_grid.append(X_query_slice)


	print('Begin sources refined')

	tree_lats = cKDTree(lat_range_events.reshape(-1,1))
	for n in range(len(srcs)):

		inearest = tree_lats.query(srcs[n,0].reshape(1,1))[1][0] # np.argmin(np.abs())
		X_query_val = np.copy(X_query_grid[inearest]) # [:,0:3]
		X_query_val[:,0:3] = X_query_grid[inearest][:,0:3] - X_query_grid[inearest][:,0:3].mean(0, keepdims = True) + srcs[n,0:3].reshape(1,-1)
		inside = np.where((X_query_val[:,0] > lat_range[0])*(X_query_val[:,0] < lat_range[1])*(X_query_val[:,1] > lon_range[0])*(X_query_val[:,1] < lon_range[1])*(X_query_val[:,2] > depth_range[0])*(X_query_val[:,2] < depth_range[1]))[0]
		if len(inside) == 0:
			inside = np.where((X_query_val[:,0] > lat_range_extend[0])*(X_query_val[:,0] < lat_range_extend[1])*(X_query_val[:,1] > lon_range_extend[0])*(X_query_val[:,1] < lon_range_extend[1])*(X_query_val[:,2] > depth_range[0])*(X_query_val[:,2] < depth_range[1]))[0]
		X_query_val = X_query_val[inside]
		X_query_cart_val = torch.Tensor(ftrns1(X_query_val)).to(device)

		## Extract inputs
		out_vals = np.zeros(len(X_query_val))
		for inc, x_grid_ind in enumerate(x_grid_ind_list_1):

			[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = extract_input_from_data(trv_pairwise, P, srcs[[n],3], ind_use, locs, x_grids[x_grid_ind], A_src_in_sta_l[x_grid_ind], trv_times = x_grids_trv[x_grid_ind], max_t = max_t, min_t = min_t, kernel_sig_t = pred_params[1], dt = dt_embed_discretize, use_sign_input = use_sign_input, device = device)
			assert(len(Inpts) == 1)

			if use_phase_types == False: ## Does this check lp_phases correctly?
				for i in range(len(Inpts)):
					Inpts[i][:,2::] = 0.0 ## Phase type informed features zeroed out
					Masks[i][:,2::] = 0.0

			out = mz_list[x_grid_ind].forward_fixed_source(Inpts[0], Masks[0], torch.Tensor(lp_times[0]).to(device), torch.Tensor(lp_stations[0]).long().to(device), torch.Tensor(lp_phases[0].reshape(-1,1)).float().to(device), torch.Tensor(ftrns1(locs_use)).to(device), x_grids_cart_torch[x_grid_ind], torch.Tensor(x_grids[x_grid_ind][:,3].reshape(-1,1)).to(device), X_query_cart_val, torch.Tensor(X_query_val[:,[3]]).to(device)) # n_reshape = len(tq_search)
			out_vals += out[1].reshape(-1).cpu().detach().numpy()/len(x_grid_ind_list_1)

		max_val, iargmax = out_vals.max(), np.argmax(out_vals)
		if (max_val >= thresh)*(len(lp_times[0]) > 0): ## Save local maxima values
			src_max = np.copy(X_query_val[iargmax]).reshape(1,-1)
			src_max[0,3] += srcs[n,3]
			srcs_refined_l.append(np.concatenate((src_max, np.array([max_val]).reshape(1,1)), axis = 1))
			trv_out_srcs_slice = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_refined_l[-1].reshape(1,-1)).to(device)).detach() # .cpu().detach().numpy() # + srcs[:,3].reshape(-1,1,1)		
			trv_out_srcs_l.append(trv_out_srcs_slice.cpu())

			# X_save[:,2] = src_max[i,2]
			X_save = np.copy(src_max)
			X_save_cart = torch.Tensor(ftrns1(X_save)).to(device)

			for inc, x_grid_ind in enumerate(x_grid_ind_list_1):

				## For these cases, re-extract inputs and compute the association predictions
				[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = extract_input_from_data(trv_pairwise, P, src_max[[0],3], ind_use, locs, x_grids[x_grid_ind], A_src_in_sta_l[x_grid_ind], trv_times = x_grids_trv[x_grid_ind], max_t = max_t, min_t = min_t, kernel_sig_t = pred_params[1], dt = dt_embed_discretize, use_sign_input = use_sign_input, device = device)
				ipick, tpick = lp_stations[0].astype('int'), lp_times[0]
				assert(len(Inpts) == 1)

				if use_phase_types == False:
					for i in range(len(Inpts)):
						Inpts[i][:,2::] = 0.0 ## Phase type informed features zeroed out
						Masks[i][:,2::] = 0.0
			
				if inc == 0:
					Out_p_save = [np.zeros(len(lp_times[j])) for j in range(len(Inpts))]
					Out_s_save = [np.zeros(len(lp_times[j])) for j in range(len(Inpts))]
					Save_picks.append(np.concatenate((tpick.reshape(-1,1), ipick.reshape(-1,1)), axis = 1))
					lp_meta_l.append(lp_meta[0])

				# out = mz_list[x_grid_ind].forward_fixed(torch.Tensor(Inpts[i]).to(device), torch.Tensor(Masks[i]).to(device), torch.Tensor(lp_times[i]).to(device), torch.Tensor(lp_stations[i]).long().to(device), torch.Tensor(lp_phases[i].reshape(-1,1)).long().to(device), torch.Tensor(ftrns1(locs_use)).to(device), x_grids_cart_torch[x_grid_ind], torch.Tensor(x_grids[x_grid_ind][:,3].reshape(-1,1)).to(device), X_save_cart, torch.Tensor(ftrns1(srcs_refined[i,0:3].reshape(1,-1))).to(device), tq, torch.zeros(1).to(device), trv_out_srcs_slice[[i],:,:])
				out = mz_list[x_grid_ind].forward_fixed(Inpts[0], Masks[0], torch.Tensor(lp_times[0]).to(device), torch.Tensor(lp_stations[0]).long().to(device), torch.Tensor(lp_phases[0].reshape(-1,1)).long().to(device), torch.Tensor(ftrns1(locs_use)).to(device), x_grids_cart_torch[x_grid_ind], torch.Tensor(x_grids[x_grid_ind][:,3].reshape(-1,1)).to(device), X_save_cart, torch.Tensor(ftrns1(src_max[0,0:3].reshape(1,-1))).to(device), torch.Tensor([src_max[0,3]]).reshape(1,1).to(device), torch.zeros(1).to(device), trv_out_srcs_slice[[0],:,:])

				# Out_save[i,:,:] += out[1][:,:,0].cpu().detach().numpy()/n_scale_x_grid_1
				Out_p_save[0] += out[2][0,:,0].cpu().detach().numpy()/n_scale_x_grid_1
				Out_s_save[0] += out[3][0,:,0].cpu().detach().numpy()/n_scale_x_grid_1

			for i in range(len(Inpts)):
				Out_p_save_l.append(Out_p_save[i])
				Out_s_save_l.append(Out_s_save[i])

		# print('Located %d (refined): %0.3f, %0.3f, %0.3f, %0.3f, %0.3f (%0.3f)'%(n, src_max[0,0], src_max[0,1], src_max[0,2], src_max[0,3], max_val, srcs[n,4]))

	srcs_refined = np.vstack(srcs_refined_l)
	iarg_sort = np.argsort(srcs_refined[:,3])
	srcs_refined = srcs_refined[iarg_sort]
	trv_out_srcs_l = [trv_out_srcs_l[j] for j in iarg_sort]
	Out_p_save_l = [Out_p_save_l[j] for j in iarg_sort]
	Out_s_save_l = [Out_s_save_l[j] for j in iarg_sort]
	lp_meta_l = [lp_meta_l[j] for j in iarg_sort]
	Save_picks = [Save_picks[j] for j in iarg_sort]


	############### ############### ############### ###############
	     ############### Remove Merged ###############
	############### ############### ############### ###############

	mp = LocalMarching(device = device)
	srcs_refined_1 = mp(srcs_refined, ftrns1, tc_win = tc_win, sp_win = sp_win, scale_depth = scale_depth_clustering, n_steps_max = 2, use_directed = False) # tc_win = 2*dt_win, sp_win = 2*dist_offset, scale_depth = scale_depth_clustering, use_directed = False, n_steps_max = 5


	## Rather than this matching, use bipartite assignment (however this can have memory issues)
	tree_refined = cKDTree(np.concatenate((ftrns1(srcs_refined), scale_time*srcs_refined[:,[3]]), axis = 1))
	ip_retained = tree_refined.query(np.concatenate((ftrns1(srcs_refined_1), scale_time*srcs_refined_1[:,[3]]), axis = 1))[1]
	ip_retained = np.unique(ip_retained)

	# tree_refined = cKDTree(ftrns1(srcs_refined))
	# ip_retained = tree_refined.query(ftrns1(srcs_refined_1))[1]

	Out_p_save = [Out_p_save_l[i] for i in ip_retained]
	Out_s_save = [Out_s_save_l[i] for i in ip_retained]
	lp_meta_l = [lp_meta_l[i] for i in ip_retained]
	Save_picks = [Save_picks[i] for i in ip_retained]
	srcs_refined = srcs_refined[ip_retained]
	

	# st_time = time.time()
	print('Begin competetive assignment')
	iargsort = np.argsort(srcs_refined[:,3])
	srcs_refined = srcs_refined[iargsort]
	# trv_out_srcs = trv_out_srcs[iargsort]
	Out_p_save = [Out_p_save[i] for i in iargsort]
	Out_s_save = [Out_s_save[i] for i in iargsort]
	Save_picks = [Save_picks[i] for i in iargsort]
	lp_meta = [lp_meta_l[i] for i in iargsort]

	trv_out_srcs = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_refined[:,0:3]).to(device)).cpu().detach()
	trv_out_srcs_init2 = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_refined[:,0:3]).to(device)).cpu().detach().numpy() + srcs_refined[:,3].reshape(-1,1,1)
	print('Number sources (after sources refined and second local marching): %d (Time %0.4f)'%(len(srcs_refined), (time.time() - st_process)))
	# print('Continuous processing time %0.4f'%(time.time() - st_process))

	     ########### ########## ########### ###########
	########### ############### ############### ###############

	    ########### Part 3 : Competitive Assignment ##########
	
	############### ############### ############### ############
	     ########### ########## ########### ###########

	n_skipped = 0
	cnt_false = 0
	cnt_false1 = 0


	st_process = time.time()
	repeat_iters = 2 ## Repeat competitive assignment to allow recovering picks assigned to removed events
	for inc_repeat in range(repeat_iters):

		if inc_repeat == 0:

			srcs_refined_init = np.copy(srcs_refined)
			ind_srcs_retain = []

		else:

			## Considering suppressing associations from "del_arv_p" and "del_arv_s" of first pass (e.g., quality control picks)
			## however, there is slight chance that after adding new picks to an event, the previous "bad" picks are no longer bad
			## So, can identify events that have new picks after competitive assignment, and for these do not suppress the poor quality
			## picks; but for events that have no new picks, then suppress the previously identified bad picks. This should allow those
			## events to only undergo one location per event

			# ind_not_used
			ind_srcs_not_used = np.unique(np.concatenate((np.delete(np.arange(len(srcs_refined_init)), ind_srcs_retain, axis = 0), ind_srcs_retain[iremove]), axis = 0)).astype('int')
			ind_srcs_used = np.delete(np.arange(len(srcs_refined_init)), ind_srcs_not_used, axis = 0)

			ind_srcs_used = ind_srcs_retain[ikeep]


			print('Initial pick counts')
			print('Mean cnt_p: %0.4f'%(cnt_p.mean()))
			print('Mean cnt_s: %0.4f'%(cnt_s.mean()))

			print('Begin competetive assignment (second iteration)')
			# iargsort = np.argsort(srcs_refined[:,3])
			srcs_refined = np.copy(srcs_refined_init[ind_srcs_used]) # srcs_refined[iargsort]
			trv_out_srcs = trv_out_srcs[ind_srcs_used]
			Out_p_save = [Out_p_save[i] for i in ind_srcs_used]
			Out_s_save = [Out_s_save[i] for i in ind_srcs_used]
			Save_picks = [Save_picks[i] for i in ind_srcs_used]
			# lp_meta = [lp_meta_l[i] for i in ind_srcs_used]
			lp_meta = [lp_meta[i] for i in ind_srcs_used]
			
			srcs_refined_init = np.copy(srcs_refined)
			ind_srcs_retain = []


		use_expanded_competitive_assignment = True
		if use_expanded_competitive_assignment == True:

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
			unique_picks = np.unique(all_picks, axis = 0)

			# ip_sort_unique = np.lexsort((unique_picks[:,0], unique_picks[:,1])) # sort by station
			ip_sort_unique = np.lexsort((unique_picks[:,1], unique_picks[:,0])) # sort by time
			unique_picks = unique_picks[ip_sort_unique]
			len_unique_picks = len(unique_picks)

			tree_picks_unique_select = cKDTree(unique_picks[:,0:2])

			matched_src_arrival_indices = []
			matched_src_arrival_indices_p = []
			matched_src_arrival_indices_s = []

			# min_picks = 4

			for i in range(len(lp_meta)):

				if len(lp_meta[i]) == 0:
					continue

				matched_arv_indices_val = tree_picks_unique_select.query(lp_meta[i][:,0:2])
				assert(matched_arv_indices_val[0].max() == 0)
				matched_arv_indices = matched_arv_indices_val[1]

				ifind_p = np.where(Out_p_save[i] > thresh_assoc)[0]
				ifind_s = np.where(Out_s_save[i] > thresh_assoc)[0]
				assert(len(Out_p_save[i]) == len(Out_s_save[i]))

				# Check for minimum number of picks, otherwise, skip source
				if ((len(ifind_p) + len(ifind_s)) >= min_required_picks)*(len(set(lp_meta[i][ifind_p,1].astype('int')).union(lp_meta[i][ifind_s,1].astype('int'))) >= min_required_sta):

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

			## From this, we may not have memory issues with competitive assignment. If so,
			## can still reduce the size of disjoint groups.

			if len(matched_src_arrival_indices) == 0: 
				print('No sources detected')
				continue ## Skipping rest of day
			matched_src_arrival_indices = np.hstack(matched_src_arrival_indices)
			matched_src_arrival_indices_p = np.hstack(matched_src_arrival_indices_p)
			matched_src_arrival_indices_s = np.hstack(matched_src_arrival_indices_s)

			## Convert to linear graph, find disconected components, apply CA

			w_edges = np.concatenate((matched_src_arrival_indices[0,:][None,:], matched_src_arrival_indices[1,:][None,:] + len_unique_picks, matched_src_arrival_indices[2,:].reshape(1,-1)), axis = 0)
			wp_edges = np.concatenate((matched_src_arrival_indices_p[0,:][None,:], matched_src_arrival_indices_p[1,:][None,:] + len_unique_picks, matched_src_arrival_indices_p[2,:].reshape(1,-1)), axis = 0)
			ws_edges = np.concatenate((matched_src_arrival_indices_s[0,:][None,:], matched_src_arrival_indices_s[1,:][None,:] + len_unique_picks, matched_src_arrival_indices_s[2,:].reshape(1,-1)), axis = 0)
			assert(np.abs(wp_edges[0:2,:] - ws_edges[0:2,:]).max() == 0)

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

			finish_splits = False
			max_sources = 15 ## per competitive assignment run
			max_splits = 30
			num_splits = 0
			while finish_splits == False:

				remove_edges_from = []

				discon_components = list(nx.connected_components(G_nx))
				discon_components = [np.sort(np.array(list(discon_components[i])).astype('int')) for i in range(len(discon_components))]

				## Should the below line really use a where function? It seems like this is a "where" on a scalar velue everytime, so it is guarenteed to evaluate as 1
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

					elif len(ifind_src_inds) > max_sources:

						## Create a source-source index graph, based on how much they "share" arrivals. Then find min-cut on this graph,
						## to seperate sources. Modify the discon_components so the sources are split.

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
						
						w_src_adj = np.zeros((len(ifind_src_inds), len(ifind_src_inds)))

						for j in range(len(ifind_src_inds)):
							for k in range(len(ifind_src_inds)):
								if j == k:
									continue
								if (len(lp_src_ind[j]) > 0)*(len(lp_src_ind[k]) > 0):
									w_src_adj[j,k] = len(list(set(iarv[lp_src_ind[j]]).intersection(iarv[lp_src_ind[k]])))

						## Simply split sources into groups of two (need to make sure this rarely cuts off indidual sources)
						clusters = SpectralClustering(n_clusters = 2, affinity = 'precomputed').fit_predict(w_src_adj)

						i1, i2 = np.where(clusters == 0)[0], np.where(clusters == 1)[0]

						## Optimize all (union) of picks between split sources, so can determine which edges (between arrivals and sources) to delete
						## This should `trim' the source-arrival graphs and increase amount of disconnected components.

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
						else:
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
				if (len(ipick) == 0) or (len(arv_src_slice) == 0):
					continue

				# thresh_assoc = 0.125
				wp_slice[wp_slice <= thresh_assoc] = 0.0
				ws_slice[ws_slice <= thresh_assoc] = 0.0
				wp_slice_init = np.copy(wp_slice)
				ws_slice_init = np.copy(ws_slice)
				

				use_modified_weights = True
				if use_modified_weights == True:
					scale_weights = 0.2
					wp_slice[wp_slice > 0] = wp_slice[wp_slice > 0]*scale_weights + 1.0
					ws_slice[ws_slice > 0] = ws_slice[ws_slice > 0]*scale_weights + 1.0
					cost_value = 1.0*min_required_picks


				restrict = []
				if len(wp_slice) > 1:
					thresh_restrict = 0.5
					for jj in range(len(wp_slice)):
						ifind_p = np.where(wp_slice[jj,:] > 0)[0]
						ifind_s = np.where(ws_slice[jj,:] > 0)[0]
						vec_w = np.concatenate((wp_slice[jj,ifind_p], ws_slice[jj,ifind_s]), axis = 0)
						norm_vec_w = np.linalg.norm(vec_w)
						for kk in range(len(wp_slice)):
							if (jj == kk) or (norm_vec_w == 0): continue
							vec_w1 = np.concatenate((wp_slice[kk,ifind_p], ws_slice[kk,ifind_s]), axis = 0)
							norm_vec_w1 = np.linalg.norm(vec_w1)
							if norm_vec_w1 == 0: continue
							pairwise_similarity = np.dot(vec_w, vec_w1)/(norm_vec_w*norm_vec_w1)
							if pairwise_similarity > thresh_restrict:
								restrict.append(np.array([jj, kk]))


				use_restrict_trv_times = True
				if use_restrict_trv_times == True:
					trv_out_srcs_pairs = trv(torch.Tensor(locs).to(device), torch.Tensor(srcs_refined[arv_src_slice][:,0:3]).to(device)).cpu().detach().numpy() + srcs_refined[arv_src_slice][:,3].reshape(-1,1,1)
					for jj in range(len(wp_slice)):
						for kk in range(len(ws_slice)):

							if (jj == kk): continue

							ifind_p = np.unique(np.concatenate((np.where(wp_slice[jj,:] > 0)[0], np.where(wp_slice[kk,:] > 0)[0]), axis = 0), axis = 0)
							ifind_s = np.unique(np.concatenate((np.where(ws_slice[jj,:] > 0)[0], np.where(ws_slice[kk,:] > 0)[0]), axis = 0), axis = 0)

							ifind_p = unique_picks[arv_ind_slice][ifind_p,1].astype('int')
							ifind_s = unique_picks[arv_ind_slice][ifind_s,1].astype('int')						
							
							trv_vec1 = np.concatenate((trv_out_srcs_pairs[jj,ifind_p,0], trv_out_srcs_pairs[jj,ifind_s,1]), axis = 0)
							trv_vec2 = np.concatenate((trv_out_srcs_pairs[kk,ifind_p,0], trv_out_srcs_pairs[kk,ifind_s,1]), axis = 0)

							rms_residual = np.linalg.norm(trv_vec1 - trv_vec2)/np.sqrt(len(trv_vec1))

							# if rms_residual < 2.0*kernel_sig_t:
							if rms_residual < 1.5*kernel_sig_t:
								restrict.append(np.array([jj, kk]))	

				if len(restrict) == 0: restrict = None

				if use_restrict == True:
					assignments1, srcs_active1 = competitive_assignment([wp_slice, ws_slice], ipick, cost_value) ## force 1 source?
					assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, cost_value, restrict = restrict) ## force 1 source?
				else:
					assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, cost_value) ## force 1 source?
					assignments1, srcs_active1 = competitive_assignment([wp_slice, ws_slice], ipick, cost_value, restrict = restrict) ## force 1 source?

				n_remove += (len(srcs_active) - len(srcs_active1)) #  if inc_repeat == (repeat_iters - 1) else 0

				# assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, 1.5, force_n_sources = 1) ## force 1 source?
				# assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, 1.5, force_n_sources = 1) ## force 1 source?
				# assignments, srcs_active = competitive_assignment([wp_slice, ws_slice], ipick, cost_value) ## force 1 source?

				if len(srcs_active) > 0:
					for j in range(len(srcs_active)):
						srcs_retained.append(srcs_refined[arv_src_slice[srcs_active[j]]].reshape(1,-1))

						wp_val = wp_slice_init[srcs_active[j], assignments[j][0]]
						ws_val = ws_slice_init[srcs_active[j], assignments[j][1]]
						
						p_assign = np.concatenate((unique_picks[arv_ind_slice[assignments[j][0]],:], cnt_src*np.ones(len(assignments[j][0])).reshape(-1,1), wp_val.reshape(-1,1)), axis = 1) ## Note: could concatenate ip_picks, if desired here, so all picks in Picks_P lists know the index of the absolute pick index.
						s_assign = np.concatenate((unique_picks[arv_ind_slice[assignments[j][1]],:], cnt_src*np.ones(len(assignments[j][1])).reshape(-1,1), ws_val.reshape(-1,1)), axis = 1)
						p_assign_perm = np.copy(p_assign)
						s_assign_perm = np.copy(s_assign)
						p_assign_perm[:,1] = perm_vec[p_assign_perm[:,1].astype('int')]
						s_assign_perm[:,1] = perm_vec[s_assign_perm[:,1].astype('int')]
						Picks_P.append(p_assign)
						Picks_S.append(s_assign)
						Picks_P_perm.append(p_assign_perm)
						Picks_S_perm.append(s_assign_perm)
						ind_srcs_retain.append(arv_src_slice[srcs_active[j]])

						cnt_src += 1

				print('%d : %d of %d'%(i, len(srcs_active), len(arv_src_slice)))

			if len(srcs_retained) == 0:
				print('No events left after competitive assignment (e.g., cost is too high w.r.t. amount of available picks)')
				continue
			
			srcs_refined = np.vstack(srcs_retained)
			ind_srcs_retain = np.hstack(ind_srcs_retain)

			## Find unique set of arrival indices, write to subset of matrix weights
			## for wp and ws.

			## Then solve CA. Need to scale weights so that: (i). Primarily, the cost is related to the number
			## of picks per event, and (ii). It still identifies "good" fit and "bad" fit source-arrival pairs,
			## based on the source-arrival weights.

		print('Number sources (after competitive assignment): %d (Time %0.4f)'%(len(srcs_refined), time.time() - st_process))


		     ########### ########## ########### ###########
		########### ############### ############### ###############

		    ########### Part 4 : Location ##########
		
		############### ############### ############### ############
		     ########### ########## ########### ###########

		## Add quality check on location offset w.r.t. true source
		## Also add search window around initial source
		## Also suppress display
		## Check mutation amount
		## Check population amount

		st_process = time.time()
		srcs_trv, srcs_sigma = [], []
		del_arv_p, del_arv_s = [], []
		iwhere_cnts = np.zeros(0)
		torch.set_grad_enabled(False)


		use_overwrite_locations = True
		if (inc_repeat == (repeat_iters - 1))*(inc_repeat > 0):

			## Check if any current pick sets are same as a previously located event
			len_p_picks = np.array([len(Picks_P[j]) for j in range(len(Picks_P))])
			len_s_picks = np.array([len(Picks_S[j]) for j in range(len(Picks_S))])
			id_picks = [np.concatenate((Picks_P[j][np.argsort(Picks_P[j][:,0]),0:2], Picks_S[j][np.argsort(Picks_S[j][:,0]),0:2]), axis = 0) for j in range(len(Picks_P))]

			tree_cnts = cKDTree(np.concatenate((len_p_picks_.reshape(-1,1), len_s_picks_.reshape(-1,1)), axis = 1))
			query_cnts = tree_cnts.query_ball_point(np.concatenate((len_p_picks.reshape(-1,1), len_s_picks.reshape(-1,1)), axis = 1), r = 0)
			iwhere_cnts = np.where([len(query_cnts[j]) > 0 for j in range(len(Picks_P))])[0] # np.where([np.abs(id_picks[])])
			iwhere_cnts = iwhere_cnts[np.where([np.abs(np.concatenate([np.expand_dims(id_picks_[j], axis = 0) for j in query_cnts[k]], axis = 0) - np.expand_dims(id_picks[k], axis = 0)).max(2).max(1).min(0) < 1e-2 for k in iwhere_cnts])[0]]

			## Now for these sources find the location of matched previous events
			imatched_ind = np.array([query_cnts[k][np.where(np.abs(np.concatenate([np.expand_dims(id_picks_[j], axis = 0) for j in query_cnts[k]], axis = 0) - np.expand_dims(id_picks[k], axis = 0)).max(2).max(1) < 1e-2)[0][0]] for k in iwhere_cnts])
			src_matched = np.copy(src_location_[imatched_ind])
			# iwhere_cnts = np.zeros(0)


		for i in range(srcs_refined.shape[0]):

			arv_p, ind_p, arv_s, ind_s = np.copy(Picks_P_perm[i][:,0]), np.copy(Picks_P_perm[i][:,1].astype('int')), np.copy(Picks_S_perm[i][:,0]), np.copy(Picks_S_perm[i][:,1].astype('int'))
			ind_unique_arrivals = np.sort(np.unique(np.concatenate((ind_p, ind_s), axis = 0)).astype('int'))

			if len(ind_unique_arrivals) == 0:
				srcs_trv.append(np.nan*np.ones((1, 4)))
				del_arv_p.append(0)
				del_arv_s.append(0)
				continue			
			

			perm_vec_arrivals = -1*np.ones(locs_use.shape[0]).astype('int')
			perm_vec_arrivals[ind_unique_arrivals] = np.arange(len(ind_unique_arrivals))
			locs_use_slice = locs_use[ind_unique_arrivals]
			ind_p_perm_slice = perm_vec_arrivals[ind_p]
			ind_s_perm_slice = perm_vec_arrivals[ind_s]
			if len(ind_p_perm_slice) > 0:
				assert(ind_p_perm_slice.min() > -1)
			if len(ind_s_perm_slice) > 0:
				assert(ind_s_perm_slice.min() > -1)


			overwrite_val = False ## Use previous location (since picks are the same)
			if (inc_repeat == (repeat_iters - 1))*(inc_repeat > 0)*(use_overwrite_locations == True)*(i in iwhere_cnts):
				xmle = src_matched[np.where(i == iwhere_cnts)[0][0]]
				xmle, origin = xmle[0:3].reshape(1,-1), xmle[3]
				logprob, skipped_p_ind, skipped_s_ind = np.nan, [], []
				overwrite_val = True

				# # xmle, origin, logprob, skipped_p_ind, skipped_s_ind = differential_evolution_location_trim(trv, locs_use_slice, arv_p - srcs_refined[i,3], ind_p_perm_slice, arv_s - srcs_refined[i,3], ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-max_t/2.0, max_t/2.0], surface_profile = surface_profile, device = device)
				# inpt_rel_p = np.copy(arv_p - srcs_refined[i,3])
				# inpt_rel_s = np.copy(arv_s - srcs_refined[i,3])
				# xmle1, origin_rel1, _, _, _ = differential_evolution_location_trim(trv, locs_use_slice, inpt_rel_p, ind_p_perm_slice, inpt_rel_s, ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-3*src_t_kernel, 3*src_t_kernel], surface_profile = surface_profile, device = device)
				# origin1 = srcs_refined[i,3] + origin_rel1

				# try:
				# 	assert(np.linalg.norm(ftrns1(xmle) - ftrns1(xmle1), axis = 1).max() < 30e3)
				# 	assert(np.abs(origin - origin1).item() < 10)
				# except:
				# 	print('Not same [1]')
				# 	cnt_false += 1

			else:

				# xmle, origin, logprob, skipped_p_ind, skipped_s_ind = differential_evolution_location_trim(trv, locs_use_slice, arv_p - srcs_refined[i,3], ind_p_perm_slice, arv_s - srcs_refined[i,3], ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-max_t/2.0, max_t/2.0], surface_profile = surface_profile, device = device)
				inpt_rel_p = np.copy(arv_p - srcs_refined[i,3])
				inpt_rel_s = np.copy(arv_s - srcs_refined[i,3])
				xmle, origin_rel, logprob, skipped_p_ind, skipped_s_ind = differential_evolution_location_trim(trv, locs_use_slice, inpt_rel_p, ind_p_perm_slice, inpt_rel_s, ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-3*src_t_kernel, 3*src_t_kernel], surface_profile = surface_profile, device = device)
				origin = srcs_refined[i,3] + origin_rel

			if use_offset_quality_control == True:
				offset_dist = np.linalg.norm(ftrns1(xmle.reshape(1,-1)) - ftrns1(srcs_refined[i,:].reshape(1,-1)), axis = 1)
				if offset_dist > offset_ratio_quality_control*src_x_kernel:
					print('Removing event based on offset: %0.4d (%0.4f)'%(offset_dist, offset_dist/src_x_kernel))
					xmle = np.nan*np.ones((1, 3))
					n_skipped += 1


			if np.isnan(xmle).sum() > 0:
				srcs_trv.append(np.nan*np.ones((1, 4)))
				del_arv_p.append(0)
				del_arv_s.append(0)
				continue

			pred_out = trv(torch.Tensor(locs_use_slice).to(device), torch.Tensor(xmle).to(device)).cpu().detach().numpy() + origin
			res_p = pred_out[0,ind_p_perm_slice,0] - arv_p
			res_s = pred_out[0,ind_s_perm_slice,1] - arv_s

			if (use_quality_check == True)*(overwrite_val == False):
				tval_p = pred_out[0,ind_p_perm_slice,0] - origin
				tval_s = pred_out[0,ind_s_perm_slice,1] - origin
				tval_p[tval_p <= 0] = 0.01
				tval_s[tval_s <= 0] = 0.01
				rel_error_p = np.abs(res_p/tval_p)
				rel_error_s = np.abs(res_s/tval_s)
				# idel_p = np.where((rel_error_p > max_relative_error)*((pred_out[0,ind_p_perm_slice,0] - origin) > min_time_buffer))[0]
				# idel_s = np.where((rel_error_s > max_relative_error)*((pred_out[0,ind_s_perm_slice,1] - origin) > min_time_buffer))[0]
				idel_p = np.where((rel_error_p > max_relative_error)*(np.abs(res_p) > min_time_buffer))[0]
				idel_s = np.where((rel_error_s > max_relative_error)*(np.abs(res_s) > min_time_buffer))[0]
				del_arv_p.append(len(idel_p))
				del_arv_s.append(len(idel_s))
						  
				if len(idel_p) > 0:
					arv_p = np.delete(arv_p, idel_p, axis = 0)
					ind_p = np.delete(ind_p, idel_p, axis = 0)
					Picks_P[i] = np.delete(Picks_P[i], idel_p, axis = 0)
					Picks_P_perm[i] = np.delete(Picks_P_perm[i], idel_p, axis = 0)

				if len(idel_s) > 0:
					arv_s = np.delete(arv_s, idel_s, axis = 0)
					ind_s = np.delete(ind_s, idel_s, axis = 0)
					Picks_S[i] = np.delete(Picks_S[i], idel_s, axis = 0)
					Picks_S_perm[i] = np.delete(Picks_S_perm[i], idel_s, axis = 0)
				
				ind_unique_arrivals = np.sort(np.unique(np.concatenate((ind_p, ind_s), axis = 0)).astype('int'))
		
				if len(ind_unique_arrivals) == 0:
					srcs_trv.append(np.nan*np.ones((1, 4)))
					# srcs_sigma.append(np.nan)
					continue			
				
				perm_vec_arrivals = -1*np.ones(locs_use.shape[0]).astype('int')
				perm_vec_arrivals[ind_unique_arrivals] = np.arange(len(ind_unique_arrivals))
				locs_use_slice = locs_use[ind_unique_arrivals]
				ind_p_perm_slice = perm_vec_arrivals[ind_p]
				ind_s_perm_slice = perm_vec_arrivals[ind_s]
				
				if len(ind_p_perm_slice) > 0:
					assert(ind_p_perm_slice.min() > -1)
				if len(ind_s_perm_slice) > 0:
					assert(ind_s_perm_slice.min() > -1)

				if ((len(idel_p) > 0) + (len(idel_s) > 0)) > 0: ## If arrivals have been removed, re-locate
					if (min_required_picks is not False)*(min_required_sta is not False):
						if ((len(ind_unique_arrivals) == 0) + ((len(arv_p) + len(arv_s)) < min_required_picks) + (len(np.unique(np.concatenate((ind_p, ind_s), axis = 0))) < min_required_sta)) > 0:
							srcs_trv.append(np.nan*np.ones((1, 4)))
							continue
		
					else:
						if len(ind_unique_arrivals) == 0:
							srcs_trv.append(np.nan*np.ones((1, 4)))
							continue
					
					# if (len(list(set(skipped_p_ind).intersection(set(idel_p)))) == len(idel_p))*(len(list(set(skipped_s_ind).intersection(set(idel_s)))) == len(idel_s)):
					if (set(idel_p).issubset(skipped_p_ind))*(set(idel_s).issubset(skipped_s_ind)):
						print('Overlapped deleted indices %d %d'%(len(idel_p), len(idel_s)))
						# pass ## In this case, quality removed picks are same as trimmed skipped indices

						# inpt_rel_p = np.copy(arv_p - srcs_refined[i,3])
						# inpt_rel_s = np.copy(arv_s - srcs_refined[i,3])
						# # xmle, origin_rel, logprob, skipped_p_ind, skipped_s_ind = differential_evolution_location_trim(trv, locs_use_slice, inpt_rel_p, ind_p_perm_slice, inpt_rel_s, ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-max_t/2.0, max_t/2.0], surface_profile = surface_profile, device = device)
						# xmle1, origin_rel1, _, _, _ = differential_evolution_location_trim(trv, locs_use_slice, inpt_rel_p, ind_p_perm_slice, inpt_rel_s, ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-3*src_t_kernel, 3*src_t_kernel], surface_profile = surface_profile, device = device)
						# origin1 = srcs_refined[i,3] + origin_rel1

						# try:
						# 	assert(np.linalg.norm(ftrns1(xmle) - ftrns1(xmle1), axis = 1).max() < 30e3)
						# 	assert(np.abs(origin - origin1).item() < 10)
						# except:
						# 	print('Not same [2]')
						# 	cnt_false1 += 1

					else:

						inpt_rel_p = np.copy(arv_p - srcs_refined[i,3])
						inpt_rel_s = np.copy(arv_s - srcs_refined[i,3])
						xmle, origin_rel, logprob, skipped_p_ind, skipped_s_ind = differential_evolution_location_trim(trv, locs_use_slice, inpt_rel_p, ind_p_perm_slice, inpt_rel_s, ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, [-3*src_t_kernel, 3*src_t_kernel], surface_profile = surface_profile, device = device)
						origin = srcs_refined[i,3] + origin_rel

				if use_offset_quality_control == True:
					offset_dist = np.linalg.norm(ftrns1(xmle.reshape(1,-1)) - ftrns1(srcs_refined[i,:].reshape(1,-1)), axis = 1)
					if offset_dist > offset_ratio_quality_control*src_x_kernel:
						print('Removing event based on offset: %0.4d (%0.4f)'%(offset_dist, offset_dist/src_x_kernel))
						xmle = np.nan*np.ones((1, 3))
						n_skipped += 1

				if np.isnan(xmle).sum() > 0:
					srcs_trv.append(np.nan*np.ones((1, 4)))
					continue

			else:
				del_arv_p.append(0)
				del_arv_s.append(0)


			srcs_trv.append(np.concatenate((xmle.reshape(-1)[0:3].reshape(1,-1), np.array([origin]).reshape(1,-1)), axis = 1))
		
		srcs_trv = np.vstack(srcs_trv)
		del_arv_p = np.hstack(del_arv_p)
		del_arv_s = np.hstack(del_arv_s)
		del_arv_p_init = np.copy(del_arv_p)
		del_arv_s_init = np.copy(del_arv_s)
		assert(len(srcs_trv) == len(del_arv_p))
		assert(len(srcs_trv) == len(del_arv_s))
		assert(len(srcs_trv) == len(Picks_P))
		assert(len(srcs_trv) == len(Picks_S))
		assert(len(srcs_trv) == len(srcs_refined))
		###### Only keep events with minimum number of picks and observing stations #########
		print('Number sources (after travel time locations and quality control): %d (Time %0.4f)'%(len(srcs_trv), time.time() - st_process))
		

		# Count number of P and S picks
		cnt_p, cnt_s = np.zeros(srcs_refined.shape[0]), np.zeros(srcs_refined.shape[0])
		for i in range(srcs_refined.shape[0]):
			cnt_p[i] = Picks_P[i].shape[0]
			cnt_s[i] = Picks_S[i].shape[0]
		

		if (min_required_picks is not False)*(min_required_sta is not False):
		
			ikeep_not_nan = np.where(np.isnan(srcs_trv[:,0]) == 0)[0]
			ikeep_picks = np.where((cnt_p + cnt_s) >= min_required_picks)[0]
			ikeep_sta = np.where(np.array([len(np.unique(np.concatenate((Picks_P[j][:,1], Picks_S[j][:,1]), axis = 0))) for j in range(len(srcs_refined))]) >= min_required_sta)[0]
			ikeep = np.sort(np.array(list(set(np.array(list(set(ikeep_picks).intersection(ikeep_sta)))).intersection(ikeep_not_nan))))
			# ikeep = np.sort(np.array(list(set(ikeep).intersection(ikeep_not_nan))))
			iremove = np.delete(np.arange(len(srcs_refined)), ikeep, axis = 0).astype('int')

			srcs_refined = srcs_refined[ikeep]
			srcs_trv = srcs_trv[ikeep]
			# srcs_sigma = srcs_sigma[ikeep]
			del_arv_p = del_arv_p[ikeep]
			del_arv_s = del_arv_s[ikeep]
			cnt_p = cnt_p[ikeep]
			cnt_s = cnt_s[ikeep]

			if len(srcs_trv) == 0:
				print('No events left after minimum pick requirements')
				continue
		
			Picks_P = [Picks_P[j] for j in ikeep]
			Picks_S = [Picks_S[j] for j in ikeep]
		
			Picks_P_perm = [Picks_P_perm[j] for j in ikeep]
			Picks_S_perm = [Picks_S_perm[j] for j in ikeep]

		print('Number sources (after minimum number of picks and stations): %d'%len(srcs_trv))
		
		####################################################################################

		# if len(ind_unique_arrivals) == 0:
		# 	srcs_trv.append(np.nan*np.ones((1, 4)))
		# 	continue				
		
		## For picks not removed by quality control in first location
		## Can "force" trim to use the retained picks within trim
		## in second location (e.g., relax trim for allowed picks)

		## Alternatively: adapt location - apply twice with first pass identifying outliers with trim
		## Then second (warm start and narrowed) and re-add trimmed picks with low errors

		## Note: this would reduce "skip" criteria


		## Save pick data for check
		if inc_repeat != (repeat_iters - 1): ## On first iteration
			len_p_picks_ = np.array([len(Picks_P[j]) for j in range(len(Picks_P))])
			len_s_picks_ = np.array([len(Picks_S[j]) for j in range(len(Picks_S))])
			id_picks_ = [np.concatenate((Picks_P[j][np.argsort(Picks_P[j][:,0]),0:2], Picks_S[j][np.argsort(Picks_S[j][:,0]),0:2]), axis = 0) for j in range(len(Picks_P))]
			src_location_ = np.copy(srcs_trv)

		if inc_repeat == (repeat_iters - 1):

			## Now compute empirical uncertainities
			srcs_sigma = []
			torch.set_grad_enabled(True)
			for i in range(srcs_refined.shape[0]):

				arv_p, ind_p, arv_s, ind_s = Picks_P_perm[i][:,0], Picks_P_perm[i][:,1].astype('int'), Picks_S_perm[i][:,0], Picks_S_perm[i][:,1].astype('int')	
				ind_unique_arrivals = np.sort(np.unique(np.concatenate((ind_p, ind_s), axis = 0)).astype('int'))
				
				if np.isnan(srcs_trv[i,0]) == True:
					srcs_sigma.append(np.nan)
					continue		
				
				perm_vec_arrivals = -1*np.ones(locs_use.shape[0]).astype('int')
				perm_vec_arrivals[ind_unique_arrivals] = np.arange(len(ind_unique_arrivals))
				locs_use_slice = locs_use[ind_unique_arrivals]
				ind_p_perm_slice = perm_vec_arrivals[ind_p]
				ind_s_perm_slice = perm_vec_arrivals[ind_s]


				xmle, origin = srcs_trv[i,0:3].reshape(1,-1), srcs_trv[i,3]
				pred_out = trv(torch.Tensor(locs_use_slice).to(device), torch.Tensor(xmle[0,0:3].reshape(1,-1)).to(device)).cpu().detach().numpy() + origin # srcs_trv[-1][0,3]
				res_p = pred_out[0,ind_p_perm_slice,0] - arv_p
				res_s = pred_out[0,ind_s_perm_slice,1] - arv_s
				
				scale_val1 = 100.0*np.linalg.norm(ftrns1(xmle[0,0:3].reshape(1,-1)) - ftrns1(xmle[0,0:3].reshape(1,-1) + np.array([0.01, 0, 0]).reshape(1,-1)), axis = 1)[0]
				scale_val2 = 100.0*np.linalg.norm(ftrns1(xmle[0,0:3].reshape(1,-1)) - ftrns1(xmle[0,0:3].reshape(1,-1) + np.array([0.0, 0.01, 0]).reshape(1,-1)), axis = 1)[0]
				scale_val = 0.5*(scale_val1 + scale_val2)
		
				scale_partials = (1/60.0)*np.array([1.0, 1.0, scale_val]).reshape(1,-1)
				src_input_p = Variable(torch.Tensor(xmle[0,0:3].reshape(1,-1)).repeat(len(ind_p_perm_slice),1).to(device), requires_grad = True)
				src_input_s = Variable(torch.Tensor(xmle[0,0:3].reshape(1,-1)).repeat(len(ind_s_perm_slice),1).to(device), requires_grad = True)
				trv_out_p = trv_pairwise1(torch.Tensor(locs_use_slice[ind_p_perm_slice]).to(device), src_input_p, method = 'direct')[:,0]
				trv_out_s = trv_pairwise1(torch.Tensor(locs_use_slice[ind_s_perm_slice]).to(device), src_input_s, method = 'direct')[:,1]
				# trv_out = trv_out[np.arange(len(trv_out)), arrivals[n_inds_picks[i],4].astype('int')] # .cpu().detach().numpy() ## Select phase type
				d_p = scale_partials*torch.autograd.grad(inputs = src_input_p, outputs = trv_out_p, grad_outputs = torch.ones(len(trv_out_p)).to(device), retain_graph = True, create_graph = True, allow_unused = True)[0].cpu().detach().numpy()
				d_s = scale_partials*torch.autograd.grad(inputs = src_input_s, outputs = trv_out_s, grad_outputs = torch.ones(len(trv_out_s)).to(device), retain_graph = True, create_graph = True, allow_unused = True)[0].cpu().detach().numpy()
				
				d_grad = np.concatenate((d_p, d_s), axis = 0)
				sig_d = 0.15 ## Assumed pick uncertainty (seconds)
				chi_pdf = chi2(df = 3).pdf(0.99)
				
				var = (d_grad/scale_partials)
				var = np.linalg.pinv(var.T@var)*(sig_d**2)
				var = var*chi_pdf
				#Variances.append(np.expand_dims(var, axis = 0))
				var_cart = (d_grad/scale_partials)/np.array([scale_val1, scale_val2, 1.0]).reshape(1,-1)
				var_cart = np.linalg.pinv(var_cart.T@var_cart)*(sig_d**2)
				var_cart = var_cart*chi_pdf
				sigma_cart = np.linalg.norm(np.diag(var_cart)**(0.5))
				srcs_sigma.append(sigma_cart)
				## Append the final location and origin time

			# srcs_trv = np.vstack(srcs_trv)
			srcs_sigma = np.hstack(srcs_sigma)
			assert(len(srcs_trv) == len(srcs_sigma))
			torch.set_grad_enabled(False)

		####################################################################################


	############### ############### ############### ###############
	     ############### Compute Magnitude ###############
	############### ############### ############### ###############

	## Compute magnitudes.
	# min_log_amplitude_val = -2.0 ## Choose this value to ignore very small amplitudes
	if (compute_magnitudes == True)*(loaded_mag_model == True):
		mag_r = []
		mag_trv = []		
		for i in range(srcs_refined.shape[0]):
			ind_p, log_amp_p = Picks_P[i][:,1].astype('int'), np.log10(Picks_P[i][:,2])
			ind_s, log_amp_s = Picks_S[i][:,1].astype('int'), np.log10(Picks_S[i][:,2])
			mag_p = Mag(torch.Tensor(ind_p).long().to(device), torch.Tensor(srcs_refined[i,0:3].reshape(1,-1)).to(device), torch.Tensor(log_amp_p).to(device), torch.zeros(len(ind_p)).long().to(device))
			mag_s = Mag(torch.Tensor(ind_s).long().to(device), torch.Tensor(srcs_refined[i,0:3].reshape(1,-1)).to(device), torch.Tensor(log_amp_s).to(device), torch.ones(len(ind_s)).long().to(device))
			mag_pred = np.median(np.concatenate((mag_p.cpu().detach().numpy().reshape(-1), mag_s.cpu().detach().numpy().reshape(-1)), axis = 0))
			mag_r.append(mag_pred)
			mag_p = Mag(torch.Tensor(ind_p).long().to(device), torch.Tensor(srcs_trv[i,0:3].reshape(1,-1)).to(device), torch.Tensor(log_amp_p).to(device), torch.zeros(len(ind_p)).long().to(device))
			mag_s = Mag(torch.Tensor(ind_s).long().to(device), torch.Tensor(srcs_trv[i,0:3].reshape(1,-1)).to(device), torch.Tensor(log_amp_s).to(device), torch.ones(len(ind_s)).long().to(device))
			mag_pred = np.median(np.concatenate((mag_p.cpu().detach().numpy().reshape(-1), mag_s.cpu().detach().numpy().reshape(-1)), axis = 0))
			mag_trv.append(mag_pred)
		mag_r = np.hstack(mag_r)
		mag_trv = np.hstack(mag_trv)
	else:
		mag_r = np.nan*np.ones(srcs_trv.shape[0])
		mag_trv = np.nan*np.ones(srcs_trv.shape[0])		


	####################################################################################

	######################## Do additional quality control #############################

	####################################################################################

	# use_additional_quality_control = True
	if use_additional_quality_control == True:

		## Remove based on high spatial perturbation (what causes this? small events and bad associations? Merging two seperate events)
		if max_perturb_offset is not None:
			## Distance offset
			offset = np.linalg.norm(ftrns1(srcs_trv) - ftrns1(srcs_refined), axis = 1)
			ioffset = np.where(offset > max_perturb_offset)[0] ## Remove based on distance offset
		else:
			ioffset = np.array([]).astype('int')

		## Remove based on anomolously high magnitudes and low pick counts (split and mislocated events)
		if mag_thresh_check is not None:
			ioutlier_magnitude_and_count = np.where((mag_trv > mag_thresh_check)*((cnt_p + cnt_s) < min_picks_check))[0]
		else:
			ioutlier_magnitude_and_count = np.array([]).astype('int')

		## Sigma removal (what causes this? small events and bad associations? Merging two seperate events)
		if max_sigma is not None:
			isigma = np.where(srcs_sigma > max_sigma)[0]
		else:
			isigma = np.array([]).astype('int')

		iremove = np.unique(np.concatenate((ioffset, ioutlier_magnitude_and_count, isigma), axis = 0)) # iremove_min_counts
		ikeep = np.array(list(set(np.arange(len(srcs_trv))).difference( iremove )))

		srcs_refined = srcs_refined[ikeep]
		srcs_trv = srcs_trv[ikeep]
		srcs_sigma = srcs_sigma[ikeep]
		mag_r = mag_r[ikeep]
		mag_trv = mag_trv[ikeep]
		del_arv_p = del_arv_p[ikeep]
		del_arv_s = del_arv_s[ikeep]
		cnt_p = cnt_p[ikeep]
		cnt_s = cnt_s[ikeep]

		Picks_P = [Picks_P[j] for j in ikeep]
		Picks_S = [Picks_S[j] for j in ikeep]
	
		Picks_P_perm = [Picks_P_perm[j] for j in ikeep]
		Picks_S_perm = [Picks_S_perm[j] for j in ikeep]		



	trv_out1 = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_refined[:,0:3]).to(device)).cpu().detach().numpy() + srcs_refined[:,3].reshape(-1,1,1)
	trv_out1_all = compute_travel_times(trv, locs, [srcs_refined], device = device)[0] + srcs_refined[:,3].reshape(-1,1,1)

	trv_out2 = np.nan*np.zeros((srcs_trv.shape[0], locs_use.shape[0], 2))
	ifind_not_nan = np.where(np.isnan(srcs_trv[:,0]) == 0)[0]
	trv_out2_all = np.nan*np.zeros((srcs_trv.shape[0], locs.shape[0], 2))
	
	if len(ifind_not_nan) > 0:
		trv_out2[ifind_not_nan,:,:] = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_trv[ifind_not_nan,0:3]).to(device)).cpu().detach().numpy() + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)
		trv_out2_all[ifind_not_nan,:,:] = compute_travel_times(trv, locs, [srcs_trv[ifind_not_nan]], device = device)[0] + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)
		# trv_out2_all[ifind_not_nan,:,:] = trv(torch.Tensor(locs).to(device), torch.Tensor(srcs_trv[ifind_not_nan,0:3]).to(device)).cpu().detach().numpy() + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)

	if apply_location_shift == True:
		locs = pseudo_lla_to_real_lla(locs, ftrns1, mn_shift, rbest_shift)
		locs_use = pseudo_lla_to_real_lla(locs_use, ftrns1, mn_shift, rbest_shift)
		srcs_trv[:,0:3] = pseudo_lla_to_real_lla(srcs_trv[:,0:3], ftrns1, mn_shift, rbest_shift)
		srcs_refined[:,0:3] = pseudo_lla_to_real_lla(srcs_refined[:,0:3], ftrns1, mn_shift, rbest_shift)
		srcs[:,0:3] = pseudo_lla_to_real_lla(srcs[:,0:3], ftrns1, mn_shift, rbest_shift)
		X_query[:,0:3] = pseudo_lla_to_real_lla(X_query[:,0:3], ftrns1, mn_shift, rbest_shift)
		
		# if initialize is None: # else: [lat_range, lon_range, ]
		domain_ = get_domain_bounds(locs_use, scale = scale_domain)
		lat_range, lon_range = domain_['lat_range'], domain_['lon_range']

	# if ('corr1' in globals())*('corr2' in globals()):
	# 	srcs_refined[:,0:3] = srcs_refined[:,0:3] + corr1 - corr2
	# 	srcs_trv[:,0:3] = srcs_trv[:,0:3] + corr1 - corr2

	############### ############### ############### ###############
	     ############### Find Matched Events ###############
	############### ############### ############### ###############


	find_matched_events = True ## Check this if use_shift is true
	if 	find_matched_events == True:

		try:
			t0 = UTCDateTime(date[0], date[1], date[2])
			min_magnitude = 0.1
			srcs_known = download_catalog(lat_range, lon_range, min_magnitude, t0, t0 + 3600*24, t0 = t0, client = 'USGS')[0] # Choose client
			print('Processing %d known events'%len(srcs_known))
	
	
			temporal_win_match = 5.0
			spatial_win_match = 35e3
			matches1 = maximize_bipartite_assignment_wrapper(srcs_known, srcs_refined, ftrns1, ftrns2, temporal_win = temporal_win_match, spatial_win = spatial_win_match)[0]
			if len(ifind_not_nan) > 0:
				matches2 = maximize_bipartite_assignment_wrapper(srcs_known, srcs_trv[ifind_not_nan], ftrns1, ftrns2, temporal_win = temporal_win_match, spatial_win = spatial_win_match)[0]
				matches2[:,1] = ifind_not_nan[matches2[:,1]]
			else:
				matches2 = np.nan*np.zeros((0,2))
		except:
			print('Failed on finding matched events')
			find_matched_events = False
	
	extra_save = process_config.get('extra_save', False)
	save_on = process_config.get('save_on', True)

	if save_on == True:
		if process_known_events == True:
			file_name_ext = 'known_events'
		else:
			file_name_ext = 'continuous_days'
		
		ext_save = path_to_file + 'Catalog' + seperator + '%d'%yr + seperator + '%s_results_%s_%d_%d_%d_ver_%d.hdf5'%(name_of_project, file_name_ext, date[0], date[1], date[2], n_save_ver)
		if (os.path.isdir(path_to_file + 'Catalog' + seperator + '%d'%yr) == 0): os.makedirs(path_to_file + 'Catalog' + seperator + '%d'%yr, exist_ok = True)
		file_save = h5py.File(ext_save, 'w')
		julday = int((UTCDateTime(date[0], date[1], date[2]) - UTCDateTime(date[0], 1, 1))/(day_len)) + 1

		## Note: the solution is in srcs or srcs_trv (lat, lon, depth, origin time)
		## (the GNN prediction location and travel-time based location based on the GNN prediction associations)
		## The associated picks for each event are in Picks/{n}_Picks_P and Picks/{n}_Picks_S
		## for each source index n
		
		file_save['P'] = P
		file_save['P_perm'] = P_perm
		file_save['srcs'] = srcs_refined ## These are the direct locations predicted by the GNN (usually has some spatial bias due to locations of source nodes)
		file_save['srcs_trv'] = srcs_trv ## These are the travel time located sources using associated picks (usually the most accurate!)
		file_save['srcs_w'] = srcs_refined[:,4] ## The detection likelihood value for each source (e.g., > thresh, and usually < 1).
		file_save['srcs_sigma'] = srcs_sigma ## The detection likelihood value for each source (e.g., > thresh, and usually < 1).
		file_save['locs'] = locs
		file_save['locs_use'] = locs_use
		file_save['ind_use'] = ind_use
		file_save['stas'] = stas.astype('S')
		file_save['date'] = np.array([date[0], date[1], date[2], julday])
		# file_save['%d_%d_%d_%d_res1'%(date[0], date[1], date[2], julday)] = res1
		# file_save['%d_%d_%d_%d_res2'%(date[0], date[1], date[2], julday)] = res2
		file_save['cnt_p'] = cnt_p ## Number of P picks per event
		file_save['cnt_s'] = cnt_s ## Number of S picks per event
		file_save['del_arv_p'] = del_arv_p ## Number of deleted P picks during quality check
		file_save['del_arv_s'] = del_arv_s ## Number of deleted S picks during quality check
		# file_save['tsteps_abs'] = tsteps_abs
		file_save['mag_r'] = mag_r
		file_save['mag_trv'] = mag_trv
		file_save['x_grid_ind_list'] = x_grid_ind_list
		file_save['x_grid_ind_list_1'] = x_grid_ind_list_1
		file_save['trv_out1'] = trv_out1
		file_save['trv_out2'] = trv_out2
		file_save['trv_out1_all'] = trv_out1_all
		file_save['trv_out2_all'] = trv_out2_all
		file_save['trv_srcs_init1'] = trv_out_srcs_init1
		file_save['trv_srcs_init2'] = trv_out_srcs_init2
		file_save['n_remove'] = n_remove
		file_save['time'] = st_time - time.time()
		file_save['cnt_isolated_picks'] = cnt_isolated_picks
		file_save['rbest'] = rbest
		file_save['mn'] = mn
		file_save['pred_prams'] = pred_params
		file_save['srcs_init'] = srcs ## These are initial local maxima after Local Marching
		file_save['X_query'] = X_query
		file_save['lat_range'] = lat_range
		file_save['lon_range'] = lon_range

		if find_matched_events == True:
			file_save['srcs_known'] = srcs_known
			file_save['matches1'] = matches1
			file_save['matches2'] = matches2

		# file_save['X_query'] = X_query
		
		if (process_known_events == True):
			if len(srcs_known) > 0:
				file_save['srcs_known'] = srcs_known
				file_save['izmatch1'] = matches1
				file_save['izmatch2'] = matches2

		if apply_location_shift == True:
			file_save['mn_shift'] = mn_shift
			file_save['rbest_shift'] = rbest_shift
		
		if extra_save == True: # This is the continuous space-time output, it can be useful for visualization/debugging, but is memory itensive
			file_save['Out'] = Out_2_sparse ## Is this heavy?

		for j in range(len(Picks_P)):

			file_save['Picks/%d_Picks_P'%j] = Picks_P[j] ## Since these are lists, but they be appended seperatley?
			file_save['Picks/%d_Picks_S'%j] = Picks_S[j]
			file_save['Picks/%d_Picks_P_perm'%j] = Picks_P_perm[j]
			file_save['Picks/%d_Picks_S_perm'%j] = Picks_S_perm[j]

		# success_count = success_count + 1
		file_save.close()
		print('Saved file %0.4f'%(time.time() - st_time))

		write_HypoDD_file = True
		if write_HypoDD_file == True:
			
			mags = np.copy(mag_trv)
			icheck_p = np.where([len(Picks_P[j]) > 0 for j in range(len(srcs_trv))])[0]
			icheck_s = np.where([len(Picks_S[j]) > 0 for j in range(len(srcs_trv))])[0]
			min_assoc_val = min([min([Picks_P[j][:,-1].min() for j in icheck_p]), min([Picks_S[j][:,-1].min() for j in icheck_s])])
			max_assoc_val = max([max([Picks_P[j][:,-1].max() for j in icheck_p]), max([Picks_S[j][:,-1].max() for j in icheck_s])])
			
			max_assoc_val = max([1.0, max_assoc_val])
			pval = np.polyfit([min_assoc_val, max_assoc_val], [0.5, 1.0], 1)
			pmap = lambda x: np.polyval(pval, x)
			use_fixed_pmap = True
			if use_fixed_pmap == True:
				pmap = lambda x: x ## Do not apply transformation, so catalogs will be consistent between days

			# Why re-compute these
			trv_out1 = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_refined[:,0:3]).to(device)).cpu().detach().numpy() + srcs_refined[:,3].reshape(-1,1,1)
			# trv_out1_all = trv(torch.Tensor(locs).to(device), torch.Tensor(srcs_refined[:,0:3]).to(device)).cpu().detach().numpy() + srcs_refined[:,3].reshape(-1,1,1) 
			
			trv_out2 = np.nan*np.zeros((srcs_trv.shape[0], locs_use.shape[0], 2))
			# trv_out2_all = np.nan*np.zeros((srcs_trv.shape[0], locs.shape[0], 2))
			ifind_not_nan = np.where(np.isnan(srcs_trv[:,0]) == 0)[0]
			if len(ifind_not_nan) > 0:
			    trv_out2[ifind_not_nan,:,:] = trv(torch.Tensor(locs_use).to(device), torch.Tensor(srcs_trv[ifind_not_nan,0:3]).to(device)).cpu().detach().numpy() + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)
			    # trv_out2_all[ifind_not_nan,:,:] = trv(torch.Tensor(locs).to(device), torch.Tensor(srcs_trv[ifind_not_nan,0:3]).to(device)).cpu().detach().numpy() + srcs_trv[ifind_not_nan,3].reshape(-1,1,1)
			    
			res_p = [trv_out2[j,Picks_P_perm[j][:,1].astype('int'),0] - Picks_P_perm[j][:,0] for j in range(len(srcs_trv))]
			res_s = [trv_out2[j,Picks_S_perm[j][:,1].astype('int'),1] - Picks_S_perm[j][:,0] for j in range(len(srcs_trv))]
			rms = np.array([np.linalg.norm(np.concatenate((res_p[j], res_s[j]), axis = 0))/np.sqrt(len(res_p[j]) + len(res_s[j])) for j in range(len(srcs_trv))])
			
			# ph2dt accepts hypocenter, followed by its travel time data in the following format:
			#, YR, MO, DY, HR, MN, SC, LAT, LON, DEP, MAG, EH, EZ, RMS, ID

			ext_save = path_to_file + 'Catalog' + seperator + '%d'%yr + seperator + '%s_ph2dt_file_%d_%d_%d_ver_%d.txt'%(name_of_project, date[0], date[1], date[2], n_save_ver)
			
			f = open(ext_save, 'w')
			for i in range(len(srcs_trv)):
				
				t0 = UTCDateTime(date[0], date[1], date[2]) + srcs_trv[i,3]
				sec_res = t0 - UTCDateTime(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
				f.write('# %d %d %d %d %d %0.3f %0.4f %0.4f %0.3f %0.3f %0.3f %0.3f %0.3f %d \n'%(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + sec_res, srcs_trv[i,0], srcs_trv[i,1], -1.0*srcs_trv[i,2]/1000.0, mags[i], srcs_sigma[i]/1000.0, srcs_sigma[i]/1000.0, rms[i], i + 1))
				
				for j in range(len(Picks_P[i])):
					f.write('%s %0.3f %0.2f %s \n'%(stas[int(Picks_P[i][j,1])], Picks_P[i][j,0] - srcs_trv[i,3], pmap(Picks_P[i][j,-1]), 'P'))
			
				for j in range(len(Picks_S[i])):
					f.write('%s %0.3f %0.2f %s \n'%(stas[int(Picks_S[i][j,1])], Picks_S[i][j,0] - srcs_trv[i,3], pmap(Picks_S[i][j,-1]), 'S'))
			
			f.close()
			
			print('Saved HypoDD ph2dt file')
			print(f)
			print('\n')
		
		print('Detected %d events'%(len(srcs_trv)))
		print('Finished saving file %d %d %d'%(date[0], date[1], date[2]))



