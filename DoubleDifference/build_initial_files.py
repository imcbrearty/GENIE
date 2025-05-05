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
from scipy.stats import chi2
import pathlib
import itertools
import shutil
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
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', use_physics_informed = use_physics_informed, device = device)
	# trv_pairwise1 = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, use_physics_informed = use_physics_informed, device = device)


if (use_differential_evolution_location == False)*(config['train_travel_time_neural_network'] == False):
	hull = ConvexHull(X)
	hull = hull.points[hull.vertices]
else:
	hull = []

## Load calibration config parameters

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


n_ver_events = 1

## Load catalog results
st_load = glob.glob(path_to_file + 'Catalog/19*') # Load years 1900's
st_load.extend(glob.glob(path_to_file + 'Catalog/20*')) # Load years 2000's
iarg = np.argsort([int(st_load[i].split(seperator)[-1]) for i in range(len(st_load))])
st_load = [st_load[i] for i in iarg]
st_load_l = []
for i in range(len(st_load)):
	st = glob.glob(st_load[i] + seperator + '%s*continuous*ver_%d.hdf5'%(name_of_project, n_ver_events))
	if len(st) > 0:
		st_load_l.extend(st)
days = np.vstack([np.array([int(x) for x in st_load_l[i].split(seperator)[-1].split('_')[4:7]]).reshape(1,-1) for i in range(len(st_load_l))])
iarg = np.argsort([(UTCDateTime(days[i,0], days[i,1], days[i,2]) - UTCDateTime(2000, 1, 1)) for i in range(len(st_load_l))])
st_load = [st_load_l[i] for i in iarg] ## This is the list of files to check for matched events
days = days[iarg]
print('Loading %d detected files for comparisons'%len(days))

lat_lim = []
lon_lim = []

## Check for reference catalog for each of these days
srcs_l = []
srcs_ref_l = []
srcs_w_l = []
srcs_sigma_l = []
cnt_p_l = []
cnt_s_l = []
Matches = []
Picks_P = []
Picks_S = []
Times = []
c1, c2 = 0, 0

max_sigma = None # 10e3 ## Remove events with uncertainity higher than this
tree_stas = cKDTree(ftrns1(locs))

# n_ver_events = 8

cnt_inc = 0

min_num_sta = None # 5

for i in range(len(days)):
	yr, mo, dy = days[i,:]
	path_read = path_to_file + 'Calibration/%d/%s_reference_%d_%d_%d_ver_%d.npz'%(yr, name_of_project, yr, mo, dy, n_ver_reference)
	path_flag = os.path.exists(path_read)
	if path_flag == True:

		## Load reference events
		z = np.load(path_read)
		srcs_ref = z['srcs_ref']
		if srcs_ref.shape[1] < 4:
			srcs_ref = np.concatenate((srcs_ref, np.ones((len(srcs_ref),1))), axis = 1) ## No magnitudes given with catalog
		# assert(np.abs(z['date'].reshape(-1) - days[i,:]).max() == 0)
		z.close()
	else:
		srcs_ref = np.zeros((0,5))

	## Load detected events
	z = h5py.File(st_load[i], 'r')
	srcs = z['srcs_trv'][:]
	srcs_w = z['srcs'][:,4] ## Weight value of detection likelihood
	sigma = z['srcs_sigma'][:]
	locs_use = z['locs_use'][:]
	ista_match = tree_stas.query(ftrns1(locs_use)) # [1]
	assert(ista_match[0].max() < 1e3)
	ista_match = ista_match[1]
	perm_assign = (-10000*np.ones(len(locs_use))).astype('int')
	perm_assign[:] = np.copy(ista_match)

	ikeep = np.arange(len(srcs))
	ikeep1 = np.arange(len(srcs_ref))
	if len(lat_lim) > 0:
		ikeep = np.where((srcs[:,0] < lat_lim[1])*(srcs[:,0] > lat_lim[0])*(srcs[:,1] < lon_lim[1])*(srcs[:,1] > lon_lim[0]))[0]
		ikeep1 = np.where((srcs_ref[:,0] < lat_lim[1])*(srcs_ref[:,0] > lat_lim[0])*(srcs_ref[:,1] < lon_lim[1])*(srcs_ref[:,1] > lon_lim[0]))[0]
		# srcs_ref = srcs_ref[ikeep1]

	if max_sigma is not None:
		ikeep = np.array(list(set(ikeep).intersection(np.where(sigma < max_sigma)[0]))).astype('int')

	if min_num_sta is not None:
		num_sta_found = np.array([len(np.unique(np.concatenate((z['Picks/%d_Picks_P_perm'%j][:,1], z['Picks/%d_Picks_S_perm'%j][:,1]), axis = 0))) for j in range(len(srcs))])
		ikeep = np.array(list(set(ikeep).intersection(np.where(num_sta_found >= min_num_sta)[0]))).astype('int')

	## Subset sources
	srcs_ref = srcs_ref[ikeep1]
	srcs = srcs[ikeep]
	srcs_w = srcs_w[ikeep]
	sigma = sigma[ikeep]

	if len(srcs) > 0:
		cnt_p_slice = [len(z['Picks/%d_Picks_P_perm'%j]) for j in ikeep]
		cnt_s_slice = [len(z['Picks/%d_Picks_S_perm'%j]) for j in ikeep]
		cnt_p_l.extend(cnt_p_slice)
		cnt_s_l.extend(cnt_s_slice)

	for inc, j in enumerate(ikeep):
		Picks_P.append(z['Picks/%d_Picks_P_perm'%j][:])
		Picks_S.append(z['Picks/%d_Picks_S_perm'%j][:])
		Picks_P[-1][:,1] = perm_assign[Picks_P[-1][:,1].astype('int')]
		Picks_S[-1][:,1] = perm_assign[Picks_S[-1][:,1].astype('int')]
		# z1['Picks_P_%d'%cnt_inc] = Picks_P[-1]
		# z1['Picks_S_%d'%cnt_inc] = Picks_S[-1]
		t0 = UTCDateTime(yr, mo, dy) + srcs[inc,3]
		seconds_frac = t0 - UTCDateTime(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
		Times.append(np.array([t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + seconds_frac]).reshape(1,-1))
		cnt_inc += 1

	z.close()

	temporal_match = 10.0
	spatial_match = 45e3
	## Find matched events
	matches = maximize_bipartite_assignment(srcs_ref, srcs, ftrns1, ftrns2, temporal_win = temporal_match, spatial_win = spatial_match, verbose = False)[0]

	if len(matches) > 0:
		Matches.append(matches + np.array([c1, c2]).reshape(1,-1))

	if len(srcs) > 0:
		srcs_l.append(srcs)
		srcs_w_l.append(srcs_w)
		srcs_sigma_l.append(sigma)
		c2 += len(srcs)

	if len(srcs_ref) > 0:
		srcs_ref_l.append(srcs_ref)
		c1 += len(srcs_ref)

	print('\n \n Finished %d \n \n '%i)

	# z1['srcs_%d'%i] = srcs_l[-1]
	# z1['srcs_ref_%d'%i] = srcs_ref_l[-1]
	# z1['srcs_w_%d'%i] = srcs_w_l[-1]
	# z1['Picks_P_%d'%i] = Picks_P[-1]
	# z1['Picks_S_%d'%i] = Picks_S[-1]
	# z1['Matches_%d'%i] = Matches[-1]
	# z1['cnt_p_%d'%i] = np.array(cnt_p_l[-1])
	# z1['cnt_s_%d'%i] = np.array(cnt_s_l[-1])

# z1.close()

num_sta = np.array([len(np.unique(np.concatenate((Picks_P[i][:,1], Picks_S[i][:,1]), axis = 0))) for i in range(len(Picks_P))])

Matches = np.vstack(Matches)
srcs = np.vstack(srcs_l)
srcs_w = np.hstack(srcs_w_l)
srcs_sigma = np.hstack(srcs_sigma_l)
cnt_p = np.hstack(cnt_p_l)
cnt_s = np.hstack(cnt_s_l)
srcs_ref = np.vstack(srcs_ref_l)
Times = np.vstack(Times)
res = srcs[Matches[:,1],0:4] - srcs_ref[Matches[:,0], 0:4]

n_catalog_ver = 1

z1 = h5py.File(path_to_file + '%s_catalog_ver_%d.hdf5'%(name_of_project, n_catalog_ver), 'w')
z1['days'] = days
z1['locs'] = locs
z1['stas'] = stas.astype('S')

z1['Matches'] = Matches
z1['srcs'] = srcs
z1['srcs_ref'] = srcs_ref
z1['srcs_w'] = srcs_w
z1['srcs_sigma'] = srcs_sigma
z1['cnt_p'] = cnt_p
z1['cnt_s'] = cnt_s
z1['res'] = res
z1['num_sta'] = num_sta
z1['Times'] = Times


inc_p, inc_s = 0, 0
ind_pick_vec_p = []
ind_pick_vec_s = []
for i in range(len(srcs)):
	ind_pick_vec_p.append(np.ones(len(Picks_P[i]))*i)
	ind_pick_vec_s.append(np.ones(len(Picks_S[i]))*i)
	# inc_p += len(ind_pick_vec_p[-1])
	# inc_s += len(ind_pick_vec_s[-1])
ind_pick_vec_p = np.hstack(ind_pick_vec_p)
ind_pick_vec_s = np.hstack(ind_pick_vec_s)
Picks_P_stack = np.vstack([Picks_P[i] for i in range(len(srcs))])
Picks_S_stack = np.vstack([Picks_S[i] for i in range(len(srcs))])

z1['ind_pick_vec_p'] = ind_pick_vec_p
z1['ind_pick_vec_s'] = ind_pick_vec_s
z1['Picks_P_stack'] = Picks_P_stack
z1['Picks_S_stack'] = Picks_S_stack
z1.close()

## Print summary residuals
print('\nBulk residuals matched events')
print('Detected %d/%d (%0.2f) events'%(len(Matches), len(srcs_ref), len(Matches)/len(srcs_ref)))
print('Lat residual %0.3f (+/- %0.3f)'%(res[:,0].mean(), res[:,0].std()))
print('Lon residual %0.3f (+/- %0.3f)'%(res[:,1].mean(), res[:,1].std()))
print('Depth residual %0.3f (+/- %0.3f)'%(res[:,2].mean(), res[:,2].std()))
print('Origin Time residual %0.3f (+/- %0.3f) \n'%(res[:,3].mean(), res[:,3].std()))

mag_levels = [1,2,3,4]
for mag in mag_levels:
	ip = np.where(srcs_ref[:,4] >= mag)[0]
	ip1 = np.where(srcs_ref[Matches[:,0],4] >= mag)[0]
	if (len(ip) == 0) or (len(ip1) == 0):
		continue
	res_slice = srcs[Matches[ip1,1],0:4] - srcs_ref[Matches[ip1,0], 0:4]
	print('M > %0.2f'%mag)
	print('Detected %d/%d (%0.2f) events'%(len(ip1), len(ip), len(ip1)/len(ip)))
	print('Lat residual %0.3f (+/- %0.3f)'%(res_slice[:,0].mean(), res_slice[:,0].std()))
	print('Lon residual %0.3f (+/- %0.3f)'%(res_slice[:,1].mean(), res_slice[:,1].std()))
	print('Depth residual %0.3f (+/- %0.3f)'%(res_slice[:,2].mean(), res_slice[:,2].std()))
	print('Origin Time residual %0.3f (+/- %0.3f) \n'%(res_slice[:,3].mean(), res_slice[:,3].std()))


print('Number of sources %d'%len(srcs))
print('Number p picks %d'%sum(cnt_p))
print('Number s picks %d'%sum(cnt_s))


################## Part 2 #####################

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

# n_ver_events = 1


shutil.copyfile(path_to_file + '%s_catalog_ver_%d.hdf5'%(name_of_project, n_catalog_ver), path_to_file + '%s_catalog_ver_%s_copy_%d.hdf5'%(name_of_project, n_catalog_ver, argvs[1]))

z = h5py.File(path_to_file + '%s_catalog_ver_%s_copy_%d.hdf5'%(name_of_project, n_catalog_ver, argvs[1]), 'r')
keys_list = list(z.keys())
days = z['days'][:]

Matches = z['Matches'][:]
srcs = z['srcs'][:]
srcs_w = z['srcs_w'][:]
srcs_sigma = z['srcs_sigma'][:]
cnt_p = z['cnt_p'][:]
cnt_s = z['cnt_s'][:]
srcs_ref = z['srcs_ref'][:]
res = z['res'][:]
Times = z['Times'][:]

if len(Matches) > 0:
	assert(Matches[:,1].max() < len(srcs))
	assert(Matches[:,0].max() < len(srcs_ref))


ind_pick_vec_p = z['ind_pick_vec_p'][:]
ind_pick_vec_s = z['ind_pick_vec_s'][:]
Picks_P_stack = z['Picks_P_stack'][:]
Picks_S_stack = z['Picks_S_stack'][:]

tree_p = cKDTree(ind_pick_vec_p.reshape(-1,1))
tree_s = cKDTree(ind_pick_vec_s.reshape(-1,1))

Picks_P = []
Picks_S = []

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
	for j in range(len(ind_batches_unravel[n])):
		Picks_P.append(Picks_P_stack[ip_query[j]])
		Picks_S.append(Picks_S_stack[is_query[j]])
	print('Unraveled %d of %d'%(n, len(ind_batches_unravel)))


os.remove(path_to_file + '%s_catalog_ver_%d_copy_%d.hdf5'%(name_of_project, n_catalog_ver, argvs[1]))


if (len(srcs_ref) > 0): ## Else, it will break

	## Print summary residuals
	print('\nBulk residuals matched events')
	print('Detected %d/%d (%0.2f) events'%(len(Matches), len(srcs_ref), len(Matches)/len(srcs_ref)))
	print('Lat residual %0.3f (+/- %0.3f)'%(res[:,0].mean(), res[:,0].std()))
	print('Lon residual %0.3f (+/- %0.3f)'%(res[:,1].mean(), res[:,1].std()))
	print('Depth residual %0.3f (+/- %0.3f)'%(res[:,2].mean(), res[:,2].std()))
	print('Origin Time residual %0.3f (+/- %0.3f) \n'%(res[:,3].mean(), res[:,3].std()))

	mag_levels = [1,2,3,4]
	for mag in mag_levels:
		ip = np.where(srcs_ref[:,4] >= mag)[0]
		ip1 = np.where(srcs_ref[Matches[:,0],4] >= mag)[0]
		if (len(ip) == 0) or (len(ip1) == 0):
			continue
		res_slice = srcs[Matches[ip1,1],0:4] - srcs_ref[Matches[ip1,0], 0:4]
		print('M > %0.2f'%mag)
		print('Detected %d/%d (%0.2f) events'%(len(ip1), len(ip), len(ip1)/len(ip)))
		print('Lat residual %0.3f (+/- %0.3f)'%(res_slice[:,0].mean(), res_slice[:,0].std()))
		print('Lon residual %0.3f (+/- %0.3f)'%(res_slice[:,1].mean(), res_slice[:,1].std()))
		print('Depth residual %0.3f (+/- %0.3f)'%(res_slice[:,2].mean(), res_slice[:,2].std()))
		print('Origin Time residual %0.3f (+/- %0.3f) \n'%(res_slice[:,3].mean(), res_slice[:,3].std()))


print('Number of sources %d'%len(srcs))
print('Number p picks %d'%sum(cnt_p))
print('Number s picks %d'%sum(cnt_s))

## Absolute pick indices correspond to the unraveling of P_picks + S_picks for each event.

arrivals = [] ## Note: overwrite the phase types, since these are "associated" phases
cnt_reassigned_phase_p = 0
cnt_reassigned_phase_s = 0
for i in range(len(srcs)):
	phase_p = np.copy(Picks_P[i])
	phase_s = np.copy(Picks_S[i])
	if phase_p.shape[1] < 6:
		phase_p = np.concatenate((phase_p, np.zeros((len(phase_p),1))), axis = 1)
		phase_s = np.concatenate((phase_s, np.ones((len(phase_s),1))), axis = 1)

	if len(phase_p) > 0:
		phase_p[:,0] = phase_p[:,0] - srcs[i,3] ## Subtract origin time
		cnt_reassigned_phase_p += (phase_p[:,4] == 1).sum()
		phase_p[:,4] = 0.0
		phase_p[:,5] = i ## Save source index for each arrival
		arrivals.append(phase_p)
	if len(phase_s) > 0:
		phase_s[:,0] = phase_s[:,0] - srcs[i,3] ## Subtract origin time
		cnt_reassigned_phase_s += (phase_s[:,4] == 0).sum()
		phase_s[:,4] = 1.0
		phase_s[:,5] = i ## Save source index for each arrival
		arrivals.append(phase_s)
arrivals = np.vstack(arrivals)
n_arrivals = len(arrivals)
print('Total picks %d'%n_arrivals)
print('%d reassigned s to p'%(cnt_reassigned_phase_p))
print('%d reassigned p to s'%(cnt_reassigned_phase_s))
cnt_p_initial = np.copy(cnt_p) ## Save these, so that can check which sources have updated set of picks
cnt_s_initial = np.copy(cnt_s)

## Pre-build residuals and partial derivatives for the full dataset

Residuals = []
TrvTimes_Initial = []
Phase_type = []
Src_ind = []
n_number_picks_batch = int(50e3)
n_batches_travel_times = int(np.floor(n_arrivals/n_number_picks_batch))
n_inds_picks = [np.arange(n_number_picks_batch) + i*n_number_picks_batch for i in range(np.maximum(1, n_batches_travel_times))]
if n_inds_picks[-1][-1] < (n_arrivals - 1):
	n_inds_picks.append(np.arange(n_inds_picks[-1][-1] + 1, n_arrivals))

if len(n_inds_picks) == 1: ## If only one batch of picks, then only use as many indices as picks
	n_inds_picks[0] = np.arange(n_arrivals)

for i in range(len(n_inds_picks)):
	trv_out = trv_pairwise(torch.Tensor(locs[arrivals[n_inds_picks[i],1].astype('int')]).to(device), torch.Tensor(srcs[arrivals[n_inds_picks[i],5].astype('int')]).to(device)) # , method = 'direct'
	trv_out = trv_out[np.arange(len(trv_out)), arrivals[n_inds_picks[i],4].astype('int')].cpu().detach().numpy() ## Select phase type
	res = arrivals[n_inds_picks[i],0] - trv_out
	Residuals.append(res)
	TrvTimes_Initial.append(trv_out)
	Phase_type.append(arrivals[n_inds_picks[i],4].astype('int'))
	Src_ind.append(arrivals[n_inds_picks[i],5].astype('int'))
	print('Finished residuals (%d/%d)'%(i, len(n_inds_picks)))

Residuals = np.hstack(Residuals)
TrvTimes_Initial = np.hstack(TrvTimes_Initial)
Phase_type = np.hstack(Phase_type)
Src_ind = np.hstack(Src_ind)

Partials = []
scale_partials = (1/60.0)*np.array([1.0, 1.0, 100e3]).reshape(1,-1)
for i in range(len(n_inds_picks)):
	src_input = Variable(torch.Tensor(srcs[arrivals[n_inds_picks[i],5].astype('int'),0:3]).to(device), requires_grad = True)
	trv_out = trv_pairwise(torch.Tensor(locs[arrivals[n_inds_picks[i],1].astype('int')]).to(device), src_input) # method = 'direct'
	trv_out = trv_out[np.arange(len(trv_out)), arrivals[n_inds_picks[i],4].astype('int')] # .cpu().detach().numpy() ## Select phase type
	d1 = torch.autograd.grad(inputs = src_input, outputs = trv_out, grad_outputs = torch.ones(len(trv_out)).to(device), retain_graph = True, create_graph = True)[0].cpu().detach().numpy()
	Partials.append(d1*scale_partials)
	print('Finished partials (%d/%d)'%(i, len(n_inds_picks)))

Partials = np.vstack(Partials)

## Clean up picks slightly
remove_high_relative_error_picks = True ## Find picks with high relative error and remove
check_for_unstable_sources = False ## Remove sources that have too few picks, following removing bad picks
remove_outlier_location_sources = False ## Also remove sources


k_min_degree = 10 ## Require this many neighboring stations within max_source_pair_distance_check
max_source_pair_distance_check = 10e3


if remove_high_relative_error_picks == True:
	rel_error_max = 0.15 ## Above this relative error
	min_time_buffer = 0.35 ## Also at least this large of residual (else, relative error measurements are unstable)
	idel = np.where((np.abs(Residuals/TrvTimes_Initial) > rel_error_max)*(np.abs(Residuals) > min_time_buffer))[0]
	idel_arv = np.copy(idel)
	print('Deleting %d / %d (%0.3f) arrivals'%(len(idel), len(arrivals), len(idel)/len(arrivals)))
	arrivals = np.delete(arrivals, idel, axis = 0)
	Residuals = np.delete(Residuals, idel, axis = 0)
	TrvTimes_Initial = np.delete(TrvTimes_Initial, idel, axis = 0)
	Phase_type = np.delete(Phase_type, idel, axis = 0)
	Src_ind = np.delete(Src_ind, idel, axis = 0)
	Partials = np.delete(Partials, idel, axis = 0)
	n_arrivals = len(arrivals)
	cnt_p = degree(torch.Tensor(arrivals[np.where(arrivals[:,4] == 0)[0],5]).long(), num_nodes = len(srcs)).cpu().detach().numpy()
	cnt_s = degree(torch.Tensor(arrivals[np.where(arrivals[:,4] == 1)[0],5]).long(), num_nodes = len(srcs)).cpu().detach().numpy()

	print('Find way to delete P_picks_stack entries')

	## Check if any sources now have too few picks (or too few unique stations)
	if (check_for_unstable_sources == True) or (remove_outlier_location_sources == True):
		min_picks = 6

		if check_for_unstable_sources == True:
			idel_srcs = np.where((cnt_p + cnt_s) < min_picks)[0]
		else:
			idel_srcs = np.array([]).astype('int')

		if remove_outlier_location_sources == True:
			## Find sources, who don't have k-nearest neighbors within such distance
			## append to the remove sources list.
			# pass
			inearest = knn(torch.Tensor(ftrns1(srcs)/1000.0), torch.Tensor(ftrns1(srcs)/1000.0), k = k_min_degree + 1).flip(0).contiguous()
			dist_vals = global_max_pool(torch.norm(torch.Tensor(ftrns1(srcs))[inearest[0]] - torch.Tensor(ftrns1(srcs))[inearest[1]], dim = 1, keepdim = True), inearest[1]).cpu().detach().numpy().reshape(-1)
			idel_srcs1 = np.where(dist_vals > max_source_pair_distance_check)[0]
			print('Deleting %d (of %d) isolated sources'%(len(idel_srcs1), len(srcs)))
			idel_srcs = np.unique(np.concatenate((idel_srcs, idel_srcs1), axis = 0))

		print('Deleting %d / %d (%0.3f) sources'%(len(idel_srcs), len(srcs), len(idel_srcs)/len(srcs)))
		ikeep = np.delete(np.arange(len(srcs)), idel_srcs, axis = 0)
		perm_vec = -1*np.ones(len(srcs)).astype('int')
		perm_vec[ikeep] = np.arange(len(ikeep))
		srcs_all = np.copy(srcs)
		srcs_del = np.copy(srcs[idel_srcs])

		## Map matched vectors to match values
		if len(Matches) > 0:
			Matches[:,1] = perm_vec[Matches[:,1]]
			idel = np.where(Matches[:,1] < 0)[0]
			Matches = np.delete(Matches, idel, axis = 0) ## Remove matched events that are not part of selected events

		srcs = srcs[ikeep]
		srcs_w = srcs_w[ikeep]
		srcs_sigma = srcs_sigma[ikeep]
		Times = Times[ikeep]
		Picks_P = [Picks_P[i] for i in ikeep]
		Picks_S = [Picks_S[i] for i in ikeep]
		cnt_p = cnt_p[ikeep]
		cnt_s = cnt_s[ikeep]

		## Find picks connected to these sources, and remove them.
		tree_del_sources = cKDTree(idel_srcs.reshape(-1,1))
		ip_del_arrivals = np.where(tree_del_sources.query(arrivals[:,5].reshape(-1,1))[0] == 0)[0]
		arrivals = np.delete(arrivals, ip_del_arrivals, axis = 0)
		Residuals = np.delete(Residuals, ip_del_arrivals, axis = 0)
		TrvTimes_Initial = np.delete(TrvTimes_Initial, ip_del_arrivals, axis = 0)
		Phase_type = np.delete(Phase_type, ip_del_arrivals, axis = 0)
		Src_ind = np.delete(Src_ind, ip_del_arrivals, axis = 0)
		Partials = np.delete(Partials, ip_del_arrivals, axis = 0)
		n_arrivals = len(arrivals)
		arrivals[:,5] = perm_vec[arrivals[:,5].astype('int')] ## Update source indices to the new set of unique sources
		Src_ind = perm_vec[Src_ind]
		cnt_p = degree(torch.Tensor(arrivals[np.where(arrivals[:,4] == 0)[0],5]).long(), num_nodes = len(srcs)).cpu().detach().numpy()
		cnt_s = degree(torch.Tensor(arrivals[np.where(arrivals[:,4] == 1)[0],5]).long(), num_nodes = len(srcs)).cpu().detach().numpy()		


## Extract list of unique indices of picks for all stations, from arrivals
lp_ind_arrival_srcs = cKDTree(arrivals[:,[5]]).query_ball_point(np.arange(len(srcs)).reshape(-1,1), r = 0)

## Assert results
assert(np.abs(Src_ind - arrivals[:,5]).max() == 0)
assert(len(Residuals) == len(arrivals))
assert(len(Residuals) == len(TrvTimes_Initial))
assert(len(Residuals) == len(Phase_type))
assert(len(Residuals) == len(Src_ind))
assert(len(Residuals) == len(Partials))
assert(np.max(cKDTree(Src_ind.reshape(-1,1)).query(arrivals[:,[5]])[0]) == 0)
assert((Phase_type == arrivals[:,4]).min() == 1)
assert(Src_ind.max() == (len(srcs) - 1))
assert((cnt_p + cnt_s).min() > 0)


## Measure uncertainties
tree = cKDTree(Src_ind.reshape(-1,1))
lp_srcs = tree.query_ball_point(np.arange(len(srcs)).reshape(-1,1), r = 0)
sig_d = 0.5

chi_pdf = chi2(df = 3).pdf(0.99)
Variances = []
Variances_cart = []
for i in range(len(srcs)):
	## Compute uncertainities around sources
	var = (Partials[lp_srcs[i]]/scale_partials)
	var = np.linalg.pinv(var.T@var)*(sig_d**2)
	var = var*chi_pdf
	Variances.append(np.expand_dims(var, axis = 0))
	var = (Partials[lp_srcs[i]]/scale_partials)/np.array([110e3, 110e3, 1.0]).reshape(1,-1)
	var = np.linalg.pinv(var.T@var)*(sig_d**2)
	var = var*chi_pdf
	Variances_cart.append(np.expand_dims(var, axis = 0))
	assert(np.abs(np.array(lp_ind_arrival_srcs[i]) - np.array(lp_srcs[i])).max() == 0)
Variances = np.vstack(Variances)
Variances_cart = np.vstack(Variances_cart)


z = h5py.File(path_to_file + '%s_catalog_data_ver_%d.hdf5'%(name_of_project, n_catalog_ver), 'w')
z['srcs'] = srcs
z['srcs_w'] = srcs_w
z['srcs_sigma'] = srcs_sigma
z['srcs_ref'] = srcs_ref
z['Residuals'] = Residuals
z['arrivals'] = arrivals
z['TrvTimes_Initial'] = TrvTimes_Initial
z['Phase_type'] = Phase_type
z['Src_ind'] = Src_ind
z['Partials'] = Partials
z['Variances'] = Variances
z['Variances_cart'] = Variances_cart
z['Matches'] = Matches
z['cnt_p'] = cnt_p
z['cnt_s'] = cnt_s
z['Times'] = Times


inc_p, inc_s = 0, 0
ind_pick_vec_p = []
ind_pick_vec_s = []
for i in range(len(srcs)):
	ind_pick_vec_p.append(np.ones(len(Picks_P[i]))*i)
	ind_pick_vec_s.append(np.ones(len(Picks_S[i]))*i)
	# inc_p += len(ind_pick_vec_p[-1])
	# inc_s += len(ind_pick_vec_s[-1])
ind_pick_vec_p = np.hstack(ind_pick_vec_p)
ind_pick_vec_s = np.hstack(ind_pick_vec_s)
Picks_P_stack = np.vstack(Picks_P) # np.vstack([Picks_P[i] for i in ifind])
Picks_S_stack = np.vstack(Picks_S) # np.vstack([Picks_S[i] for i in ifind])

z['ind_pick_vec_p'] = ind_pick_vec_p
z['ind_pick_vec_s'] = ind_pick_vec_s
# z['ind_vec_p'] = ind_pick_vec_p
# z['ind_vec_s'] = ind_pick_vec_s
z['Picks_P_stack'] = Picks_P_stack
z['Picks_S_stack'] = Picks_S_stack

z['lp_ind_arrival_srcs_vec'] = np.hstack([j*np.ones(len(lp_ind_arrival_srcs[j])) for j in range(len(lp_ind_arrival_srcs))])
z['lp_ind_arrival_srcs_stack'] = np.hstack(lp_ind_arrival_srcs)

z.close()