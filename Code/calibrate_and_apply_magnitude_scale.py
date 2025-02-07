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
from sklearn.metrics import r2_score
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

## Load Processing settings
n_ver_load = process_config['n_ver_load']
n_step_load = process_config['n_step_load']
n_save_ver = process_config['n_save_ver']
n_ver_picks = process_config['n_ver_picks']

template_ver = process_config['template_ver']
vel_model_ver = process_config['vel_model_ver']
process_days_ver = process_config['process_days_ver']

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

	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, vel_model_ver, use_physics_informed = use_physics_informed, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, vel_model_ver, method = 'direct', use_physics_informed = use_physics_informed, device = device)
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


temporal_match =  calibration_config['temporal_match'] ## window for matched events (seconds)
spatial_match = calibration_config['spatial_match'] ## spatial distance for matched events (m)
min_picks = calibration_config['min_picks'] ## minimum number of total picks to use event in calibration
min_threshold = calibration_config['min_threshold'] ## minimum detection threshold value to use event in calibration
compute_relocations = calibration_config['compute_relocations'] ## Compute example event relocations with travel time corrections
n_relocations = calibration_config['n_relocations'] ## Number of example event relocations with travel time corrections
save_with_data = calibration_config['save_with_data'] ## Flag whether to save data with the calibration file


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


cnt_inc = 0
min_num_sta = None # 5
n_catalog_ver = 1


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

	print('\n Finished %d \n '%i)



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


################## Part 2 ##########################

#### Calibrate and apply a magnitude estimation ####


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


shutil.copyfile(path_to_file + '%s_catalog_ver_%d.hdf5'%(name_of_project, n_catalog_ver), path_to_file + '%s_catalog_ver_%s_copy_%d.hdf5'%(name_of_project, n_catalog_ver, argvs[1]))
# shutil.copyfile(ext_save + 'merged_%s_catalog_all_ver_%d.hdf5'%(name_of_project, n_ver_events), '/scratch/users/imcbrear/GCalifornia/merged_catalog_central_ver_%d_copy_%d.hdf5'%(n_ver_events, int(argvs[1])))

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

## Setup probability of sampling different matched events
prob = np.ones(len(Matches))
for i in range(len(Matches)):
	if (cnt_p[Matches[i,1]] == 0) or (cnt_s[Matches[i,1]] == 0): ## Skip events missing all of p or s (otherwise might have an indexing issue)
		prob[i] = 0.0
	if srcs_w[Matches[i,1]] < min_threshold:
		prob[i] = 0.0
	if (cnt_p[Matches[i,1]] + cnt_s[Matches[i,1]]) < min_picks:
		prob[i] = 0.0

## Fit magnitude model

prob = prob/prob.sum()
print('Retained %0.8f of matches'%(len(np.where(prob > 0)[0])/len(Matches)))

# ## Setup spatial graph and create laplacian
# k_spc_edges = 25 # 50 ## smooth the spatial coefficients
# x_grid = x_grids[0]
# A_src_src = knn(torch.Tensor(ftrns1(x_grid)/1000.0), torch.Tensor(ftrns1(x_grid)/1000.0), k = k_spc_edges + 1).flip(0).long().contiguous().to(device) # )[0]
# lap = get_laplacian(A_src_src, normalization = 'rw')

# ## Initilize Laplace classes
# k_interp = 15
# Lap = Laplacian(lap[0], lap[1])
# # Interp = InterpolateAnisotropicStations(k = k_interp, device = device)

use_scalar_station_corrections = True
if use_scalar_station_corrections == True:
	mag_grid = locs.mean(0).reshape(1,-1)
	k_grid = 1
else:
	n_mag_grid = 30
	mag_grid = kmeans_packing_sampling_points(scale_x_extend, offset_x_extend, 3, n_mag_grid, ftrns1, n_batch = 3000, n_steps = 3000, n_sim = 1)[0]
	k_grid = 5

Mag = Magnitude(torch.Tensor(locs).to(device), torch.Tensor(mag_grid).to(device), ftrns1_diff, ftrns2_diff, k = k_grid, device = device).to(device)

optimizer = optim.Adam(Mag.parameters(), lr = 0.01)
schedular = StepLR(optimizer, 1000, gamma = 0.8)
loss_func = nn.MSELoss()


n_updates = 5000
n_batch = 500
use_difference_loss = True
iuse_p = np.where([len(Picks_P[i]) >= 2 for i in range(len(Picks_P))])[0]
iuse_s = np.where([len(Picks_S[i]) >= 2 for i in range(len(Picks_S))])[0]

## Applying fitting
losses = []
for i in range(n_updates):

	optimizer.zero_grad()

	i0 = np.random.choice(len(Matches), size = n_batch, p = prob)

	ref_ind, srcs_ind = Matches[i0,0], Matches[i0,1]

	arv_p = torch.Tensor(np.hstack([Picks_P[j][:,0].astype('int') for j in srcs_ind])).to(device)
	arv_s = torch.Tensor(np.hstack([Picks_S[j][:,0].astype('int') for j in srcs_ind])).to(device)
	amp_p = torch.Tensor(np.hstack([Picks_P[j][:,2] for j in srcs_ind])).to(device)
	amp_s = torch.Tensor(np.hstack([Picks_S[j][:,2] for j in srcs_ind])).to(device)
	num_p = np.array([len(Picks_P[j]) for j in srcs_ind])
	num_s = np.array([len(Picks_S[j]) for j in srcs_ind])
	ind_p = np.hstack([Picks_P[j][:,1].astype('int') for j in srcs_ind])
	ind_s = np.hstack([Picks_S[j][:,1].astype('int') for j in srcs_ind])

	cat_slice_single = torch.Tensor(np.concatenate((srcs[srcs_ind,0:3], srcs_ref[ref_ind,4].reshape(-1,1)), axis = 1)).to(device) # .repeat_interleave(torch.Tensor(num_p).to(device).long(), dim = 0)

	cat_slice_p = torch.Tensor(cat_slice_single).to(device).repeat_interleave(torch.Tensor(num_p).to(device).long(), dim = 0)
	cat_slice_s = torch.Tensor(cat_slice_single).to(device).repeat_interleave(torch.Tensor(num_s).to(device).long(), dim = 0)


	log_amp_p = Mag.train(torch.Tensor(ind_p).long().to(device), cat_slice_p[:,0:3], cat_slice_p[:,3], torch.zeros(len(ind_p)).long().to(device))
	log_amp_s = Mag.train(torch.Tensor(ind_s).long().to(device), cat_slice_s[:,0:3], cat_slice_s[:,3], torch.ones(len(ind_s)).long().to(device))

	loss1 = loss_func(torch.log10(amp_p), log_amp_p)
	loss2 = loss_func(torch.log10(amp_s), log_amp_s)
	loss = 0.5*loss1 + 0.5*loss2

	if use_difference_loss == True: ## If True, also compute pairwise log_amplitude differences (for different stations, and fixed sources), since
		## these cancel out the effect of the magnitude; and hence, this provides an unsupervised target to constrain the amplitude-distance
		## attenuation relationships (i.e., Trugman, 2024; SRL: A High‐Precision Earthquake Catalog for Nevada).
		ichoose_p = np.random.choice(iuse_p, size = int(n_batch/2))
		ichoose_s = np.random.choice(iuse_s, size = int(n_batch/2))
		ichoose_p1 = [np.random.choice(len(Picks_P[ichoose_p[j]]), size = 2, replace = False) for j in range(len(ichoose_p))]
		ichoose_s1 = [np.random.choice(len(Picks_S[ichoose_s[j]]), size = 2, replace = False) for j in range(len(ichoose_s))]
		
		ind_p1, ind_p2 = np.array([Picks_P[ichoose_p[j]][ichoose_p1[j][0],1] for j in range(len(ichoose_p))]), np.array([Picks_P[ichoose_p[j]][ichoose_p1[j][1],1] for j in range(len(ichoose_p))])
		amp_p1, amp_p2 = np.array([Picks_P[ichoose_p[j]][ichoose_p1[j][0],2] for j in range(len(ichoose_p))]), np.array([Picks_P[ichoose_p[j]][ichoose_p1[j][1],2] for j in range(len(ichoose_p))])

		ind_s1, ind_s2 = np.array([Picks_S[ichoose_s[j]][ichoose_s1[j][0],1] for j in range(len(ichoose_s))]), np.array([Picks_S[ichoose_s[j]][ichoose_s1[j][1],1] for j in range(len(ichoose_s))])
		amp_s1, amp_s2 = np.array([Picks_S[ichoose_s[j]][ichoose_s1[j][0],2] for j in range(len(ichoose_s))]), np.array([Picks_S[ichoose_s[j]][ichoose_s1[j][1],2] for j in range(len(ichoose_s))])

		## Differential P amplitude loss
		log_amp_p1 = Mag.train(torch.Tensor(ind_p1).long().to(device), torch.Tensor(srcs[ichoose_p,0:3]).to(device), torch.ones(len(ichoose_p)).to(device), torch.zeros(len(ind_p1)).long().to(device)) ## Note: effect of magnitude  will be canceled out
		log_amp_p2 = Mag.train(torch.Tensor(ind_p2).long().to(device), torch.Tensor(srcs[ichoose_p,0:3]).to(device), torch.ones(len(ichoose_p)).to(device), torch.zeros(len(ind_p2)).long().to(device)) ## Note: effect of magnitude  will be canceled out
		log_amp_p_diff = log_amp_p1 - log_amp_p2
		trgt_amp_p_diff = torch.Tensor(np.log10(amp_p1) - np.log10(amp_p2)).to(device)
		loss_diff_p = loss_func(log_amp_p_diff, trgt_amp_p_diff)

		## Differential S amplitude loss
		log_amp_s1 = Mag.train(torch.Tensor(ind_s1).long().to(device), torch.Tensor(srcs[ichoose_s,0:3]).to(device), torch.ones(len(ichoose_s)).to(device), torch.ones(len(ind_s1)).long().to(device)) ## Note: effect of magnitude  will be canceled out
		log_amp_s2 = Mag.train(torch.Tensor(ind_s2).long().to(device), torch.Tensor(srcs[ichoose_s,0:3]).to(device), torch.ones(len(ichoose_s)).to(device), torch.ones(len(ind_s2)).long().to(device)) ## Note: effect of magnitude  will be canceled out
		log_amp_s_diff = log_amp_s1 - log_amp_s2
		trgt_amp_s_diff = torch.Tensor(np.log10(amp_s1) - np.log10(amp_s2)).to(device)
		loss_diff_s = loss_func(log_amp_s_diff, trgt_amp_s_diff)

		loss_diff = 0.5*loss_diff_p + 0.5*loss_diff_s

		## Take the mean loss
		loss = 0.5*loss + 0.5*loss_diff


	loss.backward()
	optimizer.step()
	schedular.step()
	losses.append(loss.item())
	print('%d %0.5f'%(i, loss.item()))

	assert(torch.abs(log_amp_p).max().item() < 100)
	assert(torch.abs(log_amp_s).max().item() < 100)


write_training_file = path_to_file + seperator + 'Grids' + seperator
torch.save(Mag.state_dict(), write_training_file + 'trained_magnitude_model_ver_%d.h5'%(n_ver_save))
torch.save(optimizer.state_dict(), write_training_file + 'trained_magnitude_model_ver_%d_optimizer.h5'%(n_ver_save))
np.savez_compressed(write_training_file + 'trained_magnitude_model_ver_%d_supplemental.npz'%(n_ver_save), mag_grid = mag_grid, k_grid = k_grid, losses = losses)
print('saved magnitude model %d'%(n_ver_save))



#### Apply model and compute magnitudes ####

write_catalog_file = True
apply_magnitude_model = True
if apply_magnitude_model == True:

	Mag_pred = []
	for i in range(len(srcs)):

		ind_p, log_amp_p = Picks_P[i][:,1].astype('int'), np.log10(Picks_P[i][:,2])
		ind_s, log_amp_s = Picks_S[i][:,1].astype('int'), np.log10(Picks_S[i][:,2])

		mag_p = Mag(torch.Tensor(ind_p).long().to(device), torch.Tensor(srcs[i,0:3].reshape(1,-1)).to(device), torch.Tensor(log_amp_p).to(device), torch.zeros(len(ind_p)).long().to(device))
		mag_s = Mag(torch.Tensor(ind_s).long().to(device), torch.Tensor(srcs[i,0:3].reshape(1,-1)).to(device), torch.Tensor(log_amp_s).to(device), torch.ones(len(ind_s)).long().to(device))

		mag_pred = np.median(np.concatenate((mag_p.cpu().detach().numpy().reshape(-1), mag_s.cpu().detach().numpy().reshape(-1)), axis = 0))
		Mag_pred.append(mag_pred)
		if np.mod(i, 50) == 0:
			print(i)

	mag_pred = np.hstack(Mag_pred)

	if len(Matches) > 0:

		res_mag = mag_pred[Matches[:,1]] - srcs_ref[Matches[:,0],4]

		print('Station corrections:')
		print(Mag.bias)

		print('\n Magnitude residual quantiles:')
		print(np.round(np.quantile(res_mag, np.arange(0, 1.1, 0.1)), 2))
		print('r2 score: %0.4f'%(r2_score(srcs_ref[Matches[:,0],4], mag_pred[Matches[:,1]])))
		print('mean residual: %0.4f (+/- %0.4f)'%(res_mag.mean(), res_mag.std()))

	if write_catalog_file == True: ## Can also write to individual files, and create the summary catalog file
		z = h5py.File(path_to_file + '%s_catalog_ver_%d.hdf5'%(name_of_project, n_catalog_ver), 'a')
		z['mags'] = mag_pred
		z.close()
