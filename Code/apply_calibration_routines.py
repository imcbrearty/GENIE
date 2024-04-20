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
from process_utils import *
from calibration_utils import *


path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

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

## Load Processing settings
n_ver_load = process_config['n_ver_load']
n_step_load = process_config['n_step_load']
template_ver = process_config['template_ver']
vel_model_ver = process_config['vel_model_ver']

### Begin automated processing ###
print('Beginning calibration')

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

name_of_project = config['name_of_project']

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
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, device = device)
	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', device = device)

if (use_differential_evolution_location == False)*(config['train_travel_time_neural_network'] == False):
	hull = ConvexHull(X)
	hull = hull.points[hull.vertices]
else:
	hull = []

z = np.load(write_training_file + 'trained_gnn_model_step_%d_ver_%d_losses.npz'%(n_step_load, n_ver_load))
training_params = z['training_params']
graph_params = z['graph_params']
pred_params = z['pred_params']
z.close()

x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs, max_t = load_templates_region(trv, locs, x_grids, ftrns1, training_params, graph_params, pred_params, device = device)
x_grids_cart_torch = [torch.Tensor(ftrns1(x_grids[i])) for i in range(len(x_grids))]

## Load calibration config parameters

with open('calibration_config.yaml', 'r') as file:
    calibration_config = yaml.safe_load(file)

interp_type = calibration_config['interp_type'] ## Type of spatial interpolation (mean, weighted, anisotropic)
n_ver_events = calibration_config['n_ver_events']
n_ver_reference = calibration_config['n_ver_reference']
n_ver_save = calibration_config['n_ver_save']
interp_type = calibration_config['interp_type'] ## Type of spatial interpolation (mean, weighted, anisotropic)
k_spc_lap = calibration_config['k_spc_lap'] ## k-nn value for laplacian smoothing
k_sta_lap = calibration_config['k_sta_lap'] ## k-nn value for laplacian smoothing station coefficients
k_spc_lap_ker = calibration_config['k_spc_lap_ker'] ## k-nn value for laplacian smoothing spatial coefficients kernel
k_sta_lap_ker = calibration_config['k_sta_lap_ker'] ## k-nn value for laplacian smoothing station coefficients kernel
k_spc_interp = calibration_config['k_spc_interp'] ## k-nn value for interpolation
sig_ker = calibration_config['sig_ker'] ## spatial kernel (in km) of the weighting kernel (fixed for weighted, starting value for anisotropic) 
grid_index = calibration_config['grid_index'] ## grid index choice (of x_grids)
n_batch = calibration_config['n_batch'] ## 1000 earthquakes per batch
n_updates = calibration_config['n_updates'] ## Find automatic convergence criteria
use_spc = calibration_config['use_spc'] ## laplacian penality on spatial coefficients
use_sta = calibration_config['use_sta'] ## laplacian penality station coefficients (anisotropic case)
use_ker = calibration_config['use_ker'] ## laplacian penality on kernel of spatial coefficients (anisotropic case)
use_norm = calibration_config['use_norm'] ## norm penality on spatial coefficients
lam_spc = calibration_config['lam_spc'] ## weighting of laplacian regularization loss for spatial coefficients
lam_sta = calibration_config['lam_sta'] ## weighting of laplacian loss on station coefficients
lam_spc_ker = calibration_config['lam_spc_ker'] ## weighting of laplacian loss on station coefficients kernel
lam_sta_ker = calibration_config['lam_sta_ker'] ## weighting of laplacian regularization loss for station coefficients
lam_norm = calibration_config['lam_norm'] ## weighting of norm loss
temporal_match =  calibration_config['temporal_match'] ## window for matched events (seconds)
spatial_match = calibration_config['spatial_match'] ## spatial distance for matched events (m)
min_picks = calibration_config['min_picks'] ## minimum number of total picks to use event in calibration
min_threshold = calibration_config['min_threshold'] ## minimum detection threshold value to use event in calibration
compute_relocations = calibration_config['compute_relocations'] ## Compute example event relocations with travel time corrections
n_relocations = calibration_config['n_relocations'] ## Number of example event relocations with travel time corrections
save_with_data = calibration_config['save_with_data'] ## Flag whether to save data with the calibration file
lat_lim = calibration_config['lat_lim']
lon_lim = calibration_config['lon_lim']
depth_lim = calibration_config['depth_lim']
device = calibration_config['device']

## Now load catalog and calibration catalog and find matched events

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
for i in range(len(days)):
	yr, mo, dy = days[i,:]
	path_read = path_to_file + 'Calibration/%d/%s_reference_%d_%d_%d_ver_%d.npz'%(yr, name_of_project, yr, mo, dy, n_ver_reference)
	path_flag = os.path.exists(path_read)
	if path_flag == True:

		## Load reference events
		z = np.load(path_read)
		srcs_ref = z['srcs_ref']
		assert(np.abs(z['date'].reshape(-1) - days[i,:]).max() == 0)
		z.close()

		## Load detected events
		z = h5py.File(st_load[i], 'r')
		srcs = z['srcs_trv'][:]
		srcs_w = z['srcs'][:,4] ## Weight value of detection likelihood
		if len(srcs) > 0:
			cnt_p_slice = [len(z['Picks/%d_Picks_P'%j]) for j in range(len(srcs))]
			cnt_s_slice = [len(z['Picks/%d_Picks_S'%j]) for j in range(len(srcs))]
			cnt_p.extend(cnt_p_slice)
			cnt_s.extend(cnt_s_slice)

		for j in range(len(srcs)):
			Picks_P.append(z['Picks/%d_Picks_P'%j][:])
			Picks_S.append(z['Picks/%d_Picks_S'%j][:])

		z.close()

		## Find matched events
		matches = maximize_bipartite_assignment(srcs_ref, srcs, ftrns1, ftrns2, temporal_win = temporal_match, spatial_win = spatial_match, verbose = False)[0]

		if len(matches) > 0:
			Matches.append(matches + np.array([c1, c2]).reshape(1,-1))
			c1 += len(srcs_ref)
			c2 += len(srcs)

		if len(srcs) > 0:
			srcs_l.append(srcs)
			srcs_w_l.append(srcs_w)

		if len(srcs_ref) > 0:
			srcs_ref_l.append(srcs_ref)

Matches = np.vstack(Matches)
srcs = np.vstack(srcs_l)
srcs_w = np.hstack(srcs_w_l)
cnt_p = np.hstack(cnt_p)
cnt_s = np.hstack(cnt_s)
srcs_ref = np.vstack(srcs_ref_l)
res = srcs[Matches[:,1],0:4] - srcs_ref[Matches[:,0], 0:4]

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

skip_missing_phase_type = True

## Setup probability of sampling different matched events
prob = np.ones(len(Matches))
for i in range(len(Matches)):
	if (cnt_p[Matches[i,1]] == 0) or (cnt_s[Matches[i,1]] == 0): ## Skip events missing all of p or s (otherwise might have an indexing issue)
		prob[i] = 0.0
	if srcs_w[Matches[i,1]] < min_threshold:
		prob[i] = 0.0
	if (cnt_p[Matches[i,1]] + cnt_s[Matches[i,1]]) < min_picks:
		prob[i] = 0.0
	if lat_lim is not False:
		if (srcs[Matches[i,1],0] < lat_lim[0]) or (srcs[Matches[i,1],0] > lat_lim[1]):
			prob[i] = 0.0
	if lon_lim is not False:
		if (srcs[Matches[i,1],1] < lon_lim[0]) or (srcs[Matches[i,1],1] > lon_lim[1]):
			prob[i] = 0.0
	if depth_lim is not False:
		if (srcs[Matches[i,1],2] < depth_lim[0]) or (srcs[Matches[i,1],2] > depth_lim[1]):
			prob[i] = 0.0
	## Skip events missing all of one phase type, as might a cause a bug later.
	if (skip_missing_phase_type == True) and ((len(Picks_P[Matches[i,1]]) == 0) or (len(Picks_S[Matches[i,1]]) == 0)):
		prob[i] = 0.0


prob = prob/prob.sum()
print('Retained %0.8f of matches'%(len(np.where(prob > 0)[0])/len(Matches)))

## Choose spatial grid
x_grid = x_grids[grid_index]

## Create Laplacians
A_spc_spc = knn(torch.Tensor(ftrns1(x_grid)/1000.0), torch.Tensor(ftrns1(x_grid)/1000.0), k = k_spc_lap + 1).flip(0).long().contiguous().to(device) # )[0]
lap_spc = get_laplacian(A_spc_spc, normalization = 'rw')

A_spc_spc1 = knn(torch.Tensor(ftrns1(x_grid)/1000.0), torch.Tensor(ftrns1(x_grid)/1000.0), k = k_spc_lap_ker + 1).flip(0).long().contiguous().to(device) # )[0]
lap_spc_ker = get_laplacian(A_spc_spc1, normalization = 'rw')

A_sta_sta = knn(torch.Tensor(ftrns1(locs)/1000.0), torch.Tensor(ftrns1(locs)/1000.0), k = k_sta_lap + 1).flip(0).long().contiguous().to(device) # )[0]
lap_sta = get_laplacian(A_sta_sta, normalization = 'rw')

A_sta_sta1 = knn(torch.Tensor(ftrns1(locs)/1000.0), torch.Tensor(ftrns1(locs)/1000.0), k = k_sta_lap_ker + 1).flip(0).long().contiguous().to(device) # )[0]
lap_sta_ker = get_laplacian(A_sta_sta1, normalization = 'rw')

## Initilize Laplace classes
Lap_spc = Laplacian(lap_spc[0], lap_spc[1])
Lap_spc_ker = Laplacian(lap_spc_ker[0], lap_spc_ker[1])
Lap_sta = Laplacian(lap_sta[0], lap_sta[1])
Lap_sta_ker = Laplacian(lap_sta_ker[0], lap_sta_ker[1])
x_grid = torch.Tensor(x_grid).to(device)

## Initilize Interpolation and Laplace classes
if interp_type == 'mean':
	Interp = Interpolate(ftrns1_diff, k = k_spc_interp, device = device)
elif interp_type == 'weighted':
	Interp = InterpolateWeighted(ftrns1_diff, k = k_spc_interp, sig = sig_ker, device = device)
elif interp_type == 'anisotropic':
	Interp = InterpolateAnisotropic(ftrns1_diff, k = k_spc_interp, sig = sig_ker, device = device)

## Setup calibration parameters
coefs = Variable(torch.zeros((len(x_grid), len(locs), 2)).to(device), requires_grad = True) ## Coefficients are initilized for all spatial grid points and stations
coefs_ker = Variable(sig_ker*torch.ones((len(x_grid), locs.shape[0], 3)).to(device), requires_grad = True) ## Coefficients are initilized for all spatial grid points and stations
## Note: coefs ker should have a different entry for each phase type

optimizer = optim.Adam([coefs, coefs_ker], lr = 0.01)
schedular = StepLR(optimizer, 1000, gamma = 0.8)
loss_func = nn.MSELoss()

trgt_spc = torch.zeros(x_grid.shape[0], locs.shape[0]).to(device)
trgt_norm = torch.zeros(x_grid.shape[0], locs.shape[0], 2).to(device)
trgt_sta = torch.zeros(locs.shape[0], x_grid.shape[0]).to(device)

## Target kernel fields are Laplacian applied to uniform field
trgt_spc_ker = Lap_spc_ker(coefs_ker[:,:,0]).detach()
trgt_sta_ker = Lap_sta_ker(coefs_ker[:,:,0].T).detach()

## Applying fitting
losses = []
for i in range(n_updates):

	optimizer.zero_grad()

	i0 = np.random.choice(len(Matches), size = n_batch, p = prob)

	ref_ind, srcs_ind = Matches[i0,0], Matches[i0,1]

	arv_p = torch.Tensor(np.hstack([Picks_P[j][:,0].astype('int') for j in srcs_ind])).to(device)
	arv_s = torch.Tensor(np.hstack([Picks_S[j][:,0].astype('int') for j in srcs_ind])).to(device)
	num_p = np.array([len(Picks_P[j]) for j in srcs_ind])
	num_s = np.array([len(Picks_S[j]) for j in srcs_ind])
	ind_p = np.hstack([Picks_P[j][:,1].astype('int') for j in srcs_ind])
	ind_s = np.hstack([Picks_S[j][:,1].astype('int') for j in srcs_ind])

	## Only use non-duplicated sources for computing interpolation values
	cat_slice_single = torch.Tensor(srcs_ref[ref_ind]).to(device) # .repeat_interleave(torch.Tensor(num_p).to(device).long(), dim = 0)
	# cat_slice_s_single = torch.Tensor(cat[ref_ind]).to(device) # .repeat_interleave(torch.Tensor(num_p).to(device).long(), dim = 0)

	cat_slice_p = torch.Tensor(srcs_ref[ref_ind]).to(device).repeat_interleave(torch.Tensor(num_p).to(device).long(), dim = 0)
	cat_slice_s = torch.Tensor(srcs_ref[ref_ind]).to(device).repeat_interleave(torch.Tensor(num_s).to(device).long(), dim = 0)
	locs_slice_p = torch.Tensor(locs[ind_p]).to(device)
	locs_slice_s = torch.Tensor(locs[ind_s]).to(device)
	isrc_p = np.arange(len(cat_slice_p))
	isrc_s = np.arange(len(cat_slice_s))

	pred_p = trv_pairwise(locs_slice_p, cat_slice_p)[:,0].detach() + cat_slice_p[:,3] # .reshape(-1,1)
	pred_s = trv_pairwise(locs_slice_s, cat_slice_s)[:,1].detach() + cat_slice_s[:,3] # .reshape(-1,1)

	corr = Interp(x_grid, cat_slice_single, coefs, coefs_ker)
	corr_p = corr.repeat_interleave(torch.Tensor(num_p).to(device).long(), dim = 0)[isrc_p,ind_p,0]
	corr_s = corr.repeat_interleave(torch.Tensor(num_s).to(device).long(), dim = 0)[isrc_s,ind_s,1]
	# corr_p = Interp(trv_grid, cat_slice_p, coefs)

	loss1 = loss_func(pred_p + corr_p, arv_p)
	loss2 = loss_func(pred_s + corr_s, arv_s)
	loss = 0.5*loss1 + 0.5*loss2

	if use_spc:

		lap_spc1 = loss_func(Lap_spc(coefs[:,:,0]), trgt_spc)
		lap_spc2 = loss_func(Lap_spc(coefs[:,:,1]), trgt_spc)
		loss = loss + 0.5*lam_spc*lap_spc1 + 0.5*lam_spc*lap_spc2

	if use_sta == True:

		lap_sta1 = loss_func(Lap_sta(coefs[:,:,0].T), trgt_sta)
		lap_sta2 = loss_func(Lap_sta(coefs[:,:,1].T), trgt_sta)
		loss = loss + 0.5*lam_sta*lap_sta1 + 0.5*lam_sta*lap_sta2

	if (use_ker == True)*(use_spc == True):

		## Could also do this with reshaping, or broadcasting
		lap_spc_ker1 = loss_func(Lap_spc_ker(coefs_ker[:,:,0]), trgt_spc_ker)
		lap_spc_ker2 = loss_func(Lap_spc_ker(coefs_ker[:,:,1]), trgt_spc_ker)
		lap_spc_ker3 = loss_func(Lap_spc_ker(coefs_ker[:,:,2]), trgt_spc_ker)

		loss_spc_ker = (1/3.0)*lap_spc_ker1 + (1/3.0)*lap_spc_ker2 + (1/3.0)*lap_spc_ker3
		loss = loss + lam_spc_ker*loss_spc_ker

	if (use_ker == True)*(use_sta == True):

		## Could also do this with reshaping, or broadcasting
		lap_sta_ker1 = loss_func(Lap_sta_ker(coefs_ker[:,:,0].T), trgt_sta_ker)
		lap_sta_ker2 = loss_func(Lap_sta_ker(coefs_ker[:,:,1].T), trgt_sta_ker)
		lap_sta_ker3 = loss_func(Lap_sta_ker(coefs_ker[:,:,2].T), trgt_sta_ker)

		loss_sta_ker = (1/3.0)*lap_sta_ker1 + (1/3.0)*lap_sta_ker2 + (1/3.0)*lap_sta_ker3
		loss = loss + lam_sta_ker*loss_sta_ker

	if use_norm == True:

		loss_norm = loss_func(coefs, trgt_norm)
		loss = loss + lam_norm*loss_norm

	loss.backward()
	optimizer.step()
	schedular.step()
	losses.append(loss.item())
	print('%d %0.5f'%(i, loss.item()))

	assert(torch.abs((pred_p + corr_p) - arv_p).max().item() < 100)
	assert(torch.abs((pred_s + corr_s) - arv_s).max().item() < 100)

## Save calibration result
params = np.array([interp_type, k_spc_interp, k_spc_lap, sig_ker, grid_index])
event_ind_p = torch.arange(len(srcs)).repeat_interleave(torch.Tensor(cnt_p).long()).cpu().detach().numpy()
event_ind_s = torch.arange(len(srcs)).repeat_interleave(torch.Tensor(cnt_s).long()).cpu().detach().numpy()

print('Saving calibration result (version %d)'%n_ver_save)
print('Max corr: %0.2f'%coefs.max().item())
print('Min corr: %0.2f'%coefs.min().item())
if save_with_data == False:
	np.savez_compressed(path_to_file + 'Grids/%s_calibrated_travel_time_corrections_%d.npz'%(name_of_project, n_ver_save), coefs = coefs.cpu().detach().numpy(), coefs_ker = coefs_ker.cpu().detach().numpy(), x_grid = x_grid.cpu().detach().numpy(), srcs = srcs, srcs_ref = srcs_ref, Matches = Matches, params = params, losses = losses, loss1 = loss1.item(), loss2 = loss2.item())
elif save_with_data == True:
	np.savez_compressed(path_to_file + 'Grids/%s_calibrated_travel_time_corrections_%d.npz'%(name_of_project, n_ver_save), coefs = coefs.cpu().detach().numpy(), coefs_ker = coefs_ker.cpu().detach().numpy(), x_grid = x_grid.cpu().detach().numpy(), srcs = srcs, srcs_ref = srcs_ref, Matches = Matches, Picks_P = np.vstack(Picks_P), Picks_S = np.vstack(Picks_S), event_ind_p = event_ind_p, event_ind_s = event_ind_s, params = params, losses = losses, loss1 = loss1.item(), loss2 = loss2.item())
else:
	error('Set save_with_data flag')

## Re-locate example events
if compute_relocations == True:

	srcs_target = []
	srcs_initial = []
	srcs_relocated = []

	for i in range(n_relocations):

		i0 = np.random.choice(len(Matches), p = prob)

		ref_ind = Matches[i0,0]
		srcs_ind = Matches[i0,1]

		srcs_target.append(srcs_ref[ref_ind].reshape(1,-1))

		srcs_initial.append(srcs[srcs_ind].reshape(1,-1))

		arv_p, arv_s = Picks_P[srcs_ind][:,0], Picks_S[srcs_ind][:,0]
		ind_p, ind_s = Picks_P[srcs_ind][:,1].astype('int'), Picks_S[srcs_ind][:,1].astype('int')

		ind_unique = np.sort(np.unique(np.concatenate((ind_p, ind_s), axis = 0)))
		perm_vec = -1*np.ones(len(locs)).astype('int')
		perm_vec[ind_unique] = np.arange(len(ind_unique))
		ind_p_perm = perm_vec[ind_p]
		ind_s_perm = perm_vec[ind_s]
		locs_slice = locs[ind_unique]

		# trv_corr = TrvTimesCorrectionAnisotropic(trv, x_grid, coefs[:,ind_unique,:], coefs_ker, k = k_spc_interp)

		trv_corr = TrvTimesCorrection(trv, x_grid, locs_slice, coefs[:,ind_unique,:], ftrns1_diff, coefs_ker = coefs_ker[:,ind_unique,:], interp_type = 'anisotropic', k = k_spc_interp, sig = sig_ker)

		xmle, logprob = differential_evolution_location(trv_corr, locs_slice, arv_p, ind_p_perm, arv_s, ind_s_perm, lat_range_extend, lon_range_extend, depth_range, device = device)

		## Update origin time

		pred_out = trv_corr(torch.Tensor(locs_slice), torch.Tensor(xmle)).cpu().detach().numpy() + srcs[srcs_ind,3]
		res_p = pred_out[0,ind_p_perm,0] - arv_p
		res_s = pred_out[0,ind_s_perm,1] - arv_s

		mean_shift = 0.0
		cnt_phases = 0
		if len(res_p) > 0:
			mean_shift += np.median(res_p)*(len(res_p)/(len(res_p) + len(res_s)))
			cnt_phases += 1

		if len(res_s) > 0:
			mean_shift += np.median(res_s)*(len(res_s)/(len(res_p) + len(res_s)))
			cnt_phases += 1

		srcs_relocated.append(np.concatenate((xmle, np.array([srcs[srcs_ind,3] - mean_shift]).reshape(1,-1)), axis = 1))


		plot_on = False
		if plot_on == True:
			fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
			pred_out1 = trv_corr(torch.Tensor(locs_slice), torch.Tensor(srcs[srcs_ind].reshape(1,-1))).cpu().detach().numpy() + srcs[srcs_ind,3]
			pred_out2 = trv_corr(torch.Tensor(locs_slice), torch.Tensor(xmle)).cpu().detach().numpy() + srcs_relocated[-1][0,3]
			# pred_out1 = trv_corr(torch.Tensor(locs_slice), torch.Tensor(xmle)).cpu().detach().numpy() + srcs[srcs_ind,3]
			ax[0].scatter(arv_p, ind_p_perm)
			ax[0].scatter(arv_s, ind_s_perm)
			ax[1].scatter(arv_p, ind_p_perm)
			ax[1].scatter(arv_s, ind_s_perm)
			ax[0].plot(pred_out1[0,:,0], np.arange(len(locs_slice)), 'b')
			ax[0].plot(pred_out1[0,:,1], np.arange(len(locs_slice)), 'r')
			ax[1].plot(pred_out2[0,:,0], np.arange(len(locs_slice)), 'b')
			ax[1].plot(pred_out2[0,:,1], np.arange(len(locs_slice)), 'r')
			res_p1 = pred_out1[0,ind_p_perm,0] - arv_p
			res_s1 = pred_out1[0,ind_s_perm,1] - arv_s
			res_p2 = pred_out2[0,ind_p_perm,0] - arv_p
			res_s2 = pred_out2[0,ind_s_perm,1] - arv_s
			ax[0].set_title('%0.2f %0.2f'%(np.sqrt((res_p1**2).sum()/len(res_p1)), np.sqrt((res_s1**2).sum()/len(res_s1))))
			ax[1].set_title('%0.2f %0.2f'%(np.sqrt((res_p2**2).sum()/len(res_p2)), np.sqrt((res_s2**2).sum()/len(res_s2))))
			fig.set_size_inches([15,12])
			# plt.savefig('D:/Projects/GCalifornia/Plots/relocated_event_%d.png'%i)



	## Concatenate result
	srcs_target = np.vstack(srcs_target)
	srcs_initial = np.vstack(srcs_initial)
	srcs_relocated = np.vstack(srcs_relocated)

	## Residuals
	res1 = srcs_initial[:,0:4] - srcs_target[:,0:4]
	res2 = srcs_relocated[:,0:4] - srcs_target[:,0:4]

	print('\nMislocation (initial)')
	print('Lat: %0.3f (+/- %0.3f)'%(res1[:,0].mean(), res1[:,0].std()))
	print('Lon: %0.3f (+/- %0.3f)'%(res1[:,1].mean(), res1[:,1].std()))
	print('Depth: %0.3f (+/- %0.3f)'%(res1[:,2].mean(), res1[:,2].std()))
	print('Origin Time: %0.3f (+/- %0.3f)'%(res1[:,3].mean(), res1[:,3].std()))

	print('\nMislocation (relocated)')
	print('Lat: %0.3f (+/- %0.3f)'%(res2[:,0].mean(), res2[:,0].std()))
	print('Lon: %0.3f (+/- %0.3f)'%(res2[:,1].mean(), res2[:,1].std()))
	print('Depth: %0.3f (+/- %0.3f)'%(res2[:,2].mean(), res2[:,2].std()))
	print('Origin Time: %0.3f (+/- %0.3f)'%(res2[:,3].mean(), res2[:,3].std()))

	## Estimate reduction in local bias
	tree = cKDTree(ftrns1(srcs_target))
	ip_list = tree.query_ball_point(ftrns1(x_grid.cpu().detach().numpy()), r = sig_ker*1000*5) # sig_ker*1000*5
	bias1 = []
	bias2 = []
	ind_bias = []
	for i in range(len(ip_list)):
		if len(ip_list[i]) > 0:
			ip_slice = np.array(list(ip_list[i]))
			bias1.append(np.vstack([srcs_initial[[j],0:4] - srcs_target[[j],0:4] for j in ip_slice]).mean(0, keepdims = True))
			bias2.append(np.vstack([srcs_relocated[[j],0:4] - srcs_target[[j],0:4] for j in ip_slice]).mean(0, keepdims = True))
			ind_bias.append(i)
	bias1 = np.vstack(bias1)
	bias2 = np.vstack(bias2)
	ind_bias = np.hstack(ind_bias)

	print('\nBias (initial)')
	print('Lat: %0.3f'%(np.abs(bias1[:,0]).mean()))
	print('Lon: %0.3f'%(np.abs(bias1[:,1]).mean()))
	print('Depth: %0.3f'%(np.abs(bias1[:,2]).mean()))
	print('Origin Time: %0.3f'%(np.abs(bias1[:,3]).mean()))

	print('\nBias (relocated)')
	print('Lat: %0.3f'%(np.abs(bias2[:,0]).mean()))
	print('Lon: %0.3f'%(np.abs(bias2[:,1]).mean()))
	print('Depth: %0.3f'%(np.abs(bias2[:,2]).mean()))
	print('Origin Time: %0.3f'%(np.abs(bias2[:,3]).mean()))

	print('\nTotal Error (initial)')
	print('Horizontal: %0.4f'%(np.linalg.norm(ftrns1(srcs_initial)[:,0:2] - ftrns1(srcs_target)[:,0:2], axis = 1).mean()))
	print('Vertical: %0.4f'%(np.linalg.norm(ftrns1(srcs_initial)[:,[2]] - ftrns1(srcs_target)[:,[2]], axis = 1).mean()))

	print('\nTotal Error (relocated)')
	print('Horizontal: %0.4f'%(np.linalg.norm(ftrns1(srcs_relocated)[:,0:2] - ftrns1(srcs_target)[:,0:2], axis = 1).mean()))
	print('Vertical: %0.4f'%(np.linalg.norm(ftrns1(srcs_relocated)[:,[2]] - ftrns1(srcs_target)[:,[2]], axis = 1).mean()))

	np.savez_compressed(path_to_file + 'Grids/%s_calibrated_travel_time_corrections_relocations_%d.npz'%(name_of_project, n_ver_save), srcs_initial = srcs_initial, srcs_relocated = srcs_relocated, srcs_target = srcs_target, res1 = res1, res2 = res2, bias1 = bias1, bias2 = bias2, coefs = coefs.cpu().detach().numpy(), coefs_ker = coefs_ker.cpu().detach().numpy(), x_grid = x_grid.cpu().detach().numpy(), ind_bias = ind_bias)


plot_results = True
close_plots = True
if plot_results == True:

	## Visualize relocations
	fig, ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = [18,10])
	ax[0].scatter(locs[:,1], locs[:,0], c = 'r', marker = '^', alpha = 0.3, s = 2)
	ax[1].scatter(locs[:,1], locs[:,0], c = 'r', marker = '^', alpha = 0.3, s = 2)
	ax[0].scatter(srcs_target[:,1], srcs_target[:,0], s = 8, alpha = 0.5, label = 'Target')
	ax[1].scatter(srcs_target[:,1], srcs_target[:,0], s = 8, alpha = 0.5, label = 'Target')
	ax[0].scatter(srcs_initial[:,1], srcs_initial[:,0], s = 8, alpha = 0.5, label = 'Ours')
	ax[1].scatter(srcs_relocated[:,1], srcs_relocated[:,0], s = 8, alpha = 0.5, label = 'Ours')
	ax[0].set_aspect(1.0/np.cos(np.mean(locs[:,0])*np.pi/180.0))
	ax[1].set_aspect(1.0/np.cos(np.mean(locs[:,0])*np.pi/180.0))
	ax[0].set_title('Initial')
	ax[1].set_title('Relocated')
	ax[0].legend()
	ax[1].legend()
	fig.savefig(path_to_file + 'Plots' + seperator + 'event_relocations_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)
	ax[0].plot(np.concatenate((srcs_initial[:,[1]], srcs_target[:,[1]]), axis = 1).T, np.concatenate((srcs_initial[:,[0]], srcs_target[:,[0]]), axis = 1).T, c = 'black', alpha = 0.2)
	ax[1].plot(np.concatenate((srcs_relocated[:,[1]], srcs_target[:,[1]]), axis = 1).T, np.concatenate((srcs_relocated[:,[0]], srcs_target[:,[0]]), axis = 1).T, c = 'black', alpha = 0.2)
	fig.savefig(path_to_file + 'Plots' + seperator + 'event_relocations_matches_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

	## Visualize coefficients
	coefs = coefs.cpu().detach().numpy()
	coefs_ker = coefs_ker.cpu().detach().numpy()

	## Spatial grid
	for i in range(15):
		i0 = np.random.randint(0, high = coefs.shape[1])
		norm = Normalize(coefs[:,i0,:].min(), coefs[:,i0,:].max())
		fig, ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = [12, 8])
		ax[0].scatter(x_grid[:,1], x_grid[:,0], c = coefs[:,i0,0], norm = norm)
		ax[1].scatter(x_grid[:,1], x_grid[:,0], c = coefs[:,i0,1], norm = norm)
		# ax[0].scatter(locs[i0,1], locs[i0,0], c = 'r', marker = '^')
		fig.savefig(path_to_file + 'Plots' + seperator + 'spatial_grid_coefficients_example_%d_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

	## Stations
	for i in range(15):
		i0 = np.random.randint(0, high = coefs.shape[0])
		norm = Normalize(coefs[i0,:,:].min(), coefs[i0,:,:].max())
		fig, ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = [12, 8])
		ax[0].scatter(locs[:,1], locs[:,0], c = coefs[i0,:,0], norm = norm)
		ax[1].scatter(locs[:,1], locs[:,0], c = coefs[i0,:,1], norm = norm)
		fig.savefig(path_to_file + 'Plots' + seperator + 'station_coefficients_example_%d_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)


	## Spatial grid (kernel)
	for i in range(15):
		i0 = np.random.randint(0, high = coefs.shape[1])
		norm = Normalize(coefs_ker[:,i0,:].min(), coefs_ker[:,i0,:].max())
		fig, ax = plt.subplots(1,3, sharex = True, sharey = True, figsize = [12, 8])
		for j in range(3):
			ax[j].scatter(x_grid[:,1], x_grid[:,0], c = coefs_ker[:,i0,j], norm = norm)
			ax[j].scatter(x_grid[:,1], x_grid[:,0], c = coefs_ker[:,i0,j], norm = norm)
		fig.savefig(path_to_file + 'Plots' + seperator + 'spatial_grids_kernel_example_%d_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)


	## Stations (kernel)
	for i in range(15):
		i0 = np.random.randint(0, high = coefs.shape[0])
		norm = Normalize(coefs_ker[i0,:,j].min(), coefs_ker[i0,:,j].max())
		fig, ax = plt.subplots(1,3, sharex = True, sharey = True, figsize = [12, 8])
		for j in range(3):
			ax[j].scatter(locs[:,1], locs[:,0], c = coefs_ker[i0,:,j], norm = norm)
			ax[j].scatter(locs[:,1], locs[:,0], c = coefs_ker[i0,:,j], norm = norm)
		fig.savefig(path_to_file + 'Plots' + seperator + 'station_kernel_example_%d_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

	if close_plots == True:
		plt.close('all')
