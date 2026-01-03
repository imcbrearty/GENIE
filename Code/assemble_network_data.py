
import yaml
import numpy as np
import os
import torch
from torch import optim, nn
import shutil
from collections import defaultdict
from sklearn.metrics import r2_score
# import pandas
import pathlib
import json
import pdb
import scipy
from math import floor, sqrt
from torch_cluster import knn
from itertools import product
from torch_geometric.utils import from_networkx
from scipy.optimize import differential_evolution
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import sort_edge_index, subgraph # , is_sorted # , is_coalesced
from torch_geometric.nn import MessagePassing
from skopt.utils import use_named_args
from scipy.optimize import minimize
from scipy.optimize import minimize, fsolve
from torch_scatter import scatter
from scipy.stats import pearsonr
import scipy.stats.qmc as qmc
from scipy.spatial import KDTree
from skopt import gp_minimize
from skopt.space import Real
from scipy import stats
import networkx as nx
from utils import *


def optimize_with_differential_evolution(center_loc):
    """
    Optimize using the differential evolution algorithm to minimize a loss function based on geospatial transformations.
    
    Parameters:
    - center_loc (numpy.ndarray): The central location to optimize around.
    
    Returns:
    - soln (object): Solution object from the differential evolution algorithm.
    """
    
    loss_coef = [1, 1, 1.0, 0]

    # Calculate initial ecef values as they don't depend on x
    norm_lat_ecef = lla2ecef(np.concatenate((center_loc, center_loc + [0.001, 0.0, 0.0]), axis=0))
    norm_vert_ecef = lla2ecef(np.concatenate((center_loc, center_loc + [0.0, 0.0, 10.0]), axis=0))
    norm_lat = np.linalg.norm(norm_lat_ecef[1] - norm_lat_ecef[0])
    norm_vert = np.linalg.norm(norm_vert_ecef[1] - norm_vert_ecef[0])

    trgt_lat = np.array([0, 1.0, 0]).reshape(1, -1)
    trgt_vert = np.array([0, 0, 1.0]).reshape(1, -1)
    trgt_center = np.zeros(3)

    def loss_function(x):
        rbest = rotation_matrix_full_precision(x[0], x[1], x[2])

        center_out = ftrns1(center_loc, rbest, x[3:].reshape(1, -1))
        out_unit_lat = (ftrns1(center_loc + [0.001, 0.0, 0.0], rbest, x[3:].reshape(1, -1)) - center_out) / norm_lat
        out_unit_vert = (ftrns1(center_loc + [0.0, 0.0, 10.0], rbest, x[3:].reshape(1, -1)) - center_out) / norm_vert

        # If locs are global, then include this line
        # out_locs = ftrns1(locs, rbest, x[3:].reshape(1, -1))

        loss1 = np.linalg.norm(trgt_lat - out_unit_lat, axis=1)
        loss2 = np.linalg.norm(trgt_vert - out_unit_vert, axis=1)
        loss3 = np.linalg.norm(trgt_center.reshape(1, -1) - center_out, axis=1)
        loss = loss_coef[0] * loss1 + loss_coef[1] * loss2 + loss_coef[2] * loss3

        return loss

    bounds = [(0, 2.0 * np.pi) for _ in range(3)] + [(-1e7, 1e7) for _ in range(3)]
    soln = differential_evolution(loss_function, bounds, popsize=30, maxiter=1000, disp=True)

    return soln




path_to_file = str(pathlib.Path().absolute())
path_to_file += '\\' if '\\' in path_to_file else '/'
seperator = '\\' if '\\' in path_to_file else '/'
print(f'Working in the directory: {path_to_file}')


# Load configuration from YAML
config = load_config(path_to_file + 'config.yaml')
name_of_project = config['name_of_project']
num_steps = config['number_of_update_steps']
with_density = config['with_density']
use_spherical = config['use_spherical']
depth_importance_weighting_value_for_spatial_graphs = config['depth_importance_weighting_value_for_spatial_graphs']
fix_nominal_depth = config['fix_nominal_depth']
use_time_shift = config.get('use_time_shift', False)
number_of_spatial_nodes = config['number_of_spatial_nodes']
num_grids = config['number_of_grids']
if use_time_shift == True:
	time_shift_range = config.get('time_shift_range', 10.0)/2.0 ## Note this scaling (the full window is time shift range)
	scale_time = config.get('scale_time', 5000.0)
else:
	time_shift_range = 10.0 ## Not used
	scale_time = 1.0


depth_upscale_factor = 1.0
time_upscale_factor = 1.0
use_effective_time_scale = True


if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# Station file
z = np.load(path_to_file + 'stations.npz')
locs, stas = z['locs'], z['stas']
z.close()

print('\n Using stations:')
print(stas)
print('\n Using locations:')
print(locs)

# Region file
z = np.load(path_to_file + 'region.npz', allow_pickle = True)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
years = z['years']
# n_spatial_nodes = config['number_of_spatial_nodes']
load_initial_files = z['load_initial_files'][0]
use_pretrained_model = z['use_pretrained_model'][0]
if use_pretrained_model == 'None':
	use_pretrained_model = None
if with_density == 'None':
	with_density = None
z.close()
shutil.copy(path_to_file + 'region.npz', path_to_file + f'{config["name_of_project"]}_region.npz')



lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)


## Check if using full Earth, set target sampling bounds ##
if (lat_range_extend[0] <= -89.98)*(lat_range_extend[1] >= 89.98)*(lon_range_extend[0] <= -179.98)*(lon_range_extend[1] >= 179.98):
	use_global = True
	lat_range_extend = [-90.0, 90.0]
	lon_range_extend = [-180.0, 180.0]
else:
	use_global = False

## Fit projection coordinates and create spatial grids
if use_spherical == True:

	earth_radius = 6371e3
	ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
	ftrns2 = lambda x, rbest, mn: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean

else:

	earth_radius = 6378137.0
	ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
	ftrns2 = lambda x, rbest, mn: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean

## Unit lat, vertical vectors; point positive y, and outward normal
## mean centered stations. Keep the vertical depth, consistent.

if use_global == True: 
	print('\n Using global')
else:
	print('\n Using domain')

print('\n Latitude:')
print(lat_range)
print('\n Longitude:')
print(lon_range)
print('\n Depths:')
print(depth_range)

if fix_nominal_depth == True:
	nominal_depth = 0.0 ## Can change the target depth projection if prefered
else:
	nominal_depth = locs[:,2].mean() ## Can change the target depth projection if prefered

center_loc = np.array([lat_range[0] + 0.5*np.diff(lat_range)[0], lon_range[0] + 0.5*np.diff(lon_range)[0], nominal_depth]).reshape(1,-1)
# center_loc = locs.mean(0, keepdims = True)

use_differential_evolution = True
if (use_differential_evolution == True)*(use_global == False):

	## This is prefered fitting method
	
	## Unit lat, vertical vectors; point positive y, and outward normal
	## mean centered stations. Keep the vertical depth, consistent.
	
	print('\n Using domain')
	print('\n Latitude:')
	print(lat_range)
	print('\n Longitude:')
	print(lon_range)
	
	fix_nominal_depth = True
	assert(fix_nominal_depth == True)
	if fix_nominal_depth == True:
		nominal_depth = 0.0 ## Can change the target depth projection if prefered
	else:
		nominal_depth = locs[:,2].mean() ## Can change the target depth projection if prefered
	
	center_loc = np.array([lat_range[0] + 0.5*np.diff(lat_range)[0], lon_range[0] + 0.5*np.diff(lon_range)[0], nominal_depth]).reshape(1,-1)
	
	# from scipy.optimize import differential_evolution
	
	# os.rename(ext_dir + 'stations.npz', ext_dir + '%s_stations_backup.npz'%name_of_project)
	soln = optimize_with_differential_evolution(center_loc)
	rbest = rotation_matrix_full_precision(soln.x[0], soln.x[1], soln.x[2])
	mn = soln.x[3::].reshape(1,-1)

else:

	## For global, do not use local corrections
	mn = np.zeros((1,3))
	rbest = np.eye(3)


## Save station file with projection functions
np.savez_compressed(path_to_file + f'{config["name_of_project"]}_stations.npz', locs = locs, stas = stas, rbest = rbest, mn = mn)
print('Saved station file \n')

mn_cuda = torch.tensor(mn, device = device)
rbest_cuda = torch.tensor(rbest, device = device)

corr1 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)
corr2 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)


## Set updated projection functions

if use_spherical == True:

	earth_radius = 6371e3
	ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius)  # invert ftrns1

	ftrns1_abs = lambda x: lla2ecef(x, e = 0.0, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, e = 0.0, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_abs = lambda x: ecef2lla(x, e = 0.0, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, e = 0.0, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


else:

	earth_radius = 6378137.0	
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1

	ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device)  # invert ftrns1

	# ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	# ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius)  # invert ftrns1	

	ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1

	ftrns1_diff_abs = lambda x: lla2ecef_diff(x, a = torch.tensor(earth_radius, device = device), device = device) if x.shape[1] == 3 else torch.cat((lla2ecef_diff(x, a = torch.tensor(earth_radius, device = device), device = device), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_diff_abs = lambda x: ecef2lla_diff(x, a = torch.tensor(earth_radius, device = device), device = device) if x.shape[1] == 3 else torch.cat((ecef2lla_diff(x, a = torch.tensor(earth_radius, device = device), device = device), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


## Compute expected scales of domain

def compute_expected_spacing(
	N,  # total number of points
	lat_range=lat_range_extend,
	lon_range=lon_range_extend,
	depth_range=depth_range,
	time_range=time_shift_range,  # T, full range = 2T
	use_time=use_time_shift,
	scale_time=scale_time,  # w_scale: length per unit time
	use_global=use_global,
	earth_radius=6378137.0
):
	# --- Spatial volume ---
	if use_global:
		Area = 4 * np.pi * earth_radius**2
		Volume_space = (4.0 * np.pi / 3.0) * (r_max**3 - r_min**3)
	else:
		# dlon = np.deg2rad(lon_range[1] - lon_range[0])
		dlon_deg = (lon_range[1] - lon_range[0]) % 360
		if dlon_deg == 0 and lon_range[1] != lon_range[0]: 
			dlon_deg = 360 # Full circle
		dlon = np.deg2rad(dlon_deg)

		sin_diff = np.sin(np.deg2rad(lat_range[1])) - np.sin(np.deg2rad(lat_range[0]))
		Area = earth_radius**2 * dlon * sin_diff
		Volume_space = Area * (r_max**3 - r_min**3) / (3.0 * earth_radius**2)

	# 3D hexagonal packing factor
	hex_factor_3d = 0.74048

	nominal_spacing_space = (Volume_space / (hex_factor_3d * N)) ** (1.0 / 3.0)

	if not use_time:
		return nominal_spacing_space, nominal_spacing_space, 0.0

	# --- 4D hypervolume (spatial volume × scaled time range) ---
	hypervolume_4d = Volume_space * (scale_time * 2.0 * time_range)

	# 4D joint spacing (no reliable hex factor → use cubic = 1.0)
	nominal_spacing_4d = (hypervolume_4d / N) ** (1.0 / 4.0)

	# Raw time spacing (unscaled)
	nominal_spacing_time = (2.0 * time_range) / (N ** (1.0 / 4.0))

	return Volume, Volume_space, nominal_spacing_4d, nominal_spacing_space, nominal_spacing_time


def compute_warped_expected_spacing(
	N,  # total number of points
	lat_range=lat_range_extend,
	lon_range=lon_range_extend,
	depth_range=depth_range,
	time_range=time_shift_range,  # T, full range = 2T
	use_time=use_time_shift,
	scale_time=scale_time,  # w_scale: length per unit time
	depth_boost = depth_upscale_factor,
	use_global=use_global,
	earth_radius=6378137.0
):
    # 1. --- METRIC AREA ---
    # Because of 1/r warping, the area is constant at all depths in metric space.
    # We use the surface area at earth_radius as the reference cross-section.
    if use_global:
        Area_metric = 4 * np.pi * earth_radius**2
    else:
        dlon_deg = (lon_range[1] - lon_range[0]) % 360
        if dlon_deg == 0 and lon_range[1] != lon_range[0]: 
            dlon_deg = 360
        dlon = np.deg2rad(dlon_deg)
        sin_diff = np.sin(np.deg2rad(lat_range[1])) - np.sin(np.deg2rad(lat_range[0]))
        Area_metric = earth_radius**2 * dlon * sin_diff

    # 2. --- METRIC DEPTH (Z-Stretch) ---
    # The physical thickness is stretched by the depth_boost factor.
    thickness_phys = abs(depth_range[1] - depth_range[0])
    # --- 1. Compute Metric Spatial Volume (The "Unrolled Slab") ---
    # Use earth_radius area to account for 1/r warp and thickness*depth_boost
    thickness_metric = (r_max - r_min) * depth_boost
    Volume_space_metric = Area_metric * thickness_metric    

    # --- 2. 3D Nominal Spacing (Spatial Projection) ---
    # This tells you: "If I only had 3D space, what would the spacing be?"
    hex_factor_3d = 0.74048
    nominal_spacing_space = (Volume_space_metric / (hex_factor_3d * N)) ** (1.0 / 3.0)    

    # --- 3. 4D Hypervolume (Space x Scaled Time) ---
    # Total time span (2T) stretched by scale_time
    Volume_4d_metric = Volume_space_metric * (2.0 * time_range * scale_time)    

    # --- 4. 4D Joint Spacing (The FPS Target) ---
    # This is the actual distance (in metric units) FPS will enforce
    nominal_spacing_4d = (Volume_4d_metric / N) ** (1.0 / 4.0)    

    # --- 5. "Raw" Nominal Time Spacing ---
    # This represents the temporal "slot" width in seconds.
    # We use the 4th-root of N to show how the 4D density partitions time.
    nominal_spacing_time = (2.0 * time_range) / (N ** (1.0 / 4.0))

    return Volume_4d_metric, Volume_space_metric, Area_metric, nominal_spacing_4d, nominal_spacing_space, nominal_spacing_time


def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_min
	return r_val


def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_max
	return -r_val

bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_min, bounds, popsize = 50, maxiter = 1000, disp = True)
r_min = optimize_r_min(np.array([soln.x]))[0]; print('\n')

bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True)
r_max = -1.0*optimize_r_max(np.array([soln.x]))[0]; print('\n')
assert(r_max >= r_min)


# strategy = 'KEEP_FIXED'
# strategy = 'USE_ISOTROPIC'
# strategy = 'ADJUST_N'
# strategy = 'ADJUST_TIME_WINDOW'
# strategy = 'JOINT_OPTIMIZATION'
strategy = "TRIPLE_OPTIMIZATION"

N_target = number_of_spatial_nodes

# --- PHASE 1: Baseline Spatial Resolution ---
baseline = compute_warped_expected_spacing(
    N_target, lat_range=lat_range_extend, lon_range=lon_range_extend,
    depth_range=depth_range, time_range=time_shift_range,
    scale_time=1.0, depth_boost=depth_upscale_factor, use_global=use_global
)


# Constants from Phase 1
V3 = baseline[1]  # The Metric 3D Volume (m^3) - Constant
T_init = float(time_shift_range)
N_init = float(number_of_spatial_nodes)
W_init = float(scale_time)
scale_time_init = float(scale_time)


if strategy == "USE_ISOTROPIC":
    # Solve for scale_time such that dx(N_init) = scale_time * dt(N_init, T_init)
    dx = (V3 / N_init)**(1/3)
    dt = (2.0 * T_init) / (N_init**0.25)
    scale_time = dx / dt
    N_target = N_init

elif strategy == "ADJUST_N":
    # We solve: (V3 / N)^(1/3) = W_init * (2*T_init / N^(1/4))
    # Analytical solution: N = ( V3^(1/3) / (W_init * 2 * T_init) )^12
    N_required = ((V3**(1/3)) / (W_init * 2.0 * T_init))**12
    N_target = int(np.clip(N_required, 5, 25000))
    # Note: scale_time stays at W_init

elif strategy == "ADJUST_TIME_WINDOW":
    # Solve for T such that dx(N_init) = W_init * dt(N_init, T)
    dx = (V3 / N_init)**(1/3)
    # T = (dx * N_init^0.25) / (2 * W_init)
    time_shift_range = (dx * (N_init**0.25)) / (2.0 * W_init)
    N_target = N_init

elif strategy == "JOINT_OPTIMIZATION":

    # Starting values
    ln_N_init = np.log(float(number_of_spatial_nodes))
    ln_T_init = np.log(float(time_shift_range))
    W_init = float(scale_time)
    V3_const = baseline[1] # Metric Volume

    def objective(params):
        ln_N, ln_T = params
        curr_N = np.exp(ln_N)
        curr_T = np.exp(ln_T)
        
        # 1. Calculate resolutions at this specific N and T
        dx = (V3_const / curr_N)**(1/3)
        dt = (2.0 * curr_T) / (curr_N**0.25)
        
        # 2. Distance from intent (Movers Distance)
        # alpha_weight = 1.0 (Equal relative penalty for N and T)
        dist_cost = (ln_N - ln_N_init)**2 + (ln_T - ln_T_init)**2
        
        # 3. Isotropy Constraint (dx must equal dt * W_init)
        # We use log-ratio so 0 is perfect isotropy
        iso_error = (np.log(dx) - np.log(dt * W_init))**2
        
        return dist_cost + 1e8 * iso_error

    res = minimize(objective, x0=[ln_N_init, ln_T_init], 
                   bounds=[(np.log(100), np.log(100000)), (np.log(0.1), np.log(3600))])
    
    N_target = int(np.exp(res.x[0]))
    time_shift_range = np.exp(res.x[1])

elif strategy == "TRIPLE_OPTIMIZATION":

    # from scipy.optimize import minimize
    # 1. Tuning Knobs (Increase to make a variable "stiff")
    alpha_N = 1.0  # Cost of changing Node Count
    alpha_T = 5.0  # Cost of changing Time Window
    alpha_W = 2.5  # Cost of changing Velocity Scale
    alpha_V = 2.5  # Cost of missingt the target v_eff

    # 2. Initial State
    N_init = float(number_of_spatial_nodes)
    T_init = float(time_shift_range)
    W_init = float(scale_time)
    V3_const = baseline[1] # Metric Volume from Phase 1

    # 3. Objective Function
    def objective(params, use_veff_error = True):
        ln_N, ln_T, ln_W = params
        curr_N, curr_T, curr_W = np.exp(ln_N), np.exp(ln_T), np.exp(ln_W)
        
        # Geometry Math
        dx = (V3_const / curr_N)**(1/3)
        dt = (2.0 * curr_T) / (curr_N**0.25)
        
        # Weighted Mover's Distance (Relative % change)
        # We use the sensitivities to balance the 'effort'
        cost = (
            alpha_N * (ln_N - np.log(N_init))**2 + 
            alpha_T * (ln_T - np.log(T_init))**2 + 
            alpha_W * (ln_W - np.log(W_init))**2
        )
        

		# # 1. Expected spacing from Phase 1 volume
        # dx = (V3_const / curr_N)**(1/3) 
        # dt = (2.0 * curr_T) / (curr_N**0.25)
        # # 2. The predicted effective velocity
        # # We target this to be exactly equal to curr_W
        predicted_veff = dx / (dt + 1e-9)
        veff_error = (use_veff_error == True)*(alpha_V * 1e9 * (np.log(predicted_veff) - np.log(curr_W))**2)

        # Isotropy Constraint (Log-error ensures relative scaling)
        iso_error = 1e8 * (np.log(dx) - np.log(dt * curr_W))**2
        
        return cost + iso_error + veff_error

    # 4. Run Optimization
    res = minimize(objective, 
                   x0=[np.log(N_init), np.log(T_init), np.log(W_init)], 
                   bounds=[
                       (np.log(100), np.log(50000)), # N bounds
                       (np.log(1), np.log(300)),     # T bounds
                       (np.log(500), np.log(10000))  # W bounds
                   ],
                   method='L-BFGS-B')
    
    # 5. Map back to variables
    N_target = int(np.exp(res.x[0]))
    time_shift_range = np.exp(res.x[1])
    scale_time = np.exp(res.x[2])
    
    # Update global variable
    number_of_spatial_nodes = N_target




# --- SUMMARY TABLE (Pure Python) ---

# 1. Update the node count variable
number_of_spatial_nodes = N_target 
scale_time_effective = float(scale_time)

# 2. Final Metric Pass (Ground Truth)
metrics = compute_warped_expected_spacing(
    number_of_spatial_nodes, 
    lat_range=lat_range_extend, 
    lon_range=lon_range_extend,
    depth_range=depth_range, 
    time_range=time_shift_range,
    scale_time=scale_time, 
    depth_boost=depth_upscale_factor, 
    use_global=use_global
)

# Unpack using your exact variable names
Volume, Volume_space, Area, nominal_spacing, nominal_spacing_space, nominal_spacing_time = metrics

# Table Preparation
dt_init_m = ( (2.0 * T_init) / (N_init**0.25) ) * W_init
dt_final_m = nominal_spacing_time * scale_time

dx = (Volume_space / number_of_spatial_nodes)**(1/3)
dt = (2.0 * time_shift_range) / (number_of_spatial_nodes**0.25)
v_eff = dx / (dt + 1e-9)

headers = ["Metric", "Initial", "Final", "Delta %"]
rows = [
    ["Nodes (N)", f"{int(N_init):,}", f"{number_of_spatial_nodes:,}", f"{(number_of_spatial_nodes/N_init-1)*100:+.1f}%"],
    ["Window (±s)", f"{T_init:.2f}", f"{time_shift_range:.2f}", f"{(time_shift_range/T_init-1)*100:+.1f}%"],
    ["Scale (m/s)", f"{W_init:.1f}", f"{scale_time:.1f}", f"{(scale_time/W_init-1)*100:+.1f}%"],
    ["Time Res (m*)", f"{dt_init_m:.1f}", f"{dt_final_m:.1f}", f"{(dt_final_m/dt_init_m-1)*100:+.1f}%"],
    ["Space Res (m)", f"{baseline[4]:.1f}", f"{nominal_spacing_space:.1f}", f"{(nominal_spacing_space/baseline[4]-1)*100:+.1f}%"],
    ["Time Res (m*)", f"{dt_init_m:.1f}", f"{dt_final_m:.1f}", f"{(dt_final_m/dt_init_m-1)*100:+.1f}%"]
]

# Print Table
col_w = [18, 12, 12, 10]
print("\n" + "="*55)
print(f"      TRIPLE EQUILIBRIUM: {strategy}".center(55))
print("="*55)
print(f"{headers[0]:<{col_w[0]}} {headers[1]:<{col_w[1]}} {headers[2]:<{col_w[2]}} {headers[3]:<{col_w[3]}}")
print("-" * 55)
for r in rows:
    print(f"{r[0]:<{col_w[0]}} {r[1]:<{col_w[1]}} {r[2]:<{col_w[2]}} {r[3]:<{col_w[3]}}")
print("-" * 55)
print(f"Isotropy Ratio: {nominal_spacing_space / dt_final_m:.3f} (Ideal: 1.0)")
print("="*55 + "\n")
print(f"Velocity effective: {v_eff:.3f} m/s (Chosen scale_time: {scale_time:.3f} m/s)")

if number_of_spatial_nodes in [5, 25000]:
    print("!! WARNING: N hit boundary limits. Closeness to target restricted.")
print("="*55 + "\n")




## Move these steps to make_initial_files (and possibly the station projection as well; also, remove the saving of 1D velocity model; always base this on the config file)

## Make necessary directories
if os.path.isfile(path_to_file + '%s_process_days_list_ver_1.txt'%config["name_of_project"]) == 0:
	f = open(path_to_file + '%s_process_days_list_ver_1.txt'%config["name_of_project"], 'w')
	for j in range(5): ## Arbitrary 5 days to process
		f.write('%d/%d/%d \n'%(years[0], 1, j + 1)) ## Process days list (write which days want to process)
	f.close()

os.makedirs(path_to_file + 'Picks', exist_ok=True)
os.makedirs(path_to_file + 'Catalog', exist_ok=True)
os.makedirs(path_to_file + 'Calibration', exist_ok=True)
for year in years:
	os.makedirs(path_to_file + f'Picks/{year}', exist_ok=True)
	os.makedirs(path_to_file + f'Catalog/{year}', exist_ok=True)
	os.makedirs(path_to_file + f'Calibration/{year}', exist_ok=True)

os.makedirs(path_to_file + 'Plots', exist_ok=True)
os.makedirs(path_to_file + 'GNN_TrainedModels', exist_ok=True)
os.makedirs(path_to_file + 'Grids', exist_ok=True)
os.makedirs(path_to_file + '1D_Velocity_Models_Regional', exist_ok=True)

n_ver_velocity_model = 1
# seperator = '\\' if '\\' in path_to_file else '/'
shutil.copy(path_to_file + '1d_velocity_model.npz', path_to_file + '1D_Velocity_Models_Regional' + seperator + f'{config["name_of_project"]}_1d_velocity_model_ver_{n_ver_velocity_model}.npz')
os.makedirs(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'TravelTimeData', exist_ok=True)



## Create graph functions

def dlon_diff(lon_range):
	dlon_deg = (lon_range[1] - lon_range[0]) % 360
	if dlon_deg == 0 and lon_range[1] != lon_range[0]: 
		dlon_deg = 360 # Full circle
	# dlon = np.deg2rad(dlon_deg)
	return dlon_deg

def is_in_lon_range(lon, lon_min, lon_max):
    # Normalize everything to [0, 360]
    lon = lon % 360
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        return (lon >= lon_min) & (lon <= lon_max)
    else: # Crossing the seam
        return (lon >= lon_min) | (lon <= lon_max)



# def u_to_geodetic_lat(u_random, lat_range):
#     """
#     Converts a uniform random variable u [0, 1] to Geodetic Latitude 
#     using an Authalic (equal-area) mapping for the WGS84 ellipsoid.
#     """
#     # WGS84 Constant: e (eccentricity)
#     e = 0.0818191908426
    
#     def get_q(lat_deg):
#         # q is the authalic part of the projection
#         sin_lat = np.sin(np.deg2rad(lat_deg)) # sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
#         # sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
#         term1 = sin_lat / (1 - e**2 * sin_lat**2)
#         term2 = (1 / (2 * e)) * np.log((1 - e * sin_lat) / (1 + e * sin_lat))
#         return (1 - e**2) * (term1 - term2)

#     # Calculate q limits for your range
#     q_min = get_q(lat_range[0])
#     q_max = get_q(lat_range[1])
    
#     # Target q for this point
#     # pdb.set_trace()
#     q_target = q_min + u_random * (q_max - q_min)
    
#     # Map q back to Latitude (Approximation for WGS84)
#     # The difference between Geodetic and Authalic is very small, 
#     # so we can use a high-order series or a simple arcsin of (q / q_polar)
#     q_polar = get_q(90.0)
#     return np.rad2deg(np.arcsin(q_target / q_polar))


# def u_to_geodetic_lat(u_random, lat_range):
#     # WGS84 Constant: e (eccentricity)
#     e = 0.0818191908426
#     e2 = e**2
    
#     def get_q(lat_deg):
#         phi = np.deg2rad(lat_deg)
#         sin_phi = np.sin(phi)
#         # Standard Authalic series
#         term1 = sin_phi / (1 - e2 * sin_phi**2)
#         term2 = (1 / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
#         return (1 - e2) * (term1 - term2)

#     # 1. Map u to the Authalic space (q)
#     q_min = get_q(lat_range[0])
#     q_max = get_q(lat_range[1])
#     q_target = q_min + u_random * (q_max - q_min)
    
#     # 2. Convert q to Authalic Latitude (beta)
#     q_polar = get_q(90.0)
#     beta = np.arcsin(q_target / q_polar) # This is in Radians
    
#     # 3. Convert Authalic (beta) -> Geodetic (phi)
#     # Using the 3rd order series expansion for WGS84
#     phi = beta + (e2/3 + 31*e2**2/180 + 517*e2**3/5040) * np.sin(2*beta) + \
#                  (23*e2**2/360 + 251*e2**3/3780) * np.sin(4*beta) + \
#                  (761*e2**3/45360) * np.sin(6*beta)

#     return np.rad2deg(phi)


def u_to_geodetic_lat(u_random, lat_range):
    # WGS84 Constants
    e = 0.0818191908426215
    e2 = e**2
    
    def get_q(lat_deg):
        phi = np.deg2rad(lat_deg)
        s = np.sin(phi)
        # Using the standard USGS/Snyder form for q
        return (1 - e2) * ( (s / (1 - e2 * s**2)) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)) )

    # Step 1: Linear mapping in q-space (Area-preserving)
    q_min = get_q(lat_range[0])
    q_max = get_q(lat_range[1])
    q_target = q_min + u_random * (q_max - q_min)
    
    # Step 2: Inverse to Authalic Latitude (beta)
    q_polar = get_q(90.0)
    # Ensure numerical safety for arcsin
    beta = np.arcsin(np.clip(q_target / q_polar, -1.0, 1.0))
    
    # Step 3: Convert Authalic (beta) to Geodetic (phi)
    # WGS84 series coefficients
    P1 = (e2/3 + 31*e2**2/180 + 517*e2**3/5040)
    P2 = (23*e2**2/360 + 251*e2**3/3780)
    P3 = (761*e2**3/45360)
    
    phi_rad = beta + P1*np.sin(2*beta) + P2*np.sin(4*beta) + P3*np.sin(6*beta)
    return np.rad2deg(phi_rad)

def get_q_wgs84(lat_val):
    # WGS84 Constants
    e = 0.0818191908426215
    e2 = e**2
    
    def get_q(lat_deg):
        phi = np.deg2rad(lat_deg)
        s = np.sin(phi)
        # Using the standard USGS/Snyder form for q
        return (1 - e2) * ( (s / (1 - e2 * s**2)) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)) )

    # Step 1: Linear mapping in q-space (Area-preserving)
    q_val = get_q(lat_val)

    return q_val


def get_warped_metric_space(x_grid, depth_boost, scale_t, R_ref = earth_radius, scale_val = 10000.0, return_physical_units = False):
    """
    Unified 4D Metric Space for Global Gridding.
    Fixes: Vertical Shearing, Time-Space Coupling, and Radial Density Bias.
    """
    # 1. Physical Projection (True Geodetic to ECEF)
    # x_grid: [Lat, Lon, Depth, Time]
    xyz_ecef = ftrns1_abs(x_grid[:, :3])
    r_true = np.linalg.norm(xyz_ecef, axis=1, keepdims=True)
    
    # 2. Get Geodetic Normal (Straight 'Up' for WGS84)
    a, b = 6378137.0, 6356752.314245
    n = xyz_ecef / np.array([a**2, a**2, b**2])
    n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
    
    # 3. Apply Depth Boost along the Normal (The Anchor)
    # We move the point further from the surface to increase vertical resolution
    true_depths = x_grid[:, 2:3]
    xyz_boosted = xyz_ecef + (n_unit * true_depths * (depth_boost - 1.0))
    
    # 4. Apply Time Boost
    time_boosted = x_grid[:, 3:4] * scale_t
    
    # 5. Volume Warping (The 1/r Fix)
    # R_ref keeps the ratio near 1.0 for numerical stability
    radius_ratio = r_true / R_ref
    
    # Combine into 4D and warp the entire manifold
    p4d_boosted = np.hstack([xyz_boosted, time_boosted])
    p4d_warped = p4d_boosted / radius_ratio
    
    # 6. Precision Centering & Scaling
    # Centering on the batch prevents large-coordinate precision loss on GPU

    if return_physical_units == False:
	    origin = p4d_warped.mean(axis=0, keepdims=True)
	    p4d_scaled = (p4d_warped - origin) / scale_val
    else:
    	p4d_scaled = 1.0*p4d_warped

    return p4d_scaled


# def regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor, N_target = None, buffer_scale = 0.0, run_checks = False):

# 	if use_spherical == False:
# 		a = 6378137.0
# 		b = 6356752.314245
# 	else:
# 		a = 6371e3
# 		b = 6371e3

# 	if buffer_scale > 0.0:
# 		## Use a buffer around min-max regions. How to estimate? First estimate volume
# 		# Volume, Volume_space, _, nominal_spacing_space, nominal_spacing_time = compute_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
# 		Volume, Volume_space, Area_metric, _, nominal_spacing_space, nominal_spacing_time = compute_warped_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor)

# 		lat_mid = np.mean(lat_range)
# 		pad_lat = (nominal_spacing_space * buffer_scale / earth_radius) * (180 / np.pi)
# 		pad_lon = (nominal_spacing_space * buffer_scale / (earth_radius * np.cos(np.deg2rad(lat_mid)))) * (180 / np.pi) # Adjust lon padding for the convergence of meridians
# 		pad_depth = nominal_spacing_space * buffer_scale # 3. Calculate Depth and Time Padding
# 		pad_time = nominal_spacing_time * buffer_scale

# 		# 4. Define New Ranges
# 		expanded_lat = list(np.array([lat_range[0] - pad_lat, lat_range[1] + pad_lat]).clip(-90.0, 90.0)) if use_global == False else lat_range
# 		expanded_lon = [lon_range[0] - pad_lon, lon_range[1] + pad_lon] if use_global == False else lon_range
# 		expanded_depth = [depth_range[0] - pad_depth, depth_range[1] + pad_depth]
# 		expanded_time = time_range + pad_time # [time_range - pad_time, time_range + pad_time] # If time_range is half-width
# 		# Volume_expanded, Volume_space_expanded, _, _, _ = compute_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
# 		Volume_expanded, Volume_space_expanded, _, _, _, _ = compute_warped_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor)

# 		N_updated = int(np.ceil(N * (Volume_expanded/Volume)))
	
# 	# u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)
# 	# longitude = (2*np.pi*u[:,0] - np.pi)*180.0/np.pi
# 	# latitude = np.arccos(((a**2)*(1 - u[:,1]) + (b**2)*u[:,1] - a**2 + b**2) / (b**2 - a**2))

# 	# m = int(np.ceil(np.log2(N)))
# 	# initial_points = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random_base2(m = m)[0:N]


# 	if buffer_scale > 0.0:

# 		m = int(np.ceil(np.log2(N_updated)))
# 		N_sobol = 2**m

# 		u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random_base2(m)  # Sobol 4D
# 		assert((len(u) == N_sobol)*(len(u) >= N_updated))
# 		# assert(len(u) >= N_updated)

# 		if use_global == False:
# 			# phi = expanded_lon[0] + u[:,0]*(expanded_lon[1] - expanded_lon[0]) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
# 			phi = expanded_lon[0] + u[:,0]*dlon_diff(expanded_lon) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
# 			u_min = (1.0 + np.sin(np.deg2rad(expanded_lat[0])))/2.0
# 			u_max = (1.0 + np.sin(np.deg2rad(expanded_lat[1])))/2.0
# 			# theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
# 			# theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)
# 			theta = u_to_geodetic_lat(u[:,1], expanded_lat)

# 		else:
# 			phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
# 			# theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
# 			# theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)
# 			theta = u_to_geodetic_lat(u[:,1], [-90.0, 90.0])

# 		phi_wrapped = (phi + 180) % 360 - 180
# 		# r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
# 		# r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
# 		xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
# 		# r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)

# 		n = xyz_surface / np.array([a**2, a**2, b**2])
# 		n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
# 		# Local radius from center to surface point
# 		r_surface = np.linalg.norm(xyz_surface, axis=1, keepdims=True)
# 		# --- STEP B: Cubic Height Sampling ---
# 		# depth_range[0] is Top (+), depth_range[1] is Bottom (-)
# 		h_top = depth_range[0]
# 		h_bot = depth_range[1]
# 		# u is Sobol variable [0, 1]
# 		# We use the cubic formula to get 'h' that respects volume growth
# 		r_top = r_surface + h_top
# 		r_bot = r_surface + h_bot
# 		# Corrected height:
# 		h_sampled = (r_bot**3 + u[:, [2]] * (r_top**3 - r_bot**3))**(1/3.0) - r_surface
# 		# --- STEP C: Final Positioning ---
# 		# Move from the surface XYZ along the Normal by h_sampled
# 		xyz = xyz_surface + (n_unit * h_sampled)
# 		x_grid = ftrns2_abs(xyz)

# 		# r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
# 		# r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
# 		# xyz = (r*xyz_surface)/r_surface
# 		# x_grid = ftrns2_abs(xyz) 

# 		if use_time == True:
# 			t = -expanded_time + 2.0 * expanded_time * u[:, [3]]
# 			x_grid = np.concatenate((x_grid, t), axis = 1)

# 		if use_global == False:

# 			lons_wrapped = (x_grid[:,1] + 180) % 360 - 180
# 			mask_points = (x_grid[:,0] >= lat_range[0]) & (x_grid[:,0] <= lat_range[1]) & \
# 			                   is_in_lon_range(lons_wrapped, lon_range[0], lon_range[1]) & \
# 			                  (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
# 			                  (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range)) 

# 		else:

# 			# lons_wrapped = (x_grid[:,1] + 180) % 360 - 180
# 			mask_points = (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
# 			                  (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range)) 



# 		# mask_points = (x_grid[:,0] >= lat_range[0]) & (x_grid[:,0] <= lat_range[1]) & \
# 		#                   (lons_wrapped >= lon_range[0]) & (lons_wrapped <= lon_range[1]) & \
# 		#                   (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
# 		#                   (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range))


# 		## Now retain only the fraction of boundary nodes that will emulate the right density of the target number of nodes
# 		if N_target is not None:
# 			ratio = (Volume_expanded - Volume)/Volume
# 			n_boundary_retain = int(N_target*ratio)
# 			ichoose = np.concatenate((np.where(mask_points == 1)[0], np.random.choice(np.where(mask_points == 0)[0], \
# 				size = n_boundary_retain, replace = False)), axis = 0)
# 			x_grid = x_grid[ichoose]
# 			mask_points = mask_points[ichoose]


# 		# --- CORRECTED DENSITY SANITY CHECK ---
# 		if (N_target is not None)*(run_checks == True):
# 			# 1. The density the Core WILL have after FPS finishes
# 			target_density_core = N_target / Volume
# 			# 2. The density the Buffer HAS right now (the ghosts we are keeping)
# 			n_buffer_retained = np.sum(mask_points == 0)
# 			actual_density_buffer = n_buffer_retained / (Volume_expanded - Volume)
# 			# 3. The Ratio (Target is 1.0)
# 			# This proves the "Wall of Ghosts" matches the "Future Grid"
# 			density_ratio = actual_density_buffer / target_density_core
# 			print(f"--- FPS Ghost-Pressure Match ---")
# 			print(f"Target Core Nodes: {N_target}")
# 			print(f"Retained Ghosts:   {n_buffer_retained}")
# 			print(f"Expected Core Density: {target_density_core:.2e}")
# 			print(f"Actual Ghost Density:  {actual_density_buffer:.2e}")
# 			print(f"Pressure Match Ratio:  {density_ratio:.4f} (Ideal: 1.0000)")


# 		return x_grid, mask_points


# 	else: # buffer_scale == 1.0:

# 		u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)  # Sobol 4D
# 		if use_global == False:
# 			# phi = lon_range[0] + u[:,0]*(lon_range[1] - lon_range[0]) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
# 			phi = lon_range[0] + u[:,0]*dlon_diff(lon_range) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
# 			u_min = (1.0 + np.sin(np.deg2rad(lat_range[0])))/2.0
# 			u_max = (1.0 + np.sin(np.deg2rad(lat_range[1])))/2.0
# 			# theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
# 			# theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)
# 			theta = u_to_geodetic_lat(u[:,1], lat_range)

# 		else:
# 			phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
# 			# theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
# 			# theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)
# 			theta = u_to_geodetic_lat(u[:,1], [-90.0, 90.0])


# 		r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), depth_range[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
# 		r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), depth_range[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
# 		xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
# 		# r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
# 		# r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
# 		# xyz = (r*xyz_surface)/r_surface
# 		# x_grid = ftrns2_abs(xyz)

# 		n = xyz_surface / np.array([a**2, a**2, b**2])
# 		n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
# 		# Local radius from center to surface point
# 		r_surface = np.linalg.norm(xyz_surface, axis=1, keepdims=True)
# 		# --- STEP B: Cubic Height Sampling ---
# 		# depth_range[0] is Top (+), depth_range[1] is Bottom (-)
# 		h_top = depth_range[0]
# 		h_bot = depth_range[1]
# 		# u is Sobol variable [0, 1]
# 		# We use the cubic formula to get 'h' that respects volume growth
# 		r_top = r_surface + h_top
# 		r_bot = r_surface + h_bot
# 		# Corrected height:
# 		h_sampled = (r_bot**3 + u[:, [2]] * (r_top**3 - r_bot**3))**(1/3.0) - r_surface
# 		# --- STEP C: Final Positioning ---
# 		# Move from the surface XYZ along the Normal by h_sampled
# 		xyz = xyz_surface + (n_unit * h_sampled)
# 		x_grid = ftrns2_abs(xyz)


# 		if use_time == True:
# 			t = -time_shift_range + 2 * time_shift_range * u[:, [3]]
# 			x_grid = np.concatenate((x_grid, t), axis = 1)

# 		return x_grid



def get_ellipsoid_paddings(lat_min, lat_max, buffer_m):
    # WGS84 Constants
    a = 6378137.0
    e2 = 0.00669437999014
    
    def get_radii(lat_deg):
        phi = np.deg2rad(np.clip(lat_deg, -89.9999, 89.9999)) # # lat_deg np.clip(lat_deg, -89.9999, 89.9999)
        denom = (1.0 - e2 * np.sin(phi)**2)**1.5
        M = a * (1.0 - e2) / denom  # Radius for Latitude
        N = a / np.sqrt(1.0 - e2 * np.sin(phi)**2) # Radius for Longitude
        return M, N

    # Calculate radii at both boundaries
    M_min, N_min = get_radii(lat_min)
    M_max, N_max = get_radii(lat_max)

    # Latitude padding (Degrees) - can vary slightly min vs max
    pad_lat_min = np.rad2deg(buffer_m / M_min)
    pad_lat_max = np.rad2deg(buffer_m / M_max)

    # Longitude padding (Degrees) - varies significantly min vs max
    pad_lon_min = np.rad2deg(buffer_m / (N_min * np.cos(np.deg2rad(lat_min))))
    pad_lon_max = np.rad2deg(buffer_m / (N_max * np.cos(np.deg2rad(lat_max))))

    return (pad_lat_min, pad_lat_max), (pad_lon_min, pad_lon_max)

# --- Implementation ---
# (pLat_min, pLat_max), (pLon_min, pLon_max) = get_ellipsoid_paddings(lat_range[0], lat_range[1], spatial_buffer_m)

# expanded_lat = [lat_range[0] - pLat_min, lat_range[1] + pLat_max]
# We take the larger Lon pad to ensure the entire physical buffer is covered
# max_pLon = max(pLon_min, pLon_max) 
# expanded_lon = [lon_range[0] - max_pLon, lon_range[1] + max_pLon]


def regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor, N_target = None, buffer_scale = 0.0, run_checks = False):

	if use_spherical == False:
		a = 6378137.0
		b = 6356752.314245
	else:
		a = 6371e3
		b = 6371e3

	if buffer_scale > 0.0:
		## Use a buffer around min-max regions. How to estimate? First estimate volume
		# Volume, Volume_space, _, nominal_spacing_space, nominal_spacing_time = compute_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
		Volume, Volume_space, Area_metric, _, nominal_spacing_space, nominal_spacing_time = compute_warped_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor)

		# lat_mid = np.mean(lat_range)
		spatial_buffer_m = nominal_spacing_space * buffer_scale  # / earth_radius)
		(pLat_min, pLat_max), (pLon_min, pLon_max) = get_ellipsoid_paddings(lat_range[0], lat_range[1], spatial_buffer_m)
		max_pLon = max(pLon_min, pLon_max)
		pad_depth = nominal_spacing_space * buffer_scale # 3. Calculate Depth and Time Padding
		pad_time = nominal_spacing_time * buffer_scale

		# pad_lat = (nominal_spacing_space * buffer_scale / earth_radius) * (180 / np.pi)
		# pad_lon = (nominal_spacing_space * buffer_scale / (earth_radius * np.cos(np.deg2rad(lat_mid)))) * (180 / np.pi) # Adjust lon padding for the convergence of meridians

		# 4. Define New Ranges
		expanded_lat = list(np.array([lat_range[0] - pLat_min, lat_range[1] + pLat_max]).clip(-90.0, 90.0)) if use_global == False else lat_range
		expanded_lon = [lon_range[0] - max_pLon, lon_range[1] + max_pLon] if use_global == False else lon_range
		# expanded_lat = list(np.array([lat_range[0] - pad_lat, lat_range[1] + pad_lat]).clip(-90.0, 90.0)) if use_global == False else lat_range
		# expanded_lon = [lon_range[0] - pad_lon, lon_range[1] + pad_lon] if use_global == False else lon_range
		expanded_depth = [depth_range[0] - pad_depth, depth_range[1] + pad_depth]
		expanded_time = time_range + pad_time # [time_range - pad_time, time_range + pad_time] # If time_range is half-width
		# Volume_expanded, Volume_space_expanded, _, _, _ = compute_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
		Volume_expanded, Volume_space_expanded, _, _, _, _ = compute_warped_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor)

		N_updated = int(np.ceil(N * (Volume_expanded/Volume)))
	
	# u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)
	# longitude = (2*np.pi*u[:,0] - np.pi)*180.0/np.pi
	# latitude = np.arccos(((a**2)*(1 - u[:,1]) + (b**2)*u[:,1] - a**2 + b**2) / (b**2 - a**2))

	# m = int(np.ceil(np.log2(N)))
	# initial_points = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random_base2(m = m)[0:N]


	if buffer_scale > 0.0:

		m = int(np.ceil(np.log2(N_updated)))
		N_sobol = 2**m

		u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random_base2(m)  # Sobol 4D
		assert((len(u) == N_sobol)*(len(u) >= N_updated))
		# assert(len(u) >= N_updated)

		if use_global == False:
			# phi = expanded_lon[0] + u[:,0]*(expanded_lon[1] - expanded_lon[0]) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
			phi = expanded_lon[0] + u[:,0]*dlon_diff(expanded_lon) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
			u_min = (1.0 + np.sin(np.deg2rad(expanded_lat[0])))/2.0
			u_max = (1.0 + np.sin(np.deg2rad(expanded_lat[1])))/2.0
			# theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
			# theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)
			theta = u_to_geodetic_lat(u[:,1], expanded_lat)

		else:
			phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
			# theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
			# theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)
			theta = u_to_geodetic_lat(u[:,1], [-90.0, 90.0])

		phi_wrapped = (phi + 180) % 360 - 180
		# r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
		# r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
		xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
		# r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)

		n = xyz_surface / np.array([a**2, a**2, b**2])
		n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
		# Local radius from center to surface point
		r_surface = np.linalg.norm(xyz_surface, axis=1, keepdims=True)
		# --- STEP B: Cubic Height Sampling ---
		# depth_range[0] is Top (+), depth_range[1] is Bottom (-)
		h_top = depth_range[0]
		h_bot = depth_range[1]
		# u is Sobol variable [0, 1]
		# We use the cubic formula to get 'h' that respects volume growth
		r_top = r_surface + h_top
		r_bot = r_surface + h_bot
		# Corrected height:
		h_sampled = (r_bot**3 + u[:, [2]] * (r_top**3 - r_bot**3))**(1/3.0) - r_surface
		# --- STEP C: Final Positioning ---
		# Move from the surface XYZ along the Normal by h_sampled
		xyz = xyz_surface + (n_unit * h_sampled)
		x_grid = ftrns2_abs(xyz)

		# r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
		# r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
		# xyz = (r*xyz_surface)/r_surface
		# x_grid = ftrns2_abs(xyz) 

		if use_time == True:
			t = -expanded_time + 2.0 * expanded_time * u[:, [3]]
			x_grid = np.concatenate((x_grid, t), axis = 1)

		if use_global == False:

			lons_wrapped = (x_grid[:,1] + 180) % 360 - 180
			mask_points = (x_grid[:,0] >= lat_range[0]) & (x_grid[:,0] <= lat_range[1]) & \
			                   is_in_lon_range(lons_wrapped, lon_range[0], lon_range[1]) & \
			                  (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
			                  (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range)) 

		else:

			# lons_wrapped = (x_grid[:,1] + 180) % 360 - 180
			mask_points = (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
			                  (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range)) 



		# mask_points = (x_grid[:,0] >= lat_range[0]) & (x_grid[:,0] <= lat_range[1]) & \
		#                   (lons_wrapped >= lon_range[0]) & (lons_wrapped <= lon_range[1]) & \
		#                   (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
		#                   (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range))


		## Now retain only the fraction of boundary nodes that will emulate the right density of the target number of nodes
		if N_target is not None:
			ratio = (Volume_expanded - Volume)/Volume
			n_boundary_retain = int(N_target*ratio)
			ichoose = np.concatenate((np.where(mask_points == 1)[0], np.random.choice(np.where(mask_points == 0)[0], \
				size = n_boundary_retain, replace = False)), axis = 0)
			x_grid = x_grid[ichoose]
			mask_points = mask_points[ichoose]


		# --- CORRECTED DENSITY SANITY CHECK ---
		if (N_target is not None)*(run_checks == True):
			# 1. The density the Core WILL have after FPS finishes
			target_density_core = N_target / Volume
			# 2. The density the Buffer HAS right now (the ghosts we are keeping)
			n_buffer_retained = np.sum(mask_points == 0)
			actual_density_buffer = n_buffer_retained / (Volume_expanded - Volume)
			# 3. The Ratio (Target is 1.0)
			# This proves the "Wall of Ghosts" matches the "Future Grid"
			density_ratio = actual_density_buffer / target_density_core
			print(f"--- FPS Ghost-Pressure Match ---")
			print(f"Target Core Nodes: {N_target}")
			print(f"Retained Ghosts:   {n_buffer_retained}")
			print(f"Expected Core Density: {target_density_core:.2e}")
			print(f"Actual Ghost Density:  {actual_density_buffer:.2e}")
			print(f"Pressure Match Ratio:  {density_ratio:.4f} (Ideal: 1.0000)")


		return x_grid, mask_points


	else: # buffer_scale == 1.0:

		u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)  # Sobol 4D
		if use_global == False:
			# phi = lon_range[0] + u[:,0]*(lon_range[1] - lon_range[0]) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
			phi = lon_range[0] + u[:,0]*dlon_diff(lon_range) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
			u_min = (1.0 + np.sin(np.deg2rad(lat_range[0])))/2.0
			u_max = (1.0 + np.sin(np.deg2rad(lat_range[1])))/2.0
			# theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
			# theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)
			theta = u_to_geodetic_lat(u[:,1], lat_range)

		else:
			phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
			# theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
			# theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)
			theta = u_to_geodetic_lat(u[:,1], [-90.0, 90.0])


		r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), depth_range[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
		r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), depth_range[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
		xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
		# r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
		# r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
		# xyz = (r*xyz_surface)/r_surface
		# x_grid = ftrns2_abs(xyz)

		n = xyz_surface / np.array([a**2, a**2, b**2])
		n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
		# Local radius from center to surface point
		r_surface = np.linalg.norm(xyz_surface, axis=1, keepdims=True)
		# --- STEP B: Cubic Height Sampling ---
		# depth_range[0] is Top (+), depth_range[1] is Bottom (-)
		h_top = depth_range[0]
		h_bot = depth_range[1]
		# u is Sobol variable [0, 1]
		# We use the cubic formula to get 'h' that respects volume growth
		r_top = r_surface + h_top
		r_bot = r_surface + h_bot
		# Corrected height:
		h_sampled = (r_bot**3 + u[:, [2]] * (r_top**3 - r_bot**3))**(1/3.0) - r_surface
		# --- STEP C: Final Positioning ---
		# Move from the surface XYZ along the Normal by h_sampled
		xyz = xyz_surface + (n_unit * h_sampled)
		x_grid = ftrns2_abs(xyz)


		if use_time == True:
			t = -time_shift_range + 2 * time_shift_range * u[:, [3]]
			x_grid = np.concatenate((x_grid, t), axis = 1)

		return x_grid



def farthest_point_sampling(points_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = None, device = device):
    
    """
    points: [N, 3] or [N, 4] (already scaled/transformed)
    target_N: Number of 'real' points to collect
    mask_candidates: Tensor/Array of 1s (real) and 0s (buffer/mirrored)
    """    

    # scale_val = 10000.0
    points = points_candidates.copy()
    # points, radii = get_metric_space(points, depth_boost, scale_time)
    points = get_warped_metric_space(points, depth_boost, scale_time)


    # if points.shape[1] == 4: points[:, 3] *= scale_time
    # if depth_boost != 1.0: points = ftrns1_abs(ftrns2_abs(points) * np.array([1.0, 1.0, depth_boost, 1.0]))
    # origin = points[:, :3].mean(axis = 0, keepdims = True)
    # points[:, :3] -= origin
    points = torch.as_tensor(points, device = device, dtype = torch.float64)
    # radii = torch.as_tensor(radii / scale_val, device = device, dtype = torch.float64)

    if mask_candidates is None: mask_candidates = np.ones(len(points))
    mask = torch.as_tensor(mask_candidates, device = device, dtype = torch.bool)
    assert(len(mask) == len(points))
    assert(mask.sum().item() >= target_N)
    N, C = points.shape
    
    # 1. Initialize distance array
    # If we have boundary points (mask == 0), we pre-calculate distances to them
    distance = torch.full((N,), float('inf'), device = device, dtype = torch.float64)
    
    boundary_indices = torch.where(~mask)[0]
    real_indices = torch.where(mask)[0]

    # 2. Pre-process boundary Points (The repulsion field)
    if len(boundary_indices) > 0:
        # Optimization: Update distance array with the proximity to any ghost point
        # For very large N, we do this in chunks to avoid OOM
        for i in range(0, len(boundary_indices), 500):
            batch = boundary_indices[i:i+500]
            # dists shape: [len(batch), N]
            dists = torch.cdist(points[batch], points, p=2)**2
            distance = torch.min(distance, torch.min(dists, dim=0)[0])
        
        # Ensure ghost points themselves are never selected
        distance[boundary_indices] = -1.0

    # 3. Choose the first REAL point
    # Instead of random, we pick the point farthest from the boundary ghosts
    # If no ghosts exist, we default to the point closest to the centroid
    if len(boundary_indices) > 0:
        farthest = torch.argmax(distance).item()
    else:
        centroid = points[real_indices].mean(0, keepdims=True)
        dist_centroid = torch.sum((points[real_indices] - centroid)**2, dim=1)
        farthest = real_indices[torch.argmin(dist_centroid)].item()

    collected_indices = []
    cnt_found = 0

    # 4. Main FPS Loop
    while cnt_found < target_N:
        collected_indices.append(farthest)
        cnt_found += 1 # We only ever pick real points now
            
        centroid_pt = points[farthest, :].view(1, C)
        dist = torch.sum((points - centroid_pt) ** 2, dim=-1)
        
        distance = torch.min(distance, dist)
        distance[farthest] = -1.0
        
        if cnt_found < target_N:
            farthest = torch.argmax(distance).item()

    # Final Filter
    final_indices = torch.tensor(collected_indices, device=device)
    # return ftrns2_abs(points_candidates[final_indices.cpu().numpy()])
    return points_candidates[final_indices.cpu().numpy()]



# use_poisson_filtering = False 
# use_farthest_point_filtering = True 


def get_wgs84_area_val(lat_deg):

    """Computes the WGS84 area-proportional value for a given latitude."""
    lat_rad = np.deg2rad(lat_deg)
    e2 = 0.00669437999014  # WGS84 eccentricity squared
    e = np.sqrt(e2)
    sin_phi = np.sin(lat_rad)
    sin_phi = np.clip(sin_phi, -0.999999, 0.999999)

    # Standard formula for ellipsoidal surface area relative to latitude
    term1 = (1 - e2) * sin_phi / (1 - e2 * sin_phi**2)
    term2 = ((1 - e2) / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
    return term1 - term2
   

def get_natural_scale_t(Volume_space, time_range_total, target_V_eff=8000.0):
    """
    Volume_space: from your compute_expected_spacing (in m^3)
    time_range_total: full width of time (e.g., 500s)
    target_V_eff: meters per second (seismic P-wave speed approx 8000 m/s)
    """
    # Characteristic spatial length (L = V^(1/3))
    L_space = Volume_space**(1.0/3.0)
    
    # We want (scale_t * time) to be comparable to L_space.
    # However, to be physically resonant, scale_t should be near the velocity 
    # that 'connects' the dimensions.
    
    # Method A: Isotropic (Geometry only)
    isotropic_scale = L_space / (time_range_total + 1e-9)
    
    # Method B: Physical (Seismic Velocity)
    # This is often the 'sweet spot' for seismic GNNs
    physical_scale = target_V_eff 
    
    # Suggest an upper bound that allows the optimizer to find either 
    # a geometry-driven or physics-driven balance.
    upper_bound = max(isotropic_scale, physical_scale) * 5.0
    return upper_bound



def get_simple_density_ratio(normalized_data, N, frac_edge=0.1):
    num_bins = int(1.0 / frac_edge)
    # Histogram on the [0, 1] range
    counts, _ = np.histogram(normalized_data, bins=num_bins, range=(0.0, 1.0))
    
    expected_per_bin = N / num_bins
    
    # Boundary Ratio: (Average of both edge bins) / (Expected)
    # A value of 1.0 means the edges have the same density as the bulk.
    boundary_ratio = (counts[0] + counts[-1]) / (2 * expected_per_bin + 1e-9)
    
    return boundary_ratio, counts

def check_boundary_densities(x_grid, lat_range, lon_range, depth_range, time_range, use_global = use_global):

    N = len(x_grid)
    boundary_health = {}
    e = 0.0818191908426  # WGS84 Eccentricity

    def get_q_wgs84(lat_deg):
        sin_phi = np.sin(np.deg2rad(lat_deg))
        sin_phi = np.clip(sin_phi, -0.999999, 0.999999)
        t1 = sin_phi / (1 - e**2 * sin_phi**2)
        t2 = (1 / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
        return (1 - e**2) * (t1 - t2)

    # --- 1. LATITUDE (Authalic Transformation) ---
    q_actual = get_q_wgs84(x_grid[:, 0])
    q_min = get_q_wgs84(lat_range[0])
    q_max = get_q_wgs84(lat_range[1])
    val_lat = (q_actual - q_min) / (q_max - q_min + 1e-12)

    # --- 2. DEPTH (Volumetric Transformation) ---
    r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
    r_bound_0 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[0])]), axis=1)
    r_bound_1 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[1])]), axis=1)
    
    r_min_i = np.minimum(r_bound_0, r_bound_1)
    r_max_i = np.maximum(r_bound_0, r_bound_1)
    val_depth = (r_actual**3 - r_min_i**3) / (r_max_i**3 - r_min_i**3 + 1e-12)

    # --- 3. TIME & LON (Linear Transformation) ---
    val_time = (x_grid[:, 3] - (-time_range)) / (2 * time_range + 1e-12)
    
    if not use_global:
        val_lon = (x_grid[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0] + 1e-12)
    else:
        val_lon = None # Skip boundary check for global periodic lon

    # --- 4. UNIFIED DENSITY CALCULATION ---
    processed_dims = {
        'Lat': val_lat,
        'Lon': val_lon,
        'Depth': val_depth, 
        'Time': val_time, 
    }

    for name, val in processed_dims.items():
        if val is None:
            boundary_health[name] = 1.0
            continue
            
        # Every dimension now uses the same simple "flat" check
        ratio, counts = get_simple_density_ratio(val, N)
        boundary_health[name] = ratio
        
    return boundary_health




def compute_cdf_analysis(x_grid, ranges):

    N = len(x_grid)
    empirical_cdf = np.linspace(0, 1, N)
    cdf_loss = 0
    
    for i in range(x_grid.shape[1]):

        if i == 0: # LATITUDE
            q_vals = get_wgs84_area_val(x_grid[:, 0])
            q_bounds = sorted([get_wgs84_area_val(ranges[0][0]), get_wgs84_area_val(ranges[0][1])])
            val = (q_vals - q_bounds[0]) / (q_bounds[1] - q_bounds[0] + 1e-12)

        elif i == 1: # LONGITUDE
            # For longitude, we use dlon_diff to handle the wrap-around correctly
            val = (x_grid[:, 1] - ranges[1][0]) / (dlon_diff([ranges[1][0], ranges[1][1]]) + 1e-12)

        elif i == 2: # DEPTH
            # Get actual distances from geocenter
            r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
            
            # Project the bounds to absolute radii
            r_b1 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, ranges[2][0])]), axis=1)
            r_b2 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, ranges[2][1])]), axis=1)
            
            # Use absolute min/max to ensure (r_max^3 - r_min^3) is always positive
            r_min = np.minimum(r_b1, r_b2)
            r_max = np.maximum(r_b1, r_b2)
            
            val = (r_actual**3 - r_min**3) / (r_max**3 - r_min**3 + 1e-12)

        elif i == 3: # TIME
            val = (x_grid[:, 3] - ranges[3][0]) / (ranges[3][1] - ranges[3][0] + 1e-12)

        # CRITICAL: Clip to [0, 1] to prevent the optimizer from chasing outliers
        val = np.clip(val, 0.0, 1.0)
        
        sorted_vals = np.sort(val)
        cdf_loss += np.mean(np.abs(sorted_vals - empirical_cdf))

    return cdf_loss / x_grid.shape[1]


class SamplingTuner:

    def __init__(self, target_N, lat_range, lon_range, depth_range, time_range, use_global = use_global, device = device):

        from skopt.space import Real
        self.target_N = target_N
        # Store ranges as [min, max] pairs
        self.ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
        self.device = device
        self.time_range = time_range
        self.use_global = use_global
        
        # 1. Define Search Space
        # scale_t: km/s
        # depth_boost: dimensionless vertical stretch
        # buffer_scale: multiplier for the nominal spacing
        self.space = [
            Real(1e3, 15e3, prior = 'log-uniform', name='scale_t'),      
            Real(1.0, 3.0, name='depth_boost'),   
            Real(0.5, 2.5, name='buffer_scale')    # prior='log-uniform',
        ]

    def optimize(self, n_calls = 90):

        """Runs Bayesian Optimization to find the triplet of parameters."""


        @use_named_args(self.space)
        def objective(scale_t, depth_boost, buffer_scale):
            # 1. GENERATE CANDIDATES
            # We use a high up-sample factor to give FPS a dense pool to pick from
            up_sample_factor = 20 if use_time_shift else 10
            number_candidate_nodes = up_sample_factor * self.target_N        

            trial_points, mask_points = regular_sobolov(
                number_candidate_nodes, 
                lat_range=self.ranges[0], 
                lon_range=self.ranges[1], 
                depth_range=self.ranges[2], 
                time_range=self.time_range, 
                use_time=use_time_shift, 
                use_global=self.use_global, 
                scale_time=scale_t, 
                N_target=self.target_N, 
                buffer_scale=buffer_scale
            )                 

            # 2. RUN FPS (Physical -> Scaled Search Space)
            x_grid = farthest_point_sampling(
                trial_points, 
                self.target_N, 
                scale_time=scale_t, 
                depth_boost=depth_boost, 
                mask_candidates=mask_points
            )                 

            # 3. COMPUTE 4D METRICS (Scaled Space)
            # Get the warped metric for local regularity checks
            x_proj_scaled = get_warped_metric_space(x_grid, depth_boost, scale_t)
            tree = cKDTree(x_proj_scaled)
            dist_scaled, idx_nn = tree.query(x_proj_scaled, k=2)
            nn_dist_scaled = dist_scaled[:, 1]        

            x_proj_scaled_abs = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units = True)
            tree_abs = cKDTree(x_proj_scaled_abs)
            dist_scaled_abs, idx_nn_abs = tree_abs.query(x_proj_scaled_abs, k=2)
            nn_dist_scaled_abs = dist_scaled_abs[:, 1]
            
            # CV: Local Regularity (Lower is better)
            cv = np.std(nn_dist_scaled) / (np.mean(nn_dist_scaled) + 1e-9)        

            # 4. COMPUTE VOLUME EFFICIENCY (Normalized Mean)
            # Get physical volume for the density expectation
            metrics = compute_warped_expected_spacing(
                self.target_N, 
                lat_range=self.ranges[0], lon_range=self.ranges[1],
                depth_range=self.ranges[2], time_range=self.time_range,
                scale_time=scale_t, depth_boost=depth_boost, 
                use_global=self.use_global
            )
            volume_4d_warped = metrics[0]
            
            # 0.463 is the 4D Poisson constant for a hypersphere
            expected_mean_scaled = 0.463 * (volume_4d_warped / self.target_N)**(0.25)
            norm_mean = np.mean(nn_dist_scaled_abs) / expected_mean_scaled
            
            # Hinge Loss: Penalty only if below the 1.4 "Physical Floor"
            penalty_norm = max(0, 1.4 - norm_mean) ** 2        

            # 5. COMPUTE ACTUAL GRAPH VELOCITY (The Balance Check)
            # x_grid is [Lat, Lon, Depth, Time]
            nn_indices = idx_nn[:, 1]
            x_neighbors = x_grid[nn_indices]
            
            # Spatial distance (km) using the ellipsoidal transformation
            dx_vec = ftrns1_abs(x_grid[:, :3]) - ftrns1_abs(x_neighbors[:, :3])
            dist_space_km = np.linalg.norm(dx_vec, axis=1) / 1000.0
            
            # Temporal distance (seconds)
            dt_sec = np.abs(x_grid[:, 3] - x_neighbors[:, 3])
            
            # Median velocity of neighbors (km/s)
            actual_v = np.median(dist_space_km / (dt_sec + 1e-9))
            target_v = scale_t / 1000.0
            loss_isotropy = (np.log10(actual_v / target_v)) ** 2        

            # 6. BOUNDARY & GEOGRAPHY HEALTH
            densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.time_range, self.use_global)
            
            # Total Boundary Bias (Squared to penalize large drifts like your 1.24 depth bias)
            penalty_boundaries = (
                (1.0 - densities['Depth'])**2 + 
                (1.0 - densities['Time'])**2 + 
                (1.0 - densities['Lat'])**2 + 
                (1.0 - densities['Lon'])**2
            )
            
            # CDF Loss: Sub-meter transparency check
            cdf_loss = compute_cdf_analysis(x_grid, self.ranges)        

            # 7. REGULARIZATION
            # Prevents BO from "cheating" with extreme depth_boost or scale_t
            reg_penalty = (
                0.10 * (np.log10(scale_t / scale_time_effective)**2) + 
                0.50 * (depth_boost - 1.0)**2
            )        

            # FINAL WEIGHTED LOSS
            return (
                1.0 * cv +                  # Local smoothness
                2.0 * penalty_norm +        # 4D Volume floor
                2.0 * loss_isotropy +       # Graph physical balance
                5.0 * cdf_loss +            # WGS84 Geodetic truth
                2.0 * penalty_boundaries +  # Edge density balance
                1.0 * reg_penalty)           # Parameter stability


        # res = gp_minimize(objective, self.space, n_calls = n_calls, n_initial_points = 5, initial_point_generator = 'sobol', verbose = True) # random_state = 42
        res = gp_minimize(objective, self.space, n_calls = n_calls, verbose = True) # random_state = 42

        best_scale_t, best_depth_boost, best_buffer_scale = res.x
        

        _, _, _, nominal_spacing_4d, _, _ = compute_warped_expected_spacing(
        	self.target_N,  # total number of points
        	lat_range=self.ranges[0],
        	lon_range=self.ranges[1],
        	depth_range=self.ranges[2],
        	time_range=self.time_range,  # T, full range = 2T
        	use_time=use_time_shift,
        	scale_time=best_scale_t,  # w_scale: length per unit time
        	use_global=use_global,
        	earth_radius=earth_radius)

        # nominal_spacing = (total_vol / self.target_N)**(1/4)
        # buffer_width_phys = best_buffer_scale * nominal_spacing
        buffer_width_phys = best_buffer_scale * nominal_spacing_4d # [0]

        print("\n--- Optimization Results ---")
        print(f"Optimal scale_t:     {best_scale_t:.3f} m/s")
        print(f"Optimal depth_boost: {best_depth_boost:.3f}")
        print(f"Optimal buffer_scale: {best_buffer_scale:.3f}")
        print(f"Effective Padding:   {buffer_width_phys:.2f} (units)")

        return {
            'scale_t': best_scale_t,
            'depth_boost': best_depth_boost,
            'buffer_scale': best_buffer_scale,
            'buffer_width_phys': buffer_width_phys
        }






        # @use_named_args(self.space)
        # def objective(scale_t, depth_boost, buffer_scale):

        #     # 1. GENERATE CANDIDATES
        #     up_sample_factor = 20 if use_time_shift else 10
        #     number_candidate_nodes = up_sample_factor * self.target_N

        #     trial_points, mask_points = regular_sobolov(
        #         number_candidate_nodes, 
        #         lat_range=self.ranges[0], 
        #         lon_range=self.ranges[1], 
        #         depth_range=self.ranges[2], 
        #         time_range=self.time_range, 
        #         use_time=use_time_shift, 
        #         use_global=self.use_global, 
        #         scale_time=scale_t, 
        #         N_target=self.target_N, 
        #         buffer_scale=buffer_scale
        #     )        

        #     x_grid = farthest_point_sampling(
        #         trial_points, 
        #         self.target_N, 
        #         scale_time=scale_t, 
        #         depth_boost=depth_boost, 
        #         mask_candidates=mask_points
        #     )       

        #     # 3. PROJECT TO SCALED METRIC SPACE (For CV and Anisotropy)
        #     # Use depth_boost on the 3rd column and scale_t on the 4th
        #     # scaling_vector = np.array([1.0, 1.0, depth_boost, scale_t])
        #     # x_proj_scaled = ftrns1_abs(x_grid * scaling_vector)        

        #     x_proj_scaled = get_warped_metric_space(x_grid, depth_boost, scale_t)
        #     # --- PART A: Uniformity (Scaled World) ---
        #     tree = cKDTree(x_proj_scaled)
        #     nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
        #     cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)    

        #     # 1. Use the warped metric for distances
	    #     x_metric_4d1 = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units=True)
	    #     tree_4d = cKDTree(x_metric_4d1)
	    #     dist_4d1, idx_nn1 = tree_4d.query(x_metric_4d1, k=2)
	    #     nn_4d1 = dist_4d1[:, 1]
	    #     nn_indices1 = idx_nn1[:, 1]

	    #     # 2. Final Metric Pass (Ground Truth)
	    #     metrics = compute_warped_expected_spacing(
	    #         self.target_N, 
	    #         lat_range=self.ranges[0], 
	    #         lon_range=self.ranges[1],
	    #         depth_range=self.ranges[2], 
	    #         time_range=self.time_range,
	    #         scale_time=scale_t, 
	    #         depth_boost=depth_boost, 
	    #         use_global=self.use_global
	    #     )

	    #     # Unpack using your exact variable names
	    #     volume_4d_warped, _, _, _, _, _ = metrics
	    #     expected_mean = 0.463 * (volume_4d_warped / self.target_N)**(0.25)
	    #     norm_mean = np.mean(nn_4d1) / expected_mean
	    #     penalty_norm = max(0, 1.4 - norm_mean) ** 2
	    #     # cv = cv + 1.0*penalty_norm

        #     densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3][1], self.use_global)
        #     # Weighted penalty: we care most about Depth (Surface) and Time boundaries
        #     penalty_boundary = (abs(1.0 - densities['Depth']) * 1.0 +  ## Make these weights proportional to volume
        #                abs(1.0 - densities['Time']) * 1.0 + 
        #                abs(1.0 - densities['Lat']) * 1.0 + 
        #                abs(1.0 - densities['Lon']) * 1.0)/4.0

        #     # 4. CDF ANALYSIS (The WGS84 "Gold Standard")
        #     cdf_loss = compute_cdf_analysis(x_grid, self.ranges)


        #     # --- THE FINAL LOSS COMPOSITION ---            

        #     # A. Local Regularity (Lower is better)
        #     loss_regularity = cv             

        #     # B. Volume Health (Penalty only if it drops below the physical floor)
        #     # Once norm_mean hits 1.4, this penalty becomes 0.0 and stops "pushing"
        #     penalty_norm = max(0, 1.4 - norm_mean) ** 2            

        #     # C. Velocity Alignment (Log-space makes 0.5x and 2.0x errors equally expensive)
        #     # actual_graph_velocity is in km/s, scale_t is in m/s
        #     target_v = scale_t / 1000.0
        #     loss_isotropy = (np.log10(actual_graph_velocity / target_v)) ** 2            

        #     # D. WGS84 Transparency
        #     loss_transparency = cdf_loss # High weight to ensure ellipsoidal truth            

        #     # E. Boundary Pressure (Targeting that 1.24 Depth Bias)
        #     loss_boundaries = (abs(1.0 - densities['Depth']))**2 + (abs(1.0 - densities['Time']))**2            

        #     # --- Final Weighted Sum ---
        #     return (
        #         1.0 * loss_regularity + 
        #         2.0 * penalty_norm +     # "The Floor"
        #         1.5 * loss_isotropy +     # "The Balance"
        #         5.0 * loss_transparency + # "The Geography"
        #         2.0 * loss_boundaries +   # "The Edges"
        #         1.0 * reg_penalty
        #     )



        # @use_named_args(self.space)
        # def objective(scale_t, depth_boost, buffer_scale, use_normalized_mean = True):

        #     # 1. GENERATE CANDIDATES
        #     up_sample_factor = 20 if use_time_shift else 10
        #     number_candidate_nodes = up_sample_factor * self.target_N

        #     trial_points, mask_points = regular_sobolov(
        #         number_candidate_nodes, 
        #         lat_range=self.ranges[0], 
        #         lon_range=self.ranges[1], 
        #         depth_range=self.ranges[2], 
        #         time_range=self.time_range, 
        #         use_time=use_time_shift, 
        #         use_global=self.use_global, 
        #         scale_time=scale_t, 
        #         N_target=self.target_N, 
        #         buffer_scale=buffer_scale
        #     )        

        #     # 2. RUN FPS (Physical -> Scaled Search Space)
        #     # fps returns the selected physical [Lat, Lon, Depth, Time] points
        #     # x_grid = farthest_point_sampling(
        #     #     ftrns1_abs(trial_points), 
        #     #     self.target_N, 
        #     #     scale_time=scale_t, 
        #     #     depth_boost=depth_boost, 
        #     #     mask_candidates=mask_points
        #     # )        

        #     x_grid = farthest_point_sampling(
        #         trial_points, 
        #         self.target_N, 
        #         scale_time=scale_t, 
        #         depth_boost=depth_boost, 
        #         mask_candidates=mask_points
        #     )       

        #     # 3. PROJECT TO SCALED METRIC SPACE (For CV and Anisotropy)
        #     # Use depth_boost on the 3rd column and scale_t on the 4th
        #     # scaling_vector = np.array([1.0, 1.0, depth_boost, scale_t])
        #     # x_proj_scaled = ftrns1_abs(x_grid * scaling_vector)        

        #     x_proj_scaled = get_warped_metric_space(x_grid, depth_boost, scale_t)
        #     # --- PART A: Uniformity (Scaled World) ---
        #     tree = cKDTree(x_proj_scaled)
        #     nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
        #     cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)    


        #     # 1. Use the warped metric for distances
        #     if use_normalized_mean == True:
	    #         x_metric_4d1 = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units=True)
	    #         tree_4d = cKDTree(x_metric_4d1)
	    #         dist_4d1, idx_nn1 = tree_4d.query(x_metric_4d1, k=2)
	    #         nn_4d1 = dist_4d1[:, 1]
	    #         nn_indices1 = idx_nn1[:, 1]

	    #         # 2. Final Metric Pass (Ground Truth)
	    #         metrics = compute_warped_expected_spacing(
	    #             self.target_N, 
	    #             lat_range=self.ranges[0], 
	    #             lon_range=self.ranges[1],
	    #             depth_range=self.ranges[2], 
	    #             time_range=self.time_range,
	    #             scale_time=scale_t, 
	    #             depth_boost=depth_boost, 
	    #             use_global=self.use_global
	    #         )
	    #         # Unpack using your exact variable names
	    #         volume_4d_warped, _, _, _, _, _ = metrics
	    #         # cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)
	    #         # 2. Use the warped volume for the density expectation
	    #         # Standard 4D Poisson constant is 0.463
	    #         # This accounts for the 4D hypersphere volume constant
	    #         expected_mean = 0.463 * (volume_4d_warped / self.target_N)**(0.25)
	    #         norm_mean = np.mean(nn_4d1) / expected_mean
	    #         penalty_norm = max(0, 1.4 - norm_mean) ** 2
	    #         cv = cv + 1.0*penalty_norm


        #     densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3][1], self.use_global)
        #     # Weighted penalty: we care most about Depth (Surface) and Time boundaries
        #     penalty_boundary = (abs(1.0 - densities['Depth']) * 1.0 +  ## Make these weights proportional to volume
        #                abs(1.0 - densities['Time']) * 1.0 + 
        #                abs(1.0 - densities['Lat']) * 1.0 + 
        #                abs(1.0 - densities['Lon']) * 1.0)/4.0

        #     # penalty = (abs(1.0 - densities['Depth']) * 15.0 + 
        #     #            abs(1.0 - densities['Time']) * 10.0 + 
        #     #            abs(1.0 - densities['Lat']) * 5.0)

        #     # 4. CDF ANALYSIS (The WGS84 "Gold Standard")
        #     cdf_loss = compute_cdf_analysis(x_grid, self.ranges)


        #     # reg_penalty = (
        #     #     0.005 * (np.log10(scale_t / 5000)**2) +      # Prefer scale_t near 5000
        #     #     0.010 * (depth_boost - 1.0)**2 +             # Prefer depth_boost near 1.0
        #     #     0.010 * (buffer_scale - 1.0)**2              # Prefer buffer_scale near 1.0
        #     # )

        #     reg_penalty = (
        #         0.1 * (np.log10(scale_t / scale_time_effective)**2)      # Prefer scale_t near 5000
        #     )

        #     # cdf_loss = cdf_loss/x_grid.shape[1] ## Normalize by number of dimensions used
        #     # --- PART C: The Sanity Check (Anisotropy) ---
        #     # spreads = np.std(x_proj_scaled, axis=0)
        #     # anisotropy = np.max(spreads) / (np.min(spreads) + 1e-9)
        #     # penalty = np.maximum(0, np.log10(anisotropy) - 1.0)**2   

        #     return cv + (3.0 * cdf_loss) + (0.2 * penalty_boundary) + (1.0 * reg_penalty)




## Note: this version works fairly well

class SamplingTuner:

    def __init__(self, target_N, lat_range, lon_range, depth_range, time_range, use_global = use_global, device = device):

        from skopt.space import Real
        self.target_N = target_N
        # Store ranges as [min, max] pairs
        self.ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
        self.device = device
        self.time_range = time_range
        self.use_global = use_global
        
        # 1. Define Search Space
        # scale_t: km/s
        # depth_boost: dimensionless vertical stretch
        # buffer_scale: multiplier for the nominal spacing
        self.space = [
            Real(1e3, 15e3, prior = 'log-uniform', name='scale_t'),      
            Real(1.0, 3.0, name='depth_boost'),   
            Real(0.5, 2.5, name='buffer_scale')    # prior='log-uniform',
        ]

    def optimize(self, n_calls = 90):

        """Runs Bayesian Optimization to find the triplet of parameters."""

        @use_named_args(self.space)
        def objective(scale_t, depth_boost, buffer_scale, use_normalized_mean = True):

            # 1. GENERATE CANDIDATES
            up_sample_factor = 20 if use_time_shift else 10
            number_candidate_nodes = up_sample_factor * self.target_N

            trial_points, mask_points = regular_sobolov(
                number_candidate_nodes, 
                lat_range=self.ranges[0], 
                lon_range=self.ranges[1], 
                depth_range=self.ranges[2], 
                time_range=self.time_range, 
                use_time=use_time_shift, 
                use_global=self.use_global, 
                scale_time=scale_t, 
                N_target=self.target_N, 
                buffer_scale=buffer_scale
            )        

            # 2. RUN FPS (Physical -> Scaled Search Space)
            # fps returns the selected physical [Lat, Lon, Depth, Time] points
            # x_grid = farthest_point_sampling(
            #     ftrns1_abs(trial_points), 
            #     self.target_N, 
            #     scale_time=scale_t, 
            #     depth_boost=depth_boost, 
            #     mask_candidates=mask_points
            # )        

            x_grid = farthest_point_sampling(
                trial_points, 
                self.target_N, 
                scale_time=scale_t, 
                depth_boost=depth_boost, 
                mask_candidates=mask_points
            )       

            # 3. PROJECT TO SCALED METRIC SPACE (For CV and Anisotropy)
            # Use depth_boost on the 3rd column and scale_t on the 4th
            # scaling_vector = np.array([1.0, 1.0, depth_boost, scale_t])
            # x_proj_scaled = ftrns1_abs(x_grid * scaling_vector)        

            x_proj_scaled = get_warped_metric_space(x_grid, depth_boost, scale_t)
            # --- PART A: Uniformity (Scaled World) ---
            tree = cKDTree(x_proj_scaled)
            nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
            cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)    


            # 1. Use the warped metric for distances
            if use_normalized_mean == True:
	            x_metric_4d1 = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units=True)
	            tree_4d = cKDTree(x_metric_4d1)
	            dist_4d1, idx_nn1 = tree_4d.query(x_metric_4d1, k=2)
	            nn_4d1 = dist_4d1[:, 1]
	            nn_indices1 = idx_nn1[:, 1]

	            # 2. Final Metric Pass (Ground Truth)
	            metrics = compute_warped_expected_spacing(
	                self.target_N, 
	                lat_range=self.ranges[0], 
	                lon_range=self.ranges[1],
	                depth_range=self.ranges[2], 
	                time_range=self.time_range,
	                scale_time=scale_t, 
	                depth_boost=depth_boost, 
	                use_global=self.use_global
	            )
	            # Unpack using your exact variable names
	            volume_4d_warped, _, _, _, _, _ = metrics
	            # cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)
	            # 2. Use the warped volume for the density expectation
	            # Standard 4D Poisson constant is 0.463
	            # This accounts for the 4D hypersphere volume constant
	            expected_mean = 0.463 * (volume_4d_warped / self.target_N)**(0.25)
	            norm_mean = np.mean(nn_4d1) / expected_mean
	            penalty_norm = max(0, 1.4 - norm_mean) ** 2
	            cv = cv + 1.0*penalty_norm


            densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3][1], self.use_global)
            # Weighted penalty: we care most about Depth (Surface) and Time boundaries
            
            # penalty_boundary = (abs(1.0 - densities['Depth']) * 1.0 +  ## Make these weights proportional to volume
            #            abs(1.0 - densities['Time']) * 1.0 + 
            #            abs(1.0 - densities['Lat']) * 1.0 + 
            #            abs(1.0 - densities['Lon']) * 1.0)/4.0

            penalty_boundary = (
                (1.0 - densities['Depth'])**2 + 
                (1.0 - densities['Time'])**2 + 
                (1.0 - densities['Lat'])**2 + 
                (1.0 - densities['Lon'])**2
            ) / 4.0


            # penalty = (abs(1.0 - densities['Depth']) * 15.0 + 
            #            abs(1.0 - densities['Time']) * 10.0 + 
            #            abs(1.0 - densities['Lat']) * 5.0)

            # 4. CDF ANALYSIS (The WGS84 "Gold Standard")
            cdf_loss = compute_cdf_analysis(x_grid, self.ranges)


            # reg_penalty = (
            #     0.005 * (np.log10(scale_t / 5000)**2) +      # Prefer scale_t near 5000
            #     0.010 * (depth_boost - 1.0)**2 +             # Prefer depth_boost near 1.0
            #     0.010 * (buffer_scale - 1.0)**2              # Prefer buffer_scale near 1.0
            # )

            reg_penalty = (
                0.1 * (np.log10(scale_t / scale_time_effective)**2)      # Prefer scale_t near 5000
            )

            # cdf_loss = cdf_loss/x_grid.shape[1] ## Normalize by number of dimensions used
            # --- PART C: The Sanity Check (Anisotropy) ---
            # spreads = np.std(x_proj_scaled, axis=0)
            # anisotropy = np.max(spreads) / (np.min(spreads) + 1e-9)
            # penalty = np.maximum(0, np.log10(anisotropy) - 1.0)**2   

            return cv + (3.0 * cdf_loss) + (0.2 * penalty_boundary) + (1.0 * reg_penalty)

        # res = gp_minimize(objective, self.space, n_calls = n_calls, n_initial_points = 5, initial_point_generator = 'sobol', verbose = True) # random_state = 42
        res = gp_minimize(objective, self.space, n_calls = n_calls, verbose = True) # random_state = 42

        best_scale_t, best_depth_boost, best_buffer_scale = res.x
        

        _, _, _, nominal_spacing_4d, _, _ = compute_warped_expected_spacing(
        	self.target_N,  # total number of points
        	lat_range=self.ranges[0],
        	lon_range=self.ranges[1],
        	depth_range=self.ranges[2],
        	time_range=self.time_range,  # T, full range = 2T
        	use_time=use_time_shift,
        	scale_time=best_scale_t,  # w_scale: length per unit time
        	use_global=use_global,
        	earth_radius=earth_radius)

        # nominal_spacing = (total_vol / self.target_N)**(1/4)
        # buffer_width_phys = best_buffer_scale * nominal_spacing
        buffer_width_phys = best_buffer_scale * nominal_spacing_4d # [0]

        print("\n--- Optimization Results ---")
        print(f"Optimal scale_t:     {best_scale_t:.3f} m/s")
        print(f"Optimal depth_boost: {best_depth_boost:.3f}")
        print(f"Optimal buffer_scale: {best_buffer_scale:.3f}")
        print(f"Effective Padding:   {buffer_width_phys:.2f} (units)")

        return {
            'scale_t': best_scale_t,
            'depth_boost': best_depth_boost,
            'buffer_scale': best_buffer_scale,
            'buffer_width_phys': buffer_width_phys
        }



def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, buffer_scale, volume_4d_warped):
    
    N = len(x_grid)
    
    # --- 1. COORDINATE PROJECTIONS ---
    # scaling_4d = np.array([1.0, 1.0, depth_boost, scale_t])
    # x_metric_4d = ftrns1_abs(x_grid * scaling_4d)
    
    # x_metric_4d = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units = True)

    # # --- 2. GLOBAL 4D METRICS (The "O" Sliders) ---
    # tree_4d = cKDTree(x_metric_4d)
    # dist_4d, idx_nn = tree_4d.query(x_metric_4d, k=2)
    # nn_4d = dist_4d[:, 1]
    # nn_indices = idx_nn[:, 1] # Index of the 4D nearest neighbor
    
    # ## Can replace these nearest neighbors with the warped metric
    # cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)
    # v_4d = volume_space * (2.0 * time_range * scale_t)
    # expected_mean = 0.65 * (v_4d / N)**(1/4)
    # norm_mean = np.mean(nn_4d) / expected_mean


    # 1. Use the warped metric for distances
    x_metric_4d = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units=True)
    tree_4d = cKDTree(x_metric_4d)
    dist_4d, idx_nn = tree_4d.query(x_metric_4d, k=2)
    nn_4d = dist_4d[:, 1]
    nn_indices = idx_nn[:, 1]
    cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)

    # 2. Use the warped volume for the density expectation
    # Standard 4D Poisson constant is 0.463
    # This accounts for the 4D hypersphere volume constant
    expected_mean = 0.463 * (volume_4d_warped / N)**(0.25)
    norm_mean = np.mean(nn_4d) / expected_mean
    void_ratio = np.quantile(nn_4d, 0.99) / (np.mean(nn_4d) + 1e-9)

    # --- 3. GRAPH MARGINALS & EFFECTIVE VELOCITY ---
    # Spatial part (in km)
    x_phys_3d_km = ftrns1_abs(x_grid[:, :3]) / 1000.0
    dist_space = np.linalg.norm(x_phys_3d_km - x_phys_3d_km[nn_indices], axis=1)
    
    # Temporal part (in seconds)
    dist_time = np.abs(x_grid[:, 3] - x_grid[nn_indices, 3])
    
    avg_space_km = np.mean(dist_space)
    avg_time_s = np.mean(dist_time)
    min_space_km = np.min(dist_space)
    min_time_s = np.min(dist_time)
    
    # # Physical Spatial Gap (meters -> km)
    # dist_space_km = np.linalg.norm(x_phys_3d_m - x_phys_3d_m[nn_indices], axis=1) / 1000.0
    # # Physical Temporal Gap (seconds)
    # dist_time_s = np.abs(x_grid[:, 3] - x_grid[nn_indices, 3])
    # # The "True" Physical v_eff
    # # This tells you: "How fast is the graph connection?"
    # avg_space = np.mean(dist_space_km)
    # avg_time = np.mean(dist_time_s)
    # v_eff = avg_space / (avg_time + 1e-9)

    # V_eff: How fast does one have to travel to reach the 4D neighbor?
    # Higher scale_t = Lower V_eff (because time is 'longer')
    v_eff = avg_space_km / (avg_time_s + 1e-9)
    cv_space_4d = np.std(dist_space) / (avg_space_km + 1e-9)
    cv_time_4d = np.std(dist_time) / (avg_time_s + 1e-9)


    # --- 4. MARGINAL VOID ANALYSIS ---
    # Spatial Void (3D): Largest spatial jump in the 4D neighbor graph
    x_phys_3d = ftrns1_abs(x_grid[:, :3])
    dist_space_m = np.linalg.norm(x_phys_3d - x_phys_3d[nn_indices], axis=1)
    # Void ratio = 99th percentile distance / Mean distance
    space_void_ratio = np.quantile(dist_space_m, 0.99) / (np.mean(dist_space_m) + 1e-9)
    
    # Time Void (1D): Largest temporal gap between events
    dist_time_s = np.abs(x_grid[:, 3] - x_grid[nn_indices, 3])
    time_void_ratio = np.quantile(dist_time_s, 0.99) / (np.mean(dist_time_s) + 1e-9)



    # # # --- 3. EXPANDED BOUNDARY HEALTH CHECK ---
    # # A. Temporal Boundary (1D)
    # t_min, t_max = -time_range * scale_t, time_range * scale_t
    # dist_to_t_bound = np.minimum(np.abs(x_metric_4d[:, 3] - t_min), np.abs(x_metric_4d[:, 3] - t_max))
    # t_bias = np.mean(dist_to_t_bound) / (np.mean(nn_4d) + 1e-9)

    # # B. Radial / Depth Boundary (1D in Metric Space)
    # # We use the metric depth which includes depth_boost
    # r_metric = np.linalg.norm(x_phys_3d, axis=1) * depth_boost
    # r_min_m, r_max_m = r_min * depth_boost, r_max * depth_boost
    # dist_to_r_bound = np.minimum(np.abs(r_metric - r_min_m), np.abs(r_metric - r_max_m))
    # r_bias = np.mean(dist_to_r_bound) / (np.mean(nn_4d) + 1e-9)


    # # C. Lateral Boundary (If Regional)
    # if not use_global:
    #     # Check proximity to Lat/Lon edges in degrees (roughly converted to metric for ratio)
    #     lat_dist = np.minimum(np.abs(x_grid[:, 0] - lat_range[0]), np.abs(x_grid[:, 0] - lat_range[1])) * 111000
    #     lon_dist = np.minimum(np.abs(x_grid[:, 1] - lon_range[0]), np.abs(x_grid[:, 1] - lon_range[1])) * 111000 * np.cos(np.deg2rad(x_grid[:,0]))
    #     lat_lon_bias = np.mean(np.minimum(lat_dist, lon_dist)) / (np.mean(nn_4d) + 1e-9)
    # else:
    #     lat_lon_bias = 0.5 # Global has no lateral "edges" in Lon, and Pole effects are handled by WGS84 logic

    # compute_boundary_biases(x_grid, x_phys_3d, nn_4d, lat_range, lon_range, depth_range, time_range, scale_t, depth_boost
    # bias, bias_lat, bias_lon, bias_depth, bias_time, bias_masks = compute_boundary_biases(x_grid, nn_4d, lat_range, lon_range, depth_range, time_range) # scale_t, depth_boost
    boundary_health = check_boundary_densities(x_grid, lat_range, lon_range, depth_range, time_range)
    bias_lat = boundary_health['Lat']
    bias_lon = boundary_health['Lon']
    bias_depth = boundary_health['Depth']
    bias_time = boundary_health['Time']


    # --- 5. WGS84 TRANSPARENCY (CDF R2) ---
    cdf_r2s = {}
    emp_cdf = np.arange(N) / (N - 1)
    

    # Lat (Authalic)
    q = get_wgs84_area_val(x_grid[:, 0])
    q_min, q_max = get_wgs84_area_val(lat_range[0]), get_wgs84_area_val(lat_range[1])
    cdf_r2s["Lat"] = r2_score(emp_cdf, np.sort((q - q_min)/(q_max - q_min + 1e-12)))
    
    # Lon (Linear)
    cdf_r2s["Lon"] = r2_score(emp_cdf, np.sort((x_grid[:, 1] - lon_range[0])/(lon_range[1] - lon_range[0])))
    
    # Depth (WGS84 Vol Frac)
    r_min_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[0])])), axis=1)
    r_max_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[1])])), axis=1)
    r_act = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
    vol_f = (r_act**3 - r_min_l**3) / (r_max_l**3 - r_min_l**3 + 1e-12)
    cdf_r2s["Depth"] = r2_score(emp_cdf, np.sort(vol_f))
    
    # Time (Linear)
    cdf_r2s["Time"] = r2_score(emp_cdf, np.sort((x_grid[:, 3] - (-time_range))/(2*time_range)))

    # --- 5. TERMINAL OUTPUT ---
    print(f"\n{'='*65}")
    print(f"       GEOMETRIC GRID HEALTH REPORT (WGS84-4D)")
    print(f"{'='*65}")
    print(f"Nodes (N): {N:<8} | scale_t: {scale_t:<8.1f} | time range: {time_range:<.2f} | d_boost: {depth_boost:<.2f} | buffer_scale: {buffer_scale:<.2f}")
    
    def get_bar(val, ideal_min, ideal_max):
        bar = ["-"] * 20
        pos = int(min(max(val / (ideal_max * 1.5), 0), 1) * 19)
        bar[pos] = "O"
        return "".join(bar)

    print(f"\n[1] 4D Metric Uniformity")
    print(f"    CV NN:           {cv_4d:.4f}  [{get_bar(cv_4d, 0.2, 0.3)}] (Goal: <0.2)")
    print(f"    Normalized Mean: {norm_mean:.4f}  [{get_bar(norm_mean, 1.4, 1.7)}] (Goal: >1.4)")
    print(f"    Void Ratio:      {void_ratio:.4f}  [{get_bar(void_ratio, 2.0, 3.0)}] (Goal: <3.0)")

    print(f"\n[2] Graph Neighbor Marginals (4D Links)")
    print(f"    Avg Spatial Gap: {avg_space_km:.2f} km  (CV: {cv_space_4d:.3f})")
    print(f"    Avg Temporal Gap: {avg_time_s:.2f} s   (CV: {cv_time_4d:.3f})")
    print(f"    Min Spatial Gap: {min_space_km:.2f} km ")
    print(f"    Min Temporal Gap: {min_time_s:.2f} s  \n")

    print(f"\n[3] Voids and Clustering ")
    print(f"    Effective Velocity: {v_eff:.2f} km/s {'[PHYSICAL]' if 4<v_eff<10 else '[STRETCHED]'}")
    print(f"    Void Ratio (space):  {space_void_ratio:.4f}  [{get_bar(space_void_ratio, 2.0, 3.0)}] (Goal: <3.0)")
    print(f"    Void Ratio (time):  {time_void_ratio:.4f}  [{get_bar(time_void_ratio, 2.0, 3.0)}] (Goal: <3.0)")
    collision_count = np.sum(nn_4d < nn_4d.mean()*0.75) ## Anything within half the average distance
    print(f"    Collision Check: {collision_count} nodes < half avg. distance apart.")


    # --- [3] Boundary & Edge Health ---
    print(f"\n[4] Boundary & Edge Health (Bias Ratio)")
    def format_bias(name, val):
        status = "OK" if 0.7 < val < 1.3 else "BIASED"
        return f"    {name:12}: {val:.3f} [{get_bar(val, 0.5, 1.5)}] ({status})"
    print(format_bias("Temporal", bias_time))
    print(format_bias("Depth/Radial", bias_depth))
    print(format_bias("Lat", bias_lat))
    if not use_global:
        print(format_bias("Lon", bias_lon))


    # --- 6. VELOCITY & ISOTROPY DIAGNOSTIC ---
    # The target is for v_eff (physical) to be close to scale_t (metric)
    v_mismatch = v_eff / (scale_t / 1000.0) # Ratio of Graph Speed to Metric Speed    

    print(f"\n[5] Physical Graph Balance")
    print(f"    Target Velocity (Scale): {scale_t/1000.0:.2f} km/s")
    print(f"    Actual Graph Velocity:   {v_eff:.2f} km/s")
    status_v = "BALANCED" if 0.8 < v_mismatch < 1.25 else "STRETCHED"
    print(f"    Velocity Mismatch:       {v_mismatch:.3f}x [{get_bar(v_mismatch, 0.5, 2.0)}] ({status_v})")    

    # Predicted vs Actual spacing
    expected_dx_km = (volume_4d_warped / (N * scale_t * 2 * time_range))**(1/3) / 1000.0
    print(f"    Predicted Space Res:     {expected_dx_km:.2f} km")
    print(f"    Actual Space Res:        {avg_space_km:.2f} km")


    print(f"\n[6] WGS84 Transparency (CDF R2 Scores)")
    for name, score in cdf_r2s.items():
        status = "PASS" if score > 0.98 else "WARN"
        print(f"    {name:6} R2: {score:.6f}  [{'#'*int(score*20):<20}] {status}")
    

    print(f"{'='*65}\n")
    return {"cv_4d": cv_4d, "min_dist": np.min(dist_space), "collisions": collision_count, "cdf_r2s": cdf_r2s, "v_eff": v_eff}


def perform_ks_density_test(x_grid, lat_range):
    # 1. Transform Latitudes to normalized q-space [0, 1]
    # (Using your get_q_wgs84 function)
    q_actual = get_q_wgs84(x_grid[:, 0])
    q_min = get_q_wgs84(lat_range[0])
    q_max = get_q_wgs84(lat_range[1])
    
    # These are our samples for the KS test
    samples = (q_actual - q_min) / (q_max - q_min + 1e-12)
    samples = np.clip(samples, 0, 1) # Ensure no floating point overshoot
    
    # 2. Run KS Test against a uniform distribution
    # D is the maximum distance between the distributions
    d_stat, p_val = stats.kstest(samples, 'uniform')
    
    # 3. Interpret results
    print(f"\n[6] KS Density Significance (Latitude)")
    print(f"    Max Deviation (D): {d_stat:.4f}")
    print(f"    P-Value:           {p_val:.4f}")
    
    if p_val < 0.05:
        print("    RESULT: SIGNIFICANT BIAS DETECTED (p < 0.05)")
    else:
        print("    RESULT: PHYSICALLY UNBIASED (Uniform on Ellipsoid)")
        
    return d_stat, p_val



# def perform_ks_depth_test(x_grid, depth_range, r_surface):
#     # depth_range[0] is Top (+), depth_range[1] is Bottom (-)
#     # 1. Convert depths to Radii
#     r_actual = r_surface + x_grid[:, 2]
#     r_top = r_surface + depth_range[0]
#     r_bot = r_surface + depth_range[1]
    
#     # 2. Transform to Cubic Space (Volume is proportional to r^3)
#     # This 'un-warps' the spherical shell growth
#     vol_actual = r_actual**3
#     vol_min = r_bot**3
#     vol_max = r_top**3
    
#     samples = (vol_actual - vol_min) / (vol_max - vol_min + 1e-12)
#     samples = np.clip(samples, 0, 1)
    
#     d_stat, p_val = stats.kstest(samples, 'uniform')
#     print(f"\n[7] KS Density Significance (Depth/Volume)")
#     print(f"    P-Value: {p_val:.4f}")
#     return d_stat, p_val


def perform_ks_depth_test_ellipsoid(x_grid, depth_range):
    # depth_range: [top, bottom] e.g., [0, -40]
    
    # 1. Get local surface radius for every point (using WGS84)
    # If your ftrns1_abs handles the ellipsoid, we can derive the local R
    # Let's assume you have a helper to get R_local based on Latitude
    r_local_surface = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) # Lat-dependent
    
    # 2. Calculate actual distance from Earth Center
    r_actual = r_local_surface + x_grid[:, 2] # Depth is negative
    
    # 3. Calculate the local Shell Boundaries
    r_top = r_local_surface + depth_range[0]
    r_bot = r_local_surface + depth_range[1]
    
    # 4. Transform to Volume Space (r^3)
    # For a perfect ellipsoid, V is proportional to a*b*c, 
    # but the shell-ratio within a small depth range 
    # is still dominated by the r^3 scaling of the local radius.
    vol_actual = r_actual**3
    vol_min = r_bot**3
    vol_max = r_top**3
    
    samples = (vol_actual - vol_min) / (vol_max - vol_min + 1e-12)
    samples = np.clip(samples, 0, 1)
    
    d_stat, p_val = stats.kstest(samples, 'uniform')
    print(f"\n[7] KS Density Significance (Depth/Volume)")
    print(f"    P-Value: {p_val:.4f}")
    return d_stat, p_val


def export_automated_metadata(filename, x_grid, stats_report, params):
    """
    Automated export using actual run statistics.
    stats_report: The dictionary containing your 'Avg Spatial Gap', etc.
    params: The winning [scale_t, depth_boost, buffer_scale]
    """
    avg_space = 1000.0 * stats_report['avg_spatial_gap_km']
    avg_time = stats_report['avg_temporal_gap_s']
    
    # Calculate GNN sigmas dynamically
    sigma_x = avg_space / 2.0
    sigma_t = avg_time / 2.0
    
    metadata = {
        "run_parameters": params,
        "grid_health": {
            "cv_nn": stats_report['cv_nn'],
            "eff_velocity": stats_report['effective_velocity_km_s'],
            "v_mismatch": stats_report['velocity_mismatch'],
            "normalized_mean": stats_report['normalized_mean']
        },
        "gnn_configuration": {
            "edge_construction": "K-NN in Warped Metric Space",
            "recommended_k": 18,
            "label_sigma_space_m": round(sigma_x, 3),
            "label_sigma_time_s": round(sigma_t, 3),
            "node_resolution_spatial_m": round(avg_space, 3),
            "node_resolution_temporal_s": round(avg_time, 3)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)


# import numpy as np

def export_gnn_ready_metadata(filename, x_grid_warped, stats_report, params, K_neighbors=18):
    """
    Exports metadata for GNN training.
    x_grid_warped: The grid in the fully warped metric space [Lat_q, Lon_rad, Depth_r, Time_s]
    """
    # 1. Compute the actual mean distance to the K-th neighbor in warped space
    from scipy.spatial import cKDTree
    tree = cKDTree(x_grid_warped)
    # k=K+1 because index 0 is the node itself
    dists, _ = tree.query(x_grid_warped, k=K_neighbors + 1)
    
    # Use the mean distance of all neighbors in the K-shell to define sigma
    # This is more robust than just using the 1st neighbor
    mean_warped_dist = np.mean(dists[:, 1:]) 
    
    # 2. Derive physical Sigmas from the stats report
    # Sigma = (Avg Neighbor Distance) / 2
    sigma_x = stats_report['avg_spatial_gap_km'] / 2.0
    sigma_t = stats_report['avg_temporal_gap_s'] / 2.0
    
    metadata = {
        "physical_constants": {
            "scale_time_w": params['scale_t'],
            "depth_boost": params['depth_boost'],
            "effective_v_km_s": stats_report['effective_velocity_km_s']
        },
        "gnn_hyperparameters": {
            "knn_k_neighbors": K_neighbors,
            "has_self_loop": False,
            "metric_space": "Warped (Authalic + 1/r + Scale_t)",
            "label_sigma_space_km": round(sigma_x, 3),
            "label_sigma_time_s": round(sigma_t, 3),
            "mean_warped_neighbor_dist": round(mean_warped_dist, 5)
        },
        "grid_quality": {
            "cv_nn": stats_report['cv_nn'],
            "ks_p_value": stats_report.get('ks_p_value', "N/A"),
            "depth_bias": stats_report.get('depth_bias', 1.24)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"GNN sidecar saved: {filename}")


# def perform_ks_depth_test(x_grid, depth_range, r_surface):
#     # depth_range[0] is Top (+), depth_range[1] is Bottom (-)
#     # 1. Convert depths to Radii
#     r_actual = r_surface + x_grid[:, 2]
#     r_top = r_surface + depth_range[0]
#     r_bot = r_surface + depth_range[1]
    
#     # 2. Transform to Cubic Space (Volume is proportional to r^3)
#     # This 'un-warps' the spherical shell growth
#     vol_actual = r_actual**3
#     vol_min = r_bot**3
#     vol_max = r_top**3
    
#     samples = (vol_actual - vol_min) / (vol_max - vol_min + 1e-12)
#     samples = np.clip(samples, 0, 1)
    
#     d_stat, p_val = stats.kstest(samples, 'uniform')
#     print(f"\n[7] KS Density Significance (Depth/Volume)")
#     print(f"    P-Value: {p_val:.4f}")
#     return d_stat, p_val




def save_grid_metadata(path_to_file, grid_ind, params, health):
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "params": params,
        "health": health,
        "coord_system": "WGS84_Ellipsoidal_4D"
    }
    with open(f"{path_to_file}/grid_{grid_ind}_meta.json", 'w') as f:
        json.dump(meta, f, indent=4)



def save_grid_diagnostic_plot(x_grid, path_to_file, grid_ind):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = ["Lat", "Lon", "Depth", "Time"]
    for i in range(4):
        axes[i].hist(x_grid[:, i], bins=50, color='teal', alpha=0.7)
        axes[i].set_title(f"Distribution: {names[i]}")
        axes[i].set_xlabel(names[i])
    plt.tight_layout()
    plt.savefig(f"{path_to_file}/grid_diag_{grid_ind}.png")
    plt.close()





use_tuning = True
if use_tuning == True:
	## Run the auto tuning strategy to refine some scale parameters
	m = SamplingTuner(number_of_spatial_nodes, lat_range_extend, lon_range_extend, depth_range, time_shift_range)
	params = m.optimize()
	scale_time, depth_upscale_factor, buffer_scale = params['scale_t'], params['depth_boost'], params['buffer_scale']

else:
	# depth_upscale_factor = 1.0
	buffer_scale = 2.0




## Now build all spatial grids using optimal sampling strategy
x_grids = []
for n in range(num_grids):

	print('Beginning FPS sampling [%d]'%n)
	up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
	number_candidate_nodes = up_sample_factor*number_of_spatial_nodes
	trial_points, mask_points = regular_sobolov(number_candidate_nodes, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = number_of_spatial_nodes, buffer_scale = buffer_scale) # lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0
	# x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = mask_points)
	x_grid = farthest_point_sampling(trial_points, number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = mask_points)

	tol_frac = 0.01
	assert(x_grid[:,0].min() >= (lat_range_extend[0] - tol_frac*np.diff(lat_range_extend)))
	assert(x_grid[:,0].max() <= (lat_range_extend[1] + tol_frac*np.diff(lat_range_extend)))
	assert(x_grid[:,1].min() >= (lon_range_extend[0] - tol_frac*np.diff(lon_range_extend))) if use_global == False else 1
	assert(x_grid[:,1].max() <= (lon_range_extend[1] + tol_frac*np.diff(lon_range_extend))) if use_global == False else 1
	assert(x_grid[:,2].min() >= (depth_range[0] - tol_frac*np.diff(depth_range)))
	assert(x_grid[:,2].max() <= (depth_range[1] + tol_frac*np.diff(depth_range)))
	assert(len(x_grid) == number_of_spatial_nodes)

	if n == 0:
		compute_final_grid_health(x_grid, scale_time, depth_upscale_factor, lat_range_extend, lon_range_extend, depth_range, time_shift_range, buffer_scale, Volume)
		perform_ks_density_test(x_grid, lat_range_extend)
		perform_ks_depth_test_ellipsoid(x_grid, depth_range)

	# loss_metrics(x_grid, plot_on = True, grid_ind = n)
	# # nn_distance_stats(ftrns1_abs(x_grid)/1000.0, w_scale = scale_time/1000.0)
	# nn_distance_stats(ftrns1_abs(x_grid), scale_time, Volume_space, time_shift_range)
	x_grids.append(np.expand_dims(x_grid, axis = 0))



x_grids = np.vstack(x_grids)
x_grids_cart = np.vstack([np.expand_dims(ftrns1(x_grids[i]), axis = 0) for i in range(len(x_grids)) for i in range(num_grids)])
x_grids_warped = np.vstack([np.expand_dims(get_warped_metric_space(x_grids[i], depth_upscale_factor, scale_time, return_physical_units = True), axis = 0) for i in range(num_grids)])
mean_x_grid_warped = x_grids_warped[:,:,0:3].mean(0, keepdims = True).mean(0, keepdims = True)
x_grids_warped_scaled = np.copy(x_grids_warped)
x_grids_warped_scaled[:,:,0:3] -= mean_x_grid_warped
x_grids_warped_scaled = x_grids_warped_scaled/(10e3)

k_edges = 18 ## An effective edge number in 4D
edges = np.vstack([np.expand_dims(sort_edge_index(remove_self_loops(knn(torch.Tensor(x_grids_warped_scaled[i]), torch.Tensor(x_grids_warped_scaled[i]), k = k_edges + 1))[0].flip(0)).contiguous().cpu().detach().numpy(), axis = 0) for i in range(num_grids)])
np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_templates_ver_1.npz'%name_of_project, x_grids = x_grids, x_grids_cart = x_grids_cart, x_grids_warped = x_grids_warped, x_grids_warped_scaled = x_grids_warped_scaled, scale_time = scale_time, depth_boost = depth_upscale_factor, time_shift_range = time_shift_range, number_of_spatial_nodes = number_of_spatial_nodes, edges = edges, corr1 = np.zeros((1,3)), corr2 = np.zeros((1,3)))


# print('Stable graphs will typically have:')
# print('R2 expected depth >0.95')
# print('R2 expected time >0.97')
# print('CV NN full < 0.15')
# print('Normalized Mean >1.5')
# print('Correlation of Space-time nearest neighbors < 0.1')
# print('Void ratio (spatial) < 2')
# print('Void ratio (temporal) < 5')
# print('Should add metrics computed near boundary')



## Now build expander graphs
build_expander_graphs = True
if build_expander_graphs == True:


	def run_large_graph_forensics(edge_index, num_nodes):
	    data = Data(edge_index=edge_index, num_nodes=num_nodes)
	    G = to_networkx(data, to_undirected=True)
	    
	    # 1. Diameter (This is actually the slowest part for large graphs)
	    # We use an approximation if the graph is huge, but 5.9k is okay for exact
	    if nx.is_connected(G):
	        diam = nx.diameter(G)
	    else:
	        print("Graph is disconnected. Calculating metrics for largest component.")
	        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
	        diam = nx.diameter(G)

	    # 2. Sparse Eigenvector Decomposition (Spectral Gap)
	    # We use 'SM' (Smallest Magnitude). 
	    # The smallest is 0, the second smallest is the Fiedler value.
	    L = nx.laplacian_matrix(G).astype(float)
	    evals = eigsh(L, k=2, which='SM', return_eigenvectors=False)
	    fiedler_value = sorted(evals)[1]
	    
	    # 3. Density / Regularity
	    avg_degree = (2 * G.number_of_edges()) / G.number_of_nodes()
	    
	    print(f"--- Large Graph Forensics ---")
	    print(f"Diameter: {diam}")
	    print(f"Fiedler Value: {fiedler_value:.5f}")
	    print(f"Avg Degree: {avg_degree:.2f}")
	    
	    return diam, fiedler_value


	def compute_expansion_stats(edge_index, num_nodes):
	    # 1. Create Data/NetworkX object
	    data = Data(edge_index=edge_index, num_nodes=num_nodes)
	    G = to_networkx(data, to_undirected=True)
	    
	    # 2. Get Laplacian (Sparse)
	    L = nx.laplacian_matrix(G).astype(float)
	    
	    # 3. Compute Normalized Spectral Gap
	    # L_norm = D^-1/2 * L * D^-1/2
	    degrees = np.array(L.diagonal())
	    d_inv_sqrt = np.power(degrees, -0.5)
	    D_inv_sqrt_mat = sp.diags(d_inv_sqrt)
	    L_norm = D_inv_sqrt_mat @ L @ D_inv_sqrt_mat
	    
	    # We want the second smallest eigenvalue (Fiedler value of normalized Laplacian)
	    evals = eigsh(L_norm, k=2, which='SM', return_eigenvectors=False)
	    spectral_gap_norm = sorted(evals)[1]
	    
	    # 4. Physical Expansion Estimate (Cheeger Lower Bound)
	    # Using lambda_2 from the standard Laplacian
	    evals_std = eigsh(L, k=2, which='SM', return_eigenvectors=False)
	    lambda_2 = sorted(evals_std)[1]
	    h_lower_bound = lambda_2 / 2

	    print(f"Normalized Spectral Gap: {spectral_gap_norm:.6f}")
	    print(f"Cheeger Expansion Lower Bound: {h_lower_bound:.6f}")

	    return spectral_gap_norm, h_lower_bound, lambda_2



	def make_cayleigh_graph(n):

		generators = [np.array([1, 1, 0, 1]).reshape(2,2), np.array([1, 0, 1, 1]).reshape(2,2)]
		nodes = np.vstack([generators[0].reshape(1,-1), generators[1].reshape(1,-1)])
		edges = []

		new = np.inf
		cnt = 0

		while new > 0:

			print('iteration %d, num nodes %d'%(cnt, len(nodes)))

			tree = cKDTree(nodes)
			len_nodes = len(nodes)

			new_nodes_1 = []
			new_nodes_2 = []

			for i in range(len(nodes)):

				new_nodes_1.append(np.mod(nodes[i].reshape(2,2) @ generators[0], n).reshape(1,-1))
				new_nodes_2.append(np.mod(nodes[i].reshape(2,2) @ generators[1], n).reshape(1,-1))

			new_nodes_1 = np.vstack(new_nodes_1) # has size nodes
			new_nodes_2 = np.vstack(new_nodes_2)
			new_nodes = np.unique(np.concatenate((new_nodes_1, new_nodes_2), axis = 0), axis = 0)

			q = tree.query(new_nodes)[0]
			inew = np.where(q > 0)[0]
			new_nodes = new_nodes[inew]

			if len(inew) == 0:
				new = 0
				continue # Break loop

			## Now need to find which entries in new_nodes are linked for each input node
			tree_new = cKDTree(new_nodes)
			ip = tree_new.query(new_nodes_1)
			ip1 = np.where(ip[0] == 0)[0] ## Points to current absolute node indices that are linked to new node
			edges_new_1 = np.concatenate((ip1.reshape(-1,1), len_nodes + ip[1][ip1].reshape(-1,1)), axis = 1)

			ip = tree_new.query(new_nodes_2)
			ip1 = np.where(ip[0] == 0)[0] ## Points to current absolute node indices that are linked to new node
			edges_new_2 = np.concatenate((ip1.reshape(-1,1), len_nodes + ip[1][ip1].reshape(-1,1)), axis = 1)

			# edges.append(np.unique(np.concatenate((edges_new_1, edges_new_2), axis = 0), axis = 0))
			nodes = np.concatenate((nodes, new_nodes), axis = 0)
			
			cnt += 1

		## Find inverses to generators
		inv_indices_1 = []
		inv_indices_2 = []
		for i in range(len(nodes)):
			if np.abs(np.mod(generators[0] @ nodes[i].reshape(2,2), n) - np.eye(2)).max() == 0:
				inv_indices_1.append(i)
			if np.abs(np.mod(generators[1] @ nodes[i].reshape(2,2), n) - np.eye(2)).max() == 0:
				inv_indices_2.append(i)

		assert(len(inv_indices_1) == 1)
		assert(len(inv_indices_2) == 1)

		generators_inverses = [nodes[inv_indices_1[0]].reshape(2,2), nodes[inv_indices_2[0]].reshape(2,2)]

		## Now must add missing edges between all previously created nodes. (can do this outside of the loop)
		for i in range(len(nodes)):
			for j in range(len(nodes)):
				dist1 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators[0], n) - nodes[j].reshape(2,2)).max()
				dist2 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators[1], n) - nodes[j].reshape(2,2)).max()
				if ((dist1 == 0) + (dist2 == 0)) > 0:
					edges.append(np.array([i,j]).reshape(1,-1))

		for i in range(len(nodes)):
			for j in range(len(nodes)):
				dist1 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators_inverses[0], n) - nodes[j].reshape(2,2)).max()
				dist2 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators_inverses[1], n) - nodes[j].reshape(2,2)).max()
				if ((dist1 == 0) + (dist2 == 0)) > 0:
					edges.append(np.array([i,j]).reshape(1,-1))

		edges = np.unique(np.vstack(edges), axis = 0)
		## Check for all edges, if each node is really linked to the declared nodes.
		for i in range(len(edges)):
			dist1 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators[0], n) - nodes[edges[i,1]].reshape(2,2)).max()
			dist2 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators[1], n) - nodes[edges[i,1]].reshape(2,2)).max()
			dist3 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators_inverses[0], n) - nodes[edges[i,1]].reshape(2,2)).max()
			dist4 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators_inverses[1], n) - nodes[edges[i,1]].reshape(2,2)).max()
			assert(((dist1 == 0) + (dist2 == 0) + (dist3 == 0) + (dist4 == 0)) > 0)
			print(i)

		return edges

	def make_labeled_cayley_graph(n):
	    s1 = np.array([[1, 1], [0, 1]])
	    s2 = np.array([[1, 0], [1, 1]])
	    
	    def get_inv(m):
	        inv = np.array([[m[1,1], -m[0,1]], [-m[1,0], m[0,0]]])
	        return np.mod(inv, n)

	    # Define the 4 algebraic relations
	    gens = [s1, s2, get_inv(s1), get_inv(s2)]
	    
	    identity = np.eye(2, dtype=int)
	    nodes_list = [identity]
	    nodes_dict = {tuple(identity.flatten()): 0}
	    
	    edge_list = []
	    edge_types = []
	    
	    queue = [0]
	    while queue:
	        u_idx = queue.pop(0)
	        u_mat = nodes_list[u_idx]
	        
	        for g_type, g in enumerate(gens):
	            v_mat = np.mod(u_mat @ g, n)
	            v_tuple = tuple(v_mat.flatten())
	            
	            if v_tuple not in nodes_dict:
	                v_idx = len(nodes_list)
	                nodes_dict[v_tuple] = v_idx
	                nodes_list.append(v_mat)
	                queue.append(v_idx)
	            else:
	                v_idx = nodes_dict[v_tuple]
	            
	            edge_list.append([u_idx, v_idx])
	            edge_types.append(g_type) # 0:S1, 1:S2, 2:S1_inv, 3:S2_inv

	    return torch.tensor(edge_list).t(), torch.tensor(edge_types)




	## Choose type of expander graph (or none)
	use_gabber = True
	if number_of_spatial_nodes > 2500: use_gabber = True

	if use_gabber == False: ## Then use cayley graphs

		# A_c_l = []
		n_max = 0
		cnt = 2
		while n_max < number_of_spatial_nodes:
			A_edges = make_cayleigh_graph(cnt)
			# A_c_l.append(A_edges)
			n_max = A_edges.max() + 1
			cnt += 1
		Ac = subgraph(torch.arange(number_of_spatial_nodes), torch.Tensor(A_edges.T).long())[0].flip(0).cpu().detach().numpy() # .to(device)

	else:

		int_need = int(np.ceil(np.sqrt(number_of_spatial_nodes)))

		use_networkx_construction = False
		if use_networkx_construction == True:

			A_edges_c = from_networkx(nx.margulis_gabber_galil_graph(int_need)).edge_index.long().flip(0).contiguous()
			A_edges_c = torch.Tensor(np.random.permutation(A_edges_c.max().item() + 1)).long()[A_edges_c]
			Ac = subgraph(torch.arange(number_of_spatial_nodes), A_edges_c)[0].cpu().detach().numpy() # .to(device)
			# A_edges_c = perm_vec[A_edges_c]

		else:


			def make_labeled_mgg_graph(m: int):
			    edge_index_list = []
			    edge_type_list = []			

			    for x in range(m):
			        for y in range(m):
			            u = x * m + y			

			            # 4 base directed neighbors (as in NetworkX source code)
			            neighbors = [
			                ((x + 2 * y) % m, y),                     # type 0
			                ((x + 2 * y + 1) % m, y),                 # type 1
			                (x, (y + 2 * x) % m),                     # type 2
			                (x, (y + 2 * x + 1) % m),                 # type 3
			            ]			

			            for typ, (tx, ty) in enumerate(neighbors):
			                v = tx * m + ty
			                # Add forward edge with type typ (0-3)
			                edge_index_list.append([u, v])
			                edge_type_list.append(typ)			

			                # Add reverse edge with type typ + 4 (4-7) if not self-loop
			                if u != v:
			                    edge_index_list.append([v, u])
			                    edge_type_list.append(typ + 4)			

			    edge_index = torch.tensor(edge_index_list).t().contiguous().long()
			    edge_type = torch.tensor(edge_type_list).long()			

			    return edge_index, edge_type


			A_edges_c, edge_type = make_labeled_mgg_graph(int_need)
			A_edges_c = torch.Tensor(np.random.permutation(A_edges_c.max().item() + 1)).long()[A_edges_c]
			Ac, edge_type = subgraph(torch.arange(number_of_spatial_nodes), A_edges_c, edge_attr = edge_type) # [0].cpu().detach().numpy() # .to(device)
			Ac, edge_type = Ac.cpu().detach().numpy(), edge_type.cpu().detach().numpy()

			# A_edges_c = from_networkx(A_edges_c)

	np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_expanders_ver_1.npz'%name_of_project, Ac = Ac)


print("All files saved successfully!")
print("✔ Script execution: Done")


