
import yaml
import numpy as np
import os
import torch
from torch import optim, nn
import shutil
from collections import defaultdict
from sklearn.metrics import r2_score
import pathlib
import pdb
import scipy
from math import floor, sqrt
from torch_cluster import knn
from itertools import product
from torch_geometric.utils import from_networkx
from scipy.optimize import differential_evolution
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import MessagePassing
from skopt.utils import use_named_args
from torch_scatter import scatter
from scipy.stats import pearsonr
import scipy.stats.qmc as qmc
from scipy.spatial import KDTree
from skopt import gp_minimize
from skopt.space import Real
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
	time_shift_range = config['time_shift_range']/2.0 ## Note this scaling (the full window is time shift range)
	scale_time = config['scale_time']
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


# ### Define Fibonnaci sampling routine ####

# Area_globe = 4*np.pi*(earth_radius**2)
# if use_global == True:
#     Area = 4*np.pi*(earth_radius**2)
#     Volume = (4.0*np.pi/3.0)*(r_max**3 - r_min**3)
#     Volume_space = 1.0*Volume

# else:
#     Area = (earth_radius**2)*(np.deg2rad(lon_range_extend[1]) - np.deg2rad(lon_range_extend[0]))*(np.sin(np.pi*lat_range_extend[1]/180.0) - np.sin(np.pi*lat_range_extend[0]/180.0))
#     Volume = Area*(r_max**3 - r_min**3)/(3*(earth_radius**2))
#     Volume_space = 1.0*Volume


# ## Estimate an optimal time scaling for isotropic spacing
# if use_time_shift == True:
#     dx = (Volume/number_of_spatial_nodes)**(1/3)
#     dt = 2*time_shift_range/(number_of_spatial_nodes**(1/4))
#     scale_time_effective = dx/dt
#     print('Isotropic scaling effective time scale: %0.4f m/s'%scale_time_effective) ## For a given spatial and temporal volume
#     ## Could use this to guide how much time window can be increased or decreased

#     dx = (Volume_space)**(1/3)
#     dt = 2*time_shift_range # / # (number_of_spatial_nodes**(1/4))
#     scale_time_effective = dx/dt
#     print('Isotropic scaling effective time scale: %0.4f m/s'%scale_time_effective) ## For a given spatial and temporal volume
#     ## Could use this to guide how much time window can be increased or decreased


#     if use_effective_time_scale == True:
#     	print('Overwriting time scale as effective time scale: %0.8f (from %0.8f)'%(scale_time_effective, scale_time))
#     	scale_time = scale_time_effective # [0]

# def get_initial_spacing_estimates(N, Volume_space, time_range_total, scale_t):
#     # 1. Total 4D Hypervolume
#     V_4d = Volume_space * (scale_t * time_range_total)
    
#     # 2. The Theoretical 4D spacing (if perfectly uniform)
#     # Using 1.0 as the packing factor for a simple 4D hypercube
#     d_4d = (V_4d / N)**(1/4)
    
#     # 3. Marginalized back to physical units
#     # In the metric space, dx = dy = dz = dt_scaled = d_4d
#     expected_dx_meters = d_4d  # Because spatial scaling is 1.0
#     expected_dt_seconds = d_4d / scale_t
    
#     return expected_dx_meters, expected_dt_seconds


# if use_time_shift == True:
# 	Volume = Volume*(2*scale_time*time_shift_range)

# ## Determine nominal node spacing
# if use_time_shift == False:
# 	nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing
# 	nominal_spacing_space = (Volume_space/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing

# else:
# 	nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/4) ## Hex-based spacing
# 	nominal_spacing_space = (Volume_space/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing




# # --- ONE CALL TO RULE THEM ALL ---
# (Volume, Volume_space, Area, nominal_spacing, nominal_spacing_space, nominal_spacing_time) = compute_warped_expected_spacing(
#     N=number_of_spatial_nodes,
#     lat_range=lat_range_extend,
#     lon_range=lon_range_extend,
#     depth_range=depth_range,
#     time_range=time_shift_range,
#     scale_time=scale_time,
#     depth_boost=depth_upscale_factor,
#     use_global=use_global
# )


# print(f"--- Metric Insight Report ---")
# print(f"Expected 4D Metric Spacing: {nominal_spacing:.2f} meters")
# print(f"Projected Spatial Resolution: {nominal_spacing_space:.2f} meters")
# print(f"Projected Temporal Resolution: {nominal_spacing_time:.2f} seconds")





########## Determine initial nominal scales ###############

# N_target = number_of_spatial_nodes

# # --- PHASE 1: Baseline Spatial Resolution ---
# # d_space tells us the physical meters between nodes in 3D
# baseline = compute_warped_expected_spacing(
#     N_target, lat_range=lat_range_extend, lon_range=lon_range_extend,
#     depth_range=depth_range, time_range=time_shift_range,
#     scale_time=1.0, depth_boost=depth_upscale_factor, use_global=use_global
# )
# d_space = baseline[4] # nominal_spacing_space

# # Ideal temporal budget to maintain d_space (seconds)
# dt_slice_ideal = d_space / scale_time 


# initilization_strategy = 'KEEP_FIXED'
# initilization_strategy = 'USE_ISOTROPIC'
# initilization_strategy = 'ADJUST_N'
# initilization_strategy = 'ADJUST_TIME_WINDOW'
# initilization_strategy = 'JOINT_OPTIMIZATION'


# # --- PHASE 2: Selection Logic ---
# if strategy == "KEEP_FIXED":
#     print(f"Strategy: Fixed. Scale: {scale_time}")

# elif strategy == "USE_ISOTROPIC":
#     # Match the 'Velocity' to the current N and Volume
#     dt_slice_current = (2.0 * time_shift_range) / (N_target**0.25)
#     scale_time = d_space / dt_slice_current
#     print(f"Strategy: Isotropic. New Scale: {scale_time:.4f}")

# elif strategy == "ADJUST_N":
#     # Match N to the desired Scale (with Safety Cap)
#     MAX_N = int(15e3) # 500_000 
#     N_required = int(((scale_time * 2.0 * time_shift_range) / d_space)**4)
    
#     if N_required > MAX_N:
#         print(f"Warning: Adjusted N ({N_required}) exceeds cap. Clipping to {MAX_N}")
#         N_target = MAX_N
#     else:
#         N_target = N_required
#     print(f"Strategy: Adjust N. New N: {N_target}")

# elif strategy == "ADJUST_TIME_WINDOW":
#     # Strategy 4: Change the 2T window so N points are isotropic at scale_time
#     # Solving scale_time = d_space / ( (2*T_new) / N^(1/4) ) for T_new:
#     # 2*T_new = (d_space * N^(1/4)) / scale_time
#     new_total_span = (d_space * (N_target**0.25)) / scale_time
#     time_shift_range = new_total_span / 2.0
#     print(f"Strategy: Adjust Window. New Time Range: +/-{time_shift_range:.2f}s")

# elif strategy == "JOINT_OPTIMIZATION":
#     # 1. Set N to your maximum computational budget
#     MAX_N_BUDGET = 200_000 
#     N_target = MAX_N_BUDGET
    
#     # 2. Force scale_time to be isotropic (or a specific preferred value)
#     # We'll use the scale_time provided in the config as the "target velocity"
    
#     # 3. Solve for the Time Window that makes this N and Scale isotropic:
#     # 2*T = (d_space * N^(1/4)) / scale_time
#     new_total_span = (d_space * (N_target**0.25)) / scale_time
#     time_shift_range = new_total_span / 2.0
#     print(f"Strategy: Joint Optimization. Maxed N to {N_target}, adjusted Window to +/-{time_shift_range:.2f}s")


# # --- PHASE 3: Final Execution ---
# number_of_spatial_nodes = N_target ## Update assignmnet of number of nodes if adjusted
# metrics = compute_warped_expected_spacing(
#     number_of_spatial_nodes, lat_range=lat_range_extend, lon_range=lon_range_extend,
#     depth_range=depth_range, time_range=time_shift_range,
#     scale_time=scale_time, depth_boost=depth_upscale_factor, use_global=use_global
# )

# Volume, Volume_space, Area, nominal_spacing, nominal_spacing_space, nominal_spacing_time = metrics

# summary_data = {
#     "Metric": ["Total Nodes (N)", "Time Scale (m/s)", "Time Window (±s)", "Spatial Res (m)", "Temporal Res (s)", "4D Joint Res (m)"],
#     "Value": [
#         f"{N_target:,}", 
#         f"{scale_time:.2f}", 
#         f"{time_shift_range:.2f}", 
#         f"{metrics[4]:.1f}",  # nominal_spacing_space
#         f"{metrics[5]:.3f}",  # nominal_spacing_time
#         f"{metrics[3]:.1f}"   # nominal_spacing_4d
#     ],
#     "Role": ["Cost", "Velocity", "Coverage", "XY-Resolution", "T-Resolution", "FPS-Metric"]
# }

# df_summary = pd.DataFrame(summary_data)
# print("\n" + "="*40)
# print("      GRID CONFIGURATION SUMMARY")
# print("="*40)
# print(df_summary.to_string(index=False))
# print("="*40)



N_target = number_of_spatial_nodes

# --- PHASE 1: Baseline Spatial Resolution ---
baseline = compute_warped_expected_spacing(
    N_target, lat_range=lat_range_extend, lon_range=lon_range_extend,
    depth_range=depth_range, time_range=time_shift_range,
    scale_time=1.0, depth_boost=depth_upscale_factor, use_global=use_global
)
d_space = baseline[4] # nominal_spacing_space

# --- PHASE 2: Selection Logic ---
# (Assumes 'strategy' variable is set previously)
if strategy == "KEEP_FIXED":
    print(f"Strategy: Fixed. Scale: {scale_time}")

elif strategy == "USE_ISOTROPIC":
    dt_slice_current = (2.0 * time_shift_range) / (N_target**0.25)
    scale_time = d_space / dt_slice_current
    print(f"Strategy: Isotropic. New Scale: {scale_time:.4f}")

elif strategy == "ADJUST_N":
    MAX_N = int(15e3) 
    N_required = int(((scale_time * 2.0 * time_shift_range) / d_space)**4)
    if N_required > MAX_N:
        print(f"Warning: Adjusted N ({N_required}) exceeds cap. Clipping to {MAX_N}")
        N_target = MAX_N
    else:
        N_target = N_required
    print(f"Strategy: Adjust N. New N: {N_target}")

elif strategy == "ADJUST_TIME_WINDOW":
    new_total_span = (d_space * (N_target**0.25)) / scale_time
    time_shift_range = new_total_span / 2.0
    print(f"Strategy: Adjust Window. New Time Range: +/-{time_shift_range:.2f}s")

elif strategy == "JOINT_OPTIMIZATION": 

    # Starting intents
    N_init = float(N_target)
    T_init = float(time_shift_range)
    target_W = scale_time 
    # User-specified weight (Default 1.0)
    # Higher alpha_weight = "Don't change N as much"

	# Manual weighting (Replace 'getattr' with a simple variable/default)
    # alpha_weight > 1.0 makes N "stiffer" (harder to change)
    alpha_weight = 1.0

    # alpha_weight = getattr(self, 'alpha_weight', 1.0)
    def objective(params):
        ln_N, ln_T = params
        # Current scale in the warped metric: W = (d_space * N^0.25) / (2T)
        current_W = (d_space * np.exp(ln_N)**0.25) / (2.0 * np.exp(ln_T))
        # MOVERS DISTANCE (Log-space differences)
        # We multiply the N-distance by alpha_weight
        dist_N = alpha_weight * (ln_N - np.log(N_init))**2
        dist_T = (ln_T - np.log(T_init))**2
        # Constraint: Achieve target scale
        constraint_penalty = 1e8 * (np.log(current_W) - np.log(target_W))**2
        return dist_N + dist_T + constraint_penalty

    # Optimize
    res = minimize(objective, x0=[np.log(N_init), np.log(T_init)],  ## Initial bounds for N : 50 to 15e3; bounds for time: 5 s to 3600 s
                   bounds=[(np.log(50), np.log(int(15e3))), (np.log(5.0), np.log(3600))])
    N_target = int(np.round(np.exp(res.x[0])))
    time_shift_range = np.exp(res.x[1])
    # Calculate Percent Changes for the report
    pct_change_N = (N_target / N_init - 1) * 100
    pct_change_T = (time_shift_range / T_init - 1) * 100
    print(f"Joint Optimization (α_weight={alpha_weight}):")
    print(f"  -> N: {N_init:,.0f} to {N_target:,.0f} ({pct_change_N:+.1f}%)")
    print(f"  -> T: ±{T_init:.2f}s to ±{time_shift_range:.2f}s ({pct_change_T:+.1f}%)")


# Sync the main variable
number_of_spatial_nodes = N_target 

# --- PHASE 3: Final Execution ---
metrics = compute_warped_expected_spacing(
    N_target, lat_range=lat_range_extend, lon_range=lon_range_extend,
    depth_range=depth_range, time_range=time_shift_range,
    scale_time=scale_time, depth_boost=depth_upscale_factor, use_global=use_global
)

# Unpack for the summary
Volume, Volume_space, Area, nominal_spacing_4d, nominal_spacing_space, nominal_spacing_time = metrics

# --- SUMMARY TABLE ---
summary_data = {
    "Metric": ["Total Nodes (N)", "Time Scale (m/s)", "Time Window (±s)", "Spatial Res (m)", "Temporal Res (s)", "4D Joint Res (m)"],
    "Value": [
        f"{N_target:,}", 
        f"{scale_time:.2f}", 
        f"{time_shift_range:.2f}", 
        f"{nominal_spacing_space:.1f}", 
        f"{nominal_spacing_time:.3f}", 
        f"{nominal_spacing_4d:.1f}"
    ],
    "Role": ["Cost", "Velocity", "Coverage", "XY-Resolution", "T-Resolution", "FPS-Metric"]
}

df_summary = pd.DataFrame(summary_data)
print("\n" + "="*45)
print(f"      GRID CONFIGURATION: {strategy}")
print("="*45)
print(df_summary.to_string(index=False))
print("="*45)


# getattr(self, 'alpha_weight', 1.0)


########## Determine initial nominal scales ###############




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
    Volume_space_metric = Area_at_surface * thickness_metric    

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

# # --- 1. Compute Metric Spatial Volume (The "Unrolled Slab") ---
# # Use earth_radius area to account for 1/r warp and thickness*depth_boost
# thickness_metric = (r_max - r_min) * depth_boost
# Volume_space_metric = Area_at_surface * thickness_metric

# # --- 2. 3D Nominal Spacing (Spatial Projection) ---
# # This tells you: "If I only had 3D space, what would the spacing be?"
# hex_factor_3d = 0.74048
# nominal_spacing_space = (Volume_space_metric / (hex_factor_3d * N)) ** (1.0 / 3.0)

# # --- 3. 4D Hypervolume (Space x Scaled Time) ---
# # Total time span (2T) stretched by scale_time
# hypervolume_4d_metric = Volume_space_metric * (2.0 * time_range * scale_time)

# # --- 4. 4D Joint Spacing (The FPS Target) ---
# # This is the actual distance (in metric units) FPS will enforce
# nominal_spacing_4d = (hypervolume_4d_metric / N) ** (1.0 / 4.0)

# # --- 5. "Raw" Nominal Time Spacing ---
# # This represents the temporal "slot" width in seconds.
# # We use the 4th-root of N to show how the 4D density partitions time.
# nominal_spacing_time = (2.0 * time_range) / (N ** (1.0 / 4.0))


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



def u_to_geodetic_lat(u_random, lat_range):
    """
    Converts a uniform random variable u [0, 1] to Geodetic Latitude 
    using an Authalic (equal-area) mapping for the WGS84 ellipsoid.
    """
    # WGS84 Constant: e (eccentricity)
    e = 0.0818191908426
    
    def get_q(lat_deg):
        # q is the authalic part of the projection
        sin_lat = np.sin(np.deg2rad(lat_deg)) # sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
        # sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
        term1 = sin_lat / (1 - e**2 * sin_lat**2)
        term2 = (1 / (2 * e)) * np.log((1 - e * sin_lat) / (1 + e * sin_lat))
        return (1 - e**2) * (term1 - term2)

    # Calculate q limits for your range
    q_min = get_q(lat_range[0])
    q_max = get_q(lat_range[1])
    
    # Target q for this point
    # pdb.set_trace()
    q_target = q_min + u_random * (q_max - q_min)
    
    # Map q back to Latitude (Approximation for WGS84)
    # The difference between Geodetic and Authalic is very small, 
    # so we can use a high-order series or a simple arcsin of (q / q_polar)
    q_polar = get_q(90.0)
    return np.rad2deg(np.arcsin(q_target / q_polar))


# # 1. Get the surface position and the normal vector at that latitude/longitude
# # Assuming ftrns1_abs returns XYZ in meters
# xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), np.zeros((len(phi),1))), axis=1))

# # 2. For WGS84, the normal vector 'n' at the surface is easy to compute 
# # if you have the XYZ of a surface point:
# # n = [x/(a^2), y/(a^2), z/(b^2)] then normalize
# a = 6378137.0
# b = 6356752.314245
# n = xyz_surface / np.array([a**2, a**2, b**2])
# n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)

# # 3. Instead of scaling the vector from the center, 
# # you start at the surface and move ALONG the normal.
# # Note: Since your 'r' is distance from center, we need 'h' (height).
# # Height h = r_actual - r_surface (roughly), but better to use your depth_range directly.

# # Better Approach:
# # Let Sobol sample 'h' (altitude/depth) directly instead of 'r'
# h = expanded_depth[0] + u[:, [2]] * (expanded_depth[1] - expanded_depth[0])

# # XYZ = Surface_XYZ + (Normal_Vector * Height)
# xyz = xyz_surface + (n_unit * h.reshape(-1,1))

# def regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0):

# 	if use_spherical == False:
# 		a = 6378137.0
# 		b = 6356752.3142
# 	else:
# 		a = 6371e3
# 		b = 6371e3

# 	if buffer_scale > 0.0:
# 		## Use a buffer around min-max regions. How to estimate? First estimate volume
# 		Volume, Volume_space, _, nominal_spacing_space, nominal_spacing_time = compute_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
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
# 		Volume_expanded, Volume_space_expanded, _, _, _ = compute_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
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
# 		r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
# 		r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
# 		xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
# 		r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
# 		r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
# 		xyz = (r*xyz_surface)/r_surface
# 		x_grid = ftrns2_abs(xyz) 

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
# 		r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
# 		r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
# 		xyz = (r*xyz_surface)/r_surface
# 		x_grid = ftrns2_abs(xyz)

# 		if use_time == True:
# 			t = -time_shift_range + 2 * time_shift_range * u[:, [3]]
# 			x_grid = np.concatenate((x_grid, t), axis = 1)

# 		return x_grid



# def get_metric_space(x_grid, depth_boost, scale_t):

#     # 1. True ECEF
#     xyz_ecef = ftrns1_abs(x_grid[:, :3])
#     radii = np.linalg.norm(xyz_ecef, axis = 1)
    
#     # 2. WGS84 Normal Vector
#     a, b = 6378137.0, 6356752.314245 ## Note: can technially adjust a and b locally for this formula by the depth
#     n = xyz_ecef / np.array([a**2, a**2, b**2])
#     n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
    
#     # 3. Apply depth_boost along the normal
#     # We move the point from its true ECEF position to a 'boosted' position
#     # that represents its 'importance' to the FPS algorithm.
#     true_depths = x_grid[:, 2:3]
#     xyz_boosted = xyz_ecef + (n_unit * true_depths * (depth_boost - 1.0))
    
#     # 4. Temporal component
#     time_boosted = x_grid[:, 3:4] * scale_t
    
#     return np.hstack([xyz_boosted, time_boosted]), radii


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



# def regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, depth_boost = depth_upscale_factor, N_target = None, buffer_scale = 0.0):

# 	if use_spherical == False:
# 		a = 6378137.0
# 		b = 6356752.314245
# 	else:
# 		a = 6371e3
# 		b = 6371e3

# 	if buffer_scale > 0.0:
# 		## Use a buffer around min-max regions. How to estimate? First estimate volume
# 		Volume, Volume_space, _, nominal_spacing_space, nominal_spacing_time = compute_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
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
# 		Volume_expanded, Volume_space_expanded, _, _, _ = compute_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
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

		lat_mid = np.mean(lat_range)
		pad_lat = (nominal_spacing_space * buffer_scale / earth_radius) * (180 / np.pi)
		pad_lon = (nominal_spacing_space * buffer_scale / (earth_radius * np.cos(np.deg2rad(lat_mid)))) * (180 / np.pi) # Adjust lon padding for the convergence of meridians
		pad_depth = nominal_spacing_space * buffer_scale # 3. Calculate Depth and Time Padding
		pad_time = nominal_spacing_time * buffer_scale

		# 4. Define New Ranges
		expanded_lat = list(np.array([lat_range[0] - pad_lat, lat_range[1] + pad_lat]).clip(-90.0, 90.0)) if use_global == False else lat_range
		expanded_lon = [lon_range[0] - pad_lon, lon_range[1] + pad_lon] if use_global == False else lon_range
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




# def farthest_point_sampling(points_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = None, device = device):
    
#     """
#     points: [N, 3] or [N, 4] (already scaled/transformed)
#     target_N: Number of 'real' points to collect
#     mask_candidates: Tensor/Array of 1s (real) and 0s (buffer/mirrored)
#     """    

#     scale_val = 10000.0
#     points = points_candidates.copy()
#     if points.shape[1] == 4: points[:, 3] *= scale_time
#     if depth_boost != 1.0: points = ftrns1_abs(ftrns2_abs(points) * np.array([1.0, 1.0, depth_boost, 1.0]))
#     origin = points[:, :3].mean(axis = 0, keepdims = True)
#     points[:, :3] -= origin
#     points = torch.as_tensor(points / scale_val, device = device, dtype = torch.float64)

#     if mask_candidates is None: mask_candidates = np.ones(len(points))
#     mask = torch.as_tensor(mask_candidates, device = device, dtype = torch.bool)
#     assert(len(mask) == len(points))
#     assert(mask.sum().item() >= target_N)
#     N, C = points.shape
    
#     # 1. Initialize distance array
#     # If we have boundary points (mask == 0), we pre-calculate distances to them
#     distance = torch.full((N,), float('inf'), device = device, dtype = torch.float64)
    
#     boundary_indices = torch.where(~mask)[0]
#     real_indices = torch.where(mask)[0]

#     # 2. Pre-process boundary Points (The repulsion field)
#     if len(boundary_indices) > 0:
#         # Optimization: Update distance array with the proximity to any ghost point
#         # For very large N, we do this in chunks to avoid OOM
#         for i in range(0, len(boundary_indices), 500):
#             batch = boundary_indices[i:i+500]
#             # dists shape: [len(batch), N]
#             dists = torch.cdist(points[batch], points, p=2)**2
#             distance = torch.min(distance, torch.min(dists, dim=0)[0])
        
#         # Ensure ghost points themselves are never selected
#         distance[boundary_indices] = -1.0

#     # 3. Choose the first REAL point
#     # Instead of random, we pick the point farthest from the boundary ghosts
#     # If no ghosts exist, we default to the point closest to the centroid
#     if len(boundary_indices) > 0:
#         farthest = torch.argmax(distance).item()
#     else:
#         centroid = points[real_indices].mean(0, keepdims=True)
#         dist_centroid = torch.sum((points[real_indices] - centroid)**2, dim=1)
#         farthest = real_indices[torch.argmin(dist_centroid)].item()

#     collected_indices = []
#     cnt_found = 0

#     # 4. Main FPS Loop
#     while cnt_found < target_N:
#         collected_indices.append(farthest)
#         cnt_found += 1 # We only ever pick real points now
            
#         centroid_pt = points[farthest, :].view(1, C)
#         dist = torch.sum((points - centroid_pt) ** 2, dim=-1)
        
#         distance = torch.min(distance, dist)
#         distance[farthest] = -1.0
        
#         if cnt_found < target_N:
#             farthest = torch.argmax(distance).item()

#     # Final Filter
#     final_indices = torch.tensor(collected_indices, device=device)
#     return ftrns2_abs(points_candidates[final_indices.cpu().numpy()])



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



use_poisson_filtering = False 
use_farthest_point_filtering = True 


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


# def compute_boundary_biases(x_grid, nn_4d, lat_range, lon_range, depth_range, time_range, scale_t, earth_radius=earth_radius):
#     """
#     Computes biases by comparing physical distances to boundaries 
#     against the 4D metric nearest-neighbor distance.
#     """
#     N = len(x_grid)
#     x_phys_3d = ftrns1_abs(x_grid[:, :3]) # Your WGS84 -> XYZ conversion
    
#     # 1. Temporal Bias (1D is simple)
#     t_min, t_max = -time_range, time_range
#     dist_t = np.minimum(np.abs(x_grid[:, 3] - t_min), np.abs(x_grid[:, 3] - t_max)) * scale_t
#     t_bias = np.mean(dist_t) / (np.mean(nn_4d) + 1e-9)

#     # 2. Radial/Depth Bias (Ellipsoidal Shell)
#     # Distance from the center of the Earth to the node
#     r_node = np.linalg.norm(x_phys_3d, axis=1)
#     # The actual shell boundaries (meters from center)
#     r_top = earth_radius - depth_range[0]
#     r_bottom = earth_radius - depth_range[1]
#     dist_r = np.minimum(np.abs(r_node - r_top), np.abs(r_node - r_bottom))
#     r_bias = np.mean(dist_r) / (np.mean(nn_4d) + 1e-9)

#     # 3. Lateral Bias (Regional Only)
#     # We find the distance to the 4 vertical 'planes' defined by lat/lon
#     # Using a simplified but accurate ellipsoidal distance for regional bounds
#     lat_min, lat_max = lat_range
#     lon_min, lon_max = lon_range
    
#     # Haversine distance to Lat/Lon lines (in meters)
#     def haversine_m(lat1, lon1, lat2, lon2):
#         dlat = np.deg2rad(lat2 - lat1)
#         dlon = np.deg2rad(lon2 - lon1)
#         a = np.sin(dlat/2)**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon/2)**2
#         return 2 * earth_radius * np.arcsin(np.sqrt(a))

#     # Distance to the nearest Latitude boundary
#     d_lat = haversine_m(x_grid[:, 0], x_grid[:, 1], 
#                         np.clip(x_grid[:, 0], lat_min, lat_max), x_grid[:, 1])
#     # Note: clip creates the 'closest point' on the boundary line
#     d_lat = np.minimum(haversine_m(x_grid[:,0], x_grid[:,1], lat_min, x_grid[:,1]),
#                        haversine_m(x_grid[:,0], x_grid[:,1], lat_max, x_grid[:,1]))

#     # Distance to the nearest Longitude boundary
#     d_lon = np.minimum(haversine_m(x_grid[:,0], x_grid[:,1], x_grid[:,0], lon_min),
#                        haversine_m(x_grid[:,0], x_grid[:,1], x_grid[:,0], lon_max))

#     lat_lon_bias = np.mean(np.minimum(d_lat, d_lon)) / (np.mean(nn_4d) + 1e-9)
    
#     return t_bias, r_bias, lat_lon_bias

# def compute_boundary_biases(x_grid, x_phys_3d, nn_4d, lat_range, lon_range, depth_range, time_range, scale_t, depth_boost, use_global = use_global, earth_radius=6378137.0):
#     """
#     Computes boundary biases in the 4D metric space.
#     Target bias is 0.5 (perfectly centered between wall and neighbors).
#     """
#     N = len(x_grid)
    
#     # 1. Temporal Boundary (Metric Space)
#     # distance in seconds * scale_t = distance in 'scaled meters'
#     dist_t = np.minimum(np.abs(x_grid[:, 3] - (-time_range)), 
#                         np.abs(x_grid[:, 3] - time_range)) * scale_t
#     t_bias = np.mean(dist_t) / (np.mean(nn_4d) + 1e-9)

#     # 2. Radial/Depth Boundary (Metric Space)
#     # We must apply depth_boost here to match the 4D metric distortion
#     r_nodes = np.linalg.norm(x_phys_3d, axis=1)
#     r_top = earth_radius - depth_range[0]
#     r_bottom = earth_radius - depth_range[1]
    
#     # Apply boost to the differences to stay in 'metric meters'
#     dist_r = np.minimum(np.abs(r_nodes - r_top), np.abs(r_nodes - r_bottom)) * depth_boost
#     r_bias = np.mean(dist_r) / (np.mean(nn_4d) + 1e-9)

#     # 3. Lateral Boundary (Regional Only)
#     if not use_global:
#         # Proper Haversine-based distance to boundary lines
#         def dist_to_line(lats, lons, target_lat, target_lon, mode='lat'):
#             if mode == 'lat': # Distance to a constant latitude line
#                 dlat = np.deg2rad(target_lat - lats)
#                 return np.abs(dlat) * earth_radius
#             else: # Distance to a constant longitude line
#                 dlon = np.deg2rad(target_lon - lons)
#                 return np.abs(dlon) * earth_radius * np.cos(np.deg2rad(lats))

#         d_lat = np.minimum(dist_to_line(x_grid[:,0], x_grid[:,1], lat_range[0], None, 'lat'),
#                            dist_to_line(x_grid[:,0], x_grid[:,1], lat_range[1], None, 'lat'))
#         d_lon = np.minimum(dist_to_line(x_grid[:,0], x_grid[:,1], None, lon_range[0], 'lon'),
#                            dist_to_line(x_grid[:,0], x_grid[:,1], None, lon_range[1], 'lon'))
        
#         # Lateral scaling is 1.0, so no boost needed here
#         lat_lon_bias = np.mean(np.minimum(d_lat, d_lon)) / (np.mean(nn_4d) + 1e-9)
#     else:
#         # Global Lon is periodic (no boundary); Poles are points, not edges.
#         lat_lon_bias = 0.5 

#     return t_bias, r_bias, lat_lon_bias


# def compute_boundary_biases(x_grid, nn_4d, lat_range, lon_range, depth_range, time_range, scale_t, depth_boost, use_global = use_global):
#     """
#     Corrected Boundary Bias:
#     Calculates the 4D metric distance to the nearest "hyper-plane" boundary.
#     """
#     N = len(x_grid)
    
#     # --- 1. PRECISE RADIAL BOUNDARIES ---
#     # Use your actual projection to find the exact metric radii

#     r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_boost*depth_range[0])])), axis=1)
#     r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_boost*depth_range[1])])), axis=1)
#     r_nodes = np.linalg.norm(ftrns1_abs(x_grid[:, :3]*np.array([1.0, 1.0, depth_boost]).reshape(1,-1)), axis=1)
    
#     # Boundary radial values (these vary slightly with latitude in WGS84)
#     # For a point-by-point check:
#     # r_top_nodes = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[0])]), axis=1)
#     # r_bot_nodes = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[1])]), axis=1)
    
#     # --- 2. 4D METRIC DISTANCES TO BOUNDARIES ---
    
#     # Distance to Depth boundaries (Metric)
#     dist_r_metric = np.minimum(np.abs(r_nodes - r_max_local), 
#                                np.abs(r_nodes - r_min_local))
    
#     # Distance to Time boundaries (Metric)
#     dist_t_metric = np.minimum(np.abs(x_grid[:, 3] - (-time_range)), 
#                                np.abs(x_grid[:, 3] - time_range)) * scale_t

#     # Distance to Lateral boundaries (Metric - Scaled 1.0)
#     if not use_global:
#         # Using the actual metric distance (Cartesian) to the boundary planes
#         # simplified here as haversine for regional scaling
#         d_lat = np.minimum(np.abs(x_grid[:,0] - lat_range[0]), np.abs(x_grid[:,0] - lat_range[1])) * 111319.5
#         d_lon = np.minimum(np.abs(x_grid[:,1] - lon_range[0]), np.abs(x_grid[:,1] - lon_range[1])) * 111319.5 * np.cos(np.deg2rad(x_grid[:,0]))
#         dist_lat_metric = np.minimum(d_lat, d_lon)
#     else:
#         dist_lat_metric = np.full(N, np.inf) # No lateral boundaries in global

#     # --- 3. THE "TRUE" 4D PROXIMITY ---
#     # For each point, what is the distance to the CLOSEST of all 4D boundaries?
#     dist_to_any_boundary = np.minimum(np.minimum(dist_r_metric, dist_t_metric), dist_lat_metric)
    
#     # The bias is the ratio of [Dist to closest Wall] / [Dist to closest Neighbor]
#     # In a balanced FPS grid, this ratio should converge toward 0.5.
#     total_bias = np.mean(dist_to_any_boundary) / (np.mean(nn_4d) + 1e-9)
    
#     # Breakdown for diagnostic purposes
#     r_bias_diag = np.mean(dist_r_metric) / (np.mean(nn_4d) + 1e-9)
#     t_bias_diag = np.mean(dist_t_metric) / (np.mean(nn_4d) + 1e-9)

#     return total_bias, r_bias_diag, t_bias_diag

# def compute_boundary_biases(x_grid, nn_4d, lat_range, lon_range, depth_range, time_range, use_global = use_global):
#     """
#     Computes bias by comparing the neighbor distances of points near the 
#     edges vs the bulk average. No unsafe coordinate projections required.
#     """
#     N = len(x_grid)
#     d_bulk = np.mean(nn_4d)
    
#     # Define 'Edge' as the outer 5% of the range in any dimension
#     def get_edge_mask(data, bounds, fraction=0.1):

#         span = bounds[1] - bounds[0]
#         lower = bounds[0] + fraction * span
#         upper = bounds[1] - fraction * span

#         return (data < lower) | (data > upper)

#     # Masks for each dimension
#     if use_global == False:
#     	m_lat = get_edge_mask(x_grid[:, 0], lat_range)
#     	m_lon = get_edge_mask(x_grid[:, 1], lon_range)
#     else:
#     	m_lat = np.

#     m_dep = get_edge_mask(x_grid[:, 2], depth_range)
#     m_tim = get_edge_mask(x_grid[:, 3], [-time_range, time_range])
    
#     # Combined mask for any spatial or temporal boundary
#     edge_mask = m_lat | m_lon | m_dep | m_tim
    
#     if np.any(edge_mask):
#         d_edge = np.mean(nn_4d[edge_mask])
#         # Bias = How much 'roomier' or 'tighter' the edges are vs the bulk
#         boundary_bias = d_edge / (d_bulk + 1e-9)
#     else:
#         boundary_bias = 1.0

#     return boundary_bias, edge_mask

# distribution, a point on the edge should be roughly half as far from the boundary as it is from its nearest neighbor.

# If Dist 
# Boundary
# ​	
#  ≪0.5×d 
# NN
# ​	
#  , the point is "slamming" into the wall (Crowding).

# If Dist 
# Boundary
# ​	
#  ≫0.5×d 
# NN
# ​	
#  , the point is "retreating" from the wall (Erosion).

# def compute_boundary_biases(x_grid, nn_4d, lat_range, lon_range, depth_range, time_range, frac_edge = 0.05, use_global = use_global):

#     N = len(x_grid)
#     d_bulk = np.mean(nn_4d)
#     # frac = 0.1 # Edge zone definition
    
#     def get_linear_edge_mask(data, bounds):
#         span = bounds[1] - bounds[0]
#         return (data < (bounds[0] + frac_edge*span) ) | (data > (bounds[1] - frac_edge*span))

#     # 1. Linear dimensions are easy
#     m_lat = get_linear_edge_mask(x_grid[:, 0], lat_range)
#     m_depth = get_linear_edge_mask(x_grid[:, 2], depth_range)
#     m_time = get_linear_edge_mask(x_grid[:, 3], [-time_range, time_range])
    
#     # 2. Longitude: The Wraparound Check
#     if use_global:
#         # In a global model, longitude has no "edge." 
#         # Points at 180 are neighbors with -180.
#         m_lon = np.zeros(N, dtype=bool)
#     else:
#         # Shortest angular distance to the lon_range boundaries
#         lon_min, lon_max = lon_range
#         lon_span = (lon_max - lon_min) % 360
#         if lon_span == 0: lon_span = 360 # Handle full circle cases
        
#         # Distance from each point to the 'start' of the lon range
#         dist_to_start = (x_grid[:, 1] - lon_min) % 360
#         # Distance from each point to the 'end'
#         dist_to_end = (lon_max - x_grid[:, 1]) % 360
        
#         m_lon = (dist_to_start < frac_edge * lon_span) | (dist_to_end < frac_edge * lon_span)

#     # Combined Edge Mask
#     edge_mask = m_lat | m_lon | m_depth | m_time
#     # pdb.set_trace()
    
#     bias = np.mean(nn_4d[edge_mask]) / (d_bulk + 1e-9) if np.any(edge_mask) else 1.0
#     bias_lat = np.mean(nn_4d[m_lat]) / (d_bulk + 1e-9) if np.any(m_lat) else 1.0
#     bias_lon = np.mean(nn_4d[m_lon]) / (d_bulk + 1e-9) if np.any(m_lon) else 1.0
#     bias_depth = np.mean(nn_4d[m_depth]) / (d_bulk + 1e-9) if np.any(m_depth) else 1.0
#     bias_time = np.mean(nn_4d[m_time]) / (d_bulk + 1e-9) if np.any(m_time) else 1.0


#     return bias, bias_lat, bias_lon, bias_depth, bias_time, {"lat": m_lat, "lon": m_lon, "dep": m_depth, "tim": m_time}


# def compute_boundary_biases(x_grid, lat_range, lon_range, depth_range, time_range, frac_edge = 0.1, use_global = use_global):
#     """
#     Computes density ratios at the boundaries. 
#     Ratio > 1.0: Clumping (Excess points at the edge)
#     Ratio < 1.0: Erosion (Empty space at the edge)
#     """
#     N = len(x_grid)
#     results = {}
    
#     # We check the first/last 10% of the coordinate ranges
#     def get_dim_density(data, bounds, name):

#         # Define 10 bins
#         total_span = bounds[1] - bounds[0]        
#         bin_number = int(1.0/frac_edge)
#         counts, edges = np.histogram(data, bins = bin_number, range = (bounds[0], bounds[1]))
        
#         # In a perfect grid, every bin has N/10 points.
#         # However, for Lat/Depth, we must normalize by volume!
#         expected_counts = np.full(bin_number, N/bin_number)
        
#         if name == 'Lat':

#             # Volume of a latitude slice is prop to sin(lat2) - sin(lat1)
#             bin_lats = np.linspace(bounds[0], bounds[1], 11)
#             vols = np.sin(np.deg2rad(bin_lats[1:])) - np.sin(np.deg2rad(bin_lats[:-1]))
#             expected_counts = (vols / np.sum(vols)) * N

#         elif name == 'Depth':

#             # Volume of a shell slice is prop to r_outer^3 - r_inner^3
#             # (Simplifying here, but ideally uses your r_max logic)
#             bin_depths = np.linspace(bounds[0], bounds[1], 11)
#             r = 6371 - bin_depths # earth radius - depth
#             vols = np.abs(r[:-1]**3 - r[1:]**3)
#             expected_counts = (vols / np.sum(vols)) * N
            
#         # Density Ratio = Actual / Expected
#         # We average the two boundary bins (index 0 and index 9)
#         boundary_density = (counts[0] + counts[-1]) / (expected_counts[0] + expected_counts[-1] + 1e-9)

#         return boundary_density

#     results['Time'] = get_dim_density(x_grid[:, 3], [-time_range, time_range], 'Time')
#     results['Depth'] = get_dim_density(x_grid[:, 2], depth_range, 'Depth')
#     results['Lat'] = get_dim_density(x_grid[:, 0], lat_range, 'Lat')
    
#     return results


# def get_q_wgs84(lat_deg):
#         sin_lat = np.sin(np.deg2rad(lat_deg))
#         # Ensure we don't have log(0) at poles (though unlikely in regional)
#         sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
#         term1 = sin_lat / (1 - e**2 * sin_lat**2)
#         term2 = (1 / (2 * e)) * np.log((1 - e * sin_lat) / (1 + e * sin_lat))
#         return (1 - e**2) * (term1 - term2)

# def get_dim_density_ellipsoidal(data, bounds, name, N, frac_edge = 0.03, earth_radius=earth_radius):

#     num_bins = int(1.0/frac_edge)
#     counts, edges = np.histogram(data, bins = num_bins, range = (bounds[0], bounds[1]))
    
#     # WGS84 Eccentricity
#     e = 0.0818191908426

#     def get_q_wgs84(lat_deg):
#         sin_lat = np.sin(np.deg2rad(lat_deg))
#         # Ensure we don't have log(0) at poles (though unlikely in regional)
#         sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
#         term1 = sin_lat / (1 - e**2 * sin_lat**2)
#         term2 = (1 / (2 * e)) * np.log((1 - e * sin_lat) / (1 + e * sin_lat))
#         return (1 - e**2) * (term1 - term2)

#     # --- 1. ELLIPSOIDAL WEIGHTING ---
#     if name == 'Lat':
#         # Area of an ellipsoidal belt is proportional to the difference in q values
#         q_edges = get_q_wgs84(edges)
#         bin_areas = np.diff(q_edges)
#         expected_counts = (bin_areas / np.sum(bin_areas)) * N
        
#     elif name == 'Depth':
#         # Even on an ellipsoid, the radial 'shell' volume is dominated by r^3.
#         # However, for perfection, we use r = (R_local - depth)
#         # Here we use a standard r^3 weighting:
#         r_edges = earth_radius - edges
#         bin_volumes = np.abs(np.diff(r_edges**3))
#         expected_counts = (bin_volumes / np.sum(bin_volumes)) * N
        
#     else: # Time or Longitude
#         expected_counts = np.full(num_bins, N / num_bins)

#     # --- 2. BOUNDARY RATIO CALCULATION ---
#     # We look at the very first and very last bin
#     # Ratio > 1.0: Clumping | Ratio < 1.0: Erosion
#     boundary_ratio = (counts[0] + counts[-1]) / (expected_counts[0] + expected_counts[-1] + 1e-9)
    
#     return boundary_ratio, counts, expected_counts


# def check_boundary_densities(x_grid, lat_range, lon_range, depth_range, time_range, use_global = use_global):
#     """
#     Calls the ellipsoidal density check for each dimension and returns a health dict.
#     """
#     N = len(x_grid)
#     boundary_health = {}
    
#     # Mapping dimensions to their grid column index and bounds
#     dims = {
#         'Lat':   {'idx': 0, 'bounds': lat_range},
#         'Lon':   {'idx': 1, 'bounds': lon_range},
#         'Depth': {'idx': 2, 'bounds': depth_range},
#         'Time':  {'idx': 3, 'bounds': [-time_range, time_range]} # [-time_range, time_range]
#     }
    
#     for name, config in dims.items():
#         # If global, we skip Lon as it has no boundary (periodic)
#         if name == 'Lon' and use_global:
#             health_results[name] = 1.0 # Perfect seating by definition
#             continue

#         if name == 'Depth':
#             val = np.linalg.norm(ftrns1_abs(x_grid), axis = 1)
#             r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[0])])), axis=1).min()
#             r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[1])])), axis=1).max()
#             bounds = [r_min_local, r_max_local]

#         else:
#             val = x_grid[:, config['idx']]
#             bounds = config['bounds']
            
#         ratio, counts, expected = get_dim_density_ellipsoidal(
#             data=val,
#             bounds=bounds,
#             name=name,
#             N=N
#         )
#         boundary_health[name] = ratio
        
#     return boundary_health



# def get_dim_density_ellipsoidal(data, bounds, name, N, frac_edge = 0.03, r_context = None, earth_radius = earth_radius):

#     num_bins = int(1.0/frac_edge)
#     counts, edges = np.histogram(data, bins = num_bins, range = (bounds[0], bounds[1]))
    
#     # WGS84 Eccentricity
#     e = 0.0818191908426

#     def get_q_wgs84(lat_deg):
#         sin_lat = np.sin(np.deg2rad(lat_deg))
#         # Ensure we don't have log(0) at poles (though unlikely in regional)
#         sin_lat = np.clip(sin_lat, -0.999999, 0.999999)
#         term1 = sin_lat / (1 - e**2 * sin_lat**2)
#         term2 = (1 / (2 * e)) * np.log((1 - e * sin_lat) / (1 + e * sin_lat))
#         return (1 - e**2) * (term1 - term2)

#     # --- 1. ELLIPSOIDAL WEIGHTING ---
#     if name == 'Lat':
#         # Area of an ellipsoidal belt is proportional to the difference in q values
#         q_edges = get_q_wgs84(edges)
#         bin_areas = np.diff(q_edges)
#         expected_counts = (bin_areas / np.sum(bin_areas)) * N
        
#     # elif name == 'Depth':
#     #     # Even on an ellipsoid, the radial 'shell' volume is dominated by r^3.
#     #     # However, for perfection, we use r = (R_local - depth)
#     #     # Here we use a standard r^3 weighting:
#     #     r_edges = earth_radius - edges
#     #     bin_volumes = np.abs(np.diff(r_edges**3))
#     #     expected_counts = (bin_volumes / np.sum(bin_volumes)) * N

#     elif name == 'Depth' and r_context is not None:

#         r_min, delta_r = r_context
#         # edges are norm values 0 to 1
#         # r_edges = r_min + (linear_step * delta_r)
#         r_edges = r_min + edges * delta_r
        
#         # Volume of a shell slice is r_outer^3 - r_inner^3
#         # Since r_edges is strictly increasing, diff is positive
#         bin_volumes = np.diff(r_edges**3)
        
#         expected_counts = (bin_volumes / (np.sum(bin_volumes) + 1e-12)) * N

        
#     else: # Time or Longitude
#         expected_counts = np.full(num_bins, N / num_bins)

#     # --- 2. BOUNDARY RATIO CALCULATION ---
#     # We look at the very first and very last bin
#     # Ratio > 1.0: Clumping | Ratio < 1.0: Erosion
#     boundary_ratio = (counts[0] + counts[-1]) / (expected_counts[0] + expected_counts[-1] + 1e-9)
    
#     return boundary_ratio, counts, expected_counts


# def check_boundary_densities(x_grid, lat_range, lon_range, depth_range, time_range, use_global = use_global):
#     """
#     Calls the ellipsoidal density check for each dimension and returns a health dict.
#     """
#     N = len(x_grid)
#     boundary_health = {}
    
#     # Mapping dimensions to their grid column index and bounds
#     dims = {
#         'Lat':   {'idx': 0, 'bounds': lat_range},
#         'Lon':   {'idx': 1, 'bounds': lon_range},
#         'Depth': {'idx': 2, 'bounds': depth_range},
#         'Time':  {'idx': 3, 'bounds': [-time_range, time_range]} # [-time_range, time_range]
#     }
    
#     for name, config in dims.items():

#         # If global, we skip Lon as it has no boundary (periodic)
#         if name == 'Lon' and use_global:
#             health_results[name] = 1.0 # Perfect seating by definition

#             continue

#         if name == 'Depth':

#             # 1. Project actual radii (distance from geocenter)
#             r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
            
#             # 2. Project local radii for your specific depth limits
#             # depth_range[0] is typically the 'start' and [1] the 'end'
#             # We determine which is physically lower vs higher
#             r_bound_0 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[0])]), axis=1)
#             r_bound_1 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[1])]), axis=1)
            
#             r_min_local = np.minimum(r_bound_0, r_bound_1)
#             r_max_local = np.maximum(r_bound_0, r_bound_1)
            
#             # 3. Normalize: 0.0 (Deepest/Bottom) to 1.0 (Highest/Top)
#             val = (r_actual**3 - r_min_local**3) / (r_max_local**3 - r_min_local**3 + 1e-12)
#             bounds = [0.0, 1.0]
            
#             # 4. Context for the r^3 expectation
#             avg_r_min = np.mean(r_min_local)
#             avg_delta_r = np.mean(r_max_local - r_min_local)
            
#             ratio, counts, expected = get_dim_density_ellipsoidal(
#                 data=val,
#                 bounds=bounds,
#                 name=name,
#                 N=N,
#                 r_context=(avg_r_min, avg_delta_r)
#             )

#         else:

#             val = x_grid[:, config['idx']]
#             bounds = config['bounds']
            
#         ratio, counts, expected = get_dim_density_ellipsoidal(
#             data=val,
#             bounds=bounds,
#             name=name,
#             N=N
#         )
#         boundary_health[name] = ratio
        
#     return boundary_health

# def get_dim_density_simple(data, bounds, N, frac_edge=0.03):

#     num_bins = int(1.0/frac_edge)
#     counts, _ = np.histogram(data, bins=num_bins, range=(bounds[0], bounds[1]))
#     # In this normalized space, every bin SHOULD have exactly N/num_bins points
#     expected_per_bin = N / num_bins
#     # Ratio of Actual Boundary counts to Expected
#     boundary_ratio = (counts[0] + counts[-1]) / (2 * expected_per_bin + 1e-9)

#     return boundary_ratio, counts, expected_per_bin

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


## Could add Ricci curvature to the loss function


## Expected velocity scaling
# # Penalty if scale_t wanders too far from 5000
# reg_penalty = 0.01 * ((scale_t - 5000) / 5000)**2
# return total_cdf_loss + seating_penalty + reg_penalty


## Coupled time and depth scaling
# Example: If you expect scale_t ~ 5000 and depth_boost ~ 1.0
# Penalty = abs(log(scale_t / 5000)) + abs(log(depth_boost / 1.0))



# def compute_cdf_analysis(x_grid, lat_range, lon_range, depth_range, time_range):
#     N = len(x_grid)
#     empirical_cdf = np.linspace(0, 1, N)
#     total_cdf_loss = 0
    
#     # 1. Latitude: Authalic (Area-Corrected)
#     q_vals = get_wgs84_area_val(x_grid[:, 0])
#     q_min, q_max = get_wgs84_area_val(lat_range[0]), get_wgs84_area_val(lat_range[1])
#     val_lat = (q_vals - q_min) / (q_max - q_min + 1e-12)
#     total_cdf_loss += np.mean(np.abs(np.sort(val_lat) - empirical_cdf))

#     # 2. Longitude: Linear (or Circular)
#     val_lon = (x_grid[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0] + 1e-12)
#     total_cdf_loss += np.mean(np.abs(np.sort(val_lon) - empirical_cdf))

#     # 3. Depth: Volumetric (using our Normal + Cubic logic)
#     # We use the Radius-Cubed transformation to linearize the volume
#     r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
    
#     # We need the local boundaries for EVERY point to normalize correctly
#     r_bound_bot = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[0])]), axis=1)
#     r_bound_top = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[1])]), axis=1)
    
#     # Volume Fraction: (r^3 - r_min^3) / (r_max^3 - r_min^3)
#     val_depth = (r_actual**3 - r_bound_bot**3) / (r_bound_top**3 - r_bound_bot**3 + 1e-12)
#     total_cdf_loss += np.mean(np.abs(np.sort(val_depth) - empirical_cdf))

#     # 4. Time: Linear
#     val_time = (x_grid[:, 3] - (-time_range)) / (2 * time_range + 1e-12)
#     total_cdf_loss += np.mean(np.abs(np.sort(val_time) - empirical_cdf))

#     return total_cdf_loss / 4.0


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


            densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3][1], self.use_global)
            # Weighted penalty: we care most about Depth (Surface) and Time boundaries
            penalty_boundary = (abs(1.0 - densities['Depth']) * 1.0 +  ## Make these weights proportional to volume
                       abs(1.0 - densities['Time']) * 1.0 + 
                       abs(1.0 - densities['Lat']) * 1.0 + 
                       abs(1.0 - densities['Lon']) * 1.0)/4.0

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
        

        _, _, nominal_spacing_4d, _, _ = compute_expected_spacing(
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

            # cdf_loss = 0
            # N = len(x_grid)
            # empirical_cdf = np.arange(N) / (N - 1)
            # for i in range(4):
            #     if i == 0: # LATITUDE: WGS84 Authalic correction
            #         # Map latitudes to their area-proportional values
            #         q_vals = get_wgs84_area_val(x_grid[:, 0])
            #         q_min = get_wgs84_area_val(self.ranges[0][0])
            #         q_max = get_wgs84_area_val(self.ranges[0][1])
            #         val_to_sort = (q_vals - q_min) / (q_max - q_min + 1e-12)

            #     elif i == 1: # LONGITUDE: Linear
            #         val_to_sort = (x_grid[:, 1] - self.ranges[1][0]) / (self.ranges[1][1] - self.ranges[1][0] + 1e-12)

            #     elif i == 2: # DEPTH: WGS84 Volume Fraction (as we wrote before)
            #         r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), self.ranges[2][0])])), axis=1)
            #         r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), self.ranges[2][1])])), axis=1)
            #         r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
            #         val_to_sort = (r_actual**3 - r_min_local**3) / (r_max_local**3 - r_min_local**3 + 1e-12)

            #     elif i == 3: # TIME: Linear
            #         val_to_sort = (x_grid[:, 3] - self.ranges[3][0]) / (self.ranges[3][1] - self.ranges[3][0] + 1e-12)       

            #     sorted_vals = np.sort(val_to_sort)
            #     cdf_loss += np.mean(np.abs(sorted_vals - empirical_cdf))

            # scaling_vector = np.array([1.0, 1.0, depth_boost, scale_t])
            # x_proj_scaled = ftrns1_abs(x_grid * scaling_vector)        
            # # --- PART A: Uniformity (Scaled World) ---
            # tree = cKDTree(x_proj_scaled)
            # nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
            # cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)    

# class SamplingTuner:

#     def __init__(self, target_N, lat_range, lon_range, depth_range, time_range, use_global = use_global, device = device):

#         from skopt.space import Real
#         self.target_N = target_N
#         # Store ranges as [min, max] pairs
#         self.ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
#         self.device = device
#         self.time_range = time_range
#         self.use_global = use_global
        
#         # 1. Define Search Space
#         # scale_t: km/s
#         # depth_boost: dimensionless vertical stretch
#         # buffer_scale: multiplier for the nominal spacing
#         self.space = [
#             Real(1e3, 15e3, name='scale_t'),      
#             Real(1.0, 5.0, name='depth_boost'),   
#             Real(1.1, 2.5, name='buffer_scale')    # prior='log-uniform',
#         ]

#     def optimize(self, n_calls = 30):

#         """Runs Bayesian Optimization to find the triplet of parameters."""

#         @use_named_args(self.space)
#         def objective(scale_t, depth_boost, buffer_scale):

#             # 1. GENERATE CANDIDATES
#             up_sample_factor = 20 if use_time_shift else 10
#             number_candidate_nodes = up_sample_factor * self.target_N

#             trial_points, mask_points = regular_sobolov(
#                 number_candidate_nodes, 
#                 lat_range=self.ranges[0], 
#                 lon_range=self.ranges[1], 
#                 depth_range=self.ranges[2], 
#                 time_range=self.time_range, 
#                 use_time=use_time_shift, 
#                 use_global=self.use_global, 
#                 scale_time=scale_t, 
#                 N_target=self.target_N, 
#                 buffer_scale=buffer_scale
#             )        

#             # 2. RUN FPS (Physical -> Scaled Search Space)
#             # fps returns the selected physical [Lat, Lon, Depth, Time] points
#             x_grid = farthest_point_sampling(
#                 ftrns1_abs(trial_points), 
#                 self.target_N, 
#                 scale_time=scale_t, 
#                 depth_boost=depth_boost, 
#                 mask_candidates=mask_points
#             )        

#             # 3. PROJECT TO SCALED METRIC SPACE (For CV and Anisotropy)
#             # Use depth_boost on the 3rd column and scale_t on the 4th
#             scaling_vector = np.array([1.0, 1.0, depth_boost, scale_t])
#             x_proj_scaled = ftrns1_abs(x_grid * scaling_vector)        
#             # --- PART A: Uniformity (Scaled World) ---
#             tree = cKDTree(x_proj_scaled)
#             nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
#             cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)    


#             densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3][1], self.use_global)
#             # Weighted penalty: we care most about Depth (Surface) and Time boundaries
#             penalty_boundary = (abs(1.0 - densities['Depth']) * 1.0 +  ## Make these weights proportional to volume
#                        abs(1.0 - densities['Time']) * 1.0 + 
#                        abs(1.0 - densities['Lat']) * 1.0 + 
#                        abs(1.0 - densities['Lon']) * 1.0)/4.0

#             # penalty = (abs(1.0 - densities['Depth']) * 15.0 + 
#             #            abs(1.0 - densities['Time']) * 10.0 + 
#             #            abs(1.0 - densities['Lat']) * 5.0)

#             # 4. CDF ANALYSIS (The WGS84 "Gold Standard")
#             cdf_loss = 0
#             N = len(x_grid)
#             empirical_cdf = np.arange(N) / (N - 1)
#             for i in range(4):
#                 if i == 0: # LATITUDE: WGS84 Authalic correction
#                     # Map latitudes to their area-proportional values
#                     q_vals = get_wgs84_area_val(x_grid[:, 0])
#                     q_min = get_wgs84_area_val(self.ranges[0][0])
#                     q_max = get_wgs84_area_val(self.ranges[0][1])
#                     val_to_sort = (q_vals - q_min) / (q_max - q_min + 1e-12)

#                 elif i == 1: # LONGITUDE: Linear
#                     val_to_sort = (x_grid[:, 1] - self.ranges[1][0]) / (self.ranges[1][1] - self.ranges[1][0] + 1e-12)

#                 elif i == 2: # DEPTH: WGS84 Volume Fraction (as we wrote before)
#                     r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), self.ranges[2][0])])), axis=1)
#                     r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), self.ranges[2][1])])), axis=1)
#                     r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
#                     val_to_sort = (r_actual**3 - r_min_local**3) / (r_max_local**3 - r_min_local**3 + 1e-12)

#                 elif i == 3: # TIME: Linear
#                     val_to_sort = (x_grid[:, 3] - self.ranges[3][0]) / (self.ranges[3][1] - self.ranges[3][0] + 1e-12)       

#                 sorted_vals = np.sort(val_to_sort)
#                 cdf_loss += np.mean(np.abs(sorted_vals - empirical_cdf))

#             cdf_loss = cdf_loss/x_grid.shape[1] ## Normalize by number of dimensions used
#             # --- PART C: The Sanity Check (Anisotropy) ---
#             spreads = np.std(x_proj_scaled, axis=0)
#             anisotropy = np.max(spreads) / (np.min(spreads) + 1e-9)
#             penalty = np.maximum(0, np.log10(anisotropy) - 1.0)**2   

#             return cv + (3.0 * cdf_loss) + (1.0 * penalty) + (2.0 * penalty_boundary)

#         # res = gp_minimize(objective, self.space, n_calls = n_calls, n_initial_points = 5, initial_point_generator = 'sobol', verbose = True) # random_state = 42
#         res = gp_minimize(objective, self.space, n_calls = n_calls, verbose = True) # random_state = 42

#         best_scale_t, best_depth_boost, best_buffer_scale = res.x
        

#         _, _, nominal_spacing_4d, _, _ = compute_expected_spacing(
#         	self.target_N,  # total number of points
#         	lat_range=self.ranges[0],
#         	lon_range=self.ranges[1],
#         	depth_range=self.ranges[2],
#         	time_range=self.time_range,  # T, full range = 2T
#         	use_time=use_time_shift,
#         	scale_time=best_scale_t,  # w_scale: length per unit time
#         	use_global=use_global,
#         	earth_radius=earth_radius)

#         # nominal_spacing = (total_vol / self.target_N)**(1/4)
#         # buffer_width_phys = best_buffer_scale * nominal_spacing
#         buffer_width_phys = best_buffer_scale * nominal_spacing_4d # [0]

#         print("\n--- Optimization Results ---")
#         print(f"Optimal scale_t:     {best_scale_t:.3f} m/s")
#         print(f"Optimal depth_boost: {best_depth_boost:.3f}")
#         print(f"Optimal buffer_scale: {best_buffer_scale:.3f}")
#         print(f"Effective Padding:   {buffer_width_phys:.2f} (units)")

#         return {
#             'scale_t': best_scale_t,
#             'depth_boost': best_depth_boost,
#             'buffer_scale': best_buffer_scale,
#             'buffer_width_phys': buffer_width_phys
#         }

def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, buffer_scale, volume_space):
    
    N = len(x_grid)
    
    # --- 1. COORDINATE PROJECTIONS ---
    # scaling_4d = np.array([1.0, 1.0, depth_boost, scale_t])
    # x_metric_4d = ftrns1_abs(x_grid * scaling_4d)
    
    x_metric_4d = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units = True)

    # --- 2. GLOBAL 4D METRICS (The "O" Sliders) ---
    tree_4d = cKDTree(x_metric_4d)
    dist_4d, idx_nn = tree_4d.query(x_metric_4d, k=2)
    nn_4d = dist_4d[:, 1]
    nn_indices = idx_nn[:, 1] # Index of the 4D nearest neighbor
    
    ## Can replace these nearest neighbors with the warped metric
    cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)
    v_4d = volume_space * (2.0 * time_range * scale_t)
    expected_mean = 0.65 * (v_4d / N)**(1/4)
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
    print(f"Nodes (N): {N:<8} | scale_t: {scale_t:<8.1f} | d_boost: {depth_boost:<.2f} | buffer_scale: {buffer_scale:<.2f}")
    
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


    print(f"    Effective Velocity: {v_eff:.2f} km/s {'[PHYSICAL]' if 4<v_eff<10 else '[STRETCHED]'}")
    print(f"    Void Ratio (space):  {space_void_ratio:.4f}  [{get_bar(space_void_ratio, 2.0, 3.0)}] (Goal: <3.0)")
    print(f"    Void Ratio (time):  {time_void_ratio:.4f}  [{get_bar(time_void_ratio, 2.0, 3.0)}] (Goal: <3.0)")

    # --- [3] Boundary & Edge Health ---
    print(f"\n[3] Boundary & Edge Health (Bias Ratio)")
    def format_bias(name, val):
        status = "OK" if 0.7 < val < 1.3 else "BIASED"
        return f"    {name:12}: {val:.3f} [{get_bar(val, 0.5, 1.5)}] ({status})"
    print(format_bias("Temporal", bias_time))
    print(format_bias("Depth/Radial", bias_depth))
    print(format_bias("Lat", bias_lat))
    if not use_global:
        print(format_bias("Lon", bias_lon))


    print(f"\n[3] WGS84 Transparency (CDF R2 Scores)")
    for name, score in cdf_r2s.items():
        status = "PASS" if score > 0.98 else "WARN"
        print(f"    {name:6} R2: {score:.6f}  [{'#'*int(score*20):<20}] {status}")
    
    # Collisions
    collision_count = np.sum(nn_4d < nn_4d.mean()*0.75) ## Anything within half the average distance
    print(f"\n[4] Collision Check: {collision_count} nodes < half avg. distance apart.")

    print(f"{'='*65}\n")
    return {"cv_4d": cv_4d, "min_dist": np.min(dist_space), "collisions": collision_count, "cdf_r2s": cdf_r2s, "v_eff": v_eff}


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


# moi


use_tuning = True
if use_tuning == True:
	## Run the auto tuning strategy to refine some scale parameters
	m = SamplingTuner(number_of_spatial_nodes, lat_range_extend, lon_range_extend, depth_range, time_shift_range)
	params = m.optimize()
	scale_time, depth_upscale_factor, buffer_scale = params['scale_t'], params['depth_boost'], params['buffer_scale']

else:
	# depth_upscale_factor = 1.0
	buffer_scale = 2.0


# moi


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
		compute_final_grid_health(x_grid, scale_time, depth_upscale_factor, lat_range_extend, lon_range_extend, depth_range, time_shift_range, buffer_scale, Volume_space)

	# loss_metrics(x_grid, plot_on = True, grid_ind = n)
	# # nn_distance_stats(ftrns1_abs(x_grid)/1000.0, w_scale = scale_time/1000.0)
	# nn_distance_stats(ftrns1_abs(x_grid), scale_time, Volume_space, time_shift_range)
	x_grids.append(np.expand_dims(x_grid, axis = 0))


            # trial_points, mask_points = regular_sobolov(
            #     number_candidate_nodes, 
            #     lat_range=self.ranges[0], 
            #     lon_range=self.ranges[1], 
            #     depth_range=self.ranges[2], 
            #     time_range=self.time_range, 
            #     use_time=use_time_shift, 
            #     use_global=self.use_global, 
            #     scale_time=scale_t, 
            #     N_target=self.target_N, 
            #     buffer_scale=buffer_scale
            # )        

            # # 2. RUN FPS (Physical -> Scaled Search Space)
            # # fps returns the selected physical [Lat, Lon, Depth, Time] points
            # x_grid = farthest_point_sampling(
            #     ftrns1_abs(trial_points), 
            #     self.target_N, 
            #     scale_time=scale_t, 
            #     depth_boost=depth_boost, 
            #     mask_candidates=mask_points
            # )        


x_grids = np.vstack(x_grids)
np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_templates_ver_1.npz'%name_of_project, x_grids = x_grids, corr1 = np.zeros((1,3)), corr2 = np.zeros((1,3)))

print('Stable graphs will typically have:')
print('R2 expected depth >0.95')
print('R2 expected time >0.97')
print('CV NN full < 0.15')
print('Normalized Mean >1.5')
print('Correlation of Space-time nearest neighbors < 0.1')
print('Void ratio (spatial) < 2')
print('Void ratio (temporal) < 5')

print('Should add metrics computed near boundary')



# # [2]. Sorted depths
# n = len(x_grid)
# # u = np.arange(1, n+1) / n
# u = np.arange(n) / (n - 1)
# iarg = np.argsort(x_grid[:,2])
# d_sorted = np.sort(x_grid[:,2])
# # expected CDF
# r_surface = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1)
# # r = r_surface[iarg] + d_sorted
# r = r_surface.mean() + d_sorted
# F_expected = (r**3 - r_min**3) / (r_max**3 - r_min**3)
# r2_loss = r2_score(F_expected, u)
# print('R2 of expected depth distribution: %0.8f'%r2_loss)

# # diagnostic plot
# if plot_on == True:
# 	plt.figure()
# 	plt.plot(d_sorted, u, label="empirical")
# 	plt.plot(d_sorted, F_expected, "--", label="expected")
# 	plt.legend()
# 	fig = plt.gcf()
# 	fig.set_size_inches([8,8])
# 	plt.savefig(path_to_file + 'Plots' + seperator + 'grid_sorted_depths_ver_%d.png'%grid_ind, bbox_inches = 'tight', pad_inches = 0.1)





## Now build expander graphs
build_expander_graphs = True
if build_expander_graphs == True:

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
			A_edges_c = torch.Tensor(np.random.permutation(A_edges_c.max() + 1)).long()[A_edges_c]
			Ac = subgraph(torch.arange(number_of_spatial_nodes), A_edges_c)[0].cpu().detach().numpy() # .to(device)
			# A_edges_c = perm_vec[A_edges_c]

		else:


			def make_labeled_mgg_graph(m):
			    # m is the side of the grid (nodes = m*m)
			    edge_index = []
			    edge_type = []
			    
			    for x in range(m):
			        for y in range(m):
			            u = x * m + y
			            # Define the standard MGG transformations
			            targets = [
			                ((x + y) % m, y),           # T1
			                ((x - y) % m, y),           # T1 inverse
			                (x, (y + x) % m),           # T2
			                (x, (y - x) % m),           # T2 inverse
			                ((x + 2*y + 1) % m, y),     # Shifted T
			                ((x - 2*y - 1) % m, y)      # Shifted T inverse
			            ]
			            
			            for i, (tx, ty) in enumerate(targets):
			                v = tx * m + ty
			                edge_index.append([u, v])
			                edge_type.append(i) # Label 0 through 5
			                
			    return torch.tensor(edge_index).t(), torch.tensor(edge_type)

			A_edges_c, edge_type = make_labeled_mgg_graph(int_need)
			A_edges_c = torch.Tensor(np.random.permutation(A_edges_c.max() + 1)).long()[A_edges_c]
			Ac, edge_type = subgraph(torch.arange(number_of_spatial_nodes), A_edges_c, edge_attr = edge_type) # [0].cpu().detach().numpy() # .to(device)
			Ac, edge_type = Ac.cpu().detach().numpy(), edge_type.cpu().detach().numpy()

			# A_edges_c = from_networkx(A_edges_c)

	np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_expanders_ver_1.npz'%name_of_project, Ac = Ac)


print("All files saved successfully!")
print("✔ Script execution: Done")







# def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, volume_space):
#     """
#     Terminal-optimized health report for WGS84 4D grids.
#     """
#     N = len(x_grid)
#     full_ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
    
#     # 1. PREP COORDINATE SPACES
#     # Scaled Metric Space (for NN stats)
#     scaling_vec = np.array([1.0, 1.0, depth_boost, scale_t])
#     x_metric = ftrns1_abs(x_grid * scaling_vec)
    
#     # 2. NEAREST NEIGHBOR ANALYSIS (Metric Space)
#     tree = cKDTree(x_metric)
#     nn_dists = tree.query(x_metric, k=2)[0][:, 1]
    
#     cv_nn = np.std(nn_dists) / (np.mean(nn_dists) + 1e-9)
#     hypervolume_4d = volume_space * (2.0 * time_range * scale_t)
#     expected_random_mean = 0.65 * (hypervolume_4d / N)**(1/4)
#     norm_mean = np.mean(nn_dists) / expected_random_mean
#     void_ratio = np.quantile(nn_dists, 0.99) / (np.mean(nn_dists) + 1e-9)

#     # 3. CDF ANALYSIS (WGS84 Corrected)
#     cdf_r2s = {}
#     empirical_cdf = np.arange(N) / (N - 1)

#     for i, name in enumerate(["Lat", "Lon", "Depth", "Time"]):
#         if i == 0: # LATITUDE: WGS84 Authalic
#             q_vals = get_wgs84_area_val(x_grid[:, 0])
#             q_min = get_wgs84_area_val(lat_range[0])
#             q_max = get_wgs84_area_val(lat_range[1])
#             val_to_sort = (q_vals - q_min) / (q_max - q_min + 1e-12)
            
#         elif i == 1: # LON: Linear
#             val_to_sort = (x_grid[:, i] - lon_range[0]) / (lon_range[1] - lon_range[0] + 1e-12)
            
#         elif i == 2: # DEPTH: WGS84 Local Volume Fraction
#             r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[0])])), axis=1)
#             r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[1])])), axis=1)
#             r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
#             val_to_sort = (r_actual**3 - r_min_local**3) / (r_max_local**3 - r_min_local**3 + 1e-12)
            
#         else: # TIME: Linear
#             val_to_sort = (x_grid[:, i] - (-time_range)) / (2.0 * time_range + 1e-12)

#         sorted_vals = np.sort(val_to_sort)
#         cdf_r2s[name] = r2_score(empirical_cdf, sorted_vals)

#     # 4. TERMINAL DIAGNOSTIC PRINT
#     print(f"\n{'='*60}")
#     print(f"       GEOMETRIC GRID HEALTH REPORT (WGS84-4D)")
#     print(f"{'='*60}")
#     print(f"Nodes (N): {N:<10} | scale_t: {scale_t:<8.1f} | d_boost: {depth_boost:<.2f}")
    
#     print(f"\n--- Uniformity Metrics (Metric Space) ---")
#     def get_bar(val, ideal_min, ideal_max):
#         bar = ["-"] * 20
#         pos = int(min(max(val / (ideal_max * 1.5), 0), 1) * 19)
#         bar[pos] = "O"
#         return "".join(bar)

#     print(f"CV NN:           {cv_nn:.4f}  [{get_bar(cv_nn, 0.2, 0.3)}] (Goal: 0.2-0.3)")
#     print(f"Normalized Mean: {norm_mean:.4f}  [{get_bar(norm_mean, 1.4, 1.7)}] (Goal: >1.4)")
#     print(f"Void Ratio:      {void_ratio:.4f}  [{get_bar(void_ratio, 2.0, 3.0)}] (Goal: <3.0)")
    
#     print(f"\n--- Transparency Metrics (CDF R2 Scores) ---")
#     for name, score in cdf_r2s.items():
#         status = "PASS" if score > 0.999 else "WARN"
#         bar_len = int(score * 20)
#         bar = "#" * bar_len + " " * (20 - bar_len)
#         print(f"{name:6} R2: {score:.6f}  [{bar}] {status}")
    
#     print(f"{'='*60}\n")
    
#     return cdf_r2s


# import numpy as np
# from scipy.spatial import cKDTree
# from sklearn.metrics import r2_score

# def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, volume_space):
#     """
#     Complete Pre-Training Health Check for 4D Seismic Source Grids.
#     Accounts for WGS84 Ellipsoid and provides terminal diagnostics.
#     """
#     N = len(x_grid)
    
#     # --- 1. COORDINATE PROJECTIONS ---
#     # Metric 4D: For global uniformity
#     scaling_4d = np.array([1.0, 1.0, depth_boost, scale_t])
#     x_metric_4d = ftrns1_abs(x_grid * scaling_4d)
    
#     # Spatial 3D: For spatial voids/clustering
#     scaling_3d = np.array([1.0, 1.0, depth_boost])
#     x_metric_3d = ftrns1_abs(x_grid[:, :3] * scaling_3d)
    
#     # Physical 3D: For real-world collision checks (km)
#     x_phys_3d_km = ftrns1_abs(x_grid[:, :3]) / 1000.0

#     # --- 2. UNIFORMITY & COLLISION LOGIC ---
#     def get_nn_stats(pts, vol, dim):
#         dists = cKDTree(pts).query(pts, k=2)[0][:, 1]
#         cv = np.std(dists) / (np.mean(dists) + 1e-9)
#         k_dim = {1: 1.0, 3: 0.55, 4: 0.65}[dim] # Empirical Blue Noise constants
#         norm_mean = np.mean(dists) / (k_dim * (vol / N)**(1/dim))
#         void_ratio = np.quantile(dists, 0.99) / (np.mean(dists) + 1e-9)
#         return cv, norm_mean, void_ratio, dists

#     # Volumes for normalization
#     v_4d = volume_space * (2.0 * time_range * scale_t)
#     v_3d = volume_space
#     v_1d = 2.0 * time_range * scale_t

#     # Compute Stats
#     cv4, nm4, vr4, _ = get_nn_stats(x_metric_4d, v_4d, 4)
#     cv3, nm3, vr3, _ = get_nn_stats(x_metric_3d, v_3d, 3)
#     cv1, nm1, vr1, nn1d = get_nn_stats((x_grid[:, 3] * scale_t).reshape(-1, 1), v_1d, 1)
    
#     # Physical Collisions
#     nn_phys = cKDTree(x_phys_3d_km).query(x_phys_3d_km, k=2)[0][:, 1]
#     min_dist = np.min(nn_phys)
#     mean_dist = np.mean(nn_phys)
#     collision_count = np.sum(nn_phys < 0.5) # Points < 500m apart

#     # --- 3. WGS84 TRANSPARENCY (CDF R2) ---
#     cdf_r2s = {}
#     emp_cdf = np.arange(N) / (N - 1)
    
#     # Lat (Authalic)
#     q = get_wgs84_area_val(x_grid[:, 0])
#     q_min, q_max = get_wgs84_area_val(lat_range[0]), get_wgs84_area_val(lat_range[1])
#     cdf_r2s["Lat"] = r2_score(emp_cdf, np.sort((q - q_min)/(q_max - q_min + 1e-12)))
    
#     # Lon (Linear)
#     cdf_r2s["Lon"] = r2_score(emp_cdf, np.sort((x_grid[:, 1] - lon_range[0])/(lon_range[1] - lon_range[0])))
    
#     # Depth (WGS84 Vol Frac)
#     r_min_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[0])])), axis=1)
#     r_max_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[1])])), axis=1)
#     r_act = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
#     vol_f = (r_act**3 - r_min_l**3) / (r_max_l**3 - r_min_l**3 + 1e-12)
#     cdf_r2s["Depth"] = r2_score(emp_cdf, np.sort(vol_f))
    
#     # Time (Linear)
#     cdf_r2s["Time"] = r2_score(emp_cdf, np.sort((x_grid[:, 3] - (-time_range))/(2*time_range)))

#     # --- 4. THE TERMINAL REPORT ---
#     print(f"\n{'='*65}")
#     print(f"       PRE-TRAINING GRID CERTIFICATION (WGS84-4D)")
#     print(f"{'='*65}")
#     print(f"Nodes: {N} | scale_t: {scale_t:.2f} | d_boost: {depth_boost:.2f}")

#     print(f"\n[1] UNIFORMITY (CV NN - Goal: 0.20-0.35)")
#     print(f"    Full 4D:    {cv4:.4f} {'[PASS]' if 0.2<=cv4<=0.4 else '[WARN]'}")
#     print(f"    Spatial 3D: {cv3:.4f} {'[PASS]' if 0.2<=cv3<=0.4 else '[WARN]'}")
#     print(f"    Temporal 1D:{cv1:.4f} {'[PASS]' if 0.2<=cv1<=0.4 else '[WARN]'}")

#     print(f"\n[2] SPACING & COLLISIONS")
#     print(f"    Min / Mean Spacing: {min_dist:.3f} / {mean_dist:.3f} km")
#     print(f"    Collisions (<500m): {collision_count} {'[CLEAN]' if collision_count==0 else '[DIRTY]'}")
    
#     print(f"\n[3] WGS84 TRANSPARENCY (R2 Score - Goal: >0.999)")
#     for name, r2 in cdf_r2s.items():
#         print(f"    {name:6} R2: {r2:.6f} {'[OK]' if r2 > 0.999 else '[FAIL]'}")

#     # --- 5. BEST PARAMETER SUGGESTION LOGIC ---
#     total_score = (cv4 < 0.35) + (all(r > 0.99 for r in cdf_r2s.values())) + (collision_count == 0)
    
#     print(f"\n{'='*65}")
#     if total_score == 3:
#         print(">>> SUGGESTION: GRID IS OPTIMAL. Proceed to model training.")
#     elif collision_count > 0:
#         print(">>> SUGGESTION: REDUCE N or INCREASE buffer_scale to avoid collisions.")
#     elif cv4 > 0.4:
#         print(">>> SUGGESTION: INCREASE up_sample_factor to improve Blue Noise quality.")
#     else:
#         print(">>> SUGGESTION: Adjust scale_t / depth_boost; CDF or CV is lagging.")
#     print(f"{'='*65}\n")

#     return cdf_r2s


# def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, volume_space):
#     N = len(x_grid)
    
#     # --- 1. COORDINATE PROJECTIONS ---
#     scaling_4d = np.array([1.0, 1.0, depth_boost, scale_t])
#     x_metric_4d = ftrns1_abs(x_grid * scaling_4d)
#     x_phys_3d_km = ftrns1_abs(x_grid[:, :3]) / 1000.0

#     # --- 2. GLOBAL 4D METRICS (The "O" Sliders) ---
#     tree_4d = cKDTree(x_metric_4d)
#     nn_4d = tree_4d.query(x_metric_4d, k=2)[0][:, 1]
    
#     cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)
#     v_4d = volume_space * (2.0 * time_range * scale_t)
#     # Expected mean for 4D Blue Noise ~ 0.65
#     expected_mean = 0.65 * (v_4d / N)**(1/4)
#     norm_mean = np.mean(nn_4d) / expected_mean
#     void_ratio = np.quantile(nn_4d, 0.99) / (np.mean(nn_4d) + 1e-9)

#     # --- 3. SUB-GRID & COLLISION LOGIC ---
#     def get_cv(pts):
#         d = cKDTree(pts).query(pts, k=2)[0][:, 1]
#         return np.std(d) / (np.mean(d) + 1e-9)

#     cv_3d = get_cv(ftrns1_abs(x_grid[:, :3] * np.array([1.0, 1.0, depth_boost])))
#     cv_1d = get_cv((x_grid[:, 3] * scale_t).reshape(-1, 1))
    
#     nn_phys = cKDTree(x_phys_3d_km).query(x_phys_3d_km, k=2)[0][:, 1]
#     min_dist = np.min(nn_phys)
#     collision_count = np.sum(nn_phys < 0.5)

#     # --- 4. WGS84 TRANSPARENCY (CDF R2) ---
#     cdf_r2s = {}
#     emp_cdf = np.arange(N) / (N - 1)
    
#     # Lat (Authalic)
#     q = get_wgs84_area_val(x_grid[:, 0])
#     q_min, q_max = get_wgs84_area_val(lat_range[0]), get_wgs84_area_val(lat_range[1])
#     cdf_r2s["Lat"] = r2_score(emp_cdf, np.sort((q - q_min)/(q_max - q_min + 1e-12)))
    
#     # Lon (Linear)
#     cdf_r2s["Lon"] = r2_score(emp_cdf, np.sort((x_grid[:, 1] - lon_range[0])/(lon_range[1] - lon_range[0])))
    
#     # Depth (WGS84 Vol Frac)
#     r_min_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[0])])), axis=1)
#     r_max_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[1])])), axis=1)
#     r_act = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
#     vol_f = (r_act**3 - r_min_l**3) / (r_max_l**3 - r_min_l**3 + 1e-12)
#     cdf_r2s["Depth"] = r2_score(emp_cdf, np.sort(vol_f))
    
#     # Time (Linear)
#     cdf_r2s["Time"] = r2_score(emp_cdf, np.sort((x_grid[:, 3] - (-time_range))/(2*time_range)))

#     # --- 5. TERMINAL OUTPUT ---
#     print(f"\n{'='*60}")
#     print(f"       GEOMETRIC GRID HEALTH REPORT (WGS84-4D)")
#     print(f"{'='*60}")
#     print(f"Nodes (N): {N:<8} | scale_t: {scale_t:<8.1f} | d_boost: {depth_boost:<.2f}")
    
#     def get_bar(val, ideal_min, ideal_max):
#         bar = ["-"] * 20
#         pos = int(min(max(val / (ideal_max * 1.5), 0), 1) * 19)
#         bar[pos] = "O"
#         return "".join(bar)

#     print(f"\n--- Uniformity Metrics (Metric Space) ---")
#     print(f"CV NN:           {cv_4d:.4f}  [{get_bar(cv_4d, 0.2, 0.3)}] (Goal: 0.2-0.3)")
#     print(f"Normalized Mean: {norm_mean:.4f}  [{get_bar(norm_mean, 1.4, 1.7)}] (Goal: >1.4)")
#     print(f"Void Ratio:      {void_ratio:.4f}  [{get_bar(void_ratio, 2.0, 3.0)}] (Goal: <3.0)")

#     print(f"\n--- Sub-Grid Uniformity (CV) ---")
#     print(f"Spatial 3D CV: {cv_3d:.4f} | Temporal 1D CV: {cv_1d:.4f}")
    
#     print(f"\n--- Spacing & Collisions ---")
#     print(f"Min Spacing: {min_dist:.3f} km | Collisions (<0.5km): {collision_count}")

#     print(f"\n--- Transparency Metrics (CDF R2 Scores) ---")
#     for name, score in cdf_r2s.items():
#         status = "PASS" if score > 0.999 else "WARN"
#         print(f"{name:6} R2: {score:.6f}  [{'#'*int(score*20):<20}] {status}")
    
#     # Recommendation Logic
#     print(f"\n{'='*60}")
#     if all(r > 0.999 for r in cdf_r2s.values()) and 0.2 < cv_4d < 0.4:
#         print(">>> SUGGESTION: GRID IS OPTIMAL.")
#     else:
#         print(">>> SUGGESTION: ADJUST PARAMETERS (Check WARN fields).")
#     print(f"{'='*60}\n")

#     return {"cv_4d": cv_4d, "min_dist": min_dist, "collisions": collision_count, "cdf_r2s": cdf_r2s}


# import numpy as np
# import json
# import datetime
# from scipy.spatial import cKDTree
# from sklearn.metrics import r2_score

# def get_wgs84_area_val(lat_deg):
#     """Computes the WGS84 authalic (equal-area) value for a given latitude."""
#     lat_rad = np.deg2rad(lat_deg)
#     e2 = 0.00669437999014  # WGS84 eccentricity squared
#     e = np.sqrt(e2)
#     sin_phi = np.sin(lat_rad)
#     term1 = (1 - e2) * sin_phi / (1 - e2 * sin_phi**2)
#     term2 = ((1 - e2) / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
#     return term1 - term2




# trial_points = collect_trial_points(number_candidate_nodes)
# x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])

# if use_poisson_filtering == True:

# 	up_sample_factor = 10 if use_time_shift == False else 20
# 	number_candidate_nodes = up_sample_factor*number_of_spatial_nodes

# 	print('Beginning  Poisson filtering [%d]'%n)
# 	p = [1.0, False, False] ## Optimize this choice (on the first grid built)
# 	trial_points = regular_sobolov(number_candidate_nodes)
# 	x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])

# elif use_farthest_point_filtering == True:

# else:

# 	print('Using standard Sobolov sampling [%d]'%n)
# 	x_grid = regular_sobolov(number_of_spatial_nodes)


# else:
# 	p = [1.0, False, False] ## Optimize this choice (on the first grid built)
# 	trial_points = regular_sobolov(number_candidate_nodes)
# 	x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])




# # --- PART B: Boundary Transparency (Measured in Physical World) ---
# # We check the CDF here to ensure there's no 'curling' at the 0km or 40km edges.
# cdf_loss = 0
# for i in range(4):
#     sorted_axis = np.sort(x_grid[:, i])
#     theoretical = np.linspace(self.ranges[i][0], self.ranges[i][1], len(x_grid))
#     span = self.ranges[i][1] - self.ranges[i][0]
#     cdf_loss += np.mean(np.abs(sorted_axis - theoretical)) / (span + 1e-9)

# def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, earth_radius=earth_radius):

#     N = len(x_grid)
#     ranges = [lat_range, lon_range, depth_range]
    
#     # 1. PREP COORDINATE SPACES
#     scaling_vec = np.array([1.0, 1.0, depth_boost, scale_t])
#     x_metric = ftrns1_abs(x_grid * scaling_vec)
    
#     # 2. NEAREST NEIGHBOR ANALYSIS (Metric Space)
#     tree = cKDTree(x_metric)
#     nn_dists = tree.query(x_metric, k=2)[0][:, 1]
#     cv_nn = np.std(nn_dists) / np.mean(nn_dists)
    
#     hypervolume_4d = volume_space * (2.0 * time_range * scale_t)
#     expected_random_mean = 0.65 * (hypervolume_4d / N)**(1/4)
#     norm_mean = np.mean(nn_dists) / expected_random_mean
#     void_ratio = np.quantile(nn_dists, 0.99) / np.mean(nn_dists)

#     # 3. CDF ANALYSIS (Physical Space)
#     cdf_r2s = {}
#     empirical_cdf = np.arange(N) / (N - 1)

#     # --- Linear Dimensions (Lat, Lon, Time) ---
#     for i, name in [(0, "Lat"), (1, "Lon"), (3, "Time")]:
#         sorted_data = np.sort(x_grid[:, i])
#         theoretical_cdf = (sorted_data - ranges[i][0]) / (ranges[i][1] - ranges[i][0])
#         cdf_r2s[name] = r2_score(empirical_cdf, theoretical_cdf)

#     # --- Spherical Dimension (Depth/Radius) ---
#     # r_max = surface, r_min = bottom of crust
#     r_max = earth_radius - ranges[2][0] 
#     r_min = earth_radius - ranges[2][1]
    
#     # Calculate physical radii of the nodes
#     # We use the raw depths (x_grid[:, 2]) to find r
#     d_sorted = np.sort(x_grid[:, 2])
#     # Note: radius decreases as depth increases: r = earth_radius - depth
#     r_sorted = earth_radius - d_sorted 
    
#     # The expected CDF for uniform volume sampling in a sphere:
#     # F(r) = (r^3 - r_min^3) / (r_max^3 - r_min^3)
#     # We sort r ascending (from r_min to r_max) to match empirical_cdf
#     r_sorted_asc = np.sort(r_sorted) 
#     theoretical_cdf_r = (r_sorted_asc**3 - r_min**3) / (r_max**3 - r_min**3)
    
#     cdf_r2s["Depth"] = r2_score(empirical_cdf, theoretical_cdf_r)

#     # 4. PRINT SUMMARY
#     print(f"\n{'='*40}\n FINAL GRID HEALTH REPORT (SPHERICAL) \n{'='*40}")
#     print(f"Nodes (N): {N} | scale_t: {scale_t:.1f} | d_boost: {depth_boost:.2f}")
#     print(f"\n--- Uniformity (Metric Space) ---")
#     print(f"CV NN:           {cv_nn:.4f}  (Ideal: 0.2-0.3)")
#     print(f"Normalized Mean: {norm_mean:.4f}  (Ideal: >1.4)")
#     print(f"Void Ratio:      {void_ratio:.4f}  (Ideal: <3.0)")
    
#     print(f"\n--- Transparency (CDF R2 Scores) ---")
#     for name, score in cdf_r2s.items():
#         status = "PASS" if score > 0.99 else "WARN"
#         print(f"{name:6} R2: {score:.6f}  [{status}]")
    
#     return cdf_r2s



# import numpy as np
# from scipy.spatial import cKDTree
# from sklearn.metrics import r2_score


# @use_named_args(self.space) # self.target_N, self.space, self.ranges, self.time_range, self.use_global
# def objective(scale_t, depth_boost, buffer_scale, return_metrics = False):
#     """ This replaces the simple calculate_loss function """
#     # 1. GENERATE THE NODES
#     # Your pipeline should return the real coordinates AND the scaled ones
#     # nodes_phys: [Lat, Lon, Depth, Time] 
#     # nodes_scaled: [Lat, Lon, Depth * depth_boost, Time * scale_t]

#     # nodes_phys, nodes_scaled = self.run_pipeline(scale_t, depth_boost, buffer_scale)
#     up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
#     number_candidate_nodes = up_sample_factor*self.target_N
#     ## Increase efficiency of this script
#     trial_points, mask_points = regular_sobolov(number_candidate_nodes, lat_range = self.ranges[0], lon_range = self.ranges[1], depth_range = self.ranges[2], time_range = self.time_range, use_time = use_time_shift, use_global = self.use_global, scale_time = scale_t, N_target = self.target_N, buffer_scale = buffer_scale) # lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0
#     x_grid = farthest_point_sampling(ftrns1_abs(trial_points), self.target_N, scale_time = scale_t, depth_boost = depth_boost, mask_candidates = mask_points)
#     x_proj_scaled = ftrns1_abs(x_grid*np.array([1.0, 1.0, depth_boost, scale_t]).reshape(1,-1))

#     # --- PART A: Uniformity (Measured in Scaled World) ---
#     # If we measured this in 'Physical' world, the 40km depth would 
#     # make the CV look weird regardless of how good the sampling is.
#     tree = cKDTree(x_proj_scaled)
#     nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
#     cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)

#     # 3. CDF ANALYSIS (The WGS84 "Gold Standard")
#     cdf_loss = 0
#     N = len(x_grid)
#     empirical_cdf = np.arange(N) / (N - 1)
    
#     for i in range(4):
#         if i == 0: # LATITUDE: WGS84 Authalic correction
#             # Map latitudes to their area-proportional values
#             q_vals = get_wgs84_area_val(x_grid[:, 0])
#             q_min = get_wgs84_area_val(self.ranges[0][0])
#             q_max = get_wgs84_area_val(self.ranges[0][1])
#             val_to_sort = (q_vals - q_min) / (q_max - q_min + 1e-12)
            
#         elif i == 1: # LONGITUDE: Linear
#             val_to_sort = (x_grid[:, 1] - self.ranges[1][0]) / (self.ranges[1][1] - self.ranges[1][0] + 1e-12)
            
#         elif i == 2: # DEPTH: WGS84 Volume Fraction (as we wrote before)
#             r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), self.ranges[2][0])])), axis=1)
#             r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), self.ranges[2][1])])), axis=1)
#             r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
#             val_to_sort = (r_actual**3 - r_min_local**3) / (r_max_local**3 - r_min_local**3 + 1e-12)
            
#         elif i == 3: # TIME: Linear
#             val_to_sort = (x_grid[:, 3] - self.ranges[3][0]) / (self.ranges[3][1] - self.ranges[3][0] + 1e-12)        

#         sorted_vals = np.sort(val_to_sort)
#         cdf_loss += np.mean(np.abs(sorted_vals - empirical_cdf))


#     # --- PART C: The Sanity Check (Measured in Scaled World) ---
#     # This prevents the optimizer from making one axis 10,000x longer than the others.
#     spreads = np.std(x_proj_scaled, axis=0)
#     anisotropy = np.max(spreads) / (np.min(spreads) + 1e-9)
#     penalty = np.maximum(0, np.log10(anisotropy) - 1.0)**2
#     # Return the combined loss (Lower is Better)

#     if return_metrics == False:
#     	return cv + (3.0 * cdf_loss) + (1.0 * penalty)
#     else:
#     	return cv + (3.0 * cdf_loss) + (1.0 * penalty), [cv, cdf_loss, penalty]


# # --- PART B: Boundary Transparency (WGS84 Volume Aware) ---
# cdf_loss = 0
# empirical_cdf = np.arange(len(x_grid)) / (len(x_grid) - 1)

# for i in range(4):
#     if i == 2: # Depth/Radius dimension (WGS84 Spherical Shell)
#         # Compute local radius bounds for each point's Lat/Lon
#         r_min_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((len(x_grid),1), self.ranges[2][0])])), axis=1)
#         r_max_local = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((len(x_grid),1), self.ranges[2][1])])), axis=1)
#         r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
        
#         # Volume fraction normalized locally (0 to 1)
#         vol_frac = (r_actual**3 - r_min_local**3) / (r_max_local**3 - r_min_local**3 + 1e-9)
#         sorted_vals = np.sort(vol_frac)
#         theoretical = empirical_cdf # In perfect volume sampling, frac is linear
#     else:
#         # Lat, Lon, and Time use standard linear spacing
#         sorted_vals = (np.sort(x_grid[:, i]) - self.ranges[i][0]) / (self.ranges[i][1] - self.ranges[i][0] + 1e-9)
#         theoretical = empirical_cdf        

#     cdf_loss += np.mean(np.abs(sorted_vals - theoretical))    



# def calculate_loss(self, nodes):
#     """Standardized loss: Uniformity + CDF Linearity + Anisotropy Guardrail."""
#     # --- 1. Uniformity (Blue Noise Quality) ---
#     tree = cKDTree(nodes)
#     dists, _ = tree.query(nodes, k=2)
#     nn_dist = dists[:, 1]
#     cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)

#     # --- 2. Boundary Flatness (CDF Deviation) ---
#     # This is the primary check for edge-curling
#     cdf_loss = 0
#     for i in range(4):
#         sorted_axis = np.sort(nodes[:, i])
#         theoretical = np.linspace(self.ranges[i][0], self.ranges[i][1], len(nodes))
#         span = self.ranges[i][1] - self.ranges[i][0]
#         cdf_loss += np.mean(np.abs(sorted_axis - theoretical)) / (span + 1e-9)

#     # --- 3. Anisotropy Penalty (The Sanity Check) ---
#     spreads = np.std(nodes, axis=0)
#     anisotropy = np.max(spreads) / (np.min(spreads) + 1e-9)
#     # Penalize log-linearly if aspect ratio exceeds 10:1
#     penalty = np.maximum(0, np.log10(anisotropy) - 1.0)**2

#     # Weighted Sum: CDF loss is weighted highest to ensure 'transparency'
#     return cv + (3.0 * cdf_loss) + (1.0 * penalty)




## User: Input stations and spatial region
## (must have station and region files at
## (ext_dir + 'stations.npz'), and
## (ext_dir + 'region.npz')



# Calculate the resulting physical buffer width for the user
# This uses the nominal spacing formula for 4D
# total_vol = (self.ranges[0][1]-self.ranges[0][0]) * \
#             (self.ranges[1][1]-self.ranges[1][0]) * \
#             (self.ranges[2][1]-self.ranges[2][0]) * \
#             (self.ranges[3][1]-self.ranges[3][0])

# nominal_spacing = (total_vol / self.target_N)**(1/4)
# buffer_width_phys = best_buffer_scale * nominal_spacing





# def u_to_geodetic_lat(u, lat_min, lat_max):
#     # Mapping u [0,1] to sin(lat) is spherical equal-area.
#     # For ellipsoidal (Authalic), the formula involves the eccentricity squared (e^2).
#     # e2 = 0.00669437999014 for WGS84.
#     u_min = (1.0 + np.sin(np.deg2rad(lat_min))) / 2.0
#     u_max = (1.0 + np.sin(np.deg2rad(lat_max))) / 2.0
#     u_val = u_min + u * (u_max - u_min)
#     return np.arcsin(2 * u_val - 1) * (180.0 / np.pi)




# @use_named_args(self.space)
# def objective(scale_t, depth_boost, buffer_scale):
#     # This calls your external pipeline (Sobol -> Ghost Density -> FPS)
#     # nodes = run_full_pipeline(
#     #     N=self.target_N, 
#     #     scale_t=scale_t, 
#     #     depth_boost=depth_boost, 
#     #     buffer_scale=buffer_scale
#     # )

#     up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
#     number_candidate_nodes = up_sample_factor*number_of_spatial_nodes
#     print('Beginning FPS sampling [%d]'%n)
#     ## Increase efficiency of this script
#     trial_points, mask_points = regular_sobolov(lat_range = self.ranges[0], lon_range = self.ranges[1], depth_range = self.ranges[2], time_range = self.time_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_t, N_target = number_of_spatial_nodes, buffer_scale = buffer_scale) # lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0
#     x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes, scale_time = scale_t, depth_boost = depth_boost, mask_candidates = mask_points)
#     # x_grid_scaled

#     return self.calculate_loss(nodes)






# ## Sampling options
# perm_option1 = [1.0, True, True]
# perm_option2 = [1.5, True, True]
# perm_option3 = [2.0, True, True]
# perm_option4 = [1.0, False, True]
# perm_option5 = [1.0, False, False]
# perm_options = [perm_option1, perm_option2, perm_option3, perm_option4, perm_option5]
# use_relaxation = True

# ## Now implement Poisson disk filtering to obtain the target number of nodes
# trial_points = collect_trial_points(number_candidate_nodes)
# R2_losses = [] ## Base decision on depth R2 loss (or use the RMS loss)

# for p in perm_options: ## Note the increased tol_fraction for this search
# 	# x_grid, _ = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, tol_fraction = 0.01, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])
# 	x_grid = regular_sobolov(number_of_spatial_nodes)



## First do tuning

# def evaluate_sampling_quality(nodes, lat_range, lon_range, depth_range, time_range):
#     """
#     nodes: [N, 4] array (Lat, Lon, Depth, Time)
#     """
#     N = nodes.shape[0]
    
#     # 1. Uniformity Check: Coefficient of Variation (CV)
#     # A perfectly uniform (Blue Noise) set has a very low CV of NN distances.
#     tree = KDTree(nodes)
#     dist, _ = tree.query(nodes, k=2) # k=2 because k=1 is the point itself
#     nn_distances = dist[:, 1]
    
#     cv = np.std(nn_distances) / np.mean(nn_distances)
    
#     # 2. Boundary "Curling" Check
#     # We check if the density in the outer 10% of the volume matches the inner 90%
#     def get_edge_mask(pts):
#         masks = []
#         ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
#         for i, r in enumerate(ranges):
#             margin = (r[1] - r[0]) * 0.1
#             masks.append((pts[:, i] < r[0] + margin) | (pts[:, i] > r[1] - margin))
#         return np.any(masks, axis=0)

#     edge_mask = get_edge_mask(nodes)
#     edge_density = np.sum(edge_mask) / N
#     # Expected edge density for a 4D hypercube's outer 10% shell is roughly:
#     # 1 - (0.8)^4 = 0.59 (59% of points should be in the 'edge' zones)
    
#     # 3. Anisotropy Check (Aspect Ratio)
#     # We look at the standard deviation of distances along each axis
#     axis_scales = np.std(nodes, axis=0)
#     aspect_ratios = axis_scales / axis_scales[0] # Relative to Latitude
    
#     print("--- Sampling Diagnostics ---")
#     print(f"Nearest Neighbor CV: {cv:.4f} (Lower is better, target < 0.15)")
#     print(f"Edge Density: {edge_density:.4f} (Target vs. 0.59)")
#     print(f"Axis Aspect Ratios (Lat/Lon/Dep/Time): {aspect_ratios}")
    
#     return cv, edge_density, aspect_ratios









# import torch

# def loss_metrics(x_grid, grid_ind = 0, depth_scale = depth_upscale_factor, use_time_shift = use_time_shift, plot_on = False):

# 	## Compute quality checks:
# 	# [1]. Flat in depth
# 	n_bins = 30
# 	r = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)), axis = 1)
# 	h_vals = np.histogram(r**3, bins = int(len(x_grid)/n_bins)) # [0]
# 	mean_loss = np.mean(n_bins*np.ones(len(h_vals[0])) - h_vals[0])
# 	rms_loss = (np.sqrt(((n_bins*np.ones(len(h_vals[0])) - h_vals[0])**2).sum()/len(h_vals[0]))/n_bins)
# 	print('\nMean deviation of radius flatness: %0.8f'%mean_loss)
# 	print('RMS deviation of radius flatness: %0.8f'%rms_loss)

# 	# [2]. Sorted depths
# 	n = len(x_grid)
# 	# u = np.arange(1, n+1) / n
# 	u = np.arange(n) / (n - 1)
# 	iarg = np.argsort(x_grid[:,2])
# 	d_sorted = np.sort(x_grid[:,2])
# 	# expected CDF
# 	r_surface = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1)
# 	# r = r_surface[iarg] + d_sorted
# 	r = r_surface.mean() + d_sorted
# 	F_expected = (r**3 - r_min**3) / (r_max**3 - r_min**3)
# 	r2_loss = r2_score(F_expected, u)
# 	print('R2 of expected depth distribution: %0.8f'%r2_loss)

# 	# diagnostic plot
# 	if plot_on == True:
# 		plt.figure()
# 		plt.plot(d_sorted, u, label="empirical")
# 		plt.plot(d_sorted, F_expected, "--", label="expected")
# 		plt.legend()
# 		fig = plt.gcf()
# 		fig.set_size_inches([8,8])
# 		plt.savefig(path_to_file + 'Plots' + seperator + 'grid_sorted_depths_ver_%d.png'%grid_ind, bbox_inches = 'tight', pad_inches = 0.1)

# 	# [3]. Nearest neighbor distances, and nearest neighbors as function of depth
# 	if use_time_shift == False:
# 		tree = cKDTree(ftrns1(x_grid*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)))
# 		q = tree.query(ftrns1(x_grid*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)), k = 2)[0][:,1]
# 		min_dist = q.min()/1000.0
# 		mean_dist = q.mean()/1000.0
# 		std_dist = q.std()/1000.0
# 		print('Nearest neighbors: Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(min_dist, mean_dist, std_dist))

# 	else:
# 		tree = cKDTree(ftrns1(x_grid*np.array([1.0, 1.0, depth_scale, scale_time]).reshape(1,-1)))
# 		q = tree.query(ftrns1(x_grid*np.array([1.0, 1.0, depth_scale, scale_time]).reshape(1,-1)), k = 2) # [0][:,1]
# 		q = np.linalg.norm(ftrns1(x_grid[:,0:3]) - ftrns1(x_grid[q[1][:,1],0:3]), axis = 1)		
# 		min_dist = q.min()/1000.0
# 		mean_dist = q.mean()/1000.0
# 		std_dist = q.std()/1000.0
# 		print('Nearest neighbors: Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(min_dist, mean_dist, std_dist))


# 	nn_cv = nn_distance_cv(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)), scale_time)
# 	knn_cv = knn_volume_cv(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)), k=8, scale_time=scale_time)
# 	print(f"Spatial: NN-CV={nn_cv:.3f}, kNN-CV={knn_cv:.3f}")

# 	if x_grid.shape[1] == 4:
# 		nn_cv = nn_distance_cv(ftrns1_abs(x_grid*np.array([1.0, 1.0, depth_scale, 1.0]).reshape(1,-1)), scale_time) ## Note, time scaling happens inside these functions
# 		knn_cv = knn_volume_cv(ftrns1_abs(x_grid*np.array([1.0, 1.0, depth_scale, 1.0]).reshape(1,-1)), k=8, scale_time=scale_time)
# 		print(f"Full: NN-CV={nn_cv:.3f}, kNN-CV={knn_cv:.3f}")

# 	if use_time_shift == True:
# 		# [4]. R2 of expected time distribution
# 		n = len(x_grid)
# 		# u = np.arange(1, n+1) / n
# 		u = np.arange(n) / (n - 1)
# 		iarg = np.argsort(x_grid[:,3])
# 		t_sorted = np.sort(x_grid[:,3])
# 		F_expected = (t_sorted - (-time_shift_range)) / (time_shift_range - (-time_shift_range))
# 		r2_loss_time = r2_score(F_expected, u)
# 		print('R2 of expected time distribution: %0.8f'%r2_loss_time)

# 		# [5]. Check nearest neighbors
# 		tree = cKDTree(ftrns1_abs(x_grid*np.array([1.0, 1.0, depth_scale, scale_time])).reshape(1,-1))
# 		q = tree.query(ftrns1_abs(x_grid*np.array([1.0, 1.0, depth_scale, scale_time])).reshape(1,-1), k = 2) # [0][:,1]
# 		q_min = q[0][:,1].min()/1000.0
# 		q_mean = q[0][:,1].mean()/1000.0
# 		q_std = q[0][:,1].std()/1000.0
# 		print('Nearest neighbors (scaled): Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(q_min, q_mean, q_std))


# 		# [6]. Space offsets
# 		dist_space = np.linalg.norm(ftrns1(x_grid[:,0:3]*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)) - ftrns1(x_grid[q[1][:,1],0:3]*np.array([1.0, 1.0, depth_scale]).reshape(1,-1)), axis = 1)
# 		dist_time = np.abs(x_grid[:,3] - x_grid[q[1][:,1],3])
# 		# min_spc_dist = dist_space.min()/1000.0
# 		# mean_spc_dist = dist_space.mean()/1000.0
# 		# std_spc_dist = dist_space.std()/1000.0
# 		# print('Nearest neighbors: Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(min_spc_dist, mean_spc_dist, std_spc_dist))

# 		# [6]. Correlation of space and time nearest neighbors
# 		pearsonr_val = pearsonr(dist_space, dist_time).statistic
# 		print('Correlation of space and time nearest neighbors: %0.8f'%pearsonr_val)

# 		# [7]. Ratio of small dt
# 		ratio_within_time_radius = len(np.where(dist_time*scale_time < 0.005*nominal_spacing)[0])/len(dist_time)
# 		print('Ratio of small time offset nearest neighbors: %0.8f \n'%ratio_within_time_radius)


# 	return mean_loss, rms_loss, r2_loss, mean_dist, std_dist



# import numpy as np
# from scipy.spatial import cKDTree

# def nn_distance_stats(
#     xyz_t, 
#     scale_time, 
#     Volume_space,  # from compute_expected_spacing
#     time_range,    # T, full span = 2 * time_range
#     cluster_threshold=0.5,  # tunable: nn_dist < threshold * mean_nn = cluster
# ):
#     points_scaled = xyz_t.copy()
#     points_scaled[:, 3] *= scale_time

#     tree = cKDTree(points_scaled)
#     dists, _ = tree.query(points_scaled, k=2)
#     nn_dists = dists[:, 1]
    
#     mean_nn = np.mean(nn_dists)
#     median_nn = np.median(nn_dists)
#     min_nn = np.min(nn_dists)
#     max_nn = np.quantile(nn_dists, 0.99)
#     std_nn = np.std(nn_dists)
#     cv_nn = std_nn / mean_nn if mean_nn > 0 else 0.0  # Coefficient of Variation
    
#     hypervolume = Volume_space * (scale_time * 2 * time_range)
#     expected_random_mean = 0.65 * (hypervolume / len(xyz_t)) ** (1/4)
#     normalized_mean = mean_nn / expected_random_mean
    
#     # Void ratio: max gap relative to mean spacing
#     void_ratio = max_nn / mean_nn  # >2-3 = notable voids
    
#     # Cluster ratio: fraction of small NN distances
#     cluster_count = np.sum(nn_dists < (cluster_threshold * mean_nn))
#     cluster_ratio = cluster_count / len(nn_dists)  # 0 = no clusters, >0.1 = some crowding
    
#     # Spatial-only (3D projection) stats for space voids/clusters
#     tree_space = cKDTree(xyz_t[:, :3])
#     dists_space, _ = tree_space.query(xyz_t[:, :3], k=2)
#     nn_dists_space = dists_space[:, 1]
#     mean_nn_space = np.mean(nn_dists_space)
#     void_ratio_space = np.quantile(nn_dists_space, 0.99) / mean_nn_space
#     cluster_ratio_space = np.sum(nn_dists_space < (cluster_threshold * mean_nn_space)) / len(nn_dists_space)
    
#     # Time-only (1D projection) stats for time voids/clusters
#     t_reshaped = xyz_t[:, 3].reshape(-1, 1)  # 1D for KDTree
#     tree_time = cKDTree(t_reshaped)
#     dists_time, _ = tree_time.query(t_reshaped, k=2)
#     nn_dists_time = dists_time[:, 1]
#     mean_nn_time = np.mean(nn_dists_time)
#     void_ratio_time = np.quantile(nn_dists_time, 0.99) / mean_nn_time
#     cluster_ratio_time = np.sum(nn_dists_time < (cluster_threshold * mean_nn_time)) / len(nn_dists_time)
    
#     stats = {
#         'mean_nn_dist': mean_nn,
#         'median_nn_dist': median_nn,
#         'min_nn_dist': min_nn,
#         'max_nn_dist': max_nn,
#         'cv_nn': cv_nn,  # new: 0.2-0.3 = excellent, >0.5 = random/clustered
#         'normalized_mean': normalized_mean[0],
#         'void_ratio': void_ratio,  # new: 2-3 = balanced, >4 = voids present
#         'cluster_ratio': cluster_ratio,  # new: <0.05 = minimal clustering
#         'void_ratio_space': void_ratio_space,
#         'cluster_ratio_space': cluster_ratio_space,
#         'void_ratio_time': void_ratio_time,
#         'cluster_ratio_time': cluster_ratio_time,
#         'nn_distances': nn_dists
#     }

#     # print(stats)
#     # Usage
#     # stats = nn_distance_stats(xyz_t, w_scale=w_scale)
#     print(f"Min NN distance: {stats['min_nn_dist']:.1f} m (spatial+time)")
#     print(f"Mean NN distance: {stats['mean_nn_dist']:.1f} m (spatial+time)")
#     print(f"Median NN distance: {stats['median_nn_dist']:.1f} m (spatial+time)")
#     print(f"\nNormalized mean: {stats['normalized_mean']:.3f} (spatial+time)")
#     print(f"Interpretation: 1.0 = random, 1.3–1.6 = good, >1.7 = excellent low-discrepancy\n")
#     print(f"CV NN: {stats['cv_nn']:.2f} m (spatial+time) [=0.2-0.3 = excellent, >0.5 = random/clustered]")
#     print(f"Void ratio: {stats['void_ratio']:.2f} (spatial+time) [2-3 = balanced, >4 = voids present]")
#     print(f"Cluster ratio: {stats['cluster_ratio']:.2f} (spatial+time) [<0.05 = minimal clustering]")
#     print(f"\nVoid ratio: {stats['void_ratio_space']:.2f} (spatial)")
#     print(f"Cluster ratio: {stats['cluster_ratio_space']:.2f} (spatial)")
#     print(f"\nVoid ratio: {stats['void_ratio_time']:.2f} (time)")
#     print(f"Cluster ratio: {stats['cluster_ratio_time']:.2f} (time)")


#     return stats

# Tuning T: Yes, if T is arbitrary, use the process inversely: Fix scale_time to physics, generate points, check time-specific metrics (void_ratio_time, cluster_ratio_time). If void_ratio_time >3 (large gaps), shrink T; if cluster_ratio_time >0.1 (crowding), expand T. Or set T such that nominal_spacing_time ≈ seismic resolution (e.g., 1/10 of interference length).
# Tuning N: Use kernel needs: Required N ≈ (Volume_space / (kernel_radius / 3)^3) * (2T / kernel_time)^1 for ~3-5 nodes per kernel in space-time. If metrics show over-clustering (high cluster_ratio), increase N.

# CV: <0.3 = highly regular; 0.3-0.4 = good; >0.5 = check for clusters.
# Void ratio: ~2 = tight packing; >3-4 = potential voids (investigate max_nn locations).
# Cluster ratio: <0.05 = clean; >0.1 = tune to spread points.
# Space/time breakdowns: If void_ratio_time >> void_ratio_space, time is under-sampled → increase scale_time or decrease time_shift_range

# def scale_points(points, scale_time = scale_time):
#     if points.shape[1] == 4:
#         P = points.copy()
#         P[:, 3] *= scale_time
#         return P
#     return points

# def nn_distance_cv(points, scale_time=scale_time):
#     P = scale_points(points, scale_time)
#     tree = cKDTree(P)
#     d, _ = tree.query(P, k=2)   # d[:,0] = 0 (self)
#     nn = d[:, 1]
#     return np.std(nn) / np.mean(nn)

# # CV	Interpretation
# # > 0.25	poor
# # 0.15–0.25	OK (Poisson only)
# # 0.08–0.15	good
# # < 0.08	excellent

# def knn_volume_cv(points, k=8, scale_time=scale_time):
#     P = scale_points(points, scale_time)
#     tree = cKDTree(P)
#     d, _ = tree.query(P, k=k+1)  # includes self
#     rk = d[:, -1]
#     dim = P.shape[1]
#     volumes = rk**dim
#     return np.std(volumes) / np.mean(volumes)


# def r_min_func(points):
#     r_min_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) # + depth_range[0]
#     return r_min_vals

# def r_max_func(points):
#     r_max_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) # + depth_range[1]
#     return r_max_vals


# def r_min_func(points):
#     r_min_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) # + depth_range[0]
#     return r_min_vals

# def r_max_func(points):
#     r_max_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) # + depth_range[1]
#     return r_max_vals


# def scale_points(points, scale_time = scale_time):
#     if points.shape[1] == 4:
#         P = points.copy()
#         P[:, 3] *= scale_time
#         return P
#     return points


# up_sample_factor = 10 if use_time_shift == False else 20
# number_candidate_nodes = up_sample_factor*number_of_spatial_nodes







# def collect_regular_lattice(n_trgt, lat_range = lat_range_extend, lon_range = lon_range_extend, use_global = use_global, tol_fraction = 0.01, max_iter = 100):

# 	r = Area/Area_globe
# 	n_low = max(1, int((n_trgt / r) * 0.2))
# 	n_high = int((n_trgt / r) * 3.0) + 1
# 	n_current = int(0.5*(n_low + n_high)) if use_global == False else n_trgt

# 	## Set tolerance as ~1% of grid, and then retain only this fraction
# 	tol = int(np.floor(tol_fraction*n_trgt))


# 	def random_rotation_matrix():
# 	    u1, u2, u3 = np.random.rand(3)

# 	    q = np.array([
# 	        np.sqrt(1 - u1) * np.sin(2*np.pi*u2),
# 	        np.sqrt(1 - u1) * np.cos(2*np.pi*u2),
# 	        np.sqrt(u1)     * np.sin(2*np.pi*u3),
# 	        np.sqrt(u1)     * np.cos(2*np.pi*u3)
# 	    ])

# 	    w, x, y, z = q
# 	    return np.array([
# 	        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
# 	        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
# 	        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
# 	    ])


# 	def fibonacci_sphere_latlon(N):

# 	    # Golden ratio constants
# 	    phi = (1 + 5**0.5) / 2
# 	    alpha = 1 / phi

# 	    # Random phases
# 	    delta_phi, delta_z = np.random.rand(2)

# 	    n = np.arange(N)

# 	    # Fibonacci sphere (vectorized)
# 	    z = 1 - 2 * (n + delta_z) / N
# 	    theta = 2 * np.pi * (n * alpha + delta_phi)
# 	    r = np.sqrt(1 - z*z)

# 	    x = r * np.cos(theta)
# 	    y = r * np.sin(theta)

# 	    P = np.column_stack((x, y, z))

# 	    # Random global rotation
# 	    R = random_rotation_matrix()
# 	    P = P @ R.T

# 	    # Cartesian → lat/lon
# 	    lon = 180.0*np.arctan2(P[:, 1], P[:, 0])/np.pi        # [-pi, pi)
# 	    lat = 180.0*np.arcsin(np.clip(P[:, 2], -1, 1))/np.pi  # [-pi/2, pi/2]

# 	    return np.concatenate((lat.reshape(-1,1), lon.reshape(-1,1)), axis = 1)


# 	iter_cnt = 0
# 	found_grid = False
# 	while (iter_cnt < max_iter)*(found_grid == False):


# 		points = fibonacci_sphere_latlon(n_current)
# 		if use_global == False:
# 			ifind = np.where((points[:,0] < lat_range[1])*(points[:,0] > lat_range[0])*(points[:,1] < lon_range[1])*(points[:,1] > lon_range[0]))[0]
# 			points = points[ifind]

# 		n_pts = len(points)
# 		if (np.abs(n_pts - n_trgt) <= tol)*(n_pts >= n_trgt):
# 			found_grid = True

# 		else:
# 			if n_pts < n_trgt:
# 				n_low = n_current
# 			else:
# 				n_high = n_current
# 			n_current = int(0.5*(n_low + n_high))

# 			# n_current = int(0.5*(n_low + n_high))
# 		iter_cnt += 1
# 		print('Iter: %d, Diff: %d'%(iter_cnt, n_pts - n_trgt))

# 	if n_pts > n_trgt:
# 		points = points[0:n_trgt] # , size = n_trgt, replace = False)]

# 	return points

# # def knn_distance(x_proj1, x_proj2, idx1, idx2, centroid_proj, k = 10):

# # 	if isinstance(x_proj1, np.ndarray):

# # 		dist_ref = knn(torch.Tensor(centroid_proj[idx1]).to(device), torch.Tensor(centroid_proj[idx1]).to(device))

# def knn_distance(
# 	x_rel_query,      # Tensor [M, 3] float32: relative coords of query points
# 	x_rel_db,         # Tensor [N, 3] float32: relative coords of database points (can == query for self)
# 	idx_query,        # LongTensor [M]: centroid indices for queries
# 	idx_db,           # LongTensor [N]: centroid indices for database
# 	centroids,        # Tensor [C, 3] float64: absolute centroid positions
# 	k=10,
# 	device='cuda' if torch.cuda.is_available() else 'cpu',
# 	return_edges = True,
# 	use_self_loops = True
# ):

# 	if isinstance(x_rel_query, np.ndarray):
# 		"""
# 		Returns K-nearest neighbors (distances and indices) using relative coords + centroids.
# 		"""
# 		# Move everything to device
# 		x_rel_query = torch.Tensor(x_rel_query).to(device)
# 		x_rel_db = torch.Tensor(x_rel_db).to(device)
# 		idx_query = torch.Tensor(idx_query).long().to(device)
# 		idx_db = torch.Tensor(idx_db).long().to(device)
# 		centroids = torch.Tensor(centroids).to(device)  # float64 preserved
    
# 	# Reconstruct effective absolute positions
# 	abs_query = centroids[idx_query] + x_rel_query  # [M, 3]
# 	abs_db = centroids[idx_db] + x_rel_db          # [N, 3]
    
# 	# Pairwise distances (exact, GPU-optimized)
# 	D = torch.cdist(abs_query, abs_db)  # [M, N], float64 if centroids dominate
    
# 	# Top-K smallest
# 	if use_self_loops == False:
# 		distances, indices = torch.topk(D, k + 1, dim=1, largest=False, sorted=True) ## This distance not reliable due to limited GPU ram, must re-compute for subset of indices
# 		distances = distances[:,1::]
# 		indices = indices[:,1::]
# 	else:
# 		distances, indices = torch.topk(D, k, dim=1, largest=False, sorted=True) ## This distance not reliable due to limited GPU ram, must re-compute for subset of indices


# 	rel_refs = (~(idx_query.reshape(-1,1) == idx_db[indices])).unsqueeze(2)*(centroids[idx_query].unsqueeze(1) - centroids[idx_db[indices]])
# 	rel_local = x_rel_query.unsqueeze(1) - x_rel_db[indices]
# 	distances = torch.norm(rel_refs + rel_local, dim = 2)

# 	# distances = torch.norm()

# 	if return_edges == True:

# 		edges = torch.cat((indices.reshape(1,-1), torch.arange(len(x_rel_query), device = device).repeat_interleave(k).reshape(1,-1)), dim = 0)

# 		return edges, distances.reshape(-1,1), (rel_refs + rel_local).reshape(-1,rel_refs.shape[2])

# 	else:

# 		return distances, indices  # distances: [M, k], indices: [M, k] (into db)











# ## Create relative centroid lattice (for removing means and establishing stable numerical values)
# ## Could also use this approach to create a dense grid for estimating densities?
# n_frac_lattice = 0.1
# scale_km = 1000.0
# centroid_lattice = collect_regular_lattice(int(len(x_grid)*n_frac_lattice))
# centroid_lattice_proj = ftrns1_abs(np.concatenate((centroid_lattice, np.mean(depth_range)*np.ones((len(centroid_lattice),1))), axis = 1))/scale_km
# if use_time_shift == True: 
# 	# centroid_lattice = np.concatenate((centroid_lattice, np.zeros((len(centroid_lattice),1))), axis = 1)
# 	centroid_lattice_proj = np.concatenate((centroid_lattice_proj, np.zeros((len(centroid_lattice_proj),1))), axis = 1)

# tree_centroids = cKDTree(centroid_lattice_proj[:,:3])
# x_grid_id = tree_centroids.query(ftrns1_abs(x_grid[:,:3]))[1]
# x_grid_proj_local = ftrns1_abs(x_grid[:,:3])/scale_km - centroid_lattice_proj[x_grid_id,:3]
# if use_time_shift == True: x_grid_proj_local = np.concatenate((x_grid_proj_local, x_grid[:,[3]]/scale_km), axis = 1)



# from scipy.spatial import KDTree
# import numpy as np

# # Oversampled points
# xyz_t_over = np.hstack([xyz_over, t_over.reshape(-1,1)])

# # Scale for emphasis on depth (z is ~depth axis, but use full metric)
# w_scale = 6371e3 / (2 * time_shift_range)  # as before
# points_scaled = xyz_t_over.copy()
# points_scaled[:, 3] *= w_scale  # time
# # To prioritize depth, artificially scale the depth direction (approximate as radial norm variation)
# r_over = np.linalg.norm(xyz_over, axis=1)
# depth_scale = 5.0  # arbitrary factor >1 to make depth "larger" in metric → denser sampling preserved
# points_scaled[:, :3] *= (r_over[:, np.newaxis] / np.mean(r_over)) * depth_scale  # or just scale z: points_scaled[:,2] *= depth_scale

# # Thin to N points by greedy farthest-point sampling (approximates higher depth density)
# keep_idx = []
# remaining = set(range(N_over))
# current = np.random.choice(list(remaining))  # start random
# keep_idx.append(current)
# remaining.remove(current)

# tree = KDTree(points_scaled[list(remaining)])

# while len(keep_idx) < N and remaining:
#     dists = tree.query(points_scaled[keep_idx[-1]])[0]
#     farthest = np.argmax(dists)
#     next_idx = list(remaining)[farthest]
#     keep_idx.append(next_idx)
#     remaining.remove(next_idx)
#     # Update tree (inefficient for large N; use batch for speed)
#     tree = KDTree(points_scaled[list(remaining)])

# xyz = xyz_over[keep_idx]
# t = t_over[keep_idx]



# def nn_distance_stats1(xyz_t, w_scale=scale_time):
#     """
#     Compute statistics on nearest-neighbor distances in scaled space-time.
    
#     Returns:
#     - mean_nn_dist
#     - median_nn_dist  
#     - normalized_mean (higher = better uniformity)
#     """
#     points_scaled = xyz_t.copy()
#     points_scaled[:, 3] *= w_scale  # scale time dimension
    
#     tree = cKDTree(points_scaled)
#     dists, _ = tree.query(points_scaled, k=2)  # k=2 to get nearest non-self
#     nn_dists = dists[:, 1]  # nearest-neighbor distances for all points
    
#     mean_nn = np.mean(nn_dists)
#     median_nn = np.median(nn_dists)
#     min_nn = np.min(nn_dists)
    
#     # Approximate expected mean NN distance for uniform random points in 4D
#     # Using Gamma function approximation: E[NN] ≈ 0.65 * (V / N)^{1/4} in 4D
#     N = len(xyz_t)
#     ranges = np.ptp(xyz_t, axis=0)
#     vol_spatial = np.prod(ranges[:3])
#     vol_time_scaled = ranges[3] * w_scale
#     total_vol = vol_spatial * vol_time_scaled
    
#     # More accurate constant for 4D unit ball, scaled
#     expected_random_mean = 0.65 * (total_vol / N) ** (1/4)
    
#     normalized_mean = mean_nn / expected_random_mean
    
#     stats = {
#         'mean_nn_dist': mean_nn,
#         'median_nn_dist': median_nn,
#         'min_nn_dist': min_nn,
#         'normalized_mean': normalized_mean,   # >1.0 = better than random
#         'nn_distances': nn_dists  # for histograms if desired
#     }

#     # Usage
#     # stats = nn_distance_stats(xyz_t, w_scale=w_scale)
#     print(f"Mean NN distance: {stats['mean_nn_dist']:.1f} m (spatial+time)")
#     print(f"Normalized mean: {stats['normalized_mean']:.3f}")
#     print(f"Interpretation: 1.0 = random, 1.3–1.6 = good, >1.7 = excellent low-discrepancy")

#     return stats

# def nn_distance_stats1(
#     xyz_t, 
#     scale_time, 
#     Volume_space,  # actual spatial shell volume from compute_expected_spacing
#     time_range  # T, so full time span = 2 * time_range
# ):
#     """
#     Compute statistics on nearest-neighbor distances in scaled space-time.
    
#     Returns dict with:
#     - mean_nn_dist
#     - median_nn_dist
#     - min_nn_dist
#     - normalized_mean (higher = better uniformity)
#     - nn_distances (array for histograms)
#     """
#     points_scaled = xyz_t.copy()
#     points_scaled[:, 3] *= scale_time  # scale time dimension
    
#     tree = cKDTree(points_scaled)
#     dists, _ = tree.query(points_scaled, k=2)  # k=2 to get nearest non-self
#     nn_dists = dists[:, 1]  # nearest-neighbor distances for all points
    
#     mean_nn = np.mean(nn_dists)
#     median_nn = np.median(nn_dists)
#     min_nn = np.min(nn_dists)
    
#     # Actual hypervolume: spatial shell volume * scaled time range
#     N = len(xyz_t)
#     hypervolume = Volume_space * (scale_time * 2 * time_range)
    
#     # Approximate expected mean NN for uniform random in 4D (constant tuned for ball-like domain)
#     expected_random_mean = 0.65 * (hypervolume / N) ** (1/4)
    
#     normalized_mean = mean_nn / expected_random_mean
    
#     stats = {
#     	'min_nn_dist': min_nn,
#         'mean_nn_dist': mean_nn,
#         'median_nn_dist': median_nn,
#         'normalized_mean': normalized_mean[0],   # >1.0 = better than random
#         'nn_distances': nn_dists  # for histograms if desired
#     }

#     # print(stats)
#     # Usage
#     # stats = nn_distance_stats(xyz_t, w_scale=w_scale)
#     print(f"Min NN distance: {stats['min_nn_dist']:.1f} m (spatial+time)")
#     print(f"Mean NN distance: {stats['mean_nn_dist']:.1f} m (spatial+time)")
#     print(f"Median NN distance: {stats['median_nn_dist']:.1f} m (spatial+time)")
#     print(f"Normalized mean: {stats['normalized_mean']:.3f} (spatial+time)")
#     print(f"Interpretation: 1.0 = random, 1.3–1.6 = good, >1.7 = excellent low-discrepancy")

#     return stats



# def farthest_point_sampling_global_shell(
#     xyz_t_candidates, 
#     target_N, 
#     scale_time=1.0, 
#     depth_boost=1.0, 
#     mask_candidates=None, 
#     device='cuda'
# ):
#     # 1. Convert to Tensor
#     # xyz_t_candidates: [N, 4] where 0:3 are ECEF meters, 3 is time
#     points = torch.as_tensor(xyz_t_candidates, device=device, dtype=torch.float64)
#     N = points.shape[0]
    
#     # 2. Extract Components
#     coords = points[:, :3]
#     time = points[:, 3:] * scale_time
    
#     # 3. Normalize Radius and Isolate Depth
#     # Calculate radius of every point
#     radii = torch.norm(coords, dim=1, keepdim=True)
#     mean_radius = torch.mean(radii)
    
#     # Normalized surface coordinates (Unit Sphere)
#     unit_coords = coords / radii
    
#     # Relative depth (distance from mean radius, e.g., -20km to +20km)
#     # Scale this relative to the unit sphere (1.0 = one Earth Radius)
#     relative_depth = (radii - mean_radius) / mean_radius
    
#     # 4. Reconstruct Feature Space for Sampling
#     # We amplify the depth so the algorithm 'cares' about the 40km thickness
#     sampling_space = torch.cat([
#         unit_coords,                 # Surface position (Unit Scale)
#         relative_depth * depth_boost, # Amplified Depth
#         time / mean_radius           # Time scaled relative to Earth Radius
#     ], dim=1)

#     # 5. FPS Logic
#     distance = torch.full((N,), float('inf'), device=device, dtype=torch.float64)
#     mask = torch.as_tensor(mask_candidates, device=device, dtype=torch.bool)
    
#     # Start at a random REAL point
#     real_indices = torch.where(mask)[0]
#     farthest = real_indices[torch.randint(0, len(real_indices), (1,))].item()
    
#     collected_indices = []
#     cnt_found = 0
    
#     while cnt_found < target_N:
#         collected_indices.append(farthest)
#         if mask[farthest]:
#             cnt_found += 1
            
#         centroid = sampling_space[farthest, :].view(1, -1)
#         # Calculate squared distance in our custom 'boosted' space
#         dist = torch.sum((sampling_space - centroid) ** 2, dim=-1)
        
#         distance = torch.min(distance, dist)
#         distance[farthest] = -1.0
#         farthest = torch.argmax(distance).item()
        
#     final_indices = torch.tensor(collected_indices, device=device)
#     final_indices = final_indices[mask[final_indices]]
    
#     return final_indices.cpu().numpy()






# def poisson_disk_filter(
#     points,
#     h,
#     use_time = use_time_shift,
#     scale_time = scale_time,
#     t_min = -time_shift_range,
#     t_max =  time_shift_range,
#     use_mirrored = True,
#     use_mirrored_time = True,
#     use_probablistic_acceptance = True,
#     prob_factor = 1.5,
#     mc_samples = 300,
# ):
#     """
#     Dimension-agnostic Poisson disk filter.
#     Works for 3D or 4D (space-time).
#     """

#     M, D = points.shape
#     assert D == 3 or (D == 4 and use_time)

#     N = 4 if use_time else 3   # Poisson dimension
#     h2 = h * h
#     cell_size = h / sqrt(N)

#     grid = defaultdict(list)
#     accepted = []

#     order = np.random.permutation(M)

#     for idx in order:
#         p = points[idx]

#         # --- scaled coordinate for hashing ---
#         if use_time:
#             pN = np.array([p[0], p[1], p[2], scale_time * p[3]], dtype = float)
#         else:
#             # pN = p[:3].astype('float')
#             pN = np.array([p[0], p[1], p[2]], dtype = float)

#         cell = tuple(int(floor(pN[i] / cell_size)) for i in range(N))

#         # --- radial geometry (3D only) ---
#         x = p[:3]
#         rp = np.linalg.norm(x)
#         u = x / rp

#         r_min = r_min_func(x.reshape(1,-1))
#         r_max = r_max_func(x.reshape(1,-1))

#         need_min = ((rp - r_min) < h)*(use_mirrored == True)
#         need_max = ((r_max - rp) < h)*(use_mirrored == True)

#         ok = True

#         # --- neighbor search ---
#         for offset in product([-1, 0, 1], repeat=N):
#             nbr_cell = tuple(cell[i] + offset[i] for i in range(N))
#             if nbr_cell not in grid:
#                 continue

#             for q in grid[nbr_cell]:

#                 # --- original ---
#                 if use_time:
#                     dq = np.array([
#                         p[0] - q[0],
#                         p[1] - q[1],
#                         p[2] - q[2],
#                         scale_time * (p[3] - q[3]),
#                     ])
#                 else:
#                     dq = p[:3] - q[:3]

#                 if np.dot(dq, dq) < h2:
#                     ok = False
#                     break

#                 # --- radial mirrors ---
#                 qx = q[:3]
#                 rq = np.linalg.norm(qx)
#                 uq = qx / rq

#                 if need_min:
#                     qmin = q.copy()
#                     qmin[:3] = uq * (2 * r_min_func(uq.reshape(1,-1)*rq) - rq)

#                     if use_time:
#                         dq = np.array([
#                             p[0] - qmin[0],
#                             p[1] - qmin[1],
#                             p[2] - qmin[2],
#                             scale_time * (p[3] - qmin[3]),
#                         ])
#                     else:
#                         dq = p[:3] - qmin[:3]

#                     if np.dot(dq, dq) < h2:
#                         ok = False
#                         break

#                 if need_max:
#                     qmax = q.copy()
#                     qmax[:3] = uq * (2 * r_max_func(uq.reshape(1,-1)*rq) - rq)

#                     if use_time:
#                         dq = np.array([
#                             p[0] - qmax[0],
#                             p[1] - qmax[1],
#                             p[2] - qmax[2],
#                             scale_time * (p[3] - qmax[3]),
#                         ])
#                     else:
#                         dq = p[:3] - qmax[:3]

#                     if np.dot(dq, dq) < h2:
#                         ok = False
#                         break

#                 # --- time mirroring ---
#                 if use_time and use_mirrored_time:
#                     if (p[3] - t_min) < h / scale_time:
#                         qtm = q.copy()
#                         qtm[3] = 2 * t_min - q[3]
#                         dq = np.array([
#                             p[0] - qtm[0],
#                             p[1] - qtm[1],
#                             p[2] - qtm[2],
#                             scale_time * (p[3] - qtm[3]),
#                         ])
#                         if np.dot(dq, dq) < h2:
#                             ok = False
#                             break

#                     if (t_max - p[3]) < h / scale_time:
#                         qtp = q.copy()
#                         qtp[3] = 2 * t_max - q[3]
#                         dq = np.array([
#                             p[0] - qtp[0],
#                             p[1] - qtp[1],
#                             p[2] - qtp[2],
#                             scale_time * (p[3] - qtp[3]),
#                         ])
#                         if np.dot(dq, dq) < h2:
#                             ok = False
#                             break

#             if not ok:
#                 break

#         if not ok:
#             continue

#         # --- probabilistic acceptance ---
#         if use_probablistic_acceptance:
#             Nmc = mc_samples
#             v = np.random.randn(Nmc, N)
#             v /= np.linalg.norm(v, axis=1)[:, None]
#             v *= (h / 2) * (np.random.rand(Nmc) ** (1 / N))[:, None]

#             if use_time:
#                 samples = np.zeros((Nmc, 4))
#                 samples[:, :3] = p[:3] + v[:, :3]
#                 samples[:, 3]  = p[3] + v[:, 3] / scale_time
#             else:
#                 samples = p[:3] + v

#             rs = np.linalg.norm(samples[:, :3], axis=1)
#             inside = (rs >= r_min) & (rs <= r_max)
#             f = inside.mean()

#             # --- compute local occupancy ---
#             gamma = 1.0
#             occupancy = 0
#             for offset in product([-1,0,1], repeat=N):
#                 nbr_cell = tuple(cell[i]+offset[i] for i in range(N))
#                 occupancy += len(grid.get(nbr_cell, []))

#             adj_factor = (h**N / (occupancy + 1e-6))**gamma
#             if np.random.rand() > (f * adj_factor)**prob_factor:
#                 continue
#                     # if np.random.rand() > (inside.mean() ** prob_factor):
#                     #     continue

#         # --- accept ---
#         grid[cell].append(p)
#         accepted.append(p)

#     return np.array(accepted)

# def poisson_exact_count(points, target_N, h0, max_iter = 300, tol_fraction = 0.001, prob_factor = 1.5, use_probablistic_acceptance = True, use_mirrored = True):
#     """
#     points   : candidate points (oversampled)
#     target_N : desired number of accepted points
#     h0       : initial spacing guess
#     """
#     h_low = 0.5 * h0
#     h_high = 2.0 * h0

#     ## Set tolerance as ~1% of grid, and then retain only this fraction
#     tol = int(np.floor(tol_fraction*target_N))

#     best_pts = None

#     for iter_count in range(max_iter):
        
#         h = 0.5 * (h_low + h_high)

#         # if use_mirrored == False:
#         pts = poisson_disk_filter(points, h, prob_factor = prob_factor, use_probablistic_acceptance = use_probablistic_acceptance, use_mirrored = use_mirrored)
#         # else:
#         #   pts = poisson_disk_filter_mirrored(points, h)

#         # pts = poisson_disk_filter(points, h)
#         n = len(pts)

#         if n > target_N:
#             h_low = h
#             best_pts = pts

#         else:
#             h_high = h
#             best_pts = pts

#         print('Finished iteration %d (diff %d)'%(iter_count, n - target_N))

#         if (abs(n - target_N) <= tol)*(n >= target_N):
#             break

#     # If slightly too many, truncate safely
#     if len(best_pts) > target_N:
#         best_pts = best_pts[np.random.choice(len(best_pts), size = target_N, replace = False)]

#     return ftrns2_abs(best_pts), h

# def farthest_point_sampling1(xyz_t_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor):

#     points_scaled = xyz_t_candidates.copy()
#     points_scaled[:, 3] *= scale_time*time_upscale_factor

#     if depth_boost != 1:
#     	points_scaled = ftrns1_abs(ftrns2_abs(xyz_t_candidates)*np.array([1.0, 1.0, depth_boost, 1.0]))
#         # points_scaled[:, 2] *= depth_boost  # or radial as before
    
#     M = len(points_scaled)
#     keep_idx = [np.random.randint(M)]  # start with random seed
#     remaining = list(set(range(M)) - set(keep_idx))
    
#     tree = cKDTree(points_scaled[keep_idx])
    
#     while len(keep_idx) < target_N and remaining:
#         dists = tree.query(points_scaled[remaining])[0]
#         farthest = np.argmax(dists)
#         next_idx = remaining.pop(farthest)
#         keep_idx.append(next_idx)
#         tree = cKDTree(points_scaled[keep_idx])  # update tree
    
#     return ftrns2_abs(xyz_t_candidates[keep_idx])


# def farthest_point_sampling1(xyz_t_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = None, use_cuda = True):

# 	points_scaled = xyz_t_candidates.copy()
# 	points_scaled[:, 3] *= scale_time*time_upscale_factor

# 	if depth_boost != 1:
# 		points_scaled = ftrns1_abs(ftrns2_abs(xyz_t_candidates)*np.array([1.0, 1.0, depth_boost, 1.0]))
# 		# points_scaled[:, 2] *= depth_boost  # or radial as before
    
# 	M = len(points_scaled)
# 	keep_idx = [np.random.randint(M)]  # start with random seed
# 	remaining = list(set(range(M)) - set(keep_idx))
# 	cnt_found = 0

# 	if mask_candidates is None: mask_candidates = np.ones(len(xyz_t_candidates)) ## Can use mask to create dummy points
# 	assert(mask_candidates.sum() >= target_N)
# 	assert(len(mask_candidates) == len(xyz_t_candidates))

# 	if (torch.cuda.is_available() == True)*(use_cuda == True):
# 		# current_points = torch.Tensor(points_scaled[keep_idx], device = device)
# 		# remaining_points = torch.Tensor(points_scaled[remaining], device = device)
# 		# keep_idx = torch.tensor(keep_idx, device = device).long()

# 		points_scaled = torch.tensor(points_scaled, device = device)

# 		while cnt_found < target_N and remaining:
# 			farthest = torch.argmax(torch.cdist(points_scaled[keep_idx], \
# 				points_scaled[remaining]).amin(0))
# 			next_idx = remaining.pop(farthest)
# 			keep_idx.append(next_idx)
# 			cnt_found += 1 if mask_candidates[next_idx] == 1 else 0
# 			if (cnt_found % 100) == 0: print(cnt_found)
# 	else:

# 		tree = cKDTree(points_scaled[keep_idx])
# 		while cnt_found < target_N and remaining:
# 			dists = tree.query(points_scaled[remaining])[0]
# 			farthest = np.argmax(dists)
# 			next_idx = remaining.pop(farthest)
# 			keep_idx.append(next_idx)
# 			tree = cKDTree(points_scaled[keep_idx])  # update tree
# 			cnt_found += 1 if mask_candidates[next_idx] == 1 else 0
# 			if (cnt_found % 100) == 0: print(cnt_found)

# 	keep_idx = np.array(keep_idx)
# 	keep_idx = keep_idx[np.where(mask_candidates[keep_idx] == 1)[0]]

# 	return ftrns2_abs(xyz_t_candidates[keep_idx])





# def compute_expected_spacing1(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, scale_time = scale_time, time_upscale_factor = time_upscale_factor, depth_scale = depth_upscale_factor, use_global = use_global):

# 	Area_globe = 4*np.pi*(earth_radius**2)
# 	if use_global == True:
# 		Area = 4*np.pi*(earth_radius**2)
# 		Volume = (4.0*np.pi/3.0)*(r_max**3 - r_min**3)
# 		Volume_space = 1.0*Volume

# 	else:
# 		Area = (earth_radius**2)*(np.deg2rad(lon_range_extend[1]) - np.deg2rad(lon_range_extend[0]))*(np.sin(np.pi*lat_range_extend[1]/180.0) - np.sin(np.pi*lat_range_extend[0]/180.0))
# 		Volume = Area*(r_max**3 - r_min**3)/(3*(earth_radius**2))
# 		Volume_space = 1.0*Volume

# 	if use_time == True:
# 		Volume = Volume*(2*scale_time*time_shift_range) # time_upscale_factor

# 	## Determine nominal node spacing
# 	if use_time == False:
# 		hex_factor = 0.74048
# 		nominal_spacing = (Volume/(hex_factor*N))**(1/3) ## Hex-based spacing
# 		nominal_spacing_space = (Volume_space/(hex_factor*N))**(1/3) ## Hex-based spacing
# 		nominal_spacing_time = 0.0 # (2 * time_shift_range) / (target_N ** (1/4)) / scale_time

# 	else:
# 		hex_factor = 1.0 # ≈0.125
# 		nominal_spacing = (Volume/(hex_factor*N))**(1/4) ## Hex-based spacing
# 		nominal_spacing_space = (Volume_space/(hex_factor*N))**(1/3) ## Hex-based spacing
# 		nominal_spacing_time = (2 * time_shift_range) / (N ** (1/4)) # / scale_time

# 	return nominal_spacing, nominal_spacing_space, nominal_spacing_time





# def farthest_point_sampling1(points_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = None, device='cuda'):
#     """
#     points: [N, 3] or [N, 4] (already scaled/transformed)
#     target_N: Number of 'real' points to collect
#     mask_candidates: Tensor/Array of 1s (real) and 0s (buffer/mirrored)
#     """

#     scale_val = (1000.0)*10.0
#     points = points_candidates.copy()
#     if points.shape[1] == 4: points[:,3] *= scale_time
#     if depth_boost != 1.0: points = ftrns1_abs(ftrns2_abs(points)*np.array([1.0, 1.0, depth_boost, 1.0]))
#     origin = points[:, :3].mean(axis = 0, keepdims = True)
#     points[:,:3] = points[:,:3] - origin
#     points = torch.as_tensor(points/scale_val, device = device, dtype = torch.float64)

#     if mask_candidates is None: mask_candidates = np.ones(len(points))
#     mask = torch.as_tensor(mask_candidates, device = device, dtype = torch.bool)
#     assert(len(mask) == len(points))
#     assert(mask.sum().item() >= target_N)

#     N, C = points.shape
    
#     # Trackers
#     # We don't know the exact total length, so we use a list or a large buffer
#     collected_indices = []
#     distance = torch.full((N,), float('inf'), device = device, dtype = torch.float64)
    
#     # Start with a random REAL point
#     real_indices = torch.where(mask)[0]
#     dist_centroid = torch.sum((points[real_indices] - points[real_indices].mean(0, keepdims = True))**2, dim = 1)
#     init_index = torch.argmin(dist_centroid)
#     farthest = real_indices[init_index].item()
    
#     cnt_found = 0
    
#     # Optimization: pre-fetch mask to avoid device transfers in loop
#     mask_np = mask.cpu().numpy() if device == 'cuda' else mask.numpy()

#     while cnt_found < target_N:
#         collected_indices.append(farthest)
        
#         # Increment counter only if it's a real point
#         if mask[farthest]:
#             cnt_found += 1
            
#         # Standard FPS Update: O(N)
#         centroid = points[farthest, :].view(1, C)
#         dist = torch.sum((points - centroid) ** 2, dim = -1)
        
#         # Update distances to the closest selected point
#         distance = torch.min(distance, dist)
        
#         # Crucial: Set the distance of the point we just picked to -1 
#         # so it is never picked again
#         distance[farthest] = -1.0
        
#         # Next farthest
#         farthest = torch.argmax(distance).item()

#     # Final Filter: Only keep the indices that were 'real' points
#     all_selected = torch.tensor(collected_indices, device=device)
#     final_indices = all_selected[mask[all_selected]]
    
#     return ftrns2_abs(points_candidates[final_indices.cpu().detach().numpy()])


# def farthest_point_sampling1(points_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = None, device='cuda'):
#     """
#     points: [N, 3] or [N, 4] (already scaled/transformed)
#     target_N: Number of 'real' points to collect
#     mask_candidates: Tensor/Array of 1s (real) and 0s (buffer/mirrored)
#     """

#     scale_val = (1000.0)*10.0
#     points = points_candidates.copy()
#     if points.shape[1] == 4: points[:,3] *= scale_time
#     if depth_boost != 1.0: points = ftrns1_abs(ftrns2_abs(points)*np.array([1.0, 1.0, depth_boost, 1.0]))
#     origin = points[:, :3].mean(axis = 0, keepdims = True)
#     points[:,:3] = points[:,:3] - origin
#     points = torch.as_tensor(points/scale_val, device = device, dtype = torch.float64)

#     if mask_candidates is None: mask_candidates = np.ones(len(points))
#     mask = torch.as_tensor(mask_candidates, device = device, dtype = torch.bool)
#     assert(len(mask) == len(points))
#     assert(mask.sum().item() >= target_N)

#     N, C = points.shape
    
#     # Trackers
#     # We don't know the exact total length, so we use a list or a large buffer
#     collected_indices = []
#     distance = torch.full((N,), float('inf'), device = device, dtype = torch.float64)
    

#     # Start with a random REAL point
#     real_indices = torch.where(mask)[0]
#     dist_centroid = torch.sum((points[real_indices] - points[real_indices].mean(0, keepdims = True))**2, dim = 1)
#     init_index = torch.argmin(dist_centroid)
#     farthest = real_indices[init_index].item()
    
#     cnt_found = 0
    
#     # Optimization: pre-fetch mask to avoid device transfers in loop
#     mask_np = mask.cpu().numpy() if device == 'cuda' else mask.numpy()

#     while cnt_found < target_N:
#         collected_indices.append(farthest)
        
#         # Increment counter only if it's a real point
#         if mask[farthest]:
#             cnt_found += 1
            
#         # Standard FPS Update: O(N)
#         centroid = points[farthest, :].view(1, C)
#         dist = torch.sum((points - centroid) ** 2, dim = -1)
        
#         # Update distances to the closest selected point
#         distance = torch.min(distance, dist)
        
#         # Crucial: Set the distance of the point we just picked to -1 
#         # so it is never picked again
#         distance[farthest] = -1.0
        
#         # Next farthest
#         farthest = torch.argmax(distance).item()

#     # Final Filter: Only keep the indices that were 'real' points
#     all_selected = torch.tensor(collected_indices, device=device)
#     final_indices = all_selected[mask[all_selected]]
    
#     return ftrns2_abs(points_candidates[final_indices.cpu().detach().numpy()])



