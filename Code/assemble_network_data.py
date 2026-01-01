
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
from torch_scatter import scatter
from scipy.stats import pearsonr
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


## User: Input stations and spatial region
## (must have station and region files at
## (ext_dir + 'stations.npz'), and
## (ext_dir + 'region.npz')

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
r_min = optimize_r_min(np.array([soln.x])); print('\n')

bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True)
r_max = -1.0*optimize_r_max(np.array([soln.x])); print('\n')
assert(r_max >= r_min)


### Define Fibonnaci sampling routine ####

Area_globe = 4*np.pi*(earth_radius**2)
if use_global == True:
    Area = 4*np.pi*(earth_radius**2)
    Volume = (4.0*np.pi/3.0)*(r_max**3 - r_min**3)
    Volume_space = 1.0*Volume

else:
    Area = (earth_radius**2)*(np.deg2rad(lon_range_extend[1]) - np.deg2rad(lon_range_extend[0]))*(np.sin(np.pi*lat_range_extend[1]/180.0) - np.sin(np.pi*lat_range_extend[0]/180.0))
    Volume = Area*(r_max**3 - r_min**3)/(3*(earth_radius**2))
    Volume_space = 1.0*Volume


## Estimate an optimal time scaling for isotropic spacing
if use_time_shift == True:
    dx = (Volume/number_of_spatial_nodes)**(1/3)
    dt = 2*time_shift_range/(number_of_spatial_nodes**(1/4))
    scale_time_effective = dx/dt
    print('Isotropic scaling effective time scale: %0.4f m/s'%scale_time_effective) ## For a given spatial and temporal volume
    ## Could use this to guide how much time window can be increased or decreased

    if use_effective_time_scale == True:
    	print('Overwriting time scale as effective time scale: %0.8f (from %0.8f)'%(scale_time_effective, scale_time))
    	scale_time = scale_time_effective[0]


if use_time_shift == True:
	Volume = Volume*(2*scale_time*time_shift_range)

## Determine nominal node spacing
if use_time_shift == False:
	nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing
	nominal_spacing_space = (Volume_space/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing

else:
	nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/4) ## Hex-based spacing
	nominal_spacing_space = (Volume_space/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing




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


# def u_to_geodetic_lat(u, lat_min, lat_max):
#     # Mapping u [0,1] to sin(lat) is spherical equal-area.
#     # For ellipsoidal (Authalic), the formula involves the eccentricity squared (e^2).
#     # e2 = 0.00669437999014 for WGS84.
#     u_min = (1.0 + np.sin(np.deg2rad(lat_min))) / 2.0
#     u_max = (1.0 + np.sin(np.deg2rad(lat_max))) / 2.0
#     u_val = u_min + u * (u_max - u_min)
#     return np.arcsin(2 * u_val - 1) * (180.0 / np.pi)


def u_to_geodetic_lat(u_random, lat_range):
    """
    Converts a uniform random variable u [0, 1] to Geodetic Latitude 
    using an Authalic (equal-area) mapping for the WGS84 ellipsoid.
    """
    # WGS84 Constant: e (eccentricity)
    e = 0.0818191908426
    
    def get_q(lat_deg):
        # q is the authalic part of the projection
        sin_lat = np.sin(np.deg2rad(lat_deg))
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


def regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0):

	if use_spherical == False:
		a = 6378137.0
		b = 6356752.3142
	else:
		a = 6371e3
		b = 6371e3

	if buffer_scale > 0.0:
		## Use a buffer around min-max regions. How to estimate? First estimate volume
		Volume, Volume_space, _, nominal_spacing_space, nominal_spacing_time = compute_expected_spacing(N, lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, time_range = time_range, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
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
		Volume_expanded, Volume_space_expanded, _, _, _ = compute_expected_spacing(N, lat_range = expanded_lat, lon_range = expanded_lon, depth_range = expanded_depth, time_range = expanded_time, use_time = use_time, use_global = use_global, scale_time = scale_time)  # w_scale: length per unit time use_global=use_global,)  # T, full range = 2T use_time=use_time_shift, scale_time=scale_time,  # w_scale: length per unit time use_global=use_global,)
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
		r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
		r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
		xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
		r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
		r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
		xyz = (r*xyz_surface)/r_surface
		x_grid = ftrns2_abs(xyz) 

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
		r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
		r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
		xyz = (r*xyz_surface)/r_surface
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

    scale_val = 10000.0
    points = points_candidates.copy()
    if points.shape[1] == 4: points[:, 3] *= scale_time
    if depth_boost != 1.0: points = ftrns1_abs(ftrns2_abs(points) * np.array([1.0, 1.0, depth_boost, 1.0]))
    origin = points[:, :3].mean(axis = 0, keepdims = True)
    points[:, :3] -= origin
    points = torch.as_tensor(points / scale_val, device = device, dtype = torch.float64)

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
    return ftrns2_abs(points_candidates[final_indices.cpu().numpy()])



# import torch



def collect_regular_lattice(n_trgt, lat_range = lat_range_extend, lon_range = lon_range_extend, use_global = use_global, tol_fraction = 0.01, max_iter = 100):

	r = Area/Area_globe
	n_low = max(1, int((n_trgt / r) * 0.2))
	n_high = int((n_trgt / r) * 3.0) + 1
	n_current = int(0.5*(n_low + n_high)) if use_global == False else n_trgt

	## Set tolerance as ~1% of grid, and then retain only this fraction
	tol = int(np.floor(tol_fraction*n_trgt))

	iter_cnt = 0
	found_grid = False
	while (iter_cnt < max_iter)*(found_grid == False):


		points = fibonacci_sphere_latlon(n_current)
		if use_global == False:
			ifind = np.where((points[:,0] < lat_range[1])*(points[:,0] > lat_range[0])*(points[:,1] < lon_range[1])*(points[:,1] > lon_range[0]))[0]
			points = points[ifind]

		n_pts = len(points)
		if (np.abs(n_pts - n_trgt) <= tol)*(n_pts >= n_trgt):
			found_grid = True

		else:
			if n_pts < n_trgt:
				n_low = n_current
			else:
				n_high = n_current
			n_current = int(0.5*(n_low + n_high))

			# n_current = int(0.5*(n_low + n_high))
		iter_cnt += 1
		print('Iter: %d, Diff: %d'%(iter_cnt, n_pts - n_trgt))

	if n_pts > n_trgt:
		points = points[0:n_trgt] # , size = n_trgt, replace = False)]

	return points

# def knn_distance(x_proj1, x_proj2, idx1, idx2, centroid_proj, k = 10):

# 	if isinstance(x_proj1, np.ndarray):

# 		dist_ref = knn(torch.Tensor(centroid_proj[idx1]).to(device), torch.Tensor(centroid_proj[idx1]).to(device))

def knn_distance(
	x_rel_query,      # Tensor [M, 3] float32: relative coords of query points
	x_rel_db,         # Tensor [N, 3] float32: relative coords of database points (can == query for self)
	idx_query,        # LongTensor [M]: centroid indices for queries
	idx_db,           # LongTensor [N]: centroid indices for database
	centroids,        # Tensor [C, 3] float64: absolute centroid positions
	k=10,
	device='cuda' if torch.cuda.is_available() else 'cpu',
	return_edges = True,
	use_self_loops = True
):

	if isinstance(x_rel_query, np.ndarray):
		"""
		Returns K-nearest neighbors (distances and indices) using relative coords + centroids.
		"""
		# Move everything to device
		x_rel_query = torch.Tensor(x_rel_query).to(device)
		x_rel_db = torch.Tensor(x_rel_db).to(device)
		idx_query = torch.Tensor(idx_query).long().to(device)
		idx_db = torch.Tensor(idx_db).long().to(device)
		centroids = torch.Tensor(centroids).to(device)  # float64 preserved
    
	# Reconstruct effective absolute positions
	abs_query = centroids[idx_query] + x_rel_query  # [M, 3]
	abs_db = centroids[idx_db] + x_rel_db          # [N, 3]
    
	# Pairwise distances (exact, GPU-optimized)
	D = torch.cdist(abs_query, abs_db)  # [M, N], float64 if centroids dominate
    
	# Top-K smallest
	if use_self_loops == False:
		distances, indices = torch.topk(D, k + 1, dim=1, largest=False, sorted=True) ## This distance not reliable due to limited GPU ram, must re-compute for subset of indices
		distances = distances[:,1::]
		indices = indices[:,1::]
	else:
		distances, indices = torch.topk(D, k, dim=1, largest=False, sorted=True) ## This distance not reliable due to limited GPU ram, must re-compute for subset of indices


	rel_refs = (~(idx_query.reshape(-1,1) == idx_db[indices])).unsqueeze(2)*(centroids[idx_query].unsqueeze(1) - centroids[idx_db[indices]])
	rel_local = x_rel_query.unsqueeze(1) - x_rel_db[indices]
	distances = torch.norm(rel_refs + rel_local, dim = 2)

	# distances = torch.norm()

	if return_edges == True:

		edges = torch.cat((indices.reshape(1,-1), torch.arange(len(x_rel_query), device = device).repeat_interleave(k).reshape(1,-1)), dim = 0)

		return edges, distances.reshape(-1,1), (rel_refs + rel_local).reshape(-1,rel_refs.shape[2])

	else:

		return distances, indices  # distances: [M, k], indices: [M, k] (into db)



def loss_metrics(x_grid, grid_ind = 0, use_time_shift = use_time_shift, plot_on = False):

	## Compute quality checks:
	# [1]. Flat in depth
	n_bins = 30
	r = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]), axis = 1)
	h_vals = np.histogram(r**3, bins = int(len(x_grid)/n_bins)) # [0]
	mean_loss = np.mean(n_bins*np.ones(len(h_vals[0])) - h_vals[0])
	rms_loss = (np.sqrt(((n_bins*np.ones(len(h_vals[0])) - h_vals[0])**2).sum()/len(h_vals[0]))/n_bins)
	print('\nMean deviation of radius flatness: %0.8f'%mean_loss)
	print('RMS deviation of radius flatness: %0.8f'%rms_loss)

	# [2]. Sorted depths
	n = len(x_grid)
	# u = np.arange(1, n+1) / n
	u = np.arange(n) / (n - 1)
	iarg = np.argsort(x_grid[:,2])
	d_sorted = np.sort(x_grid[:,2])
	# expected CDF
	r_surface = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1)
	# r = r_surface[iarg] + d_sorted
	r = r_surface.mean() + d_sorted
	F_expected = (r**3 - r_min**3) / (r_max**3 - r_min**3)
	r2_loss = r2_score(F_expected, u)
	print('R2 of expected depth distribution: %0.8f'%r2_loss)

	# diagnostic plot
	if plot_on == True:
		plt.figure()
		plt.plot(d_sorted, u, label="empirical")
		plt.plot(d_sorted, F_expected, "--", label="expected")
		plt.legend()
		fig = plt.gcf()
		fig.set_size_inches([8,8])
		plt.savefig(path_to_file + 'Plots' + seperator + 'grid_sorted_depths_ver_%d.png'%grid_ind, bbox_inches = 'tight', pad_inches = 0.1)

	# [3]. Nearest neighbor distances, and nearest neighbors as function of depth
	if use_time_shift == False:
		tree = cKDTree(ftrns1(x_grid))
		q = tree.query(ftrns1(x_grid), k = 2)[0][:,1]
		min_dist = q.min()/1000.0
		mean_dist = q.mean()/1000.0
		std_dist = q.std()/1000.0
		print('Nearest neighbors: Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(min_dist, mean_dist, std_dist))

	else:
		tree = cKDTree(x_grid*np.array([1.0, 1.0, 1.0, scale_time]).reshape(1,-1))
		q = tree.query(x_grid*np.array([1.0, 1.0, 1.0, scale_time]).reshape(1,-1), k = 2) # [0][:,1]
		q = np.linalg.norm(ftrns1(x_grid[:,0:3]) - ftrns1(x_grid[q[1][:,1],0:3]), axis = 1)		
		min_dist = q.min()/1000.0
		mean_dist = q.mean()/1000.0
		std_dist = q.std()/1000.0
		print('Nearest neighbors: Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(min_dist, mean_dist, std_dist))


	nn_cv = nn_distance_cv(ftrns1_abs(x_grid[:,0:3]), scale_time)
	knn_cv = knn_volume_cv(ftrns1_abs(x_grid[:,0:3]), k=8, scale_time=scale_time)
	print(f"Spatial: NN-CV={nn_cv:.3f}, kNN-CV={knn_cv:.3f}")

	if x_grid.shape[1] == 4:
		nn_cv = nn_distance_cv(ftrns1_abs(x_grid), scale_time)
		knn_cv = knn_volume_cv(ftrns1_abs(x_grid), k=8, scale_time=scale_time)
		print(f"Full: NN-CV={nn_cv:.3f}, kNN-CV={knn_cv:.3f}")

	if use_time_shift == True:
		# [4]. R2 of expected time distribution
		n = len(x_grid)
		# u = np.arange(1, n+1) / n
		u = np.arange(n) / (n - 1)
		iarg = np.argsort(x_grid[:,3])
		t_sorted = np.sort(x_grid[:,3])
		F_expected = (t_sorted - (-time_shift_range)) / (time_shift_range - (-time_shift_range))
		r2_loss_time = r2_score(F_expected, u)
		print('R2 of expected time distribution: %0.8f'%r2_loss_time)

		# [5]. Check nearest neighbors
		tree = cKDTree(x_grid*np.array([1.0, 1.0, 1.0, scale_time]).reshape(1,-1))
		q = tree.query(x_grid*np.array([1.0, 1.0, 1.0, scale_time]).reshape(1,-1), k = 2) # [0][:,1]
		q_min = q[0][:,1].min()/1000.0
		q_mean = q[0][:,1].mean()/1000.0
		q_std = q[0][:,1].std()/1000.0
		print('Nearest neighbors (scaled): Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(q_min, q_mean, q_std))


		# [6]. Space offsets
		dist_space = np.linalg.norm(ftrns1(x_grid[:,0:3]) - ftrns1(x_grid[q[1][:,1],0:3]), axis = 1)
		dist_time = np.abs(x_grid[:,3] - x_grid[q[1][:,1],3])
		# min_spc_dist = dist_space.min()/1000.0
		# mean_spc_dist = dist_space.mean()/1000.0
		# std_spc_dist = dist_space.std()/1000.0
		# print('Nearest neighbors: Min: %0.4f, Mean: %0.4f (+/- %0.4f) km \n'%(min_spc_dist, mean_spc_dist, std_spc_dist))

		# [6]. Correlation of space and time nearest neighbors
		pearsonr_val = pearsonr(dist_space, dist_time).statistic
		print('Correlation of space and time nearest neighbors: %0.8f'%pearsonr_val)

		# [7]. Ratio of small dt
		ratio_within_time_radius = len(np.where(dist_time*scale_time < 0.005*nominal_spacing)[0])/len(dist_time)
		print('Ratio of small time offset nearest neighbors: %0.8f \n'%ratio_within_time_radius)


	return mean_loss, rms_loss, r2_loss, mean_dist, std_dist



# import numpy as np
# from scipy.spatial import cKDTree

def nn_distance_stats(
    xyz_t, 
    scale_time, 
    Volume_space,  # from compute_expected_spacing
    time_range,    # T, full span = 2 * time_range
    cluster_threshold=0.5  # tunable: nn_dist < threshold * mean_nn = cluster
):
    points_scaled = xyz_t.copy()
    points_scaled[:, 3] *= scale_time
    
    tree = cKDTree(points_scaled)
    dists, _ = tree.query(points_scaled, k=2)
    nn_dists = dists[:, 1]
    
    mean_nn = np.mean(nn_dists)
    median_nn = np.median(nn_dists)
    min_nn = np.min(nn_dists)
    max_nn = np.quantile(nn_dists, 0.99)
    std_nn = np.std(nn_dists)
    cv_nn = std_nn / mean_nn if mean_nn > 0 else 0.0  # Coefficient of Variation
    
    hypervolume = Volume_space * (scale_time * 2 * time_range)
    expected_random_mean = 0.65 * (hypervolume / len(xyz_t)) ** (1/4)
    normalized_mean = mean_nn / expected_random_mean
    
    # Void ratio: max gap relative to mean spacing
    void_ratio = max_nn / mean_nn  # >2-3 = notable voids
    
    # Cluster ratio: fraction of small NN distances
    cluster_count = np.sum(nn_dists < (cluster_threshold * mean_nn))
    cluster_ratio = cluster_count / len(nn_dists)  # 0 = no clusters, >0.1 = some crowding
    
    # Spatial-only (3D projection) stats for space voids/clusters
    tree_space = cKDTree(xyz_t[:, :3])
    dists_space, _ = tree_space.query(xyz_t[:, :3], k=2)
    nn_dists_space = dists_space[:, 1]
    mean_nn_space = np.mean(nn_dists_space)
    void_ratio_space = np.quantile(nn_dists_space, 0.99) / mean_nn_space
    cluster_ratio_space = np.sum(nn_dists_space < (cluster_threshold * mean_nn_space)) / len(nn_dists_space)
    
    # Time-only (1D projection) stats for time voids/clusters
    t_reshaped = xyz_t[:, 3].reshape(-1, 1)  # 1D for KDTree
    tree_time = cKDTree(t_reshaped)
    dists_time, _ = tree_time.query(t_reshaped, k=2)
    nn_dists_time = dists_time[:, 1]
    mean_nn_time = np.mean(nn_dists_time)
    void_ratio_time = np.quantile(nn_dists_time, 0.99) / mean_nn_time
    cluster_ratio_time = np.sum(nn_dists_time < (cluster_threshold * mean_nn_time)) / len(nn_dists_time)
    
    stats = {
        'mean_nn_dist': mean_nn,
        'median_nn_dist': median_nn,
        'min_nn_dist': min_nn,
        'max_nn_dist': max_nn,
        'cv_nn': cv_nn,  # new: 0.2-0.3 = excellent, >0.5 = random/clustered
        'normalized_mean': normalized_mean[0],
        'void_ratio': void_ratio,  # new: 2-3 = balanced, >4 = voids present
        'cluster_ratio': cluster_ratio,  # new: <0.05 = minimal clustering
        'void_ratio_space': void_ratio_space,
        'cluster_ratio_space': cluster_ratio_space,
        'void_ratio_time': void_ratio_time,
        'cluster_ratio_time': cluster_ratio_time,
        'nn_distances': nn_dists
    }

    # print(stats)
    # Usage
    # stats = nn_distance_stats(xyz_t, w_scale=w_scale)
    print(f"Min NN distance: {stats['min_nn_dist']:.1f} m (spatial+time)")
    print(f"Mean NN distance: {stats['mean_nn_dist']:.1f} m (spatial+time)")
    print(f"Median NN distance: {stats['median_nn_dist']:.1f} m (spatial+time)")
    print(f"\nNormalized mean: {stats['normalized_mean']:.3f} (spatial+time)")
    print(f"Interpretation: 1.0 = random, 1.3–1.6 = good, >1.7 = excellent low-discrepancy\n")
    print(f"CV NN: {stats['cv_nn']:.2f} m (spatial+time) [=0.2-0.3 = excellent, >0.5 = random/clustered]")
    print(f"Void ratio: {stats['void_ratio']:.2f} (spatial+time) [2-3 = balanced, >4 = voids present]")
    print(f"Cluster ratio: {stats['cluster_ratio']:.2f} (spatial+time) [<0.05 = minimal clustering]")
    print(f"\nVoid ratio: {stats['void_ratio_space']:.2f} (spatial)")
    print(f"Cluster ratio: {stats['cluster_ratio_space']:.2f} (spatial)")
    print(f"\nVoid ratio: {stats['void_ratio_time']:.2f} (time)")
    print(f"Cluster ratio: {stats['cluster_ratio_time']:.2f} (time)")


    return stats

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

def nn_distance_cv(points, scale_time=scale_time):
    P = scale_points(points, scale_time)
    tree = cKDTree(P)
    d, _ = tree.query(P, k=2)   # d[:,0] = 0 (self)
    nn = d[:, 1]
    return np.std(nn) / np.mean(nn)

# CV	Interpretation
# > 0.25	poor
# 0.15–0.25	OK (Poisson only)
# 0.08–0.15	good
# < 0.08	excellent

def knn_volume_cv(points, k=8, scale_time=scale_time):
    P = scale_points(points, scale_time)
    tree = cKDTree(P)
    d, _ = tree.query(P, k=k+1)  # includes self
    rk = d[:, -1]
    dim = P.shape[1]
    volumes = rk**dim
    return np.std(volumes) / np.mean(volumes)


def r_min_func(points):
    r_min_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) + depth_range[0]
    return r_min_vals

def r_max_func(points):
    r_max_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) + depth_range[1]
    return r_max_vals


def scale_points(points, scale_time = scale_time):
    if points.shape[1] == 4:
        P = points.copy()
        P[:, 3] *= scale_time
        return P
    return points


# up_sample_factor = 10 if use_time_shift == False else 20
# number_candidate_nodes = up_sample_factor*number_of_spatial_nodes


use_poisson_filtering = False 
use_farthest_point_filtering = True 


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

def evaluate_sampling_quality(nodes, lat_range, lon_range, depth_range, time_range):
    """
    nodes: [N, 4] array (Lat, Lon, Depth, Time)
    """
    N = nodes.shape[0]
    
    # 1. Uniformity Check: Coefficient of Variation (CV)
    # A perfectly uniform (Blue Noise) set has a very low CV of NN distances.
    tree = KDTree(nodes)
    dist, _ = tree.query(nodes, k=2) # k=2 because k=1 is the point itself
    nn_distances = dist[:, 1]
    
    cv = np.std(nn_distances) / np.mean(nn_distances)
    
    # 2. Boundary "Curling" Check
    # We check if the density in the outer 10% of the volume matches the inner 90%
    def get_edge_mask(pts):
        masks = []
        ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
        for i, r in enumerate(ranges):
            margin = (r[1] - r[0]) * 0.1
            masks.append((pts[:, i] < r[0] + margin) | (pts[:, i] > r[1] - margin))
        return np.any(masks, axis=0)

    edge_mask = get_edge_mask(nodes)
    edge_density = np.sum(edge_mask) / N
    # Expected edge density for a 4D hypercube's outer 10% shell is roughly:
    # 1 - (0.8)^4 = 0.59 (59% of points should be in the 'edge' zones)
    
    # 3. Anisotropy Check (Aspect Ratio)
    # We look at the standard deviation of distances along each axis
    axis_scales = np.std(nodes, axis=0)
    aspect_ratios = axis_scales / axis_scales[0] # Relative to Latitude
    
    print("--- Sampling Diagnostics ---")
    print(f"Nearest Neighbor CV: {cv:.4f} (Lower is better, target < 0.15)")
    print(f"Edge Density: {edge_density:.4f} (Target vs. 0.59)")
    print(f"Axis Aspect Ratios (Lat/Lon/Dep/Time): {aspect_ratios}")
    
    return cv, edge_density, aspect_ratios






## Now build all spatial grids using optimal sampling strategy
x_grids = []
for n in range(num_grids):

	# trial_points = collect_trial_points(number_candidate_nodes)
	# x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])

	if use_poisson_filtering == True:

		up_sample_factor = 10 if use_time_shift == False else 20
		number_candidate_nodes = up_sample_factor*number_of_spatial_nodes

		print('Beginning  Poisson filtering [%d]'%n)
		p = [1.0, False, False] ## Optimize this choice (on the first grid built)
		trial_points = regular_sobolov(number_candidate_nodes)
		x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])

	elif use_farthest_point_filtering == True:

		up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
		number_candidate_nodes = up_sample_factor*number_of_spatial_nodes

		print('Beginning FPS sampling [%d]'%n)
		## Increase efficiency of this script
		trial_points, mask_points = regular_sobolov(number_candidate_nodes, N_target = number_of_spatial_nodes, buffer_scale = 2.0)
		x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes, mask_candidates = mask_points)

	else:

		print('Using standard Sobolov sampling [%d]'%n)
		x_grid = regular_sobolov(number_of_spatial_nodes)


	# else:
	# 	p = [1.0, False, False] ## Optimize this choice (on the first grid built)
	# 	trial_points = regular_sobolov(number_candidate_nodes)
	# 	x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])


	tol_frac = 0.01
	assert(x_grid[:,0].min() >= (lat_range_extend[0] - tol_frac*np.diff(lat_range_extend)))
	assert(x_grid[:,0].max() <= (lat_range_extend[1] + tol_frac*np.diff(lat_range_extend)))
	assert(x_grid[:,1].min() >= (lon_range_extend[0] - tol_frac*np.diff(lon_range_extend))) if use_global == False else 1
	assert(x_grid[:,1].max() <= (lon_range_extend[1] + tol_frac*np.diff(lon_range_extend))) if use_global == False else 1
	assert(x_grid[:,2].min() >= (depth_range[0] - tol_frac*np.diff(depth_range)))
	assert(x_grid[:,2].max() <= (depth_range[1] + tol_frac*np.diff(depth_range)))
	assert(len(x_grid) == number_of_spatial_nodes)

	loss_metrics(x_grid, plot_on = True, grid_ind = n)
	# nn_distance_stats(ftrns1_abs(x_grid)/1000.0, w_scale = scale_time/1000.0)
	nn_distance_stats(ftrns1_abs(x_grid), scale_time, Volume_space, time_shift_range)
	x_grids.append(np.expand_dims(x_grid, axis = 0))


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
			# A_edges_c = torch.Tensor(np.random.permutation(A_edges_c.max() + 1)).long()[A_edges_c]
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
			# A_edges_c = torch.Tensor(np.random.permutation(A_edges_c.max() + 1)).long()[A_edges_c]
			Ac, edge_type = subgraph(torch.arange(number_of_spatial_nodes), A_edges_c, edge_attr = edge_type) # [0].cpu().detach().numpy() # .to(device)
			Ac, edge_type = Ac.cpu().detach().numpy(), edge_type.cpu().detach().numpy()

			# A_edges_c = from_networkx(A_edges_c)

	np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_expanders_ver_1.npz'%name_of_project, Ac = Ac)


print("All files saved successfully!")
print("✔ Script execution: Done")



