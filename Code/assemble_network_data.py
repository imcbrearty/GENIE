
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



## Create graph functions

def regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global):

	if use_spherical == False:
		a = 6378137.0
		b = 6356752.3142
	else:
		a = 6371e3
		b = 6371e3

	# u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)
	# longitude = (2*np.pi*u[:,0] - np.pi)*180.0/np.pi
	# latitude = np.arccos(((a**2)*(1 - u[:,1]) + (b**2)*u[:,1] - a**2 + b**2) / (b**2 - a**2))

	# m = int(np.ceil(np.log2(N)))
	# initial_points = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random_base2(m = m)[0:N]

	u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)  # Sobol 4D
	if use_global == False:
		phi = lon_range[0] + u[:,0]*(lon_range[1] - lon_range[0])
		u_min = (1.0 + np.sin(np.deg2rad(lat_range[0])))/2.0
		u_max = (1.0 + np.sin(np.deg2rad(lat_range[1])))/2.0
		theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
		theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)

	else:
		phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
		# theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
		theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)

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

def poisson_disk_filter(
    points,
    h,
    use_time = use_time_shift,
    scale_time = scale_time,
    t_min = -time_shift_range,
    t_max =  time_shift_range,
    use_mirrored = True,
    use_mirrored_time = True,
    use_probablistic_acceptance = True,
    prob_factor = 1.5,
    mc_samples = 300,
):
    """
    Dimension-agnostic Poisson disk filter.
    Works for 3D or 4D (space-time).
    """

    M, D = points.shape
    assert D == 3 or (D == 4 and use_time)

    N = 4 if use_time else 3   # Poisson dimension
    h2 = h * h
    cell_size = h / sqrt(N)

    grid = defaultdict(list)
    accepted = []

    order = np.random.permutation(M)

    for idx in order:
        p = points[idx]

        # --- scaled coordinate for hashing ---
        if use_time:
            pN = np.array([p[0], p[1], p[2], scale_time * p[3]], dtype = float)
        else:
            # pN = p[:3].astype('float')
            pN = np.array([p[0], p[1], p[2]], dtype = float)

        cell = tuple(int(floor(pN[i] / cell_size)) for i in range(N))

        # --- radial geometry (3D only) ---
        x = p[:3]
        rp = np.linalg.norm(x)
        u = x / rp

        r_min = r_min_func(x.reshape(1,-1))
        r_max = r_max_func(x.reshape(1,-1))

        need_min = ((rp - r_min) < h)*(use_mirrored == True)
        need_max = ((r_max - rp) < h)*(use_mirrored == True)

        ok = True

        # --- neighbor search ---
        for offset in product([-1, 0, 1], repeat=N):
            nbr_cell = tuple(cell[i] + offset[i] for i in range(N))
            if nbr_cell not in grid:
                continue

            for q in grid[nbr_cell]:

                # --- original ---
                if use_time:
                    dq = np.array([
                        p[0] - q[0],
                        p[1] - q[1],
                        p[2] - q[2],
                        scale_time * (p[3] - q[3]),
                    ])
                else:
                    dq = p[:3] - q[:3]

                if np.dot(dq, dq) < h2:
                    ok = False
                    break

                # --- radial mirrors ---
                qx = q[:3]
                rq = np.linalg.norm(qx)
                uq = qx / rq

                if need_min:
                    qmin = q.copy()
                    qmin[:3] = uq * (2 * r_min_func(uq.reshape(1,-1)*rq) - rq)

                    if use_time:
                        dq = np.array([
                            p[0] - qmin[0],
                            p[1] - qmin[1],
                            p[2] - qmin[2],
                            scale_time * (p[3] - qmin[3]),
                        ])
                    else:
                        dq = p[:3] - qmin[:3]

                    if np.dot(dq, dq) < h2:
                        ok = False
                        break

                if need_max:
                    qmax = q.copy()
                    qmax[:3] = uq * (2 * r_max_func(uq.reshape(1,-1)*rq) - rq)

                    if use_time:
                        dq = np.array([
                            p[0] - qmax[0],
                            p[1] - qmax[1],
                            p[2] - qmax[2],
                            scale_time * (p[3] - qmax[3]),
                        ])
                    else:
                        dq = p[:3] - qmax[:3]

                    if np.dot(dq, dq) < h2:
                        ok = False
                        break

                # --- time mirroring ---
                if use_time and use_mirrored_time:
                    if (p[3] - t_min) < h / scale_time:
                        qtm = q.copy()
                        qtm[3] = 2 * t_min - q[3]
                        dq = np.array([
                            p[0] - qtm[0],
                            p[1] - qtm[1],
                            p[2] - qtm[2],
                            scale_time * (p[3] - qtm[3]),
                        ])
                        if np.dot(dq, dq) < h2:
                            ok = False
                            break

                    if (t_max - p[3]) < h / scale_time:
                        qtp = q.copy()
                        qtp[3] = 2 * t_max - q[3]
                        dq = np.array([
                            p[0] - qtp[0],
                            p[1] - qtp[1],
                            p[2] - qtp[2],
                            scale_time * (p[3] - qtp[3]),
                        ])
                        if np.dot(dq, dq) < h2:
                            ok = False
                            break

            if not ok:
                break

        if not ok:
            continue

        # --- probabilistic acceptance ---
        if use_probablistic_acceptance:
            Nmc = mc_samples
            v = np.random.randn(Nmc, N)
            v /= np.linalg.norm(v, axis=1)[:, None]
            v *= (h / 2) * (np.random.rand(Nmc) ** (1 / N))[:, None]

            if use_time:
                samples = np.zeros((Nmc, 4))
                samples[:, :3] = p[:3] + v[:, :3]
                samples[:, 3]  = p[3] + v[:, 3] / scale_time
            else:
                samples = p[:3] + v

            rs = np.linalg.norm(samples[:, :3], axis=1)
            inside = (rs >= r_min) & (rs <= r_max)
            f = inside.mean()

            # --- compute local occupancy ---
            gamma = 1.0
            occupancy = 0
            for offset in product([-1,0,1], repeat=N):
                nbr_cell = tuple(cell[i]+offset[i] for i in range(N))
                occupancy += len(grid.get(nbr_cell, []))

            adj_factor = (h**N / (occupancy + 1e-6))**gamma
            if np.random.rand() > (f * adj_factor)**prob_factor:
                continue
                    # if np.random.rand() > (inside.mean() ** prob_factor):
                    #     continue

        # --- accept ---
        grid[cell].append(p)
        accepted.append(p)

    return np.array(accepted)

def poisson_exact_count(points, target_N, h0, max_iter = 300, tol_fraction = 0.001, prob_factor = 1.5, use_probablistic_acceptance = True, use_mirrored = True):
    """
    points   : candidate points (oversampled)
    target_N : desired number of accepted points
    h0       : initial spacing guess
    """
    h_low = 0.5 * h0
    h_high = 2.0 * h0

    ## Set tolerance as ~1% of grid, and then retain only this fraction
    tol = int(np.floor(tol_fraction*target_N))

    best_pts = None

    for iter_count in range(max_iter):
        
        h = 0.5 * (h_low + h_high)

        # if use_mirrored == False:
        pts = poisson_disk_filter(points, h, prob_factor = prob_factor, use_probablistic_acceptance = use_probablistic_acceptance, use_mirrored = use_mirrored)
        # else:
        #   pts = poisson_disk_filter_mirrored(points, h)

        # pts = poisson_disk_filter(points, h)
        n = len(pts)

        if n > target_N:
            h_low = h
            best_pts = pts

        else:
            h_high = h
            best_pts = pts

        print('Finished iteration %d (diff %d)'%(iter_count, n - target_N))

        if (abs(n - target_N) <= tol)*(n >= target_N):
            break

    # If slightly too many, truncate safely
    if len(best_pts) > target_N:
        best_pts = best_pts[np.random.choice(len(best_pts), size = target_N, replace = False)]

    return ftrns2_abs(best_pts), h

def farthest_point_sampling(xyz_t_candidates, target_N, scale_time = scale_time, depth_boost = depth_upscale_factor):

    points_scaled = xyz_t_candidates.copy()
    points_scaled[:, 3] *= scale_time*time_upscale_factor

    if depth_boost != 1:
    	points_scaled = ftrns1_abs(ftrns2_abs(xyz_t_candidates)*np.array([1.0, 1.0, depth_boost, 1.0]))
        # points_scaled[:, 2] *= depth_boost  # or radial as before
    
    M = len(points_scaled)
    keep_idx = [np.random.randint(M)]  # start with random seed
    remaining = list(set(range(M)) - set(keep_idx))
    
    tree = cKDTree(points_scaled[keep_idx])
    
    while len(keep_idx) < target_N and remaining:
        dists = tree.query(points_scaled[remaining])[0]
        farthest = np.argmax(dists)
        next_idx = remaining.pop(farthest)
        keep_idx.append(next_idx)
        tree = cKDTree(points_scaled[keep_idx])  # update tree
    
    return ftrns2_abs(xyz_t_candidates[keep_idx])

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

def nn_distance_stats(xyz_t, w_scale=scale_time):
    """
    Compute statistics on nearest-neighbor distances in scaled space-time.
    
    Returns:
    - mean_nn_dist
    - median_nn_dist  
    - normalized_mean (higher = better uniformity)
    """
    points_scaled = xyz_t.copy()
    points_scaled[:, 3] *= w_scale  # scale time dimension
    
    tree = cKDTree(points_scaled)
    dists, _ = tree.query(points_scaled, k=2)  # k=2 to get nearest non-self
    nn_dists = dists[:, 1]  # nearest-neighbor distances for all points
    
    mean_nn = np.mean(nn_dists)
    median_nn = np.median(nn_dists)
    min_nn = np.min(nn_dists)
    
    # Approximate expected mean NN distance for uniform random points in 4D
    # Using Gamma function approximation: E[NN] ≈ 0.65 * (V / N)^{1/4} in 4D
    N = len(xyz_t)
    ranges = np.ptp(xyz_t, axis=0)
    vol_spatial = np.prod(ranges[:3])
    vol_time_scaled = ranges[3] * w_scale
    total_vol = vol_spatial * vol_time_scaled
    
    # More accurate constant for 4D unit ball, scaled
    expected_random_mean = 0.65 * (total_vol / N) ** (1/4)
    
    normalized_mean = mean_nn / expected_random_mean
    
    stats = {
        'mean_nn_dist': mean_nn,
        'median_nn_dist': median_nn,
        'min_nn_dist': min_nn,
        'normalized_mean': normalized_mean,   # >1.0 = better than random
        'nn_distances': nn_dists  # for histograms if desired
    }

    # Usage
    # stats = nn_distance_stats(xyz_t, w_scale=w_scale)
    print(f"Mean NN distance: {stats['mean_nn_dist']:.1f} m (spatial+time)")
    print(f"Normalized mean: {stats['normalized_mean']:.3f}")
    print(f"Interpretation: 1.0 = random, 1.3–1.6 = good, >1.7 = excellent low-discrepancy")

    return stats

def scale_points(points, scale_time = scale_time):
    if points.shape[1] == 4:
        P = points.copy()
        P[:, 3] *= scale_time
        return P
    return points

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

def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_min
	return r_val


def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_max
	return -r_val

def scale_points(points, scale_time = scale_time):
    if points.shape[1] == 4:
        P = points.copy()
        P[:, 3] *= scale_time
        return P
    return points


## Not actually used but useful base statistics

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


if use_time_shift == True:
    Volume = Volume*(2*scale_time*time_shift_range)

## Determine nominal node spacing
if use_time_shift == False:
    nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing
    nominal_spacing_space = (Volume_space/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing

else:
    nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/4) ## Hex-based spacing
    nominal_spacing_space = (Volume_space/(0.74048*number_of_spatial_nodes))**(1/3) ## Hex-based spacing

up_sample_factor = 10 if use_time_shift == False else 20
number_candidate_nodes = up_sample_factor*number_of_spatial_nodes


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

# # moi

# for p in perm_options: ## Note the increased tol_fraction for this search
# 	# x_grid, _ = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, tol_fraction = 0.01, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])
# 	x_grid = regular_sobolov(number_of_spatial_nodes)


## Now build all spatial grids using optimal sampling strategy
x_grids = []
for n in range(num_grids):

	# trial_points = collect_trial_points(number_candidate_nodes)
	# x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])

	if use_poisson_filtering == True:

		p = [1.0, False, False] ## Optimize this choice (on the first grid built)
		trial_points = regular_sobolov(number_candidate_nodes)
		x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])

	elif use_farthest_point_filtering == True:

		## Increase efficiency of this script
		trial_points = regular_sobolov(number_candidate_nodes)
		x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes)

	else:

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
	nn_distance_stats(ftrns1_abs(x_grid))
	x_grids.append(np.expand_dims(x_grid, axis = 0))

x_grids = np.vstack(x_grids)
np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_templates_ver_1.npz'%name_of_project, x_grids = x_grids, corr1 = np.zeros((1,3)), corr2 = np.zeros((1,3)))

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
		A_edges_c = from_networkx(nx.margulis_gabber_galil_graph(int_need)).edge_index.long().flip(0).contiguous()
		Ac = subgraph(torch.arange(number_of_spatial_nodes), A_edges_c)[0].cpu().detach().numpy() # .to(device)

	np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_expanders_ver_1.npz'%name_of_project, Ac = Ac)


print("All files saved successfully!")
print("✔ Script execution: Done")
