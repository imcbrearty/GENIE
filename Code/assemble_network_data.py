
import yaml
import numpy as np
import os
import torch
from torch import optim, nn
import shutil
from collections import defaultdict
from sklearn.metrics import r2_score
import pathlib
from math import floor, sqrt
from itertools import product
from torch_geometric.utils import from_networkx
from scipy.stats import pearsonr
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

print('\n Using domain')
print('\n Latitude:')
print(lat_range)
print('\n Longitude:')
print(lon_range)


if fix_nominal_depth == True:
	nominal_depth = 0.0 ## Can change the target depth projection if prefered
else:
	nominal_depth = locs[:,2].mean() ## Can change the target depth projection if prefered

center_loc = np.array([lat_range[0] + 0.5*np.diff(lat_range)[0], lon_range[0] + 0.5*np.diff(lon_range)[0], nominal_depth]).reshape(1,-1)
# center_loc = locs.mean(0, keepdims = True)

use_differential_evolution = True
if use_differential_evolution == True:

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
	
	from scipy.optimize import differential_evolution
	
	# os.rename(ext_dir + 'stations.npz', ext_dir + '%s_stations_backup.npz'%name_of_project)
	soln = optimize_with_differential_evolution(center_loc)
	rbest = rotation_matrix_full_precision(soln.x[0], soln.x[1], soln.x[2])
	mn = soln.x[3::].reshape(1,-1)

## Save station file with projection functions
np.savez_compressed(path_to_file + f'{config["name_of_project"]}_stations.npz', locs = locs, stas = stas, rbest = rbest, mn = mn)
print('Saved station file \n')


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

	# ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	# ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius)  # invert ftrns1	

	ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)


### Build spatial graphs using Poisson disk sampling and elliptical Earth centric density sampling ###

def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_min
	return r_val


def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_max
	return -r_val


def random_rotation_matrix():
    u1, u2, u3 = np.random.rand(3)

    q = np.array([
        np.sqrt(1 - u1) * np.sin(2*np.pi*u2),
        np.sqrt(1 - u1) * np.cos(2*np.pi*u2),
        np.sqrt(u1)     * np.sin(2*np.pi*u3),
        np.sqrt(u1)     * np.cos(2*np.pi*u3)
    ])

    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def fibonacci_sphere_latlon(N):

    # Golden ratio constants
    phi = (1 + 5**0.5) / 2
    alpha = 1 / phi

    # Random phases
    delta_phi, delta_z = np.random.rand(2)

    n = np.arange(N)

    # Fibonacci sphere (vectorized)
    z = 1 - 2 * (n + delta_z) / N
    theta = 2 * np.pi * (n * alpha + delta_phi)
    r = np.sqrt(1 - z*z)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    P = np.column_stack((x, y, z))

    # Random global rotation
    R = random_rotation_matrix()
    P = P @ R.T

    # Cartesian → lat/lon
    lon = 180.0*np.arctan2(P[:, 1], P[:, 0])/np.pi        # [-pi, pi)
    lat = 180.0*np.arcsin(np.clip(P[:, 2], -1, 1))/np.pi  # [-pi/2, pi/2]

    return np.concatenate((lat.reshape(-1,1), lon.reshape(-1,1)), axis = 1)


def hash_coords(points, cell_size):
    """Convert points to integer grid coordinates"""
    return np.floor(points / cell_size).astype(np.int32)


def hash_coords_4d(points, cell_size):
    return np.floor(points / cell_size).astype(np.int32)


# def mirror_radial(points, rmin, rmax):
#     r = np.linalg.norm(points, axis=1)
#     u = points / r[:, None]

#     p_low = u * (2*rmin - r)[:, None]
#     p_high = u * (2*rmax - r)[:, None]

#     return np.vstack([points, p_low, p_high])

def mirror_radial(points, rmin, rmax):
    r = np.linalg.norm(points, axis=1)
    u = points / r[:, None]

    p_low = u * (2*rmin - r)[:, None]
    p_high = u * (2*rmax - r)[:, None]

    return np.vstack([p_low, p_high])

def mirror_time(points, tmin, tmax):

	time_vals = points[:,3]
	points_left = np.copy(points)
	points_right = np.copy(points)
	points_left[:,3] = 2*tmin - points_left[:,3]
	points_right[:,3] = 2*tmax - points_left[:,3]

	return np.vstack([points_left, points_right])

def r_min_func(points):
    r_min_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) + depth_range[0]
    return r_min_vals

def r_max_func(points):
    r_max_vals = np.linalg.norm(ftrns1_abs(ftrns2_abs(points[:,0:3])*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) + depth_range[1]
    return r_max_vals


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

            if np.random.rand() > (inside.mean() ** prob_factor):
                continue

        # --- accept ---
        grid[cell].append(p)
        accepted.append(p)

    return np.array(accepted)



# def poisson_disk_filter(points, h, use_mirrored = True, use_probablistic_acceptance = True, prob_factor = 1.5):
#     """
#     points : (M,3) candidate points
#     h      : minimum spacing
#     """
#     cell_size = h / np.sqrt(3)
#     # cell_size = h / np.sqrt(2)
#     grid = defaultdict(list)

#     accepted = []
#     accepted_pts = []

#     h2 = h * h

#     ## Can add lat lon mirroring as well
#     if use_mirrored == True:

#         points_lat_lon = ftrns2_abs(points)[:,0:2]
#         points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon),1))), axis = 1)
#         r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[0]
#         r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[1]
#         points_mirrored = ftrns1_abs(ftrns2_abs(mirror_radial(ftrns1_abs(points_lat_lon), r_min_vals, r_max_vals)))
#         # points_mirrored = ftrns1(ftrns2_abs(mirror_radial(ftrns1_abs(points_lat_lon), r_min, r_max)))
#         mask_mirrored = np.concatenate((np.zeros(len(points)), np.ones(len(points_mirrored))), axis = 0)
#         points = np.concatenate((points, points_mirrored), axis = 0)

#     else:

#     	mask_mirrored = np.zeros(len(points))

#     # Shuffle candidates (important!)
#     order = np.random.permutation(len(points))

#     for idx in order:

#         p = points[idx]
#         cell = tuple(np.floor(p / cell_size).astype(int))

#         ok = True
#         # Check neighboring cells
#         for dx in (-1, 0, 1):
#             for dy in (-1, 0, 1):
#                 for dz in (-1, 0, 1):
#                     nbr_cell = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
#                     if nbr_cell in grid:
#                         q = np.array(grid[nbr_cell])
#                         if np.any(np.sum((q - p)**2, axis=1) < h2):
#                             ok = False
#                             break
#                 if not ok: break
#             if not ok: break


#         if (ok == True)*(mask_mirrored[idx] == 0):

#             # use_probablistic_acceptance = True
#             if use_probablistic_acceptance == True:

#                 # Compute volume fraction f
#                 r_min_ball = h / 2  # Exclusion radius
#                 # Monte Carlo approx: Sample N points in ball around p
#                 N = 300
#                 offsets = np.random.randn(N, 3)  # Gaussian, scale to uniform ball later
#                 offsets /= np.linalg.norm(offsets, axis=1)[:, None]
#                 offsets *= (np.random.rand(N)[:, None] ** (1/3)) * r_min_ball  # Uniform in ball
#                 samples = p + offsets                

#                 # Check which samples are in domain (need your ftrns funcs for r_min/max)
#                 samples_lat_lon = ftrns2(samples)[:, 0:2]
#                 samples_lat_lon = np.concatenate((samples_lat_lon, np.zeros((len(samples_lat_lon), 1))), axis=1)
#                 r_samples = np.linalg.norm(samples, axis=1)
#                 r_min_vals = np.linalg.norm(ftrns1_abs(samples_lat_lon), axis=1) + depth_range[0]
#                 r_max_vals = np.linalg.norm(ftrns1_abs(samples_lat_lon), axis=1) + depth_range[1]
#                 inside = (r_samples >= r_min_vals) & (r_samples <= r_max_vals)  # Plus any angular constraints if needed
    
#                 f = np.sum(inside) / N
#                 if np.random.rand() < (f**prob_factor):  # Or f**2 for stronger correction
#                     grid[cell].append(p)
#                     accepted_pts.append(p)

#             else:

#                 grid[cell].append(p)
#                 accepted_pts.append(p)

#     return np.array(accepted_pts)

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
        # 	pts = poisson_disk_filter_mirrored(points, h)

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



def relax_poisson(
    points,
    h,
    use_time = use_time_shift,
    scale_time = scale_time,
    t_min = -time_shift_range,
    t_max = time_shift_range,
    iters = 30,
    alpha = 0.3,
    relax_factor = 1.8,
):
    """
    Dimension-agnostic Lloyd-style relaxation for Poisson points.
    """

    N, D = points.shape
    dim = 4 if use_time else 3
    interact_h = relax_factor * h
    interact_h2 = interact_h ** 2

    for _ in range(iters):

        disp = np.zeros_like(points)

        # scaled coordinates for distance computation
        if use_time:
            P = np.column_stack([points[:, :3], scale_time * points[:, 3]])
        else:
            P = points[:, :3]

        for i in range(N):
            d = P - P[i]
            r2 = np.sum(d * d, axis=1)

            mask = (r2 > 1e-12) & (r2 < interact_h2)
            if not np.any(mask):
                continue

            r = np.sqrt(r2[mask])
            w = (1 - r / interact_h)
            w = np.maximum(0, w) ** 2

            u = d[mask] / r[:, None]

            if use_time:
                disp[i, :3] += np.sum(w[:, None] * u[:, :3], axis=0)
                disp[i, 3]  += np.sum(w * u[:, 3]) / scale_time
            else:
                disp[i] += np.sum(w[:, None] * u, axis=0)

        # limit step size
        norm = np.linalg.norm(disp[:, :3], axis=1, keepdims=True)
        max_step = 0.5 * h
        scale = np.minimum(1.0, max_step / (norm + 1e-12))
        disp *= scale

        points += alpha * disp

        # --- project back to admissible domain ---
        x = points[:, :3]
        r = np.linalg.norm(x, axis=1)
        u = x / r[:, None]

        # print(r.shape)
        # print(u.shape)
        rmin = np.array([r_min_func(ri*ui.reshape(1,-1)) for ui, ri in zip(u, r)]).reshape(-1)
        rmax = np.array([r_max_func(ri*ui.reshape(1,-1)) for ui, ri in zip(u, r)]).reshape(-1)

        # pdb.set_trace()
        r_new = np.clip(r, rmin, rmax)
        points[:, :3] = u * r_new[:, None]

        if use_time:
            points[:, 3] = np.clip(points[:, 3], t_min, t_max)

    return ftrns2_abs(points)




bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_min, bounds, popsize = 50, maxiter = 1000, disp = True)
r_min = optimize_r_min(np.array([soln.x])); print('\n')

bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True)
r_max = -1.0*optimize_r_max(np.array([soln.x])); print('\n')
assert(r_max >= r_min)


### Define Fibonnaci sampling routine ####
## Check if using full Earth, set target sampling bounds ##
if (lat_range_extend[0] <= -89.98)*(lat_range_extend[1] >= 89.98)*(lon_range_extend[0] <= -179.98)*(lon_range_extend[1] >= 179.98):
	use_global = True
else:
	use_global = False

if use_global == True:
	Area = 4*np.pi*(earth_radius**2)
	Volume = (4.0*np.pi/3.0)*(r_max**3 - r_min**3)

else:
	Area = (earth_radius**2)*(np.deg2rad(lon_range_extend[1]) - np.deg2rad(lon_range_extend[0]))*(np.sin(np.pi*lat_range_extend[1]/180.0) - np.sin(np.pi*lat_range_extend[0]/180.0))
	Volume = Area*(r_max**3 - r_min**3)/(3*(earth_radius**2))


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
else:
	nominal_spacing = (Volume/(0.74048*number_of_spatial_nodes))**(1/4) ## Hex-based spacing


up_sample_factor = 10 if use_time_shift == False else 20
number_candidate_nodes = up_sample_factor*number_of_spatial_nodes


def collect_trial_points(number_candidate_nodes, use_time_shift = use_time_shift):

	## Collect trial points
	trial_points = []
	n_collect_points = 0
	while n_collect_points < number_candidate_nodes:
		points = fibonacci_sphere_latlon(number_candidate_nodes)
		ifind = np.where((points[:,0] < lat_range_extend[1])*(points[:,0] > lat_range_extend[0])*(points[:,1] < lon_range_extend[1])*(points[:,1] > lon_range_extend[0]))[0]
		trial_points.append(points[ifind])
		n_collect_points += len(ifind)
	trial_points = np.vstack(trial_points)
	trial_points = trial_points[np.random.choice(len(trial_points), size = number_candidate_nodes, replace = False)]

	## Now sample depths for each point

	r_min_vals = np.linalg.norm(ftrns1_abs(np.concatenate((trial_points, np.zeros((len(trial_points),1))), axis = 1)), axis = 1) + depth_range[0]
	r_max_vals = np.linalg.norm(ftrns1_abs(np.concatenate((trial_points, np.zeros((len(trial_points),1))), axis = 1)), axis = 1) + depth_range[1]
	radius_vals = (np.random.rand(number_candidate_nodes)*(r_max_vals**3 - r_min_vals**3) + r_min_vals**3)**(1/3)
	depth_vals = radius_vals - np.linalg.norm(ftrns1_abs(np.concatenate((trial_points, np.zeros((len(trial_points),1))), axis = 1)), axis = 1)
	trial_points = np.concatenate((trial_points, depth_vals.reshape(-1,1)), axis = 1)
	# trial_points_abs = np.concatenate((trial_points, radius_vals.reshape(-1,1)), axis = 1)

	if use_time_shift == True:
		time_samples = np.random.uniform(-time_shift_range, time_shift_range, size = number_candidate_nodes) # scale_time*
		trial_points = np.concatenate((trial_points, time_samples.reshape(-1,1)), axis = 1)

	return trial_points # , trial_points_abs



def loss_metrics(x_grid, grid_ind = 0, use_time_shift = use_time_shift, plot_on = False):

	## Compute quality checks:
	# [1]. Flat in depth
	n_bins = 30
	r = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]), axis = 1)
	h_vals = np.histogram(r**3, bins = int(len(x_grid)/n_bins)) # [0]
	mean_loss = np.mean(n_bins*np.ones(len(h_vals[0])) - h_vals[0])
	rms_loss = (np.sqrt(((n_bins*np.ones(len(h_vals[0])) - h_vals[0])**2).sum()/len(h_vals[0]))/n_bins)
	print('Mean deviation of radius flatness: %0.8f'%mean_loss)
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
		print('Ratio of small time offset nearest neighbors: %0.8f'%ratio_within_time_radius)


	return mean_loss, rms_loss, r2_loss, mean_dist, std_dist

##### Determine optimal sampling strategy ######

## Sampling options
perm_option1 = [1.0, True, True]
perm_option2 = [1.5, True, True]
perm_option3 = [2.0, True, True]
perm_option4 = [1.0, False, True]
perm_option5 = [1.0, False, False]
perm_options = [perm_option1, perm_option2, perm_option3, perm_option4, perm_option5]
use_relaxation = True

## Now implement Poisson disk filtering to obtain the target number of nodes
trial_points = collect_trial_points(number_candidate_nodes)
R2_losses = [] ## Base decision on depth R2 loss (or use the RMS loss)

for p in perm_options: ## Note the increased tol_fraction for this search
	x_grid, _ = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, tol_fraction = 0.01, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])
	# print(p)
	R2_losses.append(loss_metrics(x_grid)[2])
	assert(x_grid[:,0].min() >= lat_range_extend[0])
	assert(x_grid[:,0].max() <= lat_range_extend[1])
	assert(x_grid[:,1].min() >= lon_range_extend[0]) if use_global == False else 1
	assert(x_grid[:,1].max() <= lon_range_extend[1]) if use_global == False else 1
	assert(x_grid[:,2].min() >= depth_range[0])
	assert(x_grid[:,2].max() <= depth_range[1])
	assert(len(x_grid) == number_of_spatial_nodes)

print('R2 losses:')
print(R2_losses)
iarg = np.argmax(np.array(R2_losses))
p = perm_options[iarg]
print('\nOptimal sampling strategy')
print(p); print('\n')



## Now build all spatial grids using optimal sampling strategy
x_grids = []
for n in range(num_grids):

	trial_points = collect_trial_points(number_candidate_nodes)
	x_grid, nominal_spacing = poisson_exact_count(ftrns1_abs(trial_points), number_of_spatial_nodes, nominal_spacing, prob_factor = p[0], use_probablistic_acceptance = p[1], use_mirrored = p[2])
	assert(x_grid[:,0].min() >= lat_range_extend[0])
	assert(x_grid[:,0].max() <= lat_range_extend[1])
	assert(x_grid[:,1].min() >= lon_range_extend[0]) if use_global == False else 1
	assert(x_grid[:,1].max() <= lon_range_extend[1]) if use_global == False else 1
	assert(x_grid[:,2].min() >= depth_range[0])
	assert(x_grid[:,2].max() <= depth_range[1])
	assert(len(x_grid) == number_of_spatial_nodes)

	if use_relaxation == True:
		x_grid_init = np.copy(x_grid)
		x_grid_updated = relax_poisson(ftrns1_abs(x_grid), 1.5*nominal_spacing)
		x_grid = np.copy(x_grid_updated) # relax_poisson(ftrns1_abs(x_grid), 1.5*nominal_spacing)

	loss_metrics(x_grid, plot_on = True, grid_ind = n)
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










# def relax_poisson3(points, h, iters=50, alpha=0.55, relax_factor=2.2):
#     """
#     points: (N, 3) array from your Poisson disk filter
#     h: your original nominal minimum spacing
#     relax_factor: multiplier for interaction range (2.0–2.5 for stronger regularity)
#     """
#     interact_h = h * relax_factor  # Longer range for more global ordering
    
#     for _ in range(iters):
#         disp = np.zeros_like(points)
#         for i, p in enumerate(points):
#             d = points - p  # vectors to all other points
#             r2 = np.sum(d * d, axis=1)
#             mask = (r2 > 1e-8) & (r2 < interact_h**2)  # avoid self, up to interact_h
            
#             if np.any(mask):
#                 r = np.sqrt(r2[mask])
#                 # Cubic weights for stronger near-repulsion
#                 weights = np.maximum(0, 1 - r / interact_h) ** 3
#                 unit_d = d[mask] / r[:, None]
#                 disp[i] += np.sum(weights[:, None] * unit_d, axis=0)
        
#         # Limit max displacement per iter
#         disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
#         max_disp = 0.6 * h
#         disp = np.where(disp_norm > max_disp, disp * (max_disp / disp_norm), disp)
        
#         points += alpha * disp
        
#         # Reproject to valid shell (unchanged)
#         r = np.linalg.norm(points, axis=1)
#         points_lat_lon = ftrns2_abs(points)[:, 0:2]
#         points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon), 1))), axis=1)
#         r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis=1) + depth_range[0]
#         r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis=1) + depth_range[1]
#         r = np.clip(r, r_min_vals, r_max_vals)
#         points = points / np.linalg.norm(points, axis=1)[:, None] * r[:, None]
    
#     return ftrns2_abs(points)  # or return Cartesian if preferred


# def relax_poisson1(points, h, iters = 30, alpha = 0.4, relax_factor = 1.8):
#     """
#     points: (N, 3) array from your Poisson disk filter
#     h: your original nominal minimum spacing
#     relax_factor: multiplier for interaction range (1.5–2.0 recommended)
#     """
#     interact_h = h * relax_factor  # e.g., 1.8 * h → soft long-range push
    
#     for _ in range(iters):
#         disp = np.zeros_like(points)
#         for i, p in enumerate(points):
#             d = points - p  # vectors to all other points
#             r2 = np.sum(d * d, axis=1)
#             mask = (r2 > 1e-8) & (r2 < interact_h**2)  # avoid self, up to interact_h
            
#             if np.any(mask):
#                 r = np.sqrt(r2[mask])
#                 # Stronger near, tapering to zero at interact_h
#                 weights = np.maximum(0, 1 - r / interact_h) ** 2
#                 unit_d = d[mask] / r[:, None]
#                 disp[i] += np.sum(weights[:, None] * unit_d, axis=0)
        
#         # Optional: limit max displacement per iter to prevent instability
#         disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
#         max_disp = 0.5 * h
#         disp = np.where(disp_norm > max_disp, disp * (max_disp / disp_norm), disp)
        
#         points += alpha * disp
        
#         # Your existing reprojection to valid shell (crucial!)
#         r = np.linalg.norm(points, axis=1)
#         points_lat_lon = ftrns2_abs(points)[:, 0:2]
#         points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon), 1))), axis=1)
#         r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis=1) + depth_range[0]
#         r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis=1) + depth_range[1]
#         r = np.clip(r, r_min_vals, r_max_vals)
#         points = points / np.linalg.norm(points, axis=1)[:, None] * r[:, None]
    
#     return ftrns2_abs(points)  # or return Cartesian if preferred





# def poisson_disk_filter1(points, h, use_time_shift = use_time_shift, use_mirrored = True, use_mirrored_time = True, use_probablistic_acceptance = True, prob_factor = 1.5):
#     """
#     points : (M,3) candidate points
#     h      : minimum spacing
#     """
#     cell_size = h / np.sqrt(3)
#     # cell_size = h / np.sqrt(2)
#     grid = defaultdict(list)

#     accepted = []
#     accepted_pts = []

#     h2 = h * h

#     ## Can add lat lon mirroring as well
#     if use_mirrored == True:

#         points_lat_lon = ftrns2_abs(points)[:,0:2]
#         points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon),1))), axis = 1)
#         r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[0]
#         r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[1]
#         points_mirrored = ftrns1_abs(ftrns2_abs(mirror_radial(ftrns1_abs(points_lat_lon), r_min_vals, r_max_vals)))
#         # points_mirrored = ftrns1(ftrns2_abs(mirror_radial(ftrns1_abs(points_lat_lon), r_min, r_max)))
#         mask_mirrored = np.concatenate((np.zeros(len(points)), np.ones(len(points_mirrored))), axis = 0)
#         points = np.concatenate((points, points_mirrored), axis = 0)

#     else:

#     	mask_mirrored = np.zeros(len(points))


#     # if (use_time_shift == True)*(use_mirrored_time == True):

#     # 	points_time_mirrored = 


#     # Shuffle candidates (important!)
#     order = np.random.permutation(len(points))

#     for idx in order:

#         p = points[idx]
#         cell = tuple(np.floor(p / cell_size).astype(int))

#         ok = True
#         # Check neighboring cells
#         for dx in (-1, 0, 1):
#             for dy in (-1, 0, 1):
#                 for dz in (-1, 0, 1):
#                     nbr_cell = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
#                     if nbr_cell in grid:
#                         q = np.array(grid[nbr_cell])
#                         if np.any(np.sum((q - p)**2, axis=1) < h2):
#                             ok = False
#                             break
#                 if not ok: break
#             if not ok: break


#         if (ok == True)*(mask_mirrored[idx] == 0):

#             # use_probablistic_acceptance = True
#             if use_probablistic_acceptance == True:

#                 # Compute volume fraction f
#                 r_min_ball = h / 2  # Exclusion radius
#                 # Monte Carlo approx: Sample N points in ball around p
#                 N = 300
#                 offsets = np.random.randn(N, 3)  # Gaussian, scale to uniform ball later
#                 offsets /= np.linalg.norm(offsets, axis=1)[:, None]
#                 offsets *= (np.random.rand(N)[:, None] ** (1/3)) * r_min_ball  # Uniform in ball
#                 samples = p + offsets                

#                 # Check which samples are in domain (need your ftrns funcs for r_min/max)
#                 samples_lat_lon = ftrns2(samples)[:, 0:2]
#                 samples_lat_lon = np.concatenate((samples_lat_lon, np.zeros((len(samples_lat_lon), 1))), axis=1)
#                 r_samples = np.linalg.norm(samples, axis=1)
#                 r_min_vals = np.linalg.norm(ftrns1_abs(samples_lat_lon), axis=1) + depth_range[0]
#                 r_max_vals = np.linalg.norm(ftrns1_abs(samples_lat_lon), axis=1) + depth_range[1]
#                 inside = (r_samples >= r_min_vals) & (r_samples <= r_max_vals)  # Plus any angular constraints if needed
    
#                 f = np.sum(inside) / N
#                 if np.random.rand() < (f**prob_factor):  # Or f**2 for stronger correction
#                     grid[cell].append(p)
#                     accepted_pts.append(p)

#             else:

#                 grid[cell].append(p)
#                 accepted_pts.append(p)

#     return np.array(accepted_pts)






# if (load_initial_files == True)*(use_pretrained_model == False):
# 	step_load = 20000
# 	ver_load = 1
# 	if os.path.exists(path_to_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(step_load, ver_load)):
# 		shutil.move(path_to_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(step_load, ver_load), path_to_file + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(config["name_of_project"], step_load, ver_load))

# 	ver_load = 1
# 	if os.path.exists(path_to_file + '1d_travel_time_grid_ver_%d.npz'%ver_load):
# 		shutil.move(path_to_file + '1d_travel_time_grid_ver_%d.npz'%ver_load, path_to_file + '1D_Velocity_Models_Regional/%s_1d_travel_time_grid_ver_%d.npz'%(config["name_of_project"], ver_load))

# 	ver_load = 1
# 	if os.path.exists(path_to_file + 'seismic_network_templates_ver_%d.npz'%ver_load):
# 		shutil.move(path_to_file + 'seismic_network_templates_ver_%d.npz'%ver_load, path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(config["name_of_project"], ver_load))

# ## Make spatial grids




# if use_pretrained_model is not None:
# 	shutil.move(path_to_file + 'Pretrained/trained_gnn_model_step_%d_ver_%d.h5'%(20000, use_pretrained_model), path_to_file + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(config["name_of_project"], 20000, 1))
# 	shutil.move(path_to_file + 'Pretrained/1d_travel_time_grid_ver_%d.npz'%use_pretrained_model, path_to_file + '1D_Velocity_Models_Regional/%s_1d_travel_time_grid_ver_%d.npz'%(config["name_of_project"], 1))
# 	shutil.move(path_to_file + 'Pretrained/seismic_network_templates_ver_%d.npz'%use_pretrained_model, path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(use_pretrained_model, 1))

# 	## Find offset corrections if using one of the pre-trained models
# 	## Load these and apply offsets for runing "process_continuous_days.py"
# 	z = np.load(path_to_file + 'Pretrained/stations_ver_%d.npz'%use_pretrained_model)['locs']
# 	sta_loc, rbest, mn = z['locs'], z['rbest'], z['mn']
# 	corr1 = locs.mean(0, keepdims = True)
# 	corr2 = sta_loc.mean(0, keepdims = True)
# 	z.close()

# 	z = np.load(path_to_file + 'Pretrained/region_ver_%d.npz'%use_pretrained_model)
# 	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
# 	z.close()

# 	locs = np.copy(locs) - corr1 + corr2
# 	shutil.copy(path_to_file + 'Pretrained/region_ver_%d.npz'%use_pretrained_model, path_to_file + f'{config["name_of_project"]}_region.npz')

# else:
# 	corr1 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)
# 	corr2 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)




# else:

# 	## If optimizing projection coefficients with this option, need 
# 	## ftrns1 and ftrns2 to accept torch Tensors instead of numpy arrays
# 	if use_spherical == True:

# 		earth_radius = 6371e3
# 		ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef_diff(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
# 		ftrns2 = lambda x, rbest, mn: ecef2lla_diff((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean
	
# 	else:
	
# 		earth_radius = 6378137.0
# 		ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef_diff(x) - mn).T).T # just subtract mean
# 		ftrns2 = lambda x, rbest, mn: ecef2lla_diff((rbest.T @ x.T).T + mn) # just subtract mean
	
# 	## Iterative optimization, does not converge as well

# 	n_attempts = 10

# 	unit_lat = np.array([0.01, 0.0, 0.0]).reshape(1,-1) + center_loc
# 	unit_vert = np.array([0.0, 0.0, 1000.0]).reshape(1,-1) + center_loc
	
# 	norm_lat = torch.Tensor(np.linalg.norm(np.diff(lla2ecef(np.concatenate((center_loc, unit_lat), axis = 0)), axis = 0), axis = 1))
# 	norm_vert = torch.Tensor(np.linalg.norm(np.diff(lla2ecef(np.concatenate((center_loc, unit_vert), axis = 0)), axis = 0), axis = 1))
	
# 	trgt_lat = torch.Tensor([0,1.0,0]).reshape(1,-1)
# 	trgt_vert = torch.Tensor([0,0,1.0]).reshape(1,-1)
# 	trgt_center = torch.zeros(2)
	
# 	loss_func = nn.MSELoss()
	
# 	losses = []
# 	losses1, losses2, losses3, losses4 = [], [], [], []
# 	loss_coef = [1,1,1,0]
	
# 	## Based on initial conditions, sometimes this converges to a projection plane that is flipped polarity.
# 	## E.g., "up" in the lat-lon domain is "down" in the Cartesian domain, and vice versa.
# 	## So try a few attempts to make sure it has the correct polarity.
# 	for attempt in range(n_attempts):
	
# 		vec = nn.Parameter(2.0*np.pi*torch.rand(3))
	
# 		# mn = nn.Parameter(torch.Tensor(lla2ecef(locs).mean(0, keepdims = True)))
# 		mn = nn.Parameter(torch.Tensor(lla2ecef(center_loc.reshape(1,-1)).mean(0, keepdims = True)))
	
# 		optimizer = optim.Adam([vec, mn], lr = 0.001)
	
# 		print('\n Optimize the projection coefficients \n')
	
# 		n_steps_optimize = 5000
# 		for i in range(n_steps_optimize):
	
# 			optimizer.zero_grad()
	
# 			rbest = rotation_matrix(vec[0], vec[1], vec[2])
	
# 			norm_lat = lla2ecef_diff(torch.Tensor(np.concatenate((center_loc, unit_lat), axis = 0)))
# 			norm_vert = lla2ecef_diff(torch.Tensor(np.concatenate((center_loc, unit_vert), axis = 0)))
# 			norm_lat = torch.norm(norm_lat[1] - norm_lat[0])
# 			norm_vert = torch.norm(norm_vert[1] - norm_vert[0])
	
# 			center_out = ftrns1(torch.Tensor(center_loc), rbest, mn)
	
# 			out_unit_lat = ftrns1(torch.Tensor(unit_lat), rbest, mn)
# 			out_unit_lat = (out_unit_lat - center_out)/norm_lat
	
# 			out_unit_vert = ftrns1(torch.Tensor(unit_vert), rbest, mn)
# 			out_unit_vert = (out_unit_vert - center_out)/norm_vert
	
# 			out_locs = ftrns1(torch.Tensor(locs), rbest, mn)
	
# 			loss1 = loss_func(trgt_lat, out_unit_lat)
# 			loss2 = loss_func(trgt_vert, out_unit_vert)
# 			loss3 = loss_func(0.1*trgt_center, 0.1*center_out[0,0:2]) ## Scaling loss down
	
# 			loss = loss_coef[0]*loss1 + loss_coef[1]*loss2 + loss_coef[2]*loss3 # + loss_coef[3]*loss4
# 			loss.backward()
# 			optimizer.step()
	
# 			losses.append(loss.item())
# 			losses1.append(loss1.item())
# 			losses2.append(loss2.item())
# 			losses3.append(loss3.item())
	
# 			if np.mod(i, 50) == 0:
# 				print('%d %0.8f'%(i, loss.item()))
	
# 		## Save approriate files and make extensions for directory
# 		print('\n Loss of lat and lon: %0.4f \n'%(loss_coef[0]*loss1 + loss_coef[1]*loss2))
	
# 		if (loss_coef[0]*loss1 + loss_coef[1]*loss2) < 1e-1:
# 			print('\n Finished converging \n')
# 			break
# 		else:
# 			print('\n Did not converge, restarting (%d) \n'%attempt)
	
# 	# os.rename(ext_dir + 'stations.npz', ext_dir + '%s_stations_backup.npz'%name_of_project)
	
# 	rbest = rbest.cpu().detach().numpy()
# 	mn = mn.cpu().detach().numpy()


# def extend_grid(offset, scale, deg_scale, depth_scale, extend_grids):
#     """
#     Extend a spatial grid based on randomized extensions.
    
#     Parameters:
#     - offset (numpy.ndarray): The offset values of the grid.
#     - scale (numpy.ndarray): The scale values of the grid.
#     - deg_scale (float): Degree scaling factor.
#     - depth_scale (float): Depth scaling factor.
#     - extend_grids (bool, optional): Flag to determine if grid should be extended. Default is True.
    
#     Returns:
#     - offset (numpy.ndarray): Updated offset values.
#     - scale (numpy.ndarray): Updated scale values.
#     """
    
#     if extend_grids:
#         extend1, extend2, extend3, extend4 = (np.random.rand(4) - 0.5) * deg_scale
#         extend5 = (np.random.rand() - 0.5) * depth_scale
        
#         offset[0, 0] += extend1
#         offset[0, 1] += extend2
#         scale[0, 0] += extend3
#         scale[0, 1] += extend4
#         offset[0, 2] += extend5
#     return offset, scale

# def get_offset_scale_slices(offset_x_extend, scale_x_extend):
#     """Extract slices from the offset and scale matrices."""
#     offset_slice = np.array([offset_x_extend[0, 0], offset_x_extend[0, 1], offset_x_extend[0, 2]]).reshape(1, -1)
#     scale_slice = np.array([scale_x_extend[0, 0], scale_x_extend[0, 1], scale_x_extend[0, 2]]).reshape(1, -1)
#     return offset_slice, scale_slice

# def get_grid_params(offset_slice, scale_slice, eps_extra, eps_extra_depth, scale_up):
#     """Calculate parameters for the grid."""
#     offset_x_grid = scale_up * (offset_slice - eps_extra * scale_slice)
#     offset_x_grid[0, 2] -= eps_extra_depth * scale_slice[0, 2]
    
#     scale_x_grid = scale_up * (scale_slice + 2.0 * eps_extra * scale_slice)
#     scale_x_grid[0, 2] += 2.0 * eps_extra_depth * scale_slice[0, 2]
    
#     return offset_x_grid, scale_x_grid

# def calculate_density(if_density, kernel, bandwidth, data):
#     """
#     Calculate and return kernel density if the density flag is set.
    
#     Parameters:
#     - if_density (bool): Flag indicating whether to compute density.
#     - kernel (str): Type of kernel to use for density estimation.
#     - bandwidth (float): Bandwidth for the kernel density estimation.
#     - data (numpy.ndarray): Data to compute the kernel density on.
    
#     Returns:
#     - KernelDensity (object, None): Returns KernelDensity instance if if_density is True, else returns None.
#     """
#     if if_density:
#         from sklearn.neighbors import KernelDensity
#         return KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data[:, 0:2])
#     return None

# def create_grid(using_density, m_density, weight_vector, scale_x_grid, offset_x_grid, n_cluster, ftrns1, n_steps, lr):
#     """Create a grid based on density or default method."""
#     if using_density:
#         return kmeans_packing_weight_vector_with_density(m_density, weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch=10000, n_steps=n_steps, n_sim=1, lr=lr)[0] / SCALE_UP
#     return kmeans_packing_weight_vector(weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch=10000, n_steps=n_steps, n_sim=1, lr=lr)[0] / SCALE_UP

# def assemble_grids(scale_x_extend, offset_x_extend, n_grids, n_cluster, n_steps=5000, extend_grids=False, with_density=None, density_kernel=0.15):
#     """
#     Assemble a set of spatial grids based on various parameters.
    
#     Parameters:
#     - scale_x_extend (numpy.ndarray): Extended scale values for the grid.
#     - offset_x_extend (numpy.ndarray): Extended offset values for the grid.
#     - n_grids (int): Number of grids to assemble.
#     - n_cluster (int): Number of clusters to use in the k-means algorithm.
#     - n_steps (int, optional): Number of steps for the k-means algorithm. Default is 5000.
#     - extend_grids (bool, optional): Flag to determine if grids should be extended. Default is True.
#     - with_density (numpy.ndarray, None, optional): Data to use for density calculations. Default is None.
#     - density_kernel (float, optional): Kernel bandwidth for density estimation. Default is 0.15.
    
#     Returns:
#     - x_grids (list): List of assembled grids.
#     """
    
#     m_density = calculate_density(with_density, 'gaussian', density_kernel, with_density)
#     x_grids = []
    
#     weight_vector = np.array([1.0, 1.0, depth_importance_weighting_value_for_spatial_graphs]).reshape(1, -1)
#     depth_scale = (np.diff(depth_range) * 0.02)
#     deg_scale = ((0.5 * np.diff(lat_range) + 0.5 * np.diff(lon_range)) * 0.08)

#     for i in range(n_grids):
#         offset_slice, scale_slice = get_offset_scale_slices(offset_x_extend, scale_x_extend)
#         offset_slice, scale_slice = extend_grid(offset_slice, scale_slice, deg_scale, depth_scale, extend_grids)
        
#         print(f'\nOptimize for spatial grid ({i + 1} / {n_grids})')
        
#         offset_x_grid, scale_x_grid = get_grid_params(offset_slice, scale_slice, EPS_EXTRA, EPS_EXTRA_DEPTH, SCALE_UP)
        
#         x_grid = create_grid(with_density, m_density, weight_vector, scale_x_grid, offset_x_grid, n_cluster, ftrns1, n_steps, lr=0.005)
        
#         x_grid = x_grid[np.argsort(x_grid[:, 0])]
#         x_grids.append(x_grid)

#     return x_grids




# def poisson_disk_filter(points, h, use_probablistic_acceptance = True):
#     """
#     points : (M,3) candidate points
#     h      : minimum spacing
#     """
#     cell_size = h / np.sqrt(3)
#     # cell_size = h / np.sqrt(2)
#     grid = defaultdict(list)

#     accepted = []
#     accepted_pts = []

#     h2 = h * h

#     # Shuffle candidates (important!)
#     order = np.random.permutation(len(points))

#     for idx in order:
#         p = points[idx]
#         cell = tuple(np.floor(p / cell_size).astype(int))

#         ok = True
#         # Check neighboring cells
#         for dx in (-1, 0, 1):
#             for dy in (-1, 0, 1):
#                 for dz in (-1, 0, 1):
#                     nbr_cell = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
#                     if nbr_cell in grid:
#                         q = np.array(grid[nbr_cell])
#                         if np.any(np.sum((q - p)**2, axis=1) < h2):
#                             ok = False
#                             break
#                 if not ok: break
#             if not ok: break


#         if ok == True:

#             # use_probablistic_acceptance = True
#             if use_probablistic_acceptance == True:

#                 # Compute volume fraction f
#                 r_min_ball = h / 2  # Exclusion radius
#                 # Monte Carlo approx: Sample N points in ball around p
#                 N = 30
#                 offsets = np.random.randn(N, 3)  # Gaussian, scale to uniform ball later
#                 offsets /= np.linalg.norm(offsets, axis=1)[:, None]
#                 offsets *= (np.random.rand(N)[:, None] ** (1/3)) * r_min_ball  # Uniform in ball
#                 samples = p + offsets                

#                 # Check which samples are in domain (need your ftrns funcs for r_min/max)
#                 samples_lat_lon = ftrns2_abs(samples)[:, 0:2]
#                 samples_lat_lon = np.concatenate((samples_lat_lon, np.zeros((len(samples_lat_lon), 1))), axis=1)
#                 r_samples = np.linalg.norm(samples, axis=1)
#                 r_min_vals = np.linalg.norm(ftrns1_abs(samples_lat_lon), axis=1) + depth_range[0]
#                 r_max_vals = np.linalg.norm(ftrns1_abs(samples_lat_lon), axis=1) + depth_range[1]
#                 inside = (r_samples >= r_min_vals) & (r_samples <= r_max_vals)  # Plus any angular constraints if needed
    
#                 f = np.sum(inside) / N
#                 if np.random.rand() < f**1.5:  # Or f**2 for stronger correction
#                     grid[cell].append(p)
#                     accepted_pts.append(p)

#             else:

#                 grid[cell].append(p)
#                 accepted_pts.append(p)


#         # if ok:
#         #     grid[cell].append(p)
#         #     accepted_pts.append(p)

#     return np.array(accepted_pts)



# def poisson_disk_filter_mirrored(points, h):
#     """
#     points : (M,3) candidate points
#     h      : minimum spacing
#     """
#     cell_size = h / np.sqrt(3)
#     grid = defaultdict(list)

#     accepted = []
#     accepted_pts = []

#     h2 = h * h

#     points_lat_lon = ftrns2(points)[:,0:2]
#     points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon),1))), axis = 1)
#     r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[0]
#     r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[1]
#     points_mirrored = ftrns1(ftrns2_abs(mirror_radial(ftrns1_abs(points_lat_lon), r_min_vals, r_max_vals)))
#     # points_mirrored = ftrns1(ftrns2_abs(mirror_radial(ftrns1_abs(points_lat_lon), r_min, r_max)))

#     mask_mirrored = np.concatenate((np.zeros(len(points)), np.ones(len(points_mirrored))), axis = 0)
#     points = np.concatenate((points, points_mirrored), axis = 0)

#     # Shuffle candidates (important!)
#     order = np.random.permutation(len(points))

#     for idx in order:
#         p = points[idx]
#         cell = tuple(np.floor(p / cell_size).astype(int))

#         ok = True
#         # Check neighboring cells
#         for dx in (-1, 0, 1):
#             for dy in (-1, 0, 1):
#                 for dz in (-1, 0, 1):
#                     nbr_cell = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
#                     if nbr_cell in grid:
#                         q = np.array(grid[nbr_cell])
#                         if np.any(np.sum((q - p)**2, axis=1) < h2):
#                             ok = False
#                             break
#                 if not ok: break
#             if not ok: break

#         if (ok == True):
#             grid[cell].append(p)
#         if (ok == True)*(mask_mirrored[idx] == 0):
#         	accepted_pts.append(p)

#     return np.array(accepted_pts)



# def fibonacci_sphere(n):

#     i = np.arange(n)
# 	theta0 = 2*np.pi*np.random.rand(n)
# 	theta = i*golden_angle + theta0
#     theta = np.pi*(1 + 5**0.5)*i
#     phi = np.arccos(1 - 2*(i+0.5)/n)

#     return phi, theta



# def relax_poisson1(points, h, iters=5, alpha=0.2):

#     for _ in range(iters):
#         disp = np.zeros_like(points)

#         for i, p in enumerate(points):
#             d = points - p
#             r2 = np.sum(d*d, axis=1)
#             mask = (r2 > 0) & (r2 < h*h)

#             if np.any(mask):
#                 r = np.sqrt(r2[mask])
#                 disp[i] += np.sum(
#                     (1 - r/h)[:,None] * d[mask] / r[:,None],
#                     axis=0
#                 )

#         points += alpha * disp

#         # Reproject to valid shell
#         r = np.linalg.norm(points, axis=1)

#         points_lat_lon = ftrns2_abs(points)[:,0:2]
#         points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon),1))), axis = 1)
#         r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[0]
#         r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[1]

#         r = np.clip(r, r_min_vals, r_max_vals)
#         points = points / np.linalg.norm(points, axis=1)[:,None] * r[:,None]

#     return ftrns2_abs(points)


# def relax_poisson2(points, h, iters=20, alpha=0.5):
#     for _ in range(iters):
#         disp = np.zeros_like(points)
#         for i, p in enumerate(points):
#             d = points - p
#             r2 = np.sum(d*d, axis=1)
#             mask = (r2 > 0) & (r2 < (2*h)**2)  # Wider range for softer push
#             if np.any(mask):
#                 r = np.sqrt(r2[mask])
#                 weights = (1 - r / (2*h)) ** 2  # Stronger near, zero at 2h
#                 disp[i] += np.sum(
#                     weights[:, None] * d[mask] / r[:, None],
#                     axis=0
#                 )
#         # Normalize disp magnitude if too large
#         disp_mag = np.linalg.norm(disp, axis=1)
#         mask_large = disp_mag > h
#         disp[mask_large] *= (h / disp_mag[mask_large])[:, None]
        
#         points += alpha * disp
        
#         # Reproject (your code — good for constraining)
#         r = np.linalg.norm(points, axis=1)
#         points_lat_lon = ftrns2_abs(points)[:,0:2]
#         points_lat_lon = np.concatenate((points_lat_lon, np.zeros((len(points_lat_lon),1))), axis = 1)
#         r_min_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[0]
#         r_max_vals = np.linalg.norm(ftrns1_abs(points_lat_lon), axis = 1) + depth_range[1]
#         r = np.clip(r, r_min_vals, r_max_vals)
#         points = points / np.linalg.norm(points, axis=1)[:,None] * r[:,None]  # Directions preserved, r clipped
    
#     return ftrns2_abs(points)


##### Finished ######
