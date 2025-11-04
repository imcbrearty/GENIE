import yaml
import numpy as np
import os
import torch
from torch import optim, nn
import shutil
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

def extend_grid(offset, scale, deg_scale, depth_scale, extend_grids):
    """
    Extend a spatial grid based on randomized extensions.
    
    Parameters:
    - offset (numpy.ndarray): The offset values of the grid.
    - scale (numpy.ndarray): The scale values of the grid.
    - deg_scale (float): Degree scaling factor.
    - depth_scale (float): Depth scaling factor.
    - extend_grids (bool, optional): Flag to determine if grid should be extended. Default is True.
    
    Returns:
    - offset (numpy.ndarray): Updated offset values.
    - scale (numpy.ndarray): Updated scale values.
    """
    
    if extend_grids:
        extend1, extend2, extend3, extend4 = (np.random.rand(4) - 0.5) * deg_scale
        extend5 = (np.random.rand() - 0.5) * depth_scale
        
        offset[0, 0] += extend1
        offset[0, 1] += extend2
        scale[0, 0] += extend3
        scale[0, 1] += extend4
        offset[0, 2] += extend5
    return offset, scale

def get_offset_scale_slices(offset_x_extend, scale_x_extend):
    """Extract slices from the offset and scale matrices."""
    offset_slice = np.array([offset_x_extend[0, 0], offset_x_extend[0, 1], offset_x_extend[0, 2]]).reshape(1, -1)
    scale_slice = np.array([scale_x_extend[0, 0], scale_x_extend[0, 1], scale_x_extend[0, 2]]).reshape(1, -1)
    return offset_slice, scale_slice

def get_grid_params(offset_slice, scale_slice, eps_extra, eps_extra_depth, scale_up):
    """Calculate parameters for the grid."""
    offset_x_grid = scale_up * (offset_slice - eps_extra * scale_slice)
    offset_x_grid[0, 2] -= eps_extra_depth * scale_slice[0, 2]
    
    scale_x_grid = scale_up * (scale_slice + 2.0 * eps_extra * scale_slice)
    scale_x_grid[0, 2] += 2.0 * eps_extra_depth * scale_slice[0, 2]
    
    return offset_x_grid, scale_x_grid

def calculate_density(if_density, kernel, bandwidth, data):
    """
    Calculate and return kernel density if the density flag is set.
    
    Parameters:
    - if_density (bool): Flag indicating whether to compute density.
    - kernel (str): Type of kernel to use for density estimation.
    - bandwidth (float): Bandwidth for the kernel density estimation.
    - data (numpy.ndarray): Data to compute the kernel density on.
    
    Returns:
    - KernelDensity (object, None): Returns KernelDensity instance if if_density is True, else returns None.
    """
    if if_density:
        from sklearn.neighbors import KernelDensity
        return KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data[:, 0:2])
    return None

def create_grid(using_density, m_density, weight_vector, scale_x_grid, offset_x_grid, n_cluster, ftrns1, n_steps, lr):
    """Create a grid based on density or default method."""
    if using_density:
        return kmeans_packing_weight_vector_with_density(m_density, weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch=10000, n_steps=n_steps, n_sim=1, lr=lr)[0] / SCALE_UP
    return kmeans_packing_weight_vector(weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch=10000, n_steps=n_steps, n_sim=1, lr=lr)[0] / SCALE_UP

def assemble_grids(scale_x_extend, offset_x_extend, n_grids, n_cluster, n_steps=5000, extend_grids=False, with_density=None, density_kernel=0.15):
    """
    Assemble a set of spatial grids based on various parameters.
    
    Parameters:
    - scale_x_extend (numpy.ndarray): Extended scale values for the grid.
    - offset_x_extend (numpy.ndarray): Extended offset values for the grid.
    - n_grids (int): Number of grids to assemble.
    - n_cluster (int): Number of clusters to use in the k-means algorithm.
    - n_steps (int, optional): Number of steps for the k-means algorithm. Default is 5000.
    - extend_grids (bool, optional): Flag to determine if grids should be extended. Default is True.
    - with_density (numpy.ndarray, None, optional): Data to use for density calculations. Default is None.
    - density_kernel (float, optional): Kernel bandwidth for density estimation. Default is 0.15.
    
    Returns:
    - x_grids (list): List of assembled grids.
    """
    
    m_density = calculate_density(with_density, 'gaussian', density_kernel, with_density)
    x_grids = []
    
    weight_vector = np.array([1.0, 1.0, depth_importance_weighting_value_for_spatial_graphs]).reshape(1, -1)
    depth_scale = (np.diff(depth_range) * 0.02)
    deg_scale = ((0.5 * np.diff(lat_range) + 0.5 * np.diff(lon_range)) * 0.08)

    for i in range(n_grids):
        offset_slice, scale_slice = get_offset_scale_slices(offset_x_extend, scale_x_extend)
        offset_slice, scale_slice = extend_grid(offset_slice, scale_slice, deg_scale, depth_scale, extend_grids)
        
        print(f'\nOptimize for spatial grid ({i + 1} / {n_grids})')
        
        offset_x_grid, scale_x_grid = get_grid_params(offset_slice, scale_slice, EPS_EXTRA, EPS_EXTRA_DEPTH, SCALE_UP)
        
        x_grid = create_grid(with_density, m_density, weight_vector, scale_x_grid, offset_x_grid, n_cluster, ftrns1, n_steps, lr=0.005)
        
        x_grid = x_grid[np.argsort(x_grid[:, 0])]
        x_grids.append(x_grid)

    return x_grids

## User: Input stations and spatial region
## (must have station and region files at
## (ext_dir + 'stations.npz'), and
## (ext_dir + 'region.npz')

# Load configuration from YAML
config = load_config('config.yaml')

num_steps = config['number_of_update_steps']

with_density = config['with_density']
use_spherical = config['use_spherical']
depth_importance_weighting_value_for_spatial_graphs = config['depth_importance_weighting_value_for_spatial_graphs']
fix_nominal_depth = config['fix_nominal_depth']
use_time_shift = config['use_time_shift']

EPS_EXTRA = 0.0 # 0.1
EPS_EXTRA_DEPTH = 0.0 # 0.02
SCALE_UP = 1.0

path_to_file = str(pathlib.Path().absolute())
path_to_file += '\\' if '\\' in path_to_file else '/'

print(f'Working in the directory: {path_to_file}')

# Station file

# z = np.load(ext_dir + '%s_stations.npz'%name_of_project)
z = np.load(path_to_file + 'stations.npz')
locs, stas = z['locs'], z['stas']
z.close()

print('\n Using stations:')
print(stas)
print('\n Using locations:')
print(locs)

# Region file
# z = np.load(ext_dir + '%s_region.npz'%name_of_project)
z = np.load(path_to_file + 'region.npz', allow_pickle = True)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
years = z['years']
num_grids = config['number_of_grids']
n_spatial_nodes = config['number_of_spatial_nodes']
load_initial_files = z['load_initial_files'][0]
use_pretrained_model = z['use_pretrained_model'][0]

if use_pretrained_model == 'None':
	use_pretrained_model = None

if with_density == 'None':
	with_density = None

z.close()

shutil.copy(path_to_file + 'region.npz', path_to_file + f'{config["name_of_project"]}_region.npz')

# else, set with_density = srcs with srcs[:,0] == lat, srcs[:,1] == lon, srcs[:,2] == depth
## to preferentially focus the spatial graphs closer around reference sources. 

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

else:

	## If optimizing projection coefficients with this option, need 
	## ftrns1 and ftrns2 to accept torch Tensors instead of numpy arrays
	if use_spherical == True:

		earth_radius = 6371e3
		ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef_diff(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
		ftrns2 = lambda x, rbest, mn: ecef2lla_diff((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean
	
	else:
	
		earth_radius = 6378137.0
		ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef_diff(x) - mn).T).T # just subtract mean
		ftrns2 = lambda x, rbest, mn: ecef2lla_diff((rbest.T @ x.T).T + mn) # just subtract mean
	
	## Iterative optimization, does not converge as well

	n_attempts = 10

	unit_lat = np.array([0.01, 0.0, 0.0]).reshape(1,-1) + center_loc
	unit_vert = np.array([0.0, 0.0, 1000.0]).reshape(1,-1) + center_loc
	
	norm_lat = torch.Tensor(np.linalg.norm(np.diff(lla2ecef(np.concatenate((center_loc, unit_lat), axis = 0)), axis = 0), axis = 1))
	norm_vert = torch.Tensor(np.linalg.norm(np.diff(lla2ecef(np.concatenate((center_loc, unit_vert), axis = 0)), axis = 0), axis = 1))
	
	trgt_lat = torch.Tensor([0,1.0,0]).reshape(1,-1)
	trgt_vert = torch.Tensor([0,0,1.0]).reshape(1,-1)
	trgt_center = torch.zeros(2)
	
	loss_func = nn.MSELoss()
	
	losses = []
	losses1, losses2, losses3, losses4 = [], [], [], []
	loss_coef = [1,1,1,0]
	
	## Based on initial conditions, sometimes this converges to a projection plane that is flipped polarity.
	## E.g., "up" in the lat-lon domain is "down" in the Cartesian domain, and vice versa.
	## So try a few attempts to make sure it has the correct polarity.
	for attempt in range(n_attempts):
	
		vec = nn.Parameter(2.0*np.pi*torch.rand(3))
	
		# mn = nn.Parameter(torch.Tensor(lla2ecef(locs).mean(0, keepdims = True)))
		mn = nn.Parameter(torch.Tensor(lla2ecef(center_loc.reshape(1,-1)).mean(0, keepdims = True)))
	
		optimizer = optim.Adam([vec, mn], lr = 0.001)
	
		print('\n Optimize the projection coefficients \n')
	
		n_steps_optimize = 5000
		for i in range(n_steps_optimize):
	
			optimizer.zero_grad()
	
			rbest = rotation_matrix(vec[0], vec[1], vec[2])
	
			norm_lat = lla2ecef_diff(torch.Tensor(np.concatenate((center_loc, unit_lat), axis = 0)))
			norm_vert = lla2ecef_diff(torch.Tensor(np.concatenate((center_loc, unit_vert), axis = 0)))
			norm_lat = torch.norm(norm_lat[1] - norm_lat[0])
			norm_vert = torch.norm(norm_vert[1] - norm_vert[0])
	
			center_out = ftrns1(torch.Tensor(center_loc), rbest, mn)
	
			out_unit_lat = ftrns1(torch.Tensor(unit_lat), rbest, mn)
			out_unit_lat = (out_unit_lat - center_out)/norm_lat
	
			out_unit_vert = ftrns1(torch.Tensor(unit_vert), rbest, mn)
			out_unit_vert = (out_unit_vert - center_out)/norm_vert
	
			out_locs = ftrns1(torch.Tensor(locs), rbest, mn)
	
			loss1 = loss_func(trgt_lat, out_unit_lat)
			loss2 = loss_func(trgt_vert, out_unit_vert)
			loss3 = loss_func(0.1*trgt_center, 0.1*center_out[0,0:2]) ## Scaling loss down
	
			loss = loss_coef[0]*loss1 + loss_coef[1]*loss2 + loss_coef[2]*loss3 # + loss_coef[3]*loss4
			loss.backward()
			optimizer.step()
	
			losses.append(loss.item())
			losses1.append(loss1.item())
			losses2.append(loss2.item())
			losses3.append(loss3.item())
	
			if np.mod(i, 50) == 0:
				print('%d %0.8f'%(i, loss.item()))
	
		## Save approriate files and make extensions for directory
		print('\n Loss of lat and lon: %0.4f \n'%(loss_coef[0]*loss1 + loss_coef[1]*loss2))
	
		if (loss_coef[0]*loss1 + loss_coef[1]*loss2) < 1e-1:
			print('\n Finished converging \n')
			break
		else:
			print('\n Did not converge, restarting (%d) \n'%attempt)
	
	# os.rename(ext_dir + 'stations.npz', ext_dir + '%s_stations_backup.npz'%name_of_project)
	
	rbest = rbest.cpu().detach().numpy()
	mn = mn.cpu().detach().numpy()

if use_pretrained_model is not None:
	shutil.move(path_to_file + 'Pretrained/trained_gnn_model_step_%d_ver_%d.h5'%(20000, use_pretrained_model), path_to_file + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(config["name_of_project"], 20000, 1))
	shutil.move(path_to_file + 'Pretrained/1d_travel_time_grid_ver_%d.npz'%use_pretrained_model, path_to_file + '1D_Velocity_Models_Regional/%s_1d_travel_time_grid_ver_%d.npz'%(config["name_of_project"], 1))
	shutil.move(path_to_file + 'Pretrained/seismic_network_templates_ver_%d.npz'%use_pretrained_model, path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(use_pretrained_model, 1))

	## Find offset corrections if using one of the pre-trained models
	## Load these and apply offsets for runing "process_continuous_days.py"
	z = np.load(path_to_file + 'Pretrained/stations_ver_%d.npz'%use_pretrained_model)['locs']
	sta_loc, rbest, mn = z['locs'], z['rbest'], z['mn']
	corr1 = locs.mean(0, keepdims = True)
	corr2 = sta_loc.mean(0, keepdims = True)
	z.close()

	z = np.load(path_to_file + 'Pretrained/region_ver_%d.npz'%use_pretrained_model)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	locs = np.copy(locs) - corr1 + corr2
	shutil.copy(path_to_file + 'Pretrained/region_ver_%d.npz'%use_pretrained_model, path_to_file + f'{config["name_of_project"]}_region.npz')

else:
	corr1 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)
	corr2 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)

np.savez_compressed(path_to_file + f'{config["name_of_project"]}_stations.npz', locs = locs, stas = stas, rbest = rbest, mn = mn)

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
seperator = '\\' if '\\' in path_to_file else '/'
shutil.copy(path_to_file + '1d_velocity_model.npz', path_to_file + '1D_Velocity_Models_Regional' + seperator + f'{config["name_of_project"]}_1d_velocity_model_ver_{n_ver_velocity_model}.npz')
os.makedirs(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'TravelTimeData', exist_ok=True)


if (load_initial_files == True)*(use_pretrained_model == False):
	step_load = 20000
	ver_load = 1
	if os.path.exists(path_to_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(step_load, ver_load)):
		shutil.move(path_to_file + 'trained_gnn_model_step_%d_ver_%d.h5'%(step_load, ver_load), path_to_file + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(config["name_of_project"], step_load, ver_load))

	ver_load = 1
	if os.path.exists(path_to_file + '1d_travel_time_grid_ver_%d.npz'%ver_load):
		shutil.move(path_to_file + '1d_travel_time_grid_ver_%d.npz'%ver_load, path_to_file + '1D_Velocity_Models_Regional/%s_1d_travel_time_grid_ver_%d.npz'%(config["name_of_project"], ver_load))

	ver_load = 1
	if os.path.exists(path_to_file + 'seismic_network_templates_ver_%d.npz'%ver_load):
		shutil.move(path_to_file + 'seismic_network_templates_ver_%d.npz'%ver_load, path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(config["name_of_project"], ver_load))

## Make spatial grids

if use_spherical == True:

	earth_radius = 6371e3
	ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius)  # invert ftrns1

else:

	earth_radius = 6378137.0	
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1
	

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

skip_making_grid = False
if load_initial_files == True:
	if os.path.exists('Grids/%s_seismic_network_templates_ver_%d.npz'%(config["name_of_project"], ver_load)) == True:
		skip_making_grid = True

if (skip_making_grid == False)*(use_time_shift == False):
  
	x_grids = assemble_grids(scale_x_extend, offset_x_extend, num_grids, n_spatial_nodes, n_steps = num_steps, with_density = with_density)

	np.savez_compressed(path_to_file + 'Grids/%s_seismic_network_templates_ver_1.npz'%config["name_of_project"], x_grids = [x_grids[i] for i in range(len(x_grids))], corr1 = corr1, corr2 = corr2)



create_dense_graphs = True
if (create_dense_graphs == True)*(use_time_shift == True):

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	import yaml
	import numpy as np
	from matplotlib import pyplot as plt
	import torch
	from torch import nn, optim
	from sklearn.metrics import pairwise_distances as pd
	from scipy.signal import fftconvolve
	from scipy.spatial import cKDTree
	from scipy.stats import gamma, beta
	import time
	from torch_cluster import knn
	from torch_geometric.utils import remove_self_loops, subgraph
	from torch_geometric.utils import from_networkx
	from sklearn.neighbors import KernelDensity
	from torch_geometric.data import Data
	from torch_geometric.nn import MessagePassing
	from torch_geometric.utils import softmax
	from torch_geometric.utils import degree
	from sklearn.cluster import KMeans
	from torch.nn import Softplus
	from torch_scatter import scatter
	from numpy.matlib import repmat
	from scipy.stats import gamma
	from scipy.stats import chi2
	import pdb
	import pathlib
	import glob
	import sys

	from utils import *
	from module import *
	from process_utils import *
	# from generate_synthetic_data import generate_synthetic_data 
	## For now not using the seperate files definition of generate_synthetic_data

	## Note: you should try changing the synthetic data parameters and visualizing the 
	## results some, some values are better than others depending on region and stations

	# Load configuration from YAML
	with open('config.yaml', 'r') as file:
	    config = yaml.safe_load(file)

	# Load training configuration from YAML
	with open('train_config.yaml', 'r') as file:
	    train_config = yaml.safe_load(file)

	name_of_project = config['name_of_project']

	path_to_file = str(pathlib.Path().absolute())
	path_to_file += '\\' if '\\' in path_to_file else '/'
	seperator = '\\' if '\\' in path_to_file else '/'

	## Graph params
	k_sta_edges = config['k_sta_edges']
	k_spc_edges = config['k_spc_edges']
	k_time_edges = config['k_time_edges']
	use_physics_informed = config['use_physics_informed']
	use_phase_types = config['use_phase_types']
	use_subgraph = config['use_subgraph']
	use_sign_input = config.get('use_sign_input', False)
	use_topography = config['use_topography']
	use_station_corrections = config.get('use_station_corrections', False)
	number_of_spatial_nodes = config['number_of_spatial_nodes']
	number_of_grids = config['number_of_grids']
	if use_subgraph == True:
	    max_deg_offset = config['max_deg_offset']
	    k_nearest_pairs = config['k_nearest_pairs']	

	graph_params = [k_sta_edges, k_spc_edges, k_time_edges]

	# File versions
	template_ver = train_config['template_ver'] # spatial grid version
	vel_model_ver = train_config['vel_model_ver'] # velocity model version
	n_ver = train_config['n_ver'] # GNN save version


	device = torch.device(config['device']) ## or use cpu

	if torch.cuda.is_available() == False:
		print('No GPU available')
		device = torch.device('cpu')
		if config['device'] == 'cuda':
			print('Overwritting cuda to cpu since no gpu available')

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
	seperator = '\\' if '\\' in path_to_file else '/'
	write_training_file = path_to_file + 'GNN_TrainedModels' + seperator + name_of_project + '_'

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
		ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T
		ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius)

		ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T
		ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device)

	else:

		earth_radius = 6378137.0
		ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T
		ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)

		ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T
		ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device)



	ftrns1_center = lambda x: lla2ecef(x) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_center = lambda x: ecef2lla(x) # invert ftrns1



	def spherical_packing_nodes(n, depth_range, use_depth_scale = True, use_rotate = True, use_rand_phase = True):

		## Based on The Fibonacci Lattice
		## https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

		ftrns1_sphere_unit = lambda pos: lla2ecef(pos, a = 1.0, e = 0.0) # a = 6378137.0, e = 8.18191908426215e-2
		ftrns2_sphere_unit = lambda pos: ecef2lla(pos, a = 1.0, e = 0.0)

		# n = 30000
		# n_init = np.random.randint()
		i = np.arange(0, n).astype('float') + 0.5
		phi = np.arccos(1 - 2*i/n)
		goldenRatio = (1 + 5**0.5)/2
		theta = 2*np.pi * i / goldenRatio
		rand_phase = np.random.rand()*2*np.pi if use_rand_phase == True else 0.0
		x, y, z = np.cos(theta + rand_phase) * np.sin(phi + rand_phase), np.sin(theta + rand_phase) * np.sin(phi + rand_phase), np.cos(phi + rand_phase);
		xlat = ftrns2_sphere_unit(np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis = 1))

		def sample_radial(n):
			earth_radius = 6378137.0
			p = lambda x: 4.0*np.pi*(x >= offset_x[0,2])*(x <= (offset_x[0,2] + scale_x[0,2]))*((((earth_radius + x)/1000.0)**2))
			# p =  # trapezoid(y, x=None, dx=1.0
			x_grid = np.linspace(offset_x[0,2], offset_x[0,2] + scale_x[0,2], 1000)
			p_val = p(x_grid)
			# integrand = scipy.trapezoid(p_val, x = x_grid) # , dx=1.0
			mass = scipy.integrate.trapezoid(p_val, dx = np.diff(x_grid[0:2])) # , dx=1.0
			p_val = p_val/mass
			p_val = p_val/p_val.sum()
			# print(p_val.sum())
			# q_vals = np.quantile()
			# cum_pdf = np.cumsum(p_val)
			x = x_grid[np.random.choice(len(x_grid), size = n, p = p_val)] + np.random.rand(n)*np.diff(x_grid[0:2])

			# val = 4.0*np.pi*(x >= offset_x[0,2])*(x <= (offset_x[0,2] + scale_x[0,2]))*((x)**2)
			return x

		r_max = earth_radius + depth_range[1]
		r_min = earth_radius + depth_range[0]
		u = np.random.rand(n)
		r = ((u * (r_max**3 - r_min**3)) + r_min**3)**(1.0/3.0)

		xlat[:,2] = r - earth_radius # sample_radial(len(xlat))

		return xlat

	# import numpy as np

	## ChatGPT Fibonnaci lattice with random phase
	def fibonacci_lattice_4d(N, domain_min, domain_max, seed=None):
	    if seed is not None:
	        np.random.seed(seed)
	    
	    phi = (1 + np.sqrt(5)) / 2  # golden ratio
	    # irrational direction vector
	    alpha = np.array([1/phi**i for i in range(1,5)])
	    # random phase shift (per dimension)
	    phase = np.random.rand(4)
	    
	    i = np.arange(N)[:, None]  # shape (N,1)
	    pts = (i * alpha + phase) % 1.0  # modulo 1 (fractional part)
	    
	    # rescale to domain
	    domain_min = np.array(domain_min)
	    domain_max = np.array(domain_max)
	    pts = domain_min + pts * (domain_max - domain_min)
	    return pts

	# Example usage
	domain_min = [0, 0, 0, 0]
	domain_max = [1, 1, 1, 0.2]  # e.g., [x,y,z,t]
	points = fibonacci_lattice_4d(5000, domain_min, domain_max, seed=42)

	print(points.shape, points[:5])


	def rotate_vectors_from_z_to_c(vectors, c):
		"""
		Rotate vectors so that unit vector (0,0,1) maps to unit vector c.
		vectors: (n,3) array
		c: length-3 array, unit vector
		"""
		c = np.asarray(c, dtype=float)
		assert np.isclose(np.linalg.norm(c), 1.0)
		k = np.cross([0,0,1.0], c)
		k_norm = np.linalg.norm(k)
		if k_norm < 1e-12:
			# already aligned or anti-aligned
			if c[2] > 0:
				return vectors.copy()
			else:
				return vectors * np.array([1.0, -1.0, -1.0])  # rotate 180 deg about x (or any axis)
		k = k / k_norm
		angle = np.arccos(np.clip(np.dot([0,0,1.0], c), -1.0, 1.0))
		K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
		R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
		return vectors @ R.T

	def sobol_sphere_sampling(n, trgt_point = None, theta_max = None, scramble = True, skip = 0):
		""" ## This sampling function generated by ChatGPT for quasi uniform sampling of a section of a sphere
		Return n quasi-uniform points on the unit-sphere cap 0<=theta<=theta_max (polar axis = +z).
		theta_max_rad : angular radius in radians
		Returns: array (n,3) of unit vectors.
		"""
		sampler = qmc.Sobol(d = 2, scramble = scramble)
		# some Sobol implementations require specifying n as a power-of-two when using skip/resume;
		# SciPy's sampler.generate can produce any n.
		pts = sampler.random(n = n + skip)[skip: n+skip]  # shape (n,2)
		u = pts[:, 0]   # for phi
		v = pts[:, 1]   # for area (maps to cos(theta))

		phi = 2.0 * np.pi * u
		cos_theta_max = np.cos(theta_max*np.pi/180.0)
		# theta = (180.0/np.pi)*np.arccos(1.0 - v * (1.0 - cos_theta_max))
		# theta = 90.0 - theta ## Map points to being centered on north pole
		cos_theta = 1.0 - v * (1.0 - cos_theta_max)   # area-preserving
		sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))

	    ## x, y, z points centered on north pole. How to map to the pole of the chosen vector?
		x = sin_theta * np.cos(phi)
		y = sin_theta * np.sin(phi)
		z = cos_theta

		xx = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis = 1)
		trgt_unit_vec = ftrns1_center(trgt_point.reshape(1,-1))
		trgt_unit_vec = trgt_unit_vec/np.linalg.norm(trgt_unit_vec, axis = 1)
		xx = rotate_vectors_from_z_to_c(xx, trgt_unit_vec.reshape(-1))
		xx = earth_radius*xx/np.linalg.norm(xx, axis = 1, keepdims = True)
		xx = ftrns2_center(xx)
		xx[:,2] = 0.0

		return xx # ftrns2((r @ proj.T).T)

	def sobol_sphere_sampling_band(n, trgt_point = None, theta_min = 0.0, theta_max = None, scramble = True, skip = 0):
		""" ## This sampling function generated by ChatGPT for quasi uniform sampling of a section of a sphere
		Return n quasi-uniform points on the unit-sphere cap 0<=theta<=theta_max (polar axis = +z).
		theta_max_rad : angular radius in radians
		Returns: array (n,3) of unit vectors.
		"""
		sampler = qmc.Sobol(d = 2, scramble = scramble)
		# some Sobol implementations require specifying n as a power-of-two when using skip/resume;
		# SciPy's sampler.generate can produce any n.
		pts = sampler.random(n = n + skip)[skip: n+skip]  # shape (n,2)
		u = pts[:, 0]   # for phi
		v = pts[:, 1]   # for area (maps to cos(theta))

		phi = 2.0 * np.pi * u
		cos_theta_max = np.cos(theta_max*np.pi/180.0)
		cos_theta_min = np.cos(theta_min*np.pi/180.0)
		# theta = (180.0/np.pi)*np.arccos(1.0 - v * (1.0 - cos_theta_max))
		# theta = 90.0 - theta ## Map points to being centered on north pole
		# cos_theta = 1.0 - v * (1.0 - cos_theta_max)   # area-preserving
		cos_theta = cos_theta_min + v*(cos_theta_max - cos_theta_min)
		sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))

	    ## x, y, z points centered on north pole. How to map to the pole of the chosen vector?
		x = sin_theta * np.cos(phi)
		y = sin_theta * np.sin(phi)
		z = cos_theta

		xx = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis = 1)
		trgt_unit_vec = ftrns1_center(trgt_point.reshape(1,-1))
		trgt_unit_vec = trgt_unit_vec/np.linalg.norm(trgt_unit_vec, axis = 1)
		xx = rotate_vectors_from_z_to_c(xx, trgt_unit_vec.reshape(-1))
		xx = earth_radius*xx/np.linalg.norm(xx, axis = 1, keepdims = True)
		xx = ftrns2_center(xx)
		xx[:,2] = 0.0

		return xx # ftrns2((r @ proj.T).T)

	# if use_station_corrections == True:
	# 	n_ver_corrections = 1
	# 	path_station_corrections = path_to_file + 'Grids' + seperator + 'station_corrections_ver_%d.npz'%n_ver_corrections
	# 	if os.path.isfile(path_station_corrections) == False:
	# 		print('No station corrections available')
	# 		locs_corr, corrs = None, None
	# 	else:
	# 		z = np.load(path_station_corrections)
	# 		locs_corr, corrs = z['locs_corr'], z['corrs']
	# 		z.close()
	# else:
	# 	locs_corr, corrs = None, None


	# if config['train_travel_time_neural_network'] == False:

	# 	## Load travel times
	# 	z = np.load(path_to_file + '1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))
		
	# 	Tp = z['Tp_interp']
	# 	Ts = z['Ts_interp']
		
	# 	locs_ref = z['locs_ref']
	# 	X = z['X']
	# 	z.close()
		
	# 	x1 = np.unique(X[:,0])
	# 	x2 = np.unique(X[:,1])
	# 	x3 = np.unique(X[:,2])
	# 	assert(len(x1)*len(x2)*len(x3) == X.shape[0])
		
	# 	## Load fixed grid for velocity models
	# 	Xmin = X.min(0)
	# 	Dx = [np.diff(x1[0:2]),np.diff(x2[0:2]),np.diff(x3[0:2])]
	# 	Mn = np.array([len(x3), len(x1)*len(x3), 1]) ## Is this off by one index? E.g., np.where(np.diff(xx[:,0]) != 0)[0] isn't exactly len(x3)
	# 	N = np.array([len(x1), len(x2), len(x3)])
	# 	X0 = np.array([locs_ref[0,0], locs_ref[0,1], 0.0]).reshape(1,-1)
		
	# 	trv = interp_1D_velocity_model_to_3D_travel_times(X, locs_ref, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2, device = device) # .to(device)

	# 	z.close()

	# elif config['train_travel_time_neural_network'] == True:

	# 	n_ver_trv_time_model_load = vel_model_ver # 1
	# 	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	# 	trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	# 	trv_pairwise1 = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)

	## To implement Floyds algorithm, we must just randomly sample with rejection sampling
	## Then create veronal cells (note: all random points assigned to nearest "cluster centroid").
	## Then the distinct "volumes" of random points all assigned to point cluster centroid become the
	## new "unit", and it's centroid becomes the new centroid. Then repeat until convergence.
	## Could in theory use the random spherical sampling approach (Sobolov points), and just project to the finite domains.
	## This would help account for depth sensitivity.
	## Hence for space-time sampling we simply sample 3D Sobolov points and then time points, merge in the metric
	## to run Floyd refinement. Should produce a quasi-regular grid in 4D.

	time_range = 30.0
	scale_t = 10000.0 # 6500.0 ## 1 second is 1000 m
	scale_depth = 10.0 # 6.5

	n_samples_fraction = 150 # 300
	trgt_samples = int(n_samples_fraction*number_of_spatial_nodes)
	n_batch = int(100000)

	def inside_domain(x, l1, l2, l3):
		return np.where((x[:,0] < l1[1])*(x[:,0] > l1[0])*(x[:,1] < l2[1])*(x[:,1] > l2[0])*(x[:,2] < l3[1])*(x[:,2] > l3[0]))[0] 

	clusters_l = []
	for n in range(number_of_grids):

		inc_collect = 0
		nodes_cnt = 0
		samples_collect = []
		print('Target samples %d'%trgt_samples)
		while nodes_cnt < trgt_samples:

			samples = spherical_packing_nodes(n_batch, depth_range)
			ifind = inside_domain(samples, lat_range_extend, lon_range_extend, depth_range)
			samples_collect.append(samples[ifind])
			nodes_cnt += len(ifind)
			inc_collect += 1
			if np.mod(inc_collect, 10) == 0: print('%d %d'%(inc_collect, nodes_cnt))

		samples_collect = np.vstack(samples_collect)
		print('Samples')
		print(len(np.unique(samples_collect, axis = 0)))
		print(len(np.unique(samples_collect[:,0:2], axis = 0)))
		print(np.quantile(samples_collect[:,0], np.arange(0, 1.1, 0.1))); print('\n')
		print(np.quantile(samples_collect[:,1], np.arange(0, 1.1, 0.1))); print('\n')
		print(np.quantile(samples_collect[:,2], np.arange(0, 1.1, 0.1))); print('\n')
		# print(np.quantile(samples_collect[:,3], np.arange(0, 1.1, 0.1)))

		time_samples = np.random.uniform(-time_range, time_range, size = len(samples_collect)).reshape(-1,1)
		# samples_collect = np.concatenate((samples_collect, np.random.uniform(-time_range, time_range, size = len(samples_collect)).reshape(-1,1)), axis = 1)

		# scale_dist = np.array([1.0, 1.0, scale_depth, scale_t]).reshape(1,-1)/1000.0
		scale_dist = np.array([1.0, 1.0, scale_depth]).reshape(1,-1) # /1000.0

		inpt = np.concatenate((ftrns1(samples_collect*scale_dist), time_samples*scale_t), axis = 1)/1000.0 # *scale_dist # /1000.0

		# scale_dist = np.array([1.0, 1.0, scale_depth, scale_t]).reshape(1,-1)/1000.0

		print('Clusters')
		clusters = 1000.0*KMeans(n_clusters = number_of_spatial_nodes).fit(inpt).cluster_centers_
		clusters_x = ftrns2(clusters[:,0:3])/scale_dist
		clusters_t = clusters[:,3]/scale_t
		clusters = np.concatenate((clusters_x, clusters_t.reshape(-1,1)), axis = 1)
		print(np.quantile((clusters_x[:,0] - lat_range_extend[0])/(lat_range_extend[1] - lat_range_extend[0]), np.arange(0, 1.1, 0.1))); print('\n')
		print(np.quantile((clusters_x[:,1] - lon_range_extend[0])/(lon_range_extend[1] - lon_range_extend[0]), np.arange(0, 1.1, 0.1))); print('\n')
		print(np.quantile((clusters_x[:,2] - depth_range[0])/(depth_range[1] - depth_range[0]), np.arange(0, 1.1, 0.1))); print('\n')
		print(np.quantile((clusters_t - 0*time_range)/(1*time_range), np.arange(0, 1.1, 0.1))); print('\n')

		## Now use relaxation forces
		## Must interleave: (i). Quasi-uniform sampling (e.g., Fibonnaci or Sobolov)
		## (ii). KMeans or Floyd
		## (iii). Repuslive relaxation (with boundary constraint potential)
		## Then interleave (ii). and (iii).

		clusters_l.append(np.expand_dims(clusters, axis = 0))

	clusters_l = np.vstack(clusters_l)


	np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_templates_ver_1.npz'%name_of_project, x_grids = clusters_l, corr1 = np.zeros((1,3)), corr2 = np.zeros((1,3)))
	np.savez_compressed(path_to_file + 'Grids' + seperator + 'grid_time_shift_ver_1.npz', time_shifts = clusters_l[:,:,3])


	build_Cayley_graphs = True
	if build_Cayley_graphs == True:


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

		np.savez_compressed(path_to_file + 'Grids' + seperator + '%s_seismic_network_expanders_ver_1.npz'%name_of_project, x_grids = clusters_l, Ac = Ac, corr1 = np.zeros((1,3)), corr2 = np.zeros((1,3)))


print("All files saved successfully!")
print("✔ Script execution: Done")
