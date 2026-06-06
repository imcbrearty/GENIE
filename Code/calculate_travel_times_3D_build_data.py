import numpy as np
from scipy.io import loadmat
from runpy import run_path
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import multiprocessing
import skfmm
from scipy.interpolate import RegularGridInterpolator
from numpy import interp
import shutil
import pathlib
# from module import TravelTimes
from torch.autograd import Variable
from sklearn.metrics import r2_score
from utils import *
from module import TravelTimesPN
from torch.autograd import Variable
from torch_geometric.utils import get_laplacian
from scipy.optimize import differential_evolution
# from scipy.metrics import pairwise_distances as pd
from sklearn.metrics import pairwise_distances as pd
from process_utils import * # differential_evolution_location
import glob
import sys
import os

argvs = sys.argv
if len(argvs) == 1:
	argvs.append(0)

def compute_travel_times_parallel(xx, xx_r, h, h1, dx_v, x11, x12, x13, num_cores = 10):

	def step_test(args):

		yval, dx_v, h, h1, x11, x12, x13, ind = args
		print(yval.shape); print(x11.shape); print(x12.shape); print(x13.shape)

		phi_xy = (x11 - yval[0,0])**2 + (x12 - yval[0,1])**2
		phi_v = (x13 - yval[0,2])**2

		phi = np.sqrt(phi_xy + phi_v)
		# phi = phi - phi.min() - np.mean(dx_v)/5.0 ## Why include np.mean(dx_v)?
		phi = phi - phi.min() # - np.mean(dx_v)/5.0 ## Why include np.mean(dx_v)?
		assert((phi == 0).sum() == 1)

		v = np.copy(h).reshape(x11.shape) # correct?
		v1 = np.copy(h1).reshape(x11.shape) # correct?

		# t = skfmm.travel_time(phi, v, dx = [dx_v[0], dx_v[1], dx_v[2]])
		# t1 = skfmm.travel_time(phi, v1, dx = [dx_v[0], dx_v[1], dx_v[2]])

		t = skfmm.travel_time(phi, v, dx = [dx_v[1], dx_v[0], dx_v[2]])
		t1 = skfmm.travel_time(phi, v1, dx = [dx_v[1], dx_v[0], dx_v[2]])

		return t, t1, phi, ind

	tp_times, ts_times = np.nan*np.zeros((h.shape[0], xx_r.shape[0])), np.nan*np.zeros((h.shape[0], xx_r.shape[0]))

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [xx_r[i,:][None,:], dx_v, h, h1, x11, x12, x13, i] ) for i in range(xx_r.shape[0]))

	for i in range(xx_r.shape[0]):

		## Make sure to write results to correct station, based on ind
		tp_times[:,results[i][-1]] = results[i][0].reshape(-1)
		ts_times[:,results[i][-1]] = results[i][1].reshape(-1)

	return tp_times, ts_times


def grid_loss_function(x, v_min, target_error, max_regional_R, depth_profile, cpu_point_budget, C):
    dx_1, dx_2, dx_3 = x[0], x[1], x[2]
    
    # 1. Enforce strict hierarchy
    if dx_1 >= dx_2 or dx_2 >= dx_3:
        return 1e12 

    # 2. Physics boundaries
    R_max_1 = (C * (dx_2 ** 2)) / (target_error * v_min)
    R_max_2 = (C * (dx_3 ** 2)) / (target_error * v_min)

    # 3. Structural distance constraints (Adjusted for massive 3900km Lat/Lon domain)
    # Let Tier 1 go up to 5%, Tier 2 up to 35% of total domain radius
    if R_max_1 > (0.05 * max_regional_R) or R_max_1 < (0.001 * max_regional_R):
        return 1e12  
        
    if R_max_2 > (0.35 * max_regional_R) or R_max_2 <= R_max_1:
        return 1e12  

    if R_max_2 >= max_regional_R:
        return 1e12

    # --- RECTANGULAR BOX POINT ESTIMATION (MATCHING YOUR REAL GRID CODE) ---
    # Define padding rules exactly like your code
    pad_xy = 20
    pad_z = 4

    # Tier 1 Box: Extends from -R_max_1 to +R_max_1 relative to source
    t1_x_points = ((2 * R_max_1) / dx_1) + (2 * pad_xy)
    t1_y_points = ((2 * R_max_1) / dx_1) + (2 * pad_xy)
    t1_z_points = (depth_profile / dx_1) + (2 * pad_z)
    points_1 = max(1, t1_x_points) * max(1, t1_y_points) * max(1, t1_z_points)

    # Tier 2 Box
    t2_x_points = ((2 * R_max_2) / dx_2) + (2 * pad_xy)
    t2_y_points = ((2 * R_max_2) / dx_2) + (2 * pad_xy)
    t2_z_points = (depth_profile / dx_2) + (2 * pad_z)
    points_2 = max(1, t2_x_points) * max(1, t2_y_points) * max(1, t2_z_points)

    # Tier 3 Box (Covers the full regional domain size!)
    # Your max_regional_R is a radius, so total span is 2 * max_regional_R
    t3_x_points = ((2 * max_regional_R) / dx_3) + (2 * pad_xy)
    t3_y_points = ((2 * max_regional_R) / dx_3) + (2 * pad_xy)
    t3_z_points = (depth_profile / dx_3) + (2 * pad_z)
    points_3 = max(1, t3_x_points) * max(1, t3_y_points) * max(1, t3_z_points)

    # Total simulated points across all three grids
    total_points = points_1 + points_2 + points_3

    # 4. Check memory/CPU constraint
    if total_points > cpu_point_budget:
        return (1e9 * (total_points / cpu_point_budget)) # Direct penalty scaling

    # 5. Combined objective: Minimize dx (Maximize resolution)
    resolution_loss = dx_1 + dx_2 + dx_3
    budget_utilization_loss = (cpu_point_budget - total_points) / cpu_point_budget
    
    total_loss = resolution_loss + 1000.0 * budget_utilization_loss
    
    return float(np.asarray(total_loss).item())/1e7


def optimize_grid_resolutions(v_min, target_error=0.01, max_regional_R=150000.0, depth_profile=40000.0, cpu_point_budget=10_000_000):
    """
    Main execution wrapper for the grid optimization pipeline.
    """
    C = 0.25  # FMM geometric error constant

    # Bundle all structural parameters into a tuple for the DE solver
    optimization_args = (v_min, target_error, max_regional_R, depth_profile, cpu_point_budget, C)

    # Search boundaries for [dx_1, dx_2, dx_3] in meters
    bounds = [
        (50.0, 500.0),     # Fine grid limits
        (400.0, 2000.0),   # Mid grid limits
        (1500.0, 10000.0)   # Coarse grid limits
    ]

    print("--- Starting DE Multi-Res Grid Optimization ---")
    soln = differential_evolution(
        grid_loss_function, 
        bounds, 
        args=optimization_args,  # This injects the parameters safely into the loss loop
        popsize=20, 
        maxiter=500, 
        disp=True
    )
    
    # Process final results
    opt_dx1, opt_dx2, opt_dx3 = soln.x
    opt_R_max1 = (C * (opt_dx2 ** 2)) / (target_error * v_min)
    opt_R_max2 = (C * (opt_dx3 ** 2)) / (target_error * v_min)
    
    print(f"\nOptimized Grid Architecture Settings:")
    print(f"  Tier 1 (Short-Range Fine):   dx = {opt_dx1:.1f} m | Range: 0.00 to {opt_R_max1/1000:.2f} km")
    print(f"  Tier 2 (Mid-Range Medium):   dx = {opt_dx2:.1f} m | Range: {opt_R_max1/1000:.2f} to {opt_R_max2/1000:.2f} km")
    print(f"  Tier 3 (Long-Range Coarse):  dx = {opt_dx3:.1f} m | Range: {opt_R_max2/1000:.2f} to {max_regional_R/1000:.2f} km")
    
    return soln.x, (opt_R_max1, opt_R_max2)


# Load configuration from YAML
config = load_config('config.yaml')
name_of_project = config['name_of_project']
num_cores = config['num_cores']


## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())
seperator =  '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

# template_ver = 1
vel_model_type = config['vel_model_type']
use_topography = config['use_topography']
vel_model_ver = config.get('vel_model_ver', 1)

# Load region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load stations
z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
z.close()

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

## Overwrite range based on station locations and buffer
d_pad = deg_pad # 0.15
lat_range_extend = [np.minimum(lat_range_extend[0], locs[:,0].min() - d_pad), np.maximum(lat_range_extend[1], locs[:,0].max() + d_pad)]
lon_range_extend = [np.minimum(lon_range_extend[0], locs[:,1].min() - d_pad), np.maximum(lon_range_extend[1], locs[:,1].max() + d_pad)]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1

vs = np.array(config['velocity_model']['Vs']) if vel_model_type == 1 else np.load(path_to_file + '3d_velocity_model.npz')['Vs']
vs_min = np.quantile(vs.reshape(-1), 0.2)
query_proj = ftrns1(locs)

## Determine grid resolution

# lat_grid = np.linspace(lat_range_extend[0], lat_range_extend[1], 2)  
# lon_grid = np.linspace(lon_range_extend[0], lon_range_extend[1], 2)
# depth_grid = np.linspace(depth_range[0], depth_range[1], 2)
# x11, x12, x13 = np.meshgrid(lat_grid, lon_grid, depth_grid)
# X = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
# query_proj = ftrns1(X)

n_jobs = config['n_jobs']
n_batch = int(np.ceil(len(locs)/n_jobs))
ind_use = [np.arange(n_batch) + n_batch*i for i in range(n_jobs)]
if n_jobs > 1:
	ind_use[-1] = np.arange(ind_use[-2][-1] + 1, len(locs))
ind_use = ind_use[int(argvs[1])]


# n_optimal_points = config.get('target_grid_resolution', np.array([300, 300, 150])) # np.array([300, 300, 150])

n_optimal_points = np.array([300, 300, 150])
n_optimal_points = config.get('target_grid_resolution', n_optimal_points) # np.array([300, 300, 150])


## Load velocity model
if vel_model_type == 1:
	vp = np.array(config['velocity_model']['Vp'])
	vs = np.array(config['velocity_model']['Vs'])
	x_vel = np.array(config['velocity_model']['Depths'])
	vs_min = np.quantile(vs, 0.2)

else:

	z = np.load(path_to_file + '3d_velocity_model.npz')
	x_vel, vp, vs = z['X'], z['Vp'], z['Vs'] ## lat, lon, depth (x_vel) and velocity values
	z.close()
	vs_min = np.quantile(vs, 0.2)


def initilize_velocity_model(x, vp, vs, xx, dx_res, vel_type = 1):

	if vel_type == 1:

		iarg = np.argsort(x)

		dx_depth = dx_res # config.get('dx_depth', dx)
		depths_fine = np.arange(x.min(), x.max() + dx_depth/10.0, dx_depth/10.0)
		vp_fine = np.interp(depths_fine, depths[iarg], vp[iarg])
		vs_fine = np.interp(depths_fine, depths[iarg], vs[iarg])

		tree = cKDTree(depths_fine.reshape(-1,1))
		ip_nearest = tree.query(ftrns2(xx)[:,2].reshape(-1,1))[1]
		Vp = vp_fine[ip_nearest]
		Vs = vs_fine[ip_nearest]

		# tree = cKDTree(depths_fine.reshape(-1,1))
		# ip_nearest = tree.query(ftrns2(xx)[:,2].reshape(-1,1))[1]
		# Vp = vp_fine[ip_nearest]
		# Vs = vs_fine[ip_nearest]

		return Vp, Vs

	else:

		tree = cKDTree(ftrns1(x)) ## Assigns the velocity values to the computation grid (xx) using nearest neighbors (e.g., the input 3D model can include any number of points, anywhere, and interpolation will fill in the values elsewhere)
		ip_nearest = tree.query(xx)[1]
		Vp = vp[ip_nearest]
		Vs = vs[ip_nearest]

		return Vp, Vs


for sta_ind in ind_use:

	loc_proj = ftrns1(locs[sta_ind].reshape(1,-1))
	max_dist = np.linalg.norm(query_proj - loc_proj, axis = 1) ## Create grid centered on point with this radius


	target_error = 0.02
	n_optimal_points1 = np.prod(n_optimal_points)
	print('Lat range: %0.2f, %0.2f'%(lat_range_extend[0], lat_range_extend[1]))
	print('Lon range: %0.2f, %0.2f'%(lon_range_extend[0], lon_range_extend[1]))
	# v_min = np.quantile(Vs.reshape(-1), 0.2)
	optim, (opt_R_max1, opt_R_max2) = optimize_grid_resolutions(vs_min, target_error = target_error, max_regional_R = pd(query_proj).max().item(), depth_profile = 2.0*np.diff(depth_range)[0], cpu_point_budget = n_optimal_points1)
	print('Optim')
	print(optim)

	data = {}
	data['res'] = optim
	data['loc'] = locs[sta_ind].reshape(1,-1)
	data['loc_proj'] = loc_proj

	# Tp = [] # [[] for j in range(len(optim))]
	# Ts = [] # [[] for j in range(len(optim))]
	# X = [] # [[] for j in range(len(optim))]
	# X_cart = [] # [[] for j in range(len(optim))]

	for inc_res, dx_res in enumerate(optim):


		## Boundary of domain, in Cartesian coordinates
		elev = locs[:,2].max() + 1000.0
		z1 = np.array([lat_range_extend[0], lon_range_extend[0], elev])[None,:]
		z2 = np.array([lat_range_extend[0], lon_range_extend[1], elev])[None,:]
		z3 = np.array([lat_range_extend[1], lon_range_extend[1], elev])[None,:]
		z4 = np.array([lat_range_extend[1], lon_range_extend[0], elev])[None,:]
		z5 = np.array([np.mean(lat_range_extend).item(), lon_range_extend[0], elev])[None,:]
		z6 = np.array([np.mean(lat_range_extend).item(), lon_range_extend[1], elev])[None,:]
		z7 = np.array([lat_range_extend[0], np.mean(lon_range_extend).item(), elev])[None,:]
		z8 = np.array([lat_range_extend[1], np.mean(lon_range_extend).item(), elev])[None,:]
		z9 = np.array([np.mean(lat_range_extend).item(), np.mean(lon_range_extend).item(), elev])[None,:]

		# z6 = np.array([lat_range_extend[0], lon_range_extend[1], elev])[None,:]
		z = np.concatenate((z1, z2, z3, z4, z5, z6, z7, z8, z9), axis = 0)
		zz = ftrns1(z)

		n1 = n_optimal_points[0] + 1 if n_optimal_points[0] % 2 == 0 else n_optimal_points[0]
		n2 = n_optimal_points[1] + 1 if n_optimal_points[1] % 2 == 0 else n_optimal_points[1]
		n3 = n_optimal_points[2] + 1 if n_optimal_points[2] % 2 == 0 else n_optimal_points[2]

		x1 = np.linspace(0, n1 - 1, n1)*dx_res
		x2 = np.linspace(0, n2 - 1, n2)*dx_res
		x3 = np.linspace(0, n3 - 1, n3)*dx_res
		x1 = (x1 - x1.mean()) + loc_proj[0,0]
		x2 = (x2 - x2.mean()) + loc_proj[0,1]
		x3 = (x3 - x3.mean()) + loc_proj[0,2]
		x3 = x3 - x3.max() + zz[:,2].max()
		inearest = np.argmin(np.abs(x3 - loc_proj[0,2]))
		diff_val = x3[inearest] - loc_proj[0,2]
		x3 = x3 - diff_val

		x11, x12, x13 = np.meshgrid(x1, x2, x3)
		xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
		dx_v = np.array([np.diff(x1)[0], np.diff(x2)[0], np.diff(x3)[0]])
		assert(np.allclose(dx_v, dx_v.mean()))
		assert(np.allclose(np.array([x1[int(n1/2)], x2[int(n2/2)], x3[inearest]]), loc_proj[0,:])) 
		src_index = (int(n2/2) * n1 * n3) + (int(n1/2) * n3) + inearest
		assert(np.allclose(xx[src_index], loc_proj[0,:]))
		print('dx_v')
		print(dx_v)

		# moi
		Vp, Vs = initilize_velocity_model(x_vel, vp, vs, xx, dx_res, vel_type = vel_model_type)
		X = ftrns2(xx)


		## Apply topography clipping to velocity model
		if (use_topography == True)*(os.path.isfile(path_to_file + 'surface_elevation.npz') == True):

			## Load "Points" field that specifies surface elevation (columns of lat, lon, elevation (meters)). Points outside convex hull of Points will be treated as zero elevation.
			z = np.load(path_to_file + 'surface_elevation.npz')
			Points = z['Points']
			z.close()

			## Concatenate station elevations
			Points = np.concatenate((Points, locs), axis = 0)
			
			d_deg = dx_res/110e3
			## First interpolate uniform surface over all lat-lon based on Points (fill in missing values as sea level)
			tree = cKDTree(ftrns1(Points*np.array([1.0, 1.0, 0.0]).reshape(1,-1)))
			x1_s, x2_s = np.arange(lat_range_extend[0], lat_range_extend[1] + d_deg/5.0, d_deg/5.0), np.arange(lon_range_extend[0], lon_range_extend[1] + d_deg/5.0, d_deg/5.0)
			x11_s, x12_s = np.meshgrid(x1_s, x2_s)
			surface_profile = np.concatenate((x11_s.reshape(-1,1), x12_s.reshape(-1,1)), axis = 1)
			ip_match = tree.query(ftrns1(np.concatenate((surface_profile, np.zeros((len(surface_profile),1))), axis = 1)))
			val = Points[ip_match[1],2] ## Surface elevations of regular grid
			hull = ConvexHull(Points[:,0:2])
			ioutside_hull = np.where(in_hull(surface_profile,  hull.points[hull.vertices]) == 0)[0]
			val[ioutside_hull] = 0.0 ## Setting points on regular grid far from reference points to sea level
			surface_profile = np.concatenate((surface_profile, val.reshape(-1,1)), axis = 1)
			if os.path.isfile(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project) == False:
				np.savez_compressed(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project, surface_profile = surface_profile)
				
			## Check if stations are beneath surface
			tol_elev_val = 150.0 ## Stations must be within 100 meters of being beneath surface or else assume there is an error
			tree = cKDTree(ftrns1(surface_profile))
			unit_out = ftrns1(locs + np.concatenate((np.zeros((len(locs),2)), 1.0*np.ones((len(locs),1))), axis = 1))
			dist_near = tree.query(ftrns1(locs))[0]
			dist_perturb = tree.query(unit_out)[0]
			iabove_surface = np.where(dist_perturb > dist_near)[0]
			if len(iabove_surface) > 0: assert(np.abs(locs[iabove_surface,2] - surface_profile[tree.query(ftrns1(locs))[1][iabove_surface],2]).max() < tol_elev_val)

			## Add a pertubation to elevation, check if the point is moving further away or closer to the nearest point on the surface		
			inear_surface = np.where(ftrns2(xx)[:,2] >= np.minimum((0.8*(depth_range[1] - depth_range[0]) + depth_range[0]), 0.0))[0]
			unit_out = ftrns1(ftrns2(xx[inear_surface]) + np.concatenate((np.zeros((len(inear_surface),2)), 1.0*np.ones((len(inear_surface),1))), axis = 1))
			dist_near = tree.query(xx[inear_surface])[0]
			dist_perturb = tree.query(unit_out)[0]
			iabove_surface = np.where(dist_perturb > dist_near)[0]
			
			## Set points above surface to air wave speeds (or find a way to mask)
			Vp[inear_surface[iabove_surface]] = 343.0 ## Assumed acoustic p wave speed
			Vs[inear_surface[iabove_surface]] = 343.0 ## Setting to P wave speed, so that it will reflect acoustic to S wave coupling (rather than masking)


		results = compute_travel_times_parallel(xx, loc_proj, Vp, Vs, dx_v, x11, x12, x13, num_cores = num_cores)
		assert(np.allclose(results[0].min(), 0.0))
		assert(np.allclose(results[1].min(), 0.0))
		

		sample_points = True
		if sample_points == True:

			scale_factor = len(locs)/25
			n_zero_inputs = int(100000/scale_factor)
			n_per_station = int(150000/scale_factor)
			n_per_station1 = int(100000/scale_factor)

			# # p = np.zeros(locs_ref.shape[0])
			# p[sta_ind] = 1
			# isample = np.sort(np.random.choice(len(p), size = n_zero_inputs, p = p/p.sum(), replace = True))

			p = 1.0/np.maximum(results[0][:,0], 0.1)
			isample = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station, len(p)), p = p/p.sum(), replace = False))

			p = (1.0/np.maximum(results[0][:,0], 0.1))**2
			isample1 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, len(p)), p = p/p.sum(), replace = False))

			# grab_near_boundaries_samples
			p = 1.0/np.maximum(results[0].max() - results[0][:,0], 0.1)
			isample2 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, len(p)), p = p/p.sum(), replace = False))

			# grab_interior_samples
			p = 1.0*np.ones(X.shape[0]) # /np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
			isample3 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, len(p)), p = p/p.sum(), replace = False))
			isample_vald = np.random.choice(np.delete(np.arange(len(p)), np.unique(np.concatenate((isample, isample1, isample2, isample3), axis = 0)), axis = 0), size = n_per_station)
			
			isample = np.random.permutation(np.concatenate((isample, isample1, isample2, isample3), axis = 0))
			Tp_sample = results[0][isample] # np.concatenate((results[0][isample], results[0][isample1], results[0][isample2], results[0][isample3]), axis = 0)
			Ts_sample = results[1][isample] # np.concatenate((results[1][isample], results[1][isample1], results[1][isample2], results[1][isample3]), axis = 0)
			Vp_sample = Vp[isample] # np.concatenate((Vp[isample], Vp[isample1], Vp[isample2], Vp[isample3]), axis = 0)
			Vs_sample = Vs[isample] # np.concatenate((Vs[isample], Vs[isample1], Vs[isample2], Vs[isample3]), axis = 0)
			X_sample = X[isample] # np.concatenate((X[isample], X[isample1], X[isample2], X[isample3]), axis = 0)
			xx_sample = xx[isample] # np.concatenate((xx[isample], xx[isample1], xx[isample2], xx[isample3]), axis = 0)


			Tp_sample_vald = results[0][isample_vald] # , results[0][isample1], results[0][isample2], results[0][isample3]), axis = 0)
			Ts_sample_vald = results[1][isample_vald]
			Vp_sample_vald = Vp[isample_vald]
			Vs_sample_vald = Vs[isample_vald]
			X_sample_vald = X[isample_vald]
			xx_sample_vald = xx[isample_vald] # , xx[isample1], xx[isample2], xx[isample3]), axis = 0)

			Tp_boundary = np.zeros((n_zero_inputs, 1))
			Ts_boundary = np.zeros((n_zero_inputs, 1))
			Vp_boundary = Vp[src_index].repeat(n_zero_inputs, axis = 0)
			Vs_boundary = Vs[src_index].repeat(n_zero_inputs, axis = 0)
			X_boundary = X[src_index].repeat(n_zero_inputs, axis = 0)
			xx_boundary = xx[src_index].repeat(n_zero_inputs, axis = 0)

			data['Tp_%d'%inc_res] = Tp_sample
			data['Ts_%d'%inc_res] = Ts_sample
			data['X_%d'%inc_res] = X_sample
			data['X_cart_%d'%inc_res] = xx_sample
			data['Vp_%d'%inc_res] = Vp_sample
			data['Vs_%d'%inc_res] = Vs_sample
			data['Dist_%d'%inc_res] = np.linalg.norm(xx_sample - loc_proj, axis = 1)

			data['Tp_vald_%d'%inc_res] = Tp_sample_vald
			data['Ts_vald_%d'%inc_res] = Ts_sample_vald
			data['X_vald_%d'%inc_res] = X_sample_vald
			data['X_cart_vald_%d'%inc_res] = xx_sample_vald
			data['Vp_vald_%d'%inc_res] = Vp_sample_vald
			data['Vs_vald_%d'%inc_res] = Vs_sample_vald
			data['Dist_vald_%d'%inc_res] = np.linalg.norm(xx_sample_vald - loc_proj, axis = 1)

			data['Tp_boundary_%d'%inc_res] = Tp_boundary
			data['Ts_boundary_%d'%inc_res] = Ts_boundary
			data['X_boundary_%d'%inc_res] = X_boundary
			data['X_cart_boundary_%d'%inc_res] = xx_boundary
			data['Vp_boundary_%d'%inc_res] = Vp_boundary
			data['Vs_boundary_%d'%inc_res] = Vs_boundary

		else:

			data['Tp_%d'%inc_res] = results[0]
			data['Ts_%d'%inc_res] = results[1]
			data['X_%d'%inc_res] = ftrns2(xx)
			data['X_cart_%d'%inc_res] = xx
			data['Vp_%d'%inc_res] = Vp
			data['Vs_%d'%inc_res] = Vs			


	np.savez_compressed(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'TravelTimeData' + seperator + '%s_1d_velocity_model_station_%d_ver_%d.npz'%(name_of_project, sta_ind, vel_model_ver), **data)



print("All files saved successfully!")
print("✔ Script execution: Done")



