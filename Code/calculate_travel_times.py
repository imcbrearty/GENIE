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
from utils import *

def compute_travel_times_parallel(xx, xx_r, h, h1, dx_v, x11, x12, x13, num_cores = 10):

	def step_test(args):

		yval, dx_v, h, h1, x11, x12, x13, ind = args
		print(yval.shape); print(x11.shape); print(x12.shape); print(x13.shape)

		phi_xy = (x11 - yval[0,0])**2 + (x12 - yval[0,1])**2
		phi_v = (x13 - yval[0,2])**2

		phi = np.sqrt(phi_xy + phi_v)
		phi = phi - phi.min() - 100.0

		v = np.copy(h).reshape(x11.shape) # correct?
		v1 = np.copy(h1).reshape(x11.shape) # correct?

		t = skfmm.travel_time(phi, v, dx = [dx_v[0], dx_v[1], dx_v[2]])
		t1 = skfmm.travel_time(phi, v1, dx = [dx_v[0], dx_v[1], dx_v[2]])

		return t, t1, phi, ind

	tp_times, ts_times = np.nan*np.zeros((h.shape[0], xx_r.shape[0])), np.nan*np.zeros((h.shape[0], xx_r.shape[0]))

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [xx_r[i,:][None,:], dx_v, h, h1, x11, x12, x13, i] ) for i in range(xx_r.shape[0]))

	for i in range(xx_r.shape[0]):

		## Make sure to write results to correct station, based on ind
		tp_times[:,results[i][-1]] = results[i][0].reshape(-1)
		ts_times[:,results[i][-1]] = results[i][1].reshape(-1)

	return tp_times, ts_times

def compute_interpolation_parallel(x1, x2, x3, Tp, Ts, ftrns1, num_cores = 10):

	def step_test(args):

		x1, x2, x3, tp, ts, ftrns1, ind = args[0], args[1], args[2], args[3], args[4], args[5], args[6]

		mp = RegularGridInterpolator((x1, x2, x3), tp.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		ms = RegularGridInterpolator((x1, x2, x3), ts.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		tp_val = mp(ftrns1(X))
		ts_val = ms(ftrns1(X))

		return tp_val, ts_val, ind

	n_grid = Tp.shape[0]
	n_sta = Tp.shape[1]
	Tp_interp = np.zeros((n_grid, n_sta))
	Ts_interp = np.zeros((n_grid, n_sta))
	assert(Tp_interp.shape[1] == Ts_interp.shape[1])

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [x1, x2, x3, Tp[:,i], Ts[:,i], lambda x: ftrns1(x), i] ) for i in range(n_sta))

	for i in range(n_sta):

		## Make sure to write results to correct station, based on ind
		Tp_interp[:,results[i][2]] = results[i][0].reshape(-1)
		Ts_interp[:,results[i][2]] = results[i][1].reshape(-1)

	return Tp_interp, Ts_interp

# Load configuration from YAML
config = load_config('config.yaml')

name_of_project = config['name_of_project']
num_cores = config['num_cores']

dx = config['dx']
d_deg = config['d_deg']
dx_depth = config['dx_depth']
depth_steps = config['depth_steps']

depth_steps = np.arange(depth_steps['min_elevation'], depth_steps['max_elevation'] + depth_steps['elevation_step'], depth_steps['elevation_step']) # Elevation steps to compute travel times from 
## (These are reference points for stations; can be a regular grid, and each station looks up the
## travel times with respect to these values. It is discretized (nearest-neighber) rather than continous.

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())
path_to_file += '\\' if '\\' in path_to_file else '/'

template_ver = 1
vel_model_ver = 1
## Load Files

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

## Load travel times
shutil.copy(path_to_file + '1d_velocity_model.npz', path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model.npz'%name_of_project)
z = np.load(path_to_file + '1D_Velocity_Models_Regional/%s_1d_velocity_model.npz'%name_of_project)
depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
z.close()

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1

lat_grid = np.arange(lat_range_extend[0], lat_range_extend[1] + d_deg, d_deg)
lon_grid = np.arange(lon_range_extend[0], lon_range_extend[1] + d_deg, d_deg)
depth_grid = np.arange(depth_range[0], depth_range[1] + dx_depth, dx_depth)

use_interp_for_velocity_model = True
if use_interp_for_velocity_model == True:

	## Velocity model must be specified at 
	## increasing depth values
	assert(np.diff(depths).min() > 0)

	Vp_profile = interp(depth_grid, depths, vp)
	Vs_profile = interp(depth_grid, depths, vs)

else:

	tree = cKDTree(depths.reshape(-1,1))
	ip_query = tree.query(depth_grid.reshape(-1,1))[1]

	Vp_profile = vp[ip_query]
	Vs_profile = vs[ip_query]
	
replace_zero = True

x11, x12, x13 = np.meshgrid(lat_grid, lon_grid, depth_grid)
X = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)

query_proj = ftrns1(X)

## Check number of workers available. Note: should apply parallel threads for running over stations.
print('\n Total possible threads %d'%multiprocessing.cpu_count())
print(f'Actually using {num_cores} cores')

## Boundary of domain, in Cartesian coordinates
elev = locs[:,2].max() + 1000.0
z1 = np.array([lat_range_extend[0], lon_range_extend[0], elev])[None,:]
z2 = np.array([lat_range_extend[0], lon_range_extend[1], elev])[None,:]
z3 = np.array([lat_range_extend[1], lon_range_extend[1], elev])[None,:]
z4 = np.array([lat_range_extend[1], lon_range_extend[0], elev])[None,:]
z = np.concatenate((z1, z2, z3, z4), axis = 0)
zz = ftrns1(z)

ip = np.where((query_proj[:,0] >= zz[:,0].min())*(query_proj[:,0] <= zz[:,0].max())*(query_proj[:,1] >= zz[:,1].min())*(query_proj[:,1] <= zz[:,1].max()))[0]
ys = query_proj[ip,:] # not actually used for anything (non-trivial.)

## If the queries are out of bounts when calling 
x1 = np.arange(ys[:,0].min() - 20*dx, ys[:,0].max() + 20*dx, dx)
x2 = np.arange(ys[:,1].min() - 20*dx, ys[:,1].max() + 20*dx, dx)
x3 = np.arange(ys[:,2].min() - 4*dx, ys[:,2].max() + 4*dx, dx)
x11, x12, x13 = np.meshgrid(x1, x2, x3)
xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
dx_v = np.array([np.diff(x1)[0], np.diff(x2)[0], np.diff(x3)[0]])

Xmin = xx.min(0)
Dx = [np.diff(x1[0:2]),np.diff(x2[0:2]),np.diff(x3[0:2])]
Mn = np.array([len(x3), len(x1)*len(x3), 1]) ## Is this off by one index? E.g., np.where(np.diff(xx[:,0]) != 0)[0] isn't exactly len(x3)

## By using the reference point in the corner of the region, might
## induce a slight bias to the estimates as lon and lat vary widely
## (due to elliptical shape of Earth)
locs_ref = np.array([lat_range_extend[0] + 0.04, lon_range_extend[0] + 0.04, 0.0]).reshape(1,-1).repeat(len(depth_steps), axis = 0)
locs_ref[:,2] = depth_steps

reciever_proj = ftrns1(locs_ref) # for all elevs.

hull = ConvexHull(xx)
inside_hull = in_hull(reciever_proj, hull.points[hull.vertices])
assert(inside_hull.sum() == locs_ref.shape[0])

n_ver = 1

Vp = interp(ftrns2(xx)[:,2], depth_grid, Vp_profile) ## This may be locating some depths at incorrect positions, due to projection.
Vs = interp(ftrns2(xx)[:,2], depth_grid, Vs_profile)

## Travel times
if num_cores == 1:

	Tp = []
	Ts = []
	for j in range(reciever_proj.shape[0]):
		results = compute_travel_times_parallel(xx, reciever_proj[j][None,:], Vp, Vs, dx_v, x11, x12, x13, num_cores = num_cores)
		Tp.append(results[0])
		Ts.append(results[1])
	Tp = np.hstack(Tp)
	Ts = np.hstack(Ts)

else:

	results = compute_travel_times_parallel(xx, reciever_proj, Vp, Vs, dx_v, x11, x12, x13, num_cores = num_cores)
	Tp = results[0]
	Ts = results[1]

## Interpolate onto regular elliptical grid
if num_cores == 1:

	Tp_interp = np.zeros((X.shape[0], reciever_proj.shape[0]))
	Ts_interp = np.zeros((X.shape[0], reciever_proj.shape[0]))

	for j in range(reciever_proj.shape[0]): ## Need to check if it's ok to interpolate with 3D grids defined on lat,lon,depth, since depth
											## has a very different scale.
		mp = RegularGridInterpolator((x1, x2, x3), Tp[:,j].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		ms = RegularGridInterpolator((x1, x2, x3), Ts[:,j].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		Tp_interp[:,j] = mp(ftrns1(X))
		Ts_interp[:,j] = ms(ftrns1(X))

	else:

		Tp_interp, Ts_interp = compute_interpolation_parallel(x1, x2, x3, Tp, Ts, ftrns1, num_cores = num_cores)

np.savez_compressed(path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, n_ver), X = X, locs_ref = locs_ref, Tp_interp = Tp_interp, Ts_interp = Ts_interp, Vp_profile = Vp_profile, Vs_profile = Vs_profile, depth_grid = depth_grid)

print("All files saved successfully!")
print("✔ Script execution: Done")
