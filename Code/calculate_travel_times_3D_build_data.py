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
from process_utils import * # differential_evolution_location
import glob
import sys

def lla2ecef_diff(p, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	# p = p.detach().clone().float().to(device) # why include detach here?
	pi = torch.Tensor([np.pi]).to(device)
	# p[:,0:2] = p[:,0:2]*torch.Tensor([pi/180.0, pi/180.0]).view(1,-1).to(device)
	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0]*pi/180.0)**2)
	# results:
	x = (N + p[:,2])*torch.cos(p[:,0]*pi/180.0)*torch.cos(p[:,1]*pi/180.0)
	y = (N + p[:,2])*torch.cos(p[:,0]*pi/180.0)*torch.sin(p[:,1]*pi/180.0)
	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0]*pi/180.0)

	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

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

def compute_interpolation_parallel(x1, x2, x3, Tp, Ts, X, ftrns1, num_cores = 10):

	def step_test(args):

		x1, x2, x3, tp, ts, X, ftrns1, ind = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]

		mp = RegularGridInterpolator((x1, x2, x3), tp.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		ms = RegularGridInterpolator((x1, x2, x3), ts.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		tp_val = mp(ftrns1(X))
		ts_val = ms(ftrns1(X))

		print('Finished interpolation %d'%ind)
		
		return tp_val, ts_val, ind

	n_grid = X.shape[0]
	n_sta = Tp.shape[1]
	Tp_interp = np.zeros((n_grid, n_sta))
	Ts_interp = np.zeros((n_grid, n_sta))
	assert(Tp_interp.shape[1] == Ts_interp.shape[1])

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [x1, x2, x3, Tp[:,i], Ts[:,i], X, lambda x: ftrns1(x), i] ) for i in range(n_sta))

	for i in range(n_sta):

		## Make sure to write results to correct station, based on ind
		Tp_interp[:,results[i][2]] = results[i][0].reshape(-1)
		Ts_interp[:,results[i][2]] = results[i][1].reshape(-1)

	return Tp_interp, Ts_interp

class Laplacian(MessagePassing):

	def __init__(self, lap, lap_w):
		super(Laplacian, self).__init__('sum', node_dim = 0) # consider mean
		self.lap = lap # edges
		self.lap_w = lap_w.reshape(-1,1) # edge weights

	def forward(self, x):

		## Assumes x is a square matrix
		## (what about batch?, e.g. multiple stations)

		return self.propagate(self.lap, x = x, edge_attr = self.lap_w)

	def message(self, x_j, edge_attr):

		return edge_attr*x_j


# Load configuration from YAML
config = load_config('config.yaml')

name_of_project = config['name_of_project']
num_cores = config['num_cores']

dx = config['dx']
d_deg = config['d_deg']
dx_depth = config['dx_depth']
depth_steps = config['depth_steps']

## Load travel time neural network settings
save_dense_travel_time_data = config['save_dense_travel_time_data']
train_travel_time_neural_network = config['train_travel_time_neural_network']
use_relative_1d_profile = config['use_relative_1d_profile']
# using_3D = config['using_3D']
# using_1D = config['using_1D']

if use_relative_1d_profile == True:
	print('Overwritting num cores, because using relative 1d profile option')
	num_cores = 1

depth_steps = np.arange(depth_steps['min_elevation'], depth_steps['max_elevation'] + depth_steps['elevation_step'], depth_steps['elevation_step']) # Elevation steps to compute travel times from 
## (These are reference points for stations; can be a regular grid, and each station looks up the
## travel times with respect to these values. It is discretized (nearest-neighber) rather than continous.

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())
seperator =  '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

template_ver = 1
vel_model_ver = config['vel_model_ver']
vel_model_type = config['vel_model_type']
use_topography = config['use_topography']
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

## Note: if going to loop over stations, and only compute
## travel times to a certain distance, then can clip
## spatial domain travel times are computed for during the
## building travel time step for each station

## Load velocity model
# load_model_type = 1

if vel_model_type == 1:
	z = np.load(path_to_file + '1d_velocity_model.npz')
	depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
	z.close()

	tree = cKDTree(depths.reshape(-1,1))
	ip_nearest = tree.query(ftrns2(xx)[:,2].reshape(-1,1))[1]
	Vp = vp[ip_nearest]
	Vs = vs[ip_nearest]

elif vel_model_type == 2:

	z = np.load(path_to_file + '3d_velocity_model.npz')
	x_vel, vp_vel, vs_vel = z['X'], z['Vp'], z['Vs'] ## lat, lon, depth (x_vel) and velocity values
	z.close()

	tree = cKDTree(ftrns1(x_vel)) ## Assigns the velocity values to the computation grid (xx) using nearest neighbors (e.g., the input 3D model can include any number of points, anywhere, and interpolation will fill in the values elsewhere)
	ip_nearest = tree.query(xx)[1]
	Vp = vp_vel[ip_nearest]
	Vs = vs_vel[ip_nearest]

elif vel_model_type == 3:

	z = h5py.File(path_to_file + 'Vel_models.hdf5', 'r') ## Using a series of 1d velocity models for different areas
	Depths_l, Coor_l, Vp_l, Vs_l, Radius_l = [], [], [], [], []
	keys = list(z.keys())
	n_profiles = len(list(filter(lambda x: 'Depths' in x, keys)))
	for n in range(n_profiles):
		Depths_l.append(z['Depths_%d'%n][:])
		Vp_l.append(z['Vp_%d'%n][:])
		Vs_l.append(z['Vs_%d'%n][:])
		Coor_l.append(z['Coor_%d'%n][:])
		Radius_l.append(z['Radius_%d'%n][:])
	z.close()

	interp_type = 1
	if interp_type == 1: ## Use depth coordinates with each coor point
		dist_max = np.inf*np.ones(len(xx))
		xx_ind = (-1*np.ones(len(xx))).astype('int')
		xx_depth_ind = (-1*np.ones(len(xx))).astype('int')
		Vp = np.inf*np.ones(len(xx))
		Vs = np.inf*np.ones(len(xx))
		for i in range(len(Depths_l)):
			coors_slice = Coor_l[i].repeat(len(Depths_l[i]), axis = 0)
			coors_slice = np.concatenate((coors_slice, np.tile(Depths_l[i].reshape(-1,1), (len(Coor_l[i]), 1))), axis = 1)
			tree = cKDTree(ftrns1(coors_slice))
			dist_v = tree.query(xx) # [0]/np.mean(Radius_l[i]*1000.0)
			dist, dist_ind = dist_v[0]/np.mean(Radius_l[i]*1000.0), dist_v[1]
			tree_depths = cKDTree(Depths_l[i].reshape(-1,1))
			i1 = np.where(dist < dist_max)[0]
			imatch_depth = tree_depths.query(coors_slice[dist_ind[i1],2].reshape(-1,1))[1]

			## Update nearest matching point
			dist_max[i1] = dist[i1]
			xx_ind[i1] = i
			xx_depth_ind[i1] = imatch_depth
			Vp[i1] = Vp_l[i][imatch_depth]
			Vs[i1] = Vs_l[i][imatch_depth]
			print('Finished %d'%i)

## Apply topography clipping to velocity model
if (use_topography == True)*(os.path.isfile(path_to_file + 'surface_elevation.npz') == True):

	## Load "Points" field that specifies surface elevation (columns of lat, lon, elevation (meters)). Points outside convex hull of Points will be treated as zero elevation.
	z = np.load(path_to_file + 'surface_elevation.npz')
	Points = z['Points']
	z.close()
	
	## First interpolate uniform surface over all lat-lon based on Points (fill in missing values as sea level)
	tree = cKDTree(ftrns1(Points*np.array([1.0, 1.0, 0.0]).reshape(1,-1)))
	x1_s, x2_s = np.arange(lat_range_extend[0], lat_range_extend[1] + d_deg, d_deg), np.arange(lon_range_extend[0], lon_range_extend[1] + d_deg, d_deg)
	x11_s, x12_s = np.meshgrid(x1_s, x2_s)
	xx_surface = np.concatenate((x11_s.reshape(-1,1), x12_s.reshape(-1,1)), axis = 1)
	ip_match = tree.query(ftrns1(np.concatenate((xx_surface, np.zeros((len(xx_surface),1))), axis = 1)))
	val = Points[ip_match[1],2] ## Surface elevations of regular grid
	hull = ConvexHull(Points[:,0:2])
	ioutside_hull = np.where(in_hull(xx_surface,  hull.points[hull.vertices]) == 0)[0]
	val[ioutside_hull] = 0.0 ## Setting points on regular grid far from reference points to sea level
	xx_surface = np.concatenate((xx_surface, val.reshape(-1,1)), axis = 1)
	if os.path.isfile(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project) == False:
		np.savez_compressed(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project, xx_surface = xx_surface)
		
	## Check if stations are beneath surface
	tol_elev_val = 100.0 ## Stations must be within 100 meters of being beneath surface or else assume there is an error
	tree = cKDTree(ftrns1(xx_surface))
	unit_out = ftrns1(locs + np.concatenate((np.zeros((len(locs),2)), 1.0*np.ones((len(locs),1))), axis = 1))
	dist_near = tree.query(ftrns1(locs))[0]
	dist_perturb = tree.query(unit_out)[0]
	iabove_surface = np.where(dist_perturb > dist_near)[0]
	if len(iabove_surface) > 0: assert(np.abs(locs[iabove_surface,2] - xx_surface[tree.query(ftrns1(locs))[1][iabove_surface],2]).max() < tol_elev_val)

	## Add a pertubation to elevation, check if the point is moving further away or closer to the nearest point on the surface		
	inear_surface = np.where(ftrns2(xx)[:,2] >= np.minimum((0.8*(depth_range[1] - depth_range[0]) + depth_range[0]), 0.0))[0]
	unit_out = ftrns1(ftrns2(xx[inear_surface]) + np.concatenate((np.zeros((len(inear_surface),2)), 1.0*np.ones((len(inear_surface),1))), axis = 1))
	dist_near = tree.query(xx[inear_surface])[0]
	dist_perturb = tree.query(unit_out)[0]
	iabove_surface = np.where(dist_perturb > dist_near)[0]
	
	## Set points above surface to air wave speeds (or find a way to mask)
	Vp[inear_surface[iabove_surface]] = 343.0 ## Assumed acoustic p wave speed
	Vs[inear_surface[iabove_surface]] = 343.0 ## Setting to P wave speed, so that it will reflect acoustic to S wave coupling (rather than masking)

## Using 3D domain, so must use actual station coordinates
locs_ref = np.copy(locs)
reciever_proj = ftrns1(locs_ref) # for all elevs.


hull = ConvexHull(xx)
inside_hull = in_hull(reciever_proj, hull.points[hull.vertices])
print('Num sta inside hull %d'%inside_hull.sum())
print('Num total sta %d'%len(locs_ref))
# assert(inside_hull.sum() == locs_ref.shape[0])


mp = RegularGridInterpolator((x1, x2, x3), Vp.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
ms = RegularGridInterpolator((x1, x2, x3), Vs.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
Vp_interp = mp(ftrns1(X))
Vs_interp = ms(ftrns1(X))


argvs = sys.argv
if len(argvs) == 1:
	argvs.append(0)

n_jobs = 50
n_batch = int(np.ceil(len(locs)/n_jobs))
ind_use = [np.arange(n_batch) + n_batch*i for i in range(n_jobs)]
ind_use[-1] = np.arange(ind_use[-2][-1] + 1, len(locs))
ind_use = ind_use[int(argvs[1])]

vel_model_ver = 1
use_relative_1d_profile = False

compute_reference_times = True
if compute_reference_times == True:

	print('Can sub-sample the real training data to limit memory cost for large networks')

	## Travel times
	num_cores = 1 ## This next routine is very expensive in parallel
	if num_cores == 1:

		for j in ind_use:

			if inside_hull[j] == 0:
				print('Outside hull')
				continue

			# if use_relative_1d_profile == True: ## If True, shift the profile so that the same value occurs at each stations elevation (e.g., the profile varies with the surface)
			# 	Vp = interp(ftrns2(xx)[:,2], depth_grid + locs_ref[j,2], Vp_profile) ## This may be locating some depths at incorrect positions, due to projection.
			# 	Vs = interp(ftrns2(xx)[:,2], depth_grid + locs_ref[j,2], Vs_profile)
			
			results = compute_travel_times_parallel(xx, reciever_proj[j][None,:], Vp, Vs, dx_v, x11, x12, x13, num_cores = num_cores)

			mp = RegularGridInterpolator((x1, x2, x3), results[0].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
			ms = RegularGridInterpolator((x1, x2, x3), results[1].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
			tp = mp(ftrns1(X))
			ts = ms(ftrns1(X))

			np.savez_compressed(path_to_file + 'TravelTimeData' + seperator + '%s_1d_velocity_model_station_%d_ver_%d.npz'%(name_of_project, j, vel_model_ver), X = X, xx = xx, loc = locs_ref[j], locs_ref = locs_ref, Tp_interp = tp, Ts_interp = ts, Tp = results[0], Ts = results[1], Vp = Vp, Vs = Vs, Vp_interp = Vp_interp, Vs_interp = Vs_interp, depth_grid = depth_grid)

