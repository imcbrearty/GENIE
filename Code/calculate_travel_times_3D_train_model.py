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
from module import VModel, TravelTimesPN
from torch.autograd import Variable
from torch_geometric.utils import get_laplacian
from process_utils import * # differential_evolution_location
import glob
import os

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

# template_ver = 1
vel_model_ver = config['vel_model_ver']
vel_model_type = config['vel_model_type']
use_topography = config['use_topography']
## Load Files

# Load region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load templates
# z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
# x_grids = z['x_grids']
# z.close()

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

## Load velocity model
if vel_model_type == 1:
	z = np.load(path_to_file + '1d_velocity_model.npz')
	depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
	iarg = np.argsort(depths)
	z.close()
	depths_fine = np.arange(depths.min(), depths.max() + dx_depth/10.0, dx_depth/10.0)
	vp_fine = np.interp(depths_fine, depths[iarg], vp[iarg])
	vs_fine = np.interp(depths_fine, depths[iarg], vs[iarg])

	tree = cKDTree(depths_fine.reshape(-1,1))
	ip_nearest = tree.query(ftrns2(xx)[:,2].reshape(-1,1))[1]
	Vp = vp_fine[ip_nearest]
	Vs = vs_fine[ip_nearest]

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


compute_reference_times = True
if compute_reference_times == True:
	pass

## Determine which stations have data
# ver_vel_model = 1
st_sta = glob.glob(path_to_file + 'TravelTimeData/*station*ver_%d.npz'%vel_model_ver)
iarg = np.argsort([int(st_sta[j].split('/')[-1].split('_')[5]) for j in range(len(st_sta))])
st_sta = [st_sta[j] for j in iarg]
sta_ind = np.array([int(st_sta[j].split('/')[-1].split('_')[5]) for j in range(len(st_sta))]).astype('int')


if train_travel_time_neural_network == True:

	rbest_cuda = torch.Tensor(rbest).to(device)
	mn_cuda = torch.Tensor(mn).to(device)

	## If optimizing projection coefficients with this option, need 
	## ftrns1 and ftrns2 to accept torch Tensors instead of numpy arrays

	# use_spherical = False
	if config['use_spherical'] == True:

		earth_radius = 6371e3
		ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T # just subtract mean
		ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device) # just subtract mean
	
	else:
	
		earth_radius = 6378137.0
		ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T # just subtract mean
		ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device) # just subtract mean

	# assert((using_3D + using_1D) == 1)

	X_samples, Tp_samples, Ts_samples, Locs_samples = [], [], [], []
	X_samples_vald, Tp_samples_vald, Ts_samples_vald, Locs_samples_vald = [], [], [], []
	Vp_samples, Vs_samples, Vp_samples_vald, Vs_samples_vald = [], [], [], []
	Locs_samples_inds = []

	Vp_samples_boundary, Vs_samples_boundary = [], []
	X_samples_boundary, Locs_samples_boundary = [], []

	Vp_samples_boundary_vald, Vs_samples_boundary_vald = [], []
	X_samples_boundary_vald, Locs_samples_boundary_vald = [], []


	using_3D = True
	using_1D = False
	if using_3D == True:

		print('Check if locs_ref are fixed or span 3D space')


		grab_near_station_samples = True
		print('Starting near station samples')
		if grab_near_station_samples == True:

			## Usually locs_ref ~= 25, so N * 25 = Target; 
			scale_factor = len(sta_ind)/25

			n_zero_inputs = int(100000/scale_factor)

			use_source_zero_points = False
			if use_source_zero_points == True: ## We shouldn't add zero travel time points at the "sources", as then the "slot" for station locations is corrupted in the feature space
				
				for inc, n in enumerate(sta_ind):

					p = np.zeros(locs_ref.shape[0])
					p[sta_ind] = 1

					# p1 = np.ones(locs.shape[0])
					isample = np.sort(np.random.choice(len(p), size = n_zero_inputs, p = p/p.sum(), replace = False))
					isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = 10000)
					# isample1 = np.sort(np.random.choice(len(p1), size = n_zero_inputs, p = p1/p1.sum(), replace = False))


					X_samples_boundary.append(X[isample])
					Locs_samples_boundary.append(X[isample])
					Vp_samples_boundary.append(Vp_interp[isample])
					Vs_samples_boundary.append(Vs_interp[isample])

					X_samples_boundary_vald.append(X[isample_vald])
					Locs_samples_boundary_vald.append(X[isample_vald])
					Vp_samples_boundary_vald.append(Vp_interp[isample_vald])
					Vs_samples_boundary_vald.append(Vs_interp[isample_vald])

			else:

				for inc, n in enumerate(sta_ind):

					p = np.zeros(locs_ref.shape[0])
					p[sta_ind] = 1

					# p1 = np.ones(locs.shape[0])
					isample = np.sort(np.random.choice(len(p), size = n_zero_inputs, p = p/p.sum(), replace = True))
					# isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = 10000)

					## Can't really use validation samples for fixed station boundary conditions
					isample_vald = np.sort(np.random.choice(len(p), size = n_zero_inputs, p = p/p.sum(), replace = True))


					X_samples_boundary.append(locs_ref[isample])
					Locs_samples_boundary.append(locs_ref[isample])
					Vp_samples_boundary.append(Vp_interp[isample])
					Vs_samples_boundary.append(Vs_interp[isample])

					X_samples_boundary_vald.append(locs_ref[isample_vald])
					Locs_samples_boundary_vald.append(locs_ref[isample_vald])
					Vp_samples_boundary_vald.append(Vp_interp[isample_vald])
					Vs_samples_boundary_vald.append(Vs_interp[isample_vald])



			for inc, n in enumerate(sta_ind):

				n_per_station = int(150000/scale_factor)

				z = np.load(st_sta[inc])
				tp = z['Tp_interp']
				ts = z['Ts_interp']

				z = np.load(st_sta[inc])
				p = 1.0/np.maximum(tp, 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_samples.append(X[isample])
				if compute_reference_times == True:
					Tp_samples.append(tp[isample])
					Ts_samples.append(ts[isample])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))
				Locs_samples_inds.append(n*np.ones(len(isample)))
				Vp_samples.append(Vp_interp[isample])
				Vs_samples.append(Vs_interp[isample])


				n_per_station = int(100000/scale_factor)
				# z = np.load(st_sta[inc])
				p = (1.0/np.maximum(tp, 0.1))**2
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_samples.append(X[isample])
				if compute_reference_times == True:
					Tp_samples.append(tp[isample])
					Ts_samples.append(ts[isample])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))
				Locs_samples_inds.append(n*np.ones(len(isample)))
				Vp_samples.append(Vp_interp[isample])
				Vs_samples.append(Vs_interp[isample])


				# grab_near_boundaries_samples
				n_per_station = int(100000/scale_factor)
				# z = np.load(st_sta[inc])
				p = 1.0/np.maximum(tp.max() - tp, 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_samples.append(X[isample])
				if compute_reference_times == True:
					Tp_samples.append(tp[isample])
					Ts_samples.append(ts[isample])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))
				Locs_samples_inds.append(n*np.ones(len(isample)))
				Vp_samples.append(Vp_interp[isample])
				Vs_samples.append(Vs_interp[isample])

				# grab_interior_samples
				n_per_station = int(100000/scale_factor)
				# z = np.load(st_sta[inc])
				p = 1.0*np.ones(X.shape[0]) # /np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))
				isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = 10000)

				X_samples.append(X[isample])
				if compute_reference_times == True:
					Tp_samples.append(tp[isample])
					Ts_samples.append(ts[isample])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))
				Locs_samples_inds.append(n*np.ones(len(isample)))
				Vp_samples.append(Vp_interp[isample])
				Vs_samples.append(Vs_interp[isample])

				X_samples_vald.append(X[isample_vald])
				if compute_reference_times == True:
					Tp_samples_vald.append(tp[isample_vald])
					Ts_samples_vald.append(ts[isample_vald])
				Locs_samples_vald.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample_vald), axis = 0))
				Vp_samples_vald.append(Vp_interp[isample_vald])
				Vs_samples_vald.append(Vs_interp[isample_vald])
				z.close()

				print('Finished station %d'%n)
		
	# Concatenate training dataset
	X_samples = np.vstack(X_samples)
	if compute_reference_times == True:
		Tp_samples = np.hstack(Tp_samples)
		Ts_samples = np.hstack(Ts_samples)
	Locs_samples = np.vstack(Locs_samples)
	Vp_samples = np.hstack(Vp_samples)
	Vs_samples = np.hstack(Vs_samples)
	Locs_samples_inds = np.hstack(Locs_samples_inds)
	Inds_fixed_station = []
	for n in sta_ind:
		i1 = np.where(Locs_samples_inds == n)[0]
		Inds_fixed_station.append(np.concatenate((i1.reshape(1,-1), n*np.ones((1, len(i1)))), axis = 0))
	Inds_fixed_station = torch.Tensor(np.hstack(Inds_fixed_station)).long().to(device)
	assert(Inds_fixed_station.shape[1] == len(Locs_samples_inds))
	assert(len(Locs_samples) == len(Locs_samples_inds))
	# Locs_unique = np.unique(Locs_samples, axis = 0)

	# Concatenate boundary training data
	X_samples_boundary = np.vstack(X_samples_boundary)
	Locs_samples_boundary = np.vstack(Locs_samples_boundary)
	Vp_samples_boundary = np.hstack(Vp_samples_boundary)
	Vs_samples_boundary = np.hstack(Vs_samples_boundary)
	n_dataset_boundary = len(X_samples_boundary)

	# Concatenate boundary validation data
	X_samples_boundary_vald = np.vstack(X_samples_boundary_vald)
	Locs_samples_boundary_vald = np.vstack(Locs_samples_boundary_vald)
	Vp_samples_boundary_vald = np.hstack(Vp_samples_boundary_vald)
	Vs_samples_boundary_vald = np.hstack(Vs_samples_boundary_vald)
	n_dataset_boundary_vald = len(X_samples_boundary_vald)

	# Concatenate validation dataset
	X_samples_vald = np.vstack(X_samples_vald)
	if compute_reference_times == True:
		Tp_samples_vald = np.hstack(Tp_samples_vald)
		Ts_samples_vald = np.hstack(Ts_samples_vald)
	Locs_samples_vald = np.vstack(Locs_samples_vald)
	Vp_samples_vald = np.hstack(Vp_samples_vald)
	Vs_samples_vald = np.hstack(Vs_samples_vald)


	n_dataset = len(X_samples)
	n_dataset_vald = len(X_samples_vald)

	irand_sample = np.random.choice(n_dataset, size = int(0.2*n_dataset))
	scale_val = np.round(2.0*np.linalg.norm(ftrns1(X_samples[irand_sample]) - ftrns1(Locs_samples[irand_sample]), axis = 1).max())
	if compute_reference_times == True:
		trav_val = np.round(1.25*Ts_samples.max())
	else:
		## Assume homogenous space to estimate trav_val
		default_vs = 3400.0
		trav_val = 1.25*np.linalg.norm(ftrns1(X_samples[irand_sample]) - ftrns1(Locs_samples[irand_sample]), axis = 1).max()/default_vs

	t_scale_val = trav_val*0.03
	
	v_mean = np.array([Vp_interp.mean(), Vs_interp.mean()])

	print('Using a single model for both phase types')
	# Note: training a seperate model for either phase type can be more accurate

	use_real_data = False
	# if use_real_data == True:
		
	# 	## Load dataset (from Catalog)
	# 	st_list = []
	# 	Data_obs = []
	# 	Phase_type = []
	# 	Sta_ind = []
	# 	Src_pos = []
	# 	n_events_ver = 1
	# 	yrs = np.arange(2000, 2025)
	# 	for yr in yrs:
	# 		st_list.extend(glob.glob(path_to_file + 'Catalog/%d/*continuous*ver_%d.hdf5'%(yr, n_events_ver)))
	# 	for i in range(len(st_list)):
	# 		z = h5py.File(st_list[i], 'r')
	# 		src_slice = z['srcs_trv'][:]
	# 		for j in range(len(src_slice)):
	# 			picks_p = z['Picks/%d_Picks_P'%j][:]
	# 			picks_s = z['Picks/%d_Picks_S'%j][:]
	# 			for k in range(len(picks_p)):
	# 				Data_obs.append(picks_p[k,0] - src_slice[j,3])
	# 				Phase_type.append(0)
	# 				Sta_ind.append(picks_p[k,1])
	# 				Src_pos.append(src_slice[j,0:3])

	# 			for k in range(len(picks_s)):
	# 				Data_obs.append(picks_s[k,0] - src_slice[j,3])
	# 				Phase_type.append(1)
	# 				Sta_ind.append(picks_s[k,1])
	# 				Src_pos.append(src_slice[j,0:3])
	# 		z.close()

	# 	Data_obs = np.hstack(Data_obs)
	# 	Phase_type = np.hstack(Phase_type)
	# 	Sta_ind = np.hstack(Sta_ind).astype('int')
	# 	Src_pos = np.vstack(Src_pos)
	# 	n_data = len(Src_pos)

		# Src_pos.append()


	vp_max = Vp.max() ## 9000.0
	vs_min = Vs.min() ## 1000.0
	x_pos = np.vstack([np.array([lat_range[0], lon_range[0], depth_range[0]]), np.array([lat_range[0], lon_range[1], depth_range[0]]), np.array([lat_range[1], lon_range[1], depth_range[0]]), np.array([lat_range[1], lon_range[0], depth_range[0]])])
	x_pos = np.concatenate((x_pos, np.concatenate((x_pos[:,0:2], depth_range[1]*np.ones((len(x_pos),1))), axis = 1)), axis = 0)
	max_dist = np.linalg.norm(np.expand_dims(ftrns1(x_pos), axis = 0) - np.expand_dims(ftrns1(x_pos), axis = 1), axis = 2).max()
	max_time = max_dist/vs_min

	scale_norm_factor = (max_time*vp_max/max_dist)

	norm_pos = lambda x: x/max_dist
	norm_time = lambda t: t/max_time
	norm_vel = lambda v: v/vp_max

	inorm_pos = lambda x: x*max_dist
	inorm_time = lambda t: t*max_time
	inorm_vel = lambda v: v*vp_max

	## Determine conversion factor
	# base_val = torch.norm(sta_proj - src_proj, dim = 1, keepdim = True)/self.v_mean_norm.view(1,-1)
	base_val_1 = np.linalg.norm(np.expand_dims(ftrns1(locs), axis = 0) - np.expand_dims(ftrns1(x_pos), axis = 1), axis = 2, keepdims = True)/v_mean.reshape(1,1,-1)
	base_val_2 = np.linalg.norm(np.expand_dims(norm_pos(ftrns1(locs)), axis = 0) - np.expand_dims(norm_pos(ftrns1(x_pos)), axis = 1), axis = 2, keepdims = True)/norm_vel(v_mean.reshape(1,1,-1))
	assert(np.allclose(base_val_1/base_val_2, (base_val_1/base_val_2).mean()))
	conversion_factor = (base_val_1/base_val_2).mean()
	conversion_factor = conversion_factor/max_time

	scale_params = [max_dist, max_time, vp_max, vs_min, scale_norm_factor, conversion_factor]
	m = TravelTimesPN(ftrns1_diff, ftrns2_diff, n_phases = 2, v_mean = v_mean, norm_pos = norm_pos, inorm_pos = inorm_pos, inorm_time = inorm_time, norm_vel = norm_vel, conversion_factor = conversion_factor, device = device).to(device)


	# m_p = TravelTimes(ftrns1_diff, ftrns2_diff, n_phases = 1, device = device).to(device)
	# m_s = TravelTimes(ftrns1_diff, ftrns2_diff, n_phases = 1, device = device).to(device)

	optimizer = optim.Adam(m.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 10000, gamma = 0.9)
	loss_func = nn.L1Loss()
	loss_func1 = nn.L1Loss()
	# loss_func1 = nn.BCELoss()

	n_batch = 30000
	n_steps = 150001 # 50000
	n_ver_save = vel_model_ver
	use_causual_loss = True

	# assert((using_3D + using_1D) == 1)

	
	losses = []
	losses_vald = []
	vald_steps = 10
	loss_vald = 0.0
	n_burn_in = 1000

	dz_offset = np.diff(depth_steps)[0]
	add_random_vertical_shift = False

  
	use_data = False
	n_batch_data = 5000


	use_initial_model_damping = True


	use_regularization = False
	n_seed_regularize = 30
	regularize_weight = 0.1
	k_nearest_regularize = 5 ## 15
	val_scale_self = 10.0
	torch_one_vec = torch.ones(n_batch).reshape(-1,1).to(device)
  

	losses_pde = []
	losses_data = []
	losses_pde_vald = []
	losses_data_vald = []
	loss_data = torch.Tensor([0.0]).to(device)
	loss_data_vald = torch.Tensor([0.0]).to(device)

	## Note: also don't have to sample input on regular grid
	## (especially if velocity can be sampled continuously, e.g., with the interp function)

	for i in range(n_steps):

		optimizer.zero_grad()

		isample = np.random.randint(0, high = n_dataset, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos = torch.Tensor(X_samples[isample]).to(device)

		if compute_reference_times == True: ## Can include a data loss
			travel_times_p = Tp_samples[isample] #
			travel_times_s = Ts_samples[isample] # = sample_inputs_unweighted(n_batch)
			trgt_data = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

		pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
		# pred_perturb = pred_perturb1 + pred_perturb2

		vel_pred = m.vmodel(src_pos_cart, src_embed)

		if use_regularization == True:

			# ind_slice = Inds_fixed_station[:,isample]
			irand = np.random.choice(n_batch, size = n_seed_regularize)

			# inds_slice = subgraph(torch.Tensor(isample[irand]).long().to(device), ind_slice)
			sig_weight = 0.01*max_dist/1000.0
			sta_slice = torch.cat((val_scale_self*ftrns1_diff(sta_pos[irand])/1000.0, ftrns1_diff(src_pos[irand])/1000.0), dim = 1)
			k_nearest = knn(torch.cat((val_scale_self*ftrns1_diff(sta_pos)/1000.0, ftrns1_diff(src_pos)/1000.0), dim = 1), sta_slice, k = k_nearest_regularize).flip(0).contiguous()
			weights = torch.exp(-0.5*torch.norm(ftrns1_diff(src_pos[irand][k_nearest[1]])/1000.0 - ftrns1_diff(src_pos[k_nearest[0]])/1000.0, dim = 1)/(sig_weight**2))
			unique_ind = np.sort(np.unique(np.concatenate((k_nearest[0].cpu().detach().numpy(), irand), axis = 0)))
			perm_ind_vec = -1*np.ones(n_batch)
			perm_ind_vec[unique_ind] = np.arange(len(unique_ind))
			perm_ind_vec = torch.Tensor(perm_ind_vec).long().to(device)
			slice_data = (pred_base + pred_perturb)[unique_ind]
			k_nearest[0] = perm_ind_vec[k_nearest[0]]
			k_nearest[1] = perm_ind_vec[torch.Tensor(irand).long().to(device)[k_nearest[1]]]
			assert(k_nearest.min() > -1)
			# sta_pos[irand][k_nearest[1]] - sta_pos[k_nearest[0]] are mostly 0.0

			lap_spc = get_laplacian(k_nearest, normalization = 'sym', edge_weight = weights)
			Lap_spc = Laplacian(lap_spc[0], lap_spc[1])
			pred_regularize = Lap_spc(slice_data)
			loss_regularize = loss_func(pred_regularize, torch.zeros(pred_regularize.shape).to(device))

		if use_data == True:

			irand = np.random.choice(n_data, size = n_batch_data)
			obs_arv = norm_time(torch.Tensor(Data_obs[irand]).to(device))
			sta_pos_data = torch.Tensor(locs[Sta_ind[irand]]).to(device)
			src_pos_data = torch.Tensor(Src_pos[irand]).to(device)

			pred_base_data, pred_perturb_data, src_pos_cart_data, src_embed_data = m(sta_pos_data, src_pos_data, method = 'direct', train = True)
			pred_data = (pred_base_data + pred_perturb_data)[torch.arange(n_batch_data).long().to(device), torch.Tensor(Phase_type[irand]).long().to(device)]

			loss_data = loss_func(pred_data, obs_arv)


		scale_loss = 1.0
		# trgt = norm_vel(torch.Tensor(np.concatenate((Vp_samples[isample].reshape(-1,1), Vs_samples[isample].reshape(-1,1)), axis = 1))).to(device)
		# trgt = torch.Tensor(np.concatenate((Vp_samples[isample].reshape(-1,1), Vs_samples[isample].reshape(-1,1)), axis = 1)).to(device)
		trgt = vel_pred

		## Compute the predicted PDE value
		scale_val_grad = 1.0
		grad_base_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_base_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		grad_perturb_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_perturb_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		pred_val_p = (grad_base_p*grad_base_p).sum(1) + (grad_perturb_p*grad_perturb_p).sum(1) + 2.0*(grad_base_p*grad_perturb_p).sum(1)
		pred_val_s = (grad_base_s*grad_base_s).sum(1) + (grad_perturb_s*grad_perturb_s).sum(1) + 2.0*(grad_base_s*grad_perturb_s).sum(1)

		## Convert to predict velocity, and scale by pde factor
		pred_val_p = (1.0/scale_norm_factor)*(pred_val_p**(-0.5))
		pred_val_s = (1.0/scale_norm_factor)*(pred_val_s**(-0.5))

		loss_weight = torch.exp(-torch.abs(inorm_time(pred_base[:,0]))/t_scale_val)
		loss_weight = loss_weight/loss_weight.sum()

		# loss_pde_p = 0.5*loss_func(scale_loss*pred_val_p**(-0.5), scale_loss*trgt[:,0]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_p**(-0.5), loss_weight*scale_loss*trgt[:,0])
		# loss_pde_s = 0.5*loss_func(scale_loss*pred_val_s**(-0.5), scale_loss*trgt[:,1]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_s**(-0.5), loss_weight*scale_loss*trgt[:,1])

		loss_pde_p = 0.5*loss_func(scale_loss*pred_val_p, scale_loss*trgt[:,0]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_p, loss_weight*scale_loss*trgt[:,0])
		loss_pde_s = 0.5*loss_func(scale_loss*pred_val_s, scale_loss*trgt[:,1]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_s, loss_weight*scale_loss*trgt[:,1])

		# loss_pde_s = 0.5*loss_func(scale_loss*pred_val_s, scale_loss*trgt[:,1]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_s, loss_weight*scale_loss*trgt[:,1])

		## Boundary loss

		isample_boundary = np.random.randint(0, high = n_dataset_boundary, size = n_batch)
		sta_pos_boundary = torch.Tensor(Locs_samples_boundary[isample_boundary]).to(device)
		src_pos_boundary = torch.Tensor(X_samples_boundary[isample_boundary]).to(device)
		pred_base_boundary, pred_perturb_boundary, src_pos_boundary_cart, src_embed_boundary = m(sta_pos_boundary, src_pos_boundary, method = 'direct', train = True)
		# pred_perturb_boundary = pred_perturb_boundary1 + pred_perturb_boundary2
		assert(pred_base_boundary.max().item() < 1e-2)
		loss_boundary = loss_func(pred_perturb_boundary, torch.zeros(pred_perturb_boundary.shape).to(device))

		if i > n_burn_in:

			loss = 0.25*loss_pde_p + 0.25*loss_pde_s + 0.5*loss_boundary

		else:

			## Add loss of "null" prediction

			loss = loss_boundary ## Initialize model

		if compute_reference_times == True:

			trgt_data = torch.Tensor(norm_time(np.concatenate((Tp_samples[isample].reshape(-1,1), Ts_samples[isample].reshape(-1,1)), axis = 1))).to(device)
			pred_data = pred_base + pred_perturb
			# loss_data = loss_func(pred_data, trgt_data)
			loss_data = loss_func1(pred_data, trgt_data)

			loss = 0.5*loss + 0.5*loss_data

		if use_causual_loss == True:

			sign_offset = 1.0*(torch.sign(pred_base + pred_perturb) != 1)

			loss_causual = loss_func(torch.zeros(pred_base.shape).to(device), sign_offset*(pred_base + pred_perturb))

			loss = 0.99*loss + 0.01*loss_causual

		if use_regularization == True:

			loss = 0.95*loss + 0.05*loss_regularize

		if use_data == True:

			loss = 0.9*loss + 0.1*loss_data

		if use_initial_model_damping == True:

			trgt_vel = norm_vel(torch.Tensor(np.concatenate((Vp_samples[isample].reshape(-1,1), Vs_samples[isample].reshape(-1,1)), axis = 1))).to(device)

			loss_initial = loss_func(trgt_vel, vel_pred)

			loss = 0.85*loss + 0.15*loss_initial

		# loss = 0.5*loss_func(out/m.tscale, trgt/m.tscale) + 0.5*loss_func(loss_weight.reshape(-1,1)*out/m.tscale, loss_weight.reshape(-1,1)*trgt/m.tscale)

		loss.backward()
		optimizer.step()
		scheduler.step()
		losses.append(loss.item())
		losses_pde.append(loss_pde_p.item() + loss_pde_s.item())
		losses_data.append(loss_data.item())

		if np.mod(i, vald_steps) == 0:

			isample = np.random.randint(0, high = n_dataset_vald, size = n_batch)

			sta_pos = torch.Tensor(Locs_samples_vald[isample]).to(device)
			src_pos = torch.Tensor(X_samples_vald[isample]).to(device)


			if compute_reference_times == True: ## Can include a data loss
				travel_times_p = Tp_samples_vald[isample] #
				travel_times_s = Ts_samples_vald[isample] # = sample_inputs_unweighted(n_batch)
				trgt_data = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

			pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
			# pred_perturb = pred_perturb1 + pred_perturb2

			vel_pred = m.vmodel(src_pos_cart, src_embed)

			if use_regularization == True:

				# ind_slice = Inds_fixed_station[:,isample]
				irand = np.random.choice(n_batch, size = n_seed_regularize)
				# locs_inds = torch.Tensor(Locs_samples_inds[isample[irand]]).long().to(device)

				# inds_slice = subgraph(torch.Tensor(isample[irand]).long().to(device), ind_slice)
				sig_weight = 0.01*max_dist/1000.0
				sta_slice = torch.cat((val_scale_self*ftrns1_diff(sta_pos[irand])/1000.0, ftrns1_diff(src_pos[irand])/1000.0), dim = 1)
				k_nearest = knn(torch.cat((val_scale_self*ftrns1_diff(sta_pos)/1000.0, ftrns1_diff(src_pos)/1000.0), dim = 1), sta_slice, k = k_nearest_regularize).flip(0).contiguous()
				weights = torch.exp(-0.5*torch.norm(ftrns1_diff(src_pos[irand][k_nearest[1]])/1000.0 - ftrns1_diff(src_pos[k_nearest[0]])/1000.0, dim = 1)/(sig_weight**2))
				unique_ind = np.sort(np.unique(np.concatenate((k_nearest[0].cpu().detach().numpy(), irand), axis = 0)))
				perm_ind_vec = -1*np.ones(n_batch)
				perm_ind_vec[unique_ind] = np.arange(len(unique_ind))
				perm_ind_vec = torch.Tensor(perm_ind_vec).long().to(device)
				slice_data = (pred_base + pred_perturb)[unique_ind]
				k_nearest[0] = perm_ind_vec[k_nearest[0]]
				k_nearest[1] = perm_ind_vec[torch.Tensor(irand).long().to(device)[k_nearest[1]]]
				assert(k_nearest.min() > -1)
				# sta_pos[irand][k_nearest[1]] - sta_pos[k_nearest[0]] are mostly 0.0

				lap_spc = get_laplacian(k_nearest, normalization = 'rw', edge_weight = weights)
				Lap_spc = Laplacian(lap_spc[0], lap_spc[1])
				pred_regularize = Lap_spc(slice_data)
				loss_regularize = loss_func(pred_regularize, torch.zeros(pred_regularize.shape).to(device))

			if use_data == True:

				irand = np.random.choice(n_data, size = n_batch_data)
				obs_arv = norm_time(torch.Tensor(Data_obs[irand]).to(device))
				sta_pos_data = torch.Tensor(locs[Sta_ind[irand]]).to(device)
				src_pos_data = torch.Tensor(Src_pos[irand]).to(device)

				pred_base_data, pred_perturb_data, src_pos_cart_data, src_embed_data = m(sta_pos_data, src_pos_data, method = 'direct', train = True)
				pred_data = (pred_base_data + pred_perturb_data)[torch.arange(n_batch_data).long().to(device), torch.Tensor(Phase_type[irand]).long().to(device)]

				loss_data = loss_func(pred_data, obs_arv)

			scale_loss = 1.0
			trgt = vel_pred # norm_vel(torch.Tensor(np.concatenate((Vp_samples_vald[isample].reshape(-1,1), Vs_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)

			## Compute the predicted PDE value
			scale_val_grad = 1.0
			grad_base_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			grad_base_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

			grad_perturb_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
			grad_perturb_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

			pred_val_p = (grad_base_p*grad_base_p).sum(1) + (grad_perturb_p*grad_perturb_p).sum(1) + 2.0*(grad_base_p*grad_perturb_p).sum(1)
			pred_val_s = (grad_base_s*grad_base_s).sum(1) + (grad_perturb_s*grad_perturb_s).sum(1) + 2.0*(grad_base_s*grad_perturb_s).sum(1)

			## Convert to predict velocity, and scale by pde factor
			pred_val_p = (1.0/scale_norm_factor)*(pred_val_p**(-0.5))
			pred_val_s = (1.0/scale_norm_factor)*(pred_val_s**(-0.5))

			loss_weight = torch.exp(-torch.abs(inorm_time(pred_base[:,0]))/t_scale_val)
			loss_weight = loss_weight/loss_weight.sum()

			# loss_pde_p = 0.5*loss_func(scale_loss*pred_val_p**(-0.5), scale_loss*trgt[:,0]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_p**(-0.5), loss_weight*scale_loss*trgt[:,0])
			# loss_pde_s = 0.5*loss_func(scale_loss*pred_val_s**(-0.5), scale_loss*trgt[:,1]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_s**(-0.5), loss_weight*scale_loss*trgt[:,1])

			loss_pde_p = 0.5*loss_func(scale_loss*pred_val_p, scale_loss*trgt[:,0]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_p, loss_weight*scale_loss*trgt[:,0])
			loss_pde_s = 0.5*loss_func(scale_loss*pred_val_s, scale_loss*trgt[:,1]) + 0.5*loss_func(loss_weight*scale_loss*pred_val_s, loss_weight*scale_loss*trgt[:,1])

			## Boundary loss

			isample_boundary = np.random.randint(0, high = n_dataset_boundary_vald, size = n_batch)
			sta_pos_boundary = torch.Tensor(Locs_samples_boundary_vald[isample_boundary]).to(device)
			src_pos_boundary = torch.Tensor(X_samples_boundary_vald[isample_boundary]).to(device)
			pred_base_boundary, pred_perturb_boundary, src_pos_boundary_cart, src_embed_boundary = m(sta_pos_boundary, src_pos_boundary, method = 'direct', train = True)
			# pred_perturb_boundary = pred_perturb_boundary1 + pred_perturb_boundary2
			assert(pred_base_boundary.max().item() < 1e-2)
			loss_boundary = loss_func(pred_perturb_boundary, torch.zeros(pred_perturb_boundary.shape).to(device))

			if i > n_burn_in:

				loss_vald = 0.25*loss_pde_p + 0.25*loss_pde_s + 0.5*loss_boundary

			else:

				## Add loss of "null" prediction

				# loss1 = loss_func(pred_perturb2, torch.zeros(pred_perturb2.shape).to(device))
				# loss2 = loss_func(pred_perturb2, torch.zeros(pred_perturb2.shape).to(device))

				loss_vald = loss_boundary ## Initialize model

				# loss = loss_boundary ## Initialize model

			if compute_reference_times == True:

				trgt_data = torch.Tensor(norm_time(np.concatenate((Tp_samples_vald[isample].reshape(-1,1), Ts_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)
				pred_data = pred_base + pred_perturb
				loss_data = loss_func1(pred_data, trgt_data)

				loss_vald = 0.5*loss_vald + 0.5*loss_data

			if use_causual_loss == True:

				sign_offset = 1.0*(torch.sign(pred_base + pred_perturb) != 1)

				loss_causual = loss_func(torch.zeros(pred_base.shape).to(device), sign_offset*(pred_base + pred_perturb))

				loss_vald = 0.99*loss_vald + 0.01*loss_causual

			if use_regularization == True:

				loss_vald = 0.95*loss_vald + 0.05*loss_regularize

			if use_data == True:

				loss_vald = 0.9*loss_vald + 0.1*loss_data

			if use_initial_model_damping == True:

				trgt_vel = norm_vel(torch.Tensor(np.concatenate((Vp_samples_vald[isample].reshape(-1,1), Vs_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)

				loss_initial = loss_func(trgt_vel, vel_pred)

				loss_vald = 0.85*loss_vald + 0.15*loss_initial


			print('%d %0.8f %0.8f'%(i, loss.item(), loss_vald.item()))
			losses_vald.append(loss_vald.item())
			losses_pde_vald.append(loss_pde_p.item() + loss_pde_s.item())
			losses_data_vald.append(loss_data.item())

	phase = 'p_s'
	path_save = path_to_file + '1D_Velocity_Models_Regional' + seperator

	with torch.no_grad():

		## Mesure loss on training and validation and make residual plot

		isample = np.random.randint(0, high = n_dataset, size = n_batch)
		sta_pos1 = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos1 = torch.Tensor(X_samples[isample]).to(device)
		if compute_reference_times == True:
			travel_times_p = Tp_samples[isample] #
			travel_times_s = Ts_samples[isample] # = sample_inputs_unweighted(n_batch)
			trgt1 = np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)
		else:
			trgt1 = []
		out1 = m(sta_pos1, src_pos1, method = 'direct').cpu().detach().numpy()


		isample = np.random.randint(0, high = n_dataset_vald, size = n_batch)
		sta_pos2 = torch.Tensor(Locs_samples_vald[isample]).to(device)
		src_pos2 = torch.Tensor(X_samples_vald[isample]).to(device)
		if compute_reference_times == True:
			travel_times_p = Tp_samples_vald[isample] #
			travel_times_s = Ts_samples_vald[isample] # = sample_inputs_unweighted(n_batch)
			trgt2 = np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)
		else:
			trgt2 = []
		out2 = m(sta_pos2, src_pos2, method = 'direct').cpu().detach().numpy()

		sta_pos1 = sta_pos1.cpu().detach().numpy()
		sta_pos2 = sta_pos2.cpu().detach().numpy()
		src_pos1 = src_pos1.cpu().detach().numpy()
		src_pos2 = src_pos2.cpu().detach().numpy()

	check_correlation = True
	if check_correlation == True:

		## Training
		isample = np.random.randint(0, high = n_dataset, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos = torch.Tensor(X_samples[isample]).to(device)

		if compute_reference_times == True: ## Can include a data loss
			travel_times_p = Tp_samples[isample] #
			travel_times_s = Ts_samples[isample] # = sample_inputs_unweighted(n_batch)
			trgt_data = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

		pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
		# pred_perturb = pred_perturb1 + pred_perturb2

		vel_pred = m.vmodel(src_pos_cart, src_embed)

		scale_loss = 1.0
		trgt = norm_vel(torch.Tensor(np.concatenate((Vp_samples[isample].reshape(-1,1), Vs_samples[isample].reshape(-1,1)), axis = 1))).to(device)

		## Compute the predicted PDE value
		scale_val_grad = 1.0
		grad_base_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_base_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		grad_perturb_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_perturb_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		pred_val_p = (grad_base_p*grad_base_p).sum(1) + (grad_perturb_p*grad_perturb_p).sum(1) + 2.0*(grad_base_p*grad_perturb_p).sum(1)
		pred_val_s = (grad_base_s*grad_base_s).sum(1) + (grad_perturb_s*grad_perturb_s).sum(1) + 2.0*(grad_base_s*grad_perturb_s).sum(1)

		## Convert to predict velocity, and scale by pde factor
		pred_val_p = (1.0/scale_norm_factor)*(pred_val_p**(-0.5))
		pred_val_s = (1.0/scale_norm_factor)*(pred_val_s**(-0.5))
		r2_vp = r2_score(trgt[:,0].cpu().detach().numpy(), pred_val_p.cpu().detach().numpy()) 
		r2_vs = r2_score(trgt[:,1].cpu().detach().numpy(), pred_val_s.cpu().detach().numpy()) 

		## Validation
		isample = np.random.randint(0, high = n_dataset_vald, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples_vald[isample]).to(device)
		src_pos = torch.Tensor(X_samples_vald[isample]).to(device)

		if compute_reference_times == True: ## Can include a data loss
			travel_times_p = Tp_samples_vald[isample] #
			travel_times_s = Ts_samples_vald[isample] # = sample_inputs_unweighted(n_batch)
			trgt_data = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

		pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
		# pred_perturb = pred_perturb1 + pred_perturb2

		vel_pred = m.vmodel(src_pos_cart, src_embed)

		scale_loss = 1.0
		trgt = norm_vel(torch.Tensor(np.concatenate((Vp_samples_vald[isample].reshape(-1,1), Vs_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)

		## Compute the predicted PDE value
		scale_val_grad = 1.0
		grad_base_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_base_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		grad_perturb_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_perturb_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		pred_val_p = (grad_base_p*grad_base_p).sum(1) + (grad_perturb_p*grad_perturb_p).sum(1) + 2.0*(grad_base_p*grad_perturb_p).sum(1)
		pred_val_s = (grad_base_s*grad_base_s).sum(1) + (grad_perturb_s*grad_perturb_s).sum(1) + 2.0*(grad_base_s*grad_perturb_s).sum(1)

		## Convert to predict velocity, and scale by pde factor
		pred_val_p = (1.0/scale_norm_factor)*(pred_val_p**(-0.5))
		pred_val_s = (1.0/scale_norm_factor)*(pred_val_s**(-0.5))
		r2_vp_vald = r2_score(trgt[:,0].cpu().detach().numpy(), pred_val_p.cpu().detach().numpy()) 
		r2_vs_vald = r2_score(trgt[:,1].cpu().detach().numpy(), pred_val_s.cpu().detach().numpy()) 

	else:

		r2_vp, r2_vs, r2_vp_vald, r2_vs_vald = 0.0, 0.0, 0.0, 0.0

	print('Velocity correlations are: \n')
	print('%.2f [vp], %0.2f [vs], %0.2f [vp vald], %0.2f [vs vald]'%(r2_vp, r2_vs, r2_vp_vald, r2_vs_vald))

	r_vals = [r2_vp, r2_vs, r2_vp_vald, r2_vs_vald]


	check_correlation_self = True
	if check_correlation_self == True:

		## Training
		isample = np.random.randint(0, high = n_dataset, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos = torch.Tensor(X_samples[isample]).to(device)

		if compute_reference_times == True: ## Can include a data loss
			travel_times_p = Tp_samples[isample] #
			travel_times_s = Ts_samples[isample] # = sample_inputs_unweighted(n_batch)
			trgt_data = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

		pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
		# pred_perturb = pred_perturb1 + pred_perturb2

		vel_pred = m.vmodel(src_pos_cart, src_embed)

		scale_loss = 1.0
		trgt = vel_pred # norm_vel(torch.Tensor(np.concatenate((Vp_samples[isample].reshape(-1,1), Vs_samples[isample].reshape(-1,1)), axis = 1))).to(device)

		## Compute the predicted PDE value
		scale_val_grad = 1.0
		grad_base_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_base_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		grad_perturb_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_perturb_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		pred_val_p = (grad_base_p*grad_base_p).sum(1) + (grad_perturb_p*grad_perturb_p).sum(1) + 2.0*(grad_base_p*grad_perturb_p).sum(1)
		pred_val_s = (grad_base_s*grad_base_s).sum(1) + (grad_perturb_s*grad_perturb_s).sum(1) + 2.0*(grad_base_s*grad_perturb_s).sum(1)

		## Convert to predict velocity, and scale by pde factor
		pred_val_p = (1.0/scale_norm_factor)*(pred_val_p**(-0.5))
		pred_val_s = (1.0/scale_norm_factor)*(pred_val_s**(-0.5))
		r2_vp1 = r2_score(trgt[:,0].cpu().detach().numpy(), pred_val_p.cpu().detach().numpy()) 
		r2_vs1 = r2_score(trgt[:,1].cpu().detach().numpy(), pred_val_s.cpu().detach().numpy()) 

		## Validation
		isample = np.random.randint(0, high = n_dataset_vald, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples_vald[isample]).to(device)
		src_pos = torch.Tensor(X_samples_vald[isample]).to(device)

		if compute_reference_times == True: ## Can include a data loss
			travel_times_p = Tp_samples_vald[isample] #
			travel_times_s = Ts_samples_vald[isample] # = sample_inputs_unweighted(n_batch)
			trgt_data = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

		pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
		# pred_perturb = pred_perturb1 + pred_perturb2

		vel_pred = m.vmodel(src_pos_cart, src_embed)

		scale_loss = 1.0
		trgt = vel_pred # norm_vel(torch.Tensor(np.concatenate((Vp_samples_vald[isample].reshape(-1,1), Vs_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)

		## Compute the predicted PDE value
		scale_val_grad = 1.0
		grad_base_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_base_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_base[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		grad_perturb_p = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[0]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]
		grad_perturb_s = scale_val_grad*torch.autograd.grad(inputs = src_pos_cart, outputs = pred_perturb[:,[1]], grad_outputs = torch_one_vec, retain_graph = True, create_graph = True)[0]

		pred_val_p = (grad_base_p*grad_base_p).sum(1) + (grad_perturb_p*grad_perturb_p).sum(1) + 2.0*(grad_base_p*grad_perturb_p).sum(1)
		pred_val_s = (grad_base_s*grad_base_s).sum(1) + (grad_perturb_s*grad_perturb_s).sum(1) + 2.0*(grad_base_s*grad_perturb_s).sum(1)

		## Convert to predict velocity, and scale by pde factor
		pred_val_p = (1.0/scale_norm_factor)*(pred_val_p**(-0.5))
		pred_val_s = (1.0/scale_norm_factor)*(pred_val_s**(-0.5))
		r2_vp_vald1 = r2_score(trgt[:,0].cpu().detach().numpy(), pred_val_p.cpu().detach().numpy()) 
		r2_vs_vald1 = r2_score(trgt[:,1].cpu().detach().numpy(), pred_val_s.cpu().detach().numpy()) 

	else:

		r2_vp1, r2_vs1, r2_vp_vald1, r2_vs_vald1 = 0.0, 0.0, 0.0, 0.0

	print('Velocity correlations are: \n')
	print('%.2f [vp], %0.2f [vs], %0.2f [vp vald], %0.2f [vs vald]'%(r2_vp1, r2_vs1, r2_vp_vald1, r2_vs_vald1))

	r_vals1 = [r2_vp1, r2_vs1, r2_vp_vald1, r2_vs_vald1]

	n_ver_save = 1

	m = m.cpu()
	torch.save(m.state_dict(), path_save + 'travel_time_neural_network_physics_informed_%s_ver_%d.h5'%(phase, n_ver_save))
	torch.save(optimizer.state_dict(), path_save + 'travel_time_neural_network_physics_informed_%s_optimizer_ver_%d.h5'%(phase, n_ver_save))
	# np.savez_compressed(path_save + 'travel_time_neural_network_physics_informed_%s_losses_ver_%d.npz'%(phase, n_ver_save), out1 = out1, out2 = out2, trgt1 = trgt1, trgt2 = trgt2, sta_pos1 = sta_pos1.cpu().detach().numpy(), src_pos1 = src_pos1.cpu().detach().numpy(), sta_pos2 = sta_pos2.cpu().detach().numpy(), src_pos2 = src_pos2.cpu().detach().numpy(), v_mean = v_mean, scale_params = scale_params, losses = losses, losses_vald = losses_vald)
	np.savez_compressed(path_save + 'travel_time_neural_network_physics_informed_%s_losses_ver_%d.npz'%(phase, n_ver_save), out1 = out1, out2 = out2, trgt1 = trgt1, trgt2 = trgt2, sta_pos1 = sta_pos1, src_pos1 = src_pos1, sta_pos2 = sta_pos2, src_pos2 = src_pos2, v_mean = v_mean, scale_params = scale_params, r_vals = r_vals, losses = losses, losses_vald = losses_vald)
	m = m.to(device)

	make_plot = True
	if make_plot == True:

		# path_to_file = str(pathlib.Path().absolute())

		fig, ax = plt.subplots(1, figsize = [8,5])
		ax.plot(losses, label = 'Train')
		ax.plot(vald_steps*np.arange(len(losses_vald)), losses_vald, label = 'Vald')
		ax.set_yscale('log')
		ax.set_xlabel('Update Step')
		ax.set_ylabel('Losses')
		ax.legend()
		fig.savefig(path_to_file + 'Plots' + seperator + 'losses_travel_time_model_physics_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

		if compute_reference_times == True:
			fig, ax = plt.subplots(2, 2, figsize = [12,10])
			ax[0,0].scatter(trgt1[:,0], out1[:,0] - trgt1[:,0], 30, alpha = 0.75, label = 'Train')
			ax[0,0].scatter(trgt2[:,0], out2[:,0] - trgt2[:,0], 30, alpha = 0.75, label = 'Vald')
			ax[0,0].set_xlabel('Travel Time (P wave)')
			ax[0,0].set_ylabel('Travel Time Residual (P wave)')
			ax[0,0].legend()

			ax[0,1].scatter(trgt1[:,1], out1[:,1] - trgt1[:,1], 30, alpha = 0.75, label = 'Train')
			ax[0,1].scatter(trgt2[:,1], out2[:,1] - trgt2[:,1], 30, alpha = 0.75, label = 'Vald')
			ax[0,1].set_xlabel('Travel Time (S wave)')
			ax[0,1].set_ylabel('Travel Time Residual (S wave)')
			ax[0,1].legend()

			ax[1,0].hist(out1[:,0] - trgt1[:,0], 30, alpha = 0.75, label = 'Train')
			ax[1,0].hist(out2[:,0] - trgt2[:,0], 30, alpha = 0.75, label = 'Vald')
			ax[1,0].set_xlabel('Travel Time Residual (P wave)')
			ax[1,0].set_ylabel('Counts')
			ax[1,0].legend()

			ax[1,1].hist(out1[:,1] - trgt1[:,1], 30, alpha = 0.75, label = 'Train')
			ax[1,1].hist(out2[:,1] - trgt2[:,1], 30, alpha = 0.75, label = 'Vald')
			ax[1,1].set_xlabel('Travel Time Residual (S wave)')
			ax[1,1].set_ylabel('Counts')
			ax[1,1].legend()
			fig.savefig(path_to_file + 'Plots' + seperator + 'residuals_travel_time_model_physics_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

		fig, ax = plt.subplots(1,2, figsize = [10,5])
		if compute_reference_times == True:
			ax[0].scatter(np.linalg.norm(ftrns1(sta_pos1) - ftrns1(src_pos1), axis = 1)/1000.0, trgt1[:,0], alpha = 0.5, label = 'Trgt')
		ax[0].scatter(np.linalg.norm(ftrns1(sta_pos1) - ftrns1(src_pos1), axis = 1)/1000.0, out1[:,0], alpha = 0.5, label = 'Pred')
		ax[0].set_xlabel('Source - Reciever Distance (km)')
		ax[0].set_ylabel('Travel Time (P wave)')
		ax[0].legend()

		if compute_reference_times == True:
			ax[1].scatter(np.linalg.norm(ftrns1(sta_pos1) - ftrns1(src_pos1), axis = 1)/1000.0, trgt1[:,1], alpha = 0.5, label = 'Trgt')
		ax[1].scatter(np.linalg.norm(ftrns1(sta_pos1) - ftrns1(src_pos1), axis = 1)/1000.0, out1[:,1], alpha = 0.5, label = 'Pred')
		ax[1].set_xlabel('Source - Reciever Distance (km)')
		ax[1].set_ylabel('Travel Time (S wave)')
		ax[1].legend()
		fig.savefig(path_to_file + 'Plots' + seperator + 'travel_time_vs_distance_physics_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

		fig, ax = plt.subplots(2,1, figsize = [10,5])
		ones_vec = np.array([1.0, 1.0, 0.0]).reshape(1,-1) # .to(device)
		ax[0].scatter(np.linalg.norm(ftrns1(sta_pos1*ones_vec) - ftrns1(src_pos1*ones_vec), axis = 1)/1000.0, src_pos1[:,2]/1000.0, c = out1[:,0], label = 'P')
		ax[1].scatter(np.linalg.norm(ftrns1(sta_pos1*ones_vec) - ftrns1(src_pos1*ones_vec), axis = 1)/1000.0, src_pos1[:,2]/1000.0, c = out1[:,1], label = 'S')
		ax[1].set_xlabel('Source - Reciever Distance (km)')
		ax[0].set_ylabel('Source Depth (km)')
		ax[1].set_ylabel('Source Depth (km)')
		ax[0].legend()
		ax[1].legend()
		fig.savefig(path_to_file + 'Plots' + seperator + 'travel_time_vs_distance_depth_physics_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)

		plt.close('all')

	## Predict contour maps of travel time
	x1 = np.linspace(locs[:,0].min(), locs[:,0].max(), 50)
	x2 = np.linspace(locs[:,1].min(), locs[:,1].max(), 50)
	x3 = np.array([-5e3])
	# x3 = np.array([-20e3, -10e3, -5e3, 0e3])
	x11, x12, x13 = np.meshgrid(x1, x2, x3)
	x_contour = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
	trv_out = m(torch.Tensor(x_contour).to(device), torch.Tensor(locs).to(device))

	## Note: may be able to use only the relative version of travel times, and hence still sample the sources at fixed locations during training

	## Make transects and profiles of velocity
	x1 = np.array([locs[:,0].mean()])
	x2 = np.array([locs[:,1].mean()])
	x3 = np.arange(depth_range[0], depth_range[1], 50.0)
	x11, x12, x13 = np.meshgrid(x1, x2, x3)
	xx1 = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)

	pred_base, pred_perturb, src_pos_cart, src_embed = m(torch.Tensor(locs[[0],:]).to(device).repeat(len(xx1),1), torch.Tensor(xx1).to(device), method = 'direct', train = True)
	# pred_perturb = pred_perturb1 + pred_perturb2
	vel_pred = inorm_vel(m.vmodel(src_pos_cart, src_embed).cpu().detach().numpy())

	vel_pred_levels = []
	for i in range(len(x3)):
		xv = np.copy(x_contour)
		xv[:,2] = x3[i]
		pred_base, pred_perturb, src_pos_cart, src_embed = m(torch.Tensor(locs[[0],:]).to(device).repeat(len(xv),1), torch.Tensor(xv).to(device), method = 'direct', train = True)
		# pred_perturb = pred_perturb1 + pred_perturb2
		vel_pred_levels.append(np.expand_dims(m.vmodel(src_pos_cart, src_embed).cpu().detach().numpy(), axis = 0))
	vel_pred_levels = inorm_vel(np.vstack(vel_pred_levels))

	if make_plot == True:
		plt.figure()
		plt.plot(vel_pred[:,0], x3/1000.0, label = 'Vp')
		plt.plot(vel_pred[:,1], x3/1000.0, label = 'Vs')
		plt.legend()
		plt.xlabel('Velocity (m/s)')
		plt.ylabel('Depth (km)')
		fig = plt.gcf()
		fig.set_size_inches([15,12])
		fig.savefig(path_to_file + 'Plots' + seperator + 'velocity_as_function_of_depth_physics_ver_%d.png'%n_ver_save, bbox_inches = 'tight', pad_inches = 0.2)
	
	np.savez_compressed(path_to_file + 'training_results_ver_%d.npz'%n_ver_save, trv_out = trv_out.cpu().detach().numpy(), vel_pred = vel_pred, vel_pred_levels = vel_pred_levels, pos_contour = x_contour, xx1 = xx1, x3 = x3, losses = losses, losses_vald = losses_vald, losses_pde = losses_pde, losses_pde_vald = losses_pde_vald, losses_data = losses_data, losses_data_vald = losses_data_vald)


	# 	for s in st_list:
	# 		z = h5py.File(s, 'r')
	# 		srcs_slice = z['srcs_trv'][:]
	# 		locs_use = z['locs_use'][:]

	# 		for j in range(len(srcs_slice)):

	# 			arv_p, ind_p, arv_s, ind_s = z['Picks/%d_Picks_P_perm'%j][:,0], z['Picks/%d_Picks_P_perm'%j][:,1].astype('int'), z['Picks/%d_Picks_S_perm'%j][:,0], z['Picks/%d_Picks_S_perm'%j][:,1].astype('int')

	# 			ind_unique_arrivals = np.sort(np.unique(np.concatenate((ind_p, ind_s), axis = 0)).astype('int'))

	# 			if len(ind_unique_arrivals) == 0:
	# 				srcs_trv.append(np.nan*np.ones((1, 4)))
	# 				continue			
				
	# 			perm_vec_arrivals = -1*np.ones(locs_use.shape[0]).astype('int')
	# 			perm_vec_arrivals[ind_unique_arrivals] = np.arange(len(ind_unique_arrivals))
	# 			locs_use_slice = locs_use[ind_unique_arrivals]
	# 			ind_p_perm_slice = perm_vec_arrivals[ind_p]
	# 			ind_s_perm_slice = perm_vec_arrivals[ind_s]
	# 			if len(ind_p_perm_slice) > 0:
	# 				assert(ind_p_perm_slice.min() > -1)
	# 			if len(ind_s_perm_slice) > 0:
	# 				assert(ind_s_perm_slice.min() > -1)

	# 			xmle, logprob = differential_evolution_location(m, locs_use_slice, arv_p, ind_p_perm_slice, arv_s, ind_s_perm_slice, lat_range_extend, lon_range_extend, depth_range, device = device)

	# 			pred_out = m(torch.Tensor(locs_use_slice).to(device), torch.Tensor(xmle).to(device)).cpu().detach().numpy() + srcs_slice[j,3]

	# 			res_p = pred_out[0,ind_p_perm_slice,0] - arv_p
	# 			res_s = pred_out[0,ind_s_perm_slice,1] - arv_s

	# 			mean_shift = 0.0
	# 			cnt_phases = 0
	# 			if len(res_p) > 0:
	# 				mean_shift += np.median(res_p)*(len(res_p)/(len(res_p) + len(res_s)))
	# 				cnt_phases += 1

	# 			if len(res_s) > 0:
	# 				mean_shift += np.median(res_s)*(len(res_s)/(len(res_p) + len(res_s)))
	# 				cnt_phases += 1

	# 			Srcs_initial.append(srcs_slice[j,:])
	# 			Srcs_relocate.append(np.concatenate((xmle, np.array([srcs_slice[j,3] - mean_shift]).reshape(1,-1)), axis = 1))

	# 		z.close()


	######################## Run FMM checks ########################
	# m = m.to(device)

	run_fast_marching_method_quality_check = False
	if run_fast_marching_method_quality_check == True:
		## Run FMM on random samples of sources and recievers and compute travel time error.
		
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

		## Using 3D domain, so must use actual station coordinates
		locs_ref = np.copy(locs)
		reciever_proj = ftrns1(locs_ref) # for all elevs.

		Vp = interp(ftrns2(xx)[:,2], depth_grid, Vp_profile) ## This may be locating some depths at incorrect positions, due to projection.
		Vs = interp(ftrns2(xx)[:,2], depth_grid, Vs_profile)

		## Note: can add this to training data as well
		sta_check = []

		Res_p = []
		Res_s = []
		Rel_p = []
		Rel_s = []

		Res_p1 = []
		Res_s1 = []
		Rel_p1 = []
		Rel_s1 = []

		n_ver_trv_time_model_load = 1
		trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, device = device)

		n_sta_check = 10
		n_queries_check = 10000

		# assert(len(locs_ref) == len(reciever_proj))
		for j in range(n_sta_check):

			i0 = np.random.choice(ind_sta)
			if use_relative_1d_profile == True: ## If True, shift the profile so that the same value occurs at each stations elevation (e.g., the profile varies with the surface)
				Vp = interp(ftrns2(xx)[:,2], depth_grid + locs[i0,2], Vp_profile) ## This may be locating some depths at incorrect positions, due to projection.
				Vs = interp(ftrns2(xx)[:,2], depth_grid + locs[i0,2], Vs_profile)
			
			results = compute_travel_times_parallel(xx, ftrns1(locs[i0][None,:]), Vp, Vs, dx_v, x11, x12, x13, num_cores = num_cores)

			## Check residuals and relative residuals

			irand_query = np.sort(np.random.choice(len(xx), size = n_queries_check, replace = False))
			trgt_val_p = results[0][irand_query,0]
			trgt_val_s = results[1][irand_query,0]

			pred_val = m(torch.Tensor(locs[i0][None,:]).to(device), torch.Tensor(ftrns2(xx[irand_query])).to(device)).cpu().detach().numpy()
			res_p = pred_val[:,0,0] - trgt_val_p
			res_s = pred_val[:,0,1] - trgt_val_s
			rel_res_p = res_p/trgt_val_p
			rel_res_s = res_s/trgt_val_s
			Res_p.append(res_p)
			Res_s.append(res_s)
			Rel_p.append(rel_res_p)
			Rel_s.append(rel_res_s)

			pred_val1 = trv(torch.Tensor(locs[i0][None,:]).to(device), torch.Tensor(ftrns2(xx[irand_query])).to(device)).cpu().detach().numpy()
			res_p = pred_val1[:,0,0] - trgt_val_p
			res_s = pred_val1[:,0,1] - trgt_val_s
			rel_res_p = res_p/trgt_val_p
			rel_res_s = res_s/trgt_val_s
			Res_p1.append(res_p)
			Res_s1.append(res_s)
			Rel_p1.append(rel_res_p)
			Rel_s1.append(rel_res_s)

		Res_p = np.hstack(Res_p)
		Res_s = np.hstack(Res_s)
		Rel_p = np.hstack(Rel_p)
		Rel_s = np.hstack(Rel_s)

		Res_p1 = np.hstack(Res_p1)
		Res_s1 = np.hstack(Res_s1)
		Rel_p1 = np.hstack(Rel_p1)
		Rel_s1 = np.hstack(Rel_s1)

	remove_travel_time_files = True
	if remove_travel_time_files == True:
		for s in st_sta:
			os.remove(s)

print("All files saved successfully!")
print("✔ Script execution: Done")
