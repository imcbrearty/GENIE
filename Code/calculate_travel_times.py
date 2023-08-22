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
seperator =  '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

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

	Tp_interp, Ts_interp = compute_interpolation_parallel(x1, x2, x3, Tp, Ts, X, ftrns1, num_cores = num_cores)

save_dense_travel_time_data = True
train_travel_time_neural_network = True
if ((train_travel_time_neural_network == False) + (save_dense_travel_time_data == True)) > 0:

	np.savez_compressed(path_to_file + '1D_Velocity_Models_Regional' + seperator + '%s_1d_velocity_model_ver_%d.npz'%(name_of_project, n_ver), X = X, locs_ref = locs_ref, Tp_interp = Tp_interp, Ts_interp = Ts_interp, Vp_profile = Vp_profile, Vs_profile = Vs_profile, depth_grid = depth_grid)

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


	using_3D = False
	using_1D = True
	assert((using_3D + using_1D) == 1)

	X_samples, Tp_samples, Ts_samples, Locs_samples = [], [], [], []
	X_samples_vald, Tp_samples_vald, Ts_samples_vald, Locs_samples_vald = [], [], [], []

	if using_3D == True:

		print('Check if locs_ref are fixed or span 3D space')

		## For 3D, randomly shift the source and reciever to emulate being in 3D (if using fixed at specific node coordinates, or other
		## wise accept locs_ref as is)

		## Note: instead of locs_ref could use actually locs locations, and obtain a 3D model

		grab_near_station_samples = True
		if grab_near_station_samples == True:

			t_scale_sample = 5.0
			n_per_station = 15000

			for n in range(locs_ref.shape[0]):

				p = 1.0/np.maximum(Tp_interp[:,n], 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_samples.append(X[isample])
				Tp_samples.append(Tp_interp[isample,n])
				Ts_samples.append(Ts_interp[isample,n])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))

		grab_near_boundaries_samples = True
		if grab_near_boundaries_samples == True:

			t_scale_sample = 5.0
			n_per_station = 20000

			for n in range(locs_ref.shape[0]):

				p = 1.0/np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_samples.append(X[isample])
				Tp_samples.append(Tp_interp[isample,n])
				Ts_samples.append(Ts_interp[isample,n])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))

		grab_interior_samples = True
		if grab_interior_samples == True:

			t_scale_sample = 5.0
			n_per_station = 100000

			for n in range(locs_ref.shape[0]):

				p = 1.0*np.ones(Tp_interp.shape[0]) # /np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))
				isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = 10000)

				X_samples.append(X[isample])
				Tp_samples.append(Tp_interp[isample,n])
				Ts_samples.append(Ts_interp[isample,n])
				Locs_samples.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample), axis = 0))

				X_samples_vald.append(X[isample_vald])
				Tp_samples_vald.append(Tp_interp[isample_vald,n])
				Ts_samples_vald.append(Ts_interp[isample_vald,n])
				Locs_samples_vald.append(locs_ref[n,:].reshape(1,-1).repeat(len(isample_vald), axis = 0))

	elif using_1D == True:

		## For 3D, randomly shift the source and reciever to emulate being in 3D (if using fixed at specific node coordinates, or other
		## wise accept locs_ref as is)

		one_vec = np.array([1.0, 1.0, 0.0]).reshape(1,-1)

		grab_near_station_samples = True
		if grab_near_station_samples == True:

			t_scale_sample = 5.0
			n_per_station = 15000

			for n in range(locs_ref.shape[0]):

				p = 1.0/np.maximum(Tp_interp[:,n], 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_offset_sample = locs_ref[n,:].reshape(1,-1)*one_vec - X[isample]*one_vec
				X_offset_sample[:,0:2] = X_offset_sample[:,0:2]*np.random.choice([1.0, -1.0], size = (len(isample), 2))
				locs_rand = locs[np.random.randint(0, high = locs.shape[0], size = len(isample))]
				X_offset_sample = X_offset_sample + locs_rand
				locs_rand[:,2] = locs_red[n,2]
				X_offset_sample[:,2] = X[isample,2]

				X_samples.append(X_offset_sample)
				Tp_samples.append(Tp_interp[isample,n])
				Ts_samples.append(Ts_interp[isample,n])
				Locs_samples.append(locs_rand)

		grab_near_boundaries_samples = True
		if grab_near_boundaries_samples == True:

			t_scale_sample = 5.0
			n_per_station = 20000

			for n in range(locs_ref.shape[0]):

				p = 1.0/np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_offset_sample = locs_ref[n,:].reshape(1,-1)*one_vec - X[isample]*one_vec
				X_offset_sample[:,0:2] = X_offset_sample[:,0:2]*np.random.choice([1.0, -1.0], size = (len(isample), 2))
				locs_rand = locs[np.random.randint(0, high = locs.shape[0], size = len(isample))]
				X_offset_sample = X_offset_sample + locs_rand
				locs_rand[:,2] = locs_red[n,2]
				X_offset_sample[:,2] = X[isample,2]

				X_samples.append(X_offset_sample)
				Tp_samples.append(Tp_interp[isample,n])
				Ts_samples.append(Ts_interp[isample,n])
				Locs_samples.append(locs_rand)

		grab_interior_samples = True
		if grab_interior_samples == True:

			t_scale_sample = 5.0
			n_per_station = 100000

			for n in range(locs_ref.shape[0]):

				p = 1.0*np.ones(Tp_interp.shape[0]) # /np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)

				isample = np.sort(np.random.choice(len(p), size = n_per_station, p = p/p.sum(), replace = False))

				X_offset_sample = locs_ref[n,:].reshape(1,-1)*one_vec - X[isample]*one_vec
				X_offset_sample[:,0:2] = X_offset_sample[:,0:2]*np.random.choice([1.0, -1.0], size = (len(isample), 2))
				locs_rand = locs[np.random.randint(0, high = locs.shape[0], size = len(isample))]
				X_offset_sample = X_offset_sample + locs_rand
				locs_rand[:,2] = locs_red[n,2]
				X_offset_sample[:,2] = X[isample,2]

				X_samples.append(X_offset_sample)
				Tp_samples.append(Tp_interp[isample,n])
				Ts_samples.append(Ts_interp[isample,n])
				Locs_samples.append(locs_rand)

				isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = 10000)

				X_offset_sample = locs_ref[n,:].reshape(1,-1)*one_vec - X[isample_vald]*one_vec
				X_offset_sample[:,0:2] = X_offset_sample[:,0:2]*np.random.choice([1.0, -1.0], size = (len(isample_vald), 2))
				locs_rand = locs[np.random.randint(0, high = locs.shape[0], size = len(isample_vald))]
				X_offset_sample = X_offset_sample + locs_rand
				X_offset_sample[:,2] = X[isample_vald,2]

				X_samples_vald.append(X_offset_sample)
				Tp_samples_vald.append(Tp_interp[isample_vald,n])
				Ts_samples_vald.append(Ts_interp[isample_vald,n])
				Locs_samples_vald.append(locs_rand)
	# Concatenate training dataset
	X_samples = np.vstack(X_samples)
	Tp_samples = np.hstack(Tp_samples)
	Ts_samples = np.hstack(Ts_samples)
	Locs_samples = np.vstack(Locs_samples)

	# Concatenate validation dataset
	X_samples_vald = np.vstack(X_samples_vald)
	Tp_samples_vald = np.hstack(Tp_samples_vald)
	Ts_samples_vald = np.hstack(Ts_samples_vald)
	Locs_samples_vald = np.vstack(Locs_samples_vald)

	n_dataset = len(X_samples)
	n_dataset_vald = len(X_samples_vald)

	irand_sample = np.random.choice(n_dataset, size = int(0.2*n_dataset))
	scale_val = np.round(2.0*np.linalg.norm(ftrns1(X_samples[irand_sample]) - ftrns1(Locs_samples[irand_sample]), axis = 1).max())
	trav_val = np.round(1.25*Ts_samples.max())

	print('Using a single model for both phase types')
	# Note: training a seperate model for either phase type can be more accurate
	m = TravelTimes(ftrns1_diff, ftrns2_diff, scale_val = scale_val, trav_val = trav_val, n_phases = 2, device = device).to(device)

	# m_p = TravelTimes(ftrns1_diff, ftrns2_diff, n_phases = 1, device = device).to(device)
	# m_s = TravelTimes(ftrns1_diff, ftrns2_diff, n_phases = 1, device = device).to(device)

	optimizer = optim.Adam(m.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 10000, gamma = 0.9)
	loss_func = nn.MSELoss()
	# loss_func1 = nn.BCELoss()

	n_batch = 5000
	n_steps = 20001 # 50000
	n_ver_save = 1

	assert((using_3D + using_1D) == 1)
	# if use_3D == True:
	# 	method = 'direct'
	# elif use_1D == True:
	# 	method = 'direct relative'


	# moi

	losses = []
	losses_vald = []
	vald_steps = 10
	loss_vald = 0.0

	for i in range(n_steps):

		optimizer.zero_grad()

		isample = np.random.randint(0, high = n_dataset, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos = torch.Tensor(X_samples[isample]).to(device)
		travel_times_p = Tp_samples[isample] #
		travel_times_s = Ts_samples[isample] # = sample_inputs_unweighted(n_batch)

		trgt = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

		if using_3D == True:
			out = m.forward_relative_train(sta_pos, src_pos, method = 'direct', p = 0.3)

		elif using_1D == True:
			out = m.forward_relative(sta_pos, src_pos, method = 'direct')

		loss = loss_func(out/m.tscale, trgt/m.tscale)

		# sta_pos, src_pos, masks = sample_masks_unweighted(n_batch)
		# out = m.forward_mask_train(torch.Tensor(sta_pos).to(device), torch.Tensor(src_pos).to(device), method = 'direct', p = 0.3)
		# loss = loss + 0.3*loss_func1(out, torch.Tensor(masks.reshape(-1,1)).to(device))

		loss.backward()
		optimizer.step()
		scheduler.step()
		losses.append(loss.item())

		if np.mod(i, vald_steps) == 0:

			with torch.no_grad():
				isample = np.random.randint(0, high = n_dataset_vald, size = n_batch)

				sta_pos = torch.Tensor(Locs_samples_vald[isample]).to(device)
				src_pos = torch.Tensor(X_samples_vald[isample]).to(device)
				travel_times_p = Tp_samples_vald[isample] #
				travel_times_s = Ts_samples_vald[isample] # = sample_inputs_unweighted(n_batch)

				trgt = torch.Tensor(np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)).to(device)

				if using_3D == True:
					out = m.forward_relative_train(sta_pos, src_pos, method = 'direct', p = 0.3)

				elif using_1D == True:
					out = m.forward_relative(sta_pos, src_pos, method = 'direct')

			loss_vald = loss_func(out/m.tscale, trgt/m.tscale)

			losses_vald.append(loss_vald.item())

		print('%d %0.8f %0.8f'%(i, loss.item(), loss_vald.item()))

	with torch.no_grad():

		## Mesure loss on training and validation and make residual plot

		isample = np.random.randint(0, high = n_dataset, size = n_batch)
		sta_pos1 = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos1 = torch.Tensor(X_samples[isample]).to(device)
		travel_times_p = Tp_samples[isample] #
		travel_times_s = Ts_samples[isample] # = sample_inputs_unweighted(n_batch)
		trgt1 = np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)
		if using_3D == True:
			out1 = m.forward_relative_train(sta_pos1, src_pos1, method = 'direct', p = 0.3).cpu().detach().numpy()
		elif using_1D == True:
			out1 = m.forward_relative(sta_pos1, src_pos1, method = 'direct').cpu().detach().numpy()


		isample = np.random.randint(0, high = n_dataset_vald, size = n_batch)
		sta_pos2 = torch.Tensor(Locs_samples_vald[isample]).to(device)
		src_pos2 = torch.Tensor(X_samples_vald[isample]).to(device)
		travel_times_p = Tp_samples_vald[isample] #
		travel_times_s = Ts_samples_vald[isample] # = sample_inputs_unweighted(n_batch)
		trgt2 = np.concatenate((travel_times_p.reshape(-1,1), travel_times_s.reshape(-1,1)), axis = 1)
		if using_3D == True:
			out2 = m.forward_relative_train(sta_pos2, src_pos2, method = 'direct', p = 0.3).cpu().detach().numpy()
		elif using_1D == True:
			out2 = m.forward_relative(sta_pos2, src_pos2, method = 'direct').cpu().detach().numpy()

	make_plot = True
	if make_plot == True:

		# path_to_file = str(pathlib.Path().absolute())

		fig, ax = plt.subplots(1, figsize = [8,5])
		ax.plot(losses, label = 'Train')
		ax.plot(vald_steps*np.arange(len(losses_vald)), losses_vald, label = 'Vald')
		ax.set_yscale('log')
		ax.legend()
		fig.savefig(path_to_file + 'Plots' + seperator + 'losses_travel_time_model.png', bbox_inches = 'tight', pad_inches = 0.2)

		fig, ax = plt.subplots(2, 2, figsize = [10,10])
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
		fig.savefig(path_to_file + 'Plots' + seperator + 'residuals_travel_time_model.png', bbox_inches = 'tight', pad_inches = 0.2)

		fig, ax = plt.subplots(1,2, figsize = [10,5])
		ax[0].scatter(torch.norm(ftrns1_diff(sta_pos1) - ftrns1_diff(src_pos1), dim = 1).cpu().detach().numpy()/1000.0, trgt1[:,0], alpha = 0.5, label = 'Trgt')
		ax[0].scatter(torch.norm(ftrns1_diff(sta_pos1) - ftrns1_diff(src_pos1), dim = 1).cpu().detach().numpy()/1000.0, out1[:,0], alpha = 0.5, label = 'Pred')
		ax[0].set_xlabel('Source - Reciever Distance (km)')
		ax[0].set_ylabel('Travel Time (P wave)')
		ax[0].legend()

		ax[1].scatter(torch.norm(ftrns1_diff(sta_pos1) - ftrns1_diff(src_pos1), dim = 1).cpu().detach().numpy()/1000.0, trgt1[:,1], alpha = 0.5, label = 'Trgt')
		ax[1].scatter(torch.norm(ftrns1_diff(sta_pos1) - ftrns1_diff(src_pos1), dim = 1).cpu().detach().numpy()/1000.0, out1[:,1], alpha = 0.5, label = 'Pred')
		ax[1].set_xlabel('Source - Reciever Distance (km)')
		ax[1].set_ylabel('Travel Time (S wave)')
		ax[1].legend()
		fig.savefig(path_to_file + 'Plots' + seperator + 'travel_time_vs_distance.png', bbox_inches = 'tight', pad_inches = 0.2)

		fig, ax = plt.subplots(2,1, figsize = [10,5])
		ones_vec = torch.Tensor([1.0, 1.0, 0.0]).reshape(1,-1).to(device)
		ax[0].scatter(torch.norm(ftrns1_diff(sta_pos1*ones_vec) - ftrns1_diff(src_pos1*ones_vec), dim = 1).cpu().detach().numpy()/1000.0, src_pos1[:,2].cpu().detach().numpy()/1000.0, c = out1[:,0], label = 'P')
		ax[1].scatter(torch.norm(ftrns1_diff(sta_pos1*ones_vec) - ftrns1_diff(src_pos1*ones_vec), dim = 1).cpu().detach().numpy()/1000.0, src_pos1[:,2].cpu().detach().numpy()/1000.0, c = out1[:,1], label = 'S')
		ax[1].set_xlabel('Source - Reciever Distance (km)')
		ax[0].set_ylabel('Source Depth (km)')
		ax[1].set_ylabel('Source Depth (km)')
		ax[0].legend()
		ax[1].legend()
		fig.savefig(path_to_file + 'Plots' + seperator + 'travel_time_vs_distance_depth.png', bbox_inches = 'tight', pad_inches = 0.2)

		plt.close('all')

	phase = 'p_s'
	scale_val = m.scale.cpu().detach().numpy()
	trav_val = m.tscale.cpu().detach().numpy()
	m = m.cpu()
	path_save = path_to_file + '1D_Velocity_Models_Regional' + seperator
	torch.save(m.state_dict(), path_save + 'travel_time_neural_network_%s_ver_%d.h5'%(phase, n_ver_save))
	torch.save(optimizer.state_dict(), path_save + 'travel_time_neural_network_%s_optimizer_ver_%d.h5'%(phase, n_ver_save))
	np.savez_compressed(path_save + 'travel_time_neural_network_%s_losses_ver_%d.npz'%(phase, n_ver_save), out1 = out1, out2 = out2, trgt1 = trgt1, trgt2 = trgt2, sta_pos1 = sta_pos1.cpu().detach().numpy(), src_pos1 = src_pos1.cpu().detach().numpy(), sta_pos2 = sta_pos2.cpu().detach().numpy(), src_pos2 = src_pos2.cpu().detach().numpy(), scale_val = scale_val, trav_val = trav_val, losses = losses, losses_vald = losses_vald)

print("All files saved successfully!")
print("✔ Script execution: Done")
