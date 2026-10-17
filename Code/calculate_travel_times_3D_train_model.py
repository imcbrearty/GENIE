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


# Load configuration from YAML
config = load_config('config.yaml')
name_of_project = config['name_of_project']
vel_model_ver = config['vel_model_ver']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load travel time neural network settings
train_travel_time_neural_network = config['train_travel_time_neural_network']

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())
seperator =  '\\' if '\\' in path_to_file else '/'
path_to_file += seperator


z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load stations
z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
z.close()

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1

## Determine which stations have data
st_sta = glob.glob(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'TravelTimeData' + seperator + '*station*ver_%d.npz'%vel_model_ver)
iarg = np.argsort([int(st_sta[j].split('/')[-1].split(name_of_project)[-1].split('_')[5]) for j in range(len(st_sta))])
st_sta = [st_sta[j] for j in iarg]
sta_ind = np.array([int(st_sta[j].split('/')[-1].split(name_of_project)[-1].split('_')[5]) for j in range(len(st_sta))]).astype('int')


if train_travel_time_neural_network == True:

	rbest_cuda = torch.Tensor(rbest).to(device)
	mn_cuda = torch.Tensor(mn).to(device)

	# use_spherical = False
	if config['use_spherical'] == True:
		earth_radius = 6371e3
		ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, e = 0.0, a = earth_radius, device = device) - mn_cuda).T).T # just subtract mean
		ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, e = 0.0, a = earth_radius, device = device) # just subtract mean
	
	else:
		earth_radius = 6378137.0
		ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T # just subtract mean
		ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device) # just subtract mean


	X_samples, Tp_samples, Ts_samples, Locs_samples = [], [], [], []
	X_samples_vald, Tp_samples_vald, Ts_samples_vald, Locs_samples_vald = [], [], [], []
	Vp_samples, Vs_samples, Vp_samples_vald, Vs_samples_vald = [], [], [], []
	Locs_samples_inds = []

	Vp_samples_boundary, Vs_samples_boundary = [], []
	X_samples_boundary, Locs_samples_boundary = [], []

	Vp_samples_boundary_vald, Vs_samples_boundary_vald = [], []
	X_samples_boundary_vald, Locs_samples_boundary_vald = [], []


	for inc in range(len(sta_ind)):

		z = np.load(st_sta[inc])
		flag_boundary = np.where(np.abs(z['Tp']) < 1e-5)[0]
		iflag_boundary = np.delete(np.arange(len(z['X'])), flag_boundary, axis = 0)

		## Training
		X_samples.append(z['X'][iflag_boundary])
		Tp_samples.append(z['Tp'][iflag_boundary])
		Ts_samples.append(z['Ts'][iflag_boundary])
		Vp_samples.append(z['Vp'][iflag_boundary])
		Vs_samples.append(z['Vs'][iflag_boundary])
		Locs_samples.append(z['loc'].reshape(1,-1).repeat(len(X_samples[-1]), axis = 0))

		X_samples_boundary.append(z['X'][flag_boundary])
		Vp_samples_boundary.append(z['Vp'][flag_boundary])
		Vs_samples_boundary.append(z['Vs'][flag_boundary])
		Locs_samples_boundary.append(z['loc'].reshape(1,-1).repeat(len(X_samples_boundary[-1]), axis = 0))

		## Validation
		flag_boundary = np.where(np.abs(z['Tp_vald']) < 1e-5)[0]
		iflag_boundary = np.delete(np.arange(len(z['X_vald'])), flag_boundary, axis = 0)
		X_samples_vald.append(z['X_vald'][iflag_boundary])
		Tp_samples_vald.append(z['Tp_vald'][iflag_boundary])
		Ts_samples_vald.append(z['Ts_vald'][iflag_boundary])
		Vp_samples_vald.append(z['Vp_vald'][iflag_boundary])
		Vs_samples_vald.append(z['Vs_vald'][iflag_boundary])
		Locs_samples_vald.append(z['loc'].reshape(1,-1).repeat(len(X_samples_vald[-1]), axis = 0))

		X_samples_boundary_vald.append(z['X_vald'][flag_boundary])
		Vp_samples_boundary_vald.append(z['Vp_vald'][flag_boundary])
		Vs_samples_boundary_vald.append(z['Vs_vald'][flag_boundary])
		Locs_samples_boundary_vald.append(z['loc'].reshape(1,-1).repeat(len(X_samples_boundary_vald[-1]), axis = 0))


	# Concatenate training dataset
	X_samples = np.vstack(X_samples)
	compute_reference_times = True
	if compute_reference_times == True:
		Tp_samples = np.hstack(Tp_samples)
		Ts_samples = np.hstack(Ts_samples)
	Locs_samples = np.vstack(Locs_samples)
	Vp_samples = np.hstack(Vp_samples)
	Vs_samples = np.hstack(Vs_samples)

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
	v_mean = np.array([Vp_samples.mean(), Vs_samples.mean()])

	print('Using a single model for both phase types')

	vp_max = Vp_samples.max() ## 9000.0
	vs_min = Vs_samples.min() ## 1000.0
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
	base_val_1 = np.linalg.norm(np.expand_dims(ftrns1(locs), axis = 0) - np.expand_dims(ftrns1(x_pos), axis = 1), axis = 2, keepdims = True)/v_mean.reshape(1,1,-1)
	base_val_2 = np.linalg.norm(np.expand_dims(norm_pos(ftrns1(locs)), axis = 0) - np.expand_dims(norm_pos(ftrns1(x_pos)), axis = 1), axis = 2, keepdims = True)/norm_vel(v_mean.reshape(1,1,-1))
	assert(np.allclose(base_val_1/base_val_2, (base_val_1/base_val_2).mean()))
	conversion_factor = (base_val_1/base_val_2).mean()
	conversion_factor = conversion_factor/max_time

	scale_params = [max_dist, max_time, vp_max, vs_min, scale_norm_factor, conversion_factor]
	m = TravelTimesPN(ftrns1_diff, ftrns2_diff, n_phases = 2, v_mean = v_mean, norm_pos = norm_pos, inorm_pos = inorm_pos, inorm_time = inorm_time, norm_vel = norm_vel, conversion_factor = conversion_factor, device = device).to(device)


	optimizer = optim.Adam(m.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 10000, gamma = 0.9)
	loss_func = nn.L1Loss()
	loss_func1 = nn.L1Loss()


	n_batch = 30000
	n_steps = 250001 # 50000
	n_ver_save = vel_model_ver
	use_causual_loss = True

	losses = []
	losses_vald = []
	vald_steps = 10
	loss_vald = 0.0
	n_burn_in = 1000

	use_initial_model_damping = True
	torch_one_vec = torch.ones(n_batch).reshape(-1,1).to(device)
  
	losses_pde = []
	losses_data = []
	losses_pde_vald = []
	losses_data_vald = []
	loss_data = torch.Tensor([0.0]).to(device)
	loss_data_vald = torch.Tensor([0.0]).to(device)


	for i in range(n_steps):

		optimizer.zero_grad()

		isample = np.random.randint(0, high = n_dataset, size = n_batch)

		sta_pos = torch.Tensor(Locs_samples[isample]).to(device)
		src_pos = torch.Tensor(X_samples[isample]).to(device)

		if compute_reference_times == True:
			travel_times_p = Tp_samples[isample]
			travel_times_s = Ts_samples[isample]

		pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
		vel_pred = m.vmodel(src_pos_cart, src_embed)

		# 1. Total travel times
		pred_total_p = pred_base[:, [0]] + pred_perturb[:, [0]]
		pred_total_s = pred_base[:, [1]] + pred_perturb[:, [1]]

		# 2. Optimized single-pass autograd
		grad_p = torch.autograd.grad(
			inputs = src_pos_cart, 
			outputs = pred_total_p, 
			grad_outputs = torch_one_vec, 
			retain_graph = True, 
			create_graph = True
		)[0]

		grad_s = torch.autograd.grad(
			inputs = src_pos_cart, 
			outputs = pred_total_s, 
			grad_outputs = torch_one_vec, 
			retain_graph = False, 
			create_graph = True
		)[0]

		# 3. Compute gradient norm squared
		grad_sq_p = (grad_p * grad_p).sum(dim=1)
		grad_sq_s = (grad_s * grad_s).sum(dim=1)

		# 4. Option B: Safe Slowness Magnitude S = ||grad_T|| (Linear scale matching original setup)
		eps = 1e-6
		grad_s_p = torch.sqrt(grad_sq_p + eps)
		grad_s_s = torch.sqrt(grad_sq_s + eps)

		target_slowness_p = 1.0 / (scale_norm_factor * vel_pred[:, 0])
		target_slowness_s = 1.0 / (scale_norm_factor * vel_pred[:, 1])

		# 5. Distance-weighted Eikonal PDE loss in slowness space
		loss_weight = torch.exp(-torch.abs(inorm_time(pred_base[:, 0])) / t_scale_val)
		loss_weight = loss_weight / loss_weight.sum()

		loss_pde_p = 0.5 * loss_func(grad_s_p, target_slowness_p) + \
		             0.5 * loss_func(loss_weight * grad_s_p, loss_weight * target_slowness_p)

		loss_pde_s = 0.5 * loss_func(grad_s_s, target_slowness_s) + \
		             0.5 * loss_func(loss_weight * grad_s_s, loss_weight * target_slowness_s)

		## Boundary loss
		isample_boundary = np.random.randint(0, high = n_dataset_boundary, size = n_batch)
		sta_pos_boundary = torch.Tensor(Locs_samples_boundary[isample_boundary]).to(device)
		src_pos_boundary = torch.Tensor(X_samples_boundary[isample_boundary]).to(device)
		pred_base_boundary, pred_perturb_boundary, src_pos_boundary_cart, src_embed_boundary = m(sta_pos_boundary, src_pos_boundary, method = 'direct', train = True)
		assert(pred_base_boundary.max().item() < 1e-2)
		loss_boundary = loss_func(pred_perturb_boundary, torch.zeros(pred_perturb_boundary.shape).to(device))

		# Composite loss (Original 0.25 weights preserved)
		if i > n_burn_in:
			loss = 0.25 * loss_pde_p + 0.25 * loss_pde_s + 0.5 * loss_boundary
		else:
			loss = loss_boundary

		if compute_reference_times == True:
			trgt_data = torch.Tensor(norm_time(np.concatenate((Tp_samples[isample].reshape(-1,1), Ts_samples[isample].reshape(-1,1)), axis = 1))).to(device)
			pred_data = pred_base + pred_perturb
			loss_data = loss_func1(pred_data, trgt_data)
			loss = 0.5 * loss + 0.5 * loss_data

		if use_causual_loss == True:
			sign_offset = 1.0 * (torch.sign(pred_base + pred_perturb) != 1)
			loss_causual = loss_func(torch.zeros(pred_base.shape).to(device), sign_offset * (pred_base + pred_perturb))
			loss = 0.99 * loss + 0.01 * loss_causual

		if use_initial_model_damping == True:
			trgt_vel = norm_vel(torch.Tensor(np.concatenate((Vp_samples[isample].reshape(-1,1), Vs_samples[isample].reshape(-1,1)), axis = 1))).to(device)
			loss_initial = loss_func(trgt_vel, vel_pred)
			loss = 0.85 * loss + 0.15 * loss_initial

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

			if compute_reference_times == True:
				travel_times_p = Tp_samples_vald[isample]
				travel_times_s = Ts_samples_vald[isample]

			pred_base, pred_perturb, src_pos_cart, src_embed = m(sta_pos, src_pos, method = 'direct', train = True)
			vel_pred = m.vmodel(src_pos_cart, src_embed)

			pred_total_p = pred_base[:, [0]] + pred_perturb[:, [0]]
			pred_total_s = pred_base[:, [1]] + pred_perturb[:, [1]]

			grad_p = torch.autograd.grad(
				inputs = src_pos_cart, 
				outputs = pred_total_p, 
				grad_outputs = torch_one_vec, 
				retain_graph = True, 
				create_graph = False
			)[0]

			grad_s = torch.autograd.grad(
				inputs = src_pos_cart, 
				outputs = pred_total_s, 
				grad_outputs = torch_one_vec, 
				retain_graph = False, 
				create_graph = False
			)[0]

			grad_sq_p = (grad_p * grad_p).sum(dim=1)
			grad_sq_s = (grad_s * grad_s).sum(dim=1)

			eps = 1e-6
			grad_s_p = torch.sqrt(grad_sq_p + eps)
			grad_s_s = torch.sqrt(grad_sq_s + eps)

			target_slowness_p = 1.0 / (scale_norm_factor * vel_pred[:, 0])
			target_slowness_s = 1.0 / (scale_norm_factor * vel_pred[:, 1])

			loss_weight = torch.exp(-torch.abs(inorm_time(pred_base[:, 0])) / t_scale_val)
			loss_weight = loss_weight / loss_weight.sum()

			loss_pde_p = 0.5 * loss_func(grad_s_p, target_slowness_p) + \
			             0.5 * loss_func(loss_weight * grad_s_p, loss_weight * target_slowness_p)

			loss_pde_s = 0.5 * loss_func(grad_s_s, target_slowness_s) + \
			             0.5 * loss_func(loss_weight * grad_s_s, loss_weight * target_slowness_s)

			## Boundary loss
			if n_dataset_boundary_vald > 0:
				isample_boundary = np.random.randint(0, high = n_dataset_boundary_vald, size = n_batch)
				sta_pos_boundary = torch.Tensor(Locs_samples_boundary_vald[isample_boundary]).to(device)
				src_pos_boundary = torch.Tensor(X_samples_boundary_vald[isample_boundary]).to(device)
				pred_base_boundary, pred_perturb_boundary, src_pos_boundary_cart, src_embed_boundary = m(sta_pos_boundary, src_pos_boundary, method = 'direct', train = True)
				assert(pred_base_boundary.max().item() < 1e-2)
				loss_boundary = loss_func(pred_perturb_boundary, torch.zeros(pred_perturb_boundary.shape).to(device))
			else:
				loss_boundary = torch.zeros(1).to(device)

			if i > n_burn_in:
				loss_vald = 0.25 * loss_pde_p + 0.25 * loss_pde_s + 0.5 * loss_boundary
			else:
				loss_vald = loss_boundary

			if compute_reference_times == True:
				trgt_data = torch.Tensor(norm_time(np.concatenate((Tp_samples_vald[isample].reshape(-1,1), Ts_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)
				pred_data = pred_base + pred_perturb
				loss_data = loss_func1(pred_data, trgt_data)
				loss_vald = 0.5 * loss_vald + 0.5 * loss_data

			if use_causual_loss == True:
				sign_offset = 1.0 * (torch.sign(pred_base + pred_perturb) != 1)
				loss_causual = loss_func(torch.zeros(pred_base.shape).to(device), sign_offset * (pred_base + pred_perturb))
				loss_vald = 0.99 * loss_vald + 0.01 * loss_causual

			if use_initial_model_damping == True:
				trgt_vel = norm_vel(torch.Tensor(np.concatenate((Vp_samples_vald[isample].reshape(-1,1), Vs_samples_vald[isample].reshape(-1,1)), axis = 1))).to(device)
				loss_initial = loss_func(trgt_vel, vel_pred)
				loss_vald = 0.85 * loss_vald + 0.15 * loss_initial

			print('%d %0.8f %0.8f' % (i, loss.item(), loss_vald.item()))
			losses_vald.append(loss_vald.item())
			losses_pde_vald.append(loss_pde_p.item() + loss_pde_s.item())
			losses_data_vald.append(loss_data.item())

			optimizer.zero_grad()


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


	m = m.cpu()
	torch.save(m.state_dict(), path_save + 'travel_time_neural_network_physics_informed_%s_ver_%d.h5'%(phase, n_ver_save))
	torch.save(optimizer.state_dict(), path_save + 'travel_time_neural_network_physics_informed_%s_optimizer_ver_%d.h5'%(phase, n_ver_save))
	np.savez_compressed(path_save + 'travel_time_neural_network_physics_informed_%s_losses_ver_%d.npz'%(phase, n_ver_save), out1 = out1, out2 = out2, trgt1 = trgt1, trgt2 = trgt2, sta_pos1 = sta_pos1, src_pos1 = src_pos1, sta_pos2 = sta_pos2, src_pos2 = src_pos2, v_mean = v_mean, scale_params = scale_params, r_vals = r_vals, losses = losses, losses_vald = losses_vald)
	m = m.to(device)

	make_plot = True
	if make_plot == True:

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
	x11, x12, x13 = np.meshgrid(x1, x2, x3)
	x_contour = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
	trv_out = m(torch.Tensor(x_contour).to(device), torch.Tensor(locs).to(device))


	## Make transects and profiles of velocity
	x1 = np.array([locs[:,0].mean()])
	x2 = np.array([locs[:,1].mean()])
	x3 = np.arange(depth_range[0], depth_range[1], 50.0)
	x11, x12, x13 = np.meshgrid(x1, x2, x3)
	xx1 = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)

	pred_base, pred_perturb, src_pos_cart, src_embed = m(torch.Tensor(locs[[0],:]).to(device).repeat(len(xx1),1), torch.Tensor(xx1).to(device), method = 'direct', train = True)
	vel_pred = inorm_vel(m.vmodel(src_pos_cart, src_embed).cpu().detach().numpy())

	vel_pred_levels = []
	for i in range(len(x3)):
		xv = np.copy(x_contour)
		xv[:,2] = x3[i]
		pred_base, pred_perturb, src_pos_cart, src_embed = m(torch.Tensor(locs[[0],:]).to(device).repeat(len(xv),1), torch.Tensor(xv).to(device), method = 'direct', train = True)
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


	remove_travel_time_files = True
	if remove_travel_time_files == True:
		for s in st_sta:
			os.remove(s)

print("All files saved successfully!")
print("✔ Script execution: Done")

