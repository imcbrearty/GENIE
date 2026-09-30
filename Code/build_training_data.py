import glob
import h5py
import os
import pathlib
import sys
import time
import yaml

import numpy as np

# PyTorch & PyG Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_cluster import knn
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.utils import degree, remove_self_loops, softmax, subgraph
from torch_scatter import scatter
from obspy.core import UTCDateTime

# SciPy Utilities
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import beta, chi2, gamma
from sklearn.neighbors import KernelDensity
from torch_geometric.utils import degree

# Custom Project Imports
from module import *
from utils import *
from graph_utils import *

# Set primary CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Optional Experiment Tracking
use_wandb_logging = False
if use_wandb_logging:
	import wandb

	wandb.init(project="GENIE")


# -----------------------------------------------------------------------------
# 1. Configuration & Global Environment Setup
# -----------------------------------------------------------------------------
with open("config.yaml", "r") as file:
	config = yaml.safe_load(file)

with open("train_config.yaml", "r") as file:
	train_config = yaml.safe_load(file)

name_of_project = config["name_of_project"]
path_to_file = str(pathlib.Path().absolute())
seperator = "\\" if "\\" in path_to_file else "/"
path_to_file += seperator


# Graph Structure & Toggle Settings
k_sta_edges = config["k_sta_edges"]
k_spc_edges = config["k_spc_edges"]
use_topography = config.get("use_topography", False)
path_to_data = train_config.get('path_to_data', './')
graph_params = [k_sta_edges, k_spc_edges, k_time_edges]

# path_to_data = '/scratch/users/imcbrear/DetectionModel/TrainingData/'

use_subgraph = True
use_time_shift = True
use_physics_informed = True

use_phase_types = config["use_phase_types"]
use_expanded = config["use_phase_types"]

# Versions
template_ver = train_config["template_ver"]
vel_model_ver = train_config["vel_model_ver"]
n_ver = train_config["n_ver"]

# Training parameters
n_batch = train_config["n_batch"]
n_epochs = train_config["n_epochs"]
n_spc_query = train_config["n_spc_query"]
n_src_query = train_config["n_src_query"]
training_params = [n_spc_query, n_src_query]
n_ver_training_data = train_config.get('n_ver_training_data', 1)

use_real_data = train_config.get('use_real_data', True)
n_real_fraction = train_config.get('n_real_fraction', 0.3)



## Main settings (e.g., on / off switches)



## Event rate paremeters



## Domain parameters (e.g., random domain clipping / corruption)



## Graph parameters (subgraph, optimize, etc.)



## Real training data parameters



## Extra: Multiple host "domains" or folders / travel time functions / depth ranges



## Base parameters (e.g., base kernel sizes if no adaptive domain)





# Device Setup
device = torch.device(config["device"])
if not torch.cuda.is_available():
	print("No GPU available, falling back to CPU execution.")
	device = torch.device("cpu")
else:
	torch.cuda.set_device(0)
	torch.ones(1).cuda()  # Warm up GPU context

# -----------------------------------------------------------------------------
# 2. Regional Domain, Networks, and Coordinate Matrices
# -----------------------------------------------------------------------------
z_region = np.load(path_to_file + "%s_region.npz" % name_of_project)
lat_range, lon_range, depth_range, deg_pad = (
	z_region["lat_range"],
	z_region["lon_range"],
	z_region["depth_range"],
	z_region["deg_pad"],
)
z_region.close()

# Station Geometry Matrix
z_stations = np.load(path_to_file + "%s_stations.npz" % name_of_project)
locs, stas, mn, rbest = (
	z_stations["locs"],
	z_stations["stas"],
	z_stations["mn"],
	z_stations["rbest"],
)
z_stations.close()

rbest_cuda = torch.Tensor(rbest).to(device)
mn_cuda = torch.Tensor(mn).to(device)

# Seismic Discretization Templates
z_templates = np.load(
	path_to_file + f"Grids/{name_of_project}_seismic_network_templates_ver_{template_ver}.npz"
)
x_grids = z_templates["x_grids"]
scale_time = z_templates["scale_time"] / 1000.0
time_shift_range = z_templates["time_shift_range"]
Ac = z_templates["Ac"]
z_templates.close()

# Projection functions (Sphere vs Earth Ellipsoid)
if config.get("use_spherical", False):
	earth_radius = 6371e3
	ftrns1 = lambda x: (rbest @ (lla2ecef(x, e=0.0, a=earth_radius) - mn).T).T
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e=0.0, a=earth_radius)
	ftrns1_diff = lambda x: (
		rbest_cuda @ (lla2ecef_diff(x, e=0.0, a=earth_radius, device=device) - mn_cuda).T
	).T
	ftrns2_diff = lambda x: ecef2lla_diff(
		(rbest_cuda.T @ x.T).T + mn_cuda, e=0.0, a=earth_radius, device=device
	)
else:
	earth_radius = 6378137.0
	ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T
	ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)
	ftrns1_diff = lambda x: (
		rbest_cuda @ (lla2ecef_diff(x, device=device) - mn_cuda).T
	).T
	ftrns2_diff = lambda x: ecef2lla_diff(
		(rbest_cuda.T @ x.T).T + mn_cuda, device=device
	)

if use_topography:
	surface_profile = np.load(
		path_to_file + f"Grids/{name_of_project}_surface_elevation.npz"
	)["surface_profile"]
	tree_surface = cKDTree(surface_profile[:, 0:2])

# -----------------------------------------------------------------------------
# 3. Hyperparameter Kernels & Adaptive Windowing
# -----------------------------------------------------------------------------
kernel_sig_t = train_config["kernel_sig_t"]
src_t_kernel = train_config["src_t_kernel"]
src_t_arv_kernel = train_config["src_t_arv_kernel"]
src_x_kernel = train_config["src_x_kernel"]
src_x_arv_kernel = train_config["src_x_arv_kernel"]
src_depth_kernel = train_config["src_depth_kernel"]

src_kernel_mean = np.mean([src_x_kernel, src_x_kernel, src_depth_kernel])
src_spatial_kernel = np.array(
	[src_x_kernel, src_x_kernel, src_depth_kernel]
).reshape(1, 1, -1)

use_adaptive_window = True
if use_adaptive_window:
	n_resolution = 9
	t_win = np.round(np.copy(np.array([2 * src_t_kernel]))[0], 2)
	dt_win = np.diff(np.linspace(-t_win / 2.0, t_win / 2.0, n_resolution))[0]
else:
	dt_win = 1.0
	t_win = 10.0

pred_params = [
	t_win,
	kernel_sig_t,
	src_t_kernel,
	src_x_kernel,
	src_depth_kernel,
]

# -----------------------------------------------------------------------------
# 4. Fixed Subnetworks Loading
# -----------------------------------------------------------------------------
load_subnetworks = train_config.get("fixed_subnetworks", False)
Ind_subnetworks = None

if load_subnetworks:
	subnetwork_path = path_to_file + f"{name_of_project}_subnetworks.hdf5"
	if os.path.exists(subnetwork_path):
		with h5py.File(subnetwork_path, "r") as h_sub:
			Ind_subnetworks = [h_sub[k][:].astype("int") for k in h_sub.keys()]

# -----------------------------------------------------------------------------
# 5. Pack Parameter Vectors for Synthetic Data Generators
# -----------------------------------------------------------------------------
training_params_2 = [
	train_config["spc_random"],
	train_config["sig_t"],
	train_config["spc_thresh_rand"],
	train_config["min_sta_arrival"],
	train_config["coda_rate"],
	np.array(train_config["coda_win"]),
	train_config["max_num_spikes"],
	train_config["spike_time_spread"],
	train_config["s_extra"],
	train_config["use_stable_association_labels"],
	train_config["thresh_noise_max"],
	train_config["min_misfit_allowed"],
	train_config["total_bias"],
]

dist_range = train_config["dist_range"]
max_rate_events = train_config["max_rate_events"]
max_miss_events = train_config["max_miss_events"]
max_false_events = max_rate_events * train_config["max_false_events"]

training_params_3 = [
	train_config.get("n_batch", 32),
	dist_range,
	max_rate_events,
	max_miss_events,
	max_false_events,
	train_config["miss_pick_fraction"],
	train_config["T"],
	train_config["dt"],
	train_config["tscale"],
	train_config["n_sta_range"],
	train_config["use_sources"],
	train_config["use_full_network"],
	Ind_subnetworks if Ind_subnetworks else False,
	train_config["use_preferential_sampling"],
	train_config["use_shallow_sources"],
	train_config["use_extra_nearby_moveouts"],
]

min_sta_ref = int(train_config["n_sta_range"][0] * len(locs))

print(
	f"Initialization complete. Loaded {len(locs)} stations across project `{name_of_project}`."
)



def generate_synthetic_data(
	trv,
	locs,
	spatial_bounds,
	ftrns1=None,
	use_travel_time_noise = True,
	src_spatial_kernel=10.0,
	src_t_kernel=1.0,
	duration=100.0,
	event_rate=0.03,
	dist_thresh_range=(20.0, 350.0),
	total_bias=0.03,
	drop_pick_prob=0.2,
	coda_rate=0.025,
	coda_win=(1.0, 20.0),
	perturb_phase = [0.1,0.25],
	false_pick_rate=0.02,
	spc_thresh_rand=8.0,
	min_picks = 6,
	min_sta = 4,
	device = 'cpu'
):
	"""Generates synthetic pick dataset along with target source labels for optional

	query points.
	"""
	if ftrns1 is None:
		ftrns1 = lambda x: x

	n_sta = len(locs)
	locs_cart = ftrns1(locs)

	# 1. Sample synthetic sources in LLA spatial bounds & time
	n_src = max(1, np.random.poisson(duration * event_rate))
	src_times = np.sort(np.random.uniform(0, duration, size=n_src))

	(lat_min, lat_max), (lon_min, lon_max), (z_min, z_max) = spatial_bounds
	src_lat = np.random.uniform(lat_min, lat_max, size=n_src)
	src_lon = np.random.uniform(lon_min, lon_max, size=n_src)
	src_z = np.random.uniform(z_min, z_max, size=n_src)
	src_mag = np.random.uniform(-1.0, 7.0, size=n_src)

	src_lla = np.column_stack([src_lat, src_lon, src_z])
	sources = np.column_stack([src_lla, src_times, src_mag])
	src_cart = ftrns1(src_lla)

	# 2. Compute theoretical relative travel times (using LLA inputs)
	tt_raw = trv(torch.Tensor(locs).to(device), torch.Tensor(src_lla).to(device)).cpu().detach().numpy()

	# Apply event-level velocity scale bias
	std_scale_p = total_bias / 2.0
	scale_bias_p = np.random.normal(0.0, std_scale_p, size=(n_src, 1, 1))
	scale_bias_s_ratio = (
		np.random.normal(0.0, std_scale_p, size=(n_src, 1, 1)) * 0.3
	)
	scale_multiplier = 1.0 + np.concatenate(
		[scale_bias_p, scale_bias_p + scale_bias_s_ratio], axis=2
	)

	tt_biased = tt_raw * scale_multiplier

	# Apply fine-grained travel-time noise if provided
	if use_travel_time_noise:
		shape1, shape2 = tt_biased.shape[0], tt_biased.shape[1]
		noise_p = generate_travel_time_noise(
			tt_biased[:, :, 0].reshape(-1), phase_input="P", distribution="laplace"
		)[0]
		noise_s = generate_travel_time_noise(
			tt_biased[:, :, 1].reshape(-1), phase_input="S", distribution="laplace"
		)[0]
		tt_biased[:, :, 0] += noise_p.reshape(shape1, shape2)
		tt_biased[:, :, 1] += noise_s.reshape(shape1, shape2)

	abs_arrivals = tt_biased + src_times[:, None, None]

	# 3. Calculate spatial thresholds using Cartesian coordinates
	sr_distances = np.linalg.norm(
		src_cart[:, None, :] - locs_cart[None, :, :], axis=2
	)/1000.0
	dist_thresh = np.random.uniform(
		dist_thresh_range[0], dist_thresh_range[1], size=(n_src, 1)
	)

	dist_thresh_p = dist_thresh + spc_thresh_rand * np.random.laplace(
		size=(n_src, 1)
	)
	dist_thresh_s = dist_thresh + spc_thresh_rand * np.random.laplace(
		size=(n_src, 1)
	)

	p_mask = sr_distances < dist_thresh_p
	s_mask = sr_distances < dist_thresh_s

	p_indices = np.argwhere(p_mask)
	s_indices = np.argwhere(s_mask)

	# 4. Assemble candidate true picks
	p_picks = np.column_stack(
		[
			abs_arrivals[p_indices[:, 0], p_indices[:, 1], 0],
			p_indices[:, 1],
			p_indices[:, 0],
			src_times[p_indices[:, 0]],
			np.zeros(len(p_indices)),
		]
	)

	s_picks = np.column_stack(
		[
			abs_arrivals[s_indices[:, 0], s_indices[:, 1], 1],
			s_indices[:, 1],
			s_indices[:, 0],
			src_times[s_indices[:, 0]],
			np.ones(len(s_indices)),
		]
	)

	arrivals = (
		np.vstack([p_picks, s_picks])
		if len(p_picks) or len(s_picks)
		else np.empty((0, 5))
	)

	# 5. Randomly drop picks
	if len(arrivals) > 0:
		drop_mask = np.random.rand(len(arrivals)) < drop_pick_prob
		arrivals = arrivals[~drop_mask]

	# 6. Generate false Coda arrivals
	coda_picks = []
	if len(arrivals) > 0 and coda_rate > 0:
		coda_mask = np.random.rand(len(arrivals)) < coda_rate
		n_coda = np.sum(coda_mask)
		if n_coda > 0:
			coda_times = arrivals[coda_mask, 0] + np.random.uniform(
				coda_win[0], coda_win[1], size=n_coda
			)
			coda_picks = np.column_stack(
				[
					coda_times,
					arrivals[coda_mask, 1],
					-1.0 * np.ones(n_coda),
					np.zeros(n_coda),
					-1.0 * np.ones(n_coda),
				]
			)

	# 7. Generate uncorrelated false arrivals
	n_false = np.random.poisson(false_pick_rate * duration * n_sta)
	false_times = np.random.uniform(0, duration, size=n_false)
	false_stas = np.random.choice(n_sta, size=n_false)
	false_picks = np.column_stack(
		[
			false_times,
			false_stas,
			-1.0 * np.ones(n_false),
			np.zeros(n_false),
			-1.0 * np.ones(n_false),
		]
	)

	# Combine and sort picks by time
	all_groups = [p for p in [arrivals, coda_picks, false_picks] if len(p) > 0]
	picks = (
		np.vstack(all_groups)[np.vstack(all_groups)[:, 0].argsort()]
		if len(all_groups) > 0
		else np.empty((0, 5))
	)

	# # 8. Compute target query labels if grid points are provided
	# labels = None
	# if x_query is not None and x_query_t is not None:
	# 	labels = compute_source_labels(
	# 		x_query,
	# 		x_query_t,
	# 		src_lla,
	# 		src_times,
	# 		src_spatial_kernel,
	# 		src_t_kernel,
	# 		ftrns1,
	# 	)


	## Subset active sources
	tree = cKDTree(picks[:,2].reshape(-1,1))
	if len(sources) > 0:
		ip_list = tree.query_ball_point(np.arange(len(sources)).reshape(-1,1), r = 0)
		count_picks = np.array([len(ip_list[j]) for j in range(len(ip_list))]) # np.array([len(ip) for ip in tree.query_ball_point(np.)])
		count_sta = np.array([len(np.unique(picks[ip_list[j],1])) for j in range(len(ip_list))])
		iwhere_source = np.where((count_picks >= min_picks)*(count_sta >= min_sta))[0]
		perm_source = (-1*np.ones(len(sources))).astype('int')
		perm_source[iwhere_source] = np.arange(len(iwhere_source))
		source_inds = picks[:,2].astype('int')
		valid_inds = source_inds >= 0
		picks[valid_inds,2] = perm_source[source_inds[valid_inds]]
		sources = sources[iwhere_source]


	# count_picks = np.bincount(picks[picks[:,2] > -1,2], minlength = len(sources)) if len(sources) > 0 else np.zeros(0)

	picks_true = np.copy(picks)
	irand = np.where(picks[:,4] == -1)[0]
	picks[irand,4] = np.random.choice([0,1], size = len(irand))
	if perturb_phase[0] > 0:
		iflip = np.random.choice(len(picks), size = min(len(picks), int(len(picks)*np.random.uniform(perturb_phase[0], perturb_phase[1]))), replace = False)
		picks[iflip,4] = np.mod(picks[iflip,4] + 1, 2)

	return picks, picks_true, sources



def generate_sample_with_domain_slice(
	trv_fn,
	locs_ref,
	lat_range_full,
	lon_range_full,
	depth_range_full,
	ftrns1,
	ftrns2,
	n_spc_query=8000,
	min_domain_fraction=0.1,
	station_keep_range=(0.2, 1.0),
	time_window_W=10.0,
	duration=3*3600.0,
	active_source_bias_prob=0.5,
	n_frac_focused_queries=0.2,
	n_frac_random_focused=0.2,
	src_x_kernel=10000.0,
	src_depth_kernel=5000.0,
	src_t_kernel=1.0,
	src_spatial_kernel=None,
	use_global=False,
	# **synthetic_data_kwargs,
	x_grid = None,
	rbest = None,
	mn = None,
	max_t = 200.0,
	min_sta_domain = 10,
	min_picks = 6,
	min_sta = 4,
	n_samples = 30,
	estimate_domain_params = True
):
	"""Wrapper function to crop domain slices, sub-sample stations, generate

	synthetic picks, center a time window, and compute spatio-temporal query
	labels.
	"""
	# -------------------------------------------------------------------------
	# 1. Generate Domain Slice (Crop Lat/Lon Bounds with Aspect Ratio Shift)
	# -------------------------------------------------------------------------

	cnt_slice, found_slice = 0, False
	while (found_slice == False) and cnt_slice < 30:

		cnt_slice += 1
		lat_span_full = lat_range_full[1] - lat_range_full[0]
		lon_span_full = lon_range_full[1] - lon_range_full[0]	

		# Sample slice fractions allowing aspect-ratio distortion
		frac_lat = np.random.uniform(min_domain_fraction, 1.0)
		frac_lon = np.random.uniform(min_domain_fraction, 1.0)	

		slice_lat_span = lat_span_full * frac_lat
		slice_lon_span = lon_span_full * frac_lon	

		# Randomly select centroids within allowable bounds
		lat_min = np.random.uniform(
			lat_range_full[0], lat_range_full[1] - slice_lat_span
		)
		lat_max = lat_min + slice_lat_span	

		lon_min = np.random.uniform(
			lon_range_full[0], lon_range_full[1] - slice_lon_span
		)
		lon_max = lon_min + slice_lon_span	

		spatial_bounds_slice = (
			(lat_min, lat_max),
			(lon_min, lon_max),
			(depth_range_full[0], depth_range_full[1]),
		)	

		# -------------------------------------------------------------------------
		# 2. Sub-sample Station Subset within Domain Slice
		# -------------------------------------------------------------------------
		# Identify stations falling inside the domain slice
		in_slice_mask = (
			(locs_ref[:, 0] >= lat_min)
			& (locs_ref[:, 0] <= lat_max)
			& (locs_ref[:, 1] >= lon_min)
			& (locs_ref[:, 1] <= lon_max)
		)	

		slice_station_indices = np.where(in_slice_mask)[0]

		if len(slice_station_indices) >= min_sta_domain:
			found_slice = True

	if (cnt_slice == 30) and found_slice == False:
		raise ValueError("Slice count reached 30 but no slice was found.")


	# Randomly retain a fraction of the available stations (e.g. 20% to 100%)
	keep_fraction = np.random.uniform(
		station_keep_range[0], station_keep_range[1]
	)
	n_select = max(
		min_sta_domain, int(np.ceil(len(slice_station_indices) * keep_fraction))
	)
	selected_sta_idx = np.random.choice(
		slice_station_indices, size=n_select, replace=False
	)
	sub_locs = locs_ref[selected_sta_idx]
	stas_use = np.array([str(j) for j in selected_sta_idx])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	### Estimate dynamic domain parameters ###
	if estimate_domain_params == True:

		Vc = 6500.0 # Vc = 3500.0
		scale_domain = 1.05
		deg_padding = np.nan ## Use hueristic
		n_trgt_nodes = config.get('n_trgt_nodes', int(200e3))
		num_grids = 1 # config.get('number_of_grids', 1)
		number_of_spatial_nodes = config.get('number_of_spatial_nodes', 5000)
		optimize_source_graphs = False # config.get('optimize_source_graphs', False)
		optimize_station_graphs = False # config.get('optimize_station_graphs', False)
		use_paths = False # config.get('use_paths', False) # process_config.get('use_paths', False) # False
		use_tuner = False # config.get('use_tuner', True) # process_config.get('use_tuner', True) # False
		use_domain_approximate = False # config.get('use_domain_approximate', False)
		m_domain, initialize = None, None	

		data_save, domain_params = build_graphs_domain(m_domain, sub_locs, stas_use, scale_domain, deg_padding, number_of_spatial_nodes, config['k_spc_edges'], config['k_sta_edges'], depth_range_full, 
			ftrns1, ftrns2, use_global = use_global, assign_based_on_grid = False, max_nodes = number_of_spatial_nodes, n_trgt_nodes = n_trgt_nodes, n_grids = num_grids, Vc = Vc, rbest = rbest, 
			mn = mn, file_index = 0, date = UTCDateTime(2000, 1, 1), use_paths = use_paths, optimize_station_graphs = optimize_station_graphs, optimize_source_graphs = optimize_source_graphs, 
			use_domain_approximate = use_domain_approximate, initialize = initialize, name_of_project = '', use_tuner = use_tuner, device = device)

		x_grid, scale_time, kernel_sig_t, src_spatial_kernel, src_x_kernel, src_depth_kernel, src_t_kernel = data_save['x_grid'], data_save['scale_time'], data_save['sigma_input'], data_save['source_label_width'], data_save['source_label_width'], data_save['source_label_width'], data_save['source_label_width_t']
		src_x_arv_kernel, src_t_arv_kernel = data_save['src_x_arv_kernel'], data_save['src_t_arv_kernel']


		time_window_W = (x_grid[:,3].max() - x_grid[:,3].min())/2.0
		max_t = compute_travel_times(trv_fn, sub_locs, np.expand_dims(x_grid, axis = 0), device = device)[0].max()*1.05
		# pdb.set_trace()

		lat_range, lon_range, depth_range = data_save['lat_range'], data_save['lon_range'], data_save['depth_range']
		lat_range_extend, lon_range_extend = data_save['lat_range_extend'], data_save['lon_range_extend']
		deg_padding = data_save['deg_padding']

		spatial_bounds_slice = (
			(lat_range_extend[0], lat_range_extend[1]),
			(lon_range_extend[0], lon_range_extend[1]),
			(depth_range[0], depth_range[1]),
		)	
		lat_min, lat_max = spatial_bounds_slice[0][0], spatial_bounds_slice[0][1]
		lon_min, lon_max = spatial_bounds_slice[1][0], spatial_bounds_slice[1][1]

		# lat_range, lon_range, depth_range = z_dom["lat_range"], z_dom["lon_range"], z_dom["depth_range"]
		# lat_range_extend = z_dom["lat_range_extend"]
		# lon_range_extend = z_dom["lon_range_extend"]
		# deg_pad = z_dom["deg_padding"]
		# time_shift_range = z_dom["time_shift_range"]

		use_perm_expand = False
		if use_perm_expand == True:
			perm_vec_expand = np.random.permutation(np.arange(x_grid.shape[0])).astype('int')
			Ac_src_src = torch.Tensor(data_save['Ac'].copy()).long().to(device)
		else:
			perm_vec_expand = np.arange(x_grid.shape[0]).astype('int')
			Ac_src_src = torch.Tensor(data_save['Ac'].copy()).long().to(device)
		Ac_prod_src_src = build_src_src_product(Ac_src_src, torch.Tensor(data_save['A_src_in_sta'][0:2,:]).to(device).long(), data_save['locs_use'], x_grid, device = device)

		data_save['Ac_src'] = Ac_src_src[0:2,:].cpu().detach().numpy().astype('int')
		data_save['Ac_prod_src_src'] = Ac_prod_src_src[0:2,:].cpu().detach().numpy().astype('int')


		spatial_vals = torch.cat((torch.Tensor((ftrns1(x_grid[data_save['A_src_in_sta'][1,:].astype('int')][:,0:3]) - 
			ftrns1(data_save['locs_use'][data_save['A_src_in_sta'][0,:].astype('int')]))/(30*data_save['source_label_width'])).to(device), 
			torch.Tensor(x_grid[data_save['A_src_in_sta'][1,:].astype('int')][:,[3]]).to(device)/(10.0*data_save['source_label_width_t'])), dim = 1).cpu().detach().numpy()

		data_save['spatial_vals'] = spatial_vals



	# -------------------------------------------------------------------------
	# 3. Call Synthetic Data Generator for Selected Domain/Station Slice
	# -------------------------------------------------------------------------

	## Set some per domain adaptive parameters - e.g., max moveouts

	picks, picks_true, sources = generate_synthetic_data(
		trv=trv_fn,
		locs=sub_locs,
		spatial_bounds=spatial_bounds_slice,
		ftrns1=ftrns1,
		duration=duration,
		src_spatial_kernel=(
			src_spatial_kernel
			if src_spatial_kernel is not None
			else src_x_kernel
		),
		src_t_kernel=src_t_kernel,
		min_picks = min_picks,
		min_sta = min_sta,
		device = device
		# **synthetic_data_kwargs,
	)

	sample_dict_l = {}


	for n in range(n_samples):

		# -------------------------------------------------------------------------
		# 4. Choose Origin Time Window [-W, +W]
		# -------------------------------------------------------------------------
		if len(sources) > 0 and (np.random.rand() < active_source_bias_prob):
			# Biased selection near an active source origin time
			chosen_src_idx = np.random.choice(len(sources))
			center_time = sources[chosen_src_idx, 3] + np.random.uniform(-time_window_W, time_window_W) # np.random.randn()*(time_window_W/3)
		else:
			# Uniform random origin time selection
			center_time = np.random.uniform(0.0, duration)

		time_shift_range = [
			center_time - time_window_W,
			center_time + time_window_W,
		]

		# Filter picks that fall within the slice window
		if len(picks) > 0:
			valid_picks_mask = (picks[:, 0] >= time_shift_range[0]) & (
				picks[:, 0] <= (time_shift_range[1] + max_t)
			)
			slice_picks = np.copy(picks[valid_picks_mask])
			slice_picks_true = np.copy(picks_true[valid_picks_mask])
		else:
			slice_picks = np.copy(picks)
			slice_picks_true = np.copy(picks_true)

		# Filter sources that fall within the slice window
		if len(sources) > 0:
			# pdb.set_trace()
			valid_src_mask = ((sources[:, 3] >= (time_shift_range[0] - 0.2*time_window_W)) & 
				(sources[:, 3] <= (time_shift_range[1] + 0.2*time_window_W)))
			slice_sources = np.copy(sources[valid_src_mask])

			iwhere_source = np.where(valid_src_mask == 1)[0] # np.where((count_picks >= min_picks)*(count_sta >= min_sta))[0]
			perm_source = (-1*np.ones(len(sources))).astype('int')
			perm_source[iwhere_source] = np.arange(len(iwhere_source))
			source_inds = slice_picks[:,2].astype('int')
			valid_inds = source_inds >= 0
			slice_picks[valid_inds,2] = perm_source[source_inds[valid_inds]]
			slice_picks_true[valid_inds,2] = perm_source[source_inds[valid_inds]]

		else:
			slice_sources = np.copy(sources)

		apply_time_shift = True
		if apply_time_shift == True:
			slice_picks[:,0] = slice_picks[:,0] - center_time
			slice_picks_true[:,0] = slice_picks_true[:,0] - center_time
			slice_sources[:,3] = slice_sources[:,3] - center_time 
			time_shift_range = [time_shift_range[0] - center_time, time_shift_range[1] - center_time]

		# pdb.set_trace()

		# ---------------------------------
		# ----------------------------------------
		# 5. Generate Mixture Queries & Target Ground Truth Labels
		# -------------------------------------------------------------------------
		x_query, x_query_t = sample_random_queries(
			lp_srcs=slice_sources,
			n_src_query=n_spc_query,
			n_frac_focused=n_frac_focused_queries,
			n_frac_random_focused=n_frac_random_focused,
			src_x_kernel_m=src_x_kernel,
			src_depth_kernel_m=src_depth_kernel,
			src_t_kernel=src_t_kernel,
			lat_range=(lat_min, lat_max),
			lon_range=(lon_min, lon_max),
			depth_range=depth_range_full,
			time_shift_range=time_shift_range[1] - time_shift_range[0],
			is_global_lon=use_global,
		)

		x_src_query, x_src_query_t = sample_random_queries(
			lp_srcs=slice_sources,
			n_src_query=n_src_query,
			n_frac_focused=n_frac_focused_queries,
			n_frac_random_focused=n_frac_random_focused,
			src_x_kernel_m=src_x_arv_kernel,
			src_depth_kernel_m=src_depth_kernel,
			src_t_kernel=src_t_arv_kernel,
			lat_range=(lat_min, lat_max),
			lon_range=(lon_min, lon_max),
			depth_range=depth_range_full,
			time_shift_range=time_shift_range[1] - time_shift_range[0],
			is_global_lon=use_global,
		)

		labels = compute_source_labels(
			x_query=x_query,
			x_query_t=x_query_t,
			src_x=slice_sources[:, :3] if len(slice_sources) > 0 else np.empty((0, 3)),
			src_t=slice_sources[:, 3] if len(slice_sources) > 0 else np.empty((0,)),
			src_spatial_kernel=(
				src_spatial_kernel
				if src_spatial_kernel is not None
				else src_x_kernel
			),
			src_t_kernel=src_t_kernel,
			ftrns1=ftrns1,
		)
		mask_labels = (x_query[:,0:1] < lat_range[1])*(x_query[:,0:1] > lat_range[0])*(x_query[:,1:2] < lon_range[1])*(x_query[:,1:2] > lon_range[0])
		labels = labels*mask_labels

		# labels_grid = None
		# if x_grid is not None:
		labels_grid = compute_source_labels(
			x_query=x_grid[:,0:3],
			x_query_t=x_grid[:,3],
			src_x=slice_sources[:, :3] if len(slice_sources) > 0 else np.empty((0, 3)),
			src_t=slice_sources[:, 3] if len(slice_sources) > 0 else np.empty((0,)),
			src_spatial_kernel=(
				src_spatial_kernel
				if src_spatial_kernel is not None
				else src_x_kernel
			),
			src_t_kernel=src_t_kernel,
			ftrns1=ftrns1,
		)
		mask_labels = (x_grid[:,0:1] < lat_range[1])*(x_grid[:,0:1] > lat_range[0])*(x_grid[:,1:2] < lon_range[1])*(x_grid[:,1:2] > lon_range[0])
		labels_grid = labels_grid*mask_labels   	

		slice_picks_meta = slice_picks_true[:,[4,2]] ## phase, and source index
		pick_lbls = compute_pick_labels(x_src_query, x_src_query_t, slice_picks_meta, slice_sources, lat_range, lon_range, ftrns1, 
			sig_x = src_x_arv_kernel, sig_t = src_t_arv_kernel)

		sample_dict = {
			"picks": slice_picks,
			"picks_true": slice_picks_true,
			"sources": slice_sources,
			"x_query": x_query,
			"x_query_t": x_query_t,
			"x_src_query": x_src_query,
			"x_src_query_t": x_src_query_t,
			"labels": labels,
			"labels_grid" : labels_grid,
			"pick_lbls" : pick_lbls,
			"selected_sta_idx": selected_sta_idx,
			"spatial_bounds": spatial_bounds_slice,
			"center_time": center_time,
			"time_shift_range": time_window_W, # time_shift_range,
			"x_grid" : x_grid,
			"scale_time" : scale_time,
			"kernel_sig_t" : kernel_sig_t,
			"src_spatial_kernel" : src_spatial_kernel,
			"src_x_kernel" : src_x_kernel,
			"src_depth_kernel" : src_depth_kernel,
			"src_t_kernel" : src_t_kernel,
			"src_x_arv_kernel" : src_x_arv_kernel,
			"src_t_arv_kernel" : src_t_arv_kernel
		} # x_grid, scale_time, kernel_sig_t, src_spatial_kernel, src_x_kernel, src_depth_kernel, src_t_kernel


		for k in sample_dict.keys():
			sample_dict_l[k + '_%d'%n] = sample_dict[k]

		# pdb.set_trace()

	return sample_dict_l, data_save



def compute_travel_times(trv, locs, x_grids, n_max_chunks = int(50e3), device = 'cpu'):

	x_grids_trv = []
	# locs_cuda = torch.Tensor(locs).to(device)
	for i in range(len(x_grids)):
		
		n_sta, n_temp = len(locs), len(x_grids[i])
		n_chunks = int(np.maximum(1, int((n_sta*n_temp)/n_max_chunks)))
		n_int = max(int(len(locs)/n_chunks), 1)
		n_chunks = np.minimum(n_chunks, len(locs))
		inds = [np.arange(n_int) + n_int*j for j in range(n_chunks)]

		# pdb.set_trace()
		if len(inds) == 0: inds = np.arange(len(locs))
		if (inds[-1][-1] < len(locs))*(len(inds) > 1): inds[-1] = np.arange(inds[-2][-1] + 1, len(locs))
		if (inds[-1][-1] < len(locs))*(len(inds) == 1): inds[-1] = np.arange(0, len(locs))
		if inds[-1][-1] > (len(locs) - 1): inds[-1] = np.arange(inds[-1][0], len(locs))
		assert(np.abs(np.hstack(inds) - np.arange(len(locs))).max() == 0)
	
		trv_out_l = []
		x_grid_cuda = torch.Tensor(x_grids[i]).to(device)
		for j in range(len(inds)):
			# trv_out_l.append(trv(locs_cuda[inds[j]], x_grid_cuda).cpu().detach().numpy())
			trv_out_l.append(trv(torch.Tensor(locs[inds[j]]).to(device), x_grid_cuda).cpu().detach().numpy())
		# trv_out = np.concatenate(trv_out_l, axis = 1)
		x_grids_trv.append(np.concatenate(trv_out_l, axis = 1))

	return x_grids_trv



use_station_corrections = False
if use_station_corrections == True:
	n_ver_corrections = 1
	path_station_corrections = path_to_file + 'Grids' + seperator + 'station_corrections_ver_%d.npz'%n_ver_corrections
	if os.path.isfile(path_station_corrections) == False:
		print('No station corrections available')
		locs_corr, corrs = None, None
	else:
		z = np.load(path_station_corrections)
		locs_corr, corrs = z['locs_corr'], z['corrs']
		z.close()
else:
	locs_corr, corrs = None, None


	
if config['train_travel_time_neural_network'] == False:

	## Load travel times
	z = np.load(path_to_file + '1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, vel_model_ver))
	
	Tp = z['Tp_interp']
	Ts = z['Ts_interp']
	
	locs_ref = z['locs_ref']
	X = z['X']
	z.close()
	
	x1 = np.unique(X[:,0])
	x2 = np.unique(X[:,1])
	x3 = np.unique(X[:,2])
	assert(len(x1)*len(x2)*len(x3) == X.shape[0])
	
	## Load fixed grid for velocity models
	Xmin = X.min(0)
	Dx = [np.diff(x1[0:2]),np.diff(x2[0:2]),np.diff(x3[0:2])]
	Mn = np.array([len(x3), len(x1)*len(x3), 1]) ## Is this off by one index? E.g., np.where(np.diff(xx[:,0]) != 0)[0] isn't exactly len(x3)
	N = np.array([len(x1), len(x2), len(x3)])
	X0 = np.array([locs_ref[0,0], locs_ref[0,1], 0.0]).reshape(1,-1)
	
	trv = interp_1D_velocity_model_to_3D_travel_times(X, locs_ref, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2, device = device) # .to(device)

	z.close()

elif config['train_travel_time_neural_network'] == True:

	n_ver_trv_time_model_load = vel_model_ver # 1
	trv = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	# trv_pairwise = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)
	# trv_pairwise1 = load_travel_time_neural_network(path_to_file, ftrns1_diff, ftrns2_diff, n_ver_trv_time_model_load, method = 'direct', return_model = True, locs_corr = locs_corr, corrs = corrs, use_physics_informed = use_physics_informed, device = device)


## Check if knn is working on cuda
if device.type == 'cuda' or device.type == 'cpu':
	check_len = knn(torch.rand(10,3).to(device), torch.rand(10,3).to(device), k = 5).numel()
	if check_len != 100: # If it's less than 2 * 10 * 5, there's an issue
		raise SystemError('Issue with knn on cuda for some versions of pytorch geometric and cuda')

	check_len = knn(10.0*torch.rand(200,3).to(device), 10.0*torch.rand(100,3).to(device), k = 15).numel()
	if check_len != 3000: # If it's less than 2 * 10 * 5, there's an issue
		raise SystemError('Issue with knn on cuda for some versions of pytorch geometric and cuda')



# lat_range_interior = [lat_range[0], lat_range[1]]
# lon_range_interior = [lon_range[0], lon_range[1]]


build_training_data = True
if build_training_data == True:


	n_repeat = train_config['n_batches_per_job'] ## Number of batches to make per job

	argvs = sys.argv
	if len(argvs) < 2:
		argvs.append(0)

	job_number = int(argvs[1]) ## Choose job index

	print('Build and save training data on job index %d'%job_number)

	print('Note set t_win in input')


	oversample_factor = 30
	n_samples = min(oversample_factor, n_repeat) ## Oversample per graph; disribute to training files
	n_repeat_loop = n_repeat // n_samples
	total_files = int(n_repeat_loop*n_samples)

	## Will use M base files to create a batch file of m_samples; over sample by n_samples, so we create n_samples per outer loop
	file_inc = 0

	for n in range(n_repeat_loop):

		## Each of these loops will create n_samples new files
		sample_dict_store = []
		data_save_store = []

		for batch_ind in range(n_batch):

			# file_index = n_repeat*job_number + n ## Unique file index

			# if os.path.isfile(path_to_data + 'training_data_slice_%d_ver_%d.hdf5'%(file_index, n_ver_training_data)) == True:
			# 	continue

			print('Make source labels require > min stations')
			print('Need to mask active sources')
			sample_dict, data_save = generate_sample_with_domain_slice(
				trv,
				locs,
				lat_range,
				lon_range,
				depth_range,
				ftrns1,
				ftrns2,
				n_spc_query=n_spc_query, # 8000
				min_domain_fraction=0.1,
				station_keep_range=(0.2, 1.0),
				time_window_W=10.0,
				duration=3600.0,
				active_source_bias_prob=0.5,
				n_frac_focused_queries=0.2,
				n_frac_random_focused=0.2,
				src_x_kernel=10000.0,
				src_depth_kernel=10000.0,
				src_t_kernel=1.5,
				src_spatial_kernel=None,
				use_global=False,
				# **synthetic_data_kwargs,
				min_sta_domain = 10,
				n_samples = n_samples,
				estimate_domain_params = True)

			## Remove edge weight information
			for k, d in data_save.items():
				if ((k[0:2] == 'A_') or (k[0:3] == 'Ac_')) and (d.shape[0] > 2) and np.ndim(d) > 1:
					data_save[k] = d[0:2,:].astype('int')

			## Save global graph information
			data_save_store.append(data_save)

			locs_use = data_save['locs_use']
			A_src_in_sta = data_save['A_src_in_sta'][0:2,:].astype('int')
			x_grid = data_save['x_grid']
			x_grids_trv = compute_travel_times(trv, locs_use, np.expand_dims(data_save['x_grid'], axis = 0), device=device)[0]
			x_grids_trv += x_grid[:,3].reshape(-1,1,1)
			kernel_sig_t = data_save['sigma_input']
			min_t, max_t = x_grids_trv.min(), x_grids_trv.max()

			## Extract input features; interleave training samples
			for i in range(n_samples):

				engine = SeismicEmbeddingEngine(
					P=sample_dict['picks_%d'%i],
					locs=locs_use,
					ind_use=np.arange(len(locs_use)), # sample_dict['selected_sta_idx_%d'%i]
					A_src_in_sta=A_src_in_sta,
					trv_times=x_grids_trv,
					dt=kernel_sig_t/15.0, # dt_embed_discretize = np.round(pred_params[1] / 15.0, 2) # kernel_sig_t
					kernel_sig_t=kernel_sig_t,
					t_pad=3.0*kernel_sig_t,
					use_sign_input=False,
					precompute=True,  # Builds GPU global grid once for continuous O(1) sampling
					device=device,
				)

				[Inpts, Masks], [lp_times, lp_stations, lp_phases, lp_meta] = engine.extract_inputs(t0 = np.array([0.0]), min_t = min_t, max_t = max_t, t_win = 2.0*kernel_sig_t)
				# sample_dict['Inpts_%d'%i] = Inpts[0]
				# sample_dict['Masks_%d'%i] = Masks[0]
				sample_dict['Inpts_%d'%i] = Inpts[0].detach().cpu().numpy() if hasattr(Inpts[0], 'cpu') else Inpts[0]
				sample_dict['Masks_%d'%i] = Masks[0].detach().cpu().numpy() if hasattr(Masks[0], 'cpu') else Masks[0]
				sample_dict['lp_times_%d'%i] = lp_times[0]
				sample_dict['lp_stations_%d'%i] = lp_stations[0]
				sample_dict['lp_phases_%d'%i] = lp_phases[0]
				sample_dict['lp_meta_%d'%i] = lp_meta[0]

			## Store the data per sample (and per graph instance)
			sample_dict_store.append(sample_dict)


		## Need to make product expander graph edges
		print('Need to make product expander graph edges')
		print('Need to make spatial vals (offset distances), and normalize')

		## Now write n_sample new training files

		for file_ind in range(n_samples):

			file_index = total_files*job_number + file_inc

			file_inc += 1

			with h5py.File(path_to_data + 'training_data_slice_%d_ver_%d.hdf5'%(file_index, n_ver_training_data), 'w') as h:

				for ind_sample in range(n_batch):

					## Input features
					h['Inpts_%d'%ind_sample] = sample_dict_store[ind_sample]['Inpts_%d'%file_ind]
					h['Masks_%d'%ind_sample] = sample_dict_store[ind_sample]['Masks_%d'%file_ind]
					h['X_query_%d'%ind_sample] = np.concatenate((sample_dict_store[ind_sample]['x_query_%d'%file_ind], sample_dict_store[ind_sample]['x_query_t_%d'%file_ind].reshape(-1,1)), axis = 1)
					h['X_query_cart_%d'%ind_sample] = ftrns1(sample_dict_store[ind_sample]['x_query_%d'%file_ind])
					h['X_src_query_%d'%ind_sample] = np.concatenate((sample_dict_store[ind_sample]['x_src_query_%d'%file_ind], sample_dict_store[ind_sample]['x_src_query_t_%d'%file_ind].reshape(-1,1)), axis = 1)
					h['X_src_cart_%d'%ind_sample] = ftrns1(sample_dict_store[ind_sample]['x_src_query_%d'%file_ind])
					h['Lbls_%d'%ind_sample] = sample_dict_store[ind_sample]['labels_grid_%d'%file_ind]
					h['Lbls_query_%d'%ind_sample] = sample_dict_store[ind_sample]['labels_%d'%file_ind]
					h['Lbls_picks_%d'%ind_sample] = sample_dict_store[ind_sample]['pick_lbls_%d'%file_ind]
					h['lp_times_%d'%ind_sample] = sample_dict_store[ind_sample]['lp_times_%d'%file_ind]
					h['lp_stations_%d'%ind_sample] = sample_dict_store[ind_sample]['lp_stations_%d'%file_ind]
					h['lp_phases_%d'%ind_sample] = sample_dict_store[ind_sample]['lp_phases_%d'%file_ind]
					h['lp_meta_%d'%ind_sample] = sample_dict_store[ind_sample]['lp_meta_%d'%file_ind]
					h['lp_srcs_%d'%ind_sample] = sample_dict_store[ind_sample]['sources_%d'%file_ind] ## Need to mask active sources
					h['data_%d'%ind_sample] = sample_dict_store[ind_sample]['picks_%d'%file_ind]
					h['data_true_%d'%ind_sample] = sample_dict_store[ind_sample]['picks_true_%d'%file_ind]

					## Graphs
					h['A_sta_%d'%ind_sample] = data_save_store[ind_sample]['A_sta'] # [0:2,:].astype('int')
					h['A_src_%d'%ind_sample] = data_save_store[ind_sample]['A_src'] # [0:2,:].astype('int')
					h['A_src_in_sta_%d'%ind_sample] = data_save_store[ind_sample]['A_src_in_sta'] # [0:2,:].astype('int')
					h['A_src_in_prod_%d'%ind_sample] = data_save_store[ind_sample]['A_src_in_prod'] # [:,0:2].astype('int')
					h['A_prod_sta_sta_%d'%ind_sample] = data_save_store[ind_sample]['A_prod_sta_sta'] # [0:2,:].astype('int')
					h['A_prod_src_src_%d'%ind_sample] = data_save_store[ind_sample]['A_prod_src_src'] # [0:2,:].astype('int')

					h['Ac_src_%d'%ind_sample] = data_save_store[ind_sample]['Ac_src'] # [0:2,:].astype('int')
					h['Ac_prod_src_src_%d'%ind_sample] = data_save_store[ind_sample]['Ac_prod_src_src'] # [0:2,:].astype('int')
					h['spatial_vals_%d'%ind_sample] = data_save_store[ind_sample]['spatial_vals'] # [0:2,:].astype('int')


					## Add product distances (normalized)

					## Add grid information
					h['X_fixed_%d'%ind_sample] = data_save_store[ind_sample]['x_grid']
					h['X_fixed_cart_%d'%ind_sample] = ftrns1(data_save_store[ind_sample]['x_grid'])
					h['Locs_%d'%ind_sample] = data_save_store[ind_sample]['locs_use']
					h['Locs_cart_%d'%ind_sample] = ftrns1(data_save_store[ind_sample]['locs_use'])
					h['depth_boost_%d'%ind_sample] = data_save_store[ind_sample]['depth_boost']

					## Add domain parameters
					h['lat_range_%d'%ind_sample] = data_save_store[ind_sample]['lat_range']
					h['lon_range_%d'%ind_sample] = data_save_store[ind_sample]['lon_range']
					h['lat_range_extend_%d'%ind_sample] = data_save_store[ind_sample]['lat_range_extend']
					h['lon_range_extend_%d'%ind_sample] = data_save_store[ind_sample]['lon_range_extend']
					h['depth_range_%d'%ind_sample] = data_save_store[ind_sample]['depth_range']
					h['kernel_sig_t_%d'%ind_sample] = data_save_store[ind_sample]['sigma_input']
					h['src_x_kernel_%d'%ind_sample] = data_save_store[ind_sample]['source_label_width']
					h['src_t_kernel_%d'%ind_sample] = data_save_store[ind_sample]['source_label_width_t']
					h['src_depth_kernel_%d'%ind_sample] = data_save_store[ind_sample]['source_label_width']
					h['scale_time_%d'%ind_sample] = data_save_store[ind_sample]['scale_time']
					h['src_x_arv_kernel_%d'%ind_sample] = data_save_store[ind_sample]['src_x_arv_kernel']
					h['src_t_arv_kernel_%d'%ind_sample] = data_save_store[ind_sample]['src_t_arv_kernel']
					# h['src_x_arv_kernel_%d'%ind_sample] = data_save_store[ind_sample]['src_x_arv_kernel']
					# h['src_t_arv_kernel_%d'%ind_sample] = data_save_store[ind_sample]['src_t_arv_kernel']
					h['time_shift_range_%d'%ind_sample] = data_save_store[ind_sample]['x_grid'][:,3].max() - data_save_store[ind_sample]['x_grid'][:,3].min() # data_save_store[ind_sample]['time_shift_range']

			print('Saved file %d \n'%file_index)

		# Clear stores explicitly at the end of each outer chunk iteration
		del sample_dict_store, data_save_store
		import gc; gc.collect()

















# def perturb_wgs84(
# 	base_lat_deg,
# 	base_lon_deg,
# 	base_depth_m,
# 	base_t_s,
# 	src_x_kernel_m,
# 	src_depth_kernel_m,
# 	src_t_kernel,
# 	lat_range,
# 	lon_range,
# 	depth_range,
# 	time_shift_range,
# 	is_global_lon=True,
# 	a=6378137.0,
# 	e=8.18191908426215e-2,
# ):
# 	"""Refactored WGS84 perturbation using exact 3D ECEF tangent rotation.
	
# 	Prevents division-by-zero division at poles and seamlessly handles
# 	cross-polar meridian crossings.
# 	"""
# 	n_pts = len(base_lat_deg)
# 	if n_pts == 0:
# 		return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

# 	half_t_window = time_shift_range / 2.0

# 	# 1. Convert Base points to ECEF (depth_m acts directly as height in lla2ecef)
# 	lla_base = np.column_stack((base_lat_deg, base_lon_deg, base_depth_m))
# 	ecef_base = lla2ecef(lla_base, a=a, e=e)

# 	# 2. Local ENU 2D spatial perturbations
# 	dE = np.random.normal(0, src_x_kernel_m, size=n_pts)
# 	dN = np.random.normal(0, src_x_kernel_m, size=n_pts)

# 	# 3. Compute local ENU unit basis vectors in ECEF frame
# 	phi = np.radians(base_lat_deg)
# 	lam = np.radians(base_lon_deg)

# 	sin_phi, cos_phi = np.sin(phi), np.cos(phi)
# 	sin_lam, cos_lam = np.sin(lam), np.cos(lam)

# 	# East unit vector (dE)
# 	e_x = -sin_lam
# 	e_y = cos_lam
# 	e_z = np.zeros(n_pts)

# 	# North unit vector (dN)
# 	n_x = -sin_phi * cos_lam
# 	n_y = -sin_phi * sin_lam
# 	n_z = cos_phi

# 	# 4. Apply metric displacements in 3D ECEF space
# 	ecef_pert = np.empty_like(ecef_base)
# 	ecef_pert[:, 0] = ecef_base[:, 0] + dE * e_x + dN * n_x
# 	ecef_pert[:, 1] = ecef_base[:, 1] + dE * e_y + dN * n_y
# 	ecef_pert[:, 2] = ecef_base[:, 2] + dE * e_z + dN * n_z

# 	# 5. Convert back to LLA (handles polar singularities & 180 wrapping automatically)
# 	lla_pert = ecef2lla(ecef_pert, a=a, e=e)
# 	new_lat = lla_pert[:, 0]
# 	new_lon = lla_pert[:, 1]

# 	# 6. Apply Boundary Conditions
# 	if is_global_lon:
# 		new_lon = (new_lon + 180.0) % 360.0 - 180.0
# 	else:
# 		new_lon = reflect_bounds(new_lon, lon_range[0], lon_range[1])

# 	new_lat = reflect_bounds(new_lat, lat_range[0], lat_range[1])

# 	# Depth & Time Perturbations
# 	dz = np.random.normal(0, src_depth_kernel_m, size=n_pts)
# 	new_depth = reflect_bounds(base_depth_m + dz, depth_range[0], depth_range[1])

# 	dt = np.random.normal(0, src_t_kernel, size=n_pts)
# 	new_t = reflect_bounds(base_t_s + dt, -half_t_window, half_t_window)

# 	return new_lat, new_lon, new_depth, new_t


# def sample_random_queries(
# 	lp_srcs,
# 	n_src_query,
# 	n_frac_focused=0.2,
# 	n_frac_random_focused=0.2,
# 	src_x_kernel_m=5000.0,
# 	src_depth_kernel_m=5000.0,
# 	src_t_kernel=1.0,
# 	lat_range=(-90.0, 90.0),
# 	lon_range=(-180.0, 180.0),
# 	depth_range=(-700000.0, 0.0),
# 	time_shift_range=10.0,
# 	is_global_lon=True,
# ):
# 	"""Generates spatial-temporal queries via uniform equal-area sampling and target perturbations."""
# 	half_t_window = time_shift_range / 2.0

# 	# 1. Fallback: If no true sources exist, reallocate true-focused to random-focused
# 	if len(lp_srcs) == 0 and n_frac_focused > 0:
# 		n_frac_random_focused += n_frac_focused
# 		n_frac_focused = 0.0

# 	# Calculate exact allocation counts AFTER updating fractions
# 	n_focused_true = int(n_frac_focused * n_src_query)
# 	n_focused_rand = int(n_frac_random_focused * n_src_query)
# 	n_total_focused = n_focused_true + n_focused_rand

# 	# 2. Background Equal-Area Sampling (Full Set)
# 	sin_min = np.sin(np.radians(lat_range[0]))
# 	sin_max = np.sin(np.radians(lat_range[1]))
	
# 	u_lat = np.random.uniform(sin_min, sin_max, size=n_src_query)
# 	bg_lats = np.degrees(np.arcsin(np.clip(u_lat, -1.0, 1.0)))  # Safety clip

# 	bg_lons = np.random.uniform(lon_range[0], lon_range[1], size=n_src_query)
# 	bg_depths = np.random.uniform(depth_range[0], depth_range[1], size=n_src_query)
# 	bg_times = np.random.uniform(-half_t_window, half_t_window, size=n_src_query)

# 	x_src_query = np.column_stack((bg_lats, bg_lons, bg_depths))
# 	tq_sample = bg_times

# 	if n_total_focused > 0:
# 		# Pick indices to overwrite without replacement
# 		ind_overwrite_all = np.random.choice(
# 			n_src_query, size=n_total_focused, replace=False
# 		)
		
# 		ind_overwrite_true = ind_overwrite_all[:n_focused_true]
# 		ind_overwrite_rand = ind_overwrite_all[n_focused_true:]

# 		# -------------------------------------------------------------------------
# 		# 3a. Focused Perturbation around True Sources
# 		# -------------------------------------------------------------------------
# 		if n_focused_true > 0 and len(lp_srcs) > 0:
# 			ind_sources = np.random.choice(len(lp_srcs), size=n_focused_true)

# 			new_lat, new_lon, new_depth, new_t = perturb_wgs84(
# 				base_lat_deg=lp_srcs[ind_sources, 0],
# 				base_lon_deg=lp_srcs[ind_sources, 1],
# 				base_depth_m=lp_srcs[ind_sources, 2],
# 				base_t_s=lp_srcs[ind_sources, 3],
# 				src_x_kernel_m=src_x_kernel_m,
# 				src_depth_kernel_m=src_depth_kernel_m,
# 				src_t_kernel=src_t_kernel,
# 				lat_range=lat_range,
# 				lon_range=lon_range,
# 				depth_range=depth_range,
# 				time_shift_range=time_shift_range,
# 				is_global_lon=is_global_lon,
# 			)

# 			x_src_query[ind_overwrite_true] = np.column_stack((new_lat, new_lon, new_depth))
# 			tq_sample[ind_overwrite_true] = new_t

# 		# -------------------------------------------------------------------------
# 		# 3b. Focused Perturbation around Random Background Points
# 		# -------------------------------------------------------------------------
# 		if n_focused_rand > 0:
# 			rand_u_lat = np.random.uniform(sin_min, sin_max, size=n_focused_rand)
# 			rand_center_lats = np.degrees(np.arcsin(np.clip(rand_u_lat, -1.0, 1.0)))
# 			rand_center_lons = np.random.uniform(lon_range[0], lon_range[1], size=n_focused_rand)
# 			rand_center_depths = np.random.uniform(depth_range[0], depth_range[1], size=n_focused_rand)
# 			rand_center_times = np.random.uniform(-half_t_window, half_t_window, size=n_focused_rand)

# 			new_lat_r, new_lon_r, new_depth_r, new_t_r = perturb_wgs84(
# 				base_lat_deg=rand_center_lats,
# 				base_lon_deg=rand_center_lons,
# 				base_depth_m=rand_center_depths,
# 				base_t_s=rand_center_times,
# 				src_x_kernel_m=src_x_kernel_m,
# 				src_depth_kernel_m=src_depth_kernel_m,
# 				src_t_kernel=src_t_kernel,
# 				lat_range=lat_range,
# 				lon_range=lon_range,
# 				depth_range=depth_range,
# 				time_shift_range=time_shift_range,
# 				is_global_lon=is_global_lon,
# 			)

# 			x_src_query[ind_overwrite_rand] = np.column_stack((new_lat_r, new_lon_r, new_depth_r))
# 			tq_sample[ind_overwrite_rand] = new_t_r

# 	return x_src_query, tq_sample


# def compute_source_labels(
# 	x_query, x_query_t, src_x, src_t, src_spatial_kernel, src_t_kernel, ftrns1
# ):
# 	"""Computes Gaussian source probability labels for spatio-temporal query

# 	points.

# 	Parameters
# 	----------
# 	x_query : np.ndarray
# 		Query locations in LLA coordinates [N_query, 3].
# 	x_query_t : np.ndarray
# 		Query timestamps [N_query].
# 	src_x : np.ndarray
# 		Source locations in LLA coordinates [N_src, 3].
# 	src_t : np.ndarray
# 		Source origin times [N_src].
# 	src_spatial_kernel : float
# 		Standard deviation for spatial Gaussian decay (in Cartesian units).
# 	src_t_kernel : float
# 		Standard deviation for temporal Gaussian decay (in seconds).
# 	ftrns1 : callable
# 		Projection function from LLA -> 3D Cartesian coordinates.

# 	Returns
# 	-------
# 	np.ndarray
# 		Maximum Gaussian label value for each query point of shape [N_query, 1].
# 	"""
# 	if len(src_x) == 0:
# 		return np.zeros((len(x_query), 1))

# 	# Project coordinates to Cartesian space for metric distance calculation
# 	x_query_cart = ftrns1(x_query)  # [N_query, 3]
# 	src_x_cart = ftrns1(src_x)  # [N_src, 3]

# 	# Spatial Gaussian weight: exp(-0.5 * ||x_q - x_s||^2 / sigma_x^2)
# 	spatial_dist_sq = (
# 		(x_query_cart[:, None, :] - src_x_cart[None, :, :]) ** 2
# 	).sum(axis=2)
# 	spatial_weight = np.exp(-0.5 * (spatial_dist_sq / (src_spatial_kernel**2)))

# 	# Temporal Gaussian weight: exp(-0.5 * (t_q - t_s)^2 / sigma_t^2)
# 	time_diff_sq = (x_query_t.reshape(-1, 1) - src_t.reshape(1, -1)) ** 2
# 	temporal_weight = np.exp(-0.5 * (time_diff_sq / (src_t_kernel**2)))

# 	# Combined space-time label (max value across all active sources)
# 	labels = (spatial_weight * temporal_weight).max(axis=1, keepdims=True)
# 	return labels


