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
from graph_utils import *
import glob
import sys
import os



def compute_travel_times_parallel(xx, xx_r, h, h1, dx_v, x11, x12, x13, num_cores=10):
	"""
	Computes 3D P and S travel time fields in parallel using Fast Marching (scikit-fmm).
	Includes sub-grid source placement, zero-contour guards, and HPC job-array logging.
	"""
	# -------------------------------------------------------------------------
	# 0. Initial Validation & Grid Boundary Check
	# -------------------------------------------------------------------------
	n_sources = xx_r.shape[0]
	grid_shape = x11.shape

	print(f"[INFO] Initializing compute_travel_times_parallel", flush=True)
	print(f"[INFO] Grid Shape: {grid_shape} | Total Nodes: {h.shape[0]:,}", flush=True)
	print(f"[INFO] Total Sources to Process: {n_sources} across {num_cores} cores", flush=True)
	# print(f"[INFO] Grid Extents X: [{x11.min():.2f}, {x11.max():.2f}] | Y: [{x12.min():.2f}, {x12.max():.2f}] | Z: [{x13.min():.2f}, {x13.max():.2f}] | dx: {dx_v:.2f}", flush=True)
	print(f"[INFO] Grid Extents X: [{x11.min():.2f}, {x11.max():.2f}] | Y: [{x12.min():.2f}, {x12.max():.2f}] | Z: [{x13.min():.2f}, {x13.max():.2f}]", flush=True)

	# Check for non-physical/corrupted velocities prior to job dispatch
	v_min, v1_min = np.nanmin(h), np.nanmin(h1)
	if v_min <= 0 or v1_min <= 0:
		print(f"[WARNING] Velocity fields contain non-positive values (Vp_min={v_min:.1f}, Vs_min={v1_min:.1f})!", flush=True)
		print(f"[WARNING] Negative/zero velocities will cause Fast Marching crashes. Ensure arrays are clipped prior to passing.", flush=True)

	print('\n')
	print(x11.shape); print(x12.shape); print(x13.shape)
	print('\n')
	
	# -------------------------------------------------------------------------
	# 1. Parallel Worker Routine
	# -------------------------------------------------------------------------

	def step_test(args):
		yval, dx_v, h, h1, x11, x12, x13, ind = args
		source_coords = yval.ravel()

		if np.isnan(source_coords).any() or np.isinf(source_coords).any():
			print(f"[ERROR][Task {ind}] Invalid source coordinates (NaN/Inf): {source_coords}. Returning NaNs.", flush=True)
			empty = np.full(grid_shape, np.nan)
			return empty, empty, ind

		phi = np.sqrt(
			(x11 - source_coords[0])**2
			+ (x12 - source_coords[1])**2
			+ (x13 - source_coords[2])**2
		)

		r0 = 0.5 * np.min(dx_v)
		phi = phi - r0

		# Default: interior / normal nucleation — correct by sphere radius r0
		t0_radius = r0

		if not (phi < 0).any():
			min_idx = np.unravel_index(np.argmin(phi), grid_shape)
			min_dist = float(phi[min_idx] + r0)  # Euclidean dist source → nearest node
			phi[min_idx] = -r0
			t0_radius = min_dist  # exterior (or failed level set): pay full distance
			print(
				f"[WARN][Task {ind}] Source at ({source_coords[0]:.2f}, {source_coords[1]:.2f}, {source_coords[2]:.2f}) "
				f"has NO negative level set (Nearest node dist: {min_dist:.2f}). "
				f"Forcing closest node negative; t0 uses min_dist.",
				flush=True,
			)

		v_grid = h.reshape(grid_shape)
		v1_grid = h1.reshape(grid_shape)
		dx = [dx_v[0], dx_v[1], dx_v[2]]

		try:
			t = skfmm.travel_time(phi, v_grid, dx=dx)
			t1 = skfmm.travel_time(phi, v1_grid, dx=dx)
		except Exception as e:
			print(f"[FATAL][Task {ind}] skfmm failed at source {source_coords}: {str(e)}", flush=True)
			empty = np.full(grid_shape, np.nan)
			return empty, empty, ind

		source_idx = np.unravel_index(np.argmin(phi), grid_shape)
		t = t + (t0_radius / v_grid[source_idx])
		t1 = t1 + (t0_radius / v1_grid[source_idx])

		if ind % 50 == 0 or ind == n_sources - 1:
			print(f"[PROGRESS] Completed source index {ind}/{n_sources} at coords {source_coords.round(1)}", flush=True)

		return t, t1, ind

	# -------------------------------------------------------------------------
	# 2. Execution & Data Assembly
	# -------------------------------------------------------------------------
	tp_times = np.full((h.shape[0], n_sources), np.nan, dtype=np.float32)
	ts_times = np.full((h.shape[0], n_sources), np.nan, dtype=np.float32)

	# Run Parallel execution
	print(f"[INFO] Dispatching Parallel jobs...", flush=True)
	results = Parallel(n_jobs=num_cores)(
		delayed(step_test)([xx_r[i, :][None, :], dx_v, h, h1, x11, x12, x13, i])
		for i in range(n_sources)
	)

	# Reassemble results deterministically based on station index
	print(f"[INFO] Assembling travel time matrices...", flush=True)
	for res in results:
		t_p, t_s, station_ind = res[0], res[1], res[2]
		tp_times[:, station_ind] = t_p.reshape(-1)
		ts_times[:, station_ind] = t_s.reshape(-1)

	print(f"[SUCCESS] Travel time calculations finished successfully.", flush=True)
	return tp_times, ts_times


def make_merged_travel_time_interp(mp_coarse, mp_local, sta_xyz, half_xy, half_z, blend_width=None):
	sta_xyz = np.asarray(sta_xyz, float).reshape(3)
	if blend_width is None:
		blend_width = 0.1 * float(half_xy)
	blend_width = max(float(blend_width), 1.0)

	def interp(pts_cart):
		pts_cart = np.asarray(pts_cart, float).reshape(-1, 3)
		# normalized box distance (0 at station, 1 on faces)
		q = np.maximum.reduce([
			np.abs(pts_cart[:, 0] - sta_xyz[0]) / half_xy,
			np.abs(pts_cart[:, 1] - sta_xyz[1]) / half_xy,
			np.abs(pts_cart[:, 2] - sta_xyz[2]) / half_z,
		])
		t_c = np.asarray(mp_coarse(pts_cart)).ravel()
		use_loc = q <= 1.0
		t = t_c.copy()
		if np.any(use_loc):
			t_l = np.asarray(mp_local(pts_cart[use_loc])).ravel()
			# blend in outer 10% of the box
			w = np.clip((q[use_loc] - (1.0 - blend_width / max(half_xy, 1.0))) /
						(blend_width / max(half_xy, 1.0) + 1e-12), 0.0, 1.0)
			bad = ~np.isfinite(t_l)
			t_l = np.where(bad, t_c[use_loc], t_l)
			w = np.where(bad, 1.0, w)
			t[use_loc] = (1.0 - w) * t_l + w * t_c[use_loc]
		return t
	return interp


def sample_stratified_queries_for_station(
	sta_lla,
	lat_range,
	lon_range,
	depth_range,
	ftrns1,
	ftrns2,
	tp_interp=None,
	ts_interp=None,
	vp_interp=None,
	vs_interp=None,
	n_very_near=40_000,
	n_near=50_000,
	n_far=40_000,
	n_uniform=40_000,
	n_boundary=10_000,
	n_vald=10_000,
	n_candidates_factor=4,
	# optional near densification as fraction of domain width (None = off)
	near_frac=0.15,
	n_near_boost=10_000,		  # e.g. 20_000 extra candidates in near ball; 0 = pure domain
	_sobol_box=None,
	xx_min=None,
	xx_max=None,
	seed=0,
):
	"""
	Continuous stratified queries for one station (save & train as-is).
	Mirrors old lattice weights: 1/t^2, 1/t, 1/(tmax-t), uniform, + station boundary.
	"""
	sta_lla = np.asarray(sta_lla, dtype=float).reshape(3)
	low_lla = np.array([lat_range[0], lon_range[0], depth_range[0]], float)
	high_lla = np.array([lat_range[1], lon_range[1], depth_range[1]], float)
	sta_xyz = ftrns1(sta_lla.reshape(1, 3))[0]

	def clip_lla(X):
		X = np.asarray(X, dtype=float).reshape(-1, 3)
		return np.minimum(np.maximum(X, low_lla), high_lla)

	def clip_xyz_to_fmm(pts_xyz, margin=0.0):
		pts_xyz = np.asarray(pts_xyz, dtype=float).reshape(-1, 3)
		if xx_min is None or xx_max is None:
			return pts_xyz
		return np.minimum(np.maximum(pts_xyz, xx_min + margin), xx_max - margin)

	def domain_candidates(n):
		n = int(max(n, 1))
		return clip_lla(np.asarray(_sobol_box(n)))

	def near_boost_candidates(n, frac):
		"""Optional: extra continuous points near station (domain-relative radius)."""
		if n <= 0 or frac is None or frac <= 0:
			return np.zeros((0, 3))
		if xx_min is not None and xx_max is not None:
			L = float(np.minimum(xx_max[0] - xx_min[0], xx_max[1] - xx_min[1]))
		else:
			# fallback: rough scale from LLA corners
			c = ftrns1(np.array([
				[low_lla[0], low_lla[1], sta_lla[2]],
				[high_lla[0], high_lla[1], sta_lla[2]],
			]))
			L = float(np.linalg.norm(c[1] - c[0]))
		radius_m = frac * L
		need = int(n * 2)
		rng_i = np.random.default_rng(seed + 7)
		pts = sta_xyz + rng_i.uniform(-radius_m, radius_m, size=(need, 3))
		pts = pts[np.linalg.norm(pts - sta_xyz, axis=1) <= radius_m]
		if len(pts) < n:
			extra = sta_xyz + rng_i.normal(scale=radius_m / 2.5, size=(n, 3))
			pts = np.vstack([pts, extra]) if len(pts) else extra
		pts = clip_xyz_to_fmm(pts)
		return clip_lla(ftrns2(pts[:n]))

	def weighted_pick(pts_lla, tp, n, mode, rng, exclude=None):
		n = int(n)
		if n <= 0 or len(pts_lla) == 0:
			return np.zeros((0, 3)), np.zeros(0)
		tp = np.maximum(np.asarray(tp).ravel(), 0.1)
		m = np.ones(len(tp), dtype=bool)
		if exclude is not None and len(exclude):
			m[np.asarray(exclude, dtype=int)] = False
		pts_lla, tp = pts_lla[m], tp[m]
		if len(tp) == 0:
			return np.zeros((0, 3)), np.zeros(0)
		n = min(n, len(tp))
		if mode == "inv_t":
			w = 1.0 / tp
		elif mode == "inv_t2":
			w = 1.0 / tp**2
		elif mode == "far":
			w = 1.0 / np.maximum(tp.max() - tp, 0.1)
		else:
			w = np.ones_like(tp)
		w = w / w.sum()
		idx = rng.choice(len(tp), size=n, replace=False, p=w)
		return pts_lla[idx], tp[idx]

	def eval_tp(pts_lla):
		return np.asarray(tp_interp(ftrns1(pts_lla))).ravel()

	rng = np.random.default_rng(seed)

	# --- one large domain pool (like old X), optional near boost mixed in ---
	n_pool = int(n_candidates_factor * (
		n_very_near + n_near + n_far + n_uniform
	))
	cand = domain_candidates(n_pool)
	if n_near_boost > 0:
		cand = np.vstack([cand, near_boost_candidates(n_near_boost, near_frac)])
		cand = clip_lla(cand)

	tp = eval_tp(cand)

	# sequential without-replacement strata on the same pool
	used = []
	def take(n, mode):
		nonlocal used
		# map exclude through current mask is awkward; rebuild pool each time
		# simpler: pick from full cand but zero weight on already chosen via mask
		mask = np.ones(len(cand), dtype=bool)
		if used:
			mask[np.unique(used)] = False
		pts_m, tp_m = cand[mask], tp[mask]
		if len(pts_m) == 0:
			return np.zeros((0, 3))
		n = min(int(n), len(pts_m))
		if mode == "inv_t":
			w = 1.0 / np.maximum(tp_m, 0.1)
		elif mode == "inv_t2":
			w = 1.0 / np.maximum(tp_m, 0.1)**2
		elif mode == "far":
			w = 1.0 / np.maximum(tp_m.max() - tp_m, 0.1)
		else:
			w = np.ones(len(tp_m))
		w = w / w.sum()
		idx_m = rng.choice(len(pts_m), size=n, replace=False, p=w)
		# record global indices
		global_idx = np.flatnonzero(mask)[idx_m]
		used.extend(global_idx.tolist())
		return pts_m[idx_m]

	X_vn = take(n_very_near, "inv_t2")
	X_ne = take(n_near, "inv_t")
	X_fa = take(n_far, "far")
	X_un = take(n_uniform, "uniform")
	X_bd = np.repeat(sta_lla.reshape(1, 3), int(n_boundary), axis=0)

	X_train = clip_lla(np.vstack([X_vn, X_ne, X_fa, X_un, X_bd]))
	if n_boundary > 0:
		X_train[-n_boundary:] = sta_lla.reshape(1, 3)

	cand_v = domain_candidates(int(n_candidates_factor * n_vald))
	X_vald, _ = weighted_pick(
		cand_v, eval_tp(cand_v), n_vald, "uniform",
		np.random.default_rng(seed + 99),
	)
	X_vald = clip_lla(X_vald)

	def eval_all(pts):
		xyz = ftrns1(pts)
		out = {"X": pts, "Tp": np.asarray(tp_interp(xyz)).ravel()}
		if ts_interp is not None:
			out["Ts"] = np.asarray(ts_interp(xyz)).ravel()
		if vp_interp is not None:
			out["Vp"] = np.asarray(vp_interp(xyz)).ravel()
		if vs_interp is not None:
			out["Vs"] = np.asarray(vs_interp(xyz)).ravel()
		return out

	train = eval_all(X_train)
	vald = eval_all(X_vald)
	if n_boundary > 0:
		train["Tp"][-n_boundary:] = 0.0
		if "Ts" in train:
			train["Ts"][-n_boundary:] = 0.0
	return train, vald


# Load configuration from YAML
config = load_config('config.yaml')

name_of_project = config['name_of_project']
num_cores = config['num_cores']

dx = config['dx']
dx_depth = config.get('dx_depth', dx)
# d_deg = config.get('d_deg', dx/1e5) ## Roughly convert meters to degrees

## Load travel time neural network settings
save_dense_travel_time_data = config['save_dense_travel_time_data']
train_travel_time_neural_network = config['train_travel_time_neural_network']


# use_relative_1d_profile = config['use_relative_1d_profile']
# if use_relative_1d_profile == True:
# 	print('Overwritting num cores, because using relative 1d profile option')
# 	num_cores = 1

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())
seperator =  '\\' if '\\' in path_to_file else '/'
path_to_file += seperator
num_cores = 1


# template_ver = 1
vel_model_ver = config['vel_model_ver']
vel_model_type = config['vel_model_type']
use_topography = config['use_topography']


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
depth_grid = np.arange(depth_range[0], depth_range[1] + dx_depth, dx_depth)


ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1


## Using 3D domain, so must use actual station coordinates
locs_ref = np.copy(locs)
reciever_proj = ftrns1(locs_ref) # for all elevs.


## Per job slice of stations
argvs = sys.argv
if len(argvs) == 1:
	argvs.append(0)

n_jobs = config['n_jobs']
n_batch = int(np.ceil(len(locs)/n_jobs))
ind_use = [np.arange(n_batch) + n_batch*i for i in range(n_jobs)]
if n_jobs > 1:
	ind_use[-1] = np.arange(ind_use[-2][-1] + 1, len(locs))
ind_use = ind_use[int(argvs[1])]


print('\n Total possible threads %d'%multiprocessing.cpu_count())
print(f'Actually using {num_cores} cores')


spanned_domain, domain_pad_cnt = False, 0
while (spanned_domain == False) and (domain_pad_cnt < 20):

	# LLA domain used for queries / stations (already extended)
	lat0, lat1 = lat_range_extend
	lon0, lon1 = lon_range_extend
	dep0, dep1 = depth_range[0], depth_range[1]

	# Sample boundary of the LLA box (corners + edge mids + a few edge samples)
	def lla_box_boundary(n_edge=5):

		lats = np.linspace(lat0, lat1, n_edge)
		lons = np.linspace(lon0, lon1, n_edge)
		deps = np.linspace(dep0, dep1, max(3, n_edge // 2))

		x11, x12, x13 = np.meshgrid(lats, lons, deps)
		pts = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)   

		return np.unique(np.asarray(pts, float), axis=0)

	bnd_lla = lla_box_boundary(11)
	bnd_xyz = ftrns1(bnd_lla)

	margin_xy = (5.0 + domain_pad_cnt) * dx   # physical margin (m); scale-free if dx is your resolution unit
	margin_z  =  (1.0 + domain_pad_cnt * (1/5) ) * dx

	x1 = np.arange(bnd_xyz[:, 0].min() - margin_xy, bnd_xyz[:, 0].max() + margin_xy + dx, dx)
	x2 = np.arange(bnd_xyz[:, 1].min() - margin_xy, bnd_xyz[:, 1].max() + margin_xy + dx, dx)
	x3 = np.arange(bnd_xyz[:, 2].min() - margin_z,  bnd_xyz[:, 2].max() + margin_z  + dx, dx)

	x11, x12, x13 = np.meshgrid(x1, x2, x3)
	xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
	dx_v = np.array([np.diff(x1)[0], np.diff(x2)[0], np.diff(x3)[0]])
	xx_min = xx.min(0)
	xx_max = xx.max(0)

	# hull = ConvexHull(xx)
	# inside_hull = in_hull(reciever_proj, hull.points[hull.vertices])
	inside_hull = np.all((reciever_proj >= xx.min(0))*(reciever_proj <= xx.max(0)), axis = 1)
	print('Num sta inside hull %d (%d)'%(inside_hull.sum(), domain_pad_cnt))
	print('Num total sta %d (%d)'%(len(locs_ref), domain_pad_cnt))

	if (inside_hull.sum() == locs_ref.shape[0]):
		spanned_domain = True
	else:
		domain_pad_cnt += 1

## Assert that domain spans all stations
assert(inside_hull.sum() == locs_ref.shape[0]) ## If failt, expand the pad

## High resolution grid
xy_span = float(np.minimum(xx_max[0] - xx_min[0], xx_max[1] - xx_min[1]))
z_span  = float(xx_max[2] - xx_min[2])

# Local box = fixed fraction of domain (application-scale free)
frac_xy = 0.20		  # 20% of domain width on each side of station → 40% span
frac_z  = 0.50		  # larger fraction in depth is often fine (smaller absolute size)

half_xy = 0.5 * frac_xy * xy_span
half_z  = 0.5 * frac_z  * z_span


if vel_model_type == 1:
	vp = np.array(config['velocity_model']['Vp'])
	vs = np.array(config['velocity_model']['Vs'])
	depths = np.array(config['velocity_model']['Depths'])
	iarg = np.argsort(depths)
	# z = np.load(path_to_file + '1d_velocity_model.npz')
	# depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
	# z.close()
	
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

	## Concatenate station elevations
	Points = np.concatenate((Points, locs), axis = 0)
	d_deg = config.get('d_deg', dx/1e5) ## Roughly convert meters to degrees

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


## Check the extrapolation behavior (can replace with nearest neighbor)
mp_vp = RegularGridInterpolator((x1, x2, x3), Vp.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear', bounds_error = False, fill_value = None)
ms_vs = RegularGridInterpolator((x1, x2, x3), Vs.reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear', bounds_error = False, fill_value = None)


## Setup Sobolov sampler
earth_radius = 6378137.0
ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
	# r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_min
	return r_val


def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
	# r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
	r_val = r_surface + h_max
	return -r_val


bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_min, bounds, popsize = 50, maxiter = 1000, disp = True)
r_min = optimize_r_min(np.array([soln.x]))[0]; print('\n')


bounds = [(lat_range_extend[0], lat_range_extend[1])]
soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True)
r_max = -1.0*optimize_r_max(np.array([soln.x]))[0]; print('\n')
assert(r_max >= r_min)

sobolov_sampler = lambda N: regular_sobolov(N, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, use_time = False, use_global = False, r_min = r_min, r_max = r_max)[0]
# def regular_sobolov(N, lat_range = None, lon_range = None, depth_range = None, time_range = None, use_time = True, use_global = None, scale_time = None, depth_boost = 1.0, N_target = None, use_station_density = False, buffer_scale = 0.0, r_min = None, r_max = None, use_spherical = False, run_checks = False)


# use_min_velocity = True
# if use_min_velocity == True:


# vel_model_ver = 1
# use_relative_1d_profile = False

compute_reference_times = True
if compute_reference_times == True:

	print('Can sub-sample the real training data to limit memory cost for large networks')

	## Travel times
	num_cores = 1 ## This next routine is very expensive in parallel
	if num_cores == 1:

		for j in ind_use:


			results = compute_travel_times_parallel(xx, reciever_proj[j][None,:], Vp, Vs, dx_v, x11, x12, x13, num_cores = num_cores)
			mp = RegularGridInterpolator((x1, x2, x3), results[0].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear', bounds_error = False, fill_value = None)
			ms = RegularGridInterpolator((x1, x2, x3), results[1].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear', bounds_error = False, fill_value = None)

			# ----- local high-res box (domain fractions set once outside the loop) -----
			sta_xyz = reciever_proj[j]

			half_xy_j = min(half_xy, sta_xyz[0] - xx_min[0], xx_max[0] - sta_xyz[0])
			half_xy_j = min(half_xy_j, sta_xyz[1] - xx_min[1], xx_max[1] - sta_xyz[1])
			half_z_j  = min(half_z,  sta_xyz[2] - xx_min[2], xx_max[2] - sta_xyz[2])

			# avoid degenerate boxes at domain edge
			half_xy_j = max(half_xy_j, 5 * dx)
			half_z_j  = max(half_z_j,  5 * dx)

			N_target = 0.5 * len(xx)
			V = max((2 * half_xy_j) ** 2 * (2 * half_z_j), 1.0)
			dx_loc = float(np.clip((V / N_target) ** (1.0 / 3.0), dx / 4.0, dx))

			x1_loc = np.arange(sta_xyz[0] - half_xy_j, sta_xyz[0] + half_xy_j + dx_loc, dx_loc)
			x2_loc = np.arange(sta_xyz[1] - half_xy_j, sta_xyz[1] + half_xy_j + dx_loc, dx_loc)
			x3_loc = np.arange(sta_xyz[2] - half_z_j,  sta_xyz[2] + half_z_j  + dx_loc, dx_loc)

			x11_loc, x12_loc, x13_loc = np.meshgrid(x1_loc, x2_loc, x3_loc)  # same convention as global
			xx_loc = np.column_stack([
				x11_loc.ravel(), x12_loc.ravel(), x13_loc.ravel()
			])
			dx_v_loc = np.array([
				np.diff(x1_loc)[0], np.diff(x2_loc)[0], np.diff(x3_loc)[0]
			])

			# Vp/Vs on local grid from global Cartesian interpolants (or KDTree NN)
			Vp_loc = np.asarray(mp_vp(xx_loc)).ravel()
			Vs_loc = np.asarray(ms_vs(xx_loc)).ravel()
			Vp_loc = np.maximum(Vp_loc, 200.0)  # keep FMM stable
			Vs_loc = np.maximum(Vs_loc, 100.0)

			results_loc = compute_travel_times_parallel(
				xx_loc,
				sta_xyz[None, :],
				Vp_loc,
				Vs_loc,
				dx_v_loc,
				x11_loc,
				x12_loc,
				x13_loc,
				num_cores=num_cores,
			)
			mp_loc = RegularGridInterpolator(
				(x1_loc, x2_loc, x3_loc),
				results_loc[0].reshape(len(x2_loc), len(x1_loc), len(x3_loc)).transpose([1, 0, 2]),
				method="linear",
				bounds_error=False,
				fill_value=None,
			)
			ms_loc = RegularGridInterpolator(
				(x1_loc, x2_loc, x3_loc),
				results_loc[1].reshape(len(x2_loc), len(x1_loc), len(x3_loc)).transpose([1, 0, 2]),
				method="linear",
				bounds_error=False,
				fill_value=None,
			)

			r_loc = half_xy_j  # merge radius in Cartesian horizontal sense (or use min half)
			blend = 0.1 * r_loc

			# def make_merged_travel_time_interp(mp_coarse, mp_local, sta_xyz, half_xy, half_z, blend_width=None):
			# tp_merged = make_merged_travel_time_interp(mp, mp_loc, sta_xyz, r_loc, blend_width=blend)
			# ts_merged = make_merged_travel_time_interp(ms, ms_loc, sta_xyz, r_loc, blend_width=blend)
			tp_merged = make_merged_travel_time_interp(mp, mp_loc, sta_xyz, half_xy_j, half_z_j, blend_width = blend)
			ts_merged = make_merged_travel_time_interp(ms, ms_loc, sta_xyz, half_xy_j, half_z_j, blend_width = blend)


			train, vald = sample_stratified_queries_for_station(
				locs_ref[j],
				lat_range_extend,
				lon_range_extend,
				depth_range,
				ftrns1,
				ftrns2,
				tp_interp = tp_merged,		 # callable: Cartesian (N,3) -> (N,) travel time P
				ts_interp = ts_merged,
				vp_interp = mp_vp,
				vs_interp = ms_vs,
				# counts (per station) — tune to match old totals / scale_factor
				n_very_near=40_000,
				n_near=50_000,
				n_far=40_000,
				n_uniform=40_000,
				n_boundary=10_000,	   # station itself; increase if you want more t≈0 copies
				n_vald=10_000,
				n_candidates_factor=4,  # candidate pool = factor * each stratum before reweight
				_sobol_box = sobolov_sampler,
				xx_min = xx_min,
				xx_max = xx_max,
				seed=np.random.choice(10000))

			np.savez_compressed(
				path_to_file
				+ "1D_Velocity_Models_Regional"
				+ seperator
				+ "TravelTimeData"
				+ seperator
				+ "%s_1d_velocity_model_station_%d_ver_%d.npz"
				% (name_of_project, j, vel_model_ver),
				
				X=train["X"],
				X_cart=ftrns1(train["X"]),
				Tp_interp=train["Tp"],
				Ts_interp=train["Ts"],
				Vp_interp=train.get("Vp"),
				Vs_interp=train.get("Vs"),

				X_vald=vald["X"],
				X_vald_cart=ftrns1(vald["X"]),
				Tp_vald=vald["Tp"],
				Ts_vald=vald["Ts"],
				Vp_vald=vald.get("Vp"),
				Vs_vald=vald.get("Vs"),

				loc=locs_ref[j],
				locs_ref=locs_ref,
				xx=xx,
				depth_grid=depth_grid,
				# optional full FMM fields if you still want them:
				# Tp=results[0], Ts=results[1], Vp=Vp, Vs=Vs,
			)



print("All files saved successfully!")
print("✔ Script execution: Done")




# def make_merged_travel_time_interp(
#	 mp_coarse,
#	 mp_local,
#	 sta_xyz,
#	 r_loc,
#	 blend_width=None,
# ):
#	 """
#	 Returns callable pts_cart (N,3) -> (N,) travel times.
#	 Inside r_loc - blend: local; outside r_loc: coarse; shell: smooth blend.
#	 """
#	 sta_xyz = np.asarray(sta_xyz, dtype=float).reshape(3)
#	 if blend_width is None:
#		 blend_width = 0.1 * float(r_loc)  # 10% of local half-width
#	 blend_width = max(float(blend_width), 1.0)
#	 r0 = max(float(r_loc) - blend_width, 0.0)
#	 r1 = float(r_loc)

#	 def interp(pts_cart):
#		 pts_cart = np.asarray(pts_cart, dtype=float).reshape(-1, 3)
#		 d = np.linalg.norm(pts_cart - sta_xyz, axis=1)

#		 t_c = np.asarray(mp_coarse(pts_cart)).ravel()

#		 # local only where it can be valid (inside/near box)
#		 use_loc = d <= r1
#		 t = t_c.copy()
#		 if np.any(use_loc):
#			 t_l = np.asarray(mp_local(pts_cart[use_loc])).ravel()
#			 # weight 0 at r0 (all local), 1 at r1 (all coarse)
#			 w = np.clip((d[use_loc] - r0) / (r1 - r0 + 1e-12), 0.0, 1.0)
#			 # if local returns NaN (outside local grid), fall back to coarse
#			 bad = ~np.isfinite(t_l)
#			 t_l = np.where(bad, t_c[use_loc], t_l)
#			 w = np.where(bad, 1.0, w)
#			 t[use_loc] = (1.0 - w) * t_l + w * t_c[use_loc]
#		 return t

#	 return interp

