
import numpy as np
import torch
from torch import nn, optim
from scipy.io import loadmat
from matplotlib import pyplot as plt
from runpy import run_path
from argparse import Namespace
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import time
from joblib import Parallel, delayed
import multiprocessing
import skfmm
import h5py
from joblib import Parallel, delayed
from numpy.matlib import repmat
from scipy.interpolate import RegularGridInterpolator
# import density_field_library as DFL
from scipy.interpolate import interp1d
from numpy import interp
# import netCDF4 as nc
import sys
import glob
import shutil
import pathlib

## Initilize
import os

dx = 500.0 # Cartesian distance between nodes in FMM computation
d_deg = 0.005 # Degree distance between saved interpolation query points
dx_depth = 500.0 # Depth distance between nodes in FMM computation and saved query points
depth_steps = np.arange(-500.0, 4000.0 + 150, 150) # Elevation steps to compute travel times from 
## (These are reference points for stations; can be a regular grid, and each station looks up the
## travel times with respect to these values. It is discretized (nearest-neighber) rather than continous.

def lla2ecef(p, a = 6378137.0, e = 8.18191908426215e-2): # 0.0818191908426215, previous 8.1819190842622e-2
	p = p.copy().astype('float')
	p[:,0:2] = p[:,0:2]*np.array([np.pi/180.0, np.pi/180.0]).reshape(1,-1)
	N = a/np.sqrt(1 - (e**2)*np.sin(p[:,0])**2)
    # results:
	x = (N + p[:,2])*np.cos(p[:,0])*np.cos(p[:,1])
	y = (N + p[:,2])*np.cos(p[:,0])*np.sin(p[:,1])
	z = ((1-e**2)*N + p[:,2])*np.sin(p[:,0])
	return np.concatenate((x[:,None],y[:,None],z[:,None]), axis = 1)

def ecef2lla(x, a = 6378137.0, e = 8.18191908426215e-2):
	x = x.copy().astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	b = np.sqrt((a**2)*(1 - e**2))
	ep = np.sqrt((a**2 - b**2)/(b**2))
	p = np.sqrt(x[:,0]**2 + x[:,1]**2)
	th = np.arctan2(a*x[:,2], b*p)
	lon = np.arctan2(x[:,1], x[:,0])
	lat = np.arctan2((x[:,2] + (ep**2)*b*(np.sin(th)**3)), (p - (e**2)*a*(np.cos(th)**3)))
	N = a/np.sqrt(1 - (e**2)*(np.sin(lat)**2))
	alt = p/np.cos(lat) - N
	# lon = np.mod(lon, 2.0*np.pi) # don't use!
	k = (np.abs(x[:,0]) < 1) & (np.abs(x[:,1]) < 1)
	alt[k] = np.abs(x[k,2]) - b
	return np.concatenate((180.0*lat[:,None]/np.pi, 180.0*lon[:,None]/np.pi, alt[:,None]), axis = 1)

def remove_mean(x, axis):
	
	return x - np.nanmean(x, axis = axis, keepdims = True)

## From https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(p, hull):
	"""
	Test if points in `p` are in `hull`
	`p` should be a `NxK` coordinates of `N` points in `K` dimensions
	`hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
	coordinates of `M` points in `K`dimensions for which Delaunay triangulation
	will be computed
	"""
	from scipy.spatial import Delaunay
	if not isinstance(hull,Delaunay):
	    hull = Delaunay(hull)

	return hull.find_simplex(p)>=0

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

def interp_3D_return_function_adaptive(X, Xmin, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2):

	nsta = Tp.shape[1]
	i1 = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]])
	x10, x20, x30 = Xmin
	Xv = X - np.array([x10,x20,x30])[None,:] 

	def evaluate_func(y, x):

		xv = x - np.array([x10,x20,x30])[None,:]
		nx = np.shape(x)[0] # nx is the number of query points in x
		nz_vals = np.array([np.rint(np.floor(xv[:,0]/Dx[0])),np.rint(np.floor(xv[:,1]/Dx[1])),np.rint(np.floor(xv[:,2]/Dx[2]))]).T
		nz_vals1 = np.minimum(nz_vals, N - 2)

		nz = (np.reshape(np.dot((np.repeat(nz_vals1, 8, axis = 0) + repmat(i1,nx,1)),Mn.T),(nx,8)).T).astype('int')
		val_p = Tp[nz,:]
		val_s = Ts[nz,:]

		x0 = np.reshape(xv,(1,nx,3)) - Xv[nz,:]
		x0 = (1 - abs(x0[:,:,0])/Dx[0])*(1 - abs(x0[:,:,1])/Dx[1])*(1 - abs(x0[:,:,2])/Dx[2])

		val_p = np.sum(val_p*x0[:,:,None], axis = 0)
		val_s = np.sum(val_s*x0[:,:,None], axis = 0)

		return np.concatenate((val_p[:,:,None], val_s[:,:,None]), axis = 2)

	return lambda y, x: evaluate_func(y, ftrns1(x))

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())

template_ver = 1
vel_model_ver = 1
## Load Files

if '\\' in path_to_file: ## Windows

	# Load region
	name_of_project = path_to_file.split('\\')[-1] ## Windows
	z = np.load(path_to_file + '\\%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + '\\Grids\\%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	x_grids = z['x_grids']
	z.close()

	# Load stations
	z = np.load(path_to_file + '\\%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	## Load travel times
	shutil.copy(path_to_file + '\\1d_velocity_model.npz', path_to_file + '\\1D_Velocity_Models_Regional\\%s_1d_velocity_model.npz'%name_of_project)
	z = np.load(path_to_file + '\\1D_Velocity_Models_Regional\\%s_1d_velocity_model.npz'%name_of_project)
	depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
	z.close()

else: ## Linux or Unix

	# Load region
	name_of_project = path_to_file.split('/')[-1] ## Windows
	z = np.load(path_to_file + '/%s_region.npz'%name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + '/Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, template_ver))
	x_grids = z['x_grids']
	z.close()

	# Load stations
	z = np.load(path_to_file + '/%s_stations.npz'%name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	## Load travel times
	shutil.copy(path_to_file + '/1d_velocity_model.npz', path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model.npz'%name_of_project)
	z = np.load(path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model.npz'%name_of_project)
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
rbest_cuda = torch.Tensor(rbest).cuda()
mn_cuda = torch.Tensor(mn).cuda()
ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x) - mn_cuda).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda)

lat_grid = np.arange(lat_range_extend[0], lat_range_extend[1] + d_deg, d_deg)
lon_grid = np.arange(lon_range_extend[0], lon_range_extend[1] + d_deg, d_deg)
depth_grid = np.arange(depth_range[0], depth_range[1] + dx_depth, dx_depth)

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

num_cores = multiprocessing.cpu_count()

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
x1 = np.arange(ys[:,0].min() - 15*dx, ys[:,0].max() + 15*dx, dx)
x2 = np.arange(ys[:,1].min() - 15*dx, ys[:,1].max() + 20*dx, dx)
x3 = np.arange(ys[:,2].min() - 2*dx, ys[:,2].max() + 2*dx, dx)
x11, x12, x13 = np.meshgrid(x1, x2, x3)
xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)
dx_v = np.array([np.diff(x1)[0], np.diff(x2)[0], np.diff(x3)[0]])

Xmin = xx.min(0)
Dx = [np.diff(x1[0:2]),np.diff(x2[0:2]),np.diff(x3[0:2])]
Mn = np.array([len(x3), len(x1)*len(x3), 1]) ## Is this off by one index? E.g., np.where(np.diff(xx[:,0]) != 0)[0] isn't exactly len(x3)

num_cores = 1

n_ver = 1
n_updates = 100

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

for nn in [0]: ## Only compute one travel time model

	Vp = interp(ftrns2(xx)[:,2], depth_grid, Vp_profile) ## This may be locating some depths at incorrect positions, due to projection.
	Vs = interp(ftrns2(xx)[:,2], depth_grid, Vs_profile)

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

	Tp_interp = np.zeros((X.shape[0], reciever_proj.shape[0]))
	Ts_interp = np.zeros((X.shape[0], reciever_proj.shape[0]))
	for j in range(reciever_proj.shape[0]): ## Need to check if it's ok to interpolate with 3D grids defined on lat,lon,depth, since depth
											## has a very different scale.
		mp = RegularGridInterpolator((x1, x2, x3), Tp[:,j].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		ms = RegularGridInterpolator((x1, x2, x3), Ts[:,j].reshape(len(x2), len(x1), len(x3)).transpose([1,0,2]), method = 'linear')
		Tp_interp[:,j] = mp(ftrns1(X))
		Ts_interp[:,j] = ms(ftrns1(X))

	np.savez_compressed(path_to_file + '/1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz'%(name_of_project, n_ver), xx = xx, xx_latlon = ftrns2(xx), X = X, locs_ref = locs_ref, Tp = Tp, Ts = Ts, Tp_interp = Tp_interp, Ts_interp = Ts_interp, Vp = Vp, Vs = Vs, Vp_profile = Vp_profile, Vs_profile = Vs_profile, depth_grid = depth_grid)
