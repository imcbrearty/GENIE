
import numpy as np
import torch
from scipy.spatial import cKDTree
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import gamma, beta
import time
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from numpy.matlib import repmat
import itertools
import pathlib
import yaml

device = torch.device('cuda') ## or use cpu


def load_config(file_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
### Projections

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

def lla2ecef_diff(p, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	p = p.detach().clone().float().to(device) # why include detach here?
	pi = torch.Tensor([np.pi]).to(device)
	p[:,0:2] = p[:,0:2]*torch.Tensor([pi/180.0, pi/180.0]).view(1,-1).to(device)
	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0])**2)
	# results:
	x = (N + p[:,2])*torch.cos(p[:,0])*torch.cos(p[:,1])
	y = (N + p[:,2])*torch.cos(p[:,0])*torch.sin(p[:,1])
	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0])

	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

def ecef2lla_diff(x, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	a = a.to(device)
	e = e.to(device)
	pi = torch.Tensor([np.pi]).to(device)
	b = torch.sqrt((a**2)*(1 - e**2))
	ep = torch.sqrt((a**2 - b**2)/(b**2))
	p = torch.sqrt(x[:,0]**2 + x[:,1]**2)
	th = torch.atan2(a*x[:,2], b*p)
	lon = torch.atan2(x[:,1], x[:,0])
	lat = torch.atan2((x[:,2] + (ep**2)*b*(torch.sin(th)**3)), (p - (e**2)*a*(torch.cos(th)**3)))
	N = a/torch.sqrt(1 - (e**2)*(torch.sin(lat)**2))
	alt = p/torch.cos(lat) - N
	# lon = np.mod(lon, 2.0*np.pi) # don't use!
	k = (torch.abs(x[:,0]) < 1) & (torch.abs(x[:,1]) < 1)
	alt[k] = torch.abs(x[k,2]) - b
	
	return torch.cat((180.0*lat[:,None]/pi, 180.0*lon[:,None]/pi, alt[:,None]), axis = 1)

def rotation_matrix(a, b, c):

	# a, b, c = vec

	rot = torch.zeros(3,3)
	rot[0,0] = torch.cos(b)*torch.cos(c)
	rot[0,1] = torch.sin(a)*torch.sin(b)*torch.cos(c) - torch.cos(a)*torch.sin(c)
	rot[0,2] = torch.cos(a)*torch.sin(b)*torch.cos(c) + torch.sin(a)*torch.sin(c)

	rot[1,0] = torch.cos(b)*torch.sin(c)
	rot[1,1] = torch.sin(a)*torch.sin(b)*torch.sin(c) + torch.cos(a)*torch.cos(c)
	rot[1,2] = torch.cos(a)*torch.sin(b)*torch.sin(c) - torch.sin(a)*torch.cos(c)

	rot[2,0] = -torch.sin(b)
	rot[2,1] = torch.sin(a)*torch.cos(b)
	rot[2,2] = torch.cos(a)*torch.cos(b)

	return rot

def rotation_matrix_full_precision(a, b, c):

	# a, b, c = vec

	rot = np.zeros((3,3))
	rot[0,0] = np.cos(b)*np.cos(c)
	rot[0,1] = np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c)
	rot[0,2] = np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)

	rot[1,0] = np.cos(b)*np.sin(c)
	rot[1,1] = np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c)
	rot[1,2] = np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c)

	rot[2,0] = -np.sin(b)
	rot[2,1] = np.sin(a)*np.cos(b)
	rot[2,2] = np.cos(a)*np.cos(b)

	return rot

### K-means scripts

def kmeans_packing(scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 5000, n_sim = 1, lr = 0.01):

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

			tree = cKDTree(ftrns1(v))
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x
			q, ip = tree.query(ftrns1(x))

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz

def kmeans_packing_weight_vector(weight_vector, scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 5000, n_sim = 1, lr = 0.01):

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

			tree = cKDTree(ftrns1(v)*weight_vector)
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x
			q, ip = tree.query(ftrns1(x)*weight_vector)

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz
	
def kmeans_packing_weight_vector_with_density(m_density, weight_vector, scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 1000, n_sim = 1, frac = 0.75, lr = 0.01):

	## Frac specifies how many of the random samples are from the density versus background

	n1 = int(n_clusters*frac) ## Number to sample from density
	n2 = n_clusters - n1 ## Number to sample uniformly

	n1_sample = int(n_batch*frac)
	n2_sample = n_batch - n1_sample

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v1 = m_density.sample(n1)
				v1 = np.concatenate((v1, np.random.rand(n1).reshape(-1,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
				v2 = np.random.rand(n2, ndim)*scale_x + offset_x
				v = np.concatenate((v1, v2), axis = 0)

				iremove = np.where(((v[:,0] > (offset_x[0,0] + scale_x[0,0])) + ((v[:,1] > (offset_x[0,1] + scale_x[0,1]))) + (v[:,0] < offset_x[0,0]) + (v[:,1] < offset_x[0,1])) > 0)[0]
				if len(iremove) > 0:
					v[iremove] = np.random.rand(len(iremove), ndim)*scale_x + offset_x

			tree = cKDTree(ftrns1(v)*weight_vector)
			x1 = m_density.sample(n1)
			x1 = np.concatenate((x1, np.random.rand(n1).reshape(-1,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
			x2 = np.random.rand(n2, ndim)*scale_x + offset_x
			x = np.concatenate((x1, x2), axis = 0)
			iremove = np.where(((x[:,0] > (offset_x[0,0] + scale_x[0,0])) + ((x[:,1] > (offset_x[0,1] + scale_x[0,1]))) + (x[:,0] < offset_x[0,0]) + (x[:,1] < offset_x[0,1])) > 0)[0]
			if len(iremove) > 0:
				x[iremove] = np.random.rand(len(iremove), ndim)*scale_x + offset_x

			q, ip = tree.query(ftrns1(x)*weight_vector)

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz

def kmeans_packing_sampling_points(scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 1000, n_sim = 3, lr = 0.01):

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

			tree = cKDTree(ftrns1(v))
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x
			q, ip = tree.query(ftrns1(x))

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz

### TRAVEL TIMES ###

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

def interp_1D_velocity_model_to_3D_travel_times(X, locs, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2):

	i1 = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]])
	x10, x20, x30 = Xmin
	Xv = X - np.array([x10,x20,x30])[None,:]
	depth_grid = np.copy(locs[:,2]).reshape(1,-1)
	mask = np.array([1.0,1.0,0.0]).reshape(1,-1)

	print('Check way X0 is used')

	def evaluate_func(y, x):

		y, x = y.cpu().detach().numpy(), x.cpu().detach().numpy()

		ind_depth = np.tile(np.argmin(np.abs(y[:,2].reshape(-1,1) - depth_grid), axis = 1), x.shape[0])
		rel_pos = (np.expand_dims(x, axis = 1) - np.expand_dims(y*mask, axis = 0)).reshape(-1,3)
		rel_pos[:,0:2] = np.abs(rel_pos[:,0:2]) ## Postive relative pos, relative to X0.
		x_relative = X0 + rel_pos

		xv = x_relative - Xmin[None,:]
		nx = np.shape(rel_pos)[0] # nx is the number of query points in x
		nz_vals = np.array([np.rint(np.floor(xv[:,0]/Dx[0])),np.rint(np.floor(xv[:,1]/Dx[1])),np.rint(np.floor(xv[:,2]/Dx[2]))]).T
		nz_vals1 = np.minimum(nz_vals, N - 2)

		nz = (np.reshape(np.dot((np.repeat(nz_vals1, 8, axis = 0) + repmat(i1,nx,1)),Mn.T),(nx,8)).T).astype('int')

		val_p = Tp[nz,ind_depth]
		val_s = Ts[nz,ind_depth]

		x0 = np.reshape(xv,(1,nx,3)) - Xv[nz,:]
		x0 = (1 - abs(x0[:,:,0])/Dx[0])*(1 - abs(x0[:,:,1])/Dx[1])*(1 - abs(x0[:,:,2])/Dx[2])

		val_p = np.sum(val_p*x0, axis = 0).reshape(-1, y.shape[0])
		val_s = np.sum(val_s*x0, axis = 0).reshape(-1, y.shape[0])

		return torch.Tensor(np.concatenate((val_p[:,:,None], val_s[:,:,None]), axis = 2)).to(device)

	return lambda y, x: evaluate_func(y, x)


### PREP INPUTS ###
def assemble_time_pointers_for_stations(trv_out, dt = 1.0, k = 10, win = 10.0, tbuffer = 10.0):

	n_temp, n_sta = trv_out.shape[0:2]
	dt_partition = np.arange(-win, win + trv_out.max() + dt + tbuffer, dt)

	edges_p = []
	edges_s = []
	for i in range(n_sta):
		tree_p = cKDTree(trv_out[:,i,0][:,np.newaxis])
		tree_s = cKDTree(trv_out[:,i,1][:,np.newaxis])
		q_p, ip_p = tree_p.query(dt_partition[:,np.newaxis], k = k)
		q_s, ip_s = tree_s.query(dt_partition[:,np.newaxis], k = k)
		# ip must project to Lg indices.
		edges_p.append((ip_p*n_sta + i).reshape(-1)) # Lg indices are each source x n_sta + sta_ind. The reshape places each subsequence of k, per time step, per station.
		edges_s.append((ip_s*n_sta + i).reshape(-1)) # Lg indices are each source x n_sta + sta_ind. The reshape places each subsequence of k, per time step, per station.
		# Overall, each station, each time step, each k sets of edges.
	edges_p = np.hstack(edges_p)
	edges_s = np.hstack(edges_s)

	return edges_p, edges_s, dt_partition

def assemble_time_pointers_for_stations_multiple_grids(trv_out, max_t, dt = 1.0, k = 10, win = 10.0):

	n_temp, n_sta = trv_out.shape[0:2]
	dt_partition = np.arange(-win, win + max_t + dt, dt)

	edges_p = []
	edges_s = []
	for i in range(n_sta):
		tree_p = cKDTree(trv_out[:,i,0][:,np.newaxis])
		tree_s = cKDTree(trv_out[:,i,1][:,np.newaxis])
		q_p, ip_p = tree_p.query(dt_partition[:,np.newaxis], k = k)
		q_s, ip_s = tree_s.query(dt_partition[:,np.newaxis], k = k)
		# ip must project to Lg indices.
		edges_p.append((ip_p*n_sta + i).reshape(-1))
		edges_s.append((ip_s*n_sta + i).reshape(-1))
		# Overall, each station, each time step, each k sets of edges.
	edges_p = np.hstack(edges_p)
	edges_s = np.hstack(edges_s)

	return edges_p, edges_s, dt_partition


## ACTUAL UTILS
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


## Load Files
def load_files(path_to_file, name_of_project, template_ver, vel_model_ver):

	# Load region
	z = np.load(path_to_file + '%s_region.npz' % name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz' % (name_of_project, template_ver))
	x_grids = z['x_grids']
	z.close()

	# Load stations
	z = np.load(path_to_file + '%s_stations.npz' % name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	# Load travel times
	z = np.load(path_to_file + '1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz' % (name_of_project, vel_model_ver))
	depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
	z.close()
	
	# You can extract further variables from z if needed here

	# Create path to write files
	write_training_file = path_to_file + 'GNN_TrainedModels/' + name_of_project + '_'

	return lat_range, lon_range, depth_range, deg_pad, x_grids, locs, stas, mn, rbest, write_training_file, depths, vp, vs

def load_files_with_travel_times(path_to_file, name_of_project, template_ver, vel_model_ver):

	# Load region
	z = np.load(path_to_file + '%s_region.npz' % name_of_project)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	# Load templates
	z = np.load(path_to_file + 'Grids/%s_seismic_network_templates_ver_%d.npz' % (name_of_project, template_ver))
	x_grids = z['x_grids']
	z.close()

	# Load stations
	z = np.load(path_to_file + '%s_stations.npz' % name_of_project)
	locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
	z.close()

	# Load travel times
	z = np.load(path_to_file + '1D_Velocity_Models_Regional/%s_1d_velocity_model_ver_%d.npz' % (name_of_project, vel_model_ver))
	depths, vp, vs = z['Depths'], z['Vp'], z['Vs']
	
	Tp = z['Tp_interp']
	Ts = z['Ts_interp']

	locs_ref = z['locs_ref']
	X = z['X']
	z.close()
	
	# You can extract further variables from z if needed here

	# Create path to write files
	write_training_file = path_to_file + 'GNN_TrainedModels/' + name_of_project + '_'

	return lat_range, lon_range, depth_range, deg_pad, x_grids, locs, stas, mn, rbest, write_training_file, depths, vp, vs, Tp, Ts, locs_ref, X
