
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
import cvxpy as cp
import itertools
import pathlib
import yaml


if torch.cuda.is_available() == True:
	device = torch.device('cuda') ## or use cpu
else:
	device = torch.device('cpu')


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
			x1 = m_density.sample(n1_sample)
			x1 = np.concatenate((x1, np.random.rand(n1_sample).reshape(-1,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
			x2 = np.random.rand(n2_sample, ndim)*scale_x + offset_x
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

def interp_1D_velocity_model_to_3D_travel_times(X, locs, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2, device = 'cuda'):

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

def optimize_station_selection(cnt_per_station, n_total):

	## Randomize order of counts
        iperm = np.random.permutation(np.arange(len(cnt_per_station)))
        cnt_per_station_perm = cnt_per_station[iperm] ## Permuted cnts per station

        ## Optimzation vector
        c = -np.copy(cnt_per_station_perm)
        A = np.ones((1,len(cnt_per_station_perm)))
        A[0,:] = cnt_per_station_perm
        b = n_total*np.ones((1,1))

        # Solve ILP
        x = cp.Variable(len(cnt_per_station_perm), integer = True)
        prob = cp.Problem(cp.Minimize(c.T@x), constraints = [A@x <= b.reshape(-1), 0 <= x, x <= 1])
        prob.solve()
        soln = np.round(x.value)

        assert prob.status == 'optimal', 'competitive assignment solution is not optimal'

        sta_grab = iperm[np.where(soln > 0)[0]]

        return sta_grab

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

def load_travel_time_neural_network(path_to_file, ftrns1, ftrns2, n_ver_load, phase = 'p_s', device = 'cuda', method = 'relative pairs'):

	from module import TravelTimes
	seperator = '\\' if '\\' in path_to_file else '/'
	
	z = np.load(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'travel_time_neural_network_%s_losses_ver_%d.npz'%(phase, n_ver_load))
	n_phases = z['out1'].shape[1]
	scale_val = float(z['scale_val'])
	trav_val = float(z['trav_val'])
	z.close()
	
	m = TravelTimes(ftrns1, ftrns2, scale_val = scale_val, trav_val = trav_val, n_phases = n_phases, device = device).to(device)
	m.load_state_dict(torch.load(path_to_file + '/1D_Velocity_Models_Regional/travel_time_neural_network_%s_ver_%d.h5'%(phase, n_ver_load), map_location = torch.device(device)))
	m.eval()

	if method == 'relative pairs':

		trv = lambda sta_pos, src_pos: m.forward_relative(sta_pos, src_pos, method = 'pairs')

	if method == 'direct':

		trv = lambda sta_pos, src_pos: m.forward_relative(sta_pos, src_pos, method = 'direct')
	
	return trv

def load_station_corrections(trv, locs, path_to_file, name_of_project, n_ver_corrections, ftrns1_diff, ind_use = None, trv_direct = None, device = 'cpu'):

	from calibration_utils import TrvTimesCorrection

	path_to_corrections = path_to_file + 'Grids/%s_calibrated_travel_time_corrections_%d.npz'%(name_of_project, n_ver_corrections)
	z = np.load(path_to_corrections)
	coefs, coefs_ker, x_grid_corr = torch.Tensor(z['coefs']).to(device), torch.Tensor(z['coefs_ker']).to(device), torch.Tensor(z['x_grid']).to(device)

	if ind_use is not None:
		coefs = coefs[:,ind_use,:]
		coefs_ker = coefs_ker[:,ind_use,:]
		locs_use = np.copy(locs[ind_use])
	else:
		locs_use = np.copy(locs)


	interp_type, k_spc_interp, _, sig_ker, grid_index = z['params']
	k_spc_interp = int(k_spc_interp)
	sig_ker = float(sig_ker)
	grid_index = int(grid_index)
	# assert(np.abs(x_grid_corr.cpu().detach().numpy() - x_grids[grid_index]).max() == 0)
	z.close()

	## Can we overwrite the function?
	return TrvTimesCorrection(trv, x_grid_corr, locs_use, coefs, ftrns1_diff, coefs_ker = coefs_ker, interp_type = 'anisotropic', k = k_spc_interp, trv_direct = trv_direct, sig = sig_ker)

def load_templates_region(trv, locs, x_grids, ftrns1, training_params, graph_params, pred_params, dt_embed = 1.0, device = 'cpu'):

	k_sta_edges, k_spc_edges, k_time_edges = graph_params

	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	x_grids_trv = []
	x_grids_trv_pointers_p = []
	x_grids_trv_pointers_s = []
	x_grids_trv_refs = []
	x_grids_edges = []

	for i in range(len(x_grids)):

		trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
		x_grids_trv.append(trv_out.cpu().detach().numpy())

		edge_index = knn(torch.Tensor(ftrns1(x_grids[i])).to(device), torch.Tensor(ftrns1(x_grids[i])).to(device), k = k_spc_edges).flip(0).contiguous()
		edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
		x_grids_edges.append(edge_index)

	max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # + 10.0

	for i in range(len(x_grids)):

		A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations_multiple_grids(x_grids_trv[i], max_t)
		x_grids_trv_pointers_p.append(A_edges_time_p)
		x_grids_trv_pointers_s.append(A_edges_time_s)
		x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?

	return x_grids, x_grids_edges, x_grids_trv, x_grids_trv_pointers_p, x_grids_trv_pointers_s, x_grids_trv_refs, max_t

def load_picks(path_to_file, date, thresh_cut = None, use_quantile = None, min_amplitude = None, n_ver = 1, spr_picks = 1):
	
	if '\\' in path_to_file:
		z = np.load(path_to_file + 'Picks\\%d\\%d_%d_%d_ver_%d.npz'%(date[0], date[0], date[1], date[2], n_ver))
	elif '/' in path_to_file:
		z = np.load(path_to_file + 'Picks/%d/%d_%d_%d_ver_%d.npz'%(date[0], date[0], date[1], date[2], n_ver))
		
	P = z['P']
	z.close()
	P[:,0] = P[:,0]/spr_picks
	
	if use_quantile is not None:
		iz = np.where(P[:,3] > np.quantile(P[:,3], use_quantile))[0]
		P = P[iz]

	if thresh_cut is not None:
		iz = np.where(P[:,3] > thresh_cut)[0]
		P = P[iz]

	ind_use = np.unique(P[:,1]).astype('int')
	
	# if min_amplitude is not None:
	# 	iz = np.where(P[:,2] < min_amplitude)[0]
	# 	P = np.delete(P, iz, axis = 0) # remove picks with amplitude less than min possible amplitude

	return P, ind_use # Note: this permutation of locs_use.

# def load_picks(path_to_file, date, locs, stas, lat_range, lon_range, thresh_cut = None, use_quantile = None, permute_indices = False, min_amplitude = None, n_ver = 1, spr_picks = 100):

# 	from obspy.core import UTCDateTime
	
# 	if '\\' in path_to_file:
# 		z = np.load(path_to_file + 'Picks\\%d\\%d_%d_%d_ver_%d.npz'%(date[0], date[0], date[1], date[2], n_ver))
# 	elif '/' in path_to_file:
# 		z = np.load(path_to_file + 'Picks/%d/%d_%d_%d_ver_%d.npz'%(date[0], date[0], date[1], date[2], n_ver))

# 	yr, mn, dy = date[0], date[1], date[2]
# 	t0 = UTCDateTime(yr, mn, dy)
# 	P, sta_names_use, sta_ind_use = z['P'], z['sta_names_use'], z['sta_ind_use']
	
# 	if use_quantile is not None:
# 		iz = np.where(P[:,3] > np.quantile(P[:,3], use_quantile))[0]
# 		P = P[iz]

# 	if thresh_cut is not None:
# 		iz = np.where(P[:,3] > thresh_cut)[0]
# 		P = P[iz]

# 	if min_amplitude is not None:
# 		iz = np.where(P[:,2] < min_amplitude)[0]
# 		P = np.delete(P, iz, axis = 0) # remove picks with amplitude less than min possible amplitude

# 	P_l = []
# 	locs_use = []
# 	sta_use = []
# 	ind_use = []
# 	sc = 0
# 	for i in range(len(sta_names_use)):
# 		iz = np.where(sta_names_use[i] == stas)[0]
# 		if len(iz) == 0:
# 			# print('no match')
# 			continue
# 		iz1 = np.where(P[:,1] == sta_ind_use[i])[0]
# 		if len(iz1) == 0:
# 			# print('no picks')
# 			continue		
# 		p_select = P[iz1]
# 		if permute_indices == True:
# 			p_select[:,1] = sc ## New indices
# 		else:
# 			p_select[:,1] = iz ## Absolute indices
# 		P_l.append(p_select)
# 		locs_use.append(locs[iz])
# 		sta_use.append(stas[iz])
# 		ind_use.append(iz)
# 		sc += 1

# 	P_l = np.vstack(P_l)
# 	P_l[:,0] = P_l[:,0]/spr_picks ## Convert pick indices to time (note: if spr_picks = 1, then picks are already in absolute time)
# 	locs_use = np.vstack(locs_use)
# 	sta_use = np.hstack(sta_use)
# 	ind_use = np.hstack(ind_use)

# 	## Make sure ind_use is unique set. Then re-select others.
# 	ind_use = np.sort(np.unique(ind_use))
# 	locs_use = locs[ind_use]
# 	sta_use = stas[ind_use]

# 	if permute_indices == True:
# 		argsort = np.argsort(ind_use)
# 		P_l_1 = []
# 		sc = 0
# 		for i in range(len(argsort)):
# 			iz1  = np.where(P_l[:,1] == argsort[i])[0]
# 			p_slice = P_l[iz1]
# 			p_slice[:,1] = sc
# 			P_l_1.append(p_slice)
# 			sc += 1
# 		P_l = np.vstack(P_l)

# 	# julday = int((UTCDateTime(yr, mn, dy) - UTCDateTime(yr, 1, 1))/(3600*24.0) + 1)

# 	if download_catalog == True:
# 		cat, _, event_type = download_catalog(lat_range, lon_range, min_magnitude, t0, t0 + 3600*24.0, t0 = t0)
# 	else:
# 		cat, event_type = [], []

# 	z.close()

# 	return P_l, ind_use # Note: this permutation of locs_use.

def download_catalog(lat_range, lon_range, min_magnitude, startime, endtime, t0 = None, client = 'NCEDC', include_arrivals = False):

	from obspy.core import UTCDateTime
	from obspy.clients.fdsn import Client	
	
	if t0 is None:
		t0 = UTCDateTime(2000, 1, 1)
	
	client = Client(client)
	cat_l = client.get_events(starttime = startime, endtime = endtime, minlatitude = lat_range[0], maxlatitude = lat_range[1], minlongitude = lon_range[0], maxlongitude = lon_range[1], minmagnitude = min_magnitude, includearrivals = include_arrivals, orderby = 'time-asc')

	# t0 = UTCDateTime(2021,4,1) ## zero time, for relative processing.
	time = np.array([cat_l[i].origins[0].time - t0 for i in np.arange(len(cat_l))])
	latitude = np.array([cat_l[i].origins[0].latitude for i in np.arange(len(cat_l))])
	longitude = np.array([cat_l[i].origins[0].longitude for i in np.arange(len(cat_l))])
	depth = np.array([cat_l[i].origins[0].depth for i in np.arange(len(cat_l))])
	mag = np.array([cat_l[i].magnitudes[0].mag for i in np.arange(len(cat_l))])
	event_type = np.array([cat_l[i].event_type for i in np.arange(len(cat_l))])

	cat = np.hstack([latitude.reshape(-1,1), longitude.reshape(-1,1), -1.0*depth.reshape(-1,1), time.reshape(-1,1), mag.reshape(-1,1)])

	return cat, cat_l, event_type
	
def visualize_predictions(out, lbls_query, pick_lbls, x_query, lp_times, lp_stations, locs_slice, data, ind, ext_save, depth_window = 10e3, deg_window = 1.0, thresh_source = 0.2, thresh_picks = 0.2, n_step = 0, n_ver = 1, min_norm_val = 0.3, close_plots = True):

	from matplotlib.colors import Normalize

	# Note: ind is the ind of the batch in train_GENIE_model.py
	
	raw_picks = False ## Only add this if we zoom in on the relevant part of data
	if raw_picks == True:

		fig, ax = plt.subplots(2,1, figsize = [12,8], sharex = True)
		ax[0].scatter(data[0][:,0], data[0][:,1])
		ax[1].scatter(data[0][:,0], data[0][:,1], c = (data[0][:,2] > -1))
		fig.savefig(ext_save + 'predictions_raw_picks_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	## Add theoretical moveout curves to this

	raw_picks_permuted = True
	if raw_picks_permuted == True:

		fig, ax = plt.subplots(1, figsize = [8,5], sharex = True)
		ax.scatter(lp_times, lp_stations)
		fig.savefig(ext_save + 'predictions_raw_picks_sorted_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	map_view_all_depths = True
	if map_view_all_depths == True:

		fig, ax = plt.subplots(1,2, figsize = [12,8])
		norm_scale = Normalize(0, np.maximum(lbls_query[:,5].max(), min_norm_val))
		ax[0].scatter(x_query[:,1], x_query[:,0], c = lbls_query[:,5], norm = norm_scale)
		ax[1].scatter(x_query[:,1], x_query[:,0], c = out[1][:,5,0].cpu().detach().numpy(), norm = norm_scale)
		fig.savefig(ext_save + 'predictions_map_view_all_depths_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	cross_section_fixed_depth = True
	if cross_section_fixed_depth == True:

		i1 = np.where(np.abs(x_query[:,2] - x_query[np.argmax(lbls_query[:,5]),2]) < depth_window)[0]

		fig, ax = plt.subplots(1,2, figsize = [12,8])
		norm_scale = Normalize(0, np.maximum(lbls_query[:,5].max(), min_norm_val))
		ax[0].scatter(x_query[i1,1], x_query[i1,0], c = lbls_query[i1,5], norm = norm_scale)
		ax[1].scatter(x_query[i1,1], x_query[i1,0], c = out[1][i1,5,0].cpu().detach().numpy(), norm = norm_scale)
		fig.savefig(ext_save + 'predictions_map_view_fixed_depth_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	cross_section_fixed_lat = True
	if cross_section_fixed_lat == True:

		i1 = np.where(np.abs(x_query[:,0] - x_query[np.argmax(lbls_query[:,5]),0]) < deg_window)[0]

		fig, ax = plt.subplots(1,2, figsize = [12,8])
		norm_scale = Normalize(0, np.maximum(lbls_query[:,5].max(), min_norm_val))
		ax[0].scatter(x_query[i1,1], x_query[i1,2], c = lbls_query[i1,5], norm = norm_scale)
		ax[1].scatter(x_query[i1,1], x_query[i1,2], c = out[1][i1,5,0].cpu().detach().numpy(), norm = norm_scale)
		fig.savefig(ext_save + 'predictions_cross_section_fixed_lat_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	cross_section_fixed_lon = True
	if cross_section_fixed_lon == True:

		i1 = np.where(np.abs(x_query[:,1] - x_query[np.argmax(lbls_query[:,5]),1]) < deg_window)[0]

		fig, ax = plt.subplots(1,2, figsize = [12,8])
		norm_scale = Normalize(0, np.maximum(lbls_query[:,5].max(), min_norm_val))
		ax[0].scatter(x_query[i1,0], x_query[i1,2], c = lbls_query[i1,5], norm = norm_scale)
		ax[1].scatter(x_query[i1,0], x_query[i1,2], c = out[1][i1,5,0].cpu().detach().numpy(), norm = norm_scale)
		fig.savefig(ext_save + 'predictions_cross_section_fixed_lon_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	associated_p_and_s_phases = True
	if associated_p_and_s_phases == True:

		ind_max = np.argmax(pick_lbls.sum(2).sum(1).cpu().detach().numpy())
		norm_scale = Normalize(0, pick_lbls[ind_max,:,:].max().cpu().detach().numpy())

		fig, ax = plt.subplots(2,2, figsize = [12,10])
		ax[0,0].scatter(lp_times, lp_stations, c = pick_lbls[ind_max,:,0].cpu().detach().numpy(), norm = norm_scale)
		ax[0,1].scatter(lp_times, lp_stations, c = pick_lbls[ind_max,:,1].cpu().detach().numpy(), norm = norm_scale)
		ax[1,0].scatter(lp_times, lp_stations, c = out[2][ind_max,:,0].cpu().detach().numpy(), norm = norm_scale)
		ax[1,1].scatter(lp_times, lp_stations, c = out[3][ind_max,:,0].cpu().detach().numpy(), norm = norm_scale)
		fig.savefig(ext_save + 'predictions_associated_p_and_s_phases_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)

	## Add plot to show multiple nearby source association predictions (using different colors for either source)
	
	map_view_associated_stations = True
	if map_view_associated_stations == True:

		itrue_sources = np.where(lbls_query[:,5] > thresh_source)[0]
		ipred_sources = np.where(out[1][:,5].cpu().detach().numpy() > thresh_source)[0]
		ind_max = np.argmax(pick_lbls.sum(2).sum(1).cpu().detach().numpy())

		itrue_picks = np.where(pick_lbls[ind_max,:,:].cpu().detach().numpy().max(1) > thresh_picks)[0]
		ipred_picks = np.where(np.concatenate((out[2][ind_max,:,0].cpu().detach().numpy().reshape(1,-1), out[3][ind_max,:,0].cpu().detach().numpy().reshape(1,-1)), axis = 0).max(0) > thresh_picks)[0]

		itrue_picks = lp_stations[itrue_picks.astype('int')].astype('int')
		ipred_picks = lp_stations[ipred_picks.astype('int')].astype('int')

		fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
		ax[0].scatter(x_query[itrue_sources,1], x_query[itrue_sources,0], c = lbls_query[itrue_sources,5], alpha = 0.2)
		ax[0].scatter(locs_slice[:,1], locs_slice[:,0], c = 'grey', marker = '^')
		ax[0].scatter(locs_slice[itrue_picks,1], locs_slice[itrue_picks,0], c = 'red', marker = '^')

		ax[1].scatter(x_query[ipred_sources,1], x_query[ipred_sources,0], c = out[1][ipred_sources,5,0].cpu().detach().numpy(), alpha = 0.2)
		ax[1].scatter(locs_slice[:,1], locs_slice[:,0], c = 'grey', marker = '^')
		ax[1].scatter(locs_slice[ipred_picks,1], locs_slice[ipred_picks,0], c = 'red', marker = '^')
		fig.savefig(ext_save + 'predictions_map_view_associated_phases_%d_step_%d_ver_%d.png'%(ind, n_step, n_ver), bbox_inches = 'tight', pad_inches = 0.2)
	
	if close_plots == True:

		plt.close('all')

	return True
