
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

# def lla2ecef_diff(p, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):
# 	# x = x.astype('float')
# 	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
# 	a = a.to(device)
# 	e = e.to(device)
# 	p = p.detach().clone().float().to(device) # why include detach here?
# 	pi = torch.Tensor([np.pi]).to(device)
# 	p[:,0:2] = p[:,0:2]*torch.Tensor([pi/180.0, pi/180.0]).view(1,-1).to(device)
# 	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0])**2)
# 	# results:
# 	x = (N + p[:,2])*torch.cos(p[:,0])*torch.cos(p[:,1])
# 	y = (N + p[:,2])*torch.cos(p[:,0])*torch.sin(p[:,1])
# 	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0])

# 	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

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

def generate_pseudo_lla_for_new_region(lla_B, mn_A, rbest_A, ftrns2_A):
    """
    Teleports Region B stations into Region A's footprint, 
    returning "pseudo" LLA coordinates that preserve local distances.
    """
    # --- Step 1: Compute a local reference frame for Region B ---
    # We use the mean of Region B to minimize distortion during the flattening phase
    mn_B = np.mean(lla2ecef(lla_B), axis=0, keepdims=True)
    
    # Generate a local rotation matrix (rbest_B) for Region B's center
    # This aligns Z with local up, X with East, Y with North for Region B
    lat_B_center = np.mean(lla_B[:, 0]) * np.pi / 180.0
    lon_B_center = np.mean(lla_B[:, 1]) * np.pi / 180.0
    
    # Standard local ENU (East-North-Up) rotation matrix formation
    rbest_B = np.array([
        [-np.sin(lon_B_center), np.cos(lon_B_center), 0],
        [-np.sin(lat_B_center)*np.cos(lon_B_center), -np.sin(lat_B_center)*np.sin(lon_B_center), np.cos(lat_B_center)],
        [np.cos(lat_B_center)*np.cos(lon_B_center), np.cos(lat_B_center)*np.sin(lon_B_center), np.sin(lat_B_center)]
    ])
    
    # --- Step 2: Convert Region B to its own local Cartesian space ---
    # (Distance and depth geometry are now frozen into a flat metric grid)
    xyz_local_B = (rbest_B @ (lla2ecef(lla_B) - mn_B).T).T
    
    # --- Step 3: Reverse project using Region A's parameters ---
    # We pretend xyz_local_B is actually native to Region A, and ask:
    # "What global LLA coordinates would produce this exact local Cartesian layout in A?"
    pseudo_lla_B = ftrns2_A(xyz_local_B)
    
    return pseudo_lla_B, mn_B, rbest_B

def pseudo_lla_to_real_lla(pseudo_lla_events, ftrns1_A, mn_B, rbest_B):
    """
    Takes events/points located in the pseudo-LLA domain and maps them
    back to their true, real-world global LLA coordinates.
    """
    # --- Step 1: Project pseudo-points into Region A's local Cartesian space ---
    # This recovers the exact, un-distorted local (X, Y, Z) geometry
    xyz_local = ftrns1_A(pseudo_lla_events)
    
    # --- Step 2: Un-rotate and Un-center using Region B's parameters ---
    # We invert the local transformation: (rbest_B.T @ xyz_local) + mn_B
    # (Note: Since rbest_B is an orthogonal rotation matrix, its transpose is its inverse)
    ecef_real = (rbest_B.T @ xyz_local.T).T + mn_B
    
    # --- Step 3: Convert absolute ECEF back to real-world LLA ---
    real_lla_events = ecef2lla(ecef_real)
    
    return real_lla_events

def get_analytical_transform(center_loc):
    """
    Analytically computes the exact rbest and mn to center and align 
    the local coordinate system, replacing differential evolution.
    
    center_loc: numpy array of shape (1, 3) -> [lat, lon, alt] (in degrees)
    """
    # 1. Compute mn (The center point in absolute ECEF)
    mn = lla2ecef(center_loc.reshape(1, -1)) # Shape: (1, 3)
    
    # Convert lat/lon of center to radians
    lat_rad = np.radians(center_loc[0, 0])
    lon_rad = np.radians(center_loc[0, 1])
    
    # 2. Compute the exact local East, North, and Up unit vectors
    # Row 0: East direction (points along +X in your local system)
    east  = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    
    # Row 1: North direction (points along +Y in your local system)
    north = np.array([-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)])
    
    # Row 2: Up direction (points along +Z in your local system)
    up    = np.array([np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)])
    
    # Stack them to create the clean rbest rotation matrix
    rbest = np.vstack([east, north, up]) # Shape: (3, 3)
    
    return rbest, mn

def hash_rows(val):
	return val[:,0].to(torch.int64) << 32 | val[:,1].to(torch.int64)

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

def kmeans_packing_spherical(scale_x, offset_x, ndim, n_clusters, ftrns1, ftrns2, n_batch = 3000, n_steps = 5000, weights = [1.0, 1.0, 2.0], n_sim = 1, lr = 0.01):

	# ftrns1_sphere_unit = lambda pos: lla2ecef(pos, e = 0.0, a = 1.0) # a = 6378137.0, e = 8.18191908426215e-2
	# ftrns2_sphere_unit = lambda pos: ecef2lla(pos, e = 0.0, a = 1.0)

	ftrns1_sphere_unit = lambda pos: lla2ecef(pos, a = 1.0, e = 0.0) # a = 6378137.0, e = 8.18191908426215e-2
	ftrns2_sphere_unit = lambda pos: ecef2lla(pos, a = 1.0, e = 0.0)
	from scipy.stats import beta

	if weights is not None:
		weights = np.array(weights).reshape(1,-1)
	else:
		weights = np.array([1.0, 1.0, 1.0]).reshape(1,-1)

	def spherical_packing_nodes(n, use_depth_scale = True, use_rotate = True, izero = 0.65):

		## Based on The Fibonacci Lattice
		## https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

		# n = 30000
		i = np.arange(0, n).astype('float') + 0.5
		phi = np.arccos(1 - 2*i/n)
		goldenRatio = (1 + 5**0.5)/2
		theta = 2*np.pi * i / goldenRatio
		x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
		xlat = ftrns2_sphere_unit(np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis = 1))

		if use_rotate == True:
			rand_val = [np.random.rand()*2.0*np.pi for i in range(3)]
			rotate = rotation_matrix_full_precision(rand_val[0], rand_val[1], rand_val[2])
			xlat = ftrns2_sphere_unit(ftrns1_sphere_unit(xlat) @ rotate)

		if use_depth_scale is not None:
			xlat[:,2] = offset_x[0][2] + np.random.rand()*scale_x[0][2]

			if izero is not None:

				ichoose = np.random.choice(len(xlat), size = int(izero*len(xlat)))
				xlat[ichoose,2] = (1.0 - beta.rvs(1.0, 3.0, size = len(ichoose)))*scale_x[0,2] + offset_x[0,2]

				ichoose = np.random.choice(len(xlat), size = int(izero*len(xlat)))
				xlat[ichoose,2] = (1.0 - beta.rvs(1.0, 12.0, size = len(ichoose)))*scale_x[0,2] + offset_x[0,2]

		return xlat

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = spherical_packing_nodes(n_clusters)
				v1 = ftrns1(np.copy(v)*weights)
				v = ftrns1(v) # np.random.rand(n_clusters, ndim)*scale_x + offset_x
				# v1 = ftrns1(v1)

			tree = cKDTree(v1)
			x = spherical_packing_nodes(n_batch)
			x1 = ftrns1(np.copy(x)*weights)
			x = ftrns1(x)
			q, ip = tree.query(x1)

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
		x = spherical_packing_nodes(n_batch)
		x1 = ftrns1(np.copy(x)*weights)
		x = ftrns1(x)		
		q, ip = tree.query(x1)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return ftrns2(V_results[ibest]), V_results, Losses, losses, rz

def kmeans_packing_fit_sources(srcs, scale_x, offset_x, ndim, n_clusters, ftrns1, ftrns2, n_batch = 3000, n_steps = 5000, blur_sigma = 30e3, weights = [1.0, 1.0, 2.0], n_sim = 1, lr = 0.01):

	# ftrns1_sphere_unit = lambda pos: lla2ecef(pos, e = 0.0, a = 1.0) # a = 6378137.0, e = 8.18191908426215e-2
	# ftrns2_sphere_unit = lambda pos: ecef2lla(pos, e = 0.0, a = 1.0)

	ftrns1_sphere_unit = lambda pos: lla2ecef(pos, a = 1.0, e = 0.0) # a = 6378137.0, e = 8.18191908426215e-2
	ftrns2_sphere_unit = lambda pos: ecef2lla(pos, a = 1.0, e = 0.0)
	from scipy.stats import beta

	if weights is not None:
		weights = np.array(weights).reshape(1,-1)
	else:
		weights = np.array([1.0, 1.0, 1.0]).reshape(1,-1)

	def sample_sources(n):

		xlat = ftrns2(ftrns1(srcs[np.random.choice(len(srcs), size = n),0:3]) + np.random.randn(n,3)*blur_sigma)
		iwhere = np.where(((xlat[:,2] < offset_x[0,2]) + (xlat[:,2] > (offset_x[0,2] + scale_x[0,2]))) > 0)[0]
		xlat[iwhere,2] = np.random.rand(len(iwhere))*scale_x[0,2] + offset_x[0,2]

		return xlat

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = sample_sources(n_clusters)
				v1 = ftrns1(np.copy(v)*weights)
				v = ftrns1(v) # np.random.rand(n_clusters, ndim)*scale_x + offset_x
				# v1 = ftrns1(v1)

			tree = cKDTree(v1)
			x = sample_sources(n_batch)
			x1 = ftrns1(np.copy(x)*weights)
			x = ftrns1(x)
			q, ip = tree.query(x1)

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
		x = sample_sources(n_batch)
		x1 = ftrns1(np.copy(x)*weights)
		x = ftrns1(x)		
		q, ip = tree.query(x1)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return ftrns2(V_results[ibest]), V_results, Losses, losses, rz


def collect_regular_lattice(n_trgt, lat_range = None, lon_range = None, use_global = False, tol_fraction = 0.01, max_iter = 100):

	r = Area/Area_globe
	n_low = max(1, int((n_trgt / r) * 0.2))
	n_high = int((n_trgt / r) * 3.0) + 1
	n_current = int(0.5*(n_low + n_high)) if use_global == False else n_trgt

	## Set tolerance as ~1% of grid, and then retain only this fraction
	tol = int(np.floor(tol_fraction*n_trgt))


	def random_rotation_matrix():
	    u1, u2, u3 = np.random.rand(3)

	    q = np.array([
	        np.sqrt(1 - u1) * np.sin(2*np.pi*u2),
	        np.sqrt(1 - u1) * np.cos(2*np.pi*u2),
	        np.sqrt(u1)     * np.sin(2*np.pi*u3),
	        np.sqrt(u1)     * np.cos(2*np.pi*u3)
	    ])

	    w, x, y, z = q
	    return np.array([
	        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
	        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
	        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
	    ])


	def fibonacci_sphere_latlon(N):

	    # Golden ratio constants
	    phi = (1 + 5**0.5) / 2
	    alpha = 1 / phi

	    # Random phases
	    delta_phi, delta_z = np.random.rand(2)

	    n = np.arange(N)

	    # Fibonacci sphere (vectorized)
	    z = 1 - 2 * (n + delta_z) / N
	    theta = 2 * np.pi * (n * alpha + delta_phi)
	    r = np.sqrt(1 - z*z)

	    x = r * np.cos(theta)
	    y = r * np.sin(theta)

	    P = np.column_stack((x, y, z))

	    # Random global rotation
	    R = random_rotation_matrix()
	    P = P @ R.T

	    # Cartesian → lat/lon
	    lon = 180.0*np.arctan2(P[:, 1], P[:, 0])/np.pi        # [-pi, pi)
	    lat = 180.0*np.arcsin(np.clip(P[:, 2], -1, 1))/np.pi  # [-pi/2, pi/2]

	    return np.concatenate((lat.reshape(-1,1), lon.reshape(-1,1)), axis = 1)


	iter_cnt = 0
	found_grid = False
	while (iter_cnt < max_iter)*(found_grid == False):


		points = fibonacci_sphere_latlon(n_current)
		if use_global == False:
			ifind = np.where((points[:,0] < lat_range[1])*(points[:,0] > lat_range[0])*(points[:,1] < lon_range[1])*(points[:,1] > lon_range[0]))[0]
			points = points[ifind]

		n_pts = len(points)
		if (np.abs(n_pts - n_trgt) <= tol)*(n_pts >= n_trgt):
			found_grid = True

		else:
			if n_pts < n_trgt:
				n_low = n_current
			else:
				n_high = n_current
			n_current = int(0.5*(n_low + n_high))

			# n_current = int(0.5*(n_low + n_high))
		iter_cnt += 1
		print('Iter: %d, Diff: %d'%(iter_cnt, n_pts - n_trgt))

	if n_pts > n_trgt:
		points = points[0:n_trgt] # , size = n_trgt, replace = False)]

	return points



# def knn_distance(x_proj1, x_proj2, idx1, idx2, centroid_proj, k = 10):

# 	if isinstance(x_proj1, np.ndarray):

# 		dist_ref = knn(torch.Tensor(centroid_proj[idx1]).to(device), torch.Tensor(centroid_proj[idx1]).to(device))

def knn_distance(
	x_rel_query,      # Tensor [M, 3] float32: relative coords of query points
	x_rel_db,         # Tensor [N, 3] float32: relative coords of database points (can == query for self)
	idx_query,        # LongTensor [M]: centroid indices for queries
	idx_db,           # LongTensor [N]: centroid indices for database
	centroids,        # Tensor [C, 3] float64: absolute centroid positions
	k=10,
	device='cuda' if torch.cuda.is_available() else 'cpu',
	return_edges = True,
	use_self_loops = True
):

	if isinstance(x_rel_query, np.ndarray):
		"""
		Returns K-nearest neighbors (distances and indices) using relative coords + centroids.
		"""
		# Move everything to device
		x_rel_query = torch.Tensor(x_rel_query).to(device)
		x_rel_db = torch.Tensor(x_rel_db).to(device)
		idx_query = torch.Tensor(idx_query).long().to(device)
		idx_db = torch.Tensor(idx_db).long().to(device)
		centroids = torch.Tensor(centroids).to(device)  # float64 preserved
    
	# Reconstruct effective absolute positions
	abs_query = centroids[idx_query] + x_rel_query  # [M, 3]
	abs_db = centroids[idx_db] + x_rel_db          # [N, 3]
    
	# Pairwise distances (exact, GPU-optimized)
	D = torch.cdist(abs_query, abs_db)  # [M, N], float64 if centroids dominate
    
	# Top-K smallest
	if use_self_loops == False:
		distances, indices = torch.topk(D, k + 1, dim=1, largest=False, sorted=True) ## This distance not reliable due to limited GPU ram, must re-compute for subset of indices
		distances = distances[:,1::]
		indices = indices[:,1::]
	else:
		distances, indices = torch.topk(D, k, dim=1, largest=False, sorted=True) ## This distance not reliable due to limited GPU ram, must re-compute for subset of indices


	rel_refs = (~(idx_query.reshape(-1,1) == idx_db[indices])).unsqueeze(2)*(centroids[idx_query].unsqueeze(1) - centroids[idx_db[indices]])
	rel_local = x_rel_query.unsqueeze(1) - x_rel_db[indices]
	distances = torch.norm(rel_refs + rel_local, dim = 2)

	# distances = torch.norm()

	if return_edges == True:

		edges = torch.cat((indices.reshape(1,-1), torch.arange(len(x_rel_query), device = device).repeat_interleave(k).reshape(1,-1)), dim = 0)

		return edges, distances.reshape(-1,1), (rel_refs + rel_local).reshape(-1,rel_refs.shape[2])

	else:

		return distances, indices  # distances: [M, k], indices: [M, k] (into db)


# def generate_travel_time_noise(
#     t_r, phase_type="P", sigma_pick=0.05, sigma_path_max=1.20, T_c=150.0, scale_extra = 1.0
# ):
#     """Generates path-independent, scale-invariant travel-time noise.

#     Noise for a given ray path depends ONLY on its own travel time t_r.
#     Default parameters reflect global P-wave residual budgets (e.g. AK135 /
#     IASP91).
#     """
#     multiplier = 2.2 if str(phase_type).upper() == "S" else 1.0

#     # Calculate per-ray sigma (strictly local to each t_r value)
#     sigma_sec = (
#         sigma_pick + sigma_path_max * (1.0 - np.exp(-t_r / T_c))
#     ) * multiplier

#     return np.random.normal(loc=0.0, scale = scale_extra*sigma_sec)


# import numpy as np


# def generate_travel_time_noise(
#     t_r,
#     phase_type = "P",
#     distribution = "laplace",  # Options: "laplace" or "gaussian"
#     sigma_pick = 0.08, # 0.05 (or 0.1)
#     sigma_path_max = 1.20,
#     T_c = 150.0,
# 	scale_extra = 1.0
# ): # Clean / Benchmark Data (Current Defaults):sigma_pick = 0.05, sigma_path_max = 1.2 $\rightarrow$ ($\text{Core Spread } \approx \pm 2\text{s}$)Noisy / Automated Picker Data:sigma_pick = 0.15, sigma_path_max = 2.0 $\rightarrow$ ($\text{Core Spread } \approx \pm 3.5\text{s}$)
#     """Generates path-independent travel-time noise using either Laplace (heavy-

#     tailed) or Gaussian distributions.

#     Parameters:
#         t_r (np.ndarray): Travel times in seconds.
#         phase_type (str): "P" or "S".
#         distribution (str): "laplace" (realistic, L1 robust) or "gaussian" (L2).
#         sigma_pick (float): Base picking/clock error in seconds.
#         sigma_path_max (float): Asymptote for global path heterogeneity in
#           seconds.
#         T_c (float): Characteristic saturation time scale in seconds.
#     """
#     # 1. S-wave variance multiplier (~2.2x)
#     multiplier = 2.2 if str(phase_type).upper() == "S" else 1.0

#     # 2. Calculate per-ray target standard deviation (sigma)
#     sigma_sec = (
#         sigma_pick + sigma_path_max * (1.0 - np.exp(-t_r / T_c))
#     ) * multiplier

#     # 3. Sample based on requested distribution
#     if distribution.lower() == "laplace":
#         # For Laplace: std_dev = sqrt(2) * beta  =>  beta = sigma / sqrt(2)
#         beta = sigma_sec / np.sqrt(2.0)
#         return np.random.laplace(loc=0.0, scale = scale_extra*beta)
#     elif distribution.lower() in ["gaussian", "normal"]:
#         return np.random.normal(loc=0.0, scale = scale_extra*sigma_sec)
#     else:
#         raise ValueError(f"Unsupported distribution: {distribution}")


import numpy as np


def generate_travel_time_noise(
    t_r,
    phase_input="P",  # Can be a string ("P"/"S") OR an array/list of 0s and 1s
    distribution="laplace",  # Options: "laplace" or "gaussian"
    sigma_pick=0.08,  # ~80ms base jitter for ML auto-pickers
    sigma_path_max=1.20,  # Asymptote for global path heterogeneity
    T_c=150.0,  # Characteristic saturation time scale
    scale_extra=1.0,  # Global multiplier to scale noise up/down
    s_wave_multiplier=2.2,  # Relative variance factor for S-waves
    excess_threshold_sigma=2.0,  # Outlier threshold factor (N * sigma)
    return_sigma=False,  # Option to return sigma_sec array
):
    """Generates path-independent travel-time noise for single-phase strings or

    mixed P/S phase arrays.

    Parameters:
        t_r (np.ndarray or float): Exact travel times in seconds.
        phase_input (str, list, or np.ndarray): Either "P"/"S" string for all
          picks, OR array-like of 0s (P) and 1s (S).
        distribution (str): "laplace" (heavy-tailed) or "gaussian".
        sigma_pick (float): Base picking/clock error in seconds for P-waves.
        sigma_path_max (float): Asymptote for global path heterogeneity in
          seconds for P-waves.
        T_c (float): Characteristic saturation time scale in seconds.
        scale_extra (float): Global multiplier applied to noise level.
        s_wave_multiplier (float): Factor applied to S-wave noise (default:
          2.2).
        excess_threshold_sigma (float): Multiplier on sigma_sec to flag
          excess-noise outliers.
        return_sigma (bool): Whether to return sigma_sec alongside noise and
          masks.
    """
    t_r = np.asarray(t_r)

    # 1. Parse phase_input: Handle String vs. Vector Array
    if isinstance(phase_input, str):
        # String case: e.g., "P" or "S"
        mult_val = s_wave_multiplier if phase_input.upper() == "S" else 1.0
        multiplier = mult_val  # Scalar or broadcasted array
    else:
        # Array/Vector case: 0 for P, 1 for S
        phase_mask = np.asarray(phase_input)
        multiplier = np.where(phase_mask == 1, s_wave_multiplier, 1.0)

    # 2. Vectorized pointwise target standard deviation (sigma_sec)
    sigma_sec = (
        (sigma_pick + sigma_path_max * (1.0 - np.exp(-t_r / T_c)))
        * multiplier
        * scale_extra
    )

    # 3. Sample noise based on distribution
    if distribution.lower() == "laplace":
        # For Laplace: std_dev = sqrt(2) * beta  =>  beta = sigma / sqrt(2)
        beta = sigma_sec / np.sqrt(2.0)
        noise_values = np.random.laplace(loc=0.0, scale=beta)
    elif distribution.lower() in ["gaussian", "normal"]:
        noise_values = np.random.normal(loc=0.0, scale=sigma_sec)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # 4. Pointwise excess noise flag (|noise| > N * sigma_sec)
    threshold = excess_threshold_sigma * sigma_sec
    is_excess = np.abs(noise_values) > threshold

    if return_sigma:
        return noise_values, is_excess, sigma_sec
    return noise_values, is_excess



## With correlated bias
# def generate_travel_time_noise(
#     t_r,
#     phase_input="P",  # String ("P"/"S") OR array of 0s (P) and 1s (S)
#     distribution="laplace",
#     sigma_pick=0.08,
#     sigma_path_max=1.20,
#     T_c=150.0,
#     scale_extra=1.0,
#     s_wave_multiplier=2.2,
#     excess_threshold_sigma=2.0,  # Tightened conservative threshold
#     # --- Systemic Model Bias Parameters ---
#     origin_shift_std=0.8,  # Systemic constant offset (seconds) per event
#     velocity_scale_std=0.03,  # Systemic velocity error (e.g., +/- 3% scale factor)
#     apply_systemic_bias=True,  # Set True during training to simulate bad velocity models
#     return_sigma=False,
# ):
#     """Generates uncorrelated Laplace travel-time noise AND optional systemic

#     velocity model bias (origin shifts and slowness scaling).
#     """
#     t_r = np.asarray(t_r)

#     # 1. Parse Phase Multiplier
#     if isinstance(phase_input, str):
#         multiplier = s_wave_multiplier if phase_input.upper() == "S" else 1.0
#     else:
#         phase_mask = np.asarray(phase_input)
#         multiplier = np.where(phase_mask == 1, s_wave_multiplier, 1.0)

#     # 2. Uncorrelated Per-Ray Standard Deviation (sigma_sec)
#     sigma_sec = (
#         (sigma_pick + sigma_path_max * (1.0 - np.exp(-t_r / T_c)))
#         * multiplier
#         * scale_extra
#     )

#     # 3. Uncorrelated Random Noise (Pick-level)
#     if distribution.lower() == "laplace":
#         beta = sigma_sec / np.sqrt(2.0)
#         uncorrelated_noise = np.random.laplace(loc=0.0, scale=beta)
#     elif distribution.lower() in ["gaussian", "normal"]:
#         uncorrelated_noise = np.random.normal(loc=0.0, scale=sigma_sec)
#     else:
#         raise ValueError(f"Unsupported distribution: {distribution}")

#     # 4. Inject Systemic / Correlated Model Bias (Event-level)
#     if apply_systemic_bias:
#         # A. Baseline Shift: Shifts ALL picks for this event by a common offset
#         systemic_shift = np.random.normal(0.0, origin_shift_std)

#         # B. Slowness Scale Perturbation: Simulates reference velocity being e.g. 3% too fast/slow
#         systemic_scale = np.random.normal(0.0, velocity_scale_std)
#         slowness_error = t_r * systemic_scale

#         correlated_bias = systemic_shift + slowness_error
#     else:
#         correlated_bias = 0.0

#     # Total synthetic noise
#     total_noise = uncorrelated_noise + correlated_bias

#     # 5. Outlier Detection: Evaluated ONLY against the uncorrelated component!
#     # (Because systemic shifts do NOT invalidate phase coherence/associations)
#     threshold = excess_threshold_sigma * sigma_sec
#     is_excess = np.abs(uncorrelated_noise) > threshold

#     if return_sigma:
#         return total_noise, is_excess, sigma_sec
#     return total_noise, is_excess


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

def interp_1D_velocity_model_to_3D_travel_times(X, locs, Xmin, X0, Dx, Mn, Tp, Ts, N, ftrns1, ftrns2, method = 'pairs', device = 'cuda'):

	i1 = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]])
	x10, x20, x30 = Xmin
	Xv = X - np.array([x10,x20,x30])[None,:]
	depth_grid = np.copy(locs[:,2]).reshape(1,-1)
	mask = np.array([1.0,1.0,0.0]).reshape(1,-1)

	print('Check way X0 is used')

	if method == 'pairs':
	
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

	elif method == 'direct': ## Not finished being implemented yet

		def evaluate_func(y, x):
	
			y, x = y.cpu().detach().numpy(), x.cpu().detach().numpy()
	
			ind_depth = np.tile(np.argmin(np.abs(y[:,2].reshape(-1,1) - depth_grid), axis = 1), x.shape[0])
			rel_pos = x - y*mask
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
def assemble_time_pointers_for_stations(trv_out, max_t = None, min_t = None, dt = 1.0, k = 10, win = 10.0):

	n_temp, n_sta = trv_out.shape[0:2]
	if max_t is None: max_t = trv_out.max()
	if min_t is None: min_t = 0.0
	dt_partition = np.arange(-win + min_t, win + max_t + dt, dt) # + tbuffer

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

def assemble_time_pointers_for_stations_multiple_grids(trv_out, max_t, min_t = None, dt = 1.0, k = 10, win = 10.0):

	n_temp, n_sta = trv_out.shape[0:2]
	if min_t is None: min_t = 0.0
	dt_partition = np.arange(-win + min_t, win + max_t + dt, dt)

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
	
def compute_travel_times(trv, locs, x_grids, n_max_chunks = int(50e3), device = 'cpu'):

	x_grids_trv = []
	# locs_cuda = torch.Tensor(locs).to(device)
	for i in range(len(x_grids)):
		
		# n_sta, n_temp = len(locs), len(x_grids[i])
		# n_chunks = int(np.maximum(1, int((n_sta*n_temp)/n_max_chunks)))
		# n_int = int(len(locs)/n_chunks)
		# inds = [np.arange(n_int) + n_int*j for j in range(n_chunks)]

		n_sta, n_temp = len(locs), len(x_grids[i])
		n_chunks = int(np.maximum(1, int((n_sta*n_temp)/n_max_chunks)))
		n_int = max(int(len(locs)/n_chunks), 1)
		n_chunks = np.minimum(n_chunks, len(locs))
		inds = [np.arange(n_int) + n_int*j for j in range(n_chunks)]
		
		
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

	# x_grids_trv = []
	# for i in range(len(x_grids)):
		
	# 	if locs.shape[0]*x_grids[i].shape[0] > 150e3:
	# 		trv_out_l = []
	# 		for j in range(locs.shape[0]):
	# 			trv_out = trv(torch.Tensor(locs[j,:].reshape(1,-1)).to(device), torch.Tensor(x_grids[i]).to(device))
	# 			trv_out_l.append(trv_out.cpu().detach().numpy())
	# 		trv_out = torch.Tensor(np.concatenate(trv_out_l, axis = 1)).to(device)
	# 	else:
	# 		trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
		
	# 	# trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
	# 	x_grids_trv.append(trv_out.cpu().detach().numpy())


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

def load_travel_time_neural_network(path_to_file, ftrns1, ftrns2, n_ver_load, phase = 'p_s', device = 'cuda', method = 'relative pairs', corrs = None, locs_corr = None, return_model = False, use_physics_informed = False):

	if use_physics_informed == False:
	
		from module import TravelTimes
		seperator = '\\' if '\\' in path_to_file else '/'
		
		z = np.load(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'travel_time_neural_network_%s_losses_ver_%d.npz'%(phase, n_ver_load))
		n_phases = len(z['v_mean'])
		scale_val = float(z['scale_val'])
		trav_val = float(z['trav_val'])
		z.close()
		
		m = TravelTimes(ftrns1, ftrns2, scale_val = scale_val, trav_val = trav_val, n_phases = n_phases, device = device).to(device)
		m.load_state_dict(torch.load(path_to_file + '/1D_Velocity_Models_Regional/travel_time_neural_network_%s_ver_%d.h5'%(phase, n_ver_load), map_location = torch.device(device)))
		
		if return_model == False:
			m.eval()
	
		if method == 'relative pairs':
	
			trv = lambda sta_pos, src_pos: m.forward_relative(sta_pos, src_pos, method = 'pairs')
	
		if method == 'direct':
	
			trv = lambda sta_pos, src_pos: m.forward_relative(sta_pos, src_pos, method = 'direct')

		if return_model == True:

			return m

		else:
		
			return trv

	else:

		from module import VModel, TravelTimesPN
		seperator = '\\' if '\\' in path_to_file else '/'
		
		z = np.load(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'travel_time_neural_network_physics_informed_%s_losses_ver_%d.npz'%(phase, n_ver_load))
		n_phases = len(z['v_mean'])
		v_mean, scale_params = z['v_mean'], z['scale_params']
		# scale_val = float(z['scale_val'])
		# trav_val = float(z['trav_val'])
		z.close()
	
		max_dist, max_time, vp_max, vs_min, scale_norm_factor, conversion_factor = scale_params
		
		norm_pos = lambda x: x/max_dist
		norm_time = lambda t: t/max_time
		norm_vel = lambda v: v/vp_max
	
		inorm_pos = lambda x: x*max_dist
		inorm_time = lambda t: t*max_time
		inorm_vel = lambda v: v*vp_max	
		
		m = TravelTimesPN(ftrns1, ftrns2, n_phases = n_phases, v_mean = v_mean, norm_pos = norm_pos, inorm_pos = inorm_pos, inorm_time = inorm_time, norm_vel = norm_vel, conversion_factor = conversion_factor, corrs = corrs, locs_corr = locs_corr, device = device).to(device)
		m.load_state_dict(torch.load(path_to_file + '/1D_Velocity_Models_Regional/travel_time_neural_network_physics_informed_%s_ver_%d.h5'%(phase, n_ver_load), map_location = torch.device(device)))
		if return_model == False:
			m.eval()
	
		if method == 'relative pairs':
	
			trv = lambda sta_pos, src_pos: m(sta_pos, src_pos, method = 'pairs')
	
		if method == 'direct':
	
			trv = lambda sta_pos, src_pos: m(sta_pos, src_pos, method = 'direct')

		if return_model == True:

			return m

		else:
		
			return trv		

# def load_travel_time_neural_network_physics_informed(path_to_file, ftrns1, ftrns2, n_ver_load, phase = 'p_s', device = 'cuda', method = 'relative pairs'):

# 	from module import TravelTimesPN
# 	seperator = '\\' if '\\' in path_to_file else '/'
	
# 	z = np.load(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'travel_time_neural_network_physics_informed_%s_losses_ver_%d.npz'%(phase, n_ver_load))
# 	n_phases = z['out1'].shape[1]
# 	v_mean, scale_params = z['v_mean'], z['scale_params']
# 	# scale_val = float(z['scale_val'])
# 	# trav_val = float(z['trav_val'])
# 	z.close()

# 	max_dist, max_time, vp_max, vs_min, scale_norm_factor, conversion_factor = scale_params
	
# 	norm_pos = lambda x: x/max_dist
# 	norm_time = lambda t: t/max_time
# 	norm_vel = lambda v: v/vp_max

# 	inorm_pos = lambda x: x*max_dist
# 	inorm_time = lambda t: t*max_time
# 	inorm_vel = lambda v: v*vp_max	
	
# 	m = TravelTimesPN(ftrns1, ftrns2, n_phases = n_phases, v_mean = v_mean, norm_pos = norm_pos, inorm_pos = inorm_pos, inorm_time = inorm_time, norm_vel = norm_vel, conversion_factor = conversion_factor, device = device).to(device)
# 	m.load_state_dict(torch.load(path_to_file + '/1D_Velocity_Models_Regional/travel_time_neural_network_physics_informed_%s_ver_%d.h5'%(phase, n_ver_load), map_location = torch.device(device)))
# 	m.eval()

# 	if method == 'relative pairs':

# 		trv = lambda sta_pos, src_pos: m(sta_pos, src_pos, method = 'pairs')

# 	if method == 'direct':

# 		trv = lambda sta_pos, src_pos: m(sta_pos, src_pos, method = 'direct')
	
# 	return trv

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
	return TrvTimesCorrection(trv, x_grid_corr, locs_use, coefs, ftrns1_diff, coefs_ker = coefs_ker, interp_type = interp_type, k = k_spc_interp, trv_direct = trv_direct, sig = sig_ker)

def load_templates_region(trv, locs, x_grids, ftrns1, training_params, graph_params, pred_params, max_t = None, min_t = None, time_shifts = None, dt_embed = 1.0, t_win = 10.0, device = 'cpu'):

	k_sta_edges, k_spc_edges, k_time_edges = graph_params

	# t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	x_grids_trv = []
	x_grids_trv_pointers_p = []
	x_grids_trv_pointers_s = []
	x_grids_trv_refs = []
	x_grids_edges = []

	## Replacing the loop with this seperate function
	x_grids_trv = compute_travel_times(trv, locs, x_grids, device = device)
	if time_shifts is not None:
		assert(len(x_grids_trv) == len(time_shifts))
		for i in range(len(x_grids_trv)):
			x_grids_trv[i] = x_grids_trv[i] + time_shifts[i].reshape(-1,1,1)
	
	# for i in range(len(x_grids)):

	# 	# trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))

	# 	if locs.shape[0]*x_grids[i].shape[0] > 150e3:
	# 		trv_out_l = []
	# 		for j in range(locs.shape[0]):
	# 			trv_out = trv(torch.Tensor(locs[j,:].reshape(1,-1)).to(device), torch.Tensor(x_grids[i]).to(device))
	# 			trv_out_l.append(trv_out.cpu().detach().numpy())
	# 		trv_out = torch.Tensor(np.concatenate(trv_out_l, axis = 1)).to(device)
	# 	else:
	# 		trv_out = trv(torch.Tensor(locs).to(device), torch.Tensor(x_grids[i]).to(device))
		
	# 	x_grids_trv.append(trv_out.cpu().detach().numpy())

		## Removing creation of edges here
		# edge_index = knn(torch.Tensor(ftrns1(x_grids[i])).to(device), torch.Tensor(ftrns1(x_grids[i])).to(device), k = k_spc_edges).flip(0).contiguous()
		# edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
		# x_grids_edges.append(edge_index)

	if max_t is None: max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # *1.1 # + 10.0
	if min_t is None: min_t = 0.0
	# if max_t is None: max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # *1.1 # + 10.0

	for i in range(len(x_grids)):

		A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations_multiple_grids(x_grids_trv[i], max_t, min_t = min_t, dt = dt_embed, win = t_win)
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
	P = P.astype('float')
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




















