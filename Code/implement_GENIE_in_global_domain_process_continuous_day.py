
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import glob
from obspy.geodetics.base import calc_vincenty_inverse

## Make this file self-contained.
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import h5py
import os
import obspy
from obspy.core import UTCDateTime
from obspy.clients.fdsn.client import Client
from sklearn.metrics import pairwise_distances as pd
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
import time
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from sklearn.neighbors import KernelDensity
from torch_geometric.data import Data
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
from torch_geometric.utils import k_hop_subgraph
from scipy.stats import gamma, beta
import itertools
import sys
import pdb

from scipy.signal import find_peaks
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from obspy.geodetics.base import calc_vincenty_inverse
import matplotlib.gridspec as gridspec
import networkx as nx
import cvxpy as cp
import glob

## Establish local vs. remote
userhome = os.path.expanduser('~')
if '/home/users/imcbrear' in userhome:
	ext_type = 'server'
elif 'imcbrear' in userhome:
	ext_type = 'remote'
elif ('imcbr' in userhome) and ('C:\\' in userhome):
	ext_type = 'local'

device = torch.device('cuda')

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

	a = a.to(device)
	e = e.to(device)

	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	p = p.detach().clone().float()
	pi = torch.Tensor([np.pi]).to(device)
	p[:,0:2] = p[:,0:2]*torch.Tensor([pi/180.0, pi/180.0]).to(device).view(1,-1)
	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0])**2)
    # results:
	x = (N + p[:,2])*torch.cos(p[:,0])*torch.cos(p[:,1])
	y = (N + p[:,2])*torch.cos(p[:,0])*torch.sin(p[:,1])
	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0])

	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

def ecef2lla_diff(x, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2]), device = 'cpu'):

	a = a.to(device)
	e = e.to(device)

	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
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

def ecef2latlon(X, a = 6378137.0, e = 8.1819190842622e-2):
    """
    Convert Earth-Centered-Earth-Fixed (ECEF) cartesian coordinates
    to lat,lon,depth.
    Args:
        x: A numpy array (or scalar value) of x coordinates (meters).
        y: A numpy array (or scalar value) of y coordinates (meters).
        z: A numpy array (or scalar value) of y coordinates (meters),
            positive UP.
    Return:
        Tuple of lat,lon,depth numpy arrays, where lat,lon are in dd and depth
        is in km and positive DOWN.
    """
    x, y, z = X[:,0], X[:,1], X[:,2]

    DEGREES_TO_RADIANS = np.pi / 180.0
    RADIANS_TO_DEGREES = 180.0 / np.pi

    inputIsScalar = False
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
        inputIsScalar = True
    # WGS84 ellipsoid constants:
    # a = 6378137
    # e = 8.1819190842622e-2

    # calculations:
    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th)**3),
        (p - e**2 * a * np.cos(th)**3))
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    # return lon in range [0,2*pi)
    lon = np.mod(lon, 2 * np.pi)

    # correct for numerical instability in altitude near exact poles:
    # (after this correction, error is about 2 millimeters, which is about
    # the same as the numerical precision of the overall function)
    k = (np.abs(x) < 1) & (np.abs(y) < 1)
    alt[k] = np.abs(z[k]) - b

    # convert lat,lon to dd, and alt to depth positive DOWN in km
    lat = lat * RADIANS_TO_DEGREES
    lon = lon * RADIANS_TO_DEGREES
    lon[lon > 180] = lon[lon > 180] - 360.0
    dep = -alt / 1000.0
    # if input values were scalar, give that back to them
    if inputIsScalar:
        lat = lat[0]
        lon = lon[0]
        dep = dep[0]
        alt = alt[0]

    return np.concatenate((lat[:,None], lon[:,None], alt[:,None]), axis = 1)

def extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs, ind_use, x_grid, ftrns1, ftrns2, max_deg_offset = 5.0, k_nearest_pairs = 30, k_sta_edges = 10, k_spc_edges = 15, verbose = False, scale_deg = 110e3):

	## Connect all source-reciever pairs to their k_nearest_pairs, and those connections within max_deg_offset.
	## By using the K-nn neighbors as well as epsilon-pairs, this ensures all source nodes are at least
	## linked to some stations.

	## Note: can also make the src-src and sta-sta graphs as a combination of k-nn and epsilon-distance graphs

	if verbose == True:
		st = time.time()

	n_sta = locs.shape[0]
	n_spc = x_grid.shape[0]
	n_sta_slice = len(ind_use)

	assert(np.max(abs(ind_use - np.arange(locs.shape[0]))) == 0) ## For now, assume ind_use is all of locs

	k_sta_edges = np.minimum(k_sta_edges, len(ind_use) - 2)

	perm_vec = -1*np.ones(locs.shape[0])
	perm_vec[ind_use] = np.arange(len(ind_use))

	## This will put more edges on the longitude or latitude axis, due to the shape of the Earth?
	A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[ind_use])/1000.0), torch.Tensor(ftrns1(locs[ind_use])/1000.0), k = k_sta_edges + 1).flip(0).long().contiguous())[0]
	A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grid)/1000.0), torch.Tensor(ftrns1(x_grid)/1000.0), k = k_spc_edges + 1).flip(0).long().contiguous())[0]

	## Either only keep edges less than certain distance, in which case, some sources have no source-station edges.
	## Or can add several nearest stations for each source.
	## Or, can change distance threshold for all source nodes so that it includes some number of stations.
	## Can also use kmeans to only sample source nodes within some convex hull of nearby stations,
	## or nearby known seismicity.
	## Can also sample spatial nodes proportional to seismicity density.

	## Make "incoming" edges for all sources based on epsilon-distance graphs, since these are the nodes that need nearest stations
	# dist_pairwise_src_locs = np.expand_dims(x_grid[:,0:2], axis = 1) - np.expand_dims(locs[:,0:2], axis = 0)
	dist_pairwise_src_locs = np.expand_dims(ftrns1(x_grid), axis = 1) - np.expand_dims(ftrns1(locs), axis = 0)
	# dist_pairwise_src_locs[:,:,1] = np.mod(dist_pairwise_src_locs[:,:,1], 360.0) ## Modulus on longitude distance
	pairwise_src_locs_distances = np.linalg.norm(dist_pairwise_src_locs, axis = 2)
	ind_src_keep, ind_sta_keep = np.where(pairwise_src_locs_distances < scale_deg*max_deg_offset)
	A_src_in_sta_epsilon = torch.cat((torch.Tensor(ind_sta_keep.reshape(1,-1)), torch.Tensor(ind_src_keep.reshape(1,-1))), dim = 0).long()

	## Make "incoming" edges for all sources based on knn, since these are the nodes that need nearest stations
	A_src_in_sta_knn = knn(torch.Tensor(ftrns1(locs[ind_use])/1000.0), torch.Tensor(ftrns1(x_grid)/1000.0), k = k_nearest_pairs).flip(0).long().contiguous()

	## Also add a number of random edges between each source node and any distant station nodes, to improve coverage
	## Can potentially choose these so that 2-hop and 3-hop versions of graphs are highly connected

	## Combine edges to a single source-station pairwise set of edges
	A_src_in_sta = torch.cat((A_src_in_sta_epsilon, A_src_in_sta_knn), dim = 1)
	A_src_in_sta = np.unique(A_src_in_sta.cpu().detach().numpy(), axis = 1)
	isort = np.argsort(A_src_in_sta[1,:]) ## Don't need this sort, if using the one below
	A_src_in_sta = A_src_in_sta[:,isort]
	isort = np.lexsort((A_src_in_sta[0], A_src_in_sta[1]))
	A_src_in_sta = torch.Tensor(A_src_in_sta[:,isort]).long()

	## Create "subgraph" Cartesian product graph edges
	## E.g., the "subgraph" Cartesian product is only "nodes" of pairs of sources-recievers in A_src_in_sta, rather than all pairs locs*x_grid.

	degree_of_src_nodes = degree(A_src_in_sta[1])
	cum_count_degree_of_src_nodes = np.concatenate((np.array([0]), np.cumsum(degree_of_src_nodes.cpu().detach().numpy())), axis = 0).astype('int')

	sta_ind_lists = []
	for i in range(x_grid.shape[0]):
		ind_list = -1*np.ones(locs.shape[0])
		ind_list[A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]].cpu().detach().numpy()] = np.arange(degree_of_src_nodes[i])
		sta_ind_lists.append(ind_list)
	sta_ind_lists = np.hstack(sta_ind_lists).astype('int')


	## A_src_in_prod : For each entry in A_src, need to determine which subset of A_src_in_sta is incoming.
	## E.g., which set of incoming stations for each pair of (source-station); so, all point source_1 == source_2
	## for source_1 in Source graph and source_2 in Cartesian product graph
	tree_srcs_in_prod = cKDTree(A_src_in_sta[1][:,None])
	lp_src_in_prod = tree_srcs_in_prod.query_ball_point(np.arange(x_grid.shape[0])[:,None], r = 0)
	A_src_in_prod = torch.Tensor(np.hstack([np.concatenate((np.array(lp_src_in_prod[j]).reshape(1,-1), j*np.ones(len(lp_src_in_prod[j])).reshape(1,-1)), axis = 0) for j in range(x_grid.shape[0])])).long()
	# spatial_vals = torch.Tensor(x_grid[A_src_in_sta[1]] - locs_use[A_src_in_sta[0]] ## This approach assumes all station indices are ordered
	# spatial_vals = torch.Tensor((ftrns1(x_grid[A_src_in_prod[1]]) - ftrns1(locs_use[A_src_in_sta[0][A_src_in_prod[0]]]))/110e3*scale_src_in_prod)
	spatial_vals = torch.Tensor((ftrns1(x_grid[A_src_in_prod[1]]) - ftrns1(locs_use[A_src_in_sta[0][A_src_in_prod[0]]]))/(5.0*1000.0*1000.0))
	A_src_in_prod = Data(x = spatial_vals, edge_index = A_src_in_prod)

	## A_prod_sta_sta : connect any two nodes in A_src_in_sta (where these edges are distinct nodes in subgraph Cartesian product graph) 
	## if the stations are linked in A_sta_sta, and the sources are equal. 

	## Two approaches seem possible, but both have drawbacks: 
	## (i). Create the dense, full Cartesian product, with edges, then take subgraph based on retained nodes in the subgraph Cartesian product. [high memory cost]
	## (ii). Loop through each sub-segment of the subgraph Cartesian product (for fixed sources), and check which of the station pairs are linked.

	A_prod_sta_sta = []
	A_prod_src_src = []

	tree_src_in_sta = cKDTree(A_src_in_sta[0].reshape(-1,1).cpu().detach().numpy())
	lp_fixed_stas = tree_src_in_sta.query_ball_point(np.arange(locs.shape[0]).reshape(-1,1), r = 0)

	for i in range(x_grid.shape[0]):
		# slice_edges = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, relabel_nodes = False)[0]
		slice_edges = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, relabel_nodes = True)[0]
		A_prod_sta_sta.append(slice_edges + cum_count_degree_of_src_nodes[i])

	for i in range(locs.shape[0]):
	
		slice_edges = subgraph(A_src_in_sta[1,np.array(lp_fixed_stas[i])], A_src_src, relabel_nodes = False)[0]

		## This can happen when a station is only linked to one source
		if slice_edges.shape[1] == 0:
			continue

		shift_ind = sta_ind_lists[slice_edges*n_sta + i]
		assert(shift_ind.min() >= 0)
		## For each source, need to find where that station index is in the "order" of the subgraph Cartesian product
		A_prod_src_src.append(torch.Tensor(cum_count_degree_of_src_nodes[slice_edges] + shift_ind))

	## Make cartesian product graphs
	A_prod_sta_sta = torch.hstack(A_prod_sta_sta).long()
	A_prod_src_src = torch.hstack(A_prod_src_src).long()
	isort = np.lexsort((A_prod_src_src[0], A_prod_src_src[1])) # Likely not actually necessary
	A_prod_src_src = A_prod_src_src[:,isort]

	return [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

def kmeans_packing(scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 1000, n_sim = 3, lr = 0.01):

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

def kmeans_packing_cartesian(scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 1000, n_sim = 3, lr = 0.01):

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x
				v = ftrns1(v)

			tree = cKDTree(v)
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x
			x = ftrns1(x)
			q, ip = tree.query(x)

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

def kmeans_packing_weight_vector(weight_vector, scale_x, offset_x, ndim, n_clusters, ftrns1, n_batch = 3000, n_steps = 1000, n_sim = 3, lr = 0.01):

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

def kmeans_packing_global(depth_range, ndim, n_clusters, ftrns1, ftrns2, ftrns1_sphere, ftrns2_sphere, wgs84_radius = 6378137.0, n_batch = 3000, n_steps = 1000, n_sim = 3, lr = 0.01):

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				## Make random points on sphere
				v = np.random.randn(n_clusters, ndim)
				v = wgs84_radius*v/np.linalg.norm(v, axis = 1, keepdims = True)
				## Project to lat-lon values
				v = ftrns2_sphere(v)
				## Assign random depths
				v[:,2] = np.random.rand(v.shape[0])*(depth_range[1] - depth_range[0]) + depth_range[0]


			tree = cKDTree(ftrns1(v))
			## Make random points on sphere
			x = np.random.randn(n_batch, ndim)
			x = wgs84_radius*x/np.linalg.norm(x, axis = 1, keepdims = True)
			## Project to lat-lon values
			x = ftrns2_sphere(x)
			## Assign random depths
			x[:,2] = np.random.rand(x.shape[0])*(depth_range[1] - depth_range[0]) + depth_range[0]

			# *scale_x + offset_x
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

		# # Evaluate loss (5 times batch size)
		# x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		# q, ip = tree.query(x)
		# Losses.append(q.mean())
		# V_results.append(np.copy(v))

	# Losses = np.array(Losses)
	# ibest = np.argmin(Losses)

	return v

def kmeans_packing_global_cartesian(depth_range, ndim, n_clusters, ftrns1, ftrns2, ftrns1_sphere, ftrns2_sphere, wgs84_radius = 6378137.0, n_batch = 3000, n_steps = 1000, n_sim = 3, lr = 0.01):

	V_results = []
	Losses = []
	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				## Make random points on sphere
				v = np.random.randn(n_clusters, ndim)
				v = wgs84_radius*v/np.linalg.norm(v, axis = 1, keepdims = True)
				## Project to lat-lon values
				v = ftrns2_sphere(v)
				## Assign random depths
				v[:,2] = np.random.rand(v.shape[0])*(depth_range[1] - depth_range[0]) + depth_range[0]

				v = ftrns1(v)


			tree = cKDTree(v)
			## Make random points on sphere
			x = np.random.randn(n_batch, ndim)
			x = wgs84_radius*x/np.linalg.norm(x, axis = 1, keepdims = True)
			## Project to lat-lon values
			x = ftrns2_sphere(x)
			## Assign random depths
			x[:,2] = np.random.rand(x.shape[0])*(depth_range[1] - depth_range[0]) + depth_range[0]

			x = ftrns1(x)

			# *scale_x + offset_x
			q, ip = tree.query(x)

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

		# # Evaluate loss (5 times batch size)
		# x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		# q, ip = tree.query(x)
		# Losses.append(q.mean())
		# V_results.append(np.copy(v))

	# Losses = np.array(Losses)
	# ibest = np.argmin(Losses)

	return ftrns2(v)

def spherical_packing_nodes(n, ftrns1_sphere_unit, ftrns2_sphere_unit):

	## Based on The Fibonacci Lattice
	## https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

	# n = 30000
	i = np.arange(0, n).astype('float') + 0.5
	phi = np.arccos(1 - 2*i/n)
	goldenRatio = (1 + 5**0.5)/2
	theta = 2*np.pi * i / goldenRatio
	x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);

	return ftrns2_sphere_unit(np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis = 1))

	# fig = plt.figure()
	# ax = fig.add_subplot(projection = '3d')
	# ax.scatter(x, y, z, s = 1, alpha = 0.1)



def compute_pairwise_distances(locs, x_grid, A_prod_sta_sta, A_prod_src_src):

	n_sta = len(locs)
	n_spc = len(x_grid)

	ind_sta_A_prod_sta_sta = A_prod_sta_sta - n_sta*torch.floor(A_prod_sta_sta/n_sta).long()
	ind_src_A_prod_sta_sta = torch.floor(A_prod_sta_sta/n_sta).long()

	ind_sta_A_prod_src_src = A_prod_src_src - n_sta*torch.floor(A_prod_src_src/n_sta).long()
	ind_src_A_prod_src_src = torch.floor(A_prod_src_src/n_sta).long()

	mean_sta_loc_A_prod_sta_sta = locs[ind_sta_A_prod_sta_sta].mean(0)
	mean_src_loc_A_prod_sta_sta = x_grid[ind_src_A_prod_sta_sta].mean(0)
	dist_A_prod_sta_sta = np.linalg.norm(mean_sta_loc_A_prod_sta_sta[:,0:2] - mean_src_loc_A_prod_sta_sta[:,0:2], axis = 1)

	mean_sta_loc_A_prod_src_src = locs[ind_sta_A_prod_src_src].mean(0)
	mean_src_loc_A_prod_src_src = x_grid[ind_src_A_prod_src_src].mean(0)
	dist_A_prod_src_src = np.linalg.norm(mean_sta_loc_A_prod_src_src[:,0:2] - mean_src_loc_A_prod_src_src[:,0:2], axis = 1)

	return dist_A_prod_sta_sta, dist_A_prod_src_src


def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, ftrns1, ftrns2, n_batch = 75, max_rate_events = 5000/8, max_miss_events = 3500/8, max_false_events = 2000/8, T = 3600.0*3.0, dt = 30, tscale = 3600.0, n_sta_range = [0.35, 1.0], use_sources = False, use_full_network = False, fixed_subnetworks = None, use_preferential_sampling = False, use_shallow_sources = False, plot_on = False, verbose = False):

	if verbose == True:
		st = time.time()


	scale_x = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)
	n_sta = locs.shape[0]
	locs_tensor = torch.Tensor(locs).cuda()

	tsteps = np.arange(0, T + dt, dt)
	tvec = np.arange(-tscale*4, tscale*4 + dt, dt)
	tvec_kernel = np.exp(-(tvec**2)/(2.0*(tscale**2)))
	# p_rate_events = np.convolve(np.random.randn(len(tsteps)), tvec_kernel, 'same')
	# scipy.signal.convolve(in1, in2, mode='full'

	## Can augment global parameters at a different rate.
	p_rate_events = fftconvolve(np.random.randn(2*locs.shape[0] + 3, len(tsteps)), tvec_kernel.reshape(1,-1).repeat(2*locs.shape[0] + 3,0), 'same', axes = 1)
	c_cor = (p_rate_events@p_rate_events.T) ## Not slow!
	global_event_rate, global_miss_rate, global_false_rate = p_rate_events[0:3,:]

	# Process global event rate, to physical units.
	global_event_rate = (global_event_rate - global_event_rate.min())/(global_event_rate.max() - global_event_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_rate_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_rate_events - min_add) + 0.5*max_rate_events
	global_event_rate = global_event_rate*scale + min_add

	global_miss_rate = (global_miss_rate - global_miss_rate.min())/(global_miss_rate.max() - global_miss_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
	global_miss_rate = global_miss_rate*scale + min_add

	global_false_rate = (global_false_rate - global_false_rate.min())/(global_false_rate.max() - global_false_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_false_events - min_add) + 0.5*max_false_events
	global_false_rate = global_false_rate*scale + min_add

	station_miss_rate = p_rate_events[3 + np.arange(n_sta),:]
	station_miss_rate = (station_miss_rate - station_miss_rate.min(1, keepdims = True))/(station_miss_rate.max(1, keepdims = True) - station_miss_rate.min(1, keepdims = True)) # [0,1] scale
	min_add = np.random.rand(n_sta,1)*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand(n_sta,1)*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
	station_miss_rate = station_miss_rate*scale + min_add

	station_false_rate = p_rate_events[3 + n_sta + np.arange(n_sta),:]
	station_false_rate = (station_false_rate - station_false_rate.min(1, keepdims = True))/(station_false_rate.max(1, keepdims = True) - station_false_rate.min(1, keepdims = True))
	min_add = np.random.rand(n_sta,1)*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand(n_sta,1)*(0.5*max_false_events - min_add) + 0.5*max_false_events
	station_false_rate = station_false_rate*scale + min_add


	## Sample events.
	vals = np.random.poisson(dt*global_event_rate/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average
	src_times = np.sort(np.hstack([np.random.rand(vals[j])*dt + tsteps[j] for j in range(len(vals))]))
	n_src = len(src_times)
	src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
	src_magnitude = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)


	if use_shallow_sources == True:
		sample_random_depths = gamma(1.75, 0.0).rvs(n_src)
		sample_random_grab = np.where(sample_random_depths > 5)[0] # Clip the long tails, and place in uniform, [0,5].
		sample_random_depths[sample_random_grab] = 5.0*np.random.rand(len(sample_random_grab))
		sample_random_depths = sample_random_depths/sample_random_depths.max() # Scale to range
		sample_random_depths = -sample_random_depths*(scale_x[0,2] - 2e3) + (offset_x[0,2] + scale_x[0,2] - 2e3) # Project along axis, going negative direction. Removing 2e3 on edges.
		src_positions[:,2] = sample_random_depths


	# m1 = [0.5761163, -0.5]
	m1 = [0.5761163, -0.21916288]
	m2 = 1.15
	dist_range = [15e3, 1000e3]
	spc_random = 30e3
	sig_t = 0.03 # 3 percent of travel time.
	# Just use, random, per-event distance threshold, combined with random miss rate, to determine extent of moveouts.

	amp_thresh = 1.0
	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

	use_uniform_distance_threshold = False
	## This previously sampled a skewed distribution by default, not it samples a uniform
	## distribution of the maximum source-reciever distances allowed for each event.
	if use_uniform_distance_threshold == True:
		dist_thresh = np.random.rand(n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
	else:
		## Use beta distribution to generate more samples with smaller moveouts
		# dist_thresh = -1.0*np.log(np.sqrt(np.random.rand(n_src))) ## Sort of strange dist threshold set!
		# dist_thresh = (dist_thresh*dist_range[1]/10.0 + dist_range[0]).reshape(-1,1)
		dist_thresh = beta(2,5).rvs(size = n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]

	# create different distance dependent thresholds.
	dist_thresh_p = dist_thresh + 50e3*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
	dist_thresh_s = dist_thresh + 50e3*np.random.laplace(size = dist_thresh.shape[0])[:,None]

	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

	arrivals_theoretical = trv(torch.Tensor(locs).cuda(), torch.Tensor(src_positions[:,0:3]).cuda()).cpu().detach().numpy()
	arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
	arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
	src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

	# arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), log_amp_p[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1)), axis = 1)
	# arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), log_amp_s[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1)), axis = 1)
	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
	arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

	s_extra = 0.0
	t_inc = np.floor(arrivals[:,3]/dt).astype('int')
	p_miss_rate = 0.5*station_miss_rate[arrivals[:,1].astype('int'), t_inc] + 0.5*global_miss_rate[t_inc]
	idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < dt*p_miss_rate/T)[0]


	arrivals = np.delete(arrivals, idel, axis = 0)
	# 0.5 sec to here

	## Determine which sources are active, here.
	n_events = len(src_times)
	min_sta_arrival = 4


	source_tree_indices = cKDTree(arrivals[:,2].reshape(-1,1))
	lp = source_tree_indices.query_ball_point(np.arange(n_events).reshape(-1,1), r = 0)
	lp_backup = [lp[j] for j in range(len(lp))]
	n_unique_station_counts = np.array([len(np.unique(arrivals[lp[j],1])) for j in range(n_events)])
	active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
	non_active_sources = np.delete(np.arange(n_events), active_sources, axis = 0)
	src_positions_active = src_positions[active_sources]
	src_times_active = src_times[active_sources]
	src_magnitude_active = src_magnitude[active_sources] ## Not currently used
	# src_indices_active = src_indices[active_sources] # Not apparent this is needed
	## Additional 0.15 sec

	## Determine which sources (absolute indices) are inside the interior region.
	# inside_interior = np.where((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))[0]
	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))


	coda_rate = 0.035 # 5 percent arrival have code. Probably more than this? Increased from 0.035.
	coda_win = np.array([0, 25.0]) # coda occurs within 0 to 15 s after arrival (should be less?) # Increased to 25, from 20.0
	icoda = np.where(np.random.rand(arrivals.shape[0]) < coda_rate)[0]
	if len(icoda) > 0:
		false_coda_arrivals = np.random.rand(len(icoda))*(coda_win[1] - coda_win[0]) + coda_win[0] + arrivals[icoda,0] + arrivals[icoda,3]
		false_coda_arrivals = np.concatenate((false_coda_arrivals.reshape(-1,1), arrivals[icoda,1].reshape(-1,1), -1.0*np.ones((len(icoda),1)), np.zeros((len(icoda),1)), -1.0*np.ones((len(icoda),1))), axis = 1)
		arrivals = np.concatenate((arrivals, false_coda_arrivals), axis = 0)

	## Base false events
	station_false_rate_eval = 0.5*station_false_rate + 0.5*global_false_rate
	vals = np.random.poisson(dt*station_false_rate_eval/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average

	## Too slow!
	# false_times = np.hstack([np.hstack([np.random.rand(vals[k,j])*dt + tsteps[j] for j in range(vals.shape[1])]) for k in range(n_sta)])
	# false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])

	# How to speed up this part?
	i1, i2 = np.where(vals > 0)
	v_val, t_val = vals[i1,i2], tsteps[i2]
	false_times = np.repeat(t_val, v_val) + np.random.rand(vals.sum())*dt
	false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])
	n_false = len(false_times)
	false_arrivals = np.concatenate((false_times.reshape(-1,1), false_indices.reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
	arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)

	# print('make spikes!')
	## Make false spikes!
	n_spikes = np.random.randint(0, high = int(80*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
	if n_spikes > 0:
		n_spikes_extent = np.random.randint(1, high = n_sta, size = n_spikes) ## This many stations per spike
		time_spikes = np.random.rand(n_spikes)*T
		sta_ind_spikes = np.hstack([np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)])
		sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*0.15 for j in range(n_spikes)])
		false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
		# -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1)
		arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes
	# print('make spikes!')

	## Compute arrival times, exactly.
	# 3.5 % error of theoretical, laplace distribution
	iz = np.where(arrivals[:,4] >= 0)[0]
	arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]

	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
	phase_observed = np.copy(arrivals[:,-1]).astype('int')

	if len(iwhere_false) > 0: # For false picks, assign a random phase type
		phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))

	perturb_phases = True # For true picks, randomly flip a fraction of phases
	if (len(phase_observed) > 0)*(perturb_phases == True):
		n_switch = int(np.random.rand()*(0.2*len(iwhere_real))) # switch up to 20% phases
		iflip = np.random.choice(iwhere_real, size = n_switch, replace = False)
		phase_observed[iflip] = np.mod(phase_observed[iflip] + 1, 2)


	t_win = 10.0
	scale_vec = np.array([1,2*t_win]).reshape(1,-1)

	## Should try reducing kernel sizes
	src_t_kernel = 10.0 # Should decrease kernel soon!
	src_x_kernel = 50e3 # Add parallel secondary resolution prediction (condition on the low fre. resolution prediction)
	src_depth_kernel = 50e3 # was 5e3, increase depth kernel? ignore?
	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

	## Note, this used to be a Guassian, nearby times of know sources.
	if use_sources == False:
		time_samples = np.sort(np.random.rand(n_batch)*T) ## Uniform


	elif use_sources == True:
		time_samples = src_times_active[np.sort(np.random.choice(len(src_times_active), size = n_batch))]

	# focus_on_sources = True # Re-pick oorigin times close to sources
	l_src_times_active = len(src_times_active)
	if (use_preferential_sampling == True)*(len(src_times_active) > 1):
		for j in range(n_batch):
			if np.random.rand() > 0.5: # 30% of samples, re-focus time. # 0.7
				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()

	time_samples = np.sort(time_samples)

	max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))


	tree_src_times_all = cKDTree(src_times[:,np.newaxis])
	tree_src_times = cKDTree(src_times_active[:,np.newaxis])
	lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)
	lp_src_times = tree_src_times.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)


	st = time.time()
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 
	# print(time.time() - st)

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
	if len(lp_concat) == 0:
		lp_concat = np.array([0]) # So it doesnt fail?
	arrivals_select = arrivals[lp_concat]
	phase_observed_select = phase_observed[lp_concat]
	# tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec) ## Commenting out, as not used
	# print(time.time() - st)

	Trv_subset_p = []
	Trv_subset_s = []
	Station_indices = []
	Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	if (fixed_subnetworks is not None):
		fixed_subnetworks_flag = 1
	else:
		fixed_subnetworks_flag = 0		

	active_sources_per_slice_l = []
	src_positions_active_per_slice_l = []
	src_times_active_per_slice_l = []

	for i in range(n_batch):
		## Can also select sub-networks that are real, realizations.
		i0 = np.random.randint(0, high = len(x_grids))
		n_spc = x_grids[i0].shape[0]
		if use_full_network == True:
			n_sta_select = n_sta
			ind_sta_select = np.arange(n_sta)

		else:
			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
				n_sta_select = len(fixed_subnetworks[isub_network])
				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.
			
			else:
				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

		Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Station_indices.append(ind_sta_select) # record subsets used
		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
		Grid_indices.append(i0)
		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
		sc += len(Sample_indices[-1])


		active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

		# active_sources_per_slice = np.where(n_unique_station_counts_per_slice >= min_sta_arrival)[0] # subset of sources
		active_sources_per_slice_l.append(active_sources_per_slice)


	Trv_subset_p = np.vstack(Trv_subset_p)
	Trv_subset_s = np.vstack(Trv_subset_s)
	Batch_indices = np.hstack(Batch_indices)
	# print(time.time() - st)

	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
	n_arvs = arrivals_select.shape[0]
	# arrivals_select = np.concatenate((arrivals_0, arrivals_select, arrivals_1), axis = 0)

	# Rather slow!
	iargsort = np.argsort(arrivals_select[:,0])
	arrivals_select = arrivals_select[iargsort]
	phase_observed_select = phase_observed_select[iargsort]

	iwhere_p = np.where(phase_observed_select == 0)[0]
	iwhere_s = np.where(phase_observed_select == 1)[0]
	n_arvs_p = len(iwhere_p)
	n_arvs_s = len(iwhere_s)

	query_time_p = Trv_subset_p[:,0] + Batch_indices*offset_per_batch + Trv_subset_p[:,1]*offset_per_station
	query_time_s = Trv_subset_s[:,0] + Batch_indices*offset_per_batch + Trv_subset_s[:,1]*offset_per_station

	kernel_sig_t = 25.0 # Can speed up by only using matches.


	## No phase type information
	ip_p = np.searchsorted(arrivals_select[:,0], query_time_p)
	ip_s = np.searchsorted(arrivals_select[:,0], query_time_s)

	ip_p_pad = ip_p.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s_pad = ip_s.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
	ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

	rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)

	## With phase type information
	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

	rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)


	k_sta_edges = 10 # 10
	k_spc_edges = 15
	k_time_edges = 10 ## Make sure is same as in train_regional_GNN.py
	n_queries = 4500 ## 3000 ## Why only 3000?
	time_vec_slice = np.arange(k_time_edges)

	Inpts = []
	Masks = []
	Lbls = []
	Lbls_query = []
	X_fixed = []
	X_query = []
	Locs = []
	Trv_out = []

	A_sta_sta_l = []
	A_src_src_l = []
	A_prod_sta_sta_l = []
	A_prod_src_src_l = []
	A_src_in_prod_l = []
	A_edges_time_p_l = []
	A_edges_time_s_l = []
	A_edges_ref_l = []

	lp_times = []
	lp_stations = []
	lp_phases = []
	lp_meta = []
	lp_srcs = []
	lp_srcs_active = []

	print('Inputs need to be saved on the subset of Cartesian product graph, and can use the exctract adjacencies script to obtain the subset of entries on Cartesian product')

	# src_positions_active[lp_src_times[i]]

	pdb.set_trace()

	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grids[grid_select].shape[0]
		n_sta_slice = len(sta_select)


		## Based on sta_select, find the subset of station-source pairs needed for each of the `subgraphs' of Cartesian product graph
		[A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] = extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs[sta_select], np.arange(len(sta_select)), x_grids[grid_select], ftrns1, ftrns2, max_deg_offset = 5.0, k_nearest_pairs = 30)
		ind_sta_subset = sta_select[A_src_in_sta[0]]



		inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))


		## Select the subset of entries relevant for the input (it's only a subset of pairs of sources and stations...)
		## The if direct indexing not possible, might need to reshape into a full (all sources and stations, and features
		## tensor, and then selecting the subset of all pairs from this list).
		inpt_reshape = inpt.reshape(-1,4) ## This should be indices of the reshaped inpt needed for the subset of pairs of Cartesian product graph
		needed_indices = A_src_in_sta[1]*len(sta_select) + sta_select[A_src_in_sta[0]] # A_src_in_sta[0] # ind_sta_subset
		## inpt[5,20,2] == inpt_reshape[1425*5 + 20,2]

		## Does trv_out also need to be sub-selected
		trv_out = x_grids_trv[grid_select][:,sta_select,:] ## Subsetting, into sliced indices.
		Inpts.append(inpt[:,sta_select,:]) # sub-select, subset of stations.
		Masks.append(1.0*(inpt[:,sta_select,:] > thresh_mask))
		Trv_out.append(trv_out)
		Locs.append(locs[sta_select])
		X_fixed.append(x_grids[grid_select])

		## Assemble pick datasets
		perm_vec = -1*np.ones(n_sta)
		perm_vec[sta_select] = np.arange(len(sta_select))
		meta = arrivals[lp[i],:]
		phase_vals = phase_observed[lp[i]]
		times = meta[:,0]
		indices = perm_vec[meta[:,1].astype('int')]
		ineed = np.where(indices > -1)[0]
		times = times[ineed] ## Overwrite, now. Double check if this is ok.
		indices = indices[ineed]
		phase_vals = phase_vals[ineed]
		meta = meta[ineed]

		active_sources_per_slice = np.array(lp_src_times_all[i])[np.array(active_sources_per_slice_l[i])]
		ind_inside = np.where(inside_interior[active_sources_per_slice.astype('int')] > 0)[0]
		active_sources_per_slice = active_sources_per_slice[ind_inside]

		# Find pick specific, sources
		## Comment out
		ind_src_unique = np.unique(meta[meta[:,2] > -1.0,2]).astype('int') # ignore -1.0 entries.

		if len(ind_src_unique) > 0:
			ind_src_unique = np.sort(np.array(list(set(ind_src_unique).intersection(active_sources_per_slice)))).astype('int')

		src_subset = np.concatenate((src_positions[ind_src_unique], src_times[ind_src_unique].reshape(-1,1) - time_samples[i]), axis = 1)
		if len(ind_src_unique) > 0:
			perm_vec_meta = np.arange(ind_src_unique.max() + 1)
			perm_vec_meta[ind_src_unique] = np.arange(len(ind_src_unique))
			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)
			# ifind = np.where(meta[:,2] > -1.0)[0] ## Need to find picks with a source index inside the active_sources_per_slice
			ifind = np.where([meta[j,2] in ind_src_unique for j in range(meta.shape[0])])[0]
			meta[ifind,-1] = perm_vec_meta[meta[ifind,2].astype('int')] # save pointer to active source, for these picks (in new, local index, of subset of sources)
		else:
			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)

		# Do these really need to be on cuda?
		lex_sort = np.lexsort((times, indices)) ## Make sure lexsort doesn't cause any problems
		lp_times.append(times[lex_sort] - time_samples[i])
		lp_stations.append(indices[lex_sort])
		lp_phases.append(phase_vals[lex_sort])
		lp_meta.append(meta[lex_sort]) # final index of meta points into 
		lp_srcs.append(src_subset)

		# A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[sta_select])/1000.0).cuda(), torch.Tensor(ftrns1(locs[sta_select])/1000.0).cuda(), k = k_sta_edges + 1).flip(0).contiguous())[0]
		# A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).cuda(), torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).cuda(), k = k_spc_edges + 1).flip(0).contiguous())[0]
		# ## Cross-product graph is: source node x station node. Order as, for each source node, all station nodes.

		# # Cross-product graph, nodes connected by: same source node, connected stations
		# A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).cuda()).contiguous()
		# A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).cuda()).contiguous()	
		# # ind_spc = torch.floor(A_prod_src_src/torch.Tensor([n_sta]).cuda()) # Note: using scalar division causes issue at exact n_sta == n_sta indices.

		# # For each unique spatial point, sum in all edges.
		# # A_spc_in_prod = torch.cat(((torch.arange(n_sta).repeat(n_spc) + n_sta*torch.arange(n_spc).repeat_interleave(n_sta)).view(1,-1), torch.arange(n_spc)
		# A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1)), dim = 0).cuda().contiguous()

		## Sub-selecting from the time-arrays, is easy, since the time-arrays are indexed by station (triplet indexing; )
		len_dt = len(x_grids_trv_refs[grid_select])

		### Note: A_edges_time_p needs to be augmented: by removing stations, we need to re-label indices for subsequent nodes,
		### To the "correct" number of stations. Since, not n_sta shows up in definition of edges. "assemble_pointers.."
		A_edges_time_p = x_grids_trv_pointers_p[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
		A_edges_time_s = x_grids_trv_pointers_s[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
		## Need to convert these edges again. Convention is:
		## subtract i (station index absolute list), divide by n_sta, mutiply by N stations, plus ith station (in permutted indices)
		# shape is len_dt*k_time_edges*len(sta_select)
		# one_vec = np.repeat(sta_select*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere
		# A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
		# A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
		# # print('permute indices 1')
		# assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
		# assert(A_edges_time_s.max() < n_spc*n_sta_slice)

		A_sta_sta_l.append(A_sta_sta.cpu().detach().numpy())
		A_src_src_l.append(A_src_src.cpu().detach().numpy())
		A_prod_sta_sta_l.append(A_prod_sta_sta.cpu().detach().numpy())
		A_prod_src_src_l.append(A_prod_src_src.cpu().detach().numpy())
		A_src_in_prod_l.append(A_src_in_prod.cpu().detach().numpy())
		A_edges_time_p_l.append(A_edges_time_p)
		A_edges_time_s_l.append(A_edges_time_s)
		A_edges_ref_l.append(x_grids_trv_refs[grid_select])

		x_query = np.random.rand(n_queries, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.

		if len(lp_srcs[-1]) > 0:
			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]

		# t_query = (np.random.rand(n_queries)*10.0 - 5.0) + time_samples[i]
		t_slice = np.arange(-5.0, 5.0 + 1.0, 1.0) ## Window, over which to extract labels.

		if len(active_sources_per_slice) == 0:
			lbls_grid = np.zeros((x_grids[grid_select].shape[0],len(t_slice)))
			lbls_query = np.zeros((n_queries,len(t_slice)))
		else:
			active_sources_per_slice = active_sources_per_slice.astype('int')

			lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

		# Append, either if there is data or not
		X_query.append(x_query)
		Lbls.append(lbls_grid)
		Lbls_query.append(lbls_query)

	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
	data = [arrivals, srcs, active_sources]		

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

def generate_synthetic_data_simple(trv, locs, x_grids, x_grids_trv, x_query, A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, ftrns1, ftrns2, n_batch = 1, src_density = None, verbose = True):

	if verbose == True:
		st = time.time()

	## Can embed the three nearest arrivals in time, not just one, to handle 
	## large numbers of picks in a short window
	## and the large kernel embedding size

	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
	locs_cuda = torch.Tensor(locs).to(device)
	x_grid_cuda = torch.Tensor(x_grid).to(device)
	n_sta = locs.shape[0]

	## Set up time window
	max_t = 500.0
	t_window = 5.0*max_t
	min_sta_cnt = 5

	src_t_kernel = 15.0
	src_x_kernel = 200e3

	## Choose number of events between 0 and 10 within a max time window of 5*max_t
	## Up-weight the probability of choosing 0 events
	## max_t ~= 500 s, to be consistent with training data
	ratio_noise = 2
	prob_events = np.ones(30)
	prob_events[0] = ratio_noise
	prob_events = prob_events/prob_events.sum()

	n_events = np.random.choice(np.arange(len(prob_events)), p = prob_events)

	if src_density is None:
		pos_srcs = np.random.rand(n_events,3)*scale_x + offset_x
		# pos_srcs = np.random.rand(n_events,3)*(np.array([20.0, 20.0, 100e3]).reshape(1,-1)) + np.array([-10.0, -10.0, -100e3]).reshape(1,-1) + locs[np.random.choice(n_sta, size = n_events),:].reshape(-1,3)
	else:
		pos_srcs = src_density.sample(n_events)
		pos_srcs = np.concatenate((pos_srcs.reshape(-1,2), np.random.rand(n_events,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)

	src_origin = np.sort(np.random.rand(n_events)*(t_window + max_t) - max_t)
	print('Number events: %d'%n_events)
	print('Max t: %f'%max_t)

	## Create moveouts of all source-reciever phases, but `optionally' delete all arrivals > max_t
	## Randomly delete fraction of arrivals
	## Randomly add fraction of false arrivals
	## Add random error to theoretical arrival times

	# avg_pick_rate_per_time_range = [1/150.0, 1/120.0] ## Ratio of false picks
	avg_pick_rate_per_time_range = [2*6.25e-5, 2*0.001] ## Ratio of false picks
	n_rand_picks_per_station = (t_window + max_t)*(np.random.rand(locs.shape[0])*(avg_pick_rate_per_time_range[1] - avg_pick_rate_per_time_range[0]) + avg_pick_rate_per_time_range[0])

	if n_events == 0:

		arrivals_slice = np.zeros((0,4))

	else: ## n_events > 0

		use_uniform_dist = True
		del_ratio = [0.1, 0.5] ## Ratio of randomly deleted picks
		del_vals = np.random.rand(n_events)*(del_ratio[1] - del_ratio[0]) + del_ratio[0]
		dist_range = [200e3, 1500e3] ## Ratio of distance cutoff
		rnd_space_thresh = 50e3

		if use_uniform_dist == True: ## Have a different distance threshold for each phase type
			dist_vals = np.random.rand(n_events*4)*(dist_range[1] - dist_range[0]) + dist_range[0]
		else:
			dist_vals = beta(2,5).rvs(size = n_events*4)*(dist_range[1] - dist_range[0]) + dist_range[0]

		percent_trv_time_error = [0.02, 0.05]
		scale_trv_time_error = np.random.rand(n_events, 1, 1)*(percent_trv_time_error[1] - percent_trv_time_error[0]) + percent_trv_time_error[0]

		# for i in range(n_events):
		## First column, time, second column, station index, third column, amplitude, fourth column, phase type
		trv_out = trv(locs_cuda, torch.Tensor(pos_srcs).to(device)).cpu().detach().numpy()
		trv_out_noise = trv_out + src_origin.reshape(-1,1,1) + np.random.laplace(size = trv_out.shape)*np.abs(trv_out)*scale_trv_time_error
		true_moveouts = np.copy(trv_out) + src_origin.reshape(-1,1,1)

		## Use max travel time as proxy for distance, and delete arrival past max distance (since unlikely to have accurate arrival time)
		## and/or add higher noise.
		isrc, ista = np.where(trv_out.max(2) < max_t)

		arv_Pg = trv_out_noise[isrc,ista,0]
		arv_Pn = trv_out_noise[isrc,ista,1]
		arv_Sn = trv_out_noise[isrc,ista,2]
		arv_Lg = trv_out_noise[isrc,ista,3]
		ind = np.tile(ista,4)
		phase_type = np.arange(4).repeat(len(isrc))
		event_ind = np.tile(isrc,4)
		arrivals_slice_1 = np.concatenate((arv_Pg, arv_Pn, arv_Sn, arv_Lg), axis = 0).reshape(-1,1)
		# arrivals_slice = np.concatenate((arrivals_slice_1, ind[:,None], event_ind[:,None], phase_type[:,None]), axis = 1) ## Why does this not work?
		arrivals_slice = np.concatenate((arrivals_slice_1, ind.reshape(-1,1), event_ind.reshape(-1,1), phase_type.reshape(-1,1)), axis = 1)

		## Per event, delete all events with source-reciever pairs > dist_vals + noise per pick and per events
		pd_distances = np.linalg.norm(ftrns1(locs[arrivals_slice[:,1].astype('int')].reshape(-1,3)) - ftrns1(pos_srcs[arrivals_slice[:,2].astype('int')].reshape(-1,3)), axis = 1)
		idel = np.where((pd_distances + rnd_space_thresh*np.random.randn(len(pd_distances)))/dist_vals[arrivals_slice[:,2].astype('int') + n_events*arrivals_slice[:,3].astype('int')] > 1.0)[0]
		# print(pd_distances.shape)
		# print(arrivals_slice.shape)
		# print(idel.shape)
		# print(pd_distances)
		# print(dist_vals[arrivals_slice[:,2].astype('int')])

		arrivals_slice = np.delete(arrivals_slice, idel, axis = 0)

		# pdb.set_trace()

		## Per event, delete del_vals ratio of all true picks
		tree_event_inds = cKDTree(arrivals_slice[:,2][:,None])
		lp_event_inds = tree_event_inds.query_ball_point(np.arange(n_events)[:,None], r = 0)
		lp_event_cnts = np.array([len(lp_event_inds[j]) for j in range(len(lp_event_inds))])
		lp_event_ratios = lp_event_cnts*del_vals
		prob_del = lp_event_ratios[arrivals_slice[:,2].astype('int')]

		if prob_del.sum() > 0:

			prob_del = prob_del/prob_del.sum()
			# print(prob_del.shape)
			# print(prob_del)
			# print(arrivals_slice.shape[0])

			sample_random_del = np.unique(np.random.choice(np.arange(arrivals_slice.shape[0]), p = prob_del, size = int(lp_event_ratios.sum()))) # Not using replace = False
			arrivals_slice = np.delete(arrivals_slice, sample_random_del, axis = 0)

	## Add random rate of false picks per station
	arrivals_false = np.random.rand(int(n_rand_picks_per_station.sum()))*(t_window + max_t) - max_t
	arrivals_false_ind = np.random.choice(locs.shape[0], p = n_rand_picks_per_station/n_rand_picks_per_station.sum(), size = int(n_rand_picks_per_station.sum()))
	arrivals_false = np.concatenate((arrivals_false[:,None], arrivals_false_ind[:,None], -1*np.ones((len(arrivals_false),2))), axis = 1)

	arrivals = np.concatenate((arrivals_slice, arrivals_false), axis = 0)

	## Check which sources still active (or wait until the subset of stations is chosen..)
	## Alteratively, and more simply, just use all stations available, and sub-select inputs of stations
	## the station indices returned would be directly in terms of the inputs locs

	# if n_events > 0:
	# 	tree_event_inds = cKDTree(arrivals_slice[:,2][:,None])
	# 	lp_event_inds = tree_event_inds.query_ball_point(np.arange(n_events)[:,None], r = 0)
	# 	lp_event_cnts = np.array([len(lp_event_inds[j]) for j in range(len(lp_event_inds))])
	# 	#iactive_srcs = np.where(lp_event_cnts >= min_pick_cnt)[0]

	# else:
	# 	iactive_srcs = np.array([])

	## Make inputs (on Cartesian product graph - can base this on
	## the input Cartesian product graph parameters, which can be built before,
	## based on a subset of stations. E.g., do the sub-selecting of which active
	## stations before calling this script).

	## Can use the 1d embedding strategy used in regular generate_synthetic_data and the pre-computed x_grids_trv to
	## measure the nearest matches for all src-reciever pairs. Can also only consider moveouts < max_t, for consistency.

	iactive_arvs = np.where(arrivals[:,2] > -1)[0]
	iunique_srcs = np.unique(arrivals[iactive_arvs,2]).astype('int')
	cnt_src_picks = []
	for i in range(len(iunique_srcs)):
		cnt_src_picks.append(len(np.unique(arrivals[arrivals[:,2] == iunique_srcs[i],1])) >= min_sta_cnt)
	cnt_src_picks = np.array(cnt_src_picks)
	active_sources = iunique_srcs[np.where(cnt_src_picks > 0)[0]]
	src_times_active = src_origin[active_sources]

	# n_batch = 1
	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	use_uniform_sampling = False
	if use_uniform_sampling == True:

		time_samples = np.random.rand(n_batch)*max_t

	else:

		time_samples = np.random.rand(n_batch)*max_t
		l_src_times_active = len(src_times_active)
		if len(src_times_active) > 1:
			for j in range(n_batch):
				if np.random.rand() > 0.2: # 30% of samples, re-focus time. # 0.7
					time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()

	time_samples = np.sort(time_samples)

	# print('Need to implement not uniform origin time sampling')

	t_win = 10.0
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = 2.0*t_win + max_t/2.0) 
	# print(time.time() - st)

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
	if len(lp_concat) == 0:
		lp_concat = np.array([0]) # So it doesnt fail?
	arrivals_select = arrivals[lp_concat]

	# phase_observed_select = phase_observed[lp_concat]
	# tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec) ## Commenting out, as not used
	# print(time.time() - st)

	Trv_subset_Pg = []
	Trv_subset_Pn = []
	Trv_subset_Sn = []
	Trv_subset_Lg = []
	Station_indices = []
	Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	# if (fixed_subnetworks is not None):
	# 	fixed_subnetworks_flag = 1
	# else:
	# 	fixed_subnetworks_flag = 0		

	# active_sources_per_slice_l = []
	# src_positions_active_per_slice_l = []
	# src_times_active_per_slice_l = []

	for i in range(n_batch):
		## Can also select sub-networks that are real, realizations.
		i0 = np.random.randint(0, high = len(x_grids))
		n_spc = x_grids[i0].shape[0]

		use_full_network = True
		if use_full_network == True:
			n_sta_select = n_sta
			ind_sta_select = np.arange(n_sta)

		else:

			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
				n_sta_select = len(fixed_subnetworks[isub_network])
				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.
			
			else:
				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

		## Could potentially already sub-select src-reciever pairs here
		ind_pairs_repeat = np.tile(ind_sta_select, n_spc).reshape(-1,1)
		ind_src_pairs_repeat = np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1)
		Trv_subset_Pg.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , , len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_Pn.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_Sn.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,2].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_Lg.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,3].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication

		Station_indices.append(ind_sta_select) # record subsets used
		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
		Grid_indices.append(i0)
		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
		sc += len(Sample_indices[-1])

		# active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

		# active_sources_per_slice = np.where(n_unique_station_counts_per_slice >= min_sta_arrival)[0] # subset of sources
		# active_sources_per_slice_l.append(active_sources_per_slice)

	Trv_subset_Pg = np.vstack(Trv_subset_Pg)
	Trv_subset_Pn = np.vstack(Trv_subset_Pn)
	Trv_subset_Sn = np.vstack(Trv_subset_Sn)
	Trv_subset_Lg = np.vstack(Trv_subset_Lg)
	Batch_indices = np.hstack(Batch_indices)
	# print(time.time() - st)

	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(3)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
	n_arvs = arrivals_select.shape[0]
	# arrivals_select = np.concatenate((arrivals_0, arrivals_select, arrivals_1), axis = 0)

	# Rather slow!
	iargsort = np.argsort(arrivals_select[:,0])
	arrivals_select = arrivals_select[iargsort]
	# phase_observed_select = phase_observed_select[iargsort]

	# iwhere_p = np.where(phase_observed_select == 0)[0]
	# iwhere_s = np.where(phase_observed_select == 1)[0]
	# n_arvs_p = len(iwhere_p)
	# n_arvs_s = len(iwhere_s)

	mask_moveout_Pg = (Trv_subset_Pg[:,0] > max_t)
	mask_moveout_Pn = (Trv_subset_Pn[:,0] > max_t)
	mask_moveout_Sn = (Trv_subset_Sn[:,0] > max_t)
	mask_moveout_Lg = (Trv_subset_Lg[:,0] > max_t)

	query_time_Pg = Trv_subset_Pg[:,0] + Batch_indices*offset_per_batch + Trv_subset_Pg[:,1]*offset_per_station
	query_time_Pn = Trv_subset_Pn[:,0] + Batch_indices*offset_per_batch + Trv_subset_Pn[:,1]*offset_per_station
	query_time_Sn = Trv_subset_Sn[:,0] + Batch_indices*offset_per_batch + Trv_subset_Sn[:,1]*offset_per_station
	query_time_Lg = Trv_subset_Lg[:,0] + Batch_indices*offset_per_batch + Trv_subset_Lg[:,1]*offset_per_station

	## No phase type information
	ip_Pg = np.searchsorted(arrivals_select[:,0], query_time_Pg)
	ip_Pn = np.searchsorted(arrivals_select[:,0], query_time_Pn)
	ip_Sn = np.searchsorted(arrivals_select[:,0], query_time_Sn)
	ip_Lg = np.searchsorted(arrivals_select[:,0], query_time_Lg)

	## Wouldn't have to do these min and max bounds if arrivals was appended 
	## to have 2 extra picks at very large negative and positive times
	ip_Pg_pad = np.minimum(np.maximum(ip_Pg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_Pn_pad = np.minimum(np.maximum(ip_Pn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_Sn_pad = np.minimum(np.maximum(ip_Sn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_Lg_pad = np.minimum(np.maximum(ip_Lg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	# ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
	# ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

	kernel_sig_t = 20.0 # Can speed up by only using matches.
	use_nearest_arrival = True
	if use_nearest_arrival == True:
		rel_t_Pg = kernel_sig_t*5*mask_moveout_Pg + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_Pn = kernel_sig_t*5*mask_moveout_Pn + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_Sn = kernel_sig_t*5*mask_moveout_Sn + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_Lg = kernel_sig_t*5*mask_moveout_Lg + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	else: ## Use 3 nearest arrivals
		## Add mask for large source-station offsets, in terms of relative time >> kernel_sig_t
		rel_t_Pg = kernel_sig_t*5*np.expand_dims(mask_moveout_Pg, axis = 2) + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_Pn = kernel_sig_t*5*np.expand_dims(mask_moveout_Pn, axis = 2) + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_Sn = kernel_sig_t*5*np.expand_dims(mask_moveout_Sn, axis = 2) + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		rel_t_Lg = kernel_sig_t*5*np.expand_dims(mask_moveout_Lg, axis = 2) + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.


	# pdb.set_trace()

	## Based on sta_select, find the subset of station-source pairs needed for each of the `subgraphs' of Cartesian product graph
	# [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] = extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs[sta_select], np.arange(len(sta_select)), x_grids[grid_select], ftrns1, ftrns2, max_deg_offset = 5.0, k_nearest_pairs = 30)

	## Should also grab inputs from different possible source depths
	## In general, input feature dimension will be higher

	Inpts = []
	Masks = []
	Lbls = []
	X_query = []

	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grids[grid_select].shape[0]
		n_sta_slice = len(sta_select)

		# Rather than fully populate inpt for all x_grids and locs, why not only select subset inside the subset of Cartesian product graph
		## E.g., can find all pairswise matches of full set to an element in subset using cKDTree, then can extract these pairs,
		## and can ensure the ordering is the in the order of all used stations for each source node (e.g., the same ordering given
		## by A_src_in_prod and the nodes of A_prod.
		ind_sta_subset = sta_select[A_src_in_sta[0]]

		## In general, we only end up using ~2% of these pairs.
		## Ideally, we could only create a fraction of these entries to begin with.

		feature_Pg = np.exp(-0.5*(rel_t_Pg[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice) ## Could give sign information if prefered
		feature_Pn = np.exp(-0.5*(rel_t_Pn[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
		feature_Sn = np.exp(-0.5*(rel_t_Sn[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
		feature_Lg = np.exp(-0.5*(rel_t_Lg[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)

		## Now need to only grab the necessary pairs, and reshape

		feature_Pg_subset = feature_Pg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
		feature_Pn_subset = feature_Pn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
		feature_Sn_subset = feature_Sn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
		feature_Lg_subset = feature_Lg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)

		Inpts.append(np.concatenate((feature_Pg_subset, feature_Pn_subset, feature_Sn_subset, feature_Lg_subset), axis = 1))
		Masks.append(Inpts[-1] > thresh_mask)


		# t_query = (np.random.rand(n_queries)*10.0 - 5.0) + time_samples[i]
		t_slice = np.arange(-5.0, 5.0 + 1.0, 1.0) ## Window, over which to extract labels.

		n_extra_queries = 3000
		if len(active_sources) > 0:
			x_query_focused = np.random.rand(n_extra_queries, 3)*np.array([10.0, 10.0, 0.0]).reshape(1,-1) + np.array([-5.0, -5.0, 0.0]).reshape(1,-1) + pos_srcs[np.random.choice(active_sources, size = n_extra_queries)]
			x_query_focused[:,2] = np.random.rand(n_extra_queries)*(x_query[:,2].max() - x_query[:,2].min()) + x_query[:,2].min()
			x_query_focused = np.concatenate((x_query, x_query_focused), axis = 0)

		else:
			x_query_focused = np.concatenate((x_query, np.random.rand(n_extra_queries, 3)*(x_query.max(0, keepdims = True) - x_query.min(0, keepdims = True)) + x_query.min(0, keepdims = True)), axis = 0)


		X_query.append(x_query_focused)

		if len(active_sources) == 0:
			lbls_grid = np.zeros((x_grids[grid_select].shape[0], len(t_slice)))
			lbls_query = np.zeros((x_query_focused.shape[0], len(t_slice)))
		else:

			ignore_depth = True
			if ignore_depth == True:
				lbl_vec = np.array([1.0, 1.0, 0.0]).reshape(1,-1)

			else:
				lbl_vec = np.array([1.0, 1.0, 1.0]).reshape(1,-1)


			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query_focused), axis = 1) - np.expand_dims(ftrns1(pos_srcs[active_sources]*lbl_vec), axis = 0))**2)/(src_x_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_origin[active_sources].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

		Lbls.append(lbls_query)

		## Currently this should be shaped as for each source node, all station nodes

		# inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
		# inpt[Trv_subset_Pg[ind_select,2].astype('int'), Trv_subset_Pg[ind_select,1].astype('int'), 0] = feature_Pg
		# inpt[Trv_subset_Pn[ind_select,2].astype('int'), Trv_subset_Pn[ind_select,1].astype('int'), 1] = feature_Pn
		# inpt[Trv_subset_Sn[ind_select,2].astype('int'), Trv_subset_Sn[ind_select,1].astype('int'), 2] = feature_Sn
		# inpt[Trv_subset_Lg[ind_select,2].astype('int'), Trv_subset_Lg[ind_select,1].astype('int'), 3] = feature_Lg

		## Make labels

	## Again, need to only consider pairs that are within source-reciever distance (or time, of max_t)

	## Make labels

	if verbose == True:
		print('Time to generate batch: %0.2f s \n'%(time.time() - st))

	if len(active_sources) == 0:
		true_moveouts = None

	data = [arrivals, np.concatenate((pos_srcs, src_origin.reshape(-1,1)), axis = 1), active_sources, true_moveouts, time_samples]

	return [Inpts, Masks, X_query, Lbls], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

	# pw_distances = pd(ftrns1(pos_srcs), ftrns1(locs))
	# idel1, idel2 = np.where(pw_distaces/dist_vals.reshape(-1,1) < 1.0) ## Find where, per source, distances are less than threshold
	# tree_ind_keep = cKDTree(np.concatenate((isrc.reshape(-1,1), ista.reshape(-1,1)), axis = 1))
	# idist = np.where(tree_ind_keep.query(np.concatenate((idel1.reshape(-1,1), idel2.reshape(-1,1)), axis = 1)) == 0)[0] # Find large distances also contained in this set

	## Subselect a set of stations (or not, optionally, for simplicity)

	## Determine which pairs of source-stations needed, given the subgraph of the Cartesian product graph

	## Create the Input and Mask tensors (sub-selected for the subgraph of Cartesian product graph)
	## When creating these, use the pre-computed x_grids_trv arrays
	## Create inputs for unlabelled input picks, for all four phase types of Pg, Pn, Sn, Lg

	## Determine which events have >= min_number_picks

	## Create spatial labels for all events (and only have positive
	## labels for active sources).

def sample_inputs(trv, arrivals, locs, x_grids, x_grids_trv, time_samples, A_src_in_sta, lat_range, lon_range, depth_range, ftrns1, ftrns2, verbose = True):

	if verbose == True:
		st = time.time()

	## Can embed the three nearest arrivals in time, not just one, to handle
	## large numbers of picks in a short window
	## and the large kernel embedding size

	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
	locs_cpu = torch.Tensor(locs).to(device)
	x_grid_cpu = torch.Tensor(x_grids[0]).to(device)
	n_sta = locs.shape[0]
	n_spc = x_grids[0].shape[0]

	## Set up time window
	max_t = 500.0
	t_window = 5.0*max_t
	min_sta_cnt = 5
	n_batch = len(time_samples)

	# n_batch = 1
	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	t_win = 10.0
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = 2.0*t_win + max_t/2.0) 
	# print(time.time() - st)

	ind_sta_select = np.arange(locs.shape[0])
	sta_select = np.arange(locs.shape[0])

	## Could potentially already sub-select src-reciever pairs here
	i0 = 0
	ind_pairs_repeat = np.tile(ind_sta_select, n_spc).reshape(-1,1)
	ind_src_pairs_repeat = np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1)
	Trv_subset_Pg = np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , , len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
	Trv_subset_Pn = np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
	Trv_subset_Sn = np.concatenate((x_grids_trv[i0][:,ind_sta_select,2].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
	Trv_subset_Lg = np.concatenate((x_grids_trv[i0][:,ind_sta_select,3].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication

	mask_moveout_Pg = (Trv_subset_Pg[:,0] > max_t)
	mask_moveout_Pn = (Trv_subset_Pn[:,0] > max_t)
	mask_moveout_Sn = (Trv_subset_Sn[:,0] > max_t)
	mask_moveout_Lg = (Trv_subset_Lg[:,0] > max_t)

	query_time_Pg = Trv_subset_Pg[:,0] + Trv_subset_Pg[:,1]*offset_per_station
	query_time_Pn = Trv_subset_Pn[:,0] + Trv_subset_Pn[:,1]*offset_per_station
	query_time_Sn = Trv_subset_Sn[:,0] + Trv_subset_Sn[:,1]*offset_per_station
	query_time_Lg = Trv_subset_Lg[:,0] + Trv_subset_Lg[:,1]*offset_per_station

	Inpts = []
	Masks = []

	thresh_mask = 0.01
	n_spc = x_grids[0].shape[0]
	n_sta_slice = len(sta_select)
	ind_sta_subset = sta_select[A_src_in_sta[0]]

	for i in range(len(time_samples)):

		offset_per_batch = 1.5*max_t
		offset_per_station = 1.5*n_batch*offset_per_batch

		arrivals_offset = -time_samples[i] + offset_per_station*arrivals[lp[i],1] ## Actually, make disjoint, both in station axis, and in batch number.
		one_vec = np.concatenate((np.ones(1), np.zeros(3)), axis = 0).reshape(1,-1)
		arrivals_select = arrivals[lp[i]] + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
		n_arvs = arrivals_select.shape[0]
		# arrivals_select = np.concatenate((arrivals_0, arrivals_select, arrivals_1), axis = 0)

		# Rather slow!
		iargsort = np.argsort(arrivals_select[:,0])
		arrivals_select = arrivals_select[iargsort]
		# phase_observed_select = phase_observed_select[iargsort]

		# iwhere_p = np.where(phase_observed_select == 0)[0]
		# iwhere_s = np.where(phase_observed_select == 1)[0]
		# n_arvs_p = len(iwhere_p)
		# n_arvs_s = len(iwhere_s)

		## No phase type information
		ip_Pg = np.searchsorted(arrivals_select[:,0], query_time_Pg)
		ip_Pn = np.searchsorted(arrivals_select[:,0], query_time_Pn)
		ip_Sn = np.searchsorted(arrivals_select[:,0], query_time_Sn)
		ip_Lg = np.searchsorted(arrivals_select[:,0], query_time_Lg)

		## Wouldn't have to do these min and max bounds if arrivals was appended 
		## to have 2 extra picks at very large negative and positive times
		ip_Pg_pad = np.minimum(np.maximum(ip_Pg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
		ip_Pn_pad = np.minimum(np.maximum(ip_Pn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
		ip_Sn_pad = np.minimum(np.maximum(ip_Sn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
		ip_Lg_pad = np.minimum(np.maximum(ip_Lg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
		# ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
		# ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

		kernel_sig_t = 20.0 # Can speed up by only using matches.
		use_nearest_arrival = True
		if use_nearest_arrival == True:
			rel_t_Pg = kernel_sig_t*5*mask_moveout_Pg + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_Pn = kernel_sig_t*5*mask_moveout_Pn + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_Sn = kernel_sig_t*5*mask_moveout_Sn + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_Lg = kernel_sig_t*5*mask_moveout_Lg + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
		else: ## Use 3 nearest arrivals
			## Add mask for large source-station offsets, in terms of relative time >> kernel_sig_t
			rel_t_Pg = kernel_sig_t*5*np.expand_dims(mask_moveout_Pg, axis = 2) + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_Pn = kernel_sig_t*5*np.expand_dims(mask_moveout_Pn, axis = 2) + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_Sn = kernel_sig_t*5*np.expand_dims(mask_moveout_Sn, axis = 2) + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
			rel_t_Lg = kernel_sig_t*5*np.expand_dims(mask_moveout_Lg, axis = 2) + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.

		# Rather than fully populate inpt for all x_grids and locs, why not only select subset inside the subset of Cartesian product graph
		## E.g., can find all pairswise matches of full set to an element in subset using cKDTree, then can extract these pairs,
		## and can ensure the ordering is the in the order of all used stations for each source node (e.g., the same ordering given
		## by A_src_in_prod and the nodes of A_prod.

		## In general, we only end up using ~2% of these pairs.
		## Ideally, we could only create a fraction of these entries to begin with.

		## Instead of reshaping, could directly extract needed entries

		feature_Pg = np.exp(-0.5*(rel_t_Pg**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice) ## Could give sign information if prefered
		feature_Pn = np.exp(-0.5*(rel_t_Pn**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
		feature_Sn = np.exp(-0.5*(rel_t_Sn**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
		feature_Lg = np.exp(-0.5*(rel_t_Lg**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)

		## Now need to only grab the necessary pairs, and reshape

		feature_Pg_subset = feature_Pg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
		feature_Pn_subset = feature_Pn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
		feature_Sn_subset = feature_Sn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
		feature_Lg_subset = feature_Lg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)

		Inpts.append(np.concatenate((feature_Pg_subset, feature_Pn_subset, feature_Sn_subset, feature_Lg_subset), axis = 1))
		Masks.append(Inpts[-1] > thresh_mask)

	if verbose == True:
		print('Time to generate batch: %0.2f s \n'%(time.time() - st))

	# data = [arrivals, np.concatenate((pos_srcs, src_origin.reshape(-1,1)), axis = 1), active_sources, true_moveouts, time_samples]

	return Inpts, Masks ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)


def sample_inputs_efficient(arrivals, trv_out, A_src_in_sta, sample_times, dt = 0.5, kernel_sig_t = 20.0):

	## First, embed all picks to a discrete dt sampling, for each station seperately

	## Only a subset of stations needed for each source node.

	## For each source node, a different moveout across all stations

	t_win = 10.0
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = 2.0*t_win + 3.0*kernel_sig_t + max_t/2.0) 	

	trv_out_subset_Pg = trv_out[A_src_in_sta[0], A_src_in_sta[1], 0]
	trv_out_subset_Pn = trv_out[A_src_in_sta[0], A_src_in_sta[1], 1]
	trv_out_subset_Sn = trv_out[A_src_in_sta[0], A_src_in_sta[1], 2]
	trv_out_subset_Lg = trv_out[A_src_in_sta[0], A_src_in_sta[1], 3]

	## For each sample, make embedding into a discrete n_sta x n_time_steps one hot encoding

	for i in range(len(time_samples)):

		arrivals_subset = arrivals[lp[i]]



		embed_indices = (np.floor(arrivals_subset[:,0] - time_samples[i])/dt).astype('int')

		assert(embed_indices.min() >= 0)







# def sample_inputs_backup(trv, arrivals, locs, x_grids, x_grids_trv, x_query, time_samples, lat_range, lon_range, depth_range, ftrns1, ftrns2, n_batch = 1, src_density = None, verbose = True):

# 	if verbose == True:
# 		st = time.time()

# 	## Can embed the three nearest arrivals in time, not just one, to handle 39
# 	## large numbers of picks in a short window
# 	## and the large kernel embedding size

# 	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
# 	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
# 	locs_cpu = torch.Tensor(locs).to(device)
# 	x_grid_cpu = torch.Tensor(x_grid).to(device)
# 	n_sta = locs.shape[0]

# 	## Set up time window
# 	max_t = 500.0
# 	t_window = 5.0*max_t
# 	min_sta_cnt = 5
# 	n_batch = len(time_samples)

# 	# n_batch = 1
# 	offset_per_batch = 1.5*max_t
# 	offset_per_station = 1.5*n_batch*offset_per_batch

# 	t_win = 10.0
# 	tree = cKDTree(arrivals[:,0][:,None])
# 	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = 2.0*t_win + max_t/2.0) 
# 	# print(time.time() - st)

# 	# lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
# 	# if len(lp_concat) == 0:
# 	# 	lp_concat = np.array([0]) # So it doesnt fail?
# 	# arrivals_select = arrivals[lp_concat]

# 	# phase_observed_select = phase_observed[lp_concat]
# 	# tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec) ## Commenting out, as not used
# 	# print(time.time() - st)

# 	ind_sta_select = np.arange(locs.shape[0])

# 	## Could potentially already sub-select src-reciever pairs here
# 	ind_pairs_repeat = np.tile(ind_sta_select, n_spc).reshape(-1,1)
# 	ind_src_pairs_repeat = np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1)
# 	Trv_subset_Pg = np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , , len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 	Trv_subset_Pn = np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 	Trv_subset_Sn = np.concatenate((x_grids_trv[i0][:,ind_sta_select,2].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 	Trv_subset_Lg = np.concatenate((x_grids_trv[i0][:,ind_sta_select,3].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication

# 	# Station_indices.append(ind_sta_select) # record subsets used
# 	# Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
# 	# Grid_indices.append(i0)
# 	# Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
# 	# sc += len(Sample_indices[-1])

# 	offset_per_batch = 1.5*max_t
# 	offset_per_station = 1.5*n_batch*offset_per_batch

# 	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
# 	one_vec = np.concatenate((np.ones(1), np.zeros(3)), axis = 0).reshape(1,-1)
# 	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
# 	n_arvs = arrivals_select.shape[0]
# 	# arrivals_select = np.concatenate((arrivals_0, arrivals_select, arrivals_1), axis = 0)

# 	# Rather slow!
# 	iargsort = np.argsort(arrivals_select[:,0])
# 	arrivals_select = arrivals_select[iargsort]
# 	# phase_observed_select = phase_observed_select[iargsort]

# 	# iwhere_p = np.where(phase_observed_select == 0)[0]
# 	# iwhere_s = np.where(phase_observed_select == 1)[0]
# 	# n_arvs_p = len(iwhere_p)
# 	# n_arvs_s = len(iwhere_s)

# 	mask_moveout_Pg = (Trv_subset_Pg[:,0] > max_t)
# 	mask_moveout_Pn = (Trv_subset_Pn[:,0] > max_t)
# 	mask_moveout_Sn = (Trv_subset_Sn[:,0] > max_t)
# 	mask_moveout_Lg = (Trv_subset_Lg[:,0] > max_t)

# 	query_time_Pg = Trv_subset_Pg[:,0] + Batch_indices*offset_per_batch + Trv_subset_Pg[:,1]*offset_per_station
# 	query_time_Pn = Trv_subset_Pn[:,0] + Batch_indices*offset_per_batch + Trv_subset_Pn[:,1]*offset_per_station
# 	query_time_Sn = Trv_subset_Sn[:,0] + Batch_indices*offset_per_batch + Trv_subset_Sn[:,1]*offset_per_station
# 	query_time_Lg = Trv_subset_Lg[:,0] + Batch_indices*offset_per_batch + Trv_subset_Lg[:,1]*offset_per_station

# 	## No phase type information
# 	ip_Pg = np.searchsorted(arrivals_select[:,0], query_time_Pg)
# 	ip_Pn = np.searchsorted(arrivals_select[:,0], query_time_Pn)
# 	ip_Sn = np.searchsorted(arrivals_select[:,0], query_time_Sn)
# 	ip_Lg = np.searchsorted(arrivals_select[:,0], query_time_Lg)

# 	## Wouldn't have to do these min and max bounds if arrivals was appended 
# 	## to have 2 extra picks at very large negative and positive times
# 	ip_Pg_pad = np.minimum(np.maximum(ip_Pg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_Pn_pad = np.minimum(np.maximum(ip_Pn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_Sn_pad = np.minimum(np.maximum(ip_Sn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_Lg_pad = np.minimum(np.maximum(ip_Lg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	# ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
# 	# ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

# 	kernel_sig_t = 20.0 # Can speed up by only using matches.
# 	use_nearest_arrival = True
# 	if use_nearest_arrival == True:
# 		rel_t_Pg = kernel_sig_t*5*mask_moveout_Pg + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Pn = kernel_sig_t*5*mask_moveout_Pn + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Sn = kernel_sig_t*5*mask_moveout_Sn + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Lg = kernel_sig_t*5*mask_moveout_Lg + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	else: ## Use 3 nearest arrivals
# 		## Add mask for large source-station offsets, in terms of relative time >> kernel_sig_t
# 		rel_t_Pg = kernel_sig_t*5*np.expand_dims(mask_moveout_Pg, axis = 2) + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Pn = kernel_sig_t*5*np.expand_dims(mask_moveout_Pn, axis = 2) + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Sn = kernel_sig_t*5*np.expand_dims(mask_moveout_Sn, axis = 2) + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Lg = kernel_sig_t*5*np.expand_dims(mask_moveout_Lg, axis = 2) + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.


# 	# pdb.set_trace()

# 	## Based on sta_select, find the subset of station-source pairs needed for each of the `subgraphs' of Cartesian product graph
# 	# [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] = extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs[sta_select], np.arange(len(sta_select)), x_grids[grid_select], ftrns1, ftrns2, max_deg_offset = 5.0, k_nearest_pairs = 30)

# 	## Should also grab inputs from different possible source depths
# 	## In general, input feature dimension will be higher

# 	Inpts = []
# 	Masks = []
# 	Lbls = []
# 	X_query = []

# 	thresh_mask = 0.01
# 	for i in range(n_batch):
# 		# Create inputs and mask
# 		grid_select = Grid_indices[i]
# 		ind_select = Sample_indices[i]
# 		sta_select = Station_indices[i]
# 		n_spc = x_grids[grid_select].shape[0]
# 		n_sta_slice = len(sta_select)

# 		# Rather than fully populate inpt for all x_grids and locs, why not only select subset inside the subset of Cartesian product graph
# 		## E.g., can find all pairswise matches of full set to an element in subset using cKDTree, then can extract these pairs,
# 		## and can ensure the ordering is the in the order of all used stations for each source node (e.g., the same ordering given
# 		## by A_src_in_prod and the nodes of A_prod.
# 		ind_sta_subset = sta_select[A_src_in_sta[0]]

# 		## In general, we only end up using ~2% of these pairs.
# 		## Ideally, we could only create a fraction of these entries to begin with.

# 		feature_Pg = np.exp(-0.5*(rel_t_Pg[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice) ## Could give sign information if prefered
# 		feature_Pn = np.exp(-0.5*(rel_t_Pn[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
# 		feature_Sn = np.exp(-0.5*(rel_t_Sn[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
# 		feature_Lg = np.exp(-0.5*(rel_t_Lg[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)

# 		## Now need to only grab the necessary pairs, and reshape

# 		feature_Pg_subset = feature_Pg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
# 		feature_Pn_subset = feature_Pn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
# 		feature_Sn_subset = feature_Sn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
# 		feature_Lg_subset = feature_Lg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)

# 		Inpts.append(np.concatenate((feature_Pg_subset, feature_Pn_subset, feature_Sn_subset, feature_Lg_subset), axis = 1))
# 		Masks.append(Inpts[-1] > thresh_mask)


# 		# t_query = (np.random.rand(n_queries)*10.0 - 5.0) + time_samples[i]
# 		t_slice = np.arange(-5.0, 5.0 + 1.0, 1.0) ## Window, over which to extract labels.

# 		n_extra_queries = 3000
# 		if len(active_sources) > 0:
# 			x_query_focused = np.random.rand(n_extra_queries, 3)*np.array([10.0, 10.0, 0.0]).reshape(1,-1) + np.array([-5.0, -5.0, 0.0]).reshape(1,-1) + pos_srcs[np.random.choice(active_sources, size = n_extra_queries)]
# 			x_query_focused[:,2] = np.random.rand(n_extra_queries)*(x_query[:,2].max() - x_query[:,2].min()) + x_query[:,2].min()
# 			x_query_focused = np.concatenate((x_query, x_query_focused), axis = 0)

# 		else:
# 			x_query_focused = np.concatenate((x_query, np.random.rand(n_extra_queries, 3)*(x_query.max(0, keepdims = True) - x_query.min(0, keepdims = True)) + x_query.min(0, keepdims = True)), axis = 0)


# 		X_query.append(x_query_focused)


# 	if verbose == True:
# 		print('Time to generate batch: %0.2f s \n'%(time.time() - st))

# 	if len(active_sources) == 0:
# 		true_moveouts = None

# 	data = [arrivals, np.concatenate((pos_srcs, src_origin.reshape(-1,1)), axis = 1), active_sources, true_moveouts, time_samples]

# 	return [Inpts, Masks, X_query, Lbls], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

# def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, ftrns1, ftrns2, n_batch = 75, max_rate_events = 5000/8, max_miss_events = 3500/8, max_false_events = 2000/8, T = 3600.0*3.0, dt = 30, tscale = 3600.0, n_sta_range = [0.35, 1.0], use_sources = False, use_full_network = False, fixed_subnetworks = None, use_preferential_sampling = False, use_shallow_sources = False, plot_on = False, verbose = False):

# 	if verbose == True:
# 		st = time.time()


# 	scale_x = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
# 	offset_x = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)
# 	n_sta = locs.shape[0]
# 	locs_tensor = torch.Tensor(locs).cuda()

# 	tsteps = np.arange(0, T + dt, dt)
# 	tvec = np.arange(-tscale*4, tscale*4 + dt, dt)
# 	tvec_kernel = np.exp(-(tvec**2)/(2.0*(tscale**2)))
# 	# p_rate_events = np.convolve(np.random.randn(len(tsteps)), tvec_kernel, 'same')
# 	# scipy.signal.convolve(in1, in2, mode='full'

# 	## Can augment global parameters at a different rate.
# 	p_rate_events = fftconvolve(np.random.randn(2*locs.shape[0] + 3, len(tsteps)), tvec_kernel.reshape(1,-1).repeat(2*locs.shape[0] + 3,0), 'same', axes = 1)
# 	c_cor = (p_rate_events@p_rate_events.T) ## Not slow!
# 	global_event_rate, global_miss_rate, global_false_rate = p_rate_events[0:3,:]

# 	# Process global event rate, to physical units.
# 	global_event_rate = (global_event_rate - global_event_rate.min())/(global_event_rate.max() - global_event_rate.min()) # [0,1] scale
# 	min_add = np.random.rand()*0.25*max_rate_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand()*(0.5*max_rate_events - min_add) + 0.5*max_rate_events
# 	global_event_rate = global_event_rate*scale + min_add

# 	global_miss_rate = (global_miss_rate - global_miss_rate.min())/(global_miss_rate.max() - global_miss_rate.min()) # [0,1] scale
# 	min_add = np.random.rand()*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand()*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
# 	global_miss_rate = global_miss_rate*scale + min_add

# 	global_false_rate = (global_false_rate - global_false_rate.min())/(global_false_rate.max() - global_false_rate.min()) # [0,1] scale
# 	min_add = np.random.rand()*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand()*(0.5*max_false_events - min_add) + 0.5*max_false_events
# 	global_false_rate = global_false_rate*scale + min_add

# 	station_miss_rate = p_rate_events[3 + np.arange(n_sta),:]
# 	station_miss_rate = (station_miss_rate - station_miss_rate.min(1, keepdims = True))/(station_miss_rate.max(1, keepdims = True) - station_miss_rate.min(1, keepdims = True)) # [0,1] scale
# 	min_add = np.random.rand(n_sta,1)*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand(n_sta,1)*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
# 	station_miss_rate = station_miss_rate*scale + min_add

# 	station_false_rate = p_rate_events[3 + n_sta + np.arange(n_sta),:]
# 	station_false_rate = (station_false_rate - station_false_rate.min(1, keepdims = True))/(station_false_rate.max(1, keepdims = True) - station_false_rate.min(1, keepdims = True))
# 	min_add = np.random.rand(n_sta,1)*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
# 	scale = np.random.rand(n_sta,1)*(0.5*max_false_events - min_add) + 0.5*max_false_events
# 	station_false_rate = station_false_rate*scale + min_add


# 	## Sample events.
# 	vals = np.random.poisson(dt*global_event_rate/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average
# 	src_times = np.sort(np.hstack([np.random.rand(vals[j])*dt + tsteps[j] for j in range(len(vals))]))
# 	n_src = len(src_times)
# 	src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
# 	src_magnitude = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)


# 	if use_shallow_sources == True:
# 		sample_random_depths = gamma(1.75, 0.0).rvs(n_src)
# 		sample_random_grab = np.where(sample_random_depths > 5)[0] # Clip the long tails, and place in uniform, [0,5].
# 		sample_random_depths[sample_random_grab] = 5.0*np.random.rand(len(sample_random_grab))
# 		sample_random_depths = sample_random_depths/sample_random_depths.max() # Scale to range
# 		sample_random_depths = -sample_random_depths*(scale_x[0,2] - 2e3) + (offset_x[0,2] + scale_x[0,2] - 2e3) # Project along axis, going negative direction. Removing 2e3 on edges.
# 		src_positions[:,2] = sample_random_depths


# 	# m1 = [0.5761163, -0.5]
# 	m1 = [0.5761163, -0.21916288]
# 	m2 = 1.15
# 	dist_range = [15e3, 1000e3]
# 	spc_random = 30e3
# 	sig_t = 0.03 # 3 percent of travel time.
# 	# Just use, random, per-event distance threshold, combined with random miss rate, to determine extent of moveouts.

# 	amp_thresh = 1.0
# 	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

# 	use_uniform_distance_threshold = False
# 	## This previously sampled a skewed distribution by default, not it samples a uniform
# 	## distribution of the maximum source-reciever distances allowed for each event.
# 	if use_uniform_distance_threshold == True:
# 		dist_thresh = np.random.rand(n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
# 	else:
# 		## Use beta distribution to generate more samples with smaller moveouts
# 		# dist_thresh = -1.0*np.log(np.sqrt(np.random.rand(n_src))) ## Sort of strange dist threshold set!
# 		# dist_thresh = (dist_thresh*dist_range[1]/10.0 + dist_range[0]).reshape(-1,1)
# 		dist_thresh = beta(2,5).rvs(size = n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]

# 	# create different distance dependent thresholds.
# 	dist_thresh_p = dist_thresh + 50e3*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
# 	dist_thresh_s = dist_thresh + 50e3*np.random.laplace(size = dist_thresh.shape[0])[:,None]

# 	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
# 	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

# 	arrivals_theoretical = trv(torch.Tensor(locs).cuda(), torch.Tensor(src_positions[:,0:3]).cuda()).cpu().detach().numpy()
# 	arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
# 	arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
# 	src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

# 	# arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), log_amp_p[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1)), axis = 1)
# 	# arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), log_amp_s[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1)), axis = 1)
# 	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
# 	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
# 	arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

# 	s_extra = 0.0
# 	t_inc = np.floor(arrivals[:,3]/dt).astype('int')
# 	p_miss_rate = 0.5*station_miss_rate[arrivals[:,1].astype('int'), t_inc] + 0.5*global_miss_rate[t_inc]
# 	idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < dt*p_miss_rate/T)[0]


# 	arrivals = np.delete(arrivals, idel, axis = 0)
# 	# 0.5 sec to here

# 	## Determine which sources are active, here.
# 	n_events = len(src_times)
# 	min_sta_arrival = 4


# 	source_tree_indices = cKDTree(arrivals[:,2].reshape(-1,1))
# 	lp = source_tree_indices.query_ball_point(np.arange(n_events).reshape(-1,1), r = 0)
# 	lp_backup = [lp[j] for j in range(len(lp))]
# 	n_unique_station_counts = np.array([len(np.unique(arrivals[lp[j],1])) for j in range(n_events)])
# 	active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
# 	non_active_sources = np.delete(np.arange(n_events), active_sources, axis = 0)
# 	src_positions_active = src_positions[active_sources]
# 	src_times_active = src_times[active_sources]
# 	src_magnitude_active = src_magnitude[active_sources] ## Not currently used
# 	# src_indices_active = src_indices[active_sources] # Not apparent this is needed
# 	## Additional 0.15 sec

# 	## Determine which sources (absolute indices) are inside the interior region.
# 	# inside_interior = np.where((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))[0]
# 	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))


# 	coda_rate = 0.035 # 5 percent arrival have code. Probably more than this? Increased from 0.035.
# 	coda_win = np.array([0, 25.0]) # coda occurs within 0 to 15 s after arrival (should be less?) # Increased to 25, from 20.0
# 	icoda = np.where(np.random.rand(arrivals.shape[0]) < coda_rate)[0]
# 	if len(icoda) > 0:
# 		false_coda_arrivals = np.random.rand(len(icoda))*(coda_win[1] - coda_win[0]) + coda_win[0] + arrivals[icoda,0] + arrivals[icoda,3]
# 		false_coda_arrivals = np.concatenate((false_coda_arrivals.reshape(-1,1), arrivals[icoda,1].reshape(-1,1), -1.0*np.ones((len(icoda),1)), np.zeros((len(icoda),1)), -1.0*np.ones((len(icoda),1))), axis = 1)
# 		arrivals = np.concatenate((arrivals, false_coda_arrivals), axis = 0)

# 	## Base false events
# 	station_false_rate_eval = 0.5*station_false_rate + 0.5*global_false_rate
# 	vals = np.random.poisson(dt*station_false_rate_eval/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average

# 	## Too slow!
# 	# false_times = np.hstack([np.hstack([np.random.rand(vals[k,j])*dt + tsteps[j] for j in range(vals.shape[1])]) for k in range(n_sta)])
# 	# false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])

# 	# How to speed up this part?
# 	i1, i2 = np.where(vals > 0)
# 	v_val, t_val = vals[i1,i2], tsteps[i2]
# 	false_times = np.repeat(t_val, v_val) + np.random.rand(vals.sum())*dt
# 	false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])
# 	n_false = len(false_times)
# 	false_arrivals = np.concatenate((false_times.reshape(-1,1), false_indices.reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
# 	arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)

# 	# print('make spikes!')
# 	## Make false spikes!
# 	n_spikes = np.random.randint(0, high = int(80*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
# 	if n_spikes > 0:
# 		n_spikes_extent = np.random.randint(1, high = n_sta, size = n_spikes) ## This many stations per spike
# 		time_spikes = np.random.rand(n_spikes)*T
# 		sta_ind_spikes = np.hstack([np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)])
# 		sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*0.15 for j in range(n_spikes)])
# 		false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
# 		# -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1)
# 		arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes
# 	# print('make spikes!')

# 	## Compute arrival times, exactly.
# 	# 3.5 % error of theoretical, laplace distribution
# 	iz = np.where(arrivals[:,4] >= 0)[0]
# 	arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]

# 	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
# 	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
# 	phase_observed = np.copy(arrivals[:,-1]).astype('int')

# 	if len(iwhere_false) > 0: # For false picks, assign a random phase type
# 		phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))

# 	perturb_phases = True # For true picks, randomly flip a fraction of phases
# 	if (len(phase_observed) > 0)*(perturb_phases == True):
# 		n_switch = int(np.random.rand()*(0.2*len(iwhere_real))) # switch up to 20% phases
# 		iflip = np.random.choice(iwhere_real, size = n_switch, replace = False)
# 		phase_observed[iflip] = np.mod(phase_observed[iflip] + 1, 2)


# 	t_win = 10.0
# 	scale_vec = np.array([1,2*t_win]).reshape(1,-1)

# 	## Should try reducing kernel sizes
# 	src_t_kernel = 10.0 # Should decrease kernel soon!
# 	src_x_kernel = 50e3 # Add parallel secondary resolution prediction (condition on the low fre. resolution prediction)
# 	src_depth_kernel = 50e3 # was 5e3, increase depth kernel? ignore?
# 	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

# 	## Note, this used to be a Guassian, nearby times of know sources.
# 	if use_sources == False:
# 		time_samples = np.sort(np.random.rand(n_batch)*T) ## Uniform


# 	elif use_sources == True:
# 		time_samples = src_times_active[np.sort(np.random.choice(len(src_times_active), size = n_batch))]

# 	# focus_on_sources = True # Re-pick oorigin times close to sources
# 	l_src_times_active = len(src_times_active)
# 	if (use_preferential_sampling == True)*(len(src_times_active) > 1):
# 		for j in range(n_batch):
# 			if np.random.rand() > 0.5: # 30% of samples, re-focus time. # 0.7
# 				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()

# 	time_samples = np.sort(time_samples)

# 	max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))


# 	tree_src_times_all = cKDTree(src_times[:,np.newaxis])
# 	tree_src_times = cKDTree(src_times_active[:,np.newaxis])
# 	lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)
# 	lp_src_times = tree_src_times.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)


# 	st = time.time()
# 	tree = cKDTree(arrivals[:,0][:,None])
# 	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 
# 	# print(time.time() - st)

# 	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
# 	if len(lp_concat) == 0:
# 		lp_concat = np.array([0]) # So it doesnt fail?
# 	arrivals_select = arrivals[lp_concat]
# 	phase_observed_select = phase_observed[lp_concat]
# 	# tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec) ## Commenting out, as not used
# 	# print(time.time() - st)

# 	Trv_subset_p = []
# 	Trv_subset_s = []
# 	Station_indices = []
# 	Grid_indices = []
# 	Batch_indices = []
# 	Sample_indices = []
# 	sc = 0

# 	if (fixed_subnetworks is not None):
# 		fixed_subnetworks_flag = 1
# 	else:
# 		fixed_subnetworks_flag = 0		

# 	active_sources_per_slice_l = []
# 	src_positions_active_per_slice_l = []
# 	src_times_active_per_slice_l = []

# 	for i in range(n_batch):
# 		## Can also select sub-networks that are real, realizations.
# 		i0 = np.random.randint(0, high = len(x_grids))
# 		n_spc = x_grids[i0].shape[0]
# 		if use_full_network == True:
# 			n_sta_select = n_sta
# 			ind_sta_select = np.arange(n_sta)

# 		else:
# 			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
# 				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
# 				n_sta_select = len(fixed_subnetworks[isub_network])
# 				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.
			
# 			else:
# 				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
# 				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

# 		Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Station_indices.append(ind_sta_select) # record subsets used
# 		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
# 		Grid_indices.append(i0)
# 		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
# 		sc += len(Sample_indices[-1])


# 		active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

# 		# active_sources_per_slice = np.where(n_unique_station_counts_per_slice >= min_sta_arrival)[0] # subset of sources
# 		active_sources_per_slice_l.append(active_sources_per_slice)


# 	Trv_subset_p = np.vstack(Trv_subset_p)
# 	Trv_subset_s = np.vstack(Trv_subset_s)
# 	Batch_indices = np.hstack(Batch_indices)
# 	# print(time.time() - st)

# 	offset_per_batch = 1.5*max_t
# 	offset_per_station = 1.5*n_batch*offset_per_batch

# 	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
# 	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
# 	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
# 	n_arvs = arrivals_select.shape[0]
# 	# arrivals_select = np.concatenate((arrivals_0, arrivals_select, arrivals_1), axis = 0)

# 	# Rather slow!
# 	iargsort = np.argsort(arrivals_select[:,0])
# 	arrivals_select = arrivals_select[iargsort]
# 	phase_observed_select = phase_observed_select[iargsort]

# 	iwhere_p = np.where(phase_observed_select == 0)[0]
# 	iwhere_s = np.where(phase_observed_select == 1)[0]
# 	n_arvs_p = len(iwhere_p)
# 	n_arvs_s = len(iwhere_s)

# 	query_time_p = Trv_subset_p[:,0] + Batch_indices*offset_per_batch + Trv_subset_p[:,1]*offset_per_station
# 	query_time_s = Trv_subset_s[:,0] + Batch_indices*offset_per_batch + Trv_subset_s[:,1]*offset_per_station

# 	kernel_sig_t = 25.0 # Can speed up by only using matches.


# 	## No phase type information
# 	ip_p = np.searchsorted(arrivals_select[:,0], query_time_p)
# 	ip_s = np.searchsorted(arrivals_select[:,0], query_time_s)

# 	ip_p_pad = ip_p.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_s_pad = ip_s.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
# 	ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
# 	ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

# 	rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)

# 	## With phase type information
# 	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
# 	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

# 	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
# 	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
# 	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

# 	rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)


# 	k_sta_edges = 10 # 10
# 	k_spc_edges = 15
# 	k_time_edges = 10 ## Make sure is same as in train_regional_GNN.py
# 	n_queries = 4500 ## 3000 ## Why only 3000?
# 	time_vec_slice = np.arange(k_time_edges)

# 	Inpts = []
# 	Masks = []
# 	Lbls = []
# 	Lbls_query = []
# 	X_fixed = []
# 	X_query = []
# 	Locs = []
# 	Trv_out = []

# 	A_sta_sta_l = []
# 	A_src_src_l = []
# 	A_prod_sta_sta_l = []
# 	A_prod_src_src_l = []
# 	A_src_in_prod_l = []
# 	A_edges_time_p_l = []
# 	A_edges_time_s_l = []
# 	A_edges_ref_l = []

# 	lp_times = []
# 	lp_stations = []
# 	lp_phases = []
# 	lp_meta = []
# 	lp_srcs = []
# 	lp_srcs_active = []

# 	print('Inputs need to be saved on the subset of Cartesian product graph, and can use the exctract adjacencies script to obtain the subset of entries on Cartesian product')

# 	# src_positions_active[lp_src_times[i]]

# 	pdb.set_trace()

# 	thresh_mask = 0.01
# 	for i in range(n_batch):
# 		# Create inputs and mask
# 		grid_select = Grid_indices[i]
# 		ind_select = Sample_indices[i]
# 		sta_select = Station_indices[i]
# 		n_spc = x_grids[grid_select].shape[0]
# 		n_sta_slice = len(sta_select)


# 		## Based on sta_select, find the subset of station-source pairs needed for each of the `subgraphs' of Cartesian product graph
# 		[A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] = extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs[sta_select], np.arange(len(sta_select)), x_grids[grid_select], ftrns1, ftrns2, max_deg_offset = 5.0, k_nearest_pairs = 30)
# 		ind_sta_subset = sta_select[A_src_in_sta[0]]



# 		inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
# 		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
# 		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
# 		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
# 		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))


# 		## Select the subset of entries relevant for the input (it's only a subset of pairs of sources and stations...)
# 		## The if direct indexing not possible, might need to reshape into a full (all sources and stations, and features
# 		## tensor, and then selecting the subset of all pairs from this list).
# 		inpt_reshape = inpt.reshape(-1,4) ## This should be indices of the reshaped inpt needed for the subset of pairs of Cartesian product graph
# 		needed_indices = A_src_in_sta[1]*len(sta_select) + sta_select[A_src_in_sta[0]] # A_src_in_sta[0] # ind_sta_subset
# 		## inpt[5,20,2] == inpt_reshape[1425*5 + 20,2]

# 		## Does trv_out also need to be sub-selected
# 		trv_out = x_grids_trv[grid_select][:,sta_select,:] ## Subsetting, into sliced indices.
# 		Inpts.append(inpt[:,sta_select,:]) # sub-select, subset of stations.
# 		Masks.append(1.0*(inpt[:,sta_select,:] > thresh_mask))
# 		Trv_out.append(trv_out)
# 		Locs.append(locs[sta_select])
# 		X_fixed.append(x_grids[grid_select])

# 		## Assemble pick datasets
# 		perm_vec = -1*np.ones(n_sta)
# 		perm_vec[sta_select] = np.arange(len(sta_select))
# 		meta = arrivals[lp[i],:]
# 		phase_vals = phase_observed[lp[i]]
# 		times = meta[:,0]
# 		indices = perm_vec[meta[:,1].astype('int')]
# 		ineed = np.where(indices > -1)[0]
# 		times = times[ineed] ## Overwrite, now. Double check if this is ok.
# 		indices = indices[ineed]
# 		phase_vals = phase_vals[ineed]
# 		meta = meta[ineed]

# 		active_sources_per_slice = np.array(lp_src_times_all[i])[np.array(active_sources_per_slice_l[i])]
# 		ind_inside = np.where(inside_interior[active_sources_per_slice.astype('int')] > 0)[0]
# 		active_sources_per_slice = active_sources_per_slice[ind_inside]

# 		# Find pick specific, sources
# 		## Comment out
# 		ind_src_unique = np.unique(meta[meta[:,2] > -1.0,2]).astype('int') # ignore -1.0 entries.

# 		if len(ind_src_unique) > 0:
# 			ind_src_unique = np.sort(np.array(list(set(ind_src_unique).intersection(active_sources_per_slice)))).astype('int')

# 		src_subset = np.concatenate((src_positions[ind_src_unique], src_times[ind_src_unique].reshape(-1,1) - time_samples[i]), axis = 1)
# 		if len(ind_src_unique) > 0:
# 			perm_vec_meta = np.arange(ind_src_unique.max() + 1)
# 			perm_vec_meta[ind_src_unique] = np.arange(len(ind_src_unique))
# 			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)
# 			# ifind = np.where(meta[:,2] > -1.0)[0] ## Need to find picks with a source index inside the active_sources_per_slice
# 			ifind = np.where([meta[j,2] in ind_src_unique for j in range(meta.shape[0])])[0]
# 			meta[ifind,-1] = perm_vec_meta[meta[ifind,2].astype('int')] # save pointer to active source, for these picks (in new, local index, of subset of sources)
# 		else:
# 			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)

# 		# Do these really need to be on cuda?
# 		lex_sort = np.lexsort((times, indices)) ## Make sure lexsort doesn't cause any problems
# 		lp_times.append(times[lex_sort] - time_samples[i])
# 		lp_stations.append(indices[lex_sort])
# 		lp_phases.append(phase_vals[lex_sort])
# 		lp_meta.append(meta[lex_sort]) # final index of meta points into 
# 		lp_srcs.append(src_subset)

# 		# A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[sta_select])/1000.0).cuda(), torch.Tensor(ftrns1(locs[sta_select])/1000.0).cuda(), k = k_sta_edges + 1).flip(0).contiguous())[0]
# 		# A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).cuda(), torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).cuda(), k = k_spc_edges + 1).flip(0).contiguous())[0]
# 		# ## Cross-product graph is: source node x station node. Order as, for each source node, all station nodes.

# 		# # Cross-product graph, nodes connected by: same source node, connected stations
# 		# A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).cuda()).contiguous()
# 		# A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).cuda()).contiguous()	
# 		# # ind_spc = torch.floor(A_prod_src_src/torch.Tensor([n_sta]).cuda()) # Note: using scalar division causes issue at exact n_sta == n_sta indices.

# 		# # For each unique spatial point, sum in all edges.
# 		# # A_spc_in_prod = torch.cat(((torch.arange(n_sta).repeat(n_spc) + n_sta*torch.arange(n_spc).repeat_interleave(n_sta)).view(1,-1), torch.arange(n_spc)
# 		# A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1)), dim = 0).cuda().contiguous()

# 		## Sub-selecting from the time-arrays, is easy, since the time-arrays are indexed by station (triplet indexing; )
# 		len_dt = len(x_grids_trv_refs[grid_select])

# 		### Note: A_edges_time_p needs to be augmented: by removing stations, we need to re-label indices for subsequent nodes,
# 		### To the "correct" number of stations. Since, not n_sta shows up in definition of edges. "assemble_pointers.."
# 		A_edges_time_p = x_grids_trv_pointers_p[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
# 		A_edges_time_s = x_grids_trv_pointers_s[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
# 		## Need to convert these edges again. Convention is:
# 		## subtract i (station index absolute list), divide by n_sta, mutiply by N stations, plus ith station (in permutted indices)
# 		# shape is len_dt*k_time_edges*len(sta_select)
# 		# one_vec = np.repeat(sta_select*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere
# 		# A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
# 		# A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
# 		# # print('permute indices 1')
# 		# assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
# 		# assert(A_edges_time_s.max() < n_spc*n_sta_slice)

# 		A_sta_sta_l.append(A_sta_sta.cpu().detach().numpy())
# 		A_src_src_l.append(A_src_src.cpu().detach().numpy())
# 		A_prod_sta_sta_l.append(A_prod_sta_sta.cpu().detach().numpy())
# 		A_prod_src_src_l.append(A_prod_src_src.cpu().detach().numpy())
# 		A_src_in_prod_l.append(A_src_in_prod.cpu().detach().numpy())
# 		A_edges_time_p_l.append(A_edges_time_p)
# 		A_edges_time_s_l.append(A_edges_time_s)
# 		A_edges_ref_l.append(x_grids_trv_refs[grid_select])

# 		x_query = np.random.rand(n_queries, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.

# 		if len(lp_srcs[-1]) > 0:
# 			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]

# 		# t_query = (np.random.rand(n_queries)*10.0 - 5.0) + time_samples[i]
# 		t_slice = np.arange(-5.0, 5.0 + 1.0, 1.0) ## Window, over which to extract labels.

# 		if len(active_sources_per_slice) == 0:
# 			lbls_grid = np.zeros((x_grids[grid_select].shape[0],len(t_slice)))
# 			lbls_query = np.zeros((n_queries,len(t_slice)))
# 		else:
# 			active_sources_per_slice = active_sources_per_slice.astype('int')

# 			lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
# 			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

# 		# Append, either if there is data or not
# 		X_query.append(x_query)
# 		Lbls.append(lbls_grid)
# 		Lbls_query.append(lbls_query)

# 	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitude.reshape(-1,1)), axis = 1)
# 	data = [arrivals, srcs, active_sources]		

# 	if verbose == True:
# 		print('batch gen time took %0.2f'%(time.time() - st))

# 	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

# def generate_synthetic_data_simple_backup(trv, locs, x_grids, x_grids_trv, x_query, A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, ftrns1, ftrns2, n_batch = 1, src_density = None, verbose = True):

# 	if verbose == True:
# 		st = time.time()

# 	## Can embed the three nearest arrivals in time, not just one, to handle 
# 	## large numbers of picks in a short window
# 	## and the large kernel embedding size

# 	scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
# 	offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
# 	locs_cuda = torch.Tensor(locs).to(device)
# 	x_grid_cuda = torch.Tensor(x_grid).to(device)
# 	n_sta = locs.shape[0]

# 	## Set up time window
# 	max_t = 500.0
# 	t_window = 5.0*max_t
# 	min_sta_cnt = 4

# 	src_t_kernel = 15.0
# 	src_x_kernel = 150e3

# 	## Choose number of events between 0 and 10 within a max time window of 5*max_t
# 	## Up-weight the probability of choosing 0 events
# 	## max_t ~= 500 s, to be consistent with training data
# 	ratio_noise = 2
# 	prob_events = np.ones(15)
# 	prob_events[0] = ratio_noise
# 	prob_events = prob_events/prob_events.sum()

# 	n_events = np.random.choice(np.arange(len(prob_events)), p = prob_events)

# 	if src_density is None:
# 		pos_srcs = np.random.rand(n_events,3)*scale_x + offset_x
# 		# pos_srcs = np.random.rand(n_events,3)*(np.array([20.0, 20.0, 100e3]).reshape(1,-1)) + np.array([-10.0, -10.0, -100e3]).reshape(1,-1) + locs[np.random.choice(n_sta, size = n_events),:].reshape(-1,3)
# 	else:
# 		pos_srcs = src_density.sample(n_events)
# 		pos_srcs = np.concatenate((pos_srcs.reshape(-1,2), np.random.rand(n_events,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)

# 	src_origin = np.sort(np.random.rand(n_events)*(t_window + max_t) - max_t)
# 	print('Number events: %d'%n_events)
# 	print('Max t: %f'%max_t)

# 	## Create moveouts of all source-reciever phases, but `optionally' delete all arrivals > max_t
# 	## Randomly delete fraction of arrivals
# 	## Randomly add fraction of false arrivals
# 	## Add random error to theoretical arrival times

# 	# avg_pick_rate_per_time_range = [1/150.0, 1/120.0] ## Ratio of false picks
# 	avg_pick_rate_per_time_range = [2*6.25e-5, 2*0.001] ## Ratio of false picks
# 	n_rand_picks_per_station = (t_window + max_t)*(np.random.rand(locs.shape[0])*(avg_pick_rate_per_time_range[1] - avg_pick_rate_per_time_range[0]) + avg_pick_rate_per_time_range[0])

# 	if n_events == 0:

# 		arrivals_slice = np.zeros((0,4))

# 	else: ## n_events > 0

# 		use_uniform_dist = True
# 		del_ratio = [0.1, 0.65] ## Ratio of randomly deleted picks
# 		del_vals = np.random.rand(n_events)*(del_ratio[1] - del_ratio[0]) + del_ratio[0]
# 		dist_range = [100e3, 1500e3] ## Ratio of distance cutoff
# 		rnd_space_thresh = 50e3

# 		if use_uniform_dist == True:
# 			dist_vals = np.random.rand(n_events)*(dist_range[1] - dist_range[0]) + dist_range[0]
# 		else:
# 			dist_vals = beta(2,5).rvs(size = n_events)*(dist_range[1] - dist_range[0]) + dist_range[0]

# 		percent_trv_time_error = [0.02, 0.05]
# 		scale_trv_time_error = np.random.rand(n_events, 1, 1)*(percent_trv_time_error[1] - percent_trv_time_error[0]) + percent_trv_time_error[0]

# 		# for i in range(n_events):
# 		## First column, time, second column, station index, third column, amplitude, fourth column, phase type
# 		trv_out = trv(locs_cuda, torch.Tensor(pos_srcs).to(device)).cpu().detach().numpy()
# 		trv_out_noise = trv_out + src_origin.reshape(-1,1,1) + np.random.laplace(size = trv_out.shape)*np.abs(trv_out)*scale_trv_time_error
# 		true_moveouts = np.copy(trv_out) + src_origin.reshape(-1,1,1)

# 		## Use max travel time as proxy for distance, and delete arrival past max distance (since unlikely to have accurate arrival time)
# 		## and/or add higher noise.
# 		isrc, ista = np.where(trv_out.max(2) < max_t)

# 		arv_Pg = trv_out_noise[isrc,ista,0]
# 		arv_Pn = trv_out_noise[isrc,ista,1]
# 		arv_Sn = trv_out_noise[isrc,ista,2]
# 		arv_Lg = trv_out_noise[isrc,ista,3]
# 		ind = np.tile(ista,4)
# 		phase_type = np.arange(4).repeat(len(isrc))
# 		event_ind = np.tile(isrc,4)
# 		arrivals_slice_1 = np.concatenate((arv_Pg, arv_Pn, arv_Sn, arv_Lg), axis = 0).reshape(-1,1)
# 		# arrivals_slice = np.concatenate((arrivals_slice_1, ind[:,None], event_ind[:,None], phase_type[:,None]), axis = 1) ## Why does this not work?
# 		arrivals_slice = np.concatenate((arrivals_slice_1, ind.reshape(-1,1), event_ind.reshape(-1,1), phase_type.reshape(-1,1)), axis = 1)

# 		## Per event, delete all events with source-reciever pairs > dist_vals + noise per pick and per events
# 		pd_distances = np.linalg.norm(ftrns1(locs[arrivals_slice[:,1].astype('int')].reshape(-1,3)) - ftrns1(pos_srcs[arrivals_slice[:,2].astype('int')].reshape(-1,3)), axis = 1)
# 		idel = np.where((pd_distances + rnd_space_thresh*np.random.randn(len(pd_distances)))/dist_vals[arrivals_slice[:,2].astype('int')] > 1.0)[0]
# 		# print(pd_distances.shape)
# 		# print(arrivals_slice.shape)
# 		# print(idel.shape)
# 		# print(pd_distances)
# 		# print(dist_vals[arrivals_slice[:,2].astype('int')])

# 		arrivals_slice = np.delete(arrivals_slice, idel, axis = 0)

# 		# pdb.set_trace()

# 		## Per event, delete del_vals ratio of all true picks
# 		tree_event_inds = cKDTree(arrivals_slice[:,2][:,None])
# 		lp_event_inds = tree_event_inds.query_ball_point(np.arange(n_events)[:,None], r = 0)
# 		lp_event_cnts = np.array([len(lp_event_inds[j]) for j in range(len(lp_event_inds))])
# 		lp_event_ratios = lp_event_cnts*del_vals
# 		prob_del = lp_event_ratios[arrivals_slice[:,2].astype('int')]

# 		if prob_del.sum() > 0:

# 			prob_del = prob_del/prob_del.sum()
# 			# print(prob_del.shape)
# 			# print(prob_del)
# 			# print(arrivals_slice.shape[0])

# 			sample_random_del = np.unique(np.random.choice(np.arange(arrivals_slice.shape[0]), p = prob_del, size = int(lp_event_ratios.sum()))) # Not using replace = False
# 			arrivals_slice = np.delete(arrivals_slice, sample_random_del, axis = 0)

# 	## Add random rate of false picks per station
# 	arrivals_false = np.random.rand(int(n_rand_picks_per_station.sum()))*(t_window + max_t) - max_t
# 	arrivals_false_ind = np.random.choice(locs.shape[0], p = n_rand_picks_per_station/n_rand_picks_per_station.sum(), size = int(n_rand_picks_per_station.sum()))
# 	arrivals_false = np.concatenate((arrivals_false[:,None], arrivals_false_ind[:,None], -1*np.ones((len(arrivals_false),2))), axis = 1)

# 	arrivals = np.concatenate((arrivals_slice, arrivals_false), axis = 0)

# 	## Check which sources still active (or wait until the subset of stations is chosen..)
# 	## Alteratively, and more simply, just use all stations available, and sub-select inputs of stations
# 	## the station indices returned would be directly in terms of the inputs locs

# 	# if n_events > 0:
# 	# 	tree_event_inds = cKDTree(arrivals_slice[:,2][:,None])
# 	# 	lp_event_inds = tree_event_inds.query_ball_point(np.arange(n_events)[:,None], r = 0)
# 	# 	lp_event_cnts = np.array([len(lp_event_inds[j]) for j in range(len(lp_event_inds))])
# 	# 	#iactive_srcs = np.where(lp_event_cnts >= min_pick_cnt)[0]

# 	# else:
# 	# 	iactive_srcs = np.array([])

# 	## Make inputs (on Cartesian product graph - can base this on
# 	## the input Cartesian product graph parameters, which can be built before,
# 	## based on a subset of stations. E.g., do the sub-selecting of which active
# 	## stations before calling this script).

# 	## Can use the 1d embedding strategy used in regular generate_synthetic_data and the pre-computed x_grids_trv to
# 	## measure the nearest matches for all src-reciever pairs. Can also only consider moveouts < max_t, for consistency.

# 	iactive_arvs = np.where(arrivals[:,2] > -1)[0]
# 	iunique_srcs = np.unique(arrivals[iactive_arvs,2]).astype('int')
# 	cnt_src_picks = []
# 	for i in range(len(iunique_srcs)):
# 		cnt_src_picks.append(len(np.unique(arrivals[arrivals[:,2] == iunique_srcs[i],1])) >= min_sta_cnt)
# 	cnt_src_picks = np.array(cnt_src_picks)
# 	active_sources = iunique_srcs[np.where(cnt_src_picks > 0)[0]]
# 	src_times_active = src_origin[active_sources]

# 	# n_batch = 1
# 	offset_per_batch = 1.5*max_t
# 	offset_per_station = 1.5*n_batch*offset_per_batch

# 	use_uniform_sampling = False
# 	if use_uniform_sampling == True:

# 		time_samples = np.random.rand(n_batch)*max_t

# 	else:

# 		time_samples = np.random.rand(n_batch)*max_t
# 		l_src_times_active = len(src_times_active)
# 		if len(src_times_active) > 1:
# 			for j in range(n_batch):
# 				if np.random.rand() > 0.25: # 30% of samples, re-focus time. # 0.7
# 					time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (1.5/3.0)*src_t_kernel*np.random.laplace()

# 	time_samples = np.sort(time_samples)

# 	print('Need to implement not uniform origin time sampling')

# 	t_win = 10.0
# 	tree = cKDTree(arrivals[:,0][:,None])
# 	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = 2.0*t_win + max_t/2.0) 
# 	# print(time.time() - st)

# 	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
# 	if len(lp_concat) == 0:
# 		lp_concat = np.array([0]) # So it doesnt fail?
# 	arrivals_select = arrivals[lp_concat]

# 	# phase_observed_select = phase_observed[lp_concat]
# 	# tree_select = cKDTree(arrivals_select[:,0:2]*scale_vec) ## Commenting out, as not used
# 	# print(time.time() - st)

# 	Trv_subset_Pg = []
# 	Trv_subset_Pn = []
# 	Trv_subset_Sn = []
# 	Trv_subset_Lg = []
# 	Station_indices = []
# 	Grid_indices = []
# 	Batch_indices = []
# 	Sample_indices = []
# 	sc = 0

# 	# if (fixed_subnetworks is not None):
# 	# 	fixed_subnetworks_flag = 1
# 	# else:
# 	# 	fixed_subnetworks_flag = 0		

# 	# active_sources_per_slice_l = []
# 	# src_positions_active_per_slice_l = []
# 	# src_times_active_per_slice_l = []

# 	for i in range(n_batch):
# 		## Can also select sub-networks that are real, realizations.
# 		i0 = np.random.randint(0, high = len(x_grids))
# 		n_spc = x_grids[i0].shape[0]

# 		use_full_network = True
# 		if use_full_network == True:
# 			n_sta_select = n_sta
# 			ind_sta_select = np.arange(n_sta)

# 		else:

# 			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
# 				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
# 				n_sta_select = len(fixed_subnetworks[isub_network])
# 				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.
			
# 			else:
# 				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
# 				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

# 		## Could potentially already sub-select src-reciever pairs here
# 		ind_pairs_repeat = np.tile(ind_sta_select, n_spc).reshape(-1,1)
# 		ind_src_pairs_repeat = np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1)
# 		Trv_subset_Pg.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , , len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Trv_subset_Pn.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Trv_subset_Sn.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,2].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
# 		Trv_subset_Lg.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,3].reshape(-1,1), ind_pairs_repeat, ind_src_pairs_repeat), axis = 1)) # , np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication

# 		Station_indices.append(ind_sta_select) # record subsets used
# 		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
# 		Grid_indices.append(i0)
# 		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
# 		sc += len(Sample_indices[-1])

# 		# active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

# 		# active_sources_per_slice = np.where(n_unique_station_counts_per_slice >= min_sta_arrival)[0] # subset of sources
# 		# active_sources_per_slice_l.append(active_sources_per_slice)

# 	Trv_subset_Pg = np.vstack(Trv_subset_Pg)
# 	Trv_subset_Pn = np.vstack(Trv_subset_Pn)
# 	Trv_subset_Sn = np.vstack(Trv_subset_Sn)
# 	Trv_subset_Lg = np.vstack(Trv_subset_Lg)
# 	Batch_indices = np.hstack(Batch_indices)
# 	# print(time.time() - st)

# 	offset_per_batch = 1.5*max_t
# 	offset_per_station = 1.5*n_batch*offset_per_batch

# 	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
# 	one_vec = np.concatenate((np.ones(1), np.zeros(3)), axis = 0).reshape(1,-1)
# 	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
# 	n_arvs = arrivals_select.shape[0]
# 	# arrivals_select = np.concatenate((arrivals_0, arrivals_select, arrivals_1), axis = 0)

# 	# Rather slow!
# 	iargsort = np.argsort(arrivals_select[:,0])
# 	arrivals_select = arrivals_select[iargsort]
# 	# phase_observed_select = phase_observed_select[iargsort]

# 	# iwhere_p = np.where(phase_observed_select == 0)[0]
# 	# iwhere_s = np.where(phase_observed_select == 1)[0]
# 	# n_arvs_p = len(iwhere_p)
# 	# n_arvs_s = len(iwhere_s)

# 	mask_moveout_Pg = (Trv_subset_Pg[:,0] > max_t)
# 	mask_moveout_Pn = (Trv_subset_Pn[:,0] > max_t)
# 	mask_moveout_Sn = (Trv_subset_Sn[:,0] > max_t)
# 	mask_moveout_Lg = (Trv_subset_Lg[:,0] > max_t)

# 	query_time_Pg = Trv_subset_Pg[:,0] + Batch_indices*offset_per_batch + Trv_subset_Pg[:,1]*offset_per_station
# 	query_time_Pn = Trv_subset_Pn[:,0] + Batch_indices*offset_per_batch + Trv_subset_Pn[:,1]*offset_per_station
# 	query_time_Sn = Trv_subset_Sn[:,0] + Batch_indices*offset_per_batch + Trv_subset_Sn[:,1]*offset_per_station
# 	query_time_Lg = Trv_subset_Lg[:,0] + Batch_indices*offset_per_batch + Trv_subset_Lg[:,1]*offset_per_station

# 	## No phase type information
# 	ip_Pg = np.searchsorted(arrivals_select[:,0], query_time_Pg)
# 	ip_Pn = np.searchsorted(arrivals_select[:,0], query_time_Pn)
# 	ip_Sn = np.searchsorted(arrivals_select[:,0], query_time_Sn)
# 	ip_Lg = np.searchsorted(arrivals_select[:,0], query_time_Lg)

# 	## Wouldn't have to do these min and max bounds if arrivals was appended 
# 	## to have 2 extra picks at very large negative and positive times
# 	ip_Pg_pad = np.minimum(np.maximum(ip_Pg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_Pn_pad = np.minimum(np.maximum(ip_Pn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_Sn_pad = np.minimum(np.maximum(ip_Sn.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	ip_Lg_pad = np.minimum(np.maximum(ip_Lg.reshape(-1,1) + np.array([-1,0]).reshape(1,-1), 0), n_arvs - 1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
# 	# ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
# 	# ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

# 	kernel_sig_t = 15.0 # Can speed up by only using matches.
# 	use_nearest_arrival = True
# 	if use_nearest_arrival == True:
# 		rel_t_Pg = kernel_sig_t*5*mask_moveout_Pg + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Pn = kernel_sig_t*5*mask_moveout_Pn + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Sn = kernel_sig_t*5*mask_moveout_Sn + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Lg = kernel_sig_t*5*mask_moveout_Lg + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 	else: ## Use 3 nearest arrivals
# 		## Add mask for large source-station offsets, in terms of relative time >> kernel_sig_t
# 		rel_t_Pg = kernel_sig_t*5*np.expand_dims(mask_moveout_Pg, axis = 2) + np.abs(query_time_Pg[:, np.newaxis] - arrivals_select[ip_Pg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Pn = kernel_sig_t*5*np.expand_dims(mask_moveout_Pn, axis = 2) + np.abs(query_time_Pn[:, np.newaxis] - arrivals_select[ip_Pn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Sn = kernel_sig_t*5*np.expand_dims(mask_moveout_Sn, axis = 2) + np.abs(query_time_Sn[:, np.newaxis] - arrivals_select[ip_Sn_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
# 		rel_t_Lg = kernel_sig_t*5*np.expand_dims(mask_moveout_Lg, axis = 2) + np.abs(query_time_Lg[:, np.newaxis] - arrivals_select[ip_Lg_pad, 0]) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.


# 	# pdb.set_trace()

# 	## Based on sta_select, find the subset of station-source pairs needed for each of the `subgraphs' of Cartesian product graph
# 	# [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] = extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs[sta_select], np.arange(len(sta_select)), x_grids[grid_select], ftrns1, ftrns2, max_deg_offset = 5.0, k_nearest_pairs = 30)

# 	## Should also grab inputs from different possible source depths
# 	## In general, input feature dimension will be higher

# 	Inpts = []
# 	Masks = []
# 	Lbls = []
# 	X_query = []

# 	thresh_mask = 0.01
# 	for i in range(n_batch):
# 		# Create inputs and mask
# 		grid_select = Grid_indices[i]
# 		ind_select = Sample_indices[i]
# 		sta_select = Station_indices[i]
# 		n_spc = x_grids[grid_select].shape[0]
# 		n_sta_slice = len(sta_select)

# 		# Rather than fully populate inpt for all x_grids and locs, why not only select subset inside the subset of Cartesian product graph
# 		## E.g., can find all pairswise matches of full set to an element in subset using cKDTree, then can extract these pairs,
# 		## and can ensure the ordering is the in the order of all used stations for each source node (e.g., the same ordering given
# 		## by A_src_in_prod and the nodes of A_prod.
# 		ind_sta_subset = sta_select[A_src_in_sta[0]]

# 		## In general, we only end up using ~2% of these pairs.
# 		## Ideally, we could only create a fraction of these entries to begin with.

# 		feature_Pg = np.exp(-0.5*(rel_t_Pg[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice) ## Could give sign information if prefered
# 		feature_Pn = np.exp(-0.5*(rel_t_Pn[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
# 		feature_Sn = np.exp(-0.5*(rel_t_Sn[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)
# 		feature_Lg = np.exp(-0.5*(rel_t_Lg[ind_select]**2)/(kernel_sig_t**2)).reshape(n_spc, n_sta_slice)

# 		## Now need to only grab the necessary pairs, and reshape

# 		feature_Pg_subset = feature_Pg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
# 		feature_Pn_subset = feature_Pn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
# 		feature_Sn_subset = feature_Sn[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)
# 		feature_Lg_subset = feature_Lg[A_src_in_sta[1], A_src_in_sta[0]].reshape(-1,1)

# 		Inpts.append(np.concatenate((feature_Pg_subset, feature_Pn_subset, feature_Sn_subset, feature_Lg_subset), axis = 1))
# 		Masks.append(Inpts[-1] > thresh_mask)


# 		# t_query = (np.random.rand(n_queries)*10.0 - 5.0) + time_samples[i]
# 		t_slice = np.arange(-5.0, 5.0 + 1.0, 1.0) ## Window, over which to extract labels.

# 		n_extra_queries = 3000
# 		if len(active_sources) > 0:
# 			x_query_focused = np.random.rand(n_extra_queries, 3)*np.array([10.0, 10.0, 0.0]).reshape(1,-1) + np.array([-5.0, -5.0, 0.0]).reshape(1,-1) + pos_srcs[np.random.choice(active_sources, size = n_extra_queries)]
# 			x_query_focused[:,2] = np.random.rand(n_extra_queries)*(x_query[:,2].max() - x_query[:,2].min()) + x_query[:,2].min()
# 			x_query_focused = np.concatenate((x_query, x_query_focused), axis = 0)

# 		else:
# 			x_query_focused = np.concatenate((x_query, np.random.rand(n_extra_queries, 3)*(x_query.max(0, keepdims = True) - x_query.min(0, keepdims = True)) + x_query.min(0, keepdims = True)), axis = 0)


# 		X_query.append(x_query_focused)

# 		if len(active_sources) == 0:
# 			lbls_grid = np.zeros((x_grids[grid_select].shape[0], len(t_slice)))
# 			lbls_query = np.zeros((x_query_focused.shape[0], len(t_slice)))
# 		else:

# 			ignore_depth = True
# 			if ignore_depth == True:
# 				lbl_vec = np.array([1.0, 1.0, 0.0]).reshape(1,-1)

# 			else:
# 				lbl_vec = np.array([1.0, 1.0, 1.0]).reshape(1,-1)


# 			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query_focused), axis = 1) - np.expand_dims(ftrns1(pos_srcs[active_sources]*lbl_vec), axis = 0))**2)/(src_x_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_origin[active_sources].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

# 		Lbls.append(lbls_query)

# 		## Currently this should be shaped as for each source node, all station nodes

# 		# inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
# 		# inpt[Trv_subset_Pg[ind_select,2].astype('int'), Trv_subset_Pg[ind_select,1].astype('int'), 0] = feature_Pg
# 		# inpt[Trv_subset_Pn[ind_select,2].astype('int'), Trv_subset_Pn[ind_select,1].astype('int'), 1] = feature_Pn
# 		# inpt[Trv_subset_Sn[ind_select,2].astype('int'), Trv_subset_Sn[ind_select,1].astype('int'), 2] = feature_Sn
# 		# inpt[Trv_subset_Lg[ind_select,2].astype('int'), Trv_subset_Lg[ind_select,1].astype('int'), 3] = feature_Lg

# 		## Make labels



# 	## Again, need to only consider pairs that are within source-reciever distance (or time, of max_t)

# 	## Make labels

# 	if len(active_sources) == 0:
# 		true_moveouts = None

# 	data = [arrivals, np.concatenate((pos_srcs, src_origin.reshape(-1,1)), axis = 1), active_sources, true_moveouts, time_samples]

# 	return [Inpts, Masks, X_query, Lbls], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)


def assemble_time_pointers_for_stations(trv_out, dt = 1.0, k = 10, eps_distance = 30, win = 10.0, tbuffer = 10.0, knn_graphs = True):

	print('Issue: this is implemented assuming the full Cartesian product graph')

	n_temp, n_sta = trv_out.shape[0:2]
	dt_partition = np.arange(-win, win + trv_out.max() + dt + tbuffer, dt)

	if knn_graphs == True:

		# For each station, for each time step, find nearest k positions in Lg.

		# Make time access -win : max_moveout + win. This way, any time
		# in this window is perhaps

		# For each station, for each time step, find pointers to nearest k sources (indexed in Lg)
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

	else:

		print('Implement epsilon-graphs for arrival time edges instead. Or, just make these dynamically during predictions/training')
		print('Need to make sure the down-stream PhaseAssociationModules dont assume exactly k edges for each arrival for this case')

	# def return_indices(sta_ind, dt_partitiontime):

	return edges_p, edges_s, dt_partition

class TravelTimes(nn.Module):

	def __init__(self, ftrns1, ftrns2, n_phases = 1, device = 'cuda'):
		super(TravelTimes, self).__init__()

		## Components:
		# (1). An average travel time, directly calculated from bulk statistics,
		# for any source-reciever (in projected domain) offset positions (or great
		# circle arc distances)

		# (2). Add a contribution based on the relative offset (this can capture the spherical Earth
		# approximation).

		# (3). An absolute position prediction based on the absolute reciever and source coordinate
		# positions (in projected domain).

		## Relative offset prediction [2]
		self.fc1 = nn.Sequential(nn.Linear(3, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Absolute position prediction [3]
		self.fc2 = nn.Sequential(nn.Linear(6, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Relative offset prediction [2]
		self.fc3 = nn.Sequential(nn.Linear(3, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Absolute position prediction [3]
		self.fc4 = nn.Sequential(nn.Linear(6, 80), nn.ReLU(), nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, n_phases))

		## Projection functions
		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2
		self.scale = torch.Tensor([5e6]).to(device) ## Might want to scale inputs before converting to Tensor
		self.tscale = torch.Tensor([200.0]).to(device)
		self.device = device
		# self.Tp_average

	def forward(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return self.tscale*(self.fc1(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + self.fc2(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return self.tscale*(self.fc1(sta_repeat - src_repeat) + self.fc2(torch.cat((sta_repeat, src_repeat), dim = 1))).reshape(len(src), len(sta), 1)

	def forward_train(self, sta, src, p = 0.5, method = 'pairs'):

		## In training mode, drop p percent of the `local' corrections, so that the default source-reciever distance
		## model is trained to be accurate enough in a stand-alone fashion. This way, travel time estimates can still
		## be obtained for non-sampled source regions, which can help with the global application and simulated data
		## in non-sampeled regions.

		if method == 'direct':

			rand_mask = (torch.rand(sta.shape[0],1).to(self.device) > p).float()

			return self.tscale*(self.fc1(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + rand_mask*self.fc2(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			rand_mask = (torch.rand(sta_repeat.shape[0],1).to(self.device) > p).float()

			return self.tscale*(self.fc1(sta_repeat - src_repeat) + rand_mask*self.fc2(torch.cat((sta_repeat, src_repeat), dim = 1))).reshape(len(src), len(sta), 1)

	def forward_relative(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return self.tscale*(self.fc1(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return self.tscale*(self.fc1(sta_repeat - src_repeat)).reshape(len(src), len(sta), 1)

	def forward_mask(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return torch.sigmoid(self.fc3(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + self.fc4(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return torch.sigmoid(self.fc3(sta_repeat - src_repeat) + self.fc4(torch.cat((sta_repeat, src_repeat), dim = 1))).reshape(len(src), len(sta), 1)

	def forward_mask_train(self, sta, src, p = 0.5, method = 'pairs'):

		## In training mode, drop p percent of the `local' corrections, so that the default source-reciever distance
		## model is trained to be accurate enough in a stand-alone fashion. This way, travel time estimates can still
		## be obtained for non-sampled source regions, which can help with the global application and simulated data
		## in non-sampeled regions.

		if method == 'direct':

			rand_mask = (torch.rand(sta.shape[0],1).to(self.device) > p).float()

			return torch.sigmoid(self.fc3(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale) + rand_mask*self.fc4(torch.cat((self.ftrns1(sta)/self.scale, self.ftrns1(src)/self.scale), dim = 1)))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			rand_mask = (torch.rand(sta_repeat.shape[0],1).to(self.device) > p).float()

			return torch.sigmoid(self.fc3(sta_repeat - src_repeat) + rand_mask*self.fc4(torch.cat((sta_repeat, src_repeat), dim = 1))).reshape(len(src), len(sta), 1)

	def forward_mask_relative(self, sta, src, method = 'pairs'):

		if method == 'direct':

			return torch.sigmoid(self.fc3(self.ftrns1(sta)/self.scale - self.ftrns1(src)/self.scale))

		elif method == 'pairs':

			## First, create all pairs of srcs and recievers

			src_repeat = self.ftrns1(src).repeat_interleave(len(sta), dim = 0)/self.scale
			sta_repeat = self.ftrns1(sta).repeat(len(src), 1)/self.scale

			return torch.sigmoid(self.fc3(sta_repeat - src_repeat)).reshape(len(src), len(sta), 1)

def apply_travel_times(trv_l):

	def trv_times(sta, src, method = 'direct forward'):

		if method == 'direct forward':

			trv_out1 = trv_l[0](sta, src, method = 'direct')
			trv_out2 = trv_l[1](sta, src, method = 'direct')
			trv_out3 = trv_l[2](sta, src, method = 'direct')
			trv_out4 = trv_l[3](sta, src, method = 'direct')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 1)

		elif method == 'pairs forward':

			trv_out1 = trv_l[0](sta, src, method = 'pairs')
			trv_out2 = trv_l[1](sta, src, method = 'pairs')
			trv_out3 = trv_l[2](sta, src, method = 'pairs')
			trv_out4 = trv_l[3](sta, src, method = 'pairs')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 2)

		if method == 'direct relative':

			trv_out1 = trv_l[0].forward_relative(sta, src, method = 'direct')
			trv_out2 = trv_l[1].forward_relative(sta, src, method = 'direct')
			trv_out3 = trv_l[2].forward_relative(sta, src, method = 'direct')
			trv_out4 = trv_l[3].forward_relative(sta, src, method = 'direct')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 1)

		elif method == 'pairs relative':

			trv_out1 = trv_l[0].forward_relative(sta, src, method = 'pairs')
			trv_out2 = trv_l[1].forward_relative(sta, src, method = 'pairs')
			trv_out3 = trv_l[2].forward_relative(sta, src, method = 'pairs')
			trv_out4 = trv_l[3].forward_relative(sta, src, method = 'pairs')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 2)

		if method == 'direct mask forward':

			trv_out1 = trv_l[0].forward_mask(sta, src, method = 'direct')
			trv_out2 = trv_l[1].forward_mask(sta, src, method = 'direct')
			trv_out3 = trv_l[2].forward_mask(sta, src, method = 'direct')
			trv_out4 = trv_l[3].forward_mask(sta, src, method = 'direct')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 1)

		elif method == 'pairs mask forward':

			trv_out1 = trv_l[0].forward_mask(sta, src, method = 'pairs')
			trv_out2 = trv_l[1].forward_mask(sta, src, method = 'pairs')
			trv_out3 = trv_l[2].forward_mask(sta, src, method = 'pairs')
			trv_out4 = trv_l[3].forward_mask(sta, src, method = 'pairs')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 2)

		if method == 'direct mask relative':

			trv_out1 = trv_l[0].forward_mask_relative(sta, src, method = 'direct')
			trv_out2 = trv_l[1].forward_mask_relative(sta, src, method = 'direct')
			trv_out3 = trv_l[2].forward_mask_relative(sta, src, method = 'direct')
			trv_out4 = trv_l[3].forward_mask_relative(sta, src, method = 'direct')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 1)

		elif method == 'pairs mask relative':

			trv_out1 = trv_l[0].forward_mask_relative(sta, src, method = 'pairs')
			trv_out2 = trv_l[1].forward_mask_relative(sta, src, method = 'pairs')
			trv_out3 = trv_l[2].forward_mask_relative(sta, src, method = 'pairs')
			trv_out4 = trv_l[3].forward_mask_relative(sta, src, method = 'pairs')

			return torch.cat((trv_out1, trv_out2, trv_out3, trv_out4), dim = 2)

	return lambda y, x, method: trv_times(y, x, method = method)

def plot_results(x_query, lbls, pred, ext_write, ind, norm = True, save = True, close = True, n_ver_save = 1, zoom = True):

	fig, ax = plt.subplots(2, 1, figsize = [12,8], sharex = True, sharey = True)

	if norm == True:
		norm = Normalize(lbls[:,5].min(), lbls[:,5].max())
		ax[0].scatter(x_query[:,1], x_query[:,0], c = lbls[:,5], norm = norm)
		ax[1].scatter(x_query[:,1], x_query[:,0], c = pred[:,5], norm = norm)
	else:
		ax[0].scatter(x_query[:,1], x_query[:,0], c = lbls[:,5])
		ax[1].scatter(x_query[:,1], x_query[:,0], c = pred[:,5])	

	if save == True:
		fig.savefig(ext_write + 'Plots/example_prediction_%d_ver_%d.png'%(ind, n_ver_save), bbox_inches = 'tight', pad_inches = 0.2)	

		if zoom == True:
			i0 = np.argmax(lbls[:,5])
			ax[0].set_xlim(x_query[i0,1] - 10.0, x_query[i0,1] + 10.0)
			ax[1].set_xlim(x_query[i0,1] - 10.0, x_query[i0,1] + 10.0)
			ax[0].set_ylim(x_query[i0,0] - 10.0, x_query[i0,0] + 10.0)
			ax[1].set_ylim(x_query[i0,0] - 10.0, x_query[i0,0] + 10.0)
			fig.savefig(ext_write + 'Plots/example_prediction_zoom_%d_ver_%d.png'%(ind, n_ver_save), bbox_inches = 'tight', pad_inches = 0.2)	


	if close == True:
		plt.close('all')

	return fig, ax

class DataAggregation(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, n_dim_mask = 4):
		super(DataAggregation, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_hidden = n_hidden

		self.activate = nn.PReLU() # can extend to each channel
		self.init_trns = nn.Linear(in_channels + n_dim_mask, n_hidden)

		self.l1_t1_1 = nn.Linear(n_hidden, n_hidden)
		self.l1_t1_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)

		self.l1_t2_1 = nn.Linear(in_channels, n_hidden)
		self.l1_t2_2 = nn.Linear(2*n_hidden + n_dim_mask, n_hidden)
		self.activate11 = nn.PReLU() # can extend to each channel
		self.activate12 = nn.PReLU() # can extend to each channel
		self.activate1 = nn.PReLU() # can extend to each channel

		self.l2_t1_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t1_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)

		self.l2_t2_1 = nn.Linear(2*n_hidden, n_hidden)
		self.l2_t2_2 = nn.Linear(3*n_hidden + n_dim_mask, out_channels)
		self.activate21 = nn.PReLU() # can extend to each channel
		self.activate22 = nn.PReLU() # can extend to each channel
		self.activate2 = nn.PReLU() # can extend to each channel

	def forward(self, tr, mask, A_in_sta, A_in_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr)), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr)), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr))), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr))), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_edges = 3):
		super(BipartiteGraphOperator, self).__init__('mean') ## add
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, ndim_in)
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, A_src_in_edges, mask, n_sta, n_temp):

		N = inpt.shape[0]
		M = n_temp

		return self.activate2(self.fc2(self.propagate(A_src_in_edges.edge_index, size = (N, M), x = mask.max(1, keepdims = True)[0]*self.activate1(self.fc1(torch.cat((inpt, A_src_in_edges.x), dim = -1))))))

## Note: changing the spatial node offsets to be measured in lat-lon domain, so also changing scale_rel
class SpatialAggregation(MessagePassing): # make equivelent version with sum operations. (no need for "complex" concatenation version). Assuming concat version is worse/better?
	def __init__(self, in_channels, out_channels, scale_rel = 100.0, n_dim = 3, n_global = 5, n_hidden = 30, device = 'cuda'):
		super(SpatialAggregation, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.fc1 = nn.Linear(in_channels + n_dim + n_global, n_hidden)
		self.fc2 = nn.Linear(n_hidden + in_channels, out_channels) ## NOTE: correcting out_channels, here.
		self.fglobal = nn.Linear(in_channels, n_global)
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.scale_rel = scale_rel # .reshape(1,-1).to(device)

	def forward(self, tr, A_src, pos):

		## Use Cartesian pos inputs

		return self.activate2(self.fc2(torch.cat((tr, self.propagate(A_src, x = tr, pos = pos)), dim = -1)))

	def message(self, x_j, pos_i, pos_j):

		## Use Cartesian pos inputs

		pos_rel = (pos_i/1000.0 - pos_j/1000.0)/self.scale_rel
		# pos_rel[:,1] = torch.remainder(pos_rel[:,1], 360.0)

		# global state has activation before aggregation?
		# Interesting to ask, what if we "perturb" the global state, to make sure it isn't too reliant on it?
		return self.activate1(self.fc1(torch.cat((x_j, pos_rel, self.activate3(self.fglobal(x_j)).mean(0, keepdims = True).repeat(x_j.shape[0], 1)), dim = -1))) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

class SpatialAttention(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim, n_latent, scale_rel = 100.0, n_hidden = 30, n_heads = 5):
		super(SpatialAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.f_context = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_dim, n_heads*n_latent) # add second layer transformation.
		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.proj = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		# self.activate3 = nn.PReLU()

	def forward(self, inpts, x_query, x_context, k = 10): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		edge_index = knn(x_context/1000.0, x_query/1000.0, k = k).flip(0)
		edge_attr = (x_query[edge_index[1]]/1000.0 - x_context[edge_index[0]]/1000.0)/self.scale_rel # /scale_x

		return self.activate2(self.proj(self.propagate(edge_index, x = inpts, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1))) # mean over different heads

	def message(self, x_j, index, edge_attr):

		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

		alpha = softmax(alpha, index)

		return alpha.unsqueeze(-1)*value_embed

class TemporalAttention(MessagePassing): ## Hopefully replace this.
	def __init__(self, inpt_dim, out_channels, n_latent, scale_t = 10.0, n_hidden = 30, n_heads = 5):
		super(TemporalAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.

		self.temporal_query_1 = nn.Linear(1, n_hidden)
		self.temporal_query_2 = nn.Linear(n_hidden, n_heads*n_latent)
		self.f_context_1 = nn.Linear(inpt_dim, n_hidden) # add second layer transformation.
		self.f_context_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.f_values_1 = nn.Linear(inpt_dim, n_hidden) # add second layer transformation.
		self.f_values_2 = nn.Linear(n_hidden, n_heads*n_latent) # add second layer transformation.

		self.proj_1 = nn.Linear(n_latent, n_hidden) # can remove this layer possibly.
		self.proj_2 = nn.Linear(n_hidden, out_channels) # can remove this layer possibly.

		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_t = scale_t

		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		self.activate5 = nn.PReLU()

	def forward(self, inpts, t_query):

		context = self.f_context_2(self.activate1(self.f_context_1(inpts))).view(-1, self.n_heads, self.n_latent) # add more non-linear transform here?
		values = self.f_values_2(self.activate2(self.f_values_1(inpts))).view(-1, self.n_heads, self.n_latent) # add more non-linear transform here?
		query = self.temporal_query_2(self.activate3(self.temporal_query_1(t_query/self.scale_t))).view(-1, self.n_heads, self.n_latent) # must repeat t output for all spatial coordinates.

		return self.proj_2(self.activate5(self.proj_1(self.activate4((((context.unsqueeze(1)*query.unsqueeze(0)).sum(-1, keepdims = True)/self.scale)*values.unsqueeze(1)).mean(2))))) # linear.

# class GCN_Detection_Network_extended_fixed_adjacencies(nn.Module):
# 	def __init__(self, ftrns1, ftrns2, device = 'cpu'):
# 		super(GCN_Detection_Network_extended_fixed_adjacencies, self).__init__()
# 		# Define modules and other relavent fixed objects (scaling coefficients.)
# 		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
# 		self.DataAggregation = DataAggregation(4, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
# 		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 3).to(device) # 30, 15
# 		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
# 		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
# 		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
# 		self.SpatialDirect = SpatialDirect(30, 30).to(device) # 15, 30
# 		self.SpatialAttention = SpatialAttention(30, 30, 3, 15).to(device)
# 		self.TemporalAttention = TemporalAttention(30, 1, 15).to(device)

# 		## Now, tinker with adding components.
# 		# Bipartite read-out operation.
# 		self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15)
# 		self.DataAggregationAssociationPhase = DataAggregationAssociationPhase(15, 15) # need to add concatenation
# 		self.LocalSliceLgCollapseP = LocalSliceLgCollapse(30, 15) # need to add concatenation. Should it really shrink dimension? Probably not..
# 		self.LocalSliceLgCollapseS = LocalSliceLgCollapse(30, 15) # need to add concatenation. Should it really shrink dimension? Probably not..
# 		self.Arrivals = StationSourceAttentionMergedPhases(30, 15, 2, 15, n_heads = 3)

# 		self.ftrns1 = ftrns1
# 		self.ftrns2 = ftrns2

# 	def forward(self, Slice, Mask, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src, A_edges_p, A_edges_s, dt_partition, tlatent, tpick, ipick, phase_label, locs_use, x_temp_cuda_cart, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

# 		n_line_nodes = Slice.shape[0]
# 		mask_p_thresh = 0.01
# 		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use.shape[0]

# 		# x_temp_cuda_cart = self.ftrns1(x_temp_cuda)
# 		# x = self.TemporalConvolve(Slice).view(n_line_nodes,-1) # slowest module
# 		x_latent = self.DataAggregation(Slice.view(n_line_nodes, -1), Mask, A_in_sta, A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
# 		x = self.Bipartite_ReadIn(x_latent, A_src_in_edges, Mask, locs_use.shape[0], x_temp_cuda_cart.shape[0])
# 		x = self.SpatialAggregation1(x, A_src, x_temp_cuda_cart)
# 		x = self.SpatialAggregation2(x, A_src, x_temp_cuda_cart)
# 		x_spatial = self.SpatialAggregation3(x, A_src, x_temp_cuda_cart) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
# 		y_latent = self.SpatialDirect(x) # contains data on spatial solution.
# 		y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
# 		x = self.SpatialAttention(x_spatial, x_query_cart, x_temp_cuda_cart) # second slowest module (could use this embedding to seed source source attention vector).
# 		x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
# 		x = self.TemporalAttention(x, t_query) # on random queries

# 		## Note below: why detach x_latent?
# 		mask_out = 1.0*(y[:,:,0].detach().max(1, keepdims = True)[0] > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
# 		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
# 		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, A_in_sta, A_in_src) # detach x_latent. Just a "reference"
# 		arv_p = self.LocalSliceLgCollapseP(A_edges_p, dt_partition, tpick, ipick, phase_labels, tlatent[:,0].reshape(-1,1), n_temp, n_sta)
# 		arv_s = self.LocalSliceLgCollapseS(A_edges_s, dt_partition, tpick, ipick, phase_labels, tlatent[:,1].reshape(-1,1), n_temp, n_sta)
# 		arv_p = self.ArrivalP(x_query_src_cart, tq_sample, x_src, trv_out_q[:,:,0], arv_p, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
# 		arv_s = self.ArrivalS(x_query_src_cart, tq_sample, x_src, trv_out_q[:,:,1], arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,1].view(-1)

# 		return y, x, arv_p, arv_s

# 	def set_adjacencies(self, A_in_sta, A_in_src, A_src_in_edges, A_Lg_in_src, A_src, A_edges_p, A_edges_s, dt_partition, tlatent):

# 		self.A_in_sta = A_in_sta
# 		self.A_in_src = A_in_src
# 		self.A_src_in_edges = A_src_in_edges
# 		self.A_Lg_in_src = A_Lg_in_src
# 		self.A_src = A_src
# 		self.A_edges_p = A_edges_p
# 		self.A_edges_s = A_edges_s
# 		self.dt_partition = dt_partition
# 		self.tlatent = tlatent

# 	def forward_fixed(self, Slice, Mask, tpick, ipick, phase_label, locs_use, x_temp_cuda_cart, x_query_cart, x_query_src_cart, t_query, tq_sample, trv_out_q):

# 		n_line_nodes = Slice.shape[0]
# 		mask_p_thresh = 0.01
# 		n_temp, n_sta = x_temp_cuda_cart.shape[0], locs_use.shape[0]

# 		# x_temp_cuda_cart = self.ftrns1(x_temp_cuda)
# 		# x = self.TemporalConvolve(Slice).view(n_line_nodes,-1) # slowest module
# 		x_latent = self.DataAggregation(Slice.view(n_line_nodes, -1), Mask, self.A_in_sta, self.A_in_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
# 		x = self.Bipartite_ReadIn(x_latent, self.A_src_in_edges, Mask, locs_use.shape[0], x_temp_cuda_cart.shape[0])
# 		x = self.SpatialAggregation1(x, self.A_src, x_temp_cuda_cart)
# 		x = self.SpatialAggregation2(x, self.A_src, x_temp_cuda_cart)
# 		x_spatial = self.SpatialAggregation3(x, self.A_src, x_temp_cuda_cart) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
# 		y_latent = self.SpatialDirect(x) # contains data on spatial solution.
# 		y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
# 		x = self.SpatialAttention(x_spatial, x_query_cart, x_temp_cuda_cart) # second slowest module (could use this embedding to seed source source attention vector).
# 		x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
# 		x = self.TemporalAttention(x, t_query) # on random queries

# 		## Note below: why detach x_latent?
# 		mask_out = 1.0*(y[:,:,0].detach().max(1, keepdims = True)[0] > mask_p_thresh).detach() # note: detaching the mask. This is source prediction mask. Maybe, this is't necessary?
# 		s, mask_out_1 = self.BipartiteGraphReadOutOperator(y_latent, self.A_Lg_in_src, mask_out, n_sta, n_temp) # could we concatenate masks and pass through a single one into next layer
# 		s = self.DataAggregationAssociationPhase(s, x_latent.detach(), mask_out_1, Mask, self.A_in_sta, self.A_in_src) # detach x_latent. Just a "reference"
# 		arv_p = self.LocalSliceLgCollapseP(self.A_edges_p, self.dt_partition, tpick, ipick, phase_label, s, self.tlatent[:,0].reshape(-1,1), n_temp, n_sta)
# 		arv_s = self.LocalSliceLgCollapseS(self.A_edges_s, self.dt_partition, tpick, ipick, phase_label, s, self.tlatent[:,1].reshape(-1,1), n_temp, n_sta)
# 		arv = self.Arrivals(x_query_src_cart, tq_sample, x_src, trv_out_q, arv_p, arv_s, tpick, ipick, phase_label) # trv_out_q[:,ipick,0].view(-1)
		
# 		arv_p, arv_s = arv[:,:,0].unsqueeze(-1), arv[:,:,1].unsqueeze(-1)

# 		return y, x, arv_p, arv_s

class GNN_spatial(nn.Module):
	def __init__(self, ftrns1, ftrns2, device = 'cuda'):
		super(GNN_spatial, self).__init__()
		# Define modules and other relavent fixed objects (scaling coefficients.)
		# self.TemporalConvolve = TemporalConvolve(2).to(device) # output size implicit, based on input dim
		self.DataAggregation = DataAggregation(4, 15).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.Bipartite_ReadIn = BipartiteGraphOperator(30, 15, ndim_edges = 3).to(device) # 30, 15
		self.SpatialAggregation1 = SpatialAggregation(15, 30).to(device) # 15, 30
		self.SpatialAggregation2 = SpatialAggregation(30, 30).to(device) # 15, 30
		self.SpatialAggregation3 = SpatialAggregation(30, 30).to(device) # 15, 30
		# self.SpatialDirect = SpatialDirect(30, 30).to(device) # 15, 30
		self.SpatialAttention = SpatialAttention(30, 30, 3, 15).to(device)
		self.TemporalAttention = TemporalAttention(30, 1, 15).to(device)

		# ## Now, tinker with adding components.
		# # Bipartite read-out operation.
		# self.BipartiteGraphReadOutOperator = BipartiteGraphReadOutOperator(30, 15)
		# self.DataAggregationAssociationPhase = DataAggregationAssociationPhase(15, 15) # need to add concatenation
		# self.LocalSliceLgCollapseP = LocalSliceLgCollapse(30, 15) # need to add concatenation. Should it really shrink dimension? Probably not..
		# self.LocalSliceLgCollapseS = LocalSliceLgCollapse(30, 15) # need to add concatenation. Should it really shrink dimension? Probably not..
		# self.Arrivals = StationSourceAttentionMergedPhases(30, 15, 2, 15, n_heads = 3)

		self.ftrns1 = ftrns1
		self.ftrns2 = ftrns2

	def forward(self, tr, mask, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_src, locs_use, x_grid_cart_cuda, x_query_cart_cuda, t_query):
		# A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_src, locs_use, x_grid_cart_cuda, x_query_cart_cuda, t_query

		n_line_nodes = tr.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_grid_cart_cuda.shape[0], locs_use.shape[0]

		# x_temp_cuda_cart = self.ftrns1(x_temp_cuda)
		# x = self.TemporalConvolve(Slice).view(n_line_nodes,-1) # slowest module
		x_latent = self.DataAggregation(tr, mask, A_prod_sta_sta, A_prod_src_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, A_src_in_prod, mask, locs_use.shape[0], x_grid_cart_cuda.shape[0])
		x = self.SpatialAggregation1(x, A_src_src, x_grid_cart_cuda)
		x = self.SpatialAggregation2(x, A_src_src, x_grid_cart_cuda)
		x = self.SpatialAggregation3(x, A_src_src, x_grid_cart_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		# y_latent = self.SpatialDirect(x) # contains data on spatial solution.
		# y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid
		x = self.SpatialAttention(x, x_query_cart_cuda, x_grid_cart_cuda) # second slowest module (could use this embedding to seed source source attention vector).
		# x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
		x = self.TemporalAttention(x, t_query) # on random queries

		return x

	def forward_list(self, tr, mask, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_src, locs_use, x_grid_cart_cuda, x_query_cart_cuda_l, t_query):
		# A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_src, locs_use, x_grid_cart_cuda, x_query_cart_cuda, t_query

		n_line_nodes = tr.shape[0]
		mask_p_thresh = 0.01
		n_temp, n_sta = x_grid_cart_cuda.shape[0], locs_use.shape[0]

		# x_temp_cuda_cart = self.ftrns1(x_temp_cuda)
		# x = self.TemporalConvolve(Slice).view(n_line_nodes,-1) # slowest module
		x_latent = self.DataAggregation(tr, mask, A_prod_sta_sta, A_prod_src_src) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.Bipartite_ReadIn(x_latent, A_src_in_prod, mask, locs_use.shape[0], x_grid_cart_cuda.shape[0])
		x = self.SpatialAggregation1(x, A_src_src, x_grid_cart_cuda)
		x = self.SpatialAggregation2(x, A_src_src, x_grid_cart_cuda)
		x = self.SpatialAggregation3(x, A_src_src, x_grid_cart_cuda) # Last spatial step. Passed to both x_src (association readout), and x (standard readout)
		# y_latent = self.SpatialDirect(x) # contains data on spatial solution.
		# y = self.TemporalAttention(y_latent, t_query) # prediction on fixed grid

		## Note: could easily add an adaptive sampling scheme here, or 
		## with the refined image prediction, to localize maxima

		x_out = []
		for j in range(len(x_query_cart_cuda_l)):

			x1 = self.SpatialAttention(x, x_query_cart_cuda_l[j], x_grid_cart_cuda) # second slowest module (could use this embedding to seed source source attention vector).
			# x_src = self.SpatialAttention(x_spatial, x_query_src_cart, x_temp_cuda_cart) # obtain spatial embeddings, source want to query associations for.
			x1 = self.TemporalAttention(x1, t_query) # on random queries
			x_out.append(x1[:,:,0].cpu().detach().numpy())

		return x_out

## Load Region
d_deg = 0.0
lat_range = [-90.0, 90.0]
lon_range = [-180.0, 180.0]
lat_range_extend = [lat_range[0] - d_deg, lat_range[1] + d_deg]
lon_range_extend = [lon_range[0] - d_deg, lon_range[1] + d_deg]
depth_range = [-1, 0]

lat_range_interior = [lat_range[0], lat_range[1]]
lon_range_interior = [lon_range[0], lon_range[1]]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

Ind_subnetworks = None

## Can use weighted global filters as well. Can factor basis into a basis of either component.

use_spherical = False
if use_spherical == True:

	ftrns1 = lambda pos: lla2ecef(pos, e = 0.0, a = 6371e3)
	# ftrns2_sphere = lambda pos: ecef2lla(pos, e = 0.0)
	ftrns2 = lambda pos: ecef2latlon(pos, e = 0.0, a = 6371e3)

	ftrns1_diff = lambda pos: lla2ecef_diff(pos, e = torch.Tensor([0.0]), a = torch.Tensor([6371e3]), device = device) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_diff = lambda pos: ecef2lla_diff(pos, e = torch.Tensor([0.0]), a = torch.Tensor([6371e3]), device = device)

else:

	ftrns1 = lambda pos: lla2ecef(pos)
	# ftrns2 = lambda pos: ecef2lla(pos)
	ftrns2 = lambda pos: ecef2latlon(pos)

	ftrns1_diff = lambda pos: lla2ecef_diff(pos, device = device) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
	ftrns2_diff = lambda pos: ecef2lla_diff(pos, device = device)

ftrns1_sphere_unit = lambda pos: lla2ecef(pos, e = 0.0, a = 1.0)
ftrns2_sphere_unit = lambda pos: ecef2latlon(pos, e = 0.0, a = 1.0)


## Load travel time
# vp = 7000.0
# vs = vp/1.73

# def trv(y, x):
# 	if torch.is_tensor(y) == False:
# 		return torch.Tensor(np.linalg.norm(np.expand_dims(ftrns1(y), axis = 0) - np.expand_dims(ftrns1(x), axis = 1), axis = 2, keepdims = True)/np.array([vp, vs]).reshape(1,1,-1)).to(device)		
# 	else:
# 		return torch.Tensor(np.linalg.norm(np.expand_dims(ftrns1(y.cpu().detach().numpy()), axis = 0) - np.expand_dims(ftrns1(x.cpu().detach().numpy()), axis = 1), axis = 2, keepdims = True)/np.array([vp, vs]).reshape(1,1,-1)).to(device)


vel_model_ver = 2
vel_model_step = 75000

trv_l = []
for phase in ['Pg', 'Pn', 'Sn', 'Lg']:
	trv = TravelTimes(ftrns1_diff, ftrns2_diff).to(device)
	if ext_type == 'local':
		trv.load_state_dict(torch.load('D:/Projects/Global/RSTT/TrainedModels/travel_time_neural_network_%s_ver_%d_step_%d.h5'%(phase, vel_model_ver, vel_model_step)))
	elif ext_type == 'remote':
		trv.load_state_dict(torch.load('/work/wavefront/imcbrear/Global/RSTT/TrainedModels/travel_time_neural_network_%s_ver_%d_step_%d.h5'%(phase, vel_model_ver, vel_model_step)))
	trv.eval()
	trv_l.append(trv)

trv1 = apply_travel_times(trv_l)

trv = lambda y, x: trv1(y, x, 'pairs forward')

# moi

## Load stations

if ext_type == 'local':
	z = np.load('D:/Projects/Global/example_picks.npz')
elif ext_type == 'remote':
	z = np.load('/work/wavefront/imcbrear/Global/example_picks.npz')
locs_use = z['locs_use']
locs = np.copy(locs_use)
picks = z['picks_perm']
torigin = np.nanmin(np.nanmin(z['Tval'], axis = 2), axis = 1)
torigin = torigin.repeat(5) + np.tile(np.array([-50, -40, -30, -20, -10]), len(torigin))

## Example travel times
i0 = np.random.randint(0, high = locs.shape[0], size = 30) ## Really, shouldn't query travel times from sources to stations >1000 km distance
trv_out = trv1(torch.Tensor(locs).cuda(), torch.Tensor(locs[i0]).cuda(), 'pairs relative')
mask_out = trv1(torch.Tensor(locs).cuda(), torch.Tensor(locs[i0]).cuda(), 'pairs mask relative')
## Where mask > 0.5, it is supposedly a "nan" entry

print('reducing grids for testing')
n_grids = 1 # 10 # 30
n_cluster = 5000 # 3500
kx_edges = 15
load_templates = False
save_templates = True
template_ver = 1

extend_grids = False

## Decide on elliptical versus spherical coordinates

x_grids = []
x_grids_edges = []
x_grids_trv = []
for i in range(n_grids):
	# x_grid, _ , _ , _ , _ = kmeans_packing(scale_x_extend, offset_x_extend, 3, n_cluster, ftrns1, n_batch = 5000, n_steps = 3000, n_sim = 1, lr = 0.01)

	eps_extra = 0.1
	eps_extra_depth = 0.02
	scale_up = 1.0 # 10000.0
	weight_vector = np.array([1.0, 1.0, 5.0]).reshape(1,-1)

	# x_grid, _ , _ , _ , _ = kmeans_packing(scale_x_extend, offset_x_extend, 3, n_cluster, ftrns1, n_batch = 5000, n_steps = 3000, n_sim = 1, lr = 0.01)

	offset_x_extend_slice = np.array([offset_x_extend[0,0], offset_x_extend[0,1], offset_x_extend[0,2]]).reshape(1,-1)
	scale_x_extend_slice = np.array([scale_x_extend[0,0], scale_x_extend[0,1], scale_x_extend[0,2]]).reshape(1,-1)

	if extend_grids == True:
		extend1, extend2, extend3, extend4 = (np.random.rand(4) - 0.5)*0.25
		extend5 = (np.random.rand() - 0.5)*2500.0
		offset_x_extend_slice[0,0] += extend1
		offset_x_extend_slice[0,1] += extend2
		scale_x_extend_slice[0,0] += extend3
		scale_x_extend_slice[0,1] += extend4
		offset_x_extend_slice[0,2] += extend5
		scale_x_extend_slice[0,2] = depth_range[1] - offset_x_extend_slice[0,2]

	else:
		pass

	offset_x_grid = scale_up*np.array([offset_x_extend_slice[0,0] - eps_extra*scale_x_extend_slice[0,0], offset_x_extend_slice[0,1] - eps_extra*scale_x_extend_slice[0,1], offset_x_extend_slice[0,2] - eps_extra_depth*scale_x_extend_slice[0,2]]).reshape(1,-1)
	scale_x_grid = scale_up*np.array([scale_x_extend_slice[0,0] + 2.0*eps_extra*scale_x_extend_slice[0,0], scale_x_extend_slice[0,1] + 2.0*eps_extra*scale_x_extend_slice[0,1], scale_x_extend_slice[0,2] + 2.0*eps_extra_depth*scale_x_extend_slice[0,2]]).reshape(1,-1)
	# x_grid = kmeans_packing(scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch = 10000, n_steps = 8000, n_sim = 1, lr = 0.005)[0]/scale_up # .to(device)
	# x_grid = kmeans_packing_weight_vector(weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch = 10000, n_steps = 8000, n_sim = 1, lr = 0.005)[0]/scale_up # .to(device) # 8000

	# x_grid = kmeans_packing(scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch = 10000, n_steps = 8000, n_sim = 1, lr = 0.005)[0]/scale_up # .to(device) # 8000

	# x_grid = kmeans_packing_global(depth_range, 3, n_cluster, ftrns1, ftrns2, ftrns1_sphere, ftrns2_sphere, n_batch = 10000, n_steps = 3000, n_sim = 1, lr = 0.005)/scale_up # .to(device) # 8000


	use_kmeans = False
	use_Fibonacci = True
	assert((use_kmeans + use_Fibonacci) == 1)

	if use_kmeans == True:
		x_grid = kmeans_packing_global_cartesian(depth_range, 3, n_cluster, ftrns1, ftrns2, ftrns1_sphere, ftrns2_sphere, n_batch = 10000, n_steps = 3000, n_sim = 1, lr = 0.005)/scale_up # .to(device) # 8000

	elif use_Fibonacci:
		x_grid = spherical_packing_nodes(n_cluster, ftrns1_sphere_unit, ftrns2_sphere_unit)

	# trv_out = trv(torch.Tensor(locs).cuda(), torch.Tensor(x_grid).cuda())

	# x_grid[:,0:2] += r_rand_shift.reshape(1,-1)

	iargsort = np.argsort(x_grid[:,1])
	x_grid = x_grid[iargsort]
	edge_index = knn(torch.Tensor(ftrns1(x_grid)/1000.0).cuda(), torch.Tensor(ftrns1(x_grid)/1000.0).cuda(), k = kx_edges).flip(0).contiguous()
	edge_index = remove_self_loops(edge_index)[0].cpu().detach().numpy()
	x_grids.append(x_grid)
	x_grids_edges.append(edge_index)

	# x_grids_trv_slice = []
	# for j in range(x_grid.shape[0]):
	# 	x_grids_trv_slice.append(trv(torch.Tensor(locs).to(device), torch.Tensor(x_grid[j].reshape(1,-1)).to(device)).cpu().detach().numpy())


	print('finished grid %d'%i)


	if save_templates == True: # trained_travel_time_model_NC_network_2000_2020_ver_1
		if ext_type == 'local':
			h_templates = np.savez_compressed('D:/Projects/Global/Grids/Global_seismic_network_templates_denser_%d.npz'%template_ver, x_grids = x_grids, x_grids_edges = x_grids_edges)
		elif ext_type == 'remote':
			h_templates = np.savez_compressed('/work/wavefront/imcbrear/Global/Grids/Global_seismic_network_templates_denser_%d.npz'%template_ver, x_grids = x_grids, x_grids_edges = x_grids_edges)	


# if ext_type == 'local':
# 	z = np.load('D:/Projects/Global/example_grids.npz')
# elif ext_type == 'remote':
# 	z = np.load('/work/wavefront/imcbrear/Global/example_grids.npz')
# x_grid = z['x_grid']

## Make travel time vectors
x_grids_trv = []
x_grids_trv_pointers_p = []
x_grids_trv_pointers_s = []
x_grids_trv_refs = []

t_win = 10.0 ## Needs to be same, as used later!
dt_embed = 1.0
k_time_edges = 10

# moi

# ts_max_val = trv(locs, x_grid).max().item()

for i in range(len(x_grids)):

	# trv_out = trv(torch.Tensor(locs).cuda(), torch.Tensor(x_grids[i]).cuda())

	trv_out_slice = []
	for j in range(x_grids[i].shape[0]):
		trv_out_slice.append(trv(torch.Tensor(locs).cuda(), torch.Tensor(x_grids[i][j].reshape(1,-1)).cuda()).cpu().detach().numpy())
	trv_out = np.vstack(trv_out_slice)

	ts_max_val = trv_out.max()

	x_grids_trv.append(trv_out)
	# A_edges_time_p, A_edges_time_s, dt_partition = assemble_time_pointers_for_stations(trv_out)

	assert(trv_out.min() > 0.0)
	assert(trv_out.max() < (ts_max_val + 3.0))

	# ## Make pointers to edges into time arrays, for each grid (and all stations), here.
	# dt_partition = torch.arange(-t_win, t_win + x_grids_trv[i].max() + dt_embed, dt_embed)

	# x_grids_trv_pointers_p.append(A_edges_time_p)
	# x_grids_trv_pointers_s.append(A_edges_time_s)
	# x_grids_trv_refs.append(dt_partition) # save as cuda tensor, or no?


print('Issue occuring, causing travel times to exceed max values significantly')
print('It may be caused by x_grids nodes appearing outside boundaries travel time model allows')
print('May be able to increase padding on sides of travel time model, or remove these spatial nodes')
print('max_t is not unreasonably large when x_grids are bounded')
## Right now, this is far too large
# max_t = float(np.ceil(max([x_grids_trv[i].max() for i in range(len(x_grids_trv))]))) # + 10.0

# max_t = np.copy(ts_max_val)

max_t = 500.0

x_grid = x_grids[0]

locs = np.copy(locs_use)

x_grid_cuda = torch.Tensor(x_grid).to(device) ## Need to make sure depths arn't too large compared to lat-lon coordinates 
## (might need to use km for depth)

x_grid_cart_cuda = torch.Tensor(ftrns1(x_grid)).to(device)

locs_use_cart_cuda = torch.Tensor(ftrns1(locs_use)).to(device)

## Resoltuion in detection stage needs to be decoupled from resolution in association assignment stage.
## E.g., a coarse grid can be used for detection, since a large kernel size is used, but the association
## needs more accurate source-arrival associations. Or perhaps a `local' model is used for detected events,
## to determine associations using a more refined spatial scale and source-arrival associations.

## Two waves to reduce memory:
# (i). On full cartesian product graph of all source x recievers, only use (source x reciever) - (source x reciever) edges for source-receiever pairs with small spatial offset
# (2). Only consider nodes of source-recievers that have small spatial offset, and then consider all edges of this "subgraph" of the Cartesian product graph
## Version 2 will reduce the total number of nodes and edges, while version 1 only reduces the edges.

# [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod] = extract_inputs_adjacencies_partial(locs_use, np.arange(locs_use.shape[0]), x_grid)
# [A_sta_sta1, A_src_src1, A_prod_sta_sta1, A_prod_src_src1, A_src_in_prod1] = extract_inputs_adjacencies_partial_clipped_pairwise_edges(locs_use, np.arange(locs_use.shape[0]), x_grid, max_deg_offset = 5.0)

## Note, scaling of source-reciever distance edges

max_deg_offset = 10.0
k_nearest_pairs = 30

[A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta] = extract_inputs_adjacencies_partial_clipped_pairwise_nodes_and_edges_with_projection(locs_use, np.arange(locs_use.shape[0]), x_grid, ftrns1, ftrns2, max_deg_offset = max_deg_offset, k_nearest_pairs = k_nearest_pairs)

dist_A_prod_sta_sta, dist_A_prod_src_src = compute_pairwise_distances(locs_use, x_grid, A_prod_sta_sta, A_prod_src_src)
# dist_A_prod_sta_sta1, dist_A_prod_src_src1 = compute_pairwise_distances(locs_use, x_grid, A_prod_sta_sta1, A_prod_src_src1)

## Load travel time calculator

# moi

## Notes:
## Might want to implement differential edge measurements in lat-lon domain (with modulus for longitude)
## since Cartesian domain can be very "different" in different places


# moi

## Implement prediction through GENIE model

m = GNN_spatial(ftrns1_diff, ftrns2_diff).to(device)

tr = torch.rand(A_src_in_sta.shape[1],4).to(device)
mask = 1.0*(torch.rand(A_src_in_sta.shape[1],4).to(device) > 0.5)
x_query = np.random.rand(1000, 3)*scale_x + offset_x
x_query_cart_cuda = torch.Tensor(ftrns1(x_query)).to(device)
t_query = torch.arange(-5, 5).reshape(-1,1).to(device)

out = m(tr, mask, A_prod_sta_sta.to(device), A_prod_src_src.to(device), A_src_in_prod.to(device), A_src_src.to(device), locs_use, x_grid_cart_cuda, x_query_cart_cuda, t_query)

n_batch = 20

## Find all source points within epsilon distance of a station
x1 = np.arange(lat_range[0], lat_range[1], 0.1)
x2 = np.arange(lon_range[0], lon_range[1], 0.1)
x11, x12, x13 = np.meshgrid(x1, x2, 0.0)
xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis = 1)

tree = cKDTree(ftrns1(locs))
query = tree.query(ftrns1(xx))
ip = np.where(query[0] < 1000e3)[0]
src_density = KernelDensity(bandwidth = 0.2).fit(xx[ip,0:2])

# [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data = generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range_interior, lon_range_interior, lat_range_extend, lon_range_extend, depth_range, ftrns1, ftrns2, fixed_subnetworks = Ind_subnetworks, use_preferential_sampling = True, n_batch = n_batch, verbose = True)

x_query = np.random.rand(5000,3)*scale_x + offset_x

n_batch = 1
[Inpts, Masks, X_query, Lbls], data = generate_synthetic_data_simple(trv, locs, x_grids, x_grids_trv, x_query, A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_src_in_prod, A_src_in_sta, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, ftrns1, ftrns2, n_batch = n_batch, src_density = src_density)



if ext_type == 'local':
	write_training_file = 'D:/Projects/Global/GNN_TrainedModels/'
elif ext_type == 'remote':
	write_training_file = '/work/wavefront/imcbrear/Global/GNN_TrainedModels/'


# moi

## Train

m = GNN_spatial(ftrns1_diff, ftrns2_diff).to(device)

n_ver_load = 2
n_step_load = 19500 # 17800

m.load_state_dict(torch.load(write_training_file + 'GENIE_on_large_scale_%d_step_%d.h5'%(n_ver_load, n_step_load)))


n_batch = 100
n_epochs = 10000
scale_loss = 3.0
n_ver = 2
losses = []
trgts = []
preds = []

Inpts_l = []
Masks_l = []
X_query_l = []
Lbls_l = []
Preds_l = []

generate_new_data = True
pre_load_data = False
if pre_load_data == True:
	assert(generate_new_data == False)

if generate_new_data == False:

	n_ver_load = 1
	if ext_type == 'local':
		st = glob.glob('D:/Projects/Global/TrainingData/*ver_%d.npz'%n_ver_load)
	elif ext_type == 'remote':
		st = glob.glob('/work/wavefront/imcbrear/Global/TrainingData/*ver_%d.npz'%n_ver_load)
	elif ext_type == 'server':
		st = glob.glob('/oak/stanford/schools/ees/beroza/imcbrear/Global/TrainingData/*ver_%d.npz'%n_ver_load)


	if pre_load_data == True:

		n_max_files = 5000
		ichoose = np.sort(np.random.choice(len(st), size = np.minimum(len(st), n_max_files), replace = False))
		len_files = len(ichoose)

		Inpts, Masks, X_query, Lbls = [], [], [], []
		for inc, i in enumerate(ichoose):
			z = np.load(st[i])
			Inpts.append(z['inpt'])
			Masks.append(z['mask'])
			X_query.append(z['x_query'])
			Lbls.append(z['lbls'])
			z.close()
			if np.mod(inc, 50) == 0:
				print('Loaded %d'%inc)

# moi

A_prod_sta_sta_cuda, A_prod_src_src_cuda, A_src_in_prod_cuda, A_src_src_cuda = A_prod_sta_sta.to(device), A_prod_src_src.to(device), A_src_in_prod.to(device), A_src_src.to(device)
## Why was A_src_in_prod not on device?

n_iters = 5
n_batch = 150

Inpts_l = []
Masks_l = []
X_query_l = []
Lbls_l = []
Preds_l = []

file_num = 0

# moi

t_win = 10.0
step = 10.0 # .0 # 10
step_abs = 1
day_len = 3600*24
tsteps = np.arange(0, day_len, step) ## Fixed solution grid.
tsteps_abs = np.arange(-t_win/2.0, day_len + t_win/2.0 + 1, step_abs) ## Fixed solution grid, assume 1 second
tree_tsteps = cKDTree(tsteps_abs.reshape(-1,1))

tsteps_abs_cat = cKDTree(tsteps.reshape(-1,1)) ## Make this tree, so can look up nearest time for all cat.

times_need_l = np.copy(tsteps) # 5 sec overlap, more or less
## Double check this.
n_batches = int(np.floor(len(times_need_l)/n_batch))
times_need = [times_need_l[j*n_batch:(j + 1)*n_batch] for j in range(n_batches)]
if n_batches*n_batch < len(times_need_l):
	times_need.append(times_need_l[n_batches*n_batch::]) ## Add last few samples

n_batches = int(np.floor(len(tsteps)/n_batch))
n_extra = len(tsteps) - n_batches*n_batch
n_overlap = int(t_win/step) # check this

n_samples = int(250e3)
plot_on = False
save_on = True

d_deg = 0.25
x1 = np.arange(lat_range[0], lat_range[1], d_deg)
x2 = np.arange(lon_range[0], lon_range[1], d_deg)
x11, x12 = np.meshgrid(x1, x2)
xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1)), axis = 1)
x_query = np.concatenate((xx, np.zeros((len(xx),1))), axis = 1)
x_query_cart_cuda = torch.Tensor(ftrns1(x_query)).to(device)
t_query = torch.arange(-5, 5).reshape(-1,1).to(device)

n_batch_query = 10000
n_batches_query = int(np.floor(len(x_query)/n_batch_query))
query_need = [x_query[j*n_batch_query:(j + 1)*n_batch_query] for j in range(n_batches_query)]
if n_batch_query*n_batches_query < len(x_query):
	query_need.append(x_query[n_batch_query*n_batches_query::]) ## Add last few samples

x_query_cart_cuda_l = [torch.Tensor(ftrns1(query_need[i])).to(device) for i in range(len(query_need))]

thresh_save = 0.05

# moi

client = Client('USGS')
t0 = UTCDateTime(2022, 11, 3)
tf = UTCDateTime(2022, 11, 4)
events = client.get_events(starttime = t0, endtime = tf, minlatitude = -90.0, maxlatitude = 90.0, minlongitude = -180.0, maxlongitude = 180.0, minmagnitude = 1.0, orderby = 'time-asc')

srcs_known = []
for i in range(len(events)):
	t_origin = events[i].origins[0].time - t0
	srcs_known.append(np.array([events[i].origins[0].latitude, events[i].origins[0].longitude, -1.0*events[i].origins[0].depth, t_origin, events[i].magnitudes[0].mag]).reshape(1,-1))
srcs_known = np.vstack(srcs_known)

isort = np.flip(np.argsort(srcs_known[:,4]))
srcs_known = srcs_known[isort]

## Can find which sources are 
## within ~1000 km of ~5 stations

## This way, we can check a priori, which are most
## likely missed

tree = cKDTree(ftrns1(locs*np.array([1.0, 1.0, 0.0]).reshape(1,-1)))
ip_query = tree.query(ftrns1(srcs_known[:,0:3]*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), k = 5)
ifind = np.where(ip_query[0].max(1) < 1000e3)[0]
ifind1 = np.where(srcs_known[:,4] >= 1.5)[0]
iboth = np.array(list(set(ifind).intersection(ifind1)))

## Look at residual as a function of magnitude

# Out = np.zeros((x_query.shape[0], len(tsteps_abs)))

# moi

## Can load origin times for known events and compute predictions for these

thresh_save = 0.1

process_known = True

srcs_pred = []
isrcs_found_known = []
isrcs_missed_known = []

moi

if process_known == True:

	Out_save = []

	for i in range(len(srcs_known)):

		# time_samples = times_need[i]
		tsteps_slice = np.array([srcs_known[i,3]])
		tsteps_slice_indices = tree_tsteps.query(tsteps_slice.reshape(-1,1))[1]

		Inpts, Masks = sample_inputs(trv, picks, locs, x_grids, x_grids_trv, tsteps_slice, A_src_in_sta, lat_range, lon_range, depth_range, ftrns1, ftrns2)

		for j in range(len(Inpts)):

			with torch.no_grad():

				# out = m(torch.Tensor(Inpts[j]).to(device), torch.Tensor(Masks[j]).to(device), A_prod_sta_sta_cuda, A_prod_src_src_cuda, A_src_in_prod_cuda, A_src_src_cuda, locs, x_grid_cart_cuda, x_query_cart_cuda, t_query)			
				out = m.forward_list(torch.Tensor(Inpts[j]).to(device), torch.Tensor(Masks[j]).to(device), A_prod_sta_sta_cuda, A_prod_src_src_cuda, A_src_in_prod_cuda, A_src_src_cuda, locs, x_grid_cart_cuda, x_query_cart_cuda_l, t_query)			

				# i1, i2 = np.where(out[:,:,0].cpu().detach().numpy() > thresh_save)

				ip_need = tree_tsteps.query(tsteps_abs[tsteps_slice_indices[j]] + np.arange(-t_win/2.0, t_win/2.0).reshape(-1,1))

				# Out[:,ip_need[1]] += out[:,:,0].cpu().detach().numpy()/n_overlap # /n_scale_x_grid

				Out_event = []

				for n in range(len(out)):

					i1, i2 = np.where(out[n] > thresh_save) ## Should save the indices of x_query, absolute time steps, and values

					if len(i1) > 0:

						Out_event.append(np.concatenate((i1.reshape(-1,1) + n_batch_query*n, ip_need[1][i2].reshape(-1,1), out[n][i1,i2].reshape(-1,1)), axis = 1))						

				if len(Out_event) > 0:
					
					Out_event = np.vstack(Out_event)
					Out_save.append(Out_event)

					idist = np.where(np.linalg.norm(x_query[Out_event[:,0].astype('int'), 0:2] - srcs_known[i,0:2].reshape(1,-1), axis = 1) < 8.0)[0]

					if len(idist) > 0:

						iargmax = np.argmax(Out_event[idist,2]) ## Note: the argmax over the entire domain may fail, since some sources may be co-occuring with others
						print('Max src loc: %0.2f %0.2f, Value %0.2f'%(x_query[int(Out_event[idist[iargmax],0]),0], x_query[int(Out_event[idist[iargmax],0]),1], np.max(Out_event[idist,2])))
						print('Known src loc: %0.2f %0.2f %0.2f, M%0.2f \n'%(srcs_known[i,0], srcs_known[i,1], srcs_known[i,2]/1000.0, srcs_known[i,4]))

						srcs_pred.append(x_query[int(Out_event[idist[iargmax],0]),:].reshape(1,-1))
						isrcs_found_known.append(i)

					else:

						srcs_pred.append(np.nan*np.ones((1,3)))
						isrcs_missed_known.append(i)

				else:
					
					Out_save.append(np.zeros((0,3)))
					srcs_pred.append(np.nan*np.ones((1,3)))
					isrcs_missed_known.append(i)


else: ## Process continuous days

	# Out_save = np.zeros((len(x_query_cart_cuda), 21))
	points_need = np.arange(len(tsteps_abs))
	iactive = np.arange(len(tsteps_abs))

	Out_save = []

	thresh_save = 0.2

	for i in range(n_batches):

		# time_samples = times_need[i]
		tsteps_slice = times_need[i]
		tsteps_slice_indices = tree_tsteps.query(tsteps_slice.reshape(-1,1))[1]

		Inpts, Masks = sample_inputs(trv, picks, locs, x_grids, x_grids_trv, tsteps_slice, A_src_in_sta, lat_range, lon_range, depth_range, ftrns1, ftrns2)

		for j in range(len(Inpts)):

			with torch.no_grad():

				## For each j, find point on series that needs to be saved, can use a rolling buffer

				# out = m(torch.Tensor(Inpts[j]).to(device), torch.Tensor(Masks[j]).to(device), A_prod_sta_sta_cuda, A_prod_src_src_cuda, A_src_in_prod_cuda, A_src_src_cuda, locs, x_grid_cart_cuda, x_query_cart_cuda, t_query)			
				out = m.forward_list(torch.Tensor(Inpts[j]).to(device), torch.Tensor(Masks[j]).to(device), A_prod_sta_sta_cuda, A_prod_src_src_cuda, A_src_in_prod_cuda, A_src_src_cuda, locs, x_grid_cart_cuda, x_query_cart_cuda_l, t_query)			

				# i1, i2 = np.where(out[:,:,0].cpu().detach().numpy() > thresh_save)

				ip_need = tree_tsteps.query(tsteps_abs[tsteps_slice_indices[j]] + np.arange(-t_win/2.0, t_win/2.0).reshape(-1,1))

				# Out[:,ip_need[1]] += out[:,:,0].cpu().detach().numpy()/n_overlap # /n_scale_x_grid

				for n in range(len(out)):

					i1, i2 = np.where(out[n] > thresh_save) ## Should save the indices of x_query, absolute time steps, and values

					if len(i1) > 0:

						Out_save.append(np.concatenate((i1.reshape(-1,1) + n_batch_query*n, ip_need[1][i2].reshape(-1,1), out[n][i1,i2].reshape(-1,1)), axis = 1))

				ifinished = set(np.where(points_need < ip_need[1].min())[0]).intersection(iactive)
				tree_active = cKDTree(iactive.reshape(-1,1))
				# ifinished_index = tree_active.query(ifinished.reshape(-1,1))[1]

				## Update iactive
				# iactive = np.delete(iactive, ifinished_index, axis = 0)





np.savez_compressed('/work/wavefront/imcbrear/Global/example_predictions_known_events.npz', srcs_pred = srcs_pred, srcs_known = srcs_known, isrcs_found_known = isrcs_found_known, isrcs_missed_known = isrcs_missed_known, Out_save = Out_save, ifind = ifind, ifind1 = ifind1, iboth = iboth)