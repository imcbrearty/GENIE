import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from torch_scatter import scatter
import h5py
import glob
import networkx as nx

from torch import nn, optim
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from torch_geometric.transforms import FaceToEdge, GenerateMeshNormals
from joblib import Parallel, delayed
import multiprocessing
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.spatial import cKDTree
from torch.autograd import Variable
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import yaml

def load_config(file_path: str) -> dict:
	"""Load configuration from a YAML file."""
	with open(file_path, 'r') as file:
		return yaml.safe_load(file)

def make_cayleigh_graph(n):

	generators = [np.array([1, 1, 0, 1]).reshape(2,2), np.array([1, 0, 1, 1]).reshape(2,2)]
	nodes = np.vstack([generators[0].reshape(1,-1), generators[1].reshape(1,-1)])
	edges = []

	new = np.inf
	cnt = 0

	while new > 0:

		print('iteration %d, num nodes %d'%(cnt, len(nodes)))

		tree = cKDTree(nodes)
		len_nodes = len(nodes)

		new_nodes_1 = []
		new_nodes_2 = []

		for i in range(len(nodes)):

			new_nodes_1.append(np.mod(nodes[i].reshape(2,2) @ generators[0], n).reshape(1,-1))
			new_nodes_2.append(np.mod(nodes[i].reshape(2,2) @ generators[1], n).reshape(1,-1))

		new_nodes_1 = np.vstack(new_nodes_1) # has size nodes
		new_nodes_2 = np.vstack(new_nodes_2)
		new_nodes = np.unique(np.concatenate((new_nodes_1, new_nodes_2), axis = 0), axis = 0)

		q = tree.query(new_nodes)[0]
		inew = np.where(q > 0)[0]
		new_nodes = new_nodes[inew]

		if len(inew) == 0:
			new = 0
			continue # Break loop

		## Now need to find which entries in new_nodes are linked for each input node
		tree_new = cKDTree(new_nodes)
		ip = tree_new.query(new_nodes_1)
		ip1 = np.where(ip[0] == 0)[0] ## Points to current absolute node indices that are linked to new node
		edges_new_1 = np.concatenate((ip1.reshape(-1,1), len_nodes + ip[1][ip1].reshape(-1,1)), axis = 1)

		ip = tree_new.query(new_nodes_2)
		ip1 = np.where(ip[0] == 0)[0] ## Points to current absolute node indices that are linked to new node
		edges_new_2 = np.concatenate((ip1.reshape(-1,1), len_nodes + ip[1][ip1].reshape(-1,1)), axis = 1)

		# edges.append(np.unique(np.concatenate((edges_new_1, edges_new_2), axis = 0), axis = 0))
		nodes = np.concatenate((nodes, new_nodes), axis = 0)
		
		cnt += 1

	## Find inverses to generators
	inv_indices_1 = []
	inv_indices_2 = []
	for i in range(len(nodes)):
		if np.abs(np.mod(generators[0] @ nodes[i].reshape(2,2), n) - np.eye(2)).max() == 0:
			inv_indices_1.append(i)
		if np.abs(np.mod(generators[1] @ nodes[i].reshape(2,2), n) - np.eye(2)).max() == 0:
			inv_indices_2.append(i)

	assert(len(inv_indices_1) == 1)
	assert(len(inv_indices_2) == 1)

	generators_inverses = [nodes[inv_indices_1[0]].reshape(2,2), nodes[inv_indices_2[0]].reshape(2,2)]

	## Now must add missing edges between all previously created nodes. (can do this outside of the loop)
	for i in range(len(nodes)):
		for j in range(len(nodes)):
			dist1 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators[0], n) - nodes[j].reshape(2,2)).max()
			dist2 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators[1], n) - nodes[j].reshape(2,2)).max()
			if ((dist1 == 0) + (dist2 == 0)) > 0:
				edges.append(np.array([i,j]).reshape(1,-1))

	for i in range(len(nodes)):
		for j in range(len(nodes)):
			dist1 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators_inverses[0], n) - nodes[j].reshape(2,2)).max()
			dist2 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators_inverses[1], n) - nodes[j].reshape(2,2)).max()
			if ((dist1 == 0) + (dist2 == 0)) > 0:
				edges.append(np.array([i,j]).reshape(1,-1))

	edges = np.unique(np.vstack(edges), axis = 0)
	## Check for all edges, if each node is really linked to the declared nodes.
	for i in range(len(edges)):
		dist1 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators[0], n) - nodes[edges[i,1]].reshape(2,2)).max()
		dist2 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators[1], n) - nodes[edges[i,1]].reshape(2,2)).max()
		dist3 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators_inverses[0], n) - nodes[edges[i,1]].reshape(2,2)).max()
		dist4 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators_inverses[1], n) - nodes[edges[i,1]].reshape(2,2)).max()
		assert(((dist1 == 0) + (dist2 == 0) + (dist3 == 0) + (dist4 == 0)) > 0)
		# print(i)

	return edges

## This global_mean_pool function is taken from 
## TORCH_GEOMETRIC.GRAPHGYM.MODELS.POOLING
## It relies on scatter.
def global_mean_pool(x, batch, size=None):
	"""
	Globally pool node embeddings into graph embeddings, via elementwise mean.
	Pooling function takes in node embedding [num_nodes x emb_dim] and
	batch (indices) and outputs graph embedding [num_graphs x emb_dim].

	Args:
		x (torch.tensor): Input node embeddings
		batch (torch.tensor): Batch tensor that indicates which node
		belongs to which graph
		size (optional): Total number of graphs. Can be auto-inferred.

	Returns: Pooled graph embeddings

	"""
	size = batch.max().item() + 1 if size is None else size
	return scatter(x, batch, dim=0, dim_size=size, reduce='mean')

# def batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, node_ind_max, device):

# 	inpt_batch = torch.vstack(signal_slice).to(device)
# 	mask_batch = inpt_batch[:,3::] # Only select non-position points for mask
# 	pos_batch = inpt_batch[:,0:3]
# 	query_batch = torch.Tensor(np.vstack(query_slice)).to(device)
# 	edges_batch = torch.cat([edges_slice[j] + j*node_ind_max for j in range(len(edges_slice))], dim = 1).to(device)
# 	edges_batch_c = torch.cat([edges_c_slice[j] + j*node_ind_max for j in range(len(edges_c_slice))], dim = 1).to(device)
# 	trgt_batch = torch.Tensor(np.vstack(trgt_slice)).to(device)

# 	return inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch

def batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, node_ind_max):

	inpt_batch = torch.vstack(signal_slice) # .to(device)
	mask_batch = inpt_batch[:,3::] # Only select non-position points for mask
	pos_batch = inpt_batch[:,0:3]
	query_batch = torch.vstack(query_slice) # .to(device)
	edges_batch = torch.cat([edges_slice[j] + j*node_ind_max for j in range(len(edges_slice))], dim = 1) # .to(device)
	edges_batch_c = torch.cat([edges_c_slice[j] + j*node_ind_max for j in range(len(edges_c_slice))], dim = 1) # .to(device)
	trgt_batch = torch.vstack(trgt_slice) # .to(device)

	return inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch

def kmeans_packing_logarithmic(scale_x, offset_x, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	V_results = []
	Losses = []

	center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

	a_param = 3.0
	num_thresh = 0.1 # 3.0
	n_up_sample = 10

	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

			tree = cKDTree(v)
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x
			x_r = np.random.pareto(a_param, size = 100000)
			ifind = np.where(x_r < num_thresh)[0]
			ifind = np.random.choice(ifind, size = n_up_sample*n_batch)
			x_r = x_r[ifind].reshape(-1,1)/num_thresh
			x_theta = np.random.rand(n_up_sample*n_batch)*2.0*np.pi
			x_xy = x_r*np.concatenate((np.cos(x_theta[:,None]), np.sin(x_theta[:,None])), axis = 1)*(scale_x[:,0:2]/2.0)
			x_xy = np.concatenate((x_xy + center_x, np.random.rand(n_up_sample*n_batch,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
			x = np.concatenate((x, x_xy), axis = 0)
			x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

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

def kmeans_packing_logarithmic_parallel(num_cores, scale_x_list, offset_x_list, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	def step_test(args):

		scale_x, offset_x, ndim, n_clusters = args

		V_results = []
		Losses = []

		center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

		a_param = 3.0
		num_thresh = 0.1 # 3.0
		n_up_sample = 10

		for n in range(n_sim):

			losses, rz = [], []
			for i in range(n_steps):
				if i == 0:
					v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

				tree = cKDTree(v)
				x = np.random.rand(n_batch, ndim)*scale_x + offset_x
				x_r = np.random.pareto(a_param, size = 100000)
				ifind = np.where(x_r < num_thresh)[0]
				ifind = np.random.choice(ifind, size = n_up_sample*n_batch)
				x_r = x_r[ifind].reshape(-1,1)/num_thresh
				x_theta = np.random.rand(n_up_sample*n_batch)*2.0*np.pi
				x_xy = x_r*np.concatenate((np.cos(x_theta[:,None]), np.sin(x_theta[:,None])), axis = 1)*(scale_x[:,0:2]/2.0)
				x_xy = np.concatenate((x_xy + center_x, np.random.rand(n_up_sample*n_batch,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
				x = np.concatenate((x, x_xy), axis = 0)
				x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

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

		return V_results[ibest] # , ind

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [ scale_x_list[i], offset_x_list[i], ndim, n_clusters] ) for i in range(num_cores))

	pos_grid_l = []
	for i in range(num_cores):
		pos_grid_l.append(results[i])

	return pos_grid_l


def make_graph_from_mesh(mesh):
	## Input is T

	## Check if these edges are correctly defined. E.g.,
	## are incoming and outgoing edges correctly defined?

	data = Data()
	data.face = mesh
	trns = FaceToEdge()
	edges = trns(data).edge_index

	return edges

def make_spatial_graph(pos, k_pos = 15, device = 'cuda'):

	## For every mesh node, link to k pos nodes
	## Note: we could attach all spatial nodes to
	## the nearest mesh nodes, though this seems
	## less natural (as it would introduce
	## very long range connections, linking to
	## a small part of the mesh grid. It could
	## potentially make learning the mapping
	## easier).

	## Can give absolute node locations as features

	n_pos = pos.shape[0]

	# A_edges_mesh = knn(mesh, mesh, k = k + 1).flip(0).contiguous()[0].to(device)

	## transfer
	A_edges = remove_self_loops(knn(pos, pos, k = k_pos + 1).flip(0).contiguous())[0] # .to(device)
	# edges_offset = pos[A_edges[1]] - pos[A_edges[0]]

	return A_edges # , edges_offset

def load_logarithmic_grids(ext_type, n_ver):

	if ext_type == 'local':
		
		pos_grid_l = np.load('D:/Projects/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver)['pos_grid_l']
		pos_grid_l = [torch.Tensor(pos_grid_l[j]) for j in range(pos_grid_l.shape[0])]

	elif ext_type == 'remote':

		pos_grid_l = np.load('/work/wavefront/imcbrear/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver)['pos_grid_l']
		pos_grid_l = [torch.Tensor(pos_grid_l[j]) for j in range(pos_grid_l.shape[0])]

	elif ext_type == 'server':

		pos_grid_l = np.load('/oak/stanford/schools/ees/beroza/imcbrear/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver)['pos_grid_l']
		pos_grid_l = [torch.Tensor(pos_grid_l[j]) for j in range(pos_grid_l.shape[0])]

	return pos_grid_l

def apply_models_make_displacement_prediction_heterogeneous(inpt, queries, m, m1, m2, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu'):

	## Based on inpt, create the (scaled) input and mask vectors, batch and batch query vectors,
	## Then apply predictions.

	## Lh, Lv, dz_val, queries all given in absolute unit scale, and use scale_val to normalize

	n_nodes_grid = pos_grid.shape[0]
	batch_index = torch.zeros(n_nodes_grid).long().to(device) # *j for j in range(n_batch)]).long().to(device)
	batch_index_query = torch.zeros(queries.shape[0]).long().to(device) # *j for j in range(n_batch)]).long().to(device)

	# Ra_val, Rb_val, ra2d_val, dz_val, thetax_val, thetaz_val = inpt

	# Lh_val, Lv_val = Lhv_scale
 
	one_vec = np.ones(75).reshape(1,-1)
	one_vec[0,0] = scale_val ## Have to divide the first entry, dz, by scale_val

	# signal_val = torch.Tensor(np.array([Ra_val, Rb_val, ra2d_val, dz_val/scale_val, thetax_val, thetaz_val]).reshape(1,-1)/norm_vals).to(device)

	queries_inpt = torch.Tensor(queries/scale_val).to(device)

	signal_inpt_unscaled = torch.cat((pos_grid, torch.Tensor((inpt.reshape(1,-1)/one_vec)/norm_vals).to(device)*torch.ones(n_nodes_grid,75).to(device)), dim = 1)

	mask_inpt = signal_inpt_unscaled[:,3::]

	# Predict Lh and Lv
	pred_lhv = m2(signal_inpt_unscaled, mask_inpt, queries_inpt, A_edges, A_edges_c, pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	pos = torch.Tensor(np.array([pred_lhv[0,0]/2.0, pred_lhv[0,0]/2.0, pred_lhv[0,1]]).reshape(1,-1)).to(device) # *pos_grid

	# pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_c_batch, pos_batch, batch_index, batch_index_query, n_nodes_grid)

	# signal_inpt = torch.cat((pos*pos_grid, signal_val*torch.ones(n_nodes_grid,6).to(device)), dim = 1)

	signal_inpt = torch.cat((pos*pos_grid, torch.Tensor((inpt.reshape(1,-1)/one_vec)/norm_vals).to(device)*torch.ones(n_nodes_grid,75).to(device)), dim = 1)

	A_edges_scaled = make_spatial_graph(pos*pos_grid)

	# Predict displacement
	pred = m(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	# Predict norm
	pred_norm = m1(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	return pred*np.power(10.0, pred_norm) # .cpu().detach().numpy()