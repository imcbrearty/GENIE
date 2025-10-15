

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from obspy.core import UTCDateTime
from scipy.spatial import cKDTree
from torch_scatter import scatter
from torch_geometric.utils import degree, subgraph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
from torch import nn, optim
import glob
import h5py
import torch
import pathlib

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

## Projections
path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

z = np.load(path_to_file + 'California_stations.npz')
rbest, mn = z['rbest'], z['mn']
z.close()

earth_radius = 6378137.0
ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean

ftrns1_diff = lambda x: (rbest_cuda @ (lla2ecef_diff(x, device = device) - mn_cuda).T).T # just subtract mean
ftrns2_diff = lambda x: ecef2lla_diff((rbest_cuda.T @ x.T).T + mn_cuda, device = device) # just subtract mean

if torch.cuda.is_available() == True:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


## Load catalog
n_ver = 1
## Plot with and without station corrections
z = h5py.File(path_to_file + 'California_catalog_ver_%d.hdf5'%n_ver, 'r')
Times = z['Times'][:]
srcs = z['srcs'][:]
mags = z['mags'][:]
srcs_ref = z['srcs_ref'][:]
mags_ref = srcs_ref[:,4] # z['mag_ref'][:]
matches = z['Matches'][:]
t0 = UTCDateTime(2000, 1, 1)
times = np.array([((UTCDateTime(int(Times[j,0]), int(Times[j,1]), int(Times[j,2])) + srcs[j,3]) - t0)/(3600*24) for j in range(len(Times))])
z.close()

plt.figure()
a = plt.hexbin(srcs[:,1], srcs[:,0], gridsize = 150, bins = 'log')
pos = np.flip(a.get_offsets(), axis = 1)
counts = a.get_array().data

min_counts = 20
ifind = np.where(counts > min_counts)[0]
plt.figure()
plt.scatter(pos[ifind,1], pos[ifind,0], c = counts[ifind])

## Fit clusters
n_clusters = 300
clusters = np.flip(KMeans(n_clusters = n_clusters).fit(np.flip(pos[ifind], axis = 1)).cluster_centers_, axis = 1)
clusters = np.concatenate((clusters, np.zeros((len(clusters),1))), axis = 1)

## Input features:
## source location
## source magnitude
## Node location (embedding)
## Number of events within radius of node
## Maximum magnitude of events within radius of node
## Events over ~3 month interval
## Prediction target, max magnitude within radius of node, in next ~5 - 10 days
## Could also be counts of events > M amount

time_window = 30 ## Use all events within last 30 days
time_window_predict = 5.0 ## Predict over the next 5 day window
knn_edges = 30 ## Connect all events to 50 nearest neighbors
knn_edges_srcs = 30 ## Connect all grid nodes to 30 nearest grid nodes
radius_product = 30 ## 30 km radius to link events to grid nodes
distance_weights = [1.0, 10.0] ## 1 km is 1 km, 1 day is ~10 km
radius_predict = 10.0 ## 10 km radius to assign events to max magnitudes
## Note: could make radius of connection or edges adaptive to magnitude

# moi

def build_input_graph(times, srcs, mags, clusters, scale_pos = 100.0, scale_counts = 100.0, min_trgt_mag = 0.0, vald_win = 365.0, use_vald = False):

	if use_vald == False:
		t0 = np.random.uniform(times.min() + time_window, times.max() - time_window_predict - vald_win)
	else:
		t0 = np.random.uniform(times.max() - time_window_predict - vald_win, times.max() - time_window_predict)


	ifind = np.where((times < t0)*(times > (t0 - time_window)))[0]
	ifind1 = np.where((times >= t0)*(times < (t0 + time_window_predict)))[0]

	## Build cluster-cluster edges
	# edges_clusters = cKDTree(ftrns1(clusters)).query(ftrns1(clusters), k = knn_edges_srcs + 1)[1][:,1::]
	edges_clusters = cKDTree(ftrns1(clusters)).query(ftrns1(clusters), k = knn_edges_srcs)[1]
	edges_clusters = torch.Tensor(np.hstack([np.concatenate((edges_clusters[j,:].reshape(1,-1), j*np.ones((1,knn_edges_srcs))), axis = 0) for j in range(len(clusters))])).long().to(device)

	## Build event-event edges
	embed_pos = np.concatenate((distance_weights[0]*ftrns1(srcs[ifind])/1000.0, distance_weights[1]*(times[ifind] - t0).reshape(-1,1)), axis = 1)
	# edges_events = cKDTree(embed_pos).query(embed_pos, k = knn_edges + 1)[1][:,1::]
	edges_events = cKDTree(embed_pos).query(embed_pos, k = knn_edges)[1] # [:,1::]
	edges_events = torch.Tensor(np.hstack([np.concatenate((edges_events[j,:].reshape(1,-1), j*np.ones((1,knn_edges))), axis = 0) for j in range(len(embed_pos))])).long().to(device)
	embed_pos = embed_pos/scale_pos
	embed_clusters = (distance_weights[0]*ftrns1(clusters)/1000.0)/scale_pos # , distance_weights[1]*(times[ifind] - t0).reshape(-1,1)), axis = 1)


	## Build product pairs
	product_pairs = cKDTree(ftrns1(srcs[ifind])).query_ball_point(ftrns1(clusters), r = radius_product*1000.0)
	product_pairs = torch.Tensor(np.hstack([np.concatenate((np.array(product_pairs[j]).reshape(1,-1), j*np.ones((1,len(product_pairs[j])))), axis = 0) for j in range(len(product_pairs))])).long().to(device)
	cnt_inpts = scatter(torch.ones(product_pairs.shape[1],1).to(device), product_pairs[1], dim = 0, dim_size = n_clusters, reduce = 'sum')
	mag_inpts = scatter(torch.Tensor(mags[ifind]).to(device)[product_pairs[0]].reshape(-1,1), product_pairs[1], dim = 0, dim_size = n_clusters, reduce = 'max')
	mag_inpts.clamp(min = min_trgt_mag)
	cnt_inpts = cnt_inpts/scale_counts

	## Build target max magnitude prediction
	product_pairs_target = cKDTree(ftrns1(srcs[ifind1])).query_ball_point(ftrns1(clusters), r = radius_predict*1000.0)
	product_pairs_target = torch.Tensor(np.hstack([np.concatenate((np.array(product_pairs_target[j]).reshape(1,-1), j*np.ones((1,len(product_pairs_target[j])))), axis = 0) for j in range(len(product_pairs_target))])).long().to(device)
	A_cluster_in_event = torch.Tensor(1.0*product_pairs + 0.0).long().to(device)
	cnt_targets = scatter(torch.ones(product_pairs_target.shape[1],1).to(device), product_pairs_target[1], dim = 0, dim_size = n_clusters, reduce = 'sum')
	mag_targets = scatter(torch.Tensor(mags[ifind1]).to(device)[product_pairs_target[0]].reshape(-1,1), product_pairs_target[1], dim = 0, dim_size = n_clusters, reduce = 'max')
	mag_targets.clamp(min = min_trgt_mag)
	cnt_targets = cnt_targets/scale_counts

	## Build product edges

	A_prod_events_events = []
	A_prod_clusters_clusters = []

	## product_pairs_target
	tree_cluster_in_event = cKDTree(A_cluster_in_event[0].reshape(-1,1).cpu().detach().numpy())
	lp_fixed_events = tree_cluster_in_event.query_ball_point(np.arange(embed_pos.shape[0]).reshape(-1,1), r = 0)
	## cum_count_degree_of_src_nodes = 

	degree_of_src_nodes = degree(A_cluster_in_event[1], num_nodes = len(clusters))
	cum_count_degree_of_src_nodes = np.concatenate((np.array([0]), np.cumsum(degree_of_src_nodes.cpu().detach().numpy())), axis = 0).astype('int')
	n_events = len(embed_pos)


	event_ind_lists = []
	for i in range(clusters.shape[0]):
		ind_list = -1*np.ones(n_events)
		ind_list[A_cluster_in_event[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]].cpu().detach().numpy()] = np.arange(degree_of_src_nodes[i].item())
		event_ind_lists.append(ind_list)
	event_ind_lists = np.hstack(event_ind_lists).astype('int')


	tree_clusters_in_prod = cKDTree(A_cluster_in_event[1].cpu().detach().numpy()[:,None])
	lp_cluster_in_prod = tree_clusters_in_prod.query_ball_point(np.arange(clusters.shape[0])[:,None], r = 0)
	A_cluster_in_prod = torch.Tensor(np.hstack([np.concatenate((np.array(lp_cluster_in_prod[j]).reshape(1,-1), j*np.ones(len(lp_cluster_in_prod[j])).reshape(1,-1)), axis = 0) for j in range(clusters.shape[0])])).long().to(device)


	for i in range(clusters.shape[0]):
		# slice_edges = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, relabel_nodes = False)[0]
		slice_edges = subgraph(A_cluster_in_event[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], edges_events, relabel_nodes = True)[0]
		A_prod_events_events.append(slice_edges + cum_count_degree_of_src_nodes[i])

	for i in range(n_events):
	
		slice_edges = subgraph(A_cluster_in_event[1,np.array(lp_fixed_events[i])], edges_clusters, relabel_nodes = False)[0].cpu().detach().numpy()

		## This can happen when a station is only linked to one source
		if slice_edges.shape[1] == 0:
			continue

		shift_ind = event_ind_lists[slice_edges*n_events + i]
		assert(shift_ind.min() >= 0)
		## For each source, need to find where that station index is in the "order" of the subgraph Cartesian product
		A_prod_clusters_clusters.append(torch.Tensor(cum_count_degree_of_src_nodes[slice_edges] + shift_ind).to(device))

	## Make cartesian product graphs
	A_prod_events_events = torch.hstack(A_prod_events_events).long()
	A_prod_clusters_clusters = torch.hstack(A_prod_clusters_clusters).long()
	isort = np.lexsort((A_prod_clusters_clusters[0].cpu().detach().numpy(), A_prod_clusters_clusters[1].cpu().detach().numpy())) # Likely not actually necessary
	A_prod_clusters_clusters = A_prod_clusters_clusters[:,isort]

	## Build product input features
	embed_pos = torch.Tensor(embed_pos).to(device)
	embed_clusters = torch.Tensor(np.concatenate((embed_clusters, np.zeros((len(embed_clusters),1))), axis = 1)).to(device)
	inpt = torch.cat((embed_clusters[A_cluster_in_event[1]], embed_pos[A_cluster_in_event[0]], cnt_inpts[A_cluster_in_event[1]], mag_inpts[A_cluster_in_event[1]]), dim = 1)
	assert(len(inpt) == A_cluster_in_event.shape[1])

	# assert(A_cluster_in_prod[1].max().item() == (len(clusters) - 1))

	## Merge per-events features
	## Bin summary features
	## Embedded event position features
	## Embedded grid position features

	return inpt, embed_pos, embed_clusters, mag_targets, edges_events, edges_clusters, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event


## Define GNN Architecture

class DataAggregation(MessagePassing): # make equivelent version with sum operations.
	def __init__(self, in_channels, out_channels, n_hidden = 30, scale_rel = 1.0, n_dim = 4, n_dim_mask = 2, ndim_proj = 4):
		super(DataAggregation, self).__init__('mean') # node dim
		## Use two layers of SageConv.
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

		self.scale_rel = scale_rel
		self.merge_edges = nn.Sequential(nn.Linear(n_hidden + ndim_proj, n_hidden), nn.PReLU())

	def forward(self, tr, mask, A_in_sta, A_in_src, A_src_in_sta, pos_loc, pos_src):

		tr = torch.cat((tr, mask), dim = -1)
		tr = self.activate(self.init_trns(tr))

		# embed_sta_edges = self.fproj_edges_sta(pos_loc/1e6)

		pos_rel_sta = (pos_loc[A_src_in_sta[0][A_in_sta[0]]]/1.0 - pos_loc[A_src_in_sta[0][A_in_sta[1]]]/1.0)/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)
		pos_rel_src = (pos_src[A_src_in_sta[1][A_in_src[0]]]/1.0 - pos_src[A_src_in_sta[1][A_in_src[1]]]/1.0)/self.scale_rel # , self.fproj_recieve(pos_i/1e6), self.fproj_send(pos_j/1e6)), dim = 1)

		## Could add binary edge type information to indicate data type
		tr1 = self.l1_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate11(tr), edge_attr = pos_rel_sta), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l1_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate12(tr), edge_attr = pos_rel_src), mask), dim = 1))
		tr = self.activate1(torch.cat((tr1, tr2), dim = 1))

		tr1 = self.l2_t1_2(torch.cat((tr, self.propagate(A_in_sta, x = self.activate21(self.l2_t1_1(tr)), edge_attr = pos_rel_sta), mask), dim = 1)) # could concatenate edge features here, and before.
		tr2 = self.l2_t2_2(torch.cat((tr, self.propagate(A_in_src, x = self.activate22(self.l2_t2_1(tr)), edge_attr = pos_rel_src), mask), dim = 1))
		tr = self.activate2(torch.cat((tr1, tr2), dim = 1))

		return tr # the new embedding.

	def message(self, x_j, edge_attr):

		return self.merge_edges(torch.cat((x_j, edge_attr), dim = 1)) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, ndim_mask = 11, ndim_edges = 4, scale_rel = 1.0):
		super(BipartiteGraphOperator, self).__init__('add') # add
		# include a single projection map
		self.fc1 = nn.Sequential(nn.Linear(ndim_in + ndim_edges + ndim_mask, ndim_in), nn.PReLU(), nn.Linear(ndim_in, ndim_in))
		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.
		self.scale_rel = scale_rel

	def forward(self, x, mask, A_src_in_edges, A_src_in_sta, locs_cart, src_cart):

		N = x.shape[0]
		M = src_cart.shape[0]

		# print('Bipartite Aggregation')

		return self.activate2(self.fc2(self.propagate(A_src_in_edges, x = torch.cat((x, mask), dim = 1), pos = (locs_cart[A_src_in_sta[0]], src_cart), size = (N, M))))

	def message(self, x_j, pos_i, pos_j):

		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.scale_rel), dim = 1)))

# class BipartiteGraphOperatorSta(MessagePassing):
# 	def __init__(self, ndim_in, ndim_out, ndim_mask = 11, ndim_edges = 3, scale_rel = 30e3):
# 		super(BipartiteGraphOperatorSta, self).__init__('mean') # add
# 		# include a single projection map
# 		self.fc1 = nn.Sequential(nn.Linear(ndim_in + ndim_edges + ndim_mask, ndim_in), nn.PReLU(), nn.Linear(ndim_in, ndim_in))
# 		self.fc2 = nn.Linear(ndim_in, ndim_out) # added additional layer

# 		self.activate1 = nn.PReLU() # added activation.
# 		self.activate2 = nn.PReLU() # added activation.
# 		self.scale_rel = scale_rel

# 	def forward(self, x, mask, A_src_in_edges, A_src_in_sta, locs_cart, src_cart):

# 		N = x.shape[0]
# 		M = locs_cart.shape[0]

# 		# print('Bipartite Aggregation')

# 		return self.activate2(self.fc2(self.propagate(A_src_in_edges, x = torch.cat((x, mask), dim = 1), pos = (src_cart[A_src_in_sta[1]], locs_cart), size = (N, M))))

# 	def message(self, x_j, pos_i, pos_j):

# 		return self.activate1(self.fc1(torch.cat((x_j, (pos_i - pos_j)/self.scale_rel), dim = 1)))

class GNN_predict(nn.Module):

	def __init__(self, ftrns1, ftrns2, n_inpt = 10, n_mask = 10, n_hidden = 20, n_embed = 10, n_embed_vec = 10, n_cluster = None, device = 'cuda'):

		super(GNN_predict, self).__init__()

		if n_cluster is not None:
			n_inpt = n_inpt + n_embed_vec
			n_mask = n_mask + n_embed_vec

		use_relative = True
		if use_relative == True:
			n_inpt += 4
			n_mask += 4

		self.use_relative = use_relative

		self.DataAggregation1 = DataAggregation(n_inpt, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation2 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation3 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation4 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15
		self.DataAggregation5 = DataAggregation(30, 15, n_dim_mask = n_embed).to(device) # output size is latent size for (half of) bipartite code # , 15

		## Could make attention layer for read out (if limit the number of neighbors)
		self.BipartiteReadOut = BipartiteGraphOperator(30, 15, ndim_mask = n_embed)
		# self.BipartiteReadOut2 = BipartiteGraphOperatorSta(30, 15, ndim_mask = n_embed)

		self.embed_inpt = nn.Sequential(nn.Linear(n_mask, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_embed))
		self.proj = nn.Sequential(nn.Linear(15, 30), nn.PReLU(), nn.Linear(30, 2))
		if n_cluster is not None:
			self.n_cluster = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(n_cluster, n_embed_vec))).to(device)
			self.use_cluster = True
		else:
			self.use_cluster = False

		# self.proj = nn.Sequential(nn.Linear(n_read_out, 30), nn.PReLU(), nn.Linear(30, 3))
		# self.proj_t = nn.Sequential(nn.Linear(n_read_out, 15), nn.PReLU(), nn.Linear(15, 1))

		# if use_sta_corr == True:
		# self.proj_c = nn.Sequential(nn.Linear(15, 15), nn.PReLU(), nn.Linear(15, 2))

		# if use_mask == True:
		# 	self.proj_mask = nn.Sequential(nn.Linear(30, 15), nn.PReLU(), nn.Linear(15, 2))

		# self.use_memory = use_memory
		# self.use_sta_corr = use_sta_corr
		# self.scale = torch.Tensor([scale_fixed]).to(device)
		self.device = device

	def forward(self, x, mask, A_in_events, A_in_clusters, A_src_in_product, A_src_in_sta, locs_cart, srcs_cart, memory = False):

		# if self.use_memory == True:
		# 	mask = self.embed_inpt(torch.cat((mask, memory[A_src_in_sta[1]]), dim = 1))
		# 	x = torch.cat((x, memory[A_src_in_sta[1]]), dim = 1)
		# else:
		if self.use_cluster == True:
			x = torch.cat((x, self.n_cluster[A_src_in_sta[1]]), dim = 1)
			mask = torch.cat((mask, self.n_cluster[A_src_in_sta[1]]), dim = 1)

		if self.use_relative == True:
			x = torch.cat((x, locs_cart[A_src_in_sta[0]] - srcs_cart[A_src_in_sta[1]]), dim = 1)
			mask = torch.cat((mask, locs_cart[A_src_in_sta[0]] - srcs_cart[A_src_in_sta[1]]), dim = 1)

		mask = self.embed_inpt(mask)

		x = self.DataAggregation1(x, mask, A_in_events, A_in_clusters, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation2(x, mask, A_in_events, A_in_clusters, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation3(x, mask, A_in_events, A_in_clusters, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation4(x, mask, A_in_events, A_in_clusters, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
		x = self.DataAggregation5(x, mask, A_in_events, A_in_clusters, A_src_in_sta, locs_cart, srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers

		x1 = self.BipartiteReadOut(x, mask, A_src_in_product, A_src_in_sta, locs_cart, srcs_cart)

		# if self.use_memory == True:
		# 	proj_memory = self.proj_memory(memory)
		# 	x1 = self.merge_data(torch.cat((x1, proj_memory), dim = 1)) ## Can add memory for station corrections as well

		return self.proj(x1) # self.proj_mask(x)

	# def forward_fixed(self, x, mask, memory = False):

	# 	if self.use_memory == True:
	# 		mask = self.embed_inpt(torch.cat((mask, memory[self.A_src_in_sta[1]]), dim = 1))
	# 		x = torch.cat((x, memory[self.A_src_in_sta[1]]), dim = 1)
	# 	else:
	# 		mask = self.embed_inpt(mask)

	# 	x = self.DataAggregation1(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
	# 	x = self.DataAggregation2(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
	# 	x = self.DataAggregation3(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
	# 	x = self.DataAggregation4(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers
	# 	x = self.DataAggregation5(x, mask, self.A_in_pick, self.A_in_src, self.A_src_in_sta, self.locs_cart, self.srcs_cart) # note by concatenating to downstream flow, does introduce some sensitivity to these aggregation layers

	# 	x1 = self.BipartiteReadOut1(x, mask, self.A_src_in_product, self.A_src_in_sta, self.locs_cart, self.srcs_cart)
	# 	x2 = self.BipartiteReadOut2(x, mask, self.A_sta_in_product, self.A_src_in_sta, self.locs_cart, self.srcs_cart)

	# 	if self.use_memory == True:
	# 		proj_memory = self.proj_memory(memory)
	# 		x1 = self.merge_data(torch.cat((x1, proj_memory), dim = 1)) ## Can add memory for station corrections as well

	# 	return self.scale*self.proj(x1), self.proj_t(x1), self.proj_c(x2), x # self.proj_mask(x)

	# def set_adjacencies(self, A_in_pick, A_in_src, A_src_in_product, A_sta_in_product, A_src_in_sta, locs_cart, srcs_cart): # phase_type = 'P'

	# 	self.A_in_pick = A_in_pick
	# 	self.A_in_src = A_in_src
	# 	self.A_src_in_product = A_src_in_product
	# 	self.A_sta_in_product = A_sta_in_product
	# 	self.A_src_in_sta = A_src_in_sta
	# 	self.locs_cart = locs_cart
	# 	self.srcs_cart = srcs_cart


## Sample catalog
inpt, embed_pos, embed_clusters, mag_targets, edges_events, edges_clusters, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event = build_input_graph(times, srcs, mags, clusters, scale_pos = 100.0, scale_counts = 100.0, min_trgt_mag = 0.0)

use_cluster = True

## Initilize model 
if use_cluster == True:
	m = GNN_predict(ftrns1, ftrns2, n_cluster = len(clusters), device = device).to(device)
else:
	m = GNN_predict(ftrns1, ftrns2, device = device).to(device)


optimizer = optim.Adam(m.parameters(), lr = 0.001)
loss_func = nn.HuberLoss() # nn.L1Loss()
loss_func1 = nn.BCELoss()

# moi

# use_cluster = True
# if use_cluster == True:
pred = m(inpt, inpt, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event, embed_pos, embed_clusters)
# else:
# pred = m(inpt, inpt, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event, embed_pos, embed_clusters)


losses = []
losses_vald = []
max_trgt = []
max_pred = []

n_batch = 10
n_epochs = 150001
n_vald = 50

use_zero_loss = True
load_data = True
weight_vec = 1.5

ext_load = '/scratch/users/imcbrear/California/FaultTrainingData/'
st_load = glob.glob(ext_load + '*.hdf5')
irand_choose = np.random.choice(len(st_load), size = int(0.8*len(st_load)), replace = False)
irand_vald = np.delete(np.arange(len(st_load)), irand_choose, axis = 0)
irand_choose = np.tile(irand_choose, 2*n_batch)
cnt = 0

n_restart = True
n_restart_step = 100000
if n_restart == False: n_restart_step = 0

# moi

for i in range(n_restart_step, n_epochs):

	if n_restart == True:

		path_save = path_to_file + 'FaultNetworkModels/'
		n_ver_save = 1

		m.load_state_dict(torch.load(path_save + 'trained_fault_network_model_step_%d_ver_%d.h5'%(n_restart_step, n_ver_save), map_location = device))
		optimizer.load_state_dict(torch.load(path_save + 'trained_fault_network_model_step_%d_ver_%d_optimizer.h5'%(n_restart_step, n_ver_save), map_location = device))
		losses = list(np.load(path_save + 'trained_fault_network_model_step_%d_ver_%d_losses.npz'%(n_restart_step, n_ver_save))['losses'])

	optimizer.zero_grad()

	loss_val = 0.0
	n_cnt = 0

	for j in range(n_batch):

		# if (load_data == False) or (np.mod(i, n_vald) == 0):
		if load_data == False:
			if (np.mod(i, n_vald) == 0):
				inpt, embed_pos, embed_clusters, mag_targets, edges_events, edges_clusters, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event = build_input_graph(times, srcs, mags, clusters, scale_pos = 100.0, scale_counts = 100.0, use_vald = True, min_trgt_mag = 0.0)
			else:
				inpt, embed_pos, embed_clusters, mag_targets, edges_events, edges_clusters, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event = build_input_graph(times, srcs, mags, clusters, scale_pos = 100.0, scale_counts = 100.0, min_trgt_mag = 0.0)

		else:

			if np.mod(i, n_vald) != 0:
				z = h5py.File(st_load[irand_choose[cnt]], 'r')
			else:
				z = h5py.File(st_load[np.random.choice(irand_vald)], 'r')				

			inpt = torch.Tensor(z['inpt'][:]).to(device)
			embed_pos = torch.Tensor(z['embed_pos'][:]).to(device)
			embed_clusters = torch.Tensor(z['embed_clusters'][:]).to(device)
			mag_targets = torch.Tensor(z['mag_targets'][:]).to(device)
			edges_events = torch.Tensor(z['edges_events'][:]).long().to(device)
			edges_clusters = torch.Tensor(z['edges_clusters'][:]).long().to(device)
			A_prod_events_events = torch.Tensor(z['A_prod_events_events'][:]).long().to(device)
			A_prod_clusters_clusters = torch.Tensor(z['A_prod_clusters_clusters'][:]).long().to(device)
			A_cluster_in_prod = torch.Tensor(z['A_cluster_in_prod'][:]).long().to(device)
			A_cluster_in_event = torch.Tensor(z['A_cluster_in_event'][:]).long().to(device)
			cnt += 1
			z.close()

		pred = m(inpt, inpt, A_prod_events_events, A_prod_clusters_clusters, A_cluster_in_prod, A_cluster_in_event, embed_pos, embed_clusters)

		if use_zero_loss == False:
			loss = loss_func(pred[:,0].reshape(-1,1), mag_targets)
		else:
			ifind = torch.where(mag_targets[:,0] > 0)[0]
			mask_trgt = (mag_targets[:,0] > 0).float()
			loss1 = loss_func1(torch.sigmoid(pred[:,0]), mask_trgt)

			# weight_vec = (mag_targets[ifind,0]**1.5).detach()
			weight_vec = torch.ones(len(ifind)).to(device)
			if len(ifind) > 0:
				weight_vec[torch.where(mag_targets[ifind,0] >= torch.quantile(mag_targets[ifind,0], 0.5))] = 1.5
				weight_vec[torch.where(mag_targets[ifind,0] >= torch.quantile(mag_targets[ifind,0], 0.75))] = 2.0

			loss2 = loss_func(weight_vec*pred[ifind,1], weight_vec*mag_targets[ifind,0])
			loss = loss1 + loss2


		if ((np.mod(i, n_vald) != 0))*(torch.isnan(loss) == 0):
			if j != (n_batch - 1):
				loss.backward(retain_graph = True)
			else:
				loss.backward(retain_graph = False)

		if (torch.isnan(loss) == 0):
			loss_val += loss.item() # /n_batch
			n_cnt += 1

	loss_val = loss_val/n_cnt

	if (np.mod(i, n_vald) != 0):
		optimizer.step()
		losses.append(loss_val)
		if len(ifind) > 0:
			iarg = np.argmax(mag_targets[ifind,0].cpu().detach().numpy())
			print('%d %0.8f (%0.3f, %0.3f)'%(i, loss_val, mag_targets[ifind[iarg],0].item(), pred[ifind[iarg],1].item()))
			max_trgt.append(mag_targets[ifind[iarg],0].item())
			max_pred.append(pred[ifind[iarg],1].item())
		else:
			print('%d %0.8f'%(i, loss_val))
			max_trgt.append(0.0)
			iarg = np.argmax(torch.sigmoid(pred[:,0]).cpu().detach().numpy())  # [ifind,0].cpu().detach().numpy())
			if torch.sigmoid(pred[iarg,0]).item() > 0.5:
				max_pred.append(pred[iarg,1].item())
			else:
				max_pred.append(0.0)
	else:
		losses_vald.append(loss_val)
		if len(ifind) > 0:
			iarg = np.argmax(mag_targets[ifind,0].cpu().detach().numpy())
			print('%d %0.8f (%0.3f, %0.3f) (Vald)'%(i, loss_val, mag_targets[ifind[iarg],0].item(), pred[ifind[iarg],1].item()))
		else:
			print('%d %0.8f'%(i, loss_val))

		# print('%d %0.8f (Vald)'%(i, loss_val))


	if (np.mod(i, 1000) == 0)*((i != n_restart_step) + (i == 0)):

		path_save = path_to_file + 'FaultNetworkModels/'
		n_ver_save = 1

		# srcs_perturbed = ftrns2_diff(srcs_slice_cart + pred).cpu().detach().numpy()
		# torch.save(m.state_dict(), 		# srcs_perturbed = ftrns2_diff(srcs_slice_cart + pred).cpu().detach().numpy()
		torch.save(m.state_dict(), path_save + 'trained_fault_network_model_step_%d_ver_%d.h5'%(i, n_ver_save))
		torch.save(optimizer.state_dict(), path_save + 'trained_fault_network_model_step_%d_ver_%d_optimizer.h5'%(i, n_ver_save))
		np.savez_compressed(path_save + 'trained_fault_network_model_step_%d_ver_%d_losses.npz'%(i, n_ver_save), losses = losses, losses_vald = losses_vald, max_trgt = max_trgt, max_pred = max_pred, pred = pred.cpu().detach().numpy(), mag_targets = mag_targets.cpu().detach().numpy(), clusters = clusters, embed_pos = embed_pos.cpu().detach().numpy())
		# torch.save(optimizer.state_dict(), path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_optimizer.h5'%(i, n_ver_load_files, n_ver_save))
		# np.savez_compressed(path_save + 'trained_station_correction_model_step_%d_data_%d_ver_%d_losses.npz'%(i, n_ver_load_files, n_ver_save), losses = losses, srcs_slice = srcs_slice, srcs_slice_perturbed = srcs_slice_perturbed, srcs = srcs, srcs_perturbed = srcs_perturbed, srcs_pred_perturb = srcs_pred_perturb, pred_t = pred_t.cpu().detach().numpy(), corrections_c = corrections_c, srcs_ref = srcs_ref, Matches = Matches, locs = locs, losses_abs = losses_abs, losses_sta = losses_sta, losses_cal = losses_cal, losses_cal_abs = losses_cal_abs, losses_double_diff = losses_double_diff, losses_diff = losses_diff)
		# print('saved model %s data %d step %d'%(n_ver_load_files, n_ver_save, i))



