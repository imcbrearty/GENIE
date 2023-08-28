
from joblib import Parallel, delayed
import multiprocessing

import yaml
from utils import *


print('can increase n_nodes_grid size')

config = load_config('config.yaml')

eps_extra = 0.1 # pos_grid_l
scale_up = 10000.0

n_grids = config['number_of_grids']
n_nodes_grid = config['number_of_spatial_nodes']
extend_grids = config['extend_grids']

pos_grid_l = []

# moi

print('Should shift the resulting grids slightly, proportional to the node spacing, to avoid bias due to the consistent grid spacing')

make_grid = True

use_parallel = True

num_cores = 5

save_grid = True

ext_type = 'remote'

save_grid_ver = 1

if make_grid == True:

	# for i in range(ip_unique.shape[0]):

	if use_parallel == False:

		for n in range(n_grids):

			offset_x = scale_up*np.array([-1.0 - eps_extra, -1.0 - eps_extra, -1.0 - eps_extra]).reshape(1,-1)
			scale_x = scale_up*np.array([2.0 + 2.0*eps_extra, 2.0 + 2.0*eps_extra, 1.0 + 2.0*eps_extra]).reshape(1,-1)

			if extend_grids == True:

				extend1, extend2, extend3, extend4 = (np.random.rand(4) - 0.5) * 0.1 * scale_up
				extend5 = (np.random.rand() - 0.5) * 0.1 * scale_up
		        
				offset[0, 0] += extend1
				offset[0, 1] += extend2
				scale[0, 0] += extend3
				scale[0, 1] += extend4
				offset[0, 2] += extend5		

			# n_steps = 8000

			pos_grid = kmeans_packing_logarithmic(scale_x, offset_x, 3, n_nodes_grid, n_batch = 10000, n_steps = 8000, n_sim = 1, lr = 0.005)[0]/scale_up # .to(device)

			print('\n Finished grid %d \n'%i)

			pos_grid_l.append(pos_grid)

			# edges_grid = make_spatial_graph(pos_grid, k = 15)
			# edges_grid_offsets = pos_grid[edges_grid[0]] - pos_grid[edges_grid[1]]

	else:

		scale_x_list = []
		offset_x_list = []

		for n in range(n_grids):

			offset_x = scale_up*np.array([-1.0 - eps_extra, -1.0 - eps_extra, -1.0 - eps_extra]).reshape(1,-1)
			scale_x = scale_up*np.array([2.0 + 2.0*eps_extra, 2.0 + 2.0*eps_extra, 1.0 + 2.0*eps_extra]).reshape(1,-1)

			if extend_grids == True:

				extend1, extend2, extend3, extend4 = (np.random.rand(4) - 0.5) * 0.1 * scale_up
				extend5 = (np.random.rand() - 0.5) * 0.1 * scale_up
		        
		        ## Still centered
				offset_x[0, 0] -= extend3/2.0
				offset_x[0, 1] -= extend4/2.0
				scale_x[0, 0] += extend3
				scale_x[0, 1] += extend4
				# offset_x[0, 2] += extend5				

			scale_x_list.append(scale_x)
			offset_x_list.append(offset_x)

		pos_grid_l = kmeans_packing_logarithmic_parallel(num_cores, scale_x_list, offset_x_list, 3, n_nodes_grid, n_batch = 10000, n_steps = 8000, n_sim = 1, lr = 0.005) # [0] # /scale_up
		for n in range(n_grids):
			pos_grid_l[n] = pos_grid_l[n]/scale_up


if (make_grid == True)*(save_grid == True):

	if ext_type == 'local':
		np.savez_compressed('D:/Projects/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%save_grid_ver, pos_grid_l = [pos_grid_l[j] for j in range(len(pos_grid_l))])
	elif ext_type == 'remote':
		np.savez_compressed('/work/wavefront/imcbrear/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%save_grid_ver, pos_grid_l = [pos_grid_l[j] for j in range(len(pos_grid_l))])
	elif ext_type == 'server':
		np.savez_compressed('/oak/stanford/schools/ees/beroza/imcbrear/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%save_grid_ver, pos_grid_l = [pos_grid_l[j] for j in range(len(pos_grid_l))])



## Make cayley graphs
imax = 0
inc = 3
while imax < n_nodes_grid:
	## Build cayley graph
	A_edges_c = make_cayleigh_graph(inc)
	imax = A_edges_c.max() + 1
	inc += 1

	print('Cayley graph step %d, node max %d'%(inc, imax))

A_edges_c = subgraph(torch.arange(n_nodes_grid), torch.Tensor(A_edges_c.T).long().flip(0).contiguous())[0].cpu().detach().numpy() # .to(device)

if ext_type == 'local':
	np.savez_compressed('D:/Projects/Laplace/ApplyHeterogenous/Grids/cayley_grid_heterogenous_ver_%d.npz'%save_grid_ver, A_edges_c = A_edges_c)
elif ext_type == 'remote':
	np.savez_compressed('/work/wavefront/imcbrear/Laplace/ApplyHeterogenous/Grids/cayley_grid_heterogenous_ver_%d.npz'%save_grid_ver, A_edges_c = A_edges_c)
elif ext_type == 'server':
	np.savez_compressed('/oak/stanford/schools/ees/beroza/imcbrear/Laplace/ApplyHeterogenous/Grids/cayley_grid_heterogenous_ver_%d.npz'%save_grid_ver, A_edges_c = A_edges_c)
