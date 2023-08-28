
import yaml
from utils import *
from module import *
from shape_function import *
import pathlib

## Load data script

def load_batch_data_Lh_and_Lv(st_files, grid_ind):

	if len(grid_ind) == 1:
		grid_ind = grid_ind*np.ones(len(st_files))

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_c_slice = []
	trgt_slice = []

	len_files = len(st_files)

	for i in range(len_files):

		z = h5py.File(st_files[i], 'r')

		# Trgts
		Lh = z['input/Lh'][:][0][0]/scale_val
		Lv = z['input/Lv'][:][0][0]/scale_val

		# Inpts
		dz = z['input/dz'][:][0][0]/scale_val # depth? (scalar)
		Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1)
		NormF = z['input/normF'][:].reshape(-1)
		RMax = z['input/rmax'][:].reshape(-1)

		if norm_version == 1:
			norm_val_slice = np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]]))
		elif norm_version == 2:
			norm_val_slice = np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max()

		assert((np.log10(norm_val_slice) > min_val)*(np.log10(norm_val_slice) < max_val))

		Ux = z['output/Ux'][0,0:n_samples]/norm_val_slice # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
		Uy = z['output/Uy'][0,0:n_samples]/norm_val_slice # [1 x 1440]
		Uz = z['output/Uz'][0,0:n_samples]/norm_val_slice # [1 x 1440]
		X = z['output/X'][0,0:n_samples]/scale_val # [1 x 1440]
		Y = z['output/Y'][0,0:n_samples]/scale_val # [1 x 1440]
		Z = z['output/Z'][0,0:n_samples]/scale_val # [1 x 1440]
		z.close()

		trgt = np.array([Lh, Lv]).reshape(1,-1)
		x_query = np.concatenate((X[:,None], Y[:,None], Z[:,None]), axis = 1)
		inpt = np.concatenate([np.array([dz]), Fs.reshape(-1), NormF, RMax], axis = 0).reshape(1,-1)/norm_vals

		## Scale spatial graphs (but for Lh and Lv model it is constant) 
		pos = np.array([1.0, 1.0, 1.0]).reshape(1,-1) # .to(device) # *pos_grid
		pos = torch.Tensor(pos*pos_grid_l[grid_ind[i]]).to(device)

		pos_slice.append(pos)
		signal_slice.append(torch.cat(( pos, (torch.Tensor(inpt)*torch.ones(n_nodes_grid, n_features)).to(device) ), dim = 1).to(device))
		query_slice.append(torch.Tensor(x_query).to(device))
		# edges_slice = [A_edges_l[j] for j in igrids]
		edges_slice.append(make_spatial_graph(pos, k_pos = k_spc_edges, device = device))
		edges_c_slice.append(A_edges_c)
		trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, query_slice, edges_slice, edges_c_slice, trgt_slice


# def batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, node_ind_max, device):

# 	inpt_batch = torch.vstack(signal_slice).to(device)
# 	mask_batch = inpt_batch[:,3::] # Only select non-position points for mask
# 	pos_batch = inpt_batch[:,0:3]
# 	query_batch = torch.Tensor(np.vstack(query_slice)).to(device)
# 	edges_batch = torch.cat([edges_slice[j] + j*node_ind_max for j in range(len(edges_slice))], dim = 1).to(device)
# 	edges_batch_c = torch.cat([edges_c_slice[j] + j*node_ind_max for j in range(len(edges_c_slice))], dim = 1).to(device)
# 	trgt_batch = torch.Tensor(np.vstack(trgt_slice)).to(device)

# 	return inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch

userhome = os.path.expanduser('~')
if '/home/users/imcbrear' in userhome:
	ext_type = 'server'
elif 'imcbrear' in userhome:
	ext_type = 'remote'
elif ('imcbr' in userhome) and ('C:\\' in userhome):
	ext_type = 'local'

path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_model = path_to_file + seperator + 'TrainedModels' + seperator

## Load parameters
config = load_config('config.yaml')
n_grids = config['number_of_grids']
n_nodes_grid = config['number_of_spatial_nodes']
scale_val = config['scale_val']
k_spc_edges = config['k_spc_edges']
norm_version = config['norm_version']
n_samples = config['n_samples']
n_cayley = config['n_cayley']
use_only_complex = config['use_only_complex']
path_to_data = config['path_to_data']
min_val = config['min_val']
max_val = config['max_val']
n_ver_save = config['n_ver_save']
n_epochs = config['n_epochs']
n_batch = config['n_batch']
device = config['device']
n_features = config['n_features']

z = np.load(path_to_data + 'training_files_within_norm_bounds.npz')
st, ivald, itrain = z['st'], z['itrain'], z['ivald']
z.close()

z = np.load(path_to_data + 'training_files_complex_max_values.npz')
norm_vals_complex_max = z['norm_vals_max']
z.close()

z = np.load(path_to_data + 'training_files_spheroid_max_values.npz')
norm_vals_spheroid_max = z['norm_vals_max']
z.close()


if use_only_complex == True:
	# ifind = np.where(['sph_composite' in s for s in st])[0]
	# st = np.delete(st, ifind, axis = 0)

	st = list(filter(lambda x: 'sph_composite' not in x, st))
	itrain = np.sort(np.random.choice(len(st), size = int(0.9*len(st)), replace = False))
	ivald = np.delete(np.arange(len(st)), itrain, axis = 0)
	norm_vals = np.copy(norm_vals_complex_max)

elif use_only_spheroids == True:

	st = list(filter(lambda x: 'sph_composite' in x, st))
	itrain = np.sort(np.random.choice(len(st), size = int(0.9*len(st)), replace = False))
	ivald = np.delete(np.arange(len(st)), itrain, axis = 0)
	norm_vals = np.copy(norm_vals_spheroid_max)

## Overwrite zero entries
norm_vals[norm_vals == 0] = 1.0

## Load spatial graphs
n_ver_grid = 1
pos_grid_l = np.load(path_to_file + seperator + 'Grids' + seperator + 'spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver_grid)['pos_grid_l']
A_edges_l = [make_spatial_graph(torch.Tensor(pos_grid_l[i]).to(device), k_pos = k_spc_edges, device = device) for i in range(n_grids)]
print('Note: for training non Lh and Lv model, must make the spatial graphs after saling by predicted Lh and Lv')

## Build cayley graph
# A_edges_c = make_cayleigh_graph(n_cayley)
# A_edges_c = subgraph(torch.arange(n_nodes_grid), torch.Tensor(A_edges_c.T).long().flip(0).contiguous())[0].to(device)

A_edges_c = torch.Tensor(np.load(path_to_file + seperator + 'Grids' + seperator + 'cayley_grid_heterogenous_ver_%d.npz'%n_ver_grid)['A_edges_c']).to(device).long()

## Make batch vectors
batch_index = torch.hstack([torch.ones(n_nodes_grid)*j for j in range(n_batch)]).long().to(device)
batch_index_query = torch.hstack([torch.ones(n_samples)*j for j in range(n_batch)]).long().to(device)
batch_zero = torch.zeros(n_nodes_grid).long().to(device)
batch_query_zero = torch.zeros(n_samples).long().to(device)

## Train

device = torch.device(device)

m = GNN_Network_Lh_and_Lv(device = device).to(device)

optimizer = optim.Adam(m.parameters(), lr = 0.001)
schedular = StepLR(optimizer, 25000, gamma = 0.9)

loss_func = nn.MSELoss()

n_restart = None
if n_restart is not None:
	n_begin = n_restart

	m.load_state_dict(torch.load(path_to_model + 'trained_model_heterogenous_Lh_and_Lv_prediction_ver_%d.h5'%n_ver_save))
	optimizer.load_state_dict(torch.load(path_to_model + 'trained_model_heterogenous_Lh_and_Lv_prediction_optimizer_ver_%d.h5'%n_ver_save))

else:
	n_begin = 0


n_train = len(itrain)
n_vald = len(ivald)

losses = []
losses_vald = []

n_vald_steps = 10
n_save_steps = 1000

for i in range(n_begin, n_epochs):

	optimizer.zero_grad()

	isample = np.sort(np.random.choice(itrain, size = n_batch))

	st_files = [st[isample[j]] for j in range(n_batch)]

	grid_ind = np.random.choice(n_grids, size = n_batch)

	pos_slice, signal_slice, query_slice, edges_slice, edges_c_slice, trgt_slice = load_batch_data_Lh_and_Lv(st_files, grid_ind)

	inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

	pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_batch_c, pos_batch, batch_index, batch_index_query, n_nodes_grid)

	loss = loss_func(pred/10.0, trgt_batch/10.0)

	loss.backward()

	optimizer.step()

	schedular.step()

	losses.append(loss.item())

	print('%d %0.4f'%(i, loss.item()))

	if np.mod(i, n_vald_steps) == 0:

		with torch.no_grad():

			isample = np.sort(np.random.choice(ivald, size = n_batch))

			st_files = [st[isample[j]] for j in range(n_batch)]

			grid_ind = np.random.choice(n_grids, size = n_batch)

			pos_slice, signal_slice, query_slice, edges_slice, edges_c_slice, trgt_slice = load_batch_data_Lh_and_Lv(st_files, grid_ind)

			inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

			pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_batch_c, pos_batch, batch_index, batch_index_query, n_nodes_grid)

			loss = loss_func(pred/10.0, trgt_batch/10.0)

			losses_vald.append(loss.item())

			print('%d %0.4f (Vald)'%(i, loss.item()))

	if np.mod(i, n_save_steps) == 0:

		torch.save(m.state_dict(), path_to_model + 'trained_model_heterogenous_displacement_prediction_step_%d_ver_%d.h5'%(i, n_ver_save))
		torch.save(optimizer.state_dict(), path_to_model + 'trained_model_heterogenous_displacement_prediction_optimizer_step_%d_ver_%d.h5'%(i, n_ver_save))
		np.savez_compressed(path_to_model + 'example_displacement_prediction_heterogenous_displacement_prediction_step_%d_ver_%d.npz'%(i, n_ver_save), pred = pred.cpu().detach().numpy(), trgt = trgt_batch.cpu().detach().numpy(), losses = losses, losses_vald = losses_vald, itrain = itrain, ivald = ivald)