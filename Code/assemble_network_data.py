
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from torch import optim, nn
from torch_cluster import knn ## Note torch_cluster should be installed automatically with pytorch geometric
import shutil
from scipy.spatial import cKDTree

## User: Input stations and spatial region
## (must have station and region files at
## (ext_dir + 'stations.npz'), and
## (ext_dir + 'region.npz')

ext_dir = 'D:/Projects/Mayotte/Mayotte/' ## Replace with absolute directory to location to setup folders, and where all the ".py" files from Github are located.
name_of_project = 'Mayotte' ## Replace with the name of project (a single word is prefered).

# Station file
# z = np.load(ext_dir + '%s_stations.npz'%name_of_project)
z = np.load(ext_dir + 'stations.npz')
locs, stas = z['locs'], z['stas']
z.close()

print('\n Using stations:')
print(stas)
print('\n Using locations:')
print(locs)

# Region file
# z = np.load(ext_dir + '%s_region.npz'%name_of_project)
z = np.load(ext_dir + 'region.npz', allow_pickle = True)
lat_range, lon_range, depth_range = z['lat_range'], z['lon_range'], z['depth_range'], 
deg_pad, num_grids, years = z['deg_pad'], z['num_grids'], z['years']
n_spatial_nodes = z['n_spatial_nodes']
load_initial_files = z['load_initial_files'][0]
use_pretrained_model = z['use_pretrained_model'][0]
z.close()
shutil.copy(ext_dir + 'region.npz', ext_dir + '%s_region.npz'%name_of_project)

with_density = None
# else, set with_density = srcs with srcs[:,0] == lat, srcs[:,1] == lon, srcs[:,2] == depth
## to preferentially focus the spatial graphs closer around reference sources. 

## Fit projection coordinates and create spatial grids

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

def lla2ecef_diff(p, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2])):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	p = p.detach().clone().float()
	pi = torch.Tensor([np.pi])
	p[:,0:2] = p[:,0:2]*torch.Tensor([pi/180.0, pi/180.0]).view(1,-1)
	N = a/torch.sqrt(1 - (e**2)*torch.sin(p[:,0])**2)
    # results:
	x = (N + p[:,2])*torch.cos(p[:,0])*torch.cos(p[:,1])
	y = (N + p[:,2])*torch.cos(p[:,0])*torch.sin(p[:,1])
	z = ((1-e**2)*N + p[:,2])*torch.sin(p[:,0])

	return torch.cat((x.view(-1,1), y.view(-1,1), z.view(-1,1)), dim = 1)

def ecef2lla_diff(x, a = torch.Tensor([6378137.0]), e = torch.Tensor([8.18191908426215e-2])):
	# x = x.astype('float')
	# https://www.mathworks.com/matlabcentral/fileexchange/7941-convert-cartesian-ecef-coordinates-to-lat-lon-alt
	pi = torch.Tensor([np.pi])
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

ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef_diff(x) - mn).T).T # just subtract mean
ftrns2 = lambda x, rbest, mn: ecef2lla_diff((rbest.T @ x.T).T + mn) # just subtract mean

## Unit lat, vertical vectors; point positive y, and outward normal
## mean centered stations. Keep the vertical depth, consistent.

print('\n Using domain')
print('\n Latitude:')
print(lat_range)
print('\n Longitude:')
print(lon_range)

fix_nominal_depth = False
if fix_nominal_depth == True:
	nominal_depth = 0.0 ## Can change the target depth projection if prefered
else:
	nominal_depth = locs[:,2].mean() ## Can change the target depth projection if prefered

center_loc = np.array([lat_range[0] + 0.5*np.diff(lat_range)[0], lon_range[0] + 0.5*np.diff(lon_range)[0], nominal_depth]).reshape(1,-1)
# center_loc = locs.mean(0, keepdims = True)
unit_lat = np.array([0.01, 0.0, 0.0]).reshape(1,-1) + center_loc
unit_vert = np.array([0.0, 0.0, 1000.0]).reshape(1,-1) + center_loc

norm_lat = torch.Tensor(np.linalg.norm(np.diff(lla2ecef(np.concatenate((center_loc, unit_lat), axis = 0)), axis = 0), axis = 1))
norm_vert = torch.Tensor(np.linalg.norm(np.diff(lla2ecef(np.concatenate((center_loc, unit_vert), axis = 0)), axis = 0), axis = 1))

trgt_lat = torch.Tensor([0,1.0,0]).reshape(1,-1)
trgt_vert = torch.Tensor([0,0,1.0]).reshape(1,-1)
trgt_depths = torch.Tensor([nominal_depth]) ## Not used
trgt_center = torch.zeros(2)

loss_func = nn.MSELoss()

losses = []
losses1, losses2, losses3, losses4 = [], [], [], []
loss_coef = [1,1,1,0]

n_attempts = 10
## Based on initial conditions, sometimes this converges to a projection plane that is flipped polarity.
## E.g., "up" in the lat-lon domain is "down" in the Cartesian domain, and vice versa.
## So try a few attempts to make sure it has the correct polarity.
for attempt in range(n_attempts):

	vec = nn.Parameter(2.0*np.pi*torch.rand(3))

	# mn = nn.Parameter(torch.Tensor(lla2ecef(locs).mean(0, keepdims = True)))
	mn = nn.Parameter(torch.Tensor(lla2ecef(center_loc.reshape(1,-1)).mean(0, keepdims = True)))

	optimizer = optim.Adam([vec, mn], lr = 0.001)

	print('\n Optimize the projection coefficients \n')

	n_steps_optimize = 5000
	for i in range(n_steps_optimize):

		optimizer.zero_grad()

		rbest = rotation_matrix(vec[0], vec[1], vec[2])

		norm_lat = lla2ecef_diff(torch.Tensor(np.concatenate((center_loc, unit_lat), axis = 0)))
		norm_vert = lla2ecef_diff(torch.Tensor(np.concatenate((center_loc, unit_vert), axis = 0)))
		norm_lat = torch.norm(norm_lat[1] - norm_lat[0])
		norm_vert = torch.norm(norm_vert[1] - norm_vert[0])

		center_out = ftrns1(torch.Tensor(center_loc), rbest, mn)

		out_unit_lat = ftrns1(torch.Tensor(unit_lat), rbest, mn)
		out_unit_lat = (out_unit_lat - center_out)/norm_lat

		out_unit_vert = ftrns1(torch.Tensor(unit_vert), rbest, mn)
		out_unit_vert = (out_unit_vert - center_out)/norm_vert

		out_locs = ftrns1(torch.Tensor(locs), rbest, mn)

		loss1 = loss_func(trgt_lat, out_unit_lat)
		loss2 = loss_func(trgt_vert, out_unit_vert)
		loss3 = loss_func(0.1*trgt_center, 0.1*center_out[0,0:2]) ## Scaling loss down

		loss = loss_coef[0]*loss1 + loss_coef[1]*loss2 + loss_coef[2]*loss3 # + loss_coef[3]*loss4
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		losses1.append(loss1.item())
		losses2.append(loss2.item())
		losses3.append(loss3.item())
		# losses4.append(loss4.item())

		if np.mod(i, 50) == 0:
			print('%d %0.8f'%(i, loss.item()))

	## Save approriate files and make extensions for directory
	print('\n Loss of lat and lon: %0.4f \n'%(loss_coef[0]*loss1 + loss_coef[1]*loss2))

	if (loss_coef[0]*loss1 + loss_coef[1]*loss2) < 1e-1:
		print('\n Finished converging \n')
		break
	else:
		print('\n Did not converge, restarting (%d) \n'%attempt)

# os.rename(ext_dir + 'stations.npz', ext_dir + '%s_stations_backup.npz'%name_of_project)

rbest = rbest.cpu().detach().numpy()
mn = mn.cpu().detach().numpy()

if use_pretrained_model is not None:
	shutil.move(ext_dir + 'Pretrained/trained_gnn_model_step_%d_ver_%d.h5'%(20000, use_pretrained_model), ext_dir + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(name_of_project, 20000, 1))
	shutil.move(ext_dir + 'Pretrained/1d_travel_time_grid_ver_%d.npz'%use_pretrained_model, ext_dir + '1D_Velocity_Models_Regional/%s_1d_travel_time_grid_ver_%d.npz'%(name_of_project, 1))
	shutil.move(ext_dir + 'Pretrained/seismic_network_templates_ver_%d.npz'%use_pretrained_model, ext_dir + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(use_pretrained_model, 1))

	## Find offset corrections if using one of the pre-trained models
	## Load these and apply offsets for runing "process_continuous_days.py"
	z = np.load(ext_dir + 'Pretrained/stations_ver_%d.npz'%use_pretrained_model)['locs']
	sta_loc, rbest, mn = z['locs'], z['rbest'], z['mn']
	corr1 = locs.mean(0, keepdims = True)
	corr2 = sta_loc.mean(0, keepdims = True)
	z.close()

	z = np.load(ext_dir + 'Pretrained/region_ver_%d.npz'%use_pretrained_model)
	lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
	z.close()

	locs = np.copy(locs) - corr1 + corr2
	shutil.copy(ext_dir + 'Pretrained/region_ver_%d.npz'%use_pretrained_model, ext_dir + '%s_region.npz'%name_of_project)

else:
	corr1 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)
	corr2 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)


np.savez_compressed(ext_dir + '%s_stations.npz'%name_of_project, locs = locs, stas = stas, rbest = rbest, mn = mn)

## Make necessary directories

os.mkdir(ext_dir + 'Picks')
os.mkdir(ext_dir + 'Catalog')
for year in years:
	os.mkdir(ext_dir + 'Picks/%d'%year)
	os.mkdir(ext_dir + 'Catalog/%d'%year)

os.mkdir(ext_dir + 'Plots')
os.mkdir(ext_dir + 'GNN_TrainedModels')
os.mkdir(ext_dir + 'Grids')
os.mkdir(ext_dir + '1D_Velocity_Models_Regional')

if (load_initial_files == True)*(use_pretrained_model == False):
	step_load = 20000
	ver_load = 1
	if os.path.exists(ext_dir + 'trained_gnn_model_step_%d_ver_%d.h5'%(step_load, ver_load)):
		shutil.move(ext_dir + 'trained_gnn_model_step_%d_ver_%d.h5'%(step_load, ver_load), ext_dir + 'GNN_TrainedModels/%s_trained_gnn_model_step_%d_ver_%d.h5'%(name_of_project, step_load, ver_load))

	ver_load = 1
	if os.path.exists(ext_dir + '1d_travel_time_grid_ver_%d.npz'%ver_load):
		shutil.move(ext_dir + '1d_travel_time_grid_ver_%d.npz'%ver_load, ext_dir + '1D_Velocity_Models_Regional/%s_1d_travel_time_grid_ver_%d.npz'%(name_of_project, ver_load))

	ver_load = 1
	if os.path.exists(ext_dir + 'seismic_network_templates_ver_%d.npz'%ver_load):
		shutil.move(ext_dir + 'seismic_network_templates_ver_%d.npz'%ver_load, ext_dir + 'Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, ver_load))

## Make spatial grids

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

				iremove = np.where((v[:,0] > (offset_x[0,0] + scale_x[0,0])) + ((v[:,1] > (offset_x[0,1] + scale_x[0,1]))) + (v[:,0] < offset_x[0,0]) + (v[:,1] < offset_x[0,1]))[0]
				if len(iremove) > 0:
					v[iremove] = np.random.rand(len(iremove), ndim)*scale_x + offset_x

			tree = cKDTree(ftrns1(v)*weight_vector)
			x1 = m_density.sample(n1)
			x1 = np.concatenate((x1, np.random.rand(n1).reshape(-1,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
			x2 = np.random.rand(n2, ndim)*scale_x + offset_x
			x = np.concatenate((x1, x2), axis = 0)
			iremove = np.where((x[:,0] > (offset_x[0,0] + scale_x[0,0])) + ((x[:,1] > (offset_x[0,1] + scale_x[0,1]))) + (x[:,0] < offset_x[0,0]) + (x[:,1] < offset_x[0,1]))[0]
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

def assemble_grids(scale_x_extend, offset_x_extend, n_grids, n_cluster, n_steps = 5000, extend_grids = True, with_density = None, density_kernel = 0.15):

	if with_density is not None:
		from sklearn.neighbors import KernelDensity
		m_density = KernelDensity(kernel = 'gaussian', bandwidth = density_kernel).fit(with_density[:,0:2])

	x_grids = []
	for i in range(n_grids):

		eps_extra = 0.1
		eps_extra_depth = 0.02
		scale_up = 1.0 # 10000.0
		weight_vector = np.array([1.0, 1.0, 5.0]).reshape(1,-1) ## Tries to scale importance of depth up, so that nodes fill depth-axis well

		offset_x_extend_slice = np.array([offset_x_extend[0,0], offset_x_extend[0,1], offset_x_extend[0,2]]).reshape(1,-1)
		scale_x_extend_slice = np.array([scale_x_extend[0,0], scale_x_extend[0,1], scale_x_extend[0,2]]).reshape(1,-1)

		depth_scale = (np.diff(depth_range)*0.02)
		deg_scale = ((0.5*np.diff(lat_range) + 0.5*np.diff(lon_range))*0.08)

		if extend_grids == True:
			extend1, extend2, extend3, extend4 = (np.random.rand(4) - 0.5)*deg_scale
			extend5 = (np.random.rand() - 0.5)*depth_scale
			offset_x_extend_slice[0,0] += extend1
			offset_x_extend_slice[0,1] += extend2
			scale_x_extend_slice[0,0] += extend3
			scale_x_extend_slice[0,1] += extend4
			offset_x_extend_slice[0,2] += extend5
			scale_x_extend_slice[0,2] = depth_range[1] - offset_x_extend_slice[0,2]

		else:
			pass

		print('\n Optimize for spatial grid (%d / %d)'%(i + 1, n_grids))

		offset_x_grid = scale_up*np.array([offset_x_extend_slice[0,0] - eps_extra*scale_x_extend_slice[0,0], offset_x_extend_slice[0,1] - eps_extra*scale_x_extend_slice[0,1], offset_x_extend_slice[0,2] - eps_extra_depth*scale_x_extend_slice[0,2]]).reshape(1,-1)
		scale_x_grid = scale_up*np.array([scale_x_extend_slice[0,0] + 2.0*eps_extra*scale_x_extend_slice[0,0], scale_x_extend_slice[0,1] + 2.0*eps_extra*scale_x_extend_slice[0,1], scale_x_extend_slice[0,2] + 2.0*eps_extra_depth*scale_x_extend_slice[0,2]]).reshape(1,-1)
	
		if with_density is not None:

			x_grid = kmeans_packing_weight_vector_with_density(m_density, weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch = 10000, n_steps = n_steps, n_sim = 1, lr = 0.005)[0]/scale_up # .to(device) # 8000

		else:
			x_grid = kmeans_packing_weight_vector(weight_vector, scale_x_grid, offset_x_grid, 3, n_cluster, ftrns1, n_batch = 10000, n_steps = n_steps, n_sim = 1, lr = 0.005)[0]/scale_up # .to(device) # 8000

		iargsort = np.argsort(x_grid[:,0])
		x_grid = x_grid[iargsort]
		x_grids.append(x_grid)

	return x_grids # , x_grids_edges

ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

skip_making_grid = False
if load_initial_files == True:
	if os.path.exists('Grids/%s_seismic_network_templates_ver_%d.npz'%(name_of_project, ver_load)) == True:
		skip_making_grid = True

if skip_making_grid == False:
	x_grids = assemble_grids(scale_x_extend, offset_x_extend, num_grids, n_spatial_nodes, n_steps = 5000, with_density = with_density)

	np.savez_compressed(ext_dir + 'Grids/%s_seismic_network_templates_ver_1.npz'%name_of_project, x_grids = [x_grids[i] for i in range(len(x_grids))], corr1 = corr1, corr2 = corr2)
