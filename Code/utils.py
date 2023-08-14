
import numpy as np
import torch
from scipy.spatial import cKDTree

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
