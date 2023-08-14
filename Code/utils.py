
import numpy as np
import torch

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
