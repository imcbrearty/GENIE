
import numpy as np
import yaml

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
