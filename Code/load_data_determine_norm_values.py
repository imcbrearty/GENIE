
import yaml
from utils import load_config
import glob
import numpy as np
import h5py

ext_type = 'server'

if ext_type == 'local':

	ext_data = 'D:/Projects/Laplace/Heterogenous_TrainingData/'

elif ext_type == 'remote':

	ext_data = '/work/wavefront/imcbrear/Laplace/Heterogenous_TrainingData/'

elif ext_type == 'server':

	ext_data = '/scratch/users/taiyi/ForIan/complex_08162023/'

config = load_config('config.yaml')

n_samples = config['n_samples'] # 2000

scale_val = config['scale_val'] # 50e3

norm_version = config['norm_version'] # 2

max_val = config['max_val'] # 0.1
min_val = config['min_val'] # -3.5

## Load data, save maximums
Lh = []
Lv = []

Ns = []
dp2mu = []
dx = [] # position ? (scalar). It is 0
dy = [] # position ? (scalar). It is 0
dz = [] # depth? (scalar)
mu = [] # shear modulus (scalar)
nu = [] # Poissons ratio
ra2d = []
thetax = []
thetaz = []
Fs = []
NormF = []
RMax = []

Ra = []
Rb = []

C = [] # C [3 x 3196] # Positions of mesh?
P = [] # P [3 x 1600] ## P is the positions of the mesh boundaries                                                    Are P the normal vectors of each face?
T = [] # T [3 x 3196] ## All ints (must be the connectivity matrix)

## T is base 1 indexed, and max T is 1600

Ux = [] # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
Uy = [] # [1 x 1440]
Uz = [] # [1 x 1440]
X = [] # [1 x 1440]
Y = [] # [1 x 1440]
Z = [] # [1 x 1440]

dhat = []
nhat = []
that = []

Normals = []
Normals_face = []

Norm_values = []

if ext_type == 'remote':

	st = glob.glob(ext_data + 'complex/*.mat')
	st1 = glob.glob(ext_data + 'spheroids/*.mat')

elif ext_type == 'server':

	st = glob.glob(ext_data + 'complex/*.mat')
	st1 = glob.glob(ext_data + 'sph_composite/*.mat')	

st = np.concatenate((st, st1), axis = 0)

n_files = len(st)

iwhere = []
ifail = []

for i in range(n_files):

	try:
		z = h5py.File(st[i], 'r')
	except:
		ifail.append(i)
		print('Failed on %d (%d)'%(i, len(ifail)))
		continue

	if norm_version == 1:
		norm_val = np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]]))
	elif norm_version == 2:
		norm_val = np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max()
	
	z.close()

	if (np.log10(norm_val) < max_val)*(np.log10(norm_val) > min_val):

		iwhere.append(i)

	if np.mod(i, 100) == 0:
		print(i)

st = [st[iwhere[i]] for i in range(len(iwhere))]

itrain = np.sort(np.random.choice(len(st), size = int(0.9*len(st)), replace = False))
ivald = np.delete(np.arange(len(st)), itrain, axis = 0)

np.savez_compressed(ext_data + 'training_files_within_norm_bounds.npz', st = st, itrain = itrain, ivald = ivald)

moi

n_files = len(st)

# moi

norm_vals_max = -1.0*np.inf*np.ones((1,45))

for i in range(n_files):

	z = h5py.File(st[i], 'r')

	dz = z['input/dz'][:][0][0]/scale_val # depth? (scalar)
	Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1)
	NormF = z['input/normF'][:].reshape(-1)
	RMax = z['input/rmax'][:].reshape(-1)

	norm_vals_slice = np.concatenate([np.array([np.abs(dz)]), np.abs(Fs).reshape(-1), np.abs(NormF), np.abs(RMax)], axis = 0).reshape(1,-1)

	norm_vals_max = np.concatenate((norm_vals_max, norm_vals_slice), axis = 0).max(0, keepdims = True)

	if np.mod(i, 1000) == 0:
		print(i)

	# C.append(z['output/C'][:]/scale_val) # C [3 x 3196] # Positions of mesh?
	# P.append(z['output/P'][:]/scale_val) # P [3 x 1600] ## P is the positions of the mesh boundaries                                                    Are P the normal vectors of each face?
	# T.append(z['output/T'][:] - 1) # T [3 x 3196] ## All ints (must be the connectivity matrix)

	# Ra.append(z['input/ra'][:])
	# Rb.append(z['input/rb'][:])

	## T is base 1 indexed, and max T is 1600

	# if norm_version == 1:
	# 	Norm_values.append(np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]])))
	# elif norm_version == 2:
	# 	Norm_values.append(np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max())

	# Ux.append(z['output/Ux'][:]/Norm_values[-1]) # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
	# Uy.append(z['output/Uy'][:]/Norm_values[-1]) # [1 x 1440]
	# Uz.append(z['output/Uz'][:]/Norm_values[-1]) # [1 x 1440]
	# X.append(z['output/X'][:]/scale_val) # [1 x 1440]
	# Y.append(z['output/Y'][:]/scale_val) # [1 x 1440]
	# Z.append(z['output/Z'][:]/scale_val) # [1 x 1440]

	# dhat.append(z['output/dhat'][:])
	# nhat.append(z['output/nhat'][:])
	# that.append(z['output/that'][:])
	z.close()

np.savez_compressed(ext_data + 'training_files_complex_max_values.npz', st = st, itrain = itrain, ivald = ivald, norm_vals_max = norm_vals_max)
