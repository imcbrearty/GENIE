
import yaml
from utils import load_config

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

max_val = 0.1
min_val = -3.5

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

Norm_values_init = []

for i in range(n_files):

	z = h5py.File(st[i], 'r')
	if norm_version == 1:
		Norm_values_init.append(np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]])))
	elif norm_version == 2:
		Norm_values_init.append(np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max())

	z.close()

Norm_values_init = np.array(Norm_values_init)

iwhere = np.where((np.log10(Norm_values_init) < max_val)*(np.log10(Norm_values_init) > min_val))[0]

st = [st[iwhere[i]] for i in range(len(iwhere))]

n_files = len(st)

# moi

for i in range(n_files):

	z = h5py.File(st[i], 'r')

	Lh.append(z['input/Lh'][:][0][0]/scale_val)
	Lv.append(z['input/Lv'][:][0][0]/scale_val)

	Ns.append(z['input/Ns'][:][0][0])
	dp2mu.append(z['input/dp2mu'][:][0][0])
	dx.append(z['input/dx'][:][0][0]/scale_val) # position ? (scalar). It is 0
	dy.append(z['input/dy'][:][0][0]/scale_val) # position ? (scalar). It is 0
	dz.append(z['input/dz'][:][0][0]/scale_val) # depth? (scalar)
	mu.append(z['input/mu'][:][0][0]) # shear modulus (scalar)
	nu.append(z['input/nu'][:][0][0]) # Poissons ratio
	# ra2d.append(z['input/asp'][:][0][0]) # Poissons ratio
	thetax.append(z['input/thetax'][:][0][0])
	thetaz.append(z['input/thetaz'][:][0][0])

	Fs.append(np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1))
	NormF.append(z['input/normF'][:].reshape(-1))
	RMax.append(z['input/rmax'][:].reshape(-1))

	C.append(z['output/C'][:]/scale_val) # C [3 x 3196] # Positions of mesh?
	P.append(z['output/P'][:]/scale_val) # P [3 x 1600] ## P is the positions of the mesh boundaries                                                    Are P the normal vectors of each face?
	T.append(z['output/T'][:] - 1) # T [3 x 3196] ## All ints (must be the connectivity matrix)

	# Ra.append(z['input/ra'][:])
	# Rb.append(z['input/rb'][:])

	## T is base 1 indexed, and max T is 1600

	if norm_version == 1:
		Norm_values.append(np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]])))
	elif norm_version == 2:
		Norm_values.append(np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max())

	Ux.append(z['output/Ux'][:]/Norm_values[-1]) # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
	Uy.append(z['output/Uy'][:]/Norm_values[-1]) # [1 x 1440]
	Uz.append(z['output/Uz'][:]/Norm_values[-1]) # [1 x 1440]
	X.append(z['output/X'][:]/scale_val) # [1 x 1440]
	Y.append(z['output/Y'][:]/scale_val) # [1 x 1440]
	Z.append(z['output/Z'][:]/scale_val) # [1 x 1440]

	dhat.append(z['output/dhat'][:])
	nhat.append(z['output/nhat'][:])
	that.append(z['output/that'][:])
	z.close()

Lh = np.hstack(Lh)
Lv = np.hstack(Lv)

# Ra = np.hstack(Ra).reshape(-1)
# Rb = np.hstack(Rb).reshape(-1)
# ra2d = np.array(ra2d)
dz = np.array(dz)
thetax = np.array(thetax)
thetaz = np.array(thetaz)
Norm_values = np.array(Norm_values)

Fs = np.vstack(Fs)
NormF = np.hstack(NormF)
RMax = np.hstack(RMax)

norm_vals = np.concatenate([np.array([np.abs(dz).max()]), np.abs(Fs).max(0), np.array([np.abs(NormF).max()]), np.array([np.abs(RMax).max()])]).reshape(1,-1)

assert(np.log10(Norm_values).max() < 0.1)
assert(np.log10(Norm_values).min() > -3.5)