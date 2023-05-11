import numpy as np
from matplotlib import pyplot as plt
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

lat_range = [18.8, 20.3]
lon_range = [-156.1, -154.7]
depth_range = [-40e3, 5e3]

t0 = UTCDateTime(2018, 1, 1)
tf = UTCDateTime(2023, 1, 1)

## Set up region for Hawaii
client = Client('IRIS')
stations = client.get_stations(starttime = t0, endtime = tf, network = 'HV', station = '*', minlatitude = lat_range[0], maxlatitude = lat_range[1], minlongitude = lon_range[0], maxlongitude = lon_range[1])[0]

locs = []
stas = []

for i in range(len(stations)):
  stas.append(stations[i].code)
	locs.append(np.array([stations[i].latitude, stations[i].longitude, stations[i].elevation]).reshape(1,-1))

stas = np.hstack(stas)
locs = np.vstack(locs)
iarg = np.argsort(locs[:,0])

locs = locs[iarg]
stas = stas[iarg]

path_to_file = str(pathlib.Path().absolute())

## Save network file
np.savez_compressed(path_to_file + 'stations.npz', locs = locs, stas = stas)

## Save region file
deg_pad = 0.25
num_grids = 5
n_spatial_nodes = 500
years = np.arange(2018, 2024)
load_initial_files = [False]
use_pretrained_model = [None]
np.savez_compressed(path_to_file + 'region.npz', lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, deg_pad = deg_pad, num_grids = num_grids, n_spatial_nodes = n_spatial_nodes, years = years, load_initial_files = load_initial_files, use_pretrained_model = use_pretrained_model)

## Save 1D velocity model
## Approximate model from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2013JB010820
Depths = np.arange(-40e3, 10e3, 5e3)
Vp = 1000.0*np.array([8.2, 8.2, 8.15, 8.1, 8.05, 7.4, 6.8, 6.1, 2.9, 2.9])
Vs = 1000.0*np.array([4.7, 4.7, 4.65, 4.6, 4.6, 4.2, 3.8, 3.4, 1.8, 1.8])
np.savez_compressed(path_to_file + '1d_velocity_model.npz', Depths = Depths, Vp = Vp, Vs = Vs)
