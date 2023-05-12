import numpy as np
from matplotlib import pyplot as plt
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import pathlib

## This script is an example for how to initilize the station, region, and 1d velocity model files. It creates the variables:
## locs : a numpy array of N x 3 shape, with columns of latitude, longitude, elevation (elevations are positive above sea level, and negative below sea level, in meters).
## stas: a numpy array of station names, of length N, corresponding to entries in locs
## It also creates the region.npz file (with parameters specified below), and 1d velocity model.

## The purpose of this script is only to make the three files saved below; one can create these files any way they prefer, but should follow the same
## naming scheme of variables, units and sign conventions for velocities and elevations (or equivelently, depths).

lat_range = [18.8, 20.3] # Latitude range of the region that will be processed
lon_range = [-156.1, -154.7] # Longitude range of the region that will be processed
depth_range = [-40e3, 5e3] ## Note: depths are in meters, positive above sea level, negative below sea level, and 'increasing' depth means going from deep to shallow.

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
if '\\' in path_to_file: ## Windows
	path_to_file = path_to_file + '\\'
elif '/' in path_to_file: ## Linux
	path_to_file = path_to_file + '/'	

## Save network file
np.savez_compressed(path_to_file + 'stations.npz', locs = locs, stas = stas)

## Save region file
deg_pad = 0.25 # This region is appended to the lat_range and lon_range values above, and is used as a `padding' region, where we compute
## travel times, and simulate events in this region, yet train to predict zero for all labels in this area. This way, sources just outside the domain
# of interest arn't falsely mis-located inside the region (since the model learns what `exterior' events look like, to some extent).
num_grids = 5 # Number of distinct spatial graphs to create (this reduced slight bias that can result from only using one spatial graph)
n_spatial_nodes = 500 # Number of nodes per spatial graph
years = np.arange(2018, 2024) # Number of years anticipating to be processed (only effects the folders created in Picks and Catalog sub-directories)
load_initial_files = [False] # Can set to True to initilize based on pre-existing models and files
use_pretrained_model = [None] # name of pre-existing files to load (not yet implemented)
np.savez_compressed(path_to_file + 'region.npz', lat_range = lat_range, lon_range = lon_range, depth_range = depth_range, deg_pad = deg_pad, num_grids = num_grids, n_spatial_nodes = n_spatial_nodes, years = years, load_initial_files = load_initial_files, use_pretrained_model = use_pretrained_model)

## Save 1D velocity model
## Approximate model at Hawaii from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2013JB010820
Depths = np.arange(-40e3, 10e3, 5e3) # Depths, an increasing numpy vector, with negative below sea level, positive above sea level, in meters.
Vp = 1000.0*np.array([8.2, 8.2, 8.15, 8.1, 8.05, 7.4, 6.8, 6.1, 2.9, 2.9]) # Vp values corresponding to Depths, in meters/s
Vs = 1000.0*np.array([4.7, 4.7, 4.65, 4.6, 4.6, 4.2, 3.8, 3.4, 1.8, 1.8]) # Vs values corresponding to Depths, in meters/s
np.savez_compressed(path_to_file + '1d_velocity_model.npz', Depths = Depths, Vp = Vp, Vs = Vs)
