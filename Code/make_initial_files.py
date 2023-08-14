import yaml
import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import pathlib

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

lat_range = config['latitude_range']
lon_range = config['longitude_range']
depth_range = config['depth_range']
t0 = UTCDateTime(config['time_range']['start'])
tf = UTCDateTime(config['time_range']['end'])
years = list(range(t0.year, tf.year + 1))

# Set up region based on YAML config
client = Client('IRIS')
stations = client.get_stations(
    starttime=t0,
    endtime=tf,
    network=config['network'],
    station='*',
    minlatitude=lat_range[0],
    maxlatitude=lat_range[1],
    minlongitude=lon_range[0],
    maxlongitude=lon_range[1])[0]

locs = []
stas = []
for i in range(len(stations)):
    stas.append(stations[i].code)
    locs.append(np.array([stations[i].latitude, stations[i].longitude, stations[i].elevation]).reshape(1, -1))

stas = np.hstack(stas)
locs = np.vstack(locs)
iarg = np.argsort(locs[:, 0])

locs = locs[iarg]
stas = stas[iarg]

path_to_file = str(pathlib.Path().absolute())
path_to_file += '\\' if '\\' in path_to_file else '/'

# Save network file
np.savez_compressed(path_to_file + 'stations.npz', locs=locs, stas=stas)

# Save region file
np.savez_compressed(
    path_to_file + 'region.npz',
    lat_range=lat_range,
    lon_range=lon_range,
    depth_range=depth_range,
    deg_pad=config['deg_pad'],
    num_grids=config['num_grids'],
    n_spatial_nodes=config['n_spatial_nodes'],
    years=years,
    load_initial_files=config['load_initial_files'],
    use_pretrained_model=config['use_pretrained_model'])

# Save 1D velocity model
Depths = np.array(config['velocity_model']['Depths'])
Vp = np.array(config['velocity_model']['Vp'])
Vs = np.array(config['velocity_model']['Vs'])
np.savez_compressed(path_to_file + '1d_velocity_model.npz', Depths=Depths, Vp=Vp, Vs=Vs)
