import yaml
import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import pathlib

def load_config(file_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def setup_region(client: Client, config: dict, t0: UTCDateTime, tf: UTCDateTime):
    """Set up region based on configuration."""
    return client.get_stations(
        starttime=t0,
        endtime=tf,
        network=config['network'],
        station='*',
        minlatitude=config['latitude_range'][0],
        maxlatitude=config['latitude_range'][1],
        minlongitude=config['longitude_range'][0],
        maxlongitude=config['longitude_range'][1]
    )[0]

def extract_station_data(stations):
    """Extract station code, location and elevation from the stations list."""
    stas = [station.code for station in stations]
    locs = [np.array([station.latitude, station.longitude, station.elevation]).reshape(1, -1) for station in stations]

    stas = np.hstack(stas)
    locs = np.vstack(locs)

    iarg = np.argsort(locs[:, 0])
    return locs[iarg], stas[iarg]

def save_files(base_path: str, locs, stas, config, years):
    """Save network, region, and 1D velocity model files."""
    # Save network file
    np.savez_compressed(base_path + 'stations.npz', locs=locs, stas=stas)

    # Save region file
    np.savez_compressed(
        base_path + 'region.npz',
        lat_range=config['latitude_range'],
        lon_range=config['longitude_range'],
        depth_range=config['depth_range'],
        deg_pad=config['degree_padding'],
        num_grids=config['number_of_grids'],
        n_spatial_nodes=config['number_of_spatial_nodes'],
        years=years,
        load_initial_files=config['load_initial_files'],
        use_pretrained_model=config['use_pretrained_model']
    )

    # Save 1D velocity model
    np.savez_compressed(
        base_path + '1d_velocity_model.npz',
        Depths=np.array(config['velocity_model']['Depths']),
        Vp=np.array(config['velocity_model']['Vp']),
        Vs=np.array(config['velocity_model']['Vs'])
    )

if __name__ == '__main__':
    print("Loading configuration from 'config.yaml'...")
    config = load_config('config.yaml')

    print("Setting up time range...")
    t0 = UTCDateTime(config['time_range']['start'])
    tf = UTCDateTime(config['time_range']['end'])
    years = list(range(t0.year, tf.year + 1))

    print("Connecting to IRIS client...")
    client = Client('IRIS')
    print("Setting up region based on configuration...")
    stations = setup_region(client, config, t0, tf)

    print("Extracting station data...")
    locs, stas = extract_station_data(stations)

    print("Determining base path for saving files...")
    base_path = str(pathlib.Path().absolute()) + ('\\' if '\\' in str(pathlib.Path().absolute()) else '/')
    
    print("Saving files...")
    save_files(base_path, locs, stas, config, years)
    print("All files saved successfully!")