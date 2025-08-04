import yaml
import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from scipy.spatial import cKDTree
import pathlib
import os
from utils import load_config


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
    pre_load_stations = config['pre_load_stations']

    print("Setting up time range...")
    t0 = UTCDateTime(config['time_range']['start'])
    tf = UTCDateTime(config['time_range']['end'])
    years = list(range(t0.year, tf.year + 1))

    print("Determining base path for saving files...")
    path_abs = pathlib.Path().absolute()
    seperator = '\\' if '\\' in str(path_abs) else '/'
    base_path = str(path_abs) + seperator
    
    print(f"Connecting to {config['client']} client...")
    client = Client(config['client'])

    if pre_load_stations == False:
        print("Setting up region based on configuration...")
        print('Downloading stations from Obspy instead of file since pre_load_stations = False')
        stations = setup_region(client, config, t0, tf)

        print("Extracting station data...")
        locs, stas = extract_station_data(stations)

    elif pre_load_stations == True:
        print('Loading pre-built station file')

        if os.path.isfile(base_path + 'stations.txt') == True:
            f = open(base_path + 'stations.txt', 'r')
            lines = f.readlines()
            stas, locs = [], []
            for j in range(len(lines)):
                if ('Lat' in lines[j]) or ('lat' in lines[j]) or ('Lon' in lines[j]) or ('lon' in lines[j]):
                    continue
                if ',' in lines[j]:
                    line = list(filter(lambda x: len(x) > 0, lines[j].strip().split(',')))
                else:
                    line = list(filter(lambda x: len(x) > 0, lines[j].strip().split(' ')))
                stas.append(line[0])
                locs.append(np.array([float(line[1]), float(line[2]), float(line[3])]).reshape(1,-1))
            stas = np.hstack(stas).astype('U9')
            locs = np.vstack(locs)
            print('Saving station .npz file')
            np.savez_compressed(base_path + 'stations.npz', locs = locs, stas = stas) 
        elif os.path.isfile(base_path + 'stations.npz') == True:
            z = np.load(base_path + 'stations.npz', allow_pickle = True)
            locs, stas = z['locs'], z['stas'].astype('U9')
            print('Saving station .npz file')
            z.close()
        else:
             raise Exception('No station file loaded; create the "stations.txt" or "stations.npz" file; see GitHub "Setup Details" section for more information')

    ##### Run convert initial pick files #####
    convert_picks = config.get('convert_initial_pick_files', True)
    if (convert_picks == True)*(os.path.isfile(base_path + 'picks.txt') == 1):
        print('Converting pick files')
        n_ver_write = 1
        f = open(base_path + 'picks.txt', 'r')
        lines = f.readlines()
        f.close()
        P, Dates = [], []
        iskipped = []
        filt = lambda x: len(x) > 0
        for p in lines:
            p = [s.strip() for s in list(filter(filt, p.split(',')))]
            t = UTCDateTime(p[0])
            imatch = np.where(stas == p[1].strip())[0]
            assert(len(imatch) == 1) ## Requires a match of pick names in picks.txt to station names in stations.txt
            assert(((p[4] == 'P') + (p[4] == 'S')) > 0) ## Requires a phase type of either P or S
            Dates.append(np.array([t.year, t.month, t.day]).reshape(1,-1))
            P.append(np.array([t - UTCDateTime(t.year, t.month, t.day), imatch[0], float(p[2]), float(p[3]), 0.0 if p[4] == 'P' else 1.0]).reshape(1,-1))
        Dates = np.vstack(Dates)
        P = np.vstack(P)
        dates_unique = np.unique(Dates, axis = 0)
        os.makedirs(base_path + 'Picks', exist_ok = True)
        os.makedirs(base_path + 'Catalog', exist_ok = True)
        os.makedirs(base_path + 'Calibration', exist_ok = True)
        for yr in np.unique(dates_unique[:,0]): os.makedirs(base_path + 'Picks' + seperator + '%d'%yr, exist_ok = True)
        for yr in np.unique(dates_unique[:,0]): os.makedirs(base_path + 'Catalog' + seperator + '%d'%yr, exist_ok = True)
        for yr in np.unique(dates_unique[:,0]): os.makedirs(base_path + 'Calibration' + seperator + '%d'%yr, exist_ok = True)
        ip = cKDTree(Dates).query_ball_point(dates_unique, r = 0)
        for i in range(len(dates_unique)):
            if len(ip[i]) == 0: continue
            np.savez_compressed(base_path + 'Picks' + seperator + '%d'%dates_unique[i,0] + seperator + '%d_%d_%d_ver_%d.npz'%(dates_unique[i,0], dates_unique[i,1], dates_unique[i,2], n_ver_write), P = P[ip[i]])
            print('Saved %d/%d/%d (%d picks; %d P and %d S)'%(dates_unique[i,0], dates_unique[i,1], dates_unique[i,2], len(ip[i]), len(np.where(P[ip[i],4] == 0)[0]), len(np.where(P[ip[i],4] == 1)[0])))

    ##### End convert initial pick files #####
    
    print("Saving files...")
    save_files(base_path, locs, stas, config, years)

    print("All files saved successfully!")
    print("✔ Script execution: Done")
