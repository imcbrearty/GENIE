import yaml
import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from scipy.spatial import cKDTree
import pathlib
import h5py
import torch
import os
from utils import load_config
from utils import rotation_matrix_full_precision
from graph_utils import *



def optimize_with_differential_evolution(center_loc):
    """
    Optimize using the differential evolution algorithm to minimize a loss function based on geospatial transformations.
    
    Parameters:
    - center_loc (numpy.ndarray): The central location to optimize around.
    
    Returns:
    - soln (object): Solution object from the differential evolution algorithm.
    """
    
    loss_coef = [1, 1, 1.0, 0]

    # Calculate initial ecef values as they don't depend on x
    norm_lat_ecef = lla2ecef(np.concatenate((center_loc, center_loc + [0.001, 0.0, 0.0]), axis=0))
    norm_vert_ecef = lla2ecef(np.concatenate((center_loc, center_loc + [0.0, 0.0, 10.0]), axis=0))
    norm_lat = np.linalg.norm(norm_lat_ecef[1] - norm_lat_ecef[0])
    norm_vert = np.linalg.norm(norm_vert_ecef[1] - norm_vert_ecef[0])

    trgt_lat = np.array([0, 1.0, 0]).reshape(1, -1)
    trgt_vert = np.array([0, 0, 1.0]).reshape(1, -1)
    trgt_center = np.zeros(3)

    def loss_function(x):
        rbest = rotation_matrix_full_precision(x[0], x[1], x[2])

        center_out = ftrns1(center_loc, rbest, x[3:].reshape(1, -1))
        out_unit_lat = (ftrns1(center_loc + [0.001, 0.0, 0.0], rbest, x[3:].reshape(1, -1)) - center_out) / norm_lat
        out_unit_vert = (ftrns1(center_loc + [0.0, 0.0, 10.0], rbest, x[3:].reshape(1, -1)) - center_out) / norm_vert

        # If locs are global, then include this line
        # out_locs = ftrns1(locs, rbest, x[3:].reshape(1, -1))

        loss1 = np.linalg.norm(trgt_lat - out_unit_lat, axis=1)
        loss2 = np.linalg.norm(trgt_vert - out_unit_vert, axis=1)
        loss3 = np.linalg.norm(trgt_center.reshape(1, -1) - center_out, axis=1)
        loss = loss_coef[0] * loss1 + loss_coef[1] * loss2 + loss_coef[2] * loss3

        return loss

    bounds = [(0, 2.0 * np.pi) for _ in range(3)] + [(-1e7, 1e7) for _ in range(3)]
    soln = differential_evolution(loss_function, bounds, popsize=30, maxiter=1000, disp=True)

    return soln

if __name__ == '__main__':

    print("Loading configuration from 'config.yaml'...")
    config = load_config('config.yaml')
    pre_load_stations = config['pre_load_stations']
    name_of_project = config['name_of_project']
    lat_range = np.array(config['latitude_range'])
    lon_range = np.array(config['longitude_range'])
    depth_range = np.array(config['depth_range'])

    print("Setting up time range...")
    t0 = UTCDateTime(config['time_range']['start'])
    tf = UTCDateTime(config['time_range']['end'])
    years = list(range(t0.year, tf.year + 1))

    print("Determining base path for saving files...")
    path_abs = pathlib.Path().absolute()
    seperator = '\\' if '\\' in str(path_abs) else '/'
    base_path = str(path_abs) + seperator
    fix_nominal_depth = config.get('fix_nominal_depth', True)
    use_spherical = config.get('use_spherical', False)
    

    if pre_load_stations == True:

        print('Loading pre-built station file')
        ## Default load is npz file
        if os.path.isfile(base_path + 'stations.npz') == True:
            z = np.load(base_path + 'stations.npz', allow_pickle = True)
            locs, stas = z['locs'], z['stas'].astype('U9')
            print('Loading station .npz file')
            z.close()
        

        elif os.path.isfile(base_path + 'stations.txt') == True:
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
            print('Loading station .npz file')
            np.savez_compressed(base_path + 'stations.npz', locs = locs, stas = stas) 


        else:
             raise Exception('No station file loaded; create the "stations.txt" or "stations.npz" file; see GitHub "Setup Details" section for more information')

    else:

        print(f"Connecting to {config['client']} client...")
        client = Client(config.get('client', 'IRIS'))

        print("Setting up region based on configuration...")
        print('Downloading stations from Obspy instead of file since pre_load_stations = False')
        # stations = # setup_region(client, config, t0, tf)
        stations = client.get_stations(starttime = t0,endtime = tf,network = config['network'],station = '*',minlatitude = config['latitude_range'][0],maxlatitude = config['latitude_range'][1],minlongitude = config['longitude_range'][0],maxlongitude = config['longitude_range'][1])[0]

        print("Extracting station data...")
        # locs, stas = extract_station_data(stations)
        stas = [station.code for station in stations]
        locs = [np.array([station.latitude, station.longitude, station.elevation]).reshape(1, -1) for station in stations]
        stas = np.hstack(stas)
        locs = np.vstack(locs)
        iarg = np.argsort(locs[:, 0])
        locs, stas = locs[iarg], stas[iarg]

    # elif pre_load_stations == True:
    #     print('Loading pre-built station file')


    ## Fit domain projection parameters (save station file)

    ## Check if using full Earth, set target sampling bounds ##
    if (lat_range[0] <= -89.98)*(lat_range[1] >= 89.98)*(lon_range[0] <= -179.98)*(lon_range[1] >= 179.98):
        use_global = True
        lat_range_extend = [-90.0, 90.0]
        lon_range_extend = [-180.0, 180.0]
    else:
        use_global = False

    ## Fit projection coordinates and create spatial grids
    if use_spherical == True:
        earth_radius = 6371e3
        ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
        ftrns2 = lambda x, rbest, mn: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean
    else:
        earth_radius = 6378137.0
        ftrns1 = lambda x, rbest, mn: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
        ftrns2 = lambda x, rbest, mn: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean


    if fix_nominal_depth == True:
        nominal_depth = 0.0 ## Can change the target depth projection if prefered
    else:
        nominal_depth = locs[:,2].mean() ## Can change the target depth projection if prefered


    print('\n Using domain')
    print('\n Latitude:')
    print(lat_range)
    print('\n Longitude:')
    print(lon_range)


    center_loc = np.array([lat_range[0] + 0.5*np.diff(lat_range)[0], lon_range[0] + 0.5*np.diff(lon_range)[0], nominal_depth]).reshape(1,-1)    
    soln = optimize_with_differential_evolution(center_loc)
    rbest = rotation_matrix_full_precision(soln.x[0], soln.x[1], soln.x[2])
    mn = soln.x[3::].reshape(1,-1)

    ## Fit projection coordinates and create spatial grids
    if use_spherical == True:
        earth_radius = 6371e3
        ftrns1 = lambda x: (rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - mn).T).T # just subtract mean
        ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn, e = 0.0, a = earth_radius) # just subtract mean
    else:
        earth_radius = 6378137.0
        ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # just subtract mean
        ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn) # just subtract mean


    ## Save station file with projection functions
    np.savez_compressed(base_path + f'{config["name_of_project"]}_stations.npz', locs = locs, stas = stas, rbest = rbest, mn = mn)
    device = torch.device('cuda') if torch.cuda.is_available() == True else torch.device('cpu')

    print('Saved station file \n')
    mn_cuda = torch.tensor(mn, device = device)
    rbest_cuda = torch.tensor(rbest, device = device)    
    corr1 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)
    corr2 = np.array([0.0, 0.0, 0.0]).reshape(1,-1)


    max_station_ratio = config.get('max_station_ratio', 1.0)
    if max_station_ratio < 1.0:
        iselect = np.sort(np.random.choice(len(locs), size = int(np.floor(max_station_ratio*len(locs))), replace = False))
        locs_use = locs[iselect]
        stas_use = stas[iselect]
    else:
        locs_use = np.copy(locs)
        stas_use = np.copy(stas)



    ## Add estimate of number of nodes / cartesian product size
    if config['vel_model_type'] == 1:
        # vp = np.array(config['velocity_model']['Vp'])
        vs = np.array(config['velocity_model']['Vs']).reshape(-1)
        depths = np.array(config['velocity_model']['Depths'])
        ifind = np.where(depths >= np.quantile(depths, 0.8))[0]
        Vc = np.median(vs[ifind])
        # Vc = 3500.0
    elif config['vel_model_type'] == 2:   
        z = np.load(base_path + '3d_velocity_model.npz')
        depths, vs = z['X'][:,2], z['Vs'].reshape(-1) ## lat, lon, depth (x_vel) and velocity values
        ifind = np.where(depths >= np.quantile(depths, 0.8))[0]
        Vc = np.median(vs[ifind])
        # depths, vp, vs = z['X'][:,2], z['Vp'], z['Vs'] ## lat, lon, depth (x_vel) and velocity values
        z.close()
    
    Vc = 3500.0
    scale_domain = 1.0
    deg_padding = np.nan ## Use hueristic
    n_trgt_nodes = config.get('n_trgt_nodes', int(200e3))
    num_grids = config.get('number_of_grids', 5)
    number_of_spatial_nodes = config.get('number_of_spatial_nodes', 1500)
    optimize_source_graphs = config.get('optimize_source_graphs', False)
    optimize_station_graphs = config.get('optimize_station_graphs', False)
    use_paths = config.get('use_paths', False) # process_config.get('use_paths', False) # False
    use_tuner = config.get('use_tuner', True) # process_config.get('use_tuner', True) # False
    use_domain_approximate = config.get('use_domain_approximate', False)
    if use_domain_approximate == True:
        m_domain = load_model_domain(1, device = device)
    else:
        m_domain = None

    initialize = [lat_range, lon_range, num_grids, years]
    build_graphs_domain(m_domain, locs_use, stas_use, scale_domain, deg_padding, number_of_spatial_nodes, config['k_spc_edges'], config['k_sta_edges'], depth_range, 
        ftrns1, ftrns2, use_global = use_global, assign_based_on_grid = False, max_nodes = number_of_spatial_nodes, n_trgt_nodes = n_trgt_nodes, n_grids = num_grids, Vc = Vc, rbest = rbest, 
        mn = mn, file_index = 0, date = UTCDateTime(years[0], 1, 1), use_paths = use_paths, optimize_station_graphs = optimize_station_graphs, optimize_source_graphs = optimize_source_graphs, 
        use_domain_approximate = use_domain_approximate, initialize = initialize, name_of_project = name_of_project, use_tuner = use_tuner, device = device)



    ##### Run convert initial pick files #####
    convert_picks = config.get('convert_initial_pick_files', False)
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
            p = [s.strip() for s in list(filter(filt, p.split(',')))] if ',' in p else [s.strip() for s in list(filter(filt, p.split(' ')))[0:5]]
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

        ## Make pick list file
        f = open(base_path + '%s_process_days_list_ver_%d.txt'%(config['name_of_project'], n_ver_write), 'w')
        for i in range(len(dates_unique)):
            f.write('%d/%d/%d \n'%(dates_unique[i,0], dates_unique[i,1], dates_unique[i,2]))
        f.close()
        ## Make pick list file
    ##### End convert initial pick files #####


    ####### Convert initial catalog file #########
    convert_catalog = config.get('convert_initial_catalog', False)
    if (convert_catalog == True)*(os.path.isfile(base_path + 'catalog.txt') == 1):
        print('Convert catalog')
        f = open(base_path + 'catalog.txt', 'r')
        lines = f.readlines()
        f.close()
        n_ver_write = 1
        Srcs, Dates = [], []
        Picks_P, Picks_S = [], []
        Picks_P_slice, Picks_S_slice = [], []
        Uncertainity = []
        filt = lambda x: len(x) > 0
        lines.append('#') ## Add another line, so that last pick files are saved
        for l in lines:
            if l[0] == '#': ## Entry is a source
                if len(Srcs) > 0:
                    Picks_P.append(np.vstack(Picks_P_slice)) if len(Picks_P_slice) > 0 else Picks_P.append(np.zeros((0,6))) ## Append the stacked set of picks for this source
                    Picks_S.append(np.vstack(Picks_S_slice)) if len(Picks_S_slice) > 0 else Picks_S.append(np.zeros((0,6)))
                if len(l) == 1: continue ## Last line of file
                l = list(filter(filt, l.strip().split(' ')))
                t = UTCDateTime(int(l[1]), int(l[2]), int(l[3]), int(l[4]), int(l[5]), 0) + float(l[6])
                Srcs.append(np.array([float(l[7]), float(l[8]), -1000.0*float(l[9]), t - UTCDateTime(t.year, t.month, t.day), float(l[10])]).reshape(1,-1))
                Dates.append(np.array([t.year, t.month, t.day]).reshape(1,-1))
                Uncertainity.append(np.mean([float(l[11])*1000.0, float(l[12])*1000.0]))
                Picks_P_slice = []
                Picks_S_slice = []
                src_current = Srcs[-1]
            else: ## Assume entry is a pick
                l = list(filter(filt, l.strip().split(' ')))
                i1 = np.where(stas == l[0])[0]
                assert(len(i1) == 1) ## Require a match with a station in the initial stations file
                t = src_current[0,3] + float(l[1])
                prob = float(l[2])
                assert(((l[3] == 'P') + (l[3] == 'S')) > 0) ## Require phase to be either P or S
                phase = 0 if l[3] == 'P' else 1
                if phase == 0: Picks_P_slice.append(np.array([t, i1[0], np.nan, prob, phase, prob]).reshape(1,-1)) ## Amplitudes are missing from HypoDD format files
                if phase == 1: Picks_S_slice.append(np.array([t, i1[0], np.nan, prob, phase, prob]).reshape(1,-1)) ## Amplitudes are missing from HypoDD format files
        Srcs = np.vstack(Srcs)
        Dates = np.vstack(Dates)
        Uncertainity = np.hstack(Uncertainity)
        assert(len(Srcs) == len(Dates))
        assert(len(Srcs) == len(Picks_P))
        assert(len(Srcs) == len(Picks_S))
        dates_unique = np.unique(Dates, axis = 0)
        os.makedirs(base_path + 'Picks', exist_ok = True)
        os.makedirs(base_path + 'Catalog', exist_ok = True)
        os.makedirs(base_path + 'Calibration', exist_ok = True)
        for yr in np.unique(dates_unique[:,0]): os.makedirs(base_path + 'Picks' + seperator + '%d'%yr, exist_ok = True)
        for yr in np.unique(dates_unique[:,0]): os.makedirs(base_path + 'Catalog' + seperator + '%d'%yr, exist_ok = True)
        for yr in np.unique(dates_unique[:,0]): os.makedirs(base_path + 'Calibration' + seperator + '%d'%yr, exist_ok = True)
        ## Match amplitudes using reference pick file
        match_pick_amplitudes = True
        ip = cKDTree(Dates).query_ball_point(dates_unique, r = 0)

        ## Now write Catalog files for all detected sources
        for d, i in zip(dates_unique, ip):
            if match_pick_amplitudes == True:
                time_match_tol = 1.0 ## Should be even less than this
                str_load = base_path + 'Picks' + seperator + '%d'%d[0] + seperator + '%d_%d_%d_ver_%d.npz'%(d[0], d[1], d[2], n_ver_write)
                if os.path.isfile(str_load) == 1:
                    P = np.load(str_load)['P']
                    scale_ind = P[:,0].max()*1.5 # (P[:,0].max() - P[:,0].min())
                    scale_vec = np.array([1.0, scale_ind]).reshape(1,-1)
                    # time, ind, amp = P[:,0], P[:,1], P[:,2]
                    tree_picks = cKDTree((P[:,0:2]*scale_vec).sum(1, keepdims = True))
                    for j in i: ## For each associated pick list, find matched picks
                        if len(Picks_P[j]) > 0:
                            ifind_p = tree_picks.query((Picks_P[j][:,0:2]*scale_vec).sum(1, keepdims = True))[1]
                            diff_time = Picks_P[j][:,0] - P[ifind_p,0]
                            ifind_p_match = np.where(np.abs(diff_time) <= time_match_tol)[0]
                            Picks_P[j][ifind_p_match,2] = P[ifind_p[ifind_p_match],2] ## Write in amplitude values
                        if len(Picks_S[j]) > 0:
                            ifind_s = tree_picks.query((Picks_S[j][:,0:2]*scale_vec).sum(1, keepdims = True))[1]
                            diff_time = Picks_S[j][:,0] - P[ifind_s,0]
                            ifind_s_match = np.where(np.abs(diff_time) <= time_match_tol)[0]
                            Picks_S[j][ifind_s_match,2] = P[ifind_s[ifind_s_match],2] ## Write in amplitude values

            ## Save catalog file
            file_name_ext = 'continuous_days'
            ext_save = base_path + 'Catalog' + seperator + '%d'%d[0] + seperator + '%s_results_%s_%d_%d_%d_ver_%d.hdf5'%(config['name_of_project'], file_name_ext, d[0], d[1], d[2], n_ver_write)
            str_load = base_path + 'Picks' + seperator + '%d'%d[0] + seperator + '%d_%d_%d_ver_%d.npz'%(d[0], d[1], d[2], n_ver_write)
            z = h5py.File(ext_save, 'w')
            if os.path.isfile(str_load) == 1:
                P = np.load(str_load)['P']
                ind_use = np.unique(P[:,1]).astype('int')
            else:
                P = np.concatenate((np.concatenate([Picks_P[j] for j in i], axis = 0), np.concatenate([Picks_S[j] for j in i], axis = 0)), axis = 0)
                ind_use = np.unique(P[:,1]).astype('int') # np.unique(np.concatenate((np.concatenate([Picks_P[j] for j in i], axis = 0)[:,1], np.concatenate([Picks_S[j] for j in i], axis = 0)[:,1]), axis = 0))
            perm_vec = (-1*np.ones(len(locs))).astype('int')
            perm_vec[ind_use] = np.arange(len(ind_use))
            P_perm = np.copy(P)
            P_perm[:,1] = perm_vec[P[:,1].astype('int')]
            julday = int((UTCDateTime(d[0], d[1], d[2]) - UTCDateTime(d[0], 1, 1))/(3600*24)) + 1
            cnt_p = np.array([len(Picks_P[j]) for j in i])
            cnt_s = np.array([len(Picks_S[j]) for j in i])
            assert(P_perm[:,1].min() > -1)
            z['P'] = P
            z['P_perm'] = P_perm
            z['srcs'] = np.concatenate((Srcs[i][:,0:4], np.ones((len(i),1))), axis = 1) ## Assume weight vector is all ones for each source
            z['srcs_trv'] = Srcs[i]
            z['srcs_w'] = np.ones(len(i)) ## Assume weight vector is all ones for each source
            z['srcs_sigma'] = Uncertainity[i]
            z['locs'] = locs
            z['locs_use'] = locs[ind_use]
            z['date'] = np.array([d[0], d[1], d[2], julday])
            z['cnt_p'] = cnt_p ## Number of P picks per event
            z['cnt_s'] = cnt_s ## Number of S picks per event
            z['mag_r'] = Srcs[i,4]
            z['mag_trv'] = Srcs[i,4]
            for inc, j in enumerate(i):
                p_slice = Picks_P[j]
                s_slice = Picks_S[j]
                p_slice[:,1] = perm_vec[p_slice[:,1].astype('int')]
                s_slice[:,1] = perm_vec[s_slice[:,1].astype('int')]
                z['Picks/%d_Picks_P'%inc] = Picks_P[j]
                z['Picks/%d_Picks_S'%inc] = Picks_S[j]
                z['Picks/%d_Picks_P_perm'%inc] = p_slice
                z['Picks/%d_Picks_S_perm'%inc] = s_slice
            z.close()
            print('Finished saving %d/%d/%d (%d sources; %d P and %d S waves)'%(d[0], d[1], d[2], len(i), cnt_p.sum(), cnt_s.sum()))




    # print("Saving Region file (kind of depreciated)")
    # np.savez_compressed(
    #     base_path + '%s_region.npz'%name_of_project,
    #     lat_range=config['latitude_range'],
    #     lon_range=config['longitude_range'],
    #     depth_range=config['depth_range'],
    #     # deg_pad=config['degree_padding'],
    #     deg_pad=deg_padding,
    #     num_grids=config['number_of_grids'],
    #     n_spatial_nodes=config['number_of_spatial_nodes'],
    #     years=years,
    #     # load_initial_files=config['load_initial_files'],
    #     # use_pretrained_model=config['use_pretrained_model']
    # )


    path_to_file = base_path
    ## Make necessary directories (can edit this to write the days with picks in Picks)
    if os.path.isfile(path_to_file + '%s_process_days_list_ver_1.txt'%config["name_of_project"]) == 0:
        f = open(path_to_file + '%s_process_days_list_ver_1.txt'%config["name_of_project"], 'w')
        for j in range(5): ## Arbitrary 5 days to process
            f.write('%d/%d/%d \n'%(years[0], 1, j + 1)) ## Process days list (write which days want to process)
        f.close()

    os.makedirs(path_to_file + 'Picks', exist_ok=True)
    os.makedirs(path_to_file + 'Catalog', exist_ok=True)
    os.makedirs(path_to_file + 'Calibration', exist_ok=True) ## Can add an automatic download for calibration
    for year in years:
        os.makedirs(path_to_file + f'Picks/{year}', exist_ok=True)
        os.makedirs(path_to_file + f'Catalog/{year}', exist_ok=True)
        os.makedirs(path_to_file + f'Calibration/{year}', exist_ok=True)

    os.makedirs(path_to_file + 'Plots', exist_ok=True)
    os.makedirs(path_to_file + 'GNN_TrainedModels', exist_ok=True)
    os.makedirs(path_to_file + 'Grids', exist_ok=True)
    os.makedirs(path_to_file + '1D_Velocity_Models_Regional', exist_ok=True) ## Can change the name of this

    n_ver_velocity_model = 1
    # seperator = '\\' if '\\' in path_to_file else '/'
    # shutil.copy(path_to_file + '1d_velocity_model.npz', path_to_file + '1D_Velocity_Models_Regional' + seperator + f'{config["name_of_project"]}_1d_velocity_model_ver_{n_ver_velocity_model}.npz')
    os.makedirs(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'TravelTimeData', exist_ok=True)


    print("All files saved successfully!")
    print("✔ Script execution: Done")







# def setup_region(client: Client, config: dict, t0: UTCDateTime, tf: UTCDateTime):
#     """Set up region based on configuration."""
#     return client.get_stations(
#         starttime=t0,
#         endtime=tf,
#         network=config['network'],
#         station='*',
#         minlatitude=config['latitude_range'][0],
#         maxlatitude=config['latitude_range'][1],
#         minlongitude=config['longitude_range'][0],
#         maxlongitude=config['longitude_range'][1]
#     )[0]

# def extract_station_data(stations):
#     """Extract station code, location and elevation from the stations list."""
#     stas = [station.code for station in stations]
#     locs = [np.array([station.latitude, station.longitude, station.elevation]).reshape(1, -1) for station in stations]

#     stas = np.hstack(stas)
#     locs = np.vstack(locs)

#     iarg = np.argsort(locs[:, 0])
#     return locs[iarg], stas[iarg]

# def save_files(base_path: str, locs, stas, config, years):
#     """Save network, region, and 1D velocity model files."""
#     # Save network file
#     # np.savez_compressed(base_path + 'stations.npz', locs=locs, stas=stas)
#     name_of_project = config['name_of_project']

#     # Save region file
#     np.savez_compressed(
#         base_path + '%s_region.npz'%name_of_project,
#         lat_range=config['latitude_range'],
#         lon_range=config['longitude_range'],
#         depth_range=config['depth_range'],
#         deg_pad=config['degree_padding'],
#         num_grids=config['number_of_grids'],
#         n_spatial_nodes=config['number_of_spatial_nodes'],
#         years=years,
#         load_initial_files=config['load_initial_files'],
#         use_pretrained_model=config['use_pretrained_model']
#     )

    # # Save 1D velocity model
    # np.savez_compressed(
    #     base_path + '1d_velocity_model.npz',
    #     Depths=np.array(config['velocity_model']['Depths']),
    #     Vp=np.array(config['velocity_model']['Vp']),
    #     Vs=np.array(config['velocity_model']['Vs'])
    # )
