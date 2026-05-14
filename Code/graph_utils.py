
import yaml
import numpy as np
import os
import torch
from torch import optim, nn
import shutil
from collections import defaultdict
from sklearn.metrics import r2_score
# import pandas
import pathlib
import random
import json
import pdb
import sys
import scipy
import glob
from math import floor, sqrt
from torch_cluster import knn
from itertools import product
from torch_geometric.utils import from_networkx
from scipy.optimize import differential_evolution
from torch_geometric.utils import remove_self_loops, to_networkx
from torch_geometric.utils import sort_edge_index, subgraph # , is_sorted # , is_coalesced
from torch_geometric.utils import degree # , is_sorted # , is_coalesced
from torch_geometric.nn import MessagePassing
from skopt.utils import use_named_args
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.optimize import minimize, fsolve
from scipy.sparse.linalg import eigsh
from torch_scatter import scatter
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import scipy.stats.qmc as qmc
from scipy.spatial import KDTree
from skopt import gp_minimize
from skopt.space import Real
import scipy.sparse as sp
from scipy import stats
import networkx as nx
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
# from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import kneighbors_graph
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import lsqr
import scipy.sparse.linalg as spla
from collections import Counter
from utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

        center_out = ftrns1_fit(center_loc, rbest, x[3:].reshape(1, -1))
        out_unit_lat = (ftrns1_fit(center_loc + [0.001, 0.0, 0.0], rbest, x[3:].reshape(1, -1)) - center_out) / norm_lat
        out_unit_vert = (ftrns1_fit(center_loc + [0.0, 0.0, 10.0], rbest, x[3:].reshape(1, -1)) - center_out) / norm_vert

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

def compute_warped_expected_spacing(
    N,  # total number of points
    lat_range=None,
    lon_range=None,
    depth_range=None,
    time_range=None,  # T, full range = 2T
    use_time=None,
    scale_time=None,  # w_scale: length per unit time
    depth_boost = None,
    use_global=None,
    r_min = None,
    r_max = None,
    earth_radius=6378137.0
):
    # 1. --- METRIC AREA ---
    # Because of 1/r warping, the area is constant at all depths in metric space.
    # We use the surface area at earth_radius as the reference cross-section.
    if use_global:
        Area_metric = 4 * np.pi * earth_radius**2
    else:
        dlon_deg = (lon_range[1] - lon_range[0]) % 360
        if dlon_deg == 0 and lon_range[1] != lon_range[0]: 
            dlon_deg = 360
        dlon = np.deg2rad(dlon_deg)
        sin_diff = np.sin(np.deg2rad(lat_range[1])) - np.sin(np.deg2rad(lat_range[0]))
        Area_metric = earth_radius**2 * dlon * sin_diff

    # 2. --- METRIC DEPTH (Z-Stretch) ---
    # The physical thickness is stretched by the depth_boost factor.
    thickness_phys = abs(depth_range[1] - depth_range[0])
    # --- 1. Compute Metric Spatial Volume (The "Unrolled Slab") ---
    # Use earth_radius area to account for 1/r warp and thickness*depth_boost
    thickness_metric = (r_max - r_min) * depth_boost
    Volume_space_metric = Area_metric * thickness_metric    

    # --- 2. 3D Nominal Spacing (Spatial Projection) ---
    # This tells you: "If I only had 3D space, what would the spacing be?"
    hex_factor_3d = 0.74048
    nominal_spacing_space = (Volume_space_metric / (hex_factor_3d * N)) ** (1.0 / 3.0)    

    # --- 3. 4D Hypervolume (Space x Scaled Time) ---
    # Total time span (2T) stretched by scale_time
    Volume_4d_metric = Volume_space_metric * (2.0 * time_range * scale_time)    

    # --- 4. 4D Joint Spacing (The FPS Target) ---
    # This is the actual distance (in metric units) FPS will enforce
    nominal_spacing_4d = (Volume_4d_metric / N) ** (1.0 / 4.0)    

    # --- 5. "Raw" Nominal Time Spacing ---
    # This represents the temporal "slot" width in seconds.
    # We use the 4th-root of N to show how the 4D density partitions time.
    nominal_spacing_time = (2.0 * time_range) / (N ** (1.0 / 4.0))

    return Volume_4d_metric, Volume_space_metric, Area_metric, nominal_spacing_4d, nominal_spacing_space, nominal_spacing_time


def dlon_diff(lon_range):
    dlon_deg = (lon_range[1] - lon_range[0]) % 360
    if dlon_deg == 0 and lon_range[1] != lon_range[0]: 
        dlon_deg = 360 # Full circle
    # dlon = np.deg2rad(dlon_deg)
    return dlon_deg

def is_in_lon_range(lon, lon_min, lon_max):
    # Normalize everything to [0, 360]
    lon = lon % 360
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        return (lon >= lon_min) & (lon <= lon_max)
    else: # Crossing the seam
        return (lon >= lon_min) | (lon <= lon_max)

def u_to_geodetic_lat(u_random, lat_range):
    # WGS84 Constants
    e = 0.0818191908426215
    e2 = e**2
    
    def get_q(lat_deg):
        phi = np.deg2rad(lat_deg)
        s = np.sin(phi)
        # Using the standard USGS/Snyder form for q
        return (1 - e2) * ( (s / (1 - e2 * s**2)) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)) )

    # Step 1: Linear mapping in q-space (Area-preserving)
    q_min = get_q(lat_range[0])
    q_max = get_q(lat_range[1])
    q_target = q_min + u_random * (q_max - q_min)
    
    # Step 2: Inverse to Authalic Latitude (beta)
    q_polar = get_q(90.0)
    # Ensure numerical safety for arcsin
    beta = np.arcsin(np.clip(q_target / q_polar, -1.0, 1.0))
    
    # Step 3: Convert Authalic (beta) to Geodetic (phi)
    # WGS84 series coefficients
    P1 = (e2/3 + 31*e2**2/180 + 517*e2**3/5040)
    P2 = (23*e2**2/360 + 251*e2**3/3780)
    P3 = (761*e2**3/45360)
    
    phi_rad = beta + P1*np.sin(2*beta) + P2*np.sin(4*beta) + P3*np.sin(6*beta)
    return np.rad2deg(phi_rad)


def get_ellipsoid_paddings(lat_min, lat_max, buffer_m):
    # WGS84 Constants
    a = 6378137.0
    e2 = 0.00669437999014
    
    def get_radii(lat_deg):
        phi = np.deg2rad(np.clip(lat_deg, -89.9999, 89.9999)) # # lat_deg np.clip(lat_deg, -89.9999, 89.9999)
        denom = (1.0 - e2 * np.sin(phi)**2)**1.5
        M = a * (1.0 - e2) / denom  # Radius for Latitude
        N = a / np.sqrt(1.0 - e2 * np.sin(phi)**2) # Radius for Longitude
        return M, N

    # Calculate radii at both boundaries
    M_min, N_min = get_radii(lat_min)
    M_max, N_max = get_radii(lat_max)

    # Latitude padding (Degrees) - can vary slightly min vs max
    pad_lat_min = np.rad2deg(buffer_m / M_min)
    pad_lat_max = np.rad2deg(buffer_m / M_max)

    # Longitude padding (Degrees) - varies significantly min vs max
    pad_lon_min = np.rad2deg(buffer_m / (N_min * np.cos(np.deg2rad(lat_min))))
    pad_lon_max = np.rad2deg(buffer_m / (N_max * np.cos(np.deg2rad(lat_max))))

    return (pad_lat_min, pad_lat_max), (pad_lon_min, pad_lon_max)

def get_warped_metric_space(x_grid, depth_boost, scale_t, scale_val=10000.0, 
                            R_ref=6378137.0, density_weights = None, return_physical_units=False):



    earth_radius = 6378137.0
    ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
    ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


    # 1. Project to ECEF
    xyz_ecef = ftrns1_abs(x_grid[:, :3])
    
    # 2. Get the Unit Normal (Pointing 'Up' from center)
    # WGS84 specific normal logic
    a, b = 6378137.0, 6356752.314245
    n = xyz_ecef / np.array([a**2, a**2, b**2])
    n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
    
    # 3. Calculate Physical Radius and thickness
    depths = x_grid[:, 2:3]
    r_phys = R_ref - depths
    
    # 4. The Log-Radial Warp (Thickness in Log-Space)
    # If r_phys < R_ref (Deep): thickness_log is positive (distance down)
    # If r_phys > R_ref (High): thickness_log is negative (distance up)
    thickness_log = R_ref * np.log(R_ref / r_phys) 
    
    # 5. Construct the Manifold
    # First, project the point onto the 'Shell' at R_ref
    xyz_surface = (xyz_ecef / r_phys) * R_ref
    
    # Move 'down' (negative n_unit direction) by the warped thickness
    # This keeps the world oriented correctly even for points above R_ref
    xyz_warped = xyz_surface - (n_unit * thickness_log * depth_boost)
    
    # 6. Time scaling
    t_warped = x_grid[:, 3:4] * scale_t

    if density_weights is not None:
            # Flatten weights to (N, 1) for broadcasting
            w = density_weights.reshape(-1, 1)**(0.125)
            xyz_warped *= w
            t_warped *= w

    p4d_warped = np.hstack([xyz_warped, t_warped])

    # 7. GPU Precision Management
    if not return_physical_units:
        # Use a fixed scale_val so nominal_spacing is predictable
        origin = p4d_warped.mean(axis=0, keepdims=True)
        p4d_scaled = (p4d_warped - origin) / scale_val
    else:
        p4d_scaled = p4d_warped

    return p4d_scaled


def regular_sobolov(N, lat_range = None, lon_range = None, depth_range = None, time_range = None, use_time = True, use_global = None, scale_time = None, depth_boost = 1.0, N_target = None, use_station_density = False, buffer_scale = 0.0, r_min = None, r_max = None, use_spherical = False, run_checks = False):

    if use_spherical == False:
        a = 6378137.0
        b = 6356752.31424
    else:
        a = 6371e3
        b = 6371e3

    earth_radius = a # 6378137.0
    ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
    ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1

    if buffer_scale > 0.0:
        # 1. Get the 4D metric joint spacing (L)
        # This is the 'universal ruler' in the warped manifold
        metrics = compute_warped_expected_spacing(
            N, lat_range, lon_range, depth_range, time_range, 
            use_time, scale_time, depth_boost, use_global, r_min, r_max
        )
        Volume, V_space_metric, _, L_metric, _, _ = metrics    
        # 2. Conversion to Absolute (Physical) Units
        # We un-warp the metric spacing to find the equivalent physical distance
        # Horizontal: We use the 4D joint spacing to ensure isotropy in the buffer
        spatial_buffer_m = L_metric * buffer_scale 
        (pLat_min, pLat_max), (pLon_min, pLon_max) = get_ellipsoid_paddings(
            lat_range[0], lat_range[1], spatial_buffer_m
        )
        max_pLon = max(pLon_min, pLon_max)    
        # Depth: Un-warp by the boost factor
        pad_depth = (L_metric * buffer_scale) / depth_boost    
        # Time: Un-warp by the velocity scale
        pad_time = (L_metric * buffer_scale) / scale_time    
        # 3. Define Expanded Absolute Ranges
        expanded_lat = [lat_range[0] - pLat_min, lat_range[1] + pLat_max]
        expanded_lon = [lon_range[0] - max_pLon, lon_range[1] + max_pLon]
        expanded_depth = [depth_range[0] - pad_depth, depth_range[1] + pad_depth]
        # Important: If time_range is +/- T, expand both ends
        expanded_time = time_range + pad_time     
        # 4. Calculate N_updated based on the 4D Volume Ratio
        # This ensures the interior density remains exactly what the Warm Start intended
        Volume_expanded, _, _, _, _, _ = compute_warped_expected_spacing(
            N, expanded_lat, expanded_lon, expanded_depth, expanded_time, 
            use_time, scale_time, depth_boost, use_global, r_min, r_max
        )
        N_updated = int(np.ceil(N * (Volume_expanded / Volume)))



    if buffer_scale > 0.0:

        m = int(np.ceil(np.log2(N_updated)))
        N_sobol = 2**m

        u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random_base2(m)  # Sobol 4D
        assert((len(u) == N_sobol)*(len(u) >= N_updated))
        # assert(len(u) >= N_updated)

        if use_global == False:
            # phi = expanded_lon[0] + u[:,0]*(expanded_lon[1] - expanded_lon[0]) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
            phi = expanded_lon[0] + u[:,0]*dlon_diff(expanded_lon) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
            u_min = (1.0 + np.sin(np.deg2rad(expanded_lat[0])))/2.0
            u_max = (1.0 + np.sin(np.deg2rad(expanded_lat[1])))/2.0
            # theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
            # theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)
            theta = u_to_geodetic_lat(u[:,1], expanded_lat)

        else:
            phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
            # theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
            # theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)
            theta = u_to_geodetic_lat(u[:,1], [-90.0, 90.0])

        phi_wrapped = (phi + 180) % 360 - 180
        # r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
        # r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), expanded_depth[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
        xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi_wrapped.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
        # r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)

        n = xyz_surface / np.array([a**2, a**2, b**2])
        n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
        # Local radius from center to surface point
        r_surface = np.linalg.norm(xyz_surface, axis=1, keepdims=True)
        # --- STEP B: Cubic Height Sampling ---
        # depth_range[0] is Top (+), depth_range[1] is Bottom (-)
        h_top = depth_range[0]
        h_bot = depth_range[1]
        # u is Sobol variable [0, 1]
        # We use the cubic formula to get 'h' that respects volume growth
        r_top = r_surface + h_top
        r_bot = r_surface + h_bot
        # Corrected height:
        h_sampled = (r_bot**3 + u[:, [2]] * (r_top**3 - r_bot**3))**(1/3.0) - r_surface
        # --- STEP C: Final Positioning ---
        # Move from the surface XYZ along the Normal by h_sampled
        xyz = xyz_surface + (n_unit * h_sampled)
        x_grid = ftrns2_abs(xyz)

        # r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
        # r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
        # xyz = (r*xyz_surface)/r_surface
        # x_grid = ftrns2_abs(xyz) 

        if use_time == True:
            t = -expanded_time + 2.0 * expanded_time * u[:, [3]]
            x_grid = np.concatenate((x_grid, t), axis = 1)

        if use_global == False:

            lons_wrapped = (x_grid[:,1] + 180) % 360 - 180
            mask_points = (x_grid[:,0] >= lat_range[0]) & (x_grid[:,0] <= lat_range[1]) & \
                               is_in_lon_range(lons_wrapped, lon_range[0], lon_range[1]) & \
                              (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
                              (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range)) 

        else:

            # lons_wrapped = (x_grid[:,1] + 180) % 360 - 180
            mask_points = (x_grid[:,2] <= depth_range[1]) & (x_grid[:,2] >= depth_range[0]) & \
                              (x_grid[:,3] <= time_range) & (x_grid[:,3] >= (-time_range)) 



        ## Now retain only the fraction of boundary nodes that will emulate the right density of the target number of nodes
        if N_target is not None:
            ratio = (Volume_expanded - Volume)/Volume
            n_boundary_retain = int(N_target*ratio)
            ichoose = np.concatenate((np.where(mask_points == 1)[0], np.random.choice(np.where(mask_points == 0)[0], \
                size = n_boundary_retain, replace = False)), axis = 0)
            x_grid = x_grid[ichoose]
            mask_points = mask_points[ichoose]


        # --- CORRECTED DENSITY SANITY CHECK ---
        if (N_target is not None)*(run_checks == True):
            # 1. The density the Core WILL have after FPS finishes
            target_density_core = N_target / Volume
            # 2. The density the Buffer HAS right now (the ghosts we are keeping)
            n_buffer_retained = np.sum(mask_points == 0)
            actual_density_buffer = n_buffer_retained / (Volume_expanded - Volume)
            # 3. The Ratio (Target is 1.0)
            # This proves the "Wall of Ghosts" matches the "Future Grid"
            density_ratio = actual_density_buffer / target_density_core
            print(f"--- FPS Ghost-Pressure Match ---")
            print(f"Target Core Nodes: {N_target}")
            print(f"Retained Ghosts:   {n_buffer_retained}")
            print(f"Expected Core Density: {target_density_core:.2e}")
            print(f"Actual Ghost Density:  {actual_density_buffer:.2e}")
            print(f"Pressure Match Ratio:  {density_ratio:.4f} (Ideal: 1.0000)")

        if use_station_density == True:

            # 2. Probability-based rejection
            # Points in high-prob areas are kept; points in low-prob areas are likely dropped
            mask_vals = density_grid.fast_query(x_grid[:,0:2]).reshape(-1)
            # Normalize mask so max prob is 1.0
            keep_prob = mask_vals / np.max(mask_vals) 
            random_vals = np.random.rand(len(x_grid))
            x_grid = x_grid[random_vals < keep_prob]
            mask_points = mask_points[random_vals < keep_prob]

        return x_grid, mask_points


    else: # buffer_scale == 1.0:

        u = scipy.stats.qmc.Sobol(d = 4 if use_time else 3, scramble = True).random(N)  # Sobol 4D
        if use_global == False:
            # phi = lon_range[0] + u[:,0]*(lon_range[1] - lon_range[0]) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
            phi = lon_range[0] + u[:,0]*dlon_diff(lon_range) # dlon_orig = (lon_range[1] - lon_range[0]) % 360
            u_min = (1.0 + np.sin(np.deg2rad(lat_range[0])))/2.0
            u_max = (1.0 + np.sin(np.deg2rad(lat_range[1])))/2.0
            # theta = u_min + u[:,1]*(u_max - u_min) # *(180.0/np.pi) # np.arcsin(2 * u_lat_rescaled - 1)
            # theta = np.arcsin(2 * theta - 1)*(180.0/np.pi)
            theta = u_to_geodetic_lat(u[:,1], lat_range)

        else:
            phi = ((2 * np.pi * u[:, 0]) - np.pi)*(180.0/np.pi)                # longitude
            # theta = np.arcsin(1 - 2 * u[:,1])*(180.0/np.pi)
            # theta = (np.arccos(1 - 2 * u[:, 1]) - np.pi/2.0)*(180.0/np.pi)            # colatitude (equal-area on sphere)
            theta = u_to_geodetic_lat(u[:,1], [-90.0, 90.0])


        r_min_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), depth_range[0]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
        r_max_local = np.linalg.norm(ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), depth_range[1]*np.ones((len(phi),1))), axis = 1)), axis = 1, keepdims = True)
        xyz_surface = ftrns1_abs(np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1), np.zeros((len(phi),1))), axis = 1))
        # r_surface = np.linalg.norm(xyz_surface, axis = 1, keepdims = True)
        # r = (r_min_local**3 + u[:, [2]] * (r_max_local**3 - r_min_local**3)) ** (1/3.0)
        # xyz = (r*xyz_surface)/r_surface
        # x_grid = ftrns2_abs(xyz)

        n = xyz_surface / np.array([a**2, a**2, b**2])
        n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
        # Local radius from center to surface point
        r_surface = np.linalg.norm(xyz_surface, axis=1, keepdims=True)
        # --- STEP B: Cubic Height Sampling ---
        # depth_range[0] is Top (+), depth_range[1] is Bottom (-)
        h_top = depth_range[0]
        h_bot = depth_range[1]
        # u is Sobol variable [0, 1]
        # We use the cubic formula to get 'h' that respects volume growth
        r_top = r_surface + h_top
        r_bot = r_surface + h_bot
        # Corrected height:
        h_sampled = (r_bot**3 + u[:, [2]] * (r_top**3 - r_bot**3))**(1/3.0) - r_surface
        # --- STEP C: Final Positioning ---
        # Move from the surface XYZ along the Normal by h_sampled
        xyz = xyz_surface + (n_unit * h_sampled)
        x_grid = ftrns2_abs(xyz)


        if use_time == True:
            t = -time_shift_range + 2 * time_shift_range * u[:, [3]]
            x_grid = np.concatenate((x_grid, t), axis = 1)

        if use_station_density == True:

            # 2. Probability-based rejection
            # Points in high-prob areas are kept; points in low-prob areas are likely dropped
            mask_vals = density_grid.fast_query(x_grid[:,0:2]).reshape(-1)
            # Normalize mask so max prob is 1.0
            keep_prob = mask_vals / np.max(mask_vals) 
            random_vals = np.random.rand(len(x_grid))
            x_grid = x_grid[random_vals < keep_prob]
            mask_points = mask_points[random_vals < keep_prob]


        return x_grid



def farthest_point_sampling(points_candidates, target_N, scale_time = None, depth_boost = 1.0, use_station_density = False, mask_candidates = None, device = device):
    
    """
    points: [N, 3] or [N, 4] (already scaled/transformed)
    target_N: Number of 'real' points to collect
    mask_candidates: Tensor/Array of 1s (real) and 0s (buffer/mirrored)
    """    


    earth_radius = 6378137.0
    ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
    ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


    # scale_val = 10000.0
    points = points_candidates.copy()
    # points, radii = get_metric_space(points, depth_boost, scale_time)
    # points = get_warped_metric_space(points, depth_boost, scale_time)

    if use_station_density == True:
        scale_points = density_grid.fast_query(points[:,0:2]) # .reshape(-1,1)
        points = get_warped_metric_space(points, depth_boost, scale_time, density_weights = scale_points)

    else:
        points = get_warped_metric_space(points, depth_boost, scale_time)


    # points = get_fps_specific_space(points, depth_boost, scale_time)


    # if points.shape[1] == 4: points[:, 3] *= scale_time
    # if depth_boost != 1.0: points = ftrns1_abs(ftrns2_abs(points) * np.array([1.0, 1.0, depth_boost, 1.0]))
    # origin = points[:, :3].mean(axis = 0, keepdims = True)
    # points[:, :3] -= origin
    points = torch.as_tensor(points, device = device, dtype = torch.float64)
    # radii = torch.as_tensor(radii / scale_val, device = device, dtype = torch.float64)

    if mask_candidates is None: mask_candidates = np.ones(len(points))
    mask = torch.as_tensor(mask_candidates, device = device, dtype = torch.bool)
    assert(len(mask) == len(points))
    assert(mask.sum().item() >= target_N)
    N, C = points.shape
    
    # 1. Initialize distance array
    # If we have boundary points (mask == 0), we pre-calculate distances to them
    distance = torch.full((N,), float('inf'), device = device, dtype = torch.float64)
    
    boundary_indices = torch.where(~mask)[0]
    real_indices = torch.where(mask)[0]

    # 2. Pre-process boundary Points (The repulsion field)
    if len(boundary_indices) > 0:
        # Optimization: Update distance array with the proximity to any ghost point
        # For very large N, we do this in chunks to avoid OOM
        for i in range(0, len(boundary_indices), 500):
            batch = boundary_indices[i:i+500]
            # dists shape: [len(batch), N]
            dists = torch.cdist(points[batch], points, p=2)**2
            distance = torch.min(distance, torch.min(dists, dim=0)[0])
        
        # Ensure ghost points themselves are never selected
        distance[boundary_indices] = -1.0

    # 3. Choose the first REAL point
    # Instead of random, we pick the point farthest from the boundary ghosts
    # If no ghosts exist, we default to the point closest to the centroid
    if len(boundary_indices) > 0:
        farthest = torch.argmax(distance).item()
    else:
        centroid = points[real_indices].mean(0, keepdims=True)
        dist_centroid = torch.sum((points[real_indices] - centroid)**2, dim=1)
        farthest = real_indices[torch.argmin(dist_centroid)].item()

    collected_indices = []
    cnt_found = 0

    # 4. Main FPS Loop
    while cnt_found < target_N:
        collected_indices.append(farthest)
        cnt_found += 1 # We only ever pick real points now
            
        centroid_pt = points[farthest, :].view(1, C)
        dist = torch.sum((points - centroid_pt) ** 2, dim=-1)
        
        distance = torch.min(distance, dist)
        distance[farthest] = -1.0
        
        if cnt_found < target_N:
            farthest = torch.argmax(distance).item()

    # Final Filter
    final_indices = torch.tensor(collected_indices, device=device)
    # return ftrns2_abs(points_candidates[final_indices.cpu().numpy()])
    return points_candidates[final_indices.cpu().numpy()]


def get_wgs84_area_val(lat_deg):

    """Computes the WGS84 area-proportional value for a given latitude."""
    lat_rad = np.deg2rad(lat_deg)
    e2 = 0.00669437999014  # WGS84 eccentricity squared
    e = np.sqrt(e2)
    sin_phi = np.sin(lat_rad)
    sin_phi = np.clip(sin_phi, -0.999999, 0.999999)

    # Standard formula for ellipsoidal surface area relative to latitude
    term1 = (1 - e2) * sin_phi / (1 - e2 * sin_phi**2)
    term2 = ((1 - e2) / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
    return term1 - term2


def get_q_wgs84(lat_val):
    # WGS84 Constants
    e = 0.0818191908426215
    e2 = e**2
    
    def get_q(lat_deg):
        phi = np.deg2rad(lat_deg)
        s = np.sin(phi)
        # Using the standard USGS/Snyder form for q
        return (1 - e2) * ( (s / (1 - e2 * s**2)) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)) )

    # Step 1: Linear mapping in q-space (Area-preserving)
    q_val = get_q(lat_val)

    return q_val
   
def get_simple_density_ratio(normalized_data, N, frac_edge=0.1):
    num_bins = int(1.0 / frac_edge)
    # Histogram on the [0, 1] range
    counts, _ = np.histogram(normalized_data, bins=num_bins, range=(0.0, 1.0))
    
    expected_per_bin = N / num_bins
    
    # Boundary Ratio: (Average of both edge bins) / (Expected)
    # A value of 1.0 means the edges have the same density as the bulk.
    boundary_ratio = (counts[0] + counts[-1]) / (2 * expected_per_bin + 1e-9)
    
    return boundary_ratio, counts

def check_boundary_densities(x_grid, lat_range, lon_range, depth_range, time_range, use_global = False):

    N = len(x_grid)
    boundary_health = {}
    e = 0.0818191908426  # WGS84 Eccentricity


    earth_radius = 6378137.0
    ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
    ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1



    def get_q_wgs84(lat_deg):
        sin_phi = np.sin(np.deg2rad(lat_deg))
        sin_phi = np.clip(sin_phi, -0.999999, 0.999999)
        t1 = sin_phi / (1 - e**2 * sin_phi**2)
        t2 = (1 / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
        return (1 - e**2) * (t1 - t2)

    # --- 1. LATITUDE (Authalic Transformation) ---
    q_actual = get_q_wgs84(x_grid[:, 0])
    q_min = get_q_wgs84(lat_range[0])
    q_max = get_q_wgs84(lat_range[1])
    val_lat = (q_actual - q_min) / (q_max - q_min + 1e-12)

    # --- 2. DEPTH (Volumetric Transformation) ---
    r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
    r_bound_0 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[0])]), axis=1)
    r_bound_1 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, depth_range[1])]), axis=1)
    
    r_min_i = np.minimum(r_bound_0, r_bound_1)
    r_max_i = np.maximum(r_bound_0, r_bound_1)
    val_depth = (r_actual**3 - r_min_i**3) / (r_max_i**3 - r_min_i**3 + 1e-12)

    # --- 3. TIME & LON (Linear Transformation) ---
    val_time = (x_grid[:, 3] - (-time_range)) / (2 * time_range + 1e-12)
    
    if not use_global:
        val_lon = (x_grid[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0] + 1e-12)
    else:
        val_lon = None # Skip boundary check for global periodic lon

    # --- 4. UNIFIED DENSITY CALCULATION ---
    processed_dims = {
        'Lat': val_lat,
        'Lon': val_lon,
        'Depth': val_depth, 
        'Time': val_time, 
    }

    for name, val in processed_dims.items():
        if val is None:
            boundary_health[name] = 1.0
            continue
            
        # Every dimension now uses the same simple "flat" check
        ratio, counts = get_simple_density_ratio(val, N)
        boundary_health[name] = ratio
        
    return boundary_health

def extend_geo_range(lat_range, lon_range, w_phys, multiplier=2.0):
    """
    Extends a lat/lon range by a buffer based on physical distance (meters).
    """
    buffer_m = w_phys * multiplier
    
    # 1. Average latitude for longitude scaling
    lat_avg = np.mean(lat_range)
    lat_rad = np.radians(lat_avg)
    
    # 2. Conversion factors (Meters per Degree)
    # Constants for WGS84 ellipsoid
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    
    # 3. Calculate Degree Offsets
    d_lat = buffer_m / m_per_deg_lat
    d_lon = buffer_m / m_per_deg_lon
    
    new_lat_range = [lat_range[0] - d_lat, lat_range[1] + d_lat]
    new_lon_range = [lon_range[0] - d_lon, lon_range[1] + d_lon]
    
    return new_lat_range, new_lon_range

def compute_cdf_analysis(x_grid, ranges):

    N = len(x_grid)
    empirical_cdf = np.linspace(0, 1, N)
    cdf_loss = 0
    
    for i in range(x_grid.shape[1]):

        if i == 0: # LATITUDE
            q_vals = get_wgs84_area_val(x_grid[:, 0])
            q_bounds = sorted([get_wgs84_area_val(ranges[0][0]), get_wgs84_area_val(ranges[0][1])])
            val = (q_vals - q_bounds[0]) / (q_bounds[1] - q_bounds[0] + 1e-12)

        elif i == 1: # LONGITUDE
            # For longitude, we use dlon_diff to handle the wrap-around correctly
            val = (x_grid[:, 1] - ranges[1][0]) / (dlon_diff([ranges[1][0], ranges[1][1]]) + 1e-12)

        elif i == 2: # DEPTH
            # Get actual distances from geocenter
            r_actual = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
            
            # Project the bounds to absolute radii
            r_b1 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, ranges[2][0])]), axis=1)
            r_b2 = np.linalg.norm(ftrns1_abs(np.c_[x_grid[:,:2], np.full(N, ranges[2][1])]), axis=1)
            
            # Use absolute min/max to ensure (r_max^3 - r_min^3) is always positive
            r_min = np.minimum(r_b1, r_b2)
            r_max = np.maximum(r_b1, r_b2)
            
            val = (r_actual**3 - r_min**3) / (r_max**3 - r_min**3 + 1e-12)

        elif i == 3: # TIME
            val = (x_grid[:, 3] - ranges[3][0]) / (ranges[3][1] - ranges[3][0] + 1e-12)

        # CRITICAL: Clip to [0, 1] to prevent the optimizer from chasing outliers
        val = np.clip(val, 0.0, 1.0)
        
        sorted_vals = np.sort(val)
        cdf_loss += np.mean(np.abs(sorted_vals - empirical_cdf))

    return cdf_loss / x_grid.shape[1]


def fit_domain_budget_aware(W_phys_min, W_t_min, lat_range, lon_range, depth_range, time_range, 
                            N_max=6500, N_min=150, depth_boost = 1.0, use_global=False):
    """
    Calculates the best resolution for a fixed node budget while 
    strictly respecting the West-to-East longitude span.
    """
    earth_radius = 6378137.0
    
    # 1. Calculate Metric Volume (Spatial)
    if use_global:
        Area_m = 4 * np.pi * earth_radius**2
    else:
        # --- The Stable Date Line Logic ---
        # Explicitly: Index 1 (East) minus Index 0 (West)
        dlon_deg = lon_range[1] - lon_range[0]

        # If negative, we crossed the Date Line (e.g., 170 to -170)
        if dlon_deg < 0:
            dlon_deg += 360

        dlon = np.deg2rad(dlon_deg)

        # Handle the "Full Circle" case (e.g., [-180, 180])
        if dlon == 0 and lon_range[1] != lon_range[0]:
            dlon = 2 * np.pi
            
        sin_diff = np.abs(np.sin(np.deg2rad(lat_range[1])) - np.sin(np.deg2rad(lat_range[0])))
        Area_m = earth_radius**2 * dlon * sin_diff

    # 4D Volume: Space * Depth * Time
    vol_space = Area_m * np.abs(depth_range[1] - depth_range[0]) * depth_boost
    total_time_span = 2.0 * time_range # +/- window
    
    # Aspect ratio of the physics (m/s)
    s_ideal = W_phys_min / W_t_min
    
    # 2. Calculate N needed for "Perfect" Resolution
    # Formula: N = Vol_4d / (W_phys^3 * W_t)
    N_perfect = (vol_space * total_time_span) / (W_phys_min**3 * W_t_min)
    
    # 3. Budget Logic
    if N_perfect <= N_max:
        # We are under budget! Use the sharp physical kernels.
        # But don't go below N_min or the GNN graph becomes too sparse.
        final_N = int(max(N_min, N_perfect))
        final_W_phys = W_phys_min
        final_W_t = W_t_min
    else:
        # We are over budget. Broaden kernels to fit N_target.
        # Derived from: N_target = (Vol * s_ideal) / W^4
        final_N = N_max
        final_W_phys = ((vol_space * total_time_span * s_ideal) / N_max)**0.25
        final_W_t = final_W_phys / s_ideal

    final_scale_time = final_W_phys / final_W_t
    
    return final_N, final_scale_time, final_W_phys, final_W_t

class SamplingTuner:

    def __init__(self, target_N, lat_range, lon_range, depth_range, time_range, scale_time_effective = None, use_time_shift = True, use_global = False, device = device):

        from skopt.space import Real
        self.target_N = target_N
        # Store ranges as [min, max] pairs
        self.ranges = [lat_range, lon_range, depth_range, [-time_range, time_range]]
        self.device = device
        self.time_range = time_range
        self.use_global = use_global
        self.use_time_shift = use_time_shift
        self.scale_time_effective = scale_time_effective
        
        # 1. Define Search Space
        # scale_t: km/s
        # depth_boost: dimensionless vertical stretch
        # buffer_scale: multiplier for the nominal spacing
        self.space = [
            Real(1e3, 100e3, prior = 'log-uniform', name='scale_t'),      
            Real(0.5, 3.0, name='depth_boost'),   
            Real(1.5, 2.5, name='buffer_scale')    # prior='log-uniform',
        ]

    def optimize(self, n_calls = 90):

        """Runs Bayesian Optimization to find the triplet of parameters."""

        @use_named_args(self.space)
        def objective(scale_t, depth_boost, buffer_scale, use_station_density = False, use_normalized_mean = True):

            # 1. GENERATE CANDIDATES
            up_sample_factor = 20 if self.use_time_shift else 10
            if use_station_density == True: up_sample_factor = up_sample_factor*5
            number_candidate_nodes = up_sample_factor * self.target_N

            trial_points, mask_points = regular_sobolov(
                number_candidate_nodes, 
                lat_range=self.ranges[0], 
                lon_range=self.ranges[1], 
                depth_range=self.ranges[2], 
                time_range=self.time_range, 
                use_time=self.use_time_shift, 
                use_global=self.use_global, 
                scale_time=scale_t,
                depth_boost = depth_boost, 
                N_target=self.target_N, 
                buffer_scale=buffer_scale
            )        


            x_grid = farthest_point_sampling(
                trial_points, 
                self.target_N, 
                scale_time=scale_t, 
                depth_boost=depth_boost, 
                mask_candidates=mask_points,
                device = self.device
            )       

            # farthest_point_sampling(points_candidates, target_N, scale_time = None, depth_boost = 1.0, mask_candidates = None, device = 'cpu')

            # 3. PROJECT TO SCALED METRIC SPACE (For CV and Anisotropy)
            # Use depth_boost on the 3rd column and scale_t on the 4th
            # scaling_vector = np.array([1.0, 1.0, depth_boost, scale_t])
            # x_proj_scaled = ftrns1_abs(x_grid * scaling_vector)        

            x_proj_scaled = get_warped_metric_space(x_grid, depth_boost, scale_t)
            # --- PART A: Uniformity (Scaled World) ---
            tree = cKDTree(x_proj_scaled)
            nn_dist = tree.query(x_proj_scaled, k=2)[0][:, 1]
            cv = np.std(nn_dist) / (np.mean(nn_dist) + 1e-9)    


            # 1. Use the warped metric for distances
            if use_normalized_mean == True:
                x_metric_4d1 = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units=True)
                tree_4d = cKDTree(x_metric_4d1)
                dist_4d1, idx_nn1 = tree_4d.query(x_metric_4d1, k=2)
                nn_4d1 = dist_4d1[:, 1]
                nn_indices1 = idx_nn1[:, 1]

                # 2. Final Metric Pass (Ground Truth)
                metrics = compute_warped_expected_spacing(
                    self.target_N, 
                    lat_range=self.ranges[0], 
                    lon_range=self.ranges[1],
                    depth_range=self.ranges[2], 
                    time_range=self.time_range,
                    scale_time=scale_t, 
                    depth_boost=depth_boost, 
                    use_global=self.use_global,
                    r_min = self.r_min,
                    r_max = self.r_max
                )

                volume_4d_warped, _, _, _, _, _ = metrics

                # cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)
                # 2. Use the warped volume for the density expectation
                # Standard 4D Poisson constant is 0.463
                # This accounts for the 4D hypersphere volume constant

                expected_mean = 0.463 * (volume_4d_warped / self.target_N)**(0.25)
                norm_mean = np.mean(nn_4d1) / expected_mean
                penalty_norm = max(0, 1.4 - norm_mean) ** 2
                cv = cv + 1.0*penalty_norm


            densities = check_boundary_densities(x_grid, self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3][1], self.use_global)
            # Weighted penalty: we care most about Depth (Surface) and Time boundaries
            
            # penalty_boundary = (abs(1.0 - densities['Depth']) * 1.0 +  ## Make these weights proportional to volume
            #            abs(1.0 - densities['Time']) * 1.0 + 
            #            abs(1.0 - densities['Lat']) * 1.0 + 
            #            abs(1.0 - densities['Lon']) * 1.0)/4.0

            penalty_boundary = (
                (1.0 - densities['Depth'])**2 + 
                (1.0 - densities['Time'])**2 + 
                (1.0 - densities['Lat'])**2 + 
                (1.0 - densities['Lon'])**2
            ) / 4.0


            # penalty = (abs(1.0 - densities['Depth']) * 15.0 + 
            #            abs(1.0 - densities['Time']) * 10.0 + 
            #            abs(1.0 - densities['Lat']) * 5.0)

            # 4. CDF ANALYSIS (The WGS84 "Gold Standard")
            cdf_loss = compute_cdf_analysis(x_grid, self.ranges)


            # reg_penalty = (
            #     0.005 * (np.log10(scale_t / 5000)**2) +      # Prefer scale_t near 5000
            #     0.010 * (depth_boost - 1.0)**2 +             # Prefer depth_boost near 1.0
            #     0.010 * (buffer_scale - 1.0)**2              # Prefer buffer_scale near 1.0
            # )

            reg_penalty = (
                0.1 * (np.log10(scale_t / self.scale_time_effective)**2)      # Prefer scale_t near 5000
            )

            # cdf_loss = cdf_loss/x_grid.shape[1] ## Normalize by number of dimensions used
            # --- PART C: The Sanity Check (Anisotropy) ---
            # spreads = np.std(x_proj_scaled, axis=0)
            # anisotropy = np.max(spreads) / (np.min(spreads) + 1e-9)
            # penalty = np.maximum(0, np.log10(anisotropy) - 1.0)**2   

            return cv + (3.0 * cdf_loss) + (0.2 * penalty_boundary) + (1.0 * reg_penalty)

        # res = gp_minimize(objective, self.space, n_calls = n_calls, n_initial_points = 5, initial_point_generator = 'sobol', verbose = True) # random_state = 42
        res = gp_minimize(objective, self.space, n_calls = n_calls, verbose = True) # random_state = 42

        best_scale_t, best_depth_boost, best_buffer_scale = res.x
        

        _, _, _, nominal_spacing_4d, _, _ = compute_warped_expected_spacing(
            self.target_N,  # total number of points
            lat_range=self.ranges[0],
            lon_range=self.ranges[1],
            depth_range=self.ranges[2],
            time_range=self.time_range,  # T, full range = 2T
            use_time=self.use_time_shift,
            scale_time=best_scale_t,  # w_scale: length per unit time
            depth_boost = best_depth_boost,
            use_global=use_global,
            r_min = self.r_min,
            r_max = self.r_max)

        # nominal_spacing = (total_vol / self.target_N)**(1/4)
        # buffer_width_phys = best_buffer_scale * nominal_spacing
        buffer_width_phys = best_buffer_scale * nominal_spacing_4d # [0]

        print("\n--- Optimization Results ---")
        print(f"Optimal scale_t:     {best_scale_t:.3f} m/s")
        print(f"Optimal depth_boost: {best_depth_boost:.3f}")
        print(f"Optimal buffer_scale: {best_buffer_scale:.3f}")
        print(f"Effective Padding:   {buffer_width_phys:.2f} (units)")

        return {
            'scale_t': best_scale_t,
            'depth_boost': best_depth_boost,
            'buffer_scale': best_buffer_scale,
            'buffer_width_phys': buffer_width_phys
        }

def compute_final_grid_health(x_grid, scale_t, depth_boost, lat_range, lon_range, depth_range, time_range, buffer_scale, volume_4d_warped, use_global = False):
    
    N = len(x_grid)
    

    earth_radius = 6378137.0
    ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
    ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1


    # 1. Use the warped metric for distances
    x_metric_4d = get_warped_metric_space(x_grid, depth_boost, scale_t, return_physical_units=True)
    tree_4d = cKDTree(x_metric_4d)
    dist_4d, idx_nn = tree_4d.query(x_metric_4d, k=2)
    nn_4d = dist_4d[:, 1]
    nn_indices = idx_nn[:, 1]
    cv_4d = np.std(nn_4d) / (np.mean(nn_4d) + 1e-9)

    # 2. Use the warped volume for the density expectation
    # Standard 4D Poisson constant is 0.463
    # This accounts for the 4D hypersphere volume constant
    expected_mean = 0.463 * (volume_4d_warped / N)**(0.25)
    norm_mean = np.mean(nn_4d) / expected_mean
    void_ratio = np.quantile(nn_4d, 0.99) / (np.mean(nn_4d) + 1e-9)

    # --- 3. GRAPH MARGINALS & EFFECTIVE VELOCITY ---
    # Spatial part (in km)
    x_phys_3d_km = ftrns1_abs(x_grid[:, :3]) / 1000.0
    dist_space = np.linalg.norm(x_phys_3d_km - x_phys_3d_km[nn_indices], axis=1)
    
    # Temporal part (in seconds)
    dist_time = np.abs(x_grid[:, 3] - x_grid[nn_indices, 3])
    
    avg_space_km = np.mean(dist_space)
    avg_time_s = np.mean(dist_time)
    min_space_km = np.min(dist_space)
    min_time_s = np.min(dist_time)

    # V_eff: How fast does one have to travel to reach the 4D neighbor?
    # Higher scale_t = Lower V_eff (because time is 'longer')
    v_eff = avg_space_km / (avg_time_s + 1e-9)
    cv_space_4d = np.std(dist_space) / (avg_space_km + 1e-9)
    cv_time_4d = np.std(dist_time) / (avg_time_s + 1e-9)


    # --- 4. MARGINAL VOID ANALYSIS ---
    # Spatial Void (3D): Largest spatial jump in the 4D neighbor graph
    x_phys_3d = ftrns1_abs(x_grid[:, :3])
    dist_space_m = np.linalg.norm(x_phys_3d - x_phys_3d[nn_indices], axis=1)
    # Void ratio = 99th percentile distance / Mean distance
    space_void_ratio = np.quantile(dist_space_m, 0.99) / (np.mean(dist_space_m) + 1e-9)
    
    # Time Void (1D): Largest temporal gap between events
    dist_time_s = np.abs(x_grid[:, 3] - x_grid[nn_indices, 3])
    time_void_ratio = np.quantile(dist_time_s, 0.99) / (np.mean(dist_time_s) + 1e-9)


    # compute_boundary_biases(x_grid, x_phys_3d, nn_4d, lat_range, lon_range, depth_range, time_range, scale_t, depth_boost
    # bias, bias_lat, bias_lon, bias_depth, bias_time, bias_masks = compute_boundary_biases(x_grid, nn_4d, lat_range, lon_range, depth_range, time_range) # scale_t, depth_boost
    boundary_health = check_boundary_densities(x_grid, lat_range, lon_range, depth_range, time_range, use_global = use_global)
    bias_lat = boundary_health['Lat']
    bias_lon = boundary_health['Lon']
    bias_depth = boundary_health['Depth']
    bias_time = boundary_health['Time']


    # --- 5. WGS84 TRANSPARENCY (CDF R2) ---
    cdf_r2s = {}
    emp_cdf = np.arange(N) / (N - 1)
    

    # Lat (Authalic)
    q = get_wgs84_area_val(x_grid[:, 0])
    q_min, q_max = get_wgs84_area_val(lat_range[0]), get_wgs84_area_val(lat_range[1])
    cdf_r2s["Lat"] = r2_score(emp_cdf, np.sort((q - q_min)/(q_max - q_min + 1e-12)))
    
    # Lon (Linear)
    cdf_r2s["Lon"] = r2_score(emp_cdf, np.sort((x_grid[:, 1] - lon_range[0])/(lon_range[1] - lon_range[0])))
    
    # Depth (WGS84 Vol Frac)
    r_min_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[0])])), axis=1)
    r_max_l = np.linalg.norm(ftrns1_abs(np.hstack([x_grid[:,:2], np.full((N,1), depth_range[1])])), axis=1)
    r_act = np.linalg.norm(ftrns1_abs(x_grid[:, :3]), axis=1)
    vol_f = (r_act**3 - r_min_l**3) / (r_max_l**3 - r_min_l**3 + 1e-12)
    cdf_r2s["Depth"] = r2_score(emp_cdf, np.sort(vol_f))
    
    # Time (Linear)
    cdf_r2s["Time"] = r2_score(emp_cdf, np.sort((x_grid[:, 3] - (-time_range))/(2*time_range)))

    # --- 5. TERMINAL OUTPUT ---
    print(f"\n{'='*65}")
    print(f"       GEOMETRIC GRID HEALTH REPORT (WGS84-4D)")
    print(f"{'='*65}")
    print(f"Nodes (N): {N:<8} | scale_t: {scale_t:<8.1f} | time range: {time_range:<.2f} | d_boost: {depth_boost:<.2f} | buffer_scale: {buffer_scale:<.2f}")
    
    def get_bar(val, ideal_min, ideal_max):
        bar = ["-"] * 20
        pos = int(min(max(val / (ideal_max * 1.5), 0), 1) * 19)
        bar[pos] = "O"
        return "".join(bar)

    print(f"\n[1] 4D Metric Uniformity")
    print(f"    CV NN:           {cv_4d:.4f}  [{get_bar(cv_4d, 0.2, 0.3)}] (Goal: <0.2)")
    print(f"    Normalized Mean: {norm_mean:.4f}  [{get_bar(norm_mean, 1.4, 1.7)}] (Goal: >1.4)")
    print(f"    Void Ratio:      {void_ratio:.4f}  [{get_bar(void_ratio, 2.0, 3.0)}] (Goal: <3.0)")

    print(f"\n[2] Graph Neighbor Marginals (4D Links)")
    print(f"    Avg Spatial Gap: {avg_space_km:.2f} km  (CV: {cv_space_4d:.3f})")
    print(f"    Avg Temporal Gap: {avg_time_s:.2f} s   (CV: {cv_time_4d:.3f})")
    print(f"    Min Spatial Gap: {min_space_km:.2f} km ")
    print(f"    Min Temporal Gap: {min_time_s:.2f} s  \n")

    print(f"\n[3] Voids and Clustering ")
    print(f"    Effective Velocity: {v_eff:.2f} km/s {'[PHYSICAL]' if 4<v_eff<10 else '[STRETCHED]'}")
    print(f"    Void Ratio (space):  {space_void_ratio:.4f}  [{get_bar(space_void_ratio, 2.0, 3.0)}] (Goal: <3.0)")
    print(f"    Void Ratio (time):  {time_void_ratio:.4f}  [{get_bar(time_void_ratio, 2.0, 3.0)}] (Goal: <3.0)")
    collision_count = np.sum(nn_4d < nn_4d.mean()*0.75) ## Anything within half the average distance
    print(f"    Collision Check: {collision_count} nodes < half avg. distance apart.")


    # --- [3] Boundary & Edge Health ---
    print(f"\n[4] Boundary & Edge Health (Bias Ratio)")
    def format_bias(name, val):
        status = "OK" if 0.7 < val < 1.3 else "BIASED"
        return f"    {name:12}: {val:.3f} [{get_bar(val, 0.5, 1.5)}] ({status})"
    print(format_bias("Temporal", bias_time))
    print(format_bias("Depth/Radial", bias_depth))
    print(format_bias("Lat", bias_lat))
    if not use_global:
        print(format_bias("Lon", bias_lon))


    # --- 6. VELOCITY & ISOTROPY DIAGNOSTIC ---
    # The target is for v_eff (physical) to be close to scale_t (metric)
    v_mismatch = v_eff / (scale_t / 1000.0) # Ratio of Graph Speed to Metric Speed    

    print(f"\n[5] Physical Graph Balance")
    print(f"    Target Velocity (Scale): {scale_t/1000.0:.2f} km/s")
    print(f"    Actual Graph Velocity:   {v_eff:.2f} km/s")
    status_v = "BALANCED" if 0.8 < v_mismatch < 1.25 else "STRETCHED"
    print(f"    Velocity Mismatch:       {v_mismatch:.3f}x [{get_bar(v_mismatch, 0.5, 2.0)}] ({status_v})")    

    # Predicted vs Actual spacing
    # expected_dx_km = (volume_4d_warped / (N * scale_t * 2 * time_range))**(1/3) / 1000.0
    expected_dx_km = (volume_4d_warped / (N**0.75 * scale_t * 2 * time_range))**(1/3) / 1000.0

    print(f"    Predicted Space Res:     {expected_dx_km:.2f} km")
    print(f"    Actual Space Res:        {avg_space_km:.2f} km")


    print(f"\n[6] WGS84 Transparency (CDF R2 Scores)")
    for name, score in cdf_r2s.items():
        status = "PASS" if score > 0.98 else "WARN"
        print(f"    {name:6} R2: {score:.6f}  [{'#'*int(score*20):<20}] {status}")
    

    print(f"{'='*65}\n")
    return {"cv_4d": cv_4d, "min_dist": np.min(dist_space), "collisions": collision_count, "cdf_r2s": cdf_r2s, "v_eff": v_eff}


def perform_ks_density_test(x_grid, lat_range):
    # 1. Transform Latitudes to normalized q-space [0, 1]
    # (Using your get_q_wgs84 function)
    q_actual = get_q_wgs84(x_grid[:, 0])
    q_min = get_q_wgs84(lat_range[0])
    q_max = get_q_wgs84(lat_range[1])
    
    # These are our samples for the KS test
    samples = (q_actual - q_min) / (q_max - q_min + 1e-12)
    samples = np.clip(samples, 0, 1) # Ensure no floating point overshoot
    
    # 2. Run KS Test against a uniform distribution
    # D is the maximum distance between the distributions
    d_stat, p_val = stats.kstest(samples, 'uniform')
    
    # 3. Interpret results
    print(f"\n[6] KS Density Significance (Latitude)")
    print(f"    Max Deviation (D): {d_stat:.4f}")
    print(f"    P-Value:           {p_val:.4f}")
    
    if p_val < 0.05:
        print("    RESULT: SIGNIFICANT BIAS DETECTED (p < 0.05)")
    else:
        print("    RESULT: PHYSICALLY UNBIASED (Uniform on Ellipsoid)")
        
    return d_stat, p_val

def perform_ks_depth_test_ellipsoid(x_grid, depth_range):
    # depth_range: [top, bottom] e.g., [0, -40]
    
    # 1. Get local surface radius for every point (using WGS84)
    # If your ftrns1_abs handles the ellipsoid, we can derive the local R
    # Let's assume you have a helper to get R_local based on Latitude
    r_local_surface = np.linalg.norm(ftrns1_abs(x_grid[:,0:3]*np.array([1.0, 1.0, 0.0]).reshape(1,-1)), axis = 1) # Lat-dependent
    
    # 2. Calculate actual distance from Earth Center
    r_actual = r_local_surface + x_grid[:, 2] # Depth is negative
    
    # 3. Calculate the local Shell Boundaries
    r_top = r_local_surface + depth_range[0]
    r_bot = r_local_surface + depth_range[1]
    
    # 4. Transform to Volume Space (r^3)
    # For a perfect ellipsoid, V is proportional to a*b*c, 
    # but the shell-ratio within a small depth range 
    # is still dominated by the r^3 scaling of the local radius.
    vol_actual = r_actual**3
    vol_min = r_bot**3
    vol_max = r_top**3
    
    samples = (vol_actual - vol_min) / (vol_max - vol_min + 1e-12)
    samples = np.clip(samples, 0, 1)
    
    d_stat, p_val = stats.kstest(samples, 'uniform')
    print(f"\n[7] KS Density Significance (Depth/Volume)")
    print(f"    P-Value: {p_val:.4f}\n")
    return d_stat, p_val


def convert_graph(G):

    adj = nx.to_numpy_array(G, nodelist = sorted(G.nodes()))
    i1, i2 = np.where(adj > 1e-5)
    edges = np.concatenate((i1.reshape(1,-1), i2.reshape(1,-1)), axis = 0)
    weights = adj[i1,i2]
    degree_values = np.array([degree for node, degree in G.degree(weight = 'weight')])

    return edges, weights, degree_values


def robust_fiedler_solver(L, x_start = None, tol=1e-5, use_checks = False, max_iter=200):

    n = L.shape[0]
    alpha = 1e-4
    L_s = L + alpha * sp.eye(n)
    
    # Keep your robust factorization
    solve = spla.factorized(L_s.tocsc())
    
    # 3. Power Iteration with Warm Start
    if x_start is not None and len(x_start) == n:
        x = x_start.copy()
    else:
        x = np.random.normal(size=n)
        
    ones = np.ones(n) / np.sqrt(n)
    x -= np.dot(x, ones) * ones
    x /= np.linalg.norm(x)
    
    converged = False
    for i in range(max_iter):
        x_prev = x.copy()
        x = solve(x) 
        x -= np.dot(x, ones) * ones
        x /= np.linalg.norm(x)
        if np.linalg.norm(x - x_prev) < tol:
            converged = True
            break
            
    # --- Keep all your robust checks here ---
    Lx = L.dot(x)
    fiedler_val = np.dot(x, Lx)
    # ... [Insert your existing Check A, B, and C here] ...

    if use_checks == True:

        # Check A: Orthogonality
        ortho_err = np.abs(np.dot(x, ones))
        
        # Check B: Eigenvalue sanity
        # If fiedler_val is negative or effectively zero, the graph is likely disconnected
        if fiedler_val < 1e-10:
            fiedler_val = 0.0 # Force clean zero for disconnected components
            
        # Check C: Residual (is it actually an eigenvector?)
        # Residual = ||Lx - lambda*x||
        residual = np.linalg.norm(Lx - fiedler_val * x)
        
        if not converged or ortho_err > 1e-3 or residual > 1e-1:
            # If residual is high, the graph might be too fragmented for a single vector
            return x, fiedler_val, False 
    
    return x, fiedler_val, True




def soft_component_merger(G, coords, sigma):
    """
    Connects disjoint components using boundary-to-boundary distances 
    to prevent 'far apart' connections caused by centroid logic.
    """
    import networkx as nx
    from scipy.spatial import cKDTree
    # sigma = G.graph['scale_length']

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) <= 1:
        return G

    # We use a set of nodes for the 'main' body to grow into
    # Start with the Largest Connected Component
    main_nodes = set(components[0])
    remaining_components = components[1:]
    
    added_bridges = 0

    while remaining_components:
        # Create a tree of everything already in the main body
        main_coords = coords[list(main_nodes)]
        main_tree = cKDTree(main_coords)
        main_node_indices = list(main_nodes)

        best_global_bridge = None
        min_global_dist = float('inf')
        best_comp_idx = -1

        # Find the absolute closest component to the 'main' body
        for idx, comp in enumerate(remaining_components):
            comp_nodes = list(comp)
            # Find the closest point in 'main' for every point in 'comp'
            dists, neighbors = main_tree.query(coords[comp_nodes], k=1)
            
            # Find the minimum of those distances
            local_min_idx = np.argmin(dists)
            local_min_dist = dists[local_min_idx]

            if local_min_dist < min_global_dist:
                min_global_dist = local_min_dist
                # Map back to original indices
                u = comp_nodes[local_min_idx]
                v = main_node_indices[neighbors[local_min_idx]]
                best_global_bridge = (u, v)
                best_comp_idx = idx

        # Add the best bridge found
        if best_global_bridge:
            u, v = best_global_bridge
            w = np.exp(-min_global_dist / sigma)
            G.add_edge(u, v, dist=float(min_global_dist), weight=float(w), is_bridge=True)
            
            # Merge this component into the main body and repeat
            main_nodes.update(remaining_components.pop(best_comp_idx))
            added_bridges += 1

    print(f"Soft Merger added {added_bridges} bridges using boundary-to-boundary logic.")
    return G



def compute_manifold_diagnostics(G, coords, sample_size=1000):
    # 1. Curvature Variance (Flatness)
    curvatures = [G[u][v]['ricci'] for u, v in G.edges() if 'ricci' in G[u][v]]
    global_flatness = np.var(curvatures)
    
    # 2. Community Resolution
    # High modularity suggests the manifold is 'breaking' into clusters
    communities = nx.community.louvain_communities(G, weight='weight')
    modularity = nx.community.modularity(G, communities, weight='weight')
    
    # 3. Geodesic Consistency (Distance Ratio)
    # We sample pairs to avoid O(N^2) shortest path costs
    ratios = []
    nodes = list(G.nodes())
    for _ in range(sample_size):
        u, v = np.random.choice(nodes, 2, replace=False)
        try:
            graph_dist = nx.shortest_path_length(G, source=u, target=v, weight='dist')
            euclid_dist = np.linalg.norm(coords[u] - coords[v])
            if euclid_dist > 0:
                ratios.append(graph_dist / euclid_dist)
        except nx.NetworkXNoPath:
            continue
            
    avg_stretch = np.mean(ratios) if ratios else 0
    consistency = np.std(ratios) if ratios else 0 # Lower is more uniform

    stats = {
        "flatness_variance": global_flatness,
        "modularity": modularity,
        "stretch_factor": avg_stretch,
        "stretch_consistency": consistency,
        "num_communities": len(communities)
    }

    return stats

    # return {
    #     "flatness_variance": global_flatness,
    #     "modularity": modularity,
    #     "stretch_factor": avg_stretch,
    #     "stretch_consistency": consistency,
    #     "num_communities": len(communities)
    # }


def compute_advanced_diagnostics(G, coords, n_eigen=5):
    L = nx.laplacian_matrix(G, weight='weight').astype(float)
    # Get the first few eigenvalues
    vals = sp.linalg.eigsh(L, k=n_eigen, which='SM', return_eigenvectors=False)
    
    # 1. Spectral Gap (Expansion Health)
    spectral_gap = vals[2] - vals[1] if len(vals) > 2 else 0
    
    # 2. Tortuosity (Geodesic Consistency)
    # Ratio of graph distance to Euclidean distance
    ratios = []
    # ... (sampling logic from previous message) ...
    
    # 3. Global Ricci Flatness
    curvatures = [d['ricci'] for u, v, d in G.edges(data=True) if 'ricci' in d]
    flatness_score = np.std(curvatures) # Lower is flatter/more consistent
    print('\n')

    return {
        "fiedler": vals[1],
        "spectral_gap": spectral_gap,
        "tortuosity": np.mean(ratios),
        "curvature_std": flatness_score
    }


def compute_azimuthal_gaps(G, coords):
    """
    Computes the maximum angular gap between neighbors for every node.
    coords: array of shape (N, 2)
    """
    n = len(coords)
    gaps = np.zeros(n)
    
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if len(neighbors) < 2:
            gaps[i] = 360.0  # Maximum possible gap for 0 or 1 neighbor
            continue
            
        # 1. Get vectors from node i to all neighbors
        # coords[neighbors] is (k, 2), coords[i] is (2,)
        diffs = np.vstack([(coords[int(n)] - coords[i]).reshape(1,-1) for n in neighbors])
        
        # 2. Compute angles in radians [-pi, pi]
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        # 3. Sort angles to find gaps
        sorted_angles = np.sort(angles)
        
        # 4. Compute differences between consecutive angles
        # The 'wrap-around' gap is (2*pi - (last - first))
        diff_list = np.diff(sorted_angles)
        wrap_gap = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
        
        max_gap_rad = max(np.max(diff_list), wrap_gap)
        gaps[i] = np.degrees(max_gap_rad)
        
    return gaps


def compute_expansion_statistics(G, num_nodes):
    # 1. Create Data/NetworkX object
    # data = Data(edge_index=edge_index, num_nodes=num_nodes)
    # G = to_networkx(data, to_undirected=True)
    
    # 2. Get Laplacian (Sparse)
    L = nx.laplacian_matrix(G, weight = 'weight').astype(float)
    
    # 3. Compute Normalized Spectral Gap
    # L_norm = D^-1/2 * L * D^-1/2
    degrees = np.array(L.diagonal())
    d_inv_sqrt = np.power(degrees, -0.5)
    D_inv_sqrt_mat = sp.diags(d_inv_sqrt)
    L_norm = D_inv_sqrt_mat @ L @ D_inv_sqrt_mat
    
    # We want the second smallest eigenvalue (Fiedler value of normalized Laplacian)
    evals = eigsh(L_norm, k=2, which='SM', return_eigenvectors=False)
    spectral_gap_norm = sorted(evals)[1]
    
    # 4. Physical Expansion Estimate (Cheeger Lower Bound)
    # Using lambda_2 from the standard Laplacian
    evals_std = eigsh(L, k=2, which='SM', return_eigenvectors=False)
    lambda_2 = sorted(evals_std)[1]
    h_lower_bound = lambda_2 / 2

    print(f"Normalized Spectral Gap: {spectral_gap_norm:.6f}")
    print(f"Cheeger Expansion Lower Bound: {h_lower_bound:.6f}")

    return spectral_gap_norm, h_lower_bound, lambda_2




def compute_local_metrics(G, pos, k_local = 20, use_weights = False, num_samples_dist = 10000, num_samples_local=100, name = None):
    """
    Compute local graph metrics to highlight strengths of local/spatial graphs over expanders.
    
    Args:
        edge_index: torch.LongTensor [2, num_edges] (PyTorch Geometric style)
        pos: np.array or torch.Tensor [num_nodes, dim] (node positions for spatial distances)
        k_local: Target size for small sets in local conductance (default 20, adjust based on KNN ~18)
        use_weights: If True, add edge weights as inverse Euclidean distance (stronger for closer edges)
        num_samples_dist: Number of random pairs for distance correlation (default 10k; reduce for large n>10k)
        num_samples_local: Number of samples for average local conductance (default 100; fast)
        
    Returns:
        dict of metrics:
            - average_clustering: Higher = stronger local density/triangles
            - modularity: Higher = better community structure
            - degree_assortativity: Higher positive = similar degrees connect (common in spatial graphs)
            - distance_correlation: Higher = graph distances better match spatial (geometric fidelity)
            - average_local_conductance: Lower in local graphs = tighter small clusters (strong locality);
              higher in expanders = quick expansion even locally
    Notes:
        - Robust for n=100s-1000s: Sampling keeps time O(1) relative to n, but distance calc may take seconds for n=1000s.
        - For expander: Expect low clustering/modularity/correlation, high local conductance.
        - For local variants: High clustering/modularity/correlation, low local conductance.
        - Weights (optional): Use for weighted clustering/modularity if distances add value; makes close edges stronger.
        - Assumes graph is undirected; adds edges without direction.
        - Handles possible disconnection (inf distances filtered).
    """
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    
    num_nodes = pos.shape[0]
    
    
    # 1. Average clustering coefficient

    k_local = int(np.mean([v for n,v in G.degree()]))
    clustering = nx.average_clustering(G, weight='weight' if use_weights else None)
    
    # 2. Modularity using Louvain
    partition = nx.community.louvain_communities(G, weight='weight' if use_weights else None)
    mod = nx.community.modularity(G, partition, weight='weight' if use_weights else None)
    
    # 3. Degree assortativity
    deg_assort = nx.degree_assortativity_coefficient(G, weight='weight' if use_weights else None)
    
    # 4. Graph vs Spatial distance correlation (Spearman rank)
    graph_dists = []
    spatial_dists = []
    for _ in range(num_samples_dist):
        u, v = random.sample(range(num_nodes), 2)
        spatial_dist = np.linalg.norm(pos[u] - pos[v])
        try:
            graph_dist = nx.shortest_path_length(G, u, v, weight=None)  # Unweighted paths for "hop" distance
        except nx.NetworkXNoPath:
            graph_dist = float('inf')
        graph_dists.append(graph_dist)
        spatial_dists.append(spatial_dist)
    finite_mask = np.isfinite(graph_dists)
    if np.sum(finite_mask) > 1:
        corr, _ = spearmanr(np.array(graph_dists)[finite_mask], np.array(spatial_dists)[finite_mask])
    else:
        corr = 0.0
    
    # 5. Average local conductance for small sets (vertex expansion style)
    local_cond = []
    for _ in range(num_samples_local):
        seed = random.randint(0, num_nodes - 1)
        S = {seed}
        for __ in range(k_local - 1):
            boundary = set()
            for s in S:
                boundary.update(G.neighbors(s))
            boundary -= S
            if not boundary:
                break
            S.add(random.choice(list(boundary)))
        if len(S) < 2:
            continue
        if use_weights:
            cut_size = sum(G[s][t].get('weight', 1.0) for s in S for t in G.neighbors(s) if t not in S)
        else:
            cut_size = sum(1 for s in S for t in G.neighbors(s) if t not in S)
        cond = cut_size / len(S)
        local_cond.append(cond)
    avg_local_cond = np.mean(local_cond) if local_cond else 0.0
    
    stats = {
        'average_clustering': clustering,
        'modularity': mod,
        'degree_assortativity': deg_assort,
        'distance_correlation': corr,
        'average_local_conductance': avg_local_cond
    }       

    if name is not None: print('\n' + name)
    print(f"Average Clustering Coef.: {clustering:.6f}")
    print(f"Modularity: {mod:.6f}")
    print(f"Degree Assortativity: {deg_assort:.6f}")
    print(f"Distance Correlation: {corr:.6f}")
    print(f"Average Conductance: {avg_local_cond:.6f}")     

    return stats


def extract_inputs_subgraph(locs, x_grid, A_src_in_sta, A_src_src, A_sta_sta, Ac = False, verbose = False, device = 'cpu'):

    ## Connect all source-reciever pairs to their k_nearest_pairs, and those connections within max_deg_offset.
    ## By using the K-nn neighbors as well as epsilon-pairs, this ensures all source nodes are at least
    ## linked to some stations.
    ## Note: can also make the src-src and sta-sta graphs as a combination of k-nn and epsilon-distance graphs

    if verbose == True:
        st = time.time()

    ind_use = np.arange(locs.shape[0])
    locs_use = locs[ind_use]
    n_sta = locs_use.shape[0]
    n_spc = x_grid.shape[0]
    n_sta_slice = len(ind_use)

    use_edge_attr = True if ((A_src_src.shape[0] == 3) or (A_sta_sta.shape[0] == 3)) else False
    if use_edge_attr == True:
        assert((A_src_src.shape[0] == 3)*(A_sta_sta.shape[0] == 3))
        A_src_src, A_src_src_weights = torch.Tensor(A_src_src[0:2,:]).to(device).long(), torch.Tensor(A_src_src[2]).to(device)
        A_sta_sta, A_sta_sta_weights = torch.Tensor(A_sta_sta[0:2,:]).to(device).long(), torch.Tensor(A_sta_sta[2]).to(device)
    else:
        A_src_src, A_src_src_weights = torch.Tensor(A_src_src[0:2,:]).to(device).long(), torch.ones(A_src_src.shape[1]).to(device)
        A_sta_sta, A_sta_sta_weights = torch.Tensor(A_sta_sta[0:2,:]).to(device).long(), torch.ones(A_sta_sta.shape[1]).to(device)


    degree_of_src_nodes = degree(A_src_in_sta[1])
    cum_count_degree_of_src_nodes = np.concatenate((np.array([0]), np.cumsum(degree_of_src_nodes.cpu().detach().numpy())), axis = 0).astype('int')

    sta_ind_lists = []
    for i in range(x_grid.shape[0]):
        ind_list = -1*np.ones(locs.shape[0])
        ind_list[A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]].cpu().detach().numpy()] = np.arange(degree_of_src_nodes[i].item())
        sta_ind_lists.append(ind_list)
    sta_ind_lists = np.hstack(sta_ind_lists).astype('int')


    tree_srcs_in_prod = cKDTree(A_src_in_sta[1].cpu().detach().numpy()[:,None])
    lp_src_in_prod = tree_srcs_in_prod.query_ball_point(np.arange(x_grid.shape[0])[:,None], r = 0)
    A_src_in_prod = torch.Tensor(np.hstack([np.concatenate((np.array(lp_src_in_prod[j]).reshape(1,-1), j*np.ones(len(lp_src_in_prod[j])).reshape(1,-1)), axis = 0) for j in range(x_grid.shape[0])])).long().to(device)
    

    A_prod_sta_sta = []
    A_prod_src_src = []
    A_prod_sta_sta_weights = []
    A_prod_src_src_weights = []


    tree_src_in_sta = cKDTree(A_src_in_sta[0].reshape(-1,1).cpu().detach().numpy())
    lp_fixed_stas = tree_src_in_sta.query_ball_point(np.arange(locs.shape[0]).reshape(-1,1), r = 0)


    for i in range(x_grid.shape[0]):
        # slice_edges = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, relabel_nodes = False)[0]
        slice_edges, slice_weights = subgraph(A_src_in_sta[0,cum_count_degree_of_src_nodes[i]:cum_count_degree_of_src_nodes[i+1]], A_sta_sta, edge_attr = A_sta_sta_weights, relabel_nodes = True)
        A_prod_sta_sta.append(slice_edges + cum_count_degree_of_src_nodes[i])
        A_prod_sta_sta_weights.append(slice_weights)


    for i in range(locs.shape[0]):
    
        slice_edges, slice_weights = subgraph(A_src_in_sta[1,np.array(lp_fixed_stas[i])], A_src_src, edge_attr = A_src_src_weights, relabel_nodes = False) # [0].cpu().detach().numpy()
        slice_edges = slice_edges.cpu().detach().numpy()

        ## This can happen when a station is only linked to one source
        if slice_edges.shape[1] == 0:
            continue

        shift_ind = sta_ind_lists[slice_edges*n_sta + i]
        assert(shift_ind.min() >= 0)
        ## For each source, need to find where that station index is in the "order" of the subgraph Cartesian product
        A_prod_src_src.append(torch.Tensor(cum_count_degree_of_src_nodes[slice_edges] + shift_ind).to(device))
        A_prod_src_src_weights.append(slice_weights)


    ## Make cartesian product graphs
    A_prod_sta_sta = torch.hstack(A_prod_sta_sta).long()
    A_prod_src_src = torch.hstack(A_prod_src_src).long()
    A_prod_sta_sta_weights = torch.hstack(A_prod_sta_sta_weights)
    A_prod_src_src_weights = torch.hstack(A_prod_src_src_weights)
    isort = np.lexsort((A_prod_src_src[0].cpu().detach().numpy(), A_prod_src_src[1].cpu().detach().numpy())) # Likely not actually necessary
    A_prod_src_src = A_prod_src_src[:,isort]
    A_prod_src_src_weights = A_prod_src_src_weights[isort]

    # if Ac is not False:
    #     use_perm_expand = True
    #     if use_perm_expand == True:
    #         perm_vec_expand = np.random.permutation(np.arange(x_grid.shape[0])).astype('int')
    #         Ac_src_src = torch.Tensor(perm_vec_expand[Ac]).long().to(device)
    #     else:
    #         perm_vec_expand = np.arange(x_grid.shape[0]).astype('int')
    #         Ac_src_src = torch.Tensor(perm_vec_expand[Ac]).long().to(device)
    #     Ac_prod_src_src = build_src_src_product(Ac_src_src, A_src_in_sta, locs[ind_use], x_grid, device = device)

    # if Ac is False:

    return [A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_prod_sta_sta_weights, A_prod_src_src_weights, A_src_in_prod, A_src_in_sta] ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)


def compute_local_sigma(coords, k=7):
    tree = cKDTree(coords)
    # dists[:, 0] is the node itself, dists[:, k] is the k-th neighbor
    dists, _ = tree.query(coords, k=k+1)
    # The local sigma is often the distance to the k-th neighbor
    local_sigmas = dists[:, k]
    return local_sigmas


def initialize_sensor_graph(coords, cnt = 0, min_weight = 0.05, G = None, k_trgt = 10, use_local_scale = True, set_initial_edges = None, edges_to_update = None, init_knn = None, use_rng = True):

    n_nodes, n_dim = coords.shape

    # if use_local_scale == True:
    # local_sigmas = compute_local_sigma(coords, k=7)

    if G is None:

        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i, p in enumerate(coords):
            G.nodes[i]['pos'] = p


        tree = cKDTree(coords) # Need this for both branches
        # scale_base = np.median(tree.query(coords, k = k_trgt + 1)[0][:,-1])

        scale_base = np.quantile(tree.query(coords, k = k_trgt + 1)[0][:,-1], 0.75)

        if use_local_scale == True:
            length_scales = compute_local_sigma(coords, k = k_trgt)
            for i in range(n_nodes):
                G.nodes[i]['scale'] = length_scales[i]


        initial_edges = []
        # --- HIGHER DIMENSION STRATEGY (k-NN + RNG Check) ---
        if n_dim >= 3:
            k_check = int(n_dim*2)
            distances, indices = tree.query(coords, k= k_check + 1)
            for i in range(n_nodes):
                p_i = coords[i]
                for neighbor_idx in range(1, k_check + 1):
                    j = indices[i, neighbor_idx]
                    p_j = coords[j]
                    d_ij = distances[i, neighbor_idx]

                    if use_rng:
                        midpoint = (p_i + p_j) / 2.0
                        radius = d_ij - 1e-9 
                        potential_violators = tree.query_ball_point(midpoint, radius)
                        
                        is_rng = True
                        for v_idx in potential_violators:
                            if v_idx == i or v_idx == j: continue
                            # Distance to both must be less than d_ij for it to be a 'lune' violation
                            if np.linalg.norm(p_i - coords[v_idx]) < d_ij and \
                               np.linalg.norm(p_j - coords[v_idx]) < d_ij:
                                is_rng = False
                                break
                        if not is_rng: continue

                    # Mutual check
                    if i in indices[j, 1:k_check + 1]:
                        u, v = sorted((i, j))
                        initial_edges.append((u, v, d_ij))
            
            initial_edges = list(set(initial_edges))
            print(f"Initialized with {'RNG' if use_rng else 'Mutual k-NN'} (Dim {n_dim})")

        # --- 2D STRATEGY (Gabriel Backbone) ---
        else:
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        edges.add(tuple(sorted((simplex[i], simplex[j]))))

            for i, j in edges:
                p1, p2 = coords[i], coords[j]
                midpoint = (p1 + p2) / 2.0
                radius = (np.linalg.norm(p1 - p2) / 2.0) - 1e-9
                # Faster than calculating all distances:
                if not tree.query_ball_point(midpoint, radius):
                    initial_edges.append((i, j, np.linalg.norm(p1 - p2)))
            print(f"Initialized with Gabriel Backbone (Dim {n_dim})")

        # --- SCALE & COMPONENT MANAGEMENT ---
        G_init = nx.Graph()
        G_init.add_weighted_edges_from(initial_edges, weight='dist')
        
        # Calculate scale from MST of the largest component
        comps = list(nx.connected_components(G_init))
        if comps:
            lcc = G_init.subgraph(max(comps, key=len))
            mst_edges = [d['dist'] for u, v, d in nx.minimum_spanning_tree(lcc, weight='dist').edges(data=True)]
            scale_length = np.percentile(mst_edges, 99)
            scale_length = np.maximum(scale_base, scale_length) ## Max of the distances
            G.graph['scale_length'] = scale_length
        else:
            scale_length = 1.0 # Fallback


        if use_local_scale == True:
            ## Bound the local scales to fraction of the charecteristic scale
            for i in range(n_nodes):
                # G.nodes[i]['scale'] = np.clip(G.nodes[i]['scale'], 0.2*scale_length, 2.0*scale_length)
                # G.nodes[i]['scale'] = np.clip(G.nodes[i]['scale'], 0.2*scale_length, 2.0*scale_length)
                pass ## Not re-scaling per lengths
        else:
            for i in range(n_nodes):
               G.nodes[i]['scale'] = scale_length


        if set_initial_edges is not None: ## Set initial edges if given (e.g., K-NN graph)
            # initial_egdes = set_initial_edges ## Fix initial edges 
            initial_edges = [(set_initial_edges[0,i], set_initial_edges[1,i], np.linalg.norm(coords[set_initial_edges[0,i]] - coords[set_initial_edges[1,i]])) for i in range(set_initial_edges.shape[1])]

        # Apply weights and the 'Soft Merger'
        for u, v, d in initial_edges:
            # w = np.exp(-d / scale_length)
            w = np.exp(-d / (scale_length*G.nodes[u]['scale']*G.nodes[v]['scale'])**(1/3))
            if w >= min_weight:
                G.add_edge(int(u), int(v), dist = d, weight = float(w), step = 0, immutable = True)

        if init_knn is not None:
            tree_pos = cKDTree(coords) # Need this for both branches
            knn_edges = tree_pos.query(coords, k = init_knn + 1)[1][:,1::]
            ip_edges = np.hstack([np.concatenate((knn_edges[j,:].reshape(1,-1), j*np.ones(init_knn).reshape(1,-1)), axis = 0) for j in range(len(coords))]).astype('int').T
            for u, v in ip_edges:
                d = np.linalg.norm(coords[u] - coords[v])
                w = np.exp(-d / (scale_length*G.nodes[u]['scale']*G.nodes[v]['scale'])**(1/3))
                if w >= min_weight:
                    G.add_edge(int(u), int(v), dist = d, weight = float(w), step = 0, immutable = True)


        # Final Safety Step: ensure we don't start with disconnected islands
        G = soft_component_merger(G, coords, scale_length)
        G.graph['distances'] = np.linalg.norm((np.expand_dims(coords, axis = 1) - np.expand_dims(coords, axis = 0)), axis = 2)
        scales = np.array([G.nodes[i]['scale'] for i in range(n_nodes)])
        G.graph['scale_values'] = (scale_length*scales.reshape(-1,1)*scales.reshape(1,-1))**(1/3) ## The pairwise scale lengths
        G.graph['weights'] = np.exp(-G.graph['distances'] / G.graph['scale_values']) # np.linalg.norm((np.expand_dims(coords, axis = 1) - np.expand_dims(coords, axis = 0)), axis = 1)
 
        edges_allowed = G.graph['weights'] >= min_weight
        ilist1, ilist2 = np.where(edges_allowed > 0)
        edges_allowed = np.concatenate((ilist1.reshape(-1,1), ilist2.reshape(-1,1)), axis = 1)
        G.graph['allowed_edges'] = edges_allowed[edges_allowed[:,0] < edges_allowed[:,1]]

        print(f"Final scale length: {scale_length:0.3f} | Components: {nx.number_connected_components(G)}")


    # components = list(nx.connected_components(G))
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    scale_length = G.graph['scale_length']
    for comp_id, nodes in enumerate(components):
        nodes_sorted = sorted(list(nodes))
        subG = G.subgraph(nodes_sorted)
        if (comp_id == 0)*(np.mod(cnt, 50) == 0): ## Only re-compute diameter every n steps
            G.graph['diameter'] = nx.algorithms.approximation.diameter(subG)

        for n in nodes_sorted:
            G.nodes[n]['comp_id'] = comp_id
            
        if len(nodes_sorted) > 2:
            # Compute local Fiedler

            L = nx.laplacian_matrix(subG, weight='weight').astype(float)


            # pdb.set_trace()
            # vals, vecs = lobpcg(L, X, largest=False, tol=1e-2)
            fiedler_vector, fiedler_value, flag = robust_fiedler_solver(L)
            if comp_id == 0: G.graph['fiedler_value'] = fiedler_value

            if flag == False:
                print('Fiedler vector did not converge')
                fiedler_vector[:] = 0.0
                fiedler_value = 0.0


            normalize_fiedler = True
            if normalize_fiedler == True:
                fiedler_vector = 2 * (fiedler_vector - np.min(fiedler_vector)) / (np.max(fiedler_vector) - np.min(fiedler_vector) + 1e-9) - 1
            
            for i, node_idx in enumerate(nodes_sorted):
                G.nodes[node_idx]['fiedler'] = fiedler_vector[i]

            if comp_id == 0: 
                fiedler_graph = fiedler_value

            # except:
            #     for node_idx in nodes_sorted: G.nodes[node_idx]['fiedler'] = 0.0
        else:
            for node_idx in nodes_sorted: G.nodes[node_idx]['fiedler'] = 0.0


    use_normalized_node_importance = True
    node_weights = {u: G.degree(u, weight='weight') for u in G.nodes()}
    if edges_to_update is None: edges_to_update = G.edges()
    if use_normalized_node_importance == True: ## This suppresses the per-node total curvature effect on the local ricci curvature
        # 5. Weighted (Normalized) Forman-Ricci (Global pass)
        edges = []
        for u, v in edges_to_update:
        # for u, v in G.edges():
            w_e = G[u][v]['weight']
            # w_u, w_v = node_weights[u], node_weights[v]
            sum_u = sum(np.sqrt(w_e / G[u][n]['weight']) for n in G.neighbors(u) if n != v)
            sum_v = sum(np.sqrt(w_e / G[v][n]['weight']) for n in G.neighbors(v) if n != u)
            G[u][v]['ricci'] = 2.0 - sum_u - sum_v
            edges.append(np.array([u, v, G[u][v]['ricci'], G[u][v]['dist'], G[u][v]['weight']]).reshape(-1,1))
        edges = np.hstack(edges)

    else:
        # 5. Weighted Forman-Ricci (Global pass)
        edges = []
        for u, v in edges_to_update:
        # for u, v in G.edges():
            w_e = G[u][v]['weight']
            w_u, w_v = node_weights[u], node_weights[v]
            sum_u = sum(np.sqrt(w_u / (w_e * G[u][n]['weight'])) for n in G.neighbors(u) if n != v)
            sum_v = sum(np.sqrt(w_v / (w_e * G[v][n]['weight'])) for n in G.neighbors(v) if n != u)
            G[u][v]['ricci'] = w_e * ((w_u / w_e) + (w_v / w_e) - sum_u - sum_v)
            edges.append(np.array([u, v, G[u][v]['ricci'], G[u][v]['dist'], G[u][v]['weight']]).reshape(-1,1))
        edges = np.hstack(edges)


    use_mean_total_curvature = True ## Normalized curvature per node
    if use_mean_total_curvature == True:
        for u in G.nodes():
            vals = [G[u][v]['ricci'] for v in G.neighbors(u) if 'ricci' in G[u][v]]
            G.nodes[u]['total_curvature'] = sum(vals)/np.maximum(1.0, len(vals))
    else:
         for u in G.nodes():
            G.nodes[u]['total_curvature'] = sum(G[u][v]['ricci'] for v in G.neighbors(u) if 'ricci' in G[u][v])       

    fiedler_vector = np.array([G.nodes[i]['fiedler'] for i in range(n_nodes)])
    components = np.array([G.nodes[i]['comp_id'] for i in range(n_nodes)])


    curvature = []
    clustering = []
    degree_values = []

    if np.mod(cnt, 100) == 0:

        fiedler_vector = np.array([G.nodes[i]['fiedler'] for i in range(n_nodes)])
        components = np.array([G.nodes[i]['comp_id'] for i in range(n_nodes)])
        curvature = np.array([G.nodes[i]['total_curvature'] for i in range(n_nodes)])
        clustering = np.array(list(nx.clustering(G, weight='weight').values()))
        degree_values = np.array([degree for node, degree in G.degree(weight = 'weight')])

        print('\nFiedler value: %0.3f'%fiedler_graph) ## Also add correlation between distance and graph distance
        print('Diameter: %0.3f'%G.graph['diameter'])
        print('\nCurvature distribution: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(curvature, [0, 0.25, 0.5, 0.75, 1.0])))
        print('Edge Curvature distribution: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(edges[2], [0, 0.25, 0.5, 0.75, 1.0])))
        print('Edge Weight distribution: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(edges[4], [0, 0.25, 0.5, 0.75, 1.0])))
        print('Clustering distribution: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(clustering, [0, 0.25, 0.5, 0.75, 1.0])))
        print('Degree distribution: [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(degree_values, [0, 0.25, 0.5, 0.75, 1.0])))


    edges = np.vstack(list(G.edges())).T

    return G, edges, fiedler_vector, curvature, clustering, degree_values, components, fiedler_graph, scale_length



class SpectralProductSampler:

    def __init__(self, G_A, G_B, pos_A, pos_B, sparse_threshold = 2000, k_approx = 150):

        self.G_A = G_A
        self.G_B = G_B
        self.pos_A = pos_A
        self.pos_B = pos_B
        self.k_approx = k_approx
        
        # Determine mode for each factor independently
        self.mode_A = 'sparse' if len(G_A) > sparse_threshold else 'dense'
        self.mode_B = 'sparse' if len(G_B) > sparse_threshold else 'dense'
        
        # 1. Compute Leverage Scores (tau)
        self.tau_A, self.L_A_obj = self._compute_spectral_stats(G_A, self.mode_A)
        self.tau_B, self.L_B_obj = self._compute_spectral_stats(G_B, self.mode_B)
        
        # 2. Probabilities for sampling
        self.p_A = self.tau_A / np.sum(self.tau_A)
        self.p_B = self.tau_B / np.sum(self.tau_B)

        # 3. Index mappings
        self.node_to_idx_A = {node: i for i, node in enumerate(G_A.nodes())}
        self.node_to_idx_B = {node: i for i, node in enumerate(G_B.nodes())}
        self._precompute_edge_weights()
        # self._prepare_scipy_matrices()

        # Convert the graphs to CSR format once
        self.adj_A = nx.to_scipy_sparse_array(self.G_A, weight='res_w', format='csr')
        self.adj_B = nx.to_scipy_sparse_array(self.G_B, weight='res_w', format='csr')


        # Force internal CSR structure to 32-bit
        self.adj_A.indices = self.adj_A.indices.astype(np.int32)
        self.adj_A.indptr = self.adj_A.indptr.astype(np.int32)
        self.adj_B.indices = self.adj_B.indices.astype(np.int32)
        self.adj_B.indptr = self.adj_B.indptr.astype(np.int32)


        # Store node lists for index-to-node mapping
        self.nodes_A = list(self.G_A.nodes())
        self.nodes_B = list(self.G_B.nodes())

        # # Much faster O(1) lookup
        # u_idx = self.node_to_idx_A[u] if mode == 'A' else self.node_to_idx_B[u]

    def _precompute_edge_weights(self):
        """Run this once to store weights on the graph edges."""
        for G, L_obj, mode, mapping in [
            (self.G_A, self.L_A_obj, self.mode_A, self.node_to_idx_A),
            (self.G_B, self.L_B_obj, self.mode_B, self.node_to_idx_B)
        ]:
            for u, v in G.edges():
                if mode == 'dense':
                    u_idx, v_idx = mapping[u], mapping[v]
                    # R_uv = L+uu + L+vv - 2L+uv
                    w = max(1e-6, L_obj[u_idx, u_idx] + L_obj[v_idx, v_idx] - 2 * L_obj[u_idx, v_idx])
                else:
                    # Proxy for sparse mode
                    w = 1.0 / np.sqrt(G.degree(u) * G.degree(v))
                
                G[u][v]['res_w'] = w


    def _compute_spectral_stats(self, G, mode):
        L = nx.laplacian_matrix(G).astype(float)
        n = L.shape[0]
        
        if mode == 'dense':
            # Returns the full pseudoinverse for exact resistance
            L_pinv = np.linalg.pinv(L.toarray())
            return np.diag(L_pinv), L_pinv
        else:
            # Random Projection Approximation (Spielman-Srivastava)
            # We solve Lz = r for k random vectors r
            R = np.random.randn(n, self.k_approx) / np.sqrt(self.k_approx)
            Z = np.zeros((n, self.k_approx))
            for i in range(self.k_approx):
                # Using lsqr for the singular sparse Laplacian
                sol = lsqr(L, R[:, i])[0]
                Z[:, i] = sol
            tau = np.sum(Z**2, axis=1)
            return tau, L # Return sparse L instead of pinv

    def _reconstruct_path(self, predecessors, start_idx, end_idx, node_list):
        path = []
        curr = end_idx
        while curr != -9999: # SciPy's code for "no path/start node"
            path.append(node_list[curr])
            if curr == start_idx: break
            curr = predecessors[curr]
        return path[::-1]


    def get_resistance_path(self, G, source, target):
        # No more Python function calls per edge!
        return nx.shortest_path(G, source, target, weight='res_w')


    def _get_path_from_predecessor(self, predecessors, start_idx, end_idx, mapping_list):
        path = []
        curr = end_idx
        while curr != -9999: # SciPy uses -9999 for "no predecessor"
            path.append(mapping_list[curr])
            if curr == start_idx: break
            curr = predecessors[curr]
        return path[::-1] # Reverse to get start -> end


    def select_anchors(self, n_total_target=100000, core_ratio=0.2, n_physical_src = 15, n_physical_sta = 10):
        """
        n_total_target: The rough budget for seeds
        core_ratio: Fraction of the budget allocated to the 'All-Pairs' Hub grid
        """
        nodes_A = list(self.G_A.nodes())
        nodes_B = list(self.G_B.nodes())
        anchors = set()

        n_physical_src = min(n_physical_src, len(nodes_B) - 1)

        # 1. THE SPECTRAL CORE (All-Pairs of Important Nodes)
        # We want a grid of k_a stations x k_b sources ≈ target * core_ratio
        n_core_target = int(n_total_target * core_ratio)
        # Adjust k_a and k_b proportionally to factor sizes to keep the 'grid' balanced
        ratio = len(nodes_A) / len(nodes_B)
        k_a = max(1, int(np.sqrt(n_core_target * ratio)))
        k_b = max(1, int(n_core_target / k_a))

        # Pick top nodes by leverage score
        top_a_indices = np.argsort(self.tau_A)[-k_a:]
        top_b_indices = np.argsort(self.tau_B)[-k_b:]
        
        for idx_a in top_a_indices:
            for idx_b in top_b_indices:
                anchors.add((nodes_A[idx_a], nodes_B[idx_b]))

        # 2. THE SPECTRAL DISTRIBUTION (Randomly Paired Samples)
        # This ensures global coverage of the far-field manifold
        n_remaining = n_total_target - len(anchors)
        if n_remaining > 0:
            idx_a = np.random.choice(len(nodes_A), size=n_remaining, p=self.p_A)
            idx_b = np.random.choice(len(nodes_B), size=n_remaining, p=self.p_B)
            for i in range(n_remaining):
                anchors.add((nodes_A[idx_a[i]], nodes_B[idx_b[i]]))


        coords_A = np.array([self.pos_A[n] for n in nodes_A])
        coords_B = np.array([self.pos_B[n] for n in nodes_B])

        # Build tree for stations
        tree_sta = cKDTree(coords_B)
        tree_src = cKDTree(coords_A)
        
        # physical_anchors = set()
        # Query all sources at once
        # dists: (n_sources, k), indices: (n_sources, k)
        _, indices = tree_sta.query(coords_A, k=n_physical_src)
        
        for a_idx, b_indices in enumerate(indices):
            source = nodes_A[a_idx]
            # tree.query returns a scalar if k=1, handle both
            if n_physical_src == 1:
                b_indices = [b_indices]
            for b_idx in b_indices:
                station = nodes_B[b_idx]
                anchors.add((source, station))


        _, indices = tree_src.query(coords_B, k=n_physical_sta)
        
        for b_idx, a_indices in enumerate(indices):
            station = nodes_B[b_idx]
            # tree.query returns a scalar if k=1, handle both
            if n_physical_sta == 1:
                a_indices = [a_indices]
            for a_idx in a_indices:
                source = nodes_A[a_idx]
                anchors.add((source, station))


        return list(anchors)

    def sort_anchors_greedy(self, anchors):
        """
        Sorts anchors in O(N log N) using a lexicographical spatial sort.
        This ensures the backbone 'snakes' through the map rather than 
        criss-crossing randomly.
        """
        if not anchors: return []
        
        # We sort by Station X, then Station Y, then Source X, then Source Y.
        # This keeps 'backbone steps' local to a specific station area 
        # before moving to the next.

        return sorted(list(anchors), key=lambda x: (
            self.pos_B[x[1]][0], 
            self.pos_B[x[1]][1], 
            self.pos_A[x[0]][0], 
            self.pos_A[x[0]][1]
        ))

    def build_final_subgraph(self, target_node_count=100000, anchor_fraction = 0.15, slack_factor = 2.5, skip_paths = False):
        nodes_to_retain = set()
        
        st_time = time.time()

        # 1. Seeds (10% of budget)
        # anchors = self.select_anchors(n_total_target=int(target_node_count * 0.1))
        anchors = self.select_anchors(n_total_target=int(target_node_count * anchor_fraction))

        nodes_to_retain.update(anchors)

        nodes_A = list(self.G_A.nodes())
        nodes_B = list(self.G_B.nodes())

        MAX_DEG_A = int((target_node_count / len(self.G_A)) * slack_factor)
        MAX_DEG_B = int((target_node_count / len(self.G_B)) * slack_factor)

        # Initialize appearance counters for factor nodes
        count_A = {node: 0 for node in self.G_A.nodes()}
        count_B = {node: 0 for node in self.G_B.nodes()}

        # Pre-fill with anchors
        for a, b in anchors:
            count_A[a] += 1
            count_B[b] += 1


        print('Time [1] : %0.4f'%(time.time() - st_time))

        # --- NEW: Anchor Sorting ---
        # Sort to ensure the backbone doesn't jump sporadically
        sorted_anchors = self.sort_anchors_greedy(anchors)
        
        print('Time [2] : %0.4f'%(time.time() - st_time))

        if skip_paths == False:

            n_init_nodes = len(nodes_to_retain)

            # 2. THE BACKBONE (SciPy Optimized)
            for i in range(len(sorted_anchors) - 1):
                u, v = sorted_anchors[i], sorted_anchors[i+1]
                
                # --- Factor A (Sources) ---
                u_idx_a, v_idx_a = int(np.int32(self.node_to_idx_A[u[0]])), int(np.int32(self.node_to_idx_A[v[0]]))
                if u_idx_a == v_idx_a:
                    path_a = [u[0]]
                else:
                    # returns (distances, predecessors)
                    _, pred_a = shortest_path(self.adj_A, directed=False, indices=u_idx_a, return_predecessors=True)
                    path_a = self._reconstruct_path(pred_a, np.int32(u_idx_a), np.int32(v_idx_a), self.nodes_A)
                
                # --- Factor B (Stations) ---
                u_idx_b, v_idx_b = int(np.int32(self.node_to_idx_B[u[1]])), int(np.int32(self.node_to_idx_B[v[1]]))
                if u_idx_b == v_idx_b:
                    path_b = [u[1]]
                else:
                    _, pred_b = shortest_path(self.adj_B, directed=False, indices=u_idx_b, return_predecessors=True)
                    path_b = self._reconstruct_path(pred_b, np.int32(u_idx_b), np.int32(v_idx_b), self.nodes_B)
                
                for na in path_a:
                    nodes_to_retain.update([(na, u[1]) for na in path_a])
                    count_A[na] += 1
                    count_B[u[1]] += 1

                for nb in path_b:
                    nodes_to_retain.update([(v[0], nb) for nb in path_b])
                    count_A[v[0]] += 1
                    count_B[nb] += 1

            print('Added %d nodes during path finding'%(len(nodes_to_retain) - n_init_nodes))

            print('Time [3] : %0.4f'%(time.time() - st_time))

        # 3. PRIORITY EXPANSION (Degree-Limited Blobs)
        current_seeds = list(nodes_to_retain)
        for a, b in current_seeds:
            if len(nodes_to_retain) >= target_node_count: break
            
            # Expand in G_A neighbors if 'b' isn't too popular
            if count_B[b] < MAX_DEG_B:
                for na in self.G_A.neighbors(a):
                    if (na, b) not in nodes_to_retain:
                        nodes_to_retain.add((na, b))
                        count_A[na] += 1
                        count_B[b] += 1
                        if len(nodes_to_retain) >= target_node_count: break

            # Expand in G_B neighbors if 'a' isn't too popular
            if count_A[a] < MAX_DEG_A:
                for nb in self.G_B.neighbors(b):
                    if (a, nb) not in nodes_to_retain:
                        nodes_to_retain.add((a, nb))
                        count_A[a] += 1
                        count_B[nb] += 1
                        if len(nodes_to_retain) >= target_node_count: break


        print('Time [4] : %0.4f'%(time.time() - st_time))

        # 4. STOCHASTIC FILLING (Far-field clusters)
        nodes_A_list, nodes_B_list = list(self.G_A.nodes()), list(self.G_B.nodes())
        while len(nodes_to_retain) < target_node_count:
            idx_a = np.random.choice(len(nodes_A_list), p=self.p_A)
            idx_b = np.random.choice(len(nodes_B_list), p=self.p_B)
            seed = (nodes_A_list[idx_a], nodes_B_list[idx_b])
            nodes_to_retain.add(seed)
            # Expand every new random seed as a mini-blob
            for na in self.G_A.neighbors(seed[0]):
                if len(nodes_to_retain) >= target_node_count: break
                nodes_to_retain.add((na, seed[1]))
            for nb in self.G_B.neighbors(seed[1]):
                if len(nodes_to_retain) >= target_node_count: break
                nodes_to_retain.add((seed[0], nb))

        print('Time [5] : %0.4f'%(time.time() - st_time))

        # Refined Step 5: Multi-Relational Induction
        G_sub = nx.Graph()
        G_sub.add_nodes_from(nodes_to_retain)

        for (a, b) in nodes_to_retain:
            # 1. Factor A Edges (Moving in Source Space)
            # These represent relationships between sources for a fixed station
            for na in self.G_A[a]:
                if (na, b) in nodes_to_retain:
                    G_sub.add_edge((a, b), (na, b), edge_type='factor_a')
                    
            # 2. Factor B Edges (Moving in Station Space)
            # These represent physical signal propagation between sensors for a fixed source
            for nb in self.G_B[b]:
                if (a, nb) in nodes_to_retain:
                    G_sub.add_edge((a, b), (a, nb), edge_type='factor_b')

        return G_sub


def get_domain_bounds(points_lla, scale=1.05):
    lats = points_lla[:, 0]
    lons = points_lla[:, 1]

    # 1. Handle Longitude Wrap
    if np.max(lons) - np.min(lons) > 180:
        working_lons = np.mod(lons, 360)
        is_wrapped = True
    else:
        working_lons = lons
        is_wrapped = False

    lat_min, lat_max = np.min(lats), np.max(lats)
    lon_min, lon_max = np.min(working_lons), np.max(working_lons)

    d_lat = lat_max - lat_min
    d_lon = lon_max - lon_min

    # --- CRITICAL FIX FOR GLOBAL SPAN ---
    # If the spread is already huge (e.g. > 350 deg), it's a global domain.
    # Padding it will just create messy overlaps.
    if d_lon > 320:
        return {
            "lat_range": (np.clip(lat_min - 2, -90, 90), np.clip(lat_max + 2, -90, 90)),
            "lon_range": (-180.0, 180.0),
            "is_wrapped": False
        }
    # ------------------------------------

    pad_lat = (d_lat * (scale - 1.0)) / 2.0
    pad_lon = (d_lon * (scale - 1.0)) / 2.0

    out_lat_min = np.clip(lat_min - pad_lat, -90, 90)
    out_lat_max = np.clip(lat_max + pad_lat, -90, 90)
    
    out_lon_start = lon_min - pad_lon
    out_lon_end = lon_max + pad_lon

    # Only normalize if we aren't spanning the whole world
    def norm_lon(l):
        return ((l + 180) % 360) - 180

    final_start = norm_lon(out_lon_start)
    final_end = norm_lon(out_lon_end)

    return {
        "lat_range": (out_lat_min, out_lat_max),
        "lon_range": (final_start, final_end),
        "is_wrapped": final_start > final_end or is_wrapped
    }

def build_graphs_domain(m_domain, locs_use, stas_use, scale_domain, deg_padding, number_of_spatial_nodes, k_spc_edges, k_sta_edges, depth_range, ftrns1, ftrns2, use_global = False, assign_based_on_grid = False, max_nodes = 1500, n_trgt_nodes = 100e3, Vc = 3500.0, file_index = 0, date = [2000, 1, 1], use_paths = False, rbest = None, mn = None, optimize_station_graph = False, optimize_source_graph = False, use_domain_approximate = True, device = 'cpu'):


    domain = get_domain_bounds(locs_use, scale = scale_domain)
    lat_range, lon_range = domain['lat_range'], domain['lon_range']
    locs_cart = ftrns1(locs_use)


    if (use_domain_approximate == True) and (m_domain is not None):


        k_inpts_model = tensor_to_strings(m_domain.k_inpts)
        k_trgts_model = tensor_to_strings(m_domain.k_trgts)


        if len(k_inpts_model) == 2:
            k_inpts_model = [lat_range, lon_range]
        else:
            print('Need to set deg_padding')
            lat_range_extend, lon_range_extend = [lat_range[0] - deg_padding, lat_range[1] + deg_padding], [lon_range[0] - deg_padding, lon_range[1] + deg_padding] # extend_geo_range(lat_range, lon_range, domain_scale['W_phys_m'], multiplier = 2.0)
            k_inpts_model = [lat_range, lon_range, lat_range_extend, lon_range_extend, np.array([deg_padding])]

        # pdb.set_trace()

        inpt_domain = np.hstack([np.diff(k) if len(k) > 1 else k for k in k_inpts_model]).reshape(1,-1)
        inpt_domain = torch.tensor(np.concatenate((inpt_domain, np.array([len(locs_use)]).reshape(1,1)), axis = 1), device = device) # .float()
        inpt_domain = (inpt_domain - m_domain.offset_inpt)/m_domain.scale_inpt
        scale_params = (m_domain(inpt_domain.float())*m_domain.scale_trgt + m_domain.offset_trgt).cpu().detach().numpy().reshape(-1)


        # for k in k_inpts_model:
        #     print('%0.4f'%(z[k] if len(np.array(z[k]).reshape(-1)) == 1 else np.diff(z[k])))
        # print('\n')
        for inc, k in enumerate(k_trgts_model):
            print('%s %0.4f'%(k, scale_params[inc]))
            # print(scale_params[inc])
            # print('\n')
        # z.close()

        earth_radius = 6378137.0
        ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
        ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1

        def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
            # r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
            r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
            r_val = r_surface + h_min
            return r_val


        def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
            # r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
            r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
            r_val = r_surface + h_max
            return -r_val


        if len(k_inpts_model) == 2:
            deg_padding = scale_params[7]
            lat_range_extend, lon_range_extend = [lat_range[0] - deg_padding, lat_range[1] + deg_padding], [lon_range[0] - deg_padding, lon_range[1] + deg_padding] # extend_geo_range(lat_range, lon_range, domain_scale['W_phys_m'], multiplier = 2.0)

        bounds = [(lat_range_extend[0], lat_range_extend[1])]
        soln = differential_evolution(optimize_r_min, bounds, popsize = 50, maxiter = 1000, disp = True)
        r_min = optimize_r_min(np.array([soln.x]))[0]; print('\n')

        bounds = [(lat_range_extend[0], lat_range_extend[1])]
        soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True)
        r_max = -1.0*optimize_r_max(np.array([soln.x]))[0]; print('\n')
        assert(r_max >= r_min)


        ## Set parameters

        # scale_time, depth_upscale_factor, time_shift_range, 
        scale_time = scale_params[0]
        depth_boost = scale_params[1]
        time_shift_range = scale_params[2]
        buffer_scale = scale_params[3]
        source_label_width = scale_params[4]
        source_label_width_t = scale_params[5]
        association_label_width = scale_params[4]
        association_label_width_t = scale_params[5]*(1.5/1.2)
        sigma_input = scale_params[6]

        # time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = number_of_spatial_nodes, buffer_scale = buffer_scale
        # final_N, _, final_W_phys, final_W_t = fit_domain_budget_aware(source_label_width/1.2, source_label_width_t/1.2, lat_range_extend, lon_range_extend, depth_range, time_shift_range, 
        #                          N_max = max_nodes, depth_boost=1.0, use_global=use_global)
        final_N, _, _, _ = fit_domain_budget_aware(source_label_width/1.2, source_label_width_t/1.2, lat_range_extend, lon_range_extend, depth_range, time_shift_range, 
                                 N_max = max_nodes, depth_boost=1.0, use_global=use_global)

        if final_N != number_of_spatial_nodes:
            print('Over writing number of nodes: %d to %d'%(number_of_spatial_nodes, final_N))
            number_of_spatial_nodes = final_N

        ## Sample grid:
        use_time_shift = True
        use_station_density = False
        # number_of_spatial_nodes = n_grid
        print('Beginning FPS sampling [%d]'%0)
        up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
        # if use_station_density == True: up_sample_factor = up_sample_factor*5
        number_candidate_nodes = up_sample_factor*number_of_spatial_nodes
        trial_points, mask_points = regular_sobolov(number_candidate_nodes, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = number_of_spatial_nodes, buffer_scale = buffer_scale, r_min = r_min, r_max = r_max) # lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0
        # x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = mask_points)
        x_grid = farthest_point_sampling(trial_points, number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_boost, mask_candidates = mask_points)


        ## Slight inconsistency of warped spacing
        ## Compute grid health
        metrics = compute_warped_expected_spacing(
            number_of_spatial_nodes, 
            lat_range=lat_range_extend, 
            lon_range=lon_range_extend,
            depth_range=depth_range, 
            time_range=time_shift_range,
            scale_time=scale_time, 
            depth_boost=1.0, ## Should use depth boost (check Hawaii case)
            use_global=use_global,
            r_min = r_min,
            r_max = r_max
        )

        # Unpack using your exact variable names
        Volume, Volume_space, Area, nominal_spacing, nominal_spacing_space, nominal_spacing_time = metrics

        compute_final_grid_health(x_grid, scale_time, depth_boost, lat_range_extend, lon_range_extend, depth_range, time_shift_range, buffer_scale, Volume)

        ## Find domain scale parameters
        x_grid_cart = ftrns1(x_grid)
        x_grid_proj = np.concatenate((x_grid_cart, scale_time*x_grid[:,[3]]), axis = 1)


        k_edges = k_spc_edges ## An effective edge number in 4D
        edges = np.ascontiguousarray(np.flip(sort_edge_index(remove_self_loops(knn(torch.Tensor(np.concatenate((x_grid_cart, scale_time*x_grid[:,[3]]), axis = 1)), torch.Tensor(np.concatenate((x_grid_cart, scale_time*x_grid[:,[3]]), axis = 1)), k = k_edges + 1))[0].flip(0)).contiguous().cpu().detach().numpy(), axis = 0))
        dist_arg = np.argmin(np.linalg.norm(x_grid_proj[edges[0]] - x_grid_proj[edges[1]], axis = 1).reshape(-1, k_edges), axis = 1) + k_edges*np.arange(len(x_grid))
        # mean_nearest_neighbor = scatter(torch.Tensor(np.linalg.norm(x_grid_cart[edges[0]] - x_grid_cart[edges[1]], axis = 1)).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
        # mean_nearest_neighbor_t = scatter(torch.Tensor(np.abs(x_grid[edges[0],3] - x_grid[edges[1],3])).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
        mean_nearest_neighbor = np.linalg.norm(x_grid_cart[edges[1][dist_arg]] - x_grid_cart[edges[0][dist_arg]], axis = 1).mean() # scatter(torch.Tensor(np.linalg.norm(x_grid_cart[edges[0]] - x_grid_cart[edges[1]], axis = 1)).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
        mean_nearest_neighbor_t = np.abs(x_grid[edges[1][dist_arg],3] - x_grid[edges[0][dist_arg],3]).mean() # scatter(torch.Tensor(np.abs(x_grid[edges[0],3] - x_grid[edges[1],3])).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
        final_W_phys = mean_nearest_neighbor
        final_W_t = mean_nearest_neighbor_t
        edges_src = np.copy(edges)
        edges_sta = np.ascontiguousarray(np.flip(sort_edge_index(remove_self_loops(knn(torch.Tensor(ftrns1(locs_use)/1000.0), torch.Tensor(ftrns1(locs_use)/1000.0), k = k_sta_edges + 1))[0].flip(0)).contiguous().cpu().detach().numpy(), axis = 0))


        source_label_width_grid = final_W_phys*1.2
        source_label_width_t_grid = final_W_t*1.2
        association_label_width_grid = final_W_phys*1.2
        association_label_width_t_grid = final_W_t*1.5

        try:
            assert(np.abs(source_label_width_grid - source_label_width)/source_label_width < 0.35)
            assert(np.abs(source_label_width_t_grid - source_label_width_t)/source_label_width_t < 0.35)
            assert(np.abs(association_label_width_grid - association_label_width)/association_label_width < 0.35)
            assert(np.abs(association_label_width_t_grid - association_label_width_t)/association_label_width_t < 0.35)
        except:
            print('Relative differences of parameters large [1]')

        # assign_based_on_grid = False
        if assign_based_on_grid == True:
            source_label_width = source_label_width_grid
            source_label_width_t = source_label_width_t_grid
            association_label_width = association_label_width_grid
            association_label_width_t = association_label_width_t_grid

        # The 4D Geometric Slack
        sigma_input_grid = np.sqrt((source_label_width_t/2)**2 + (source_label_width/(2*Vc))**2)

        try:
            assert(np.abs(sigma_input_grid - sigma_input)/sigma_input < 0.35)
        except:
            print('Relative differences of parameters large [2]')

        if assign_based_on_grid == True:
            sigma_input = sigma_input_grid

    else:

        ## Call fit domain
        fit_spatial_domain(locs_use, stas_use, scale_domain, deg_padding, number_of_spatial_nodes, k_spc_edges, k_sta_edges, depth_range, ftrns1, ftrns2, use_global = False, assign_based_on_grid = False, max_nodes = 1500, n_trgt_nodes = 100e3, Vc = 3500.0, file_index = file_index, date = [2000, 1, 1], rbest = None, mn = None, domain = None, n_rand_srcs = 150, quantile_times = 0.35, quantile_times_srcs = 0.5, device = 'cpu')

        z = np.load('Domains/domain_parameters_%d_%d_%d_%d_ver_1.npz'%(file_index, date[0], date[1], date[2]))
        # scale_time = scale_time, depth_boost = depth_upscale_factor, locs_use = locs_use, stas_use = stas_use, x_grid = x_grid, lat_range = lat_range, lon_range = lon_range, lat_range_extend = lat_range_extend, lon_range_extend = lon_range_extend, depth_range = depth_range, deg_padding = deg_padding, time_shift_range = time_shift_range, buffer_scale = buffer_scale, source_label_width = source_label_width, source_label_width_t = source_label_width_t, association_label_width = association_label_width, association_label_width_t = association_label_width_t, sigma_input = sigma_input)
        scale_time = z['scale_time']
        depth_boost = z['depth_boost']
        locs_use = z['locs_use']
        stas_use = z['stas_use']
        x_grid = z['x_grid']
        lat_range = z['lat_range']
        lon_range = z['lon_range']
        lat_range_extend = z['lat_range_extend']
        lon_range_extend = z['lon_range_extend']
        depth_range = z['depth_range']
        deg_padding = z['deg_padding']
        time_shift_range = z['time_shift_range']
        buffer_scale = z['buffer_scale']
        source_label_width = z['source_label_width']
        source_label_width_t = z['source_label_width_t']
        association_label_width = z['association_label_width']
        association_label_width_t = z['association_label_width_t']
        sigma_input = z['sigma_input']
        z.close()


    k_sta_edges = min(k_sta_edges, len(locs_use) - 1)

    if optimize_station_graph == True:
        G_sta, edges_sta = optimize_station_graph(locs_use, ftrns1, k_sta_edges, init_knn = 3)

    else:
        G_sta, _, _, _, _, _, _, _, _ = initialize_sensor_graph(ftrns1(locs_use)/1000.0, init_knn = k_sta_edges, k_trgt = k_sta_edges)
        edges_sta, weights_sta, weights_sta = convert_graph(G_sta)
        # edges_sta1, weights_sta1, weights_sta1 = convert_graph(G_sta)
        edges_sta = np.flip(edges_sta, axis = 0)


    if optimize_source_graph == True:
        G_src, edges_src = optimize_source_graph(x_grid, ftrns1, k_spc_edges, scale_time, k_init_ratio = 0.8)

    else:
        G_src, _, _, _, _, _, _, _, _ = initialize_sensor_graph(x_grid_proj/1000.0, init_knn = k_spc_edges, k_trgt = k_spc_edges)
        edges_src, weights_src, degrees_src = convert_graph(G_src)
        edges_src = np.flip(edges_src, axis = 0)


    # n_trgt_nodes = 100e3
    min_ratio = 0.001
    max_ratio = 1.0
    n_fraction = min(max(min_ratio, n_trgt_nodes/(x_grid_proj.shape[0]*locs_cart.shape[0])), max_ratio)


    Product = SpectralProductSampler(G_src, G_sta, x_grid_proj[:,0:3], locs_cart) # locs_cart, srcs_cart
    G_product = Product.build_final_subgraph(target_node_count = int(n_fraction*len(x_grid_proj)*len(locs_cart)), skip_paths = True if not use_paths else False)        
    A_src_in_sta = np.flip(np.vstack(list(G_product.nodes())).T, axis = 0)
    isort = np.lexsort((A_src_in_sta[0], A_src_in_sta[1]))
    A_src_in_sta = A_src_in_sta[:,isort]
    A_src_in_sta = np.concatenate((A_src_in_sta, np.log10(np.linalg.norm(locs_cart[A_src_in_sta[0]] - x_grid_proj[A_src_in_sta[1],0:3], axis = 1) + 1.0).reshape(1,-1)), axis = 0)


    ## Convert bipartite graph
    edges_sta = np.vstack([[u,v,d['weight']] for u,v,d in G_sta.edges(data = True)])
    edges_src = np.vstack([[u,v,d['weight']] for u,v,d in G_src.edges(data = True)])
    edges_sta = np.flip(edges_sta, axis = 0)
    edges_src = np.flip(edges_src, axis = 0)
    A_sta = np.unique(np.concatenate((edges_sta, np.concatenate((np.flip(edges_sta[:,0:2], axis = 1), edges_sta[:,[2]]), axis = 1)), axis = 0), axis = 0).T
    A_src = np.unique(np.concatenate((edges_src, np.concatenate((np.flip(edges_src[:,0:2], axis = 1), edges_src[:,[2]]), axis = 1)), axis = 0), axis = 0).T
    A_sta = np.ascontiguousarray(np.concatenate((np.flip(A_sta[0:2,:], axis = 0), A_sta[[2],:]), axis = 0))
    A_src = np.ascontiguousarray(np.concatenate((np.flip(A_src[0:2,:], axis = 0), A_src[[2],:]), axis = 0))

    # A_sta_sta, A_src_src, A_prod_sta_sta, A_prod_src_src, A_prod_sta_sta_weights, A_prod_src_src_weights, A_src_in_prod, A_src_in_sta
    _, _, A_prod_sta_sta, A_prod_src_src, A_prod_sta_sta_weights, A_prod_src_src_weights, A_src_in_prod, _ = extract_inputs_subgraph(locs_cart, x_grid_proj, torch.Tensor(A_src_in_sta[0:2,:]).long(), A_src, A_sta)
    remove_isolated_nodes = True
    if remove_isolated_nodes == True: ## Remove any nodes on the product that have no induced edges
        min_degree = 1
        idel, inc_cnt, max_iter = [[]], 0, 10
        while (len(idel) > 0) * (inc_cnt < max_iter):
            count_bin1 = np.bincount(A_prod_sta_sta[0].cpu().detach().numpy().astype('int'))
            count_bin2 = np.bincount(A_prod_src_src[0].cpu().detach().numpy().astype('int'))
            idel1, idel2 = np.where(count_bin1 < min_degree)[0], np.where(count_bin2 < min_degree)[0]
            # idel = np.array(list(set(idel1).union(idel2)))
            idel = np.array(list(set(idel1).intersection(idel2)))
            # idel = np.array(list(set(idel1).intersection(idel2)))
            if len(idel) == 0: break
            print('%d Deleting %d of %d (partially) isolated nodes, totally isolated: %d'%(inc_cnt, len(idel), A_src_in_sta.shape[1], len(np.array(list(set(idel1).intersection(idel2))))))
            A_src_in_sta = np.delete(A_src_in_sta, idel, axis = 1)
            isort = np.lexsort((A_src_in_sta[0], A_src_in_sta[1]))
            A_src_in_sta = A_src_in_sta[:,isort] ## Unneccessary
            _, _, A_prod_sta_sta, A_prod_src_src, A_prod_sta_sta_weights, A_prod_src_src_weights, A_src_in_prod, _ = extract_inputs_subgraph(locs_cart, x_grid_proj, torch.Tensor(A_src_in_sta[0:2,:]).long(), A_src, A_sta)
            inc_cnt += 1

    A_prod_sta_sta = A_prod_sta_sta.cpu().detach().numpy()
    A_prod_src_src = A_prod_src_src.cpu().detach().numpy()
    A_prod_sta_sta_weights = A_prod_sta_sta_weights.cpu().detach().numpy()
    A_prod_src_src_weights = A_prod_src_src_weights.cpu().detach().numpy()
    A_src_in_prod = A_src_in_prod.cpu().detach().numpy()
    ## Sanity check weights of product

    irand = np.random.choice(A_prod_sta_sta.shape[1], size = min(A_prod_sta_sta.shape[1] - 1, 10000)) ## Paired stations on product
    sta_ref_inds1 = A_src_in_sta[0][A_prod_sta_sta[0][irand].astype('int')]
    sta_ref_inds2 = A_src_in_sta[0][A_prod_sta_sta[1][irand].astype('int')]
    for sta_node1, sta_node2 in zip(sta_ref_inds1, sta_ref_inds2):
        assert(sta_node2 in G_sta.neighbors(int(sta_node1)))
        assert(sta_node1 in G_sta.neighbors(int(sta_node2)))

    irand = np.random.choice(A_prod_src_src.shape[1], size = min(A_prod_src_src.shape[1] - 1, 10000)) ## Paired stations on product
    src_ref_inds1 = A_src_in_sta[1][A_prod_src_src[0][irand].astype('int')]
    src_ref_inds2 = A_src_in_sta[1][A_prod_src_src[1][irand].astype('int')]
    for src_node1, src_node2 in zip(src_ref_inds1, src_ref_inds2):
        assert(src_node2 in G_src.neighbors(int(src_node1)))
        assert(src_node1 in G_src.neighbors(int(src_node2)))

    degrees_srcs = np.bincount(A_src[0].astype('int'))
    degrees_stas = np.bincount(A_sta[0].astype('int'))
    degrees_bipartite_sta = np.bincount(A_src_in_sta[0].astype('int'))
    degrees_bipartite_src = np.bincount(A_src_in_sta[1].astype('int'))
    degrees_src_srcs = np.bincount(A_prod_src_src[0].astype('int'))
    degrees_sta_stas = np.bincount(A_prod_sta_sta[0].astype('int'))


    print('\nSubgraph Cartesian product: %d nodes (%d total; %0.4f)'%(A_src_in_sta.shape[1], len(x_grid_proj)*len(locs_cart), A_src_in_sta.shape[1]/(len(x_grid_proj)*len(locs_cart))))
    print('Degree distribution (source): [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(degrees_srcs, [0, 0.25, 0.5, 0.75, 1.0])))
    print('Degree distribution (sta): [%0.3f, %0.3f, %0.3f, %0.3f, %0.3f]' % tuple(np.quantile(degrees_stas, [0, 0.25, 0.5, 0.75, 1.0])))
    print('Degree distribution (bipartite sta): [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f]' % tuple(np.quantile(degrees_bipartite_sta, [0, 0.25, 0.5, 0.75, 1.0])))
    print('Degree distribution (bipartite src): [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f]' % tuple(np.quantile(degrees_bipartite_src, [0, 0.25, 0.5, 0.75, 1.0])))
    print('Degree distribution (source-source): [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f]' % tuple(np.quantile(degrees_src_srcs, [0, 0.25, 0.5, 0.75, 1.0])))
    print('Degree distribution (station-station): [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f]' % tuple(np.quantile(degrees_sta_stas, [0, 0.25, 0.5, 0.75, 1.0])))


    fit_local_projection = False
    if fit_local_projection == True:

        fix_nominal_depth = True
        assert(fix_nominal_depth == True)
        if fix_nominal_depth == True:
            nominal_depth = 0.0 ## Can change the target depth projection if prefered
        else:
            nominal_depth = locs_use[:,2].mean() ## Can change the target depth projection if prefered
        
        center_loc = np.array([lat_range[0] + 0.5*np.diff(lat_range)[0], lon_range[0] + 0.5*np.diff(lon_range)[0], nominal_depth]).reshape(1,-1)

        # os.rename(ext_dir + 'stations.npz', ext_dir + '%s_stations_backup.npz'%name_of_project)
        soln = optimize_with_differential_evolution(center_loc)
        rbest = rotation_matrix_full_precision(soln.x[0], soln.x[1], soln.x[2])
        mn = soln.x[3::].reshape(1,-1)

    build_expander = True
    if build_expander == True:
        def generate_regular_expander(d, n, max_tries = 100):
            for i in range(max_tries):
                try:
                    G = nx.random_regular_graph(d, n)
                    if nx.is_connected(G):  # optional, but will always be true
                        return G
                except nx.NetworkXError:
                    pass  # retry on failure
            raise ValueError(f"Failed to generate {d}-regular graph on {n} nodes after {max_tries} tries")
        # G = nx.random_regular_graph(d = 8, n = number_of_spatial_nodes)  # d-regular on n nodes

        regular_degree = 8
        G = generate_regular_expander(regular_degree, number_of_spatial_nodes) ## Can optimize
        G = from_networkx(G)
        Ac = np.flip(G.edge_index.cpu().detach().numpy(), axis = 0)
        edge_type = np.zeros(Ac.shape[1])
    else:
        Ac = None



    # folder_path = "path/to/your/folder"
    os.makedirs('Domains', exist_ok=True)
    np.savez_compressed('Domains/domain_file_%d_%d_%d_%d_ver_1.npz'%(file_index, date[0], date[1], date[2]), A_src_in_sta = A_src_in_sta, A_sta = A_sta, A_src = A_src, Ac = Ac, A_prod_sta_sta = A_prod_sta_sta, A_prod_src_src = A_prod_src_src, A_prod_sta_sta_weights = A_prod_sta_sta_weights, A_prod_src_src_weights = A_prod_src_src_weights, A_src_in_prod = A_src_in_prod, x_grid = x_grid, scale_time = scale_time, depth_boost = depth_boost, ichoose_grid = 0, locs_use = locs_use, stas_use = stas_use, srcs_cart = x_grid_cart, locs_cart = locs_cart, lat_range = lat_range, lon_range = lon_range, lat_range_extend = lat_range_extend, lon_range_extend = lon_range_extend, depth_range = depth_range, deg_padding = deg_padding, time_shift_range = time_shift_range, source_label_width = source_label_width, source_label_width_t = source_label_width_t, association_label_width = association_label_width, association_label_width_t = association_label_width_t, sigma_input = sigma_input, rbest = rbest, mn = mn) # ind_use = np.arange(len(locs_use)) # metrics_product = metrics_product

    # print('Finished building graphs %d %d %d'%(date[0] + yr_inc, date[1], date[2]))
    print('Finished building graphs %d'%(file_index))
    print('Num nodes: %d Sta, %d Src, %d subgraph'%(len(G_sta.nodes()), len(G_src.nodes()), A_src_in_sta.shape[1]))
    print('Edges: %d Sta, %d Src, %d subgraph'%(len(G_sta.edges()), len(G_src.edges()), A_prod_src_src.shape[1] + A_prod_sta_sta.shape[1]))




def fit_spatial_domain(locs_use, stas_use, scale_domain, deg_padding, number_of_spatial_nodes, k_spc_edges, k_sta_edges, depth_range, ftrns1, ftrns2, use_global = False, assign_based_on_grid = False, max_nodes = 1500, n_trgt_nodes = 100e3, Vc = 3500.0, file_index = 0, date = [2000, 1, 1], rbest = None, mn = None, domain = None, n_rand_srcs = 150, quantile_times = 0.35, quantile_times_srcs = 0.5, device = 'cpu'):

    if domain is None:
        domain = get_domain_bounds(locs_use, scale = scale_domain)

    # domain = get_domain_bounds(locs_use, scale = scale_domain)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_scale = estimate_kernel_widths(domain, locs_use, z_range = depth_range, Vs = Vc, noise_level = 0.015, n_neighbors_trgt = 20)

    lat_range, lon_range = domain['lat_range'], domain['lon_range']
    lat_range_extend, lon_range_extend = extend_geo_range(lat_range, lon_range, domain_scale['W_phys_m'], multiplier = 2.0)
    deg_padding = np.mean([lat_range_extend[1] - lat_range[1], lat_range[0] - lat_range_extend[0], lon_range_extend[1] - lon_range[1], lon_range[0] - lon_range_extend[0]]) if use_global == False else 0.0

    Dt_offsets = []
    for i in range(n_rand_srcs):

        src_true, side_lobes, t_obs_picks, [max_radius_m, max_dt] = probe_network_sidelobes_geodetic(locs_use, lat_range_extend, lon_range_extend, depth_range,
                                         k_stations=max(8, np.random.choice(np.arange(int(0.1*len(locs_use)), int(0.5*len(locs_use))))), vel_avg=Vc, vel_min=Vc*0.75,
                                         scan_step_m=domain_scale['W_phys_m']/2.0, W_phys_m = domain_scale['W_phys_m'], W_t = domain_scale['W_t_s'], device=device)

        dt_sort = np.sort(np.abs(np.array([s['dt_offset'] for s in side_lobes])))
        Dt_offsets.append(np.quantile(dt_sort, quantile_times))

        trv_out = trv(torch.Tensor(locs_use[t_obs_picks[:,1].astype('int')]).to(device), torch.Tensor(src_true).to(device)).cpu().detach().numpy()
        trv_out_sidelobes = np.vstack([trv(torch.Tensor(locs_use[t_obs_picks[:,1].astype('int')]).to(device), torch.Tensor(s['pos_src'].reshape(1,-1)).to(device)).cpu().detach().numpy() + s['dt_offset'] for s in side_lobes])


    Dt_offsets = np.array(Dt_offsets)
    # time_shift_range = np.quantile(Dt_offsets, quantile_times)/2.0
    time_shift_range = np.round(np.quantile(Dt_offsets, quantile_times_srcs), 2) # /2.0


    def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
        r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
        r_val = r_surface + h_min
        return r_val


    def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
        r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
        r_val = r_surface + h_max
        return -r_val


    bounds = [(lat_range_extend[0], lat_range_extend[1])]
    soln = differential_evolution(optimize_r_min, bounds, popsize = 50, maxiter = 1000, disp = True)
    r_min = optimize_r_min(np.array([soln.x]))[0]; print('\n')

    bounds = [(lat_range_extend[0], lat_range_extend[1])]
    soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True)
    r_max = -1.0*optimize_r_max(np.array([soln.x]))[0]; print('\n')
    assert(r_max >= r_min)


    scale_time_base = domain_scale['W_phys_m']/domain_scale['W_t_s']


    final_N, final_scale_time, final_W_phys, final_W_t = fit_domain_budget_aware(domain_scale['W_phys_m'], domain_scale['W_t_s'], lat_range_extend, lon_range_extend, depth_range, time_shift_range, 
                             N_max = max_nodes, depth_boost=1.0, use_global=use_global)


    ## Run the auto tuning strategy to refine some scale parameters
    m = SamplingTuner(final_N, lat_range_extend, lon_range_extend, depth_range, time_shift_range, scale_time_effective = final_scale_time)
    params = m.optimize()
    scale_time, depth_boost, buffer_scale = params['scale_t'], params['depth_boost'], params['buffer_scale']

    # 2. Final Metric Pass (Ground Truth)
    metrics = compute_warped_expected_spacing(
        final_N, 
        lat_range=lat_range_extend, 
        lon_range=lon_range_extend,
        depth_range=depth_range, 
        time_range=time_shift_range,
        # scale_time=final_scale_time, 
        scale_time=scale_time, 
        depth_boost=1.0, 
        use_global=use_global
    )

    # Unpack using your exact variable names
    Volume, Volume_space, Area, nominal_spacing, nominal_spacing_space, nominal_spacing_time = metrics

    ## Create grid:

    use_time_shift = True
    use_station_density = False
    number_of_spatial_nodes = final_N
    print('Beginning FPS sampling [%d]'%n)
    up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
    if use_station_density == True: up_sample_factor = up_sample_factor*5
    number_candidate_nodes = up_sample_factor*number_of_spatial_nodes
    trial_points, mask_points = regular_sobolov(number_candidate_nodes, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = number_of_spatial_nodes, buffer_scale = buffer_scale) # lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0
    # x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = mask_points)
    x_grid = farthest_point_sampling(trial_points, number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_boost, mask_candidates = mask_points)
    x_grid_cart = ftrns1(x_grid)
    x_grid_proj = np.concatenate((x_grid_cart, scale_time*x_grid[:,[3]]), axis = 1)

    tol_frac = 0.01
    assert(x_grid[:,0].min() >= (lat_range_extend[0] - tol_frac*np.diff(lat_range_extend)))
    assert(x_grid[:,0].max() <= (lat_range_extend[1] + tol_frac*np.diff(lat_range_extend)))
    assert(x_grid[:,1].min() >= (lon_range_extend[0] - tol_frac*np.diff(lon_range_extend))) if use_global == False else 1
    assert(x_grid[:,1].max() <= (lon_range_extend[1] + tol_frac*np.diff(lon_range_extend))) if use_global == False else 1
    assert(x_grid[:,2].min() >= (depth_range[0] - tol_frac*np.diff(depth_range)))
    assert(x_grid[:,2].max() <= (depth_range[1] + tol_frac*np.diff(depth_range)))
    assert(len(x_grid) == number_of_spatial_nodes)

    # if n == 0:
    compute_final_grid_health(x_grid, scale_time, depth_boost, lat_range_extend, lon_range_extend, depth_range, time_shift_range, buffer_scale, Volume)
    compute_final_grid_health(x_grid, scale_time, depth_boost, lat_range_extend, lon_range_extend, depth_range, time_shift_range, buffer_scale, Volume)
    perform_ks_density_test(x_grid, lat_range_extend)
    perform_ks_depth_test_ellipsoid(x_grid, depth_range)

    k_edges = k_spc_edges # 18 ## An effective edge number in 4D
    edges = np.ascontiguousarray(np.flip(sort_edge_index(remove_self_loops(knn(torch.Tensor(np.concatenate((x_grid_cart, scale_time*x_grid[:,[3]]), axis = 1)), torch.Tensor(np.concatenate((x_grid_cart, scale_time*x_grid[:,[3]]), axis = 1)), k = k_edges + 1))[0].flip(0)).contiguous().cpu().detach().numpy(), axis = 0))

    dist_arg = np.argmin(np.linalg.norm(x_grid_proj[edges[0]] - x_grid_proj[edges[1]], axis = 1).reshape(-1, k_edges), axis = 1) + k_edges*np.arange(len(x_grid))
    # mean_nearest_neighbor = scatter(torch.Tensor(np.linalg.norm(x_grid_cart[edges[0]] - x_grid_cart[edges[1]], axis = 1)).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
    # mean_nearest_neighbor_t = scatter(torch.Tensor(np.abs(x_grid[edges[0],3] - x_grid[edges[1],3])).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
    mean_nearest_neighbor = np.linalg.norm(x_grid_cart[edges[1][dist_arg]] - x_grid_cart[edges[0][dist_arg]], axis = 1).mean() # scatter(torch.Tensor(np.linalg.norm(x_grid_cart[edges[0]] - x_grid_cart[edges[1]], axis = 1)).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()
    mean_nearest_neighbor_t = np.abs(x_grid[edges[1][dist_arg],3] - x_grid[edges[0][dist_arg],3]).mean() # scatter(torch.Tensor(np.abs(x_grid[edges[0],3] - x_grid[edges[1],3])).reshape(-1,1), torch.Tensor(edges[1]).long(), dim = 0, reduce = 'min').cpu().detach().numpy().mean()

    final_W_phys = mean_nearest_neighbor
    final_W_t = mean_nearest_neighbor_t

    # The 4D Geometric Slack
    dt_geometric = np.sqrt((final_W_t/2)**2 + (final_W_phys/(2*Vc))**2)


    sigma_input = 1.0 * dt_geometric
    print('Num nodes: %d'%final_N)
    print('scale_time: %0.4f'%scale_time)
    print('Time shift range: %0.2f'%time_shift_range)
    print('Degree padding: %0.2f'%deg_padding)
    print('Spatial spacing: %0.4f'%final_W_phys)
    print('Temporal spacing: %0.4f'%final_W_t)
    # print('Input kernel: %0.4f'%sigma_input)

    source_label_width = final_W_phys*1.2
    source_label_width_t = final_W_t*1.2

    association_label_width = final_W_phys*1.2
    association_label_width_t = final_W_t*1.5


    print('\nLabels widths:')
    print('Source label width (space): %0.4f'%source_label_width)
    print('Source label width (time): %0.4f'%source_label_width_t)
    print('Association label width (space): %0.4f'%association_label_width)
    print('Association label width (time): %0.4f'%association_label_width_t)
    print('Input kernel: %0.4f'%sigma_input)

    np.savez_compressed('Domains/domain_parameters_%d_%d_%d_%d_ver_1.npz'%(file_index, date[0], date[1], date[2]), scale_time = scale_time, depth_boost = depth_boost, locs_use = locs_use, stas_use = stas_use, x_grid = x_grid, lat_range = lat_range, lon_range = lon_range, lat_range_extend = lat_range_extend, lon_range_extend = lon_range_extend, depth_range = depth_range, deg_padding = deg_padding, time_shift_range = time_shift_range, buffer_scale = buffer_scale, source_label_width = source_label_width, source_label_width_t = source_label_width_t, association_label_width = association_label_width, association_label_width_t = association_label_width_t, sigma_input = sigma_input)


def optimize_station_graph(locs_use, ftrns1, k_sta_edges, init_knn = 3):

    locs_proj = ftrns1(locs_use)/(1000.0)
    locs_cart = np.copy(locs_proj)

    ## Use the target number of edges per station
    k_trgt = k_sta_edges

    G, edges, fiedler, curvature, clustering, degree_values, components, fiedler_value, scale_length = initialize_sensor_graph(locs_cart, init_knn = init_knn, k_trgt = k_trgt)
    edges_slice = edges[0:2,:].astype('int')
    tracked_values = None

    n_new_edges = int(1e9)
    edges_to_update = None
    for i in range(n_new_edges):
        if i > 0:
            G, edges, fiedler, curvature, clustering, degree_values, components, fiedler_value, scale_length = initialize_sensor_graph(locs_cart, cnt = i, k_trgt = k_trgt, G = G, edges_to_update = edges_to_update)

        if len(G.edges()) > len(locs_cart)*int(k_trgt/2):
            break

        if len(G.edges()) < (len(locs_cart)*int(k_trgt/2))/2:
            G, d, dv1, edges_to_update = get_most_impactful_edges_balanced(G, G.graph['scale_length'], cnt = i, top_k = 5, greedy_regularize = 1.0, use_triangles = False, update = True)
        else:
            G, d, dv1, edges_to_update = get_most_impactful_edges_balanced(G, G.graph['scale_length'], cnt = i, top_k = 5, greedy_regularize = 1.0, use_triangles = True, update = True)

        print('%d, Num edges: %d'%(i, len(G.edges())))


    G_sta = G.copy()
    edges_sta, weights_sta, degrees_sta = convert_graph(G_sta)
    edges_sta = np.flip(edges_sta, axis = 0)

    return G_sta, edges_sta


def optimize_source_graph(x_grid, ftrns1, k_spc_edges, scale_time, k_init_ratio = 0.8):

    k_trgt = k_spc_edges

    srcs_cart = np.concatenate((ftrns1(x_grid[:,0:3]), scale_time*x_grid[:,[3]]), axis = 1)/1000.0

    G, edges, fiedler, curvature, clustering, degree_values, components, fiedler_value, scale_length = initialize_sensor_graph(srcs_cart, init_knn = int(k_init_ratio*k_spc_edges), k_trgt = k_trgt)
    edges_slice = edges[0:2,:].astype('int')
    edges_slice = np.flip(edges_slice, axis = 0)

    tracked_values = None
    n_new_edges = int(1e9)

    edges_to_update = None
    for i in range(n_new_edges):

        if i > 0:
            G, edges, fiedler, curvature, clustering, degree_values, components, fiedler_value, scale_length = initialize_sensor_graph(srcs_cart, cnt = i, k_trgt = k_trgt, G = G, edges_to_update = edges_to_update)

        if len(G.edges()) > len(srcs_cart)*int(k_trgt/2):
            break

        if len(G.edges()) < (len(srcs_cart)*int(k_trgt/2))/2:
            G, d, dv1, edges_to_update = get_most_impactful_edges_balanced(G, G.graph['scale_length'], cnt = i, top_k = 30, greedy_regularize = 1.0, use_triangles = False, update = True)
        else:
            G, d, dv1, edges_to_update = get_most_impactful_edges_balanced(G, G.graph['scale_length'], cnt = i, top_k = 30, greedy_regularize = 1.0, use_triangles = True, update = True)

        print('%d, Num edges: %d'%(i, len(G.edges())))

    G_src = G.copy()
    edges_src, weights_src, degrees_src = convert_graph(G_src)
    edges_src = np.flip(edges_src, axis = 0)

    return G_src, edges_src



def get_most_impactful_edges_balanced(G, scale_length, cnt = 0, top_k=5, degree_p=1.0, greedy_regularize = 0.9, greedy_regularize_extra = 0.1, alpha_scale = 0.75, beta_scale = 0.1, min_weight = 0.05, exclusion_scale = 1.0, mode = 'univariate', use_triangles = False, update = False): # tracked_values = None, track_resistance = True

    nodes = list(G.nodes(data=True))
    n = len(nodes)
    n_dim = len(G.nodes[0]['pos'])

    # node_weights = {n: G.degree(n, weight='weight') for n in G.nodes()}
    node_weights = np.array([v for n, v in G.degree(weight='weight')])
    node_inv_sqrt_sum = np.array([sum(1.0 / np.sqrt(G[n][nbr]['weight']) for nbr in G.neighbors(n)) for n in G.nodes()])


    use_normalized_ricci = True
    gamma_penalty = 0.2
    tau_scale = 0.1


    if (greedy_regularize == 1.0)*(greedy_regularize_extra > 0):
        if np.random.rand() < greedy_regularize_extra:
            greedy_regularize = 1.0 - greedy_regularize_extra

    if greedy_regularize != 1.0:
        L = nx.laplacian_matrix(G, weight='weight').astype(float)
        solve = spla.factorized((L + 1e-4 * sp.eye(n)).tocsc())


    # Before the loop starts, create a mapping of component IDs to their sizes
    comp_ids = np.array([du.get('comp_id', 0) for _, du in nodes])
    # from collections import Counter
    comp_sizes = Counter(comp_ids)
    weights = G.graph['weights']
    distances = G.graph['distances']
    scale_values = G.graph['scale_values']
    edge_allowed = G.graph['allowed_edges'] # G.graph['allowed_edges']

    if mode == 'bipartite':
        n_nodes_s = G.graph['n_nodes_s']
        n_nodes_r = G.graph['n_nodes_r']
        scale_src = G.graph['scale_src']
        scale_sta = G.graph['scale_sta']

    # ilist1, ilist2 = np.where(mask_allowed > 0)
    # edge_allowed = np.concatenate((ilist1.reshape(-1,1) ilist2.reshape(-1,1)), axis = 1)
    # edge_allowed = edges_allowed[edges_allowed[:,0] <= edges_allowed[:,1]]

    fiedler_vec = np.array([v['fiedler'] for n, v in nodes])
    curvature = np.array([v['total_curvature'] for n, v in nodes])
    degrees = np.array([v for u,v in G.degree()])

    edge_slice = np.stack(list(G.edges()))
    tree_edges = cKDTree(np.concatenate((edge_slice, np.flip(edge_slice, axis = 1)), axis = 0))



    top_k_approx = np.minimum(250*top_k, int(0.1*len(edge_allowed)))
    top_k_approx1 = np.minimum(2*top_k_approx, int(0.2*len(edge_allowed)))
    if cnt == 0: print('Top k values : %d, %d, %d'%(top_k, top_k_approx, top_k_approx1)) # *len(patch)

    # mov_mean_resistance = -1
    ## Also estimate the (even larger) k min required to ensure sum - resistance - curvature is within top k set

    # candidates = [] ## How can we pre-determine the top-k pairs to try at all for the search?
    batch_size_val = int(50e3)
    ind_batches = [np.arange(int(batch_size_val)) + i*batch_size_val for i in range(int(np.ceil(len(edge_allowed)/batch_size_val)))]    
    if (ind_batches[-1][-1] < len(edge_allowed))*(len(ind_batches) > 1):
        ind_batches.append(np.arange(ind_batches[-1][-1] + 1, len(edge_allowed)))
    elif (ind_batches[-1][-1] < len(edge_allowed)):
        ind_batches[0] = np.arange(len(edge_allowed))
    elif (ind_batches[-1][-1] >= len(edge_allowed)):
        ind_batches[-1] = np.arange(ind_batches[-1][0], len(edge_allowed))
    assert(max([len(ind_batches[i]) for i in range(len(ind_batches))]) <= 1.5*batch_size_val)


    ## Initial pass
    candidates = []
    use_greedy = []

    # pdb.set_trace()
    for batch_ind in ind_batches:
        edge_slice = edge_allowed[batch_ind,:] # .T
        edge_slice = edge_slice[tree_edges.query(edge_slice)[0] > 0]
        # u_slice, v_slice = edge_slice[tree_edges.query(edge_slice)[0] > 0].T
        u_slice, v_slice = edge_slice.T

        w_new = weights[u_slice,v_slice]
        spec_log = np.log(np.abs(fiedler_vec[u_slice] - fiedler_vec[v_slice])*(comp_ids[u_slice] == comp_ids[v_slice]) + 1e-9) # + np.log(eff_res + 1e-9) ## Consider non local weight scaling for distance
        cost_log = np.log(1.0 + (distances[u_slice,v_slice] / scale_values[u_slice, v_slice])/2.0) # (dist / scale_length)/2.0 # This is the log of your 'cost' variable
        balance_log = degree_p * np.log(degrees[u_slice] + degrees[v_slice] + 1e-9)
        score_approx1 = spec_log - cost_log - balance_log

        irand_greedy = 1.0*(np.random.rand(len(edge_slice)) < greedy_regularize).reshape(-1,1)
        candidates.append(np.concatenate((edge_slice, w_new.reshape(-1,1), score_approx1.reshape(-1,1) - irand_greedy*spec_log.reshape(-1,1), spec_log.reshape(-1,1)), axis = 1))
        use_greedy.append(irand_greedy.reshape(-1))

    ## Select top percentile candidates (with greedy shortcut)
    candidates = np.vstack(candidates)
    use_greedy = np.hstack(use_greedy)

    # igrab_optimal = np.flip(np.argsort(candidates[:,3])[-top_k_approx1::])

    igrab_optimal = np.flip(np.argsort(candidates[:,3]*(use_greedy == 0))[-top_k_approx1::]).astype('int')
    igrab_greedy = np.flip(np.argsort(candidates[:,3]*(use_greedy == 1))[-top_k_approx1::]).astype('int')
    # pdb.set_trace()

    # try:
    igrab_optimal = np.array(list(set(igrab_optimal).union(igrab_greedy))).astype('int')
    # except:

    candidates = candidates[igrab_optimal]
    use_greedy = use_greedy[igrab_optimal]

    use_loop_version = False
    if use_loop_version == True:

        ## Update scores with Curvature and smoothness losses
        score_approx = []
        for c in candidates: # [0:top_k_approx]:

            u, v, w_new = c[0:3] # c['u'], c['v'], c['w_new']
            u, v = int(u), int(v)
            u_neigh = list(G.neighbors(u))
            v_neigh = list(G.neighbors(v))
            weight_u_neigh = weights[u, u_neigh]
            weight_v_neigh = weights[v, v_neigh]

            patch = list(set([u,v] + u_neigh + v_neigh))
            patch_curv = curvature[patch]

            if use_normalized_ricci == True: # u, v, w_new, deg_u, deg_v, curv_u, curv_v, u_neigh, v_neigh, weight_u_neig, weight_v_neigh, node_inv_sqrt_sum_u, node_inv_sqrt_sum_v, deg_u_neigh, deg_v_neigh, patch_curv
                ricci_delta, local_debt = calculate_ricci_delta(u, v, w_new, degrees[u], degrees[v], curvature[u], curvature[v], u_neigh, v_neigh, weight_u_neigh, weight_v_neigh, node_inv_sqrt_sum[u], node_inv_sqrt_sum[v], degrees[u_neigh], degrees[v_neigh], patch_curv, use_normalized = True, node_weights = None) # min_debt_thresh = 0.2 # node_weights = None
                # ricci_delta, local_debt = calculate_ricci_delta(G, u, v, w_new, node_inv_sqrt_sum[u], node_inv_sqrt_sum[v], use_normalized = True, node_weights = None) # min_debt_thresh = 0.2 # node_weights = None

            else:
                ricci_delta, local_debt = calculate_ricci_delta(u, v, w_new, degrees[u], degrees[v], curvature[u], curvature[v], u_neigh, v_neigh, weight_u_neigh, weight_v_neigh, node_inv_sqrt_sum[u], node_inv_sqrt_sum[v], degrees[u_neigh], degrees[v_neigh], patch_curv, use_normalized = False, node_weights = node_weights)
                # ricci_delta, local_debt = calculate_ricci_delta(G, u, v, w_new, node_inv_sqrt_sum[u], node_inv_sqrt_sum[v], use_normalized = False, node_weights = node_weights)

            # Option 3: Full log-space geometry reward
            ricci_scaled = local_debt * ricci_delta  # positive when healing
            geo_log = alpha_scale * np.log(1.0 + np.maximum(0, ricci_scaled))  # only reward healing, ignore neutral/negative
            if ricci_delta < 0:
                geo_log -= gamma_penalty * np.log(1.0 + abs(ricci_delta))            


            if use_triangles == True:
                common_neighbors = set(u_neigh) & set(v_neigh)
                use_forman = True
                if use_forman == True:
                    triangle_bonus = (w_new >= 0.0)*np.sum([(w_new*weights[u,c]*weights[v,c])**(1/3) for c in common_neighbors])
                    # triangle_bonus = (w_new >= 0.0)*np.sum([(w_new*G[u][c]['weight']*G[v][c]['weight'])**(1/3) for c in common_neighbors])
                else:
                    triangle_bonus = (w_new >= 0.0)*np.sqrt(w_new)*np.sum([np.sqrt(weights[u,c]*weights[v,c]) for c in common_neighbors]) ## Simple triangles
                    # triangle_bonus = (w_new >= 0.0)*np.sqrt(w_new)*np.sum([np.sqrt(G[u][c]['weight']*G[v][c]['weight']) for c in common_neighbors]) ## Simple triangles
                triangle_log = tau_scale * np.log(1.0 + triangle_bonus)
                geo_log += triangle_log

            # degree_smooth = calculate_weighted_smoothness_penalty(u, v, list(G.neighbors(u)), list(G.neighbors(v)), w_new, node_weights) # G, u, v, w_new, node_weights
            degree_smooth = calculate_weighted_smoothness_penalty(u, v, u_neigh, v_neigh, w_new, node_weights) # G, u, v, w_new, node_weights
            smooth_log = beta_scale * degree_smooth
            score_approx.append(c[3] + geo_log - smooth_log)

        score_approx = np.hstack(score_approx)
        # iarg_sort = np.flip(np.argsort(score_approx)[-top_k_approx::])

    else:

        pre_compute = precompute_large_graph(G, weights, node_inv_sqrt_sum, curvature, degrees) # G, weights, node_inv_sqrt_sum, curvature, degrees
        
        # print('Time [2] %0.3f'%(time.time() - st_time))

        score_approx = optimized_normalized_scores_large(
        G,
        candidates,  # np.array (N, 5+): [u, v, w_new, prior_score, ...] — uses [:,0:4]
        alpha_scale,
        beta_scale,
        gamma_penalty,
        tau_scale,
        use_triangles=use_triangles,
        use_forman=True,
        min_debt_thresh=0.2,
        skip_low_debt=True,
        weights=weights,  # n x n array for edge weights
        curvature=curvature,  # np.array (n,) node curvatures
        degrees=degrees,  # np.array (n,) node degrees
        node_inv_sqrt_sum=node_inv_sqrt_sum,  # np.array (n,) precomputed inv sqrt sums
        precomputed_data=pre_compute)  # Dict from precompute_large_graph


    # pdb.set_trace()
    igrab_optimal = np.flip(np.argsort(score_approx*(use_greedy == 0))[-top_k_approx::]).astype('int')
    igrab_greedy = np.flip(np.argsort(score_approx*(use_greedy == 1))[-top_k_approx::]).astype('int')
    igrab_optimal = np.array(list(set(igrab_optimal).union(igrab_greedy))).astype('int')


    score_approx = score_approx[igrab_optimal]
    candidates = candidates[igrab_optimal]
    use_greedy = use_greedy[igrab_optimal]


    use_batch = True
    if use_batch == False:

        score = []
        for i, c in enumerate(candidates): # [0:top_k_approx]:
            u, v = c[0:2]
            u, v = int(u), int(v)


            if use_greedy[i] == 0:

                # inside_component = (comp_nodes[u] == comp_nodes[v])
                inside_component = (comp_ids[u] == comp_ids[v])
                if inside_component: ## Can pre compute low likelihood edges and skip this step
                    rhs = np.zeros(n)
                    rhs[u], rhs[v] = 1, -1
                    z = solve(rhs)
                    eff_res = abs(z[u] - z[v])
                else:
                    size_u = comp_sizes[comp_ids[u]] # comp_sizes[comp_u]
                    size_v = comp_sizes[comp_ids[v]] # comp_sizes[comp_v]
                    eff_res = 1e5*(size_u + size_v)

            else:

                eff_res = 1.0

            score.append(np.exp(score_approx[i] + np.log(eff_res + 1e-9)))

        score = np.hstack(score)

    else:


        eff_res_batch = np.ones(len(candidates))  # default for greedy or inter-component

        # Find indices where we need to compute real eff_res
        need_eff_res = (use_greedy == 0)
        need_eff_res = need_eff_res & (comp_ids[candidates[:,0].astype(int)] == comp_ids[candidates[:,1].astype(int)])

        if np.any(need_eff_res):
            
            idxs = np.flatnonzero(need_eff_res)
            k = len(idxs)
            
            # Build batch RHS: (n, k) sparse matrix
            rows = np.concatenate([np.repeat(np.arange(k), 2), np.arange(k)])  # u and v rows
            cols = np.tile([0, 1], k) + np.repeat(np.arange(k), 2) * 2  # wait, simpler:
            
            # Simpler: use list of arrays
            us_batch = candidates[idxs, 0].astype(int)
            vs_batch = candidates[idxs, 1].astype(int)
            
            # Check if inside same component
            same_comp = (comp_ids[us_batch] == comp_ids[vs_batch])
            
            # For inter-component: set high resistance
            inter_comp = ~same_comp
            if np.any(inter_comp):
                size_u = np.array([comp_sizes[comp_ids[us_batch[j]]] for j in np.flatnonzero(inter_comp)])
                size_v = np.array([comp_sizes[comp_ids[vs_batch[j]]] for j in np.flatnonzero(inter_comp)])
                eff_res_batch[idxs[inter_comp]] = 1e5 * (size_u + size_v)
            
            # For intra-component: batch solve
            intra = same_comp
            if np.any(intra):
                k_intra = np.sum(intra)
                RHS = np.zeros((n, k_intra))
                intra_idxs = idxs[intra]
                us_intra = us_batch[intra]
                vs_intra = vs_batch[intra]
                
                RHS[us_intra, np.arange(k_intra)] = 1
                RHS[vs_intra, np.arange(k_intra)] = -1
                
                # Batch solve!
                Z = solve(RHS)  # (n, k_intra)
                
                eff_res_intra = np.abs(Z[us_intra, np.arange(k_intra)] - Z[vs_intra, np.arange(k_intra)])
                eff_res_batch[intra_idxs] = eff_res_intra

        # Now use eff_res_batch in scoring
        score = np.exp(score_approx + np.log(eff_res_batch + 1e-9))


    if greedy_regularize == 1.0:

        score = np.hstack(score)
        igrab_optimal = np.flip(np.argsort(score*(use_greedy == 1))[-int(np.ceil(top_k*greedy_regularize))::]).astype('int')
        if np.random.rand() < greedy_regularize:
            igrab_optimal = igrab_optimal[np.flip(np.argsort(score[igrab_optimal]*(use_greedy[igrab_optimal] == 1)))[0:int((use_greedy[igrab_optimal] == 1).sum())]]
             # igrab_optimal[np.flip(np.argsort(score[igrab_optimal]*(use_greedy[igrab_optimal] == 0)))[0:int((use_greedy[igrab_optimal] == 0).sum())]]), axis = 0)
        else:
            igrab_optimal = igrab_optimal[np.flip(np.argsort(score[igrab_optimal]*(use_greedy[igrab_optimal] == 1)))[0:int((use_greedy[igrab_optimal] == 1).sum())]]

    else:

        score = np.hstack(score)
        # igrab_optimal = np.flip(np.argsort(score*(use_greedy == 0))[-top_k::]).astype('int')
        igrab_optimal = np.flip(np.argsort(score*(use_greedy == 0))[-int(np.ceil(top_k*(1 - greedy_regularize)))::]).astype('int')
        igrab_greedy = np.flip(np.argsort(score*(use_greedy == 1))[-int(np.ceil(top_k*greedy_regularize))::]).astype('int')
        igrab_optimal = np.array(list(set(igrab_optimal).union(igrab_greedy)))

        if np.random.rand() < greedy_regularize:
            igrab_optimal = np.concatenate((igrab_optimal[np.flip(np.argsort(score[igrab_optimal]*(use_greedy[igrab_optimal] == 1)))[0:int((use_greedy[igrab_optimal] == 1).sum())]], \
             igrab_optimal[np.flip(np.argsort(score[igrab_optimal]*(use_greedy[igrab_optimal] == 0)))[0:int((use_greedy[igrab_optimal] == 0).sum())]]), axis = 0)
        else:
            igrab_optimal = np.concatenate((igrab_optimal[np.flip(np.argsort(igrab_optimal*(use_greedy[igrab_optimal] == 0)))[0:int((use_greedy[igrab_optimal] == 0).sum())]], \
             igrab_optimal[np.flip(np.argsort(igrab_optimal*(use_greedy[igrab_optimal] == 1)))[0:int((use_greedy[igrab_optimal] == 1).sum())]]), axis = 0)


    # iarg_sort = np.flip(np.argsort(score)[-top_k::])
    igrab_optimal = igrab_optimal.astype('int')
    candidates = candidates[igrab_optimal]
    use_greedy = use_greedy[igrab_optimal]
    score = score[igrab_optimal]

    print('Ratio %0.4f'%(use_greedy.sum()/len(use_greedy)))


    candidates_select = []
    node_used = []
    G_updated = G.copy()
    cnt_new_edge = 0

    if (update == True)*(len(candidates) > 0):

        edge_pairs = np.concatenate((candidates[:,[0]], candidates[:,[1]], score.reshape(-1,1), weights[candidates[:,0].astype('int'), candidates[:,1].astype('int')].reshape(-1,1), distances[candidates[:,0].astype('int'), candidates[:,1].astype('int')].reshape(-1,1)), axis = 1).T

        
        # Track which nodes and spatial neighborhoods we have already "serviced"
        nodes_used = set()
        neighborhood_midpoints = []

                
        # Iterate through candidates (they are already sorted by score)
        for i in range(edge_pairs.shape[1]):
            u = int(edge_pairs[0, i])
            v = int(edge_pairs[1, i])
            score = edge_pairs[2, i]
            w = edge_pairs[3, i]
            dist = edge_pairs[4, i]
            
            # --- Spatial Logic Check ---
            # 1. Don't reuse nodes in the same batch update
            if u in nodes_used or v in nodes_used:
                continue


            is_redundant = False ## Update prev_midpoint to record it's exclusion radius (and then take the mean of the two exclusion radius)
            if mode == 'univariate':

                scale_length_local = scale_values[u,v]
                exclusion_radius = 1.2*scale_length_local ## Correct?

                p1 = G.nodes[u]['pos']
                p2 = G.nodes[v]['pos']
                midpoint = (p1 + p2) / 2.0

                for prev_midpoint in neighborhood_midpoints:
                    if np.linalg.norm(midpoint - prev_midpoint) < exclusion_scale*exclusion_radius:
                        is_redundant = True
                        break
            
            elif mode == 'bipartite':

                p1 = G.nodes[u]['pos'].reshape(-1) ## Src
                p2 = G.nodes[v]['pos'].reshape(-1) ## Station
                midpoint = np.concatenate((p1, p2), axis = 0)

                scale_length_local_src = scale_src[u]
                scale_length_local_sta = scale_sta[v - n_nodes_s]
                exclusion_radius_src = 1.2*scale_length_local_src ## Correct?
                exclusion_radius_sta = 1.2*scale_length_local_sta ## Correct?
                scale_length_local = scale_values[u,v]


                # is_redundant = False ## Update prev_midpoint to record it's exclusion radius (and then take the mean of the two exclusion radius)
                for prev_midpoint in neighborhood_midpoints:
                    flag1 = (np.linalg.norm(midpoint[0:n_dim] - prev_midpoint[0:n_dim]) < exclusion_scale*exclusion_radius_src)
                    flag2 = (np.linalg.norm(midpoint[n_dim::] - prev_midpoint[n_dim::]) < exclusion_scale*exclusion_radius_sta)
                    # if (np.linalg.norm(midpoint[0:n_dim] - prev_midpoint[0:n_dim]) < exclusion_scale*exclusion_radius_src):
                    if flag1*flag2:
                        is_redundant = True
                        break

            else:
                print('Error, no type chosen')
                assert(1 == 0)

            if is_redundant:
                continue

            # pdb.set_trace()
            assert(w == np.exp(-dist / scale_length_local))
            if w < min_weight:
                continue

            G_updated.add_edge(int(u), int(v), dist = float(dist), weight = float(w), step = cnt)
            candidates_select.append(candidates[i])
            
            # Update tracking sets
            nodes_used.update([u, v])
            neighborhood_midpoints.append(midpoint)
            cnt_new_edge += 1

    edges_to_update = []
    if update and cnt_new_edge > 0:
        # Collect all nodes touched in this batch
        touched_nodes = nodes_used  # already a set from your code
        
        # Collect all edges that need Ricci update:
        # 1. New edges
        new_edges = [(int(c[0]), int(c[1])) for c in candidates_select]
        
        # 2. Existing edges incident to touched nodes
        incident_edges = []
        for node in touched_nodes:
            for nbr in G.neighbors(node):  # G is the old graph (before add_edge)
                edge = tuple(sorted((node, nbr)))
                if edge not in new_edges:  # avoid double-counting new ones
                    incident_edges.append(edge)
        
        # Combine and dedupe
        edges_to_update = set(new_edges + incident_edges)

    return G_updated, candidates, candidates_select, edges_to_update



def calculate_ricci_delta(u, v, w_new, deg_u, deg_v, curv_u, curv_v, u_neigh, v_neigh, weight_u_neigh, weight_v_neigh, node_inv_sqrt_sum_u, node_inv_sqrt_sum_v, deg_u_neigh, deg_v_neigh, patch_curv, min_debt_thresh=0.2, skip_low_debt = True, use_normalized = True, node_weights = None):
    """
    Exact Weighted Ricci Delta + Neighborhood Debt Sensing.
    Calculates how much adding (u,v) heals the surrounding manifold patch.
    Supports both normalized (vertex weights=1, with mean node curvature) and non-normalized (vertex weights=strength, with sum node curvature) modes.
    """

    ## This check should already be done
    # if G.has_edge(u, v):
    #     return -float('inf'), 0
    

    local_debt = abs(min(0, patch_curv[np.argsort(patch_curv)[int(min_debt_thresh*len(patch_curv))]]))

    if skip_low_debt and local_debt <= 1e-5:
        return 0.0, local_debt
    
    if use_normalized:

        old_mean_u = curv_u # curvature[u]
        old_mean_v = curv_v # curvature[v]

        old_sum_ric_u = old_mean_u * deg_u
        old_sum_ric_v = old_mean_v * deg_v
        
        # 3. Ricci of the New Edge (u, v)
        sqrt_w_new = np.sqrt(w_new)
        # ric_uv_new = 2.0 - sqrt_w_new * (node_inv_sqrt_sum[u] + node_inv_sqrt_sum[v])
        ric_uv_new = 2.0 - sqrt_w_new * (node_inv_sqrt_sum_u + node_inv_sqrt_sum_v)

        # 4. Ripple Effect on Incident Edges
        # Compute delta_incident for u and v sides, plus delta_means for their nbrs
        delta_incident_u = 0.0
        delta_means_nbrs_u = 0.0

        use_loop = False
        if use_loop == True:
            for nbr in G.neighbors(u):
                w_un = G[u][nbr]['weight']
                delta_ric = - np.sqrt(w_un / w_new)
                delta_incident_u += delta_ric
                # nbr's deg unchanged
                deg_nbr = G.degree(nbr)
                delta_means_nbrs_u += delta_ric / deg_nbr if deg_nbr > 0 else 0
            
            delta_incident_v = 0.0
            delta_means_nbrs_v = 0.0
            for nbr in G.neighbors(v):
                w_vn = G[v][nbr]['weight']
                delta_ric = - np.sqrt(w_vn / w_new)
                delta_incident_v += delta_ric
                deg_nbr = G.degree(nbr)
                delta_means_nbrs_v += delta_ric / deg_nbr if deg_nbr > 0 else 0
        else:

            w_un = weight_u_neigh # G[u][nbr]['weight']
            delta_ric = - np.sqrt(w_un / w_new)
            delta_incident_u = delta_ric.sum()
            # deg_nbr = deg_u_neigh # degrees[u_neighs]
            delta_means_nbrs_u = ((deg_u_neigh > 0)*(delta_ric / np.maximum(1, deg_u_neigh))).sum() # if deg_nbr > 0 else 0


            w_vn = weight_v_neigh # G[u][nbr]['weight']
            delta_ric = - np.sqrt(w_vn / w_new)
            delta_incident_v = delta_ric.sum()
            # deg_nbr = deg_v_neigh # degrees[v_neighs]
            delta_means_nbrs_v = ((deg_v_neigh > 0)*(delta_ric / np.maximum(1, deg_v_neigh))).sum() # if deg_nbr > 0 else 0



        # Updates for u and v means (account for +deg and ric updates)
        new_sum_ric_u = old_sum_ric_u + ric_uv_new + delta_incident_u
        new_deg_u = deg_u + 1
        new_mean_u = new_sum_ric_u / new_deg_u if new_deg_u > 0 else 0
        delta_mean_u = new_mean_u - old_mean_u
        
        new_sum_ric_v = old_sum_ric_v + ric_uv_new + delta_incident_v
        new_deg_v = deg_v + 1
        new_mean_v = new_sum_ric_v / new_deg_v if new_deg_v > 0 else 0
        delta_mean_v = new_mean_v - old_mean_v
        
        # Total gain: sum of delta_means over all affected nodes (u, v, nbrs_u, nbrs_v)
        # Handles shared nbrs correctly (additive deltas if connected to both)
        total_ricci_gain = delta_mean_u + delta_mean_v + delta_means_nbrs_u + delta_means_nbrs_v
    

    else:

        # Non-normalized mode: vertex weights = strength (node_weights[n])
        # total_curvature assumed to be sum of incident Riccis (not mean)
        if node_weights is None:
            raise ValueError("node_weights required for non-normalized mode")
        
        w_u_old = node_weights[u]
        w_v_old = node_weights[v]
        w_u_next = w_u_old + w_new
        w_v_next = w_v_old + w_new
        

        inv_sum_u = node_inv_sqrt_sum_u
        inv_sum_v = node_inv_sqrt_sum_v
        
        # 3. Ricci of the new edge (u,v)
        # s_u = √(w_u_next / w_new) * sum_{nbr} 1/√w_{u-nbr}
        s_u_for_new = np.sqrt(w_u_next / w_new) * inv_sum_u
        s_v_for_new = np.sqrt(w_v_next / w_new) * inv_sum_v
        
        ric_uv_new = w_new * ((w_u_next / w_new) + (w_v_next / w_new) - s_u_for_new - s_v_for_new)
        
        # 4. Ripple effect on all existing incident edges
        delta_incident = 0.0
        
        for center, w_old, w_next, inv_sum in [(u, w_u_old, w_u_next, inv_sum_u),
                                               (v, w_v_old, w_v_next, inv_sum_v)]:
            sqrt_w_old = np.sqrt(w_old)
            sqrt_w_next = np.sqrt(w_next)
            
            for nbr in G.neighbors(center):
                w_ce = G[center][nbr]['weight']          # weight of existing edge center--nbr
                inv_sqrt_w_ce = 1.0 / np.sqrt(w_ce)
                
                # Old contribution of this parallel edge to the sum at center
                contrib_old = sqrt_w_old * inv_sqrt_w_ce
                
                # New contribution after strength increases and new parallel edge added
                contrib_new = sqrt_w_next * inv_sqrt_w_ce
                
                # The sum term also gains the new parallel edge contribution: + 1/√w_new
                # So total new sum multiplier effect on this edge's Ricci
                s_old = contrib_old * (inv_sum - inv_sqrt_w_ce)   # excludes self in original sum
                s_new = contrib_new * (inv_sum - inv_sqrt_w_ce + 1.0 / np.sqrt(w_new))
                
                # Ricci change for this existing edge
                delta_ric = w_ce * ((w_next - w_old) / w_ce - (s_new - s_old))
                delta_incident += delta_ric
        
        # Each Ricci (new + changes) contributes to two nodes' total_curvature (sum)
        total_ricci_gain = 2 * (ric_uv_new + delta_incident)

    
    return total_ricci_gain, local_debt


def calculate_weighted_smoothness_penalty(u, v, u_neighs, v_neighs, w_new, node_weights):
    W_u_next = node_weights[u] + w_new
    W_v_next = node_weights[v] + w_new
    
    # u side
    # u_neighs = list(G.neighbors(u))
    sum_W_u_neighs = sum(node_weights[n] for n in u_neighs) + W_v_next
    mean_W_u_neigh = sum_W_u_neighs / (len(u_neighs) + 1)
    rel_dev_u = abs(W_u_next - mean_W_u_neigh) / (mean_W_u_neigh + 1e-6)  # relative deviation
    
    # v side
    # v_neighs = list(G.neighbors(v))
    sum_W_v_neighs = sum(node_weights[n] for n in v_neighs) + W_u_next
    mean_W_v_neigh = sum_W_v_neighs / (len(v_neighs) + 1)
    rel_dev_v = abs(W_v_next - mean_W_v_neigh) / (mean_W_v_neigh + 1e-6)
    
    total_rel_gradient = rel_dev_u + rel_dev_v
    return np.log(1.0 + total_rel_gradient)  # now bounded and log-scaled


def precompute_large_graph(G, weights, node_inv_sqrt_sum, curvature, degrees):
    """
    Precompute fixed data for large graphs (n=5k-10k). Call once per G or when structure changes.
    Assumes nodes are integers 0 to n-1, weights is n x n array (dense or sparse-capable via indexing).
    """
    n = len(G.nodes())
    # Existing edges for quick check (tuples of (min(u,v), max(u,v)))
    existing_edges = set((min(u, v), max(u, v)) for u, v in G.edges())

    # Neighbor lists, sorted for fast intersections
    neigh_lists = [np.sort(list(G.neighbors(i))) for i in range(n)]

    # node_strength = weighted degrees (sum of weights to neighbors)
    node_strength = np.array([np.sum(weights[i, neigh_lists[i]]) for i in range(n)])

    # sum_sqrt_w[i] = sum_{nbr} sqrt(weights[i, nbr])
    sum_sqrt_w = np.zeros(n)
    for i in range(n):
        nbrs = neigh_lists[i]
        sum_sqrt_w[i] = np.sum(np.sqrt(weights[i, nbrs]))

    # weighted_sum_inv_deg[i] = sum_{nbr} sqrt(weights[i, nbr]) / max(degrees[nbr], 1)
    weighted_sum_inv_deg = np.zeros(n)
    for i in range(n):
        nbrs = neigh_lists[i]
        weighted_sum_inv_deg[i] = np.sum(np.sqrt(weights[i, nbrs]) / np.maximum(degrees[nbrs], 1))

    # sum_nbr_strength[i] = sum_{nbr} node_strength[nbr]
    sum_nbr_strength = np.zeros(n)
    for i in range(n):
        nbrs = neigh_lists[i]
        sum_nbr_strength[i] = np.sum(node_strength[nbrs])

    return {
        'existing_edges': existing_edges,
        'neigh_lists': neigh_lists,
        'sum_sqrt_w': sum_sqrt_w,
        'weighted_sum_inv_deg': weighted_sum_inv_deg,
        'sum_nbr_strength': sum_nbr_strength,
        'node_strength': node_strength
    }


def optimized_normalized_scores_large(
    G,
    candidates,  # np.array (N, 5+): [u, v, w_new, prior_score, ...] — uses [:,0:4]
    alpha_scale,
    beta_scale,
    gamma_penalty,
    tau_scale,
    use_triangles=False,
    use_forman=True,
    min_debt_thresh=0.2,
    skip_low_debt=True,
    weights=None,  # n x n array for edge weights
    curvature=None,  # np.array (n,) node curvatures
    degrees=None,  # np.array (n,) node degrees
    node_inv_sqrt_sum=None,  # np.array (n,) precomputed inv sqrt sums
    precomputed_data=None  # Dict from precompute_large_graph
):
    """
    Fully optimized NumPy-vectorized scoring for large N and graphs.
    Ricci/smoothness: fully vectorized with precomputes.
    Local debt: looped but optimized with np.unique + argsort.
    Triangles: looped with searchsorted on sorted lists.
    Assumes nodes 0 to n-1; update curvature/degrees/fiedler if they change.
    """
    if precomputed_data is None:
        precomputed_data = precompute_large_graph(G, weights, node_inv_sqrt_sum, curvature, degrees)

    n = len(curvature)
    existing_edges = precomputed_data['existing_edges']
    neigh_lists = precomputed_data['neigh_lists']
    sum_sqrt_w = precomputed_data['sum_sqrt_w']
    weighted_sum_inv_deg = precomputed_data['weighted_sum_inv_deg']
    sum_nbr_strength = precomputed_data['sum_nbr_strength']
    node_strength = precomputed_data['node_strength']

    us = candidates[:, 0].astype(int)
    vs = candidates[:, 1].astype(int)

    # Safety: skip any candidates with invalid node indices
    valid_nodes = (us >= 0) & (us < n) & (vs >= 0) & (vs < n)
    if not np.all(valid_nodes):
        print(f"Warning: {np.sum(~valid_nodes)} candidates have invalid node indices")
        # You can filter or set priors to -inf, but for now just proceed

    w_news = candidates[:, 2]  # Original for checks
    priors = candidates[:, 3]
    N = len(candidates)

    # Masks
    existing = np.zeros(N, dtype=bool)
    for i in range(N):
        u, v = us[i], vs[i]
        existing[i] = (w_news[i] <= 0) or ((min(u, v), max(u, v)) in existing_edges)

    w_news = np.maximum(w_news, 1e-10)  # Clamp for computations

    # Vectorized Ricci delta
    sqrt_w_news = np.sqrt(w_news)
    inv_sqrt_w_news = 1.0 / sqrt_w_news

    ric_uv_news = 2.0 - sqrt_w_news * (node_inv_sqrt_sum[us] + node_inv_sqrt_sum[vs])

    delta_incident_us = -sum_sqrt_w[us] * inv_sqrt_w_news
    delta_incident_vs = -sum_sqrt_w[vs] * inv_sqrt_w_news

    delta_means_nbrs_us = -weighted_sum_inv_deg[us] * inv_sqrt_w_news
    delta_means_nbrs_vs = -weighted_sum_inv_deg[vs] * inv_sqrt_w_news

    old_sum_ric_us = curvature[us] * degrees[us]
    old_sum_ric_vs = curvature[vs] * degrees[vs]

    new_sum_ric_us = old_sum_ric_us + ric_uv_news + delta_incident_us
    new_sum_ric_vs = old_sum_ric_vs + ric_uv_news + delta_incident_vs

    new_deg_us = degrees[us] + 1
    new_deg_vs = degrees[vs] + 1

    new_mean_us = new_sum_ric_us / new_deg_us
    new_mean_vs = new_sum_ric_vs / new_deg_vs

    delta_mean_us = new_mean_us - curvature[us]
    delta_mean_vs = new_mean_vs - curvature[vs]

    total_ricci_gains = delta_mean_us + delta_mean_vs + delta_means_nbrs_us + delta_means_nbrs_vs
    total_ricci_gains[existing] = -np.inf

    # Optimized local_debt loop (np.unique for union, argsort for quantile)
    local_debts = np.zeros(N)
    for i in range(N):
        if existing[i]:
            continue
        u, v = us[i], vs[i]
        u_neigh = neigh_lists[u]
        v_neigh = neigh_lists[v]
        concat = np.concatenate(([u, v], u_neigh, v_neigh))
        patch = np.unique(concat)  # Fast union
        if len(patch) == 0:
            continue
        patch_curv = curvature[patch]
        sorted_curv = np.sort(patch_curv)
        idx = int(min_debt_thresh * len(sorted_curv))
        idx = min(idx, len(sorted_curv) - 1)
        q = sorted_curv[idx]
        local_debts[i] = np.abs(min(q, 0.0))

    if skip_low_debt:
        low_mask = local_debts <= 1e-5
        total_ricci_gains[low_mask] = 0.0
    local_debts[existing] = 0.0

    # Geo log
    ricci_scaleds = local_debts * total_ricci_gains
    pos_ricci = np.maximum(ricci_scaleds, 0)
    geo_logs = alpha_scale * np.log(1.0 + pos_ricci)
    neg_mask = total_ricci_gains < 0
    geo_logs[neg_mask] -= gamma_penalty * np.log(1.0 + np.abs(total_ricci_gains[neg_mask]))


    if use_triangles:
        triangle_bonuses = np.zeros(N)
        for i in range(N):
            if existing[i] or w_news[i] < 0:
                continue
            u, v = us[i], vs[i]
            u_neigh = neigh_lists[u]  # already sorted np.array
            v_neigh = neigh_lists[v]  # sorted np.array
            
            if len(u_neigh) == 0 or len(v_neigh) == 0:
                continue
            
            # Safe searchsorted intersection
            left = np.searchsorted(u_neigh, v_neigh)
            valid = left < len(u_neigh)
            matches = u_neigh[left[valid]] == v_neigh[valid]
            common = v_neigh[valid][matches]
            
            if len(common) == 0:
                continue
            
            w_new_val = w_news[i]
            w_uc = weights[u, common]
            w_vc = weights[v, common]
            
            if use_forman:
                triangle_bonuses[i] = np.sum((w_new_val * w_uc * w_vc) ** (1/3))
            else:
                triangle_bonuses[i] = np.sqrt(w_new_val) * np.sum(np.sqrt(w_uc * w_vc))
        
        geo_logs += tau_scale * np.log(1.0 + triangle_bonuses)


    # Vectorized smoothness
    W_u_nexts = node_strength[us] + w_news
    W_v_nexts = node_strength[vs] + w_news
    sum_W_u_neighs = sum_nbr_strength[us] + W_v_nexts
    sum_W_v_neighs = sum_nbr_strength[vs] + W_u_nexts
    mean_W_u_neigh = sum_W_u_neighs / (degrees[us] + 1 + 1e-6)
    mean_W_v_neigh = sum_W_v_neighs / (degrees[vs] + 1 + 1e-6)
    rel_dev_us = np.abs(W_u_nexts - mean_W_u_neigh) / (mean_W_u_neigh + 1e-6)
    rel_dev_vs = np.abs(W_v_nexts - mean_W_v_neigh) / (mean_W_v_neigh + 1e-6)
    total_rel_gradients = rel_dev_us + rel_dev_vs
    smooth_logs = beta_scale * np.log(1.0 + total_rel_gradients)

    # Final scores
    score_approx = priors + geo_logs - smooth_logs
    return score_approx


def fit_domain_model(dir_ext = 'Domains', n_ver_load = 1, n_save_ver = 1):

    import numpy as np
    import torch
    import glob
    from torch import nn, optim

    st = glob.glob('%s/*parameters*ver_%d.npz'%(dir_ext, n_ver_load))
    st = np.random.permutation(st)

    Inpts = []
    Trgts = []

    # k_inpts = ['lat_range', 'lon_range', 'lat_range_extend', 'lon_range_extend', 'deg_padding']
    # k_trgts = ['scale_time', 'depth_boost', 'time_shift_range', 'buffer_scale', 'source_label_width', 'source_label_width_t', 'sigma_input']

    k_inpts = ['lat_range', 'lon_range']
    k_trgts = ['scale_time', 'depth_boost', 'time_shift_range', 'buffer_scale', 'source_label_width', 'source_label_width_t', 'sigma_input', 'deg_padding']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for s in st:
        z = np.load(s)
        inpt = np.hstack([np.diff(z[k]) if '_range' in k else z[k] for k in k_inpts]).reshape(1,-1)
        inpt = np.concatenate((inpt, np.array([len(z['locs_use'])]).reshape(1,1)), axis = 1)
        trgt = np.array([z[k] for k in k_trgts]).reshape(1,-1)
        Inpts.append(inpt)
        Trgts.append(trgt)
        z.close()
        print('Loaded %s'%s)

    Inpts = np.vstack(Inpts)
    Trgts = np.vstack(Trgts)
    n_size = Inpts.shape[0]

    scale_inpt = Inpts.max(0, keepdims = True) - Inpts.min(0, keepdims = True)
    offset_inpt = Inpts.min(0, keepdims = True)

    scale_trgt = Trgts.max(0, keepdims = True) - Trgts.min(0, keepdims = True)
    offset_trgt = Trgts.min(0, keepdims = True)

    m = nn.Sequential(nn.Linear(Inpts.shape[1], 30), nn.ReLU(), nn.Linear(30, 30), nn.ReLU(), nn.Linear(30, Trgts.shape[1])).to(device)

    optimizer = optim.Adam(m.parameters(), lr = 0.001)

    loss_func = nn.MSELoss()

    n_batch = 300
    n_epochs = 10000
    n_vald = 0.1

    losses = []
    losses_vald = []

    for i in range(n_epochs):

        optimizer.zero_grad()
        i0 = np.sort(np.random.choice(int(n_size*(1.0 - n_vald)), size = n_batch, replace = False))
        inpt_slice = torch.Tensor((Inpts[i0] - offset_inpt)/scale_inpt).to(device)
        trgt_slice = torch.Tensor((Trgts[i0] - offset_trgt)/scale_trgt).to(device)
        out = m(inpt_slice)
        loss = loss_func(out, trgt_slice)
        loss.backward()
        optimizer.step()

        if np.mod(i, 10) == 0:
            with torch.no_grad():
                i0 = np.sort(np.random.choice(np.arange(int(n_size*(1.0 - n_vald)), n_size), size = n_batch))
                inpt_slice = torch.Tensor((Inpts[i0] - offset_inpt)/scale_inpt).to(device)
                trgt_slice = torch.Tensor((Trgts[i0] - offset_trgt)/scale_trgt).to(device)
                out = m(inpt_slice)
                loss_vald = loss_func(out, trgt_slice)

            print('%d %0.8f (%0.8f)'%(i, loss.item(), loss_vald.item()))

        else:

            print('%d %0.8f'%(i, loss.item()))

    ## Compute residuals

    Res = []
    Res_vald = []

    with torch.no_grad():
        for i in range(30):
            i0 = np.sort(np.random.choice(int(n_size*(1.0 - n_vald)), size = n_batch, replace = False))
            inpt_slice = torch.Tensor((Inpts[i0] - offset_inpt)/scale_inpt).to(device)
            # trgt_slice = torch.Tensor((Trgts[i0] - offset_trgt)/scale_trgt).to(device)
            trgt_slice = Trgts[i0] #  - offset_trgt)/scale_trgt).to(device)
            out = scale_trgt*m(inpt_slice).cpu().detach().numpy() + offset_trgt
            Res.append(out - trgt_slice)

            i0 = np.sort(np.random.choice(np.arange(int(n_size*(1.0 - n_vald)), n_size), size = n_batch, replace = False))
            inpt_slice = torch.Tensor((Inpts[i0] - offset_inpt)/scale_inpt).to(device)
            # trgt_slice = torch.Tensor((Trgts[i0] - offset_trgt)/scale_trgt).to(device)
            trgt_slice = Trgts[i0] #  - offset_trgt)/scale_trgt).to(device)
            out = scale_trgt*m(inpt_slice).cpu().detach().numpy() + offset_trgt
            Res_vald.append(out - trgt_slice)


    Res = np.vstack(Res)
    Res_vald = np.vstack(Res_vald)

    print('\n')
    print(list(Res.mean(0)))
    print(list(Res.std(0)))

    print('\n')
    print(list(Res_vald.mean(0)))
    print(list(Res_vald.std(0)))

    m.register_buffer('scale_trgt', torch.tensor(scale_trgt, device = device))
    m.register_buffer('offset_trgt', torch.tensor(offset_trgt, device = device))

    m.register_buffer('scale_inpt', torch.tensor(scale_inpt, device = device))
    m.register_buffer('offset_inpt', torch.tensor(offset_inpt, device = device))


    def strings_to_tensor(string_list):
        # Join strings with a null character and convert to bytes
        combined = "\0".join(string_list)
        return torch.ByteTensor(list(combined.encode('utf-8')))

    def tensor_to_strings(tensor):
        # Convert bytes back to string and split
        return bytes(tensor.tolist()).decode('utf-8').split("\0")

    m.register_buffer('k_inpts', strings_to_tensor(k_inpts))
    m.register_buffer('k_trgts', strings_to_tensor(k_trgts))

    torch.save(m.state_dict(), 'model_scale_parameters_ver_%d.h5'%(n_save_ver)) # trained_gnn_model_step_%d_ver_%d.h5


def load_model_domain(n_ver, device = 'cpu'):

    ## Load domain model
    state_dict = torch.load('model_scale_parameters_ver_%d.h5'%n_ver, map_location = device)
    in_dims = state_dict['0.weight'].shape[1]   # The 2nd dimension of the first layer
    out_dims = state_dict['4.weight'].shape[0]  # The 1st dimension of the last layer
    m_domain = nn.Sequential(nn.Linear(in_dims, 30), nn.ReLU(), nn.Linear(30, 30), nn.ReLU(), nn.Linear(30, out_dims)).to(device)
    # 2. Identify which keys are in the file but NOT in your model
    model_keys = set(m_domain.state_dict().keys())
    saved_keys = set(state_dict.keys())
    extra_keys = saved_keys - model_keys
    # 3. Automatically register those extra keys as buffers
    for key in extra_keys:
        # Get the data from the saved file to know the shape/type
        data = state_dict[key]
        m_domain.register_buffer(key, torch.zeros_like(data))
    # 4. Now you can load safely
    m_domain.load_state_dict(state_dict)
    k_inpts_model = tensor_to_strings(m_domain.k_inpts)
    k_trgts_model = tensor_to_strings(m_domain.k_trgts)
    # inpt = np.hstack([np.diff(z[k]) if '_range' in k else z[k] for k in k_inpts]).reshape(1,-1)

    print('Domain inputs are:')
    print(k_inpts_model)

    return m_domain


def build_sampling_grid(lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, time_shift_range, scale_time, number_of_spatial_nodes, ftrns1, ftrns2, buffer_scale = 2.0, depth_upscale_factor = 2.0, use_global = False, rbest = None, mn = None, verbose = True, device = 'cpu'):

    earth_radius = 6378137.0
    ftrns1_abs = lambda x: lla2ecef(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((lla2ecef(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
    ftrns2_abs = lambda x: ecef2lla(x, a = earth_radius) if x.shape[1] == 3 else np.concatenate((ecef2lla(x, a = earth_radius), x[:,3].reshape(-1,1)), axis = 1) # invert ftrns1

    def optimize_r_min(lat_vals, lon_mean = np.mean(lon_range), h_min = depth_range[0]):
        # r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
        r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
        r_val = r_surface + h_min
        return r_val


    def optimize_r_max(lat_vals, lon_mean = np.mean(lon_range), h_max = depth_range[1]):
        # r_surface = np.linalg.norm(ftrns1(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
        r_surface = np.linalg.norm(ftrns1_abs(np.concatenate((lat_vals.reshape(-1,1), lon_mean*np.ones((len(lat_vals),1)), np.zeros((len(lat_vals),1))), axis = 1)), axis = 1)
        r_val = r_surface + h_max
        return -r_val



    bounds = [(lat_range_extend[0], lat_range_extend[1])]
    soln = differential_evolution(optimize_r_min, bounds, popsize = 50, maxiter = 1000, disp = True if verbose == True else False)
    r_min = optimize_r_min(np.array([soln.x]))[0]; print('\n')

    bounds = [(lat_range_extend[0], lat_range_extend[1])]
    soln = differential_evolution(optimize_r_max, bounds, popsize = 50, maxiter = 1000, disp = True if verbose == True else False)
    r_max = -1.0*optimize_r_max(np.array([soln.x]))[0]; print('\n')
    assert(r_max >= r_min)


    ## Sample grid:
    use_time_shift = True
    use_station_density = False
    # number_of_spatial_nodes = n_grid
    if verbose == True: print('Beginning FPS sampling [%d]'%0)
    up_sample_factor = 10 if use_time_shift == False else 20 ## Could reduce to just 10 most likely
    # if use_station_density == True: up_sample_factor = up_sample_factor*5
    number_candidate_nodes = up_sample_factor*number_of_spatial_nodes
    trial_points, mask_points = regular_sobolov(number_candidate_nodes, lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = number_of_spatial_nodes, buffer_scale = buffer_scale, r_min = r_min, r_max = r_max) # lat_range = lat_range_extend, lon_range = lon_range_extend, depth_range = depth_range, time_range = time_shift_range, use_time = use_time_shift, use_global = use_global, scale_time = scale_time, N_target = None, buffer_scale = 0.0
    # x_grid = farthest_point_sampling(ftrns1_abs(trial_points), number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = mask_points)
    x_grid = farthest_point_sampling(trial_points, number_of_spatial_nodes, scale_time = scale_time, depth_boost = depth_upscale_factor, mask_candidates = mask_points)


    if verbose == True:

        ## Compute grid health
        metrics = compute_warped_expected_spacing(
            number_of_spatial_nodes, 
            lat_range=lat_range_extend, 
            lon_range=lon_range_extend,
            depth_range=depth_range, 
            time_range=time_shift_range,
            scale_time=scale_time, 
            depth_boost=1.0, ## Should use depth boost (check Hawaii case)
            use_global=use_global,
            r_min = r_min,
            r_max = r_max
        )

        # Unpack using your exact variable names
        Volume, Volume_space, Area, nominal_spacing, nominal_spacing_space, nominal_spacing_time = metrics

        compute_final_grid_health(x_grid, scale_time, depth_upscale_factor, lat_range_extend, lon_range_extend, depth_range, time_shift_range, buffer_scale, Volume)

    return x_grid


def estimate_kernel_widths(domain, station_locs, z_range = (-40000, 2000), Vs = 3500.0, noise_level = 0.02, n_srcs = 250, n_test_per_src = 10000, n_neighbors_trgt = 20):
    """
    Computes Gaussian coherency widths (W_phys, W_t) using a vectorized batch approach.
    Adaptive for any scale (Borehole to Global) with zero hard-coded temporal floors.
    
    Inputs:
        domain: dict with 'lat_range', 'lon_range', 'is_wrapped'
        station_locs: (n_stas, 3) LLA array
        Vs: Reference velocity (m/s)
        noise_level: Pick uncertainty (e.g., 0.02 for 2%)
    """
    # 1. Coordinate & Aperture Setup
    lat_r, lon_r = domain['lat_range'], domain['lon_range']
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Calculate representative aperture for sanity check (Corner-to-Corner ECEF)
    c1 = lla2ecef(np.array([[lat_r[0], lon_r[0], 0]]))
    c2 = lla2ecef(np.array([[lat_r[1], lon_r[1], 0]]))
    aperture_m = np.linalg.norm(c1 - c2)

    # 2. Sample Reference Sources (S, 3)
    if domain['is_wrapped']:
        width = (lon_r[1] + 360) - lon_r[0]
        lons = np.mod(np.random.uniform(lon_r[0], lon_r[0] + width, n_srcs), 360)
        lons = ((lons + 180) % 360) - 180
    else:
        lons = np.random.uniform(lon_r[0], lon_r[1], n_srcs)
        
    lats = np.random.uniform(lat_r[0], lat_r[1], n_srcs)
    zs = np.random.uniform(z_range[0], z_range[1], n_srcs)
    src_refs_lla = np.stack([lats, lons, zs], axis=1)

    st_ecef = lla2ecef(station_locs)
    ref_ecef = lla2ecef(src_refs_lla)

    # 3. Calculate Adaptive Search Limits per Source
    all_dists = cdist(ref_ecef, st_ecef) # (S, N_stas)
    nearest_idx = np.argsort(all_dists, axis=1)[:, :n_neighbors_trgt] # (S, 20)
    
    local_scales = np.zeros(n_srcs)
    time_limits = np.zeros(n_srcs)
    
    for s in range(n_srcs):
        s_dists = all_dists[s, nearest_idx[s]]
        # Spatial: Characteristic distance to the 5th neighbor
        local_scales[s] = s_dists[min(4, len(s_dists)-1)]
        
        # Temporal: Moveout across the local 20-station cluster
        moveout = (s_dists[-1] - s_dists[0]) / Vs
        
        # Noise Scale: 5x the expected pick jitter
        sigma_t_expected = np.mean(s_dists / Vs) * noise_level
        
        # Adaptive Limit: Searches enough for moveout OR noise smear
        time_limits[s] = max(moveout * 0.5, sigma_t_expected * 5.0)

    # 4. Generate Batched Perturbations
    # space_offs: (S, T, 3), time_offs: (S, T)
    space_offs = np.random.uniform(-1, 1, (n_srcs, n_test_per_src, 3)) * local_scales[:, None, None]
    time_offs = np.random.uniform(-1, 1, (n_srcs, n_test_per_src)) * time_limits[:, None]
    
    # Flatten test points for travel time call
    test_ecef_flat = (ref_ecef[:, None, :] + space_offs).reshape(-1, 3)
    test_lla_flat = ecef2lla(test_ecef_flat)

    # 5. Execute Vectorized trv
    t_obs_list = []
    t_test_list = []
    
    for s in range(n_srcs):
        s_stas = station_locs[nearest_idx[s]]
        # Observed (Reference) travel times: (1, 20, P)
        t_r = trv(torch.Tensor(s_stas).to(device), torch.Tensor(src_refs_lla[s:s+1]).to(device)).cpu().detach().numpy() # Vs = Vs
        # Add pick noise to simulate real-world uncertainty
        t_obs_list.append(t_r + np.random.normal(0, t_r * noise_level))
        
        # Test (Perturbed) travel times: (T, 20, P)
        t_t = trv(torch.Tensor(s_stas).to(device), torch.Tensor(test_lla_flat[s*n_test_per_src : (s+1)*n_test_per_src]).to(device)).cpu().detach().numpy() # Vs = Vs
        t_test_list.append(t_t)

    t_obs = np.concatenate(t_obs_list, axis=0) # (S, 20, P)
    t_test = np.stack(t_test_list, axis=0)    # (S, T, 20, P)

    # 6. Vectorized Chi-Square Misfit
    # Residuals: (S, T, 20, P) after origin-time shift (time_offs)
    residuals = t_obs[:, None, :, :] - (t_test + time_offs[:, :, None, None])
    sigma_d = np.maximum(t_test * noise_level, 1e-6)
    
    # RMS of normalized error across the 20-station neighborhood
    chi_error = np.sqrt(np.mean((residuals / sigma_d)**2, axis=(2, 3))) # (S, T)

    # 7. Extract Widths from the Likelihood Slope
    dist_s = np.linalg.norm(space_offs, axis=2) # (S, T)
    dist_t = np.abs(time_offs) # (S, T)

    # Use a mask to focus on points where the misfit escapes the noise floor
    mask = (chi_error > 0.1) & (chi_error < 5.0)
    w_phys = np.median(dist_s[mask] / chi_error[mask])
    w_t = np.median(dist_t[mask] / chi_error[mask])

    # --- FINAL SANITY CHECK & LOGGING ---
    rel_scale = w_phys / aperture_m
    print(f"\n" + "="*55)
    print(f"BATCHED ADAPTIVE COHERENCY ESTIMATION")
    print(f"="*55)
    print(f"Total Points Sampled:   {n_srcs * n_test_per_src}")
    print(f"Spatial Width (W_phys): {w_phys/1000:.4f} km")
    print(f"Temporal Width (W_t):   {w_t:.4f} s")
    print(f"Computed Aperture:      {aperture_m/1000:.2f} km")
    print(f"Relative Resolution:    {rel_scale:.4%}")
    print(f"Noise Level Used:       {noise_level*100:.1f}%")
    
    if rel_scale > 0.4:
        print("!! WARNING: Broad coherency. Network may be critically sparse.")
    elif rel_scale < 0.001:
        print(">> NOTE: Highly localized resolution. Consider high-res voxels.")
    print("="*55 + "\n")

    return {
        "W_phys_m": w_phys,
        "W_t_s": w_t,
        "rel_scale": rel_scale,
        "metadata": {
            "n_srcs": n_srcs,
            "n_test_per_src": n_test_per_src,
            "Vs": Vs
        }
    }



def probe_network_sidelobes_geodetic(station_latlonz, domain_lat_range, domain_lon_range, domain_depth_range,
                                     k_stations=20, vel_avg=3500.0, vel_min=2500.0,
                                     scan_step_m=1000.0, W_phys_m = 1000.0, W_t = 3.0, device=device):
    
    # --- 1. Project Stations to 3D for Aperture Check ---
    # station_latlonz: [Lat, Lon, Depth]
    station_xyz = torch.tensor(ftrns1(station_latlonz), device = device) # torch.cat((station_latlonz[:, 0].reshape(-1,1), station_latlonz[:, 1], station_latlonz[:, 2])
    
    dist_matrix = torch.cdist(station_xyz, station_xyz)
    d_array = torch.max(dist_matrix).item()
    
    # Limits in meters
    max_radius_m = d_array / 2.0
    max_dt = d_array / vel_min
    # dynamic_time_window = d_array / vel_min

    # --- 2. Pick Random Source (The "Truth") ---
    src_lat_c = np.random.uniform(*domain_lat_range)
    src_lon_c = np.random.uniform(*domain_lon_range)
    src_z_c = np.random.uniform(*domain_depth_range)
    
    src_true = np.array([src_lat_c, src_lon_c, src_z_c]).reshape(1,-1)
    src_true_xyz = torch.tensor(ftrns1(src_true), device = device) # , 
                                # torch.tensor([src_lon_c]), 
                                # torch.tensor([src_z_c])), dim = 1).cpu().detach().numpy()), device = device) # ).to(device)

    # --- 3. Convert Meter-Offsets to Lat/Lon Degrees ---
    # We use the conversion factors at the source's specific latitude
    lat_rad = np.radians(src_lat_c)
    m_per_deg_lat = 111132.0 
    m_per_deg_lon = 111320.0 * np.cos(lat_rad)
    
    # Create degree-based offsets
    lat_offsets = torch.arange(-max_radius_m, max_radius_m + scan_step_m, scan_step_m, device=device) / m_per_deg_lat
    lon_offsets = torch.arange(-max_radius_m, max_radius_m + scan_step_m, scan_step_m, device=device) / m_per_deg_lon
    
    grid_lat = lat_offsets + src_lat_c
    grid_lon = lon_offsets + src_lon_c
    grid_z = torch.linspace(domain_depth_range[0], domain_depth_range[1], 5, device=device)

    # Create the Lat/Lon/Depth mesh
    mesh_lat, mesh_lon, mesh_z = torch.meshgrid(grid_lat, grid_lon, grid_z, indexing='ij')
    
    # --- 4. PROJECT THE GRID TO 3D XYZ ---
    # This ensures the grid "hugs" the ellipsoid
    grid_xyz = torch.tensor(ftrns1(torch.cat((mesh_lat.reshape(-1,1), mesh_lon.reshape(-1,1), mesh_z.reshape(-1,1)), dim = 1).cpu().detach().numpy()), device = device)

    # --- 5. Vectorized Back-Projection ---
    # Get K-nearest stations (indices from station_xyz vs src_true_xyz)
    all_dists = torch.cdist(src_true_xyz, station_xyz).squeeze(0)
    _, sta_idx = torch.topk(all_dists, k=k_stations, largest=False)
    
    active_stas_xyz = station_xyz[sta_idx]
    t_obs = (torch.norm(active_stas_xyz - src_true_xyz, dim=1)) / vel_avg
    t_obs_picks = np.concatenate((t_obs.cpu().detach().numpy().reshape(-1,1), sta_idx.cpu().detach().numpy().reshape(-1,1)), axis = 1)

    t_calc = torch.cdist(grid_xyz, active_stas_xyz) / vel_avg
    residuals = t_obs.unsqueeze(0) - t_calc
    

    w_t_m = W_t * vel_avg 

    # The 4D suppression radius is the hypotenuse of the physical 'bowl' widths
    # We use a 1.5x multiplier to ensure we clear the 'shoulders' of the main peak
    suppress_r_4d = torch.sqrt(torch.tensor(W_phys_m**2 + w_t_m**2)) * 1.5


    time_step = max(0.01, max_dt / 100.0)
    dt_range = torch.arange(-max_dt, max_dt + time_step, time_step, device=device, dtype=torch.float32)
    
    max_coherence = torch.zeros(grid_xyz.shape[0], device=device, dtype=torch.float32)
    best_dt = torch.zeros(grid_xyz.shape[0], device=device, dtype=torch.float32)

    for dt in dt_range:
        # Note: 'residuals' must be float32 to avoid dtype mismatch
        mismatch = torch.abs(residuals - dt).to(torch.float32)
        coherence = torch.exp(-mismatch / 0.1).mean(dim=1)
        
        mask = coherence > max_coherence
        max_coherence[mask] = coherence[mask]
        best_dt[mask] = dt


    # --- 6. 4D Non-Maximum Suppression (Space + Time) ---
    indices = torch.argsort(max_coherence, descending=True)
    peaks = []

    for idx in indices:
        peak_pos = grid_xyz[idx]
        peak_dt = best_dt[idx]
        peak_val = max_coherence[idx].item()
        
        # Calculate 4D distance from TRUE SOURCE [src_true_xyz, 0]
        d_space_true = torch.norm(peak_pos - src_true_xyz)
        d_time_true = torch.abs(peak_dt - 0.0) * vel_avg
        dist_4d_from_true = torch.sqrt(d_space_true**2 + d_time_true**2)
        
        # 1. SKIP IF IT'S PART OF THE MAIN SOURCE PEAK
        if dist_4d_from_true < suppress_r_4d:
            continue 
            
        # 2. SKIP IF IT'S PART OF AN ALREADY IDENTIFIED SIDELOBE
        is_duplicate = False
        for p in peaks:
            d_s = torch.norm(peak_pos - p['pos'])
            d_t = torch.abs(peak_dt - p['dt_offset']) * vel_avg
            # Euclidean check in the isotropized 4D volume
            if torch.sqrt(d_s**2 + d_t**2) < suppress_r_4d:
                is_duplicate = True
                break
                
        if not is_duplicate:
            # Convert to Lat/Lon/Depth
            pos_latlonz = ftrns2(peak_pos.cpu().detach().numpy().reshape(1,-1)).reshape(-1)
            
            peaks.append({
                'pos': peak_pos, 
                'pos_src': pos_latlonz,
                'val': peak_val, 
                'dt_offset': peak_dt.item(),
                'dist_offset_m': d_space_true.item(),
                'dist_4d_m': dist_4d_from_true.item()
            })
        
        if len(peaks) >= 10: break


    return src_true, peaks, t_obs_picks, [max_radius_m, max_dt]


def strings_to_tensor(string_list):
    # Join strings with a null character and convert to bytes
    combined = "\0".join(string_list)
    return torch.ByteTensor(list(combined.encode('utf-8')))


def tensor_to_strings(tensor):
    # Convert bytes back to string and split
    return bytes(tensor.tolist()).decode('utf-8').split("\0")

    
