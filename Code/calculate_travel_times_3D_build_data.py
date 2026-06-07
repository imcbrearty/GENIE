import numpy as np
from scipy.io import loadmat
from runpy import run_path
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import multiprocessing
import skfmm
from scipy.interpolate import RegularGridInterpolator
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from numpy import interp
import shutil
import pathlib
# from module import TravelTimes
from torch.autograd import Variable
from sklearn.metrics import r2_score
from utils import *
from module import TravelTimesPN
from torch.autograd import Variable
from torch_geometric.utils import get_laplacian
from scipy.optimize import differential_evolution
# from scipy.metrics import pairwise_distances as pd
from sklearn.metrics import pairwise_distances as pd
from process_utils import * # differential_evolution_location
import platform
import psutil
import glob
import sys
import os

argvs = sys.argv
if len(argvs) == 1:
	argvs.append(0)

def compute_travel_times_parallel(xx, xx_r, h, h1, dx_v, x11, x12, x13, num_cores=10):


    print(xx.shape); print(x11.shape); print(x12.shape); print(x13.shape)

    # -------------------------------------------------------------
    # VARIANT A: OPTIMIZED LINUX SHAPE (Reads grids via scope closure)
    # -------------------------------------------------------------
    def step_test_linux(args):
        yval, dx_v, h, h1, ind = args
        
        phi_xy = (x11 - yval[0,0])**2 + (x12 - yval[0,1])**2
        phi_v = (x13 - yval[0,2])**2
        phi = np.sqrt(phi_xy + phi_v) # - phi.min()
        phi = phi - phi.min()
        assert((phi == 0).sum() == 1)

        v = np.copy(h).reshape(x11.shape)
        v1 = np.copy(h1).reshape(x11.shape)

        # fmm_dx = [
        #     float(np.abs(x12[1, 0, 0] - x12[0, 0, 0])),
        #     float(np.abs(x11[0, 1, 0] - x11[0, 0, 0])),
        #     float(np.abs(x13[0, 0, 1] - x13[0, 0, 0]))
        # ]

		# FIXED: Extract spacings directly from their corresponding arrays
        fmm_dx = [
            float(np.abs(x11[1, 0, 0] - x11[0, 0, 0])), # True dx (changes along axis 0)
            float(np.abs(x12[0, 1, 0] - x12[0, 0, 0])), # True dy (changes along axis 1)
            float(np.abs(x13[0, 0, 1] - x13[0, 0, 0]))  # True dz (changes along axis 2)
        ]

        t = skfmm.travel_time(phi, v, dx=fmm_dx)
        t1 = skfmm.travel_time(phi, v1, dx=fmm_dx)
        return t, t1, phi, ind

    # -------------------------------------------------------------
    # VARIANT B: COMPATIBLE WINDOWS SHAPE (Unpacks grids explicitly)
    # -------------------------------------------------------------
    def step_test_windows(args):
        yval, dx_v, h, h1, x11_local, x12_local, x13_local, ind = args
        
        phi_xy = (x11_local - yval[0,0])**2 + (x12_local - yval[0,1])**2
        phi_v = (x13_local - yval[0,2])**2
        phi = np.sqrt(phi_xy + phi_v) # - phi.min()
        phi = phi - phi.min()
        assert((phi == 0).sum() == 1)

        v = np.copy(h).reshape(x11_local.shape)
        v1 = np.copy(h1).reshape(x11_local.shape)

        # fmm_dx = [
        #     float(np.abs(x12_local[1, 0, 0] - x12_local[0, 0, 0])),
        #     float(np.abs(x11_local[0, 1, 0] - x11_local[0, 0, 0])),
        #     float(np.abs(x13_local[0, 0, 1] - x13_local[0, 0, 0]))
        # ]

        # # FIXED: Extract spacings directly from their corresponding arrays
        # fmm_dx = [
        #     float(np.abs(x11[1, 0, 0] - x11[0, 0, 0])), # True dx (changes along axis 0)
        #     float(np.abs(x12[0, 1, 0] - x12[0, 0, 0])), # True dy (changes along axis 1)
        #     float(np.abs(x13[0, 0, 1] - x13[0, 0, 0]))  # True dz (changes along axis 2)
        # ]

        # FIXED: Extract spacings from unpacked local grid copies on Windows
        fmm_dx = [
            float(np.abs(x11_local[1, 0, 0] - x11_local[0, 0, 0])), 
            float(np.abs(x12_local[0, 1, 0] - x12_local[0, 0, 0])), 
            float(np.abs(x13_local[0, 0, 1] - x13_local[0, 0, 0]))  
        ]

        t = skfmm.travel_time(phi, v, dx=fmm_dx)
        t1 = skfmm.travel_time(phi, v1, dx=fmm_dx)
        return t, t1, phi, ind

    # -------------------------------------------------------------
    # RUNTIME CHECK & EXECUTION DISPATCH
    # -------------------------------------------------------------
    is_windows = (platform.system().lower() == 'windows')

    tp_times = np.nan * np.zeros((h.shape[0], xx_r.shape[0]))
    ts_times = np.nan * np.zeros((h.shape[0], xx_r.shape[0]))

    if is_windows:
        # Windows needs explicit payload serialization due to its 'spawn' process behavior
        results = Parallel(n_jobs=num_cores)(
            delayed(step_test_windows)([xx_r[i,:][None,:], dx_v, h, h1, x11, x12, x13, i]) 
            for i in range(xx_r.shape[0])
        )
    else:
        # Linux / macOS uses optimized lightweight fork streams
        results = Parallel(n_jobs=num_cores)(
            delayed(step_test_linux)([xx_r[i,:][None,:], dx_v, h, h1, i]) 
            for i in range(xx_r.shape[0])
        )

    # Reconstruct outputs identically
    for i in range(xx_r.shape[0]):
        tp_times[:, results[i][-1]] = results[i][0].reshape(-1)
        ts_times[:, results[i][-1]] = results[i][1].reshape(-1)

    return tp_times, ts_times


def grid_loss_function(x, v_min, target_error, span_x, span_y, span_z, cpu_point_budget, C):
    """
    Optimizes a 3-tier multi-resolution grid architecture for Eikonal solvers.
    
    Parameters:
    -----------
    x : array-like
        The optimization variables [dx_1, dx_2, dx_3] representing cell sizes in meters.
    v_min : float
        Minimum wave velocity (m/s) in the domain.
    target_error : float
        Acceptable tracking error threshold (e.g., 0.02 for 2%).
    span_x, span_y, span_z : float
        True regional domain dimensions in meters (Cartesian coordinates).
    cpu_point_budget : int
        Maximum allowed grid points for a SINGLE tier (e.g., np.prod(n_optimal_points)).
    C : float
        Geometric error constant for the Fast Marching Method (typically 0.25).
    """
    dx_1, dx_2, dx_3 = x[0], x[1], x[2]
    
    # 1. Enforce strict grid resolution hierarchy
    if dx_1 >= dx_2 or dx_2 >= dx_3:
        return 1e12 

    # 2. Physics boundaries (Maximum accurate tracking radii based on Eikonal error scaling)
    R_max_1 = (C * (dx_2 ** 2)) / (target_error * v_min)
    R_max_2 = (C * (dx_3 ** 2)) / (target_error * v_min)

    # 3. Structural nested constraints 
    if R_max_1 >= R_max_2 or R_max_1 < 500.0:
        return 1e12  
        
    if R_max_2 <= R_max_1:
        return 1e12  

    # --- GEOGRAPHICALLY ACCURATE RECTANGULAR BOX POINT ESTIMATION ---
    pad_cells = 8

    # Tier 1 Box: Bounds dynamically adapt to wavefront radius horizontally AND vertically
    t1_span_x = min(2.0 * R_max_1, span_x)
    t1_span_y = min(2.0 * R_max_1, span_y)
    t1_span_z = min(2.0 * R_max_1, span_z) # Dynamic depth-cap removes the skyscraper anomaly
    points_1 = (int(np.ceil(t1_span_x / dx_1)) + pad_cells) * \
               (int(np.ceil(t1_span_y / dx_1)) + pad_cells) * \
               (int(np.ceil(t1_span_z / dx_1)) + pad_cells)

    # Tier 2 Box: Mid-range sub-grid
    t2_span_x = min(2.0 * R_max_2, span_x)
    t2_span_y = min(2.0 * R_max_2, span_y)
    t2_span_z = min(2.0 * R_max_2, span_z) # Dynamic depth-cap
    points_2 = (int(np.ceil(t2_span_x / dx_2)) + pad_cells) * \
               (int(np.ceil(t2_span_y / dx_2)) + pad_cells) * \
               (int(np.ceil(t2_span_z / dx_2)) + pad_cells)

    # Tier 3 Box: Natively covers the exact deep, non-square regional footprint
    points_3 = (int(np.ceil(span_x / dx_3)) + pad_cells) * \
               (int(np.ceil(span_y / dx_3)) + pad_cells) * \
               (int(np.ceil(span_z / dx_3)) + pad_cells)

    # 4. Check memory/CPU constraints INDEPENDENTLY per tier
    # Since grids run sequentially, we evaluate against the peak hardware bottleneck
    max_tier_points = max(points_1, points_2, points_3)
    if max_tier_points > cpu_point_budget:
        # Direct penalization scaling if any single layer exceeds the memory budget
        return (1e9 * (max_tier_points / cpu_point_budget)) 

    # 5. Multi-Objective (Pareto) Loss Optimization Loop
    # Normalize resolutions against the coarse step so they share a comparable loss scale
    resolution_loss = (dx_1 / dx_3) + (dx_2 / dx_3)
    
    # Calculate how well each tier maximizes its independent memory allocation pool
    util_1 = (cpu_point_budget - points_1) / cpu_point_budget
    util_2 = (cpu_point_budget - points_2) / cpu_point_budget
    util_3 = (cpu_point_budget - points_3) / cpu_point_budget
    avg_budget_utilization_loss = (util_1 + util_2 + util_3) / 3.0

    # Dynamically scale aperture evaluation relative to total domain dimensions
    domain_scale = (span_x + span_y) / 2.0
    coverage_ratio_1 = R_max_1 / domain_scale
    coverage_ratio_2 = R_max_2 / domain_scale

    # Penalize the solver if physical apertures shrink too low relative to your map size.
    # The inverse function creates a sharp wall preventing R_max from collapsing to 0.
    aperture_loss = (1.0 / (coverage_ratio_1 + 1e-6)) + (1.0 / (coverage_ratio_2 + 1e-6))

    # Balance all competitive forces (Resolution vs Aperture vs Memory)
    total_loss = resolution_loss + (100.0 * aperture_loss) + (1000.0 * avg_budget_utilization_loss)
    
    return float(np.asarray(total_loss).item()) / 1e7


def optimize_grid_resolutions(v_min, target_error, span_x, span_y, span_z, cpu_point_budget):
    """
    Updated execution wrapper parsing true geometric aspect spans.
    """
    C = 0.25  # FMM geometric error constant

    optimization_args = (v_min, target_error, span_x, span_y, span_z, cpu_point_budget, C)

    bounds = [
        (50.0, 500.0),     # Fine grid limits (meters)
        (400.0, 2000.0),   # Mid grid limits (meters)
        (1500.0, 15000.0)  # Coarse grid limits (meters)
    ]

    print("--- Starting Geometrically Tailored Grid Optimization ---")
    soln = differential_evolution(
        grid_loss_function, 
        bounds, 
        args=optimization_args,
        popsize=20, 
        maxiter=500, 
        disp=False
    )
    
    opt_dx1, opt_dx2, opt_dx3 = soln.x
    opt_R_max1 = (C * (opt_dx2 ** 2)) / (target_error * v_min)
    opt_R_max2 = (C * (opt_dx3 ** 2)) / (target_error * v_min)
    
    return soln.x, (opt_R_max1, opt_R_max2)

def estimate_safe_cores(cpu_point_budget, safety_factor=0.8):
    """
    Estimates a safe number of CPU cores to utilize based on the maximum 
    grid size budget and available system memory.
    
    Parameters:
    -----------
    cpu_point_budget : int
        The maximum number of points allowed in a single grid tier.
    safety_factor : float
        The fraction of available memory allowed to be consumed by this script 
        (defaults to 80% to leave room for the OS and background tasks).
    """
    # 1. Get system memory info
    mem_info = psutil.virtual_memory()
    available_mem_bytes = mem_info.available
    
    # 2. Calculate memory footprint per grid point
    # 8 bytes per float64 * roughly 12 concurrent tracking arrays
    BYTES_PER_POINT = 8 * 12
    
    # Base Python process overhead (roughly 100-150 MB for imports/compiled code)
    BASE_OVERHEAD_BYTES = 150 * 1024 * 1024 
    
    # Estimated total bytes required for a single worker thread
    estimated_worker_bytes = (cpu_point_budget * BYTES_PER_POINT) + BASE_OVERHEAD_BYTES
    
    # 3. Calculate how many workers can fit into the safe memory pool
    safe_memory_pool = available_mem_bytes * safety_factor
    mem_limited_cores = int(safe_memory_pool // estimated_worker_bytes)
    
    # 4. Get physical hardware constraints
    hardware_cores = multiprocessing.cpu_count()
    
    # 5. The ideal number of cores is bounded by hardware limits and memory limits
    optimal_cores = max(1, min(hardware_cores, mem_limited_cores))
    
    print("\n--- Dynamic Resource Allocation Analysis ---")
    print(f"System Available Memory : {available_mem_bytes / (1024**3):.2f} GB")
    print(f"Est. Memory Per Worker  : {estimated_worker_bytes / (1024**3):.2f} GB")
    print(f"Hardware CPU Cores      : {hardware_cores}")
    print(f"Memory-Safe Core Limit  : {mem_limited_cores}")
    print(f"Selected 'num_cores'    : {optimal_cores}\n")
    
    return optimal_cores


# def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2):
#     """
#     Computes travel times using ObsPy TauP by mapping a structural 3D point mesh
#     to a temporary 2D distance-depth lookup table, then maps results back to xx.
#     Integrates an angle-aware slant correction using the local ray parameter (p).
#     """
#     from scipy.interpolate import RegularGridInterpolator
    
#     # 1. Map local Cartesian inputs to true geographic coordinates (LLA)
#     X_lla = ftrns2(xx)
#     station_lla = ftrns2(loc_proj)[0]

#     # Extract surface velocities from TauP to compute corrections
#     # v_mod = taup_model.model.v_mod
#     # NEW FIXED LINE
#     # v_mod = taup_model.model.v_mod if hasattr(taup_model.model, 'v_mod') else taup_model.model

#     # NEW FIXED LINE: Extract the internal velocity layer structured array
#     # Safe fallback navigation to locate the internal v_mod data structure
#     if hasattr(taup_model.model, 'v_mod'):
#         v_mod = taup_model.model.v_mod
#     elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
#         v_mod = taup_model.model.s_mod.v_mod
#     else:
#         v_mod = taup_model.model

#     # Read the very first layer row (surface layer at 0.0 km depth)
#     # Convert velocities from km/s to m/s
#     vp_surface = v_mod.layers[0]['top_p_velocity'] * 1000.0
#     vs_surface = v_mod.layers[0]['top_s_velocity'] * 1000.0


#     # vp_surface = v_mod.p_velocity(0.0) * 1000.0  # Convert km/s to m/s
#     # vs_surface = v_mod.s_velocity(0.0) * 1000.0  # Convert km/s to m/s

#     # Track true elevations (Z)
#     src_z = station_lla[2]
#     grid_z = X_lla[:, 2]

#     # Convert absolute coordinates to positive downward depths in km, clipping anything above 0
#     source_depth_km = np.maximum(0.0, -src_z) / 1000.0
#     target_depths_km = np.maximum(0.0, -grid_z) / 1000.0

#     # 2. Vectorized True WGS84 Great-Circle Distance Calculation (in Degrees)
#     lat1 = np.radians(station_lla[0])
#     lon1 = np.radians(station_lla[1])
#     lat2 = np.radians(X_lla[:, 0])
#     lon2 = np.radians(X_lla[:, 1])

#     dlon = lon2 - lon1
#     cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
#     cos_clat = np.clip(cos_clat, -1.0, 1.0) 
#     distances_deg = np.degrees(np.arccos(cos_clat))

#     # 3. Dynamic Step Size Estimation
#     max_deg = distances_deg.max()
#     min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     dep_range = max_dep - min_dep

#     dist_step_deg = max(1e-4, max_deg / 200.0) if max_deg > 0 else 0.01
#     depth_step_km = max(1e-3, dep_range / 200.0) if dep_range > 0 else 0.5

#     grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

#     # 5. Initialize Lookup Tables (Now tracking Slowness 'p' fields as well)
#     lookup_tp = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_ts = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_pp = np.zeros((len(grid_deg), len(grid_dep)))  # P-wave ray parameter
#     lookup_ps = np.zeros((len(grid_deg), len(grid_dep)))  # S-wave ray parameter

#     # 6. Populate 2D Lookup Grid via TauP Ray-Tracing
#     for i, deg in enumerate(grid_deg):
#         if np.isclose(deg, 0.0, atol=1e-7):
#             continue
#         try:
#             arrivals = taup_model.get_travel_times(
#                 source_depth_in_km=source_depth_km,
#                 distance_in_degree=deg,
#                 phase_list=["p", "P", "s", "S"], 
#                 receiver_depth_in_km=grid_dep      
#             )
            
#             for arrival in arrivals:
#                 j = np.argmin(np.abs(grid_dep - arrival.receiver_depth))
#                 phase = arrival.name.upper()
                
#                 # ObsPy ray_param is given in seconds/radian. 
#                 # Convert to seconds/meter by dividing by Earth's radius (6371000 m)
#                 ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
#                 if phase in ["P", "P"]:
#                     if np.isnan(lookup_tp[i, j]) or arrival.time < lookup_tp[i, j]:
#                         lookup_tp[i, j] = arrival.time
#                         lookup_pp[i, j] = ray_param_sec_per_m
#                 elif phase in ["S", "S"]:
#                     if np.isnan(lookup_ts[i, j]) or arrival.time < lookup_ts[i, j]:
#                         lookup_ts[i, j] = arrival.time
#                         lookup_ps[i, j] = ray_param_sec_per_m
                        
#         except Exception:
#             continue

#     # === EXPLICIT SOURCE POSITION ENFORCEMENT ===
#     idx_deg_zero = 0 
#     for j, dep in enumerate(grid_dep):
#         if np.isclose(dep, source_depth_km, atol=depth_step_km * 0.5):
#             lookup_tp[idx_deg_zero, j] = 0.0
#             lookup_ts[idx_deg_zero, j] = 0.0
#         else:
#             distance_km = np.abs(dep - source_depth_km)
#             lookup_tp[idx_deg_zero, j] = distance_km / (vp_surface / 1000.0)
#             lookup_ts[idx_deg_zero, j] = distance_km / (vs_surface / 1000.0)
#         # At 0 degrees distance, horizontal slowness p is exactly 0 (vertical ray)
#         lookup_pp[idx_deg_zero, j] = 0.0
#         lookup_ps[idx_deg_zero, j] = 0.0

#     # 7. Clean up remaining NaN gaps safely
#     for table in [lookup_tp, lookup_ts]:
#         if np.any(np.isnan(table)):
#             mask = np.isnan(table)
#             table[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), table[~mask])

#     # 8. High-Speed 2D Matrix Interpolation back to the structural 3D Mesh
#     interp_tp = RegularGridInterpolator((grid_deg, grid_dep), lookup_tp, method='linear', bounds_error=False, fill_value=None)
#     interp_ts = RegularGridInterpolator((grid_deg, grid_dep), lookup_ts, method='linear', bounds_error=False, fill_value=None)
#     interp_pp = RegularGridInterpolator((grid_deg, grid_dep), lookup_pp, method='linear', bounds_error=False, fill_value=None)
#     interp_ps = RegularGridInterpolator((grid_deg, grid_dep), lookup_ps, method='linear', bounds_error=False, fill_value=None)

#     query_points = np.column_stack((distances_deg, target_depths_km))
    
#     tp_times = interp_tp(query_points)
#     ts_times = interp_ts(query_points)
#     p_wave_slowness = interp_pp(query_points)
#     s_wave_slowness = interp_ps(query_points)

#     # --- ANGLE-AWARE MOUNTAIN TOPOGRAPHIC TIME CORRECTIONS ---
#     # Apply Slant Path Correction formula: dt = z * sqrt(1/V^2 - p^2)
#     # If a point is above sea level (Z > 0), calculate its specific slant propagation delay
#     p_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_wave_slowness**2))
#     s_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_wave_slowness**2))
    
#     src_correction_p = np.maximum(0.0, src_z) * p_factor_src
#     src_correction_s = np.maximum(0.0, src_z) * s_factor_src

#     node_corrections_p = np.maximum(0.0, grid_z) * p_factor_src
#     node_corrections_s = np.maximum(0.0, grid_z) * s_factor_src

#     tp_times = tp_times + src_correction_p + node_corrections_p
#     ts_times = ts_times + src_correction_s + node_corrections_s

#     # Force the source location analytically to 0.0 to prevent interpolation bleeding.
#     station_idx = np.argmin(distances_deg + np.abs(target_depths_km - source_depth_km))
#     if distances_deg[station_idx] < 1e-4:
#         tp_times[station_idx] = 0.0
#         ts_times[station_idx] = 0.0

#     return tp_times, ts_times

# def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2):
#     """
#     Computes travel times using ObsPy TauP by mapping a structural 3D point mesh
#     to a temporary 2D distance-depth lookup table, then maps results back to xx.
#     Integrates an angle-aware slant correction using the local ray parameter (p).
#     """
#     from scipy.interpolate import RegularGridInterpolator
    
#     # 1. Map local Cartesian inputs to true geographic coordinates (LLA)
#     X_lla = ftrns2(xx)
#     station_lla = ftrns2(loc_proj)[0]

#     # Extract surface velocities from TauP to compute corrections
#     # v_mod = taup_model.model.v_mod
#     # NEW FIXED LINE
#     # v_mod = taup_model.model.v_mod if hasattr(taup_model.model, 'v_mod') else taup_model.model

#     # NEW FIXED LINE: Extract the internal velocity layer structured array
#     # Safe fallback navigation to locate the internal v_mod data structure
#     if hasattr(taup_model.model, 'v_mod'):
#         v_mod = taup_model.model.v_mod
#     elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
#         v_mod = taup_model.model.s_mod.v_mod
#     else:
#         v_mod = taup_model.model

#     # Read the very first layer row (surface layer at 0.0 km depth)
#     # Convert velocities from km/s to m/s
#     vp_surface = v_mod.layers[0]['top_p_velocity'] * 1000.0
#     vs_surface = v_mod.layers[0]['top_s_velocity'] * 1000.0


#     # vp_surface = v_mod.p_velocity(0.0) * 1000.0  # Convert km/s to m/s
#     # vs_surface = v_mod.s_velocity(0.0) * 1000.0  # Convert km/s to m/s

#     # Track true elevations (Z)
#     src_z = station_lla[2]
#     grid_z = X_lla[:, 2]

#     # Convert absolute coordinates to positive downward depths in km, clipping anything above 0
#     source_depth_km = np.maximum(0.0, -src_z) / 1000.0
#     target_depths_km = np.maximum(0.0, -grid_z) / 1000.0

#     # 2. Vectorized True WGS84 Great-Circle Distance Calculation (in Degrees)
#     lat1 = np.radians(station_lla[0])
#     lon1 = np.radians(station_lla[1])
#     lat2 = np.radians(X_lla[:, 0])
#     lon2 = np.radians(X_lla[:, 1])

#     dlon = lon2 - lon1
#     cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
#     cos_clat = np.clip(cos_clat, -1.0, 1.0) 
#     distances_deg = np.degrees(np.arccos(cos_clat))

#     # # 3. Dynamic Step Size Estimation
#     # max_deg = distances_deg.max()
#     # min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     # dep_range = max_dep - min_dep

#     # dist_step_deg = max(1e-4, max_deg / 200.0) if max_deg > 0 else 0.01
#     # depth_step_km = max(1e-3, dep_range / 200.0) if dep_range > 0 else 0.5

#     # grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     # grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

# 	# =====================================================================
#     # 3. Uncertainty-Driven Dynamic Step Size Estimation (~2% Error Bound)
#     # =====================================================================
#     max_deg = distances_deg.max()
#     min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     dep_range = max_dep - min_dep

#     # 1. GEOGRAPHICALLY ACCURATE KM-TO-DEGREE CONVERSION (WGS84 Meridian Radius)
#     # Since we are working with great-circle paths, the distance per degree of latitude 
#     # varies slightly by location. We compute this dynamically using the average latitude.
#     mean_lat_rad = np.radians(np.mean(X_lla[:, 0]))
#     # WGS84 Constants
#     a = 6378.137          # Equatorial radius in km
#     b = 6356.7523142      # Polar radius in km
#     # Compute the radius of curvature along the meridian at this latitude
#     e_sq = (a**2 - b**2) / (a**2)
#     radius_meridian = (a * (1.0 - e_sq)) / (1.0 - e_sq * np.sin(mean_lat_rad)**2)**(1.5)
#     km_per_deg = (np.pi / 180.0) * radius_meridian

#     # Convert horizontal aperture from degrees to true kilometers
#     max_dist_km = max_deg * km_per_deg


#     # =====================================================================
#     # 2. DYNAMIC VELOCITY INVERSION (UN-SKEWED BY MANTLE DEPTHS)
#     # =====================================================================
#     # Find the depth limits of your active mesh tier (in meters to match your input 'depths')
#     # Since your 'depths' array is likely positive-downward or negative-upward, 
#     # let's adaptively find the matching indices within your custom profile.
    
#     # Map active mesh depths to positive values in meters
#     mesh_min_dep_m = max(0.0, min_dep * 1000.0)
#     mesh_max_dep_m = max_dep * 1000.0
    
#     # Convert your input 'depths' array to positive downward to easily mask it
#     positive_input_depths = np.abs(depths)
    
#     # Create a mask of your input velocity model that matches the depth profile of the active mesh
#     active_depth_mask = (positive_input_depths >= mesh_min_dep_m) & (positive_input_depths <= mesh_max_dep_m)
    
#     # Fallback: If the mesh goes deeper than your custom input profile (plunging into iasp91),
#     # or if the mask is empty, just look at your local crustal profile to keep resolution sharp.
#     if np.sum(active_depth_mask) == 0:
#         active_depth_mask = positive_input_depths <= np.max(positive_input_depths)
        
#     # Extract the median velocity *only* from the layers your mesh is physically occupying
#     v_representative = float(np.median(vp[active_depth_mask])) / 1000.0  # Convert m/s to km/s
#     v_representative = max(1.5, v_representative)  # Safety floor (speed of sound in water)


#     # 2. DYNAMIC VELOCITY INVERSION
#     # Extract the true unblended representative velocity from your custom model.
#     # We use the median velocity of your target subsurface array to represent the medium.
#     # We clip it to P-wave limits to prevent dividing by zero if S-wave layers drop to 0 in the core.
#     # v_representative = float(np.median(vp)) / 1000.0  # Convert m/s to km/s
#     # v_representative = max(1.5, v_representative)     # Safety floor (speed of sound in water)

#     # Target maximum fractional interpolation error (2%)
#     target_error_fraction = 0.02  

#     # 3. PHYSICS-BASED STEP SIZE CALCULATION
#     # --- HORIZONTAL STEP ESTIMATION ---
#     if max_dist_km > 0:
#         # Allowable spacing scales with the square root of distance 
#         # because wavefront curvature flattens out away from the source anchor.
#         ideal_dx_km = np.sqrt(8.0 * target_error_fraction * max_dist_km * v_representative)
        
#         # Guardrails: Never step coarser than 5% of the total domain width
#         # Never step finer than 100 meters to protect memory
#         dist_step_km = np.clip(ideal_dx_km, 0.1, max_dist_km * 0.05)
#         dist_step_deg = dist_step_km / km_per_deg
#     else:
#         dist_step_deg = 0.01

#     # --- VERTICAL STEP ESTIMATION ---
#     if dep_range > 0:
#         # Vertical step scales based on the vertical depth range being profiled
#         ideal_dz_km = np.sqrt(8.0 * target_error_fraction * dep_range * v_representative)
#         depth_step_km = np.clip(ideal_dz_km, 0.1, dep_range * 0.05)
#     else:
#         depth_step_km = 0.5

#     # 4. BUILD THE ACCURACY-BOUNDED LOOKUP AXES
#     grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

#     # --- STRUCTURAL STABILITY SHIELD ---
#     # Guardrail: If the resulting grid contains too few nodes due to tight clipping,
#     # pad the resolution out to ensure the 2D interpolator has an adequate stencil.
#     if len(grid_deg) < 30:
#         grid_deg = np.linspace(0.0, max_deg + 2*dist_step_deg, 30)
#     if len(grid_dep) < 30:
#         grid_dep = np.linspace(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, 30)

#     # 5. Initialize Lookup Tables (Now tracking Slowness 'p' fields as well)
#     lookup_tp = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_ts = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_pp = np.zeros((len(grid_deg), len(grid_dep)))  # P-wave ray parameter
#     lookup_ps = np.zeros((len(grid_deg), len(grid_dep)))  # S-wave ray parameter

#     # 6. Populate 2D Lookup Grid via TauP Ray-Tracing
#     for i, deg in enumerate(grid_deg):
#         if np.isclose(deg, 0.0, atol=1e-7):
#             continue
#         try:
#             arrivals = taup_model.get_travel_times(
#                 source_depth_in_km=source_depth_km,
#                 distance_in_degree=deg,
#                 phase_list=["p", "P", "s", "S"], 
#                 receiver_depth_in_km=grid_dep      
#             )
            
#             for arrival in arrivals:
#                 j = np.argmin(np.abs(grid_dep - arrival.receiver_depth))
#                 phase = arrival.name.upper()
                
#                 # ObsPy ray_param is given in seconds/radian. 
#                 # Convert to seconds/meter by dividing by Earth's radius (6371000 m)
#                 ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
#                 if phase in ["P", "P"]:
#                     if np.isnan(lookup_tp[i, j]) or arrival.time < lookup_tp[i, j]:
#                         lookup_tp[i, j] = arrival.time
#                         lookup_pp[i, j] = ray_param_sec_per_m
#                 elif phase in ["S", "S"]:
#                     if np.isnan(lookup_ts[i, j]) or arrival.time < lookup_ts[i, j]:
#                         lookup_ts[i, j] = arrival.time
#                         lookup_ps[i, j] = ray_param_sec_per_m
                        
#         except Exception:
#             continue

#     # === EXPLICIT SOURCE POSITION ENFORCEMENT ===
#     idx_deg_zero = 0 
#     for j, dep in enumerate(grid_dep):
#         if np.isclose(dep, source_depth_km, atol=depth_step_km * 0.5):
#             lookup_tp[idx_deg_zero, j] = 0.0
#             lookup_ts[idx_deg_zero, j] = 0.0
#         else:
#             distance_km = np.abs(dep - source_depth_km)
#             lookup_tp[idx_deg_zero, j] = distance_km / (vp_surface / 1000.0)
#             lookup_ts[idx_deg_zero, j] = distance_km / (vs_surface / 1000.0)
#         # At 0 degrees distance, horizontal slowness p is exactly 0 (vertical ray)
#         lookup_pp[idx_deg_zero, j] = 0.0
#         lookup_ps[idx_deg_zero, j] = 0.0

#     # 7. Clean up remaining NaN gaps safely
#     for table in [lookup_tp, lookup_ts]:
#         if np.any(np.isnan(table)):
#             mask = np.isnan(table)
#             table[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), table[~mask])

#     # 8. High-Speed 2D Matrix Interpolation back to the structural 3D Mesh
#     interp_tp = RegularGridInterpolator((grid_deg, grid_dep), lookup_tp, method='linear', bounds_error=False, fill_value=None)
#     interp_ts = RegularGridInterpolator((grid_deg, grid_dep), lookup_ts, method='linear', bounds_error=False, fill_value=None)
#     interp_pp = RegularGridInterpolator((grid_deg, grid_dep), lookup_pp, method='linear', bounds_error=False, fill_value=None)
#     interp_ps = RegularGridInterpolator((grid_deg, grid_dep), lookup_ps, method='linear', bounds_error=False, fill_value=None)

#     query_points = np.column_stack((distances_deg, target_depths_km))
    
#     tp_times = interp_tp(query_points)
#     ts_times = interp_ts(query_points)
#     p_wave_slowness = interp_pp(query_points)
#     s_wave_slowness = interp_ps(query_points)

#     # --- ANGLE-AWARE MOUNTAIN TOPOGRAPHIC TIME CORRECTIONS ---
#     # Apply Slant Path Correction formula: dt = z * sqrt(1/V^2 - p^2)
#     # If a point is above sea level (Z > 0), calculate its specific slant propagation delay
#     p_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_wave_slowness**2))
#     s_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_wave_slowness**2))
    
#     src_correction_p = np.maximum(0.0, src_z) * p_factor_src
#     src_correction_s = np.maximum(0.0, src_z) * s_factor_src

#     node_corrections_p = np.maximum(0.0, grid_z) * p_factor_src
#     node_corrections_s = np.maximum(0.0, grid_z) * s_factor_src

#     tp_times = tp_times + src_correction_p + node_corrections_p
#     ts_times = ts_times + src_correction_s + node_corrections_s

#     # Force the source location analytically to 0.0 to prevent interpolation bleeding.
#     station_idx = np.argmin(distances_deg + np.abs(target_depths_km - source_depth_km))
#     if distances_deg[station_idx] < 1e-4:
#         tp_times[station_idx] = 0.0
#         ts_times[station_idx] = 0.0

#     return tp_times, ts_times


# def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, depths, vp):
#     """
#     Computes travel times using ObsPy TauP by mapping a structural 3D point mesh
#     to a temporary 2D distance-depth lookup table, then maps results back to xx.
#     Integrates an angle-aware slant correction using the local ray parameter (p).
#     """
#     from scipy.interpolate import RegularGridInterpolator
#     import numpy as np
    
#     # 1. Map local Cartesian inputs to true geographic coordinates (LLA)
#     X_lla = ftrns2(xx)
#     station_lla = ftrns2(loc_proj)[0]

#     # Safe fallback navigation to locate the internal v_mod data structure
#     if hasattr(taup_model.model, 'v_mod'):
#         v_mod = taup_model.model.v_mod
#     elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
#         v_mod = taup_model.model.s_mod.v_mod
#     else:
#         v_mod = taup_model.model

#     # Read the very first layer row (surface layer at 0.0 km depth) and convert to m/s
#     vp_surface = v_mod.layers[0]['top_p_velocity'] * 1000.0
#     vs_surface = v_mod.layers[0]['top_s_velocity'] * 1000.0

#     # Track true elevations (Z)
#     src_z = station_lla[2]
#     grid_z = X_lla[:, 2]

#     # Convert absolute coordinates to positive downward depths in km, clipping anything above 0
#     source_depth_km = np.maximum(0.0, -src_z) / 1000.0
#     target_depths_km = np.maximum(0.0, -grid_z) / 1000.0

#     # 2. Vectorized True WGS84 Great-Circle Distance Calculation (in Degrees)
#     lat1 = np.radians(station_lla[0])
#     lon1 = np.radians(station_lla[1])
#     lat2 = np.radians(X_lla[:, 0])
#     lon2 = np.radians(X_lla[:, 1])

#     dlon = lon2 - lon1
#     cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
#     cos_clat = np.clip(cos_clat, -1.0, 1.0) 
#     distances_deg = np.degrees(np.arccos(cos_clat))

#     # =====================================================================
#     # 3. Uncertainty-Driven Dynamic Step Size Estimation (~2% Error Bound)
#     # =====================================================================
#     max_deg = distances_deg.max()
#     min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     dep_range = max_dep - min_dep

#     # Geographically accurate km-to-degree conversion via WGS84 Meridian Radius
#     mean_lat_rad = np.radians(np.mean(X_lla[:, 0]))
#     a = 6378.137          # Equatorial radius in km
#     b = 6356.7523142      # Polar radius in km
#     e_sq = (a**2 - b**2) / (a**2)
#     radius_meridian = (a * (1.0 - e_sq)) / (1.0 - e_sq * np.sin(mean_lat_rad)**2)**(1.5)
#     km_per_deg = (np.pi / 180.0) * radius_meridian

#     max_dist_km = max_deg * km_per_deg

#     # Un-skewed velocity mask matching the active mesh depth limits
#     mesh_min_dep_m = max(0.0, min_dep * 1000.0)
#     mesh_max_dep_m = max_dep * 1000.0
    
#     positive_input_depths = np.abs(depths)
#     active_depth_mask = (positive_input_depths >= mesh_min_dep_m) & (positive_input_depths <= mesh_max_dep_m)
    
#     if np.sum(active_depth_mask) == 0:
#         active_depth_mask = positive_input_depths <= np.max(positive_input_depths)
        
#     v_representative = float(np.median(vp[active_depth_mask])) / 1000.0  # Convert m/s to km/s
#     v_representative = max(1.5, v_representative)

#     target_error_fraction = 0.02  

#     # Physics-based steps sizing
#     if max_dist_km > 0:
#         ideal_dx_km = np.sqrt(8.0 * target_error_fraction * max_dist_km * v_representative)
#         dist_step_km = np.clip(ideal_dx_km, 0.1, max_dist_km * 0.05)
#         dist_step_deg = dist_step_km / km_per_deg
#     else:
#         dist_step_deg = 0.01

#     if dep_range > 0:
#         ideal_dz_km = np.sqrt(8.0 * target_error_fraction * dep_range * v_representative)
#         depth_step_km = np.clip(ideal_dz_km, 0.1, dep_range * 0.05)
#     else:
#         depth_step_km = 0.5

#     # Build lookup axes dynamically
#     grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

#     if len(grid_deg) < 30:
#         grid_deg = np.linspace(0.0, max_deg + 2*dist_step_deg, 30)
#     if len(grid_dep) < 30:
#         grid_dep = np.linspace(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, 30)

#     # 5. Initialize Lookup Tables
#     lookup_tp = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_ts = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_pp = np.zeros((len(grid_deg), len(grid_dep)))  
#     lookup_ps = np.zeros((len(grid_deg), len(grid_dep)))  

#     # 6. Your original, working Vectorized Lookup Grid Population Loop
#     for i, deg in enumerate(grid_deg):
#         if np.isclose(deg, 0.0, atol=1e-7):
#             continue
#         try:
#             # Reverted to your exact working vectorized call structure
#             arrivals = taup_model.get_travel_times(
#                 source_depth_in_km=source_depth_km,
#                 distance_in_degree=deg,
#                 phase_list=["p", "P", "s", "S"], 
#                 receiver_depth_in_km=grid_dep      
#             )
            
#             for arrival in arrivals:
#                 j = np.argmin(np.abs(grid_dep - arrival.receiver_depth))
#                 phase = arrival.name.upper()
#                 ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
#                 if phase in ["P", "P"]:
#                     if np.isnan(lookup_tp[i, j]) or arrival.time < lookup_tp[i, j]:
#                         lookup_tp[i, j] = arrival.time
#                         lookup_pp[i, j] = ray_param_sec_per_m
#                 elif phase in ["S", "S"]:
#                     if np.isnan(lookup_ts[i, j]) or arrival.time < lookup_ts[i, j]:
#                         lookup_ts[i, j] = arrival.time
#                         lookup_ps[i, j] = ray_param_sec_per_m
                        
#         except Exception:
#             continue

#     # === EXPLICIT SOURCE POSITION ENFORCEMENT ===
#     idx_deg_zero = 0 
#     for j, dep in enumerate(grid_dep):
#         if np.isclose(dep, source_depth_km, atol=depth_step_km * 0.5):
#             lookup_tp[idx_deg_zero, j] = 0.0
#             lookup_ts[idx_deg_zero, j] = 0.0
#         else:
#             distance_km = np.abs(dep - source_depth_km)
#             lookup_tp[idx_deg_zero, j] = distance_km / (vp_surface / 1000.0)
#             lookup_ts[idx_deg_zero, j] = distance_km / (vs_surface / 1000.0)
#         lookup_pp[idx_deg_zero, j] = 0.0
#         lookup_ps[idx_deg_zero, j] = 0.0

#     # 7. Clean up remaining NaN gaps safely
#     for table in [lookup_tp, lookup_ts]:
#         if np.any(np.isnan(table)):
#             mask = np.isnan(table)
#             table[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), table[~mask])

#     # 8. High-Speed 2D Matrix Interpolation back to the structural 3D Mesh
#     interp_tp = RegularGridInterpolator((grid_deg, grid_dep), lookup_tp, method='linear', bounds_error=False, fill_value=None)
#     interp_ts = RegularGridInterpolator((grid_deg, grid_dep), lookup_ts, method='linear', bounds_error=False, fill_value=None)
#     interp_pp = RegularGridInterpolator((grid_deg, grid_dep), lookup_pp, method='linear', bounds_error=False, fill_value=None)
#     interp_ps = RegularGridInterpolator((grid_deg, grid_dep), lookup_ps, method='linear', bounds_error=False, fill_value=None)

#     query_points = np.column_stack((distances_deg, target_depths_km))
    
#     tp_times = interp_tp(query_points)
#     ts_times = interp_ts(query_points)
#     p_wave_slowness = interp_pp(query_points)
#     s_wave_slowness = interp_ps(query_points)

#     # --- ANGLE-AWARE MOUNTAIN TOPOGRAPHIC TIME CORRECTIONS ---
#     p_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_wave_slowness**2))
#     s_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_wave_slowness**2))
    
#     src_correction_p = np.maximum(0.0, src_z) * p_factor_src
#     src_correction_s = np.maximum(0.0, src_z) * s_factor_src

#     node_corrections_p = np.maximum(0.0, grid_z) * p_factor_src
#     node_corrections_s = np.maximum(0.0, grid_z) * s_factor_src

#     tp_times = tp_times + src_correction_p + node_corrections_p
#     ts_times = ts_times + src_correction_s + node_corrections_s

#     # Force the source location analytically to 0.0 to prevent interpolation bleeding.
#     station_idx = np.argmin(distances_deg + np.abs(target_depths_km - source_depth_km))
#     if distances_deg[station_idx] < 1e-4:
#         tp_times[station_idx] = 0.0
#         ts_times[station_idx] = 0.0

#     return tp_times, ts_times


# def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, depths, vp):
#     """
#     Computes travel times using ObsPy TauP by mapping a structural 3D point mesh
#     to a temporary 2D distance-depth lookup table, then maps results back to xx.
#     Integrates an angle-aware slant correction using the local ray parameter (p).
#     """
#     from scipy.interpolate import RegularGridInterpolator
#     import numpy as np
    
#     # 1. Map local Cartesian inputs to true geographic coordinates (LLA)
#     X_lla = ftrns2(xx)
#     station_lla = ftrns2(loc_proj)[0]

#     # Safe fallback navigation to locate the internal v_mod data structure
#     if hasattr(taup_model.model, 'v_mod'):
#         v_mod = taup_model.model.v_mod
#     elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
#         v_mod = taup_model.model.s_mod.v_mod
#     else:
#         v_mod = taup_model.model

#     # Read the very first layer row (surface layer at 0.0 km depth) and convert to m/s
#     vp_surface = v_mod.layers[0]['top_p_velocity'] * 1000.0
#     vs_surface = v_mod.layers[0]['top_s_velocity'] * 1000.0

#     # Track true elevations (Z)
#     src_z = station_lla[2]
#     grid_z = X_lla[:, 2]

#     # Convert absolute coordinates to positive downward depths in km, clipping anything above 0
#     source_depth_km = np.maximum(0.0, -src_z) / 1000.0
#     target_depths_km = np.maximum(0.0, -grid_z) / 1000.0

#     # 2. Vectorized True WGS84 Great-Circle Distance Calculation (in Degrees)
#     lat1 = np.radians(station_lla[0])
#     lon1 = np.radians(station_lla[1])
#     lat2 = np.radians(X_lla[:, 0])
#     lon2 = np.radians(X_lla[:, 1])

#     dlon = lon2 - lon1
#     cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
#     cos_clat = np.clip(cos_clat, -1.0, 1.0) 
#     distances_deg = np.degrees(np.arccos(cos_clat))

#     # =====================================================================
#     # 3. Uncertainty-Driven Dynamic Step Size Estimation (~2% Error Bound)
#     # =====================================================================
#     max_deg = distances_deg.max()
#     min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     dep_range = max_dep - min_dep

#     # Geographically accurate km-to-degree conversion via WGS84 Meridian Radius
#     mean_lat_rad = np.radians(np.mean(X_lla[:, 0]))
#     a = 6378.137          # Equatorial radius in km
#     b = 6356.7523142      # Polar radius in km
#     e_sq = (a**2 - b**2) / (a**2)
#     radius_meridian = (a * (1.0 - e_sq)) / (1.0 - e_sq * np.sin(mean_lat_rad)**2)**(1.5)
#     km_per_deg = (np.pi / 180.0) * radius_meridian

#     max_dist_km = max_deg * km_per_deg

#     # Un-skewed velocity mask matching the active mesh depth limits
#     mesh_min_dep_m = max(0.0, min_dep * 1000.0)
#     mesh_max_dep_m = max_dep * 1000.0
    
#     positive_input_depths = np.abs(depths)
#     active_depth_mask = (positive_input_depths >= mesh_min_dep_m) & (positive_input_depths <= mesh_max_dep_m)
    
#     if np.sum(active_depth_mask) == 0:
#         active_depth_mask = positive_input_depths <= np.max(positive_input_depths)
        
#     v_representative = float(np.median(vp[active_depth_mask])) / 1000.0  # Convert m/s to km/s
#     v_representative = max(1.5, v_representative)

#     target_error_fraction = 0.02  

#     # Physics-based steps sizing
#     if max_dist_km > 0:
#         ideal_dx_km = np.sqrt(8.0 * target_error_fraction * max_dist_km * v_representative)
#         dist_step_km = np.clip(ideal_dx_km, 0.1, max_dist_km * 0.05)
#         dist_step_deg = dist_step_km / km_per_deg
#     else:
#         dist_step_deg = 0.01

#     if dep_range > 0:
#         ideal_dz_km = np.sqrt(8.0 * target_error_fraction * dep_range * v_representative)
#         depth_step_km = np.clip(ideal_dz_km, 0.1, dep_range * 0.05)
#     else:
#         depth_step_km = 0.5

#     # Build lookup axes dynamically
#     grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

#     if len(grid_deg) < 30:
#         grid_deg = np.linspace(0.0, max_deg + 2*dist_step_deg, 30)
#     if len(grid_dep) < 30:
#         grid_dep = np.linspace(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, 30)

#     # 5. Initialize Lookup Tables
#     lookup_tp = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_ts = np.nan * np.zeros((len(grid_deg), len(grid_dep)))
#     lookup_pp = np.zeros((len(grid_deg), len(grid_dep)))  
#     lookup_ps = np.zeros((len(grid_deg), len(grid_dep)))  

#     # 6. Your original, working Vectorized Lookup Grid Population Loop
#     for i, deg in enumerate(grid_deg):
#         if np.isclose(deg, 0.0, atol=1e-7):
#             continue
#         try:
#             # Reverted to your exact working vectorized call structure
#             arrivals = taup_model.get_travel_times(
#                 source_depth_in_km=source_depth_km,
#                 distance_in_degree=deg,
#                 phase_list=["p", "P", "s", "S"], 
#                 receiver_depth_in_km=grid_dep      
#             )
            
#             for arrival in arrivals:
#                 j = np.argmin(np.abs(grid_dep - arrival.receiver_depth))
#                 phase = arrival.name.upper()
#                 ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
#                 if phase in ["P", "P"]:
#                     if np.isnan(lookup_tp[i, j]) or arrival.time < lookup_tp[i, j]:
#                         lookup_tp[i, j] = arrival.time
#                         lookup_pp[i, j] = ray_param_sec_per_m
#                 elif phase in ["S", "S"]:
#                     if np.isnan(lookup_ts[i, j]) or arrival.time < lookup_ts[i, j]:
#                         lookup_ts[i, j] = arrival.time
#                         lookup_ps[i, j] = ray_param_sec_per_m
                        
#         except Exception:
#             continue

#     # === EXPLICIT SOURCE POSITION ENFORCEMENT ===
#     idx_deg_zero = 0 
#     for j, dep in enumerate(grid_dep):
#         if np.isclose(dep, source_depth_km, atol=depth_step_km * 0.5):
#             lookup_tp[idx_deg_zero, j] = 0.0
#             lookup_ts[idx_deg_zero, j] = 0.0
#         else:
#             distance_km = np.abs(dep - source_depth_km)
#             lookup_tp[idx_deg_zero, j] = distance_km / (vp_surface / 1000.0)
#             lookup_ts[idx_deg_zero, j] = distance_km / (vs_surface / 1000.0)
#         lookup_pp[idx_deg_zero, j] = 0.0
#         lookup_ps[idx_deg_zero, j] = 0.0

#     # 7. Clean up remaining NaN gaps safely
#     for table in [lookup_tp, lookup_ts]:
#         if np.any(np.isnan(table)):
#             mask = np.isnan(table)
#             table[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), table[~mask])

#     # 8. High-Speed 2D Matrix Interpolation back to the structural 3D Mesh
#     interp_tp = RegularGridInterpolator((grid_deg, grid_dep), lookup_tp, method='linear', bounds_error=False, fill_value=None)
#     interp_ts = RegularGridInterpolator((grid_deg, grid_dep), lookup_ts, method='linear', bounds_error=False, fill_value=None)
#     interp_pp = RegularGridInterpolator((grid_deg, grid_dep), lookup_pp, method='linear', bounds_error=False, fill_value=None)
#     interp_ps = RegularGridInterpolator((grid_deg, grid_dep), lookup_ps, method='linear', bounds_error=False, fill_value=None)

#     query_points = np.column_stack((distances_deg, target_depths_km))
    
#     tp_times = interp_tp(query_points)
#     ts_times = interp_ts(query_points)
#     p_wave_slowness = interp_pp(query_points)
#     s_wave_slowness = interp_ps(query_points)

#     # --- ANGLE-AWARE MOUNTAIN TOPOGRAPHIC TIME CORRECTIONS ---
#     p_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_wave_slowness**2))
#     s_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_wave_slowness**2))
    
#     src_correction_p = np.maximum(0.0, src_z) * p_factor_src
#     src_correction_s = np.maximum(0.0, src_z) * s_factor_src

#     node_corrections_p = np.maximum(0.0, grid_z) * p_factor_src
#     node_corrections_s = np.maximum(0.0, grid_z) * s_factor_src

#     tp_times = tp_times + src_correction_p + node_corrections_p
#     ts_times = ts_times + src_correction_s + node_corrections_s

#     # Force the source location analytically to 0.0 to prevent interpolation bleeding.
#     station_idx = np.argmin(distances_deg + np.abs(target_depths_km - source_depth_km))
#     if distances_deg[station_idx] < 1e-4:
#         tp_times[station_idx] = 0.0
#         ts_times[station_idx] = 0.0

#     return tp_times, ts_times



# def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, depths, vp):
#     """
#     Computes travel times using ObsPy TauP by mapping a structural 3D point mesh
#     to a temporary 2D distance-depth lookup table, then maps results back to xx.
#     Integrates an angle-aware slant correction using the local ray parameter (p).
#     """
#     from scipy.interpolate import RegularGridInterpolator, griddata
#     import numpy as np
    
#     # =====================================================================
#     # 1. Coordinate Mapping and Surface Geometries
#     # =====================================================================
#     X_lla = ftrns2(xx)
#     station_lla = ftrns2(loc_proj)[0]

#     # Safe fallback navigation to locate the internal v_mod data structure
#     if hasattr(taup_model.model, 'v_mod'):
#         v_mod = taup_model.model.v_mod
#     elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
#         v_mod = taup_model.model.s_mod.v_mod
#     else:
#         v_mod = taup_model.model

#     # Extract surface velocities from 1D profile and convert to m/s
#     vp_surface = v_mod.layers[0]['top_p_velocity'] * 1000.0
#     vs_surface = v_mod.layers[0]['top_s_velocity'] * 1000.0

#     # Track true elevations (Z)
#     src_z = station_lla[2]
#     grid_z = X_lla[:, 2]

#     # Convert absolute coordinates to positive downward depths in km (clip anything above sea level)
#     source_depth_km = np.maximum(0.0, -src_z) / 1000.0
#     target_depths_km = np.maximum(0.0, -grid_z) / 1000.0

#     # =====================================================================
#     # 2. Vectorized True WGS84 Great-Circle Distance Calculation
#     # =====================================================================
#     lat1 = np.radians(station_lla[0])
#     lon1 = np.radians(station_lla[1])
#     lat2 = np.radians(X_lla[:, 0])
#     lon2 = np.radians(X_lla[:, 1])

#     dlon = lon2 - lon1
#     cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
#     cos_clat = np.clip(cos_clat, -1.0, 1.0) 
#     distances_deg = np.degrees(np.arccos(cos_clat))

#     # =====================================================================
#     # 3. Dynamic Step Size Estimation (~2% Relative Uncertainty Bound)
#     # =====================================================================
#     max_deg = distances_deg.max()
#     min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     dep_range = max_dep - min_dep

#     # Geographically accurate km-to-degree conversion via WGS84 Meridian Radius
#     mean_lat_rad = np.radians(np.mean(X_lla[:, 0]))
#     a = 6378.137          # Equatorial radius in km
#     b = 6356.7523142      # Polar radius in km
#     e_sq = (a**2 - b**2) / (a**2)
#     radius_meridian = (a * (1.0 - e_sq)) / (1.0 - e_sq * np.sin(mean_lat_rad)**2)**(1.5)
#     km_per_deg = (np.pi / 180.0) * radius_meridian

#     max_dist_km = max_deg * km_per_deg

#     # Un-skewed velocity mask matching the active mesh depth limits
#     mesh_min_dep_m = max(0.0, min_dep * 1000.0)
#     mesh_max_dep_m = max_dep * 1000.0
    
#     positive_input_depths = np.abs(depths)
#     active_depth_mask = (positive_input_depths >= mesh_min_dep_m) & (positive_input_depths <= mesh_max_dep_m)
    
#     if np.sum(active_depth_mask) == 0:
#         active_depth_mask = positive_input_depths <= np.max(positive_input_depths)
        
#     v_representative = float(np.median(vp[active_depth_mask])) / 1000.0  # Convert m/s to km/s
#     v_representative = max(1.5, v_representative)

#     target_error_fraction = 0.02  

#     # Physics-based spatial step selection
#     if max_dist_km > 0:
#         ideal_dx_km = np.sqrt(8.0 * target_error_fraction * max_dist_km * v_representative)
#         dist_step_km = np.clip(ideal_dx_km, 0.1, max_dist_km * 0.05)
#         dist_step_deg = dist_step_km / km_per_deg
#     else:
#         dist_step_deg = 0.01

#     if dep_range > 0:
#         ideal_dz_km = np.sqrt(8.0 * target_error_fraction * dep_range * v_representative)
#         depth_step_km = np.clip(ideal_dz_km, 0.1, dep_range * 0.05)
#     else:
#         depth_step_km = 0.5

#     # Build lookup axes dynamically
#     grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

#     if len(grid_deg) < 30:
#         grid_deg = np.linspace(0.0, max_deg + 2*dist_step_deg, 30)
#     if len(grid_dep) < 30:
#         grid_dep = np.linspace(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, 30)

#     # =====================================================================
#     # 4. Initialize Lookup Tables with Infinity (Enforces Minimum First Arrivals)
#     # =====================================================================
#     lookup_tp = np.full((len(grid_deg), len(grid_dep)), np.inf)
#     lookup_ts = np.full((len(grid_deg), len(grid_dep)), np.inf)
#     lookup_pp = np.zeros((len(grid_deg), len(grid_dep)))  
#     lookup_ps = np.zeros((len(grid_deg), len(grid_dep)))  

#     # =====================================================================
#     # 5. Populate 2D Lookup Grid via TauP Ray-Tracing
#     # =====================================================================
#     for i, deg in enumerate(grid_deg):
#         if np.isclose(deg, 0.0, atol=1e-7):
#             continue
#         try:
#             arrivals = taup_model.get_travel_times(
#                 source_depth_in_km=source_depth_km,
#                 distance_in_degree=deg,
#                 phase_list=["p", "P", "s", "S", "Pn", "Sn", "Pg", "Sg"], 
#                 receiver_depth_in_km=grid_dep      
#             )
            
#             for arrival in arrivals:
#                 j = np.argmin(np.abs(grid_dep - arrival.receiver_depth))
                
#                 # Guard against depth-aliasing outside bounds
#                 if np.abs(grid_dep[j] - arrival.receiver_depth) > (depth_step_km * 1.5):
#                     continue
                    
#                 phase = arrival.name.upper()
#                 ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
#                 # Ensure we strictly capture the fastest first arrival path
#                 if phase in ["P", "PN", "PG"]:
#                     if arrival.time < lookup_tp[i, j]:
#                         lookup_tp[i, j] = arrival.time
#                         lookup_pp[i, j] = ray_param_sec_per_m
#                 elif phase in ["S", "SN", "SG"]:
#                     if arrival.time < lookup_ts[i, j]:
#                         lookup_ts[i, j] = arrival.time
#                         lookup_ps[i, j] = ray_param_sec_per_m
                        
#         except Exception:
#             continue

#     # =====================================================================
#     # 6. Explicit Source Position & Vertical Path Enforcement
#     # =====================================================================
#     idx_deg_zero = 0 
#     for j, dep in enumerate(grid_dep):
#         distance_km = np.abs(dep - source_depth_km)
#         # Check if we are inside the same cell block as the source anchor
#         if np.isclose(dep, source_depth_km, atol=depth_step_km * 0.1) or distance_km < 1e-4:
#             lookup_tp[idx_deg_zero, j] = 0.0
#             lookup_ts[idx_deg_zero, j] = 0.0
#         else:
#             # Seed vertical un-reflected baseline columns
#             t_p_vert = distance_km / (vp_surface / 1000.0)
#             t_s_vert = distance_km / (vs_surface / 1000.0)
            
#             # Only update if it provides a faster valid pathway than ray tracing
#             if t_p_vert < lookup_tp[idx_deg_zero, j]:
#                 lookup_tp[idx_deg_zero, j] = t_p_vert
#             if t_s_vert < lookup_ts[idx_deg_zero, j]:
#                 lookup_ts[idx_deg_zero, j] = t_s_vert
                
#         lookup_pp[idx_deg_zero, j] = 0.0
#         lookup_ps[idx_deg_zero, j] = 0.0

#     # =====================================================================
#     # 7. 2D Coordinate-Aware Grid Interpolation (Replaces Old 1D Flat Bleed)
#     # =====================================================================
#     # Convert unmapped infinity nodes to NaN values for grid data handling
#     lookup_tp[np.isinf(lookup_tp)] = np.nan
#     lookup_ts[np.isinf(lookup_ts)] = np.nan

#     x_indices, y_indices = np.indices(lookup_tp.shape)

#     for table in [lookup_tp, lookup_ts]:
#         if np.any(np.isnan(table)):
#             valid_mask = ~np.isnan(table)
#             if np.sum(valid_mask) > 0:
#                 # Map coordinate stencils in true 2D matrix layout space
#                 coords_valid = np.column_stack((x_indices[valid_mask], y_indices[valid_mask]))
#                 coords_missing = np.column_stack((x_indices[~valid_mask], y_indices[~valid_mask]))
#                 # Fill gaps using nearest 2D neighbors to preserve structural gradients
#                 table[~valid_mask] = griddata(coords_valid, table[valid_mask], coords_missing, method='nearest')

#     # =====================================================================
#     # 8. High-Speed 2D Matrix Interpolation back to the structural 3D Mesh
#     # =====================================================================
#     interp_tp = RegularGridInterpolator((grid_deg, grid_dep), lookup_tp, method='linear', bounds_error=False, fill_value=None)
#     interp_ts = RegularGridInterpolator((grid_deg, grid_dep), lookup_ts, method='linear', bounds_error=False, fill_value=None)
#     interp_pp = RegularGridInterpolator((grid_deg, grid_dep), lookup_pp, method='linear', bounds_error=False, fill_value=None)
#     interp_ps = RegularGridInterpolator((grid_deg, grid_dep), lookup_ps, method='linear', bounds_error=False, fill_value=None)

#     query_points = np.column_stack((distances_deg, target_depths_km))
    
#     tp_times = interp_tp(query_points)
#     ts_times = interp_ts(query_points)
#     p_wave_slowness = interp_pp(query_points)
#     s_wave_slowness = interp_ps(query_points)

#     # =====================================================================
#     # 9. Angle-Aware Mountain Topographic Time Corrections
#     # =====================================================================
#     p_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_wave_slowness**2))
#     s_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_wave_slowness**2))
    
#     src_correction_p = np.maximum(0.0, src_z) * p_factor_src
#     src_correction_s = np.maximum(0.0, src_z) * s_factor_src

#     node_corrections_p = np.maximum(0.0, grid_z) * p_factor_src
#     node_corrections_s = np.maximum(0.0, grid_z) * s_factor_src

#     tp_times = tp_times + src_correction_p + node_corrections_p
#     ts_times = ts_times + src_correction_s + node_corrections_s

#     # Force the precise source node location analytically to exactly 0.0
#     station_idx = np.argmin(distances_deg + np.abs(target_depths_km - source_depth_km))
#     if distances_deg[station_idx] < 1e-4:
#         tp_times[station_idx] = 0.0
#         ts_times[station_idx] = 0.0

#     return tp_times, ts_times


# def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, depths, vp):
#     """
#     Computes travel times using ObsPy TauP by mapping a structural 3D point mesh
#     to a temporary 2D distance-depth lookup table, then maps results back to xx.
#     Integrates an angle-aware slant correction using the local ray parameter (p).
    
#     Parameters:
#     -----------
#     xx : numpy.ndarray
#         Mesh points in local Cartesian coordinates (meters), shape (N, 3).
#     loc_proj : numpy.ndarray
#         Station/Source point in local Cartesian coordinates (meters).
#     taup_model : obspy.taup.TauPyModel
#         The active TauP velocity model instance.
#     ftrns2 : callable
#         Transformation function mapping Cartesian [X, Y, Z] to Geodetic LLA [Lat, Lon, Alt].
#     depths : numpy.ndarray
#         Array of true upward-increasing Cartesian velocity depths/elevations (meters).
#     vp : numpy.ndarray
#         Array of P-wave velocities (m/s) corresponding to the input depths.
#     """
#     from scipy.interpolate import RegularGridInterpolator, griddata
#     import numpy as np
    
#     # =====================================================================
#     # 1. Coordinate Mapping and Surface Geometries
#     # =====================================================================
#     X_lla = ftrns2(xx)
#     station_lla = ftrns2(loc_proj)[0]

#     # Safe fallback navigation to locate the internal v_mod data structure
#     if hasattr(taup_model.model, 'v_mod'):
#         v_mod = taup_model.model.v_mod
#     elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
#         v_mod = taup_model.model.s_mod.v_mod
#     else:
#         v_mod = taup_model.model

#     # Extract surface velocities from 1D profile and convert to m/s
#     vp_surface = v_mod.layers[0]['top_p_velocity'] * 1000.0
#     vs_surface = v_mod.layers[0]['top_s_velocity'] * 1000.0

#     # Track true elevations (Z)
#     src_z = station_lla[2]
#     grid_z = X_lla[:, 2]

#     # Convert absolute coordinates to positive downward depths in km (clip anything above sea level)
#     source_depth_km = np.maximum(0.0, -src_z) / 1000.0
#     target_depths_km = np.maximum(0.0, -grid_z) / 1000.0

#     # =====================================================================
#     # 2. Vectorized True WGS84 Great-Circle Distance Calculation
#     # =====================================================================
#     lat1 = np.radians(station_lla[0])
#     lon1 = np.radians(station_lla[1])
#     lat2 = np.radians(X_lla[:, 0])
#     lon2 = np.radians(X_lla[:, 1])

#     dlon = lon2 - lon1
#     cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
#     cos_clat = np.clip(cos_clat, -1.0, 1.0) 
#     distances_deg = np.degrees(np.arccos(cos_clat))

#     # =====================================================================
#     # 3. Dynamic Step Size Estimation (~2% Relative Uncertainty Bound)
#     # =====================================================================
#     max_deg = distances_deg.max()
#     min_dep, max_dep = target_depths_km.min(), target_depths_km.max()
#     dep_range = max_dep - min_dep

#     # Geographically accurate km-to-degree conversion via WGS84 Meridian Radius
#     mean_lat_rad = np.radians(np.mean(X_lla[:, 0]))
#     a = 6378.137          # Equatorial radius in km
#     b = 6356.7523142      # Polar radius in km
#     e_sq = (a**2 - b**2) / (a**2)
#     radius_meridian = (a * (1.0 - e_sq)) / (1.0 - e_sq * np.sin(mean_lat_rad)**2)**(1.5)
#     km_per_deg = (np.pi / 180.0) * radius_meridian

#     max_dist_km = max_deg * km_per_deg

#     # Un-skewed velocity mask matching the active mesh depth limits
#     mesh_min_dep_m = max(0.0, min_dep * 1000.0)
#     mesh_max_dep_m = max_dep * 1000.0
    
#     positive_input_depths = np.abs(depths)
#     active_depth_mask = (positive_input_depths >= mesh_min_dep_m) & (positive_input_depths <= mesh_max_dep_m)
    
#     if np.sum(active_depth_mask) == 0:
#         active_depth_mask = positive_input_depths <= np.max(positive_input_depths)
        
#     v_representative = float(np.median(vp[active_depth_mask])) / 1000.0  # Convert m/s to km/s
#     v_representative = max(1.5, v_representative)

#     target_error_fraction = 0.02  

#     # Physics-based spatial step selection
#     if max_dist_km > 0:
#         ideal_dx_km = np.sqrt(8.0 * target_error_fraction * max_dist_km * v_representative)
#         dist_step_km = np.clip(ideal_dx_km, 0.1, max_dist_km * 0.05)
#         dist_step_deg = dist_step_km / km_per_deg
#     else:
#         dist_step_deg = 0.01

#     if dep_range > 0:
#         ideal_dz_km = np.sqrt(8.0 * target_error_fraction * dep_range * v_representative)
#         depth_step_km = np.clip(ideal_dz_km, 0.1, dep_range * 0.05)
#     else:
#         depth_step_km = 0.5

#     # Build lookup axes dynamically
#     grid_deg = np.arange(0.0, max_deg + 2*dist_step_deg, dist_step_deg)
#     grid_dep = np.arange(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, depth_step_km)

#     if len(grid_deg) < 30:
#         grid_deg = np.linspace(0.0, max_deg + 2*dist_step_deg, 30)
#     if len(grid_dep) < 30:
#         grid_dep = np.linspace(max(0.0, min_dep - 2*depth_step_km), max_dep + 2*depth_step_km, 30)

#     # =====================================================================
#     # 4. Initialize Lookup Tables with Infinity (Enforces Minimum First Arrivals)
#     # =====================================================================
#     lookup_tp = np.full((len(grid_deg), len(grid_dep)), np.inf)
#     lookup_ts = np.full((len(grid_deg), len(grid_dep)), np.inf)
#     lookup_pp = np.zeros((len(grid_deg), len(grid_dep)))  
#     lookup_ps = np.zeros((len(grid_deg), len(grid_dep)))  

#     # =====================================================================
#     # 5. Populate 2D Lookup Grid via TauP Ray-Tracing
#     # =====================================================================
#     for i, deg in enumerate(grid_deg):
#         if np.isclose(deg, 0.0, atol=1e-7):
#             continue
#         try:
#             # CRITICAL FIX: UPPERCASE phase lists only to block lowercase (p, s)
#             # upgoing reflections from forcing regional travel times down to zero.
#             arrivals = taup_model.get_travel_times(
#                 source_depth_in_km=source_depth_km,
#                 distance_in_degree=deg,
#                 phase_list=["P", "S", "Pn", "Sn", "Pg", "Sg"],
#                 receiver_depth_in_km=grid_dep      
#             )
            
#             for arrival in arrivals:
#                 j = np.argmin(np.abs(grid_dep - arrival.receiver_depth))
                
#                 # Guard against depth-aliasing outside bounds
#                 if np.abs(grid_dep[j] - arrival.receiver_depth) > (depth_step_km * 1.0):
#                     continue
                    
#                 phase = arrival.name.upper()
#                 ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
#                 # Ensure we strictly capture the absolute fastest first arrival path minimum
#                 if phase in ["P", "PN", "PG"]:
#                     if arrival.time < lookup_tp[i, j]:
#                         lookup_tp[i, j] = arrival.time
#                         lookup_pp[i, j] = ray_param_sec_per_m
#                 elif phase in ["S", "SN", "SG"]:
#                     if arrival.time < lookup_ts[i, j]:
#                         lookup_ts[i, j] = arrival.time
#                         lookup_ps[i, j] = ray_param_sec_per_m
                        
#         except Exception:
#             continue

#     # =====================================================================
#     # 6. Explicit Source Position & Vertical Path Enforcement
#     # =====================================================================
#     idx_deg_zero = 0 
#     for j, dep in enumerate(grid_dep):
#         distance_km = np.abs(dep - source_depth_km)
#         if distance_km < 1e-4:
#             lookup_tp[idx_deg_zero, j] = 0.0
#             lookup_ts[idx_deg_zero, j] = 0.0
#         else:
#             # Seed vertical un-reflected baseline paths
#             t_p_vert = distance_km / (vp_surface / 1000.0)
#             t_s_vert = distance_km / (vs_surface / 1000.0)
            
#             # Only update if it provides a faster valid pathway than ray tracing
#             if t_p_vert < lookup_tp[idx_deg_zero, j]:
#                 lookup_tp[idx_deg_zero, j] = t_p_vert
#             if t_s_vert < lookup_ts[idx_deg_zero, j]:
#                 lookup_ts[idx_deg_zero, j] = t_s_vert
                
#         lookup_pp[idx_deg_zero, j] = 0.0
#         lookup_ps[idx_deg_zero, j] = 0.0

#     # =====================================================================
#     # 7. Physical 2D Coordinate-Aware Grid Interpolation (No Index Warping)
#     # =====================================================================
#     lookup_tp[np.isinf(lookup_tp)] = np.nan
#     lookup_ts[np.isinf(lookup_ts)] = np.nan

#     # Create a coordinate mesh grid matching true physical degrees and depths
#     deg_mesh, dep_mesh = np.meshgrid(grid_deg, grid_dep, indexing='ij')

#     for table in [lookup_tp, lookup_ts]:
#         if np.any(np.isnan(table)):
#             valid_mask = ~np.isnan(table)
#             if np.sum(valid_mask) > 0:
#                 # CRITICAL FIX: Feed griddata true physical coordinates (deg, km)
#                 # to prevent vertical bleeding artifacts across axis scale changes.
#                 coords_valid = np.column_stack((deg_mesh[valid_mask], dep_mesh[valid_mask]))
#                 coords_missing = np.column_stack((deg_mesh[~valid_mask], dep_mesh[~valid_mask]))
                
#                 table[~valid_mask] = griddata(coords_valid, table[valid_mask], coords_missing, method='nearest')

#     # =====================================================================
#     # 8. High-Speed 2D Matrix Interpolation back to the structural 3D Mesh
#     # =====================================================================
#     interp_tp = RegularGridInterpolator((grid_deg, grid_dep), lookup_tp, method='linear', bounds_error=False, fill_value=None)
#     interp_ts = RegularGridInterpolator((grid_deg, grid_dep), lookup_ts, method='linear', bounds_error=False, fill_value=None)
#     interp_pp = RegularGridInterpolator((grid_deg, grid_dep), lookup_pp, method='linear', bounds_error=False, fill_value=None)
#     interp_ps = RegularGridInterpolator((grid_deg, grid_dep), lookup_ps, method='linear', bounds_error=False, fill_value=None)

#     query_points = np.column_stack((distances_deg, target_depths_km))
    
#     tp_times = interp_tp(query_points)
#     ts_times = interp_ts(query_points)
#     p_wave_slowness = interp_pp(query_points)
#     s_wave_slowness = interp_ps(query_points)

#     # =====================================================================
#     # 9. Angle-Aware Mountain Topographic Time Corrections
#     # =====================================================================
#     p_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_wave_slowness**2))
#     s_factor_src = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_wave_slowness**2))
    
#     src_correction_p = np.maximum(0.0, src_z) * p_factor_src
#     src_correction_s = np.maximum(0.0, src_z) * s_factor_src

#     node_corrections_p = np.maximum(0.0, grid_z) * p_factor_src
#     node_corrections_s = np.maximum(0.0, grid_z) * s_factor_src

#     tp_times = tp_times + src_correction_p + node_corrections_p
#     ts_times = ts_times + src_correction_s + node_corrections_s

#     # Force the precise source node location analytically to exactly 0.0
#     station_idx = np.argmin(distances_deg + np.abs(target_depths_km - source_depth_km))
#     if distances_deg[station_idx] < 1e-4:
#         tp_times[station_idx] = 0.0
#         ts_times[station_idx] = 0.0

#     return tp_times, ts_times


def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, depths, vp):
    """
    Computes travel times using ObsPy TauP by querying a high-density 1D surface 
    distance array to guarantee unbroken first-arrivals, then applies a rigorous 
    multi-layer 1D integration over the velocity profile to handle depth variations.
    """
    from scipy.interpolate import interp1d, interp2d
    import numpy as np
    
    # =====================================================================
    # 1. Coordinate Mapping and Geometries
    # =====================================================================
    X_lla = ftrns2(xx)
    station_lla = ftrns2(loc_proj)[0]

    # Navigate the internal TauP 1D model profile
    if hasattr(taup_model.model, 'v_mod'):
        v_mod = taup_model.model.v_mod
    elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
        v_mod = taup_model.model.s_mod.v_mod
    else:
        v_mod = taup_model.model

    # Extract all layers from the 1D model structure
    model_depths_km = []
    model_vp_km_s = []
    model_vs_km_s = []
    
    for layer in v_mod.layers:
        model_depths_km.append(layer['top_depth'])
        model_vp_km_s.append(layer['top_p_velocity'])
        model_vs_km_s.append(layer['top_s_velocity'])
    
    # Append bottom depth boundary to close out vectors
    model_depths_km.append(v_mod.layers[-1]['bot_depth'])
    model_vp_km_s.append(v_mod.layers[-1]['bot_p_velocity'])
    model_vs_km_s.append(v_mod.layers[-1]['bot_s_velocity'])
    
    model_depths_m = np.array(model_depths_km) * 1000.0
    model_vp_m_s = np.array(model_vp_km_s) * 1000.0
    model_vs_m_s = np.array(model_vs_km_s) * 1000.0

    vp_surface = model_vp_m_s[0]
    vs_surface = model_vs_m_s[0]

    src_z = station_lla[2]
    grid_z = X_lla[:, 2]

    source_depth_m = np.maximum(0.0, -src_z)
    target_depths_m = np.maximum(0.0, -grid_z)

    # =====================================================================
    # 2. Vectorized True WGS84 Great-Circle Distance Calculation (Degrees)
    # =====================================================================
    lat1, lon1 = np.radians(station_lla[0]), np.radians(station_lla[1])
    lat2, lon2 = np.radians(X_lla[:, 0]), np.radians(X_lla[:, 1])

    dlon = lon2 - lon1
    cos_clat = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    cos_clat = np.clip(cos_clat, -1.0, 1.0) 
    distances_deg = np.degrees(np.arccos(cos_clat))

    # =====================================================================
    # 3. Dense 1D Surface Distance Grid Seeding
    # =====================================================================
    max_deg = max(0.01, distances_deg.max())
    dense_deg_axis = np.linspace(0.0, max_deg * 1.05, 300)
    
    surf_tp = np.full_like(dense_deg_axis, np.inf)
    surf_ts = np.full_like(dense_deg_axis, np.inf)
    surf_pp = np.zeros_like(dense_deg_axis)
    surf_ps = np.zeros_like(dense_deg_axis)

    surf_tp[0] = 0.0
    surf_ts[0] = 0.0

    # =====================================================================
    # 4. Pure Surface First-Arrival Ray Population
    # =====================================================================
    for i, deg in enumerate(dense_deg_axis):
        if i == 0:
            continue
        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=source_depth_m / 1000.0,
                distance_in_degree=deg,
                phase_list=["P", "S", "Pn", "Sn", "Pg", "Sg"]
            )
            
            for arrival in arrivals:
                phase = arrival.name.upper()
                ray_param_sec_per_m = arrival.ray_param / 6371000.0
                
                if phase in ["P", "PN", "PG"]:
                    if arrival.time < surf_tp[i]:
                        surf_tp[i] = arrival.time
                        surf_pp[i] = ray_param_sec_per_m
                elif phase in ["S", "SN", "SG"]:
                    if arrival.time < surf_ts[i]:
                        surf_ts[i] = arrival.time
                        surf_ps[i] = ray_param_sec_per_m
        except Exception:
            continue

    # Clean up unpopulated ray slots via 1D linear interpolation
    for surf_table in [surf_tp, surf_ts]:
        inf_mask = np.isinf(surf_table)
        if np.any(inf_mask):
            surf_table[inf_mask] = np.interp(
                np.flatnonzero(inf_mask), np.flatnonzero(~inf_mask), surf_table[~inf_mask]
            )

    # Map surface values back to individual mesh point distances
    interp_surf_tp = interp1d(dense_deg_axis, surf_tp, kind='linear', fill_value="extrapolate")
    interp_surf_ts = interp1d(dense_deg_axis, surf_ts, kind='linear', fill_value="extrapolate")
    interp_surf_pp = interp1d(dense_deg_axis, surf_pp, kind='linear', fill_value="extrapolate")
    interp_surf_ps = interp1d(dense_deg_axis, surf_ps, kind='linear', fill_value="extrapolate")

    tp_base = interp_surf_tp(distances_deg)
    ts_base = interp_surf_ts(distances_deg)
    p_slowness = interp_surf_pp(distances_deg)
    s_slowness = interp_surf_ps(distances_deg)

    # =====================================================================
    # 5. Rigorous Multi-Layer Vertical Integration Correction
    # =====================================================================
    # We evaluate the cumulative vertical integral layer-by-layer for every single node point
    subsurface_correction_p = np.zeros_like(target_depths_m)
    subsurface_correction_s = np.zeros_like(target_depths_m)

    for idx in range(len(model_depths_m) - 1):
        z_top = model_depths_m[idx]
        z_bot = model_depths_m[idx + 1]
        v_p_layer = model_vp_m_s[idx]
        v_s_layer = model_vs_m_s[idx]

        # Determine which nodes have target depths that intercept this layer interval
        active_nodes = target_depths_m > z_top
        if not np.any(active_nodes):
            continue

        # Calculate the vertical distance segment traversing this layer block
        thickness_in_layer = np.minimum(target_depths_m[active_nodes], z_bot) - z_top

        # Evaluate the dynamic vertical slowness components (eta)
        eta_p = np.sqrt(np.maximum(0.0, (1.0 / v_p_layer**2) - p_slowness[active_nodes]**2))
        eta_s = np.sqrt(np.maximum(0.0, (1.0 / v_s_layer**2) - s_slowness[active_nodes]**2))

        # Add the integrated time chunk to the running node totals
        subsurface_correction_p[active_nodes] += thickness_in_layer * eta_p
        subsurface_correction_s[active_nodes] += thickness_in_layer * eta_s

    # =====================================================================
    # 6. Station & Topography Angular Multipliers (Using Surface Velocities)
    # =====================================================================
    p_factor_surf = np.sqrt(np.maximum(0.0, (1.0 / vp_surface**2) - p_slowness**2))
    s_factor_surf = np.sqrt(np.maximum(0.0, (1.0 / vs_surface**2) - s_slowness**2))

    src_correction_p = np.maximum(0.0, src_z) * p_factor_surf
    src_correction_s = np.maximum(0.0, src_z) * s_factor_surf

    node_topo_p = np.maximum(0.0, grid_z) * p_factor_surf
    node_topo_s = np.maximum(0.0, grid_z) * s_factor_surf

    # =====================================================================
    # 7. Consolidated Assembly
    # =====================================================================
    tp_times = tp_base + src_correction_p + node_topo_p + subsurface_correction_p
    ts_times = ts_base + src_correction_s + node_topo_s + subsurface_correction_s

    # Secure source coordinate anchor origin cleanly at 0.0
    station_idx = np.argmin(distances_deg + np.abs(target_depths_m - source_depth_m))
    if distances_deg[station_idx] < 1e-4:
        tp_times[station_idx] = 0.0
        ts_times[station_idx] = 0.0

    return tp_times, ts_times


def create_custom_taup_model(depths, vp, vs, model_name):
    import os
    import numpy as np
    from obspy.taup.taup_create import build_taup_model
    from obspy.taup import TauPyModel
    import obspy.taup

    # 1. Process your custom shallow crustal model (keep depths <= 0)
    subsurface_mask = depths <= 0
    sub_depths = -depths[subsurface_mask]  # Convert to positive downward
    
    # Reverse arrays so they start at the surface (0) and go deep
    taup_depths = sub_depths[::-1]
    taup_vp = vp[subsurface_mask][::-1] / 1000.0  # Convert m/s to km/s
    taup_vs = vs[subsurface_mask][::-1] / 1000.0  # Convert m/s to km/s
    taup_depths[0] = 0.0  # Snap nearest surface node to zero
    
    # DYNAMICALLY DETECT DEPTH PARAMETERS
    max_regional_depth_km = taup_depths[-1] / 1000.0
    total_profile_length_km = max_regional_depth_km
    
    # Define a 10% blending window at the bottom of your regional profile
    blend_window_km = 0.10 * total_profile_length_km
    blend_start_km = max_regional_depth_km - blend_window_km

    # 2. Extract iasp91 values to compute the smooth blend zone
    iasp91_obj = TauPyModel(model="iasp91").model
    iasp91_v_mod = iasp91_obj.s_mod.v_mod if hasattr(iasp91_obj, 's_mod') else iasp91_obj
    
    iasp91_depths = np.array([layer['top_depth'] for layer in iasp91_v_mod.layers])
    iasp91_vp = np.array([layer['top_p_velocity'] for layer in iasp91_v_mod.layers])
    iasp91_vs = np.array([layer['top_s_velocity'] for layer in iasp91_v_mod.layers])

    # 3. Open and write the hybrid .tvel file
    tvel_filename = model_name + ".tvel"
    with open(tvel_filename, 'w') as f:
        f.write(f"Model: Hybrid_{os.path.basename(model_name)}\n")
        f.write("6371.0\n")  # Standard Earth Radius
        
        # Write regional profile, applying the smooth blend in the final 10% section
        for d_m, vp_kms, vs_kms in zip(taup_depths, taup_vp, taup_vs):
            d_km = d_m / 1000.0
            
            if d_km >= blend_start_km:
                # Calculate linear blend weight (0 at blend_start, 1 at max_regional_depth)
                weight = (d_km - blend_start_km) / blend_window_km
                weight = np.clip(weight, 0.0, 1.0)
                
                # Sample matching iasp91 values at this exact depth via interpolation
                iasp91_vp_val = np.interp(d_km, iasp91_depths, iasp91_vp)
                iasp91_vs_val = np.interp(d_km, iasp91_depths, iasp91_vs)
                
                # Linear combination blend
                final_vp = (1.0 - weight) * vp_kms + weight * iasp91_vp_val
                final_vs = (1.0 - weight) * vs_kms + weight * iasp91_vs_val
            else:
                final_vp = vp_kms
                final_vs = vs_kms
                
            f.write(f"{d_km:7.3f} {final_vp:7.3f} {final_vs:7.3f} 2.700\n")
            
        # 4. SAFELY APPEND THE DEEP EARTH FROM OBSPY'S RAW IASP91 SOURCE FILE
        # Find where obspy keeps its built-in .tvel text databases
        taup_dir = os.path.dirname(obspy.taup.__file__)
        iasp91_src_path = os.path.join(taup_dir, "data", "iasp91.tvel")
        
        with open(iasp91_src_path, 'r') as src_f:
            for line in src_f:
                tokens = line.strip().split()
                # Skip blank lines or headers
                if not tokens or len(tokens) < 3:
                    continue
                try:
                    # Check the depth entry of the line (first column)
                    line_depth_km = float(tokens[0])
                    # If it's deeper than your regional floor, copy the line exactly!
                    if line_depth_km > max_regional_depth_km:
                        f.write(line)
                except ValueError:
                    # Skips text lines (headers like "iasp91")
                    continue
            
    # Compile the hybrid model cleanly
    build_taup_model(tvel_filename)
    if os.path.exists(tvel_filename):
        os.remove(tvel_filename)
        
    base_model_name = os.path.basename(model_name)
    return TauPyModel(model=base_model_name)



# Load configuration from YAML
config = load_config('config.yaml')
name_of_project = config['name_of_project']
# num_cores = config['num_cores']

# --- INITIALIZE THE ENGINE IN YOUR CONFIG LOADING SECTION ---
engine_type = config.get('engine_type', 'eikonal') # 'eikonal' or 'taup'

## Load travel times (train regression model, elsewhere, or, load and "initilize" 1D interpolator method)
path_to_file = str(pathlib.Path().absolute())
seperator =  '\\' if '\\' in path_to_file else '/'
path_to_file += seperator

# template_ver = 1
vel_model_type = config['vel_model_type']
use_topography = config['use_topography']
vel_model_ver = config.get('vel_model_ver', 1)

# Load region
z = np.load(path_to_file + '%s_region.npz'%name_of_project)
lat_range, lon_range, depth_range, deg_pad = z['lat_range'], z['lon_range'], z['depth_range'], z['deg_pad']
z.close()

# Load stations
z = np.load(path_to_file + '%s_stations.npz'%name_of_project)
locs, stas, mn, rbest = z['locs'], z['stas'], z['mn'], z['rbest']
z.close()

lat_range_extend = [lat_range[0] - deg_pad, lat_range[1] + deg_pad]
lon_range_extend = [lon_range[0] - deg_pad, lon_range[1] + deg_pad]

## Overwrite range based on station locations and buffer
d_pad = deg_pad # 0.15
lat_range_extend = [np.minimum(lat_range_extend[0], locs[:,0].min() - d_pad), np.maximum(lat_range_extend[1], locs[:,0].max() + d_pad)]
lon_range_extend = [np.minimum(lon_range_extend[0], locs[:,1].min() - d_pad), np.maximum(lon_range_extend[1], locs[:,1].max() + d_pad)]

scale_x = np.array([lat_range[1] - lat_range[0], lon_range[1] - lon_range[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x = np.array([lat_range[0], lon_range[0], depth_range[0]]).reshape(1,-1)
scale_x_extend = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
offset_x_extend = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)

ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T # map (lat,lon,depth) into local cartesian (x || East,y || North, z || Outward)
ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)  # invert ftrns1

vs = np.array(config['velocity_model']['Vs']) if vel_model_type == 1 else np.load(path_to_file + '3d_velocity_model.npz')['Vs']
vs_min = np.quantile(vs.reshape(-1), 0.2)
query_proj = ftrns1(locs)

## Determine grid resolution

n_jobs = config['n_jobs']
n_batch = int(np.ceil(len(locs)/n_jobs))
ind_use = [np.arange(n_batch) + n_batch*i for i in range(n_jobs)]
if n_jobs > 1:
	ind_use[-1] = np.arange(ind_use[-2][-1] + 1, len(locs))
ind_use = ind_use[int(argvs[1])]


n_optimal_points = np.array([250, 250, 125])
n_optimal_points = config.get('target_grid_resolution', n_optimal_points) # np.array([300, 300, 150])
cpu_point_budget = np.prod(n_optimal_points)
hardware_safe_cores = estimate_safe_cores(cpu_point_budget, safety_factor=0.8)
num_cores = min(config['num_cores'], hardware_safe_cores)

## Load velocity model
if engine_type == 'taup':
	print('Using 1D velocity model and TauP' if vel_model_type == 1 else 'Using 1D velocity model and TauP (overwritting vel_model_type)')
	vel_model_type = 1

if vel_model_type == 1:
	vp = np.array(config['velocity_model']['Vp'])
	vs = np.array(config['velocity_model']['Vs'])
	x_vel = np.array(config['velocity_model']['Depths'])
	vs_min = np.quantile(vs, 0.2)

	if engine_type == 'taup':
	    # Pre-compile your 1D velocity model once before entering the loop
	    taup_model = create_custom_taup_model(x_vel, vp, vs, model_name= '1D_Velocity_Models_Regional' + seperator + 'regional_%d_1d'%int(argvs[1]))

else:

	z = np.load(path_to_file + '3d_velocity_model.npz')
	x_vel, vp, vs = z['X'], z['Vp'], z['Vs'] ## lat, lon, depth (x_vel) and velocity values
	z.close()
	vs_min = np.quantile(vs, 0.2)


def initilize_velocity_model(x, vp, vs, xx, dx_res, vel_type = 1):

	if vel_type == 1:

		iarg = np.argsort(x)

		dx_depth = dx_res[2] # config.get('dx_depth', dx)
		depths_fine = np.arange(x.min(), x.max() + dx_depth/10.0, dx_depth/10.0)
		vp_fine = np.interp(depths_fine, x[iarg], vp[iarg])
		vs_fine = np.interp(depths_fine, x[iarg], vs[iarg])

		tree = cKDTree(depths_fine.reshape(-1,1))
		ip_nearest = tree.query(ftrns2(xx)[:,2].reshape(-1,1))[1]
		Vp = vp_fine[ip_nearest]
		Vs = vs_fine[ip_nearest]

		# tree = cKDTree(depths_fine.reshape(-1,1))
		# ip_nearest = tree.query(ftrns2(xx)[:,2].reshape(-1,1))[1]
		# Vp = vp_fine[ip_nearest]
		# Vs = vs_fine[ip_nearest]

		return Vp, Vs

	else:

		tree = cKDTree(ftrns1(x)) ## Assigns the velocity values to the computation grid (xx) using nearest neighbors (e.g., the input 3D model can include any number of points, anywhere, and interpolation will fill in the values elsewhere)
		ip_nearest = tree.query(xx)[1]
		Vp = vp[ip_nearest]
		Vs = vs[ip_nearest]

		return Vp, Vs


if (use_topography == True)*(os.path.isfile(path_to_file + 'surface_elevation.npz') == True):

	## Load "Points" field that specifies surface elevation (columns of lat, lon, elevation (meters)). Points outside convex hull of Points will be treated as zero elevation.
	z = np.load(path_to_file + 'surface_elevation.npz')
	Points = z['Points']
	z.close()
	## Concatenate station elevations
	Points = np.concatenate((Points, locs), axis = 0)

	station_tree = cKDTree(ftrns1(locs))
	nn_distances, _ = station_tree.query(ftrns1(locs), k=2)
	nearest_neighbor_distances = nn_distances[:, 1]
	# Set the baseline resolution to a conservative fraction of the average station spacing
	# (e.g., Average spacing divided by 4, clamped safely between 250m and 2000m)
	average_station_spacing = np.mean(nearest_neighbor_distances)
	baseline_dx = np.clip(average_station_spacing / 4.0, 10.0, 10000.0)

	d_deg = baseline_dx/110e3
	## First interpolate uniform surface over all lat-lon based on Points (fill in missing values as sea level)
	tree = cKDTree(ftrns1(Points*np.array([1.0, 1.0, 0.0]).reshape(1,-1)))

	x1_s, x2_s = np.arange(lat_range_extend[0], lat_range_extend[1] + d_deg/5.0, d_deg/5.0), np.arange(lon_range_extend[0], lon_range_extend[1] + d_deg/5.0, d_deg/5.0)
	x11_s, x12_s = np.meshgrid(x1_s, x2_s)
	surface_profile = np.concatenate((x11_s.reshape(-1,1), x12_s.reshape(-1,1)), axis = 1)
	ip_match = tree.query(ftrns1(np.concatenate((surface_profile, np.zeros((len(surface_profile),1))), axis = 1)))
	val = Points[ip_match[1],2] ## Surface elevations of regular grid
	hull = ConvexHull(Points[:,0:2])
	ioutside_hull = np.where(in_hull(surface_profile,  hull.points[hull.vertices]) == 0)[0]
	val[ioutside_hull] = 0.0 ## Setting points on regular grid far from reference points to sea level
	surface_profile = np.concatenate((surface_profile, val.reshape(-1,1)), axis = 1)
	if os.path.isfile(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project) == False:
		np.savez_compressed(path_to_file + 'Grids/%s_surface_elevation.npz'%name_of_project, surface_profile = surface_profile)
		
	## Check if stations are beneath surface
	tol_elev_val = 150.0 ## Stations must be within 100 meters of being beneath surface or else assume there is an error
	tree = cKDTree(ftrns1(surface_profile))
	unit_out = ftrns1(locs + np.concatenate((np.zeros((len(locs),2)), 1.0*np.ones((len(locs),1))), axis = 1))
	dist_near = tree.query(ftrns1(locs))[0]
	dist_perturb = tree.query(unit_out)[0]
	iabove_surface = np.where(dist_perturb > dist_near)[0]
	if len(iabove_surface) > 0: assert(np.abs(locs[iabove_surface,2] - surface_profile[tree.query(ftrns1(locs))[1][iabove_surface],2]).max() < tol_elev_val)


include_random_samples = True
if include_random_samples == True:
	ind_use_rand = np.random.choice(len(locs), size = int(3*len(ind_use)))
	locs_nn = cKDTree(ftrns1(locs)).query(ftrns1(locs[ind_use_rand]), k = 2)[0][:,1]
	locs_rand = ftrns2(locs_nn.reshape(-1,1)*np.random.randn(len(ind_use_rand),3) + ftrns1(locs[ind_use_rand]))
	irand = np.sort(np.random.choice(len(ind_use_rand), size = int(np.floor(len(ind_use_rand)/2)), replace = False)) if int(np.floor(len(ind_use_rand)/2)) > 0 else np.zeros(0).astype('int')
	locs_rand[irand] = np.random.uniform(offset_x_extend, offset_x_extend + scale_x_extend)
	locs_rand = locs_rand.clip(offset_x_extend, offset_x_extend + scale_x_extend)
	ind_use = np.concatenate((ind_use, np.arange(len(ind_use_rand)) + len(locs)), axis = 0)
	locs_concat = np.concatenate((locs, locs_rand), axis = 0)
else:
	locs_concat = np.copy(locs)



for sta_ind in ind_use:

	# loc_proj = ftrns1(locs[sta_ind].reshape(1,-1))
	loc_proj = ftrns1(locs_concat[sta_ind].reshape(1,-1))
	max_dist = np.linalg.norm(query_proj - loc_proj, axis = 1) ## Create grid centered on point with this radius

	print('\nLat range: %0.2f, %0.2f'%(lat_range_extend[0], lat_range_extend[1]))
	print('Lon range: %0.2f, %0.2f'%(lon_range_extend[0], lon_range_extend[1]))
	print('Depth range: %0.2f, %0.2f \n'%(depth_range[0], depth_range[1]))

	# === PRE-COMPUTE GEOMETRY SPANS FOR THE OPTIMIZER ===
	elev = locs[:,2].max() + 1000.0
	z_corners = np.array([
		[lat_range_extend[0], lon_range_extend[0], elev],
		[lat_range_extend[1], lon_range_extend[1], depth_range[0]]
	])
	zz_corners = ftrns1(z_corners)
	
	# Extract true ground lengths in meters directly tracking polar warping
	regional_span_x1 = float(np.abs(zz_corners[1, 0] - zz_corners[0, 0])) # East-West Span
	regional_span_x2 = float(np.abs(zz_corners[1, 1] - zz_corners[0, 1])) # North-South Span
	regional_span_x3 = float(np.abs(zz_corners[1, 2] - zz_corners[0, 2])) # Vertical Span

	target_error = 0.02
	if sta_ind < len(locs): ## Regular resolution
		n_optimal_points1 = np.prod(n_optimal_points)
	else:
		n_optimal_points1 = int(np.prod(n_optimal_points)/8)	

	# Run the geometrically updated optimizer
	optim, (opt_R_max1, opt_R_max2) = optimize_grid_resolutions(
		vs_min, 
		target_error = target_error, 
		span_x = regional_span_x1,
		span_y = regional_span_x2,
		span_z = regional_span_x3,
		cpu_point_budget = n_optimal_points1
	)
	print('Optimized dx settings:', optim)
	# print('Domain: %0.4f, %0.4f, %0.4f'%(regional_span_x1, regional_span_x2, regional_span_x3))

	data = {}
	data['res'] = optim
	data['loc'] = locs_concat[sta_ind].reshape(1,-1)
	data['loc_proj'] = loc_proj
	data['engine'] = engine_type


	for inc_res, dx_res in enumerate(optim):

		# if inc_res == (len(optim) - 1):
		# 	dx_res = dx_res*1.25

		## Boundary of domain, in Cartesian coordinates
		elev = locs[:,2].max() + 1000.0
		z1 = np.array([lat_range_extend[0], lon_range_extend[0], elev])[None,:]
		z2 = np.array([lat_range_extend[0], lon_range_extend[1], elev])[None,:]
		z3 = np.array([lat_range_extend[1], lon_range_extend[1], elev])[None,:]
		z4 = np.array([lat_range_extend[1], lon_range_extend[0], elev])[None,:]
		z5 = np.array([np.mean(lat_range_extend).item(), lon_range_extend[0], elev])[None,:]
		z6 = np.array([np.mean(lat_range_extend).item(), lon_range_extend[1], elev])[None,:]
		z7 = np.array([lat_range_extend[0], np.mean(lon_range_extend).item(), elev])[None,:]
		z8 = np.array([lat_range_extend[1], np.mean(lon_range_extend).item(), elev])[None,:]
		z9 = np.array([np.mean(lat_range_extend).item(), np.mean(lon_range_extend).item(), elev])[None,:]

		# z6 = np.array([lat_range_extend[0], lon_range_extend[1], elev])[None,:]
		z = np.concatenate((z1, z2, z3, z4, z5, z6, z7, z8, z9), axis = 0)
		zz = ftrns1(z)

		# === BEGIN GEOGRAPHICALLY AWARE NODE ESTIMATION ===
		# 1. Map the absolute corners of your extended regional domain into your local Cartesian system
		corners_lla = np.array([
			[lat_range_extend[0], lon_range_extend[0], zz[:,2].max()],
			[lat_range_extend[1], lon_range_extend[1], depth_range[0]]
		])
		corners_xyz = ftrns1(corners_lla)

		# 2. Calculate the true physical dimensions of the regional domain in meters
		regional_span_x1 = float(np.abs(corners_xyz[1, 0] - corners_xyz[0, 0])) # True East-West span
		regional_span_x2 = float(np.abs(corners_xyz[1, 1] - corners_xyz[0, 1])) # True North-South span
		regional_span_x3 = float(np.abs(corners_xyz[1, 2] - corners_xyz[0, 2])) # True Vertical span		

		# 3. Scale down BOTH horizontal and vertical spans for local fine tiers
		if inc_res == 0:
			span_x1 = np.minimum(2.0 * opt_R_max1, regional_span_x1)
			span_x2 = np.minimum(2.0 * opt_R_max1, regional_span_x2)
			# Capping depth: Look down only as far as the tier's horizontal range allows
			span_x3 = np.minimum(2.0 * opt_R_max1, regional_span_x3)
		elif inc_res == 1:
			span_x1 = np.minimum(2.0 * opt_R_max2, regional_span_x1)
			span_x2 = np.minimum(2.0 * opt_R_max2, regional_span_x2)
			span_x3 = np.minimum(2.0 * opt_R_max2, regional_span_x3)
		else:
			# # Tier 3 captures the full deep regional footprint
			# span_x1 = regional_span_x1
			# span_x2 = regional_span_x2
			# span_x3 = regional_span_x3
			# --- FIXED: TIER 3 BUDGET PROTECTION FOR VIRTUAL STATIONS ---
			if sta_ind < len(locs):
				# Regular stations capture the full deep regional footprint
				span_x1 = regional_span_x1
				span_x2 = regional_span_x2
				span_x3 = regional_span_x3
			else:
				# Random stations get a physically cropped Tier 3 window 
				# (e.g., restricted to a reasonable multi-resolution tracking radius)
				span_x1 = np.minimum(4.0 * opt_R_max2, regional_span_x1)
				span_x2 = np.minimum(4.0 * opt_R_max2, regional_span_x2)
				span_x3 = np.minimum(4.0 * opt_R_max2, regional_span_x3)


		# # 4. Derive grid counts ensuring dx == dy == dz (Perfect cubes)
		# n1_target = int(np.ceil(span_x1 / dx_res)) + 8
		# n2_target = int(np.ceil(span_x2 / dx_res)) + 8
		# n3_target = int(np.ceil(span_x3 / dx_res)) + 8

		# # 5. Force odd dimensions so the station source lands perfectly on a node center
		# n1 = n1_target + 1 if n1_target % 2 == 0 else n1_target
		# n2 = n2_target + 1 if n2_target % 2 == 0 else n2_target
		# n3 = n3_target + 1 if n3_target % 2 == 0 else n3_target


		# 4. Derive grid counts ensuring dx == dy == dz (Perfect cubes)
		n1_target = int(np.ceil(span_x1 / dx_res)) + 8
		n2_target = int(np.ceil(span_x2 / dx_res)) + 8
		
		# # --- ASYMMETRIC OVERRIDE FOR TIER 3 ---
		# use_asymmetric_depths = True
		# if (inc_res == (len(optim) - 1))*(use_asymmetric_depths == True):  # Assuming Tier 3 is index 2
		# 	# Cap the vertical step size to a much finer value (e.g., max 2000 meters)
		# 	dz_res = min(2000.0, dx_res)
		# 	n3_target = int(np.ceil(span_x3 / dz_res)) + 8
		# else:
		# 	dz_res = dx_res
		# 	n3_target = int(np.ceil(span_x3 / dx_res)) + 8

		use_asymmetric_depths = True
		if (inc_res == (len(optim) - 1)) and use_asymmetric_depths:
			# Define your structural limits dynamically:
			# 1. Guarantee a minimum number of vertical nodes (e.g., at least 60 layers)
			min_vertical_layers = 60 
			dz_from_layers = span_x3 / min_vertical_layers
			
			# 2. Prevent the cell aspect ratio (dx/dz) from exceeding a limit (e.g., max 4:1)
			max_aspect_ratio = 4.0
			dz_from_aspect = dx_res / max_aspect_ratio
			
			# Pick the finer (smaller) dz of the two constraints, but never make it coarser than dx_res
			dz_res = min(dz_from_layers, dz_from_aspect, dx_res)
			
			# Enforce a sanity floor so dz doesn't become micro-scale (e.g., never sharper than Tier 2's dx)
			if inc_res > 0:
				dz_res = max(dz_res, optim[inc_res - 1]) 
				
			n3_target = int(np.ceil(span_x3 / dz_res)) + 8
		else:
			dz_res = dx_res
			n3_target = int(np.ceil(span_x3 / dx_res)) + 8

		# 5. Force odd dimensions so the station source lands perfectly on a node center
		n1 = n1_target + 1 if n1_target % 2 == 0 else n1_target
		n2 = n2_target + 1 if n2_target % 2 == 0 else n2_target
		n3 = n3_target + 1 if n3_target % 2 == 0 else n3_target


		# --- UNIFIED MESH GRID GENERATION ---
		x1 = np.linspace(0, n1 - 1, n1) * dx_res
		x2 = np.linspace(0, n2 - 1, n2) * dx_res
		x3 = np.linspace(0, n3 - 1, n3) * dz_res

		if inc_res < (len(optim) - 1) or sta_ind >= len(locs):
			x1 = (x1 - x1.mean()) + loc_proj[0,0]
			x2 = (x2 - x2.mean()) + loc_proj[0,1]
		else:
			domain_center_xyz = corners_xyz.mean(axis=0)
			x1 = (x1 - x1.mean()) + domain_center_xyz[0]
			x2 = (x2 - x2.mean()) + domain_center_xyz[1]

			snap_idx_x1 = np.argmin(np.abs(x1 - loc_proj[0,0]))
			snap_idx_x2 = np.argmin(np.abs(x2 - loc_proj[0,1]))
			
			x1 = x1 + (loc_proj[0,0] - x1[snap_idx_x1])
			x2 = x2 + (loc_proj[0,1] - x2[snap_idx_x2])

		x3 = (x3 - x3.mean()) + loc_proj[0,2]
		x3 = x3 - x3.max() + elev
		
		inearest = np.argmin(np.abs(x3 - loc_proj[0,2]))
		x3 = x3 - (x3[inearest] - loc_proj[0,2])

		# Matrix indexing assigned properly for asymmetric cases
		x11, x12, x13 = np.meshgrid(x1, x2, x3, indexing='ij')
		xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis=1)
		X = ftrns2(xx)

		print('\nDomain: %0.4f, %0.4f, %0.4f'%(dx_res*len(x1), dx_res*len(x2), dz_res*len(x3)))


		idx_x1_src = np.argmin(np.abs(x1 - loc_proj[0,0]))
		idx_x2_src = np.argmin(np.abs(x2 - loc_proj[0,1]))
		
		assert(np.allclose(np.array([x1[idx_x1_src], x2[idx_x2_src], x3[inearest]]), loc_proj[0,:], atol = 1e-3)) 
		# src_index = (idx_x2_src * n1 * n3) + (idx_x1_src * n3) + inearest

		# OPTION A: The Stride Math Fix (Strictly matching indexing='ij')
		src_index = (idx_x1_src * n2 * n3) + (idx_x2_src * n3) + inearest
		# OPTION B: The Bulletproof Geometric Fix (Recommended)
		# Directly finds whichever row in xx physically matches the target source point
		src_index1 = np.argmin(np.linalg.norm(xx - loc_proj, axis=1))
		assert(src_index == src_index1)


		# =====================================================================
		# EXECUTION BACKEND BRANCH
		# =====================================================================
		if engine_type == 'taup':
			# Resolve times using the 1D model interpolator mapped to the 3D grid
			tp_flattened, ts_flattened = compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, -1.0*x_vel, vp)
			tp_flattened = np.asarray(tp_flattened).ravel()
			ts_flattened = np.asarray(ts_flattened).ravel()


			# 2. OVERRIDE VP AND VS FOR THE PINN TO ENSURE 100% MATHEMATICAL CONSISTENCY
			if hasattr(taup_model.model, 'v_mod'):
				compiled_v_mod = taup_model.model.v_mod
			else:
				compiled_v_mod = taup_model.model.s_mod.v_mod

			taup_internal_depths = np.array([layer['top_depth'] for layer in compiled_v_mod.layers])
			taup_internal_vp = np.array([layer['top_p_velocity'] for layer in compiled_v_mod.layers]) * 1000.0 # to m/s
			taup_internal_vs = np.array([layer['top_s_velocity'] for layer in compiled_v_mod.layers]) * 1000.0 # to m/s

			# Extract baseline surface velocities directly from the top layer for mountain zones
			vp_surface_val = taup_internal_vp[0]
			vs_surface_val = taup_internal_vs[0]

			# Safe interpolation handling for duplicate discontinuity depths
			for idx in range(1, len(taup_internal_depths)):
				if taup_internal_depths[idx] <= taup_internal_depths[idx - 1]:
					taup_internal_depths[idx] = taup_internal_depths[idx - 1] + 1e-6

			# Map your xx grid points to true LLA to find their absolute elevations (Z)
			X_lla_mesh = ftrns2(xx)
			true_z = X_lla_mesh[:, 2]

			# Convert your increasing-upward Z coordinate to TauP's positive-downward depth in km
			# ONLY clip for the subsurface interpolator
			subsurface_depths_km = np.maximum(0.0, -true_z) / 1000.0

			# 1D Interpolate the standard subsurface velocities onto your 3D mesh points
			Vp = np.interp(subsurface_depths_km, taup_internal_depths, taup_internal_vp)
			Vs = np.interp(subsurface_depths_km, taup_internal_depths, taup_internal_vs)

			# --- TOPOGRAPHIC ELEVATION FORCE OVERRIDE ---
			# For any point sitting above sea level (true_z > 0), explicitly override 
			# its velocity to match the surface correction parameters used in TauP slant math
			mountain_mask = true_z > 0
			Vp[mountain_mask] = vp_surface_val
			Vs[mountain_mask] = vs_surface_val


		else:

			Vp, Vs = initilize_velocity_model(x_vel, vp, vs, xx, [dx_res, dx_res, dz_res], vel_type=vel_model_type)
			print('dx_v %0.4f %0.4f %0.4f \n'%(dx_res, dx_res, dz_res))
			# print([dx_res, dx_res, dz_res])

			# Run parallel Eikonal solver explicitly with matrix grids
			tp_grid, ts_grid = compute_travel_times_parallel(xx, loc_proj, Vp, Vs, [dx_res, dx_res, dz_res], x11, x12, x13, num_cores=num_cores)
			tp_flattened = tp_grid.reshape(-1)
			ts_flattened = ts_grid.reshape(-1)

		# Pack answers into unified format for downstream steps
		results = [tp_flattened[:, None], ts_flattened[:, None]]
		assert(np.isclose(results[0][src_index,0], 0.0, atol=1e-2))
		assert(np.isclose(results[1][src_index,0], 0.0, atol=1e-2))




		sample_points = True
		if sample_points == True:

			scale_factor = len(locs)/25
			n_zero_inputs = int(int(100000/scale_factor) / (len(optim) - 1))
			n_per_station = int(int(150000/scale_factor) / (len(optim) - 1))
			n_per_station1 = int(int(100000/scale_factor) / (len(optim) - 1))

			if sta_ind >= len(locs):
				n_zero_inputs = int(n_zero_inputs/3)
				n_per_station = int(n_per_station/3)
				n_per_station1 = int(n_per_station1/3)

			# # p = np.zeros(locs_ref.shape[0])
			# p[sta_ind] = 1
			# isample = np.sort(np.random.choice(len(p), size = n_zero_inputs, p = p/p.sum(), replace = True))

			p = 1.0/np.maximum(results[0][:,0], 0.1)
			isample = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station, len(p)), p = p/p.sum(), replace = False))

			p = (1.0/np.maximum(results[0][:,0], 0.1))**2
			isample1 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, len(p)), p = p/p.sum(), replace = False))

			# grab_near_boundaries_samples
			p = 1.0/np.maximum(results[0].max() - results[0][:,0], 0.1)
			isample2 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, len(p)), p = p/p.sum(), replace = False))

			# grab_interior_samples
			p = 1.0*np.ones(X.shape[0]) # /np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
			isample3 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, len(p)), p = p/p.sum(), replace = False))
			# isample_vald = np.random.choice(np.delete(np.arange(len(p)), np.unique(np.concatenate((isample, isample1, isample2, isample3), axis = 0)), axis = 0), size = n_per_station)
			
			isample = np.random.permutation(np.concatenate((isample, isample1, isample2, isample3), axis = 0))
			isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = n_per_station)

			use_within_region = True
			if use_within_region == True:
				ikeep = np.where((X[isample][:,0] < (lat_range_extend[1] + deg_pad))*(X[isample][:,0] > (lat_range_extend[0] - deg_pad))*(X[isample][:,1] < (lon_range_extend[1] + deg_pad))*(X[isample][:,1] > (lon_range_extend[0] - deg_pad)))[0]
				isample = isample[ikeep]
				ikeep1 = np.where((X[isample_vald][:,0] < (lat_range_extend[1] + deg_pad))*(X[isample_vald][:,0] > (lat_range_extend[0] - deg_pad))*(X[isample_vald][:,1] < (lon_range_extend[1] + deg_pad))*(X[isample_vald][:,1] > (lon_range_extend[0] - deg_pad)))[0]
				isample_vald = isample_vald[ikeep1]


			Tp_sample = results[0][isample] # np.concatenate((results[0][isample], results[0][isample1], results[0][isample2], results[0][isample3]), axis = 0)
			Ts_sample = results[1][isample] # np.concatenate((results[1][isample], results[1][isample1], results[1][isample2], results[1][isample3]), axis = 0)
			Vp_sample = Vp[isample] # np.concatenate((Vp[isample], Vp[isample1], Vp[isample2], Vp[isample3]), axis = 0)
			Vs_sample = Vs[isample] # np.concatenate((Vs[isample], Vs[isample1], Vs[isample2], Vs[isample3]), axis = 0)
			X_sample = X[isample] # np.concatenate((X[isample], X[isample1], X[isample2], X[isample3]), axis = 0)
			xx_sample = xx[isample] # np.concatenate((xx[isample], xx[isample1], xx[isample2], xx[isample3]), axis = 0)


			Tp_sample_vald = results[0][isample_vald] # , results[0][isample1], results[0][isample2], results[0][isample3]), axis = 0)
			Ts_sample_vald = results[1][isample_vald]
			Vp_sample_vald = Vp[isample_vald]
			Vs_sample_vald = Vs[isample_vald]
			X_sample_vald = X[isample_vald]
			xx_sample_vald = xx[isample_vald] # , xx[isample1], xx[isample2], xx[isample3]), axis = 0)

			Tp_boundary = np.zeros((n_zero_inputs, 1))
			Ts_boundary = np.zeros((n_zero_inputs, 1))
			Vp_boundary = Vp[src_index].repeat(n_zero_inputs, axis = 0)
			Vs_boundary = Vs[src_index].repeat(n_zero_inputs, axis = 0)
			X_boundary = X[src_index].repeat(n_zero_inputs, axis = 0)
			xx_boundary = xx[src_index].repeat(n_zero_inputs, axis = 0)

			data['Tp_%d'%inc_res] = Tp_sample
			data['Ts_%d'%inc_res] = Ts_sample
			data['X_%d'%inc_res] = X_sample
			data['X_cart_%d'%inc_res] = xx_sample
			data['Vp_%d'%inc_res] = Vp_sample
			data['Vs_%d'%inc_res] = Vs_sample
			data['Dist_%d'%inc_res] = np.linalg.norm(xx_sample - loc_proj, axis = 1)

			data['Tp_vald_%d'%inc_res] = Tp_sample_vald
			data['Ts_vald_%d'%inc_res] = Ts_sample_vald
			data['X_vald_%d'%inc_res] = X_sample_vald
			data['X_cart_vald_%d'%inc_res] = xx_sample_vald
			data['Vp_vald_%d'%inc_res] = Vp_sample_vald
			data['Vs_vald_%d'%inc_res] = Vs_sample_vald
			data['Dist_vald_%d'%inc_res] = np.linalg.norm(xx_sample_vald - loc_proj, axis = 1)

			data['Tp_boundary_%d'%inc_res] = Tp_boundary
			data['Ts_boundary_%d'%inc_res] = Ts_boundary
			data['X_boundary_%d'%inc_res] = X_boundary
			data['X_cart_boundary_%d'%inc_res] = xx_boundary
			data['Vp_boundary_%d'%inc_res] = Vp_boundary
			data['Vs_boundary_%d'%inc_res] = Vs_boundary

		else:

			data['Tp_%d'%inc_res] = results[0]
			data['Ts_%d'%inc_res] = results[1]
			data['X_%d'%inc_res] = ftrns2(xx)
			data['X_cart_%d'%inc_res] = xx
			data['Vp_%d'%inc_res] = Vp
			data['Vs_%d'%inc_res] = Vs			


	# =========================================================================
	# CONCATENATION OUTSIDE THE LOOP (Combines Tier 0, Tier 1, and Tier 2)
	# =========================================================================
	if sample_points == True:

		# Gather keys across all tiers dynamically
		tiers = range(len(optim))
		
		# 1. Spatial Matrices (Shape: N, 3)
		data['X'] = np.concatenate([data['X_%d' % i] for i in tiers], axis=0)
		data['X_cart'] = np.concatenate([data['X_cart_%d' % i] for i in tiers], axis=0)
		
		data['X_vald'] = np.concatenate([data['X_vald_%d' % i] for i in tiers], axis=0)
		data['X_cart_vald'] = np.concatenate([data['X_cart_vald_%d' % i] for i in tiers], axis=0)
		
		data['X_boundary'] = np.concatenate([data['X_boundary_%d' % i] for i in tiers], axis=0)
		data['X_cart_boundary'] = np.concatenate([data['X_cart_boundary_%d' % i] for i in tiers], axis=0)

		# 2. Vectors (Handles 1D flat vs 2D column arrays identically)
		if len(data['Tp_0'].shape) == 1:
			data['Tp'] = np.concatenate([data['Tp_%d' % i] for i in tiers], axis=0)
			data['Ts'] = np.concatenate([data['Ts_%d' % i] for i in tiers], axis=0)
			data['Vp'] = np.concatenate([data['Vp_%d' % i] for i in tiers], axis=0)
			data['Vs'] = np.concatenate([data['Vs_%d' % i] for i in tiers], axis=0)
			data['Dist'] = np.concatenate([data['Dist_%d' % i] for i in tiers], axis=0)

			data['Tp_vald'] = np.concatenate([data['Tp_vald_%d' % i] for i in tiers], axis=0)
			data['Ts_vald'] = np.concatenate([data['Ts_vald_%d' % i] for i in tiers], axis=0)
			data['Vp_vald'] = np.concatenate([data['Vp_vald_%d' % i] for i in tiers], axis=0)
			data['Vs_vald'] = np.concatenate([data['Vs_vald_%d' % i] for i in tiers], axis=0)
			data['Dist_vald'] = np.concatenate([data['Dist_vald_%d' % i] for i in tiers], axis=0)

			data['Tp_boundary'] = np.concatenate([data['Tp_boundary_%d' % i] for i in tiers], axis=0)
			data['Ts_boundary'] = np.concatenate([data['Ts_boundary_%d' % i] for i in tiers], axis=0)
			data['Vp_boundary'] = np.concatenate([data['Vp_boundary_%d' % i] for i in tiers], axis=0)
			data['Vs_boundary'] = np.concatenate([data['Vs_boundary_%d' % i] for i in tiers], axis=0)
		else:
			# Assumes 2D column formatting (N, 1)
			data['Tp'] = np.concatenate([data['Tp_%d' % i] for i in tiers], axis=0)
			data['Ts'] = np.concatenate([data['Ts_%d' % i] for i in tiers], axis=0)
			data['Vp'] = np.concatenate([data['Vp_%d' % i] for i in tiers], axis=0)
			data['Vs'] = np.concatenate([data['Vs_%d' % i] for i in tiers], axis=0)
			data['Dist'] = np.concatenate([data['Dist_%d' % i].reshape(-1, 1) for i in tiers], axis=0)

			data['Tp_vald'] = np.concatenate([data['Tp_vald_%d' % i] for i in tiers], axis=0)
			data['Ts_vald'] = np.concatenate([data['Ts_vald_%d' % i] for i in tiers], axis=0)
			data['Vp_vald'] = np.concatenate([data['Vp_vald_%d' % i] for i in tiers], axis=0)
			data['Vs_vald'] = np.concatenate([data['Vs_vald_%d' % i] for i in tiers], axis=0)
			data['Dist_vald'] = np.concatenate([data['Dist_vald_%d' % i].reshape(-1, 1) for i in tiers], axis=0)

			data['Tp_boundary'] = np.concatenate([data['Tp_boundary_%d' % i] for i in tiers], axis=0)
			data['Ts_boundary'] = np.concatenate([data['Ts_boundary_%d' % i] for i in tiers], axis=0)
			data['Vp_boundary'] = np.concatenate([data['Vp_boundary_%d' % i] for i in tiers], axis=0)
			data['Vs_boundary'] = np.concatenate([data['Vs_boundary_%d' % i] for i in tiers], axis=0)

	else:
		# If you aren't sampling points, concatenate raw un-sampled grids
		tiers = range(len(optim))
		data['Tp'] = np.concatenate([data['Tp_%d' % i] for i in tiers], axis=0)
		data['Ts'] = np.concatenate([data['Ts_%d' % i] for i in tiers], axis=0)
		data['X'] = np.concatenate([data['X_%d' % i] for i in tiers], axis=0)
		data['X_cart'] = np.concatenate([data['X_cart_%d' % i] for i in tiers], axis=0)
		data['Vp'] = np.concatenate([data['Vp_%d' % i] for i in tiers], axis=0)
		data['Vs'] = np.concatenate([data['Vs_%d' % i] for i in tiers], axis=0)	

	if sta_ind >= len(locs):
		sta_ind += len(ind_use_rand)*int(argvs[1])

	np.savez_compressed(path_to_file + '1D_Velocity_Models_Regional' + seperator + 'TravelTimeData' + seperator + '%s_1d_velocity_model_station_%d_ver_%d.npz'%(name_of_project, sta_ind, vel_model_ver), **data)
	print('Saved %d'%sta_ind)


print("All files saved successfully!")
print("✔ Script execution: Done")



