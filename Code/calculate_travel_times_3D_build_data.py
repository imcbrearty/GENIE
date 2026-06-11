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
from scipy.interpolate import interp1d
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




def compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, depths, vp, vs=None):
    """
    Computes P and S travel times using TauP surface references across an arbitrary
    space-filling grid. Corrects for WGS84 ellipsoidal radius variation and preserves
    1D ray integration physics across variable topography completely natively.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    
    # =====================================================================
    # 1. Coordinate and Velocity Framework Extraction (WGS84 Geocentric Fix)
    # =====================================================================
    X_lla = ftrns2(xx)
    station_lla = ftrns2(loc_proj)[0]

    # WGS84 Constants
    A_EQUATOR = 6378137.0
    B_POLE = 6356752.3142
    ELLIPSOIDAL_FLATTENING = 1.0 / 298.257223563
    LAT_SCALE_FACTOR = (1.0 - ELLIPSOIDAL_FLATTENING)**2

    station_lat_geo, station_lon_geo = station_lla[0], station_lla[1]
    mesh_lats_geo, mesh_lons_geo = X_lla[:, 0], X_lla[:, 1]

    # Convert geodetic latitudes to geocentric parameters to find exact earth radius
    station_lat_centric = np.degrees(np.arctan(LAT_SCALE_FACTOR * np.tan(np.radians(station_lat_geo))))
    mesh_lats_centric = np.degrees(np.arctan(LAT_SCALE_FACTOR * np.tan(np.radians(mesh_lats_geo))))

    lat1, lon1 = np.radians(station_lat_centric), np.radians(station_lon_geo)
    lat2, lon2 = np.radians(mesh_lats_centric), np.radians(mesh_lons_geo)
    
    # Calculate exact great-circle angular separation (Delta) across the ellipsoid
    dlon = lon2 - lon1
    cos_clat = np.clip(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon), -1.0, 1.0)
    distances_deg = np.degrees(np.arccos(cos_clat))

    # Parse velocity framework configurations
    if hasattr(taup_model.model, 'v_mod'):
        v_mod = taup_model.model.v_mod
    elif hasattr(taup_model.model, 's_mod') and hasattr(taup_model.model.s_mod, 'v_mod'):
        v_mod = taup_model.model.s_mod.v_mod
    else:
        v_mod = taup_model.model

    model_depths_km = [layer['top_depth'] for layer in v_mod.layers] + [v_mod.layers[-1]['bot_depth']]
    model_depths_m = np.array(model_depths_km) * 1000.0
    model_vp_m_s = np.array([layer['top_p_velocity'] for layer in v_mod.layers] + [v_mod.layers[-1]['bot_p_velocity']]) * 1000.0
    model_vs_m_s = np.array([layer['top_s_velocity'] for layer in v_mod.layers] + [v_mod.layers[-1]['bot_s_velocity']]) * 1000.0

    vp_surface, vs_surface = model_vp_m_s[0], model_vs_m_s[0]
    src_z, grid_z = station_lla[2], X_lla[:, 2]
    
    # Calculate true ellipsoidal surface radius at the station and mesh positions
    r_surface_mesh = (A_EQUATOR * B_POLE) / np.sqrt(
        (B_POLE * np.cos(lat2))**2 + (A_EQUATOR * np.sin(lat2))**2
    )
    r_surface_station = (A_EQUATOR * B_POLE) / np.sqrt(
        (B_POLE * np.cos(lat1))**2 + (A_EQUATOR * np.sin(lat1))**2
    )
    
    # Radii from Earth's center to absolute 3D nodes
    r_source = r_surface_station + src_z
    r_target = r_surface_mesh + grid_z
    
    # Convert depth metrics relative to local columns.
    source_depth_m = r_surface_station - r_source
    target_depths_m = r_surface_mesh - r_target

    # Map the reference source depth cleanly to TauP's standard 1D column space
    source_depth_taup_km = np.maximum(0.0, source_depth_m / 1000.0)

    # =====================================================================
    # 2. Seed a Single Dense Surface Profile (Run TauP exactly ONCE)
    # =====================================================================
    max_deg = max(0.1, distances_deg.max())

    if max_deg * 1.1 > 0.2:
        near_field = np.geomspace(1e-7, 0.2, 200)
        far_field = np.linspace(0.2001, max_deg * 1.1, 300)
        dense_deg_axis = np.concatenate((near_field, far_field))
    else:
        dense_deg_axis = np.geomspace(1e-7, max_deg * 1.1, 500)

    surf_t_p = np.full_like(dense_deg_axis, np.inf)
    surf_p_p = np.zeros_like(dense_deg_axis)
    surf_t_s = np.full_like(dense_deg_axis, np.inf)
    surf_p_s = np.zeros_like(dense_deg_axis)

    for i, deg in enumerate(dense_deg_axis):
        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=source_depth_taup_km,
                distance_in_degree=deg,
                phase_list=["P", "S", "Pn", "Sn", "Pg", "Sg", "PKP", "SKS", "Pdiff", "Sdiff"]
            )
            for arrival in arrivals:
                phase = arrival.name.upper()
                p_m = arrival.ray_param / 6371000.0
                base_type = phase[0] if len(phase) > 0 else ""
                
                if base_type == "P" and "S" not in phase:
                    if arrival.time < surf_t_p[i]:
                        surf_t_p[i], surf_p_p[i] = arrival.time, p_m
                elif base_type == "S" and "P" not in phase:
                    if arrival.time < surf_t_s[i]:
                        surf_t_s[i], surf_p_s[i] = arrival.time, p_m
        except Exception:
            continue

    # Near-field dynamic fallback paths if TauP drops below local threshold
    z_limit = np.maximum(0.0, source_depth_m)
    active_layers = model_depths_m[:-1] <= z_limit
    
    if np.any(active_layers):
        avg_vp_near = np.mean(model_vp_m_s[:-1][active_layers])
        avg_vs_near = np.mean(model_vs_m_s[:-1][active_layers])
    else:
        avg_vp_near, avg_vs_near = vp_surface, vs_surface

    for surf_t, surf_p, v_near in [(surf_t_p, surf_p_p, avg_vp_near), (surf_t_s, surf_p_s, avg_vs_near)]:
        mask = np.isinf(surf_t)
        if np.any(mask):
            surf_t[mask] = (np.radians(dense_deg_axis[mask]) * r_surface_station) / v_near
            surf_p[mask] = 1.0 / v_near

    final_deg_axis = np.insert(dense_deg_axis, 0, 0.0)
    
    # Interpolate background values across the irregular spatial points
    interp_tp_base = interp1d(final_deg_axis, np.insert(surf_t_p, 0, 0.0), kind='linear', fill_value="extrapolate")(distances_deg)
    p_slowness_raw = interp1d(final_deg_axis, np.insert(surf_p_p, 0, 1.0/vp_surface), kind='linear', fill_value="extrapolate")(distances_deg)
    
    interp_ts_base = interp1d(final_deg_axis, np.insert(surf_t_s, 0, 0.0), kind='linear', fill_value="extrapolate")(distances_deg)
    s_slowness_raw = interp1d(final_deg_axis, np.insert(surf_p_s, 0, 1.0/vs_surface), kind='linear', fill_value="extrapolate")(distances_deg)

    # =====================================================================
    # 3. Adaptive Near-Vertical Conditioning Framework
    # =====================================================================
    dx_m = xx[:, 0] - loc_proj[0, 0]
    dy_m = xx[:, 1] - loc_proj[0, 1]
    horizontal_distances_m = np.sqrt(dx_m**2 + dy_m**2)

    # Determine the adaptive threshold based on grid resolution cell size
    nonzero_horiz = horizontal_distances_m[horizontal_distances_m > 1e-2]
    adaptive_threshold_m = np.min(nonzero_horiz) * 1.5 if len(nonzero_horiz) > 0 else 500.0

    # Flag and condition variables inside the near-vertical mask
    vertical_mask = horizontal_distances_m <= adaptive_threshold_m

    if np.any(vertical_mask):
        interp_tp_base[vertical_mask] = 0.0
        interp_ts_base[vertical_mask] = 0.0
        p_slowness_raw[vertical_mask] = 0.0
        s_slowness_raw[vertical_mask] = 0.0

    # =====================================================================
    # 4. Vectorized Directional Depth Integration Loop (With Topography)
    # =====================================================================
    def integrate_between_bounds(p_slowness, z_start, z_end, is_s_wave=False):
        dt = np.zeros_like(z_start)
        v_profile = model_vs_m_s if is_s_wave else model_vp_m_s
        v_surf = vs_surface if is_s_wave else vp_surface
        
        z_top_arr = np.minimum(z_start, z_end)
        z_bot_arr = np.maximum(z_start, z_end)
        
        # STEP A: Natively integrate through topography layers (depth < 0)
        topo_active = (z_top_arr < 0.0)
        if np.any(topo_active):
            thick_topo = 0.0 - np.minimum(z_top_arr[topo_active], 0.0)
            thick_topo -= np.maximum(0.0 - z_bot_arr[topo_active], 0.0)
            
            slowness_sq_topo = 1.0 / v_surf**2
            valid_ray_topo = slowness_sq_topo > (p_slowness[topo_active]**2)
            
            eta_topo = np.zeros_like(thick_topo)
            if np.any(valid_ray_topo):
                eta_topo[valid_ray_topo] = np.sqrt(
                    slowness_sq_topo - p_slowness[topo_active][valid_ray_topo]**2
                )
            dt[topo_active] += thick_topo * eta_topo
            
        # STEP B: Core structural integration for subsurface crustal layers (depth >= 0)
        for idx in range(len(model_depths_m) - 1):
            z_layer_top, z_layer_bot = model_depths_m[idx], model_depths_m[idx + 1]
            v_layer = v_profile[idx]
            
            active = (z_bot_arr > z_layer_top) & (z_top_arr < z_layer_bot)
            if not np.any(active): continue
            
            effective_top = np.maximum(z_top_arr[active], z_layer_top)
            effective_bot = np.minimum(z_bot_arr[active], z_layer_bot)
            thick = effective_bot - effective_top
            
            slowness_sq_layer = 1.0 / v_layer**2
            valid_ray_layer = slowness_sq_layer > (p_slowness[active]**2)
            
            eta = np.zeros_like(thick)
            if np.any(valid_ray_layer):
                eta[valid_ray_layer] = np.sqrt(
                    slowness_sq_layer - p_slowness[active][valid_ray_layer]**2
                )
                
            dt[active] += thick * eta
        return dt

    z_source_vector = np.full_like(target_depths_m, source_depth_m)
    
    # Execute integration sweeps
    tp_times = interp_tp_base + integrate_between_bounds(p_slowness_raw, z_source_vector, target_depths_m, False)
    ts_times = interp_ts_base + integrate_between_bounds(s_slowness_raw, z_source_vector, target_depths_m, True)

    # =====================================================================
    # 5. Absolute Spatial Pinning (Fixed Cartesian Distance Tracking)
    # =====================================================================
    dx_raw = xx[:, 0] - loc_proj[0, 0]
    dy_raw = xx[:, 1] - loc_proj[0, 1]

    if xx.shape[1] >= 3:
        dz_raw = xx[:, 2] - loc_proj[0, 2]
    else:
        dz_raw = target_depths_m - source_depth_m
        
    dist_sq = dx_raw**2 + dy_raw**2 + dz_raw**2

    # Pin the absolute closest node to zero
    true_src_idx = np.argmin(dist_sq)
    if dist_sq[true_src_idx] < 1.0:  # 1 meter tolerance threshold
        tp_times[true_src_idx] = 0.0
        ts_times[true_src_idx] = 0.0

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
	topo_tree = cKDTree(ftrns1(surface_profile))
	unit_out = ftrns1(locs + np.concatenate((np.zeros((len(locs),2)), 1.0*np.ones((len(locs),1))), axis = 1))
	dist_near = topo_tree.query(ftrns1(locs))[0]
	dist_perturb = topo_tree.query(unit_out)[0]
	iabove_surface = np.where(dist_perturb > dist_near)[0]
	if len(iabove_surface) > 0: assert(np.abs(locs[iabove_surface,2] - surface_profile[tree.query(ftrns1(locs))[1][iabove_surface],2]).max() < tol_elev_val)


include_random_samples = True
if include_random_samples == True:
	ind_use_rand = np.random.choice(len(locs), size = int(3*len(ind_use)))
	locs_dist = cKDTree(ftrns1(locs)).query(ftrns1(locs[ind_use_rand]), k = 2)[0][:,1]
	locs_rand = ftrns2(locs_dist.reshape(-1,1)*np.random.randn(len(ind_use_rand),3) + ftrns1(locs[ind_use_rand]))
	irand = np.sort(np.random.choice(len(ind_use_rand), size = int(np.floor(len(ind_use_rand)/2)), replace = False)) if int(np.floor(len(ind_use_rand)/2)) > 0 else np.zeros(0).astype('int')
	locs_rand[irand] = np.random.uniform(offset_x_extend, offset_x_extend + scale_x_extend, size=(len(irand), 3))
	locs_rand = locs_rand.clip(offset_x_extend.squeeze(), (offset_x_extend + scale_x_extend).squeeze())
	locs_rand[:,2] = np.random.uniform(locs[:,2].min() - 0.1*(locs[:,2].max() - locs[:,2].min()), locs[:,2].max(), size = len(locs_rand))
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
			span_x3 = np.minimum(2.0 * opt_R_max1, regional_span_x3)
		elif inc_res == 1:
			span_x1 = np.minimum(2.0 * opt_R_max2, regional_span_x2)
			span_x2 = np.minimum(2.0 * opt_R_max2, regional_span_x2)
			span_x3 = np.minimum(2.0 * opt_R_max2, regional_span_x3)
		else:
			# --- FIXED: TIER 3 BUDGET PROTECTION FOR VIRTUAL STATIONS ---
			if sta_ind < len(locs):
				span_x1 = regional_span_x1
				span_x2 = regional_span_x2
				span_x3 = regional_span_x3
			else:
				span_x1 = np.minimum(4.0 * opt_R_max2, regional_span_x1)
				span_x2 = np.minimum(4.0 * opt_R_max2, regional_span_x2)
				span_x3 = np.minimum(4.0 * opt_R_max2, regional_span_x3)

		# 4. Derive grid counts ensuring dx == dy == dz (Perfect cubes)
		n1_target = int(np.ceil(span_x1 / dx_res)) + 8
		n2_target = int(np.ceil(span_x2 / dx_res)) + 8
		


		# Enable asymmetric vertical nodes for all deeper/regional tiers (Tier 2 and Tier 3)
		use_asymmetric_depths = True
		if use_asymmetric_depths and inc_res >= 1:
			# 1. Guarantee a minimum number of vertical layers
			min_vertical_layers = 60 
			dz_from_layers = span_x3 / min_vertical_layers
			
			# 2. Prevent the cell aspect ratio (dx/dz) from exceeding 4:1
			max_aspect_ratio = 4.0
			dz_from_aspect = dx_res / max_aspect_ratio
			
			# Pick the finer dz constraint, but never make it coarser than the horizontal dx_res
			dz_res = min(dz_from_layers, dz_from_aspect, dx_res)
			
			# --- DYNAMIC SANITY FLOOR ---
			# Never let dz become sharper than 20% of your horizontal resolution.
			# This natively scales down to tiny numbers for small local arrays, 
			# but still protects you from micro-scale infinity traps if span_x3 is near zero.
			dz_floor = dx_res / 5.0
			dz_res = max(dz_res, dz_floor) 
				
			n3_target = int(np.ceil(span_x3 / dz_res)) + 8
		else:
			dz_res = dx_res
			n3_target = int(np.ceil(span_x3 / dx_res)) + 8
	
		# 5. Force odd dimensions so the station source lands perfectly on a node center
		n1 = n1_target + 1 if n1_target % 2 == 0 else n1_target
		n2 = n2_target + 1 if n2_target % 2 == 0 else n2_target
		n3 = n3_target + 1 if n3_target % 2 == 0 else n3_target

		# =====================================================================
		# ENGINE-DEPENDENT GEOMETRIC MESH GENERATION (WITH TIER TRUNCATION)
		# =====================================================================

		if engine_type == 'taup':
			# -----------------------------------------------------------------
			# GEOGRAPHIC PATH: True Spherical/Ellipsoidal Angular Grid Setup
			# -----------------------------------------------------------------
			station_lla = ftrns2(loc_proj)[0]

			# Extract your domain boundaries directly from the true geodetic extent
			lat_min, lat_max = lat_range_extend[0], lat_range_extend[1]
			lon_min, lon_max = lon_range_extend[0], lon_range_extend[1]

			# Calculate the total true angular spans of your region
			span_lat_deg = lat_max - lat_min
			span_lon_deg = lon_max - lon_min



			# Determine the grid center based on the resolution tier
			if inc_res < (len(optim) - 1) or sta_ind >= len(locs):
				# Fine local tiers stay centered right on the station
				center_lat = station_lla[0]
				center_lon = station_lla[1]
				
				# Derive scaling factors from the station's local curvature metrics
				METERS_PER_LAT_DEG = 111195.0
				METERS_PER_LON_DEG = 111195.0 * np.cos(np.radians(station_lla[0]))
				
				# Convert the meter spans to local angular equivalents safely
				half_span_lat_deg = (span_x1 / 2.0) / METERS_PER_LAT_DEG
				half_span_lon_deg = (span_x2 / 2.0) / METERS_PER_LON_DEG
			else:
				# Massive global/regional tiers center on the overall geographic area
				center_lat = (lat_min + lat_max) / 2.0
				center_lon = (lon_min + lon_max) / 2.0
				
				# Map to the absolute physical limits of your specified window
				half_span_lat_deg = span_lat_deg / 2.0
				half_span_lon_deg = span_lon_deg / 2.0

			# Generate uniform angular coordinates
			lats_uniform = np.linspace(center_lat - half_span_lat_deg, center_lat + half_span_lat_deg, n1)
			lons_uniform = np.linspace(center_lon - half_span_lon_deg, center_lon + half_span_lon_deg, n2)
			
			# === FIXED: SNAP STATION TO NODE CENTERS UNCONDITIONALLY ACROSS ALL TIERS ===
			snap_idx_lat = np.argmin(np.abs(lats_uniform - station_lla[0]))
			snap_idx_lon = np.argmin(np.abs(lons_uniform - station_lla[1]))
			lats_uniform += (station_lla[0] - lats_uniform[snap_idx_lat])
			lons_uniform += (station_lla[1] - lons_uniform[snap_idx_lon])
			
			# Set up the vertical geodetic altitude coordinates
			elev_lla = locs[:, 2].max() + 1000.0  
			altitudes_base = np.linspace(0, n3 - 1, n3) * dz_res
			altitudes_base = (altitudes_base - altitudes_base.mean()) + station_lla[2]
			altitudes_base = altitudes_base - altitudes_base.max() + elev_lla
			
			inearest = np.argmin(np.abs(altitudes_base - station_lla[2]))
			altitudes_uniform = altitudes_base - (altitudes_base[inearest] - station_lla[2])

			# Mesh grid assembly
			lat_mesh, lon_mesh, alt_mesh = np.meshgrid(lats_uniform, lons_uniform, altitudes_uniform, indexing='ij')
			X = np.concatenate((lat_mesh.reshape(-1,1), lon_mesh.reshape(-1,1), alt_mesh.reshape(-1,1)), axis=1)
			
			# Forward project to local meters so your Eikonal PINN handles uniform physical spacing
			xx = ftrns1(X)
			
			x11, x12, x13 = None, None, None
	
		else:
			# -----------------------------------------------------------------
			# CARTESIAN PATH: Strict dX/dY/dZ for Eikonal/FMM Finite Differences
			# -----------------------------------------------------------------
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

			x11, x12, x13 = np.meshgrid(x1, x2, x3, indexing='ij')
			xx = np.concatenate((x11.reshape(-1,1), x12.reshape(-1,1), x13.reshape(-1,1)), axis=1)
			X = ftrns2(xx)

		# print(f'\nGenerated Tier {inc_res} [{engine_type.upper()}]: {dx_res*n1:0.2f}m x {dx_res*n2:0.2f}m x {dz_res*n3:0.2f}m')		
		print(f'\nGenerated Tier {inc_res}: {dx_res*n1:0.2f} m x {dx_res*n2:0.2f} m x {dz_res*n3:0.2f} m')		

		# Universal geometric fallback locator across both engine types
		src_index = np.argmin(np.linalg.norm(xx - loc_proj, axis=1))
		assert(np.allclose(xx[src_index], loc_proj[0,:], atol = 1e-3)) 

		# =====================================================================
		# EXECUTION BACKEND BRANCH
		# =====================================================================
		if engine_type == 'taup':
			
			# Applied safely to any dense multi-resolution tier
			if xx.shape[0] > 1000000 or inc_res > 0:
				# =====================================================================
				# VOLUMETRIC SAFETY VALVE FOR GIANT DOMAINS
				# =====================================================================
				MAX_TIER_NODES = 30000000  
				if xx.shape[0] > MAX_TIER_NODES:
					print(f" Warning: xx size ({xx.shape[0]}) exceeds maximum budget. Downsampling to {MAX_TIER_NODES} points...")

					dist_pad_depth = 0.2 * deg_pad * 110e3
					valid_region_mask = (
						(X[:, 0] < (lat_range_extend[1] + deg_pad)) & 
						(X[:, 0] > (lat_range_extend[0] - deg_pad)) & 
						(X[:, 1] < (lon_range_extend[1] + deg_pad)) & 
						(X[:, 1] > (lon_range_extend[0] - deg_pad)) & 
						(X[:, 2] <= (depth_range[1] + dist_pad_depth)) & 
						(X[:, 2] >= (depth_range[0] - dist_pad_depth))
					)

					keep_indices = np.random.choice(np.where(valid_region_mask == 1)[0], size = min(MAX_TIER_NODES, int(valid_region_mask.sum())), replace = False)
					
					if src_index not in keep_indices:
						keep_indices[0] = src_index
						
					xx = xx[keep_indices]
					X = X[keep_indices]
					src_index = np.argmin(np.linalg.norm(xx - loc_proj, axis=1))

				# =====================================================================
				# MEMORY-SAFE CHUNKED EXECUTION
				# =====================================================================
				chunk_size = 10000000  
				total_nodes = xx.shape[0]
				
				tp_times_all = np.zeros(total_nodes, dtype=np.float32)
				ts_times_all = np.zeros(total_nodes, dtype=np.float32)
				
				print(f"Processing {total_nodes} nodes in {int(np.ceil(total_nodes/chunk_size))} memory batches...")
				
				for chunk_idx in range(0, total_nodes, chunk_size):
					end_idx = min(chunk_idx + chunk_size, total_nodes)
					xx_chunk = xx[chunk_idx:end_idx]
					
					tp_chunk, ts_chunk = compute_travel_times_taup_optimized(
						xx_chunk, loc_proj, taup_model, ftrns2, x_vel, vp
					)
					tp_times_all[chunk_idx:end_idx] = tp_chunk
					ts_times_all[chunk_idx:end_idx] = ts_chunk
				
				tp_flattened = tp_times_all.ravel()
				ts_flattened = ts_times_all.ravel()

			else:
				tp_flattened, ts_flattened = compute_travel_times_taup_optimized(xx, loc_proj, taup_model, ftrns2, x_vel, vp)
				tp_flattened = np.asarray(tp_flattened).ravel()
				ts_flattened = np.asarray(ts_flattened).ravel()

			# -----------------------------------------------------------------
			# MATHEMETICAL VELOCITY EXTRACTOR FOR PINN ALIGNMENT
			# -----------------------------------------------------------------
			if hasattr(taup_model.model, 'v_mod'):
				compiled_v_mod = taup_model.model.v_mod
			else:
				compiled_v_mod = taup_model.model.s_mod.v_mod

			taup_internal_depths = np.array([layer['top_depth'] for layer in compiled_v_mod.layers])
			taup_internal_vp = np.array([layer['top_p_velocity'] for layer in compiled_v_mod.layers]) * 1000.0 
			taup_internal_vs = np.array([layer['top_s_velocity'] for layer in compiled_v_mod.layers]) * 1000.0 

			vp_surface_val = taup_internal_vp[0]
			vs_surface_val = taup_internal_vs[0]

			for idx in range(1, len(taup_internal_depths)):
				if taup_internal_depths[idx] <= taup_internal_depths[idx - 1]:
					taup_internal_depths[idx] = taup_internal_depths[idx - 1] + 1e-5

			true_z = X[:, 2]
			subsurface_depths_km = np.maximum(0.0, -true_z) / 1000.0

			Vp = np.interp(subsurface_depths_km, taup_internal_depths, taup_internal_vp)
			Vs = np.interp(subsurface_depths_km, taup_internal_depths, taup_internal_vs)

			mountain_mask = true_z > 0
			Vp[mountain_mask] = vp_surface_val
			Vs[mountain_mask] = vs_surface_val

		else:
			# -----------------------------------------------------------------
			# FINITE DIFFERENCE PATH (EIKONAL/FMM)
			# -----------------------------------------------------------------
			Vp, Vs = initilize_velocity_model(x_vel, vp, vs, xx, [dx_res, dx_res, dz_res], vel_type=vel_model_type)
			print('dx_v %0.4f %0.4f %0.4f \n'%(dx_res, dx_res, dz_res))

			if use_topography and ('surface_profile' in locals() or 'surface_profile' in globals()):
				print("--> Applying global topography envelope to Eikonal computational grid...")
				
				grid_lla = ftrns2(xx)
				grid_depths = grid_lla[:, 2] 
				grid_horizontal_proj = ftrns1(np.concatenate((grid_lla[:, 0:2], np.zeros((len(grid_lla), 1))), axis=1))
				
				_, surface_match_indices = topo_tree.query(grid_horizontal_proj)
				true_surface_elevations = surface_profile[surface_match_indices, 2]
				
				air_mask = grid_depths > true_surface_elevations
				if np.any(air_mask):
					Vp[air_mask] = Vp.min() 
					Vs[air_mask] = Vs.min() 
					print(f"    Forced {np.sum(air_mask)} atmospheric grid nodes to surface velocity baselines.")

			tp_grid, ts_grid = compute_travel_times_parallel(xx, loc_proj, Vp, Vs, [dx_res, dx_res, dz_res], x11, x12, x13, num_cores=num_cores)
			tp_flattened = tp_grid.reshape(-1)
			ts_flattened = ts_grid.reshape(-1)

		# Pack answers into unified format for downstream steps
		results = [tp_flattened[:, None], ts_flattened[:, None]]
		assert(np.isclose(results[0][src_index,0], 0.0, atol=1e-2))
		assert(np.isclose(results[1][src_index,0], 0.0, atol=1e-2))



		sample_points = True
		if sample_points == True:

			# scale_factor = len(locs)/25
			# scale_factor = 1.0

			scale_factor = np.log1p(len(locs) / 25.0)
			# Ensure the scale_factor never shrinks below 1.0 for small networks
			scale_factor = max(1.0, scale_factor)
			
			n_zero_inputs = int(int(100000/scale_factor) / (len(optim) - 1))
			n_per_station = int(int(150000/scale_factor) / (len(optim) - 1))
			n_per_station1 = int(int(100000/scale_factor) / (len(optim) - 1))

			
			# 3. Dynamic Floor Guard: Ensure counts never drop below a physics-informed baseline
			# Even on massive domains, a station needs a core skeleton of nodes to resolve details.
			n_zero_inputs = max(500, n_zero_inputs)
			n_per_station = max(1000, n_per_station)
			n_per_station1 = max(500, n_per_station1)

			
			if sta_ind >= len(locs):
				n_zero_inputs = int(n_zero_inputs/3)
				n_per_station = int(n_per_station/3)
				n_per_station1 = int(n_per_station1/3)

			# # p = np.zeros(locs_ref.shape[0])
			# p[sta_ind] = 1
			# isample = np.sort(np.random.choice(len(p), size = n_zero_inputs, p = p/p.sum(), replace = True))


			# =====================================================================
			# 1. GENERATE THE REGIONAL VALID MASK UPFRONT
			# =====================================================================
			dist_pad_depth = 0.2 * deg_pad * 110e3
			
			# Identify every node in the entire grid that physically sits within your bounds
			valid_region_mask = (
				(X[:, 0] < (lat_range_extend[1] + deg_pad)) & 
				(X[:, 0] > (lat_range_extend[0] - deg_pad)) & 
				(X[:, 1] < (lon_range_extend[1] + deg_pad)) & 
				(X[:, 1] > (lon_range_extend[0] - deg_pad)) & 
				(X[:, 2] <= (depth_range[1] + dist_pad_depth)) & 
				(X[:, 2] >= (depth_range[0] - dist_pad_depth))
			)
			
			# Get a list of indices that are inside your target box
			valid_indices = np.where(valid_region_mask)[0]
			
			# Fallback protection: if the box is empty, use the whole grid
			if len(valid_indices) == 0:
				valid_indices = np.arange(X.shape[0])
				
			p_mask = np.zeros(len(X))
			p_mask[valid_indices] = 1.0
			p_mask_int = int(p_mask.sum())
			
			p = p_mask*(1.0/np.maximum(results[0][:,0], 0.1))
			isample = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station, p_mask_int), p = p/p.sum(), replace = False))

			p = p_mask*((1.0/np.maximum(results[0][:,0], 0.1))**2)
			isample1 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, p_mask_int), p = p/p.sum(), replace = False))

			# grab_near_boundaries_samples
			p = p_mask*(1.0/np.maximum(results[0].max() - results[0][:,0], 0.1))
			isample2 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, p_mask_int), p = p/p.sum(), replace = False))

			# grab_interior_samples
			p = p_mask*(1.0*np.ones(X.shape[0])) # /np.maximum(Tp_interp[:,n].max() - Tp_interp[:,n], 0.1)
			isample3 = np.sort(np.random.choice(len(p), size = np.minimum(n_per_station1, p_mask_int), p = p/p.sum(), replace = False))
			# isample_vald = np.random.choice(np.delete(np.arange(len(p)), np.unique(np.concatenate((isample, isample1, isample2, isample3), axis = 0)), axis = 0), size = n_per_station)
			
			isample = np.random.permutation(np.concatenate((isample, isample1, isample2, isample3), axis = 0))
			# isample_vald = np.random.choice(np.delete(np.arange(len(p)), isample, axis = 0), size = n_per_station)

			valid_ind = np.setdiff1d(valid_indices, isample)
			
			# If you ran out of unique points, fall back to all valid indices
			if len(valid_ind) == 0:
				valid_ind = valid_indices
				
			isample_vald = np.random.choice(valid_ind, size=np.minimum(n_per_station, len(valid_ind)), replace = False)

			# use_within_region = True
			# if use_within_region == True:

			# 	dist_pad_depth = 0.2*deg_pad*110e3
			# 	ikeep = np.where((X[isample][:,0] < (lat_range_extend[1] + deg_pad))*(X[isample][:,0] > (lat_range_extend[0] - deg_pad))*(X[isample][:,1] < (lon_range_extend[1] + deg_pad))*(X[isample][:,1] > (lon_range_extend[0] - deg_pad))*(X[isample][:,2] <= (depth_range[1] + dist_pad_depth))*(X[isample][:,2] >= (depth_range[0] - dist_pad_depth)))[0]
			# 	isample = isample[ikeep]
				
			# 	ikeep1 = np.where((X[isample_vald][:,0] < (lat_range_extend[1] + deg_pad))*(X[isample_vald][:,0] > (lat_range_extend[0] - deg_pad))*(X[isample_vald][:,1] < (lon_range_extend[1] + deg_pad))*(X[isample_vald][:,1] > (lon_range_extend[0] - deg_pad))*(X[isample_vald][:,2] <= (depth_range[1] + dist_pad_depth))*(X[isample_vald][:,2] >= (depth_range[0] - dist_pad_depth)))[0]
			# 	isample_vald = isample_vald[ikeep1]


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
			# Vp_boundary = Vp[src_index].repeat(n_zero_inputs, axis = 0)
			# Vs_boundary = Vs[src_index].repeat(n_zero_inputs, axis = 0)
			# X_boundary = X[src_index].reshape(1,-1).repeat(n_zero_inputs, axis = 0)
			# xx_boundary = xx[src_index].reshape(1,-1).repeat(n_zero_inputs, axis = 0)

			Vp_boundary = np.tile(Vp[src_index], (n_zero_inputs, 1))
			Vs_boundary = np.tile(Vs[src_index], (n_zero_inputs, 1))
			X_boundary = np.tile(X[src_index].reshape(1, -1), (n_zero_inputs, 1))
			xx_boundary = np.tile(xx[src_index].reshape(1, -1), (n_zero_inputs, 1))
			
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
