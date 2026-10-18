import os
import pathlib
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import skfmm

from graph_utils import *
from utils import *


def compute_travel_times_parallel(
    xx_r,
    h,
    h1,
    dx_v,
    x11,
    x12,
    x13,
    num_cores=10,
    vp_interp=None,
    vs_interp=None,
):
    """
    Computes 3D P and S travel time fields in parallel using Fast Marching (scikit-fmm).
    Guarantees sub-grid continuous source placement without snapping/bias errors.
    Expects x11, x12, x13 built with indexing='ij'.
    """
    n_sources = xx_r.shape[0]
    grid_shape = x11.shape  # Shape is (len(x1), len(x2), len(x3))

    def step_test(args):
        yval, dx_v, h, h1, x11, x12, x13, ind = args
        source_coords = yval.ravel()

        if np.isnan(source_coords).any() or np.isinf(source_coords).any():
            empty = np.full(grid_shape, np.nan, dtype=np.float32)
            return empty, empty, ind

        # Exact continuous Euclidean distance field
        phi_dist = np.sqrt(
            (x11 - source_coords[0]) ** 2
            + (x12 - source_coords[1]) ** 2
            + (x13 - source_coords[2]) ** 2
        )

        min_cell_dx = np.min(dx_v)
        r0 = 0.5 * min_cell_dx
        phi = phi_dist - r0

        # Dynamic Expansion Guard: handles off-grid/boundary sources safely
        if not (phi < 0).any():
            min_dist = float(np.min(phi_dist))
            r0 = min_dist + 0.1 * min_cell_dx
            phi = phi_dist - r0

        v_grid = h.reshape(grid_shape)
        v1_grid = h1.reshape(grid_shape)
        dx = [float(dx_v[0]), float(dx_v[1]), float(dx_v[2])]

        try:
            t = skfmm.travel_time(phi, v_grid, dx=dx)
            t1 = skfmm.travel_time(phi, v1_grid, dx=dx)
        except Exception:
            empty = np.full(grid_shape, np.nan, dtype=np.float32)
            return empty, empty, ind

        # Use passed-in global velocity interpolators if available, else fallback to nearest index
        if vp_interp is not None and vs_interp is not None:
            vp_source = float(np.asarray(vp_interp(source_coords)).ravel()[0])
            vs_source = float(np.asarray(vs_interp(source_coords)).ravel()[0])
            if np.isnan(vp_source) or vp_source <= 0:
                nearest_idx = np.unravel_index(np.argmin(phi_dist), grid_shape)
                vp_source = float(np.asarray(v_grid[nearest_idx]).ravel()[0])
            if np.isnan(vs_source) or vs_source <= 0:
                nearest_idx = np.unravel_index(np.argmin(phi_dist), grid_shape)
                vs_source = float(np.asarray(v1_grid[nearest_idx]).ravel()[0])
        else:
            nearest_idx = np.unravel_index(np.argmin(phi_dist), grid_shape)
            vp_source = float(np.asarray(v_grid[nearest_idx]).ravel()[0])
            vs_source = float(np.asarray(v1_grid[nearest_idx]).ravel()[0])

        # Analytical source time shift
        t += r0 / vp_source
        t1 += r0 / vs_source

        return t.astype(np.float32), t1.astype(np.float32), ind

    tp_times = np.full((h.shape[0], n_sources), np.nan, dtype=np.float32)
    ts_times = np.full((h.shape[0], n_sources), np.nan, dtype=np.float32)

    results = Parallel(n_jobs=num_cores)(
        delayed(step_test)([
            xx_r[i, :][None, :],
            dx_v,
            h,
            h1,
            x11,
            x12,
            x13,
            i,
        ])
        for i in range(n_sources)
    )

    for res in results:
        t_p, t_s, station_ind = res[0], res[1], res[2]
        tp_times[:, station_ind] = t_p.reshape(-1)
        ts_times[:, station_ind] = t_s.reshape(-1)

    return tp_times, ts_times


def make_merged_travel_time_interp(mp_coarse, mp_local, sta_xyz, half_xy, half_z):
    sta_xyz = np.asarray(sta_xyz, float).reshape(3)

    def interp(pts_cart):
        pts_cart = np.asarray(pts_cart, float).reshape(-1, 3)

        d_norm = np.sqrt(
            ((pts_cart[:, 0] - sta_xyz[0]) / half_xy) ** 2
            + ((pts_cart[:, 1] - sta_xyz[1]) / half_xy) ** 2
            + ((pts_cart[:, 2] - sta_xyz[2]) / half_z) ** 2
        )

        t_c = np.asarray(mp_coarse(pts_cart)).ravel()
        use_loc = d_norm <= 1.0
        t = t_c.copy()

        if np.any(use_loc):
            t_l = np.asarray(mp_local(pts_cart[use_loc])).ravel()

            w = np.clip((d_norm[use_loc] - 0.85) / 0.15, 0.0, 1.0)
            w_smooth = 0.5 * (1.0 - np.cos(np.pi * w))

            bad_loc = ~np.isfinite(t_l)
            t_l = np.where(bad_loc, t_c[use_loc], t_l)
            w_smooth = np.where(bad_loc, 1.0, w_smooth)

            t[use_loc] = (1.0 - w_smooth) * t_l + w_smooth * t_c[use_loc]

        return t

    return interp


def reflect_lla_into_depth_range(lla_pts, depth_range):
    """
    Reflects LLA coordinates so that depth/altitude (lla_pts[:, 2]) strictly 
    bounces back inside [depth_range[0], depth_range[1]].
    """
    lla = np.asarray(lla_pts, dtype=float).copy()
    min_dep, max_dep = float(depth_range[0]), float(depth_range[1])
    L = max_dep - min_dep

    if L > 0:
        z = lla[:, 2] - min_dep
        period = 2.0 * L
        z_mod = np.mod(z, period)
        z_reflected = np.where(z_mod > L, period - z_mod, z_mod)
        lla[:, 2] = min_dep + z_reflected

    return lla


def sample_near_station_reflected(
    sta_lla,
    ftrns1,
    ftrns2,
    n_samples,
    r_min=10.0,
    r_max=15000.0,
    depth_range=None,
    downward_ratio=0.70,
    power=1.0,
    seed=0,
):
    """
    Generates near-field points around sta_lla using log-radial sampling.
    Applies downward hemisphere bias (downward_ratio) and isotropic spread,
    with strict LLA depth reflection.
    """
    if n_samples <= 0:
        return np.zeros((0, 3))

    sta_lla = np.asarray(sta_lla, dtype=float).reshape(3)
    sta_xyz = ftrns1(sta_lla.reshape(1, 3))[0]

    rng_loc = np.random.default_rng(seed)

    # Power-adjusted log distribution to control density decay into mid-range
    u = rng_loc.uniform(0.0, 1.0, size=n_samples) ** power
    log_r = np.log(r_min) + u * (np.log(r_max) - np.log(r_min))
    r = np.exp(log_r)

    # 70% Downward / 30% Isotropic Direction Split
    n_down = int(np.round(downward_ratio * n_samples))
    n_iso = n_samples - n_down

    costheta_down = rng_loc.uniform(-1.0, 0.0, size=n_down)
    costheta_iso = rng_loc.uniform(-1.0, 1.0, size=n_iso)
    costheta = np.concatenate([costheta_down, costheta_iso])
    rng_loc.shuffle(costheta)

    phi = rng_loc.uniform(0.0, 2.0 * np.pi, size=n_samples)
    sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta**2))

    dx = r * sintheta * np.cos(phi)
    dy = r * sintheta * np.sin(phi)
    dz = r * costheta

    pts_cart = sta_xyz + np.column_stack([dx, dy, dz])
    pts_lla = ftrns2(pts_cart)

    if depth_range is not None:
        pts_lla = reflect_lla_into_depth_range(pts_lla, depth_range)

    return pts_lla


def sample_downward_cylinder(
    sta_lla,
    ftrns1,
    ftrns2,
    n_samples,
    r_cyl_max=5000.0,
    depth_range=None,
    seed=0,
):
    """
    Generates points in a vertical cylinder directly beneath the station 
    to fill the sub-receiver volume across depth, adaptively scaled to r_cyl_max.
    """
    if n_samples <= 0 or depth_range is None:
        return np.zeros((0, 3))

    sta_lla = np.asarray(sta_lla, dtype=float).reshape(3)
    rng_loc = np.random.default_rng(seed)

    r_cyl = r_cyl_max * np.sqrt(rng_loc.uniform(0.0, 1.0, size=n_samples))
    phi = rng_loc.uniform(0.0, 2.0 * np.pi, size=n_samples)

    dx = r_cyl * np.cos(phi)
    dy = r_cyl * np.sin(phi)

    min_dep, max_dep = float(depth_range[0]), float(depth_range[1])
    target_depths = rng_loc.uniform(min_dep, max_dep, size=n_samples)

    pts_lla = np.zeros((n_samples, 3))
    pts_lla[:, 0] = sta_lla[0]
    pts_lla[:, 1] = sta_lla[1]
    pts_lla[:, 2] = target_depths

    pts_cart = ftrns1(pts_lla) + np.column_stack([dx, dy, np.zeros(n_samples)])
    pts_lla_offset = ftrns2(pts_cart)

    pts_lla_offset = reflect_lla_into_depth_range(pts_lla_offset, depth_range)
    return pts_lla_offset


def sample_stratified_queries_for_station(
    sta_lla,
    lat_range,
    lon_range,
    depth_range,
    ftrns1,
    ftrns2,
    tp_interp=None,
    ts_interp=None,
    vp_interp=None,
    vs_interp=None,
    n_very_near=25_000,
    n_near=35_000,
    n_mid=25_000,
    n_sub_cylinder=15_000,
    n_far=35_000,
    n_uniform=25_000,
    n_boundary=2_000,
    n_vald=10_000,
    n_candidates_factor=4,
    _sobol_box=None,
    xx_min=None,
    xx_max=None,
    seed=0,
    dx=1000.0,
):
    sta_lla = np.asarray(sta_lla, dtype=float).reshape(3)
    rng = np.random.default_rng(seed)

    if xx_min is not None and xx_max is not None:
        L_xy = float(np.minimum(xx_max[0] - xx_min[0], xx_max[1] - xx_min[1]))
    else:
        c = ftrns1(
            np.array([
                [lat_range[0], lon_range[0], sta_lla[2]],
                [lat_range[1], lon_range[1], sta_lla[2]],
            ])
        )
        L_xy = float(np.linalg.norm(c[1] - c[0]))

    # 1. Very Near Tier (5m to 5% L_xy)
    r_min_vn = max(5.0, 0.5 * dx)
    r_max_vn = max(1000.0, 0.05 * L_xy)

    X_vn = sample_near_station_reflected(
        sta_lla,
        ftrns1,
        ftrns2,
        n_very_near,
        r_min=r_min_vn,
        r_max=r_max_vn,
        depth_range=depth_range,
        downward_ratio=0.70,
        power=1.0,
        seed=seed,
    )

    # 2. Near Tier (100m to 20% L_xy)
    r_min_ne = max(50.0, 2.0 * dx)
    r_max_ne = max(5000.0, 0.20 * L_xy)

    X_ne = sample_near_station_reflected(
        sta_lla,
        ftrns1,
        ftrns2,
        n_near,
        r_min=r_min_ne,
        r_max=r_max_ne,
        depth_range=depth_range,
        downward_ratio=0.70,
        power=1.0,
        seed=seed + 1,
    )

    # 3. Mid-Range Tier (5% L_xy to 50% L_xy)
    r_min_mid = 0.05 * L_xy
    r_max_mid = 0.50 * L_xy

    X_mid = sample_near_station_reflected(
        sta_lla,
        ftrns1,
        ftrns2,
        n_mid,
        r_min=r_min_mid,
        r_max=r_max_mid,
        depth_range=depth_range,
        downward_ratio=0.60,
        power=0.75,
        seed=seed + 2,
    )

    # 4. Sub-Station Cylinder
    r_cyl_max = max(1000.0, 0.05 * L_xy)
    X_cyl = sample_downward_cylinder(
        sta_lla,
        ftrns1,
        ftrns2,
        n_sub_cylinder,
        r_cyl_max=r_cyl_max,
        depth_range=depth_range,
        seed=seed + 3,
    )

    # 5. Far & Uniform Candidate Pools
    def domain_candidates(n):
        n = int(max(n, 1))
        n_pow2 = 1 << (n - 1).bit_length()
        pts_lla = np.asarray(_sobol_box(n_pow2))[:n]
        return reflect_lla_into_depth_range(pts_lla, depth_range)

    n_pool = int(n_candidates_factor * (n_far + n_uniform))
    cand = domain_candidates(n_pool)

    tp = np.asarray(tp_interp(ftrns1(cand))).ravel()
    used = []

    def take(n, mode):
        nonlocal used
        mask = np.ones(len(cand), dtype=bool)
        if used:
            mask[np.unique(used)] = False

        valid_tp = np.isfinite(tp) & (tp > 0)
        mask = mask & valid_tp

        pts_m, tp_m = cand[mask], tp[mask]
        if len(pts_m) == 0:
            return np.zeros((0, 3))

        n = min(int(n), len(pts_m))
        eps_t = 0.001

        if mode == "far":
            w = 1.0 / (np.maximum(0.0, tp_m.max() - tp_m) + eps_t)
        else:
            w = np.ones(len(tp_m))

        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if w.sum() <= 0:
            return np.zeros((0, 3))

        w = w / w.sum()
        idx_m = rng.choice(len(pts_m), size=n, replace=False, p=w)
        global_idx = np.flatnonzero(mask)[idx_m]
        used.extend(global_idx.tolist())
        return pts_m[idx_m]

    X_fa = take(n_far, "far")
    X_un = take(n_uniform, "uniform")
    X_bd = np.repeat(sta_lla.reshape(1, 3), int(n_boundary), axis=0)

    X_train = np.vstack([X_vn, X_ne, X_mid, X_cyl, X_fa, X_un, X_bd])
    if n_boundary > 0:
        X_train[-n_boundary:] = sta_lla.reshape(1, 3)

    cand_v = domain_candidates(int(n_candidates_factor * n_vald))
    tp_v = np.asarray(tp_interp(ftrns1(cand_v))).ravel()
    valid_v = np.isfinite(tp_v) & (tp_v > 0)
    cand_v = cand_v[valid_v]

    n_vald_actual = min(n_vald, len(cand_v))
    idx_v = rng.choice(len(cand_v), size=n_vald_actual, replace=False)
    X_vald = cand_v[idx_v]

    # 6. Evaluation
    def eval_all(pts):
        xyz = ftrns1(pts)
        tp_vals = np.asarray(tp_interp(xyz)).ravel()

        valid = np.isfinite(tp_vals)
        if ts_interp is not None:
            ts_vals = np.asarray(ts_interp(xyz)).ravel()
            valid = valid & np.isfinite(ts_vals)

        pts = pts[valid]
        tp_vals = tp_vals[valid]

        out = {"X": pts, "Tp": tp_vals}
        if ts_interp is not None:
            out["Ts"] = ts_vals[valid]
        if vp_interp is not None:
            out["Vp"] = np.asarray(vp_interp(xyz[valid])).ravel()
        if vs_interp is not None:
            out["Vs"] = np.asarray(vs_interp(xyz[valid])).ravel()
        return out

    train = eval_all(X_train)
    vald = eval_all(X_vald)

    if n_boundary > 0 and len(train["Tp"]) >= n_boundary:
        train["Tp"][-n_boundary:] = 0.0
        if "Ts" in train:
            train["Ts"][-n_boundary:] = 0.0

    return train, vald


if __name__ == "__main__":
    config = load_config("config.yaml")

    name_of_project = config["name_of_project"]
    dx = config["dx"]
    dx_depth = config.get("dx_depth", dx)

    save_dense_travel_time_data = config["save_dense_travel_time_data"]
    train_travel_time_neural_network = config[
        "train_travel_time_neural_network"
    ]

    path_to_file = str(pathlib.Path().absolute())
    seperator = "\\" if "\\" in path_to_file else "/"
    path_to_file += seperator
    num_cores = 1

    vel_model_ver = config["vel_model_ver"]
    vel_model_type = config["vel_model_type"]
    use_topography = config["use_topography"]

    z = np.load(path_to_file + "%s_region.npz" % name_of_project)
    lat_range, lon_range, depth_range, deg_pad = (
        z["lat_range"],
        z["lon_range"],
        z["depth_range"],
        z["deg_pad"],
    )
    z.close()

    z = np.load(path_to_file + "%s_stations.npz" % name_of_project)
    locs, stas, mn, rbest = z["locs"], z["stas"], z["mn"], z["rbest"]
    z.close()

    d_pad = deg_pad
    lat_range_extend = [
        np.minimum(lat_range[0] - deg_pad, locs[:, 0].min() - d_pad),
        np.maximum(lat_range[1] + deg_pad, locs[:, 0].max() + d_pad),
    ]
    lon_range_extend = [
        np.minimum(lon_range[0] - deg_pad, locs[:, 1].min() - d_pad),
        np.maximum(lon_range[1] + deg_pad, locs[:, 1].max() + d_pad),
    ]

    depth_grid = np.arange(depth_range[0], depth_range[1] + dx_depth, dx_depth)

    ftrns1 = lambda x: (rbest @ (lla2ecef(x) - mn).T).T
    ftrns2 = lambda x: ecef2lla((rbest.T @ x.T).T + mn)

    locs_ref = np.copy(locs)
    reciever_proj = ftrns1(locs_ref)

    argvs = sys.argv
    if len(argvs) == 1:
        argvs.append(0)

    n_jobs = config["n_jobs"]
    n_batch = int(np.ceil(len(locs) / n_jobs))
    ind_use = [np.arange(n_batch) + n_batch * i for i in range(n_jobs)]
    if n_jobs > 1:
        ind_use[-1] = np.arange(ind_use[-2][-1] + 1, len(locs))
    ind_use = ind_use[int(argvs[1])]

    spanned_domain, domain_pad_cnt = False, 0
    while (spanned_domain == False) and (domain_pad_cnt < 20):
        lat0, lat1 = lat_range_extend
        lon0, lon1 = lon_range_extend
        dep0, dep1 = depth_range[0], depth_range[1]

        def lla_box_boundary(n_edge=5):
            lats = np.linspace(lat0, lat1, n_edge)
            lons = np.linspace(lon0, lon1, n_edge)
            deps = np.linspace(dep0, dep1, max(3, n_edge // 2))
            x11, x12, x13 = np.meshgrid(lats, lons, deps, indexing="ij")
            pts = np.concatenate(
                (
                    x11.reshape(-1, 1),
                    x12.reshape(-1, 1),
                    x13.reshape(-1, 1),
                ),
                axis=1,
            )
            return np.unique(np.asarray(pts, float), axis=0)

        bnd_lla = lla_box_boundary(11)
        bnd_xyz = ftrns1(bnd_lla)

        margin_xy = (5.0 + domain_pad_cnt) * dx
        margin_z = (1.0 + domain_pad_cnt * (1 / 5)) * dx

        x1 = np.arange(
            bnd_xyz[:, 0].min() - margin_xy,
            bnd_xyz[:, 0].max() + margin_xy + dx,
            dx,
        )
        x2 = np.arange(
            bnd_xyz[:, 1].min() - margin_xy,
            bnd_xyz[:, 1].max() + margin_xy + dx,
            dx,
        )
        x3 = np.arange(
            bnd_xyz[:, 2].min() - margin_z,
            bnd_xyz[:, 2].max() + margin_z + dx,
            dx,
        )

        x11, x12, x13 = np.meshgrid(x1, x2, x3, indexing="ij")
        xx = np.concatenate(
            (x11.reshape(-1, 1), x12.reshape(-1, 1), x13.reshape(-1, 1)),
            axis=1,
        )
        dx_v = np.array([np.diff(x1)[0], np.diff(x2)[0], np.diff(x3)[0]])
        xx_min = xx.min(0)
        xx_max = xx.max(0)

        inside_hull = np.all(
            (reciever_proj >= xx.min(0)) * (reciever_proj <= xx.max(0)),
            axis=1,
        )

        if inside_hull.sum() == locs_ref.shape[0]:
            spanned_domain = True
        else:
            domain_pad_cnt += 1

    assert inside_hull.sum() == locs_ref.shape[0]

    xy_span = float(np.minimum(xx_max[0] - xx_min[0], xx_max[1] - xx_min[1]))
    z_span = float(xx_max[2] - xx_min[2])

    frac_xy = 0.20
    frac_z = 0.50

    half_xy = 0.5 * frac_xy * xy_span
    half_z = 0.5 * frac_z * z_span

    if vel_model_type == 1:
        vp = np.array(config["velocity_model"]["Vp"])
        vs = np.array(config["velocity_model"]["Vs"])
        depths = np.array(config["velocity_model"]["Depths"])
        iarg = np.argsort(depths)

        depths_fine = np.arange(
            depths.min(), depths.max() + dx_depth / 10.0, dx_depth / 10.0
        )
        vp_fine = np.interp(depths_fine, depths[iarg], vp[iarg])
        vs_fine = np.interp(depths_fine, depths[iarg], vs[iarg])

        tree = cKDTree(depths_fine.reshape(-1, 1))
        ip_nearest = tree.query(ftrns2(xx)[:, 2].reshape(-1, 1))[1]
        Vp = vp_fine[ip_nearest]
        Vs = vs_fine[ip_nearest]

    elif vel_model_type == 2:
        z = np.load(path_to_file + "3d_velocity_model.npz")
        x_vel, vp_vel, vs_vel = z["X"], z["Vp"], z["Vs"]
        z.close()

        tree = cKDTree(ftrns1(x_vel))
        ip_nearest = tree.query(xx)[1]
        Vp = vp_vel[ip_nearest]
        Vs = vs_vel[ip_nearest]

    mp_vp = RegularGridInterpolator(
        (x1, x2, x3),
        Vp.reshape(len(x1), len(x2), len(x3)),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    ms_vs = RegularGridInterpolator(
        (x1, x2, x3),
        Vs.reshape(len(x1), len(x2), len(x3)),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    earth_radius = 6378137.0
    ftrns1_abs = lambda x: (
        lla2ecef(x, a=earth_radius)
        if x.shape[1] == 3
        else np.concatenate(
            (lla2ecef(x, a=earth_radius), x[:, 3].reshape(-1, 1)), axis=1
        )
    )

    def optimize_r_min(
        lat_vals, lon_mean=np.mean(lon_range), h_min=depth_range[0]
    ):
        r_surface = np.linalg.norm(
            ftrns1_abs(
                np.concatenate(
                    (
                        lat_vals.reshape(-1, 1),
                        lon_mean * np.ones((len(lat_vals), 1)),
                        np.zeros((len(lat_vals), 1)),
                    ),
                    axis=1,
                )
            ),
            axis=1,
        )
        return r_surface + h_min

    def optimize_r_max(
        lat_vals, lon_mean=np.mean(lon_range), h_max=depth_range[1]
    ):
        r_surface = np.linalg.norm(
            ftrns1_abs(
                np.concatenate(
                    (
                        lat_vals.reshape(-1, 1),
                        lon_mean * np.ones((len(lat_vals), 1)),
                        np.zeros((len(lat_vals), 1)),
                    ),
                    axis=1,
                )
            ),
            axis=1,
        )
        return -(r_surface + h_max)

    bounds = [(lat_range_extend[0], lat_range_extend[1])]
    soln = differential_evolution(
        optimize_r_min, bounds, popsize=50, maxiter=1000
    )
    r_min = optimize_r_min(np.array([soln.x]))[0]

    soln = differential_evolution(
        optimize_r_max, bounds, popsize=50, maxiter=1000
    )
    r_max = -1.0 * optimize_r_max(np.array([soln.x]))[0]

    sobolov_sampler = lambda N: regular_sobolov(
        N,
        lat_range=lat_range_extend,
        lon_range=lon_range_extend,
        depth_range=depth_range,
        use_time=False,
        use_global=False,
        r_min=r_min,
        r_max=r_max,
    )[0]

    output_dir = os.path.join(
        path_to_file, "1D_Velocity_Models_Regional", "TravelTimeData"
    )
    os.makedirs(output_dir, exist_ok=True)

    for j in ind_use:
        results = compute_travel_times_parallel(
            reciever_proj[j][None, :],
            Vp,
            Vs,
            dx_v,
            x11,
            x12,
            x13,
            num_cores=num_cores,
            vp_interp=mp_vp,
            vs_interp=ms_vs,
        )
        mp = RegularGridInterpolator(
            (x1, x2, x3),
            results[0].reshape(len(x1), len(x2), len(x3)),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        ms = RegularGridInterpolator(
            (x1, x2, x3),
            results[1].reshape(len(x1), len(x2), len(x3)),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        sta_xyz = reciever_proj[j]

        half_xy_j = min(half_xy, sta_xyz[0] - xx_min[0], xx_max[0] - sta_xyz[0])
        half_xy_j = min(
            half_xy_j, sta_xyz[1] - xx_min[1], xx_max[1] - sta_xyz[1]
        )
        half_z_j = min(half_z, sta_xyz[2] - xx_min[2], xx_max[2] - sta_xyz[2])

        half_xy_j = max(half_xy_j, 5 * dx)
        half_z_j = max(half_z_j, 5 * dx)

        N_target = 0.5 * len(xx)
        V = max((2 * half_xy_j) ** 2 * (2 * half_z_j), 1.0)
        dx_loc = float(np.clip((V / N_target) ** (1.0 / 3.0), dx / 4.0, dx))

        x1_loc = np.arange(
            sta_xyz[0] - half_xy_j, sta_xyz[0] + half_xy_j + dx_loc, dx_loc
        )
        x2_loc = np.arange(
            sta_xyz[1] - half_xy_j, sta_xyz[1] + half_xy_j + dx_loc, dx_loc
        )
        x3_loc = np.arange(
            sta_xyz[2] - half_z_j, sta_xyz[2] + half_z_j + dx_loc, dx_loc
        )

        x11_loc, x12_loc, x13_loc = np.meshgrid(
            x1_loc, x2_loc, x3_loc, indexing="ij"
        )
        xx_loc = np.column_stack(
            [x11_loc.ravel(), x12_loc.ravel(), x13_loc.ravel()]
        )
        dx_v_loc = np.array(
            [np.diff(x1_loc)[0], np.diff(x2_loc)[0], np.diff(x3_loc)[0]]
        )

        Vp_loc = np.maximum(np.asarray(mp_vp(xx_loc)).ravel(), 200.0)
        Vs_loc = np.maximum(np.asarray(ms_vs(xx_loc)).ravel(), 100.0)

        results_loc = compute_travel_times_parallel(
            sta_xyz[None, :],
            Vp_loc,
            Vs_loc,
            dx_v_loc,
            x11_loc,
            x12_loc,
            x13_loc,
            num_cores=num_cores,
            vp_interp=mp_vp,
            vs_interp=ms_vs,
        )
        mp_loc = RegularGridInterpolator(
            (x1_loc, x2_loc, x3_loc),
            results_loc[0].reshape(len(x1_loc), len(x2_loc), len(x3_loc)),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        ms_loc = RegularGridInterpolator(
            (x1_loc, x2_loc, x3_loc),
            results_loc[1].reshape(len(x1_loc), len(x2_loc), len(x3_loc)),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        tp_merged = make_merged_travel_time_interp(
            mp, mp_loc, sta_xyz, half_xy_j, half_z_j
        )
        ts_merged = make_merged_travel_time_interp(
            ms, ms_loc, sta_xyz, half_xy_j, half_z_j
        )

        train, vald = sample_stratified_queries_for_station(
            locs_ref[j],
            lat_range_extend,
            lon_range_extend,
            depth_range,
            ftrns1,
            ftrns2,
            tp_interp=tp_merged,
            ts_interp=ts_merged,
            vp_interp=mp_vp,
            vs_interp=ms_vs,
            n_very_near=20_000,
            n_near=30_000,
            n_sub_cylinder=30_000,
            n_far=30_000,
            n_uniform=50_000,
            n_boundary=2_000,
            n_vald=10_000,
            n_candidates_factor=4,
            _sobol_box=sobolov_sampler,
            xx_min=xx_min,
            xx_max=xx_max,
            seed=np.random.choice(10000),
            dx=dx,
        )

        save_path = os.path.join(
            output_dir,
            f"{name_of_project}_1d_velocity_model_station_{j}_ver_{vel_model_ver}.npz",
        )
        np.savez_compressed(
            save_path,
            X=train["X"],
            X_cart=ftrns1(train["X"]),
            Tp=train["Tp"],
            Ts=train["Ts"],
            Vp=train.get("Vp"),
            Vs=train.get("Vs"),
            X_vald=vald["X"],
            X_vald_cart=ftrns1(vald["X"]),
            Tp_vald=vald["Tp"],
            Ts_vald=vald["Ts"],
            Vp_vald=vald.get("Vp"),
            Vs_vald=vald.get("Vs"),
            loc=locs_ref[j],
            locs_ref=locs_ref,
            xx=xx,
            depth_grid=depth_grid,
        )
        print(f"[SUCCESS] Saved station {j} data to {save_path}\n")
