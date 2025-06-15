import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from sklearn.neighbors import NearestNeighbors

def load_distance_magnitude_model(n_mag_ver=1, use_softplus=True):
    use_softplus = True
    dist_supp = np.load('distance_magnitude_model_ver_%d.npz'%(n_mag_ver))

    if use_softplus == False:
        poly_dist_p, poly_dist_s, min_dist = dist_supp['dist_p'], dist_supp['dist_s'], dist_supp['min_dist']
        pdist_p = lambda mag: np.maximum(min_dist[0], np.polyval(poly_dist_p, mag))
        pdist_s = lambda mag: np.maximum(min_dist[1], np.polyval(poly_dist_s, mag))
    else:
        dist_params = dist_supp['params']
        pdist_p = lambda mag: dist_params[4]*(1.0/dist_params[1])*np.log(1.0 + np.exp(dist_params[1]*mag)) + dist_params[0]
        pdist_s = lambda mag: dist_params[4]*(1.0/dist_params[3])*np.log(1.0 + np.exp(dist_params[3]*mag)) + dist_params[2]
    dist_supp.close()
    print('Will use amplitudes since a magnitude model was loaded')

    return pdist_p, pdist_s

def prepare_station_coordinates(locs, dropout_rate=0.1, scale=111320):
    """Prepare station coordinates from raw location data."""
    # Replaced by frtns
    latitudes = locs[:, 0]
    longitudes = locs[:, 1]
    print(f"longitudes: {longitudes.min()}, {longitudes.max()}")
    print(f"latitudes: {latitudes.min()}, {latitudes.max()}")
    y = scale*latitudes
    x = scale*longitudes # *np.cos(latitudes/360*2*np.pi)
    y = y - y.mean()
    x = x - x.mean()
    print(f"x: {x.min()}, {x.max()}")
    print(f"y: {y.min()}, {y.max()}")
    
    # Keep approximately 1-dropout_rate of the points using proper sampling
    if dropout_rate > 0:
        n_total = len(x)
        n_keep = int(n_total * (1 - dropout_rate))
        # Use random choice without replacement for proper dropout
        keep_indices = np.random.choice(n_total, size=n_keep, replace=False)
        x = x[keep_indices]
        y = y[keep_indices]
        real_dropout_rate = 1 - len(keep_indices) / n_total
        print(f"Dropped {real_dropout_rate*100:.1f}% of the points ({n_total - n_keep}/{n_total})")
    
    points = np.stack([x, y], axis=1)
    print(f"Final points shape: {points.shape}")
    print(f"Unique points: {len(np.unique(points, axis=0))}")
    return points

# def knn_weights_nonbinary(coords, k=8):
#     # TO DO: use real knn graph.
#     """Compute k-nearest neighbour weight matrix (row-standardised)."""
#     N = coords.shape[0]
#     d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
#     np.fill_diagonal(d2, np.inf)
#     idx_knn = np.argpartition(d2, k, axis=1)[:, :k]
    
#     W = [[] for _ in range(N)]
#     for i in range(N):
#         neighbours = idx_knn[i]
#         w = 1.0 / k
#         for j in neighbours:
#             W[i].append((j, w))
#     return W

def knn_weights(coords, k=8):
    """Compute binary k-nearest neighbour weight matrix using scikit-learn.
    
    Args:
        coords (np.ndarray): Array of shape (N, 2) containing point coordinates
        k (int): Number of nearest neighbors to consider
        
    Returns:
        list: List of lists containing (neighbor_index, weight) tuples where weight is 1.0
    """
    # Initialize and fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(coords)
    
    # Get k nearest neighbors for each point
    distances, indices = nn.kneighbors(coords)
    
    # Create binary weight matrix
    N = coords.shape[0]
    W = [[] for _ in range(N)]
    for i in range(N):
        neighbors = indices[i]
        for j in neighbors:
            if i != j:  # Skip self-loops
                W[i].append((j, 1.0))  # Binary weight: 1.0 for neighbors
    return W

def morans_I_binary(selected, W):
    """Calculate Moran's I for binary selection vector using sparse W."""
    y = selected.astype(float)
    k = y.sum()
    if k == 0 or k == len(y):
        return 0.0
    y_bar = k / len(y)
    num = 0.0
    row_sum = 0.0
    for i, neighbours in enumerate(W):
        for j, w in neighbours:
            num += w * (y[i] - y_bar) * (y[j] - y_bar)
            row_sum += w
    denom = np.sum((y - y_bar) ** 2)
    I = (len(y) / row_sum) * (num / denom)
    return I

# def compute_distance(points):
#     """Compute pairwise distances between points."""
#     N = points.shape[0]
#     distance = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             distance[i, j] = np.linalg.norm(points[i] - points[j])
#     return distance

def distances(points):
    """Compute pairwise distances between points using vectorized operations."""
    # Use broadcasting to compute all pairwise distances at once
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def compute_covariance(distance, sigma_noise=1.0):
    """Compute covariance matrix from distances."""
    N = distance.shape[0]
    covariance = np.exp(-(distance**2) / (2 * (sigma_noise**2)))
    covariance += 1e-8 * np.eye(N)
    return covariance


def inv_2x2(matrix):
    """Compute inverse of a 2x2 matrix using direct formula."""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / det

def radial_function(points, center, inv_cov, sigma_radial, p=2, scale_factor=1.0):
    """Compute Mahalanobis distances and PDF values."""
    diff = points - center
    mahalanobis2 = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    radial_pdf = scale_factor * np.exp(-(1/(2 * 9**(p-1))) * np.power(mahalanobis2/sigma_radial, 2*p))
    return radial_pdf


def generate_ellipse_parameters(scale_factor=1.0):
    """Generate random ellipse parameters."""
    angle = np.random.uniform(0, 2*np.pi)
    length1 = np.random.uniform(0.7, 1.3) * scale_factor # TODO: tune this
    length2 = np.random.uniform(0.7, 1.3) * scale_factor
    
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    D = np.array([[length1, 0],
                  [0, length2]])
    
    cov_ellipse = R @ D @ D @ R.T
    inv_cov = inv_2x2(cov_ellipse)
    
    return angle, length1, length2, cov_ellipse, inv_cov

def generate_noise(points, sigma_noise, cholesky_matrix=None):
    """Generate spatially correlated noise using Cholesky decomposition.
    TO DO: use pre-computed coefficients.
    Args:
        points (np.ndarray): Array of points
        sigma_noise (float): Covariance scale parameter
        cholesky_matrix (np.ndarray): Pre-computed Cholesky matrix
    Returns:
        np.ndarray: Spatially correlated noise
    """

    if cholesky_matrix is None:
        distance = distances(points)
        covariance = compute_covariance(distance, sigma_noise=sigma_noise)

        L = np.linalg.cholesky(covariance)
    else:
        L = cholesky_matrix
    
    z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
    noise = L @ z
    print(f'  [DEBUG] noise min: {noise.min()}, max: {noise.max()}')
    print(f'  [DEBUG] noise mean: {noise.mean()}, std: {noise.std()}')
    return noise

def logistic(x, sigma_logistic=1.0):
    """Logistic function with scaling parameter sigma_radial."""
    print(f'  [DEBUG] x: {x.min()}, {x.max()}')
    print(f'  [DEBUG] sigma_logistic: {sigma_logistic}')
    print(f'  [DEBUG] 1 / (1 + np.exp(-x/sigma_logistic)): {1 / (1 + np.exp(-x/sigma_logistic)).min()}, {1 / (1 + np.exp(-x/sigma_logistic)).max()}')
    return 1 / (1 + np.exp(-x/sigma_logistic))

def compute_final_probabilities(radial_pdf, noise, lambda_corr=0.5, sigma_logistic=1.0, decaying_factor=25):
    """Compute final selection probabilities."""
    # =========== Option 1: adding dumbly =============
    # Add the noise to the pdf values
    # alpha = 0.125
    # noise_pdf = alpha * (logistic(noise) - 0.5)
    # pdf_final = np.clip(radial_pdf + noise_pdf, 0, 1)

    # ========== Option 2: adding a logistic function with convex sum =============
    # noise_pdf = logistic(noise, sigma_logistic)
    # # Mixing function
    # def g(p, lam):
    #     return lam * p * (1 - p)
    # # Compute pdf_final (q)
    # g_vals = g(radial_pdf, lambda_corr)
    # pdf_final = (1 - g_vals) * radial_pdf + g_vals * noise_pdf

    # ========== Option 3: adding a logistic function with "proportional" sum =============
    def g_prop(radial_pdf):
        return radial_pdf
    
    def g_step(radial_pdf, decaying_factor=25):  # decaying_factor = 25 => decays from 0.25
        return 1 - (1 - radial_pdf)**decaying_factor

    noise_pdf = 2 * (logistic(noise, sigma_logistic) - 0.5) # between -1 and 1

    pdf_final = np.clip(radial_pdf + lambda_corr * g_step(radial_pdf) * noise_pdf, 0, 1)

    return pdf_final

def run_single_experiment(points, sigma_radial, sigma_noise, sigma_logistic=1.0, lambda_corr=0.5, p=2, scale_factor=1.0):
    """Run a single experiment with the given parameters.
    
    Args:
        points (np.ndarray): Array of points
        sigma_radial (float): Standard deviation for the radial function
        sigma_noise (float): Covariance scale parameter for noise
        sigma_logistic (float): Scale parameter for logistic function
        lambda_corr (float): Mixing parameter between radial and noise
        p (float): Power parameter for Mahalanobis distance
        scale_factor (float): Scale factor for ellipse parameters
        
    Returns:
        dict: Dictionary containing experiment results and parameters
    """
    # Generate ellipse parameters
    angle, length1, length2, _, inv_cov = generate_ellipse_parameters(scale_factor)
    
    # Generate random center
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    center = np.array([
        np.random.uniform(x_min, x_max),
        np.random.uniform(y_min, y_max)
    ])
    
    # Compute initial probabilities
    radial_pdf = radial_function(points, center, inv_cov, sigma_radial, p, scale_factor)
    
    # Phase A: Initial selection
    initial_mask = np.random.binomial(1, radial_pdf)
    initial_idx = np.where(initial_mask)[0]
    
    # Phase B: Add noise
    noise = generate_noise(points, sigma_noise)
    pdf_final = compute_final_probabilities(radial_pdf, noise, lambda_corr, sigma_logistic)
    
    final_mask = np.random.binomial(1, pdf_final)
    final_idx = np.where(final_mask)[0]
    
    return {
        'initial_idx': initial_idx,
        'final_idx': final_idx,
        'center': center,
        'radial_pdf': radial_pdf,
        'pdf_final': pdf_final,
        'noise': noise,
        'parameters': {
            'scale_factor': scale_factor,
            'angle': angle,
            'length1': length1,
            'length2': length2,
            'sigma_radial': sigma_radial,
            'sigma_noise': sigma_noise,
            'sigma_logistic': sigma_logistic,
            'lambda_corr': lambda_corr,
            'p': p
        }
    }

def calculate_inertia(points, indices):
    """Calculate inertia for a subset of points."""
    if len(indices) == 0:
        return 0.0
    center = points[indices].mean(axis=0)
    return np.mean((points[indices] - center)**2)

# ----------------------------- 
# Visualization functions, not useful for the experiment, but useful for debugging.
# ----------------------------- 

def create_visualization(points, initial_idx, final_idx, noise_pdf, radial_pdf, pdf_final,
                        center, inv_cov, sigma_radial, angle, length1, length2,
                        scale_factor, sigma_noise, lambda_corr, fig_num, k_neighbours=8):
    print(f'[DEBUG] create_visualization: points shape: {points.shape}, initial_idx: {initial_idx.shape}, final_idx: {final_idx.shape}')
    print(f'  [DEBUG] noise_pdf min/max: {noise_pdf.min()}/{noise_pdf.max()}')
    print(f'  [DEBUG] radial_pdf min/max: {radial_pdf.min()}/{radial_pdf.max()}')
    print(f'  [DEBUG] pdf_final min/max: {pdf_final.min()}/{pdf_final.max()}')
    print(f'  [DEBUG] center: {center}')
    assert not np.isnan(noise_pdf).any(), 'noise_pdf contains NaN!'
    assert not np.isnan(radial_pdf).any(), 'radial_pdf contains NaN!'
    assert not np.isnan(pdf_final).any(), 'pdf_final contains NaN!'
    fig = plt.figure(figsize=(15, 5))
    axes = fig.subplots(1, 3)
    
    # Create grid for density plot with appropriate number of points for large coordinates
    # Use fewer points but maintain the scale
    x_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    y_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute grid values
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    grid_diff = grid_points - center
    mahalanobis2_grid = np.sum((grid_diff @ inv_cov) * grid_diff, axis=1)
    Z = np.exp(-(mahalanobis2_grid.reshape(X.shape)**1) / (2 * (sigma_radial**2)))
    
    # Plot noise distribution
    scatter1 = axes[0].scatter(points[:, 0], points[:, 1], c=noise_pdf, cmap='viridis', s=20)
    plt.colorbar(scatter1, ax=axes[0], label='noise_pdf value')
    axes[0].set_title(f'Noise Distribution\nScale: {scale_factor:.1f}, σ: {sigma_radial:.2f}, σ_cov: {sigma_noise:.2f}')
    axes[0].set_aspect('equal', 'box')
    # Format axis labels with scientific notation
    axes[0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Plot initial subset
    scatter2 = axes[1].scatter(points[:, 0], points[:, 1], s=5, alpha=0.3, label='Background points')
    scatter3 = axes[1].scatter(points[initial_idx, 0], points[initial_idx, 1], s=20, label='Selected points')
    contour1 = axes[1].contour(X, Y, Z, levels=10, alpha=0.5, colors='gray')
    im1 = axes[1].contourf(X, Y, Z, levels=10, alpha=0.3)
    plt.colorbar(im1, ax=axes[1], label='Density')
    
    I_initial = morans_I_binary(np.isin(np.arange(len(points)), initial_idx), knn_weights(points, k=k_neighbours))
    init_inertia = calculate_inertia(points, initial_idx)
    axes[1].set_title(f'No Noise Radial Points\nAngle: {np.degrees(angle):.1f}°, Lengths: {length1:.2f}, {length2:.2f}\nMoran I: {I_initial:.3f}, Inertia: {init_inertia:.1f}')
    axes[1].set_aspect('equal', 'box')
    axes[1].legend()
    # Format axis labels with scientific notation
    axes[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Plot final subset
    scatter4 = axes[2].scatter(points[:, 0], points[:, 1], s=5, alpha=0.3, label='Background points')
    scatter5 = axes[2].scatter(points[final_idx, 0], points[final_idx, 1], s=20, label='Selected points')
    contour2 = axes[2].contour(X, Y, Z, levels=10, alpha=0.5, colors='gray')
    im2 = axes[2].contourf(X, Y, Z, levels=10, alpha=0.3)
    plt.colorbar(im2, ax=axes[2], label='Density')
    
    # Add equiprobability contours using a different interpolation approach
    # Create a regular grid for interpolation
    xi = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    yi = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Use griddata with 'nearest' method for better stability with large coordinates
    grid_pdf_final = griddata(points, pdf_final, (xi, yi), method='nearest', fill_value=0)
    
    # Add contours with adjusted levels
    all_levels = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    min_val, max_val = np.nanmin(grid_pdf_final), np.nanmax(grid_pdf_final)
    valid_levels = [lvl for lvl in all_levels if min_val < lvl < max_val]
    if valid_levels:
        colors = ['red', 'blue'] * ((len(valid_levels) + 1) // 2)
        eq_contours = axes[2].contour(xi, yi, grid_pdf_final, levels=valid_levels, colors=colors[:len(valid_levels)], linewidths=1.5, linestyles='dashed')
        axes[2].clabel(eq_contours, fmt=lambda v: f'{v:.4f}', fontsize=9)
    
    # Add statistics
    n_selected = len(final_idx)
    n_total = len(points)
    selection_ratio = n_selected / n_total
    I_final = morans_I_binary(np.isin(np.arange(len(points)), final_idx), knn_weights(points, k=k_neighbours))
    final_inertia = calculate_inertia(points, final_idx)
    
    print("Moran's I is weird. TODO")

    axes[2].set_title(f'Final Selected Points\nλ: {lambda_corr:.2f}, Selected: {n_selected}/{n_total} ({selection_ratio:.1%})\nMoran I: {I_final:.3f}, Inertia: {final_inertia:.1f}')
    axes[2].set_aspect('equal', 'box')
    axes[2].legend()
    # Format axis labels with scientific notation
    axes[2].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Add main title
    fig.suptitle(f'Figure {fig_num+1}: Scale={scale_factor:.1f}, Angle={np.degrees(angle):.1f}°, Lengths={length1:.2f},{length2:.2f}\nσ={sigma_radial:.2f}, σ_cov={sigma_noise:.2f}, λ={lambda_corr:.2f}', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures/magnitude', exist_ok=True)
    filename = f'scale{scale_factor:.1f}_id{fig_num+1}_angle{np.degrees(angle):.1f}_len{length1:.1f}_{length2:.1f}.png'
    plt.savefig(os.path.join('figures', 'magnitude', filename), bbox_inches='tight', dpi=300)  # Increased DPI for better quality
    plt.close()

    print(f'  [DEBUG] grid_pdf_final shape: {grid_pdf_final.shape}, min/max: {grid_pdf_final.min()}/{grid_pdf_final.max()}')
    assert not np.isnan(grid_pdf_final).any(), 'grid_pdf_final contains NaN!'


def plot_experiment_results(points, runs, k_neighbours=8, p=2):
    print('[DEBUG] plot_experiment_results: points shape:', points.shape)
    print('[DEBUG] plot_experiment_results: runs count:', len(runs))
    assert points.shape[0] > 0, 'No points to plot!'
    
    # Precompute weights once
    W = knn_weights(points, k=k_neighbours)
    
    # Create individual figures for each run (exactly like radial_cholesky_stations_magnitude)
    for run_idx, run in enumerate(runs):
        print(f'[DEBUG] Plotting run {run_idx+1}')
        initial_idx = run['initial_idx']
        final_idx = run['final_idx']
        radial_pdf = run['radial_pdf']
        pdf_final = run['pdf_final']
        noise = run['noise']
        center = run['center']
        params = run['parameters']
        
        # Extract parameters
        sigma_radial = params['sigma_radial']
        sigma_noise = params['sigma_noise']
        lambda_corr = params['lambda_corr']
        angle = params['angle']
        length1 = params['length1']
        length2 = params['length2']
        scale_factor = params['scale_factor']
        sigma_logistic = params['sigma_logistic']
        
        # Create rotation and covariance matrices
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        D = np.array([[length1, 0],
                      [0, length2]])
        cov_ellipse = R @ D @ D @ R.T
        inv_cov = inv_2x2(cov_ellipse)
        
        # Calculate statistics
        I_initial = morans_I_binary(np.isin(np.arange(len(points)), initial_idx), W)
        I_final = morans_I_binary(np.isin(np.arange(len(points)), final_idx), W)
        init_inertia = calculate_inertia(points, initial_idx)
        final_inertia = calculate_inertia(points, final_idx)
        
        # Compute noise_pdf (exactly like stations_magnitude)
        noise_pdf = 2 * (logistic(noise, sigma_logistic=sigma_logistic) - 0.5)  # Between -1 and 1
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(15, 5))
        axes = fig.subplots(1, 3)
        
        # Create grid for density plot
        x_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
        y_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Elliptical Mahalanobis distance for the grid
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        grid_diff = grid_points - center
        mahalanobis2_grid = np.sqrt(np.sum((grid_diff @ inv_cov) * grid_diff, axis=1))
        Z = scale_factor * np.exp(-(1/(2 * 9**(p-1))) * np.power(mahalanobis2_grid.reshape(X.shape)/sigma_radial, 2*p))
        
        # Plot noise_pdf spatial distribution
        scatter1 = axes[0].scatter(points[:, 0], points[:, 1], c=noise_pdf, cmap='viridis', s=20)
        plt.colorbar(scatter1, ax=axes[0], label='noise_pdf value')
        axes[0].set_title(f'Noise Distribution\nScale: {scale_factor:.1f}, σ: {sigma_radial:.2f}, σ_cov: {sigma_noise:.2f}\nσ_logistic: {sigma_logistic:.1e}')
        axes[0].set_aspect('equal', 'box')
        
        # Plot initial subset (no noise)
        scatter2 = axes[1].scatter(points[:, 0], points[:, 1], s=5, alpha=0.3, label='Background points')
        scatter3 = axes[1].scatter(points[initial_idx, 0], points[initial_idx, 1], s=20, label='Selected points')
        contour1 = axes[1].contour(X, Y, Z, levels=10, alpha=0.5, colors='gray')
        im1 = axes[1].contourf(X, Y, Z, levels=10, alpha=0.3)
        plt.colorbar(im1, ax=axes[1], label='Density')
        axes[1].set_title(f'No Noise Radial Points\nAngle: {np.degrees(angle):.1f}°, Lengths: {length1:.2f}, {length2:.2f}\nMoran I: {I_initial:.3f}, Inertia: {init_inertia:.1f}')
        axes[1].set_aspect('equal', 'box')
        axes[1].legend()
        
        # Plot final subset
        scatter4 = axes[2].scatter(points[:, 0], points[:, 1], s=5, alpha=0.3, label='Background points')
        scatter5 = axes[2].scatter(points[final_idx, 0], points[final_idx, 1], s=20, label='Selected points')
        contour2 = axes[2].contour(X, Y, Z, levels=10, alpha=0.5, colors='gray')
        im2 = axes[2].contourf(X, Y, Z, levels=10, alpha=0.3)
        plt.colorbar(im2, ax=axes[2], label='Density')
        
        # Add equiprobability contours for pdf_final < 0.01
        grid_pdf_final = griddata(points, pdf_final, (X, Y), method='linear', fill_value=0)
        all_levels = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        # Only keep levels that are strictly increasing and within the data range
        min_val, max_val = np.nanmin(grid_pdf_final), np.nanmax(grid_pdf_final)
        valid_levels = [lvl for lvl in all_levels if min_val < lvl < max_val]
        if valid_levels:
            colors = ['red', 'blue'] * ((len(valid_levels) + 1) // 2)
            eq_contours = axes[2].contour(X, Y, grid_pdf_final, levels=valid_levels, colors=colors[:len(valid_levels)], linewidths=1.5, linestyles='dashed')
            axes[2].clabel(eq_contours, fmt=lambda v: f'{v:.4f}', fontsize=9)
        
        # Add statistics to the final plot
        n_selected = len(final_idx)
        n_total = len(points)
        selection_ratio = n_selected / n_total
        axes[2].set_title(f'Final Selected Points\nλ: {lambda_corr:.2f}, Selected: {n_selected}/{n_total} ({selection_ratio:.1%})\nMoran I: {I_final:.3f}, Inertia: {final_inertia:.1f}')
        axes[2].set_aspect('equal', 'box')
        axes[2].legend()
        
        # Add a main title with all parameters
        fig.suptitle(f'Figure {run_idx+1}: Scale={scale_factor:.1f}, Angle={np.degrees(angle):.1f}°, Lengths={length1:.2f},{length2:.2f}\nσ={sigma_radial:.2f}, σ_cov={sigma_noise:.2f}, σ_logistic={sigma_logistic:.1e}, λ={lambda_corr:.2f}, p={p}', y=1.02)
        
        plt.tight_layout()
        
        # Create filename with parameters
        filename = f'scale{scale_factor:.1f}_id{run_idx+1}_angle{np.degrees(angle):.1f}_len{length1:.1f}_{length2:.1f}_p{p}.png'
        plt.savefig(os.path.join('figures', 'magnitude', filename), bbox_inches='tight')
        plt.show()
        plt.close()
    
    print(f"Generated {len(runs)} figures successfully!")
    return None 

def visualize_point_selection(points, selected_idx, title=None, save_path=None):
    """
    Visualize all points and highlight selected points on a 2D plane.
    Args:
        points (np.ndarray): Array of shape (N, 2) for all points.
        selected_idx (np.ndarray): Indices of selected points.
        title (str, optional): Title for the plot.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='gray', s=10, label='All points', alpha=0.5)
    if len(selected_idx) > 0:
        plt.scatter(points[selected_idx, 0], points[selected_idx, 1], c='red', s=20, label='Selected points', alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    if title:
        plt.title(title)
    plt.gca().set_aspect('equal', 'box')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show() 

def plot_experiment_results_extended(points, runs, k_neighbours=8, p=2):
    print('[DEBUG] plot_experiment_results_extended: points shape:', points.shape)
    print('[DEBUG] plot_experiment_results_extended: runs count:', len(runs))
    assert points.shape[0] > 0, 'No points to plot!'
    
    # Precompute weights once
    W = knn_weights(points, k=k_neighbours)
    
    # Create individual figures for each run
    for run_idx, run in enumerate(runs):
        print(f'[DEBUG] Plotting run {run_idx+1}')
        initial_idx = run['initial_idx']
        final_idx = run['final_idx']
        radial_pdf = run['radial_pdf']
        pdf_final = run['pdf_final']
        noise = run['noise']
        center = run['center']
        params = run['parameters']
        
        # Extract parameters
        sigma_radial = params['sigma_radial']
        sigma_noise = params['sigma_noise']
        lambda_corr = params['lambda_corr']
        angle = params['angle']
        length1 = params['length1']
        length2 = params['length2']
        scale_factor = params['scale_factor']
        sigma_logistic = params['sigma_logistic']
        
        # Create rotation and covariance matrices
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        D = np.array([[length1, 0],
                      [0, length2]])
        cov_ellipse = R @ D @ D @ R.T
        inv_cov = inv_2x2(cov_ellipse)
        
        # Calculate statistics
        I_initial = morans_I_binary(np.isin(np.arange(len(points)), initial_idx), W)
        I_final = morans_I_binary(np.isin(np.arange(len(points)), final_idx), W)
        init_inertia = calculate_inertia(points, initial_idx)
        final_inertia = calculate_inertia(points, final_idx)
        
        # Compute noise_pdf
        noise_pdf = 2 * (logistic(noise, sigma_logistic=sigma_logistic) - 0.5)  # Between -1 and 1
        
        # Create figure with 5 subplots (2 rows, 3 columns)
        fig = plt.figure(figsize=(24, 16))  # Increased figure size
        axes = fig.subplots(2, 3)
        axes = axes.flatten()
        
        # Create grid for density plot
        x_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
        y_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Elliptical Mahalanobis distance for the grid
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        grid_diff = grid_points - center
        mahalanobis2_grid = np.sqrt(np.sum((grid_diff @ inv_cov) * grid_diff, axis=1))
        Z = scale_factor * np.exp(-(1/(2 * 9**(p-1))) * np.power(mahalanobis2_grid.reshape(X.shape)/sigma_radial, 2*p))
        
        # Plot 1: noise_pdf spatial distribution
        scatter1 = axes[0].scatter(points[:, 0], points[:, 1], c=noise_pdf, cmap='RdBu_r', s=20)
        cbar1 = plt.colorbar(scatter1, ax=axes[0], label='noise_pdf value', pad=0.1)
        cbar1.ax.tick_params(labelsize=10)
        axes[0].set_title(f'Noise Distribution\nScale: {scale_factor:.1f}, σ: {sigma_radial:.2f}, σ_cov: {sigma_noise:.2f}\nσ_logistic: {sigma_logistic:.1e}', pad=20)
        axes[0].set_aspect('equal', 'box')
        
        # Plot 2: Initial subset (no noise)
        scatter2 = axes[1].scatter(points[:, 0], points[:, 1], s=5, alpha=0.3, label='Background points')
        scatter3 = axes[1].scatter(points[initial_idx, 0], points[initial_idx, 1], s=20, label='Selected points')
        contour1 = axes[1].contour(X, Y, Z, levels=10, alpha=0.5, colors='gray')
        im1 = axes[1].contourf(X, Y, Z, levels=10, alpha=0.3)
        cbar2 = plt.colorbar(im1, ax=axes[1], label='Density', pad=0.1)
        cbar2.ax.tick_params(labelsize=10)
        axes[1].set_title(f'No Noise Radial Points\nAngle: {np.degrees(angle):.1f}°, Lengths: {length1:.2f}, {length2:.2f}\nMoran I: {I_initial:.3f}, Inertia: {init_inertia:.1f}', pad=20)
        axes[1].set_aspect('equal', 'box')
        axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Plot 3: Final subset
        scatter4 = axes[2].scatter(points[:, 0], points[:, 1], s=5, alpha=0.3, label='Background points')
        scatter5 = axes[2].scatter(points[final_idx, 0], points[final_idx, 1], s=20, label='Selected points')
        contour2 = axes[2].contour(X, Y, Z, levels=10, alpha=0.5, colors='gray')
        im2 = axes[2].contourf(X, Y, Z, levels=10, alpha=0.3)
        cbar3 = plt.colorbar(im2, ax=axes[2], label='Density', pad=0.1)
        cbar3.ax.tick_params(labelsize=10)
        
        # Add equiprobability contours for pdf_final
        grid_pdf_final = griddata(points, pdf_final, (X, Y), method='linear', fill_value=0)
        all_levels = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        min_val, max_val = np.nanmin(grid_pdf_final), np.nanmax(grid_pdf_final)
        valid_levels = [lvl for lvl in all_levels if min_val < lvl < max_val]
        if valid_levels:
            colors = ['red', 'blue'] * ((len(valid_levels) + 1) // 2)
            eq_contours = axes[2].contour(X, Y, grid_pdf_final, levels=valid_levels, colors=colors[:len(valid_levels)], linewidths=1.5, linestyles='dashed')
            axes[2].clabel(eq_contours, fmt=lambda v: f'{v:.4f}', fontsize=9)
        
        # Add statistics to the final plot
        n_selected = len(final_idx)
        n_total = len(points)
        selection_ratio = n_selected / n_total
        axes[2].set_title(f'Final Selected Points\nλ: {lambda_corr:.2f}, Selected: {n_selected}/{n_total} ({selection_ratio:.1%})\nMoran I: {I_final:.3f}, Inertia: {final_inertia:.1f}', pad=20)
        axes[2].set_aspect('equal', 'box')
        axes[2].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Plot 4: Radial function probability distribution
        im3 = axes[3].contourf(X, Y, Z, levels=20, cmap='viridis')
        cbar4 = plt.colorbar(im3, ax=axes[3], label='Probability', pad=0.1)
        cbar4.ax.tick_params(labelsize=10)
        axes[3].set_title('Radial Function Probability Distribution', pad=20)
        axes[3].set_aspect('equal', 'box')
        
        # Plot 5: noise_pdf probability distribution
        grid_pdf_x = griddata(points, noise_pdf, (X, Y), method='linear', fill_value=0)
        im4 = axes[4].contourf(X, Y, grid_pdf_x, levels=20, cmap='RdBu_r')
        cbar5 = plt.colorbar(im4, ax=axes[4], label='noise_pdf value', pad=0.1)
        cbar5.ax.tick_params(labelsize=10)
        axes[4].set_title('Noise Probability Distribution', pad=20)
        axes[4].set_aspect('equal', 'box')
        
        # Plot 6: Final probability distribution
        im5 = axes[5].contourf(X, Y, grid_pdf_final, levels=20, cmap='viridis')
        cbar6 = plt.colorbar(im5, ax=axes[5], label='Probability', pad=0.1)
        cbar6.ax.tick_params(labelsize=10)
        axes[5].set_title('Final Probability Distribution', pad=20)
        axes[5].set_aspect('equal', 'box')
        
        # Add a main title with all parameters
        fig.suptitle(f'Figure {run_idx+1}: Scale={scale_factor:.1f}, Angle={np.degrees(angle):.1f}°, Lengths={length1:.2f},{length2:.2f}\nσ={sigma_radial:.2f}, σ_cov={sigma_noise:.2f}, σ_logistic={sigma_logistic:.1e}, λ={lambda_corr:.2f}, p={p}', y=0.95, fontsize=14)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create filename with parameters
        filename = f'scale{scale_factor:.1f}_id{run_idx+1}_angle{np.degrees(angle):.1f}_len{length1:.1f}_{length2:.1f}_p{p}_extended.png'
        plt.savefig(os.path.join('figures', 'magnitude', filename), bbox_inches='tight')
        plt.show()
        plt.close()
    
    print(f"Generated {len(runs)} extended figures successfully!")
    return None 