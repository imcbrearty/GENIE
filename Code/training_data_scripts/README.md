# Training Data Generation Scripts

## Required Files

The following files are required but not included in the repository:

1. `distance_magnitude_model_ver_1.npz`
   - Contains parameters for distance-magnitude relationships
   - Place in the same directory as the scripts

2. `CentralCalifornia_stations.npz`
   - Contains station coordinates for Central California
   - Place in the same directory as the scripts

## Usage

The main utility file `data_generation_utils.py` provides functions for generating synthetic training data with spatial correlations. Key functions include:

- `load_distance_magnitude_model()`: Loads the distance-magnitude relationship model
- `prepare_station_coordinates()`: Processes station coordinates with optional dropout
- `run_single_experiment()`: Generates a single synthetic dataset with:
  - Spatially correlated noise
  - Radial probability density functions
  - Logistic transformations
  - Moran's I spatial correlation metrics

Example usage:
```python
import numpy as np
from data_generation_utils import (
    prepare_station_coordinates,
    load_distance_magnitude_model,
    run_single_experiment,
    plot_experiment_results_extended
)

# Load required models and data
pdist_p, pdist_s = load_distance_magnitude_model(n_mag_ver=1, use_softplus=True)
data = np.load('CentralCalifornia_stations.npz')
locs = data['locs']

# Prepare coordinates with random dropout
dropout_rate = np.random.uniform(0.0, 0.1)
points = prepare_station_coordinates(locs, dropout_rate=dropout_rate)

# Parameters to choose
magnitude = 1.0
p = 3
sigma_radial = pdist_p(magnitude) / 6
scale_factor = 0.95
space_size = max(points[:, 0].max() - points[:, 0].min(), 
                points[:, 1].max() - points[:, 1].min())
sigma_noise = (space_size / 6) / 4
sigma_logistic = -4 / np.log(1/0.99 - 1)  # For threshold_logistic=4
lambda_corr = 0.2

# Run one experiment
run_data = run_single_experiment(
    points=points,
    sigma_radial=sigma_radial,
    sigma_noise=sigma_noise,
    sigma_logistic=sigma_logistic,
    lambda_corr=lambda_corr,
    p=p,
    scale_factor=scale_factor
)

# Visualize results
plot_experiment_results_extended(points, runs, k_neighbours=10, p=p)
```

For visualization and analysis:
- Use `plot_experiment_results_extended` by default
<!-- - Use `create_visualization()` for single experiment plots
- Use `plot_experiment_results()` for multiple experiment comparisons with less informations
- Use `visualize_point_selection()` for point selection visualization -->
