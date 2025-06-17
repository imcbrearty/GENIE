import numpy as np
import matplotlib.pyplot as plt
import os
from data_generation_utils import (
    prepare_station_coordinates,
    load_distance_magnitude_model,
    run_single_experiment,
    plot_experiment_results_extended,
    visualize_point_selection
)
    
def main():
    np.random.seed(14)

    # -----------------------------
    # Dropout rate
    # -----------------------------
    dropout_rate = np.random.uniform(0.0, 0.1)
    print(f"dropout_rate: {dropout_rate}")
    
    # Load the distance-magnitude model
    pdist_p, pdist_s = load_distance_magnitude_model(n_mag_ver=1, use_softplus=True)
    
    # Load the station data
    data = np.load('CentralCalifornia_stations.npz')
    locs = data['locs']  # This contains the station coordinates
    
    # Prepare coordinates
    print('Preparing station coordinates...')
    points = prepare_station_coordinates(locs, dropout_rate=dropout_rate)
    N = len(points)
    print(f"Number of points: {N}")
    print("Points shape: ", points.shape)
    print(f"Points min: {points.min(axis=0)}, max: {points.max(axis=0)}")
    print(f"Sample points: {points[:5]}")

    # Calculate space size for sigma_noise
    space_size = max(points[:, 0].max() - points[:, 0].min(), 
                    points[:, 1].max() - points[:, 1].min())
    print(f"Space size: {space_size}")

    # Sizes
    max_noise_spread = space_size / 6

    # -----------------------------
    # Experiment configuration
    # -----------------------------
    # Magnitude of the cluster
    magnitude = 2.0
    # Radial function sigma_radial, controls the spreading of the cluster.
    p = 3 # TODO: tune this
    sigma_radial = pdist_p(magnitude)/ 6
    # scaling factor for the radial function
    scale_factor = 0.95 # TUNABLE: 0.95 is a good default value, but can be tuned to get better results.
    
    # Covariance matrix/kernel distances sigma_radial, controls the spreading of the cluster.
    sigma_noise = max_noise_spread / 4 #TODO: tune this # adjust between small (tight cluster, many points, small values) and big (one big cluster, few points, large values)
    
    # Logistic function sigma_radial, controls the roughness of cluster border
    threshold_logistic = 4 # TUNABLE (very binary) 0 < threshol_logistic <= ~4 (diffused) (can be more than 3 but the values are below)
    max_value_logistic = 0.99 # < 1, the maximum value of the logistic function for the threshold, don't tune this.
    sigma_logistic = - threshold_logistic / np.log(1/max_value_logistic - 1) 
    
    # Mixing function lambda, controls the correlation between the radial function and the correlated noise
    lambda_corr = 0.2  # TODO: tune this # adjust between 0 (no correlation) and 1 (max allowed) 

    k_neighbours = 10 # TODO: tune this
    # -----------------------------
    
    # Run the experiment N_runs times
    N_runs = 3
    runs = []
    for i in range(N_runs):
        print(f'Running experiment {i+1}...')
        run_data = run_single_experiment(
            points=points,
            sigma_radial=sigma_radial,
            sigma_noise=sigma_noise,
            sigma_logistic=sigma_logistic,
            lambda_corr=lambda_corr,
            p=p,
            scale_factor=scale_factor
        )
        print(f'Experiment {i+1} done. Initial idx: {run_data["initial_idx"].shape}, Final idx: {run_data["final_idx"].shape}')
        print(f'  Center: {run_data["center"]}, radial_pdf min/max: {run_data["radial_pdf"].min()}/{run_data["radial_pdf"].max()}')
        print(f'  pdf_final min/max: {run_data["pdf_final"].min()}/{run_data["pdf_final"].max()}')
        runs.append(run_data)
        # Visualize selection for this run
        # visualize_point_selection(
        #     points,
        #     run_data['final_idx'],
        #     title=f'Run {i+1}: Selected Points',
        #     save_path=f'figures/magnitude/quick_selection_run{i+1}.png'
        # )
        
    # Print summary of results
    print("\nSummary of experiment runs:")
    for i, run in enumerate(runs):
        params = run['parameters']
        print(f"\nRun {i+1}:")
        print(f"  Scale factor: {params['scale_factor']:.2f}")
        print(f"  Angle: {np.degrees(params['angle']):.1f}°")
        print(f"  Lengths: {params['length1']:.2f}, {params['length2']:.2f}")
        print(f"  Sigma radial: {params['sigma_radial']:.2f}")
        print(f"  Sigma covariance: {params['sigma_noise']:.2f}")
        print(f"  Selected points: {len(run['final_idx'])}/{len(points)}")
    
    # Create and save visualization
    print('Plotting experiment results...')
    os.makedirs('figures', exist_ok=True)
    plot_experiment_results_extended(points, runs, k_neighbours=k_neighbours, p=p)
    print('Plotting done.')

if __name__ == "__main__":
    main() 