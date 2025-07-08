#!/usr/bin/env python3
"""
Compare Real vs Synthetic Earthquake Data
=========================================

This script:
1. Loads real earthquake catalog data
2. Extracts picks, sources, and station selections for P and S waves
3. Determines magnitude for the source
4. Generates synthetic data using run_single_experiment with same parameters
5. Visualizes real vs synthetic data side by side

Usage:
    python compare_real_synthetic_events.py [--event_id EVENT_ID] [--day_file DAY_FILE]
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import os
import sys
import argparse
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add paths for imports (adjust as needed)
sys.path.append('.')
sys.path.append('./training_data_scripts/')

# Import necessary functions
try:
    from training_data_scripts.data_generation_utils import (
        run_single_experiment, 
        load_distance_magnitude_model,
        prepare_station_coordinates,
        plot_experiment_results_extended,
        visualize_point_selection
    )
    from utils_simple import *
except ImportError as e:
    print(f"Warning: Could not import all required functions: {e}")
    print("Some features may not work correctly")

class RealSyntheticEventComparison:
    """Main class for comparing real and synthetic earthquake events."""
    
    def __init__(self, stations_file='CentralCalifornia_stations.npz'):
        """
        Initialize the comparison class.
        
        Args:
            stations_file (str): Path to station locations file
        """
        self.load_stations(stations_file)
        self.load_magnitude_model()
        
    def load_stations(self, stations_file): #OK
        """Load station locations and metadata."""
        try:
            z = np.load(stations_file)
            self.locs = z['locs']  # Station locations [lat, lon, elev]
            self.stas = z['stas']  # Station names

            self.mn = z['mn']
            self.rbest = z['rbest']

            earth_radius = 6371e3
            def func1(x):
                return (self.rbest @ (lla2ecef(x, e = 0.0, a = earth_radius) - self.mn).T).T
            self.ftrns1 = func1            
            self.ftrns2 = lambda x: ecef2lla((self.rbest.T @ x.T).T + self.mn, e = 0.0, a = earth_radius)

            z.close()
            print(f"Loaded {len(self.locs)} stations from {stations_file}")
        except FileNotFoundError:
            print(f"Error: Station file {stations_file} not found")
            sys.exit(1)
            
    def load_magnitude_model(self): # OK, f: mag -> dist
        """Load magnitude-distance model for synthetic data generation."""
        try:
            self.pdist_p, self.pdist_s = load_distance_magnitude_model(n_mag_ver=1, use_softplus=True)
            print("Loaded magnitude-distance model")
        except Exception as e:
            print(f"Warning: Could not load magnitude model: {e}")
            self.pdist_p = None
            self.pdist_s = None
    
    def load_real_event_data(self, catalog_file): # Load ONE DAY not catalog
        """
        Load real earthquake data from HDF5 catalog file.
        
        Args:
            catalog_file (str): Path to HDF5 catalog file
            
        Returns:
            dict: Dictionary containing event data
        """
        try:
            with h5py.File(catalog_file, 'r') as z:
                data = {
                    'locs_use': z['locs_use'][:],
                    'srcs_trv': z['srcs_trv'][:],  # [lat, lon, depth, time]
                    'mag_r': z['mag_r'][:],
                    'P_perm': z['P_perm'][:],
                    'Picks_P': [z['Picks/%d_Picks_P' % j][:] for j in range(len(z['srcs_trv'][:]))],
                    'Picks_S': [z['Picks/%d_Picks_S' % j][:] for j in range(len(z['srcs_trv'][:]))],
                    'Picks_P_perm': [z['Picks/%d_Picks_P_perm' % j][:] for j in range(len(z['srcs_trv'][:]))],
                    'Picks_S_perm': [z['Picks/%d_Picks_S_perm' % j][:] for j in range(len(z['srcs_trv'][:]))]
                }
            
            print(f"Loaded catalog with {len(data['srcs_trv'])} events from {catalog_file}")
            return data
            
        except Exception as e:
            print(f"Error loading catalog file {catalog_file}: {e}")
            return None
    
    def extract_event_picks(self, data, event_id):
        """
        Extract picks and station information for a specific event.
        
        Args:
            data (dict): Event data dictionary
            event_id (int): Index of the event to analyze
            
        Returns:
            dict: Extracted event information
        """
        if event_id >= len(data['srcs_trv']):
            print(f"Error: Event ID {event_id} out of range (max: {len(data['srcs_trv'])-1})")
            return None
            
        # Extract source information
        source = data['srcs_trv'][event_id]  # [lat, lon, depth, origin_time]
        magnitude = data['mag_r'][event_id]
        
        # Extract picks
        picks_p = data['Picks_P_perm'][event_id] if event_id < len(data['Picks_P_perm']) else np.array([])
        picks_s = data['Picks_S_perm'][event_id] if event_id < len(data['Picks_S_perm']) else np.array([])
        
        # Get station indices
        if len(picks_p) > 0:
            p_stations = np.unique(picks_p[:, 1]).astype(int) # TODO: check if this is correct
        else:
            p_stations = np.array([])
            
        if len(picks_s) > 0:
            s_stations = np.unique(picks_s[:, 1]).astype(int) # TODO: check if this is correct
        else:
            s_stations = np.array([])
        
        # Categorize stations TODO: check if this is correct
        p_only_stations = list(set(p_stations) - set(s_stations))
        s_only_stations = list(set(s_stations) - set(p_stations))
        both_stations = list(set(p_stations) & set(s_stations))
        all_detecting_stations = list(set(p_stations) | set(s_stations))
        
        event_info = {
            'event_id': event_id,
            'source': source,
            'magnitude': magnitude,
            'picks_p': picks_p,
            'picks_s': picks_s,
            'p_stations': p_stations,
            's_stations': s_stations,
            'p_only_stations': p_only_stations,
            's_only_stations': s_only_stations,
            'both_stations': both_stations,
            'all_detecting_stations': all_detecting_stations,
            'locs_use': data['locs_use']
        }
        
        print(f"\nEvent {event_id} Summary:")
        print(f"  Source: Lat={source[0]:.3f}, Lon={source[1]:.3f}, Depth={source[2]:.1f}km")
        print(f"  Magnitude: {magnitude:.2f}")
        print(f"  P-wave picks: {len(picks_p)} at {len(p_stations)} stations")
        print(f"  S-wave picks: {len(picks_s)} at {len(s_stations)} stations")
        print(f"  Station breakdown: {len(p_only_stations)} P-only, {len(s_only_stations)} S-only, {len(both_stations)} both")
        
        return event_info
    
    def generate_synthetic_data(self, event_info):
        """
        Generate synthetic data using the same source and magnitude as real event.
        
        Args:
            event_info (dict): Real event information
            
        Returns:
            dict: Synthetic event results
        """
        
        # weird conversion here but ok because we cancel it after that?
        # Prepare station coordinates (convert to 2D for radial cholesky)
        station_coords = self.ftrns1(event_info['locs_use'])[:, 0:2]
        
        # Source center in same coordinate system
        # print(f"Source shape {np.shape(event_info['source'])}")
        # src = np.array(event_info['source']).reshape(-1,1)
        # print(f"Src shape {np.shape(src)}")
        center = self.ftrns1(np.array(event_info['source']).reshape(1,-1))[0,0:2]
        
        space_size = max(station_coords[:, 0].max() - station_coords[:, 0].min(),
                        station_coords[:, 1].max() - station_coords[:, 1].min())
        max_noise_spread = space_size / 6

        sigma_radial_p_factor = 1.1096087666753762
        sigma_radial_s_factor =  1.2085845276778682
        sigma_radial_divider = 7.0

        if self.pdist_p is None or self.pdist_s is None:
            print("Warning: No magnitude model loaded, using default parameters")
            sigma_radial_p = 50000  # 50 km default
            sigma_radial_s = 60000  # 60 km default
        else:
            # Calculate detection radii based on magnitude
            magnitude = event_info['magnitude']
            if np.isnan(magnitude):
                print("Magnitude is nan, setting to 1.0")
                magnitude = 1.0
            sigma_radial_p = sigma_radial_p_factor * self.pdist_p(magnitude) / sigma_radial_divider
            sigma_radial_s = sigma_radial_s_factor * self.pdist_s(magnitude) / sigma_radial_divider
        # Calculate noise parameter based on station distribution
        
        # Set parameters for synthetic generation
        sigma_noise = 20877
        p = 2
        scale_factor = 0.963
        threshold_logistic = 7.0
        max_value_logistic = 0.99 # < 1, the maximum value of the logistic function for the threshold, don't tune this.
        sigma_logistic = - threshold_logistic / np.log(1/max_value_logistic - 1) 
        lambda_corr = 0.25

        
        print(f"\nGenerating synthetic data with:")
        print(f"  Magnitude: {magnitude:.2f}")
        print(f"  P-wave radius: {sigma_radial_p:.0f}m")
        print(f"  S-wave radius: {sigma_radial_s:.0f}m")
        print(f"  Noise correlation spread: {sigma_noise:.0f}m")
        
        # Generate P-wave synthetic data
        synthetic_p = run_single_experiment(
            points=station_coords,
            sigma_radial=sigma_radial_p,
            sigma_noise=sigma_noise,
            sigma_logistic=sigma_logistic,
            lambda_corr=lambda_corr,
            p=p,
            scale_factor=scale_factor,
            center=center
        )
        # Save the noise
        noise_p = synthetic_p['noise']
        # Generate S-wave synthetic data with slight variations
        angle_p = synthetic_p['parameters']['angle']
        length1_p = synthetic_p['parameters']['length1']
        length2_p = synthetic_p['parameters']['length2']
        
        # Add small random variations for S-waves
        angle_s = (angle_p + np.random.uniform(-np.pi/8, np.pi/8)) % (2*np.pi)
        length1_s = length1_p * np.random.uniform(0.95, 1.05)
        length2_s = length2_p * np.random.uniform(0.95, 1.05)
        
        synthetic_s = run_single_experiment(
            points=station_coords,
            sigma_radial=sigma_radial_s,
            sigma_noise=sigma_noise,
            sigma_logistic=sigma_logistic,
            lambda_corr=lambda_corr,
            p=p,
            scale_factor=scale_factor,
            center=center,
            angle=angle_s,
            length1=length1_s,
            length2=length2_s,
            noise=noise_p
        )
        
        return {
            'synthetic_p': synthetic_p,
            'synthetic_s': synthetic_s,
            'station_coords': station_coords,
            'parameters': {
                'magnitude': magnitude,
                'sigma_radial_p': sigma_radial_p,
                'sigma_radial_s': sigma_radial_s,
                'sigma_noise': sigma_noise,
                'center': center
            }
        }
    
    def plot_real_event(self, event_info, ax, title_prefix="Real"):
        """
        Plot real event data.
        
        Args:
            event_info (dict): Real event information
            ax: Matplotlib axis object
            title_prefix (str): Prefix for plot title
        """
        locs_use = event_info['locs_use']
        source = event_info['source']
        
        # Plot all stations in gray
        ax.scatter(locs_use[:, 1], locs_use[:, 0], c='lightgray', marker='^', 
                  s=10, alpha=0.5, label='All stations')
        
        # Plot stations by type
        if event_info['p_only_stations']:
            p_only_locs = locs_use[event_info['p_only_stations']]
            ax.scatter(p_only_locs[:, 1], p_only_locs[:, 0], 
                      c='red', marker='^', s=30, 
                      label=f'P-only ({len(event_info["p_only_stations"])})', alpha=0.8)
        
        if event_info['s_only_stations']:
            s_only_locs = locs_use[event_info['s_only_stations']]
            ax.scatter(s_only_locs[:, 1], s_only_locs[:, 0], 
                      c='blue', marker='^', s=30, 
                      label=f'S-only ({len(event_info["s_only_stations"])})', alpha=0.8)
        
        if event_info['both_stations']:
            both_locs = locs_use[event_info['both_stations']]
            ax.scatter(both_locs[:, 1], both_locs[:, 0], 
                      c='purple', marker='^', s=30, 
                      label=f'Both P&S ({len(event_info["both_stations"])})', alpha=0.8)
        
        # Plot source
        ax.scatter(source[1], source[0], c='orange', marker='*', s=200, 
                  label=f'Source (M{event_info["magnitude"]:.1f})', 
                  edgecolors='black', linewidth=1)
        
        # Add connection lines with low alpha
        for sta_idx in event_info['all_detecting_stations']:
            if sta_idx < len(locs_use):
                ax.plot([source[1], locs_use[sta_idx, 1]], 
                       [source[0], locs_use[sta_idx, 0]], 
                       'k-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title_prefix} Event {event_info["event_id"]} (M{event_info["magnitude"]:.1f})')
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
    
    def plot_synthetic_event(self, synthetic_data, event_info, ax, title_prefix="Synthetic"):
        """
        Plot synthetic event data.
        
        Args:
            synthetic_data (dict): Synthetic event data
            event_info (dict): Original real event info for reference
            ax: Matplotlib axis object
            title_prefix (str): Prefix for plot title
        """
        station_coords = synthetic_data['station_coords']
        synthetic_p = synthetic_data['synthetic_p']
        synthetic_s = synthetic_data['synthetic_s']
        
        # Convert station coordinates back to lat/lon using the inverse transformation
        station_coords_latlon = self.ftrns2(np.hstack([station_coords, np.zeros((station_coords.shape[0], 1))]))
        station_lats = station_coords_latlon[:, 0]
        station_lons = station_coords_latlon[:, 1]
        
        # Get selected stations
        p_selected = synthetic_p['final_idx']
        s_selected = synthetic_s['final_idx']
        
        # Categorize stations
        p_only = list(set(p_selected) - set(s_selected))
        s_only = list(set(s_selected) - set(p_selected))
        both = list(set(p_selected) & set(s_selected))
        all_selected = list(set(p_selected) | set(s_selected))
        
        # Plot all stations in gray
        ax.scatter(station_lons, station_lats, c='lightgray', marker='^', 
                  s=10, alpha=0.5, label='All stations')
        
        # Plot selected stations by type
        if p_only:
            ax.scatter(station_lons[p_only], station_lats[p_only], 
                      c='red', marker='^', s=30, 
                      label=f'P-only ({len(p_only)})', alpha=0.8)
        
        if s_only:
            ax.scatter(station_lons[s_only], station_lats[s_only], 
                      c='blue', marker='^', s=30, 
                      label=f'S-only ({len(s_only)})', alpha=0.8)
        
        if both:
            ax.scatter(station_lons[both], station_lats[both], 
                      c='purple', marker='^', s=30, 
                      label=f'Both P&S ({len(both)})', alpha=0.8)
        
        # Plot source (same as real event)
        source = event_info['source']
        ax.scatter(source[1], source[0], c='orange', marker='*', s=200, 
                  label=f'Source (M{event_info["magnitude"]:.1f})', 
                  edgecolors='black', linewidth=1)
        
        # Add connection lines
        for sta_idx in all_selected:
            ax.plot([source[1], station_lons[sta_idx]], 
                   [source[0], station_lats[sta_idx]], 
                   'k-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title_prefix} Event (M{event_info["magnitude"]:.1f})')
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
    
    def compare_events(self, event_info, synthetic_data, save_path=None):
        """
        Create side-by-side comparison of real vs synthetic events.
        
        Args:
            event_info (dict): Real event information
            synthetic_data (dict): Synthetic event data
            save_path (str): Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot real event
        self.plot_real_event(event_info, axes[0], "Real")
        
        # Plot synthetic event
        self.plot_synthetic_event(synthetic_data, event_info, axes[1], "Synthetic")
        
        # Add summary statistics
        real_p_count = len(event_info['p_stations'])
        real_s_count = len(event_info['s_stations'])
        real_total = len(event_info['all_detecting_stations'])
        
        synth_p_count = len(synthetic_data['synthetic_p']['final_idx'])
        synth_s_count = len(synthetic_data['synthetic_s']['final_idx'])
        synth_total = len(set(synthetic_data['synthetic_p']['final_idx']) | 
                         set(synthetic_data['synthetic_s']['final_idx']))
        
        fig.suptitle(f'Real vs Synthetic Event Comparison\n' +
                    f'Real: {real_total} stations ({real_p_count} P, {real_s_count} S) | ' +
                    f'Synthetic: {synth_total} stations ({synth_p_count} P, {synth_s_count} S)',
                    fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
        
        # Print detailed comparison
        print(f"\n=== DETAILED COMPARISON ===")
        print(f"Real Event:")
        print(f"  Total detecting stations: {real_total}")
        print(f"  P-wave detections: {real_p_count}")
        print(f"  S-wave detections: {real_s_count}")
        print(f"  P-only stations: {len(event_info['p_only_stations'])}")
        print(f"  S-only stations: {len(event_info['s_only_stations'])}")
        print(f"  Both P&S stations: {len(event_info['both_stations'])}")
        
        print(f"\nSynthetic Event:")
        print(f"  Total detecting stations: {synth_total}")
        print(f"  P-wave detections: {synth_p_count}")
        print(f"  S-wave detections: {synth_s_count}")
        
        p_only_synth = set(synthetic_data['synthetic_p']['final_idx']) - set(synthetic_data['synthetic_s']['final_idx'])
        s_only_synth = set(synthetic_data['synthetic_s']['final_idx']) - set(synthetic_data['synthetic_p']['final_idx'])
        both_synth = set(synthetic_data['synthetic_p']['final_idx']) & set(synthetic_data['synthetic_s']['final_idx'])
        
        print(f"  P-only stations: {len(p_only_synth)}")
        print(f"  S-only stations: {len(s_only_synth)}")
        print(f"  Both P&S stations: {len(both_synth)}")

        print(f"Synthetic p magnitude: {synthetic_data['synthetic_p']['parameters']['magnitude']}")
        print(f"Synthetic s magnitude: {synthetic_data['synthetic_s']['parameters']['magnitude']}")
        print(f"Synthetic p angle: {synthetic_data['synthetic_p']['parameters']['angle']}")
        print(f"Synthetic s angle: {synthetic_data['synthetic_s']['parameters']['angle']}")
        print(f"Synthetic p length1: {synthetic_data['synthetic_p']['parameters']['length1']}")
        print(f"Synthetic s length1: {synthetic_data['synthetic_s']['parameters']['length1']}")
        print(f"Synthetic p length2: {synthetic_data['synthetic_p']['parameters']['length2']}")
        plot_experiment_results_extended(synthetic_data['station_coords'], [synthetic_data['synthetic_p'], synthetic_data['synthetic_s']], k_neighbours=8, p=4)

def find_catalog_files(catalog_dir="Catalog"):
    """Find available catalog files from any subdirectory of the catalog directory.
    
    Args:
        catalog_dir (str): Directory to search for catalog files
        
    Returns:
        list: List of catalog file paths ordered by date (only 2023 in Catalog2)
    """
    if not os.path.exists(catalog_dir):
        print(f"Error: Catalog directory '{catalog_dir}' not found")
        return []
    
    # Look for HDF5 files in subdirectories
    pattern1 = os.path.join(catalog_dir, "*/*ver_1.hdf5")
    pattern2 = os.path.join(catalog_dir, "**/*ver_1.hdf5")
    
    files = glob.glob(pattern1) + glob.glob(pattern2, recursive=True)
    
    if not files:
        print(f"No catalog files found in {catalog_dir}")
        print("Looking for files matching patterns: */*ver_1.hdf5")
    
    return sorted(files)


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description='Compare real vs synthetic earthquake events')
    parser.add_argument('--event_id', type=int, default=0, help='Event ID to analyze (default: 0)')
    parser.add_argument('--day_file', type=str, help='Specific catalog file to use')
    parser.add_argument('--catalog_dir', type=str, default='Catalog', help='Catalog directory path')
    parser.add_argument('--stations_file', type=str, default='CentralCalifornia_stations.npz', 
                       help='Station file path')
    parser.add_argument('--save_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--random_event', action='store_true', help='Select a random event')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize comparison object
    print("Initializing comparison...")
    comparison = RealSyntheticEventComparison(stations_file=args.stations_file)
    
    # Find catalog files
    if args.day_file:
        catalog_files = [args.day_file] if os.path.exists(args.day_file) else []
    else:
        catalog_files = find_catalog_files(args.catalog_dir)
    
    if not catalog_files:
        print(f"Catalog directory: {args.catalog_dir}")
        print("No catalog files found. Please check the catalog directory or specify a file.")
        return
    
    # Select a catalog file
    if len(catalog_files) > 1:
        print(f"\nFound {len(catalog_files)} catalog files:")
        for i, f in enumerate(catalog_files[:10]):  # Show first 10
            print(f"  {i}: {f}")
        if len(catalog_files) > 10:
            print(f"  ... and {len(catalog_files)-10} more")
        
        if not args.day_file:
            catalog_file = np.random.choice(catalog_files)
            print(f"\nRandomly selected: {catalog_file}")
        else: 
            catalog_file = catalog_files[0]
    else:
        catalog_file = catalog_files[0]
    
    # Load real event data
    print(f"\nLoading catalog: {catalog_file}")
    data = comparison.load_real_event_data(catalog_file)
    if data is None:
        return
    
    # Select event
    n_events = len(data['srcs_trv'])
    if args.random_event:
        event_id = np.random.randint(0, n_events)
        print(f"Randomly selected event {event_id} out of {n_events}")
    else:
        event_id = min(args.event_id, n_events - 1)
    
    # Extract event information
    print(f"\nAnalyzing event {event_id}...")
    event_info = comparison.extract_event_picks(data, event_id)
    if event_info is None:
        return
    
    # Check if event has enough detections
    if len(event_info['all_detecting_stations']) < 3:
        print(f"Warning: Event {event_id} has only {len(event_info['all_detecting_stations'])} detecting stations")
        print("Proceeding anyway...")
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    synthetic_data = comparison.generate_synthetic_data(event_info)
    
    # Create comparison plot
    save_path = os.path.join(args.save_dir, f'real_vs_synthetic_event_{event_id}.png')
    comparison.compare_events(event_info, synthetic_data, save_path=save_path)
    
    # Generate detailed synthetic plots if requested
    create_detailed_plots = True
    if create_detailed_plots:
        print(f"\nCreating detailed synthetic plots...")
        try:
            # This requires the extended plotting function
            plot_experiment_results_extended(
                synthetic_data['station_coords'], 
                [synthetic_data['synthetic_p'], synthetic_data['synthetic_s']], 
                k_neighbours=8, 
                p=2
            )
        except Exception as e:
            print(f"Could not create detailed plots: {e}")
    
    print(f"\nComparison complete!")


if __name__ == "__main__":
    main() 