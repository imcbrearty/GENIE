#!/usr/bin/env python3
"""
Example Usage of Real vs Synthetic Event Comparison
===================================================

This script demonstrates how to use the compare_real_synthetic_events.py
functionality to analyze real earthquakes and compare them with synthetic data.
"""

import sys
import os
import numpy as np

# Import the comparison class
from compare_real_synthetic_events import RealSyntheticEventComparison, find_catalog_files

def run_basic_example():
    """Run a basic example with default settings."""
    print("=" * 60)
    print("BASIC EXAMPLE: Compare Real vs Synthetic Events")
    print("=" * 60)
    
    # Initialize the comparison object
    # This will load stations and magnitude model
    comparison = RealSyntheticEventComparison()
    
    # Find available catalog files
    catalog_files = find_catalog_files("../CentralCalifornia1/Catalog/")
    if not catalog_files:
        print("No catalog files found. Please ensure you have catalog data available.")
        return
    
    # Select a random catalog file
    catalog_file = np.random.choice(catalog_files)
    print(f"Using catalog: {catalog_file}")
    
    # Load the catalog data
    data = comparison.load_real_event_data(catalog_file)
    if data is None:
        print("Failed to load catalog data.")
        return
    
    # Select a random event with enough detections
    n_events = len(data['srcs_trv'])
    
    # Try several random events to find one with good detections
    for attempt in range(10):
        event_id = np.random.randint(0, n_events)
        event_info = comparison.extract_event_picks(data, event_id)
        
        if event_info and len(event_info['all_detecting_stations']) >= 5:
            print(f"Selected event {event_id} with {len(event_info['all_detecting_stations'])} detecting stations")
            break
    else:
        # Fallback to first event
        event_id = 0
        event_info = comparison.extract_event_picks(data, event_id)
        print(f"Using first event (ID: {event_id}) as fallback")
    
    if event_info is None:
        print("Failed to extract event information.")
        return
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_data = comparison.generate_synthetic_data(event_info)
    
    # Create comparison plots
    os.makedirs('example_plots', exist_ok=True)
    save_path = f'example_plots/example_event_{event_id}_comparison.png'
    
    print(f"\nCreating comparison plot...")
    comparison.compare_events(event_info, synthetic_data, save_path=save_path)
    
    print("\nExample complete!")

def run_specific_event_example():
    """Run example with a specific event ID."""
    print("=" * 60)
    print("SPECIFIC EVENT EXAMPLE")
    print("=" * 60)
    
    comparison = RealSyntheticEventComparison()
    catalog_files = find_catalog_files("../CentralCalifornia1/Catalog/")
    
    if not catalog_files:
        print("No catalog files found.")
        return
    
    # Use first available catalog
    catalog_file = catalog_files[0]
    data = comparison.load_real_event_data(catalog_file)
    
    if data is None:
        return
    
    # Analyze first few events
    n_events_to_analyze = min(3, len(data['srcs_trv']))
    
    for event_id in range(n_events_to_analyze):
        print(f"\n--- Analyzing Event {event_id} ---")
        
        event_info = comparison.extract_event_picks(data, event_id)
        if event_info is None:
            continue
            
        # Skip events with too few detections
        if len(event_info['all_detecting_stations']) < 3:
            print(f"Skipping event {event_id} (only {len(event_info['all_detecting_stations'])} detections)")
            continue
        
        # Generate synthetic data
        synthetic_data = comparison.generate_synthetic_data(event_info)
        
        # Create comparison plot
        os.makedirs('example_plots', exist_ok=True)
        save_path = f'example_plots/event_{event_id}_detailed_comparison.png'
        comparison.compare_events(event_info, synthetic_data, save_path=save_path)

def analyze_magnitude_effect():
    """Analyze how magnitude affects synthetic detection patterns."""
    print("=" * 60)
    print("MAGNITUDE EFFECT ANALYSIS")
    print("=" * 60)
    
    comparison = RealSyntheticEventComparison()
    catalog_files = find_catalog_files("../CentralCalifornia1/Catalog/")
    
    if not catalog_files:
        print("No catalog files found.")
        return
    
    # Load a catalog
    catalog_file = catalog_files[0]
    data = comparison.load_real_event_data(catalog_file)
    
    if data is None:
        return
    
    # Find events of different magnitudes
    magnitudes = data['mag_r']
    magnitude_ranges = [
        (1.0, 2.5, "Small"),
        (2.5, 4.0, "Medium"), 
        (4.0, 6.0, "Large")
    ]
    
    os.makedirs('magnitude_analysis', exist_ok=True)
    
    for mag_min, mag_max, size_label in magnitude_ranges:
        # Find events in this magnitude range
        in_range = (magnitudes >= mag_min) & (magnitudes < mag_max)
        event_ids = np.where(in_range)[0]
        
        if len(event_ids) == 0:
            print(f"No {size_label.lower()} events found in range [{mag_min}, {mag_max})")
            continue
        
        # Select a random event from this range
        event_id = np.random.choice(event_ids)
        
        print(f"\nAnalyzing {size_label} Event (ID: {event_id}, M{magnitudes[event_id]:.1f})")
        
        event_info = comparison.extract_event_picks(data, event_id)
        if event_info is None or len(event_info['all_detecting_stations']) < 3:
            continue
        
        # Generate synthetic data
        synthetic_data = comparison.generate_synthetic_data(event_info)
        
        # Create comparison plot
        save_path = f'magnitude_analysis/{size_label.lower()}_event_M{magnitudes[event_id]:.1f}_comparison.png'
        comparison.compare_events(event_info, synthetic_data, save_path=save_path)

def main():
    """Main function to run examples."""
    print("Real vs Synthetic Event Comparison Examples")
    print("==========================================")
    
    # Check if required files exist
    if not os.path.exists('CentralCalifornia_stations.npz'):
        print("Error: Station file 'CentralCalifornia_stations.npz' not found.")
        print("Please ensure you have the required data files.")
        return
    
    if not os.path.exists('Catalog'):
        print("Error: 'Catalog' directory not found.")
        print("Please ensure you have catalog data available.")
        return
    
    # Run examples
    try:
        print("\n1. Running basic example...")
        run_basic_example()
        
        print("\n2. Running specific event analysis...")
        run_specific_event_example()
        
        print("\n3. Running magnitude effect analysis...")
        analyze_magnitude_effect()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This may be due to missing data files or import issues.")
    
    print("\nAll examples completed!")
    print("Check the 'example_plots' and 'magnitude_analysis' directories for results.")

if __name__ == "__main__":
    main() 