# Real vs Synthetic Earthquake Event Comparison

This repository contains tools for comparing real earthquake events with synthetically generated events using the radial Cholesky method from GENIE training data generation.

## Overview

The comparison scripts allow you to:

1. **Extract real earthquake data** from HDF5 catalog files
2. **Identify P-wave and S-wave station detections** for specific events
3. **Determine event magnitudes** using existing magnitude models
4. **Generate synthetic detection patterns** using the same source location and magnitude
5. **Visualize side-by-side comparisons** of real vs synthetic events

## Files

- `compare_real_synthetic_events.py` - Main comparison script
- `example_usage.py` - Example script showing different usage patterns
- `README_comparison.md` - This documentation file

## Requirements

### Data Files
- `CentralCalifornia_stations.npz` - Station location data
- `Catalog/` directory with HDF5 earthquake catalog files (`*ver_1.hdf5`)
- `training_data_scripts/data_generation_utils.py` - Synthetic data generation functions

### Python Dependencies
```python
numpy
matplotlib
h5py
scipy
sklearn
```

## Usage

### Command Line Interface

Basic usage:
```bash
# Analyze a specific event
python compare_real_synthetic_events.py --event_id 5

# Use a specific catalog file
python compare_real_synthetic_events.py --day_file Catalog/2023/catalog_2023_1_1_ver_1.hdf5

# Select a random event
python compare_real_synthetic_events.py --random_event

# Specify output directory
python compare_real_synthetic_events.py --save_dir output_plots
```

### Programmatic Usage

```python
from compare_real_synthetic_events import RealSyntheticEventComparison

# Initialize comparison object
comparison = RealSyntheticEventComparison()

# Load catalog data
data = comparison.load_real_event_data('Catalog/2023/catalog_2023_1_1_ver_1.hdf5')

# Extract event information
event_info = comparison.extract_event_picks(data, event_id=0)

# Generate synthetic data with same parameters
synthetic_data = comparison.generate_synthetic_data(event_info)

# Create comparison plot
comparison.compare_events(event_info, synthetic_data, save_path='comparison.png')
```

### Example Scripts

Run the comprehensive examples:
```bash
python example_usage.py
```

This will run three different analyses:
1. **Basic Example** - Random event comparison
2. **Specific Events** - Analysis of first few events
3. **Magnitude Effect** - Compare events of different magnitudes

## Output

### Comparison Plots
Each comparison generates a side-by-side plot showing:

**Left Panel (Real Event):**
- All stations (gray triangles)
- P-wave only stations (red triangles)
- S-wave only stations (blue triangles)  
- Both P&S wave stations (purple triangles)
- Source location (orange star)
- Connection lines between source and detecting stations

**Right Panel (Synthetic Event):**
- Same layout but showing synthetically generated detection pattern
- Uses same source location and magnitude as real event
- Detection pattern generated using radial Cholesky method

### Console Output
Detailed statistics including:
```
Event 5 Summary:
  Source: Lat=36.123, Lon=-120.456, Depth=8.5km
  Magnitude: 3.2
  P-wave picks: 45 at 23 stations
  S-wave picks: 38 at 19 stations
  Station breakdown: 4 P-only, 0 S-only, 19 both

=== DETAILED COMPARISON ===
Real Event:
  Total detecting stations: 23
  P-wave detections: 23
  S-wave detections: 19
  P-only stations: 4
  S-only stations: 0
  Both P&S stations: 19

Synthetic Event:
  Total detecting stations: 25
  P-wave detections: 22
  S-wave detections: 18
  P-only stations: 7
  S-only stations: 3
  Both P&S stations: 15
```

## How It Works

### Real Event Processing
1. **Load catalog data** from HDF5 files containing earthquake sources and picks
2. **Extract station indices** for P and S wave detections
3. **Categorize stations** into P-only, S-only, or both wave types
4. **Get source parameters** (location, magnitude, time)

### Synthetic Data Generation
1. **Convert station coordinates** to 2D Cartesian system
2. **Calculate detection radii** based on magnitude-distance relationships
3. **Set up spatial parameters** for radial Cholesky method:
   - `sigma_radial_p/s`: Detection radius for P/S waves
   - `sigma_noise`: Spatial correlation scale
   - `lambda_corr`: Mixing parameter for noise correlation
4. **Generate detection patterns** using `run_single_experiment`
5. **Apply slight variations** between P and S wave patterns

### Key Parameters
- **sigma_radial**: Controls detection radius (based on magnitude)
- **sigma_noise**: Controls spatial correlation of noise
- **scale_factor**: Overall scaling of detection probability
- **lambda_corr**: Amount of spatial correlation in detection pattern
- **p**: Power parameter for Mahalanobis distance

## Validation

The comparison helps validate that:
1. **Synthetic detection patterns** are realistic compared to real events
2. **Magnitude-distance relationships** are properly calibrated
3. **Spatial correlation structures** match observed patterns
4. **P vs S wave detection ratios** are appropriate

## Troubleshooting

### Common Issues

**No catalog files found:**
- Ensure `Catalog/` directory exists
- Check that HDF5 files follow naming pattern `*ver_1.hdf5`

**Station file not found:**
- Ensure `CentralCalifornia_stations.npz` exists in working directory
- Or specify custom path with `--stations_file`

**Import errors:**
- Check that `training_data_scripts/data_generation_utils.py` exists
- Ensure all required Python packages are installed

**Events with few detections:**
- Script will automatically skip events with < 3 detecting stations
- Use `--random_event` to find events with more detections

### Customization

You can modify parameters in `generate_synthetic_data()`:
```python
# Adjust detection parameters
sigma_logistic = 2.0      # Change detection threshold sharpness
lambda_corr = 0.5         # Increase spatial correlation
scale_factor = 0.95       # Adjust overall detection probability
```

## Example Output Files

After running the scripts, you'll find:
- `plots/real_vs_synthetic_event_X.png` - Individual event comparisons
- `example_plots/` - Results from example script
- `magnitude_analysis/` - Magnitude-dependent analysis results

Each plot provides visual comparison of detection patterns and quantitative statistics for validation of the synthetic data generation process. 