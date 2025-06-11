# GENIE : Graph Earthquake Neural Interpretation Engine

A Graph Neural Network (GNN) based earthquake phase associator and spatio-temporal source localization model. 

The paper associated with this work is given at https://pubs.geoscienceworld.org/ssa/bssa/article/doi/10.1785/0120220182/619845/Earthquake-Phase-Association-with-Graph-Neural.

The source code is posted at https://github.com/imcbrearty/GENIE/tree/main/Code.

## Applying the model

For now, the basic workflow is to (i). install dependencies in "install_dependencies.txt", (ii). run the "make_initial_files.py" script to initialize station, region, and velocity model files, (iii). run "assemble_network_data.py" for an input set of stations and spatial region, which sets up the directory and initilizes required variables, (iv). run "calculate_travel_times_3D_build_data.py" and "calculate_travel_times_3D_train_model.py" to compute the travel time grid of P and S phases over the region of interest, for a chosen velocity model, (v). run "train_GENIE_model.py" to train the GNN for the given application, and (vi). run "process_continuous_days.py" to compute predictions and build an earthquake catalog for a given set of input picks, and the current trained GNN and velocity model.

The config files can be used to adjust most parameters. Also, station, region, and 1D or 3D velocity model files can be created following the same format as shown in make_initial_files.py without actually having to run make_initial_files.py.

Pre-trained GNN's, and pre-computed travel time fields will be supplied in the future, to faciliate easier use and allow users to only have to run steps (i-iii), and (vi). above. Running additional re-location techniques like NonLinLoc or HypoDD with the associated picks from this model can often improve event location accuracies.

## Setup Details

For a new project, first download the GitHub repo and navigate to the "Code" folder. Then (before running any .py codes),

First, (i). create the "stations.txt" (or "stations.npz") file. The station file must contain columns of (station name, latitude, longitude, elevation (m)); seperated by commas or spaces.

Then, set the (ii). region, (iii). velocity model, (iv). travel time calculation discretization parameters and (v). source graph parameters in "config.yaml".

Then, (vi). load the pick data in the GENIE format directly (saved as .npz files), or as .csv files in "Picks/2022/2022_12_20_ver_1.csv" format (where for any day 'Picks/%d/%d_%d_%d_ver_1.csv'%(year, year, month, day)). For these .csv files, the columns are simply: (station name, pick time (seconds), amplitude, pick probability, phase type); where station names must correspond with station names in the "stations.txt" file, and where "phase type" entries are either 0 and 1's, or "P" and "S", for P and S picks, respectively.

Then (vii). set the synthetic training data parameters in "train_config.yaml"

Now the (i) - (vi) set of python scripts described in "Applying the model" can be run implement the model.

## Extra information

The pre-print is given at https://arxiv.org/abs/2209.07086. The datasets used in this study are available in https://github.com/imcbrearty/GENIE/tree/main/BSSA.

For a description of a related GNN architecture applied to source localization from discrete pick datasets, see https://ieeexplore.ieee.org/document/9897468, https://arxiv.org/abs/2203.05144, https://arxiv.org/abs/2410.19323

Preliminary results of the method are shown here, https://www.scec.org/meetings/2021/am/poster/227, https://www.scec.org/meetings/2023/am/poster/013

## Tutorials

Look in Examples/Ferndale.zip for example trained models and applications to real data.
