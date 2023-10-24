
Follow order of scripts explained in "Applying the model" section. Most of the parameters can be changed in the config files (config.yaml, train_config.yaml, and process_config.yaml). Additional details for some of those scripts are given here. After training, to run an individual day of data, run process_continuous_days.py; the input format for the data is given below.

## Using train_GENIE_model.py

To train the model, the training parameters in train_config.yaml should be adapted somewhat for different settings. Several parameters that become the inputs to the generate_synthetic_data function (defined in lists training_params, training_params_2, and training_params_3) should be edited based on the source domain scale and station distribution. These control things like the average background rate of sources, missed and false pick rates, travel time uncertainity levels, source and spatial label kernel widths, etc.

## Running process_continuous_days.py

To load picks, put the pick file in the directory:

path_file + Picks/%d_%d_%d_ver_%d.npz'%(date[0], date[1], date[2], n_ver))   
date = [year, month, day] as integers   

The file must have the three fields: P, sta_names_use, sta_ind_use   

P: picks from PhaseNet, given to GENIE as input.    
first column is time index from start of day (assuming 100 Hz sampling).   
second column is station index (corresponding to indices of stations in the NC_EHZ_network.npz file).   
third column is maximum peak ground velocity from 1 s before to 2.5 s after each pick time.   
fourth column is probability of PhaseNet pick.   
fifth column is phase type (P waves, 0; S waves, 1), labeled by PhaseNet

sta_names_use: used stations on this day (referenced to the absolute network file).   

sta_ind_use: indices of stations used, corresponding to sta_names_use (referenced to the absolute network file).   

Examples pick files in this format are given in: https://github.com/imcbrearty/GENIE/tree/main/BSSA/Datasets/500%20random%20day%20test   

Each of the scripts (i - vi). should run with minimal changes, though some hyperparemeters can be changed, and a few features are hard-coded. Increased documentation will be added.   
