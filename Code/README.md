
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
fifth column is phase type (P waves, 0; S waves, 1), labeled by PhaseNet (these phase labels are not used by GENIE by default).   

sta_names_use: used stations on this day (referenced to the absolute network file).   

sta_ind_use: indices of stations used, corresponding to sta_names_use (referenced to the absolute network file).   

Examples pick files in this format are given in: https://github.com/imcbrearty/GENIE/tree/main/BSSA/Datasets/500%20random%20day%20test   

Each of the scripts (i - iv). should run with minimal changes, though some hyperparemeters can be changed, and a few features are hard-coded. Increased documentation will be added.   
