## Running model

Follow order of scripts explained in "Applying the model" section. Most of the parameters can be changed in the config files ("config.yaml", "train_config.yaml", and "process_config.yaml"). Additional details for some of those scripts are given here. After training, to run an individual day of data, run process_continuous_days.py; the input format for the data is given below.

## Using train_GENIE_model.py

To train the model, the training parameters in "train_config.yaml" should be adapted somewhat for different settings. Several parameters that become the inputs to the "generate_synthetic_data" function (defined in lists "training_params", "training_params_2", and "training_params_3") should be edited based on the source domain scale and station distribution. These control things like the average background rate of sources, missed and false pick rates, travel time uncertainity levels, source and spatial label kernel widths, etc.

## Running process_continuous_days.py

Specify which days of picks you want to run by creating the "Project name"_process_days_list_ver_1.txt file in the main directory.    

Each row is a date specified by, e.g., (2000/1/1) for year, month, day. When calling process_continuous_days.py, the first system argument specifies which day, in the "Project name"_process_days_list_ver_1.txt is run. E.g., "python process_continuous_days.py 0" runs the first day in "Project name"_process_days_list_ver_1.txt", and "python process_continuous_days.py 1" runs the second day in "Project name"_process_days_list_ver_1.txt, etc.

The output of the processing script includes the associated picks in the "Picks" output, and two versions of the source locations: "srcs" and "srcs_trv". The first is the direct prediction of location from the model, and the second is the same events located using the associated picks and standard travel-time based location optimization. Usually, "srcs_trv" is slightly more accurate, but if a few mis-associations occur, then these locations are also usually more pertubed than "srcs". In constrast, "srcs" will have less highly mislocated events, but retains some bias (and fake clustering) due to the source graph node positions. Using an additional re-location technique such as NonLinLoc or HypoDD on the output of this model can lead to more refined locations.

To load picks, put the pick file in the directory:

path_file + Picks/%d_%d_%d_ver_%d.npz'%(date[0], date[1], date[2], n_ver))   
date = [year, month, day] as integers

The pick files must have the three fields: P, sta_names_use, sta_ind_use   

P: picks from PhaseNet, given to GENIE as input.    
first column is time since start of day (or can be in terms of a sampling rate, e.g., 100 Hz sampling, specified by the spr_picks parameter in process_config.yaml). Previous default behavior assumed 100 Hz sampling.    
second column is station index (corresponding to indices of stations in the NC_EHZ_network.npz file).   
third column is maximum peak ground velocity from 1 s before to 2.5 s after each pick time.   
fourth column is probability of PhaseNet pick.   
fifth column is phase type (P waves, 0; S waves, 1), labeled by PhaseNet    

sta_names_use: used stations on this day (referenced to the absolute network file).   

sta_ind_use: indices of stations used, corresponding to sta_names_use (referenced to the absolute network file).   

Examples pick files in this format are given in: https://github.com/imcbrearty/GENIE/tree/main/BSSA/Datasets/500%20random%20day%20test. Note that, in these example picks, the pick times were specified in 100 Hz sampling, however the default behavior is now for picks to be specified in absolute time (the parameter spr_picks in process_config.yaml specifies whether absolute time or a given sampling rate is used).

Note that by default, maximum peak ground velocity and probability of PhaseNet pick are not currently used by the model.

Each of the scripts (i - vi). should run with minimal changes, though some hyperparemeters can be changed, and a few features are hard-coded. Increased documentation will be added.   
