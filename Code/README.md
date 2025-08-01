## Running model

Follow order of scripts explained in "Applying the model" section. Most of the parameters can be changed in the config files ("config.yaml", "train_config.yaml", and "process_config.yaml"). Additional details for some of those scripts are given here. After training, to run an individual day of data, run process_continuous_days.py; the input format for the data is given below.

## To initilize a GENIE environment

(i). First, in "config.yaml" set your lat_range, lon_range, depth_range area's of interest (for depth, use meters, where negative is below sea level). It is helpful to choose a representative project_name that will be appended to some of the produced files.

(ii). Choose "pre_load_stations: True", and create the station file (stations.txt : columns of station name, latitude, longitude, depth (m, negative below sea level) prior to running any scripts.

(iii). Also set the 1D model in config.yaml and choose "vel_model_type: 1", or create the 3d velocity model file "3d_velocity_model.npz", which has three fields; "X", "Vp", "Vs", where X is np.array size (N_points x 3) of columns of lat, lon, depth (m, negative below sea level), and Vp and Vs a corresponding np.array size (N_points) of Vp and Vs (in m/s). If using 3d, then set ""vel_model_type: 2".

(iv). Depending on the size of your domain, you should adjust "dx" to either a smaller of lower numbers prior to computing travel times. This adjusts the discretization of the initial fast marching method travel time computations - for large areas, e.g., ~100's of km in aperture, these initial values may be too small. For instance, "dx" can be increased to ~1000 - 3000 to decrease memory costs for large domains, or decreased to 100 - 300 m for small domains. Lastly, set "n_jobs" as the number of independent jobs that will be run to compute the travel times (over all possible stations in "stations.npz"; leave it at 1 if only looping all stations in a single script). 

(v). For the GNN, set the number of graph nodes used to represent the spatial domain with "number_of_spatial_nodes". For large problems "number_of_spatial_nodes" can be in the ~1000 - 5000's of node range; high efficiency can also be achieved by decreasing "number_of_spatial_nodes" to very low numbers (e.g., ~100's), which can be sufficient for small problems; it may be necessary to increase the kernel widths slightly to handle this case however (explained below). If the product of "number_of_spatial_nodes" and the max number of stations is too large (e.g., >50,000 - 100,000), it may be necessary to set "use_subgraph" as True. For this option, set the "max_deg_offset" to the max source-reciever distance allowed in the input graph, and "k_nearest_pairs" to force at least this many pairs for each source, if too few stations exist within "max_deg_offset". This option can significantly increase computation time for some applications.

(vi). Now, run "make_initial_files.py", then "assemble_network_data.py" and then "calculate_travel_times_3D_build_data.py 0". Repeat (or parallel call) "calculate_travel_times_3D_build_data.py < i >" for all < i > in 0 ... (n_jobs - 1). This is a niave way of easily allowing parallel jobs to compute the travel times over the stations, and hence speed up this step. After all the travel times are done computing, then run "calculate_travel_times_3D_train_model.py". This fits the neural network travel time surrogate model to the travel times.

## To train the model

(i). Now adjust the parameters in "train_config.yaml". Primarily, the "dist_range" has to be chosen to encompase the full maximum moveout distance (in m) that events are expected to travel over the array. It is normally optimal to set the lower bound at the minimum size event (e.g., 10,000 - 30,000 m), and the upper bound at the max aperture of the array (e.g., ~300,000 - 1,000,000 m); but this is problem dependent. Then set an average or slightly high event rate (to prepare for dense sequences of events) by increasing or decreasing "max_rate_events" and increasing or decreasing the proportion of false picks with "max_false_events" (which is a ratio, e.g., ~1 - 10). Choose the range of the typical size of subsets of the station network that are typically available (e.g., if only 30% - 70% of the total set of stations in stations.npz is ever typically available at one time, set "n_sta_range: [0.3, 0.7]").

(ii). You must also set the label kernel widths to be appropriate for the scale of problem: so edit "kernel_sig_t", "src_t_kernel", "src_t_arv_kernel", "src_x_kernel", "src_x_arv_kernel", "src_depth_kernel" as needed. Roughly, these capture the uncertainity level (but also discretization level of the graphs) for both how much "arrival time - predicted time" misfits are embedded into the GNN (for "kernel_sig_t"), but also the width at which we try to localize the sources in time ("src_t_kernel" and "src_t_arv_kernel") and space ("src_x_kernel" and "src_x_arv_kernel"). If they are too small (or large) for a given domain, that can be problematic. Roughly, the spatial kernel widths should allow for the target Gaussians to occupy a non-negligable part of the domain, but not too much of it at once (e.g., 10 - 25% the domain size); however the main controlling factor is whether the spatial graphs (whose density is dependent on the domain size, and "number_of_spatial_nodes") are dense enough with respect the kernel width.

(iii). It is also important to set "spc_random" and "spc_thresh_rand" (in m) to a spatial scale range that implies that randomness at this spatial level in terms of which stations "do" or "don't" see a given event, due to stochasticity in which stations observe an event at the average ~max distance (given the magnitude), roughly reflects variable spatial patterns in attenuation and noise levels. It hence may be natural to choose this on the order of the average ~station spacing (or a small multiple), so that observed randomness does occur between stations. "spc_thresh_rand" controls the differences that may occur in the thresholds between either phase type (P and S). If events are expected to only produce one dominant phase type, increasing "spc_thresh_rand" to larger numbers shows it more examples of dominantly only one phase type. Training speed can be increased by decreasing "n_batch" to ~5-10.

(iv). Some adjustments to all of: "sig_t", "total_bias", "min_sta_arrival", "min_pick_arrival", "max_num_spikes", "thresh_noise_max", "min_misfit_allowed" may also be advisable. Sig_t controls the random travel time pertubation proportional to travel time that is applied per-pick (so 0.03 is ~3% of travel time; which can be too high for some applications). Similarly "total_bias" controls the per-event random travel time scale factor that can occur (so 0.03 is ~3% of travel time; which can be too high for some applications). "min_sta_arrival", "min_pick_arrival" control the minimum number of picks/stations that are required for a positive-label event. Increasing these (and also increasing them in "process_config.yaml") can lead to a model that only detects more confident and high quality events, while missing more smaller events. 

(v). The "thresh_noise_max", "min_misfit_allowed" control a subtle aspect of the synthetic training data labelling - since random noise is added per-pick to theoretical travel times, sometimes the noise on a pick is so large, we want to overwrite it's association label to "non-associated", so the model doesn't learn to associate very anomolous picks. "thresh_noise_max" is the ratio proportional to the baseline noise level "sig_t" above which, we overwrite the picks label (i.e., "thresh_noise_max" of 2.5 implies that for travel time noise > 2.5 x sig_t x theoretical_travel_time, the label is set to zero). However this re-labelling approach is only applied if the travel time error is also > "min_misfit_allowed", which is more stable at small travel times, since "thresh_noise_max x sig_t x theoretical_travel_time" is always very small at short travel times. Changing "max_num_spikes" shows it more (or less) network-wide simultaenous glitches of picks across the network. It can be helpful to show it more of these if such cases do occur in the real data, however it will also make the model more careful in its detections, and may lead to some increase in missed small events.

(vi). The accuracy of the final model can also be improved if prior to training, all of the picks that will be processed are pre-loaded into the "Picks/" folders, following the expected format (as accessed when running "process_continuous_days.py"). This way, the model also trains on the observed subsets of stations (and hence different station graphs) that are available on a given day (in the pick files). Also, if a reference or initial catalog is saved in the "Calibration/" folders, where each day file contains the "srcs_ref" variable (columns of lat, lon, depth (m, negative beneath sea level), origin time (seconds since start of day), magnitude), and option "use_reference_spatial_density" is set as True, then these reference coordinatres (smoothed by the scale "spatial_sigma") will be used to focus some of the sampling of random sources during training.

(vii). Now the model is ready to be trained. "train_GENIE_model.py" can be run while building the training data between each batch (slower), or if prefered, you first set "build_training_data" as True in "train_config.yaml", and then call "train_GENIE_model.py < i >" for all < i > in 0 ... (N - 1) for N difference jobs computing "n_batches_per_job" different batches each, and saving them to the absolute path specified by "path_to_data". After the data is built, then set "load_training_data" as True and "build_training_data" as False, and then run "train_GENIE_model" to begin training the model. 

(viii). While training, it should create a file "GNN_TrainedModels/<name_of_project>_output_1.txt" that prints the loss during training. It also prints four very useful diagnostic metrics: the columns of the values labeled "trgts" and "preds" should over time, start to match eachother more closely, and the values of "trgts" should roughly be in the ~1 - 15 range on average (with some all zero, or very low numbers occasionally). These "trgts" represent the average max values over the source (first two columns) and association labels (last two columns) over each sample of the batch; the values "preds" are the predicted max values over the same samples. Since the targets are sparse Gaussians in a larger 3D space, at first the model will predict mostly zeros (or low numbers), but over time it will begin to match the target Gaussians, and hence the "preds" columns will begin to match the "trgts" columns more closely. It will also save the model and optimizer and produce example training predictions every 1000 steps.

## To process data

(i). After training, the model is ready to be run. By calling "process_continuous_days < i >" for all < i > for all rows in the "<name_of_project>_process_days_list_ver_1.txt" file, it will process each day of picks listed in this file. The corresponding pick data should be saved in the "Picks/" folders prior to processing, following the format where each pick file has a single variable "P", with columns of: time (seconds since start of day), station index, amplitude, pick probability, and phase type. The "station index" are indexes between (0, N - 1), for N stations in the absolute initial stations.txt (or stations.npz) file, which indicate what station each pick corresponds too in that absolute list of stations (using 0 - (N - 1) python indexing, rather than 1 - N indexing). The "phase_type" must be a binary vector indicting 0's for P waves, and 1's for S waves. The pick probabilities and amplitudes are not directly used by default and can be set to zero; if amplitudes are supplied they can be used to determine magnitudes (explained below).

(ii). The detected events and associations will be saved in the "Catalog/" folder. You can get different results by primarily changing the thresholds ("thresh" and "thresh_assoc"); oftentimes higher quality events will be recovered if the thresholds are slightly higher (e.g., ~0.5 or 0.7), while lossing some smaller events. "min_required_picks" and "min_required_sta" can also be adjusted depending on the target events.

(iii). After an initial catalog has been built for many days and you have saved reference events of a known catalog in "Calibration/" with known magnitudes, you can run "calibrate_and_apply_magnitude_scale.py" to determine how many "matched" events are recovered from the known catalog, compute some summary statistics, and to fit a local magnitude scale using the associated picks and the target magnitudes of the matched events. Then if "compute_magnitudes" is set as True in process_config.yaml, future runs of "process_continuous_days.py" will compute magnitudes of the detected events. A summary of the matched earthquakaes and detected events, and their magnitudes are also saved in a .hdf5 file during this calibration.

(iv). Once an initial catalog has been built, it's also straighforward to run the GraphDD relocation scripts to relocate the events. These files are given and described in the "Relocation" directoy.



## Additional Information:


### Using train_GENIE_model.py

To train the model, the training parameters in "train_config.yaml" should be adapted somewhat for different settings. Several parameters that become the inputs to the "generate_synthetic_data" function (defined in lists "training_params", "training_params_2", and "training_params_3") should be edited based on the source domain scale and station distribution. These control things like the average background rate of sources, missed and false pick rates, travel time uncertainity levels, source and spatial label kernel widths, max moveout distances of sources, etc. 

The training speed can be improved by first building the training data and saving it to distinct files (using build_training_data = True), and loading during training (the memory cost of the files can be ~Tb in size). By default it builds a new batch of training data between each update step.

Use the parameter "fixed_subnetworks" in train_config.yaml to train the GNN on specific instances of subnetworks available per day, as recorded in the pick data (else, only random subsets of stations are chosen).

Note: there are a few fixed scale-dependent parameters in config.yaml, such as "scale_rel", "scale_t" and "eps"; these are used to normalize typical offset distances between nodes, and arival time uncertainities. For small applications (e.g., < 50 km or so), these parameters should typically be decreased from their default values.

### Running process_continuous_days.py

Specify which days of picks you want to run by creating the "Project name"_process_days_list_ver_1.txt file in the main directory.    

Each row is a date specified by, e.g., (2000/1/1) for year, month, day. When calling process_continuous_days.py, the first system argument specifies which day, in the "Project name"_process_days_list_ver_1.txt is run. E.g., "python process_continuous_days.py 0" runs the first day in "Project name"_process_days_list_ver_1.txt", and "python process_continuous_days.py 1" runs the second day in "Project name"_process_days_list_ver_1.txt, etc.

The output of the processing script is saved in the "Catalog" folder and as one of the saved fields includes the associated picks in the "Picks" output (e.g., z = h5py.File('Catalog/2018/Project_results_continuous_days_2018_5_1_ver_1.hdf5', 'r'); z['Picks/%d_Picks_P'%0][:] are the associated P picks for the first event, z['Picks/%d_Picks_P'%1][:] for the second, and z['Picks/%d_Picks_S'%i][:] accesses the associated S waves for all i'th events, etc). The file also saves two versions of the source locations: "srcs" and "srcs_trv". The first is the direct prediction of location from the model, and the second is the same events located using the associated picks and standard travel-time based location optimization. Usually, "srcs_trv" is slightly more accurate, but if a few mis-associations occur, then these locations are also usually more pertubed than "srcs". In constrast, "srcs" will have less highly mislocated events, but retains some bias (and fake clustering) due to the source graph node positions. Using an additional re-location technique such as NonLinLoc or HypoDD on the output of this model can lead to more refined locations.

To load picks, put the pick file in the directory:

path_file + Picks/%d_%d_%d_ver_%d.npz'%(date[0], date[1], date[2], n_ver))   
date = [year, month, day] as integers

The pick files must have the field: P   

P: picks from PhaseNet, given to GENIE as input.    
first column is time since start of day (or can be in terms of a sampling rate, e.g., 100 Hz sampling, specified by the spr_picks parameter in process_config.yaml). Previous default behavior assumed 100 Hz sampling.    
second column is station index (corresponding to indices of stations in the NC_EHZ_network.npz file).   
third column is maximum peak ground velocity from 1 s before to 2.5 s after each pick time.   
fourth column is probability of PhaseNet pick.   
fifth column is phase type (P waves, 0; S waves, 1), labeled by PhaseNet    


Examples pick files in this format are given in: https://github.com/imcbrearty/GENIE/tree/main/Examples/Ferndale.zip.

Note that by default, maximum peak ground velocity and probability of PhaseNet pick are not currently used by the model.

Each of the scripts (i - vi). should run with minimal changes, though some hyperparemeters can be changed, and a few features are hard-coded. Increased documentation will be added.
