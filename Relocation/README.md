# Graph Double Difference (GraphDD)

A graph neural network based earthquake relocation framework

## Applying the model

First run "build_initial_files.py" to convert the catalog earthquake data to one reference file.

Then run "build_subsets_of_paired_sources.py" to build the sets of training input graphs.

Then "train_double_difference_model.py" to train the double difference model.

These scripts assume you have setup a GENIE environment for a study region of interest, and have trained the travel time neural network for a chosen velocity model. The catalog files also have to be formated in the GENIE format in the corresponding "Catalog/%d"%year folders.

Different training loss terms can be selected in "train_double_difference_model.py", and different graph construction parameters in "build_subsets_of_paired_sources.py".

For substantial catalogs (e.g., > 10,000's of events), you should build a large number of independent input graphs (such as >30,000); then load a batch of these (~3-10) during training, and then update the model for ~50,000 steps.

## To initilize a GENIE environment

(i). First, in "config.yaml" set your lat_range, lon_range, depth_range area's of interest (for depth, use meters, where negative is below sea level). It is helpful to choose a representative project_name that will be appended to some of the produced files.

(ii). Choose "pre_load_stations: True", and create the station file (stations.txt : columns of latitude, longitude, depth (m, negative below sea level) prior to running any scripts.

(iii). Also set the 1D model in config.yaml and choose "vel_model_type: 1", or create the 3d velocity model file "3d_velocity_model.npz", which has three fields; "X", "Vp", "Vs", where X is np.array size (N_points x 3) of columns of lat, lon, depth (m, negative below sea level), and Vp and Vs a corresponding np.array size (N_points) of Vp and Vs (in m/s). If using 3d, then set ""vel_model_type: 2".

(iv). Depending on the size of your domain, you should adjust "dx" to either a smaller of lower numbers prior to computing travel times. This adjusts the discretization of the initial fast marching method travel time computations - for large areas, e.g., ~100's of km in aperture, these initial values may be too small. For instance, "dx" can be increased to ~1000 - 3000 to decrease memory costs for large domains, or decreased to 100 - 300 m for small domains. Lastly, set "n_jobs" as the number of independent jobs that will be run to compute the travel times (over all possible stations in "stations.npz"; leave it at 1 if only looping all stations in a single script).

(v). Now, run "make_initial_files.py", then "assemble_network_data.py" and then "calculate_travel_times_3D_build_data.py 0". Repeat (or parallel call) "calculate_travel_times_3D_build_data.py < i >" for all < i > in 0 ... (n_jobs - 1). This is a niave way of easily allowing parallel jobs to compute the travel times over the stations, and hence speed up this step. After all the travel times are done computing, then run "calculate_travel_times_3D_train_model.py". This fits the neural network travel time surrogate model to the travel times.

(vi). Lastly convert the catalog information into the GENIE ".hdf5" saved format. Examples of how to create this file are shown in the Tutorial, and also at the end of the script "process_continuous_days.py". Alternatively, training and running GENIE to build the initial catalog will also create these files.

(vii). GraphDD can now be run to begin locating events. You can run "build_initial_files.py" to convert the distinct (per day) catalog files into one large dataset catalog, then "build_subsets_of_paired_sources.py" (or in parallel; "build_subsets_of_paired_sources.py < i >" for multiple different < i >) to build the subgraph training dataset, and then "train_double_difference_model.py" to train the model.

## Extra Information

The method is described in https://arxiv.org/abs/2410.19323v1 and https://earth-planets-space.springeropen.com/articles/10.1186/s40623-025-02251-4.
