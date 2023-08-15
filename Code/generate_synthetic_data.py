
## Training params list 2
# spc_random = 30e3
# sig_t = 0.03 # 3 percent of travel time error on pick times
# spc_thresh_rand = 20e3
# min_sta_arrival = 4
# coda_rate = 0.035 # 5 percent arrival have code. Probably more than this? Increased from 0.035.
# coda_win = np.array([0, 25.0]) # coda occurs within 0 to 25 s after arrival (should be less?) # Increased to 25, from 20.0
# max_num_spikes = 80
# spike_time_spread = 0.15
# s_extra = 0.0 ## If this is non-zero, it can increase (or decrease) the total rate of missed s waves compared to p waves
# use_stable_association_labels = True
# thresh_noise_max = 1.5
# training_params_2 = [spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max]

## Training params list 3
# n_batch = 75
# dist_range = [15e3, 500e3]
# max_rate_events = 6000/8
# max_miss_events = 2500/8
# max_false_events = 2500/8
# T = 3600.0*3.0
# dt = 30
# tscale = 3600.0
# n_sta_range = [0.35, 1.0]
# use_sources = False
# use_full_network = False
# fixed_subnetworks = None
# use_preferential_sampling = False
# use_shallow_sources = False
# training_params_3 = [n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources]

def generate_synthetic_data(trv, locs, x_grids, x_grids_trv, x_grids_trv_refs, x_grids_trv_pointers_p, x_grids_trv_pointers_s, lat_range, lon_range, lat_range_extend, lon_range_extend, depth_range, training_params_1, training_params_2, training_params_3, graph_params, pred_params, ftrns1, ftrns2, plot_on = False, verbose = False):

	if verbose == True:
		st = time.time()

	k_sta_edges, k_spc_edges, k_time_edges = graph_params
	t_win, kernel_sig_t, src_t_kernel, src_x_kernel, src_depth_kernel = pred_params

	n_spc_query, n_src_query = training_params_1
	spc_random, sig_t, spc_thresh_rand, min_sta_arrival, coda_rate, coda_win, max_num_spikes, spike_time_spread, s_extra, use_stable_association_labels, thresh_noise_max = training_params_2
	n_batch, dist_range, max_rate_events, max_miss_events, max_false_events, T, dt, tscale, n_sta_range, use_sources, use_full_network, fixed_subnetworks, use_preferential_sampling, use_shallow_sources = training_params_3

	## Generate synthetic events
	arrivals, src_times, src_positions, src_magnitudes = generate_synthetic_events(locs)

	## Check active sources and compute sampling points
	arrivals_select, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources, active_sources_per_slice_l, inside_interior, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices = check_active_sources_and_compute_sampling_points(locs, arrivals, src_times, src_positions, src_magnitudes)

	## Compute inputs and labels
	[Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l] = compute_inputs_and_labels(locs, arrivals, src_times, src_positions, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources_per_slice_l, inside_interior, t_win, src_t_kernel, src_x_kernel, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices)

	## Initilize source and data variables
	srcs = np.concatenate((src_positions, src_times.reshape(-1,1), src_magnitudes.reshape(-1,1)), axis = 1)
	data = [arrivals, srcs, active_sources]

	if verbose == True:
		print('batch gen time took %0.2f'%(time.time() - st))

	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l], data ## Can return data, or, merge this with the update-loss compute, itself (to save read-write time into arrays..)

def generate_synthetic_events(locs):

	assert(np.floor(n_sta_range[0]*locs.shape[0]) > k_sta_edges)

	scale_x = np.array([lat_range_extend[1] - lat_range_extend[0], lon_range_extend[1] - lon_range_extend[0], depth_range[1] - depth_range[0]]).reshape(1,-1)
	offset_x = np.array([lat_range_extend[0], lon_range_extend[0], depth_range[0]]).reshape(1,-1)
	n_sta = locs.shape[0]
	locs_tensor = torch.Tensor(locs).to(device)

	# t_slice = np.arange(-t_win/2.0, t_win/2.0 + 1.0, 1.0)

	tsteps = np.arange(0, T + dt, dt)
	tvec = np.arange(-tscale*4, tscale*4 + dt, dt)
	tvec_kernel = np.exp(-(tvec**2)/(2.0*(tscale**2)))

	p_rate_events = fftconvolve(np.random.randn(2*locs.shape[0] + 3, len(tsteps)), tvec_kernel.reshape(1,-1).repeat(2*locs.shape[0] + 3,0), 'same', axes = 1)
	c_cor = (p_rate_events@p_rate_events.T) ## Not slow!
	global_event_rate, global_miss_rate, global_false_rate = p_rate_events[0:3,:]

	# Process global event rate, to physical units.
	global_event_rate = (global_event_rate - global_event_rate.min())/(global_event_rate.max() - global_event_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_rate_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_rate_events - min_add) + 0.5*max_rate_events
	global_event_rate = global_event_rate*scale + min_add

	global_miss_rate = (global_miss_rate - global_miss_rate.min())/(global_miss_rate.max() - global_miss_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
	global_miss_rate = global_miss_rate*scale + min_add

	global_false_rate = (global_false_rate - global_false_rate.min())/(global_false_rate.max() - global_false_rate.min()) # [0,1] scale
	min_add = np.random.rand()*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand()*(0.5*max_false_events - min_add) + 0.5*max_false_events
	global_false_rate = global_false_rate*scale + min_add

	station_miss_rate = p_rate_events[3 + np.arange(n_sta),:]
	station_miss_rate = (station_miss_rate - station_miss_rate.min(1, keepdims = True))/(station_miss_rate.max(1, keepdims = True) - station_miss_rate.min(1, keepdims = True)) # [0,1] scale
	min_add = np.random.rand(n_sta,1)*0.25*max_miss_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand(n_sta,1)*(0.5*max_miss_events - min_add) + 0.5*max_miss_events
	station_miss_rate = station_miss_rate*scale + min_add

	station_false_rate = p_rate_events[3 + n_sta + np.arange(n_sta),:]
	station_false_rate = (station_false_rate - station_false_rate.min(1, keepdims = True))/(station_false_rate.max(1, keepdims = True) - station_false_rate.min(1, keepdims = True))
	min_add = np.random.rand(n_sta,1)*0.25*max_false_events ## minimum between 0 and 0.25 of max rate
	scale = np.random.rand(n_sta,1)*(0.5*max_false_events - min_add) + 0.5*max_false_events
	station_false_rate = station_false_rate*scale + min_add

	## Sample events.
	vals = np.random.poisson(dt*global_event_rate/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average
	src_times = np.sort(np.hstack([np.random.rand(vals[j])*dt + tsteps[j] for j in range(len(vals))]))
	n_src = len(src_times)
	src_positions = np.random.rand(n_src, 3)*scale_x + offset_x
	src_magnitudes = np.random.rand(n_src)*7.0 - 1.0 # magnitudes, between -1.0 and 7 (uniformly)

	if use_shallow_sources == True:
		sample_random_depths = gamma(1.75, 0.0).rvs(n_src)
		sample_random_grab = np.where(sample_random_depths > 5)[0] # Clip the long tails, and place in uniform, [0,5].
		sample_random_depths[sample_random_grab] = 5.0*np.random.rand(len(sample_random_grab))
		sample_random_depths = sample_random_depths/sample_random_depths.max() # Scale to range
		sample_random_depths = -sample_random_depths*(scale_x[0,2] - 2e3) + (offset_x[0,2] + scale_x[0,2] - 2e3) # Project along axis, going negative direction. Removing 2e3 on edges.
		src_positions[:,2] = sample_random_depths

	m1 = [0.5761163, -0.21916288]
	m2 = 1.15

	amp_thresh = 1.0
	sr_distances = pd(ftrns1(src_positions[:,0:3]), ftrns1(locs))

	use_uniform_distance_threshold = False
	## This previously sampled a skewed distribution by default, not it samples a uniform
	## distribution of the maximum source-reciever distances allowed for each event.
	if use_uniform_distance_threshold == True:
		dist_thresh = np.random.rand(n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
	else:
		## Use beta distribution to generate more samples with smaller moveouts
		# dist_thresh = -1.0*np.log(np.sqrt(np.random.rand(n_src))) ## Sort of strange dist threshold set!
		# dist_thresh = (dist_thresh*dist_range[1]/10.0 + dist_range[0]).reshape(-1,1)
		dist_thresh = beta(2,5).rvs(size = n_src).reshape(-1,1)*(dist_range[1] - dist_range[0]) + dist_range[0]
		
	# create different distance dependent thresholds.
	dist_thresh_p = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None] # Increased sig from 20e3 to 25e3 # Decreased to 10 km
	dist_thresh_s = dist_thresh + spc_thresh_rand*np.random.laplace(size = dist_thresh.shape[0])[:,None]

	ikeep_p1, ikeep_p2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_p))
	ikeep_s1, ikeep_s2 = np.where(((sr_distances + spc_random*np.random.randn(n_src, n_sta)) < dist_thresh_s))

	arrivals_theoretical = trv(torch.Tensor(locs).to(device), torch.Tensor(src_positions[:,0:3]).to(device)).cpu().detach().numpy()
	arrival_origin_times = src_times.reshape(-1,1).repeat(n_sta, 1)
	arrivals_indices = np.arange(n_sta).reshape(1,-1).repeat(n_src, 0)
	src_indices = np.arange(n_src).reshape(-1,1).repeat(n_sta, 1)

	arrivals_p = np.concatenate((arrivals_theoretical[ikeep_p1, ikeep_p2, 0].reshape(-1,1), arrivals_indices[ikeep_p1, ikeep_p2].reshape(-1,1), src_indices[ikeep_p1, ikeep_p2].reshape(-1,1), arrival_origin_times[ikeep_p1, ikeep_p2].reshape(-1,1), np.zeros(len(ikeep_p1)).reshape(-1,1)), axis = 1)
	arrivals_s = np.concatenate((arrivals_theoretical[ikeep_s1, ikeep_s2, 1].reshape(-1,1), arrivals_indices[ikeep_s1, ikeep_s2].reshape(-1,1), src_indices[ikeep_s1, ikeep_s2].reshape(-1,1), arrival_origin_times[ikeep_s1, ikeep_s2].reshape(-1,1), np.ones(len(ikeep_s1)).reshape(-1,1)), axis = 1)
	arrivals = np.concatenate((arrivals_p, arrivals_s), axis = 0)

	t_inc = np.floor(arrivals[:,3]/dt).astype('int')
	p_miss_rate = 0.5*station_miss_rate[arrivals[:,1].astype('int'), t_inc] + 0.5*global_miss_rate[t_inc]
	idel = np.where((np.random.rand(arrivals.shape[0]) + s_extra*arrivals[:,4]) < dt*p_miss_rate/T)[0]

	arrivals = np.delete(arrivals, idel, axis = 0)
	n_events = len(src_times)

	icoda = np.where(np.random.rand(arrivals.shape[0]) < coda_rate)[0]
	if len(icoda) > 0:
		false_coda_arrivals = np.random.rand(len(icoda))*(coda_win[1] - coda_win[0]) + coda_win[0] + arrivals[icoda,0] + arrivals[icoda,3]
		false_coda_arrivals = np.concatenate((false_coda_arrivals.reshape(-1,1), arrivals[icoda,1].reshape(-1,1), -1.0*np.ones((len(icoda),1)), np.zeros((len(icoda),1)), -1.0*np.ones((len(icoda),1))), axis = 1)
		arrivals = np.concatenate((arrivals, false_coda_arrivals), axis = 0)

	## Base false events
	station_false_rate_eval = 0.5*station_false_rate + 0.5*global_false_rate
	vals = np.random.poisson(dt*station_false_rate_eval/T) # This scaling, assigns to each bin the number of events to achieve correct, on averge, average

	# How to speed up this part?
	i1, i2 = np.where(vals > 0)
	v_val, t_val = vals[i1,i2], tsteps[i2]
	false_times = np.repeat(t_val, v_val) + np.random.rand(vals.sum())*dt
	false_indices = np.hstack([k*np.ones(vals[k,:].sum()) for k in range(n_sta)])
	n_false = len(false_times)
	false_arrivals = np.concatenate((false_times.reshape(-1,1), false_indices.reshape(-1,1), -1.0*np.ones((n_false,1)), np.zeros((n_false,1)), -1.0*np.ones((n_false,1))), axis = 1)
	arrivals = np.concatenate((arrivals, false_arrivals), axis = 0)

	n_spikes = np.random.randint(0, high = int(max_num_spikes*T/(3600*24))) ## Decreased from 150. Note: these may be unneccessary now. ## Up to 200 spikes per day, decreased from 200
	if n_spikes > 0:
		n_spikes_extent = np.random.randint(1, high = n_sta, size = n_spikes) ## This many stations per spike
		time_spikes = np.random.rand(n_spikes)*T
		sta_ind_spikes = np.hstack([np.random.choice(n_sta, size = n_spikes_extent[j], replace = False) for j in range(n_spikes)])
		sta_time_spikes = np.hstack([time_spikes[j] + np.random.randn(n_spikes_extent[j])*spike_time_spread for j in range(n_spikes)])
		false_arrivals_spikes = np.concatenate((sta_time_spikes.reshape(-1,1), sta_ind_spikes.reshape(-1,1), -1.0*np.ones((len(sta_ind_spikes),1)), np.zeros((len(sta_ind_spikes),1)), -1.0*np.ones((len(sta_ind_spikes),1))), axis = 1)
		arrivals = np.concatenate((arrivals, false_arrivals_spikes), axis = 0) ## Concatenate on spikes


	## Check which true picks have so much noise, they should be marked as `false picks' for the association labels
	if use_stable_association_labels == True:
		iz = np.where(arrivals[:,4] >= 0)[0]
		noise_values = np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]
		iexcess_noise = np.where(np.abs(noise_values) > thresh_noise_max*sig_t*arrivals[iz,0])[0]
		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + noise_values ## Setting arrival times equal to moveout time plus origin time plus noise
		if len(iexcess_noise) > 0: ## Set these arrivals to "false arrivals", since noise is so high
			arrivals[iz[iexcess_noise],2] = -1
			arrivals[iz[iexcess_noise],3] = 0
			arrivals[iz[iexcess_noise],4] = -1
	else: ## This was the original version
		iz = np.where(arrivals[:,4] >= 0)[0]
		arrivals[iz,0] = arrivals[iz,0] + arrivals[iz,3] + np.random.laplace(scale = 1, size = len(iz))*sig_t*arrivals[iz,0]

	return arrivals, src_times, src_positions, src_magnitudes

def check_active_sources_and_compute_sampling_points(locs, arrivals, src_times, src_positions, src_magnitudes):

	n_events = len(src_times)
	n_sta = locs.shape[0]

	## Check which sources are active
	source_tree_indices = cKDTree(arrivals[:,2].reshape(-1,1))
	lp = source_tree_indices.query_ball_point(np.arange(n_events).reshape(-1,1), r = 0)
	lp_backup = [lp[j] for j in range(len(lp))]
	n_unique_station_counts = np.array([len(np.unique(arrivals[lp[j],1])) for j in range(n_events)])
	active_sources = np.where(n_unique_station_counts >= min_sta_arrival)[0] # subset of sources
	non_active_sources = np.delete(np.arange(n_events), active_sources, axis = 0)
	src_positions_active = src_positions[active_sources]
	src_times_active = src_times[active_sources]
	src_magnitudes_active = src_magnitudes[active_sources] ## Not currently used

	inside_interior = ((src_positions[:,0] < lat_range[1])*(src_positions[:,0] > lat_range[0])*(src_positions[:,1] < lon_range[1])*(src_positions[:,1] > lon_range[0]))
	
	iwhere_real = np.where(arrivals[:,-1] > -1)[0]
	iwhere_false = np.delete(np.arange(arrivals.shape[0]), iwhere_real)
	phase_observed = np.copy(arrivals[:,-1]).astype('int')

	if len(iwhere_false) > 0: # For false picks, assign a random phase type
		phase_observed[iwhere_false] = np.random.randint(0, high = 2, size = len(iwhere_false))

	perturb_phases = True # For true picks, randomly flip a fraction of phases
	if (len(phase_observed) > 0)*(perturb_phases == True):
		n_switch = int(np.random.rand()*(0.2*len(iwhere_real))) # switch up to 20% phases
		iflip = np.random.choice(iwhere_real, size = n_switch, replace = False)
		phase_observed[iflip] = np.mod(phase_observed[iflip] + 1, 2)

	scale_vec = np.array([1,2*t_win]).reshape(1,-1)

	if use_sources == False:
		time_samples = np.sort(np.random.rand(n_batch)*T) ## Uniform

	elif use_sources == True:
		time_samples = src_times_active[np.sort(np.random.choice(len(src_times_active), size = n_batch))]

	l_src_times_active = len(src_times_active)
	if (use_preferential_sampling == True)*(len(src_times_active) > 1):
		for j in range(n_batch):
			if np.random.rand() > 0.5: # 30% of samples, re-focus time. # 0.7
				time_samples[j] = src_times_active[np.random.randint(0, high = l_src_times_active)] + (2.0/3.0)*src_t_kernel*np.random.laplace()

	time_samples = np.sort(time_samples)

	max_t = float(np.ceil(max([x_grids_trv[j].max() for j in range(len(x_grids_trv))])))

	tree_src_times_all = cKDTree(src_times[:,np.newaxis])
	tree_src_times = cKDTree(src_times_active[:,np.newaxis])
	lp_src_times_all = tree_src_times_all.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)
	lp_src_times = tree_src_times.query_ball_point(time_samples[:,np.newaxis], r = 3.0*src_t_kernel)

	st = time.time()
	tree = cKDTree(arrivals[:,0][:,None])
	lp = tree.query_ball_point(time_samples.reshape(-1,1) + max_t/2.0, r = t_win + max_t/2.0) 

	lp_concat = np.hstack([np.array(list(lp[j])) for j in range(n_batch)]).astype('int')
	if len(lp_concat) == 0:
		lp_concat = np.array([0]) # So it doesnt fail?

	arrivals_select = arrivals[lp_concat]
	phase_observed_select = phase_observed[lp_concat]

	Trv_subset_p = []
	Trv_subset_s = []
	Station_indices = []
	Grid_indices = []
	Batch_indices = []
	Sample_indices = []
	sc = 0

	if (fixed_subnetworks is not None):
		fixed_subnetworks_flag = 1
	else:
		fixed_subnetworks_flag = 0		

	active_sources_per_slice_l = []
	src_positions_active_per_slice_l = []
	src_times_active_per_slice_l = []

	for i in range(n_batch):
		i0 = np.random.randint(0, high = len(x_grids))
		n_spc = x_grids[i0].shape[0]
		if use_full_network == True:
			n_sta_select = n_sta
			ind_sta_select = np.arange(n_sta)

		else:
			if (fixed_subnetworks_flag == 1)*(np.random.rand() < 0.5): # 50 % networks are one of fixed networks.
				isub_network = np.random.randint(0, high = len(fixed_subnetworks))
				n_sta_select = len(fixed_subnetworks[isub_network])
				ind_sta_select = np.copy(fixed_subnetworks[isub_network]) ## Choose one of specific networks.
			
			else:
				n_sta_select = int(n_sta*(np.random.rand()*(n_sta_range[1] - n_sta_range[0]) + n_sta_range[0]))
				ind_sta_select = np.sort(np.random.choice(n_sta, size = n_sta_select, replace = False))

		Trv_subset_p.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,0].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Trv_subset_s.append(np.concatenate((x_grids_trv[i0][:,ind_sta_select,1].reshape(-1,1), np.tile(ind_sta_select, n_spc).reshape(-1,1), np.repeat(np.arange(n_spc).reshape(-1,1), len(ind_sta_select), axis = 1).reshape(-1,1), i*np.ones((n_spc*len(ind_sta_select),1))), axis = 1)) # not duplication
		Station_indices.append(ind_sta_select) # record subsets used
		Batch_indices.append(i*np.ones(len(ind_sta_select)*n_spc))
		Grid_indices.append(i0)
		Sample_indices.append(np.arange(len(ind_sta_select)*n_spc) + sc)
		sc += len(Sample_indices[-1])

		active_sources_per_slice = np.where(np.array([len( np.array(list(set(ind_sta_select).intersection(np.unique(arrivals[lp_backup[j],1])))) ) >= min_sta_arrival for j in lp_src_times_all[i]]))[0]

		active_sources_per_slice_l.append(active_sources_per_slice)

	Trv_subset_p = np.vstack(Trv_subset_p)
	Trv_subset_s = np.vstack(Trv_subset_s)
	Batch_indices = np.hstack(Batch_indices)

	return arrivals_select, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources, active_sources_per_slice_l, inside_interior, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices

def compute_inputs_and_labels(locs, arrivals, src_times, src_positions, phase_observed, phase_observed_select, time_samples, lp, lp_src_times_all, active_sources_per_slice_l, inside_interior, t_win, src_t_kernel, src_x_kernel, Trv_subset_p, Trv_subset_s, Station_indices, Batch_indices, Grid_indices, Sample_indices):

	n_sta = locs.shape[0]

	offset_per_batch = 1.5*max_t
	offset_per_station = 1.5*n_batch*offset_per_batch

	t_slice = np.arange(-t_win/2.0, t_win/2.0 + 1.0, 1.0)
	src_spatial_kernel = np.array([src_x_kernel, src_x_kernel, src_depth_kernel]).reshape(1,1,-1) # Combine, so can scale depth and x-y offset differently.

	arrivals_offset = np.hstack([-time_samples[i] + i*offset_per_batch + offset_per_station*arrivals[lp[i],1] for i in range(n_batch)]) ## Actually, make disjoint, both in station axis, and in batch number.
	one_vec = np.concatenate((np.ones(1), np.zeros(4)), axis = 0).reshape(1,-1)
	arrivals_select = np.vstack([arrivals[lp[i]] for i in range(n_batch)]) + arrivals_offset.reshape(-1,1)*one_vec ## Does this ever fail? E.g., when there's a missing station's
	n_arvs = arrivals_select.shape[0]

	# Rather slow!
	iargsort = np.argsort(arrivals_select[:,0])
	arrivals_select = arrivals_select[iargsort]
	phase_observed_select = phase_observed_select[iargsort]

	iwhere_p = np.where(phase_observed_select == 0)[0]
	iwhere_s = np.where(phase_observed_select == 1)[0]
	n_arvs_p = len(iwhere_p)
	n_arvs_s = len(iwhere_s)

	query_time_p = Trv_subset_p[:,0] + Batch_indices*offset_per_batch + Trv_subset_p[:,1]*offset_per_station
	query_time_s = Trv_subset_s[:,0] + Batch_indices*offset_per_batch + Trv_subset_s[:,1]*offset_per_station

	## No phase type information
	ip_p = np.searchsorted(arrivals_select[:,0], query_time_p)
	ip_s = np.searchsorted(arrivals_select[:,0], query_time_s)

	ip_p_pad = ip_p.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s_pad = ip_s.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p_pad = np.minimum(np.maximum(ip_p_pad, 0), n_arvs - 1) 
	ip_s_pad = np.minimum(np.maximum(ip_s_pad, 0), n_arvs - 1)

	rel_t_p = abs(query_time_p[:, np.newaxis] - arrivals_select[ip_p_pad, 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	rel_t_s = abs(query_time_s[:, np.newaxis] - arrivals_select[ip_s_pad, 0]).min(1)

	## With phase type information
	ip_p1 = np.searchsorted(arrivals_select[iwhere_p,0], query_time_p)
	ip_s1 = np.searchsorted(arrivals_select[iwhere_s,0], query_time_s)

	ip_p1_pad = ip_p1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) # np.array([-1,0,1]).reshape(1,-1), third digit, unnecessary.
	ip_s1_pad = ip_s1.reshape(-1,1) + np.array([-1,0]).reshape(1,-1) 
	ip_p1_pad = np.minimum(np.maximum(ip_p1_pad, 0), n_arvs_p - 1) 
	ip_s1_pad = np.minimum(np.maximum(ip_s1_pad, 0), n_arvs_s - 1)

	rel_t_p1 = abs(query_time_p[:, np.newaxis] - arrivals_select[iwhere_p[ip_p1_pad], 0]).min(1) ## To do neighborhood version, can extend this to collect neighborhoods of points linked.
	rel_t_s1 = abs(query_time_s[:, np.newaxis] - arrivals_select[iwhere_s[ip_s1_pad], 0]).min(1)

	time_vec_slice = np.arange(k_time_edges)

	Inpts = []
	Masks = []
	Lbls = []
	Lbls_query = []
	X_fixed = []
	X_query = []
	Locs = []
	Trv_out = []

	A_sta_sta_l = []
	A_src_src_l = []
	A_prod_sta_sta_l = []
	A_prod_src_src_l = []
	A_src_in_prod_l = []
	A_edges_time_p_l = []
	A_edges_time_s_l = []
	A_edges_ref_l = []

	lp_times = []
	lp_stations = []
	lp_phases = []
	lp_meta = []
	lp_srcs = []
	lp_srcs_active = []

	thresh_mask = 0.01
	for i in range(n_batch):
		# Create inputs and mask
		grid_select = Grid_indices[i]
		ind_select = Sample_indices[i]
		sta_select = Station_indices[i]
		n_spc = x_grids[grid_select].shape[0]
		n_sta_slice = len(sta_select)

		inpt = np.zeros((x_grids[Grid_indices[i]].shape[0], n_sta, 4)) # Could make this smaller (on the subset of stations), to begin with.
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 0] = np.exp(-0.5*(rel_t_p[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 1] = np.exp(-0.5*(rel_t_s[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_p[ind_select,2].astype('int'), Trv_subset_p[ind_select,1].astype('int'), 2] = np.exp(-0.5*(rel_t_p1[ind_select]**2)/(kernel_sig_t**2))
		inpt[Trv_subset_s[ind_select,2].astype('int'), Trv_subset_s[ind_select,1].astype('int'), 3] = np.exp(-0.5*(rel_t_s1[ind_select]**2)/(kernel_sig_t**2))

		trv_out = x_grids_trv[grid_select][:,sta_select,:] ## Subsetting, into sliced indices.
		Inpts.append(inpt[:,sta_select,:]) # sub-select, subset of stations.
		Masks.append(1.0*(inpt[:,sta_select,:] > thresh_mask))
		Trv_out.append(trv_out)
		Locs.append(locs[sta_select])
		X_fixed.append(x_grids[grid_select])

		## Assemble pick datasets
		perm_vec = -1*np.ones(n_sta)
		perm_vec[sta_select] = np.arange(len(sta_select))
		meta = arrivals[lp[i],:]
		phase_vals = phase_observed[lp[i]]
		times = meta[:,0]
		indices = perm_vec[meta[:,1].astype('int')]
		ineed = np.where(indices > -1)[0]
		times = times[ineed] ## Overwrite, now. Double check if this is ok.
		indices = indices[ineed]
		phase_vals = phase_vals[ineed]
		meta = meta[ineed]

		active_sources_per_slice = np.array(lp_src_times_all[i])[np.array(active_sources_per_slice_l[i])]
		ind_inside = np.where(inside_interior[active_sources_per_slice.astype('int')] > 0)[0]
		active_sources_per_slice = active_sources_per_slice[ind_inside]

		ind_src_unique = np.unique(meta[meta[:,2] > -1.0,2]).astype('int') # ignore -1.0 entries.

		if len(ind_src_unique) > 0:
			ind_src_unique = np.sort(np.array(list(set(ind_src_unique).intersection(active_sources_per_slice)))).astype('int')

		src_subset = np.concatenate((src_positions[ind_src_unique], src_times[ind_src_unique].reshape(-1,1) - time_samples[i]), axis = 1)
		if len(ind_src_unique) > 0:
			perm_vec_meta = np.arange(ind_src_unique.max() + 1)
			perm_vec_meta[ind_src_unique] = np.arange(len(ind_src_unique))
			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)
			# ifind = np.where(meta[:,2] > -1.0)[0] ## Need to find picks with a source index inside the active_sources_per_slice
			ifind = np.where([meta[j,2] in ind_src_unique for j in range(meta.shape[0])])[0]
			meta[ifind,-1] = perm_vec_meta[meta[ifind,2].astype('int')] # save pointer to active source, for these picks (in new, local index, of subset of sources)
		else:
			meta = np.concatenate((meta, -1.0*np.ones((meta.shape[0],1))), axis = 1)

		# Do these really need to be on cuda?
		lex_sort = np.lexsort((times, indices)) ## Make sure lexsort doesn't cause any problems
		lp_times.append(times[lex_sort] - time_samples[i])
		lp_stations.append(indices[lex_sort])
		lp_phases.append(phase_vals[lex_sort])
		lp_meta.append(meta[lex_sort]) # final index of meta points into 
		lp_srcs.append(src_subset)

		A_sta_sta = remove_self_loops(knn(torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), torch.Tensor(ftrns1(locs[sta_select])/1000.0).to(device), k = k_sta_edges + 1).flip(0).contiguous())[0]
		A_src_src = remove_self_loops(knn(torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), torch.Tensor(ftrns1(x_grids[grid_select])/1000.0).to(device), k = k_spc_edges + 1).flip(0).contiguous())[0]
		## Cross-product graph is: source node x station node. Order as, for each source node, all station nodes.

		# Cross-product graph, nodes connected by: same source node, connected stations
		A_prod_sta_sta = (A_sta_sta.repeat(1, n_spc) + n_sta_slice*torch.arange(n_spc).repeat_interleave(n_sta_slice*k_sta_edges).view(1,-1).to(device)).contiguous()
		A_prod_src_src = (n_sta_slice*A_src_src.repeat(1, n_sta_slice) + torch.arange(n_sta_slice).repeat_interleave(n_spc*k_spc_edges).view(1,-1).to(device)).contiguous()	

		# For each unique spatial point, sum in all edges.
		A_src_in_prod = torch.cat((torch.arange(n_sta_slice*n_spc).view(1,-1), torch.arange(n_spc).repeat_interleave(n_sta_slice).view(1,-1)), dim = 0).to(device).contiguous()

		## Sub-selecting from the time-arrays, is easy, since the time-arrays are indexed by station (triplet indexing; )
		len_dt = len(x_grids_trv_refs[grid_select])

		### Note: A_edges_time_p needs to be augmented: by removing stations, we need to re-label indices for subsequent nodes,
		### To the "correct" number of stations. Since, not n_sta shows up in definition of edges. "assemble_pointers.."
		A_edges_time_p = x_grids_trv_pointers_p[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
		A_edges_time_s = x_grids_trv_pointers_s[grid_select][np.tile(np.arange(k_time_edges*len_dt), n_sta_slice) + (len_dt*k_time_edges)*sta_select.repeat(k_time_edges*len_dt)]
		## Need to convert these edges again. Convention is:
		## subtract i (station index absolute list), divide by n_sta, mutiply by N stations, plus ith station (in permutted indices)
		# shape is len_dt*k_time_edges*len(sta_select)
		one_vec = np.repeat(sta_select*np.ones(n_sta_slice), k_time_edges*len_dt).astype('int') # also used elsewhere
		A_edges_time_p = (n_sta_slice*(A_edges_time_p - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
		A_edges_time_s = (n_sta_slice*(A_edges_time_s - one_vec)/n_sta) + perm_vec[one_vec] # transform indices, based on subsetting of stations.
		# print('permute indices 1')
		assert(A_edges_time_p.max() < n_spc*n_sta_slice) ## Can remove these, after a bit of testing.
		assert(A_edges_time_s.max() < n_spc*n_sta_slice)

		A_sta_sta_l.append(A_sta_sta.cpu().detach().numpy())
		A_src_src_l.append(A_src_src.cpu().detach().numpy())
		A_prod_sta_sta_l.append(A_prod_sta_sta.cpu().detach().numpy())
		A_prod_src_src_l.append(A_prod_src_src.cpu().detach().numpy())
		A_src_in_prod_l.append(A_src_in_prod.cpu().detach().numpy())
		A_edges_time_p_l.append(A_edges_time_p)
		A_edges_time_s_l.append(A_edges_time_s)
		A_edges_ref_l.append(x_grids_trv_refs[grid_select])

		x_query = np.random.rand(n_spc_query, 3)*scale_x + offset_x # Check if scale_x and offset_x are correct.

		if len(lp_srcs[-1]) > 0:
			x_query[0:len(lp_srcs[-1]),0:3] = lp_srcs[-1][:,0:3]

		if len(active_sources_per_slice) == 0:
			lbls_grid = np.zeros((x_grids[grid_select].shape[0], len(t_slice)))
			lbls_query = np.zeros((n_spc_query, len(t_slice)))
		else:
			active_sources_per_slice = active_sources_per_slice.astype('int')

			lbls_grid = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_grids[grid_select]), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)
			lbls_query = (np.expand_dims(np.exp(-0.5*(((np.expand_dims(ftrns1(x_query), axis = 1) - np.expand_dims(ftrns1(src_positions[active_sources_per_slice]), axis = 0))**2)/(src_spatial_kernel**2)).sum(2)), axis = 1)*np.exp(-0.5*(((time_samples[i] + t_slice).reshape(1,-1,1) - src_times[active_sources_per_slice].reshape(1,1,-1))**2)/(src_t_kernel**2))).max(2)

		X_query.append(x_query)
		Lbls.append(lbls_grid)
		Lbls_query.append(lbls_query)

	return [Inpts, Masks, X_fixed, X_query, Locs, Trv_out], [Lbls, Lbls_query, lp_times, lp_stations, lp_phases, lp_meta, lp_srcs], [A_sta_sta_l, A_src_src_l, A_prod_sta_sta_l, A_prod_src_src_l, A_src_in_prod_l, A_edges_time_p_l, A_edges_time_s_l, A_edges_ref_l] # , data
