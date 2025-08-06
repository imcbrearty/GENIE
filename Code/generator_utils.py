import numpy as np
from sklearn.metrics import pairwise_distances as pd
from utils import *
from matplotlib.colors import LinearSegmentedColormap, Normalize




def inv_2x2(matrix):
	"""Compute inverse of a 2x2 matrix using direct formula."""
	a, b = matrix[0, 0], matrix[0, 1]
	c, d = matrix[1, 0], matrix[1, 1]
	det = a * d - b * c
	return np.array([[d, -b], [-c, a]]) / det

# def radial_function(points, center, inv_cov, sigma_radial, p_exp, scale_factor):
# 	"""Compute Mahalanobis distances and PDF values."""
# 	diff = points - center
# 	mahalanobis2 = np.sqrt(np.sum((diff[:,0:2]/sigma_radial) @ inv_cov * (diff[:,0:2]/sigma_radial), axis=1))
# 	radial_pdf = scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis2, 2.0*p_exp))

# 	# mahalanobis3 = np.sqrt(np.sum(diff[:,0:2] @ inv_cov * diff[:,0:2], axis=1))
# 	# radial_pdf1= scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis3/sigma_radial, 2.0*p_exp))
# 	# assert(np.abs(radial_pdf - radial_pdf1).max() < 1e-2)

# 	return radial_pdf

def radial_function(points, center, inv_cov, p_exp=3.0, scale_factor=1.0, d_noise=None, lambda_noise=0.1):
	"""Compute Mahalanobis distances and PDF values."""
	diff = points - center
	mahalanobis2 = np.sqrt(np.sum((diff[:,0:2]) @ inv_cov * (diff[:,0:2]), axis=1))
	if d_noise is not None:
		mahalanobis2 = np.clip(mahalanobis2 + lambda_noise * d_noise, 0, None)
	try:
		radial_pdf = scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis2, 2.0*p_exp))
	except:
		pdb.set_trace()
		print(f"radial_pdf range: {radial_pdf.min():.6f} to {radial_pdf.max():.6f}")
		print(f"scale_factor: {scale_factor}")
		print(f"p_exp: {p_exp}")
		print(f"lambda_noise: {lambda_noise}")
		print(f"d_noise: {d_noise}")
		print(f"mahalanobis2: {mahalanobis2.min():.6f} to {mahalanobis2.max():.6f}")
		print(f"diff: {diff.min():.6f} to {diff.max():.6f}")
		print(f"inv_cov: {inv_cov.min():.6f} to {inv_cov.max():.6f}")
	# mahalanobis3 = np.sqrt(np.sum(diff[:,0:2] @ inv_cov * diff[:,0:2], axis=1))
	# radial_pdf1= scale_factor * np.exp(-(1/(2 * 9**(p_exp - 1))) * np.power(mahalanobis3/sigma_radial, 2.0*p_exp))
	# assert(np.abs(radial_pdf - radial_pdf1).max() < 1e-2)

	return radial_pdf

def sample_correlated_noise(covariance, ind_use, n_repeat = 1, cholesky_matrix = None):
	"""Generate spatially correlated noise using Cholesky decomposition.
	TO DO: use pre-computed coefficients.
	Args:
		points (np.ndarray): Array of points
		sigma_noise (float): Covariance scale parameter
		cholesky_matrix (np.ndarray): Pre-computed Cholesky matrix
	Returns:
	np.ndarray: Spatially correlated noise
	"""
	# covariance = compute_covariance(distance, sigma_noise=sigma_noise)
	if cholesky_matrix is None:
		L = np.linalg.cholesky(covariance[ind_use.reshape(-1,1), ind_use.reshape(1,-1)])
		z = np.random.randn(len(ind_use), n_repeat) 
		noise = (L @ z).squeeze()
	else:
		L = cholesky_matrix # Matrix on all the locations
		z = np.random.randn(len(L), n_repeat) 
		# z = np.random.multivariate_normal(np.zeros(len(points)), np.identity(len(points)))
		noise = (L @ z).squeeze()
		noise = noise[ind_use] # Select only the indices we are interested in
	return noise

def generate_ellipse_parameters(angle=None, length1=None, length2=None, sigma_radial=10000.0):
	"""Generate random ellipse parameters."""
	if angle is None:
		angle = np.random.uniform(0, 2*np.pi)
	if length1 is None: 
		length1 = sigma_radial 
	if length2 is None:
		length2 = sigma_radial
	
	R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
	D = np.array([[length1, 0],
                  [0, length2]])
    
	cov_ellipse = R @ D @ D @ R.T
	inv_cov = inv_2x2(cov_ellipse)
    
	return angle, length1, length2, cov_ellipse, inv_cov

def calculate_inertia(points, indices, scale_km = 1000.0):
	"""Calculate inertia for a subset of points."""
	if len(indices) == 0:
		return 0.0
	center = points[indices].mean(axis=0, keepdims = True)
	return np.sum(((points[indices] - center)/scale_km)**2)/len(indices) ## Shuld we actually take mean?

def compute_morans_I_metric(W, ind_use, locs_use_cart, src_cart, selected_list, return_matrix = False):
	"""Calculate Moran's I for a subset of points being inside the circle of radius maximum distance between the centroid and one of the selected points.
	This method first filters the points to create a new selected vector and new W then returns morans_I_binary(new_selected, new_W)
	"""
	# Find the maximum distance between the centroid and one of the selected points
	# centroid = np.copy(src_cart) # points[selected].mean(axis=0)
	assert(len(selected_list) <= 2)
	offset_dist = np.linalg.norm(locs_use_cart - src_cart, axis = 1)
	if len(selected_list[0]) > 0:
		filtering_distance = 1.25*np.quantile(offset_dist[selected_list[0]], 0.98) ## Base the distance threshold on the first index set in selected (e.g., the targets)
	elif len(selected_list) == 1: ## If only one input, no distance
		filtering_distance = 0.0
	elif len(selected_list[1]) > 0:
		filtering_distance = 1.25*np.quantile(offset_dist[selected_list[1]], 0.98)
	else:
		filtering_distance = 0.0

	ipoints_within_radius = np.where(offset_dist <= filtering_distance)[0] ## Want footprint of network within a proportion of radius of max association
	ind_use_subset = ind_use[ipoints_within_radius]

	
	W_select = W[ind_use_subset.reshape(-1,1), ind_use_subset.reshape(1,-1)]
	perm_vec = (-1*np.ones(len(ind_use))).astype('int')
	perm_vec[ipoints_within_radius] = np.arange(len(ipoints_within_radius))

	morans_vals = []
	for selected in selected_list:
		new_selected = perm_vec[selected] ## Map selectred indices to the new index ordering
		morans_vals.append(morans_I_binary(W_select, new_selected))

	if return_matrix == True:

		return np.array(morans_vals), W_select, ind_use_subset

	else:

		return np.array(morans_vals)

def morans_I_binary(W, selected):
	"""Calculate Moran's I for binary selection vector using sparse W."""
	N = W.shape[0]
	y = np.zeros(len(W)) # selected.astype(float)
	y[selected] = 1.0
	k = y.sum()
	if k == 0 or k == len(y):  # Return 0 if no points or all points are selected
		return 0.0
	y_bar = k / len(y)
	row_sum = W.sum()
	num = (W*(y[np.arange(N).reshape(-1,1)] - y_bar)*(y[np.arange(N).reshape(1,-1)] - y_bar)).sum()
	denom = np.sum((y - y_bar) ** 2)
	I = (len(y) / np.maximum(row_sum, 1e-5)) * (num / np.maximum(denom, 1e-5))
	return I

def sample_synthetic_moveout_pattern_generator(Srcs, Mags, Inds, locs, prob_vec, chol_params, ftrns1, pdist_p, pdist_s, n_samples = 100, srcs = None, mags = None, locs_use_list = None, inds = None, use_l1 = False, mask_noise = False, return_features = True,  Picks_P_lists=None, Picks_S_lists=None, distance_abs = None, debug=False):
	## Sample sources
	if srcs is None: ## If srcs is None, assume global dataset Srcs is available and sample from this. 
		## Otherwise, sample the given input lists of sources and magnitudes
		ichoose = np.random.choice(len(Srcs), p = prob_vec, size = n_samples)
		locs_use_list = [locs[Inds[j]] for j in ichoose]
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
		srcs_sample = Srcs[ichoose]
		mags_sample = Mags[ichoose]
		srcs_samples_cart = ftrns1(srcs_sample)

	else:
		## In this case, assume all of srcs, mags, locs_use_list, and inds, are all supplied
		n_samples = len(srcs)
		ichoose = np.arange(len(srcs))
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
		Srcs = np.copy(srcs)
		srcs_sample = np.copy(srcs)
		Mags = np.copy(mags)
		mags_sample = Mags[ichoose]
		Inds = [ind_use for ind_use in inds]
		srcs_samples_cart = ftrns1(srcs_sample)

	## Extract parameters
	p_exp = chol_params['p_exp']
	radial_factor_p = chol_params['radial_factor_p']
	radial_factor_s = chol_params['radial_factor_s']
	miss_pick_rate = chol_params['miss_pick_rate']
	sigma_noise = chol_params['sigma_noise']
	lambda_noise = chol_params['lambda_noise']
	perturb_factor = chol_params['perturb_factor']
	angle = 0.0

	scale_factor = 1 - miss_pick_rate
	
	## Note: could likely remove scale_factor, and only use these per phase type and event scale factors
	peturb_range = [1.0 - perturb_factor, 1.0 + perturb_factor]
	# perturb_factor_sample = np.random.rand(n_samples)*(peturb_range[1] - peturb_range[0]) + peturb_range[0]
	perturb_factor_p, perturb_factor_s = np.random.rand(2,n_samples)*(peturb_range[1] - peturb_range[0]) + peturb_range[0]
	
	## Radius values per sources (need to add per-source random perturb, not fixed scaling)
	sigma_radial_p = radial_factor_p * perturb_factor_p * np.array([pdist_p(magnitude) for magnitude in mags_sample])/3  # P-wave detection radius
	sigma_radial_s = radial_factor_s * perturb_factor_s * np.array([pdist_s(magnitude) for magnitude in mags_sample])/3  # S-wave detection radius
	
	# Noise correlation range
	phase_noise_corr_range = [0.1, 0.4] ## Range to "weight" the independent S wave noise probabilities compared to P waves
	phase_noise_corr_sample = np.random.rand(n_samples)*(phase_noise_corr_range[1] - phase_noise_corr_range[0]) + phase_noise_corr_range[0]

	## Setup absolute network parameters
	tol = 1e-8
	distance_abs = pd(ftrns1(locs), ftrns1(locs)) ## Absolute stations O(n^2)
	if use_l1 == False:
		covariance_abs = np.exp(-0.5*(distance_abs**2) / (sigma_noise**2)) + tol*np.eye(distance_abs.shape[0])
	else:
		covariance_abs = np.exp(-1.0*np.abs(distance_abs) / (sigma_noise**1)) + tol*np.eye(distance_abs.shape[0])
	cholesky_matrix = np.linalg.cholesky(covariance_abs) # Compute the Cholesky decomposition of the covariance matrix

	## Could cach the repeated set indices, and save cholesky factors for each
	# # pdb.set_trace()
	# pre_check_unique_sets = True
	# if pre_check_unique_sets == True:
	# 	set_equal = np.vstack([np.array([set(Inds[ichoose[i]]) == set(Inds[ichoose[j]]) for j in range(len(ichoose))]).reshape(1,-1) for i in range(len(ichoose))])
	# 	iset1, iset2 = np.where(set_equal > 0)
	# 	graph_components = nx.connected_components(to_networkx(Data(edge_index = torch.Tensor(np.concatenate((iset1.reshape(1,-1), iset2.reshape(1,-1)), axis = 0)).long())).to_undirected()) # .connected_components()

	ikeep_p1, ikeep_p2 = [], []
	ikeep_s1, ikeep_s2 = [], []

	for i in range(n_samples):

		# angle_p, length1_p, length2_p = experiment_result_p['parameters']['angle'], experiment_result_p['parameters']['length1'], experiment_result_p['parameters']['length2']
		angle, length1, length2, _, inv_cov_p = generate_ellipse_parameters(angle = angle, length1 = None, length2 = None, sigma_radial=sigma_radial_p[i])
		angle, length1, length2, _, inv_cov_s = generate_ellipse_parameters(angle = angle, length1 = None, length2 = None, sigma_radial=sigma_radial_s[i])
		# points, center, inv_cov, sigma_radial

		## Sample P and S wave noise (do not use pre-built Cholesky matrix for S wave sampling)
		## Could "cache" the cholesky factor for each unique (repeated) set of Inds[ichoose[i]]. 
		## In cases where Inds[ichoose[i]] is repeated, then the cholesky factor is fixed. (should occur in practice during sampling)
		if np.random.rand() < 0.5: ## Note: can make sigma_noise a random sample over a range
			noise_p = sample_correlated_noise(covariance_abs, Inds[ichoose[i]], n_repeat = 1, cholesky_matrix=cholesky_matrix)
			noise_s = noise_p + phase_noise_corr_sample[i]*sample_correlated_noise(covariance_abs, Inds[ichoose[i]], cholesky_matrix=cholesky_matrix, n_repeat = 1)
		else:
			noise_s = sample_correlated_noise(covariance_abs, Inds[ichoose[i]], cholesky_matrix=cholesky_matrix, n_repeat = 1)
			noise_p = noise_s + phase_noise_corr_sample[i]*sample_correlated_noise(covariance_abs, Inds[ichoose[i]], cholesky_matrix=cholesky_matrix, n_repeat = 1)			
		
		radial_pdf_p = radial_function(locs_use_cart_list[i], srcs_samples_cart[i], inv_cov_p, scale_factor=scale_factor, p_exp=p_exp, d_noise=noise_p, lambda_noise=lambda_noise) ## Note: p_exp is the exponent term
		radial_pdf_s = radial_function(locs_use_cart_list[i], srcs_samples_cart[i], inv_cov_s, scale_factor=scale_factor, p_exp=p_exp, d_noise=noise_s, lambda_noise=lambda_noise) ## Note: p_exp is the exponent term
		updated_pdf_p = np.copy(radial_pdf_p)
		updated_pdf_s = np.copy(radial_pdf_s)

		updated_mask_p = np.random.binomial(1, updated_pdf_p)
		updated_mask_s = np.random.binomial(1, updated_pdf_s)
		updated_idx_p = np.where(updated_mask_p)[0]
		updated_idx_s = np.where(updated_mask_s)[0]

		ikeep_p1.append(i*np.ones(len(updated_idx_p)))
		ikeep_p2.append(updated_idx_p)
		ikeep_s1.append(i*np.ones(len(updated_idx_s)))
		ikeep_s2.append(updated_idx_s)

		## Plot the results
		if debug == True:
			fig, axs = plt.subplots(2, 4, figsize=(18,10), sharex=True, sharey=True)
			checkpoint_dir = "comparisons"
			plt.suptitle(f'Source ({srcs_samples_cart[i][0]:.1f}, {srcs_samples_cart[i][1]:.1f}), Magnitude={Mags[ichoose[i]]:.1f}')
			
			# Plot real points
			axs[0,0].set_title('%0.3f'%Mags[ichoose[i]])
			axs[0,1].set_title('%0.3f'%Mags[ichoose[i]])
			locs_use = locs_use_cart_list[i]
			for j in range(2):
				axs[j, 0].scatter(locs_use_cart_list[i][:, 0], locs_use_cart_list[i][:, 1], c = 'grey', marker = '^')
				# axs[0, 0].set_aspect(1.0/np.cos(np.pi*locs_use[:,0].mean()/180.0))
				axs[j,0].scatter(srcs_samples_cart[i,0], srcs_samples_cart[i,1], c = 'm', marker = 's')
				axs[j][0].set_xlabel('X')
				axs[j][0].set_ylabel('Y')
				axs[j][0].legend()
				axs[j][0].set_aspect('equal')

			## Real event (P and S)
			axs[0,0].scatter(locs_use[Picks_P_lists[ichoose[i]][:,1].astype('int'),0], locs_use[Picks_P_lists[ichoose[i]][:,1].astype('int'),1], c = 'red', marker = '^')
			axs[1,0].scatter(locs_use[Picks_S_lists[ichoose[i]][:,1].astype('int'),0], locs_use[Picks_S_lists[ichoose[i]][:,1].astype('int'),1], c = 'red', marker = '^')

			# 3. Plot radial_pdf with lambda_corr (probability color))
			sc3 = axs[0,1].scatter(
				locs_use_cart_list[i][:, 0], locs_use_cart_list[i][:, 1],
				c=radial_pdf_p, cmap='Blues', s=40, marker='^'
			)
			cbar3 = plt.colorbar(sc3, ax=axs[0,1], fraction=0.046, pad=0.04)
			cbar3.set_label(f'radial_pdf (λ={lambda_noise:.2f})')
			axs[0,1].set_facecolor('#f5e9ff')

			sc4 = axs[1,1].scatter(
				locs_use_cart_list[i][:, 0], locs_use_cart_list[i][:, 1],
				c=radial_pdf_s, cmap='Blues', s=40, marker='^'
			)
			cbar4 = plt.colorbar(sc4, ax=axs[1,1], fraction=0.046, pad=0.04)
			cbar4.set_label(f'radial_pdf (λ={lambda_noise:.2f})')
			axs[1,1].set_facecolor('#f5e9ff')

			axs[0,1].scatter(locs_use[updated_idx_p,0], locs_use[updated_idx_p,1], c = 'red', marker = '^')
			axs[1,1].scatter(locs_use[updated_idx_s,0], locs_use[updated_idx_s,1], c = 'red', marker = '^')
				
			for j in range(2):
				axs[j,1].scatter(srcs_samples_cart[i][0], srcs_samples_cart[i][1], c='yellow', marker='*', s=120, label='Source')
				axs[j,1].set_title(f'Radial PDF of P with noise (λ={lambda_noise:.2f})')
				axs[j,1].set_xlabel('X')
				axs[j,1].set_ylabel('Y')
				axs[j,1].legend()
				axs[j,1].set_aspect('equal')

			# 1. Plot noise for each point (blue to red colormap)
			for j in range(2):
				axs[j][2].scatter(srcs_samples_cart[i][0], srcs_samples_cart[i][1], c='black', marker='*', s=120, label='Source')
				axs[j][2].set_title(f'Noise Visualization (sigma_noise={sigma_noise:.0f})')
				axs[j][2].set_xlabel('X')
				axs[j][2].set_ylabel('Y')
				axs[j][2].legend()
				axs[j][2].set_aspect('equal')
			sc1 = axs[0][2].scatter(
				locs_use_cart_list[i][:, 0], locs_use_cart_list[i][:, 1],
				c=noise_p.flatten(), cmap='coolwarm', s=80, marker='o'
			)
			cbar1 = plt.colorbar(sc1, ax=axs[0][2], fraction=0.046, pad=0.04)
			cbar1.set_label('Noise Value')

			sc2 = axs[1][2].scatter(
				locs_use_cart_list[i][:, 0], locs_use_cart_list[i][:, 1],
				c=noise_s.flatten(), cmap='coolwarm', s=80, marker='o'
			)
			cbar2 = plt.colorbar(sc2, ax=axs[1][2], fraction=0.046, pad=0.04)
			cbar2.set_label('Noise Value')


			# 4. Plot ellipse and radial function probability distribution
			# Create grid for density plot
			x_grid = np.linspace(locs_use_cart_list[i][:, 0].min() - 0.1, locs_use_cart_list[i][:, 0].max() + 0.1, 100)
			y_grid = np.linspace(locs_use_cart_list[i][:, 1].min() - 0.1, locs_use_cart_list[i][:, 1].max() + 0.1, 100)
			X, Y = np.meshgrid(x_grid, y_grid)
			# Elliptical Mahalanobis distance for the grid
			grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
			
			Z_p = radial_function(grid_points, srcs_samples_cart[i][:2], inv_cov_p, scale_factor=scale_factor, p_exp=p_exp, d_noise=0.0, lambda_noise=0.0)
			Z_p = Z_p.reshape(X.shape)  # Reshape back to 2D grid
			axs[0,3].set_title(f'Radial PDF, no noise \n p_exp={p_exp}, scale_factor={scale_factor} \nsigma_radial={sigma_radial_p[i]:.0f}')
			
			# Plot radial function probability distribution
			im_p = axs[0,3].contourf(X, Y, Z_p, levels=20, cmap='viridis', alpha=0.6)
			axs[0,3].contour(X, Y, Z_p, levels=10, colors='black', alpha=0.4, linewidths=0.5)
			cbar_p = plt.colorbar(im_p, ax=axs[0,3], fraction=0.046, pad=0.04)
			# Add colorbar for the probability distribution
			cbar_p.set_label('Radial Function Probability')
			
			Z_s = radial_function(grid_points, srcs_samples_cart[i][:2], inv_cov_s, scale_factor=scale_factor, p_exp=p_exp, d_noise=0.0, lambda_noise=0.0)
			Z_s = Z_s.reshape(X.shape)  # Reshape back to 2D grid
			axs[1,3].set_title(f'Radial PDF, no noise \n p_exp={p_exp}, scale_factor={scale_factor} \nsigma_radial={sigma_radial_s[i]:.0f}')
			
			# Plot radial function probability distribution
			im_s = axs[1,3].contourf(X, Y, Z_s, levels=20, cmap='viridis', alpha=0.6)
			axs[1,3].contour(X, Y, Z_s, levels=10, colors='black', alpha=0.4, linewidths=0.5)
			cbar_s = plt.colorbar(im_s, ax=axs[1,3], fraction=0.046, pad=0.04)
			# Add colorbar for the probability distribution
			cbar_s.set_label('Radial Function Probability')

			for j in range(2):
				ax4 = axs[j,3]
				# Mark the source location
				ax4.scatter(srcs_samples_cart[i][0], srcs_samples_cart[i][1], c='red', marker='*', s=200, label='Source', zorder=15)
				# Add the original data points
				ax4.scatter(locs_use_cart_list[i][:, 0], locs_use_cart_list[i][:, 1], c='white', s=30, alpha=0.8, label='Data Points', zorder=5)
				
				ax4.set_xlabel('X')
				ax4.set_ylabel('Y')
				ax4.legend()
				ax4.grid(True, alpha=0.3)
				ax4.set_aspect('equal')  # Maintain equal aspect ratio so circles appear as circles
				
			plt.tight_layout()
			plt.savefig(f'{checkpoint_dir}/comparison_{i}.png')
			plt.close()


	## Merge results
	ikeep_p1 = np.hstack(ikeep_p1).astype('int')
	ikeep_p2 = np.hstack(ikeep_p2).astype('int')
	ikeep_s1 = np.hstack(ikeep_s1).astype('int')
	ikeep_s2 = np.hstack(ikeep_s2).astype('int')

	## Add computation of features
	Features = []
	if return_features == True:
		length_scale = sigma_noise ## Is this reasonable? ## Instead, can base it on a multiple of nearest neighbors

		W_abs = np.exp(-(distance_abs**2) / (2 * (length_scale**2))) # Compute weights using a Gaussian kernel: W_ij = exp(-d^2 / (2*length_scale^2))
		np.fill_diagonal(W_abs, 0.0)  # Remove self-loops

		Morans_pred_p = []
		Morans_pred_s = []
		Inertia_pred_p = []
		Inertia_pred_s = []
		Cnt_pred_p, Cnt_pred_s = [], []

		Morans_trgt_p = []
		Morans_trgt_s = []
		Inertia_trgt_p = []
		Inertia_trgt_s = []
		Cnt_trgt_p, Cnt_trgt_s = [], []

		Intersection_p_ratio = []
		Intersection_s_ratio = []

		for i in range(n_samples):
			ind_pred_p = ikeep_p2[np.where(ikeep_p1 == i)[0]]
			ind_pred_s = ikeep_s2[np.where(ikeep_s1 == i)[0]]		
			ind_trgt_p = Picks_P_lists[ichoose[i]][:,1].astype('int')
			ind_trgt_s = Picks_S_lists[ichoose[i]][:,1].astype('int')	

			## [1]. Morans metric
			morans_metric_p = compute_morans_I_metric(W_abs, Inds[ichoose[i]], locs_use_cart_list[i], srcs_samples_cart[i], [ind_trgt_p, ind_pred_p]) ## Should pass in trgt first, as filtering scale is set by target data
			morans_metric_s = compute_morans_I_metric(W_abs, Inds[ichoose[i]], locs_use_cart_list[i], srcs_samples_cart[i], [ind_trgt_s, ind_pred_s])
			Morans_trgt_p.append(morans_metric_p[0])
			Morans_trgt_s.append(morans_metric_s[0])
			Morans_pred_p.append(morans_metric_p[1])
			Morans_pred_s.append(morans_metric_s[1])

			# morans_metric_p_trgt, morans_metric_p_pred = morans_metric_p
			# morans_metric_s_trgt, morans_metric_s_pred = morans_metric_s

			## [2]. Inertia
			Inertia_trgt_p.append(calculate_inertia(locs_use_cart_list[i], ind_trgt_p))
			Inertia_trgt_s.append(calculate_inertia(locs_use_cart_list[i], ind_trgt_s))
			Inertia_pred_p.append(calculate_inertia(locs_use_cart_list[i], ind_pred_p))
			Inertia_pred_s.append(calculate_inertia(locs_use_cart_list[i], ind_pred_s))

			## [3]. Counts
			Cnt_trgt_p.append(len(ind_trgt_p))
			Cnt_trgt_s.append(len(ind_trgt_s))
			Cnt_pred_p.append(len(ind_pred_p))
			Cnt_pred_s.append(len(ind_pred_s))

			## [4]. Intersection
			Intersection_p_ratio.append((1.0 if (len(ind_trgt_p) > 0) else np.nan)*len(set(ind_trgt_p).intersection(ind_pred_p))/np.maximum(len(ind_trgt_p), 1.0))
			Intersection_s_ratio.append((1.0 if (len(ind_trgt_s) > 0) else np.nan)*len(set(ind_trgt_s).intersection(ind_pred_s))/np.maximum(len(ind_trgt_s), 1.0))

		## Merge outputs
		Morans_pred_p = np.hstack(Morans_pred_p)
		Morans_pred_s = np.hstack(Morans_pred_s)
		Morans_trgt_p = np.hstack(Morans_trgt_p)
		Morans_trgt_s = np.hstack(Morans_trgt_s)

		Inertia_pred_p = np.hstack(Inertia_pred_p)
		Inertia_pred_s = np.hstack(Inertia_pred_s)
		Inertia_trgt_p = np.hstack(Inertia_trgt_p)
		Inertia_trgt_s = np.hstack(Inertia_trgt_s)

		Cnt_pred_p = np.hstack(Cnt_pred_p)
		Cnt_pred_s = np.hstack(Cnt_pred_s)
		Cnt_trgt_p = np.hstack(Cnt_trgt_p)
		Cnt_trgt_s = np.hstack(Cnt_trgt_s)

		Intersection_p_ratio = np.hstack(Intersection_p_ratio)
		Intersection_s_ratio = np.hstack(Intersection_s_ratio)

		Features.append([Morans_pred_p, Morans_pred_s, Morans_trgt_p, Morans_trgt_s])
		Features.append([Inertia_pred_p, Inertia_pred_s, Inertia_trgt_p, Inertia_trgt_s])
		Features.append([Cnt_pred_p, Cnt_pred_s, Cnt_trgt_p, Cnt_trgt_s])
		Features.append([Intersection_p_ratio, Intersection_s_ratio])
		# Features = [[Morans_trgt_p, Morans_trgt_s, Morans_pred_p, Morans_pred_p], [Inertia_trgt_p, Inertia_trgt_s, Inertia_pred_p, Inertia_pred_p], ]

		# error('Not yet implemented')
		# pass

	return srcs_sample, mags_sample, Features, ichoose, [ikeep_p1, ikeep_p2, ikeep_s1, ikeep_s2]

def comparison_plots(chol_params, srcs, mags, locs, inds, locs_use_list, Picks_P_lists, Picks_S_lists, n_samples = 100, srcs_sample = None, mags_sample = None, locs_use_cart_list = None, inds_use = None):
	"""Generate comparison plots for synthetic moveout patterns."""
	if srcs_sample is None:
		srcs_sample = srcs
	if mags_sample is None:
		mags_sample = mags
	if locs_use_cart_list is None:
		locs_use_cart_list = [ftrns1(l) for l in locs_use_list]
	if inds_use is None:
		inds_use = inds

	srcs_sample, mags_sample, Features, ichoose, ikeep = sample_synthetic_moveout_pattern_generator(
		srcs=srcs_sample, mags=mags_sample, locs=locs, prob_vec=None,
		chol_params=chol_params, ftrns1=ftrns1,
		pdist_p=pdist_p, pdist_s=pdist_s,
		n_samples=n_samples,
		locs_use_list=locs_use_list,
		inds=inds_use,
		Picks_P_lists=Picks_P_lists,
		Picks_S_lists=Picks_S_lists
	)

	return srcs_sample, mags_sample, Features, ichoose, ikeep

# if __name__ = "__main__":
# 	# Example usage
# 	chol_params = {
# 		'p_exp': 3.0,
# 		'radial_factor_p': 1.0,
# 		'radial_factor_s': 1.0,
# 		'miss_pick_rate': 0.1,
# 		'sigma_noise': 10000.0,
# 		'lambda_noise': 0.1,
# 		'perturb_factor': 0.2
# 	}
# 	srcs = np.random.rand(10, 2) * 100
# 	mags = np.random.rand(10) * 5 + 5
# 	locs = np.random.rand(20, 2) * 100
# 	locs_use_list = [locs[i] for i in range(20)]
# 	Picks_P_lists = [np.array([[0, i]]) for i in range(20)]
# 	Picks_S_lists = [np.array([[0, i]]) for i in range(20)]
# 	n_samples = 5

# 	comparison_plots(chol_params, srcs, mags, locs, locs_use_list, Picks_P_lists, Picks_S_lists, n_samples)