from bnt_smooth import LognormalWeakLensingSim
import numpy as np
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
from functools import partial
from bnt_smooth import ProcessMaps  # Make sure this import is present



def parent_nz(z):
    z_euc = 0.9 / 2 ** 0.5
    return (z / z_euc) ** 2. * np.exp(-(z / z_euc) ** 1.5)

def make_equal_ngal_bins(nz_func, z_grid, nbins, sigma_z0=0.05):
    """
    Split the parent n(z) into `nbins` tomographic bins with equal galaxy counts
    *and* apply the Euclid photo‑z smoothing of Eq. (115) in Blanchard et al. (2019).

    Parameters
    ----------
    nz_func   : callable
        Parent redshift distribution n(z) (does NOT need to be normalised).
    z_grid    : ndarray
        1‑D grid on which n(z) is evaluated.
    nbins     : int
        Number of tomographic bins.
    sigma_z0  : float, optional
        Photometric redshift error normalisation, σ_z = sigma_z0·(1+z).
        Default 0.05 (Table 5 of Blanchard et al.).

    Returns
    -------
    nz_bins : list
        List of tuples (z_grid, n_i_obs(z)) for each tomographic bin i, where
        each n_i_obs is individually normalised to ∫ n_i_obs dz = 1.
    edges   : ndarray
        Redshift edges z_0 … z_{nbins}.
    """
    # --- parent n(z) (normalised) ------------------------------------------------
    nz_parent = nz_func(z_grid)
    nz_parent /= np.trapz(nz_parent, z_grid)

    # --- cumulative distribution to get equal‑ngal edges -------------------------
    cdf = np.cumsum(nz_parent)
    cdf /= cdf[-1]
    inv_cdf = interp1d(np.concatenate([[0], cdf, [1]]),
                       np.concatenate([[z_grid[0]], z_grid, [z_grid[-1]]]))
    edges = inv_cdf(np.linspace(0, 1, nbins + 1))

    # --- photo‑z smoothing (Eq. 115) --------------------------------------------
    nz_bins = []
    sigma_of_z = lambda zz: sigma_z0 * (1.0 + zz)          # σ_z(z)

    for i in range(nbins):
        z_lo, z_hi = edges[i], edges[i + 1]
        sig = sigma_of_z(z_grid)

        # Eq. (115): observed distribution in bin i
        kernel = 0.5 * (erf((z_hi - z_grid) / (np.sqrt(2) * sig))
                        - erf((z_lo - z_grid) / (np.sqrt(2) * sig)))
        nz_i_obs = nz_parent * kernel

        # re‑normalise each bin to unit area
        area = np.trapz(nz_i_obs, z_grid)
        if area > 0:
            nz_i_obs /= area
        nz_bins.append((z_grid, nz_i_obs))

    return nz_bins, edges


z = np.linspace(0.01, 2.5, 500)

# --- Simulation settings ---

nbins = 3
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0/nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.
sigma8 = 0.8
seed = 1234


lognormal_shift = 1.0
#nslices = 5
#l_max = 32
#nside = 32

nslices = 50
l_max = 3000
nside = 2048

n_realizations = 100

step_frac = 0.05  # 5% step size for numerical derivative



def generate_one_realization(i, z, nz_list, n_eff_list, sigma_eps_list,
                             baryon_feedback, sigma8, lognormal_shift,
                             seed, l_max, nside, nslices):
    """
    Generate a single realization of the κ data vector and BNT-transformed data vector.
    
    Parameters
    ----------
    i : int
        Realization index used to offset the seed.
    All other arguments are simulation settings passed via partial().
    
    Returns
    -------
    data_vector : ndarray
        Standard data vector.
    bnt_data_vector : ndarray
        BNT-transformed data vector.
    """
    sim = ProcessMaps(
        z_array=z,
        nz_list=nz_list,
        n_eff_list=n_eff_list,
        sigma_eps_list=sigma_eps_list,
        baryon_feedback=baryon_feedback,
        sigma8=sigma8,
        lognormal_shift=lognormal_shift,
        seed=seed + i,
        l_max=l_max,
        nside=nside,
        nslices=nslices
    )

    kappa_maps = sim.generate_noisy_kappa_maps()
    data_vector = sim.compute_data_vector(kappa_maps)
    bnt_data_vector = sim.compute_data_vector(sim.bnt_transform_kappa_maps(kappa_maps))
    print ('completed a realisation')

    return data_vector, bnt_data_vector


# --- Function to generate data vector at given parameter values and seed ---
def generate_derivative_sample(i, param, value, shift_type):
    seed_offset = 1000 * (0 if shift_type == "up" else 1) + i
    sim = ProcessMaps(
        z_array=z,
        nz_list=nz_list,
        n_eff_list=n_eff_list,
        sigma_eps_list=sigma_eps_list,
        baryon_feedback=baryon_feedback,
        sigma8=value if param == "sigma8" else sigma8,
        lognormal_shift=value if param == "lognormal_shift" else lognormal_shift,
        seed=seed + seed_offset,
        l_max=l_max,
        nside=nside,
        nslices=nslices
    )
    kappa = sim.generate_noisy_kappa_maps()
    data_vector = sim.compute_data_vector(kappa)
    bnt_data_vector = sim.compute_data_vector(sim.bnt_transform_kappa_maps(kappa))
    return data_vector, bnt_data_vector




if __name__ == "__main__":
    import multiprocessing
    from multiprocessing import Pool


    # --- Define parameter names and dummy derivative placeholders ---
    params_fid = {"sigma8": sigma8, "lognormal_shift": lognormal_shift}
    param_names = list(params_fid.keys())
     # --- Storage ---
    d_data_vector = {}
    d_bnt_data_vector = {}


    # Create partial function with fixed arguments
    worker = partial(
        generate_one_realization,
        z=z,
        nz_list=nz_list,
        n_eff_list=n_eff_list,
        sigma_eps_list=sigma_eps_list,
        baryon_feedback=baryon_feedback,
        sigma8=sigma8,
        lognormal_shift=lognormal_shift,
        seed=seed,
        l_max=l_max,
        nside=nside,
        nslices=nslices
    )

    # Run in parallel
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(worker, range(n_realizations))

    # Split results into two arrays
    mock_data_vectors, mock_bnt_data_vectors = zip(*results)
    mock_data_vectors = np.array(mock_data_vectors)
    mock_bnt_data_vectors = np.array(mock_bnt_data_vectors)

    print("Shapes:")
    print("  Data vectors:      ", mock_data_vectors.shape)
    print("  BNT data vectors:  ", mock_bnt_data_vectors.shape)

    # Convert to arrays for later use
    mock_data_vectors = np.array(mock_data_vectors)           # shape (100, 2 * nbins)
    mock_bnt_data_vectors = np.array(mock_bnt_data_vectors)   # shape (100, 2 * nbins)

    print("Shapes:")
    print("  Data vectors:      ", mock_data_vectors.shape)
    print("  BNT data vectors:  ", mock_bnt_data_vectors.shape)

    cov_data_vector = np.cov(mock_data_vectors, rowvar=False)
    cov_bnt_data_vector = np.cov(mock_bnt_data_vectors, rowvar=False)

    print("Covariance matrix shapes:")
    print("  Standard: ", cov_data_vector.shape)
    print("  BNT:      ", cov_bnt_data_vector.shape)



    for pname in param_names:
        print(f"Computing derivative for {pname}...")

        step = step_frac * params_fid[pname]
        p_up = params_fid[pname] + step
        p_dn = params_fid[pname] - step

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            up_worker = partial(generate_derivative_sample, param=pname, value=p_up, shift_type="up")
            dn_worker = partial(generate_derivative_sample, param=pname, value=p_dn, shift_type="down")
            results_up = pool.map(up_worker, range(n_realizations))
            results_dn = pool.map(dn_worker, range(n_realizations))

        # Split into standard and BNT data vectors
        up_std, up_bnt = zip(*results_up)
        dn_std, dn_bnt = zip(*results_dn)

        # Convert to arrays and compute finite difference
        up_std = np.mean(np.array(up_std), axis=0)
        dn_std = np.mean(np.array(dn_std), axis=0)
        up_bnt = np.mean(np.array(up_bnt), axis=0)
        dn_bnt = np.mean(np.array(dn_bnt), axis=0)

        d_data_vector[pname] = (up_std - dn_std) / (2 * step)
        d_bnt_data_vector[pname] = (up_bnt - dn_bnt) / (2 * step)

    print("Derivatives computed.")


    # --- Compute Fisher matrices ---
    inv_cov = np.linalg.inv(cov_data_vector)
    inv_cov_bnt = np.linalg.inv(cov_bnt_data_vector)

    F = np.zeros((len(param_names), len(param_names)))
    F_bnt = np.zeros_like(F)

    # Fill Fisher matrices using precomputed derivatives
    for i, pi in enumerate(param_names):
        for j, pj in enumerate(param_names):
            F[i, j] = np.dot(d_data_vector[pi], inv_cov @ d_data_vector[pj])
            F_bnt[i, j] = np.dot(d_bnt_data_vector[pi], inv_cov_bnt @ d_bnt_data_vector[pj])

    # --- Output results ---
    print("\nFisher matrix (standard):\n", F)
    print("\nFisher matrix (BNT):\n", F_bnt)

    # --- Save Fisher matrices ---
    np.save("data/fisher_standard.npy", F)
    np.save("data/fisher_bnt.npy", F_bnt)

    print("Fisher matrices saved to 'fisher_standard.npy' and 'fisher_bnt.npy'")

















