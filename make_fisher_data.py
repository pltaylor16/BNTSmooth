from bnt_smooth import LognormalWeakLensingSim
import numpy as np
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d


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


# --- Simulation settings ---
z = np.linspace(0.01, 2.5, 500)
nbins = 5
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0/nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.
sigma8 = 0.8
seed = 1234
l_max = 3000
nside = 2048
lognormal_shift = 1.0
nslices = 50

# --- Initialize simulation ---
sim = ProcessMaps(
    z_array = z,
    nz_list=nz_list,
    n_eff_list=n_eff_list,
    sigma_eps_list=sigma_eps_list,
    baryon_feedback=baryon_feedback,
    sigma8=sigma8,
    lognormal_shift=lognormal_shift,
    seed=seed,
    l_max=l_max,
    nside= nside,
    nslices=nslices
)

sim.generate_noisy_kappa_maps()
kappa_maps = sim.generate_noisy_kappa_maps()
bnt_kappa_maps = sim.bnt_transform_kappa_maps()
data_vector = sim.compute_data_vector(kappa_maps)
bnt_data_vector = sim.compute_data_vector(kappa_maps)

