import multiprocessing
import numpy as np
from functools import partial
from bnt_smooth import ProcessMaps
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference
import torch


def parent_nz(z):
    z_euc = 0.9 / 2 ** 0.5
    return (z / z_euc)**2 * np.exp(-(z / z_euc)**1.5)


def make_equal_ngal_bins(nz_func, z_grid, nbins, sigma_z0=0.05):
    from scipy.special import erf
    from scipy.interpolate import interp1d

    nz_parent = nz_func(z_grid)
    nz_parent /= np.trapz(nz_parent, z_grid)

    cdf = np.cumsum(nz_parent)
    cdf /= cdf[-1]
    inv_cdf = interp1d(np.concatenate([[0], cdf, [1]]),
                       np.concatenate([[z_grid[0]], z, [z_grid[-1]]]))
    edges = inv_cdf(np.linspace(0, 1, nbins + 1))

    nz_bins = []
    sigma_of_z = lambda zz: sigma_z0 * (1.0 + zz)

    for i in range(nbins):
        z_lo, z_hi = edges[i], edges[i + 1]
        sig = sigma_of_z(z_grid)
        kernel = 0.5 * (erf((z_hi - z_grid)/(np.sqrt(2)*sig)) -
                        erf((z_lo - z_grid)/(np.sqrt(2)*sig)))
        nz_i_obs = nz_parent * kernel
        area = np.trapz(nz_i_obs, z_grid)
        if area > 0:
            nz_i_obs /= area
        nz_bins.append((z_grid, nz_i_obs))

    return nz_bins, edges


# --- Simulation settings ---
z = np.linspace(0.01, 2.5, 500)
nbins = 3
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.
l_max = 32
nside = 32
nslices = 5
seed = 1234


def worker(theta):
    sigma8_val, lognormal_shift_val = float(theta[0]), float(theta[1])
    print(f"Running simulation with sigma8 = {sigma8_val:.3f}, lognormal_shift = {lognormal_shift_val:.3f}")

    sim = ProcessMaps(
        z_array=z,
        nz_list=nz_list,
        n_eff_list=n_eff_list,
        sigma_eps_list=sigma_eps_list,
        baryon_feedback=baryon_feedback,
        sigma8=sigma8_val,
        lognormal_shift=lognormal_shift_val,
        seed=np.random.randint(1e6),
        l_max=l_max,
        nside=nside,
        nslices=nslices
    )

    kappa_maps = sim.generate_noisy_kappa_maps()
    data_vector = sim.compute_data_vector(kappa_maps)

    return data_vector


def main():
    # --- SBI settings ---
    n_simulations = 100
    prior_min = torch.tensor([0.6, 0.5])  # sigma8, lognormal_shift
    prior_max = torch.tensor([1.0, 1.5])
    prior = sbi_utils.BoxUniform(prior_min, prior_max)

    # Set up inference object
    inference = sbi_inference.SNPE(prior=prior, density_estimator="mdn")

    # --- Sample θ values ---
    theta_samples = prior.sample((n_simulations,))
    theta_np = theta_samples.numpy()

    print("Starting parallel simulations...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        x_data = pool.map(worker, theta_np)
    print("All simulations complete.\n")

    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    # --- Train SBI posterior ---
    print("Starting SBI training...")
    density_estimator = inference.append_simulations(theta_samples, x_tensor).train()
    print ('I made it this far.....')
    posterior = inference.build_posterior(density_estimator)
    print("SBI training complete.\n")

    # --- Example inference ---
    x_obs = x_tensor[0]  # Treat the first simulated example as observation
    samples = posterior.sample((100,), x=x_obs)
    print("Posterior sample shape:", samples.shape)


if __name__ == "__main__":
    main()