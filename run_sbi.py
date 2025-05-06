import multiprocessing
import numpy as np
from functools import partial
from bnt_smooth import ProcessMaps
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference
import torch
from getdist import MCSamples, plots


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
                       np.concatenate([[z_grid[0]], z_grid, [z_grid[-1]]]))
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
n_simulations = 5000
l_max = 1500
nside = 512
nslices = 15
nbins = 5
n_processes = 10
n_samples = 5000

z = np.linspace(0.01, 2.5, 500)
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.
seed = 1234


def worker(theta):
    alpha, beta = float(theta[0]), float(theta[1])
    print(f"Running simulation with alpha = {alpha:.3f}, beta = {beta:.3f}")

    sim = ProcessMaps(
        z_array=z,
        nz_list=nz_list,
        n_eff_list=n_eff_list,
        sigma_eps_list=sigma_eps_list,
        baryon_feedback=baryon_feedback,
        alpha=alpha,
        beta=beta,
        seed=np.random.randint(1e6),
        l_max=l_max,
        nside=nside,
        nslices=nslices
    )

    kappa_maps = sim.generate_noisy_kappa_maps()
    if use_bnt:
        kappa_maps = sim.bnt_transform_kappa_maps(kappa_maps)
    data_vector = sim.compute_data_vector(kappa_maps)
    return data_vector


def main():
    # --- SBI settings ---
    prior_min = torch.tensor([0.75, 0.75])  # alpha, beta
    prior_max = torch.tensor([1.25, 1.25])
    prior = sbi_utils.BoxUniform(prior_min, prior_max)

    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")

    theta_samples = prior.sample((n_simulations,))
    theta_np = theta_samples.numpy()

    print("Starting parallel simulations...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        x_data = pool.map(worker, theta_np)
    print("All simulations complete.\n")

    if use_bnt:
        np.save("data/data_vector_bnt.npy", x_data)
    else:
        np.save("data/data_vector.npy", x_data)

    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    # Use fiducial point for observation
    x_obs = torch.tensor(worker(theta=[1.0, 1.0]), dtype=torch.float32)
    theta_train = theta_samples[1:]
    x_train = x_tensor[1:]

    print("Starting SBI training...")
    density_estimator = inference.append_simulations(theta_train, x_train).train()
    posterior = inference.build_posterior(density_estimator)
    print("SBI training complete.\n")

    samples = posterior.sample((n_samples,), x=x_obs)
    if use_bnt:
        np.save("data/samples_bnt.npy", samples)
        np.save("data/x_obs_bnt.npy", x_obs)
    else:
        np.save("data/samples.npy", samples)
        np.save("data/x_obs.npy", x_obs)

    # Subset training
    print("Starting SBI training with only 100 simulations...")
    inference_100 = sbi_inference.SNPE(prior=prior, density_estimator="maf")
    density_estimator_100 = inference_100.append_simulations(theta_train[:3000], x_train[:3000]).train()
    posterior_100 = inference_100.build_posterior(density_estimator_100)
    samples_100 = posterior_100.sample((n_samples,), x=x_obs)

    # GetDist plot
    param_names = ["alpha", "beta"]
    g_all = MCSamples(samples=samples.numpy(), names=param_names, labels=param_names)
    g_100 = MCSamples(samples=samples_100.numpy(), names=param_names, labels=param_names)

    gplt = plots.get_subplot_plotter()
    gplt.triangle_plot([g_all, g_100], filled=True, legend_labels=["All", "Subset"])
    if use_bnt:
        gplt.export("data/posterior_comparison_bnt.png")
    else:
        gplt.export("data/posterior_comparison.png")

    print("Saved triangle plot to data/posterior_comparison.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SBI with or without BNT transform.")
    parser.add_argument("--use_bnt", action="store_true", help="Apply BNT transform if set.")
    args = parser.parse_args()
    use_bnt = args.use_bnt
    main()