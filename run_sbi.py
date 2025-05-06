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


# --- Settings ---
n_rounds = 10
n_simulations_per_round = 500
n_samples = 5000
l_max = 1500
nside = 512
nslices = 15
nbins = 5
n_processes = 10
seed = 1234

z = np.linspace(0.01, 2.5, 500)
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.


def worker(theta, use_bnt=False):
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
    return sim.compute_data_vector(kappa_maps)


def main():
    from functools import partial

    # --- SBI settings ---
    prior_min = torch.tensor([0.5, 0.5])  # alpha, beta
    prior_max = torch.tensor([1.5, 1.5])
    prior = sbi_utils.BoxUniform(prior_min, prior_max)

    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")
    all_theta = []
    all_x = []

    for round_idx in range(n_rounds):
        print(f"\n--- Starting round {round_idx + 1} ---")

        # Sample θ
        theta_round = prior.sample((simulations_per_round,))
        theta_np = theta_round.numpy()
        all_theta.append(theta_round)

        # Run simulations in parallel
        worker_with_flag = partial(worker, use_bnt=use_bnt)
        with multiprocessing.Pool(processes=n_processes) as pool:
            x_data = pool.map(worker_with_flag, theta_np)
        all_x.append(torch.tensor(x_data, dtype=torch.float32))

        # Append and train
        inference = inference.append_simulations(torch.cat(all_theta), torch.cat(all_x))
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)

        # Save posterior samples and triangle plot
        x_obs = torch.tensor(worker([1.0, 1.0], use_bnt=use_bnt), dtype=torch.float32)
        samples = posterior.sample((n_samples,), x=x_obs).numpy()

        np.save(f"data/samples_round{round_idx+1}_{'bnt' if use_bnt else 'nobnt'}.npy", samples)

        g = MCSamples(samples=samples, names=["alpha", "beta"], labels=["alpha", "beta"])
        gplt = plots.get_subplot_plotter()
        gplt.triangle_plot([g], filled=True)
        fname = f"data/posterior_sequential_round{round_idx+1}_{'bnt' if use_bnt else 'nobnt'}.png"
        gplt.export(fname)
        print(f"Saved {fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bnt", action="store_true")
    args = parser.parse_args()
    use_bnt = args.use_bnt
    main()