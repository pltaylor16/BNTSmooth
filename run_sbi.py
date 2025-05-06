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
#nside = 512
#l_max = 1500
#nslices = 15
nside = 32
l_max = 30
nslices = 5
n_rounds = 5
n_simulations_per_round = 10

nbins = 5
n_samples = 5000
n_processes = 10


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


def train_density_estimator(theta, x, prior, x_obs, n_samples):
    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    samples = posterior.sample((n_samples,), x=x_obs)
    return posterior, samples


def main():
    prior_min = torch.tensor([0.5, 0.5])  # alpha, beta
    prior_max = torch.tensor([1.5, 1.5])
    prior = sbi_utils.BoxUniform(prior_min, prior_max)

    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")

    theta_all = []
    x_all = []

    #must use multiprocessing to produce data o.w. crash
    with multiprocessing.Pool(1) as pool:
        x_obs_list = pool.map(worker, [[1.0, 1.0]])
    x_obs = torch.tensor(x_obs_list[0], dtype=torch.float32)

    for round_idx in range(n_rounds):

        #just put everyting in multiprocessing to play it safe
        with multiprocessing.Pool(1) as pool:
            print(f"\n--- Starting round {round_idx + 1} ---")

            # Sample theta
            if round_idx == 0:
                theta_round = prior.sample((n_simulations_per_round,))
            else:
                sample_idx = np.random.choice(len(samples), size=n_simulations_per_round, replace=False)
                theta_round = samples[sample_idx]
            theta_np = theta_round.numpy()

        # Simulate
        with multiprocessing.Pool(processes=n_processes) as pool:
            x_round = pool.map(worker, theta_np)

        # Append to all data
        theta_all.append(theta_round)
        x_all.append(torch.tensor(x_round, dtype=torch.float32))

        theta_concat = torch.cat(theta_all)
        x_concat = torch.cat(x_all)

        with multiprocessing.Pool(1) as pool:
            results = pool.starmap(
                train_density_estimator,
                [(theta_concat, x_concat, prior, x_obs, n_samples)]
            )
            posterior, samples = results[0]

        # Plot
        param_names = ["alpha", "beta"]
        g = MCSamples(samples=samples.numpy(), names=param_names, labels=param_names)

        gplt = plots.get_subplot_plotter()
        gplt.triangle_plot([g], filled=True)

        fname = f"data/posterior_sequential_bnt_round{round_idx+1}.png" if use_bnt else f"data/posterior_sequential_round{round_idx+1}.png"
        gplt.export(fname)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SBI with or without BNT transform.")
    parser.add_argument("--use_bnt", action="store_true", help="Apply BNT transform if set.")
    args = parser.parse_args()
    use_bnt = args.use_bnt
    main()

