import concurrent.futures
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
nside = 512
l_max = 1500
nslices = 15
nbins = 5
n_samples = 500
n_processes = 10
n_rounds = 5
n_simulations_per_round = 100
timeout = 180 # timeout in sec

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
    prior_min = torch.tensor([0.5, 0.5])  # alpha, beta
    prior_max = torch.tensor([1.5, 1.5])
    prior = sbi_utils.BoxUniform(prior_min, prior_max)

    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")

    theta_all = []
    x_all = []

    # Fiducial simulation (alpha=1.0, beta=1.0)
    print("Computing fiducial observation (x_obs)...")
    x_obs = torch.tensor(worker([1.0, 1.0]), dtype=torch.float32)

    for round_idx in range(n_rounds):
        print(f"\n--- Starting round {round_idx + 1} ---")

        # Sample theta
        if round_idx == 0:
            theta_round = prior.sample((n_simulations_per_round,))
        else:
            theta_round = posterior.sample((n_simulations_per_round,), x=x_obs)
        theta_np = theta_round.numpy()

        # Simulate
        x_round = []
        theta_valid = []
        print("Running simulations with individual timeouts...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = {executor.submit(worker, theta): theta for theta in theta_np}

            for i, (future, theta) in enumerate(zip(futures.keys(), futures.values())):
                try:
                    result = future.result(timeout=timeout)
                    x_round.append(result)
                    theta_valid.append(theta)
                except concurrent.futures.TimeoutError:
                    print(f"[Warning] Simulation {i} with theta={theta} timed out and was skipped.")
                except Exception as e:
                    print(f"[Error] Simulation {i} with theta={theta} failed: {e}")

        # Filter valid results (drop any failed ones)
        valid_pairs = [(theta, x) for theta, x in zip(theta_round, x_round) if x is not None]

        if len(valid_pairs) == 0:
            print(f"[Round {round_idx+1}] Warning: no valid simulations, skipping training.")
            continue

        # Split valid thetas and xs
        theta_valid = torch.stack([torch.tensor(pair[0], dtype=torch.float32) for pair in valid_pairs])
        x_valid = torch.stack([torch.tensor(pair[1], dtype=torch.float32) for pair in valid_pairs])

        # Append to full training set
        theta_all.append(theta_valid)
        x_all.append(x_valid)

        # Concatenate all so far
        theta_concat = torch.cat(theta_all)
        x_concat = torch.cat(x_all)

        # Train and build posterior
        density_estimator = inference.append_simulations(theta_concat, x_concat).train()
        posterior = inference.build_posterior(density_estimator)

        # Evaluate posterior at fiducial observation
        samples = posterior.sample((n_samples,), x=x_obs)

        # Save concatenated data so far
        prefix = "bnt_" if use_bnt else "nobnt_"
        theta_path = f"data/{prefix}theta_all_round{round_idx+1}.pt"
        x_path = f"data/{prefix}x_all_round{round_idx+1}.pt"
        torch.save(theta_concat, theta_path)
        torch.save(x_concat, x_path)
        print(f"Saved simulation data: {theta_path}, {x_path}")

        # Save triangle plot
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