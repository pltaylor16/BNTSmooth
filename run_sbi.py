import multiprocessing
import numpy as np
from multiprocessing import Process, Queue
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
        nz_i_obs /= np.trapz(nz_i_obs, z_grid)
        nz_bins.append((z_grid, nz_i_obs))

    return nz_bins, edges


def run_simulation(theta, queue, use_bnt, sim_args):
    alpha, beta = float(theta[0]), float(theta[1])
    try:
        sim = ProcessMaps(alpha=alpha, beta=beta, **sim_args)
        kappa_maps = sim.generate_noisy_kappa_maps()
        if use_bnt:
            kappa_maps = sim.bnt_transform_kappa_maps(kappa_maps)
        data_vector = sim.compute_data_vector(kappa_maps)
        queue.put(data_vector)
    except Exception as e:
        print(f"[Error] Simulation failed for theta={theta}: {e}")
        queue.put(None)


def main():
    # Settings
    nside = 512
    l_max = 1500
    nslices = 15
    nbins = 5
    n_samples = 500
    n_rounds = 5
    n_simulations_per_round = 100
    timeout = 180  # seconds

    z = np.linspace(0.01, 2.5, 500)
    nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
    n_eff_list = [30.0 / nbins] * nbins
    sigma_eps_list = [0.26] * nbins
    baryon_feedback = 7.

    sim_args = dict(
        z_array=z,
        nz_list=nz_list,
        n_eff_list=n_eff_list,
        sigma_eps_list=sigma_eps_list,
        baryon_feedback=baryon_feedback,
        l_max=l_max,
        nside=nside,
        nslices=nslices,
    )

    prior = sbi_utils.BoxUniform(torch.tensor([0.5, 0.5]), torch.tensor([1.5, 1.5]))
    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")

    theta_all, x_all = [], []

    # Fiducial point
    x_obs = torch.tensor(run_simulation([1.0, 1.0], Queue(), use_bnt, sim_args), dtype=torch.float32)

    for round_idx in range(n_rounds):
        print(f"\n--- Starting round {round_idx + 1} ---")

        # Draw samples
        if round_idx == 0:
            theta_round = prior.sample((n_simulations_per_round,))
        else:
            theta_round = posterior.sample((n_simulations_per_round,), x=x_obs)

        x_round = []
        valid_theta = []

        for i, theta in enumerate(theta_round.numpy()):
            queue = Queue()
            p = Process(target=run_simulation, args=(theta, queue, use_bnt, sim_args))
            p.start()
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                print(f"[Timeout] Simulation {i} timed out: theta = {theta}")
            else:
                result = queue.get()
                if result is not None:
                    x_round.append(torch.tensor(result, dtype=torch.float32))
                    valid_theta.append(torch.tensor(theta, dtype=torch.float32))
                else:
                    print(f"[Fail] Simulation {i} returned no result.")

        if len(x_round) == 0:
            print("[Warning] No valid simulations in this round.")
            continue

        x_valid = torch.stack(x_round)
        theta_valid = torch.stack(valid_theta)
        x_all.append(x_valid)
        theta_all.append(theta_valid)

        x_concat = torch.cat(x_all)
        theta_concat = torch.cat(theta_all)


        density_estimator = inference.append_simulations(theta_concat, x_concat).train()
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((n_samples,), x=x_obs)

        # Save triangle plot
        from getdist import MCSamples, plots
        g = MCSamples(samples=samples.numpy(), names=["alpha", "beta"], labels=["alpha", "beta"])
        gplt = plots.get_subplot_plotter()
        gplt.triangle_plot([g], filled=True)
        suffix = "bnt" if use_bnt else "nobnt"
        fname = f"data/posterior_sequential_{suffix}_round{round_idx + 1}.png"
        gplt.export(fname)
        print(f"[Saved] Posterior plot → {fname}")

        # Save simulation data
        torch.save(theta_concat, f"data/{suffix}_theta_all_round{round_idx+1}.pt")
        torch.save(x_concat, f"data/{suffix}_x_all_round{round_idx+1}.pt")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bnt", action="store_true")
    args = parser.parse_args()
    use_bnt = args.use_bnt
    main()