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
#l_max = 3000
#nside = 512
#nslices = 50
#nbins = 5
#n_processes = 10


l_max = 16
nside = 16
nslices = 5
nbins = 3
n_processes = 20


z = np.linspace(0.01, 2.5, 500)
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.
seed = 1234
n_samples = 5000
n_simulations = 200



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
    if use_bnt == False:
        data_vector = sim.compute_data_vector(kappa_maps)
    elif use_bnt == True:
        kappa_maps = sim.bnt_transform_kappa_maps(kappa_maps)
        data_vector = sim.compute_data_vector(kappa_maps)


    return data_vector


def main():
    # --- SBI settings ---
    prior_min = torch.tensor([0.6, 0.7])  # sigma8, lognormal_shift
    prior_max = torch.tensor([0.8, 1.3])
    prior = sbi_utils.BoxUniform(prior_min, prior_max)

    # Set up inference object
    inference = sbi_inference.SNPE(prior=prior, density_estimator="maf")

    # --- Sample θ values ---
    theta_samples = prior.sample((n_simulations,))
    theta_np = theta_samples.numpy()

    print("Starting parallel simulations...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        x_data = pool.map(worker, theta_np)
    print("All simulations complete.\n")

    # Save the data vector
    if use_bnt == False:
        np.save("data/data_vector.npy", x_data)
    elif use_bnt == True:
        np.save("data/data_vector_bnt.npy", x_data)
    print("Saved data_vector to data/data_vector.npy")

    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    # --- Use the first simulated point as observation and the rest for training ---
    x_obs = x_tensor[0]
    theta_train = theta_samples[1:]
    x_train = x_tensor[1:]

    # --- Train SBI posterior ---
    print("Starting SBI training...")
    density_estimator = inference.append_simulations(theta_train, x_train).train()
    posterior = inference.build_posterior(density_estimator)
    print("SBI training complete.\n")

    # --- Sample from posterior conditioned on the held-out observation ---
    samples = posterior.sample((n_samples,), x=x_obs)
    print("Posterior sample shape:", samples.shape)
    if use_bnt == False:
        np.save("data/samples.npy", samples)
        np.save("data/x_obs.npy", x_obs)

    elif use_bnt == True:
        np.save("data/samples_bnt.npy", samples)
        np.save("data/x_obs_bnt.npy", x_obs)

    # SBI with only first 100 simulations
    print("Starting SBI training with only 100 simulations...")
    inference_100 = sbi_inference.SNPE(prior=prior, density_estimator="mdn")
    density_estimator_100 = inference_100.append_simulations(theta_train[:100], x_train[:100]).train()
    posterior_100 = inference_100.build_posterior(density_estimator_100)
    samples_100 = posterior_100.sample((n_samples,), x=x_obs)


    # --- GetDist comparison ---
    param_names = ["sigma8", "lognormal_shift"]
    g_all = MCSamples(samples=samples.numpy(), names=param_names, labels=param_names)
    g_100 = MCSamples(samples=samples_100.numpy(), names=param_names, labels=param_names)

    gplt = plots.get_subplot_plotter()
    gplt.triangle_plot([g_all, g_100], filled=True, legend_labels=["All (200)", "Subset (100)"])
    if use_bnt == True:
        gplt.export("data/posterior_comparison_bnt.png")
    else:
        gplt.export("data/posterior_comparison.png")

    print("Saved triangle plot to data/posterior_comparison.png")





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SBI with or without BNT transform.")
    parser.add_argument("--use_bnt", action="store_true", help="Apply BNT transform if set.")
    args = parser.parse_args()

    use_bnt = args.use_bnt  # dynamically set global variable python run_sbi.py --use_bnt to run with bnt
    main()