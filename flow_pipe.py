import multiprocessing
import numpy as np
from functools import partial
from bnt_smooth import ProcessMaps
from tqdm import tqdm
from getdist import MCSamples, plots
from bnt_smooth import NzEuclid
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from utils import fit_to_data_weight, WeightedMaximumLikelihoodLoss
import random

# --- Simulation settings ---
nside = 16
l_max = 16
nslices = 5
n_train_per_round = 8
n_rounds = 3
n_cov_sim = 16
n_derivative_sim = 8
n_processes = 8


#nside = 512
#l_max = 1500
#nslices = 15
#n_train_per_round = 1000
#n_rounds = 3
#n_cov_sim = 200
#n_processes = 20

nbins = 5
n_samples = 5000


z = np.linspace(0.01, 2.5, 500)
Nz = NzEuclid(nbins = nbins, z=z)
nz_list = Nz.get_nz()
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins
baryon_feedback = 7.
seed = 1234


def worker(theta, use_bnt):
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



def compute_fisher_numerical(worker_fn, theta_fid, inv_cov, step_frac=0.05, n_avg=n_derivative_sim, n_processes=n_processes):
    import itertools

    theta_fid = np.array(theta_fid)
    ndim = len(theta_fid)
    steps = step_frac * np.abs(theta_fid)
    derivatives = []

    def simulate_theta(theta):
        with multiprocessing.Pool(n_processes) as pool:
            sims = pool.map(worker_fn, [theta] * n_avg)
        return np.mean(sims, axis=0)

    # Compute fiducial mean
    print("Computing fiducial mean for Fisher")
    x_fid = simulate_theta(theta_fid)

    # Compute derivatives
    for i in range(ndim):
        theta_plus = theta_fid.copy()
        theta_plus[i] += steps[i]
        theta_minus = theta_fid.copy()
        theta_minus[i] -= steps[i]

        print(f"Computing derivative wrt theta[{i}]")
        x_plus = simulate_theta(theta_plus)
        x_minus = simulate_theta(theta_minus)
        dx_dtheta = (x_plus - x_minus) / (2 * steps[i])
        derivatives.append(dx_dtheta)

    # Compute Fisher matrix
    F = np.zeros((ndim, ndim))
    for i, j in itertools.product(range(ndim), repeat=2):
        F[i, j] = derivatives[i] @ inv_cov @ derivatives[j]

    # Save the Fisher
    np.save('data/fisher.npy', F)

    return F


def main():

    bnt_tag = "bnt" if use_bnt else "nobnt"

    # create version of worker with use_bnt fixed
    worker_fn = partial(worker, use_bnt=use_bnt)

    #generate mock data vector
    with multiprocessing.Pool(1) as pool:
        x_obs_list = pool.map(worker_fn, [[1.0, 1.0]])
    x_obs = x_obs_list[0]

    #generate training data
    prior_min = [0.5, 0.5]
    prior_max = [1.5, 1.5]

    # --- Covariance estimation ---
    print("Running fiducial simulations for covariance...")
    fiducial_thetas = np.tile([[1.0, 1.0]], (n_cov_sim, 1))
    with multiprocessing.Pool(n_processes) as pool:
        x_fiducial = pool.map(worker_fn, fiducial_thetas)
    x_fiducial = np.stack(x_fiducial)
    cov = np.cov(x_fiducial.T)

    # Anderson-Hartlap correction
    n_sim = x_fiducial.shape[0]
    p = x_fiducial.shape[1]
    hartlap_factor = (n_sim - p - 2) / (n_sim - 1)
    if hartlap_factor <= 0:
        raise ValueError(f"Hartlap factor is non-positive: {hartlap_factor:.3f}. Increase number of simulations.")
    inv_cov = hartlap_factor * np.linalg.inv(cov)
    np.save(f"data/inv_cov_{bnt_tag}.npy", inv_cov)
    print(f"Saved inverse covariance matrix to data/inv_cov_{bnt_tag}.npy")


    for round_idx in range(n_rounds):
        with multiprocessing.Pool(1) as pool:
            print(f"\n--- Starting round {round_idx+1} ---")

            # Draw theta
            if round_idx == 0:
                # Step 1: Fisher-based proposal (2× broader)
                theta_fid = np.array([1.0, 1.0])
                F = compute_fisher_numerical(worker_fn, theta_fid, inv_cov)

                # Scale the covariance 2× wider than Fisher
                cov_fisher = np.linalg.inv(F)
                cov_proposal = 2.0 * cov_fisher

                # Draw initial training samples from the broader Gaussian
                theta_samples = np.random.multivariate_normal(theta_fid, cov_proposal, size=n_train_per_round)
            else:
                sample_idx = np.random.choice(posterior_samples.shape[0], size=n_train_per_round, replace=False)
                theta_samples = posterior_samples[sample_idx]

        # Step 2: Compute data vectors and log-likelihoods for reweighting
        print("Evaluating log-likelihoods at Fisher proposal samples...")

        with multiprocessing.Pool(n_processes) as pool:
            x_sims = pool.map(worker_fn, theta_samples)

        x_sims = np.stack(x_sims)
        log_weights = []

        for x_sim in x_sims:
            delta = x_obs - x_sim
            logL = -0.5 * delta @ inv_cov @ delta
            log_weights.append(logL)

        log_weights = np.array(log_weights)
        log_weights -= np.max(log_weights)  # for numerical stability
        weights = np.exp(log_weights)       # unnormalized posterior weights
        weights /= np.sum(weights)          # normalize     


        # Set up flow
        jax_seed = random.randint(0, 1e6)
        key = jax.random.PRNGKey(jax_seed)
        key, subkey = jax.random.split(key)
        flow = masked_autoregressive_flow(
            subkey,
            base_dist=Normal(jnp.zeros(theta_samples.shape[1])),
            transformer=RationalQuadraticSpline(knots=8, interval=4),
        )

        # Fit flow to weighted samples
        key, subkey = jax.random.split(key)
        flow, losses = fit_to_data_weight(
            weights=weights,
            key=subkey,
            dist=flow,
            x=theta_samples,
            learning_rate=1e-3,
            loss_fn=WeightedMaximumLikelihoodLoss(),
        )


        # --- Step 4: Draw from flow and reweight ---

        # Draw samples from the fitted flow
        key, subkey = jax.random.split(key)
        new_theta_samples_jax = flow.sample(seed=subkey, n=n_samples)
        posterior_samples = np.array(new_theta_samples_jax)

        np.save('posterior_samples_round{round_idx+1}.npy', posterior_samples)

        # Plot with GetDist
        names = ["alpha", "beta"]
        g = MCSamples(samples=posterior_samples, names=names, labels=names)
        gplt = plots.get_subplot_plotter()
        gplt.triangle_plot([g], filled=True)
        fname = f"data/posterior_triangle_{bnt_tag}_round{round_idx+1}.png"
        gplt.export(fname)
        print(f"Saved triangle plot to {fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SBI with or without BNT transform.")
    parser.add_argument("--use_bnt", action="store_true", help="Apply BNT transform if set.")
    args = parser.parse_args()
    use_bnt = args.use_bnt
    main()






