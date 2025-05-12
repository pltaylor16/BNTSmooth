import multiprocessing
import numpy as np
from functools import partial
from bnt_smooth import ProcessMaps
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import emcee
import pickle
from tqdm import tqdm
from getdist import MCSamples, plots
from bnt_smooth import NzEuclid

# --- Simulation settings ---
nside = 512
l_max = 1500
nslices = 15
n_train_per_round = 1000
n_rounds = 3
n_cov_sim = 300
n_derivative_sim = 100
n_processes = 20


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


class Emulator(nn.Module):
    def __init__(self, input_dim=2, output_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, theta):
        return self.net(theta)


def train_emulator(theta_train, x_train, inv_cov, n_epochs=1000, lr=1e-3, patience=20):
    from torch.utils.data import TensorDataset, DataLoader, random_split

    # Convert inverse covariance matrix to a tensor
    inv_cov_tensor = torch.tensor(inv_cov, dtype=torch.float32)

    # Split into training and validation sets
    dataset = TensorDataset(theta_train, x_train)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    # Define neural network model
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def chi2_loss(pred, target):
        diff = pred - target
        return torch.einsum("bi,ij,bj->b", diff, inv_cov_tensor, diff).mean()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch_theta, batch_x in train_loader:
            pred = model(batch_theta)
            loss = chi2_loss(pred, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(chi2_loss(model(t), x) for t, x in val_loader) / len(val_loader)

        print(f"Epoch {epoch+1} — Val χ²: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

# --- Log-likelihood using emulator ---
class LogPosteriorEvaluator:
    def __init__(self, model, x_obs, inv_cov):
        self.model = model
        self.x_obs = x_obs
        self.inv_cov = inv_cov

    def __call__(self, theta):
        if not (0.5 <= theta[0] <= 1.5 and 0.5 <= theta[1] <= 1.5):
            return -np.inf
        theta_tensor = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(theta_tensor).numpy().flatten()
        delta = self.x_obs.numpy() - pred
        return -0.5 * delta @ self.inv_cov @ delta

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
    torch.set_num_threads(1)

    bnt_tag = "bnt" if use_bnt else "nobnt"

    # create version of worker with use_bnt fixed
    worker_fn = partial(worker, use_bnt=use_bnt)

    #generate mock data vector
    with multiprocessing.Pool(1) as pool:
        x_obs_list = pool.map(worker_fn, [[1.0, 1.0]])
    x_obs = torch.tensor(x_obs_list[0], dtype=torch.float32)

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

    theta_train_all = []
    x_train_all = []

    for round_idx in range(n_rounds):
        with multiprocessing.Pool(1) as pool:
            print(f"\n--- Starting round {round_idx+1} ---")

            # Draw theta
            if round_idx == 0:
                theta_fid = [1.0, 1.0]
                F = compute_fisher_numerical(worker_fn, theta_fid, inv_cov)
                theta_samples = np.random.multivariate_normal(theta_fid, np.linalg.inv(F), size=n_train_per_round)
            else:
                sample_idx = np.random.choice(posterior_samples.shape[0], size=n_train_per_round, replace=False)
                theta_samples = posterior_samples[sample_idx]

        # Simulate
        with multiprocessing.Pool(processes=n_processes) as pool:
            x_train = pool.map(worker_fn, theta_samples)

        #accumalate data
        theta_train = torch.tensor(theta_samples, dtype=torch.float32)
        x_train = torch.tensor(x_train, dtype=torch.float32)

        # Accumulate data
        theta_train_all.append(theta_train)
        x_train_all.append(x_train)

        theta_concat = torch.cat(theta_train_all)
        x_concat = torch.cat(x_train_all)

        # Train emulator
        with multiprocessing.Pool(1) as pool:
            model = train_emulator(theta_concat, x_concat, inv_cov)
            torch.save(model.state_dict(), f"data/emulator_{bnt_tag}_round{round_idx+1}.pt")
            
        # MCMC
        ndim = 2
        nwalkers = 20
        nsteps = n_samples
        initial_pos = [1.0, 1.0] + 1e-2 * np.random.randn(nwalkers, ndim)

        print("Running MCMC...")
        logpost = LogPosteriorEvaluator(model, x_obs, inv_cov)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

        posterior_samples = sampler.get_chain(discard=500, thin=10, flat=True)
        np.save(f"data/emcee_samples_{bnt_tag}_round{round_idx+1}.npy", posterior_samples)
        print(f"Saved: data/emcee_samples_{bnt_tag}_round{round_idx+1}.npy")

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



