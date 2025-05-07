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
nside = 16
l_max = 16
nslices = 5
n_train_per_round = 10
n_rounds = 3
n_cov_sim = 100
use_bnt = False

#nside = 512
#l_max = 1500
#nslices = 15
#n_train_per_round = 200
#n_rounds = 3
#n_cov_sim = 200

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


def train_emulator(theta_train, x_train, n_epochs=1000, lr=1e-3, patience=20):
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import torch.nn.functional as F

    # Split into training and validation sets
    dataset = TensorDataset(theta_train, x_train)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    # Define a small neural network
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch_theta, batch_x in train_loader:
            pred = model(batch_theta)
            loss = F.mse_loss(pred, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(F.mse_loss(model(t), x) for t, x in val_loader) / len(val_loader)

        print(f"Epoch {epoch+1} — Val Loss: {val_loss:.4e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    # Load best model state
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


def main():

    #generate mock data vector
    with multiprocessing.Pool(1) as pool:
        x_obs_list = pool.map(worker, [[1.0, 1.0]])
    x_obs = torch.tensor(x_obs_list[0], dtype=torch.float32)

    #generate training data
    prior_min = [0.5, 0.5]
    prior_max = [1.5, 1.5]



    # --- Covariance estimation ---
    print("Running fiducial simulations for covariance...")
    fiducial_thetas = np.tile([[1.0, 1.0]], (n_cov_sim, 1))
    with multiprocessing.Pool(n_processes) as pool:
        x_fiducial = pool.map(worker, fiducial_thetas)
    x_fiducial = np.stack(x_fiducial)
    cov = np.cov(x_fiducial.T)
    inv_cov = np.linalg.inv(cov)




    theta_train_all = []
    x_train_all = []

    for round_idx in range(n_rounds):
        print(f"\n--- Starting round {round_idx+1} ---")

        # Draw theta
        if round_idx == 0:
            theta_samples = np.random.uniform(low=prior_min, high=prior_max, size=(n_train_per_round, 2))
        else:
            sample_idx = np.random.choice(posterior_samples.shape[0], size=n_train_per_round, replace=False)
            theta_samples = posterior_samples[sample_idx]

        # Simulate
        with multiprocessing.Pool(processes=n_processes) as pool:
            x_train = pool.map(worker, theta_samples)

        theta_train = torch.tensor(theta_samples, dtype=torch.float32)
        x_train = torch.tensor(x_train, dtype=torch.float32)

        # Accumulate data
        theta_train_all.append(theta_train)
        x_train_all.append(x_train)

        theta_concat = torch.cat(theta_train_all)
        x_concat = torch.cat(x_train_all)

        # Train emulator
        with multiprocessing.Pool(1) as pool:
            model = train_emulator(theta_concat, x_concat)
            torch.save(model.state_dict(), f"data/emulator_round{round_idx+1}.pt")

            # MCMC
            ndim = 2
            nwalkers = 20
            nsteps = 3000
            initial_pos = [1.0, 1.0] + 1e-2 * np.random.randn(nwalkers, ndim)

            print("Running MCMC...")
            logpost = LogPosteriorEvaluator(model, x_obs, inv_cov)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost)
            sampler.run_mcmc(initial_pos, nsteps, progress=True)

            posterior_samples = sampler.get_chain(discard=500, thin=10, flat=True)
            np.save(f"data/emcee_samples_round{round_idx+1}.npy", posterior_samples)
            print(f"Saved: data/emcee_samples_round{round_idx+1}.npy")

            # Plot with GetDist
            names = ["alpha", "beta"]
            g = MCSamples(samples=posterior_samples, names=names, labels=names)
            gplt = plots.get_subplot_plotter()
            gplt.triangle_plot([g], filled=True)
            fname = f"data/posterior_triangle_round{round_idx+1}.png"
            gplt.export(fname)
            print(f"Saved triangle plot to {fname}")


if __name__ == "__main__":
    main()

