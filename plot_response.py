import numpy as np
import matplotlib.pyplot as plt
from bnt_smooth import ProcessMaps  # Updated class assumed to be in bnt_smooth.py

# Define parent n(z)
def parent_nz(z):
    z_euc = 0.9 / 2 ** 0.5
    return (z / z_euc)**2 * np.exp(-(z / z_euc)**1.5)

# Binning utility
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

# Simulation settings
z = np.linspace(0.01, 2.5, 500)
nbins = 5
nz_list, _ = make_equal_ngal_bins(parent_nz, z, nbins=nbins)
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins

common_kwargs = dict(
    z_array=z,
    nz_list=nz_list,
    n_eff_list=n_eff_list,
    sigma_eps_list=sigma_eps_list,
    baryon_feedback=7.0,
    l_max=512,
    nside=256,
    nslices=40,
    seed=1234
)

# Run for two alpha values (beta fixed)
alpha_data = {}
for alpha in [0.5, 1.5]:
    sim = ProcessMaps(alpha=alpha, beta=1.0, **common_kwargs)
    maps = sim.generate_noisy_kappa_maps()
    dv = sim.compute_data_vector(maps)
    alpha_data[alpha] = dv

plt.figure(figsize=(10, 5))
for alpha, vec in alpha_data.items():
    plt.plot(vec, label=f"alpha = {alpha}")
plt.xlabel("Data Vector Index")
plt.ylabel("Moment Value")
plt.title("κ² and κ³ Moments for Different α")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/alpha.png")
print("Saved alpha plot to data/alpha.png")

# Run for two beta values (alpha fixed)
beta_data = {}
for beta in [0.5, 1.5]:
    sim = ProcessMaps(alpha=1.0, beta=beta, **common_kwargs)
    maps = sim.generate_noisy_kappa_maps()
    dv = sim.compute_data_vector(maps)
    beta_data[beta] = dv

plt.figure(figsize=(10, 5))
for beta, vec in beta_data.items():
    plt.plot(vec, label=f"beta = {beta}")
plt.xlabel("Data Vector Index")
plt.ylabel("Moment Value")
plt.title("κ² and κ³ Moments for Different β")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/beta.png")
print("Saved beta plot to data/beta.png")