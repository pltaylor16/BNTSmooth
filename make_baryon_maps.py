import multiprocessing
import numpy as np
import os
from pathlib import Path
from functools import partial
from bnt_smooth import ProcessMaps, NzEuclid

# --- Simulation settings ---
nside = 512
l_max = 1500
nslices = 15
n_sims = 500
n_processes = 20
nbins = 5

#nside = 8
#l_max = 8
#nslices = 3
#n_sims = 4
#n_processes = 4
#nbins = 2

z = np.linspace(0.01, 2.5, 500)
Nz = NzEuclid(nbins=nbins, z=z)
nz_list = Nz.get_nz()
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins

# --- Worker function ---
def worker(args):
    theta, save_dir, sim_idx, baryon_feedback = args
    alpha, beta = float(theta[0]), float(theta[1])
    print(f"[sim {sim_idx}] alpha={alpha:.3f}, beta={beta:.3f}, feedback={baryon_feedback}")

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
        nslices=nslices,
        baryon_smooth_mpc=0.3,
    )

    kappa_maps = sim.generate_noisy_kappa_maps()
    out_path = os.path.join(save_dir, f"kappa_maps_{sim_idx:04d}.npz")
    np.savez_compressed(out_path, **{f"slice{i}": m for i, m in enumerate(kappa_maps)})
    return out_path

# --- Main execution ---
def main():
    theta_fid = [1.0, 1.0]
    delta = 0.05  # 5% step
    thetas_to_run = {
        "fiducial": theta_fid,
        "alpha_plus": [1.0 + delta, 1.0],
        "alpha_minus": [1.0 - delta, 1.0],
        "beta_plus": [1.0, 1.0 + delta],
        "beta_minus": [1.0, 1.0 - delta]
    }

    for baryon_feedback in [7.0]:
        for label, theta in thetas_to_run.items():
            save_dir = f"/srv/scratch2/taylor.4264/BNTSmooth_data/maps_baryon/maps_{label}_b{int(baryon_feedback)}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            args_list = [
                (theta, save_dir, i, baryon_feedback)
                for i in range(n_sims)
            ]

            print(f"\nLaunching {label} simulations")
            with multiprocessing.Pool(n_processes) as pool:
                pool.map(worker, args_list)

            print(f"Finished saving {n_sims} maps to {save_dir}/")

if __name__ == "__main__":
    main()