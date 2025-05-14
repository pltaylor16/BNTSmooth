import numpy as np
import os
from pathlib import Path
import multiprocessing
from functools import partial
from bnt_smooth import ProcessMaps, NzEuclid

# --- Configuration ---
nside = 512
l_max = 1500
nslices = 5
nbins = 5
n_processes = 40
step_frac = 0.05
delta = step_frac
theta_fid = np.array([1.0, 1.0])
z = np.linspace(0.01, 2.5, 500)
fwhm_arcmin = 10. #arcmin

# --- Survey specification ---
Nz = NzEuclid(nbins=nbins, z=z)
nz_list = Nz.get_nz()
n_eff_list = [30.0 / nbins] * nbins
sigma_eps_list = [0.26] * nbins

# --- Simulation template ---
sim = ProcessMaps(
    z_array=z,
    nz_list=nz_list,
    n_eff_list=n_eff_list,
    sigma_eps_list=sigma_eps_list,
    baryon_feedback=None,  # Set later
    alpha=1.0,
    beta=1.0,
    seed=42,
    l_max=l_max,
    nside=nside,
    nslices=nslices
)
sim.set_cosmo()

# --- Map loading + data vector computation ---
def compute_dvec_from_file(path, baryon_feedback, use_bnt=False, naive=False):
    fname = os.path.basename(path)
    print(f"[{baryon_feedback=}, {use_bnt=}, {naive=}] Processing: {fname}", flush=True)

    loaded = np.load(path)
    kappa_maps = [loaded[f"slice{i}"] for i in range(nslices)]

    if use_bnt:
        if naive:
            kappa_maps = sim.bnt_transform_kappa_maps(kappa_maps)
            kappa_maps = sim.smooth_kappa_maps(kappa_maps, fwhm_arcmin)
        else:
            kappa_maps = sim.bnt_transform_kappa_maps(kappa_maps)
            kappa_maps = sim.smooth_kappa_maps(kappa_maps, fwhm_arcmin)
            kappa_maps = sim.inverse_bnt_transform_kappa_maps(kappa_maps)
    else:
        kappa_maps = sim.smooth_kappa_maps(kappa_maps, fwhm_arcmin)

    return sim.compute_data_vector(kappa_maps)

# --- Helper: compute Fisher matrix ---
def compute_fisher_matrix(base_dir, baryon_feedback, use_bnt=False, naive=False):
    print(f"\n=== Computing Fisher for baryon_feedback={baryon_feedback} ===")

    def get_paths(label):
        return sorted(Path(f"{base_dir}_{label}_b{int(baryon_feedback)}").glob("*.npz"))

    # Load all map paths
    paths_fid = get_paths("fiducial")
    paths_aplus = get_paths("alpha_plus")
    paths_aminus = get_paths("alpha_minus")
    paths_bplus = get_paths("beta_plus")
    paths_bminus = get_paths("beta_minus")

    # Parallel data vector computation
    with multiprocessing.Pool(n_processes) as pool:
        func = partial(compute_dvec_from_file, baryon_feedback=baryon_feedback, use_bnt=use_bnt, naive=naive)
        dvecs_fid    = pool.map(func, paths_fid)
        dvecs_aplus  = pool.map(func, paths_aplus)
        dvecs_aminus = pool.map(func, paths_aminus)
        dvecs_bplus  = pool.map(func, paths_bplus)
        dvecs_bminus = pool.map(func, paths_bminus)


    dvecs_fid = np.stack(dvecs_fid)
    dvecs_aplus = np.stack(dvecs_aplus)
    dvecs_aminus = np.stack(dvecs_aminus)
    dvecs_bplus = np.stack(dvecs_bplus)
    dvecs_bminus = np.stack(dvecs_bminus)

    # --- Mean and derivatives ---
    mu_fid = np.mean(dvecs_fid, axis=0)
    dmu_dalpha = (np.mean(dvecs_aplus, axis=0) - np.mean(dvecs_aminus, axis=0)) / (2 * delta)
    dmu_dbeta  = (np.mean(dvecs_bplus, axis=0) - np.mean(dvecs_bminus, axis=0)) / (2 * delta)

    # --- Covariance matrix + Hartlap correction ---
    N = dvecs_fid.shape[0]
    p = dvecs_fid.shape[1]
    cov = np.cov(dvecs_fid.T)
    hartlap = (N - p - 2) / (N - 1)
    if hartlap <= 0:
        raise ValueError(f"Hartlap factor non-positive: {hartlap}")
    inv_cov = hartlap * np.linalg.inv(cov)

    # --- Fisher matrix ---
    fisher = np.zeros((2, 2))
    fisher[0, 0] = dmu_dalpha @ inv_cov @ dmu_dalpha
    fisher[0, 1] = dmu_dalpha @ inv_cov @ dmu_dbeta
    fisher[1, 0] = fisher[0, 1]
    fisher[1, 1] = dmu_dbeta @ inv_cov @ dmu_dbeta

    # Save
    bnt_tag = "bnt" if use_bnt else "nobnt"
    naive_tag = "naive" if naive else "not naive"
    fname = f"results/fisher_b{int(baryon_feedback)}_{bnt_tag}_{naive_tag}.npy"
    np.save(fname, fisher)
    print(f"Saved Fisher matrix to {fname}")

    return fisher

# --- Run ---
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bnt", action="store_true", help="Apply BNT transform before computing data vector.")
    parser.add_argument("--naive", action="store_true", help="Naive BNT Smoothing")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    for b_feedback in [7.0]:
        compute_fisher_matrix(
            base_dir="/srv/scratch2/taylor.4264/BNTSmooth_data/maps/maps",
            baryon_feedback=b_feedback,
            use_bnt=args.use_bnt,
            naive=args.naive 
        )

if __name__ == "__main__":
    main()