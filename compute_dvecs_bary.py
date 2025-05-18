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
    nslices=nslices,
)
sim.set_cosmo()

# --- Map loading + data vector computation ---
def compute_dvec_from_file(path, fwhm_arcmin, baryon_feedback, physical_scale_mpc, use_bnt=False):
    fname = os.path.basename(path)
    print(f"[{baryon_feedback=}, R={physical_scale_mpc} Mpc, {use_bnt=}] Processing: {fname}", flush=True)

    loaded = np.load(path)
    kappa_maps = [loaded[f"slice{i}"] for i in range(nslices)]

    if use_bnt:
        kappa_maps = sim.bnt_smoothing(kappa_maps, physical_scale_mpc)
    else:
        kappa_maps = sim.smoothing(kappa_maps, physical_scale_mpc)
        

    return sim.compute_data_vector(kappa_maps)

# --- Helper: compute and save data vectors ---
def compute_dvecs(base_dir, fwhm_arcmin, baryon_feedback, physical_scale_mpc, use_bnt=False):
    print(f"\n=== Computing dvecs for baryon_feedback={baryon_feedback}, R={physical_scale_mpc} Mpc ===")

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
        func = partial(
            compute_dvec_from_file,
            fwhm_arcmin=fwhm_arcmin,
            baryon_feedback=baryon_feedback,
            physical_scale_mpc=physical_scale_mpc,
            use_bnt=use_bnt
        )
        dvecs_fid    = pool.map(func, paths_fid)
        dvecs_aplus  = pool.map(func, paths_aplus)
        dvecs_aminus = pool.map(func, paths_aminus)
        dvecs_bplus  = pool.map(func, paths_bplus)
        dvecs_bminus = pool.map(func, paths_bminus)

    # Stack results
    dvecs_fid = np.stack(dvecs_fid)
    dvecs_aplus = np.stack(dvecs_aplus)
    dvecs_aminus = np.stack(dvecs_aminus)
    dvecs_bplus = np.stack(dvecs_bplus)
    dvecs_bminus = np.stack(dvecs_bminus)

    # Save results
    bnt_tag = "bnt" if use_bnt else "nobnt"
    smooth_tag = f"R{int(physical_scale_mpc)}"
    Path("dvecs").mkdir(exist_ok=True)

    dvec_path = f"dvecs/bary/dvecs_b{int(baryon_feedback)}_{smooth_tag}_{bnt_tag}.npz"
    np.savez_compressed(
        dvec_path,
        fid=dvecs_fid,
        alpha_plus=dvecs_aplus,
        alpha_minus=dvecs_aminus,
        beta_plus=dvecs_bplus,
        beta_minus=dvecs_bminus,
    )
    print(f"Saved data vectors to {dvec_path}")
    return 0.

# --- Run ---
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bnt", action="store_true", help="Apply BNT transform before computing data vector.")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    for smooth_mpc in [1., 2., 4., 6., 8., 10., 12., 14.]:
        compute_dvecs(
            base_dir="/srv/scratch2/taylor.4264/BNTSmooth_data/maps_baryon/maps",
            baryon_feedback=7.0,
            fwhm_arcmin=None,
            physical_scale_mpc=smooth_mpc,
            use_bnt=args.use_bnt,
        )

if __name__ == "__main__":
    main()