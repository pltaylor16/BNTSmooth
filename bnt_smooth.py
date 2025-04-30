import numpy as np
from sbi_lens.simulator.LogNormal_field import LogNormal_field
import copy
import pyccl as ccl

class LognormalWeakLensingSim:
    def __init__(self, nz_list, n_eff_list, sigma_eps_list,
                 baryon_feedback=3.13, seed=42, sigma8=0.8,
                 lognormal_shifts=None, l_max=3000, cosmo_params=None):
        """
        Initialize the lognormal weak lensing simulation.

        Parameters
        ----------
        nz_list : list of (z, nz) tuples
            Redshift distributions for each source bin.
        n_eff_list : list of float
            Effective number densities (gal/arcmin²) for each source bin.
        sigma_eps_list : list of float
            Intrinsic ellipticity dispersion per component for each bin.
        baryon_feedback : float
            HMCode baryonic feedback strength (log10 A_baryon).
        seed : int
            Random seed for reproducibility.
        sigma8 : float
            Amplitude of matter fluctuations.
        lognormal_shifts : list of float
            Lognormal shift parameters for each bin.
        l_max : int
            Maximum multipole for power spectrum generation.
        cosmo_params : dict, optional
            Cosmological parameters (Ωm, Ωb, h, n_s, etc.).
        """
        self.nz_list = nz_list
        self.n_eff_list = n_eff_list
        self.sigma_eps_list = sigma_eps_list
        self.baryon_feedback = baryon_feedback
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sigma8 = sigma8
        self.l_max = l_max

        self.nbins = len(nz_list)
        if not (len(n_eff_list) == len(sigma_eps_list) == self.nbins):
            raise ValueError("Length of n_eff_list and sigma_eps_list must match number of tomographic bins.")

        if lognormal_shifts is None:
            self.lognormal_shifts = [0.0] * self.nbins
        else:
            if len(lognormal_shifts) != self.nbins:
                raise ValueError("Length of lognormal_shifts must match number of tomographic bins.")
            self.lognormal_shifts = lognormal_shifts

        self.cosmo_params = cosmo_params or {
            "Omega_c": 0.25,
            "Omega_b": 0.05,
            "h": 0.67,
            "n_s": 0.96
        }
        self.cosmo_params["sigma8"] = self.sigma8



    def compute_wl_cls(self, ell=None):
            """
            Compute tomographic convergence power spectra C_ell^{ij} for all i,j bins.

            Parameters
            ----------
            ell : array-like, optional
                Array of multipoles at which to evaluate the power spectra. If None, defaults to logspace up to l_max.

            Returns
            -------
            ell : ndarray
                Multipole array used.
            cl_dict : dict
                Dictionary with keys (i, j) and values C_ell^{ij} arrays.
            """
            if ell is None:
                ell = np.logspace(np.log10(10), np.log10(self.l_max), 50)

            # --- Set up PyCCL cosmology ---
            cosmo = ccl.Cosmology(
                Omega_c=self.cosmo_params["Omega_c"],
                Omega_b=self.cosmo_params["Omega_b"],
                h=self.cosmo_params["h"],
                n_s=self.cosmo_params["n_s"],
                sigma8=self.sigma8,
                matter_power_spectrum="halofit",
                extra_parameters={
                    "halofit_version": "mead2020",
                    "baryonic_feedback": self.baryon_feedback
                }
            )

            # --- Create CCL tracers for each tomographic bin ---
            tracers = []
            for z, nz in self.nz_list:
                nz_norm = nz / np.trapz(nz, z)
                tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_norm))
                tracers.append(tracer)

            # --- Compute all tomographic power spectra ---
            cl_dict = {}
            for i in range(self.nbins):
                for j in range(i, self.nbins):
                    cl_ij = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
                    cl_dict[(i, j)] = cl_ij
                    if i != j:
                        cl_dict[(j, i)] = cl_ij  # symmetry

            return ell, cl_dict


    def generate_lognormal_fields(self, n_pix=256, map_size_deg=5.0):
        """
        Generate lognormal convergence maps for each tomographic bin using sbi_lens.

        Parameters
        ----------
        n_pix : int
            Number of pixels per side in the map (assumes square map).
        map_size_deg : float
            Size of the map in degrees (assumes square map).

        Returns
        -------
        kappa_maps : list of ndarray
            List of 2D lognormal κ maps for each tomographic bin.
        """
        # Compute auto power spectra
        ell, cl_dict = self.compute_wl_cls()
        kappa_maps = []

        for i in range(self.nbins):
            cl_ii = cl_dict[(i, i)]
            shift = self.lognormal_shifts[i]
            ell_input = ell.copy()
            cl_input = cl_ii.copy()

            lognormal_sim = LogNormal_field(
                ell=ell_input,
                cl=cl_input,
                npix=n_pix,
                map_size=map_size_deg,
                shift=shift,
                seed=self.rng.integers(1e6)
            )

            kappa_map = lognormal_sim.get_field()
            kappa_maps.append(kappa_map)

        return kappa_maps


