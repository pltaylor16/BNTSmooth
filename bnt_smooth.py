import numpy as np
import pyccl as ccl
import healpy as hp


class LognormalWeakLensingSim:
    def __init__(self, nz_list, n_eff_list, sigma_eps_list,
                 baryon_feedback=3.13, seed=42, sigma8=0.8,
                 lognormal_shift=0.0, l_max=256, zmax=3.0, nslices=50, cosmo_params=None):
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
        lognormal_shift : float
            Global lognormal shift parameter used by GLASS.
        l_max : int
            Maximum multipole for power spectrum generation.
        cosmo_params : dict, optional
            Cosmological parameters.
        """
        self.nz_list = nz_list
        self.n_eff_list = n_eff_list
        self.sigma_eps_list = sigma_eps_list
        self.baryon_feedback = baryon_feedback
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sigma8 = sigma8
        self.l_max = l_max
        self.nside = l_max
        self.zmax = zmax
        self.lognormal_shift = lognormal_shift

        self.nbins = len(nz_list)
        if not (len(n_eff_list) == len(sigma_eps_list) == self.nbins):
            raise ValueError("n_eff_list and sigma_eps_list must match number of tomographic bins.")

        self.cosmo_params = cosmo_params or {
            "Omega_c": 0.25,
            "Omega_b": 0.05,
            "h": 0.67,
            "n_s": 0.96
        }
        self.cosmo_params["sigma8"] = self.sigma8
        self.nslices = nslices



    def compute_matter_cls(self):
        """
        Compute auto-spectra C_ell^{δδ} for narrow redshift slices using PyCCL.

        Returns
        -------
        gls : list of ndarray
            List of auto-spectra Cl^{ii}(ℓ) arrays for each slice i.
        """
        ell = np.arange(2, self.l_max + 1)
        z_edges = np.linspace(0, self.zmax, self.nslices + 1)

        cosmo = ccl.Cosmology(
            Omega_c=self.cosmo_params["Omega_c"],
            Omega_b=self.cosmo_params["Omega_b"],
            h=self.cosmo_params["h"],
            n_s=self.cosmo_params["n_s"],
            sigma8=self.sigma8,
            matter_power_spectrum="halofit",
            extra_parameters={"halofit_version": "mead2020", "baryonic_feedback": self.baryon_feedback},
        )

        # Create number count tracers for each redshift shell
        tracers = []
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            z = np.linspace(z0, z1, 100)
            dz = z1 - z0
            dndz = np.ones_like(z) / dz
            tracer = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=False,
                dndz=(z, dndz),
                bias=(z, np.ones_like(z))
            )
            tracers.append(tracer)

        # Compute auto-spectra only
        gls = []
        for i in range(self.nslices):
            cl_ii = ccl.angular_cl(cosmo, tracers[i], tracers[i], ell)
            gls.append(cl_ii)

        return gls


    def generate_matter_fields_from_scratch(self):
        """
        Generate lognormal random fields for matter density shells from scratch,
        applying a constant shift to the Gaussian field before exponentiation.

        Returns
        -------
        maps : list of ndarray
            List of HEALPix lognormal κ maps for each redshift shell.
        """
        cls = self.compute_matter_cls()
        nside = self.nside

        maps = []
        for cl in cls:
            # Extend Cl to full ell range for synfast
            full_cl = np.zeros(self.l_max + 1)
            full_cl[2:] = cl

            # Generate Gaussian field
            delta_g = hp.synfast(full_cl, nside=nside, verbose=False)

            # Apply lognormal transformation with shift
            delta_ln = np.exp(delta_g - self.lognormal_shift) - 1

            maps.append(delta_ln)

        return maps


