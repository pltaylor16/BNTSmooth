import numpy as np
import pyccl as ccl
import healpy as hp
from BNT import BNT as BNT

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


    def set_cosmo(self):
        self.cosmo = ccl.Cosmology(
            Omega_c=self.cosmo_params["Omega_c"],
            Omega_b=self.cosmo_params["Omega_b"],
            h=self.cosmo_params["h"],
            n_s=self.cosmo_params["n_s"],
            sigma8=self.sigma8,
            matter_power_spectrum="camb",
            extra_parameters = {"camb": {"halofit_version": "mead2020_feedback",
                             "HMCode_logT_AGN": self.baryon_feedback}},
        )

        return 0.



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

        # Create number count tracers for each redshift shell
        tracers = []
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            z = np.linspace(z0, z1, 100)
            dz = z1 - z0
            dndz = np.ones_like(z) / dz
            tracer = ccl.NumberCountsTracer(
                self.cosmo,
                has_rsd=False,
                dndz=(z, dndz),
                bias=(z, np.ones_like(z))
            )
            tracers.append(tracer)

        # Compute auto-spectra only
        gls = []
        for i in range(self.nslices):
            cl_ii = ccl.angular_cl(self.cosmo, tracers[i], tracers[i], ell)
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
            np.random.seed(self.seed)
            delta_g = hp.synfast(full_cl, nside=nside)

            # Apply lognormal transformation 
            # glass eqn 12 in https://arxiv.org/pdf/2302.01942
            delta_ln = self.lognormal_shift * (np.exp(delta_g) - 1)

            maps.append(delta_ln)

        return maps


    def get_shell_zchi(self):
        """
        Compute effective redshift, comoving distance, and shell width for each redshift shell.

        Returns
        -------
        z_eff : ndarray
            Midpoint redshift of each shell.
        chi_eff : ndarray
            Comoving radial distance at z_eff [Mpc].
        delta_chi : ndarray
            Comoving shell width (chi_{i+1} - chi_i) [Mpc].
        """
        z_edges = np.linspace(0, self.zmax, self.nslices + 1)
        z_eff = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Compute comoving distances at z_edges and z_eff
        chi_edges = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1 + z_edges))
        chi_eff = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1 + z_eff))

        # Shell widths in comoving distance
        delta_chi = chi_edges[1:] - chi_edges[:-1]

        return z_eff, chi_eff, delta_chi


    def get_lensing_kernels(self):
        """
        Compute lensing kernel q_i(chi) for each source bin at the matter shell midpoints.

        Returns
        -------
        q_list : list of ndarray
            List of arrays q_i(chi_eff) for each tomographic source bin.
        """
        # Effective z and chi for matter shells
        z_eff, chi_eff, _ = self.get_shell_zchi()

        q_list = []
        for z_nz, n_z in self.nz_list:
            # Normalize n(z)
            n_z = n_z / np.trapz(n_z, z_nz)

            # Compute lensing kernel q(chi_eff) for this source bin
            q_arr = np.zeros_like(chi_eff)
            for i, chi in enumerate(chi_eff):
                a = 1.0 / (1.0 + z_eff[i])
                chi_s = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1.0 + z_nz))
                w = np.zeros_like(z_nz)
                mask = chi_s > chi
                w[mask] = (chi_s[mask] - chi) / chi_s[mask]
                c_light = 299792.458  # speed of light in km/s
                q = 1.5 * (self.cosmo["Omega_m"]) * (self.cosmo["h"]**2) * (100 / c_light)**2
                q *= a * chi * np.trapz(w * n_z, z_nz)
                q_arr[i] = q

            q_list.append(q_arr)

        return q_list


    def compute_kappa_maps(self, matter_maps):
        """
        Compute convergence κ maps for each tomographic source bin by integrating over matter shells.

        Parameters
        ----------
        matter_maps : list of ndarray
            Lognormal matter density maps for each shell (length = nslices).

        Returns
        -------
        kappa_maps : list of ndarray
            List of κ maps for each tomographic bin (length = nbins).
        """
        if len(matter_maps) != self.nslices:
            raise ValueError("Number of matter maps must equal nslices.")

        # Get lensing kernel and comoving shell widths
        z_eff, chi_eff, delta_chi = self.get_shell_zchi()
        q_list = self.get_lensing_kernels()  # one [nslices] array per tomo bin

        npix = len(matter_maps[0])
        kappa_maps = []

        for q in q_list:  # Loop over tomographic bins
            kappa = np.zeros(npix)
            for i in range(self.nslices):
                kappa += delta_chi[i] * q[i] * matter_maps[i]
            kappa_maps.append(kappa)

        return kappa_maps


    def generate_noise_only_kappa_maps(self):
        """
        Generate pure shape noise κ maps (no signal) by Poisson sampling the galaxy counts per pixel,
        and drawing Gaussian noise with variance σ_ε² / N_pix.

        Returns
        -------
        noise_maps : list of ndarray
            List of noise-only κ maps, one per tomographic bin.
        """
        npix = hp.nside2npix(self.nside)
        omega_pix_arcmin2 = hp.nside2pixarea(self.nside, degrees=True) * 3600.0  # arcmin²

        noise_maps = []
        for sigma_eps, n_eff in zip(self.sigma_eps_list, self.n_eff_list):
            mean_ngal_per_pix = n_eff * omega_pix_arcmin2  # expected number of galaxies per pixel

            # Poisson sample actual galaxy counts
            ngal_pix = self.rng.poisson(lam=mean_ngal_per_pix, size=npix)

            # Avoid divide-by-zero issues
            ngal_pix_safe = np.where(ngal_pix == 0, 1, ngal_pix)

            # Gaussian noise with σ² = σ_ε² / N_pix
            noise_std = sigma_eps / np.sqrt(ngal_pix_safe)
            noise_map = self.rng.normal(0, noise_std, size=npix)

            # Set noise to 0 in empty pixels
            noise_map[ngal_pix == 0] = 0.0

            noise_maps.append(noise_map)

        return noise_maps


    def generate_noisy_kappa_maps(self):
        """
        Run the pipeline to generate noisy κ maps:
        1. Set cosmology
        2. Generate matter maps
        3. Compute κ signal maps
        4. Generate noise maps
        5. Add signal and noise
        6. Return the noisy κ maps for each tomographic bin

        Returns
        -------
        noisy_kappa_maps : list of ndarray
            List of κ maps (signal + noise) for each tomographic bin.
        """
        self.set_cosmo()

        matter_maps = self.generate_matter_fields_from_scratch()
        kappa_maps = self.compute_kappa_maps(matter_maps)
        noise_maps = self.generate_noise_only_kappa_maps()

        noisy_kappa_maps = [kappa + noise for kappa, noise in zip(kappa_maps, noise_maps)]

        return noisy_kappa_maps


class ProcessMaps(LognormalWeakLensingSim):
    """
    Subclass for processing κ maps after simulation.
    Inherits all setup and map generation functionality from LognormalWeakLensingSim.
    Intended for operations like smoothing, masking, moment calculation, and data vector extraction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_moments(self, maps):
        """
        Compute the second and third moments of a list of κ maps.

        Parameters
        ----------
        maps : list of ndarray
            List of κ maps (signal, noise, or combined), one per tomographic bin.

        Returns
        -------
        moments_2 : ndarray
            Second moments (⟨κ²⟩) for each tomographic bin.
        moments_3 : ndarray
            Third moments (⟨κ³⟩) for each tomographic bin.
        """
        nbins = len(maps)
        moments_2 = np.zeros(nbins)
        moments_3 = np.zeros(nbins)

        for i, kappa in enumerate(maps):
            moments_2[i] = np.mean(kappa**2)
            moments_3[i] = np.mean(kappa**3)

        return moments_2, moments_3


	def get_bnt_matrix(self):
	    """
	    Construct the BNT matrix for source tomography.

	    Returns
	    -------
	    BNT_matrix : ndarray
	        A (N, N) matrix
	    """
	    chi_list = []
	    normed_nz_list = []

	    for z, nz in self.nz_list:
	        chi = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1.0 + z))
	        nz /= np.trapz(nz, z)
	        chi_list.append(chi)
	        normed_nz_list.append(nz)
	        z_arr = z

	    B = BNT(z_arr, chi, normed_nz_list)
	    BNT_matrix = B.get_matrix

	    return BNT_matrix


	def bnt_transform_kappa_maps(self);
		pass


	def inverse_bnt_tranform_kappa_maps(self):
		pass

		


