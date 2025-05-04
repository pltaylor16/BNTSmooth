import numpy as np
import pyccl as ccl
import healpy as hp
from BNT import BNT as BNT


class TruncatedLognormalWeakLensingSim:
    def __init__(self, z_array, nz_list, n_eff_list, sigma_eps_list,
                 baryon_feedback=3.13, seed=42, sigma8=0.8, Omega_m=0.3,
                 gauss_amplitude=1.0, nongauss_amplitude=1.0,
                 l_max=256, nside=256, nslices=50, cosmo_params=None):
        """
        Initialize the truncated lognormal weak lensing simulation.

        Parameters
        ----------
        z_array : ndarray
            Common redshift array to interpolate all n(z) distributions onto.
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
        Omega_m : float
            Total matter density parameter (Ωₘ = Ω_c + Ω_b).
        gauss_amplitude : float
            Amplitude scaling for the Gaussian component of the density field.
        nongauss_amplitude : float
            Amplitude scaling for the non-Gaussian component of the density field.
        l_max : int
            Maximum multipole for power spectrum generation.
        nside : int
            HEALPix nside parameter for map resolution.
        nslices : int
            Number of redshift slices for shell integration.
        cosmo_params : dict, optional
            Additional cosmological parameters to override defaults.
        """
        self.z_array = z_array
        self.zmax = z_array.max()  # Set zmax automatically

        # Interpolate all n(z) onto shared grid
        from scipy.interpolate import interp1d
        self.nz_list = []
        for z_i, nz_i in nz_list:
            interp_func = interp1d(z_i, nz_i, bounds_error=False, fill_value=0.0)
            nz_interp = interp_func(z_array)
            self.nz_list.append((z_array, nz_interp))

        self.n_eff_list = n_eff_list
        self.sigma_eps_list = sigma_eps_list
        self.baryon_feedback = baryon_feedback
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sigma8 = sigma8
        self.Omega_m = Omega_m
        self.gauss_amplitude = gauss_amplitude
        self.nongauss_amplitude = nongauss_amplitude
        self.l_max = l_max
        self.nside = nside
        self.nslices = nslices

        self.nbins = len(self.nz_list)
        if not (len(n_eff_list) == len(sigma_eps_list) == self.nbins):
            raise ValueError("n_eff_list and sigma_eps_list must match number of tomographic bins.")

        self.cosmo_params = cosmo_params or {
            "Omega_c": 0.25,
            "Omega_b": 0.05,
            "h": 0.67,
            "n_s": 0.96
        }
        self.cosmo_params["sigma8"] = self.sigma8
        self.cosmo_params["Omega_c"] = self.Omega_m - self.cosmo_params["Omega_b"]



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
        Generate truncated lognormal random fields for matter density shells from scratch,
        applying a constant shift to the Gaussian field before exponentiation.

        Returns
        -------
        maps : list of ndarray
            List of HEALPix truncated lognormal κ maps for each redshift shell.
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

            nongauss_term = 0.5 * delta_g ** 2. + 1/6. * delta_g ** 3.
            delta_ln = self.gauss_amplitude * delta_g + self.nongauss_amplitude * nongauss_term

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

    def compute_data_vector(self, maps):
        """
        Compute ⟨κ²⟩ and ⟨κ³⟩ for each tomographic bin and return as concatenated data vector.

        Parameters
        ----------
        maps : list of ndarray
            List of κ maps (signal, noise, or combined), one per tomographic bin.

        Returns
        -------
        data_vector : ndarray
            Concatenated array [⟨κ²⟩₁, ..., ⟨κ²⟩ₙ, ⟨κ³⟩₁, ..., ⟨κ³⟩ₙ].
        """
        kappa2 = [np.mean(k**2) for k in maps]
        kappa3 = [np.mean(k**3) for k in maps]
        return np.concatenate([kappa2, kappa3])


    def get_bnt_matrix(self):
        """
        Construct the BNT matrix for source tomography.

        Returns
        -------
        BNT_matrix : ndarray
            A (N, N) matrix
        """

        normed_nz_list = []

        for z, nz in self.nz_list:
            nz /= np.trapz(nz, z)
            normed_nz_list.append(nz)
            z_arr = z

        chi = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1.0 + self.z_array))
        B = BNT(self.z_array, chi, normed_nz_list)
        BNT_matrix = B.get_matrix()

        return BNT_matrix


    def bnt_transform_kappa_maps(self, kappa_maps):
        """
        Apply the BNT transformation to a list of κ maps.

        Parameters
        ----------
        kappa_maps : list of ndarray
            List of input κ maps (length N), one per tomographic bin.

        Returns
        -------
        kappa_maps_bnt : list of ndarray
            List of BNT-transformed κ maps (length N), one per BNT bin.
        """
        # Get the BNT matrix (N x N)
        B = self.get_bnt_matrix()

        # Stack κ maps into a (N, npix) array
        kappa_array = np.stack(kappa_maps)  # shape (N, npix)

        # Apply the BNT transform (matrix multiplication)
        kappa_bnt_array = B @ kappa_array  # shape (N, npix)

        # Convert back to list of maps
        kappa_maps_bnt = [kappa_bnt_array[i] for i in range(kappa_bnt_array.shape[0])]

        return kappa_maps_bnt


    def inverse_bnt_transform_kappa_maps(self, kappa_maps_bnt):
        """
        Apply the inverse BNT transformation to a list of κ maps.

        Parameters
        ----------
        kappa_maps_bnt : list of ndarray
            List of BNT-transformed κ maps (length N), one per BNT bin.

        Returns
        -------
        kappa_maps : list of ndarray
            List of original κ maps (length N), one per tomographic bin.
        """
        # Get the BNT matrix (N x N)
        B = self.get_bnt_matrix()

        # Stack BNT κ maps into a (N, npix) array
        kappa_bnt_array = np.stack(kappa_maps_bnt)  # shape (N, npix)

        # Apply the inverse BNT transform
        kappa_array = np.linalg.inv(B) @ kappa_bnt_array  # shape (N, npix)

        # Convert back to list of maps
        kappa_maps = [kappa_array[i] for i in range(kappa_array.shape[0])]

        return kappa_maps

    def get_lensing_kernels_on_z_grid(self):
        """
        Compute lensing kernels q_i(chi) evaluated on the redshift grid z_arr 
        from the input nz_list. Assumes all n(z)s share the same z grid.

        Returns
        -------
        chi_arr : ndarray
            Comoving distance array corresponding to the shared z_arr grid.
        q_chi_list : list of ndarray
            List of lensing kernel arrays q_i(chi) for each tomographic bin.
        """
        # Use z_arr from the first source bin (assumed shared by all)
        z_arr = self.nz_list[0][0]
        chi_arr = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1.0 + z_arr))

        q_chi_list = []
        for z_nz, n_z in self.nz_list:
            # Normalize n(z)
            n_z = n_z / np.trapz(n_z, z_nz)

            # Compute q(chi) on chi_arr grid
            q = np.zeros_like(chi_arr)
            for i, chi in enumerate(chi_arr):
                a = 1.0 / (1.0 + z_arr[i])
                chi_s = ccl.comoving_radial_distance(self.cosmo, 1.0 / (1.0 + z_nz))
                w = np.zeros_like(z_nz)
                mask = chi_s > chi
                w[mask] = (chi_s[mask] - chi) / chi_s[mask]
                c_light = 299792.458  # speed of light in km/s
                prefac = 1.5 * (self.cosmo["Omega_m"]) * (self.cosmo["h"]**2) * (100 / c_light)**2
                q[i] = prefac * a * chi * np.trapz(w * n_z, z_nz)

            q_chi_list.append(q)

        return chi_arr, q_chi_list


    def bnt_transform_lensing_kernels(self):
        """
        Apply the BNT transformation to the lensing kernels q_i(chi) 
        evaluated on the original z grid shared by all n(z).

        Returns
        -------
        q_bnt_list : list of ndarray
            BNT-transformed lensing kernels for each bin.
        """
        # Get chi and original q_i(chi) values
        chi_arr, q_chi_list = self.get_lensing_kernels_on_z_grid()

        # Stack into matrix: shape (nbins, nchis)
        q_matrix = np.stack(q_chi_list)

        # Construct the BNT transformation matrix
        z_arr = self.nz_list[0][0]
        nz_normed = [nz / np.trapz(nz, z_arr) for _, nz in self.nz_list]
        B = BNT(z_arr, chi_arr, nz_normed)
        B_matrix = B.get_matrix()  # shape (nbins, nbins)

        # Apply BNT: B @ q_matrix → shape (nbins, nchis)
        q_bnt_matrix = B_matrix @ q_matrix

        # Split back into list of arrays
        q_bnt_list = [q_bnt_matrix[i] for i in range(q_bnt_matrix.shape[0])]

        return q_bnt_list

    def compute_kernel_weighted_mean_chi(self, chi_arr, q_list):
        """
        Compute the average comoving distance ⟨χ⟩ for each lensing kernel q^i(χ).

        Parameters
        ----------
        chi_arr : ndarray
            1D array of comoving distances χ corresponding to the q(χ) values.
        q_list : list of ndarray
            List of q^i(χ) arrays, one per tomographic bin.

        Returns
        -------
        mean_chis : list of float
            Average ⟨χ⟩ value for each kernel.
        """
        mean_chis = []
        for q in q_list:
            numerator = np.trapz(chi_arr * q, chi_arr)
            denominator = np.trapz(q, chi_arr)
            mean_chi = numerator / denominator if denominator != 0 else 0.0
            mean_chis.append(mean_chi)

        return mean_chis


    def compute_data_vector(self, maps):
        """
        Compute second and third moments of κ maps and return as concatenated data vector.

        Parameters
        ----------
        maps : list of ndarray
            List of κ maps (signal, noise, or combined), one per tomographic bin.

        Returns
        -------
        data_vector : ndarray
            Concatenated array [⟨κ²⟩₁, ..., ⟨κ²⟩ₙ, ⟨κ³⟩₁, ..., ⟨κ³⟩ₙ].
        """
        moments_2 = [np.mean(k**2) for k in maps]
        moments_3 = [np.mean(k**3) for k in maps]
        data_vector = np.concatenate([moments_2, moments_3])
        return data_vector







        




