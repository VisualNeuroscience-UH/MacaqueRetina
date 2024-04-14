# Numerical
import numpy as np

import pandas as pd
from scipy.signal import convolve, convolve2d
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from scipy.special import gamma as gamma_function
import scipy.fftpack as fftpack
from scipy.integrate import odeint
from skimage.transform import resize
import torch

# Viz
from tqdm import tqdm
import matplotlib.pyplot as plt

# Comput Neurosci
import brian2 as b2
import brian2.units as b2u

import brian2cuda

# Local
from retina.retina_math_module import RetinaMath
from project.project_utilities_module import Printable


# Builtin
from copy import deepcopy
import sys
import time

b2.prefs["logging.display_brian_error_message"] = False


class ReceptiveFieldsBase(Printable):
    """
    Class containing information associated with receptive fields, including
    retina parameters, the spatial and temporal filters.
    """

    def __init__(self, my_retina) -> None:
        # Parameters directly passed to the constructor
        self.my_retina = my_retina

        # Default values for computed variables
        self.spatial_filter_sidelen = 0
        self.microm_per_pix = 0.0
        self.temporal_filter_len = 0

        # Extracted and computed values from provided parameters
        self.gc_type = self.my_retina["gc_type"]
        self.response_type = self.my_retina["response_type"]
        self.deg_per_mm = self.my_retina["deg_per_mm"]
        self.DoG_model = self.my_retina["DoG_model"]
        self.spatial_model = self.my_retina["spatial_model"]
        self.temporal_model = self.my_retina["temporal_model"]


class Cones(ReceptiveFieldsBase):
    def __init__(
        self,
        my_retina,
        ret_npz,
        device,
        ND_filter,
        interpolation_function,
        lin_interp_and_double_lorenzian,
    ) -> None:
        super().__init__(my_retina)

        self.my_retina = my_retina
        self.ret_npz = ret_npz
        self.device = device
        self.ND_filter = ND_filter
        self.interpolation_function = interpolation_function
        self.lin_interp_and_double_lorenzian = lin_interp_and_double_lorenzian

        self.cones_to_gcs_weights = ret_npz["cones_to_gcs_weights"]
        self.cone_noise_parameters = ret_npz["cone_noise_parameters"]
        self.cone_general_params = my_retina["cone_general_params"]
        self.n_units = self.cones_to_gcs_weights.shape[0]

    def _cornea_photon_flux_density_to_luminance(self, F, lambda_nm=555):
        """
        Convert photon flux density at cornea to luminance using human photopic vision V(lambda).

        Parameters
        ----------
        F : float
            Photon flux density at the cornea in photons/mm²/s.
        lambda_nm : float, optional
            Wavelength of the monochromatic light in nanometers, default is 555 nm (peak of human photopic sensitivity).

        Returns
        -------
        L : float
            Luminance in cd/m².

        Notes
        -----
        The conversion uses the formula:

        L = F * (hc/lambda) * kappa * V(lambda)

        where:
        - h is Planck's constant (6.626 x 10^-34 J·s),
        - c is the speed of light (3.00 x 10^8 m/s),
        - lambda is the wavelength of light in meters,
        - kappa is the luminous efficacy of monochromatic radiation (683 lm/W at 555 nm),
        - V(lambda) is the photopic luminous efficiency function value at lambda,
        assumed to be 1 at 555 nm for peak human photopic sensitivity.
        """
        # Constants
        h = 6.626e-34  # Planck's constant in J·s
        c = 3.00e8  # Speed of light in m/s
        lambda_m = lambda_nm * 1e-9  # Convert wavelength from nm to m
        kappa = 683  # Luminous efficacy of monochromatic radiation in lm/W at 555 nm

        # Energy of a photon at wavelength lambda in joules
        E_photon = (h * c) / lambda_m

        # Convert photon flux density F to luminance L in cd/m²
        F_m2 = F * 1e6  # Convert photon flux density from mm² to m²
        L = F_m2 * E_photon * kappa

        return L

    def _luminance_to_cornea_photon_flux_density(self, L, lambda_nm=555):
        """
        Convert luminance to photon flux density at cornea using human photopic vision V(lambda).

        Parameters
        ----------
        L : float
            Luminance in cd/m².
        lambda_nm : float, optional
            Wavelength of the monochromatic light in nanometers, default is 555 nm (peak of human photopic sensitivity).

        Returns
        -------
        F : float
            Photon flux density at the cornea in photons/mm²/s.
        """
        # Constants
        h = 6.626e-34  # Planck's constant in J·s
        c = 3.00e8  # Speed of light in m/s
        lambda_m = lambda_nm * 1e-9  # Convert wavelength from nm to m
        kappa = 683  # Luminous efficacy of monochromatic radiation in lm/W at 555 nm

        # Energy of a photon at wavelength lambda in joules
        E_photon = (h * c) / lambda_m

        # Convert luminance L to photon flux density F in photons/mm²/s
        F_m2 = L / (E_photon * kappa)
        F = F_m2 / 1e6  # Convert from m² to mm²

        return F

    def _create_cone_noise(self, tvec, n_cones, *params):
        tvec = tvec / b2u.second
        freqs = fftpack.fftfreq(len(tvec), d=(tvec[1] - tvec[0]))

        white_noise = np.random.normal(0, 1, (len(tvec), n_cones))
        noise_fft = fftpack.fft(white_noise, axis=0)

        # Generate the asymmetric concave function for scaling
        f_scaled = np.abs(freqs)
        # Prevent division by zero for zero frequency
        f_scaled[f_scaled == 0] = 1e-10

        # Transfer to log scale and
        # combine the fitted amplitudes with fixed corner frequencies
        a0 = params[0]
        L1_params = np.array([params[1], self.cone_noise_wc[0]])
        L2_params = np.array([params[2], self.cone_noise_wc[1]])

        asymmetric_scale = self.lin_interp_and_double_lorenzian(
            f_scaled, a0, L1_params, L2_params, self.cone_interp_response
        )

        noise_fft = noise_fft * asymmetric_scale[:, np.newaxis]

        # Transform back to time domain
        cone_noise = np.real(fftpack.ifft(noise_fft, axis=0))

        return cone_noise

    def _create_cone_signal_clark(
        self, cone_input, p, dt, duration, tvec, pad_value=0.0
    ):
        """
        Create cone signal using Brian2. Works in video time domain.

        Parameters
        ----------
        cone_input : ndarray
            The cone input of shape (n_cones, n_timepoints).
        p : dict
            The cone signal parameters.
        dt : float
            The video time step.
        duration : float
            The duration of input video.
        tvec : ndarray
            The time vector of input video.

        Returns
        -------
        ndarray
            The cone signal of shape (n_cones, n_timepoints).
        """

        alpha = p["alpha"]
        beta = p["beta"]
        gamma = p["gamma"]
        tau_y = p["tau_y"] / b2u.second
        n_y = p["n_y"]
        tau_z = p["tau_z"] / b2u.second
        n_z = p["n_z"]
        tau_r = p["tau_r"]  # Goes into Brian which uses seconds for timed arrays
        filter_limit_time = p["filter_limit_time"]

        def simple_filter(t, n, tau):
            norm_coef = gamma_function(n + 1) * np.power(tau, n + 1)
            values = (t**n / norm_coef) * np.exp(-t / tau)
            # values = (t / tau) ** n * np.exp(-t / tau)
            return values

        tvec_idx = tvec < filter_limit_time
        tvec_filter = tvec / b2u.second

        Ky = simple_filter(tvec_filter, n_y, tau_y)
        Kz_prime = simple_filter(tvec_filter, n_z, tau_z)
        Kz = gamma * Ky + (1 - gamma) * Kz_prime

        # Cut filters for computational efficiency
        Ky = Ky[tvec_idx]
        Kz = Kz[tvec_idx]
        Kz_prime = Kz_prime[tvec_idx]

        # Normalize filters to full filter = 1.0
        Ky = Ky / Ky.sum()
        Kz = Kz / Kz.sum()

        # Prepare 2D convolution for the filters,
        Ky_2D_kernel = Ky.reshape(1, -1)
        Kz_2D_kernel = Kz.reshape(1, -1)

        pad_length = len(Ky) - 1

        assert all(cone_input[:, 0] == pad_value), "Padding failed..."

        # Pad cone input start with the initial value to avoid edge effects. Use filter limit time.
        cone_input_padded = np.pad(
            cone_input,
            ((0, 0), (pad_length, 0)),
            mode="constant",
            constant_values=((0, 0), (pad_value, 0)),
        )

        print("\nConvolving cone signal matrices...")
        y_mtx = convolve(cone_input_padded, Ky_2D_kernel, mode="full", method="direct")[
            :, pad_length : pad_length + len(tvec)
        ]
        z_mtx = convolve(cone_input_padded, Kz_2D_kernel, mode="full", method="direct")[
            :, pad_length : pad_length + len(tvec)
        ]

        print("\nRunning Brian code for cones...")
        y_mtx_ta = b2.TimedArray(y_mtx.T, dt=dt)
        z_mtx_ta = b2.TimedArray(z_mtx.T, dt=dt)

        # r(t) is the photoreceptor response = V(t) - Vrest, where
        # Vrest is the depolarized cone membrane potential in the dark.
        # Negative r means hyperpolarization from the resting potential in millivolts.
        eqs = b2.Equations(
            """
            dr/dt = (alpha * y_mtx_ta(t,i)) / tau_r - ((1 + beta * z_mtx_ta(t,i)) * r) / tau_r : 1
            """
        )
        # Assuming dr/dt is zero at t=0, a.k.a. steady illumination
        r_initial_value = alpha * y_mtx[0, 0] / (1 + beta * z_mtx[0, 0])
        # print(f"Initial value of r: {r_initial_value}")
        G = b2.NeuronGroup(self.n_units, eqs, dt=dt, method="exact")
        G.r = r_initial_value
        M = b2.StateMonitor(G, ("r"), record=True)
        b2.run(duration)

        cone_output = M.r - r_initial_value  # Get zero baseline

        return cone_output

    # Detached internal legacy functions
    def _optical_aberration(self):
        """
        Gaussian smoothing from Navarro 1993 JOSAA: 2 arcmin FWHM under 20deg eccentricity.
        """

        # Turn the optical aberration of 2 arcmin FWHM to Gaussian function sigma
        optical_aberration = self.context.my_retina["optical_aberration"]

        sigma_deg = optical_aberration / (2 * np.sqrt(2 * np.log(2)))
        sigma_pix = self.pix_per_deg * sigma_deg
        image = self.image

        # Apply Gaussian blur to each frame in the image array
        if len(image.shape) == 3:
            self.image_after_optics = gaussian_filter(
                image, sigma=[0, sigma_pix, sigma_pix]
            )
        elif len(image.shape) == 2:
            self.image_after_optics = gaussian_filter(
                image, sigma=[sigma_pix, sigma_pix]
            )
        else:
            raise ValueError("Image must be 2D or 3D, aborting...")

    def _luminance2cone_response(self):
        """
        Cone nonlinearity. Equation from Baylor_1987_JPhysiol.
        """

        # Range
        response_range = np.ptp([self.cone_sensitivity_min, self.cone_sensitivity_max])

        # Scale. Image should be between 0 and 1
        image_at_response_scale = self.image * response_range
        cone_input = image_at_response_scale + self.cone_sensitivity_min

        # Cone nonlinearity
        cone_response = self.rm * (1 - np.exp(-self.k * cone_input))

        self.cone_response = cone_response

        # Save the cone response to output folder
        filename = self.context.my_stimulus_metadata["stimulus_file"]
        self.data_io.save_cone_response_to_hdf5(filename, cone_response)

    # Public functions
    def create_signal(self, vs, n_trials):
        """
        Generates cone signal.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object containing the transduction cascade from stimulus video
            to RGC unit spike response.
        n_trials : int
            The number of trials to simulate.

        Returns
        -------
        ndarray
            The cone signal.
        """

        video_copy = vs.stimulus_video.frames.copy()
        video_copy = np.transpose(
            video_copy, (1, 2, 0)
        )  # Original frames are now [height, width, time points]
        video_copy = video_copy[np.newaxis, ...]  # Add new axis for broadcasting

        cone_pos_mm = self.ret_npz["cone_optimized_pos_mm"]
        cone_pos_deg = cone_pos_mm * vs.deg_per_mm
        q, r = vs._vspace_to_pixspace(cone_pos_deg[:, 0], cone_pos_deg[:, 1])
        q_idx = np.floor(q).astype(int)
        r_idx = np.floor(r).astype(int)

        # Ensure r_indices and q_indices are repeated for each unit
        r_indices = r_idx.reshape(-1, 1, 1, 1)
        q_indices = q_idx.reshape(-1, 1, 1, 1)
        time_points_indices = np.arange(video_copy.shape[-1])

        # Use advanced indexing for selecting pixels
        cone_input_cropped = video_copy[0, r_indices, q_indices, time_points_indices]
        cone_input = np.squeeze(cone_input_cropped)

        # Photoisomerization units need more bits to represent the signal
        cone_input = np.squeeze(cone_input_cropped).astype(np.float64)
        minl = np.min(cone_input)
        maxl = np.max(cone_input)

        cone_input = self.get_photoisomerizations_from_luminance(cone_input)

        # Neutral Density filtering factor to reduce or increase luminance
        ff = np.power(10.0, -self.ND_filter)
        cone_input = cone_input * ff
        minp = np.min(cone_input).astype(int)
        maxp = np.max(cone_input).astype(int)

        print(f"\nLuminance range: {minl * ff:.2f} to {maxl * ff:.2f} cd/m²")
        print(f"\nR* range: {minp} to {maxp} photoisomerizations/cone/s")

        # Update visual stimulus photodiode response
        vs.photodiode_response = vs.photodiode_response * ff
        vs.photodiode_Rstar_range = [minp, maxp]

        # Update mean value
        background = vs.options_from_file["background"]
        background_R = self.get_photoisomerizations_from_luminance(background)
        background_R = background_R * ff

        print(f"\nbackground_R* {background_R} photoisomerizations/cone/s")

        params_dict = self.my_retina["cone_signal_parameters"]

        tvec = vs.tvec
        dt = vs.video_dt
        duration = vs.duration

        cone_signal = self._create_cone_signal_clark(
            cone_input, params_dict, dt, duration, tvec, background_R
        )

        print("\nCone signal min:", cone_signal.min())
        print("Cone signal max:", cone_signal.max())

        vs.cone_signal = cone_signal

        return vs

    def create_noise(self, vs, n_trials):
        """
        Generates cone noise.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object containing the transduction cascade from stimulus video
            to RGC unit spike response.
        gcs : ReceptiveFields
            The receptive fields object containing the RGCs and their parameters.
        n_trials : int
            The number of trials to simulate.

        Returns
        -------
        ndarray
            The normalized cone noise.
        """
        params = self.cone_noise_parameters
        n_cones = self.n_units

        # ret_file_npz = self.data_io.get_data(self.context.my_retina["ret_file"])
        cone_frequency_data = self.ret_npz["cone_frequency_data"]
        cone_power_data = self.ret_npz["cone_power_data"]
        self.cone_interp_response = self.interpolation_function(
            cone_frequency_data, cone_power_data
        )
        self.cone_noise_wc = self.cone_general_params["cone_noise_wc"]

        # Make independent cone noise for multiple trials
        if n_trials > 1:
            for trial in range(n_trials):
                cone_noise = self._create_cone_noise(vs.tvec, n_cones, *params)
                if trial == 0:
                    trials_noise = cone_noise
                else:
                    trials_noise = np.concatenate(
                        (trials_noise, cone_noise), axis=1
                    )  # TODO CHECK MULTIPLE TRIALS

            cone_noise = trials_noise
        else:
            cone_noise = self._create_cone_noise(vs.tvec, n_cones, *params)

        # Normalize noise to have one mean and unit sd at the noise data frequencies
        cone_noise_norm = 1 + (cone_noise - cone_noise.mean()) / np.std(
            cone_noise, axis=0
        )

        vs.cone_noise = cone_noise_norm

        return vs

    def get_luminance_from_photoisomerizations(
        self, I_cone, A_pupil=9.3, A_retina=670, a_c_end_on=3.21e-5, tau_media=1.0
    ):
        """
        Calculate luminance from photoisomerizations.

        Parameters
        ----------
        I_cone : float
            The number of photoisomerizations per second per cone.
        A_pupil : float
            The area of the pupil in mm^2.
            Mean tonic pupil radiusis 3.44/2 mm in macaques, from Selezneva_2021_FrontPsychol
        A_retina : float
            The area of the retina in mm^2.
        a_c_end_on : float
            Upper limit for the effective cross-sectional area of the total
            pigment content of a photoreceptor for axially propagating light.
            Default value is 3.21e-5 mm^2, derived from cone density at 5 deg ecc.
        tau_media : float
            The transmittance of the media. Default value is 1.0.
        """

        # Calculate photon flux at cornea
        F_cornea = I_cone / (a_c_end_on * (A_pupil / A_retina) * tau_media)

        # Get the luminance from the photon flux density
        luminance = self._cornea_photon_flux_density_to_luminance(
            F_cornea, lambda_nm=555
        )

        return luminance

    def get_photoisomerizations_from_luminance(
        self, L, A_pupil=9.3, A_retina=670, a_c_end_on=3.21e-5, tau_media=1.0
    ):
        """
        Calculate the rate of photoisomerizations per cone per second from luminance.

        Parameters
        ----------
        L : float
            Luminance in cd/m².
        A_pupil : float
            The area of the pupil in mm^2.
        A_retina : float
            The area of the retina in mm^2.
        a_c_end_on : float
            The end-on collecting area for the cones in mm^2.
        tau_media : float
            The transmittance of the ocular media at wavelength λ.

        Returns
        -------
        I_cone : float
            The rate of photoisomerizations per cone per second (R* cone^-1 s^-1).
        """
        # Convert luminance to photon flux density
        F_cornea = self._luminance_to_cornea_photon_flux_density(L, lambda_nm=555)

        # Calculate the photon flux density at the retina (F_retina)
        F_retina = F_cornea * (A_pupil / A_retina) * tau_media

        # Calculate the rate of photoisomerizations per cone per second (I_cone)
        I_cone = F_retina * a_c_end_on

        return I_cone


class Bipolars(ReceptiveFieldsBase):
    def __init__(self, my_retina, ret_npz) -> None:
        super().__init__(my_retina)
        self.my_retina = my_retina
        self.ret_npz = ret_npz


class GanglionCells(ReceptiveFieldsBase):
    def __init__(
        self,
        my_retina,
        apricot_metadata,
        rfs_npz,
        gc_dataframe,
        unit_index,
        spike_generator_model,
        pol2cart_df,
    ):
        super().__init__(my_retina)

        self.spike_generator_model = spike_generator_model

        self.mask_threshold = my_retina["center_mask_threshold"]
        self.refractory_params = my_retina["refractory_params"]

        assert isinstance(
            self.mask_threshold, float
        ), "mask_threshold must be float, aborting..."
        assert (
            self.mask_threshold >= 0 and self.mask_threshold <= 1
        ), "mask_threshold must be between 0 and 1, aborting..."

        self.apricot_metadata = apricot_metadata
        self.data_microm_per_pixel = self.apricot_metadata["data_microm_per_pix"]
        self.data_filter_fps = self.apricot_metadata["data_fps"]
        self.data_filter_timesteps = self.apricot_metadata[
            "data_temporalfilter_samples"
        ]
        self.data_filter_duration = self.data_filter_timesteps * (
            1000 / self.data_filter_fps
        )

        rspace_pos_mm = pol2cart_df(gc_dataframe)
        vspace_pos = rspace_pos_mm * self.deg_per_mm
        vspace_coords_deg = pd.DataFrame(
            {"x_deg": vspace_pos[:, 0], "y_deg": vspace_pos[:, 1]}
        )
        df = pd.concat([gc_dataframe, vspace_coords_deg], axis=1)

        if self.DoG_model in ["ellipse_fixed"]:
            # Convert RF center radii to degrees as well
            df["semi_xc_deg"] = df.semi_xc_mm * self.deg_per_mm
            df["semi_yc_deg"] = df.semi_yc_mm * self.deg_per_mm
            # Drop rows (units) where semi_xc_deg and semi_yc_deg is zero.
            # These have bad (>3SD deviation in any ellipse parameter) fits
            df = df[(df.semi_xc_deg != 0) & (df.semi_yc_deg != 0)].reset_index(
                drop=True
            )
        if self.DoG_model in ["ellipse_independent"]:
            # Convert RF center radii to degrees as well
            df["semi_xc_deg"] = df.semi_xc_mm * self.deg_per_mm
            df["semi_yc_deg"] = df.semi_yc_mm * self.deg_per_mm
            df["semi_xs_deg"] = df.semi_xs_mm * self.deg_per_mm
            df["semi_ys_deg"] = df.semi_ys_mm * self.deg_per_mm
            df = df[(df.semi_xc_deg != 0) & (df.semi_yc_deg != 0)].reset_index(
                drop=True
            )
        elif self.DoG_model == "circular":
            df["rad_c_deg"] = df.rad_c_mm * self.deg_per_mm
            df["rad_s_deg"] = df.rad_s_mm * self.deg_per_mm
            df = df[(df.rad_c_deg != 0) & (df.rad_s_deg != 0)].reset_index(drop=True)

        # Drop retinal positions from the df (so that they are not used by accident)
        df = df.drop(["pos_ecc_mm", "pos_polar_deg"], axis=1)

        self.df = df

        self.spat_rf = rfs_npz["gc_img"]
        self.um_per_pix = rfs_npz["um_per_pix"]
        self.sidelen_pix = rfs_npz["pix_per_side"]

        # Run all units
        if unit_index is None:
            self.n_units = len(df.index)  # all units
            unit_indices = np.arange(self.n_units)
        # Run a subset of units
        elif isinstance(unit_index, (list)):
            unit_indices = np.array(unit_index)
            self.n_units = len(unit_indices)
        # Run one unit
        elif isinstance(unit_index, (int)):
            unit_indices = np.array([unit_index])
            self.n_units = len(unit_indices)
        else:
            raise AssertionError(
                "unit_index must be None, an integer or list, aborting..."
            )
        if isinstance(unit_indices, (int, np.int32, np.int64)):
            unit_indices = np.array([unit_indices])
        self.unit_indices = np.atleast_1d(unit_indices)

    def link_gcs_to_vs(self, vs):
        """
        Endows ganglion cells with stimulus/pixel space coordinates.

        Here we make a new dataframe df_stimpix where everything is in pixels
        """

        df_stimpix = pd.DataFrame()
        df = self.df
        # Endow RGCs with pixel coordinates.
        pixspace_pos = np.array(
            [vs._vspace_to_pixspace(gc.x_deg, gc.y_deg) for index, gc in df.iterrows()]
        )
        if self.DoG_model in ["ellipse_fixed", "circular"]:
            pixspace_coords = pd.DataFrame(
                {"q_pix": pixspace_pos[:, 0], "r_pix": pixspace_pos[:, 1]}
            )
        elif self.DoG_model == "ellipse_independent":
            # We need to here compute the pixel coordinates of the surround as well.
            # It would be an overkill to make pos_ecc_mm, pos_polar_deg forthe surround as well,
            # so we'll just compute the surround's pixel coordinates relative to the center's pixel coordinates.
            # 1) Get the experimental pixel coordinates of the center
            xoc = df.xoc_pix.values
            yoc = df.yoc_pix.values
            # 2) Get the experimental pixel coordinates of the surround
            xos = df.xos_pix.values
            yos = df.yos_pix.values
            # 3) Compute the experimental pixel coordinates of the surround relative to the center
            x_diff = xos - xoc
            y_diff = yos - yoc
            # 4) Tranform the experimental pixel coordinate difference to mm
            mm_per_exp_pix = self.context.apricot_metadata["data_microm_per_pix"] / 1000
            x_diff_mm = x_diff * mm_per_exp_pix
            y_diff_mm = y_diff * mm_per_exp_pix
            # 5) Transform the mm difference to degrees difference
            x_diff_deg = x_diff_mm * self.deg_per_mm
            y_diff_deg = y_diff_mm * self.deg_per_mm
            # 6) Scale the degrees difference with eccentricity scaling factor and
            # add to the center's degrees coordinates
            x_deg_s = x_diff_deg * df.gc_scaling_factors + df.x_deg
            y_deg_s = y_diff_deg * df.gc_scaling_factors + df.y_deg
            # 7) Transform the degrees coordinates to pixel coordinates in stimulus space
            pixspace_pos_s = np.array(
                [vs._vspace_to_pixspace(x, y) for x, y in zip(x_deg_s, y_deg_s)]
            )

            pixspace_coords = pd.DataFrame(
                {
                    "q_pix": pixspace_pos[:, 0],
                    "r_pix": pixspace_pos[:, 1],
                    "q_pix_s": pixspace_pos_s[:, 0],
                    "r_pix_s": pixspace_pos_s[:, 1],
                }
            )

        # Scale RF to stimulus pixel space.
        if self.DoG_model == "ellipse_fixed":
            df_stimpix["semi_xc"] = df.semi_xc_deg * vs.pix_per_deg
            df_stimpix["semi_yc"] = df.semi_yc_deg * vs.pix_per_deg
            df_stimpix["orient_cen_rad"] = df.orient_cen_rad
            df_stimpix["relat_sur_diam"] = df.relat_sur_diam
        elif self.DoG_model == "ellipse_independent":
            df_stimpix["semi_xc"] = df.semi_xc_deg * vs.pix_per_deg
            df_stimpix["semi_yc"] = df.semi_yc_deg * vs.pix_per_deg
            df_stimpix["semi_xs"] = df.semi_xs_deg * vs.pix_per_deg
            df_stimpix["semi_ys"] = df.semi_ys_deg * vs.pix_per_deg
            df_stimpix["orient_cen_rad"] = df.orient_cen_rad
            df_stimpix["orient_sur_rad"] = df.orient_sur_rad
        elif self.DoG_model == "circular":
            df_stimpix["rad_c"] = df.rad_c_deg * vs.pix_per_deg
            df_stimpix["rad_s"] = df.rad_s_deg * vs.pix_per_deg
            df_stimpix["orient_cen_rad"] = 0.0

        df_stimpix = pd.concat([df_stimpix, pixspace_coords], axis=1)
        pix_df = deepcopy(df_stimpix)

        if self.spatial_model == "FIT":
            # Define spatial filter sidelength in pixels in stimulus space
            # based on angular resolution and widest semimajor axis.
            # We use the general rule that the sidelength should be at least 5 times the SD
            # Remove exceptionally large values (n_sd) from surround before calculating sidelen.
            # This saves a lot of memory and computation time downstream.
            n_sd = 3  # SD
            if self.DoG_model == "ellipse_fixed":
                cut = pix_df.relat_sur_diam.mean() + pix_df.relat_sur_diam.std() * n_sd
                pix_df.relat_sur_diam[pix_df.relat_sur_diam > cut] = 0
                rf_max_pix = max(
                    max(pix_df.semi_xc * pix_df.relat_sur_diam),
                    max(pix_df.semi_yc * pix_df.relat_sur_diam),
                )

            elif self.DoG_model == "ellipse_independent":
                cut = pix_df.semi_xs.mean() + pix_df.semi_xs.std() * n_sd
                pix_df.semi_xs[pix_df.semi_xs > cut] = 0
                cut = pix_df.semi_ys.mean() + pix_df.semi_ys.std() * n_sd
                pix_df.semi_ys[pix_df.semi_ys > cut] = 0
                rf_max_pix = max(max(pix_df.semi_xs), max(pix_df.semi_ys))

            elif self.DoG_model == "circular":
                cut = pix_df.rad_c.mean() + pix_df.rad_c.std() * n_sd
                pix_df.rad_s[pix_df.rad_s > cut] = 0
                rf_max_pix = max(pix_df.rad_s)

            self.spatial_filter_sidelen = 5 * int(rf_max_pix)

            # Sidelength always odd number
            if self.spatial_filter_sidelen % 2 == 0:
                self.spatial_filter_sidelen += 1

        elif self.spatial_model == "VAE":
            # Fixed spatial filter sidelength according to VAE RF pixel resolution
            # at given eccentricity (calculated at construction)
            stim_um_per_pix = 1000 / (vs.pix_per_deg * self.deg_per_mm)
            # Same metadata in all units, thus index [0]
            self.spatial_filter_sidelen = int(
                (self.um_per_pix / stim_um_per_pix) * self.sidelen_pix
            )

        # # Set center and surround midpoints in new pixel space
        pix_scale = self.spatial_filter_sidelen / self.sidelen_pix
        if self.DoG_model in ["ellipse_fixed", "circular"]:
            xoc = xos = df.xoc_pix.values * pix_scale
            yoc = yos = df.yoc_pix.values * pix_scale
        elif self.DoG_model == "ellipse_independent":
            xoc = df.xoc_pix.values * pix_scale
            yoc = df.yoc_pix.values * pix_scale
            xos = df.xos_pix.values * pix_scale
            yos = df.yos_pix.values * pix_scale
        df_stimpix["xoc_pix"] = xoc
        df_stimpix["yoc_pix"] = yoc
        df_stimpix["xos_pix"] = xos
        df_stimpix["yos_pix"] = yos

        df_stimpix["ampl_c"] = df.ampl_c_norm
        df_stimpix["ampl_s"] = df.ampl_s_norm

        self.df_stimpix = df_stimpix

        self.microm_per_pix = (1 / self.deg_per_mm) / vs.pix_per_deg * 1000
        self.temporal_filter_len = int(self.data_filter_duration / (1000 / vs.fps))


class VisualSignal(Printable):
    """
    Class containing information associated with visual signal
    passing through the retina. This includes the stimulus video,
    its transformations, the generator potential and spikes.
    """

    def __init__(
        self,
        my_stimulus_options,
        stimulus_center,
        load_stimulus_from_videofile,
        simulation_dt,
        deg_per_mm,
        stimulus_video=None,
    ):
        # Parameters directly passed to the constructor
        self.my_stimulus_options = my_stimulus_options
        self.stimulus_center = stimulus_center
        self.load_stimulus_from_videofile = load_stimulus_from_videofile
        self.deg_per_mm = deg_per_mm

        # Default value for computed variable
        self.stimulus_video = stimulus_video

        # Load stimulus video if not already loaded
        if self.stimulus_video is None:
            self.video_file_name = self.my_stimulus_options["stimulus_video_name"]
            self.stimulus_video = self.load_stimulus_from_videofile(
                self.video_file_name
            )

            self.options_from_file = self.stimulus_video.options
            self.stimulus_width_pix = self.options_from_file["image_width"]
            self.stimulus_height_pix = self.options_from_file["image_height"]
            self.pix_per_deg = self.options_from_file["pix_per_deg"]
            self.fps = self.options_from_file["fps"]

            self.stimulus_width_deg = self.stimulus_width_pix / self.pix_per_deg
            self.stimulus_height_deg = self.stimulus_height_pix / self.pix_per_deg

        cen_x = self.options_from_file["center_pix"][0]
        cen_y = self.options_from_file["center_pix"][1]
        self.photodiode_response = self.stimulus_video.frames[:, cen_y, cen_x]

        # Assertions to ensure stimulus video properties match expected parameters
        assert (
            self.stimulus_video.video_width == self.stimulus_width_pix
            and self.stimulus_video.video_height == self.stimulus_height_pix
        ), "Check that stimulus dimensions match those of the mosaic"
        assert (
            self.stimulus_video.fps == self.fps
        ), "Check that stimulus frame rate matches that of the mosaic"
        assert (
            self.stimulus_video.pix_per_deg == self.pix_per_deg
        ), "Check that stimulus resolution matches that of the mosaic"
        # assert (
        #     np.min(self.stimulus_video.frames) >= 0
        #     and np.max(self.stimulus_video.frames) <= 255
        # ), "Stimulus pixel values must be between 0 and 255"

        self.video_dt = (1 / self.stimulus_video.fps) * b2u.second  # input
        self.stim_len_tp = self.stimulus_video.video_n_frames
        self.duration = self.stim_len_tp * self.video_dt
        self.simulation_dt = simulation_dt * b2u.second  # output
        self.tvec = range(self.stim_len_tp) * self.video_dt

    def _vspace_to_pixspace(self, x, y):
        """
        Converts visual space coordinates (in degrees; x=eccentricity, y=elevation) to pixel space coordinates.
        In pixel space, coordinates (q,r) correspond to matrix locations, ie. (0,0) is top-left.

        Parameters
        ----------
        x : float
            eccentricity (deg)
        y : float
            elevation (deg)

        Returns
        -------
        q : float
            pixel space x-coordinate
        r : float
            pixel space y-coordinate
        """

        video_width_px = self.stimulus_width_pix  # self.stimulus_video.video_width
        video_height_px = self.stimulus_height_pix  # self.stimulus_video.video_height
        pix_per_deg = self.pix_per_deg  # self.stimulus_video.pix_per_deg

        # 1) Set the video center in visual coordinates as origin
        # 2) Scale to pixel space. Mirror+scale in y axis due to y-coordinate running top-to-bottom in pixel space
        # 3) Move the origin to video center in pixel coordinates
        q = pix_per_deg * (x - self.stimulus_center.real) + (video_width_px / 2)
        r = -pix_per_deg * (y - self.stimulus_center.imag) + (video_height_px / 2)

        return q, r


class SimulateRetina(RetinaMath):
    def __init__(self, context, data_io, viz, project_data) -> None:
        self._context = context.set_context(self)
        self._data_io = data_io
        self._viz = viz
        self._project_data = project_data

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def viz(self):
        return self._viz

    @property
    def project_data(self):
        return self._project_data

    def _get_crop_pixels(self, gcs, unit_index):
        """
        Get pixel coordinates for a stimulus crop matching the spatial filter size.

        Parameters
        ----------
        unit_index : int or array-like of int
            Index or indices of the unit(s) for which to retrieve crop coordinates.

        Returns
        -------
        qmin, qmax, rmin, rmax : int or tuple of int
            Pixel coordinates defining the crop's bounding box.
            qmin and qmax specify the range in the q-dimension (horizontal),
            and rmin and rmax specify the range in the r-dimension (vertical).

        Notes
        -----
        The crop size is determined by the spatial filter's sidelength.

        """

        if isinstance(unit_index, (int, np.int32, np.int64)):
            unit_index = np.array([unit_index])
        df_stimpix = gcs.df_stimpix.iloc[unit_index]
        q_center = np.round(df_stimpix.q_pix).astype(int).values
        r_center = np.round(df_stimpix.r_pix).astype(int).values

        # crops have width = height
        side_halflen = (gcs.spatial_filter_sidelen - 1) // 2

        qmin = q_center - side_halflen
        qmax = q_center + side_halflen
        rmin = r_center - side_halflen
        rmax = r_center + side_halflen

        return qmin, qmax, rmin, rmax

    def _create_spatial_filter_FIT(self, gcs, unit_index):
        """
        Creates the spatial component of the spatiotemporal filter

        Parameters
        ----------
        unit_index : int
            Index of the unit in the df

        Returns
        -------
        spatial_filter : np.ndarray
            Spatial filter for the given unit
        """

        offset = 0.0
        s = gcs.spatial_filter_sidelen

        gc = gcs.df_stimpix.iloc[unit_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(gcs, unit_index)

        x_grid, y_grid = np.meshgrid(
            np.arange(qmin, qmax + 1, 1), np.arange(rmin, rmax + 1, 1)
        )
        # spatial_kernel is here 1-dim vector
        if gcs.DoG_model == "ellipse_fixed":
            spatial_kernel = self.DoG2D_fixed_surround(
                (x_grid, y_grid),
                gc.ampl_c,
                gc.q_pix,
                gc.r_pix,
                gc.semi_xc,
                gc.semi_yc,
                gc.orient_cen_rad,
                gc.ampl_s,
                gc.relat_sur_diam,
                offset,
            )
        elif self.DoG_model == "ellipse_independent":
            spatial_kernel = self.DoG2D_independent_surround(
                (x_grid, y_grid),
                gc.ampl_c,
                gc.q_pix,
                gc.r_pix,
                gc.semi_xc,
                gc.semi_yc,
                gc.orient_cen_rad,
                gc.ampl_s,
                gc.q_pix_s,
                gc.r_pix_s,
                gc.semi_xs,
                gc.semi_ys,
                gc.orient_sur_rad,
                offset,
            )
        elif self.DoG_model == "circular":
            spatial_kernel = self.DoG2D_circular(
                (x_grid, y_grid),
                gc.ampl_c,
                gc.q_pix,
                gc.r_pix,
                gc.rad_c,
                gc.ampl_s,
                gc.rad_s,
                offset,
            )

        spatial_kernel = np.reshape(spatial_kernel, (s, s))

        return spatial_kernel

    def _create_spatial_filter_VAE(self, gcs, unit_index):
        """
        Creates the spatial component of the spatiotemporal filter

        Parameters
        ----------
        unit_index : int
            Index of the unit in the df

        Returns
        -------
        spatial_filter : np.ndarray
            Spatial filter for the given unit
        """
        s = gcs.spatial_filter_sidelen

        spatial_kernel = resize(
            gcs.spat_rf[unit_index, :, :], (s, s), anti_aliasing=True
        )

        return spatial_kernel

    def _create_temporal_filter(self, gcs, unit_index):
        """
        Creates the temporal component of the spatiotemporal filter. Linear fixed-sum of two lowpass filters.

        Parameters
        ----------
        unit_index : int
            Index of the unit in the df

        Returns
        -------
        temporal_filter : np.ndarray
            Temporal filter for the given unit
        """

        filter_params = gcs.df.iloc[unit_index][["n", "p1", "p2", "tau1", "tau2"]]

        tvec = np.linspace(0, gcs.data_filter_duration, gcs.temporal_filter_len)
        temporal_filter = self.diff_of_lowpass_filters(tvec, *filter_params)

        # Amplitude will be scaled by abs(sum()) of the temporal_filter
        temporal_filter = temporal_filter / np.sum(np.abs(temporal_filter))

        return temporal_filter

    def _get_spatially_cropped_video(self, vs, gcs, contrast=True, reshape=False):
        """
        Crops the video to the surroundings of the specified Retinal ganglion cells (RGCs).

        The function works by first determining the pixel range to be cropped for each unit
        in unit_indices, and then selecting those pixels from the original video. The cropping
        is done for each frame of the video. If the contrast option is set to True, the video
        is also rescaled to have pixel values between -1 and 1.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object containing the transduction cascade from stimulus video to RGC unit spike response.

        gcs : ReceptiveFields
            The receptive fields object containing the RGCs and their parameters.

        contrast : bool, optional
            If True, the video is rescaled to have pixel values between -1 and 1.
            This is the Weber constrast ratio, set for the stimulus.
            By default, this option is set to True.

        reshape : bool, optional
            If True, the function will reshape the output array to have dimensions
            (number_of_cells, number_of_pixels, number_of_time_points).
            By default, this option is set to False.

        Returns
        -------
        vs.stimulus_cropped : np.ndarray
            The cropped video. The dimensions of the array are
            (number_of_cells, number_of_pixels_w, number_of_pixels_h, number_of_time_points)
            if reshape is False, and
            (number_of_cells, number_of_pixels, number_of_time_points)
            if reshape is True.

        Notes
        -----
        qmin and qmax specify the range in the q-dimension (horizontal),
        and rmin and rmax specify the range in the r-dimension (vertical).
        """

        sidelen = gcs.spatial_filter_sidelen
        unit_indices = gcs.unit_indices.copy()
        video_copy = vs.stimulus_video.frames.copy()
        video_copy = np.transpose(
            video_copy, (1, 2, 0)
        )  # Original frames are now [height, width, time points]
        video_copy = video_copy[np.newaxis, ...]  # Add new axis for broadcasting

        qmin, qmax, rmin, rmax = self._get_crop_pixels(gcs, unit_indices)

        # Adjust the creation of r_indices and q_indices for proper broadcasting
        r_indices = np.arange(sidelen) + rmin[:, np.newaxis]
        q_indices = np.arange(sidelen) + qmin[:, np.newaxis]

        # Ensure r_indices and q_indices are repeated for each unit
        r_indices = r_indices.reshape(-1, 1, sidelen, 1)
        q_indices = q_indices.reshape(-1, sidelen, 1, 1)

        # Broadcasting to create compatible shapes for indexing
        r_matrix, q_matrix = np.broadcast_arrays(r_indices, q_indices)
        time_points_indices = np.arange(video_copy.shape[-1])

        # Use advanced indexing for selecting pixels
        stimulus_cropped = video_copy[0, r_matrix, q_matrix, time_points_indices]

        if contrast:
            stimulus_cropped = stimulus_cropped / 127.5 - 1.0

        if reshape:
            n_frames = vs.stimulus_video.frames.shape[0]
            stimulus_cropped = stimulus_cropped.reshape(
                (gcs.n_units, sidelen**2, n_frames)
            )

        vs.stimulus_cropped = stimulus_cropped

        return vs

    def _get_uniformity_index(self, vs, gcs):
        """
        Calculate the uniformity index for retinal ganglion cell receptive fields.

        This function computes the uniformity index which quantifies the evenness
        of the distribution of receptive field centers over the visual stimulus area,
        using Delaunay triangulation to estimate the total area covered by receptive
        fields.

        Parameters
        ----------

        Returns
        -------
        dict
            A dictionary containing:
            - 'uniformify_index': The calculated uniformity index.
            - 'total_region': Binary mask indicating the total region covered by
            the receptive fields after Delaunay triangulation.
            - 'unity_region': Binary mask indicating regions where exactly one
            receptive field is present.
            - 'unit_region': The sum of center regions for all units.

        """
        height = vs.stimulus_height_pix
        width = vs.stimulus_width_pix
        unit_indices = gcs.unit_indices.copy()

        qmin, qmax, rmin, rmax = self._get_crop_pixels(gcs, unit_indices)

        stim_region = np.zeros((gcs.n_units, height, width), dtype=np.int32)
        center_region = np.zeros((gcs.n_units, height, width), dtype=np.int32)

        # Create the r and q indices for each unit, ensure they're integer type
        sidelen = gcs.spatial_filter_sidelen
        r_indices = (
            (np.arange(sidelen) + rmin[:, np.newaxis])
            .astype(int)
            .reshape(-1, 1, sidelen)
        )
        q_indices = (
            (np.arange(sidelen) + qmin[:, np.newaxis])
            .astype(int)
            .reshape(-1, sidelen, 1)
        )

        # Create r_matrix and q_matrix by broadcasting r_indices and q_indices
        r_matrix, q_matrix = np.broadcast_arrays(r_indices, q_indices)

        # create a unit index array
        unit_region_idx = np.arange(gcs.n_units).astype(np.int32).reshape(-1, 1, 1)

        # expand the indices arrays to the shape of r_matrix and q_matrix using broadcasting
        unit_region_idx = unit_region_idx + np.zeros_like(r_matrix, dtype=np.int32)

        # use the index arrays to select the elements from video_copy
        stim_region[unit_region_idx, r_matrix, q_matrix] = 1

        center_masks = gcs.center_masks_flat.copy()
        center_masks = center_masks.astype(bool).reshape(
            (gcs.n_units, sidelen, sidelen)
        )

        center_region[
            unit_region_idx * center_masks,
            r_matrix * center_masks,
            q_matrix * center_masks,
        ] = 1

        unit_region = np.sum(center_region, axis=0)

        # Delaunay triangulation for the total region
        gc = gcs.df_stimpix.iloc[unit_indices]
        q_center = np.round(gc.q_pix).astype(int).values
        r_center = np.round(gc.r_pix).astype(int).values

        # Create points array for Delaunay triangulation from r_center and q_center
        points = np.vstack((q_center, r_center)).T  # Shape should be (N, 2)

        # Perform Delaunay triangulation
        tri = Delaunay(points)

        # Initialize total area
        total_area = 0
        delaunay_mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate the area of each triangle and sum it up
        for triangle in tri.simplices:
            # Get the vertices of the triangle
            vertices = points[triangle]

            # Use the vertices to calculate the area of the triangle
            # Area formula for triangles given coordinates: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            total_area += 0.5 * abs(
                vertices[0, 0] * (vertices[1, 1] - vertices[2, 1])
                + vertices[1, 0] * (vertices[2, 1] - vertices[0, 1])
                + vertices[2, 0] * (vertices[0, 1] - vertices[1, 1])
            )

            # Get the bounding box of the triangle to minimize the area to check
            min_x = np.min(vertices[:, 0])
            max_x = np.max(vertices[:, 0])
            min_y = np.min(vertices[:, 1])
            max_y = np.max(vertices[:, 1])

            # Generate a grid of points representing pixels in the bounding box
            x_range = np.arange(min_x, max_x + 1)
            y_range = np.arange(min_y, max_y + 1)
            grid_x, grid_y = np.meshgrid(x_range, y_range)

            # Use the points in the grid to check if they are inside the triangle
            grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
            indicator = tri.find_simplex(grid_points) >= 0

            # Reshape the indicator back to the shape of the bounding box grid
            indicator = indicator.reshape(grid_x.shape)

            # Place the indicator in the mask image
            delaunay_mask[min_y : max_y + 1, min_x : max_x + 1] = np.logical_or(
                delaunay_mask[min_y : max_y + 1, min_x : max_x + 1], indicator
            )

        unity_region = (unit_region * delaunay_mask) == 1

        uniformify_index = np.sum(unity_region) / np.sum(delaunay_mask)

        uniformify_data = {
            "uniformify_index": uniformify_index,
            "total_region": delaunay_mask,
            "unity_region": unity_region,
            "unit_region": unit_region,
            "mask_threshold": gcs.mask_threshold,
        }

        self.project_data.simulate_retina["uniformify_data"] = uniformify_data

    def _create_temporal_signal_cg(
        self, tvec, svec, dt, params, device, show_impulse=False, impulse_contrast=1.0
    ):
        """
        Contrast gain control implemented in temporal domain according to Victor_1987_JPhysiol
        """
        # Henri aloita tästä

        Tc = torch.tensor(
            15.0, device=device
        )  # 15  # Time constant for dynamical variable c(t), ms. Victor_1987_JPhysiol

        # parameter_names for parasol gain control ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
        NL = params[0]
        NL = NL.to(torch.int)
        TL = params[1]
        HS = params[2]
        T0 = params[3]
        Chalf = params[4]
        D = params[5]

        ### Low pass filter ###

        # Calculate the low-pass impulse response function.
        h = (
            (1 / torch.math.factorial(NL))
            * (tvec / TL) ** (NL - 1)
            * torch.exp(-tvec / TL)
        )

        # Convolving two signals involves "flipping" one signal and then sliding it
        # across the other signal. PyTorch, however, does not flip the kernel, so we
        # need to do it manually.
        h_flipped = torch.flip(h, dims=[0])

        # Scale the impulse response to have unit area. This normalizes the effect of video dt.
        h_flipped = h_flipped / torch.sum(h_flipped)

        c_t = torch.tensor(0.0, device=device)
        if show_impulse is True:
            svec = svec.to(dtype=torch.float64)

            c_t_imp = torch.tensor(impulse_contrast, device=device)
            c_t = c_t_imp

        # Padding is necessary for the convolution operation to work properly.
        # Calculate padding size
        padding_size = len(tvec) - 1

        # Pad the stimulus
        svec_padded = torch.nn.functional.pad(
            svec.unsqueeze(0).unsqueeze(0), (padding_size, 0), mode="replicate"
        )

        # Convolve the stimulus with the flipped kernel
        x_t_vec = torch.nn.functional.conv1d(
            svec_padded,
            h_flipped.view(1, 1, -1),
            padding=0,
        ).squeeze()

        # Henri aloita tästä

        ### High pass stages ###
        y_t = torch.tensor(0.0, device=device)
        yvec = torch.zeros(len(tvec), device=device)
        Ts_t = T0 / (1 + c_t / Chalf)
        for idx in torch.range(1, len(tvec) - 1, dtype=torch.int):
            y_t = y_t + dt * (
                (-y_t / Ts_t)  # Ts**2 ?
                + (x_t_vec[idx] - x_t_vec[idx - 1]) / dt
                + (((1 - HS) * x_t_vec[idx]) / Ts_t)
            )
            Ts_t = T0 / (1 + c_t / Chalf)
            c_t = c_t + dt * ((torch.abs(y_t) - c_t) / Tc)
            yvec[idx] = y_t

            if show_impulse is True:
                c_t = c_t_imp

        if show_impulse is True:
            return yvec

        # Add delay
        D_tp = torch.round(D / dt).to(dtype=torch.int)
        generator_potential = torch.zeros(len(tvec) + D_tp).to(device)
        generator_potential[D_tp:] = yvec

        return generator_potential

    def _generator_to_firing_rate_dynamic(
        self,
        params_all,
        generator_potentials,
    ):
        assert (
            params_all.shape[0] == generator_potentials.shape[0]
        ), "Number of units in params_all and generator_potentials must match, aborting..."

        tonic_drive = params_all["tonic_drive"]
        # Expanding tonic_drive to match the shape of generator_potentials
        tonic_drive = np.expand_dims(tonic_drive, axis=1)
        # Apply nonlinearity
        # tonic_drive**2 is added to mimick spontaneous firing rates
        firing_rates = np.maximum(generator_potentials + tonic_drive**2, 0)

        return firing_rates

    def _generator_to_firing_rate_noise(self, vs, gcs, cones, n_trials):
        """
        Generates cone noise, scales it with mean firing rates. Multiplies the generator
        potentials with gain and finally adds the firing rates generated by the cone noise
        to the light-induced firing rates.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object containing the transduction cascade from stimulus video
            to RGC unit spike response.
        gcs : ReceptiveFields
            The receptive fields object containing the RGCs and their parameters.
        n_trials : int
            The number of trials to simulate.

        Returns
        -------
        ndarray
            The firing rates after adding noise and applying gain and mean firing rates adjustments.
        """
        params_all = gcs.df.loc[gcs.unit_indices]
        assert (
            params_all.shape[0] == vs.generator_potentials.shape[0]
        ), "Number of units in params_all and generator_potentials must match, aborting..."

        cones_to_gcs_weights = cones.cones_to_gcs_weights
        params = cones.cone_noise_parameters

        cones_to_gcs_weights = cones_to_gcs_weights[:, gcs.unit_indices]
        n_cones = cones_to_gcs_weights.shape[0]

        ret_file_npz = self.data_io.get_data(self.context.my_retina["ret_file"])
        cone_frequency_data = ret_file_npz["cone_frequency_data"]
        cone_power_data = ret_file_npz["cone_power_data"]
        self.cone_interp_response = self.interpolation_function(
            cone_frequency_data, cone_power_data
        )
        self.cone_noise_wc = cones.cone_general_params["cone_noise_wc"]

        # Normalize weights by columns (ganglion cells)
        weights_norm = cones_to_gcs_weights / np.sum(cones_to_gcs_weights, axis=0)

        # Make independent cone noise for multiple trials
        if n_trials > 1:
            for trial in range(n_trials):
                # cone_noise = _create_cone_noise(vs.tvec, n_cones, *params)
                if trial == 0:

                    gc_noise = vs.cone_noise @ weights_norm
                else:
                    gc_noise = np.concatenate(
                        (gc_noise, vs.cone_noise @ weights_norm), axis=1
                    )
        elif vs.generator_potentials.shape[0] > 1:
            # cone_noise = _create_cone_noise(vs.tvec, n_cones, *params)
            gc_noise = vs.cone_noise @ weights_norm

        # Normalize noise to have one mean and unit sd at the noise data frequencies
        gc_noise_norm = 1 + (gc_noise - gc_noise.mean()) / np.std(gc_noise, axis=0)

        # Manual multiplier from conf file
        magn = cones.cone_general_params["cone_noise_magnitude"]
        gc_noise_mean = params_all.Mean.values
        firing_rates_cone_noise = gc_noise_norm.T * gc_noise_mean[:, np.newaxis] * magn

        gc_gain = params_all.A.values
        firing_rates_light = vs.generator_potentials * gc_gain[:, np.newaxis]

        # Truncating nonlinearity
        firing_rates = np.maximum(firing_rates_light + firing_rates_cone_noise, 0)

        vs.firing_rates = firing_rates

        return vs

    def _create_temporal_signal(
        self, tvec, svec, dt, params, h, device, show_impulse=False
    ):
        """
        Dynamic temporal signal for midget units
        """

        # parameter name order for midget ["NL", "NLTL", "TS", "HS", "D", "A"]
        HS = params[3]
        TS = params[2]
        D = params[4]

        # Calculate padding size
        padding_size = len(tvec) - 1

        # Pad the stimulus
        svec_padded = torch.nn.functional.pad(
            svec.unsqueeze(0).unsqueeze(0), (padding_size, 0), mode="replicate"
        )

        # Convolve the stimulus with the flipped kernel
        x_t_vec = (
            torch.nn.functional.conv1d(
                svec_padded,
                h.view(1, 1, -1),
                padding=0,
            ).squeeze()
            * dt
        )
        ### High pass stages ###
        y_t = torch.tensor(0.0, device=device)
        yvec = tvec * torch.tensor(0.0, device=device)
        for idx in torch.range(1, len(tvec) - 1, dtype=torch.int):
            y_t = y_t + dt * (
                (-y_t / TS)
                + (x_t_vec[idx] - x_t_vec[idx - 1]) / dt
                + (((1 - HS) * x_t_vec[idx]) / TS)
            )
            yvec[idx] = y_t

        if show_impulse is True:
            return yvec

        # Add delay
        D_tp = torch.round(D / dt).to(dtype=torch.int)
        generator_potential = torch.zeros(len(tvec) + D_tp).to(device)
        generator_potential[D_tp:] = yvec

        return generator_potential

    def _create_lowpass_response(self, tvec, params):
        """
        Lowpass filter kernel for convolution for midget units
        """
        # parameter name order for midget ["NL", "NLTL", "TS", "HS", "D", "A"]
        NL = params[0]
        # Chance NL to dtype torch integer
        NL = NL.to(torch.int)
        # TL = NLTL / NL
        TL = params[1] / params[0]

        ### Low pass filter ###
        h = (
            (1 / torch.math.factorial(NL))
            * (tvec / TL) ** (NL - 1)
            * torch.exp(-tvec / TL)
        )

        # With large values of NL, the show_impulse response function runs out of humour (becomes inf or nan)
        # at later latencies. We can avoid this by setting these inf and nan values of h to zero.
        h[torch.isinf(h)] = 0
        h[torch.isnan(h)] = 0

        # Convolving two signals involves "flipping" one signal and then sliding it
        # across the other signal. PyTorch, however, does not flip the kernel, so we
        # need to do it manually.
        h_flipped = torch.flip(h, dims=[0])

        # Scale the impulse response to have unit area. This normalizes the effect of video dt.
        h_flipped = h_flipped / torch.sum(h_flipped)

        return h_flipped

    def _show_surround_and_exit(self, center_surround_filters, spatial_filters):
        """
        Internal QA image for surrounds. Call by activating at _create_dynamic_contrast

        center_surround_filters : array
            Arrays with not-surrounds masked to zero
        spatial_filters : array
            Original spatial receptive fields
        """
        n_img = center_surround_filters.shape[0]
        side_img = self.spatial_filter_sidelen
        tp_img = center_surround_filters.shape[-1] // 2
        center_surround_filters_rs = np.reshape(
            center_surround_filters, (n_img, side_img, side_img, -1)
        )
        spatial_filters_rs = np.reshape(spatial_filters, (n_img, side_img, side_img))

        for this_img in range(n_img):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(spatial_filters_rs[this_img, :, :])
            axs[0].set_title("Original RF")
            axs[1].imshow(center_surround_filters_rs[this_img, :, :, tp_img])
            axs[1].set_title("Surround (dark)")
            plt.show()
        sys.exit()

    def _create_dynamic_contrast(self, vs, gcs):
        """
        Create dynamic contrast signal by multiplying the stimulus with the spatial filter
        masks are used for midget units, where center and surround have distinct dynamics.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object containing the transduction cascade from stimulus video to RGC unit spike response.
        gcs : ReceptiveFields
            The receptive fields object containing the RGCs and their parameters.

        Returns
        -------
        vs : VisualSignal
            The visual signal object containing the dynamic contrast signal.
        """

        # Reshape masks and spatial_filters to match the dimensions of stimulus_cropped
        spatial_filters = gcs.spatial_filters_flat.copy()
        spatial_filters_reshaped = np.expand_dims(spatial_filters, axis=2)

        # victor_1987_JPhysiol: input to model is s(t)), the signed Weber contrast at the centre.
        # However, we assume that the surround suppression is early (horizontal units) and linear,
        # so we approximate s(t) = RF * stimulus
        # TODO Check svec dynamic range, should be [-1, 1] for contrast stimuli
        if gcs.gc_type == "parasol":

            # This is the stimulus contrast viewed through spatial rf filter, and summed over spatial dimension.
            # The np.einsum provides a fast and memory-efficient way to do this.
            # i is the unit, j is the spatial dimension, k is the time dimension
            vs.svecs = np.einsum(
                "ijk,ijk->ik", spatial_filters_reshaped, vs.stimulus_cropped
            )

        elif gcs.gc_type == "midget":
            masks_sur = gcs.surround_masks_flat[:, :, np.newaxis]
            vs.svecs_sur = np.einsum(
                "ijk,ijk->ik",
                spatial_filters_reshaped * masks_sur,
                vs.stimulus_cropped,
            )

            masks_cen = gcs.center_masks_flat[:, :, np.newaxis]
            vs.svecs_cen = np.einsum(
                "ijk,ijk->ik",
                spatial_filters_reshaped * masks_cen,
                vs.stimulus_cropped,
            )

        return vs

    def _get_impulse_response(self, gcs, contrasts_for_impulse, video_dt):
        """
        Provides impulse response for distinct ganglion cell and response types.
        Much of the run_cells code is copied here, but with the stimulus replaced by an impulse.
        """

        # Set filter duration the same as in Apricot data
        total_duration = (
            gcs.data_filter_timesteps * (1000 / gcs.data_filter_fps) * b2u.ms
        )
        stim_len_tp = int(np.round(total_duration / video_dt))
        tvec = range(stim_len_tp) * video_dt

        # Dummy kernel for show_impulse response
        svec = np.zeros(len(tvec))
        dt = video_dt / b2u.ms
        start_delay = 100  # ms
        idx_100_ms = int(np.round(start_delay / dt))
        svec[idx_100_ms] = 1.0

        if gcs.response_type == "off":
            # Spatial OFF filters have been inverted to max upwards for construction of RFs.
            svec = -svec

        stim_len_tp = len(tvec)
        # Append to impulse_to_show a key str(contrast) for each contrast,
        # holding empty array for impulse response

        assert contrasts_for_impulse is not None and isinstance(
            contrasts_for_impulse, list
        ), "Impulse must specify contrasts as list, aborting..."

        yvecs = np.empty((gcs.n_units, len(contrasts_for_impulse), stim_len_tp))
        if gcs.temporal_model == "dynamic":
            # cpu on purpose, less issues, very fast anyway
            device = torch.device("cpu")
            svec_t = torch.tensor(svec, device=device)
            tvec_t = torch.tensor(tvec / b2u.ms, device=device)
            video_dt_t = torch.tensor(video_dt / b2u.ms, device=device)

            for idx, this_cell_index in enumerate(gcs.unit_indices):
                if gcs.gc_type == "parasol":
                    # Get unit params
                    columns = ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
                    params = gcs.df.loc[this_cell_index, columns].values
                    params_t = torch.tensor(params, device=device)

                    for contrast in contrasts_for_impulse:
                        yvec = self._create_temporal_signal_cg(
                            tvec_t,
                            svec_t,
                            video_dt_t,
                            params_t,
                            device,
                            show_impulse=True,
                            impulse_contrast=contrast,
                        )
                        yvecs[idx, contrasts_for_impulse.index(contrast), :] = yvec

                elif gcs.gc_type == "midget":
                    columns_cen = [
                        "NL_cen",
                        "NLTL_cen",
                        "TS_cen",
                        "HS_cen",
                        "D_cen",
                        "A_cen",
                    ]
                    params_cen = gcs.df.loc[this_cell_index, columns_cen].values
                    params_cen_t = torch.tensor(params_cen, device=device)
                    lp_cen = self._create_lowpass_response(tvec_t, params_cen_t)

                    columns_sur = [
                        "NL_sur",
                        "NLTL_sur",
                        "TS_sur",
                        "HS_sur",
                        "D_cen",
                        "A_sur",
                    ]
                    params_sur = gcs.df.loc[this_cell_index, columns_sur].values
                    params_sur_t = torch.tensor(params_sur, device=device)
                    lp_sur = self._create_lowpass_response(tvec_t, params_sur_t)
                    h_cen = lp_cen / torch.sum(lp_cen + lp_sur)

                    yvec = self._create_temporal_signal(
                        tvec_t,
                        svec_t,
                        video_dt_t,
                        params_sur_t,
                        h_cen,
                        device,
                        show_impulse=True,
                    )
                    yvecs[idx, 0, :] = yvec

        elif gcs.temporal_model == "fixed":  # Linear model
            # Amplitude will be scaled by first (positive) lowpass filter.
            gcs = self._get_linear_temporal_filters(gcs)
            yvecs = np.repeat(
                gcs.temporal_filter[:, np.newaxis, :],
                len(contrasts_for_impulse),
                axis=1,
            )
            if gcs.response_type == "off":
                # Spatial OFF filters have been inverted to max upwards for construction of RFs.
                yvecs = -yvecs

            # Shift impulse response to start_delay and cut length correspondingly
            idx_start_delay = int(np.round(start_delay / (video_dt / b2u.ms)))
            # append zeros to the start of the impulse response
            yvecs = np.pad(
                yvecs,
                ((0, 0), (0, 0), (idx_start_delay, 0)),
                mode="constant",
                constant_values=0,
            )
            # cut the impulse response to the desired length
            yvecs = yvecs[:, :, :-idx_start_delay]

        impulse_to_show = {
            "tvec": tvec / b2u.second,
            "svec": svec,
        }
        impulse_to_show["start_delay"] = start_delay
        impulse_to_show["contrasts"] = contrasts_for_impulse
        impulse_to_show["impulse_responses"] = yvecs
        impulse_to_show["Unit idx"] = list(gcs.unit_indices)
        impulse_to_show["gc_type"] = gcs.gc_type
        impulse_to_show["response_type"] = gcs.response_type
        impulse_to_show["temporal_model"] = gcs.temporal_model

        self.project_data.simulate_retina["impulse_to_show"] = impulse_to_show

    def get_w_z_coords(self, gcs):
        """
        Create w_coord, z_coord for cortical and visual coordinates, respectively

        Parameters
        ----------
        None

        Returns
        -------
        w_coord : np.ndarray
            Cortical coordinates
        z_coord : np.ndarray
            Visual coordinates
        """

        # Create w_coord, z_coord for cortical and visual coordinates, respectively
        z_coord = gcs.df["x_deg"].values + 1j * gcs.df["y_deg"].values

        visual2cortical_params = self.context.my_retina["visual2cortical_params"]
        a = visual2cortical_params["a"]
        k = visual2cortical_params["k"]
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord

    def _get_linear_temporal_filters(self, gcs):
        """
        Retrieve temporal filters for an array of units.

        This function generates temporal filters for each unit specified by the
        unit indices. The temporal filter for a specific unit is obtained by calling
        the `_create_temporal_filter` method.

        Parameters
        ----------

        Returns
        -------
        temporal_filters : ndarray
            2-D array where each row corresponds to a temporal filter of a unit. The shape is
            (len(unit_indices), self.temporal_filter_len).

        Notes
        -----
        This function depends on the following instance variables:
          - self.temporal_filter_len: an integer specifying the length of a temporal filter.
        """

        temporal_filters = np.zeros((len(gcs.unit_indices), gcs.temporal_filter_len))

        for idx, unit_index in enumerate(gcs.unit_indices):
            temporal_filters[idx, :] = self._create_temporal_filter(gcs, unit_index)

        gcs.temporal_filters = temporal_filters

        return gcs

    def _get_linear_spatiotemporal_filters(self, gcs):
        """
        Generate spatiotemporal filters for given unit indices."""

        # Assuming spatial_filters.shape = (U, N) and temporal_filters.shape = (U, T)
        spatiotemporal_filters = (
            gcs.spatial_filters_flat[:, :, None] * gcs.temporal_filters[:, None, :]
        )
        gcs.spatiotemporal_filters = spatiotemporal_filters

        return gcs

    def _get_surround_masks(self, gcs, img_stack):
        """
        Generate surround masks.

        Parameters
        ----------
        gcs : ReceptiveFields
            The receptive fields object containing the RGCs and their parameters.
        img_stack : numpy.ndarray
            3D numpy array representing a stack of images. The shape of the array should be (N, H, W).

        Returns
        -------
        numpy.ndarray
            3D numpy array of boolean masks (N, H, W). In each mask, True indicates
            a pixel is part of the contour, and False indicates it is not.
        """
        df = gcs.df_stimpix
        xo = df["xos_pix"].values
        yo = df["yos_pix"].values
        if gcs.DoG_model == "ellipse_fixed":
            semi_x = df["semi_xc"].values * df["relat_sur_diam"].values
            semi_y = df["semi_yc"].values * df["relat_sur_diam"].values
            ori = df["orient_cen_rad"].values
        elif gcs.DoG_model == "ellipse_independent":
            semi_x = df["semi_xs"].values
            semi_y = df["semi_ys"].values
            ori = df["orient_sur_rad"].values
        elif gcs.DoG_model == "circular":
            semi_x = df["rad_s"].values
            semi_y = df["rad_s"].values
            ori = df["orient_cen_rad"].values

        s = gcs.spatial_filter_sidelen
        n_sd = 2
        masks = []
        for idx, img in enumerate(img_stack):
            ellipse_mask = self.create_ellipse_mask(
                xo[idx],
                yo[idx],
                semi_x[idx] * n_sd,
                semi_y[idx] * n_sd,
                -ori[idx],
                s,
            )

            min_val = np.min(img)
            mask = img < min_val * gcs.mask_threshold
            final_mask = np.logical_and(mask, ellipse_mask)
            masks.append(final_mask)

        return np.array(masks)

    def _get_gc_spatial_filters(self, gcs):
        """
        Generate spatial filters for given unit indices.

        This function takes a list of unit indices, determines the model type,
        creates a corresponding spatial filter for each unit index based on the model,
        and then reshapes the filter to 1-D. It returns a 2-D array where each row is a
        1-D spatial filter for a corresponding unit index.

        Parameters
        ----------
        gcs.spatial_filter_sidelen : int
            The side length of a spatial filter.
        gcs.spatial_model : str
            The type of model used to generate the spatial filters. Expected values are 'FIT' or 'VAE'.
        gcs.unit_indices : list
            A list of unit indices for which to generate spatial filters.
        gcs.mask_threshold : float
            The threshold for the mask.
        gcs.response_type : str
            The type of response. Expected values are 'on' or 'off'.

        Returns
        -------
        gcs.spatial_filters_flat : ndarray
            2-D array where each row corresponds to a 1-D spatial filter or mask of a unit.
            The shape is (len(unit_indices), s**2), where s is the side length of a spatial filter.
        gcs.center_masks_flat : ndarray
            2-D array where each row corresponds to a 1-D center mask of a unit.

        Raises
        ------
        ValueError
            If the model type is neither 'FIT' nor 'VAE'.

        Notes
        -----
        This function depends on the following instance variables:
          - self.spatial_model: a string indicating the type of model used. Expected values are 'FIT' or 'VAE'.
          - self.spatial_filter_sidelen: an integer specifying the side length of a spatial filter.
        """

        s = gcs.spatial_filter_sidelen
        spatial_filters = np.zeros((gcs.n_units, s, s))
        for idx, unit_index in enumerate(gcs.unit_indices):
            if gcs.spatial_model == "FIT":
                spatial_filters[idx, ...] = self._create_spatial_filter_FIT(
                    gcs, unit_index
                )
            elif gcs.spatial_model == "VAE":
                spatial_filters[idx, ...] = self._create_spatial_filter_VAE(
                    gcs, unit_index
                )
            else:
                raise ValueError("Unknown model type, aborting...")

        # Get center masks. This must be done in 2D.
        center_masks = self.get_center_masks(spatial_filters, gcs.mask_threshold)
        surround_masks = self._get_surround_masks(gcs, spatial_filters)

        # Reshape to N units, s**2 pixels
        center_masks_flat = center_masks.reshape((gcs.n_units, s**2))
        spatial_filters_flat = spatial_filters.reshape((gcs.n_units, s**2))

        # Scale spatial filters to sum one of centers for each unit to get veridical max contrast
        spatial_filters_flat_norm = (
            spatial_filters_flat
            / np.sum(spatial_filters_flat * center_masks_flat, axis=1)[:, None]
        )

        # Spatial OFF filters have been inverted to max upwards for construction of RFs.
        # We need to invert them back to max downwards for simulation.
        if gcs.response_type == "off":
            spatial_filters_flat_norm = -spatial_filters_flat_norm

        gcs.spatial_filters_flat = spatial_filters_flat_norm
        gcs.center_masks_flat = center_masks_flat
        gcs.surround_masks_flat = surround_masks.reshape((gcs.n_units, s**2))

        return gcs

    def _convolve_stimulus(self, vs, gcs):
        """
        Convolves the stimulus with the spatiotemporal filter for a given set of units.

        This function performs a convolution operation between the cropped stimulus and
        a spatiotemporal filter for each specified unit. It uses either PyTorch (if available)
        or numpy and scipy to perform the convolution. After the convolution, it adds a tonic drive to the
        generator potential of each unit.

        Parameters
        ----------

        Returns
        -------
        ndarray
            Generator potential of each unit, array of shape (num_cells, stimulus timesteps),
            after the convolution and the addition of the tonic drive.

        Raises
        ------
        AssertionError
            If there is a mismatch between the duration of the stimulus and the duration of the generator potential.

        """

        # Move to GPU if possible. Both give the same result, but PyTorch@GPU is faster.
        if "torch" in sys.modules:
            device = self.context.device
            num_units_t = torch.tensor(gcs.n_units, device=device)
            stim_len_tp_t = torch.tensor(vs.stim_len_tp, device=device)

            # Dimensions are [batch_size, num_channels, time_steps]. We use pixels as channels.
            stimulus_cropped = torch.tensor(vs.stimulus_cropped).float().to(device)
            spatiotemporal_filter = (
                torch.tensor(gcs.spatiotemporal_filters).float().to(device)
            )

            # Convolving two signals involves "flipping" one signal and then sliding it
            # across the other signal. PyTorch, however, does not flip the kernel, so we
            # need to do it manually.
            spatiotemporal_filter_flipped = torch.flip(spatiotemporal_filter, dims=[2])

            # Calculate padding size
            filter_length = spatiotemporal_filter_flipped.shape[2]
            padding_size = filter_length - 1

            # Pad the stimulus
            stimulus_padded = torch.nn.functional.pad(
                stimulus_cropped, (padding_size, 0), mode="replicate"
            )

            output = torch.empty(
                (num_units_t, stim_len_tp_t),
                device=device,
            )

            tqdm_desc = "Preparing fixed generator potential..."
            for i in tqdm(
                torch.range(0, num_units_t - 1, dtype=torch.int), desc=tqdm_desc
            ):
                output[i] = torch.nn.functional.conv1d(
                    stimulus_padded[i].unsqueeze(0),
                    spatiotemporal_filter_flipped[i].unsqueeze(0),
                    padding=0,
                )

            # Move back to CPU and convert to numpy
            generator_potential = output.cpu().squeeze().numpy()
        else:
            # Run convolution. NOTE: expensive computation. Solution without torch.
            # if mode is "valid", the output consists only of those elements that do not rely on the zero-padding
            # if baseline is shorter than filter, the output is truncated
            filter_length = gcs.spatiotemporal_filters.shape[-1]
            assert (
                self.context.my_stimulus_options["baseline_start_seconds"]
                >= filter_length * 1 / vs.stimulus_video.fps
            ), f"baseline_start_seconds must be longer than filter length ({filter_length * vs.video_dt}), aborting..."

            generator_potential = np.empty(
                (gcs.n_units, vs.stim_len_tp - filter_length + 1)
            )
            for idx in range(gcs.n_units):
                generator_potential[idx, :] = convolve(
                    vs.stimulus_cropped[idx],
                    gcs.spatiotemporal_filters[idx],
                    mode="valid",
                )

            # Add some padding to the beginning so that stimulus time and generator potential time match
            # (First time steps of stimulus are not convolved)
            n_padding = int(gcs.data_filter_duration * b2u.ms / vs.video_dt - 1)
            generator_potential = np.pad(
                generator_potential,
                ((0, 0), (n_padding, 0)),
                mode="edge",
            )

        # Internal test for convolution operation
        generator_potential_duration_tp = generator_potential.shape[-1]
        assert (
            vs.stim_len_tp == generator_potential_duration_tp
        ), "Duration mismatch, check convolution operation, aborting..."

        vs.generator_potentials = generator_potential

        return vs

    def _firing_rates2brian_timed_arrays(self, vs):
        # Let's interpolate the rate to vs.video_dt intervals
        tvec_original = np.arange(1, vs.stimulus_video.video_n_frames + 1) * vs.video_dt
        rates_func = interp1d(
            tvec_original,
            vs.firing_rates,
            axis=1,
            fill_value=0,
            bounds_error=False,
        )

        tvec_new = np.arange(0, vs.duration, vs.simulation_dt)

        # This needs to be 2D array for Brian
        interpolated_rates_array = rates_func(tvec_new)

        # Identical rates array for every trial; rows=time, columns=unit index
        inst_rates = b2.TimedArray(
            interpolated_rates_array.T * b2u.Hz, vs.simulation_dt
        )

        vs.interpolated_rates_array = interpolated_rates_array
        vs.tvec_new = tvec_new
        vs.inst_rates = inst_rates
        return vs

    def _brian_spike_generation(self, vs, gcs, n_trials):

        # Set inst_rates to locals() for Brian equation access
        inst_rates = eval("vs.inst_rates")

        # units in parallel (NG), trial iterations (repeated runs)
        n_units_or_trials = np.max([gcs.n_units, n_trials])

        if gcs.spike_generator_model == "refractory":
            # Create Brian NeuronGroup
            # calculate probability of firing for current timebin
            # draw spike/nonspike from random distribution
            # refractory_params = self.context.my_retina["refractory_params"]
            abs_refractory = gcs.refractory_params["abs_refractory"] * b2u.ms
            rel_refractory = gcs.refractory_params["rel_refractory"] * b2u.ms
            p_exp = gcs.refractory_params["p_exp"]
            clip_start = gcs.refractory_params["clip_start"] * b2u.ms
            clip_end = gcs.refractory_params["clip_end"] * b2u.ms

            neuron_group = b2.NeuronGroup(
                n_units_or_trials,
                model="""
                lambda_ttlast = inst_rates(t, i) * dt * w: 1
                t_diff = clip(t - lastspike - abs_refractory, clip_start, clip_end) : second
                w = t_diff**p_exp / (t_diff**p_exp + rel_refractory**p_exp) : 1
                """,
                threshold="rand()<lambda_ttlast",
                refractory="(t-lastspike) < abs_refractory",
                dt=vs.simulation_dt,
            )

            spike_monitor = b2.SpikeMonitor(neuron_group)
            net = b2.Network(neuron_group, spike_monitor)

        elif gcs.spike_generator_model == "poisson":
            # Create Brian PoissonGroup
            poisson_group = b2.PoissonGroup(n_units_or_trials, rates="inst_rates(t, i)")
            spike_monitor = b2.SpikeMonitor(poisson_group)
            net = b2.Network(poisson_group, spike_monitor)
        else:
            raise ValueError(
                "Missing valid spike_generator_model, check my_run_options parameters, aborting..."
            )

        # Save brian state
        net.store()
        all_spiketrains = []
        spikearrays = []
        t_start = []
        t_end = []

        # Run units/trials in parallel, trials in loop
        # tqdm_desc = "Simulating " + self.response_type + " " + self.gc_type + " mosaic"
        # for trial in tqdm(range(n_trials), desc=tqdm_desc):
        net.restore()  # Restore the initial state
        t_start.append(net.t)
        net.run(vs.duration)
        t_end.append(net.t)

        spiketrains = list(spike_monitor.spike_trains().values())
        all_spiketrains.extend(spiketrains)

        # Cxsystem spikemon save natively supports multiple monitors
        spikearrays.append(
            [
                deepcopy(spike_monitor.it[0].__array__()),
                deepcopy(spike_monitor.it[1].__array__()),
            ]
        )

        vs.spikearrays = spikearrays
        vs.n_units_or_trials = n_units_or_trials
        vs.all_spiketrains = all_spiketrains

        return vs

    def _get_dynamic_generator_potentials(self, vs, gcs):

        device = self.context.device

        # Dummy variables to avoid jump to cpu. Impulse response is called above.
        get_impulse_response = torch.tensor(False, device=device)
        contrasts_for_impulse = torch.tensor([1.0], device=device)

        KeyErrorMsg = "Parameter columns mismatch. Did you forget to build? Activate PM.construct_retina.build()."
        if gcs.gc_type == "parasol":
            columns = ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
            try:
                params = gcs.df.loc[gcs.unit_indices, columns].values
            except KeyError:
                raise KeyError(KeyErrorMsg)
            params_t = torch.tensor(params, device=device)
            svecs_t = torch.tensor(vs.svecs, device=device)
        elif gcs.gc_type == "midget":
            columns_cen = [
                "NL_cen",
                "NLTL_cen",
                "TS_cen",
                "HS_cen",
                "D_cen",
                "A_cen",
            ]
            try:
                params_cen = gcs.df.loc[gcs.unit_indices, columns_cen].values
            except KeyError:
                raise KeyError(KeyErrorMsg)

            params_cen_t = torch.tensor(params_cen, device=device)
            svecs_cen_t = torch.tensor(vs.svecs_cen, device=device)
            # Note delay (D) for sur is the same as for cen, the cen-sur delay
            # emerges from LP filter parameters
            columns_sur = [
                "NL_sur",
                "NLTL_sur",
                "TS_sur",
                "HS_sur",
                "D_cen",
                "A_sur",
            ]
            try:
                params_sur = gcs.df.loc[gcs.unit_indices, columns_sur].values
            except KeyError:
                raise KeyError(KeyErrorMsg)

            params_sur_t = torch.tensor(params_sur, device=device)
            svecs_sur_t = torch.tensor(vs.svecs_sur, device=device)

        stim_len_tp_t = torch.tensor(vs.stim_len_tp, device=device)
        num_units_t = torch.tensor(gcs.n_units, device=device)
        generator_potentials_t = torch.empty(
            (num_units_t, stim_len_tp_t), device=device
        )
        tvec_t = torch.tensor(vs.tvec / b2u.ms, device=device)
        video_dt_t = torch.tensor(vs.video_dt / b2u.ms, device=device)

        tqdm_desc = "Preparing dynamic generator potential..."
        for idx in tqdm(
            torch.range(0, num_units_t - 1, dtype=torch.int), desc=tqdm_desc
        ):
            if gcs.gc_type == "parasol":
                unit_params = params_t[idx, :]
                # Henri aloita tästä

                generator_potential = self._create_temporal_signal_cg(
                    tvec_t,
                    svecs_t[idx, :],
                    video_dt_t,
                    unit_params,
                    device,
                    show_impulse=get_impulse_response,
                    impulse_contrast=contrasts_for_impulse,
                )
                # generator_potentials were unitwise delayed at start of the stimulus
                generator_potential = generator_potential[:stim_len_tp_t]
                generator_potentials_t[idx, :] = generator_potential

            elif gcs.gc_type == "midget":
                # Migdet units' surrounds are delayed in comparison to centre.
                # Thus, we need to run cen and the sur separately.

                # Low-passing impulse response for center and surround
                unit_params_cen = params_cen_t[idx, :]
                lp_cen = self._create_lowpass_response(tvec_t, unit_params_cen)

                unit_params_sur = params_sur_t[idx, :]
                lp_sur = self._create_lowpass_response(tvec_t, unit_params_sur)

                # Scale the show_impulse response to have unit area in both calls for high-pass.
                # This corresponds to summation before high-pass stage, as in Schottdorf_2021_JPhysiol
                lp_total = torch.sum(lp_cen) + torch.sum(lp_sur)
                h_cen = lp_cen / lp_total
                h_sur = lp_sur / lp_total

                # Convolve stimulus with the low-pass filter and apply high-pass stage
                gen_pot_cen = self._create_temporal_signal(
                    tvec_t,
                    svecs_cen_t[idx, :],
                    video_dt_t,
                    unit_params_cen,
                    h_cen,
                    device,
                    show_impulse=get_impulse_response,
                )
                gen_pot_sur = self._create_temporal_signal(
                    tvec_t,
                    svecs_sur_t[idx, :],
                    video_dt_t,
                    unit_params_sur,
                    h_sur,
                    device,
                    show_impulse=get_impulse_response,
                )
                # generator_potentials are individually delayed from the beginning of the stimulus
                # This results in varying vector lengths, so we need to crop
                gen_pot_cen = gen_pot_cen[:stim_len_tp_t]
                gen_pot_sur = gen_pot_sur[:stim_len_tp_t]

                generator_potentials_t[idx, :] = gen_pot_cen + gen_pot_sur

            vs.generator_potentials = generator_potentials_t.cpu().numpy()

        return vs

    def _bind_data_for_viz(self, vs, gcs, n_trials):
        """
        Bind data to project_data container for visualization.
        """

        stim_to_show = {
            "stimulus_video": vs.stimulus_video,
            "df_stimpix": gcs.df_stimpix,
            "stimulus_height_pix": vs.stimulus_height_pix,
            "pix_per_deg": vs.pix_per_deg,
            "deg_per_mm": gcs.deg_per_mm,
            "stimulus_center": vs.stimulus_center,
            "qr_min_max": self._get_crop_pixels(gcs, gcs.unit_indices),
            "spatial_filter_sidelen": gcs.spatial_filter_sidelen,
            "stimulus_cropped": vs.stimulus_cropped,
        }

        intermediate_responses_to_show = {
            "cone_noise": vs.cone_noise,
            "cone_signal": vs.cone_signal,
            "photodiode_response": vs.photodiode_response,
            "photodiode_Rstar_range": vs.photodiode_Rstar_range,
        }

        gc_responses_to_show = {
            "n_trials": n_trials,
            "n_units": gcs.n_units,
            "all_spiketrains": vs.all_spiketrains,
            "duration": vs.duration,
            "generator_potential": vs.firing_rates,
            "video_dt": vs.video_dt,
            "tvec_new": vs.tvec_new,
        }

        # Attach data requested by other classes to project_data
        self.project_data.simulate_retina["stim_to_show"] = stim_to_show
        self.project_data.simulate_retina["gc_responses_to_show"] = gc_responses_to_show
        self.project_data.simulate_retina["intermediate_responses_to_show"] = (
            intermediate_responses_to_show
        )

        if gcs.temporal_model == "fixed":
            spat_temp_filter_to_show = {
                "spatial_filters": gcs.spatial_filters_flat,
                "temporal_filters": gcs.temporal_filters,
                "data_filter_duration": gcs.data_filter_duration,
                "temporal_filter_len": gcs.temporal_filter_len,
                "gc_type": gcs.gc_type,
                "response_type": gcs.response_type,
                "temporal_model": gcs.temporal_model,
                "spatial_filter_sidelen": gcs.spatial_filter_sidelen,
            }
            self.project_data.simulate_retina["spat_temp_filter_to_show"] = (
                spat_temp_filter_to_show
            )

    def initialize_cones(self):
        ret_npz_file = self.context.my_retina["ret_file"]
        ret_npz = self.data_io.get_data(filename=ret_npz_file)

        cones = Cones(
            self.context.my_retina,
            ret_npz,
            self.context.device,
            self.context.my_stimulus_options["ND_filter"],
            # RetinaMath methods:
            self.interpolation_function,
            self.lin_interp_and_double_lorenzian,
        )

        return cones

    def _initialize_build(
        self, stimulus, unit_index, spike_generator_model, simulation_dt
    ):
        """
        Inject dependencies to helper classes here
        """
        # This is needed also independently of the pipeline
        cones = self.initialize_cones()

        # Abstraction for clarity
        rfs_npz_file = self.context.my_retina["spatial_rfs_file"]
        rfs_npz = self.data_io.get_data(filename=rfs_npz_file)
        mosaic_file = self.context.my_retina["mosaic_file"]
        gc_dataframe = self.data_io.get_data(filename=mosaic_file)

        gcs = GanglionCells(
            self.context.my_retina,
            self.context.apricot_metadata,
            rfs_npz,
            gc_dataframe,
            unit_index,
            spike_generator_model,
            self.pol2cart_df,
        )

        ret_npz_file = self.context.my_retina["ret_file"]
        ret_npz = self.data_io.get_data(filename=ret_npz_file)

        bipolars = Bipolars(
            self.context.my_retina,
            ret_npz,
        )

        vs = VisualSignal(
            self.context.my_stimulus_options,
            self.context.my_retina["stimulus_center"],
            self.data_io.load_stimulus_from_videofile,
            simulation_dt,
            self.context.my_retina["deg_per_mm"],
            stimulus_video=stimulus,
        )

        gcs.link_gcs_to_vs(vs)

        return vs, gcs, cones, bipolars

    def run_cells(
        self,
        unit_index=None,
        n_trials=1,
        save_data=False,
        spike_generator_model="refractory",
        filename=None,
        simulation_dt=0.001,
        get_impulse_response=False,
        contrasts_for_impulse=None,
        get_uniformity_data=False,
        stimulus=None,
    ):
        """
        Executes the visual signal processing for designated ganglion cells, simulating their spiking output.

        This method is capable of running the linear-nonlinear (LN) pipeline for a single or multiple ganglion cells,
        converting visual stimuli into spike trains using the Brian2 simulator. When `get_impulse_response` is enabled,
        it bypasses the pipeline to compute impulse responses for specified unit types and contrasts.
        The method also supports the computation of spatial uniformity indices when `get_uniformity_data` is set.


        References
        ----------
        The theoretical background and models used in this simulation refer to:
        [1] Victor 1987 Journal of Physiology
        [2] Benardete & Kaplan 1997 Visual Neuroscience
        [3] Kaplan & Benardete 1999 Journal of Physiology
        [4] Chichilnisky 2001 Network
        [5] Chichilnisky & Kalmar 2002 Journal of Neuroscience
        [6] Field et al. 2010 Nature
        [8] Ala-Laurila et al. 2011 NatNeurosci
        [9] Angueyra & Rieke 2013 NatNeurosci
        """

        # Initialize
        # vs = VisualSignal, gcs = GanglionCells, cones = Cones, bipolars = Bipolars
        vs, gcs, cones, bipolars = self._initialize_build(
            stimulus, unit_index, spike_generator_model, simulation_dt
        )

        if get_impulse_response is True:
            self._get_impulse_response(gcs, contrasts_for_impulse, vs.video_dt)
            return

        # Get ganglion cell spatial_filters and center masks
        gcs = self._get_gc_spatial_filters(gcs)

        if get_uniformity_data is True:
            self._get_uniformity_index(vs, gcs)
            return

        # Get cropped stimulus, vectorized. One cropped sequence for each unit
        vs = self._get_spatially_cropped_video(vs, gcs, reshape=True)

        vs = cones.create_signal(vs, n_trials)
        vs = cones.create_noise(vs, n_trials)

        # Get generator potentials
        if gcs.temporal_model == "dynamic":

            vs = self._create_dynamic_contrast(vs, gcs)
            vs = self._get_dynamic_generator_potentials(vs, gcs)

        elif gcs.temporal_model == "fixed":  # Linear model

            gcs = self._get_linear_temporal_filters(gcs)
            gcs = self._get_linear_spatiotemporal_filters(gcs)
            vs = self._convolve_stimulus(vs, gcs)

        # From generator potential to spikes
        vs = self._generator_to_firing_rate_noise(vs, gcs, cones, n_trials)
        vs = self._firing_rates2brian_timed_arrays(vs)
        vs = self._brian_spike_generation(vs, gcs, n_trials)

        # Save retina spikes
        if save_data is True:
            vs.w_coord, vs.z_coord = self.get_w_z_coords(gcs)
            self.data_io.save_retina_output(vs, gcs, filename)

        self._bind_data_for_viz(vs, gcs, n_trials)

    def run_with_my_run_options(self):
        """
        Filter method between my_run_options and run units.
        See run_cells for parameter description.
        """

        filenames = self.context.my_run_options["gc_response_filenames"]
        unit_index = self.context.my_run_options["unit_index"]
        n_trials = self.context.my_run_options["n_trials"]
        save_data = self.context.my_run_options["save_data"]
        spike_generator_model = self.context.my_run_options["spike_generator_model"]
        simulation_dt = self.context.my_run_options["simulation_dt"]

        for filename in filenames:
            self.run_cells(
                unit_index=unit_index,
                n_trials=n_trials,
                save_data=save_data,
                spike_generator_model=spike_generator_model,
                filename=filename,
                simulation_dt=simulation_dt,
            )
