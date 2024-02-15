# Numerical
import numpy as np

import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from scipy.optimize import fsolve, curve_fit, least_squares
from scipy.ndimage import gaussian_filter
from scipy.special import erf
import scipy.fftpack as fftpack
from skimage.transform import resize
import torch

# Data IO
import cv2

# Viz
from tqdm import tqdm
import matplotlib.pyplot as plt

# Comput Neurosci
import brian2 as b2
import brian2.units as b2u

# Local
from cxsystem2.core.tools import write_to_file, load_from_file
from retina.retina_math_module import RetinaMath
from project.project_utilities_module import Printable


# Builtin
from pathlib import Path
from copy import deepcopy
import pdb
import sys
import time
from dataclasses import dataclass
from typing import Any
import inspect

b2.prefs["logging.display_brian_error_message"] = False


class PreGCProcessing:
    """
    PreGCProcessing is with SimulateRetina, because the latter needs the cone filtering for natural stimuli
    (optical aberration and nonlinear luminance response).
    """

    def __init__(self, context, data_io) -> None:
        self._context = context.set_context(self)
        self._data_io = data_io

        self.optical_aberration = self.context.my_retina["optical_aberration"]
        self.rm = self.context.my_retina["cone_general_params"]["rm"]
        self.k = self.context.my_retina["cone_general_params"]["k"]
        self.cone_sensitivity_min = self.context.my_retina["cone_general_params"][
            "sensitivity_min"
        ]
        self.cone_sensitivity_max = self.context.my_retina["cone_general_params"][
            "sensitivity_max"
        ]

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    def natural_stimuli_cone_filter(self):
        image_file_name = self.context.my_stimulus_metadata["stimulus_file"]
        self.pix_per_deg = self.context.my_stimulus_metadata["pix_per_deg"]
        self.fps = self.context.my_stimulus_metadata["fps"]

        # Process stimulus.
        self.image = self.data_io.get_data(image_file_name)

        # For videofiles, average over color channels
        filename_extension = Path(image_file_name).suffix
        if filename_extension in [".avi", ".mp4"]:
            if self.image.shape[-1] == 3:
                self.image = np.mean(self.image, axis=-1).squeeze()
            options = self.context.my_stimulus_options

        self._optical_aberration()
        self._luminance2cone_response()

    def _optical_aberration(self):
        """
        Gaussian smoothing from Navarro 1993 JOSAA: 2 arcmin FWHM under 20deg eccentricity.
        """

        # Turn the optical aberration of 2 arcmin FWHM to Gaussian function sigma
        sigma_deg = self.optical_aberration / (2 * np.sqrt(2 * np.log(2)))
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


class ReceptiveFields(Printable):
    """
    Class containing information associated with receptive fields, including
    retina parameters, the spatial and temporal filters.
    """

    def __init__(
        self,
        my_retina,
        apricot_metadata,
        get_data,
        pol2cart_df,
        cell_index,
        spike_generator_model,
    ):
        # Parameters directly passed to the constructor
        self.my_retina = my_retina
        self.apricot_metadata = apricot_metadata
        self.get_data = get_data
        self.pol2cart_df = pol2cart_df
        self.spike_generator_model = spike_generator_model

        # Default values for computed variables
        self.spatial_filter_sidelen = 0
        self.microm_per_pix = 0.0
        self.temporal_filter_len = 0

        # Extracted and computed values from provided parameters
        self.gc_type = self.my_retina["gc_type"]
        self.response_type = self.my_retina["response_type"]
        self.deg_per_mm = self.my_retina["deg_per_mm"]
        self.DoG_model = self.my_retina["DoG_model"]
        self.cone_general_params = self.my_retina["cone_general_params"]
        self.spatial_model = self.my_retina["spatial_model"]
        self.temporal_model = self.my_retina["temporal_model"]
        self.mask_threshold = my_retina["center_mask_threshold"]
        self.refractory_params = my_retina["refractory_params"]

        assert isinstance(
            self.mask_threshold, float
        ), "mask_threshold must be float, aborting..."
        assert (
            self.mask_threshold >= 0 and self.mask_threshold <= 1
        ), "mask_threshold must be between 0 and 1, aborting..."

        self.data_microm_per_pixel = self.apricot_metadata["data_microm_per_pix"]
        self.data_filter_fps = self.apricot_metadata["data_fps"]
        self.data_filter_timesteps = self.apricot_metadata[
            "data_temporalfilter_samples"
        ]
        self.data_filter_duration = self.data_filter_timesteps * (
            1000 / self.data_filter_fps
        )

        gc_dataframe = self.get_data(filename=self.my_retina["mosaic_file"])
        rspace_pos_mm = self.pol2cart_df(gc_dataframe)
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

        rfs_npz = self.get_data(filename=self.my_retina["spatial_rfs_file"])
        self.spat_rf = rfs_npz["gc_img"]
        self.um_per_pix = rfs_npz["um_per_pix"]
        self.sidelen_pix = rfs_npz["pix_per_side"]
        ret_npz = self.get_data(filename=self.my_retina["ret_file"])
        self.cones_to_gcs_weights = ret_npz["cones_to_gcs_weights"]
        self.cone_noise_parameters = ret_npz["cone_noise_parameters"]

        # Run all cells
        if cell_index is None:
            self.n_units = len(df.index)  # all cells
            cell_indices = np.arange(self.n_units)
        # Run a subset of cells
        elif isinstance(cell_index, (list)):
            cell_indices = np.array(cell_index)
            self.n_units = len(cell_indices)
        # Run one cell
        elif isinstance(cell_index, (int)):
            cell_indices = np.array([cell_index])
            self.n_units = len(cell_indices)
        else:
            raise AssertionError(
                "cell_index must be None, an integer or list, aborting..."
            )
        if isinstance(cell_indices, (int, np.int32, np.int64)):
            cell_indices = np.array([cell_indices])
        self.cell_indices = np.atleast_1d(cell_indices)

    def _link_rf_to_vs(self, vs):
        """
        Endows RGCs with stimulus/pixel space coordinates.

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
            # Remove exceptionally large values from surround before calculating sidelen.
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

        df_stimpix["ampl_c"] = df.ampl_c_norm
        df_stimpix["ampl_s"] = df.ampl_s_norm

        self.df_stimpix = df_stimpix

        self.microm_per_pix = (1 / self.deg_per_mm) / vs.pix_per_deg * 1000
        self.temporal_filter_len = int(self.data_filter_duration / (1000 / vs.fps))
        # pdb.set_trace()


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
        stimulus_video=None,
    ):
        # Parameters directly passed to the constructor
        self.my_stimulus_options = my_stimulus_options
        self.stimulus_center = stimulus_center
        self.load_stimulus_from_videofile = load_stimulus_from_videofile

        # Default value for computed variable
        self.stimulus_video = stimulus_video

        # Extracting and computing values from provided parameters
        self.video_file_name = self.my_stimulus_options["stimulus_video_name"]
        self.stimulus_width_pix = self.my_stimulus_options["image_width"]
        self.stimulus_height_pix = self.my_stimulus_options["image_height"]
        self.pix_per_deg = self.my_stimulus_options["pix_per_deg"]
        self.fps = self.my_stimulus_options["fps"]

        self.stimulus_width_deg = self.stimulus_width_pix / self.pix_per_deg
        self.stimulus_height_deg = self.stimulus_height_pix / self.pix_per_deg

        # Load stimulus video if not already loaded
        if self.stimulus_video is None:
            self.stimulus_video = self.load_stimulus_from_videofile(
                self.video_file_name
            )

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
        assert (
            np.min(self.stimulus_video.frames) >= 0
            and np.max(self.stimulus_video.frames) <= 255
        ), "Stimulus pixel values must be between 0 and 255"

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
    def __init__(self, context, data_io, cones, viz, project_data) -> None:
        self._context = context.set_context(self)
        self._data_io = data_io
        self._cones = cones
        self._viz = viz
        self._project_data = project_data

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def cones(self):
        return self._cones

    @property
    def viz(self):
        return self._viz

    @property
    def project_data(self):
        return self._project_data

    def _get_crop_pixels(self, rf, cell_index):
        """
        Get pixel coordinates for a stimulus crop matching the spatial filter size.

        Parameters
        ----------
        cell_index : int or array-like of int
            Index or indices of the cell(s) for which to retrieve crop coordinates.

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

        if isinstance(cell_index, (int, np.int32, np.int64)):
            cell_index = np.array([cell_index])
        df_stimpix = rf.df_stimpix.iloc[cell_index]
        q_center = np.round(df_stimpix.q_pix).astype(int).values
        r_center = np.round(df_stimpix.r_pix).astype(int).values

        # crops have width = height
        side_halflen = (rf.spatial_filter_sidelen - 1) // 2

        qmin = q_center - side_halflen
        qmax = q_center + side_halflen
        rmin = r_center - side_halflen
        rmax = r_center + side_halflen

        return qmin, qmax, rmin, rmax

    def _create_spatial_filter_FIT(self, rf, cell_index):
        """
        Creates the spatial component of the spatiotemporal filter

        Parameters
        ----------
        cell_index : int
            Index of the cell in the df

        Returns
        -------
        spatial_filter : np.ndarray
            Spatial filter for the given cell
        """

        offset = 0.0
        s = rf.spatial_filter_sidelen

        gc = rf.df_stimpix.iloc[cell_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(rf, cell_index)

        x_grid, y_grid = np.meshgrid(
            np.arange(qmin, qmax + 1, 1), np.arange(rmin, rmax + 1, 1)
        )
        # spatial_kernel is here 1-dim vector
        if rf.DoG_model == "ellipse_fixed":
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

    def _create_spatial_filter_VAE(self, rf, cell_index):
        """
        Creates the spatial component of the spatiotemporal filter

        Parameters
        ----------
        cell_index : int
            Index of the cell in the df

        Returns
        -------
        spatial_filter : np.ndarray
            Spatial filter for the given cell
        """
        s = rf.spatial_filter_sidelen

        spatial_kernel = resize(
            rf.spat_rf[cell_index, :, :], (s, s), anti_aliasing=True
        )

        return spatial_kernel

    def _create_temporal_filter(self, rf, cell_index):
        """
        Creates the temporal component of the spatiotemporal filter. Linear fixed-sum of two lowpass filters.

        Parameters
        ----------
        cell_index : int
            Index of the cell in the df

        Returns
        -------
        temporal_filter : np.ndarray
            Temporal filter for the given cell
        """

        filter_params = rf.df.iloc[cell_index][["n", "p1", "p2", "tau1", "tau2"]]

        tvec = np.linspace(0, rf.data_filter_duration, rf.temporal_filter_len)
        temporal_filter = self.diff_of_lowpass_filters(tvec, *filter_params)

        # Amplitude will be scaled by abs(sum()) of the temporal_filter
        temporal_filter = temporal_filter / np.sum(np.abs(temporal_filter))

        return temporal_filter

    def _get_spatially_cropped_video(self, vs, rf, contrast=True, reshape=False):
        """
        Crops the video to the surroundings of the specified Retinal Ganglion Cells (RGCs).

        The function works by first determining the pixel range to be cropped for each cell
        in cell_indices, and then selecting those pixels from the original video. The cropping
        is done for each frame of the video. If the contrast option is set to True, the video
        is also rescaled to have pixel values between -1 and 1.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object containing the stimulus video.

        rf : ReceptiveFields
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

        sidelen = rf.spatial_filter_sidelen
        cell_indices = rf.cell_indices.copy()
        video_copy = vs.stimulus_video.frames.copy()
        video_copy = np.transpose(
            video_copy, (1, 2, 0)
        )  # Original frames are now [height, width, time points]
        video_copy = video_copy[np.newaxis, ...]  # Add new axis for broadcasting

        qmin, qmax, rmin, rmax = self._get_crop_pixels(rf, cell_indices)

        # Adjust the creation of r_indices and q_indices for proper broadcasting
        r_indices = np.arange(sidelen) + rmin[:, np.newaxis]
        q_indices = np.arange(sidelen) + qmin[:, np.newaxis]

        # Ensure r_indices and q_indices are repeated for each cell
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
                (rf.n_units, sidelen**2, n_frames)
            )

        vs.stimulus_cropped = stimulus_cropped

        return vs

    def _get_uniformity_index(self, vs, rf):
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
            - 'unit_region': The sum of center regions for all cells.

        """
        height = vs.stimulus_height_pix
        width = vs.stimulus_width_pix
        cell_indices = rf.cell_indices.copy()

        qmin, qmax, rmin, rmax = self._get_crop_pixels(rf, cell_indices)

        stim_region = np.zeros((rf.n_units, height, width), dtype=np.int32)
        center_region = np.zeros((rf.n_units, height, width), dtype=np.int32)

        # Create the r and q indices for each cell, ensure they're integer type
        sidelen = rf.spatial_filter_sidelen
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

        # create a cell index array
        unit_region_idx = np.arange(rf.n_units).astype(np.int32).reshape(-1, 1, 1)

        # expand the indices arrays to the shape of r_matrix and q_matrix using broadcasting
        unit_region_idx = unit_region_idx + np.zeros_like(r_matrix, dtype=np.int32)

        # use the index arrays to select the elements from video_copy
        stim_region[unit_region_idx, r_matrix, q_matrix] = 1

        center_masks = rf.center_masks.copy()
        center_masks = center_masks.astype(bool).reshape((rf.n_units, sidelen, sidelen))

        center_region[
            unit_region_idx * center_masks,
            r_matrix * center_masks,
            q_matrix * center_masks,
        ] = 1

        unit_region = np.sum(center_region, axis=0)

        # Delaunay triangulation for the total region
        gc = rf.df_stimpix.iloc[cell_indices]
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
            "mask_threshold": rf.mask_threshold,
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
        ), "Number of cells in params_all and generator_potentials must match, aborting..."

        tonic_drive = params_all["tonic_drive"]
        # Expanding tonic_drive to match the shape of generator_potentials
        tonic_drive = np.expand_dims(tonic_drive, axis=1)
        # Apply nonlinearity
        # tonic_drive**2 is added to mimick spontaneous firing rates
        firing_rates = np.maximum(generator_potentials + tonic_drive**2, 0)

        return firing_rates

    def _generator_to_firing_rate_noise(self, vs, rf, n_trials):
        """
        Generates cone noise, scales it with mean firing rates. Multiplies the generator potentials with gain and
        finally adds the firing rates generated by the cone noise to the light-induced firing rates.
        Parameters
        ----------

        Returns
        -------
        ndarray
            The firing rates after adding Gaussian noise and applying gain and mean firing rates adjustments.
        """
        params_all = rf.df.loc[rf.cell_indices]
        assert (
            params_all.shape[0] == vs.generator_potentials.shape[0]
        ), "Number of cells in params_all and generator_potentials must match, aborting..."

        cones_to_gcs_weights = rf.cones_to_gcs_weights
        NL, TL, HS, TS, A0, M0, D = rf.cone_noise_parameters

        cones_to_gcs_weights = cones_to_gcs_weights[:, rf.cell_indices]
        n_cones = cones_to_gcs_weights.shape[0]

        # Normalize weights by columns (ganglion cells)
        weights_norm = cones_to_gcs_weights / np.sum(cones_to_gcs_weights, axis=0)

        def _create_cone_noise(tvec, n_cones, NL, TL, HS, TS, A0, M0, D):
            tvec = tvec / b2u.second
            freqs = fftpack.fftfreq(len(tvec), d=(tvec[1] - tvec[0]))

            white_noise = np.random.normal(0, 1, (len(tvec), n_cones))
            noise_fft = fftpack.fft(white_noise, axis=0)

            # Generate the asymmetric concave function for scaling
            f_scaled = np.abs(freqs)
            # Prevent division by zero for zero frequency
            f_scaled[f_scaled == 0] = 1e-10
            asymmetric_scale = self.victor_model_frequency_domain(
                f_scaled, NL, TL, HS, TS, A0, M0, D
            )

            noise_fft = noise_fft * asymmetric_scale[:, np.newaxis]

            # Transform back to time domain
            cone_noise = np.real(fftpack.ifft(noise_fft, axis=0))

            return cone_noise

        # Make independent cone noise for multiple trials
        if n_trials > 1:
            for trial in range(n_trials):
                cone_noise = _create_cone_noise(
                    vs.tvec, n_cones, NL, TL, HS, TS, A0, M0, D
                )
                if trial == 0:
                    gc_noise = cone_noise @ weights_norm
                else:
                    gc_noise = np.concatenate(
                        (gc_noise, cone_noise @ weights_norm), axis=1
                    )
        elif vs.generator_potentials.shape[0] > 1:
            cone_noise = _create_cone_noise(vs.tvec, n_cones, NL, TL, HS, TS, A0, M0, D)
            gc_noise = cone_noise @ weights_norm

        # Normalize noise to have unit variance
        gc_noise_norm = gc_noise / np.std(gc_noise, axis=0)

        # Manual multiplier from conf file
        magn = rf.cone_general_params["cone_noise_magnitude"]
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
        Dynamic temporal signal for midget cells
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
        Lowpass filter kernel for convolution for midget cells
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

    def _create_dynamic_contrast(self, vs, rf):
        """
        Create dynamic contrast signal by multiplying the stimulus with the spatial filter
        masks are used for midget cells, where center and surround have distinct dynamics.

        Parameters
        ----------


        Returns
        -------
        center_surround_filters_sum : array
            Dynamic contrast signal
        """

        # Reshape masks and spatial_filters to match the dimensions of stimulus_cropped
        spatial_filters = rf.spatial_filters.copy()
        spatial_filters_reshaped = np.expand_dims(spatial_filters, axis=2)

        # victor_1987_JPhysiol: input to model is s(t)), the signed Weber contrast at the centre.
        # However, we assume that the surround suppression is early (horizontal cells) and linear,
        # so we approximate s(t) = RF * stimulus
        if rf.gc_type == "parasol":

            # This is the stimulus contrast viewed through spatial rf filter, and summed over spatial dimension.
            # The np.einsum provides a fast and memory-efficient way to do this.
            vs.svecs = np.einsum(
                "ijk,ijk->ik", spatial_filters_reshaped, vs.stimulus_cropped
            )

        elif rf.gc_type == "midget":
            # Surround is always negative at this stage
            masks_sur = spatial_filters < 0
            masks_sur = masks_sur[:, :, np.newaxis]

            vs.svecs_sur = np.einsum(
                "ijk,ijk->ik", spatial_filters_reshaped * masks_sur, vs.stimulus_cropped
            )

            center_masks = rf.center_masks[:, :, np.newaxis]
            # Similarly, for center filters
            vs.svecs_cen = np.einsum(
                "ijk,ijk->ik",
                spatial_filters_reshaped * center_masks,
                vs.stimulus_cropped,
            )

        return vs

    def _get_impulse_response(self, rf, contrasts_for_impulse, video_dt):
        """
        Provides impulse response for distinct ganglion cell and response types.
        Much of the run_cells code is copied here, but with the stimulus replaced by an impulse.
        """

        # Set filter duration the same as in Apricot data
        total_duration = rf.data_filter_timesteps * (1000 / rf.data_filter_fps) * b2u.ms
        stim_len_tp = int(np.round(total_duration / video_dt))
        tvec = range(stim_len_tp) * video_dt

        # Dummy kernel for show_impulse response
        svec = np.zeros(len(tvec))
        dt = video_dt / b2u.ms
        start_delay = 100  # ms
        idx_100_ms = int(np.round(start_delay / dt))
        svec[idx_100_ms] = 1.0

        if rf.response_type == "off":
            # Spatial OFF filters have been inverted to max upwards for construction of RFs.
            svec = -svec

        stim_len_tp = len(tvec)
        # Append to impulse_to_show a key str(contrast) for each contrast,
        # holding empty array for impulse response

        assert contrasts_for_impulse is not None and isinstance(
            contrasts_for_impulse, list
        ), "Impulse must specify contrasts as list, aborting..."

        yvecs = np.empty((rf.n_units, len(contrasts_for_impulse), stim_len_tp))
        if rf.temporal_model == "dynamic":
            # cpu on purpose, less issues, very fast anyway
            device = torch.device("cpu")
            svec_t = torch.tensor(svec, device=device)
            tvec_t = torch.tensor(tvec / b2u.ms, device=device)
            video_dt_t = torch.tensor(video_dt / b2u.ms, device=device)

            for idx, this_cell_index in enumerate(rf.cell_indices):
                if rf.gc_type == "parasol":
                    # Get unit params
                    columns = ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
                    params = rf.df.loc[this_cell_index, columns].values
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

                elif rf.gc_type == "midget":
                    columns_cen = [
                        "NL_cen",
                        "NLTL_cen",
                        "TS_cen",
                        "HS_cen",
                        "D_cen",
                        "A_cen",
                    ]
                    params_cen = rf.df.loc[this_cell_index, columns_cen].values
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
                    params_sur = rf.df.loc[this_cell_index, columns_sur].values
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

        elif rf.temporal_model == "fixed":  # Linear model
            # Amplitude will be scaled by first (positive) lowpass filter.
            rf = self._get_linear_temporal_filters(rf)
            yvecs = np.repeat(
                rf.temporal_filter[:, np.newaxis, :], len(contrasts_for_impulse), axis=1
            )
            if rf.response_type == "off":
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
        impulse_to_show["Unit idx"] = list(rf.cell_indices)
        impulse_to_show["gc_type"] = rf.gc_type
        impulse_to_show["response_type"] = rf.response_type
        impulse_to_show["temporal_model"] = rf.temporal_model

        self.project_data.simulate_retina["impulse_to_show"] = impulse_to_show

    def get_w_z_coords(self, rf):
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
        z_coord = rf.df["x_deg"].values + 1j * rf.df["y_deg"].values

        visual2cortical_params = self.context.my_retina["visual2cortical_params"]
        a = visual2cortical_params["a"]
        k = visual2cortical_params["k"]
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord

    def _get_linear_temporal_filters(self, rf):
        """
        Retrieve temporal filters for an array of cells.

        This function generates temporal filters for each cell specified by the
        cell indices. The temporal filter for a specific cell is obtained by calling
        the `_create_temporal_filter` method.

        Parameters
        ----------

        Returns
        -------
        temporal_filters : ndarray
            2-D array where each row corresponds to a temporal filter of a cell. The shape is
            (len(cell_indices), self.temporal_filter_len).

        Notes
        -----
        This function depends on the following instance variables:
          - self.temporal_filter_len: an integer specifying the length of a temporal filter.
        """

        temporal_filters = np.zeros((len(rf.cell_indices), rf.temporal_filter_len))

        for idx, cell_index in enumerate(rf.cell_indices):
            temporal_filters[idx, :] = self._create_temporal_filter(rf, cell_index)

        rf.temporal_filters = temporal_filters

        return rf

    def _get_linear_spatiotemporal_filters(self, rf):
        """
        Generate spatiotemporal filters for given cell indices."""

        # Assuming spatial_filters.shape = (U, N) and temporal_filters.shape = (U, T)
        spatiotemporal_filters = (
            rf.spatial_filters[:, :, None] * rf.temporal_filters[:, None, :]
        )
        rf.spatiotemporal_filters = spatiotemporal_filters

        return rf

    def get_spatial_filters(self, rf):
        """
        Generate spatial filters for given cell indices.

        This function takes a list of cell indices, determines the model type,
        creates a corresponding spatial filter for each cell index based on the model,
        and then reshapes the filter to 1-D. It returns a 2-D array where each row is a
        1-D spatial filter for a corresponding cell index.

        Parameters
        ----------

        Returns
        -------
        spatial_filters : ndarray
            2-D array where each row corresponds to a 1-D spatial filter or mask of a cell.
            The shape is (len(cell_indices), s**2), where s is the side length of a spatial filter.

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

        s = rf.spatial_filter_sidelen
        spatial_filters = np.zeros((rf.n_units, s, s))
        for idx, cell_index in enumerate(rf.cell_indices):
            if rf.spatial_model == "FIT":
                spatial_filters[idx, ...] = self._create_spatial_filter_FIT(
                    rf, cell_index
                )
            elif rf.spatial_model == "VAE":
                spatial_filters[idx, ...] = self._create_spatial_filter_VAE(
                    rf, cell_index
                )
            else:
                raise ValueError("Unknown model type, aborting...")

        # Reshape to N cells, s**2 pixels
        spatial_filters = np.reshape(spatial_filters, (rf.n_units, s**2))

        # Get center masks
        center_masks = self.get_rf_masks(
            spatial_filters, mask_threshold=rf.mask_threshold
        )

        # Scale spatial filters to sum one of centers for each unit to get veridical max contrast
        spatial_filters = (
            spatial_filters / np.sum(spatial_filters * center_masks, axis=1)[:, None]
        )

        # Spatial OFF filters have been inverted to max upwards for construction of RFs.
        # We need to invert them back to max downwards for simulation.
        if rf.response_type == "off":
            spatial_filters = -spatial_filters

        rf.spatial_filters = spatial_filters
        rf.center_masks = center_masks

        return rf

    def _convolve_stimulus_batched(self, vs, rf):
        """
        Convolves the stimulus with the spatiotemporal filter for a given set of cells.

        This function performs a convolution operation between the cropped stimulus and
        a spatiotemporal filter for each specified cell. It uses either PyTorch (if available)
        or numpy and scipy to perform the convolution. After the convolution, it adds a tonic drive to the
        generator potential of each cell.

        Parameters
        ----------

        Returns
        -------
        ndarray
            Generator potential of each cell, array of shape (num_cells, stimulus timesteps),
            after the convolution and the addition of the tonic drive.

        Raises
        ------
        AssertionError
            If there is a mismatch between the duration of the stimulus and the duration of the generator potential.

        """

        # Move to GPU if possible. Both give the same result, but PyTorch@GPU is faster.
        if "torch" in sys.modules:
            device = self.context.device
            # num_cells = rf.n_units
            num_units_t = torch.tensor(rf.n_units, device=device)
            stim_len_tp_t = torch.tensor(vs.stim_len_tp, device=device)

            # Dimensions are [batch_size, num_channels, time_steps]. We use pixels as channels.
            stimulus_cropped = torch.tensor(vs.stimulus_cropped).float().to(device)
            spatiotemporal_filter = (
                torch.tensor(rf.spatiotemporal_filters).float().to(device)
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
            filter_length = rf.spatiotemporal_filters.shape[-1]
            assert (
                self.context.my_stimulus_options["baseline_start_seconds"]
                >= filter_length * 1 / vs.stimulus_video.fps
            ), f"baseline_start_seconds must be longer than filter length ({filter_length * vs.video_dt}), aborting..."

            generator_potential = np.empty(
                (rf.n_units, vs.stim_len_tp - filter_length + 1)
            )
            for idx in range(rf.n_units):
                generator_potential[idx, :] = convolve(
                    vs.stimulus_cropped[idx],
                    rf.spatiotemporal_filters[idx],
                    mode="valid",
                )

            # Add some padding to the beginning so that stimulus time and generator potential time match
            # (First time steps of stimulus are not convolved)
            n_padding = int(rf.data_filter_duration * b2u.ms / vs.video_dt - 1)
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

        # Identical rates array for every trial; rows=time, columns=cell index
        inst_rates = b2.TimedArray(
            interpolated_rates_array.T * b2u.Hz, vs.simulation_dt
        )

        vs.interpolated_rates_array = interpolated_rates_array
        vs.tvec_new = tvec_new
        vs.inst_rates = inst_rates
        return vs

    def _brian_spike_generation(self, vs, rf, n_trials):

        # Set inst_rates to locals() for Brian equation access
        inst_rates = eval("vs.inst_rates")

        # Cells in parallel (NG), trial iterations (repeated runs)
        n_units_or_trials = np.max([rf.n_units, n_trials])

        if rf.spike_generator_model == "refractory":
            # Create Brian NeuronGroup
            # calculate probability of firing for current timebin
            # draw spike/nonspike from random distribution
            # refractory_params = self.context.my_retina["refractory_params"]
            abs_refractory = rf.refractory_params["abs_refractory"] * b2u.ms
            rel_refractory = rf.refractory_params["rel_refractory"] * b2u.ms
            p_exp = rf.refractory_params["p_exp"]
            clip_start = rf.refractory_params["clip_start"] * b2u.ms
            clip_end = rf.refractory_params["clip_end"] * b2u.ms

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

        elif rf.spike_generator_model == "poisson":
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

        # Run cells/trials in parallel, trials in loop
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

    def _get_dynamic_generator_potentials(self, vs, rf):

        device = self.context.device

        # Dummy variables to avoid jump to cpu. Impulse response is called above.
        get_impulse_response = torch.tensor(False, device=device)
        contrasts_for_impulse = torch.tensor([1.0], device=device)

        KeyErrorMsg = "Parameter columns mismatch. Did you forget to build? Activate PM.construct_retina.build()."
        if rf.gc_type == "parasol":
            columns = ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
            try:
                params = rf.df.loc[rf.cell_indices, columns].values
            except KeyError:
                raise KeyError(KeyErrorMsg)
            params_t = torch.tensor(params, device=device)
            svecs_t = torch.tensor(vs.svecs, device=device)
        elif rf.gc_type == "midget":
            columns_cen = [
                "NL_cen",
                "NLTL_cen",
                "TS_cen",
                "HS_cen",
                "D_cen",
                "A_cen",
            ]
            try:
                params_cen = rf.df.loc[rf.cell_indices, columns_cen].values
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
                params_sur = rf.df.loc[rf.cell_indices, columns_sur].values
            except KeyError:
                raise KeyError(KeyErrorMsg)

            params_sur_t = torch.tensor(params_sur, device=device)
            svecs_sur_t = torch.tensor(vs.svecs_sur, device=device)

        stim_len_tp_t = torch.tensor(vs.stim_len_tp, device=device)
        num_units_t = torch.tensor(rf.n_units, device=device)
        generator_potentials_t = torch.empty(
            (num_units_t, stim_len_tp_t), device=device
        )
        tvec_t = torch.tensor(vs.tvec / b2u.ms, device=device)
        video_dt_t = torch.tensor(vs.video_dt / b2u.ms, device=device)

        tqdm_desc = "Preparing dynamic generator potential..."
        for idx in tqdm(
            torch.range(0, num_units_t - 1, dtype=torch.int), desc=tqdm_desc
        ):
            if rf.gc_type == "parasol":
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

            elif rf.gc_type == "midget":
                # Migdet cells' surrounds are delayed in comparison to centre.
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

    def _bind_for_visualization(self, vs, rf, n_trials):

        stim_to_show = {
            "stimulus_video": vs.stimulus_video,
            "df_stimpix": rf.df_stimpix,
            "stimulus_height_pix": vs.stimulus_height_pix,
            "pix_per_deg": vs.pix_per_deg,
            "deg_per_mm": rf.deg_per_mm,
            "stimulus_center": vs.stimulus_center,
            "qr_min_max": self._get_crop_pixels(rf, rf.cell_indices),
            "spatial_filter_sidelen": rf.spatial_filter_sidelen,
            "stimulus_cropped": vs.stimulus_cropped,
        }

        gc_responses_to_show = {
            "n_trials": n_trials,
            "n_units": rf.n_units,
            "all_spiketrains": vs.all_spiketrains,
            "duration": vs.duration,
            "generator_potential": vs.firing_rates,
            "video_dt": vs.video_dt,
            "tvec_new": vs.tvec_new,
        }

        # Attach data requested by other classes to project_data
        self.project_data.simulate_retina["stim_to_show"] = stim_to_show
        self.project_data.simulate_retina["gc_responses_to_show"] = gc_responses_to_show

        if rf.temporal_model == "fixed":
            spat_temp_filter_to_show = {
                "spatial_filters": rf.spatial_filters,
                "temporal_filters": rf.temporal_filters,
                "data_filter_duration": rf.data_filter_duration,
                "temporal_filter_len": rf.temporal_filter_len,
                "gc_type": rf.gc_type,
                "response_type": rf.response_type,
                "temporal_model": rf.temporal_model,
                "spatial_filter_sidelen": rf.spatial_filter_sidelen,
            }
            self.project_data.simulate_retina["spat_temp_filter_to_show"] = (
                spat_temp_filter_to_show
            )

    def run_cells(
        self,
        cell_index=None,
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
         it bypasses the pipeline to compute impulse responses for specified cell types and contrasts.
         The method also supports the computation of spatial uniformity indices when `get_uniformity_data` is set.

         Parameters
         ----------
         cell_index : int, list of int, or None, optional
             The index(es) of the cell(s) to simulate. If None, all cells are processed. Defaults to None.
         n_trials : int, optional
             The number of independent trials to simulate for the stochastic elements of the model.
             Defaults to 1.
         save_data : bool, optional
             Flag to save the output data to a file. Defaults to False.
         spike_generator_model : str, optional
             The model for spike generation: 'refractory' for a refractory model,
             'poisson' for a Poisson process.
             Defaults to 'refractory'.
         filename : str or None, optional
             The filename for saving output data. If None, no data is saved. Defaults to None.
         simulation_dt : float, optional
             The time step for the simulation in seconds. Defaults to 0.001 (1 ms).
         get_impulse_response : bool, optional
             If True, computes and returns the impulse response for the cell types specified in
             `contrasts_for_impulse`, and skips the standard LN pipeline. Defaults to False.
         contrasts_for_impulse : list of floats or None, optional
             A list of contrast values to compute impulse responses for, applicable when
             `get_impulse_response` is True. Defaults to None.
         get_uniformity_data : bool, optional
             If True, computes and returns a spatial uniformity index and data for visualization,
             and skips the standard LN pipeline. Defaults to False.

         Returns
         -------
        impulse_responses : dict or None
             A dictionary containing impulse responses if `get_impulse_response` is True, otherwise None.
         uniformity_indices : dict or None
             A dictionary containing uniformity indices if `get_uniformity_data` is True, otherwise None.

         Saves to file
         -------------
        spike_trains : dict
            A dictionary containing spike trains for each cell.
            This is saved both for CxSystem2 as gz(ipped) pickle, and
            as csv.
        unit positions (structure): csv
            A csv file containing the coordinates of each cell.

         Saves to internal dictionary for visualization
         ----------------------------------------------
        stim_to_show : dict
            A dictionary containing the stimulus and some metadata used for simulation.
        spat_temp_filter_to_show : dict
            A dictionary containing the spatial and temporal filters for each cell and some metadata.
        gc_responses_to_show : dict
            A dictionary containing the generator potentials and spike trains for each cell and some metadata.

         Raises
         ------
         AssertionError
             If `cell_index` is not None, an integer, or a list;
             if `get_impulse_response` is True but the required
             conditions (e.g., `cell_index`, cell type, or contrasts) are not met.
         ValueError
             If `spike_generator_model` is neither 'refractory' nor 'poisson'.

         Notes
         -----
         - The method can be utilized in various modes depending on the combination of boolean flags provided.
         - Saving data and obtaining impulse responses or uniformity indices are mutually exclusive operations.
         - This method handles the inversion of off-responses to a maximum negative value internally.

         References
         ----------
         For the theoretical background and models used in this simulation refer to:
         [1] Victor 1987 Journal of Physiology
         [2] Benardete & Kaplan 1997 Visual Neuroscience
         [3] Kaplan & Benardete 1999 Journal of Physiology
         [4] Chichilnisky 2001 Network
         [5] Chichilnisky 2002 Journal of Neuroscience
         [6] Field 2010 Nature
         [7] Gauthier 2009 PLoS Biology
        """

        vs = VisualSignal(
            self.context.my_stimulus_options,
            self.context.my_retina["stimulus_center"],
            self.data_io.load_stimulus_from_videofile,
            simulation_dt,
            stimulus_video=stimulus,
        )

        rf = ReceptiveFields(
            self.context.my_retina,
            self.context.apricot_metadata,
            self.data_io.get_data,
            self.pol2cart_df,
            cell_index,
            spike_generator_model,
        )

        rf._link_rf_to_vs(vs)

        if get_impulse_response is True:
            self._get_impulse_response(rf, contrasts_for_impulse, vs.video_dt)
            return

        # Get spatial_filters and center masks
        rf = self.get_spatial_filters(rf)

        if get_uniformity_data is True:
            self._get_uniformity_index(vs, rf)
            return

        # Get cropped stimulus, vectorized. One cropped sequence for each unit
        vs = self._get_spatially_cropped_video(vs, rf, reshape=True)

        # Get instantaneous firing rates
        if rf.temporal_model == "dynamic":
            # Contrast gain control depends dynamically on contrast
            # Henri aloita tästä

            # Get stimulus contrast vector
            vs = self._create_dynamic_contrast(vs, rf)

            # Get generator potentials
            vs = self._get_dynamic_generator_potentials(vs, rf)

        elif rf.temporal_model == "fixed":  # Linear model
            # Amplitude will be scaled by first (positive) lowpass filter.
            rf = self._get_linear_temporal_filters(rf)
            rf = self._get_linear_spatiotemporal_filters(rf)

            vs = self._convolve_stimulus_batched(vs, rf)

        # print(np.mean(vs.generator_potentials.flatten()))
        # print(np.std(vs.generator_potentials.flatten()))
        # plt.hist(vs.generator_potentials.flatten(), 20)
        # plt.show()
        # pdb.set_trace()
        # From generator potential to spikes
        vs = self._generator_to_firing_rate_noise(vs, rf, n_trials)
        vs = self._firing_rates2brian_timed_arrays(vs)
        vs = self._brian_spike_generation(vs, rf, n_trials)

        # Save retina spikes
        if save_data is True:
            vs.w_coord, vs.z_coord = self.get_w_z_coords(rf)
            self.data_io.save_retina_output(vs, rf, filename)

        self._bind_for_visualization(vs, rf, n_trials)

    def run_with_my_run_options(self):
        """
        Filter method between my_run_options and run cells.
        See run_cells for parameter description.
        """

        filenames = self.context.my_run_options["gc_response_filenames"]
        cell_index = self.context.my_run_options["cell_index"]
        n_trials = self.context.my_run_options["n_trials"]
        save_data = self.context.my_run_options["save_data"]
        spike_generator_model = self.context.my_run_options["spike_generator_model"]
        simulation_dt = self.context.my_run_options["simulation_dt"]

        for filename in filenames:
            self.run_cells(
                cell_index=cell_index,
                n_trials=n_trials,
                save_data=save_data,
                spike_generator_model=spike_generator_model,
                filename=filename,
                simulation_dt=simulation_dt,
            )
