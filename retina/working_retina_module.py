# Numerical
import numpy as np

import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
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

# Builtin
from pathlib import Path
from copy import deepcopy
import pdb
import sys
import time

b2.prefs["logging.display_brian_error_message"] = False


class WorkingRetina(RetinaMath):
    _properties_list = [
        "path",
        "output_folder",
        "my_retina",
        "my_stimulus_metadata",
        "my_stimulus_options",
        "my_run_options",
        "apricot_metadata",
    ]

    def __init__(self, context, data_io, viz) -> None:
        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        # viz.client_object = self  # injecting client object pointer into viz object
        self._viz = viz

        self.initialized = False

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def viz(self):
        return self._viz

    def _initialize(self):
        """
        Initialize the retina object.
        The variable gc_dataframe contains the ganglion cell parameters;
            positions are retinal coordinates
            pos_ecc_mm in mm
            pos_polar_deg in degrees

        Attributes:
            gc_type (str): Ganglion cell type
            response_type (str): Response type
            deg_per_mm (float): Degrees per mm
            stimulus_center (list): Center of stimulus in visual space
            stimulus_width_pix (int): Width of stimulus in pixels
            stimulus_height_pix (int): Height of stimulus in pixels
            pix_per_deg (float): Pixels per degree
            fps (float): Frames per second
            model_type (str): Model type
            data_microm_per_pixel (float): Micrometers per pixel
            data_filter_fps (float): Timesteps per second in data
            data_filter_timesteps (int): Timesteps in data
            data_filter_duration (float): Filter duration
            gc_df (DataFrame): Ganglion cell parameters
            gc_df_pixspace (DataFrame): Ganglion cell parameters in pixel space
            spatial_filter_sidelen (int): Spatial filter side length
            microm_per_pix (float): Micrometers per pixel

        """

        # Read fitted parameters from file
        gc_dataframe = self.data_io.get_data(
            filename=self.context.my_retina["mosaic_file"]
        )
        self.gc_type = self.context.my_retina["gc_type"]
        self.response_type = self.context.my_retina["response_type"]
        self.deg_per_mm = self.context.my_retina["deg_per_mm"]
        stimulus_center = self.context.my_retina["stimulus_center"]

        stimulus_width_pix = self.context.my_stimulus_options["image_width"]
        stimulus_height_pix = self.context.my_stimulus_options["image_height"]
        pix_per_deg = self.context.my_stimulus_options["pix_per_deg"]
        fps = self.context.my_stimulus_options["fps"]

        self.model_type = self.context.my_retina["model_type"]
        self.temporal_model = self.context.my_retina["temporal_model"]

        # Metadata for Apricot dataset.
        self.data_microm_per_pixel = self.context.apricot_metadata[
            "data_microm_per_pix"
        ]
        self.data_filter_fps = self.context.apricot_metadata["data_fps"]
        self.data_filter_timesteps = self.context.apricot_metadata[
            "data_temporalfilter_samples"
        ]
        self.data_filter_duration = self.data_filter_timesteps * (
            1000 / self.data_filter_fps
        )  # in milliseconds

        # Convert retinal positions (ecc, pol angle) to visual space positions in deg (x, y)
        rspace_pos_mm = self.pol2cart_df(gc_dataframe)
        vspace_pos = rspace_pos_mm * self.deg_per_mm
        vspace_coords_deg = pd.DataFrame(
            {"x_deg": vspace_pos[:, 0], "y_deg": vspace_pos[:, 1]}
        )
        self.gc_df = pd.concat([gc_dataframe, vspace_coords_deg], axis=1)

        # Convert RF center radii to degrees as well
        self.gc_df.semi_xc = self.gc_df.semi_xc * self.deg_per_mm
        self.gc_df.semi_yc = self.gc_df.semi_yc * self.deg_per_mm

        # Drop retinal positions from the df (so that they are not used by accident)
        self.gc_df = self.gc_df.drop(["pos_ecc_mm", "pos_polar_deg"], axis=1)

        # Simulated data
        self.simulated_spiketrains = []

        # Initialize stuff related to digital sampling
        self.stimulus_center = stimulus_center
        self.stimulus_video = None
        self.stimulus_width_pix = stimulus_width_pix
        self.stimulus_height_pix = stimulus_height_pix
        self.stimulus_width_deg = stimulus_width_pix / pix_per_deg
        self.stimulus_height_deg = stimulus_height_pix / pix_per_deg
        self.pix_per_deg = pix_per_deg  # angular resolution (eg. van Hateren 1 arcmin/pix => 60 pix/deg)
        self.fps = fps
        self.gc_df_pixspace = pd.DataFrame()
        self.spatial_filter_sidelen = 0
        self.microm_per_pix = 0
        self.temporal_filter_len = 0

        self._initialize_digital_sampling()

        self.model_type = self.context.my_retina["model_type"]

        if self.model_type == "VAE":
            self.spat_rf = self.data_io.get_data(
                filename=self.context.my_retina["spatial_rfs_file"],
            )

            mask_filename = "_".join(
                [
                    Path(self.context.my_retina["spatial_rfs_file"]).stem,
                    "center_mask.npy",
                ]
            )
            self.spat_rf_center_mask = self.data_io.get_data(
                filename=mask_filename,
            )

        self.initialized = True

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

    def _get_crop_pixels(self, cell_index):
        """
        Get pixel coordinates for stimulus crop that is the same size as the spatial filter

        :param cell_index: int
        :return:
        """
        if isinstance(cell_index, int) or isinstance(cell_index, np.int32):
            cell_index = np.array([cell_index])
        gc = self.gc_df_pixspace.iloc[cell_index]
        q_center = np.round(gc.q_pix).astype(int).values
        r_center = np.round(gc.r_pix).astype(int).values

        side_halflen = (
            self.spatial_filter_sidelen - 1
        ) // 2  # crops have width = height

        qmin = q_center - side_halflen
        qmax = q_center + side_halflen
        rmin = r_center - side_halflen
        rmax = r_center + side_halflen

        return qmin, qmax, rmin, rmax

    def _create_spatial_filter_FIT(self, cell_index, get_masks=False):
        """
        Creates the spatial component of the spatiotemporal filter

        Parameters
        ----------
        cell_index : int
            Index of the cell in the gc_df

        Returns
        -------
        spatial_filter : np.ndarray
            Spatial filter for the given cell
        """

        offset = 0.0
        s = self.spatial_filter_sidelen

        gc = self.gc_df_pixspace.iloc[cell_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)

        x_grid, y_grid = np.meshgrid(
            np.arange(qmin, qmax + 1, 1), np.arange(rmin, rmax + 1, 1)
        )

        orient_cen = gc.orient_cen * (np.pi / 180)
        # spatial_kernel is here 1-dim vector
        spatial_kernel = self.DoG2D_fixed_surround(
            (x_grid, y_grid),
            gc.ampl_c,
            gc.q_pix,
            gc.r_pix,
            gc.semi_xc,
            gc.semi_yc,
            orient_cen,
            gc.ampl_s,
            gc.relat_sur_diam,
            offset,
        )
        spatial_kernel = np.reshape(spatial_kernel, (s, s))

        # Skip scaling for now
        # # Scale the spatial filter so that its maximal gain is something reasonable
        # # TODO - how should you scale the kernel??
        # max_gain = np.max(np.abs(np.fft.fft2(spatial_kernel)))
        # # 5.3 here just to give exp(5.3) = 200 Hz max firing rate to sinusoids
        # spatial_kernel = (5.3 / max_gain) * spatial_kernel

        if get_masks:
            # Create center mask
            center_mask = np.zeros((s, s))
            center_mask[spatial_kernel > 0] = 1
            spatial_kernel = center_mask.astype(bool)

        return spatial_kernel

    def _create_spatial_filter_VAE(self, cell_index, get_masks=False):
        """
        Creates the spatial component of the spatiotemporal filter

        Parameters
        ----------
        cell_index : int
            Index of the cell in the gc_df

        Returns
        -------
        spatial_filter : np.ndarray
            Spatial filter for the given cell
        """
        offset = 0.0
        s = self.spatial_filter_sidelen

        gc = self.gc_df_pixspace.iloc[cell_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)

        x_grid, y_grid = np.meshgrid(
            np.arange(qmin, qmax + 1, 1), np.arange(rmin, rmax + 1, 1)
        )

        orient_cen = gc.orient_cen * (np.pi / 180)

        if get_masks == True:
            spatial_kernel = resize(
                self.spat_rf_center_mask[cell_index, :, :], (s, s), anti_aliasing=False
            )

        else:
            spatial_kernel = resize(
                self.spat_rf[cell_index, :, :], (s, s), anti_aliasing=True
            )

            # Skip scaling for now
            # # Scale the spatial filter so that its maximal gain is something reasonable
            # max_gain = np.max(np.abs(np.fft.fft2(spatial_kernel)))
            # # The 18 is arbitrary, to give reasonable firing rates
            # spatial_kernel = (18 / max_gain) * spatial_kernel

        return spatial_kernel

    def _create_temporal_filter(self, cell_index):
        """
        Creates the temporal component of the spatiotemporal filter. Linear fixed-sum of two lowpass filters.

        Parameters
        ----------
        cell_index : int
            Index of the cell in the gc_df

        Returns
        -------
        temporal_filter : np.ndarray
            Temporal filter for the given cell
        """

        filter_params = self.gc_df.iloc[cell_index][["n", "p1", "p2", "tau1", "tau2"]]
        if self.response_type == "off":
            filter_params[1] = (-1) * filter_params[1]
            filter_params[2] = (-1) * filter_params[2]

        tvec = np.linspace(0, self.data_filter_duration, self.temporal_filter_len)
        temporal_filter = self.diff_of_lowpass_filters(tvec, *filter_params)

        # Scale to sum of 1 for the low-pass filters. This is comparable to gain control model.
        scaling_params = self.gc_df.iloc[cell_index][["n", "p1", "tau1"]]
        scaling_filter = self.lowpass(tvec, *scaling_params)
        temporal_filter = temporal_filter / np.sum(scaling_filter)

        return temporal_filter

    def _generator_to_firing_rate(self, cell_indices, generator_potential):
        """ """
        A = 440  # TODO take this from the Benardette data
        # tonic_drive = 3.0

        def logistic_function(x, max_fr=1, k=1, x0=1):
            """
            Logistic Function

            :param x: input value
            :param max_fr: the maximum value of the curve
            :param k: steepness of the curve
            :param x0: the sigmoid's midpoint
            :return: output value
            """
            return max_fr / (1 + np.exp(-k * (x - x0)))

        def equation(k, fr, td):
            return logistic_function(0, max_fr=fr, k=k, x0=1) - td

        tonic_drives = self.gc_df.iloc[cell_indices].tonicdrive
        firing_rates = np.zeros((len(cell_indices), generator_potential.shape[1]))

        for idx, cell_idx in enumerate(cell_indices):
            tonic_drive = tonic_drives.iloc[idx]
            # Find the value of k that makes the logistic function output tonic_drive at x=0
            k = fsolve(equation, 1, args=(A, tonic_drive))[0]
            firing_rates[idx] = logistic_function(
                generator_potential[0, :], max_fr=A, k=k, x0=1
            )

        return firing_rates

    def _save_for_cxsystem(
        self, spike_mons, n_units, filename=None, analog_signal=None, dt=None
    ):
        self.w_coord, self.z_coord = self.get_w_z_coords()

        # Copied from CxSystem2\cxsystem2\core\stimuli.py The Stimuli class does not support reuse
        print(" -  Saving spikes, rgc coordinates and analog signal (if not None)...")

        data_to_save = {}
        for ii in range(len(spike_mons)):
            data_to_save["spikes_" + str(ii)] = []
            # units, i in cxsystem2
            data_to_save["spikes_" + str(ii)].append(spike_mons[ii][0])
            # times, t in cxsystem2
            data_to_save["spikes_" + str(ii)].append(spike_mons[ii][1])
        data_to_save["w_coord"] = self.w_coord
        data_to_save["z_coord"] = self.z_coord

        data_to_save["n_units"] = n_units

        if analog_signal is not None:
            data_to_save["analog_signal"] = analog_signal

        if dt is not None:
            data_to_save["dt"] = dt

        if filename is None:
            save_path = self.context.output_folder.joinpath("most_recent_spikes")
        else:
            save_path = self.context.output_folder.joinpath(filename)
        self.output_file_extension = ".gz"

        filename_full = Path(str(save_path) + self.output_file_extension)
        self.data_io.write_to_file(filename_full, data_to_save)

    def _get_extents_deg(self):
        """
        Get the stimulus/screen extents in degrees

        Parameters
        ----------
        None

        Returns
        -------
        video_extent_deg : list
            Extents of the stimulus in degrees
        """

        video_xmin_deg = self.stimulus_center.real - self.stimulus_width_deg / 2
        video_xmax_deg = self.stimulus_center.real + self.stimulus_width_deg / 2
        video_ymin_deg = self.stimulus_center.imag - self.stimulus_height_deg / 2
        video_ymax_deg = self.stimulus_center.imag + self.stimulus_height_deg / 2
        # left, right, bottom, top
        video_extent_deg = [
            video_xmin_deg,
            video_xmax_deg,
            video_ymin_deg,
            video_ymax_deg,
        ]

        return video_extent_deg

    def _initialize_digital_sampling(self):
        """
        Endows RGCs with stimulus/pixel space coordinates.
        """

        # Endow RGCs with pixel coordinates.
        # NB! Here we make a new dataframe where everything is in pixels
        pixspace_pos = np.array(
            [
                self._vspace_to_pixspace(gc.x_deg, gc.y_deg)
                for index, gc in self.gc_df.iterrows()
            ]
        )
        pixspace_coords = pd.DataFrame(
            {"q_pix": pixspace_pos[:, 0], "r_pix": pixspace_pos[:, 1]}
        )

        self.gc_df_pixspace = pd.concat([self.gc_df, pixspace_coords], axis=1)

        # Scale RF axes to pixel space
        self.gc_df_pixspace.semi_xc = self.gc_df.semi_xc * self.pix_per_deg
        self.gc_df_pixspace.semi_yc = self.gc_df.semi_yc * self.pix_per_deg

        # Define spatial filter sidelength (based on angular resolution and widest semimajor axis)
        # We use the general rule that the sidelength should be at least 5 times the SD
        # Sidelength always odd number
        self.spatial_filter_sidelen = (
            2
            * 3
            * int(
                max(
                    max(
                        self.gc_df_pixspace.semi_xc * self.gc_df_pixspace.relat_sur_diam
                    ),
                    max(
                        self.gc_df_pixspace.semi_yc * self.gc_df_pixspace.relat_sur_diam
                    ),
                )
            )
            + 1
        )

        self.microm_per_pix = (1 / self.deg_per_mm) / self.pix_per_deg * 1000

        # Get temporal parameters from stimulus video
        # self.video_fps = self.stimulus_video.fps
        self.temporal_filter_len = int(self.data_filter_duration / (1000 / self.fps))

    def _get_spatially_cropped_video(self, cell_index, contrast=True, reshape=False):
        """
        Crops the video to the surroundings of the specified Retinal Ganglion Cells (RGCs).

        The function works by first determining the pixel range to be cropped for each cell
        in cell_index, and then selecting those pixels from the original video. The cropping
        is done for each frame of the video. If the contrast option is set to True, the video
        is also rescaled to have pixel values between -1 and 1.

        Parameters
        ----------
        cell_index : array of ints
            Indices for the RGCs. The function will crop the video around each cell
            specified in this array.

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
        stimulus_cropped : np.ndarray
            The cropped video. The dimensions of the array are
            (number_of_cells, number_of_pixels_w, number_of_pixels_h, number_of_time_points)
            if reshape is False, and
            (number_of_cells, number_of_pixels, number_of_time_points)
            if reshape is True.
        """

        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)
        # video_copy = self.stimulus_video.frames.copy()
        video_copy = np.tile(
            self.stimulus_video.frames.copy(), (len(cell_index), 1, 1, 1)
        )

        sidelen = self.spatial_filter_sidelen

        # Create the r and q indices for each cell, ensure they're integer type
        r_indices = (
            (np.arange(sidelen) + rmin[:, np.newaxis])
            .astype(int)
            .reshape(-1, 1, sidelen, 1)
        )
        q_indices = (
            (np.arange(sidelen) + qmin[:, np.newaxis])
            .astype(int)
            .reshape(-1, sidelen, 1, 1)
        )

        # Create r_matrix and q_matrix by broadcasting r_indices and q_indices
        r_matrix, q_matrix = np.broadcast_arrays(r_indices, q_indices)

        # create a cell index array and a time_points index array
        cell_indices = (
            np.arange(len(cell_index)).astype(np.int32).reshape(-1, 1, 1, 1)
        )  # shape: (len(cell_index), 1, 1, 1)
        time_points_indices = np.arange(video_copy.shape[-1]).astype(
            np.int32
        )  # shape: (n_time_points,)

        # expand the indices arrays to the shape of r_matrix and q_matrix using broadcasting
        cell_indices = cell_indices + np.zeros_like(
            r_matrix, dtype=np.int32
        )  # shape: (len(cell_index), sidelen, sidelen)
        time_points_indices = time_points_indices + np.zeros(
            (1, 1, 1, video_copy.shape[-1]), dtype=np.int32
        )  # shape: (1, 1, 1, n_time_points)

        # use the index arrays to select the elements from video_copy
        stimulus_cropped = video_copy[
            cell_indices, r_matrix, q_matrix, time_points_indices
        ]

        if contrast is True:
            # Returns Weber constrast
            stimulus_cropped = stimulus_cropped / 127.5 - 1.0
        else:
            stimulus_cropped = stimulus_cropped.astype(np.uint16)

        if reshape is True:
            n_frames = np.shape(self.stimulus_video.frames)[-1]
            # reshape the video
            stimulus_cropped = stimulus_cropped.reshape(
                (len(cell_index), sidelen**2, n_frames)
            )

        return stimulus_cropped

    def get_w_z_coords(self):
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
        z_coord = self.gc_df["x_deg"].values + 1j * self.gc_df["y_deg"].values

        # Macaque values
        # a for macaques should be 0.3 - 0.9, Schwartz 1994 citing Wilson et al 1990 "The perception of form" in Visual perception: The neurophysiological foundations, Academic Press
        # k has been pretty open.
        # However, if we relate 1/M = (a/k) + (1/k) * E and M = (1/0.077) + (1/(0.082 * E)), we get
        # Andrew James, personal communication: k=1/.082, a=. 077/.082
        a = 0.077 / 0.082  # ~ 0.94
        k = 1 / 0.082  # ~ 12.2
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord

    def load_stimulus(self, stimulus_video=None):
        """
        Loads stimulus video

        Parameters
        ----------
        stimulus_video : VideoBaseClass, optional
            Visual stimulus to project to the ganglion cell mosaic. The default is None.
            If None, the stimulus video is loaded from the stimulus_video_name attribute
            of the stimulus metadata dictionary.

        Returns
        -------
        None.

        Attributes
        ----------
        stimulus_video : VideoBaseClass
            Visual stimulus to project to the ganglion cell mosaic
        """

        # Set basic working_retina attributes
        if self.initialized is False:
            self._initialize()

        if stimulus_video is None:
            video_file_name = self.context.my_stimulus_options["stimulus_video_name"]

            stimulus_video = self.data_io.load_stimulus_from_videofile(video_file_name)

        assert (stimulus_video.video_width == self.stimulus_width_pix) & (
            stimulus_video.video_height == self.stimulus_height_pix
        ), "Check that stimulus dimensions match those of the mosaic"
        assert (
            stimulus_video.fps == self.fps
        ), "Check that stimulus frame rate matches that of the mosaic"
        assert (
            stimulus_video.pix_per_deg == self.pix_per_deg
        ), "Check that stimulus resolution matches that of the mosaic"

        # Get parameters from the stimulus object
        self.stimulus_video = stimulus_video
        assert (
            np.min(stimulus_video.frames) >= 0 and np.max(stimulus_video.frames) <= 255
        ), "Stimulus pixel values must be between 0 and 255"

        # Drop RGCs whose center is not inside the stimulus
        xmin, xmax, ymin, ymax = self._get_extents_deg()
        for index, gc in self.gc_df_pixspace.iterrows():
            if (
                (gc.x_deg < xmin)
                | (gc.x_deg > xmax)
                | (gc.y_deg < ymin)
                | (gc.y_deg > ymax)
            ):
                self.gc_df.iloc[index] = 0.0  # all columns set as zero

    def create_spatiotemporal_filter(self, cell_index, called_from_loop=False):
        """
        Returns the outer product of the spatial and temporal filters in stimulus space.

        Parameters
        ----------
        cell_index : int
            Index of the RGC whose filter is to be created
        called_from_loop : bool, optional
            If True, the function is called from a loop. The default is False.

        Returns
        -------
        spatiotemporal_filter : np.ndarray
            Outer product of the spatial and temporal filters
            The row-dimension is the number of pixels in the stimulus
            The column-dimension is the number of frames in the stimulus
        """

        if self.model_type == "FIT":
            spatial_filter = self._create_spatial_filter_FIT(cell_index)
        elif self.model_type == "VAE":
            spatial_filter = self._create_spatial_filter_VAE(cell_index)
        else:
            raise ValueError("Unknown model type, aborting...")
        s = self.spatial_filter_sidelen
        spatial_filter_1d = np.array([np.reshape(spatial_filter, s**2)]).T

        temporal_filter = self._create_temporal_filter(cell_index)

        spatiotemporal_filter = (
            spatial_filter_1d * temporal_filter
        )  # (Nx1) * (1xT) = NxT

        if called_from_loop is False:
            self.spatiotemporal_filter_to_show = {
                "spatial_filter": spatial_filter,
                "temporal_filter": temporal_filter,
                "cell_index": cell_index,
            }

        return spatiotemporal_filter

    def get_temporal_filters(self, cell_indices):
        """
        Retrieve temporal filters for an array of cells.

        This function generates temporal filters for each cell specified by the
        cell indices. The temporal filter for a specific cell is obtained by calling
        the `_create_temporal_filter` method.

        Parameters
        ----------
        cell_indices : array_like
            List or 1-D array of cell indices for which to generate temporal filters.

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

        temporal_filters = np.zeros((len(cell_indices), self.temporal_filter_len))

        for idx, cell_index in enumerate(cell_indices):
            temporal_filters[idx, :] = self._create_temporal_filter(cell_index)

        return temporal_filters

    def get_spatial_filters(self, cell_indices, get_masks=False):
        """
        Generate spatial filters for given cell indices.

        This function takes a list of cell indices, determines the model type,
        creates a corresponding spatial filter for each cell index based on the model,
        and then reshapes the filter to 1-D. It returns a 2-D array where each row is a
        1-D spatial filter for a corresponding cell index.

        Parameters
        ----------
        cell_indices : array_like
            List or 1-D array of cell indices for which to generate spatial filters.
        get_masks : bool, optional
            If True, return center masks instead of spatial filters. The default is False.

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
          - self.model_type: a string indicating the type of model used. Expected values are 'FIT' or 'VAE'.
          - self.spatial_filter_sidelen: an integer specifying the side length of a spatial filter.
        """
        s = self.spatial_filter_sidelen
        spatial_filters = np.zeros((len(cell_indices), s**2))
        for idx, cell_index in enumerate(cell_indices):
            if self.model_type == "FIT":
                spatial_filter = self._create_spatial_filter_FIT(
                    cell_index, get_masks=get_masks
                )
            elif self.model_type == "VAE":
                spatial_filter = self._create_spatial_filter_VAE(
                    cell_index, get_masks=get_masks
                )
            else:
                raise ValueError("Unknown model type, aborting...")
            spatial_filter_1d = np.array([np.reshape(spatial_filter, s**2)]).T
            spatial_filters[idx, :] = spatial_filter_1d.squeeze()

        return spatial_filters

    def convolve_stimulus_batched(
        self, cell_indices, stimulus_cropped, spatiotemporal_filter
    ):
        """
        Convolves the stimulus with the spatiotemporal filter for a given set of cells.

        This function performs a convolution operation between the cropped stimulus and
        a spatiotemporal filter for each specified cell. It uses either PyTorch (if available)
        or numpy and scipy to perform the convolution. After the convolution, it adds a tonic drive to the
        generator potential of each cell.

        Parameters
        ----------
        cell_indices : array_like
            Indices of the cells to convolve the stimulus with.
        stimulus_cropped : ndarray
            Cropped stimulus to be convolved with the spatiotemporal filter, shape should be
            [num_cells, num_pixels, time_steps].
        spatiotemporal_filter : ndarray
            Spatiotemporal filter used in the convolution, shape should be
            [num_cells, num_pixels, time_steps].

        Returns
        -------
        ndarray
            Generator potential of each cell, array of shape (num_cells, stimulus timesteps),
            after the convolution and the addition of the tonic drive.

        Raises
        ------
        AssertionError
            If there is a mismatch between the duration of the stimulus and the duration of the generator potential.

        Notes
        -----
        The `num_cells` should be the same for `cell_indices`, `stimulus_cropped`, and `spatiotemporal_filter`.
        The `num_pixels` and `time_steps` should be the same for `stimulus_cropped` and `spatiotemporal_filter`.
        """

        stim_len_tp = stimulus_cropped.shape[-1]
        # stimulus_size_pix = stimulus_cropped.shape[1]
        num_cells = len(cell_indices)
        video_dt = (1 / self.stimulus_video.fps) * b2u.second

        # Move to GPU if possible. Both give the same result, but PyTorch@GPU is faster.
        if "torch" in sys.modules:  # 0:  #
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Dimensions are [batch_size, num_channels, time_steps]. We use pixels as channels.
            stimulus_cropped = torch.tensor(stimulus_cropped).float().to(device)
            spatiotemporal_filter = (
                torch.tensor(spatiotemporal_filter).float().to(device)
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
                (num_cells, stim_len_tp),
                device=device,
            )
            for i in range(num_cells):
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
            filter_length = spatiotemporal_filter.shape[-1]
            assert (
                self.context.my_stimulus_options["baseline_start_seconds"]
                >= filter_length * 1 / self.stimulus_video.fps
            ), f"baseline_start_seconds must be longer than filter length ({filter_length * video_dt}), aborting..."

            generator_potential = np.empty((num_cells, stim_len_tp - filter_length + 1))
            for idx in range(num_cells):
                generator_potential[idx, :] = convolve(
                    stimulus_cropped[idx], spatiotemporal_filter[idx], mode="valid"
                )

            # Add some padding to the beginning so that stimulus time and generator potential time match
            # (First time steps of stimulus are not convolved)
            n_padding = int(self.data_filter_duration * b2u.ms / video_dt - 1)
            generator_potential = np.pad(
                generator_potential,
                ((0, 0), (n_padding, 0)),
                mode="edge",
            )

        # Internal test for convolution operation
        generator_potential_duration_tp = generator_potential.shape[-1]
        assert (
            stim_len_tp == generator_potential_duration_tp
        ), "Duration mismatch, check convolution operation, aborting..."

        return generator_potential

    def _create_temporal_signal_gc(self, tvec, svec, dt, params):
        """
        Contrast gain control implemented in temporal domain according to Victor_1987_JPhysiol
        """

        # Convert to appropriate units
        dt = dt / b2u.ms  # sampling period in ms
        tvec = tvec / b2u.ms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move input arguments to GPU
        tvec = torch.tensor(tvec, device=device)
        svec = torch.tensor(svec, device=device)
        dt = torch.tensor(dt, device=device)

        Tc = torch.tensor(
            15.0, device=device
        )  # 15  # Time constant for dynamical variable c(t), ms. Victor_1987_JPhysiol

        # parameter_names for parasol gain control ["NL", "TL", "HS", "T0", "Chalf", "D"]
        NL = torch.tensor(int(np.round(params["NL"])), device=device)  # 30
        # 1.44  # ms Low-pass fr_cutoff = 1 / (2pi * TL) = 110 Hz
        TL = torch.tensor(params["TL"], device=device)
        HS = torch.tensor(params["HS"], device=device)
        # 37.34  # ms High-pass fr_cutoff = 1 / (2pi * T0) = 2.12 Hz
        T0 = torch.tensor(params["T0"], device=device)
        # 0.015 dummy for testing
        Chalf = torch.tensor(params["Chalf"], device=device)
        D = params["D"]

        ### Low pass filter ###

        # Calculate the impulse response function.
        h = (
            (1 / torch.math.factorial(NL))
            * (tvec / TL) ** (NL - 1)
            * torch.exp(-tvec / TL)
        )

        # # Dummy kernel for testing impulse response
        # h0 = torch.zeros(len(tvec), device=device)
        # h0[100] = 1.0
        # svec = h0.to(dtype=torch.float64)

        # Convolving two signals involves "flipping" one signal and then sliding it
        # across the other signal. PyTorch, however, does not flip the kernel, so we
        # need to do it manually.
        h_flipped = torch.flip(h, dims=[0])

        # Scale the impulse response to have unit area
        h_flipped = h_flipped / torch.sum(h_flipped)

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
                h_flipped.view(1, 1, -1),
                padding=0,
            ).squeeze()
            * dt
        )

        if self.response_type == "off":
            x_t_vec = -x_t_vec

        ### High pass stages ###
        c_t = y_t = torch.tensor(0.0, device=device)
        Ts_t = T0
        yvec = torch.zeros(len(tvec), device=device)
        Ts_vec = torch.ones(len(tvec), device=device) * T0
        c_t_vec = torch.zeros(len(tvec), device=device)
        for idx, this_time in enumerate(tvec[1:]):
            y_t = y_t + dt * (
                (-y_t / Ts_t)
                + (x_t_vec[idx] - x_t_vec[idx - 1]) / dt
                + (((1 - HS) * x_t_vec[idx]) / Ts_t)
            )
            Ts_t = T0 / (1 + c_t / Chalf)
            c_t = c_t + dt * ((torch.abs(y_t) - c_t) / Tc)
            yvec[idx] = y_t
            Ts_vec[idx] = Ts_t
            c_t_vec[idx] = c_t

        # End of pytorch loop
        yvec = yvec.cpu().numpy()
        tvec = tvec.cpu().numpy()
        dt = dt.cpu().numpy()

        # Ts_vec = Ts_vec.cpu().numpy()
        # c_t_vec = c_t_vec.cpu().numpy()
        # x_t_vec = x_t_vec.cpu().numpy()
        # plt.plot(tvec, yvec)
        # plt.plot(tvec, Ts_vec)
        # plt.plot(tvec, c_t_vec)
        # plt.plot(tvec, x_t_vec)
        # legend = ["y", "Ts", "c", "x"]
        # plt.legend(legend)

        # time shift rvec by delay D
        D_tp = int(D / dt)
        temporal_signal = np.concatenate((np.zeros(len(tvec)), np.zeros(D_tp)))
        temporal_signal[D_tp:] = yvec
        temporal_signal = temporal_signal[: len(tvec)]

        return temporal_signal

    def _create_temporal_signal(self, tvec, svec, dt, params, region):
        """
        Dynamic temporal signal for midget cells
        """

        # Convert to appropriate units
        dt = dt / b2u.ms  # sampling period in ms
        tvec = tvec / b2u.ms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move input arguments to GPU
        tvec = torch.tensor(tvec, device=device)
        svec = torch.tensor(svec, device=device)
        dt = torch.tensor(dt, device=device)

        # Get D value for surround by summing D_cen and deltaNLTL_sur
        if region == "sur":
            params["D_sur"] = params["D_cen"] + params["deltaNLTL_sur"]

        # Manipulate the index labels and D value for surround
        selected_params = params[params.index.str.contains(region)]

        # Substring to remove
        substring_to_remove = "_" + region

        # Create new index labels by removing the substring
        new_index_labels = [
            label.replace(substring_to_remove, "") for label in selected_params.index
        ]

        # Assign the new index labels to the DataFrame
        selected_params.index = new_index_labels
        params = selected_params
        # parameter_names for midget gain control ["NL", "NLTL", "TS", "HS", "D"]
        NL = torch.tensor(int(np.round(params["NL"])), device=device)  # 30
        TL = torch.tensor(params["NLTL"] / params["NL"], device=device)
        HS = torch.tensor(params["HS"], device=device)
        TS = torch.tensor(params["TS"], device=device)
        D = params["D"]

        ### Low pass filter ###
        # Calculate the impulse response function.
        h = (
            (1 / torch.math.factorial(NL))
            * (tvec / TL) ** (NL - 1)
            * torch.exp(-tvec / TL)
        )

        # With large values of NL, the impulse response function runs out of humour (becomes inf or nan)
        # at later latencies. We can avoid this by setting these inf and nan values of h to zero.
        h[torch.isinf(h)] = 0
        h[torch.isnan(h)] = 0

        # Convolving two signals involves "flipping" one signal and then sliding it
        # across the other signal. PyTorch, however, does not flip the kernel, so we
        # need to do it manually.
        h_flipped = torch.flip(h, dims=[0])

        # Scale the impulse response to have unit area
        h_flipped = h_flipped / torch.sum(h_flipped)

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
                h_flipped.view(1, 1, -1),
                padding=0,
            ).squeeze()
            * dt
        )

        if self.response_type == "off":
            x_t_vec = -x_t_vec

        ### High pass stages ###
        y_t = torch.tensor(0.0, device=device)
        yvec = torch.zeros(len(tvec), device=device)
        for idx, this_time in enumerate(tvec[1:]):
            y_t = y_t + dt * (
                (-y_t / TS)
                + (x_t_vec[idx] - x_t_vec[idx - 1]) / dt
                + (((1 - HS) * x_t_vec[idx]) / TS)
            )
            yvec[idx] = y_t

        # End of pytorch loop
        yvec = yvec.cpu().numpy()
        tvec = tvec.cpu().numpy()
        dt = dt.cpu().numpy()

        # time shift rvec by delay D
        D_tp = int(D / dt)
        temporal_signal = np.concatenate((np.zeros(len(tvec)), np.zeros(D_tp)))
        temporal_signal[D_tp:] = yvec
        temporal_signal = temporal_signal[: len(tvec)]

        return temporal_signal

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

    def _create_dynamic_contrast(
        self, stimulus_cropped, spatial_filters, gc_type, masks, surround=False
    ):
        """
        Create dynamic contrast signal by multiplying the stimulus with the spatial filter
        masks are used for midget cells, where center and surround have distinct dynamics.

        Parameters
        ----------
        stimulus_cropped : array
            Stimulus cropped to the size of the spatial filter
        spatial_filters : array
            Spatial filter
        masks : array
            Mask for center (ones), None for parasols

        Returns
        -------
        center_surround_filters_sum : array
            Dynamic contrast signal
        """

        # Reshape masks and spatial_filters to match the dimensions of stimulus_cropped
        spatial_filters_reshaped = np.expand_dims(spatial_filters, axis=2)

        if gc_type is "parasol":
            masks = np.ones_like(spatial_filters_reshaped)  # mask with all ones
        elif gc_type is "midget":
            if surround is True:
                # Surround is always negative at this stage
                masks = spatial_filters < 0
            masks = np.expand_dims(masks, axis=2)

        # Multiply the arrays using broadcasting.
        # This is the stimulus contrast viewed through spatial rf filter
        center_surround_filters = spatial_filters_reshaped * stimulus_cropped * masks

        # # Activate to show surround and exit, QA
        # if surround is True:
        #     self._show_surround_and_exit(center_surround_filters, spatial_filters)

        # Sum over spatial dimension. Collapses the filter into one temporal signal.
        center_surround_filters_sum = np.nansum(center_surround_filters, axis=1)

        # victor_1987_JPhysiol: input to model is s(t)), the signed Weber contrast at the centre.
        # However, we assume that the surround suppression is early (horizontal cells) and linear,
        # so we approximate s(t) = RF * stimulus
        svecs = center_surround_filters_sum
        return svecs

    def run_cells(
        self,
        cell_index=None,
        n_trials=1,
        save_data=False,
        spike_generator_model="refractory",
        return_monitor=False,
        filename=None,
        simulation_dt=0.001,
    ):
        """
        Runs the LNP pipeline for a single ganglion cell (spiking by Brian2)

        Parameters
        ----------
        cell_index : int or None, optional
            Index of the cell to run. If None, run all cells. The default is None.
        n_trials : int, optional
            Number of trials to run. The default is 1.
        save_data : bool, optional
            Whether to save the data. The default is False.
        spike_generator_model : str, optional
            'refractory' or 'poisson'. The default is 'refractory'.
        return_monitor : bool, optional
            Whether to return a raw Brian2 SpikeMonitor. The default is False.
        filename : str, optional
            Filename to save the data to. The default is None.
        simulation_dt : float, optional
            Time step of the simulation. The default is 0.001 (1 ms)
        """

        # Save spike generation model
        self.spike_generator_model = spike_generator_model

        video_dt = (1 / self.stimulus_video.fps) * b2u.second  # input
        stim_len_tp = self.stimulus_video.video_n_frames
        duration = stim_len_tp * video_dt
        simulation_dt = simulation_dt * b2u.second  # output

        # Run all cells
        if cell_index is None:
            n_cells = len(self.gc_df.index)  # all cells
            cell_indices = np.arange(n_cells)
        # Run one cell
        else:
            n_cells = 1
            cell_indices = np.array(cell_index)

        cell_indices = np.atleast_1d(cell_indices)  # make sure it's an array

        # Get cropped stimulus, vectorized. Time to crop:  7.05 seconds
        stimulus_cropped = self._get_spatially_cropped_video(cell_indices, reshape=True)

        # Get center masks
        center_masks = self.get_spatial_filters(cell_indices, get_masks=True)

        # Get spatiotemporal filters
        spatial_filters = self.get_spatial_filters(cell_indices)

        # Scale spatial filters to sum one of centers for each unit to get veridical max contrast
        spatial_filters = (
            spatial_filters / np.sum(spatial_filters * center_masks, axis=1)[:, None]
        )
        tvec = range(stim_len_tp) * video_dt

        if self.temporal_model == "dynamic":
            # Contrast gain control depends dynamically on contrast
            num_cells = len(cell_indices)

            # Get stimulus contrast vector:  Time to get stimulus contrast:  4.34 seconds
            if self.gc_type == "parasol":
                svecs = self._create_dynamic_contrast(
                    stimulus_cropped,
                    spatial_filters,
                    self.gc_type,
                    None,
                    surround=False,
                )
            elif self.gc_type == "midget":
                svecs_cen = self._create_dynamic_contrast(
                    stimulus_cropped,
                    spatial_filters,
                    self.gc_type,
                    center_masks,
                    surround=False,
                )
                svecs_sur = self._create_dynamic_contrast(
                    stimulus_cropped,
                    spatial_filters,
                    self.gc_type,
                    None,
                    surround=True,
                )
            
            # Get generator potentials:  Time to get generator potentials:  19.6 seconds
            generator_potentials = np.empty((num_cells, stim_len_tp))
            for idx in range(num_cells):
                params = self.gc_df.loc[cell_indices[idx]]
                if self.gc_type == "parasol":
                    generator_potentials[idx, :] = self._create_temporal_signal_gc(
                        tvec, svecs[idx, :], video_dt, params
                    )
                elif self.gc_type == "midget":
                    gen_pot_cen = self._create_temporal_signal(
                        tvec, svecs_cen[idx, :], video_dt, params, "cen"
                    )
                    gen_pot_sur = self._create_temporal_signal(
                        tvec, svecs_sur[idx, :], video_dt, params, "sur"
                    )
                    generator_potentials[idx, :] = gen_pot_cen + gen_pot_sur

        elif self.temporal_model == "fixed":  # Linear model
            # Amplitude will be scaled by first (positive) lowpass filter.
            temporal_filters = self.get_temporal_filters(cell_indices)

            # Assuming spatial_filters.shape = (U, N) and temporal_filters.shape = (U, T)
            spatiotemporal_filters = (
                spatial_filters[:, :, None] * temporal_filters[:, None, :]
            )

            print("Preparing generator potential...")
            generator_potentials = self.convolve_stimulus_batched(
                cell_indices, stimulus_cropped, spatiotemporal_filters
            )

            # Experimental scaling to match approximately contrast gain model values
            generator_potentials = generator_potentials * 8.0

        # Aplies scaling and logistic function to linear impulse firing rates to get veridical firing
        firing_rates = self._generator_to_firing_rate(
            cell_indices, generator_potentials
        )

        # Let's interpolate the rate to video_dt intervals
        tvec_original = np.arange(1, self.stimulus_video.video_n_frames + 1) * video_dt
        rates_func = interp1d(
            tvec_original,
            firing_rates,
            axis=1,
            fill_value=0,
            bounds_error=False,
        )

        tvec_new = np.arange(0, duration, simulation_dt)
        interpolated_rates_array = rates_func(
            tvec_new
        )  # This needs to be 2D array for Brian!

        # Identical rates array for every trial; rows=time, columns=cell index
        inst_rates = b2.TimedArray(interpolated_rates_array.T * b2u.Hz, simulation_dt)

        # Cells in parallel (NG), trial iterations (repeated runs)
        if spike_generator_model == "refractory":
            # Create Brian NeuronGroup
            # calculate probability of firing for current timebin (eg .1 ms)
            # draw spike/nonspike from random distribution

            abs_refractory = (
                self.context.my_retina["refractory_params"]["abs_refractory"] * b2u.ms
            )
            rel_refractory = (
                self.context.my_retina["refractory_params"]["rel_refractory"] * b2u.ms
            )
            p_exp = self.context.my_retina["refractory_params"]["p_exp"]
            clip_start = (
                self.context.my_retina["refractory_params"]["clip_start"] * b2u.ms
            )
            clip_end = self.context.my_retina["refractory_params"]["clip_end"] * b2u.ms

            neuron_group = b2.NeuronGroup(
                n_cells,
                model="""
                lambda_ttlast = inst_rates(t, i) * dt * w: 1
                t_diff = clip(t - lastspike - abs_refractory, clip_start, clip_end) : second
                w = t_diff**p_exp / (t_diff**p_exp + rel_refractory**p_exp) : 1
                """,
                threshold="rand()<lambda_ttlast",
                refractory="(t-lastspike) < abs_refractory",
                dt=simulation_dt,
            )  # This is necessary for brian2 to generate lastspike variable. Does not affect refractory behavior

            spike_monitor = b2.SpikeMonitor(neuron_group)
            net = b2.Network(neuron_group, spike_monitor)

        elif spike_generator_model == "poisson":
            # Create Brian PoissonGroup
            poisson_group = b2.PoissonGroup(n_cells, rates="inst_rates(t, i)")
            spike_monitor = b2.SpikeMonitor(poisson_group)
            net = b2.Network(poisson_group, spike_monitor)
        else:
            raise ValueError(
                "Missing valid spike_generator_model, check my_run_options parameters, aborting..."
            )

        # Save brian state
        net.store()
        all_spiketrains = []
        spikemons = []
        spikearrays = []
        t_start = []
        t_end = []

        # Run cells in parallel, trials in loop
        tqdm_desc = "Simulating " + self.response_type + " " + self.gc_type + " mosaic"
        for trial in tqdm(range(n_trials), desc=tqdm_desc):
            net.restore()  # Restore the initial state
            t_start.append((net.t / b2u.second) * b2u.second)  # pq => b2u
            net.run(duration)
            t_end.append((net.t / b2u.second) * b2u.second)

            spiketrains = list(spike_monitor.spike_trains().values())
            all_spiketrains.extend(spiketrains)

            # Cxsystem spikemon save natively supports multiple monitors
            spikemons.append(spike_monitor)
            spikearrays.append(
                [
                    deepcopy(spike_monitor.it[0].__array__()),
                    deepcopy(spike_monitor.it[1].__array__()),
                ]
            )

        if save_data is True:
            self._save_for_cxsystem(
                spikearrays,
                n_units=n_cells,
                filename=filename,
                analog_signal=interpolated_rates_array,
                dt=simulation_dt,
            )

        # For save_spikes_csv. Only 1st trial is saved.
        self.simulated_spiketrains = all_spiketrains[0]

        self.gc_responses_to_show = {
            "n_trials": n_trials,
            "n_cells": n_cells,
            "all_spiketrains": all_spiketrains,
            "exp_generator_potential": firing_rates,
            "duration": duration,
            "generator_potential": firing_rates,
            "video_dt": video_dt,
            "tvec_new": tvec_new,
        }

        if return_monitor is True:
            return spike_monitor
        else:
            return spiketrains, interpolated_rates_array.flatten()

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
                return_monitor=False,
                filename=filename,
                simulation_dt=simulation_dt,
            )

    def run_all_cells(
        self,
        spike_generator_model="refractory",
        save_data=False,
    ):
        """
        Runs the LNP pipeline for all ganglion cells (legacy function)

        Parameters
        ----------
        spike_generator_model : str
            'refractory' or 'poisson'
        save_data : bool
            Whether to save the data

        """

        self.run_cells(
            cell_index=None,
            n_trials=1,
            spike_generator_model=spike_generator_model,
            save_data=save_data,
        )

    def save_spikes_csv(self, filename=None):
        """
        Saves spikes as a csv file with rows of the form cell_index, spike_time.
        This file can be used in ViSimpl:
        visimpl.AppImage -csv parasol_structure.csv parasol_spikes.csv

        Parameters
        ----------
        filename: str, optional
            Name of the file to save the spikes to. If None, the filename will be
            generated automatically.
        """
        assert (
            len(self.simulated_spiketrains) > 0
        ), "There are no simulated spiketrains to save"

        if filename is None:
            filename = self.gc_type + "_" + self.response_type + "_spikes.csv"

        filename_full = self.context.output_folder.joinpath(filename)

        spikes_df = pd.DataFrame(columns=["cell_index", "spike_time"])
        for cell_index in range(len(self.gc_df)):
            spiketrain = self.simulated_spiketrains[cell_index]
            index_array = cell_index * np.ones(len(spiketrain))
            temp_df = pd.DataFrame(
                np.column_stack((index_array, spiketrain)),
                columns=["cell_index", "spike_time"],
            )
            spikes_df = pd.concat([spikes_df, temp_df], axis=0)

        spikes_df["cell_index"] = spikes_df["cell_index"].astype(int)
        spikes_df = spikes_df.sort_values(by="spike_time")
        spikes_df.to_csv(filename_full, index=False, header=False)

    def save_structure_csv(self, filename=None):
        """
        Saves x,y coordinates of model cells to a csv file (for use in ViSimpl).

        Parameters
        ----------
        filename: str, optional
            Name of the file to save the structure to. If None, the filename will be
            generated automatically.
        """
        if filename is None:
            filename = self.gc_type + "_" + self.response_type + "_structure.csv"

        filename_full = self.context.output_folder.joinpath(filename)

        rgc_coords = self.gc_df[["x_deg", "y_deg"]].copy()
        rgc_coords["z_deg"] = 0.0

        rgc_coords.to_csv(filename_full, header=False, index=False)


class PhotoReceptor:
    """
    This class gets one image at a time, and provides the cone response.
    After instantiation, the RGC group can get one frame at a time, and the system will give an impulse response.

    This is not necessary for GC transfer function, it is not used in Chichilnisky_2002_JNeurosci Field_2010_Nature.
    Instead, they focus the pattern directly on isolated cone mosaic.
    Nevertheless, it may be useful for comparison of image input with and w/o  explicit photoreceptor.
    """

    # self.context. attributes
    _properties_list = [
        "path",
        "input_folder",
        "output_folder",
        "my_retina",
        "my_stimulus_metadata",
        "my_stimulus_options",
    ]

    def __init__(self, context, data_io) -> None:
        self._context = context.set_context(self._properties_list)
        self._data_io = data_io

        self.optical_aberration = self.context.my_retina["optical_aberration"]
        self.rm = self.context.my_retina["cone_params"]["rm"]
        self.k = self.context.my_retina["cone_params"]["k"]
        self.cone_sensitivity_min = self.context.my_retina["cone_params"][
            "sensitivity_min"
        ]
        self.cone_sensitivity_max = self.context.my_retina["cone_params"][
            "sensitivity_max"
        ]

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    def image2cone_response(self):
        image_file_name = self.context.my_stimulus_metadata["stimulus_file"]
        self.pix_per_deg = self.context.my_stimulus_metadata["pix_per_deg"]
        self.fps = self.context.my_stimulus_options["fps"]

        # Process stimulus.
        self.image = self.data_io.get_data(image_file_name)
        self.blur_image()
        self.aberrated_image2cone_response()

    def blur_image(self):
        """
        Gaussian smoothing from Navarro 1993: 2 arcmin FWHM under 20deg eccentricity.
        """

        # Turn the optical aberration of 2 arcmin FWHM to Gaussian function sigma
        sigma_in_degrees = self.optical_aberration / (2 * np.sqrt(2 * np.log(2)))
        sigma_in_pixels = self.pix_per_deg * sigma_in_degrees

        # Turn
        kernel_size = (
            5,
            5,
        )  # Dimensions of the smoothing kernel in pixels, centered in the pixel to be smoothed
        image_after_optics = cv2.GaussianBlur(
            self.image, kernel_size, sigmaX=sigma_in_pixels
        )  # sigmaY = sigmaX

        self.image_after_optics = image_after_optics

    def aberrated_image2cone_response(self):
        """
        Cone nonlinearity. Equation from Baylor_1987_JPhysiol.
        """

        # Range
        response_range = np.ptp([self.cone_sensitivity_min, self.cone_sensitivity_max])

        # Scale
        image_at_response_scale = (
            self.image * response_range
        )  # Image should be between 0 and 1
        cone_input = image_at_response_scale + self.cone_sensitivity_min

        # Cone nonlinearity
        cone_response = self.rm * (1 - np.exp(-self.k * cone_input))

        self.cone_response = cone_response

        # Save the cone response to output folder
        filename = self.context.my_stimulus_metadata["stimulus_file"]
        self.data_io.save_cone_response_to_hdf5(filename, cone_response)


# if __name__ == "__main__":
#     pass
