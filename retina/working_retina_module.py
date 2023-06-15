# Numerical
import numpy as np

import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d
import scipy.optimize as opt
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
        gc = self.gc_df_pixspace.iloc[cell_index]
        q_center = int(gc.q_pix)
        r_center = int(gc.r_pix)

        side_halflen = (
            self.spatial_filter_sidelen - 1
        ) // 2  # crops have width = height

        qmin = q_center - side_halflen
        qmax = q_center + side_halflen
        rmin = r_center - side_halflen
        rmax = r_center + side_halflen

        return qmin, qmax, rmin, rmax

    def _create_spatial_filter_FIT(self, cell_index):
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

        # Scale the spatial filter so that its maximal gain is something reasonable
        # TODO - how should you scale the kernel??
        max_gain = np.max(np.abs(np.fft.fft2(spatial_kernel)))
        # 5.3 here just to give exp(5.3) = 200 Hz max firing rate to sinusoids
        spatial_kernel = (5.3 / max_gain) * spatial_kernel

        return spatial_kernel

    def _create_spatial_filter_VAE(self, cell_index):
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

        spatial_kernel = resize(
            self.spat_rf[cell_index, :, :], (s, s), anti_aliasing=True
        )

        # Scale the spatial filter so that its maximal gain is something reasonable
        max_gain = np.max(np.abs(np.fft.fft2(spatial_kernel)))
        # The 18 is arbitrary, to give reasonable firing rates
        spatial_kernel = (18 / max_gain) * spatial_kernel

        return spatial_kernel

    def _create_temporal_filter(self, cell_index):
        """
        Creates the temporal component of the spatiotemporal filter

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

        # Scale the temporal filter so that its maximal gain is 1
        # TODO - how should you scale the kernel??
        max_gain = np.max(np.abs(np.fft.fft(temporal_filter)))
        temporal_filter = (1 / max_gain) * temporal_filter

        return temporal_filter

    def _generator_to_firing_rate(self, generator_potential):
        firing_rate = np.power(generator_potential, 2)

        return firing_rate

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
        pdb.set_trace()

        self.microm_per_pix = (1 / self.deg_per_mm) / self.pix_per_deg * 1000

        # Get temporal parameters from stimulus video
        # self.video_fps = self.stimulus_video.fps
        self.temporal_filter_len = int(self.data_filter_duration / (1000 / self.fps))

    def _get_spatially_cropped_video(self, cell_index, contrast=True, reshape=False):
        """
        Crops the video to RGC surroundings

        Parameters
        ----------
        cell_index : int
            Index of the RGC
        contrast : bool
            If True, the video is scaled to [-1, 1] (default: True)
        reshape : bool
            If True, the video is reshaped to (n_frames, n_pixels, n_pixels, n_channels) (default: False)

        Returns
        -------
        stimulus_cropped : np.ndarray
            Cropped video
        """

        # NOTE: RGCs that are near the border of the stimulus will fail (no problem if stim is large enough)

        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)
        stimulus_cropped = self.stimulus_video.frames[
            rmin : rmax + 1, qmin : qmax + 1, :
        ].copy()

        # Scale stimulus pixel values from [0, 255] to [-1.0, 1.0]
        # TODO - This is brutal and unphysiological, at least for natural movies
        if contrast is True:
            stimulus_cropped = stimulus_cropped / 127.5 - 1.0
        else:
            # unsigned int will overflow when frame_max + frame_min >= 256
            stimulus_cropped = stimulus_cropped.astype(np.uint16)

        if reshape is True:
            sidelen = self.spatial_filter_sidelen
            n_frames = np.shape(self.stimulus_video.frames)[2]

            stimulus_cropped = np.reshape(stimulus_cropped, (sidelen**2, n_frames))

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

        pdb.set_trace()
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

    def convolve_stimulus(self, cell_index, called_from_loop=False):
        """
        Convolves the stimulus with the stimulus filter

        Parameters
        ----------
        cell_index : int
            Index of the cell to convolve the stimulus with
        called_from_loop : bool, optional
            Whether the method is called from a loop, by default False

        Returns
        -------
        np.ndarray
            Generator potential of the cell, array of shape (stimulus timesteps,)
        """
        # Get spatiotemporal filter
        pdb.set_trace()
        spatiotemporal_filter = self.create_spatiotemporal_filter(
            cell_index, called_from_loop=called_from_loop
        )
        pdb.set_trace()

        # Get cropped stimulus
        stimulus_cropped = self._get_spatially_cropped_video(cell_index, reshape=True)
        stimulus_duration_tp = stimulus_cropped.shape[-1]
        video_dt = (1 / self.stimulus_video.fps) * b2u.second

        # Move to GPU if possible
        if "torch" in sys.modules:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pad_value = stimulus_cropped.mean(axis=0)[0]

            # Reshape to 4D (adding batch_size and num_channels dimensions)
            stimulus_cropped = (
                torch.tensor(stimulus_cropped).unsqueeze(0).float().to(device)
            )
            spatiotemporal_filter = (
                torch.tensor(spatiotemporal_filter).unsqueeze(0).float().to(device)
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
                stimulus_cropped, (padding_size, 0), value=pad_value
            )

            # Run the convolution.
            # NOTE: Crops the output by filter_length - 1 for unknown reasons.
            generator_potential = torch.nn.functional.conv1d(
                stimulus_padded,
                spatiotemporal_filter_flipped,
                padding=0,
            )

            # Move back to CPU and convert to numpy
            generator_potential = generator_potential.cpu().squeeze().numpy()

        else:
            # Run convolution. NOTE: expensive computation. Solution without torch.
            # if mode is "valid", the output consists only of those elements that do not rely on the zero-padding
            # if baseline is shorter than filter, the output is truncated
            filter_length = spatiotemporal_filter.shape[-1]
            assert (
                self.context.my_stimulus_options["baseline_start_seconds"]
                > filter_length * 1 / self.stimulus_video.fps
            ), f"baseline_start_seconds must be longer than filter length ({filter_length * video_dt}), aborting..."
            generator_potential = convolve(
                stimulus_cropped, spatiotemporal_filter, mode="valid"
            )
            generator_potential = generator_potential[0, :]

            # Add some padding to the beginning so that stimulus time and generator potential time match
            # (First time steps of stimulus are not convolved)
            n_padding = int(self.data_filter_duration * b2u.ms / video_dt - 1)
            generator_potential = np.pad(
                generator_potential, (n_padding, 0), mode="constant", constant_values=0
            )

        # Internal test for convolution operation
        generator_potential_duration_tp = generator_potential.shape[-1]
        assert (
            stimulus_duration_tp == generator_potential_duration_tp
        ), "Duration mismatch, check convolution operation, aborting..."

        tonic_drive = self.gc_df.iloc[cell_index].tonicdrive

        firing_rate = self._generator_to_firing_rate(generator_potential + tonic_drive)

        if called_from_loop is False:
            self.convolved_stimulus_to_show = {
                "generator_potential": generator_potential,
                "cell_index": cell_index,
                "video_dt": video_dt,
                "tonic_drive": tonic_drive,
                "firing_rate": firing_rate,
            }

        # Return the 1-dimensional generator potential
        return generator_potential + tonic_drive

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

        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        duration = self.stimulus_video.video_n_frames * video_dt
        simulation_dt = simulation_dt * b2u.second

        # Run all cells
        if cell_index is None:
            n_cells = len(self.gc_df.index)  # all cells
            cell_index = np.arange(n_cells)
        # Run one cell
        else:
            n_cells = 1
            cell_index = np.array(cell_index)

        cell_index = np.atleast_1d(cell_index)  # python is not always so simple...

        # Get instantaneous firing rate
        print("Preparing generator potential...")
        generator_potential = np.zeros([self.stimulus_video.video_n_frames, n_cells])
        for idx, this_cell in enumerate(cell_index):
            generator_potential[:, idx] = self.convolve_stimulus(
                this_cell, called_from_loop=True
            )

        exp_generator_potential = self._generator_to_firing_rate(generator_potential)

        # Let's interpolate the rate to 1ms intervals
        tvec_original = np.arange(1, self.stimulus_video.video_n_frames + 1) * video_dt
        rates_func = interp1d(
            tvec_original,
            exp_generator_potential,
            axis=0,
            fill_value=0,
            bounds_error=False,
        )

        tvec_new = np.arange(0, duration, simulation_dt)
        interpolated_rates_array = rates_func(
            tvec_new
        )  # This needs to be 2D array for Brian!

        # Identical rates array for every trial; rows=time, columns=cell index
        inst_rates = b2.TimedArray(interpolated_rates_array * b2u.Hz, simulation_dt)

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
            "exp_generator_potential": exp_generator_potential,
            "duration": duration,
            "generator_potential": generator_potential,
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
