# Numerical
# from fileinput import filename
import numpy as np

# import scipy.optimize as opt
# import scipy.io as sio
# import scipy.stats as stats
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d

# Data IO
import cv2

# Viz
from tqdm import tqdm

# Comput Neurosci
import brian2 as b2
import brian2.units as b2u

# Local
from cxsystem2.core.tools import write_to_file, load_from_file

# from retina.apricot_fit_module import ApricotFit
from retina.retina_math_module import RetinaMath

# from retina.vae_module import ApricotVAE

# Builtin
# import sys
from pathlib import Path

# import os
from copy import deepcopy
import pdb


class WorkingRetina(RetinaMath):
    _properties_list = [
        "path",
        "output_folder",
        "my_retina",
        "my_stimulus_metadata",
        "my_stimulus_options",
        "my_run_options",
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

        :param gc_dataframe: Ganglion cell parameters; positions are retinal coordinates; positions_eccentricity in mm, positions_polar_angle in degrees
        """

        # Read fitted parameters from file
        gc_dataframe = self.data_io.get_data(
            filename=self.context.my_retina["mosaic_file_name"]
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

        # Metadata for Apricot dataset. TODO move to project_conf module
        self.data_microm_per_pixel = 60
        self.data_filter_fps = 30  # Uncertain - "30 or 120 Hz"
        self.data_filter_timesteps = 15
        self.data_filter_duration = self.data_filter_timesteps * (
            1000 / self.data_filter_fps
        )  # in milliseconds

        # Convert retinal positions (ecc, pol angle) to visual space positions in deg (x, y)
        vspace_pos = np.array(
            [
                self.pol2cart(gc.positions_eccentricity, gc.positions_polar_angle)
                for index, gc in gc_dataframe.iterrows()
            ]
        )
        vspace_pos = vspace_pos * self.deg_per_mm
        vspace_coords = pd.DataFrame(
            {"x_deg": vspace_pos[:, 0], "y_deg": vspace_pos[:, 1]}
        )

        self.gc_df = pd.concat([gc_dataframe, vspace_coords], axis=1)

        # Convert RF center radii to degrees as well
        self.gc_df.semi_xc = self.gc_df.semi_xc * self.deg_per_mm
        self.gc_df.semi_yc = self.gc_df.semi_yc * self.deg_per_mm

        # Drop retinal positions from the df (so that they are not used by accident)
        self.gc_df = self.gc_df.drop(
            ["positions_eccentricity", "positions_polar_angle"], axis=1
        )

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

        self.initialized = True

    def _vspace_to_pixspace(self, x, y):
        """
        Converts visual space coordinates (in degrees; x=eccentricity, y=elevation) to pixel space coordinates.
        In pixel space, coordinates (q,r) correspond to matrix locations, ie. (0,0) is top-left.

        :param x: eccentricity (deg)
        :param y: elevation (deg)
        :return:
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

    def _create_spatial_filter(self, cell_index):
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

        orientation_center = gc.orientation_center * (np.pi / 180)
        spatial_kernel = self.DoG2D_fixed_surround(
            (x_grid, y_grid),
            gc.amplitudec,
            gc.q_pix,
            gc.r_pix,
            gc.semi_xc,
            gc.semi_yc,
            orientation_center,
            gc.amplitudes,
            gc.sur_ratio,
            offset,
        )
        spatial_kernel = np.reshape(spatial_kernel, (s, s))

        # Scale the spatial filter so that its maximal gain is something reasonable
        # TODO - how should you scale the kernel??
        max_gain = np.max(np.abs(np.fft.fft2(spatial_kernel)))
        # 5.3 here just to give exp(5.3) = 200 Hz max firing rate to sinusoids
        spatial_kernel = (5.3 / max_gain) * spatial_kernel

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

    def _save_for_cxsystem(self, spike_mons, filename=None, analog_signal=None):

        self.w_coord, self.z_coord = self.get_w_z_coords()

        # Copied from CxSystem2\cxsystem2\core\stimuli.py The Stimuli class does not support reuse
        print(" -  Saving spikes, rgc coordinates and analog signal (if not None)...")

        data_to_save = {}
        for ii in range(len(spike_mons)):
            data_to_save["spikes_" + str(ii)] = []
            # data_to_save['spikes_' + str(ii)].append(spike_mons[ii].it[0].__array__())
            # data_to_save['spikes_' + str(ii)].append(spike_mons[ii].it[1].__array__())
            data_to_save["spikes_" + str(ii)].append(spike_mons[ii][0])
            data_to_save["spikes_" + str(ii)].append(spike_mons[ii][1])
        data_to_save["w_coord"] = self.w_coord
        data_to_save["z_coord"] = self.z_coord

        if analog_signal is not None:
            data_to_save["analog_signal"] = analog_signal

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
                    max(self.gc_df_pixspace.semi_xc * self.gc_df_pixspace.sur_ratio),
                    max(self.gc_df_pixspace.semi_yc * self.gc_df_pixspace.sur_ratio),
                )
            )
            + 1
        )

        self.microm_per_pix = (1 / self.deg_per_mm) / self.pix_per_deg * 1000

        # Get temporal parameters from stimulus video
        # self.video_fps = self.stimulus_video.fps
        self.temporal_filter_len = int(self.data_filter_duration / (1000 / self.fps))

    def _get_cropped_video(self, cell_index, contrast=True, reshape=False):
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

        # TODO - RGCs that are near the border of the stimulus will fail (no problem if stim is large enough)

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

    def _filter_from_VAE_model(self, cell_index):
        # Convolve stimulus with VAE model
        pass

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
        """

        if self.initialized is False:
            self._initialize()

        if stimulus_video is None:
            video_file_name = self.context.my_stimulus_metadata["stimulus_video_name"]

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
        # self.stimulus_video = deepcopy(stimulus_video)  # Not sure if copying the best way here...
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
        Returns the outer product of the spatial and temporal filters

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
            spatial_filter = self._create_spatial_filter(cell_index)
            s = self.spatial_filter_sidelen
            spatial_filter_1d = np.array([np.reshape(spatial_filter, s**2)]).T

            temporal_filter = self._create_temporal_filter(cell_index)

            spatiotemporal_filter = (
                spatial_filter_1d * temporal_filter
            )  # (Nx1) * (1xT) = NxT
        elif self.model_type == "VAE":
            spatiotemporal_filter = self._filter_from_VAE_model(cell_index)

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

        Returns
        -------
        generator_potential : np.ndarray
            Generator potential of the cell, array of length (stimulus timesteps)
        """

        # Get spatiotemporal filter
        spatiotemporal_filter = self.create_spatiotemporal_filter(
            cell_index, called_from_loop=called_from_loop
        )

        # Get cropped stimulus
        stimulus_cropped = self._get_cropped_video(cell_index, reshape=True)

        # Run convolution
        generator_potential = convolve(
            stimulus_cropped, spatiotemporal_filter, mode="valid"
        )
        generator_potential = generator_potential[0, :]

        # Add some padding to the beginning so that stimulus time and generator potential time match
        # (First time steps of stimulus are not convolved)
        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        n_padding = int(self.data_filter_duration * b2u.ms / video_dt - 1)
        generator_potential = np.pad(
            generator_potential, (n_padding, 0), mode="constant", constant_values=0
        )

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
    ):
        """
        Runs the LNP pipeline for a single ganglion cell (spiking by Brian2)

        :param cell_index: int or None. If None, run all cells
        :param n_trials: int
        :param show_gc_response: bool
        :param save_data: bool
        :param spike_generator_model: str, 'refractory' or 'poisson'
        :param return_monitor: bool, whether to return a raw Brian2 SpikeMonitor
        :param filename: str
        :return:
        """

        # Save spike generation model
        self.spike_generator_model = spike_generator_model

        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        duration = self.stimulus_video.video_n_frames * video_dt
        poissongen_dt = 1.0 * b2u.ms

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

        # exp_generator_potential = np.array(np.exp(generator_potential))
        # exp_generator_potential = generator_potential
        exp_generator_potential = self._generator_to_firing_rate(generator_potential)
        # exp_generator_potential = stats.norm.cdf(generator_potential)

        # Let's interpolate the rate to 1ms intervals
        tvec_original = np.arange(1, self.stimulus_video.video_n_frames + 1) * video_dt
        rates_func = interp1d(
            tvec_original,
            exp_generator_potential,
            axis=0,
            fill_value=0,
            bounds_error=False,
        )

        tvec_new = np.arange(0, duration, poissongen_dt)
        interpolated_rates_array = rates_func(
            tvec_new
        )  # This needs to be 2D array for Brian!

        # Identical rates array for every trial; rows=time, columns=cell index
        inst_rates = b2.TimedArray(interpolated_rates_array * b2u.Hz, poissongen_dt)

        # Cells in parallel (NG), trial iterations (repeated runs)
        if spike_generator_model == "refractory":
            # Create Brian NeuronGroup
            # calculate probability of firing for current timebin (eg .1 ms)
            # draw spike/nonspike from random distribution

            # Recovery function from Berry_1998_JNeurosci, Uzzell_2004_JNeurophysiol
            # abs and rel refractory estimated from Uzzell_2004_JNeurophysiol,
            # Fig 7B, bottom row, inset. Parasol ON cell
            abs_refractory = 1 * b2u.ms
            rel_refractory = 3 * b2u.ms
            p_exp = 4
            clip_start = 0 * b2u.ms
            clip_end = 100 * b2u.ms
            neuron_group = b2.NeuronGroup(
                n_cells,
                model="""
                lambda_ttlast = inst_rates(t, i) * dt * w: 1
                t_diff = clip(t - lastspike - abs_refractory, clip_start, clip_end) : second
                w = t_diff**p_exp / (t_diff**p_exp + rel_refractory**p_exp) : 1
                """,
                threshold="rand()<lambda_ttlast",
                refractory="(t-lastspike) < abs_refractory",
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

            # for old visualization
            spiketrains = np.array(list(spike_monitor.spike_trains().values()))
            all_spiketrains.append(spiketrains.flatten())

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
                spikearrays, filename=filename, analog_signal=interpolated_rates_array
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

        for filename in filenames:

            self.run_cells(
                cell_index=cell_index,
                n_trials=n_trials,
                save_data=save_data,
                spike_generator_model=spike_generator_model,
                return_monitor=False,
                filename=filename,
            )

    def run_all_cells(
        self,
        spike_generator_model="refractory",
        save_data=False,
    ):

        """
        Runs the LNP pipeline for all ganglion cells (legacy function)

        :param spike_generator_model: str, 'refractory' or 'poisson'
        :param save_data: bool
        :return:
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

        :param filename: str
        :return:
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

        :param filename: str
        :return:
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
        self.rm = self.context.my_retina["rm"]
        self.k = self.context.my_retina["k"]
        self.cone_sensitivity_min = self.context.my_retina["cone_sensitivity_min"]
        self.cone_sensitivity_max = self.context.my_retina["cone_sensitivity_max"]

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
