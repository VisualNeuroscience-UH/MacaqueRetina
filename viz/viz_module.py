# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd

# import cv2

# Comput Neurosci
import brian2.units as b2u

# Viz
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

# Builtin
import os
import sys
import pdb


class Viz:
    """
    Methods to viz_module the retina

    Some methods import object instance as call parameter (ConstructRetina, WorkingRetina, etc). 
    """

    cmap = "gist_earth"  # viridis or cividis would be best for color-blind
    _properties_list = [
        "path",
        "output_folder",
    ]

    def __init__(self, context, data_io, ana, **kwargs) -> None:

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        self._ana = ana

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # Some settings related to plotting
        self.cmap_stim = "gray"
        self.cmap_spatial_filter = "bwr"

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def ana(self):
        return self._ana

    def data_is_valid(self, data, accept_empty=False):

        try:
            data = data / data.get_best_unit()
        except:
            pass

        if accept_empty == True:
            is_valid = isinstance(data, np.ndarray)
        else:
            is_valid = isinstance(data, np.ndarray) and data.size > 0

        return is_valid


    # ApricotFits visualization
    def show_temporal_filter_response(
        self,
        mosaic,
    ):
        '''
        Show temporal filter response for each cell.
        '''
        temporal_filters_to_show = mosaic.temporal_filters_to_show
        xdata = temporal_filters_to_show['xdata']
        xdata_finer = temporal_filters_to_show['xdata_finer']
        title = temporal_filters_to_show['title']
        
        
        # get cell_ixs
        cell_ixs_list = [ci for ci in temporal_filters_to_show.keys() if ci.startswith('cell_ix_')]

        for this_cell_ix in cell_ixs_list:
            ydata = temporal_filters_to_show[f'{this_cell_ix}']['ydata']
            y_fit = temporal_filters_to_show[f'{this_cell_ix}']['y_fit']
            plt.scatter(xdata, ydata)
            plt.plot(
                xdata_finer,
                y_fit,
                c="grey",
            )
        
        N_cells = len(cell_ixs_list)
        plt.title(f'{title} ({N_cells} cells)')

    def show_spatial_filter_response(
        self,
        mosaic):
        
        spatial_filters_to_show  = mosaic.spatial_filters_to_show

        data_all_viable_cells = spatial_filters_to_show['data_all_viable_cells']
        x_grid = spatial_filters_to_show['x_grid']
        y_grid = spatial_filters_to_show['y_grid']
        surround_model = spatial_filters_to_show['surround_model']
        pixel_array_shape_x = spatial_filters_to_show['pixel_array_shape_x']
        pixel_array_shape_y = spatial_filters_to_show['pixel_array_shape_y']

        # get cell_ixs
        cell_ixs_list = [ci for ci in spatial_filters_to_show.keys() if ci.startswith('cell_ix_')]

        for this_cell_ix in cell_ixs_list:
            imshow_cmap = "bwr"
            ellipse_edgecolor = "black"

            this_cell_ix_numerical = int(this_cell_ix.split('_')[-1])
            popt = data_all_viable_cells[this_cell_ix_numerical, :]
            data_array = spatial_filters_to_show[this_cell_ix]['data_array']
            suptitle = spatial_filters_to_show[this_cell_ix]['suptitle']

            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
            plt.suptitle(suptitle, fontsize=10,)
            cen = ax1.imshow(
                data_array,
                vmin=-0.1,
                vmax=0.4,
                cmap=imshow_cmap,
                origin="lower",
                extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
            )
            fig.colorbar(cen, ax=ax1)

            # # Ellipses for DoG2D_fixed_surround

            if surround_model == 1:
                # xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset
                data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)

                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[4],
                    -popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[7] * popt[3],
                    popt[7] * popt[4],
                    -popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )
                print(
                    popt[0],
                    popt[np.array([1, 2])],
                    popt[3],
                    popt[4],
                    -popt[5] * 180 / np.pi,
                )
                print(popt[6], "sur_ratio=", popt[7], "offset=", popt[8])
            else:
                data_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[4],
                    -popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([7, 8])]),
                    popt[9],
                    popt[10],
                    -popt[11] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )
                print(
                    popt[0],
                    popt[np.array([1, 2])],
                    popt[3],
                    popt[4],
                    -popt[5] * 180 / np.pi,
                )
                print(
                    popt[6],
                    popt[np.array([7, 8])],
                    popt[9],
                    popt[10],
                    -popt[11] * 180 / np.pi,
                )

            print("\n")

            ax1.add_artist(e1)
            ax1.add_artist(e2)

            sur = ax2.imshow(
                data_fitted.reshape(pixel_array_shape_y, pixel_array_shape_x),
                vmin=-0.1,
                vmax=0.4,
                cmap=imshow_cmap,
                origin="lower",
            )
            fig.colorbar(sur, ax=ax2)
            plt.show()


    # ConstructRetina visualization
    def show_gc_positions_and_density(self, mosaic):
        """
        Show retina cell positions and receptive fields

        ConstructRetina call.
        """

        rho = mosaic.gc_df['positions_eccentricity'].to_numpy()
        phi = mosaic.gc_df['positions_polar_angle'].to_numpy()
        gc_density_func_params = mosaic.gc_density_func_params

        # to cartesian
        xcoord, ycoord = self.pol2cart(rho, phi)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(xcoord.flatten(), ycoord.flatten(), "b.", label=mosaic.gc_type)
        ax[0].axis("equal")
        ax[0].legend()
        ax[0].set_title("Cartesian retina")
        ax[0].set_xlabel("Eccentricity (mm)")
        ax[0].set_ylabel("Elevation (mm)")

        # quality control for density.
        nbins = 50
        # Fit for published data
        edge_ecc = np.linspace(np.min(rho), np.max(rho), nbins)
        my_gaussian_fit = self.gauss_plus_baseline(edge_ecc, *gc_density_func_params)
        my_gaussian_fit_current_GC = my_gaussian_fit * mosaic.gc_proportion
        ax[1].plot(edge_ecc, my_gaussian_fit_current_GC, "r")

        # Density of model cells
        index = np.all(
            [
                phi > np.min(mosaic.theta),
                phi < np.max(mosaic.theta),
                rho > np.min(mosaic.eccentricity_in_mm),
                rho < np.max(mosaic.eccentricity_in_mm),
            ],
            axis=0,
        )  # Index only cells within original requested theta
        hist, bin_edges = np.histogram(rho[index], nbins)
        center_ecc = bin_edges[:-1] + ((bin_edges[1:] - bin_edges[:-1]) / 2)
        area_for_each_bin = self.sector2area(
            bin_edges[1:], np.ptp(mosaic.theta)
        ) - self.sector2area(
            bin_edges[:-1], np.ptp(mosaic.theta)
        )  # in mm2. Vector length len(edge_ecc) - 1.
        # Cells/area
        model_cell_density = hist / area_for_each_bin  # in cells/mm2
        ax[1].plot(center_ecc, model_cell_density, "b.")

    def visualize_mosaic(self, mosaic):
        """
        Plots the full ganglion cell mosaic. Note that this is slow if you have a large patch.

        ConstructRetina call.

        :return:
        """

        rho = mosaic.gc_df["positions_eccentricity"].to_numpy()
        phi = mosaic.gc_df["positions_polar_angle"].to_numpy()

        gc_rf_models = np.zeros((len(mosaic.gc_df), 6))
        gc_rf_models[:, 0] = mosaic.gc_df["semi_xc"]
        gc_rf_models[:, 1] = mosaic.gc_df["semi_yc"]
        gc_rf_models[:, 2] = mosaic.gc_df["xy_aspect_ratio"]
        gc_rf_models[:, 3] = mosaic.gc_df["amplitudes"]
        gc_rf_models[:, 4] = mosaic.gc_df["sur_ratio"]
        gc_rf_models[:, 5] = mosaic.gc_df["orientation_center"]

        # to cartesian
        xcoord, ycoord = self.pol2cart(rho, phi)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(xcoord.flatten(), ycoord.flatten(), "b.", label = mosaic.gc_type)

        if mosaic.surround_fixed:
            # gc_rf_models parameters:'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
            # Ellipse parameters: Ellipse(xy, width, height, angle=0, **kwargs). Only possible one at the time, unfortunately.
            for index in np.arange(len(xcoord)):
                ellipse_center_x = xcoord[index]
                ellipse_center_y = ycoord[index]
                semi_xc = gc_rf_models[index, 0]
                semi_yc = gc_rf_models[index, 1]
                # angle_in_radians = gc_rf_models[index, 5]  # Orientation
                angle_in_deg = gc_rf_models[index, 5]  # Orientation
                diameter_xc = semi_xc * 2
                diameter_yc = semi_yc * 2
                e1 = Ellipse(
                    (ellipse_center_x, ellipse_center_y),
                    diameter_xc,
                    diameter_yc,
                    angle_in_deg,
                    edgecolor="b",
                    linewidth=0.5,
                    fill=False,
                )
                ax.add_artist(e1)

        ax.axis("equal")
        ax.legend()
        ax.set_title("Cartesian retina")
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel("Elevation (mm)")

    def show_spatial_statistics(self, mosaic):
        """
        Show histograms of receptive field parameters

        ConstructRetina call.
        """
        ydata = mosaic.spatial_statistics_to_show["ydata"]
        spatial_statistics_dict = mosaic.spatial_statistics_to_show["spatial_statistics_dict"]
        model_fit_data = mosaic.spatial_statistics_to_show["model_fit_data"]

        distributions = [key for key in spatial_statistics_dict.keys()]
        n_distributions = len(spatial_statistics_dict)

        # plot the distributions and fits.
        fig, axes = plt.subplots(2, 3, figsize=(13, 4))
        axes = axes.flatten()
        for index in np.arange(n_distributions):
            ax = axes[index]

            bin_values, foo, foo2 = ax.hist(ydata[:, index], bins=20, density=True)

            if model_fit_data != None:  # Assumes tuple of arrays, see below
                x_model_fit, y_model_fit = model_fit_data[0], model_fit_data[1]
                ax.plot(
                    x_model_fit[:, index],
                    y_model_fit[:, index],
                    "r-",
                    linewidth=6,
                    alpha=0.6,
                )

                spatial_statistics_dict[distributions[index]]
                shape = spatial_statistics_dict[distributions[index]]["shape"]
                loc = spatial_statistics_dict[distributions[index]]["loc"]
                scale = spatial_statistics_dict[distributions[index]]["scale"]
                model_function = spatial_statistics_dict[distributions[index]][
                    "distribution"
                ]

                if model_function == "gamma":
                    ax.annotate(
                        "shape = {0:.2f}\nloc = {1:.2f}\nscale = {2:.2f}".format(
                            shape, loc, scale
                        ),
                        xy=(0.6, 0.4),
                        xycoords="axes fraction",
                    )
                    ax.set_title(
                        "{0} fit for {1}".format(model_function, distributions[index])
                    )
                elif model_function == "beta":
                    a_parameter, b_parameter = shape[0], shape[1]
                    ax.annotate(
                        "a = {0:.2f}\nb = {1:.2f}\nloc = {2:.2f}\nscale = {3:.2f}".format(
                            a_parameter, b_parameter, loc, scale
                        ),
                        xy=(0.6, 0.4),
                        xycoords="axes fraction",
                    )
                    ax.set_title(
                        "{0} fit for {1}".format(model_function, distributions[index])
                    )

                # Rescale y axis if model fit goes high. Shows histogram better
                if y_model_fit[:, index].max() > 1.5 * bin_values.max():
                    ax.set_ylim([ax.get_ylim()[0], 1.1 * bin_values.max()])

        # Check correlations
        # distributions = ['semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center']
        fig2, axes2 = plt.subplots(2, 3, figsize=(13, 4))
        axes2 = axes2.flatten()
        ref_index = 1
        for index in np.arange(n_distributions):
            ax2 = axes2[index]
            data_all_x = ydata[:, ref_index]
            data_all_y = ydata[:, index]

            r, p = stats.pearsonr(data_all_x, data_all_y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data_all_x, data_all_y
            )
            ax2.plot(data_all_x, data_all_y, ".")
            data_all_x.sort()
            ax2.plot(data_all_x, intercept + slope * data_all_x, "b-")
            ax2.annotate(
                "\nr={0:.2g},\np={1:.2g}".format(r, p),
                xy=(0.8, 0.4),
                xycoords="axes fraction",
            )
            ax2.set_title(
                "Correlation between {0} and {1}".format(
                    distributions[ref_index], distributions[index]
                )
            )

    def show_dendrite_diam_vs_ecc(self, mosaic):
        '''
        ConstructRetina call.
        '''
        data_all_x = mosaic.dendrite_diam_vs_ecc_to_show["data_all_x"]
        data_all_y = mosaic.dendrite_diam_vs_ecc_to_show["data_all_y"]
        polynomials = mosaic.dendrite_diam_vs_ecc_to_show["polynomials"]
        dataset_name = mosaic.dendrite_diam_vs_ecc_to_show["dataset_name"]
        title = mosaic.dendrite_diam_vs_ecc_to_show["title"]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(data_all_x, data_all_y, ".")

        if dataset_name:
            if (
                len(polynomials) == 2
            ):  # check if only two parameters, ie intercept and slope
                intercept = polynomials[1]
                slope = polynomials[0]
                ax.plot(data_all_x, intercept + slope * data_all_x, "k--")
                ax.annotate(
                    "{0} : \ny={1:.1f} + {2:.1f}x".format(
                        dataset_name, intercept, slope
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )
            elif len(polynomials) == 3:
                intercept = polynomials[2]
                slope = polynomials[1]
                square = polynomials[0]
                ax.plot(
                    data_all_x,
                    intercept + slope * data_all_x + square * data_all_x**2,
                    "k--",
                )
                ax.annotate(
                    "{0}: \ny={1:.1f} + {2:.1f}x + {3:.1f}x^2".format(
                        dataset_name, intercept, slope, square
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )
            elif len(polynomials) == 4:
                intercept = polynomials[3]
                slope = polynomials[2]
                square = polynomials[1]
                cube = polynomials[0]
                ax.plot(
                    data_all_x,
                    intercept
                    + slope * data_all_x
                    + square * data_all_x**2
                    + cube * data_all_x**3,
                    "k--",
                )
                ax.annotate(
                    "{0}: \ny={1:.1f} + {2:.1f}x + {3:.1f}x^2 + {4:.1f}x^3".format(
                        dataset_name, intercept, slope, square, cube
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

        plt.title(title)

    def show_temp_stat(self, mosaic):
        '''
        Show the temporal statistics of the mosaic.
        ConstructRetina call.
        '''

        temporal_filter_parameters = mosaic.temp_stat_to_show["temporal_filter_parameters"]
        distrib_params = mosaic.temp_stat_to_show["distrib_params"]
        suptitle = mosaic.temp_stat_to_show["suptitle"]   
        all_fits_df = mosaic.temp_stat_to_show["all_fits_df"]
        good_data_indices = mosaic.temp_stat_to_show["good_data_indices"]
        
        plt.subplots(2, 3)
        plt.suptitle(suptitle)
        for i, param_name in enumerate(temporal_filter_parameters):
            plt.subplot(2, 3, i + 1)
            ax = plt.gca()
            shape, loc, scale = distrib_params[i, :]
            param_array = np.array(all_fits_df.iloc[good_data_indices][param_name])

            x_min, x_max = stats.gamma.ppf(
                [0.001, 0.999], a=shape, loc=loc, scale=scale
            )
            xs = np.linspace(x_min, x_max, 100)
            pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
            ax.plot(xs, pdf)
            ax.hist(param_array, density=True)
            ax.set_title(param_name)

    def show_tonic_drives(self, mosaic):
        '''
        ConstructRetina call.
        '''
        xs = mosaic.tonic_drives_to_show["xs"]
        pdf = mosaic.tonic_drives_to_show["pdf"]
        tonicdrive_array = mosaic.tonic_drives_to_show["tonicdrive_array"]
        title = mosaic.tonic_drives_to_show["title"]

        plt.plot(xs, pdf)
        plt.hist(tonicdrive_array, density=True)
        plt.title(title)
        plt.xlabel("Tonic drive (a.u.)")

    def show_build_process(self, mosaic, show_all_spatial_fits=False):
        '''
        
        '''
        
        # If show_all_spatial_fits is true, show the spatial fits
        if show_all_spatial_fits is True:
            self.show_spatial_filter_response(mosaic)
            return

        self.show_temporal_filter_response(mosaic)

        self.show_gc_positions_and_density(mosaic) 
        self.visualize_mosaic(mosaic)
        self.show_spatial_statistics(mosaic)
        self.show_dendrite_diam_vs_ecc(mosaic)
        self.show_temp_stat(mosaic)
        self.show_tonic_drives(mosaic)


    # WorkingRetina visualization
    def show_stimulus_with_gcs(self, retina, frame_number=0, ax=None, example_gc=5):
        """
        Plots the 1SD ellipses of the RGC mosaic

        WorkingRetina call.

        :param frame_number: int
        :param ax: matplotlib Axes object
        :return:
        """

        stimulus_video = retina.stimulus_video
        gc_df_pixspace = retina.gc_df_pixspace
        stimulus_height_pix = retina.stimulus_height_pix
        pix_per_deg = retina.pix_per_deg
        deg_per_mm = retina.deg_per_mm
        stimulus_center = retina.stimulus_center

        fig = plt.figure()
        ax = ax or plt.gca()
        ax.imshow(stimulus_video.frames[:, :, frame_number], vmin=0, vmax=255)
        ax = plt.gca()

        for index, gc in gc_df_pixspace.iterrows():
            # When in pixel coordinates, positive value in Ellipse angle is clockwise. Thus minus here.
            # Note that Ellipse angle is in degrees.
            # Width and height in Ellipse are diameters, thus x2.
            if index == example_gc:
                facecolor = "yellow"
            else:
                facecolor = "None"

            circ = Ellipse(
                (gc.q_pix, gc.r_pix),
                width=2 * gc.semi_xc,
                height=2 * gc.semi_yc,
                angle=gc.orientation_center * (-1),
                edgecolor="blue",
                facecolor=facecolor,
            )
            ax.add_patch(circ)

        # Annotate
        # Get y tics in pixels
        locs, labels = plt.yticks()

        # Remove tick marks outside stimulus
        locs = locs[locs < stimulus_height_pix]
        # locs=locs[locs>=0] # Including zero seems to shift center at least in deg
        locs = locs[locs > 0]

        # Set left y tick labels (pixels)
        left_y_labels = locs.astype(int)
        # plt.yticks(ticks=locs, labels=left_y_labels)
        plt.yticks(ticks=locs)
        ax.set_ylabel("pix")

        # Set x tick labels (degrees)
        xlocs = locs - np.mean(locs)
        down_x_labels = np.round(xlocs / pix_per_deg, decimals=2) + np.real(
            stimulus_center
        )
        plt.xticks(ticks=locs, labels=down_x_labels)
        ax.set_xlabel("deg")

        # Set right y tick labels (mm)
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.tick_params(axis="y")
        right_y_labels = np.round(
            (locs / pix_per_deg) / deg_per_mm, decimals=2
        )
        plt.yticks(ticks=locs, labels=right_y_labels)
        ax2.set_ylabel("mm")

        fig.tight_layout()

    def show_single_gc_view(self, retina, cell_index, frame_number=0, ax=None):
        """
        Plots the stimulus frame cropped to RGC surroundings

        WorkingRetina call.

        :param cell_index: int
        :param frame_number: int
        :param ax: matplotlib Axes object
        :return:
        """

        stimulus_video = retina.stimulus_video
        gc_df_pixspace = retina.gc_df_pixspace
        qmin, qmax, rmin, rmax = retina._get_crop_pixels(cell_index)

        ax = ax or plt.gca()

        gc = gc_df_pixspace.iloc[cell_index]

        # Show stimulus frame cropped to RGC surroundings & overlay 1SD center RF on top of that
        ax.imshow(
            stimulus_video.frames[:, :, frame_number],
            cmap=self.cmap_stim,
            vmin=0,
            vmax=255,
        )
        ax.set_xlim([qmin, qmax])
        ax.set_ylim([rmax, rmin])

        # When in pixel coordinates, positive value in Ellipse angle is clockwise. Thus minus here.
        # Note that Ellipse angle is in degrees.
        # Width and height in Ellipse are diameters, thus x2.
        circ = Ellipse(
            (gc.q_pix, gc.r_pix),
            width=2 * gc.semi_xc,
            height=2 * gc.semi_yc,
            angle=gc.orientation_center * (-1),
            edgecolor="white",
            facecolor="yellow",
        )
        ax.add_patch(circ)
        plt.xticks([])
        plt.yticks([])

    def plot_tf_amplitude_response(self, retina, cell_index, ax=None):
        '''
        WorkingRetina call.
        '''
        tf = retina._create_temporal_filter(cell_index)
        data_filter_duration = retina.data_filter_duration

        ax = ax or plt.gca()

        ft_tf = np.fft.fft(tf)
        timestep = data_filter_duration / len(tf) / 1000  # in seconds
        freqs = np.fft.fftfreq(tf.size, d=timestep)
        amplitudes = np.abs(ft_tf)

        ax.set_xscale("log")
        ax.set_xlim([0.1, 100])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        ax.plot(freqs, amplitudes, ".")

    def plot_midpoint_contrast(self, retina, cell_index, ax=None):
        """
        Plots the contrast in the mid-pixel of the stimulus cropped to RGC surroundings

        WorkingRetina call.

        :param cell_index:
        :return:
        """
        stimulus_cropped = retina._get_cropped_video(cell_index)
        spatial_filter_sidelen = retina.spatial_filter_sidelen
        stimulus_video = retina.stimulus_video

        midpoint_ix = (spatial_filter_sidelen - 1) // 2
        signal = stimulus_cropped[midpoint_ix, midpoint_ix, :]

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        ax = ax or plt.gca()
        ax.plot(tvec, signal)
        ax.set_ylim([-1, 1])

    def plot_local_rms_contrast(self, retina, cell_index, ax=None):
        """
        Plots local RMS contrast in the stimulus cropped to RGC surroundings.
        Note that is just a frame-by-frame computation, no averaging here

        WorkingRetina call.

        :param cell_index:
        :return:
        """
        # get stimulus intensities
        stimulus_cropped = retina._get_cropped_video(
            cell_index, contrast=False) 
        stimulus_video = retina.stimulus_video
        spatial_filter_sidelen = retina.spatial_filter_sidelen
        
        n_frames = stimulus_video.video_n_frames
        sidelen = spatial_filter_sidelen
        signal = np.zeros(n_frames)

        for t in range(n_frames):
            frame_mean = np.mean(stimulus_cropped[:, :, t])
            squared_sum = np.sum((stimulus_cropped[:, :, t] - frame_mean) ** 2)
            signal[t] = np.sqrt(1 / (frame_mean**2 * sidelen**2) * squared_sum)

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        ax = ax or plt.gca()
        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

    def plot_local_michelson_contrast(self, retina, cell_index, ax=None):
        """
        Plots local RMS contrast in the stimulus cropped to RGC surroundings.
        Note that is just a frame-by-frame computation, no averaging here

        WorkingRetina call.

        :param cell_index:
        :return:
        """
        # get stimulus intensities
        stimulus_cropped = retina._get_cropped_video(
            cell_index, contrast=False)  
        stimulus_video = retina.stimulus_video

        n_frames = stimulus_video.video_n_frames
        signal = np.zeros(n_frames)

        # unsigned int will overflow when frame_max + frame_min = 256
        stimulus_cropped = stimulus_cropped.astype(np.int16)
        for t in range(n_frames):
            frame_min = np.min(stimulus_cropped[:, :, t])
            frame_max = np.max(stimulus_cropped[:, :, t])
            signal[t] = (frame_max - frame_min) / (frame_max + frame_min)

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt
        ax = ax or plt.gca()
        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

    def show_gc_responses(self, retina):
        '''
        WorkingRetina call.
        '''
        n_trials = retina.gc_responses_to_show["n_trials"]
        n_cells = retina.gc_responses_to_show["n_cells"]
        all_spiketrains = retina.gc_responses_to_show["all_spiketrains"]
        exp_generator_potential = retina.gc_responses_to_show["exp_generator_potential"]
        duration = retina.gc_responses_to_show["duration"]
        generator_potential = retina.gc_responses_to_show["generator_potential"]
        video_dt = retina.gc_responses_to_show["video_dt"]
        tvec_new = retina.gc_responses_to_show["tvec_new"]        

        # Prepare data for manual visualization
        if n_trials > 1 and n_cells == 1:
            for_eventplot = np.array(all_spiketrains)
            for_histogram = np.concatenate(all_spiketrains)
            for_generatorplot = exp_generator_potential.flatten()
            n_samples = n_trials
            sample_name = "Trials"
        elif n_trials == 1 and n_cells > 1:
            for_eventplot = np.concatenate(all_spiketrains)
            for_histogram = np.concatenate(all_spiketrains[0])
            for_generatorplot = np.mean(exp_generator_potential, axis=1)
            n_samples = n_cells
            sample_name = "Cell #"
        else:
            print(
                "You attempted to visualize gc activity, but you have either n_trials or n_cells must be 1, and the other > 1"
            )

        plt.subplots(2, 1, sharex=True)
        plt.subplot(211)
        # plt.eventplot(spiketrains)
        plt.eventplot(for_eventplot)
        plt.xlim([0, duration / b2u.second])
        # plt.ylabel('Trials')
        plt.ylabel(sample_name)

        plt.subplot(212)
        # Plot the generator and the average firing rate
        tvec = np.arange(0, len(generator_potential), 1) * video_dt
        # plt.plot(tvec, exp_generator_potential.flatten(), label='Generator')
        plt.plot(tvec, for_generatorplot, label="Generator")
        plt.xlim([0, duration / b2u.second])

        # Compute average firing rate over trials (should approximately follow generator)
        hist_dt = 1 * b2u.ms
        # n_bins = int((duration/hist_dt))
        bin_edges = np.append(
            tvec_new, [duration / b2u.second]
        )  # Append the rightmost edge
        # hist, _ = np.histogram(spiketrains_flat, bins=bin_edges)
        hist, _ = np.histogram(for_histogram, bins=bin_edges)
        # avg_fr = hist / n_trials / (hist_dt / b2u.second)
        avg_fr = hist / n_samples / (hist_dt / b2u.second)

        xsmooth = np.arange(-15, 15 + 1)
        smoothing = stats.norm.pdf(xsmooth, scale=5)  # Gaussian smoothing with SD=5 ms
        smoothed_avg_fr = np.convolve(smoothing, avg_fr, mode="same")

        plt.plot(bin_edges[:-1], smoothed_avg_fr, label="Measured")
        plt.ylabel("Firing rate (Hz)")
        plt.xlabel("Time (s)")

        plt.legend()

    def show_spatiotemporal_filter(
        self, retina):

        '''
        WorkingRetina call.
        '''

        spatial_filter = retina.spatiotemporal_filter_to_show["spatial_filter"]
        cell_index = retina.spatiotemporal_filter_to_show["cell_index"]
        temporal_filter = retina.spatiotemporal_filter_to_show["temporal_filter"]
        gc_type = retina.gc_type
        response_type = retina.response_type
        temporal_filter_len = retina.temporal_filter_len

        vmax = np.max(np.abs(spatial_filter))
        vmin = -vmax

        plt.subplots(1, 2, figsize=(10, 4))
        plt.suptitle(gc_type + " " + response_type + " / cell ix " + str(cell_index))
        plt.subplot(121)
        plt.imshow(spatial_filter, cmap=self.cmap_spatial_filter, vmin=vmin, vmax=vmax)
        plt.colorbar()

        plt.subplot(122)
        plt.plot(range(temporal_filter_len), np.flip(temporal_filter))

        plt.tight_layout()

    def show_convolved_stimulus(self, retina):
        '''
        WorkingRetina call.
        '''

        cell_index = retina.convolved_stimulus_to_show["cell_index"]
        generator_potential = retina.convolved_stimulus_to_show["generator_potential"]
        video_dt = retina.convolved_stimulus_to_show["video_dt"]
        tonic_drive = retina.convolved_stimulus_to_show["tonic_drive"]
        firing_rate = retina.convolved_stimulus_to_show["firing_rate"]
        gc_type = retina.gc_type
        response_type = retina.response_type

        tvec = np.arange(0, len(generator_potential), 1) * video_dt

        plt.subplots(2, 1, sharex=True)
        plt.subplot(211)
        plt.plot(tvec, generator_potential + tonic_drive)
        plt.ylabel("Generator [a.u.]")

        plt.title(gc_type + " " + response_type + " / cell ix " + str(cell_index))

        plt.subplot(212)
        plt.plot(tvec, firing_rate)
        plt.xlabel("Time (s)]")
        plt.ylabel("Firing rate (Hz)]")

    # PhotoReceptor visualization
    def show_cone_response(self, image, image_after_optics, cone_response):
        '''
        PhotoReceptor call.
        '''
        fig, ax = plt.subplots(nrows=2, ncols=3)
        axs = ax.ravel()
        axs[0].hist(image.flatten(), 20)
        axs[1].hist(image_after_optics.flatten(), 20)
        axs[2].hist(cone_response.flatten(), 20)

        axs[3].imshow(image, cmap="Greys")
        axs[4].imshow(image_after_optics, cmap="Greys")
        axs[5].imshow(cone_response, cmap="Greys")

