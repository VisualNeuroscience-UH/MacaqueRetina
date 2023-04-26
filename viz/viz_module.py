# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd

# Machine learning
import torch
import torchvision.transforms.functional as TF

# import cv2

# Comput Neurosci
import brian2.units as b2u

# Viz
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from tqdm import tqdm
# import seaborn as sns

# Local
from retina.vae_module import AugmentedDataset

# Builtin
import os
from pathlib import Path
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

    # Fit visualization
    def show_temporal_filter_response(
        self,
        mosaic,
    ):
        """
        Show temporal filter response for each cell.
        """
        exp_temp_filt_to_viz = mosaic.exp_temp_filt_to_viz
        xdata = exp_temp_filt_to_viz["xdata"]
        xdata_finer = exp_temp_filt_to_viz["xdata_finer"]
        title = exp_temp_filt_to_viz["title"]

        # get cell_ixs
        cell_ixs_list = [
            ci for ci in exp_temp_filt_to_viz.keys() if ci.startswith("cell_ix_")
        ]

        for this_cell_ix in cell_ixs_list:
            ydata = exp_temp_filt_to_viz[f"{this_cell_ix}"]["ydata"]
            y_fit = exp_temp_filt_to_viz[f"{this_cell_ix}"]["y_fit"]
            plt.scatter(xdata, ydata)
            plt.plot(
                xdata_finer,
                y_fit,
                c="grey",
            )

        N_cells = len(cell_ixs_list)
        plt.title(f"{title} ({N_cells} cells)")

    def show_spatial_filter_response(
        self,
        spat_filt_to_viz,
        n_samples=np.inf,
        title="",
        pause_to_show=False,
    ):
        data_all_viable_cells = spat_filt_to_viz["data_all_viable_cells"]
        x_grid = spat_filt_to_viz["x_grid"]
        y_grid = spat_filt_to_viz["y_grid"]
        surround_model = spat_filt_to_viz["surround_model"]
        pixel_array_shape_x = spat_filt_to_viz["num_pix_x"]
        pixel_array_shape_y = spat_filt_to_viz["num_pix_y"]

        # get cell_ixs
        cell_ixs_list = [
            ci for ci in spat_filt_to_viz.keys() if ci.startswith("cell_ix_")
        ]

        if n_samples < len(cell_ixs_list):
            cell_ixs_list = np.random.choice(cell_ixs_list, n_samples, replace=False)

        for this_cell_ix in cell_ixs_list:
            imshow_cmap = "bwr"
            ellipse_edgecolor = "black"

            this_cell_ix_numerical = int(this_cell_ix.split("_")[-1])
            popt = data_all_viable_cells[this_cell_ix_numerical, :]
            spatial_data_array = spat_filt_to_viz[this_cell_ix]["spatial_data_array"]
            suptitle = spat_filt_to_viz[this_cell_ix]["suptitle"]
            suptitle = f"{title}, {suptitle})"

            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

            plt.suptitle(
                suptitle,
                fontsize=10,
            )
            cen = ax1.imshow(
                spatial_data_array,
                # vmin=-0.1,
                # vmax=0.4,
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
                if 0:
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
                if 0:
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
                # vmin=-0.1,
                # vmax=0.4,
                cmap=imshow_cmap,
                origin="lower",
            )
            fig.colorbar(sur, ax=ax2)

            if pause_to_show:
                plt.show()

    # ConstructRetina visualization
    def show_gc_positions_and_density(self, mosaic):
        """
        Show retina cell positions and receptive fields

        ConstructRetina call.
        """

        rho = mosaic.gc_df["positions_eccentricity"].to_numpy()
        phi = mosaic.gc_df["positions_polar_angle"].to_numpy()
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
        ax.plot(xcoord.flatten(), ycoord.flatten(), "b.", label=mosaic.gc_type)

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
        ydata = mosaic.exp_spat_stat_to_viz["ydata"]
        spatial_statistics_dict = mosaic.exp_spat_stat_to_viz["spatial_statistics_dict"]
        model_fit_data = mosaic.exp_spat_stat_to_viz["model_fit_data"]

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
        """
        ConstructRetina call.
        """
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
        """
        Show the temporal statistics of the mosaic.
        ConstructRetina call.
        """

        temporal_filter_parameters = mosaic.exp_temp_stat_to_viz[
            "temporal_filter_parameters"
        ]
        distrib_params = mosaic.exp_temp_stat_to_viz["distrib_params"]
        suptitle = mosaic.exp_temp_stat_to_viz["suptitle"]
        all_data_fits_df = mosaic.exp_temp_stat_to_viz["all_data_fits_df"]
        good_data_idx = mosaic.exp_temp_stat_to_viz["good_data_idx"]

        plt.subplots(2, 3)
        plt.suptitle(suptitle)
        for i, param_name in enumerate(temporal_filter_parameters):
            plt.subplot(2, 3, i + 1)
            ax = plt.gca()
            shape, loc, scale = distrib_params[i, :]
            param_array = np.array(all_data_fits_df.iloc[good_data_idx][param_name])

            x_min, x_max = stats.gamma.ppf(
                [0.001, 0.999], a=shape, loc=loc, scale=scale
            )
            xs = np.linspace(x_min, x_max, 100)
            pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
            ax.plot(xs, pdf)
            ax.hist(param_array, density=True)
            ax.set_title(param_name)

    def show_tonic_drives(self, mosaic):
        """
        ConstructRetina call.
        """
        xs = mosaic.exp_tonic_dr_to_viz["xs"]
        pdf = mosaic.exp_tonic_dr_to_viz["pdf"]
        tonicdrive_array = mosaic.exp_tonic_dr_to_viz["tonicdrive_array"]
        title = mosaic.exp_tonic_dr_to_viz["title"]

        plt.plot(xs, pdf)
        plt.hist(tonicdrive_array, density=True)
        plt.title(title)
        plt.xlabel("Tonic drive (a.u.)")

    def show_exp_build_process(self, mosaic, show_all_spatial_fits=False):
        """
        Visualize retina mosaic build process.
        """

        # If show_all_spatial_fits is true, show the spatial fits
        if show_all_spatial_fits is True:
            spat_filt_to_viz = mosaic.exp_spat_filt_to_viz
            self.show_spatial_filter_response(
                spat_filt_to_viz,
                n_samples=np.inf,
                title="Experimental",
                pause_to_show=True,
            )
            return

        self.show_temporal_filter_response(mosaic)

        self.show_gc_positions_and_density(mosaic)
        self.visualize_mosaic(mosaic)
        self.show_spatial_statistics(mosaic)
        self.show_dendrite_diam_vs_ecc(mosaic)
        self.show_temp_stat(mosaic)
        self.show_tonic_drives(mosaic)

    def show_gen_and_exp_spatial_rfs(self, mosaic, n_samples=2):
        """
        Show the experimental (fitted) and generated spatial receptive fields

        Parameters
        ----------
        mosaic : ConstructRetina object
        n_samples : int
            Number of samples to show
        """
        spat_filt_to_viz = mosaic.exp_spat_filt_to_viz
        self.show_spatial_filter_response(
            spat_filt_to_viz,
            n_samples=n_samples,
            title="Experimental",
            pause_to_show=False,
        )

        spat_filt_to_viz = mosaic.gen_spat_filt_to_viz
        self.show_spatial_filter_response(
            spat_filt_to_viz,
            n_samples=n_samples,
            title="Generated",
            pause_to_show=False,
        )

    def show_latent_space_and_samples(self, mosaic):
        """
        Plot the latent samples on top of the estimated kde, one sublot for each successive two dimensions of latent_dim

        Parameters
        ----------
        mosaic : ConstructRetina object
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        latent_samples = mosaic.gen_latent_space_to_viz["samples"]
        latent_data = mosaic.gen_latent_space_to_viz["data"]
        latent_dim = mosaic.gen_latent_space_to_viz["dim"]

        # Make a grid of subplots
        n_cols = 4
        n_rows = int(np.ceil(latent_dim / n_cols))
        if n_rows == 1:
            n_cols = latent_dim
        elif n_rows > 4:
            n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
        fig_suptitle_text = "Latent space and samples"
        axes = axes.flatten()

        # Plot the latent samples on top of the estimated kde
        for ax_idx, i in enumerate(range(0, latent_dim, 2)):
            if ax_idx > 15:
                fig_suptitle_text = (
                    "Latent space and samples (plotting only the first 32 dimensions)"
                )
                break

            # Get only two dimensions at a time
            values = latent_data[:, [i, i + 1]].T
            # Evaluate the kde using only the same two dimensions
            kernel = stats.gaussian_kde(values)
            # Construct X and Y grids using the same two dimensions
            x = np.linspace(latent_data[:, i].min(), latent_data[:, i].max(), 100)
            y = np.linspace(
                latent_data[:, i + 1].min(), latent_data[:, i + 1].max(), 100
            )
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)

            # Plot the estimated kde and samples on top of it
            axes[ax_idx].contour(X, Y, Z, levels=10)
            axes[ax_idx].scatter(latent_samples[:, i], latent_samples[:, i + 1])

            # Make marginal plots of the contours as contours and samples as histograms.
            # Place the marginal plots on the right and top of the main plot
            ax_marg_x = inset_axes(
                axes[ax_idx],
                width="100%",  # width  of parent_bbox width
                height="30%",  # height : 1 inch
                loc="upper right",
                # bbox_to_anchor=(1.05, 1.05),
                bbox_to_anchor=(0, 0.95, 1, 0.3),
                bbox_transform=axes[ax_idx].transAxes,
                borderpad=0,
            )
            ax_marg_y = inset_axes(
                axes[ax_idx],
                width="30%",  # width of parent_bbox width
                height="100%",  # height : 1 inch
                loc="lower left",
                # bbox_to_anchor=(-0.05, -0.05),
                bbox_to_anchor=(1, 0, 0.4, 1),
                bbox_transform=axes[ax_idx].transAxes,
                borderpad=0,
            )

            # Plot the marginal plots
            nx, bins, _ = ax_marg_x.hist(latent_samples[:, i], bins=20, density=True)
            ny, bins, _ = ax_marg_y.hist(
                latent_samples[:, i + 1],
                bins=20,
                density=True,
                orientation="horizontal",
            )

            # Plot the one-dimensional marginal shapes of the kde
            x_margin_contour = nx.max() * Z.mean(axis=0) / Z.mean(axis=0).max()
            y_margin_contour = ny.max() * Z.mean(axis=1) / Z.mean(axis=1).max()
            ax_marg_x.plot(x, x_margin_contour, color="r")
            ax_marg_y.plot(y_margin_contour, y, color="r")

            # Remove the ticks from the marginal plots
            ax_marg_x.tick_params(
                axis="both", which="both", bottom=False, top=False, labelbottom=False
            )
            ax_marg_y.tick_params(
                axis="both", which="both", left=False, right=False, labelleft=False
            )

            # Set the title of the main plot
            axes[ax_idx].set_title(f"Latent dims {i}, {i+1}")

        # plt.tight_layout()
        fig.suptitle(fig_suptitle_text)

    def show_gen_spat_postprocessing(self, mosaic):
        """
        Show the original experimental spatial receptive fields and
        the generated spatial receptive fields before and after postprocessing

        Parameters
        ----------
        mosaic : ConstructRetina object
        """

        # Get the keys for the cell_ix arrays
        cell_key_list = [
            key for key in mosaic.exp_spat_filt_to_viz.keys() if "cell_ix" in key
        ]
        img_shape = mosaic.exp_spat_filt_to_viz["cell_ix_0"]["spatial_data_array"].shape
        # The shape of the array is N cells, y_pixels, x_pixels
        img_exp = np.zeros([len(cell_key_list), img_shape[0], img_shape[1]])
        for i, cell_key in enumerate(cell_key_list):
            img_exp[i, :, :] = mosaic.exp_spat_filt_to_viz[cell_key][
                "spatial_data_array"
            ]

        img_pre = mosaic.gen_spat_img_to_viz["img_raw"]
        img_post = mosaic.gen_spat_img_to_viz["img_processed"]

        plt.subplot(1, 3, 1)
        plt.hist(img_exp.flatten(), bins=100)
        # plot median value as a vertical line
        plt.axvline(np.median(img_exp), color="r")
        plt.title(f"Experimental, median: {np.median(img_exp):.2f}")

        plt.subplot(1, 3, 2)
        plt.hist(img_pre.flatten(), bins=100)
        plt.axvline(np.median(img_pre), color="r")
        plt.title(f"Generated raw, median: {np.median(img_pre):.2f}")

        plt.subplot(1, 3, 3)
        plt.hist(img_post.flatten(), bins=100)
        plt.axvline(np.median(img_post), color="r")
        plt.title(f"Generated processed, median: {np.median(img_post):.2f}")

    def show_ray_experiment(self, mosaic, ray_exp, this_dep_var):
        """
        Show the results of a ray experiment. If ray_exp is None, then
        the most recent experiment is shown.

        Parameters
        ----------
        ray_exp : RayExperiment object
        """

        info_columns = ["trial_id", "iteration"]
        dep_vars = ["train_loss", "val_loss", "mse", "ssim", "kid_mean", "kid_std"]
        dep_vars_best = ["min", "min", "min", "max", "min", "min"]
        config_prefix = "config/"

        if ray_exp is None:
            most_recent = True
        else:
            most_recent = False

        result_grid = self.data_io.load_ray_results_grid(
            most_recent=most_recent, ray_exp=ray_exp
        )
        df = result_grid.get_dataframe()

        # Get configuration variables
        config_vars_all = [c for c in df.columns if config_prefix in c]

        # Drop columns that are constant in the experiment
        constant_cols = []
        for col in config_vars_all:
            if len(df[col].unique()) == 1:
                constant_cols.append(col)
        config_vars_changed = [
            col for col in config_vars_all if col not in constant_cols
        ]
        config_vars = [col.removeprefix(config_prefix) for col in config_vars_changed]

        # Remove all rows containing nan values in the dependent variables
        df = df.dropna(subset=dep_vars)

        # Collect basic data from the experiment
        n_trials = len(df)
        n_errors = result_grid.num_errors

        # Columns to describe the experiment
        exp_info_columns = info_columns + config_vars_changed + dep_vars
        print(df[exp_info_columns].describe())

        # Find the row indeces of the n best trials
        best_trials_across_dep_vars = []
        for dep_var, dep_var_best in zip(dep_vars, dep_vars_best):
            if dep_var_best == "min":
                best_trials_across_dep_vars.append(df[dep_var].idxmin())
            elif dep_var_best == "max":
                best_trials_across_dep_vars.append(df[dep_var].idxmax())
            if this_dep_var in dep_var:
                this_dep_var_best = dep_var_best

        df_filtered = df[exp_info_columns].loc[best_trials_across_dep_vars]
        # Print the exp_info_columns for the best trials
        print(f"Best trials: in order of {dep_vars=}")
        print(df_filtered)

        nrows = 6
        ncols = len(dep_vars)
        nsamples = 10

        layout = [
            ["dh0", "dh1", "dh2", "dh3", "dh4", "dh5", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            ["dv0", "dv1", "dv2", "dv3", "dv4", "dv5", ".", ".", ".", "."],
            ["im0", "im1", "im2", "im3", "im4", "im5", "im6", "im7", "im8", "im9"],
            ["re0" + str(i) for i in range(10)],
            ["re1" + str(i) for i in range(10)],
            ["re2" + str(i) for i in range(10)],
            ["re3" + str(i) for i in range(10)],
            ["re4" + str(i) for i in range(10)],
        ]
        fig, axd = plt.subplot_mosaic(layout, figsize=(nrows, ncols * 5))

        # Fraction of best = 1/4
        frac_best = 0.25
        num_best_trials = int(len(df) * frac_best)

        self._subplot_dependent_histograms(
            axd, "dh", df, dep_vars, dep_vars_best, num_best_trials
        )

        self._subplot_dependent_variables(axd, "dv", result_grid, dep_vars, best_trials_across_dep_vars)

        if hasattr(mosaic, "exp_spat_filt_to_viz"):
            exp_spat_filt_to_viz = mosaic.exp_spat_filt_to_viz
        else:
            mosaic._initialize()
            exp_spat_filt_to_viz = mosaic.exp_spat_filt_to_viz

        num_best_trials = 5
        best_trials, dep_var_vals = self._get_best_trials(
            df, this_dep_var, this_dep_var_best, num_best_trials
        )

        img, rec_img, samples = self._get_imgs(
            df, nsamples, exp_spat_filt_to_viz, best_trials[0]
        )

        title = f"Original \nimages"
        self._subplot_img_recoimg(axd, "im", None, img, samples, title)

        title = f"Reco for \n{this_dep_var} = \n{dep_var_vals[best_trials[0]]:.3f}, \nidx = {best_trials[0]}"
        self._subplot_img_recoimg(axd, "re", 0, rec_img, samples, title)

        for idx, this_trial in enumerate(best_trials[1:]):
            img, rec_img, samples = self._get_imgs(
                df, nsamples, exp_spat_filt_to_viz, this_trial
            )

            title = f"Reco for \n{this_dep_var} = \n{dep_var_vals[this_trial]:.3f}, \nidx = {this_trial}"
            # enumerate starts at 0, so add 1
            self._subplot_img_recoimg(axd, "re", idx + 1, rec_img, samples, title)

    def _get_imgs(
        self,
        df,
        nsamples,
        exp_spat_filt_to_viz,
        this_trial_idx,
    ):

        log_dir = df["logdir"][this_trial_idx]

        # Get folder name starting "checkpoint"
        checkpoint_folder_name = [f for f in os.listdir(log_dir) if "checkpoint" in f][
            0
        ]
        checkpoint_path = Path(log_dir) / checkpoint_folder_name / "model.pth"

        # Load the model
        model = torch.load(checkpoint_path)
        encoder = model.encoder
        decoder = model.decoder

        if hasattr(model, "test_data"):
            test_data = model.test_data[:nsamples, :, :, :]
        else:
            test_data = np.zeros(
                [
                    nsamples,
                    1,
                    exp_spat_filt_to_viz["num_pix_y"],
                    exp_spat_filt_to_viz["num_pix_x"],
                ]
            )
            for i in range(nsamples):
                test_data[i, 0, :, :] = exp_spat_filt_to_viz[f"cell_ix_{i}"][
                    "spatial_data_array"
                ]
        test_data = torch.from_numpy(test_data).float()
        img_size = model.decoder.unflatten.unflattened_size
        test_data = TF.resize(test_data, img_size[-2:], antialias=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = range(0, nsamples)

        encoder.eval()
        decoder.eval()
        encoder.to(self.device)
        decoder.to(self.device)

        img = test_data.to(self.device)
        with torch.no_grad():
            rec_img = decoder(encoder(img))

        img = img.cpu().squeeze().numpy()
        rec_img = rec_img.cpu().squeeze().numpy()

        return img, rec_img, samples

    def _subplot_dependent_histograms(
        self, axd, kw, df, dep_vars, dep_vars_best, num_best_trials
    ):
        """Plot dependent variables as a function of epochs."""

        # Make one subplot for each dependent variable
        for idx, dep_var in enumerate(dep_vars):
            best_is = dep_vars_best[idx]
            best_trials, dep_var_vals = self._get_best_trials(
                df, dep_var, best_is, num_best_trials
            )

            ax = axd[f"{kw}{idx}"]

            # Make histogram of the frac_best trials
            ax.hist(dep_var_vals[best_trials], bins=20)

            # Set x and y axis tick font size 8
            ax.tick_params(axis="both", which="major", labelsize=8)

            if idx==0:
                ax.set_ylabel("Frequency")
                ax.text(
                    0,
                    1.6,
                    f"Histogram of the {num_best_trials} best trials for each metrics",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=ax.transAxes,
                    fontsize=11,
                )


            ax.set_title(f"{dep_var}")

    def _get_best_trials(self, df, dep_var, best_is, num_best_trials):
        """
        Get the indices of the best trials for a dependent variable.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the results of the hyperparameter search.
        dep_var : str
            Name of the dependent variable.
        best_is : str
            Whether the best trials are the ones with the highest or lowest values.
        num_best_trials : int
            Number of best trials to return. Overrides frac_best.

        Returns
        -------
        best_trials : list
            List of indices of the best trials.
        """

        # get array of values for this dependent variable
        dep_var_vals = df[dep_var].values

        # get the indices of the num_best_trials
        if best_is == "min":
            best_trials = np.argsort(dep_var_vals)[:num_best_trials]
        elif best_is == "max":
            best_trials = np.argsort(dep_var_vals)[-num_best_trials:]
        
        return best_trials, dep_var_vals

    def _subplot_dependent_variables(self, axd, kw, result_grid, dep_vars, best_trials):
        """Plot dependent variables as a function of epochs."""

        df = result_grid.get_dataframe()
        # Find all columns with string "config/"
        config_cols = [x for x in df.columns if "config/" in x]

        # From the config_cols, identify columns where there is more than one unique value
        # These are the columns which were varied in the search space
        varied_cols = []
        for col in config_cols:
            if len(df[col].unique()) > 1:
                varied_cols.append(col)

        # Drop the "config/" part from the column names
        varied_cols = [x.replace("config/", "") for x in varied_cols]

        # # remove "model_id" from the varied columns
        # varied_cols.remove("model_id")

        num_colors = len(best_trials)
        colors = plt.cm.get_cmap("tab20", num_colors).colors

        total_n_epochs = 0
        # Make one subplot for each dependent variable
        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = axd[f"{kw}{idx}"]

            for i, result in enumerate(result_grid):
                if i not in best_trials:
                    continue

                if idx == 0:
                    label = f"{dep_vars[color_idx]}: " + ",".join(f"{x}={result.config[x]}" for x in varied_cols)
                    legend = True
                    first_ax = ax

                else:
                    label=None
                    legend = False

                result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=legend,
                )

                if len(result.metrics_dataframe) > total_n_epochs:
                    total_n_epochs = len(result.metrics_dataframe)



                # At the end (+1) of the x-axis, add mean and SD of last 50 epochs as dot and vertical line, respectively
                last_50 = result.metrics_dataframe.tail(50)
                mean = last_50[dep_var].mean()
                std = last_50[dep_var].std()
                n_epochs = result.metrics_dataframe.tail(1)["training_iteration"]
                ax.plot(
                    n_epochs + n_epochs // 5,
                    mean,
                    "o",
                    color=colors[color_idx],
                )
                ax.plot(
                    [n_epochs + n_epochs // 5] * 2,
                    [mean - std, mean + std],
                    "-",
                    color=colors[color_idx],
                )

                color_idx += 1

            if idx==0:
                ax.set_ylabel("Metrics")

            # Add legend and bring it to the front
            leg = first_ax.legend(
                loc="center left", bbox_to_anchor=((idx + 2.0), 0.5, 1.0, 0.2)
            )
            first_ax.set_zorder(1)

            # change the line width for the legend
            for line in leg.get_lines():
                line.set_linewidth(3.0)

            # Set x and y axis tick font size 8
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Change legend font size to 8
            for text in leg.get_texts():
                text.set_fontsize(8)

            ax.grid(True)

            # set x axis labels off
            ax.set_xlabel("")
            # set x ticks off
            ax.set_xticks([])

        first_ax.set_title(f"Evolution for best trials (ad {total_n_epochs} epochs)\nDot and vertical line indicate mean and SD of last 50 epochs", loc="left")

    def _subplot_img_recoimg(self, axd, kw, subidx, img, samples, title):
        """
        Plot sample images
        """
        for pos_idx, sample_idx in enumerate(samples):
            if subidx is None:
                ax = axd[f"{kw}{pos_idx}"]
            else:
                ax = axd[f"{kw}{subidx}{pos_idx}"]
            ax.imshow(img[sample_idx], cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                # ax.set_title(title, fontsize=8, fontdict={'verticalalignment': 'baseline', 'horizontalalignment': 'left'})
                # Print title to the left of the first image. The coordinates are in axes coordinates
                ax.text(
                    -1.0,
                    0.5,
                    title,
                    fontsize=8,
                    fontdict={       
                        "verticalalignment": "baseline",
                        "horizontalalignment": "left",
                    },
                    transform=ax.transAxes,
                )

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
        right_y_labels = np.round((locs / pix_per_deg) / deg_per_mm, decimals=2)
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
        """
        WorkingRetina call.
        """
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
        stimulus_cropped = retina._get_cropped_video(cell_index, contrast=False)
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
        stimulus_cropped = retina._get_cropped_video(cell_index, contrast=False)
        stimulus_video = retina.stimulus_video

        n_frames = stimulus_video.video_n_frames
        signal = np.zeros(n_frames)

        # unsigned int will overflow when frame_max + frame_min = 256
        stimulus_cropped = stimulus_cropped.astype(np.uint16)
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
        """
        WorkingRetina call.
        """
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
            for_eventplot = all_spiketrains  # list of different leght arrays
            for_histogram = np.concatenate(all_spiketrains)
            for_generatorplot = exp_generator_potential.flatten()
            n_samples = n_trials
            sample_name = "Trials"
        elif n_trials == 1 and n_cells > 1:
            for_eventplot = all_spiketrains
            for_histogram = np.concatenate(all_spiketrains)
            for_generatorplot = np.mean(exp_generator_potential, axis=1)
            n_samples = n_cells
            sample_name = "Cell #"
        else:
            raise ValueError(
                """You attempted to visualize gc activity, but either n_trials or n_cells must be 1, and the other > 1"""
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

    def show_spatiotemporal_filter(self, retina):
        """
        WorkingRetina call.
        """

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
        """
        WorkingRetina call.
        """

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
        """
        PhotoReceptor call.
        """
        fig, ax = plt.subplots(nrows=2, ncols=3)
        axs = ax.ravel()
        axs[0].hist(image.flatten(), 20)
        axs[1].hist(image_after_optics.flatten(), 20)
        axs[2].hist(cone_response.flatten(), 20)

        axs[3].imshow(image, cmap="Greys")
        axs[4].imshow(image_after_optics, cmap="Greys")
        axs[5].imshow(cone_response, cmap="Greys")

    def plot_analog_stimulus(self, analog_input):
        data = analog_input.Input

        plt.figure()
        plt.plot(data.T)
