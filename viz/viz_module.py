# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
from sklearn.manifold import TSNE

# Machine learning
import torch
import torchvision.transforms.functional as TF
from torchsummary import summary

# import cv2

# Comput Neurosci
import brian2.units as b2u

# Viz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# from tqdm import tqdm

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

    def _figsave(self, figurename="", myformat="png", subfolderpath="", suffix=""):
        """
        Save the current figure to the working directory or a specified subfolder path.

        This method saves the current figure with various customization options for the
        filename, format, and location. By default, figures are saved as 'MyFigure.png'.
        The figure's font settings are configured such that fonts are preserved as they are,
        and not converted into paths.

        Parameters
        ----------
        figurename : str, optional
            The name of the figure file. If it's specified with an extension, the figure
            is saved with that name. If it's a relative path, the figure is saved to that path.
            If not provided, the figure is saved as 'MyFigure.png'. Defaults to "".
        myformat : str, optional
            The format of the figure (e.g., 'png', 'jpg', 'svg', etc.).
            If provided with a leading ".", the "." is removed. Defaults to 'png'.
        subfolderpath : str, optional
            The subfolder within the working directory to which the figure is saved.
            If figurename is a path, this value will be overridden by the parent directory
            of figurename. Defaults to "".
        suffix : str, optional
            A suffix that is added to the end of the filename, just before the file extension.
            Defaults to "".

        Returns
        -------
        None

        Notes
        -----
        - The fonts in the figure are configured to be saved as fonts, not as paths.
        - If the specified subfolder doesn't exist, it is created.
        - If both `figurename` and `subfolderpath` are paths, `figurename` takes precedence,
        and `subfolderpath` is overridden.
        """

        plt.rcParams["svg.fonttype"] = "none"  # Fonts as fonts and not as paths
        plt.rcParams["ps.fonttype"] = "type3"  # Fonts as fonts and not as paths

        # Confirm pathlib type
        figurename = Path(figurename)
        subfolderpath = Path(subfolderpath)

        # Check if figurename is a path. If yes, parse the figurename and subfolderpath
        if str(figurename.parent) != ".":
            subfolderpath = figurename.parent
            figurename = Path(figurename.name)

        if myformat[0] == ".":
            myformat = myformat[1:]

        filename, file_extension = figurename.stem, figurename.suffix

        filename = filename + suffix

        if not file_extension:
            file_extension = "." + myformat

        if not figurename:
            figurename = "MyFigure" + file_extension
        else:
            figurename = filename + file_extension

        path = self.context.path
        figurename_fullpath = Path.joinpath(path, subfolderpath, figurename)
        full_subfolderpath = Path.joinpath(path, subfolderpath)
        if not Path.is_dir(full_subfolderpath):
            Path.mkdir(full_subfolderpath)
        print(f"Saving figure to {figurename_fullpath}")
        plt.savefig(
            figurename_fullpath,
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=file_extension[1:],
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )

    # Fit visualization
    def show_temporal_filter_response(
        self,
    ):
        """
        Show temporal filter response for each cell.
        """
        exp_temp_filt_to_viz = self.construct_retina.exp_temp_filt_to_viz
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
            # Get DoG model fit parameters to popt
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
                # xy_tuple, ampl_c, xoc, yoc, semi_xc, semi_yc, orient_cen, ampl_s, relat_sur_diam, offset
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
                    print(popt[6], "relat_sur_diam=", popt[7], "offset=", popt[8])
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
    def show_gc_positions_and_density(self):
        """
        Show retina cell positions and receptive fields

        ConstructRetina call.
        """

        rho = self.construct_retina.gc_df["pos_ecc_mm"].to_numpy()
        phi = self.construct_retina.gc_df["pos_polar_deg"].to_numpy()
        gc_density_func_params = self.construct_retina.gc_density_func_params

        # to cartesian
        xcoord, ycoord = self.pol2cart(rho, phi)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(
            xcoord.flatten(),
            ycoord.flatten(),
            "b.",
            label=self.construct_retina.gc_type,
        )
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
        my_gaussian_fit_current_GC = (
            my_gaussian_fit * self.construct_retina.gc_proportion
        )
        ax[1].plot(edge_ecc, my_gaussian_fit_current_GC, "r")

        # Density of model cells
        index = np.all(
            [
                phi > np.min(self.construct_retina.polar_lim_deg),
                phi < np.max(self.construct_retina.polar_lim_deg),
                rho > np.min(self.construct_retina.ecc_lim_mm),
                rho < np.max(self.construct_retina.ecc_lim_mm),
            ],
            axis=0,
        )  # Index only cells within original requested polar_lim_deg
        hist, bin_edges = np.histogram(rho[index], nbins)
        center_ecc = bin_edges[:-1] + ((bin_edges[1:] - bin_edges[:-1]) / 2)
        area_for_each_bin = self.sector2area(
            bin_edges[1:], np.ptp(self.construct_retina.polar_lim_deg)
        ) - self.sector2area(
            bin_edges[:-1], np.ptp(self.construct_retina.polar_lim_deg)
        )  # in mm2. Vector length len(edge_ecc) - 1.
        # Cells/area
        model_cell_density = hist / area_for_each_bin  # in cells/mm2
        ax[1].plot(center_ecc, model_cell_density, "b.")

    def visualize_mosaic(self):
        """
        Plots the full ganglion cell self.construct_retina. Note that this is slow if you have a large patch.

        ConstructRetina call.

        :return:
        """

        rho = self.construct_retina.gc_df["pos_ecc_mm"].to_numpy()
        phi = self.construct_retina.gc_df["pos_polar_deg"].to_numpy()

        gc_rf_models = np.zeros((len(self.construct_retina.gc_df), 6))
        gc_rf_models[:, 0] = self.construct_retina.gc_df["semi_xc"]
        gc_rf_models[:, 1] = self.construct_retina.gc_df["semi_yc"]
        gc_rf_models[:, 2] = self.construct_retina.gc_df["xy_aspect_ratio"]
        gc_rf_models[:, 3] = self.construct_retina.gc_df["ampl_s"]
        gc_rf_models[:, 4] = self.construct_retina.gc_df["relat_sur_diam"]
        gc_rf_models[:, 5] = self.construct_retina.gc_df["orient_cen"]

        # to cartesian
        xcoord, ycoord = self.pol2cart(rho, phi)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            xcoord.flatten(),
            ycoord.flatten(),
            "b.",
            label=self.construct_retina.gc_type,
        )

        if self.construct_retina.surround_fixed:
            # gc_rf_models parameters:'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'ampl_s','relat_sur_diam', 'orient_cen'
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

    def show_spatial_statistics(self):
        """
        Show histograms of receptive field parameters

        ConstructRetina call.
        """
        ydata = self.construct_retina.exp_spat_stat_to_viz["ydata"]
        spatial_statistics_dict = self.construct_retina.exp_spat_stat_to_viz[
            "spatial_statistics_dict"
        ]
        model_fit_data = self.construct_retina.exp_spat_stat_to_viz["model_fit_data"]

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

                # Rescale y axis if model fit goes high. Shows histogram better
                if y_model_fit[:, index].max() > 1.5 * bin_values.max():
                    ax.set_ylim([ax.get_ylim()[0], 1.1 * bin_values.max()])

        # Check correlations
        # distributions = ['semi_xc', 'semi_yc', 'xy_aspect_ratio', 'ampl_s','relat_sur_diam', 'orient_cen']
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

    def show_dendrite_diam_vs_ecc(self):
        """
        Plot dendritic diameter as a function of retinal eccentricity with linear, quadratic, or cubic fitting.
        """
        data_all_x = self.construct_retina.dd_vs_ecc_to_viz["data_all_x"]
        data_all_y = self.construct_retina.dd_vs_ecc_to_viz["data_all_y"]
        dd_fit_x = self.construct_retina.dd_vs_ecc_to_viz["dd_fit_x"]
        dd_fit_y = self.construct_retina.dd_vs_ecc_to_viz["dd_fit_y"]
        if self.construct_retina.model_type == "VAE":
            dd_vae_x = self.construct_retina.dd_vs_ecc_to_viz["dd_vae_x"]
            dd_vae_y = self.construct_retina.dd_vs_ecc_to_viz["dd_vae_y"]
        polynomials = self.construct_retina.dd_vs_ecc_to_viz["polynomials"]
        dd_model_caption = self.construct_retina.dd_vs_ecc_to_viz["dd_model_caption"]
        title = self.construct_retina.dd_vs_ecc_to_viz["title"]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(data_all_x, data_all_y, "b.", label="Data")
        ax.plot(dd_fit_x, dd_fit_y, "r.", label="Fit")
        if self.construct_retina.model_type == "VAE":
            ax.plot(dd_vae_x, dd_vae_y, "k.", label="Vae")

        ax.set_xlabel("Retinal eccentricity (mm)")
        ax.set_ylabel("Dendritic diameter (um)")
        ax.legend()

        if dd_model_caption:
            if (
                len(polynomials) == 2
            ):  # check if only two parameters, ie intercept and slope
                intercept = polynomials[1]
                slope = polynomials[0]
                ax.plot(data_all_x, intercept + slope * data_all_x, "k--")
                ax.annotate(
                    "{0} : \ny={1:.1f} + {2:.1f}x".format(
                        dd_model_caption, intercept, slope
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
                        dd_model_caption, intercept, slope, square
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
                        dd_model_caption, intercept, slope, square, cube
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

        plt.title(title)

    def show_temp_stat(self):
        """
        Show the temporal statistics of the retina units.
        """

        temporal_filter_parameters = self.construct_retina.exp_temp_stat_to_viz[
            "temporal_filter_parameters"
        ]
        distrib_params = self.construct_retina.exp_temp_stat_to_viz["distrib_params"]
        suptitle = self.construct_retina.exp_temp_stat_to_viz["suptitle"]
        all_data_fits_df = self.construct_retina.exp_temp_stat_to_viz[
            "all_data_fits_df"
        ]
        good_idx = self.construct_retina.exp_temp_stat_to_viz["good_idx"]

        plt.subplots(2, 3)
        plt.suptitle(suptitle)
        for i, param_name in enumerate(temporal_filter_parameters):
            plt.subplot(2, 3, i + 1)
            ax = plt.gca()
            shape, loc, scale = distrib_params[i, :]
            param_array = np.array(all_data_fits_df.iloc[good_idx][param_name])

            x_min, x_max = stats.gamma.ppf(
                [0.001, 0.999], a=shape, loc=loc, scale=scale
            )
            xs = np.linspace(x_min, x_max, 100)
            pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
            ax.plot(xs, pdf)
            ax.hist(param_array, density=True)
            ax.set_title(param_name)

    def show_tonic_drives(self):
        """
        ConstructRetina call.
        """
        xs = self.construct_retina.exp_tonic_dr_to_viz["xs"]
        pdf = self.construct_retina.exp_tonic_dr_to_viz["pdf"]
        tonicdrive_array = self.construct_retina.exp_tonic_dr_to_viz["tonicdrive_array"]
        title = self.construct_retina.exp_tonic_dr_to_viz["title"]

        plt.plot(xs, pdf)
        plt.hist(tonicdrive_array, density=True)
        plt.title(title)
        plt.xlabel("Tonic drive (a.u.)")

    def show_exp_build_process(self, show_all_spatial_fits=False):
        """
        Visualize the stages of retina mosaic building process.
        """

        # If show_all_spatial_fits is true, show the spatial fits
        if show_all_spatial_fits is True:
            spat_filt_to_viz = self.construct_retina.exp_spat_filt_to_viz
            self.show_spatial_filter_response(
                spat_filt_to_viz,
                n_samples=np.inf,
                title="Experimental",
                pause_to_show=True,
            )
            return

        self.show_temporal_filter_response()

        self.show_gc_positions_and_density()
        self.visualize_mosaic()
        self.show_spatial_statistics()
        self.show_dendrite_diam_vs_ecc()
        self.show_temp_stat()
        self.show_tonic_drives()

    def show_gen_exp_spatial_fit(self, n_samples=2):
        """
        Show the experimental (fitted) and generated spatial receptive fields (VAE)

        Parameters
        ----------
        n_samples : int
            Number of samples to show
        """
        spat_filt_to_viz = self.construct_retina.exp_spat_filt_to_viz
        self.show_spatial_filter_response(
            spat_filt_to_viz,
            n_samples=n_samples,
            title="Experimental",
            pause_to_show=False,
        )

        if self.construct_retina.model_type == "VAE":
            spat_filt_to_viz = self.construct_retina.gen_spat_filt_to_viz
            self.show_spatial_filter_response(
                spat_filt_to_viz,
                n_samples=n_samples,
                title="Generated",
                pause_to_show=False,
            )

    def show_latent_space_and_samples(self):
        """
        Plot the latent samples on top of the estimated kde, one sublot for each successive two dimensions of latent_dim
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        latent_samples = self.construct_retina.gen_latent_space_to_viz["samples"]
        latent_data = self.construct_retina.gen_latent_space_to_viz["data"]
        latent_dim = self.construct_retina.gen_latent_space_to_viz["dim"]

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
            # Both uniform and normal distr during learning is sampled
            # using gaussian kde estimate. The kde estimate is basically smooth histogram,
            # so it is not a problem that the data is not normal.
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
            # axes[ax_idx].scatter(latent_data[:, i], latent_data[:, i + 1])

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

    def show_gen_spat_post_hist(self):
        """
        Show the original experimental spatial receptive fields and
        the generated spatial receptive fields before and after postprocessing.
        """

        # Get the keys for the cell_ix arrays
        cell_key_list = [
            key
            for key in self.construct_retina.exp_spat_filt_to_viz.keys()
            if "cell_ix" in key
        ]
        img_shape = self.construct_retina.exp_spat_filt_to_viz["cell_ix_0"][
            "spatial_data_array"
        ].shape
        # The shape of the array is N cells, y_pixels, x_pixels
        img_exp = np.zeros([len(cell_key_list), img_shape[0], img_shape[1]])
        for i, cell_key in enumerate(cell_key_list):
            img_exp[i, :, :] = self.construct_retina.exp_spat_filt_to_viz[cell_key][
                "spatial_data_array"
            ]

        img_pre = self.construct_retina.gen_spat_img_to_viz["img_raw"]
        img_post = self.construct_retina.gen_spat_img_to_viz["img_processed"]

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

    def show_gen_exp_spatial_rf(self, ds_name="test_ds", n_samples=10):
        """
        Plot the outputs of the autoencoder.
        """
        assert (
            self.construct_retina.model_type == "VAE"
        ), "Only model type VAE is supported for show_gen_exp_spatial_rf()"
        if ds_name == "train_ds":
            ds = self.construct_retina.retina_vae.train_loader.dataset
        elif ds_name == "valid_ds":
            ds = self.construct_retina.retina_vae.val_loader.dataset
        else:
            ds = self.construct_retina.retina_vae.test_loader.dataset

        plt.figure(figsize=(16, 4.5))

        vae = self.construct_retina.retina_vae.vae
        vae.eval()
        len_ds = len(ds)
        samples = np.random.choice(len_ds, n_samples, replace=False)

        for pos_idx, sample_idx in enumerate(samples):
            ax = plt.subplot(2, len(samples), pos_idx + 1)
            img = (
                ds[sample_idx][0]
                .unsqueeze(0)
                .to(self.construct_retina.retina_vae.device)
            )
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.text(
                0.05,
                0.85,
                self.construct_retina.retina_vae.apricot_data.data_labels2names_dict[
                    ds[sample_idx][1].item()
                ],
                fontsize=10,
                color="red",
                transform=ax.transAxes,
            )
            if pos_idx == 0:
                ax.set_title("Original images")

            ax = plt.subplot(2, len(samples), len(samples) + pos_idx + 1)
            with torch.no_grad():
                rec_img = vae(img)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                ax.set_title("Reconstructed images")

        # Set the whole figure title as ds_name
        plt.suptitle(ds_name)

    def show_latent_tsne_space(self):
        train_df = self.construct_retina.retina_vae.get_encoded_samples(
            dataset=self.construct_retina.retina_vae.train_loader.dataset
        )
        valid_df = self.construct_retina.retina_vae.get_encoded_samples(
            dataset=self.construct_retina.retina_vae.val_loader.dataset
        )
        test_df = self.construct_retina.retina_vae.get_encoded_samples(
            dataset=self.construct_retina.retina_vae.test_loader.dataset
        )

        # Add a column to each df with the dataset name
        train_df["dataset"] = "train"
        valid_df["dataset"] = "valid"
        test_df["dataset"] = "test"

        # Concatenate the dfs
        encoded_samples = pd.concat([train_df, valid_df, test_df])

        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)

        if encoded_samples.shape[0] < tsne.perplexity:
            tsne.perplexity = encoded_samples.shape[0] - 1

        tsne_results = tsne.fit_transform(
            encoded_samples.drop(["label", "dataset"], axis=1)
        )

        ax0 = sns.relplot(
            # data=tsne_results,
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=encoded_samples.dataset.astype(str),
        )
        ax0.set(xlabel="tsne-2d-one", ylabel="tsne-2d-two")
        plt.title("TSNE plot of encoded samples")

    def show_ray_experiment(self, ray_exp, this_dep_var, highlight_trial=None):
        """
        Show the results of a ray experiment. If ray_exp is None, then
        the most recent experiment is shown.

        Parameters
        ----------
        ray_exp : str
            The name of the ray experiment
        this_dep_var : str
            The dependent variable to use for selecting the best trials
        highlight_trial : int
            The trial to highlight in the evolution plot
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

        self._show_tune_depvar_evolution(
            result_grid, dep_vars, highlight_trial=highlight_trial
        )

        nrows = 9
        ncols = len(dep_vars)
        nsamples = 10

        layout = [
            ["dh0", "dh1", "dh2", "dh3", "dh4", "dh5", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            # ["dv0", "dv1", "dv2", "dv3", "dv4", "dv5", ".", ".", ".", "."],
            ["im0", "im1", "im2", "im3", "im4", "im5", "im6", "im7", "im8", "im9"],
            ["re0" + str(i) for i in range(10)],
            ["re1" + str(i) for i in range(10)],
            ["re2" + str(i) for i in range(10)],
            ["re3" + str(i) for i in range(10)],
            ["re4" + str(i) for i in range(10)],
        ]
        fig, axd = plt.subplot_mosaic(layout, figsize=(ncols * 2, nrows))

        # Fraction of best = 1/4
        frac_best = 0.25
        num_best_trials = int(len(df) * frac_best)

        self._subplot_dependent_boxplots(axd, "dh", df, dep_vars, config_vars_changed)

        # self._subplot_dependent_variables(
        #     axd, "dv", result_grid, dep_vars, best_trials_across_dep_vars
        # )

        if hasattr(self.construct_retina, "exp_spat_filt_to_viz"):
            exp_spat_filt_to_viz = self.construct_retina.exp_spat_filt_to_viz
            print("construct_retina had attribute exp_spat_filt_to_viz")
        else:
            self.construct_retina._initialize()
            exp_spat_filt_to_viz = self.construct_retina.exp_spat_filt_to_viz
            print("construct_retina did not have attribute exp_spat_filt_to_viz")

        num_best_trials = 5  # Also N reco img to show
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
            # Position idx in layout: enumerate starts at 0, so add 1.
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

        if hasattr(model, "test_data"):
            test_data = model.test_data[:nsamples, :, :, :]
        else:
            # Make a list of dict keys starting "cell_ix_" from exp_spat_filt_to_viz dictionary
            keys = exp_spat_filt_to_viz.keys()
            cell_ix_names_list = [key for key in keys if "cell_ix_" in key]
            # Make a numpy array of numbers following "cell_ix_"
            cell_ix_array = np.array(
                [int(key.split("cell_ix_")[1]) for key in cell_ix_names_list]
            )

            # Take first nsamples from cell_ix_array. They have to be constant,
            # because we are showing multiple sets of images on top of each other.
            samples = cell_ix_array[:nsamples]

            test_data = np.zeros(
                [
                    nsamples,
                    1,
                    exp_spat_filt_to_viz["num_pix_y"],
                    exp_spat_filt_to_viz["num_pix_x"],
                ]
            )
            for idx, this_sample in enumerate(samples):
                test_data[idx, 0, :, :] = exp_spat_filt_to_viz[
                    f"cell_ix_{this_sample}"
                ]["spatial_data_array"]

        # Hack to reuse the AugmentedDataset._feature_scaling method. Scales to [0,1]
        test_data = AugmentedDataset._feature_scaling("", test_data)

        test_data = torch.from_numpy(test_data).float()
        img_size = model.decoder.unflatten.unflattened_size
        test_data = TF.resize(test_data, img_size[-2:], antialias=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = range(0, nsamples)

        model.eval()
        model.to(self.device)

        img = test_data.to(self.device)

        with torch.no_grad():
            rec_img = model(img)

        img = img.cpu().squeeze().numpy()
        rec_img = rec_img.cpu().squeeze().numpy()

        return img, rec_img, samples

    def _subplot_dependent_boxplots(self, axd, kw, df, dep_vars, config_vars_changed):
        """Boxplot dependent variables for one ray tune experiment"""

        # config_vars_changed list contain the varied columns in dataframe df
        # From config_vars_changed, 'config/model_id' contain the replications of the same model
        # Other config_vars_changed contain the models of interest
        # Make an seaborn boxplot for each model of interest
        config_vars_changed.remove("config/model_id")

        # If there are more than one config_vars_changed,
        # make a new dataframe column with the values of the config_vars_changed as strings
        if len(config_vars_changed) > 1:
            df["config_vars"] = (
                df[config_vars_changed].astype(str).agg(",".join, axis=1)
            )
            # config_vars_changed = ["config_vars"]
            config_vars_for_label = [
                col.removeprefix("config/") for col in config_vars_changed
            ]
            # Combine the string listed in config_vars_changed to one string
            config_vars_label = ",".join(config_vars_for_label)

        else:
            df["config_vars"] = df[config_vars_changed[0]]
            config_vars_label = [
                col.removeprefix("config/") for col in config_vars_changed
            ][0]

        # Make one subplot for each dependent variable
        # Plot labels only after the last subplot
        for idx, dep_var in enumerate(dep_vars):
            ax = axd[f"{kw}{idx}"]

            # Create the boxplot with seaborn
            ax_sns = sns.boxplot(
                x="config_vars", y=dep_var, data=df, ax=ax, whis=[0, 100]
            )
            # If any of the df["config_vars"] has length > 4, make x-label rotated 90 degrees
            if any(df["config_vars"].astype(str).str.len() > 4):
                ax_sns.set_xticklabels(ax.get_xticklabels(), rotation=90)

            # Set the title of the subplot to be the dependent variable name
            ax.set_title(dep_var)

            # Set y-axis label
            ax.set_ylabel("")

            # Set x-axis label
            if idx == 0:  # only the first subplot gets an x-axis label
                ax.set_xlabel(config_vars_label)
            else:
                ax.set_xlabel("")

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
                    label = f"{dep_vars[color_idx]}: " + ",".join(
                        f"{x}={result.config[x]}" for x in varied_cols
                    )
                    legend = True
                    first_ax = ax

                else:
                    label = None
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

            if idx == 0:
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

        first_ax.set_title(
            f"Evolution for best trials (ad {total_n_epochs} epochs)\nDot and vertical line indicate mean and SD of last 50 epochs",
            loc="left",
        )

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

    def _show_tune_depvar_evolution(self, result_grid, dep_vars, highlight_trial=None):
        """Plot results from ray tune"""

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

        num_colors = len(result_grid.get_dataframe())
        if highlight_trial is None:
            colors = plt.cm.get_cmap("tab20", num_colors).colors
            highlight_idx = None
        else:
            [highlight_idx] = [
                idx
                for idx, r in enumerate(result_grid)
                if highlight_trial in r.metrics["trial_id"]
            ]
            # set all other colors low contrast gray, and the highlight color to red
            colors = np.array(
                ["gray" if idx != highlight_idx else "red" for idx in range(num_colors)]
            )

        # Make one subplot for each dependent variable
        nrows = 2
        ncols = len(dep_vars) // 2
        plt.figure(figsize=(ncols * 5, nrows * 5))

        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = plt.subplot(nrows, ncols, idx + 1)
            label = None

            for result in result_grid:
                # Too cluttered for a legend
                # if idx == 0 and highlight_idx is None:
                #     label = ",".join(f"{x}={result.config[x]}" for x in varied_cols)
                #     legend = True
                # else:
                #     legend = False

                ax_plot = result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=False,
                )

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
            ax.set_title(f"{dep_var}")
            ax.set_ylabel(dep_var)
            ax.grid(True)

    # WorkingRetina visualization
    def show_stimulus_with_gcs(
        self, retina, frame_number=0, ax=None, example_gc=5, show_rf_id=False
    ):
        """
        Plots the 1SD ellipses of the RGC mosaic. This method is a WorkingRetina call.

        Parameters
        ----------
        retina : object
            The retina object that contains all the relevant information about the stimulus video and ganglion cells.
        frame_number : int, optional
            The index of the frame from the stimulus video to be displayed. Default is 0.
        ax : matplotlib.axes.Axes, optional
            The axes object to draw the plot on. If None, the current axes is used. Default is None.
        example_gc : int, optional
            The index of the ganglion cell to be highlighted. Default is 5.
        show_rf_id : bool, optional
            If True, the index of each ganglion cell will be printed at the center of its ellipse. Default is False..
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
            if index == example_gc:
                facecolor = "yellow"
            else:
                facecolor = "None"

            circ = Ellipse(
                (gc.q_pix, gc.r_pix),
                width=2 * gc.semi_xc,
                height=2 * gc.semi_yc,
                angle=gc.orient_cen * (-1),
                edgecolor="blue",
                facecolor=facecolor,
            )
            ax.add_patch(circ)

            # If show_rf_id is True, annotate each ellipse with the index
            if show_rf_id:
                ax.annotate(
                    str(index),
                    (gc.q_pix, gc.r_pix),
                    color="black",
                    weight="bold",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

        locs, labels = plt.yticks()

        locs = locs[locs < stimulus_height_pix]
        locs = locs[locs > 0]

        left_y_labels = locs.astype(int)
        plt.yticks(ticks=locs)
        ax.set_ylabel("pix")

        xlocs = locs - np.mean(locs)
        down_x_labels = np.round(xlocs / pix_per_deg, decimals=2) + np.real(
            stimulus_center
        )
        plt.xticks(ticks=locs, labels=down_x_labels)
        ax.set_xlabel("deg")

        ax2 = ax.twinx()
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
            angle=gc.orient_cen * (-1),
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
        ampl_s = np.abs(ft_tf)

        ax.set_xscale("log")
        ax.set_xlim([0.1, 100])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        ax.plot(freqs, ampl_s, ".")

    def plot_midpoint_contrast(self, retina, cell_index, ax=None):
        """
        Plots the contrast in the mid-pixel of the stimulus cropped to RGC surroundings

        WorkingRetina call.

        :param cell_index:
        :return:
        """
        stimulus_cropped = retina._get_spatially_cropped_video(cell_index)
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
        stimulus_cropped = retina._get_spatially_cropped_video(
            cell_index, contrast=False
        )
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
        stimulus_cropped = retina._get_spatially_cropped_video(
            cell_index, contrast=False
        )
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
            for_generatorplot = np.mean(exp_generator_potential, axis=0)
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
        tvec = np.arange(0, generator_potential.shape[-1], 1) * video_dt

        plt.plot(tvec, for_generatorplot, label="Generator")
        plt.xlim([0, duration / b2u.second])

        # Compute average firing rate over trials (should approximately follow generator)
        # hist_dt = 1 * b2u.ms
        hist_dt = self.context.my_run_options["simulation_dt"] * b2u.second
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

    def show_parasol_impulse_response(self, retina, savefigname=None):
        viz_dict = retina.impulse_for_viz_dict
        tvec = viz_dict["tvec"]
        svec = viz_dict["svec"]

        # Get keys which are not "tvec" or "svec"
        contrasts = [
            key for key in viz_dict.keys() if key not in ["tvec", "svec", "Unit idx"]
        ]

        for this_contrast in contrasts:
            plt.plot(tvec[:-1], viz_dict[this_contrast][:-1])

        # Set vertical dashed line at max (svec) time point, i.e. at the impulse time
        plt.axvline(x=tvec[np.argmax(svec)], color="k", linestyle="--")
        plt.legend(contrasts)
        plt.ylim(-0.2, 1.1)
        plt.title(
            f"Parasol impulse response for unit {str(viz_dict['Unit idx'])} at multiple contrasts"
        )

        if savefigname is not None:
            self._figsave(figurename=savefigname)

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

    def show_retina_img(self):
        """
        Show the image of whole retina with all the receptive fields summed up.
        """

        img_ret = self.construct_retina.gen_ret_to_viz["img_ret"]
        img_ret_masked = self.construct_retina.gen_ret_to_viz["img_ret_masked"]
        img_ret_adjusted = self.construct_retina.gen_ret_to_viz["img_ret_adjusted"]

        plt.figure()
        plt.subplot(221)
        plt.imshow(img_ret, cmap="gray")
        plt.colorbar()
        plt.title("Original coverage")
        plt.subplot(222)

        cmap = plt.cm.get_cmap("viridis")
        custom_cmap = mcolors.ListedColormap(
            cmap(np.linspace(0, 1, int(np.max(img_ret_masked)) + 1))
        )
        plt.imshow(img_ret_masked, cmap=custom_cmap)
        plt.colorbar()
        plt.title("Summed masks")

        plt.subplot(223)
        plt.imshow(img_ret_adjusted, cmap="gray")
        plt.colorbar()
        plt.title("Adjusted coverage")

    def show_rf_imgs(self, n_samples=10):
        """
        Show the individual RFs of the VAE retina

        img_rf: (n_cells, n_pix, n_pix)
        img_rf_mask: (n_cells, n_pix, n_pix)
        img_rfs_adjusted: (n_cells, n_pix, n_pix)
        """

        img_rf = self.construct_retina.gen_rfs_to_viz["img_rf"]
        img_rf_mask = self.construct_retina.gen_rfs_to_viz["img_rf_mask"]
        img_rfs_adjusted = self.construct_retina.gen_rfs_to_viz["img_rfs_adjusted"]

        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples, 3))
        samples = np.random.choice(img_rf.shape[0], n_samples, replace=False)
        for i, sample in enumerate(samples):
            axs[0, i].imshow(img_rf[sample], cmap="gray")
            axs[0, i].axis("off")
            axs[0, i].set_title("Cell " + str(sample))

            axs[1, i].imshow(img_rf_mask[sample], cmap="gray")
            axs[1, i].axis("off")

            axs[2, i].imshow(img_rfs_adjusted[sample], cmap="gray")
            axs[2, i].axis("off")

        # On the left side of the first axis of each row, set text labels.
        axs[0, 0].set_ylabel("RF")
        axs[0, 0].axis("on")
        axs[1, 0].set_ylabel("Mask")
        axs[1, 0].axis("on")
        axs[2, 0].set_ylabel("Adjusted RF")
        axs[2, 0].axis("on")

        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])

        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])

        # # Adjust the layout so labels are visible
        # fig.subplots_adjust(left=0.15)

    def show_rf_violinplot(self):
        """
        Show the individual RFs of the VAE retina

        img_rf: (n_cells, n_pix, n_pix)
        img_rfs_adjusted: (n_cells, n_pix, n_pix)
        """

        img_rf = self.construct_retina.gen_rfs_to_viz["img_rf"]
        img_rfs_adjusted = self.construct_retina.gen_rfs_to_viz["img_rfs_adjusted"]

        fig, axs = plt.subplots(
            2, 1, figsize=(10, 10)
        )  # I assume you want a bigger figure size.

        # reshape and transpose arrays so that we have one row per cell
        df_rf = pd.DataFrame(img_rf.reshape(img_rf.shape[0], -1).T)
        df_pruned = pd.DataFrame(
            img_rfs_adjusted.reshape(img_rfs_adjusted.shape[0], -1).T
        )

        # Show seaborn boxplot with RF values, one box for each cell
        sns.violinplot(data=df_rf, ax=axs[0])
        axs[0].set_title("RF values")
        # Put grid on
        axs[0].grid(True)
        # ...and RF_adjusted values
        sns.violinplot(data=df_pruned, ax=axs[1])
        axs[1].set_title("RF adjusted values")
        axs[1].grid(True)

    # Results visualization

    def _string_on_plot(
        self, ax, variable_name=None, variable_value=None, variable_unit=None
    ):
        plot_str = f"{variable_name} = {variable_value:6.2f} {variable_unit}"
        ax.text(
            0.05,
            0.95,
            plot_str,
            fontsize=8,
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Square,pad=0.2", fc="white", ec="white", lw=1),
        )

    def fr_response(self, exp_variables, xlog=False, ylog=False):
        """
        Plot the mean firing rate response curve.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        assert (
            len(exp_variables) == 1
        ), "Only one variable can be plotted at a time, aborting..."

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )
        data_df = pd.read_csv(
            data_folder / f"{cond_names_string}_population_means.csv", index_col=0
        )
        data_df_units = pd.read_csv(
            data_folder / f"{cond_names_string}_unit_means.csv", index_col=0
        )

        response_levels_s = experiment_df.loc["contrast", :]
        mean_response_levels_s = data_df.mean()
        response_levels_s = pd.to_numeric(response_levels_s)
        response_levels_s = response_levels_s.round(decimals=2)

        response_function_df = pd.DataFrame(
            {cond_names_string: response_levels_s, "response": mean_response_levels_s}
        )

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        sns.lineplot(
            data=response_function_df,
            x=cond_names_string,
            y="response",
            marker="o",
            color="black",
            ax=ax[0],
        )

        # Title
        ax[0].set_title(f"{cond_names_string} response function (population mean)")

        if xlog:
            ax[0].set_xscale("log")
        if ylog:
            ax[0].set_yscale("log")

        sns.boxplot(data=data_df_units, color="white", linewidth=2, whis=100, ax=ax[1])
        sns.swarmplot(data=data_df_units, color="black", size=3, ax=ax[1])

        # Title
        ax[1].set_title(f"{cond_names_string} response function (individual units)")

    def F1F2_popul_response(self, exp_variables, xlog=False, ylog=False):
        """
        Plot oF1 and  F2 frequency response curves for all conditions.
        Population response, i.e. mean across units.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        n_variables = len(exp_variables)

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )
        F_popul_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_population_means.csv", index_col=0
        )

        F_popul_long_df = pd.melt(
            F_popul_df,
            id_vars=["trial", "F_peak"],
            value_vars=F_popul_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_popul_long_df[cond] = F_popul_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        fig, ax = plt.subplots(1, n_variables, figsize=(8, 4))

        if n_variables == 1:
            sns.lineplot(
                data=F_popul_long_df,
                x=exp_variables[0],
                y="amplitudes",
                hue="F_peak",
                palette="tab10",
                ax=ax,
            )
            ax.set_title("Population amplitude spectra for " + exp_variables[0])
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

        else:
            for i, cond in enumerate(exp_variables):
                sns.lineplot(
                    data=F_popul_long_df,
                    x=cond,
                    y="amplitudes",
                    hue="F_peak",
                    palette="tab10",
                    ax=ax[i],
                )

                # Title
                ax[i].set_title("Population amplitude spectra for " + cond)
                if xlog:
                    ax[i].set_xscale("log")
                if ylog:
                    ax[i].set_yscale("log")

    def F1F2_unit_response(self, exp_variables, xlog=False, ylog=False):
        """
        Plot F1 and  F2 frequency response curves for all conditions.
        Unit response, i.e. mean across trials.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        n_variables = len(exp_variables)

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )

        F_unit_ampl_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_unit_ampl_means.csv", index_col=0
        )
        F_unit_long_df = pd.melt(
            F_unit_ampl_df,
            id_vars=["unit", "F_peak"],
            value_vars=F_unit_ampl_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_unit_long_df[cond] = F_unit_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        fig, ax = plt.subplots(1, n_variables, figsize=(8, 4))

        if n_variables == 1:
            sns.lineplot(
                data=F_unit_long_df,
                x=exp_variables[0],
                y="amplitudes",
                hue="F_peak",
                palette="tab10",
                ax=ax,
            )
            ax.set_title("Unit amplitude spectra for " + exp_variables[0])
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

        else:
            for i, cond in enumerate(exp_variables):
                sns.lineplot(
                    data=F_unit_long_df,
                    x=cond,
                    y="amplitudes",
                    hue="F_peak",
                    palette="tab10",
                    ax=ax[i],
                )

                # Title
                ax[i].set_title("Unit amplitude spectra for " + cond)
                if xlog:
                    ax[i].set_xscale("log")
                if ylog:
                    ax[i].set_yscale("log")

    def spike_raster_response(self, exp_variables, trial=0, savefigname=None):
        """
        Show spikes from a results file.

        Parameters
        ----------
        results_filename : str or None
            This name is searched from current directory, input_folder and output_folder (defined in config file). If None, the latest results file is used.
        savefigname : str or None
            If not empty, the figure is saved to this filename.

        Returns
        -------
        None
        """
        cond_names_string = "_".join(exp_variables)
        filename = f"exp_metadata_{cond_names_string}.csv"
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.columns.values
        n_trials_vec = pd.to_numeric(experiment_df.loc["n_trials", :].values)
        assert trial < np.min(n_trials_vec), "Trial id too high, aborting..."

        # Visualize

        fig, ax = plt.subplots(len(experiment_df.columns), 1, figsize=(8, 4))

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = f"Response_{cond_name}.gz"
            data_dict = self.data_io.get_data(filename)

            cond_s = experiment_df[cond_name]
            duration_seconds = pd.to_numeric(cond_s.loc["duration_seconds"])
            baseline_start_seconds = pd.to_numeric(cond_s.loc["baseline_start_seconds"])
            baseline_end_seconds = pd.to_numeric(cond_s.loc["baseline_end_seconds"])
            duration_tot = (
                duration_seconds + baseline_start_seconds + baseline_end_seconds
            )

            units, times = self.ana._get_spikes_by_interval(
                data_dict, trial, 0, duration_tot
            )

            ax[idx].plot(
                times,
                units,
                ".",
            )

            this_contrast = pd.to_numeric(experiment_df.loc["contrast", cond_name])
            ax[idx].set_title(
                f"{cond_name}, contrast {this_contrast:.2f}",
                fontsize=10,
            )

            MeanFR = self.ana._analyze_meanfr(data_dict, trial, 0, duration_tot)
            self._string_on_plot(
                ax[idx],
                variable_name="Mean FR",
                variable_value=MeanFR,
                variable_unit="Hz",
            )

        if savefigname:
            self._figsave(figurename=savefigname)

    def tf_vs_fr_cg(self, exp_variables, n_contrasts=None, xlog=False, ylog=False):
        """
        Plot F1 frequency response curves for 2D frequency-contrast experiment.
        Unit response, i.e. mean across trials.
        Subplot 1: temporal frequency vs firing rate at n_contrasts
        Subplot 2: temporal frequency vs contrast gain (cg) at n_contrasts. Contrast gain is defined as the F1 response divided by contrast.

        Parameters
        ----------
        exp_variables : list of str
            List of experiment variables to be plotted.
        n_contrasts : int
            Number of contrasts to be plotted. If None, all contrasts are plotted.
        xlog : bool
            If True, x-axis is logarithmic.
        ylog : bool
            If True, y-axis is logarithmic.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)

        # Experiment metadata
        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )

        # Results
        F_unit_ampl_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_unit_ampl_means.csv", index_col=0
        )

        F_unit_long_df = pd.melt(
            F_unit_ampl_df,
            id_vars=["unit", "F_peak"],
            value_vars=F_unit_ampl_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_unit_long_df[cond] = F_unit_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        # Make new columns cg and phase.
        F_unit_long_df["cg"] = F_unit_long_df["amplitudes"] / F_unit_long_df["contrast"]

        F_unit_long_df = F_unit_long_df[F_unit_long_df["F_peak"] == "F1"].reset_index(
            drop=True
        )

        # Select only the desired number of contrasts at about even intervals, including the lowest and the highest contrast
        if n_contrasts:
            contrasts = F_unit_long_df["contrast"].unique()
            contrasts.sort()
            contrasts = contrasts[:: int(len(contrasts) / n_contrasts)]
            contrasts = np.append(contrasts, F_unit_long_df["contrast"].max())
            F_unit_long_df = F_unit_long_df.loc[
                F_unit_long_df["contrast"].isin(contrasts)
            ]

        fig, ax = plt.subplots(2, 1, figsize=(8, 12))

        # Make the three subplots using seaborn lineplot
        sns.lineplot(
            data=F_unit_long_df,
            x="temporal_frequency",
            y="amplitudes",
            hue="contrast",
            palette="tab10",
            ax=ax[0],
        )
        ax[0].set_title("Firing rate vs temporal frequency")
        if xlog:
            ax[0].set_xscale("log")
        if ylog:
            ax[0].set_yscale("log")

        sns.lineplot(
            data=F_unit_long_df,
            x="temporal_frequency",
            y="cg",
            hue="contrast",
            palette="tab10",
            ax=ax[1],
        )
        ax[1].set_title("Contrast gain vs temporal frequency")
        if xlog:
            ax[1].set_xscale("log")
        if ylog:
            ax[1].set_yscale("log")
