# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd

# import cv2

# Viz
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse
from tqdm import tqdm

# Builtin
import os
import sys
import pdb


class Viz:
    """
    Methods to viz_module the retina
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

    def show_gc_positions_and_density(self, rho, phi, gc_density_func_params):
        """
        Show retina cell positions and receptive fields
        """

        # to cartesian
        xcoord, ycoord = self.pol2cart(rho, phi)
        # xcoord,ycoord = FitFunctionsAndMath.pol2cart(matrix_eccentricity, matrix_orientation_surround)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(xcoord.flatten(), ycoord.flatten(), "b.", label=self.gc_type)
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
        my_gaussian_fit_current_GC = my_gaussian_fit * self.gc_proportion
        ax[1].plot(edge_ecc, my_gaussian_fit_current_GC, "r")

        # Density of model cells
        index = np.all(
            [
                phi > np.min(self.theta),
                phi < np.max(self.theta),
                rho > np.min(self.eccentricity_in_mm),
                rho < np.max(self.eccentricity_in_mm),
            ],
            axis=0,
        )  # Index only cells within original requested theta
        hist, bin_edges = np.histogram(rho[index], nbins)
        center_ecc = bin_edges[:-1] + ((bin_edges[1:] - bin_edges[:-1]) / 2)
        area_for_each_bin = self.sector2area(
            bin_edges[1:], np.ptp(self.theta)
        ) - self.sector2area(
            bin_edges[:-1], np.ptp(self.theta)
        )  # in mm2. Vector length len(edge_ecc) - 1.
        # Cells/area
        model_cell_density = hist / area_for_each_bin  # in cells/mm2
        ax[1].plot(center_ecc, model_cell_density, "b.")

    def show_gc_receptive_fields(self, rho, phi, gc_rf_models, surround_fixed=0):
        """
        Show retina cell positions and receptive fields. Note that this is slow if you have a large patch.
        """

        # to cartesian
        xcoord, ycoord = self.pol2cart(rho, phi)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(xcoord.flatten(), ycoord.flatten(), "b.", label=self.gc_type)

        if self.surround_fixed:
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
                e1 = ellipse(
                    (ellipse_center_x, ellipse_center_y),
                    diameter_xc,
                    diameter_yc,
                    angle_in_deg,
                    edgecolor="b",
                    linewidth=0.5,
                    fill=False,
                )
                ax.add_artist(e1)
        # e2=ellipse((popt[np.array([1,2])]),popt[7]*popt[3],popt[7]*popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')

        ax.axis("equal")
        ax.legend()
        ax.set_title("Cartesian retina")
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel("Elevation (mm)")

    def show_spatial_statistics(
        self, ydata, spatial_statistics_dict, model_fit_data=None
    ):
        """
        Show histograms of receptive field parameters
        """

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
                        s="shape = {0:.2f}\nloc = {1:.2f}\nscale = {2:.2f}".format(
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
                        s="a = {0:.2f}\nb = {1:.2f}\nloc = {2:.2f}\nscale = {3:.2f}".format(
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
                s="\nr={0:.2g},\np={1:.2g}".format(r, p),
                xy=(0.8, 0.4),
                xycoords="axes fraction",
            )
            ax2.set_title(
                "Correlation between {0} and {1}".format(
                    distributions[ref_index], distributions[index]
                )
            )

        # Save correlation figure
        # folder_name = 'Korrelaatiot\Midget_OFF'
        # save4save = os.getcwd()
        # os.chdir(folder_name)
        # plt.savefig('Corr_{0}.png'.format(distributions[ref_index]),format='png')
        # os.chdir(save4save)

        pass

    def show_dendritic_diameter_vs_eccentricity(
        self, gc_type, dataset_x, dataset_y, polynomials, dataset_name=""
    ):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(dataset_x, dataset_y, ".")

        if dataset_name:
            if (
                len(polynomials) == 2
            ):  # check if only two parameters, ie intercept and slope
                intercept = polynomials[1]
                slope = polynomials[0]
                ax.plot(dataset_x, intercept + slope * dataset_x, "k--")
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
                    dataset_x,
                    intercept + slope * dataset_x + square * dataset_x**2,
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
                    dataset_x,
                    intercept
                    + slope * dataset_x
                    + square * dataset_x**2
                    + cube * dataset_x**3,
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

        plt.title(
            "DF diam wrt ecc for {0} type, {1} dataset".format(gc_type, dataset_name)
        )

    def show_cone_response(self, image, image_after_optics, cone_response):

        fig, ax = plt.subplots(nrows=2, ncols=3)
        axs = ax.ravel()
        axs[0].hist(image.flatten(), 20)
        axs[1].hist(image_after_optics.flatten(), 20)
        axs[2].hist(cone_response.flatten(), 20)

        axs[3].imshow(image, cmap="Greys")
        axs[4].imshow(image_after_optics, cmap="Greys")
        axs[5].imshow(cone_response, cmap="Greys")
