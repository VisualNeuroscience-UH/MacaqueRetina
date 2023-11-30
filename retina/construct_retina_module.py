# Numerical
import numpy as np
import numpy.ma as ma
import matplotlib.path as mplPath


import scipy.stats as stats
import scipy.optimize as opt
from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import griddata
import pandas as pd
import torch
import torch.nn.functional as F

# import torch.autograd.profiler as profiler

# from torch.utils.data import DataLoader

# from scipy.signal import convolve
# from scipy.interpolate import interp1d

# Image analysis
# from skimage import measure
# from skimage.transform import resize
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon

# Data IO
# import cv2
# from PIL import Image

# Viz
# from tqdm import tqdm

# Comput Neurosci
# import brian2 as b2
# import brian2.units as b2u

# Local
from retina.retina_math_module import RetinaMath
from retina.vae_module import RetinaVAE

# Builtin
from pathlib import Path
import pdb
from copy import deepcopy
import math
import sys


class ConstructRetina(RetinaMath):
    """
    Create the ganglion cell mosaic.
    All spatial parameters are saved to the dataframe *gc_df*

    Attributes
    ----------
    gc_type : str
        Type of ganglion cell, either "parasol" or "midget"
    response_type : str
        Type of response, either "on" or "off"
    eccentricity : list
        List of two floats, the eccentricity limits in degrees
    ecc_lim_mm : list
        List of two floats, the eccentricity limits in mm
    polar_lim_deg : list
        Numpy array two floats, the sector limits in degrees
    deg_per_mm : float
        Degrees per mm
    spatial_model : str
        Type of model, either "FIT"or  "VAE"
    gc_proportion : float
        Proportion of ganglion cells to be created
    gc_df : pd.DataFrame
        Dataframe containing the ganglion cell mosaic
    """

    _properties_list = []

    def __init__(self, context, data_io, viz, fit, project_data) -> None:
        # Dependency injection at ProjectManager construction
        self._context = context.set_context(self)
        self._data_io = data_io
        self._viz = viz
        self._fit = fit
        self._project_data = project_data

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

    @property
    def fit(self):
        return self._fit

    @property
    def project_data(self):
        return self._project_data

    def _initialize(self):
        """
        Initialize the ganglion cell mosaic.
            First: sets ConstructRetina instance parameters from conf file my_retina
            Second: creates empty gc_df to hold the final ganglion cell mosaics
            Third: gets gc creation model according to spatial_model
                Calls Fit or RetinaVAE classes

        See class attributes for more details.
        """

        my_retina = self.context.my_retina
        gc_type = my_retina["gc_type"]
        response_type = my_retina["response_type"]
        ecc_limits_deg = my_retina["ecc_limits_deg"]
        visual_field_limit_for_dd_fit = my_retina["visual_field_limit_for_dd_fit"]
        pol_limits_deg = my_retina["pol_limits_deg"]
        model_density = my_retina["model_density"]
        self.rf_coverage_adjusted_to_1 = my_retina["rf_coverage_adjusted_to_1"]
        self.dd_regr_model = my_retina["dd_regr_model"]
        self.deg_per_mm = my_retina["deg_per_mm"]

        self.gc_type = gc_type
        self.response_type = response_type

        self.spatial_model = my_retina["spatial_model"]
        if self.spatial_model in ["VAE"]:
            self.training_mode = my_retina["training_mode"]

        proportion_of_parasol_gc_type = my_retina["proportion_of_parasol_gc_type"]
        proportion_of_midget_gc_type = my_retina["proportion_of_midget_gc_type"]
        proportion_of_ON_response_type = my_retina["proportion_of_ON_response_type"]
        proportion_of_OFF_response_type = my_retina["proportion_of_OFF_response_type"]

        # Assertions
        assert (
            isinstance(ecc_limits_deg, list) and len(ecc_limits_deg) == 2
        ), "Wrong type or length of eccentricity, aborting"
        assert (
            isinstance(pol_limits_deg, list) and len(pol_limits_deg) == 2
        ), "Wrong type or length of pol_limits_deg, aborting"
        assert model_density <= 1.0, "Density should be <=1.0, aborting"

        # Calculate self.gc_proportion from GC type specifications
        gc_type = gc_type.lower()
        response_type = response_type.lower()
        if all([gc_type == "parasol", response_type == "on"]):
            self.gc_proportion = (
                proportion_of_parasol_gc_type
                * proportion_of_ON_response_type
                * model_density
            )
        elif all([gc_type == "parasol", response_type == "off"]):
            self.gc_proportion = (
                proportion_of_parasol_gc_type
                * proportion_of_OFF_response_type
                * model_density
            )
        elif all([gc_type == "midget", response_type == "on"]):
            self.gc_proportion = (
                proportion_of_midget_gc_type
                * proportion_of_ON_response_type
                * model_density
            )
        elif all([gc_type == "midget", response_type == "off"]):
            self.gc_proportion = (
                proportion_of_midget_gc_type
                * proportion_of_OFF_response_type
                * model_density
            )
        else:
            raise ValueError("Unknown ganglion cell type, aborting")

        self.gc_type = gc_type
        self.response_type = response_type

        self.eccentricity = ecc_limits_deg
        self.ecc_lim_mm = np.asarray(
            [r / self.deg_per_mm for r in ecc_limits_deg]
        )  # Turn list to numpy array and deg to mm
        self.visual_field_limit_for_dd_fit_mm = (
            visual_field_limit_for_dd_fit / self.deg_per_mm
        )
        self.polar_lim_deg = np.asarray(pol_limits_deg)  # Turn list to numpy array

        # Make or read fits
        if self.spatial_model == "VAE":
            # VAE RF scaling with eccentricity is dependent on DoG fit (dendritic diameter)
            # comparison btw literature and experimental data fit. We do not want the data fit
            # to vary with the DoG model. Thus, we use the same DoG model for all for the initial
            # experimental fit. The VAE generated RFs will be fitted downstream.
            DoG_model = "ellipse_fixed"
        elif self.spatial_model == "FIT":
            DoG_model = my_retina["DoG_model"]
        self.fit.initialize(
            gc_type, response_type, fit_type="experimental", DoG_model=DoG_model
        )
        (
            self.exp_stat_df,
            self.exp_cen_radius_mm,
            self.exp_sur_radius_mm,
            self.spat_DoG_fit_params,
        ) = self.fit.get_experimental_fits(DoG_model)

        self.gc_df = pd.DataFrame()

        self.device = self.context.device

        self.initialized = True

    def _get_random_samples(self, shape, loc, scale, n_cells, distribution):
        """
        Create random samples from a model distribution.

        Parameters
        ----------
        shape : float or array_like of floats
            The shape parameters of the distribution.
        loc : float or array_like of floats
            The location parameters of the distribution.
        scale : float or array_like of floats
            The scale parameters of the distribution.
        n_cells : int
            The number of cells to generate samples for.
        distribution : str
            The distribution to sample from. Supported distributions: "gamma", "vonmises", "skewnorm", "triang".

        Returns
        -------
        distribution_parameters : ndarray
            The generated random samples from the specified distribution.

        Raises
        ------
        ValueError
            If the specified distribution is not supported.
        """
        assert distribution in [
            "gamma",
            "vonmises",
            "skewnorm",
            "triang",
        ], "Distribution not supported, aborting..."

        # Check if any of the shape, loc, scale parameters are np.nan
        # If so, set distribution_parameters to np.nan and return
        if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
            distribution_parameters = np.nan * np.ones(n_cells)
            return distribution_parameters

        # Waiting times between events are relevant for gamma distribution
        if distribution == "gamma":
            distribution_parameters = stats.gamma.rvs(
                a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )  # random_state is the seed
        # Continuous probability distribution on the circle
        elif distribution == "vonmises":
            distribution_parameters = stats.vonmises.rvs(
                kappa=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )
        # Skewed normal distribution
        elif distribution == "skewnorm":
            distribution_parameters = stats.skewnorm.rvs(
                a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )
        # Triangular distribution when min max mean median and sd is available in literature
        elif distribution == "triang":
            distribution_parameters = stats.triang.rvs(
                c=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )

        return distribution_parameters

    def _read_gc_density_data(self):
        """
        Read re-digitized old literature ganglion cell density data
        """

        gc_density = self.data_io.get_data(
            self.context.literature_data_files["gc_density_fullpath"]
        )
        cell_eccentricity = np.squeeze(gc_density["Xdata"])
        # Cells are in thousands, thus the 1e3
        cell_density = np.squeeze(gc_density["Ydata"]) * 1e3
        return cell_eccentricity, cell_density

    def _fit_gc_density_data(self):
        """
        Fits a Gaussian to ganglion cell density (digitized data from Perry_1984).

        :returns a, x0, sigma, baseline (aka "gc_density_func_params")
        """

        cell_eccentricity, cell_density = self._read_gc_density_data()
        # Gaussian + baseline fit initial values for fitting
        scale, mean, sigma, baseline0 = 1000, 0, 2, np.min(cell_density)
        popt, pcov = opt.curve_fit(
            self.gauss_plus_baseline,
            cell_eccentricity,
            cell_density,
            p0=[scale, mean, sigma, baseline0],
        )

        return popt  # = gc_density_func_params

    def _fit_dd_vs_ecc(self):
        """
        Fit dendritic field diameter with respect to eccentricity. Linear, quadratic and cubic fit.

        Returns
        -------
        dict
            dictionary containing dendritic diameter parameters and related data for visualization
        """

        dd_regr_model = self.dd_regr_model
        visual_field_limit_for_dd_fit_mm = self.visual_field_limit_for_dd_fit_mm

        # Read dendritic field data and return linear fit with scipy.stats.linregress
        dendr_diam_parameters = {}

        dendr_diam1 = self.data_io.get_data(
            self.context.literature_data_files["dendr_diam1_fullpath"]
        )
        dendr_diam2 = self.data_io.get_data(
            self.context.literature_data_files["dendr_diam2_fullpath"]
        )
        dendr_diam3 = self.data_io.get_data(
            self.context.literature_data_files["dendr_diam3_fullpath"]
        )
        dendr_diam_units = self.context.literature_data_files["dendr_diam_units"]
        deg_per_mm = self.context.my_retina["deg_per_mm"]

        # Quality control. Datasets separately for visualization
        assert dendr_diam_units["data1"] == ["mm", "um"]
        data_set_1_x = np.squeeze(dendr_diam1["Xdata"])
        data_set_1_y = np.squeeze(dendr_diam1["Ydata"])
        assert dendr_diam_units["data2"] == ["mm", "um"]
        data_set_2_x = np.squeeze(dendr_diam2["Xdata"])
        data_set_2_y = np.squeeze(dendr_diam2["Ydata"])
        assert dendr_diam_units["data3"] == ["deg", "um"]
        data_set_3_x = np.squeeze(dendr_diam3["Xdata"]) / deg_per_mm
        data_set_3_y = np.squeeze(dendr_diam3["Ydata"])

        # Both datasets together
        data_all_x = np.concatenate((data_set_1_x, data_set_2_x, data_set_3_x))
        data_all_y = np.concatenate((data_set_1_y, data_set_2_y, data_set_3_y))

        # Limit eccentricities for central visual field studies to get better approximation at about 5 deg ecc (1mm)
        # x is eccentricity in mm
        # y is dendritic field diameter in micrometers
        data_all_x_index = data_all_x <= visual_field_limit_for_dd_fit_mm
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[
            data_all_x_index
        ]  # Don't forget to truncate values, too

        # Sort to ascending order
        data_all_x_index = np.argsort(data_all_x)
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[data_all_x_index]

        # Get rf diameter vs eccentricity
        # dd_regr_model is 'linear'  'quadratic' or cubic
        dict_key = "{0}_{1}".format(self.gc_type, dd_regr_model)

        if dd_regr_model == "linear":
            polynomial_order = 1
            fit_parameters = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": fit_parameters[1],
                "slope": fit_parameters[0],
            }
        elif dd_regr_model == "quadratic":
            polynomial_order = 2
            fit_parameters = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": fit_parameters[2],
                "slope": fit_parameters[1],
                "square": fit_parameters[0],
            }
        elif dd_regr_model == "cubic":
            polynomial_order = 3
            fit_parameters = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": fit_parameters[3],
                "slope": fit_parameters[2],
                "square": fit_parameters[1],
                "cube": fit_parameters[0],
            }
        elif dd_regr_model == "exponential":

            def exp_func(x, a, b):
                return a + np.exp(x / b)

            fit_parameters, pcov = opt.curve_fit(
                exp_func, data_all_x, data_all_y, p0=[0, 1]
            )
            dendr_diam_parameters[dict_key] = {
                "constant": fit_parameters[0],
                "lamda": fit_parameters[1],
            }
        elif dd_regr_model == "loglog":
            # Define the model function for the power law relationship
            # Note that we're fitting the log of the function, so we need to use the linear form
            def power_func(E, a, b):
                return a * np.power(E, b)

            fit_parameters, pcov = opt.curve_fit(
                power_func, data_all_x, data_all_y, p0=[1, 1]
            )

            a = fit_parameters[0]
            b = fit_parameters[1]

            # Save the parameters
            dendr_diam_parameters[dict_key] = {
                "a": a,
                "b": b,
            }

        dd_model_caption = f"All data {dd_regr_model} fit"

        self.project_data.construct_retina["dd_vs_ecc"] = {
            "data_all_x": data_all_x,
            "data_all_y": data_all_y,
            "fit_parameters": fit_parameters,
            "dd_model_caption": dd_model_caption,
            "title": f"DF diam wrt ecc for {self.gc_type} type, {dd_model_caption} dataset",
        }

        return dendr_diam_parameters

    def _get_ecc_from_dd(self, dendr_diam_parameters, dd_regr_model, dd):
        """
        Given the parameters of a polynomial and a dendritic diameter (dd), find the corresponding eccentricity.

        Parameters
        ----------
        dendr_diam_parameters : dict
            a dictionary containing the parameters of a polynomial that fits the dendritic diameter as a function of eccentricity
        dd_regr_model : str
            a string representing the type of polynomial ('linear', 'quadratic', or 'cubic')
        dd : float
            the dendritic diameter (micrometers) for which to find the corresponding eccentricity

        Returns
        -------
        float
            the eccentricity (millimeters from fovea, temporal equivalent) corresponding to the given dendritic diameter
        """
        # Get the parameters of the polynomial
        params = dendr_diam_parameters[f"{self.gc_type}_{dd_regr_model}"]

        if dd_regr_model == "linear":
            # For a linear equation, we can solve directly
            # y = mx + c => x = (y - c) / m
            return (dd - params["intercept"]) / params["slope"]
        elif dd_regr_model in ["quadratic", "cubic"]:
            # For quadratic and cubic equations, we need to solve numerically
            # Set up the polynomial equation
            def equation(x):
                if dd_regr_model == "quadratic":
                    return (
                        params["square"] * x**2
                        + params["slope"] * x
                        + params["intercept"]
                        - dd
                    )
                elif dd_regr_model == "cubic":
                    return (
                        params["cube"] * x**3
                        + params["square"] * x**2
                        + params["slope"] * x
                        + params["intercept"]
                        - dd
                    )

            # Solve the equation numerically and return the root
            # We use 1 as the initial guess
            return opt.root(equation, 1).x[0]

        elif dd_regr_model == "loglog":
            # For the loglog (power law) model, we can solve directly using the inversion
            # D = aE^b => E = (D/a)^(1/b)
            a = params["a"]
            b = params["b"]
            return np.power(dd / a, 1 / b)

    def _generate_DoG_with_rf_coverage_one(self):
        """
        Generate Difference of Gaussians (DoG) model with full retinal field coverage.

        This function ensures full coverage of the retinal field (coverage factor = 1).
        It updates the `gc_df` dataframe with spatial parameters converted from pixels in
        orginal experimental data space to millimeters of final retina. It applies scaling
        for retinal coverage of one at the given eccentricity.
        """
        # Create all gc units from parameters fitted to experimental data
        n_cells = len(self.gc_df)
        data_microm_per_pix = self.context.apricot_metadata["data_microm_per_pix"]
        spatial_df = self.exp_stat_df[self.exp_stat_df["domain"] == "spatial"]
        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # Change units to mm. Here the scale reflects Chichilnisky data and they are at large eccentricity
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            self.gc_df["semi_xc_mm"] = (
                self.gc_df["semi_xc_pix"] * data_microm_per_pix / 1000
            )
            self.gc_df["semi_yc_mm"] = (
                self.gc_df["semi_yc_pix"] * data_microm_per_pix / 1000
            )

        if self.context.my_retina["DoG_model"] == "ellipse_independent":
            # Add surround
            self.gc_df["semi_xs_mm"] = (
                self.gc_df["semi_xs_pix"] * data_microm_per_pix / 1000
            )
            self.gc_df["semi_ys_mm"] = (
                self.gc_df["semi_ys_pix"] * data_microm_per_pix / 1000
            )

        if self.context.my_retina["DoG_model"] == "circular":
            self.gc_df["rad_c_mm"] = (
                self.gc_df["rad_c_pix"] * data_microm_per_pix / 1000
            )
            self.gc_df["rad_s_mm"] = (
                self.gc_df["rad_s_pix"] * data_microm_per_pix / 1000
            )

        # Calculate RF diameter scaling factor for all ganglion cells. The surround in
        # ellipse_indenpendent model has the same scaling factor as the center.
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            area_rfs_cen_mm2 = (
                np.pi * self.gc_df["semi_xc_mm"] * self.gc_df["semi_yc_mm"]
            )

        elif self.context.my_retina["DoG_model"] == "circular":
            area_rfs_cen_mm2 = np.pi * self.gc_df["rad_c_mm"] ** 2

        """
        The area_of_rf contains area for all model units. Its sum must fill the whole area (coverage factor = 1).
        We do it separately for each ecc sector, step by step, to keep coverage factor at 1 despite changing gc density with ecc
        r_scaled = sqrt( (area_scaled / area) * r^2 ) => r_scaling_factor = sqrt( (area_scaled / area) )
        """
        # Calculate area scaling factors for each eccentricity group
        area_scaling_factors_coverage1 = np.zeros(area_rfs_cen_mm2.shape)
        for index, sector_area_mm2 in enumerate(self.sector_surface_areas_mm2):
            area_scaling_factor = (sector_area_mm2) / np.sum(
                area_rfs_cen_mm2[self.gc_df["ecc_group_idx"] == index]
            )

            area_scaling_factors_coverage1[
                self.gc_df["ecc_group_idx"] == index
            ] = area_scaling_factor

        radius_scaling_factors_coverage_1 = np.sqrt(area_scaling_factors_coverage1)

        # Save scaling factors for later working retina computations
        self.gc_df["gc_scaling_factors"] = radius_scaling_factors_coverage_1

        # Apply scaling factors.
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            semi_xc = radius_scaling_factors_coverage_1 * self.gc_df["semi_xc_mm"]

            semi_yc = radius_scaling_factors_coverage_1 * self.gc_df["semi_yc_mm"]

            self.gc_df["semi_xc_mm"] = semi_xc
            self.gc_df["semi_yc_mm"] = semi_yc

        if self.context.my_retina["DoG_model"] == "ellipse_independent":
            # Add surround
            semi_xs = radius_scaling_factors_coverage_1 * self.gc_df["semi_xs_mm"]

            semi_ys = radius_scaling_factors_coverage_1 * self.gc_df["semi_ys_mm"]

            self.gc_df["semi_xs_mm"] = semi_xs
            self.gc_df["semi_ys_mm"] = semi_ys

        if self.context.my_retina["DoG_model"] == "circular":
            rad_c = radius_scaling_factors_coverage_1 * self.gc_df["rad_c_mm"]
            self.gc_df["rad_c_mm"] = rad_c

            rad_s = radius_scaling_factors_coverage_1 * self.gc_df["rad_s_mm"]
            self.gc_df["rad_s_mm"] = rad_s

    def _generate_DoG_with_rf_from_literature(self, lit_dd_vs_ecc_params):
        """
        Generate Difference of Gaussians (DoG) model with dendritic field sizes from literature.

        Places all ganglion cell spatial parameters to ganglion cell object dataframe self.gc_df

        At return, all units are in mm unless stated otherwise in the the column name
        """

        dd_regr_model = self.dd_regr_model  # "linear", "quadratic", "cubic"

        # Get eccentricity data for all model cells
        gc_eccentricity = self.gc_df["pos_ecc_mm"].values

        # Get rf diameter vs eccentricity
        dict_key = "{0}_{1}".format(self.gc_type, dd_regr_model)
        diam_fit_params = lit_dd_vs_ecc_params[dict_key]

        if dd_regr_model == "linear":
            lit_cen_diameter_um = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
            )  # Units are micrometers
        elif dd_regr_model == "quadratic":
            lit_cen_diameter_um = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
                + diam_fit_params["square"] * gc_eccentricity**2
            )
        elif dd_regr_model == "cubic":
            lit_cen_diameter_um = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
                + diam_fit_params["square"] * gc_eccentricity**2
                + diam_fit_params["cube"] * gc_eccentricity**3
            )
        elif dd_regr_model == "exponential":
            lit_cen_diameter_um = diam_fit_params["constant"] + np.exp(
                gc_eccentricity / diam_fit_params["lamda"]
            )
        elif dd_regr_model == "loglog":
            lit_cen_diameter_um = diam_fit_params["a"] * np.power(
                gc_eccentricity, diam_fit_params["b"]
            )

        # Create all gc units from parameters fitted to experimental data
        n_cells = len(self.gc_df)
        spatial_df = self.exp_stat_df[self.exp_stat_df["domain"] == "spatial"]
        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # Scale factor for semi_x and semi_y from pix at data eccentricity to pix at the actual eccentricity
        # Units are pixels for the Chichilnisky data and they are at large eccentricity
        gc_scaling_factors = (lit_cen_diameter_um / 2) / (self.exp_cen_radius_mm * 1000)
        # Save scaling factors to gc_df for later use
        self.gc_df["gc_scaling_factors"] = gc_scaling_factors

        um_per_pixel = self.context.apricot_metadata["data_microm_per_pix"]
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            # Scale semi_x to pix at its actual eccentricity
            self.gc_df["semi_xc_pix_eccscaled"] = (
                self.gc_df["semi_xc_pix"] * gc_scaling_factors
            )

            # Scale semi_x to mm
            self.gc_df["semi_xc_mm"] = (
                self.gc_df["semi_xc_pix_eccscaled"] * um_per_pixel / 1000
            )

            # Scale semi_y to pix at its actual eccentricity
            self.gc_df["semi_yc_pix_eccscaled"] = (
                self.gc_df["semi_yc_pix"] * gc_scaling_factors
            )

            # Scale semi_y to mm
            self.gc_df["semi_yc_mm"] = (
                self.gc_df["semi_yc_pix_eccscaled"] * um_per_pixel / 1000
            )
        if self.context.my_retina["DoG_model"] == "ellipse_independent":
            # Surround
            # Scale semi_x to pix at its actual eccentricity
            self.gc_df["semi_xs_pix_eccscaled"] = (
                self.gc_df["semi_xs_pix"] * gc_scaling_factors
            )

            # Scale semi_x to mm
            self.gc_df["semi_xs_mm"] = (
                self.gc_df["semi_xs_pix_eccscaled"] * um_per_pixel / 1000
            )

            # Scale semi_y to pix at its actual eccentricity
            self.gc_df["semi_ys_pix_eccscaled"] = (
                self.gc_df["semi_ys_pix"] * gc_scaling_factors
            )

            # Scale semi_y to mm
            self.gc_df["semi_ys_mm"] = (
                self.gc_df["semi_ys_pix_eccscaled"] * um_per_pixel / 1000
            )

        elif self.context.my_retina["DoG_model"] == "circular":
            # Scale rad_c to pix at its actual eccentricity
            self.gc_df["rad_c_pix_eccscaled"] = (
                self.gc_df["rad_c_pix"] * gc_scaling_factors
            )

            # Scale rad_c to mm
            self.gc_df["rad_c_mm"] = (
                self.gc_df["rad_c_pix_eccscaled"] * um_per_pixel / 1000
            )

            # Same for rad_s
            self.gc_df["rad_s_pix_eccscaled"] = (
                self.gc_df["rad_s_pix"] * gc_scaling_factors
            )
            self.gc_df["rad_s_mm"] = (
                self.gc_df["rad_s_pix_eccscaled"] * um_per_pixel / 1000
            )

    def _densfunc(self, r, d0, beta):
        return d0 * (1 + beta * r) ** (-2)

    # GC placement functions
    def _initialize_positions_by_group(self, gc_density_func_params):
        """
        Initialize cell positions based on grouped eccentricities.

        Parameters
        ----------
        gc_density_func_params : tuple
            Parameters for the density function used to calculate cell density
            at a given eccentricity.

        Returns
        -------
        eccentricity_groups : list of ndarray
            A list of arrays where each array contains group indices representing
            eccentricity steps for cells.

        initial_positions : list of ndarray
            A list of arrays where each array contains cell positions for each
            eccentricity group.

        sector_surface_areas_mm2 : list of float
            Area in mm^2 for each eccentricity step.

        Notes
        -----
        The method divides the total eccentricity range into smaller steps,
        calculates cell densities and positions for each step, and returns
        group-wise cell positions and the corresponding sector surface areas.
        """
        # Loop for reasonable delta ecc to get correct density in one hand and good cell distribution from the algo on the other
        # Lets fit close to 0.1 mm intervals, which makes sense up to some 15 deg. Thereafter longer jumps would do fine.
        assert (
            self.ecc_lim_mm[0] < self.ecc_lim_mm[1]
        ), "Radii in wrong order, give [min max], aborting"
        eccentricity_in_mm_total = self.ecc_lim_mm
        fit_interval = 0.1  # mm
        n_steps = math.ceil(np.ptp(eccentricity_in_mm_total) / fit_interval)
        eccentricity_steps = np.linspace(
            eccentricity_in_mm_total[0], eccentricity_in_mm_total[1], 1 + n_steps
        )

        angle_deg = np.ptp(self.polar_lim_deg)  # The angle_deg is now == max theta_deg

        eccentricity_groups = []
        initial_positions = []
        areas_all_mm2 = []
        density_prop_all = []
        for group_idx in range(len(eccentricity_steps) - 1):
            min_ecc = eccentricity_steps[group_idx]
            max_ecc = eccentricity_steps[group_idx + 1]
            avg_ecc = (min_ecc + max_ecc) / 2
            density = self.gauss_plus_baseline(avg_ecc, *gc_density_func_params)
            # density_prop_all.append(density * self.gc_proportion)

            # Calculate area for this eccentricity group
            sector_area_remove = self.sector2area_mm2(min_ecc, angle_deg)
            sector_area_full = self.sector2area_mm2(max_ecc, angle_deg)
            sector_surface_area = sector_area_full - sector_area_remove  # in mm2
            # collect sector area for each ecc step
            areas_all_mm2.append(sector_surface_area)

            n_cells = math.ceil(sector_surface_area * density * self.gc_proportion)
            positions = self._random_positions_within_group(min_ecc, max_ecc, n_cells)
            eccentricity_groups.append(np.full(n_cells, group_idx))
            density_prop_all.append(np.full(n_cells, density * self.gc_proportion))
            initial_positions.append(positions)

        gc_density = np.concatenate(density_prop_all)

        return eccentricity_groups, initial_positions, areas_all_mm2, gc_density

    def _boundary_force(self, positions, rep, dist_th, ecc_lim_mm, polar_lim_deg):
        """
        Calculate boundary repulsive forces for given positions based on both
        eccentricity (left-right) and polar (bottom-top) constraints.

        Parameters
        ----------
        positions : torch.Tensor
            A tensor of positions (shape: [N, 2], where N is the number of nodes).
        rep : float or torch.Tensor
            Repulsion coefficient for boundary force.
        dist_th : float or torch.Tensor
            Distance threshold beyond which no force is applied.
        clamp_min : float or torch.Tensor
            Minimum distance value to avoid division by very small numbers.
        ecc_lim_mm : torch.Tensor
            A tensor representing the eccentricity limits in millimeters for
            left and right boundaries (shape: [2]).
        polar_lim_deg : torch.Tensor
            A tensor representing the polar angle limits in degrees for
            bottom and top boundaries (shape: [2]).

        Returns
        -------
        forces : torch.Tensor
            A tensor of forces (shape: [N, 2]) for each position.

        Notes
        -----
        This method calculates repulsive forces between the given positions and
        the defined boundaries based on both eccentricity and polar constraints.
        Repulsion is based on the inverse cube law. The method calculates the repulsive
        forces by determining the distances of nodes to these boundaries and applying
        the inverse cube law based on those distances.
        """

        forces = torch.zeros_like(positions)
        clamp_min = 1e-5

        # Polar angle-based calculations for bottom and top boundaries
        bottom_x, bottom_y = self._pol2cart_torch(
            ecc_lim_mm, polar_lim_deg[0].expand_as(ecc_lim_mm)
        )
        top_x, top_y = self._pol2cart_torch(
            ecc_lim_mm, polar_lim_deg[1].expand_as(ecc_lim_mm)
        )
        m_bottom = (bottom_y[1] - bottom_y[0]) / (bottom_x[1] - bottom_x[0])
        c_bottom = bottom_y[0] - m_bottom * bottom_x[0]
        m_top = (top_y[1] - top_y[0]) / (top_x[1] - top_x[0])
        c_top = top_y[0] - m_top * top_x[0]

        # Calculating distance from the line for each position
        bottom_distance = torch.abs(
            m_bottom * positions[:, 0] - positions[:, 1] + c_bottom
        ) / torch.sqrt(m_bottom**2 + 1)
        top_distance = torch.abs(
            m_top * positions[:, 0] - positions[:, 1] + c_top
        ) / torch.sqrt(m_top**2 + 1)

        # Computing repulsive forces based on these distances
        bottom_force = rep / (bottom_distance.clamp(min=clamp_min) ** 3)
        bottom_force[bottom_distance > dist_th] = 0
        forces[:, 1] -= bottom_force

        top_force = rep / (top_distance.clamp(min=clamp_min) ** 3)
        top_force[top_distance > dist_th] = 0
        forces[:, 1] += top_force

        # Eccentricity arc-based calculations for min and max arcs
        distances_to_center = torch.norm(positions, dim=1)
        min_ecc_distance = torch.abs(distances_to_center - ecc_lim_mm[0])
        max_ecc_distance = torch.abs(distances_to_center - ecc_lim_mm[1])

        # Compute forces based on these distances
        min_ecc_force = rep / (min_ecc_distance.clamp(min=clamp_min) ** 3)
        min_ecc_force[min_ecc_distance > dist_th] = 0

        max_ecc_force = rep / (max_ecc_distance.clamp(min=clamp_min) ** 3)
        max_ecc_force[max_ecc_distance > dist_th] = 0

        # Calculate direction for the forces (from point to the origin)
        directions = positions / distances_to_center.unsqueeze(1)

        # Update forces using the computed repulsive forces and their directions
        forces -= directions * min_ecc_force.unsqueeze(1)
        forces += directions * max_ecc_force.unsqueeze(1)

        return forces

    def _pol2cart_torch(self, radius, phi, deg=True):
        """
        Convert polar coordinates to Cartesian coordinates using PyTorch tensors.

        Parameters
        ----------
        radius : torch.Tensor
            Tensor representing the radius value in real distance such as mm.
        phi : list or torch.Tensor
            Tensor or list representing the polar angle value.
        deg : bool, optional
            If True, the angle is given in degrees; if False, the angle is given
            in radians. Default is True.

        Returns
        -------
        x, y : torch.Tensor
            Cartesian coordinates corresponding to the input polar coordinates.
        """

        if deg:
            theta = phi * torch.pi / 180
        else:
            theta = phi

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        return (x, y)

    def _cart2pol_torch(self, x, y, deg=True):
        """
        Convert Cartesian coordinates to polar coordinates using PyTorch tensors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor representing the x-coordinate in Cartesian coordinates.
        y : torch.Tensor
            Tensor representing the y-coordinate in Cartesian coordinates.
        deg : bool, optional
            If True, the returned angle is in degrees; if False, the angle is in
            radians. Default is True.

        Returns
        -------
        radius, phi : torch.Tensor
            Polar coordinates corresponding to the input Cartesian coordinates.
        """

        radius = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        if deg:
            phi = theta * 180 / torch.pi
        else:
            phi = theta

        return radius, phi

    def _check_boundaries(self, node_positions, ecc_lim_mm, polar_lim_deg):
        x, y = node_positions[:, 0], node_positions[:, 1]
        min_eccentricity, max_eccentricity = ecc_lim_mm
        min_polar, max_polar = polar_lim_deg

        r, theta = self._cart2pol_torch(x, y)
        # Guarding eccentricity boundaries
        r = torch.clamp(r, min=min_eccentricity, max=max_eccentricity)
        # Guarding polar boundaries
        theta = torch.clamp(theta, min=min_polar, max=max_polar)

        new_x, new_y = self._pol2cart_torch(r, theta)

        delta_x = new_x - x
        delta_y = new_y - y

        return torch.stack([delta_x, delta_y], dim=1)

    def _apply_force_based_layout(self, all_positions, gc_density):
        """
        Apply a force-based layout on the given positions.

        Parameters
        ----------
        all_positions : list or ndarray
            Initial positions of nodes.
        gc_density : float
            One local density according to eccentricity group.
        Returns
        -------
        positions : ndarray
            New positions of nodes after the force-based optimization.

        Notes
        -----
        This method applies a force-based layout to optimize node positions.
        It visualizes the progress of the layout optimization.
        """

        gc_placement_params = self.context.my_retina["gc_placement_params"]
        n_iterations = gc_placement_params["n_iterations"]
        change_rate = gc_placement_params["change_rate"]
        unit_repulsion_stregth = gc_placement_params["unit_repulsion_stregth"]
        unit_distance_threshold = gc_placement_params["unit_distance_threshold"]
        diffusion_speed = gc_placement_params["diffusion_speed"]
        border_repulsion_stength = gc_placement_params["border_repulsion_stength"]
        border_distance_threshold = gc_placement_params["border_distance_threshold"]
        show_placing_progress = gc_placement_params["show_placing_progress"]
        show_skip_steps = gc_placement_params["show_skip_steps"]

        if show_placing_progress is True:
            # Init plotting
            fig_args = self.viz.show_gc_placement_progress(all_positions, init=True)

        unit_distance_threshold = torch.tensor(unit_distance_threshold).to(self.device)
        unit_repulsion_stregth = torch.tensor(unit_repulsion_stregth).to(self.device)
        diffusion_speed = torch.tensor(diffusion_speed).to(self.device)
        n_iterations = torch.tensor(n_iterations).to(self.device)
        gc_density = torch.tensor(gc_density).to(self.device)

        rep = torch.tensor(border_repulsion_stength).to(self.device)
        dist_th = torch.tensor(border_distance_threshold).to(self.device)

        original_positions = deepcopy(all_positions)
        positions = torch.tensor(
            all_positions, requires_grad=True, dtype=torch.float64, device=self.device
        )
        change_rate = torch.tensor(change_rate).to(self.device)
        optimizer = torch.optim.Adam([positions], lr=change_rate, betas=(0.95, 0.999))

        ecc_lim_mm = torch.tensor(self.ecc_lim_mm).to(self.device)
        polar_lim_deg = torch.tensor(self.polar_lim_deg).to(self.device)
        boundary_polygon = self.viz.boundary_polygon(ecc_lim_mm, polar_lim_deg)

        # Adjust unit_distance_threshold and diffusion speed with density of the units
        # This is necessary because the density of the units are adjusted with eccentricity
        # The 1 mm ecc for parasol provides 952 units/mm2 density. This is the reference density.
        gc_distance_threshold = unit_distance_threshold * (952 / gc_density)
        gc_diffusion_speed = diffusion_speed * (952 / gc_density)

        for iteration in torch.range(0, n_iterations):
            optimizer.zero_grad()
            # Repulsive force between nodes
            diff = positions[None, :, :] - positions[:, None, :]
            dist = torch.norm(diff, dim=-1, p=2) + 1e-9

            # Clip minimum distance to avoid very high repulsion
            dist = torch.clamp(dist, min=0.00001)
            # Clip max to inf (zero repulsion) above a certain distance
            dist[dist > gc_distance_threshold] = torch.inf
            # Using inverse cube for repulsion
            repulsive_force = unit_repulsion_stregth * torch.sum(
                diff / (dist[..., None] ** 3), dim=1
            )

            # After calculating repulsive_force:
            boundary_forces = self._boundary_force(
                positions, rep, dist_th, ecc_lim_mm, polar_lim_deg
            )

            total_force = repulsive_force + boundary_forces

            # Use the force as the "loss"
            loss = torch.norm(total_force, p=2)

            loss.backward()
            optimizer.step()

            # Update positions in-place
            positions_delta = self._check_boundaries(
                positions, ecc_lim_mm, polar_lim_deg
            )

            gc_diffusion_speed_reshaped = gc_diffusion_speed.view(-1, 1)
            new_data = (
                torch.randn_like(positions) * gc_diffusion_speed_reshaped
                + positions_delta
            )
            positions.data = positions + new_data

            if show_placing_progress is True:
                # Update the visualization every 100 iterations for performance (or adjust as needed)
                if iteration % show_skip_steps == 0:
                    positions_cpu = positions.detach().cpu().numpy()
                    self.viz.show_gc_placement_progress(
                        original_positions=original_positions,
                        positions=positions_cpu,
                        iteration=iteration,
                        boundary_polygon=boundary_polygon,
                        **fig_args,
                    )

        if show_placing_progress is True:
            plt.ioff()  # Turn off interactive mode

        return positions.detach().cpu().numpy()

    def _apply_voronoi_layout(self, all_positions):
        """
        Apply a Voronoi-based layout on the given positions.

        Parameters
        ----------
        all_positions : list or ndarray
            Initial positions of nodes.

        Returns
        -------
        positions : ndarray
            New positions of nodes after the Voronoi-based optimization.

        Notes
        -----
        This method applies a Voronoi diagram to optimize node positions.
        It uses Lloyd's relaxation for iteratively adjusting seed points.
        """

        # Extract parameters from context
        gc_placement_params = self.context.my_retina["gc_placement_params"]
        n_iterations = gc_placement_params["n_iterations"]
        change_rate = gc_placement_params["change_rate"]
        show_placing_progress = gc_placement_params["show_placing_progress"]
        show_skip_steps = gc_placement_params["show_skip_steps"]

        if show_placing_progress:
            fig_args = self.viz.show_gc_placement_progress(all_positions, init=True)

        def polygon_centroid(polygon):
            """Compute the centroid of a polygon."""
            A = 0.5 * np.sum(
                polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1]
            )
            C_x = (1 / (6 * A)) * np.sum(
                (polygon[:-1, 0] + polygon[1:, 0])
                * (polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1])
            )
            C_y = (1 / (6 * A)) * np.sum(
                (polygon[:-1, 1] + polygon[1:, 1])
                * (polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1])
            )
            return np.array([C_x, C_y])

        ecc_lim_mm = self.ecc_lim_mm
        polar_lim_deg = self.polar_lim_deg
        boundary_polygon = self.viz.boundary_polygon(ecc_lim_mm, polar_lim_deg)
        original_positions = all_positions.copy()
        positions = all_positions.copy()
        boundary_polygon_shape = ShapelyPolygon(boundary_polygon)

        for iteration in range(n_iterations):
            vor = Voronoi(positions)
            new_positions = []
            old_positions = []
            intersected_polygons = []

            for region, original_seed in zip(vor.regions, original_positions):
                if not -1 in region and len(region) > 0:
                    polygon = np.array([vor.vertices[i] for i in region])
                    voronoi_cell_shape = ShapelyPolygon(polygon)

                    # Find the intersection between the Voronoi cell and the boundary polygon
                    intersection_shape = voronoi_cell_shape.intersection(
                        boundary_polygon_shape
                    )

                    if intersection_shape.is_empty:
                        new_positions.append(original_seed)
                        continue

                    intersection_polygon = np.array(intersection_shape.exterior.coords)

                    # Wannabe centroid
                    new_seed = polygon_centroid(intersection_polygon)

                    if show_placing_progress and iteration % show_skip_steps == 0:
                        # Take polygons for viz.
                        intersected_polygons.append(intersection_polygon)
                        old_positions.append(new_seed)

                    # We cool things down a bit by moving the centroid only the change_rate of the way
                    diff = new_seed - original_seed
                    partial_diff = diff * change_rate
                    new_seed = original_seed + partial_diff
                    new_positions.append(new_seed)

                else:
                    new_positions.append(original_seed)

            # Convert to torch tensor for boundary check
            positions_torch = torch.tensor(new_positions, dtype=torch.float32).to("cpu")

            # Check boundaries and adjust positions if needed
            position_deltas = self._check_boundaries(
                positions_torch,
                torch.tensor(self.ecc_lim_mm),
                torch.tensor(self.polar_lim_deg),
            )

            positions = (positions_torch + position_deltas).numpy()

            if show_placing_progress and iteration % show_skip_steps == 0:
                self.viz.show_gc_placement_progress(
                    original_positions=original_positions,
                    positions=np.array(old_positions),
                    iteration=iteration,
                    intersected_polygons=intersected_polygons,
                    boundary_polygon=boundary_polygon,
                    **fig_args,
                )

                # wait = input("Press enter to continue")

        if show_placing_progress:
            plt.ioff()

        return positions

    def _random_positions_within_group(self, min_ecc, max_ecc, n_cells):
        eccs = np.random.uniform(min_ecc, max_ecc, n_cells)
        angles = np.random.uniform(
            self.polar_lim_deg[0], self.polar_lim_deg[1], n_cells
        )
        return np.column_stack((eccs, angles))

    def _place_gc_units(self, gc_density_func_params):
        # 1. Initial Positioning by Group
        (
            eccentricity_groups,
            initial_positions,
            sector_surface_areas_mm2,
            gc_density,
        ) = self._initialize_positions_by_group(gc_density_func_params)

        # 2. Merge the Groups
        all_positions = np.vstack(initial_positions)
        all_positions_tuple = self.pol2cart(all_positions[:, 0], all_positions[:, 1])
        all_positions_mm = np.column_stack(all_positions_tuple)

        # 3 Optimize positions
        optim_algorithm = self.context.my_retina["gc_placement_params"]["algorithm"]
        if optim_algorithm == None:
            # Initial random placement.
            # Use this for testing/speed/nonvarying placements.
            optimized_positions = all_positions
        else:
            if optim_algorithm == "force":
                # Apply Force Based Layout Algorithm with Boundary Repulsion
                optimized_positions_mm = self._apply_force_based_layout(
                    all_positions_mm, gc_density
                )
            elif optim_algorithm == "voronoi":
                # Apply Voronoi-based Layout with Loyd's Relaxation
                optimized_positions_mm = self._apply_voronoi_layout(all_positions_mm)
            optimized_positions_tuple = self.cart2pol(
                optimized_positions_mm[:, 0], optimized_positions_mm[:, 1]
            )
            optimized_positions = np.column_stack(optimized_positions_tuple)

        # 4. Assign Output Variables
        self.gc_df["pos_ecc_mm"] = optimized_positions[:, 0]
        self.gc_df["pos_polar_deg"] = optimized_positions[:, 1]
        self.gc_df["ecc_group_idx"] = np.concatenate(eccentricity_groups)
        self.sector_surface_areas_mm2 = sector_surface_areas_mm2
        self.gc_density_func_params = gc_density_func_params

    # temporal filter and tonic frive functions
    def _create_fixed_temporal_rfs(self):
        n_cells = len(self.gc_df)
        temporal_df = self.exp_stat_df[self.exp_stat_df["domain"] == "temporal"]
        for param_name, row in temporal_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

    def _read_temporal_statistics_benardete_kaplan(self):
        """
        Fit temporal statistics of the temporal parameters using the triangular distribution.
        Data from Benardete & Kaplan Visual Neuroscience 16 (1999) 355-368 (parasol cells), and
        Benardete & Kaplan Visual Neuroscience 14 (1997) 169-185 (midget cells).

        Returns
        -------
        temporal_exp_stat_df : pd.DataFrame
            A DataFrame containing the temporal statistics of the temporal filter parameters, including the shape, loc,
            and scale parameters of the fitted gamma distribution, as well as the name of the distribution and the domain.

        temp_stat : dict
            A dictionary containing information needed for visualization, including the temporal filter parameters, the
            fitted distribution parameters, the super title of the plot, `self.gc_type`, `self.response_type`, and the
            `self.all_data_fits_df` DataFrame.
        """

        cell_type = self.gc_type

        if cell_type == "parasol":
            temporal_model_parameters = [
                "A",
                "NLTL",
                "NL",
                "TL",
                "HS",
                "T0",
                "Chalf",
                "D",
            ]

        elif cell_type == "midget":
            temporal_model_parameters = [
                "A_cen",
                "NLTL_cen",
                "NL_cen",
                "HS_cen",
                "TS_cen",
                "D_cen",
                "A_sur",
                "NLTL_sur",
                "NL_sur",
                "HS_sur",
                "TS_sur",
                "deltaNLTL_sur",
            ]

        col_names = ["Minimum", "Maximum", "Median", "Mean", "SD", "SEM"]
        distrib_params = np.zeros((len(temporal_model_parameters), 3))
        response_type = self.response_type.upper()

        temp_params_df = pd.read_csv(
            self.context.literature_data_files["temporal_BK_model_fullpath"]
        )

        for i, param_name in enumerate(temporal_model_parameters):
            condition = (temp_params_df["Parameter"] == param_name) & (
                temp_params_df["Type"] == response_type
            )

            param_df = temp_params_df[condition].loc[:, col_names]

            if param_df.empty:
                continue

            minimum, maximum, median, mean, sd, sem = param_df.values[0]

            # Midget type contains separate A_cen and A_sur parameters
            # Their relative snr is used to scale A_cen/A_sur at
            # _create_dynamic_temporal_rfs
            if param_name == "A_cen":
                A_cen_snr = mean / sd
            if param_name == "A_sur":
                A_sur_snr = mean / sd

            c, loc, scale = self.get_triangular_parameters(
                minimum, maximum, median, mean, sd, sem
            )
            distrib_params[i, :] = [c, loc, scale]

        temporal_exp_stat_df = pd.DataFrame(
            distrib_params,
            index=temporal_model_parameters,
            columns=["shape", "loc", "scale"],
        )
        temporal_exp_stat_df["distribution"] = "triang"
        temporal_exp_stat_df["domain"] = "temporal_BK"
        all_data_fits_df = pd.concat([self.exp_stat_df, temporal_exp_stat_df], axis=0)

        # Add snr to scale A_cen/A_sur at _create_dynamic_temporal_rfs
        if cell_type == "midget":
            temporal_exp_stat_df["snr"] = np.nan
            temporal_exp_stat_df.loc["A_cen", "snr"] = A_cen_snr
            temporal_exp_stat_df.loc["A_sur", "snr"] = A_sur_snr

        self.project_data.construct_retina["exp_temp_BK_model"] = {
            "temporal_model_parameters": temporal_model_parameters,
            "distrib_params": distrib_params,
            "suptitle": self.gc_type + " " + self.response_type,
            "all_data_fits_df": all_data_fits_df,
        }

        self.exp_stat_df = all_data_fits_df

        return temporal_exp_stat_df

    def _create_dynamic_temporal_rfs(self):
        n_cells = len(self.gc_df)

        temporal_bk_stat_df = self._read_temporal_statistics_benardete_kaplan()
        for param_name, row in temporal_bk_stat_df.iterrows():
            shape, loc, scale, distribution, *_ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # For midget type, get snr-weighted average of A_cen and A_sur
        if self.gc_type == "midget":
            snr_cen = temporal_bk_stat_df.loc["A_cen", "snr"]
            snr_sur = temporal_bk_stat_df.loc["A_sur", "snr"]
            weight_cen = snr_cen / (snr_cen + snr_sur)
            weight_sur = snr_sur / (snr_cen + snr_sur)
            self.gc_df["A"] = (
                self.gc_df["A_cen"] * weight_cen + self.gc_df["A_sur"] * weight_sur
            )

    def _scale_both_amplitudes(self, df):
        """
        Scale center amplitude to center volume of one.
        Scale surround amplitude also to center volume of one.
        Volume of 2D Gaussian = 2 * pi * sigma_x*sigma_y

        Second step of scaling is done before convolving with the stimulus.
        """
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            cen_vol_mm3 = 2 * np.pi * df["semi_xc_mm"] * df["semi_yc_mm"]
        elif self.context.my_retina["DoG_model"] == "circular":
            cen_vol_mm3 = np.pi * df["rad_c_mm"] ** 2

        df["relat_sur_ampl"] = df["ampl_s"] / df["ampl_c"]

        # This sets center volume (sum of all pixel values in data, after fitting) to one
        ampl_c_norm = 1 / cen_vol_mm3
        ampl_s_norm = df["relat_sur_ampl"] / cen_vol_mm3

        df["ampl_c_norm"] = ampl_c_norm
        df["ampl_s_norm"] = ampl_s_norm

        return df

    def _create_tonic_drive(self):
        """
        Create tonic drive for each cell.
        """
        tonic_df = self.exp_stat_df[self.exp_stat_df["domain"] == "tonic"]
        for param_name, row in tonic_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, len(self.gc_df), distribution
            )

    # spatial filter functions
    def _get_generated_rfs(self, retina_vae, n_samples=10):
        """
        Get the spatial data generated by the retina VAE.

        Parameters
        ----------
        retina_vae : RetinaVAE
            A RetinaVAE object.
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        img_flipped : numpy.ndarray
            Processed (median removed and fileed positive max up) img stack
            of shape (n_samples, img_size, img_size).
        img_reshaped : numpy.ndarray
            Image stack of shape (n_samples, img_size, img_size).
        """

        # --- 1. make a probability density function of the latent space

        latent_data = self.get_data_at_latent_space(retina_vae)

        # Make a probability density function of the latent_data
        # Both uniform and normal distr during learning is sampled
        # using gaussian kde estimate. The kde estimate is basically smooth histogram,
        # so it is not a problem that the data is not normal.
        latent_pdf = stats.gaussian_kde(latent_data.T)

        # --- 2. sample from the pdf
        latent_samples = torch.tensor(latent_pdf.resample(n_samples).T).to(
            retina_vae.device
        )
        # Change the dtype to float32
        latent_samples = latent_samples.type(torch.float32)
        latent_dim = retina_vae.latent_dim

        self.project_data.construct_retina["gen_latent_space"] = {
            "samples": latent_samples.to("cpu").numpy(),
            "dim": latent_dim,
            "data": latent_data,
        }

        # --- 3. decode the samples
        img_stack_np = retina_vae.vae.decoder(latent_samples)

        # The shape of img_stack_np is (n_samples, 1, img_size, img_size)
        # Reshape to (n_samples, img_size, img_size)
        img_reshaped = np.reshape(
            img_stack_np.detach().cpu().numpy(),
            (n_samples, img_stack_np.shape[2], img_stack_np.shape[3]),
        )

        # The images are in img_reshaped of shape (n_samples, img_size, img_size)
        # For each image, get the median value across 2D image
        medians = np.median(img_reshaped, axis=(1, 2))

        # Then, subtract the median value from the image. This sets the median value to 0.
        img_median_removed = img_reshaped - medians[:, None, None]

        # For each image,
        #   if the abs(min) > abs(max), then the image is flipped so that the strongest deviation becomes positive.
        img_flipped = img_median_removed
        for i in range(img_flipped.shape[0]):
            if abs(np.min(img_flipped[i])) > abs(np.max(img_flipped[i])):
                img_flipped[i] = -img_flipped[i]

        return img_flipped, img_reshaped

    def _get_retina_with_rf_masks(
        self,
        rf_masks,
        rspace_pos_mm,
    ):
        pass

    def _get_rf_resampling_params(self, lit_dd_vs_ecc_params):
        """
        Place rf images to pixel space
        First we calculate the dendritic diameter as a function of eccentricity for each gc unit.
        This dendritic diameter is then used to calculate the pixel size in um for each gc unit.
        The minimum pixel size is used to determine the new image stack sidelength.

        Later we will resample original VAE images to this space.

        Notes
        -----
        gc_pos_ecc_mm is expected to be slightly different each time, because of placement optimization process.
        """

        # TODO: implement Log-log line estimate of parameters: D = 12.25 * np.power(E, 0.757), from Goodchild et al. 1996 J Comp Neurol

        gc_pos_ecc_mm = np.array(self.gc_vae_df.pos_ecc_mm.values)
        exp_um_per_pix = self.context.apricot_metadata["data_microm_per_pix"]
        # Mean fitted dendritic diameter for the original experimental data
        exp_dd_um = self.exp_cen_radius_mm * 2 * 1000  # in micrometers
        exp_sidelen = self.context.apricot_metadata["data_spatialfilter_height"]

        key = list(lit_dd_vs_ecc_params.keys())[0]
        parameters = lit_dd_vs_ecc_params[key]
        if self.context.my_retina["dd_regr_model"] in ["linear", "quadratic", "cubic"]:
            lit_dd_at_gc_ecc_um = np.polyval(
                [
                    parameters.get("cube", 0),
                    parameters.get("square", 0),
                    parameters.get("slope", 0),
                    parameters.get("intercept", 0),
                ],
                gc_pos_ecc_mm,
            )
        elif self.context.my_retina["dd_regr_model"] == "exponential":
            lit_dd_at_gc_ecc_um = parameters.get("constant", 0) + np.exp(
                gc_pos_ecc_mm / parameters.get("lamda", 0)
            )
        elif self.context.my_retina["dd_regr_model"] == "loglog":
            # Calculate dendritic diameter from the power law relationship
            # D = a * E^b, where E is the eccentricity and D is the dendritic diameter
            a = parameters["a"]
            b = parameters["b"]
            # Eccentricity in mm, dendritic diameter in um
            lit_dd_at_gc_ecc_um = a * np.power(gc_pos_ecc_mm, b)
        else:
            raise ValueError(
                f"Unknown dd_regr_model: {self.context.my_retina['dd_regr_model']}"
            )

        # Assuming the experimental data reflects the eccentricity for
        # VAE mtx generation
        gc_scaling_factors = lit_dd_at_gc_ecc_um / exp_dd_um
        gc_um_per_pix = gc_scaling_factors * exp_um_per_pix

        # Get min and max values of gc_um_per_pix
        min_um_per_pix = np.min(gc_um_per_pix)
        max_um_per_pix = np.max(gc_um_per_pix)

        # Get new img stack sidelength whose pixel size = min(gc_um_per_pix),
        new_sidelen = int(np.round((max_um_per_pix / min_um_per_pix) * exp_sidelen))

        # Save scaling factors to gc_df for VAE model type
        self.gc_vae_df["gc_scaling_factors"] = gc_scaling_factors
        self.gc_vae_df["zoom_factor"] = gc_um_per_pix / min_um_per_pix

        return new_sidelen, min_um_per_pix

    def _get_resampled_scaled_rfs(
        self,
        rfs,
        new_sidelen,
    ):
        # Resample all images to new img stack. Use scipy.ndimage.zoom,
        img_upsampled = np.zeros((len(rfs), new_sidelen, new_sidelen))
        for i, img in enumerate(rfs):
            # zoom_factor = this_gc_um_per_pix / min_um_per_pix
            zoom_factor = self.gc_vae_df["zoom_factor"][i]

            # Pad the image with zeros to achieve the new dimensions
            # If new_sidelen - img.shape[0] is even:
            if (new_sidelen - img.shape[0]) % 2 == 0:
                padding = int((new_sidelen - img.shape[0]) / 2)
            elif (new_sidelen - img.shape[0]) % 2 == 1:
                padding = (
                    int((new_sidelen - img.shape[0]) / 2),
                    int((new_sidelen - img.shape[0]) / 2) + 1,
                )  # (before, after)

            img_padded = np.pad(
                img, pad_width=padding, mode="constant", constant_values=0
            )

            # Upsample the padded image
            img_temp = ndimage.zoom(img_padded, zoom_factor)

            # Crop the upsampled image to the new dimensions
            crop_length = new_sidelen / 2
            img_cropped = img_temp[
                int(img_temp.shape[0] / 2 - crop_length) : int(
                    img_temp.shape[0] / 2 + crop_length
                ),
                int(img_temp.shape[1] / 2 - crop_length) : int(
                    img_temp.shape[1] / 2 + crop_length
                ),
            ]

            img_upsampled[i] = img_cropped

        return img_upsampled

    def _get_dd_in_um(self):
        gc_df = self.gc_df
        # Add diameters to dataframe
        if self.context.my_retina["DoG_model"] == "circular":
            den_diam_um_s = gc_df["rad_c_mm"] * 2 * 1000
        elif self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            den_diam_um_s = pd.Series(
                self.ellipse2diam(
                    gc_df["semi_xc_mm"].values * 1000,
                    gc_df["semi_yc_mm"].values * 1000,
                )
            )

        self.gc_df["den_diam_um"] = den_diam_um_s

    def _update_gc_vae_df(
        self, gc_vae_df_in, um_per_pix, sidelen_pix, updated_rf_lu_pix, ret_lu_mm
    ):
        """
        Update gc_vae_df to have the same columns as gc_df with corresponding values.
        Update the remaining pixel values to mm, unless unit in is in the column name
        """
        gc_vae_df = gc_vae_df_in.reindex(columns=self.gc_vae_df.columns)
        # Calculate the eccentricity and polar angle of the receptive field center from the updated_rf_lu_pix
        # and ret_lu_mm
        xoc_mm = gc_vae_df_in.xoc_pix * um_per_pix / 1000
        yoc_mm = gc_vae_df_in.yoc_pix * um_per_pix / 1000
        rf_lu_mm = updated_rf_lu_pix * um_per_pix / 1000

        x_mm = ret_lu_mm[0] + rf_lu_mm[:, 0] + xoc_mm
        y_mm = ret_lu_mm[1] - rf_lu_mm[:, 1] - yoc_mm
        (pos_ecc_mm, pos_polar_deg) = self.cart2pol(x_mm, y_mm)

        gc_vae_df["pos_ecc_mm"] = pos_ecc_mm
        gc_vae_df["pos_polar_deg"] = pos_polar_deg
        # These values come from _place_gc_units before _create_spatial_rfs in build()
        # They are independent from FIT.
        gc_vae_df["ecc_group_idx"] = self.gc_vae_df["ecc_group_idx"]
        gc_vae_df["gc_scaling_factors"] = self.gc_vae_df["gc_scaling_factors"]

        # Save this metadata to df, although it is the same for all units
        gc_vae_df["um_per_pix"] = um_per_pix
        gc_vae_df["sidelen_pix"] = sidelen_pix

        if self.context.my_retina["DoG_model"] == "ellipse_fixed":
            gc_vae_df["relat_sur_diam"] = gc_vae_df_in["relat_sur_diam"]

        if self.context.my_retina["DoG_model"] == "ellipse_independent":
            # Scale factor for semi_x and semi_y from pix to millimeters
            gc_vae_df["semi_xs_mm"] = (
                gc_vae_df_in["semi_xs_pix"] * um_per_pix / 1000
            )  # mm
            gc_vae_df["semi_ys_mm"] = (
                gc_vae_df_in["semi_ys_pix"] * um_per_pix / 1000
            )  # mm
            gc_vae_df["orient_sur_rad"] = gc_vae_df_in["orient_sur_rad"]
            gc_vae_df["xos_pix"] = gc_vae_df_in["xos_pix"]
            gc_vae_df["yos_pix"] = gc_vae_df_in["yos_pix"]

        if self.context.my_retina["DoG_model"] == "circular":
            # Scale factor for rad_c and rad_s from pix to millimeters
            gc_vae_df["rad_c_mm"] = gc_vae_df_in["rad_c_pix"] * um_per_pix / 1000  # mm
            gc_vae_df["rad_s_mm"] = gc_vae_df_in["rad_s_pix"] * um_per_pix / 1000  # mm
            # dendritic diameter in micrometers
            gc_vae_df["den_diam_um"] = gc_vae_df["rad_c_mm"] * 2 * 1000  # um

        elif self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            # Scale factor for semi_x and semi_y from pix to millimeters
            gc_vae_df["semi_xc_mm"] = (
                gc_vae_df_in["semi_xc_pix"] * um_per_pix / 1000
            )  # mm
            gc_vae_df["semi_yc_mm"] = (
                gc_vae_df_in["semi_yc_pix"] * um_per_pix / 1000
            )  # mm
            # dendritic diameter in micrometers
            gc_vae_df["den_diam_um"] = self.ellipse2diam(
                gc_vae_df["semi_xc_mm"].values * 1000,
                gc_vae_df["semi_yc_mm"].values * 1000,
            )

            gc_vae_df["orient_cen_rad"] = gc_vae_df_in["orient_cen_rad"]

            gc_vae_df["xy_aspect_ratio"] = (
                gc_vae_df["semi_yc_mm"] / gc_vae_df["semi_xc_mm"]
            )

        gc_vae_df["ampl_c"] = gc_vae_df_in["ampl_c"]
        gc_vae_df["ampl_s"] = gc_vae_df_in["ampl_s"]

        gc_vae_df["xoc_pix"] = gc_vae_df_in["xoc_pix"]
        gc_vae_df["yoc_pix"] = gc_vae_df_in["yoc_pix"]

        return gc_vae_df

    def _get_full_retina_with_rf_images(
        self,
        rf_img,
        um_per_pix,
    ):
        """
        Build one retina image with all receptive fields. The retina sector is first rotated to
        be symmetric around the horizontal meridian. Then the image is cropped to the smallest
        rectangle that contains all receptive fields. The image is then rotated back to the original
        orientation.

        Parameters
        ----------
        rf_img : numpy.ndarray
            3D numpy array of receptive field images. The shape of the array should be (N, H, W).
        df : pandas.DataFrame
            DataFrame with gc parameters.
        um_per_pix : float
            The number of micrometers per pixel in the rf_img.
        """

        ecc_lim_mm = self.ecc_lim_mm
        polar_lim_deg = self.polar_lim_deg

        # First we need to get rotation angle of the mean meridian in degrees.
        rot_deg = np.mean(polar_lim_deg)

        # Find corner coordinates of the retina image as [left upper, right_upper, left_lower, right lower]
        # Sector is now symmetrically around the horizontal meridian
        corners_mm = np.zeros((4, 2))
        corners_mm[0, :] = self.pol2cart(ecc_lim_mm[0], polar_lim_deg[1] - rot_deg)
        corners_mm[1, :] = self.pol2cart(ecc_lim_mm[0], polar_lim_deg[0] - rot_deg)
        corners_mm[2, :] = self.pol2cart(ecc_lim_mm[1], polar_lim_deg[0] - rot_deg)
        corners_mm[3, :] = self.pol2cart(ecc_lim_mm[1], polar_lim_deg[1] - rot_deg)

        self.corners_mm = corners_mm

        # Get the max extent for rectangular image
        min_x_mm = np.min(corners_mm[:, 0])
        max_x_mm = np.max(corners_mm[:, 0])
        min_y_mm = np.min(corners_mm[:, 1])
        max_y_mm = np.max(corners_mm[:, 1])

        # Check for max hor extent
        if np.max(ecc_lim_mm) > max_x_mm:
            max_x_mm = np.max(ecc_lim_mm)

        # TODO: implement rotation

        # # Convert the rotation angle from degrees to radians
        # theta_rad = np.radians(rot_deg)

        # # Find the max and min extents in rotated coordinates
        # max_x_mm_rot = np.max(
        #     corners_mm[:, 0] * np.cos(theta_rad)
        #     - corners_mm[:, 1] * np.sin(theta_rad)
        # )
        # min_x_mm_rot = np.min(
        #     corners_mm[:, 0] * np.cos(theta_rad)
        #     - corners_mm[:, 1] * np.sin(theta_rad)
        # )
        # max_y_mm_rot = np.max(
        #     corners_mm[:, 0] * np.sin(theta_rad)
        #     + corners_mm[:, 1] * np.cos(theta_rad)
        # )
        # min_y_mm_rot = np.min(
        #     corners_mm[:, 0] * np.sin(theta_rad)
        #     + corners_mm[:, 1] * np.cos(theta_rad)
        # )

        # # Rotate back to original coordinates to get max and min extents
        # max_x_mm = max_x_mm_rot * np.cos(theta_rad) + max_y_mm_rot * np.sin(theta_rad)
        # min_x_mm = min_x_mm_rot * np.cos(theta_rad) + min_y_mm_rot * np.sin(theta_rad)
        # max_y_mm = max_y_mm_rot * np.cos(theta_rad) - max_x_mm_rot * np.sin(theta_rad)
        # min_y_mm = min_y_mm_rot * np.cos(theta_rad) - min_x_mm_rot * np.sin(theta_rad)

        # Pad with one full rf in each side. This prevents need to cutting the
        # rf imgs at the borders later on
        pad_size_x_mm = rf_img.shape[2] * um_per_pix / 1000
        pad_size_y_mm = rf_img.shape[1] * um_per_pix / 1000

        min_x_mm_im = min_x_mm - pad_size_x_mm
        max_x_mm_im = max_x_mm + pad_size_x_mm
        min_y_mm_im = min_y_mm - pad_size_y_mm
        max_y_mm_im = max_y_mm + pad_size_y_mm

        # Get retina image size in pixels
        ret_pix_x = int(np.ceil((max_x_mm_im - min_x_mm_im) * 1000 / um_per_pix))
        ret_pix_y = int(np.ceil((max_y_mm_im - min_y_mm_im) * 1000 / um_per_pix))

        ret_img_pix = np.zeros((ret_pix_y, ret_pix_x))

        # Prepare numpy nd array to hold left upeer corner pixel coordinates for each rf image
        rf_lu_pix = np.zeros((rf_img.shape[0], 2), dtype=int)

        pos_ecc_mm = self.gc_vae_df["pos_ecc_mm"].values
        pos_polar_deg = self.gc_vae_df["pos_polar_deg"].values

        # Locate left upper corner of each rf img and lay images onto retina image
        x_mm, y_mm = self.pol2cart(
            pos_ecc_mm.astype(np.float64),
            pos_polar_deg.astype(np.float64) - rot_deg,
            deg=True,
        )
        y_pix_c = (np.round((max_y_mm_im - y_mm) * 1000 / um_per_pix)).astype(np.int64)
        x_pix_c = (np.round((x_mm - min_x_mm_im) * 1000 / um_per_pix)).astype(np.int64)
        for i, row in self.gc_vae_df.iterrows():
            # Get the position of the rf upper left corner in pixels
            # The xoc and yoc are the center of the rf image in the resampled data scale.
            # TÄHÄN JÄIT: JOSTAIN SYYSTÄ MIDGET SOLULLA 295 Y JÄÄ NEGATIIVISEKSI.
            y_pix_lu = y_pix_c[i] - int(row.yoc_pix)
            x_pix_lu = x_pix_c[i] - int(row.xoc_pix)

            # Get the rf image
            this_rf_img = rf_img[i, :, :]
            # Lay the rf image onto the retina image
            ret_img_pix[
                y_pix_lu : y_pix_lu + this_rf_img.shape[0],
                x_pix_lu : x_pix_lu + this_rf_img.shape[1],
            ] += this_rf_img
            # Store the left upper corner pixel coordinates and width and height of each rf image.
            # The width and height are necessary because some are cut off at the edges of the retina image.
            rf_lu_pix[i, :] = [x_pix_lu, y_pix_lu]

        return ret_img_pix, rf_lu_pix, (min_x_mm_im, max_y_mm_im)

    def _get_vae_rfs_with_good_fits(self, retina_vae, new_sidelen, um_per_pix):
        """
        Provides eccentricity-scaled spatial receptive fields from the
        VAE model, with good DoG fits as QA.

        In the main loop "Bad fit loop"
        1) rescale according to eccentricity and resample to final rf size
        2) fit DoG model.
        3) replace outlier (bad) rfs DoG fits
        4) repeat until all fits are good

        Parameters
        ----------
        retina_vae : RetinaVAE
            Variational autoencoder model for creating spatial receptive fields.

        Returns
        -------
        img_processed : np.ndarray
            Processed image of the generated RFs.
        img_raw : np.ndarray
            Raw image of the generated RFs.
        gc_vae_df: pd.DataFrame
            Dataframe containing the FITs for accepted rfs.

        Notes
        -----
        The VAE generates a number of RFs that is larger than nsamples.
        This is to account for outliers that are not accepted.
        """

        nsamples = self.n_units

        # Get samples. We take 50% extra samples to cover the bad fits
        nsamples_extra = int(nsamples * 1.5)  # 50% extra to account for outliers
        img_processed_extra, img_raw_extra = self._get_generated_rfs(
            retina_vae, n_samples=nsamples_extra
        )

        idx_to_process = np.arange(nsamples)
        img_rfs = np.zeros((nsamples, new_sidelen, new_sidelen))
        available_idx_mask = np.ones(nsamples_extra, dtype=bool)
        available_idx_mask[idx_to_process] = False
        img_to_resample = img_processed_extra[idx_to_process, :, :]
        good_mask_compiled = np.zeros(nsamples, dtype=bool)
        gc_vae_df_temp = pd.DataFrame(
            index=np.arange(nsamples),
            columns=["xoc_pix", "yoc_pix"],
        )

        # Loop until there is no bad fits
        for _ in range(100):
            # Upsample according to smallest rf diameter
            img_after_resample = self._get_resampled_scaled_rfs(
                img_to_resample[idx_to_process, :, :],
                new_sidelen,
            )

            # Fit elliptical gaussians to the img[idx_to_process]
            # This is dependent metrics, not affecting the spatial RFs
            # other than quality assurance (below)
            # Fixed DoG model type excludes the model effect on unit selection
            # Note that this fits the img_after_resample and thus the
            # xoc_pix and yoc_pix are veridical for the upsampled data.
            self.fit.initialize(
                self.gc_type,
                self.response_type,
                fit_type="generated",
                DoG_model="ellipse_fixed",
                spatial_data=img_after_resample,
                um_per_pix=um_per_pix,
                mark_outliers_bad=True,  # False to bypass bad fit check
            )

            # 6) Discard bad fits
            good_idx_this_iter = self.fit.good_idx_generated
            good_idx_generated = idx_to_process[good_idx_this_iter]
            # save the good rfs
            img_rfs[good_idx_generated, :, :] = img_after_resample[
                good_idx_this_iter, :, :
            ]

            good_df = self.fit.all_data_fits_df.loc[good_idx_this_iter, :]
            gc_vae_df_temp.loc[
                good_idx_generated, ["yoc_pix", "xoc_pix"]
            ] = good_df.loc[:, ["yoc_pix", "xoc_pix"]].values

            good_mask_compiled[good_idx_generated] = True

            # 7) Update idx_to_process for the loop
            idx_to_process = np.setdiff1d(idx_to_process, good_idx_generated)
            print(f"bad fits to replace by new RF:s: {idx_to_process}")

            if len(idx_to_process) > 0:
                for this_miss in idx_to_process:
                    # Get next possible replacement index
                    this_replace = np.where(available_idx_mask)[0][0]
                    # Replace the bad rf with the reserve rf
                    img_to_resample[this_miss, :, :] = img_processed_extra[
                        this_replace, :, :
                    ]
                    # remove replacement from available indices
                    available_idx_mask[this_replace] = False
            else:
                break

        # For visualization of the construction process early steps
        good_idx_compiled = np.where(good_mask_compiled)[0]
        self.project_data.construct_retina["gen_spat_img"] = {
            "img_processed": img_processed_extra[good_idx_compiled, :, :],
            "img_raw": img_raw_extra[good_idx_compiled, :, :],
        }

        self.gc_vae_df.loc[:, ["xoc_pix", "yoc_pix"]] = gc_vae_df_temp

        return img_rfs

    def _apply_rf_repulsion(
        self, img_ret_shape, img_rfs, img_rfs_mask, rf_lu_pix, um_per_pix
    ):
        """
        Apply mutual repulsion to receptive fields (RFs) to ensure optimal coverage of a simulated retina.
        It involves multiple iterations to gradually move the RFs until they cover the retina
        with minimal overlapping, considering boundary effects and force gradients.

        Parameters:
        -----------
        img_ret_shape : tuple
            Shape of the retina image (height, width) in pixels.
        img_rfs : numpy.ndarray
            3D array representing the RFs, shape (n_rfs, n_pixels, n_pixels).
        img_rfs_mask : numpy.ndarray
            3D array of boolean masks for RF centers, shape (n_rfs, n_pixels, n_pixels).
        rf_lu_pix : numpy.ndarray
            2D array of the upper-left pixel coordinates of each RF, shape (n_rfs, 2).
        um_per_pix : float
            Scale factor representing micrometers per pixel in `img_rfs`.

        Returns:
        --------
        updated_img_rfs: numpy.ndarray
            The updated RFs after repulsion and transformation, shape (n_rfs, n_pixels, n_pixels).
        updated_rf_lu_pix: numpy.ndarray
            Updated upper-left pixel coordinates of each RF, shape (n_rfs, 2).
        final_retina: numpy.ndarray
            2D array representing the final state of the retina after RF adjustments, shape matching `img_ret_shape`.

        Notes:
        ------
        The method internally uses parameters from `rf_repulsion_params` in `self.context.my_retina`.
        These parameters control aspects of the repulsion process like the rate of change, number of iterations,
        and visualization options.
        """

        rf_repulsion_params = self.context.my_retina["rf_repulsion_params"]
        show_repulsion_progress = rf_repulsion_params["show_repulsion_progress"]
        change_rate = rf_repulsion_params["change_rate"]
        n_iterations = rf_repulsion_params["n_iterations"]
        show_skip_steps = rf_repulsion_params["show_skip_steps"]
        border_repulsion_stength = rf_repulsion_params["border_repulsion_stength"]
        cooling_rate = rf_repulsion_params["cooling_rate"]
        show_only_unit = rf_repulsion_params["show_only_unit"]

        n_units, H, W = img_rfs.shape
        assert H == W, "RF must be square, aborting..."

        if show_repulsion_progress is True:
            # Init plotting
            fig_args = self.viz.show_repulsion_progress(
                np.zeros(img_ret_shape),
                np.zeros(img_ret_shape),
                init=True,
                um_per_pix=um_per_pix,
                sidelen=H,
            )

        rf_positions = np.array(rf_lu_pix, dtype=float)
        rfs = np.array(img_rfs, dtype=float)
        rfs_mask = np.array(img_rfs_mask, dtype=bool)
        masked_rfs = rfs * rfs_mask
        sum_masked_rfs = np.sum(masked_rfs, axis=(1, 2))

        # Compute boundary effect
        boundary_polygon = self.viz.boundary_polygon(
            self.ecc_lim_mm,
            self.polar_lim_deg,
            um_per_pix=um_per_pix,
            sidelen=H,
        )
        boundary_polygon_path = mplPath.Path(boundary_polygon)  # x, y
        Y, X = np.meshgrid(
            np.arange(img_ret_shape[0]),
            np.arange(img_ret_shape[1]),
            indexing="ij",
        )  # y, x
        boundary_points = np.vstack((X.flatten(), Y.flatten())).T  # x,y
        inside_boundary = boundary_polygon_path.contains_points(boundary_points)
        boundary_mask = inside_boundary.reshape(img_ret_shape)  # x, y
        retina_boundary_effect = np.where(boundary_mask, 0, border_repulsion_stength)

        # Rigid body matrix
        Mrb_pre = np.tile(np.eye(3), (n_units, 1, 1))
        Mrb_pre[:, :2, 2] = rf_positions

        Y0, X0 = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        homogeneous_coords = np.stack(
            [X0.flatten(), Y0.flatten(), np.ones(H * W)], axis=0
        )

        force_y = np.empty_like(rfs)
        force_x = np.empty_like(rfs)
        new_coords = Mrb_pre @ homogeneous_coords

        # Main optimization loop
        for iteration in range(n_iterations):
            # Get the new coordinates of the RFs
            Xt = new_coords[:, 0, ...].round().reshape(n_units, H, W).astype(int)
            Yt = new_coords[:, 1, ...].round().reshape(n_units, H, W).astype(int)

            retina = np.zeros(img_ret_shape)

            for i in range(n_units):
                idx = np.where(rfs[i, ...] == np.max(rfs[i], axis=(0, 1)))  # y,x
                pos = np.stack((Xt[i, idx[1], idx[0]], Yt[i, idx[1], idx[0]]), axis=1)
                inside_boundary = boundary_polygon_path.contains_points(pos)  # x, y
                if inside_boundary:
                    retina[Yt[i], Xt[i]] += rfs[i]
                else:
                    inside = np.where(boundary_mask)  # y,x
                    choise = np.random.choice(len(inside[0]))
                    # Subtract center location within RF, shape n,y,x
                    y_start = int(inside[0][choise] - idx[0])
                    y_end = int(y_start + H)
                    x_start = int(inside[1][choise] - idx[1])
                    x_end = int(x_start + W)
                    Y1, X1 = np.meshgrid(
                        np.arange(y_start, y_end),
                        np.arange(x_start, x_end),
                        indexing="ij",
                    )
                    retina[y_start:y_end, x_start:x_end] += rfs[i]
                    Yt[i, ...] = Y1
                    Xt[i, ...] = X1
                    Mrb_pre[i, :2, 2] = [x_start, y_start]

            if iteration == 0:
                reference_retina = retina.copy()

            retina_viz = retina.copy()
            retina += retina_boundary_effect

            grad_y, grad_x = np.gradient(retina)

            for i in range(n_units):
                # Force goes downhill the gradient, thus -1
                force_y[i] = -1 * grad_y[Yt[i], Xt[i]] * rfs[i] * rfs_mask[i]
                force_x[i] = -1 * grad_x[Yt[i], Xt[i]] * rfs[i] * rfs_mask[i]

            # Centre of mass of the RF centre (masked) for all rfs in H, W coordinates
            com_y = np.sum(masked_rfs * Yt, axis=(1, 2)) / sum_masked_rfs
            com_x = np.sum(masked_rfs * Xt, axis=(1, 2)) / sum_masked_rfs

            if show_repulsion_progress is True and show_only_unit is not None:
                unit_retina = np.zeros(img_ret_shape)
                unit_idx = show_only_unit
                fig_args["additional_points"] = [com_x[unit_idx], com_y[unit_idx]]
                fig_args["unit_idx"] = unit_idx
                unit_img = np.ones(masked_rfs[unit_idx].shape) * 0.1
                unit_img += masked_rfs[unit_idx, ...]
                unit_retina[Yt[unit_idx], Xt[unit_idx]] += unit_img
                retina_viz = unit_retina.copy()

            com_y_mtx = np.tile(com_y, (H, W, 1)).transpose(2, 0, 1)
            com_x_mtx = np.tile(com_x, (H, W, 1)).transpose(2, 0, 1)

            radius_vec = np.stack([Yt - com_y_mtx, Xt - com_x_mtx], axis=-1)

            # Torque is computed as forces_y * radii_x - forces_x * radii_y
            torques = force_y * radius_vec[..., 1] - force_x * radius_vec[..., 0]
            net_torque = np.sum(torques, axis=(1, 2))
            net_force_y = np.sum(force_y, axis=(1, 2))
            net_force_x = np.sum(force_x, axis=(1, 2))

            # Normalize the net torque and net_forces to about matching effects
            net_torque = np.pi * net_torque / np.max(np.abs(net_torque))
            # With 0.01 change rate, 50 should result in 0.5 pix / iteration
            net_force_y = 50 * net_force_y / np.max(np.abs(net_force_y))
            net_force_x = 50 * net_force_x / np.max(np.abs(net_force_x))

            # Loop based update
            rot = change_rate * net_torque
            tr_x = change_rate * net_force_x
            tr_y = change_rate * net_force_y

            # Compute the change in rotation and translation
            rot_mtx = np.array(
                [
                    [np.cos(rot), -np.sin(rot), np.zeros(n_units)],
                    [np.sin(rot), np.cos(rot), np.zeros(n_units)],
                    [np.zeros(n_units), np.zeros(n_units), np.ones(n_units)],
                ]
            ).transpose(2, 0, 1)
            trans_mtx = np.array(
                [
                    [np.ones(n_units), np.zeros(n_units), tr_x],
                    [np.zeros(n_units), np.ones(n_units), tr_y],
                    [np.zeros(n_units), np.zeros(n_units), np.ones(n_units)],
                ]
            ).transpose(2, 0, 1)

            Mrb_change = trans_mtx @ rot_mtx
            Mrb = Mrb_pre @ Mrb_change
            Mrb_pre = Mrb
            new_coords = Mrb @ homogeneous_coords
            change_rate = change_rate * cooling_rate

            if show_repulsion_progress is True:
                center_mask = np.zeros(img_ret_shape)
                for i in range(n_units):
                    center_mask[Yt[i], Xt[i]] += rfs_mask[i]

                if iteration % show_skip_steps == 0:
                    self.viz.show_repulsion_progress(
                        reference_retina,
                        center_mask,
                        new_retina=retina_viz,
                        iteration=iteration,
                        um_per_pix=um_per_pix,
                        sidelen=H,
                        **fig_args,
                    )

        # Resample to rectangular H, W resolution
        updated_img_rfs = np.zeros((n_units, H, W))
        Yout = np.zeros((n_units, H, W), dtype=np.int32)
        Xout = np.zeros((n_units, H, W), dtype=np.int32)

        for i in range(n_units):
            y_top = np.round(com_y[i] - H / 2)
            x_left = np.round(com_x[i] - W / 2)
            y_out = np.arange(y_top, y_top + H).round().astype(np.int32)
            x_out = np.arange(x_left, x_left + W).round().astype(np.int32)
            y_out_grid, x_out_grid = np.meshgrid(y_out, x_out, indexing="ij")
            Yout[i] = y_out_grid
            Xout[i] = x_out_grid

            # Flatten the coordinates and image
            points = np.array([Yt[i].ravel(), Xt[i].ravel()]).T
            values = rfs[i].ravel()
            new_points = np.array([y_out_grid.ravel(), x_out_grid.ravel()]).T

            # Interpolate using griddata
            resampled_values = griddata(
                points, values, new_points, method="cubic", fill_value=0
            )

            # Reshape to 2D
            updated_img_rfs[i, ...] = resampled_values.reshape(H, W)

        updated_rf_lu_pix = np.array(
            [Xout[:, 0, 0], Yout[:, 0, 0]], dtype=np.int32
        ).T  # x, y
        com_x_local = com_x - updated_rf_lu_pix[:, 0]
        com_y_local = com_y - updated_rf_lu_pix[:, 1]

        new_retina = np.zeros(img_ret_shape)
        final_retina = np.zeros(img_ret_shape)
        for i in range(n_units):
            new_retina[Yt[i], Xt[i]] += rfs[i]
            final_retina[Yout[i], Xout[i]] += updated_img_rfs[i]

        if show_repulsion_progress is True:
            # Show one last time with the final interpolated result
            self.viz.show_repulsion_progress(
                reference_retina,
                center_mask,
                new_retina=final_retina,
                iteration=iteration,
                um_per_pix=um_per_pix,
                sidelen=H,
                **fig_args,
            )
            plt.ioff()  # Turn off interactive mode

        return (
            updated_img_rfs,
            updated_rf_lu_pix,
            final_retina,
            com_x_local,
            com_y_local,
        )

    def _create_spatial_rfs(self):
        """
        Generation of spatial receptive fields (RFs) for the retinal ganglion cells (RGCs).

        RF become resampled, and the resolution will change if
        eccentricity is different from eccentricity of the original data.
        """

        # Get fit parameters for dendritic field diameter (dd) with respect to eccentricity (ecc).
        # Data from literature (lit) Watanabe_1989_JCompNeurol, Perry_1984_Neurosci and Goodchild_1996_JCompNeurol
        lit_dd_vs_ecc_params = self._fit_dd_vs_ecc()

        # # Quality control: check that the fitted dendritic diameter is close to the original data
        # # Frechette_2005_JNeurophysiol datasets: 9.7 mm (45°); 9.0 mm (41°); 8.4 mm (38°)
        # # Estimate the orginal data eccentricity from the fit to full eccentricity range
        # exp_rad = self.exp_cen_radius_mm * 2 * 1000
        # self.visual_field_limit_for_dd_fit_mm = np.inf
        # dd_ecc_params_full = self._fit_dd_vs_ecc()
        # data_ecc_mm = self._get_ecc_from_dd(dd_ecc_params_full, dd_regr_model, exp_rad)
        # data_ecc_deg = data_ecc_mm * self.deg_per_mm  # 37.7 deg

        # Endow cells with spatial elliptical receptive fields.
        # Units become mm unless specified in column names.
        if self.spatial_model == "FIT":
            # self.gc_df is updated silently
            if self.rf_coverage_adjusted_to_1 == True:
                # Assumes that the dendritic field diameter is proportional to the coverage
                self._generate_DoG_with_rf_coverage_one()
            elif self.rf_coverage_adjusted_to_1 == False:
                # Read the dendritic field diameter from literature data
                self._generate_DoG_with_rf_from_literature(lit_dd_vs_ecc_params)

            # Add dendritic diameter to self.gc_df for visualization, in micrometers
            self._get_dd_in_um()

        elif self.spatial_model == "VAE":
            # Endow cells with spatial receptive fields using the generative variational autoencoder model

            # 1) Get variational autoencoder to generate receptive fields
            print("\nGetting VAE model...")
            retina_vae = RetinaVAE(
                self.gc_type,
                self.response_type,
                self.training_mode,
                self.context,
                save_tuned_models=True,
            )

            # The methods below will silently use and update self.gc_vae_df
            self.gc_vae_df = self.gc_df.copy()

            # 2) Get resampling parameters.
            print("\nGetting resampling parameters...")
            (
                new_sidelen,
                new_um_per_pix,
            ) = self._get_rf_resampling_params(lit_dd_vs_ecc_params)

            # 3) "Bad fit loop", provides eccentricity-scaled vae rfs with good DoG fits (error < 3SD from mean).
            print("\nBad fit loop: Generating receptive fields with good DoG fits...")
            img_rfs = self._get_vae_rfs_with_good_fits(
                retina_vae, new_sidelen, new_um_per_pix
            )

            # 4) Get center masks
            mask_th = self.context.my_retina["center_mask_threshold"]
            img_rfs_mask = self.get_rf_masks(img_rfs, mask_threshold=mask_th)

            # 5) Sum separate rf images onto one retina pixel matrix.
            # In the retina pixel matrix, for each rf get the upper left corner
            # pixel coordinates and corresponding retinal mm coordinates
            ret_pix_mtx, rf_lu_pix, ret_lu_mm = self._get_full_retina_with_rf_images(
                img_rfs,
                new_um_per_pix,
            )

            # 6) Apply repulsion adjustment to the receptive fields
            print("\nApplying repulsion between the receptive fields...")
            (
                img_rfs_final,
                updated_rf_lu_pix,
                ret_pix_mtx_final,
                com_x,
                com_y,
            ) = self._apply_rf_repulsion(
                ret_pix_mtx.shape,
                img_rfs,
                img_rfs_mask,
                rf_lu_pix,
                new_um_per_pix,
            )

            # 7) Redo the good fits for final statistics
            print("\nFinal DoG fit to generated rfs...")
            DoG_model = self.context.my_retina["DoG_model"]
            self.fit.initialize(
                self.gc_type,
                self.response_type,
                fit_type="generated",
                DoG_model=DoG_model,
                spatial_data=img_rfs_final,
                um_per_pix=new_um_per_pix,
                mark_outliers_bad=False,
            )

            (
                self.gen_stat_df,
                self.gen_spat_cen_sd,
                self.gen_spat_sur_sd,
                gc_vae_df,
                _,
            ) = self.fit.get_generated_spatial_fits(DoG_model)

            # 8) Update gc_vae_df to include new positions and DoG fits after repulsion
            # and convert units to to mm, where applicable
            print("\nUpdating ganglion cell dataframe...")
            self.gc_vae_df = self._update_gc_vae_df(
                gc_vae_df, new_um_per_pix, new_sidelen, updated_rf_lu_pix, ret_lu_mm
            )

            # 9) Get final center masks for the generated spatial rfs
            print("\nGetting final masked rfs and retina...")
            img_rfs_final_mask = self.get_rf_masks(
                img_rfs_final, mask_threshold=mask_th
            )

            # 10) Sum separate rf center masks onto one retina pixel matrix.
            ret_pix_mtx_final_masked, _, _ = self._get_full_retina_with_rf_images(
                img_rfs_final_mask,
                new_um_per_pix,
            )

            # 11) Save the generated receptive fields and masks
            print("\nSaving data...")
            output_path = self.context.output_folder
            filename_stem = self.context.my_retina["spatial_rfs_file"]

            self.data_io.save_generated_rfs(
                img_rfs_final, output_path, filename_stem=filename_stem
            )

            # Save original and new df:s. For vae, gc_df contains the original
            # positions which were updated during the repulsion step
            self.gc_df_original = self.gc_df.copy()
            self.gc_df = self.gc_vae_df

            # 12) Set project_data for later visualization
            self.project_data.construct_retina["retina_vae"] = retina_vae

            self.project_data.construct_retina["gen_rfs"] = {
                "img_rf": img_rfs,
                "img_rf_mask": img_rfs_mask,
                "img_rfs_adjusted": img_rfs_final,
                "centre_of_mass_x": com_x,
                "centre_of_mass_y": com_y,
            }

            self.project_data.construct_retina["gen_ret"] = {
                "img_ret": ret_pix_mtx,
                "img_ret_masked": ret_pix_mtx_final_masked,
                "img_ret_adjusted": ret_pix_mtx_final,
            }

        # Scale center and surround amplitude: center Gaussian volume in pixel space becomes one
        # Surround amplitude is scaled relative to center volume of one
        self.gc_df = self._scale_both_amplitudes(self.gc_df)

        # Set more project_data for later visualization
        self.project_data.construct_retina["dd_vs_ecc"][
            "dd_DoG_x"
        ] = self.gc_df.pos_ecc_mm.values
        self.project_data.construct_retina["dd_vs_ecc"][
            "dd_DoG_y"
        ] = self.gc_df.den_diam_um.values

        ### Generation of spatial receptive fields ends here ###
        #########################################################

    def build(self):
        """
        Builds the receptive field mosaic. This is the main method to call.
        """

        if self.initialized is False:
            self._initialize()

        # -- First, place the ganglion cell midpoints (units mm)
        # Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
        gc_density_func_params = self._fit_gc_density_data()

        # Place ganglion cells to desired retina.
        self._place_gc_units(gc_density_func_params)

        # Now that we have placed the units, we know their number
        self.n_units = len(self.gc_df)

        # -- Second, endow cells with spatial receptive fields
        self._create_spatial_rfs()

        # -- Third, endow cells with temporal receptive fields
        self._create_fixed_temporal_rfs()  # Chichilnisky data
        self._create_dynamic_temporal_rfs()  # Benardete & Kaplan data

        # -- Fourth, endow cells with tonic drive
        self._create_tonic_drive()

        print(f"Built RGC mosaic with {self.n_units} cells")

        # Save the receptive field mosaic
        self.save_gc_csv()

        # Save the project data
        # Attach data requested by other classes to project_data
        self.project_data.construct_retina["gc_df"] = self.gc_df

    def get_data_at_latent_space(self, retina_vae):
        """
        Get original image data as projected through encoder to the latent space
        """
        # Get the latent space data
        train_df = retina_vae.get_encoded_samples(
            dataset=retina_vae.train_loader.dataset
        )
        valid_df = retina_vae.get_encoded_samples(dataset=retina_vae.val_loader.dataset)
        test_df = retina_vae.get_encoded_samples(dataset=retina_vae.test_loader.dataset)
        latent_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)

        # Extract data from latent_df into a numpy array from columns whose title include "EncVariable"
        latent_data = latent_df.filter(regex="EncVariable").to_numpy()

        return latent_data

    def save_gc_csv(self, filename=None):
        """
        Save the mosaic to a csv file

        Parameters
        ----------
        filename : pathlib Path object, str or None
            If None, the default filename is used.
        """
        output_folder = self.context.output_folder

        # Create output folder if it does not exist, with parents
        if not output_folder.exists():
            Path.mkdir(output_folder, mode=0o771, parents=True, exist_ok=False)

        if filename is None:
            filepath = output_folder.joinpath(self.context.my_retina["mosaic_file"])
        else:
            filepath = output_folder.joinpath(filename)

        print("Saving model mosaic to %s" % filepath)
        self.gc_df.to_csv(filepath)
