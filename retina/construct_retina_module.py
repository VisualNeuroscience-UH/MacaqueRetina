# Numerical
import numpy as np
import numpy.ma as ma
import scipy.optimize as opt

# import scipy.io as sio
import scipy.stats as stats
from scipy.optimize import root
import pandas as pd
from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d

import torch

# import torch.nn.functional as F
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
    randomize_position : bool
        Whether to randomize the position of the ganglion cells
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
        ecc_limits = my_retina["ecc_limits"]
        visual_field_limit_for_dd_fit = my_retina["visual_field_limit_for_dd_fit"]
        sector_limits = my_retina["sector_limits"]
        model_density = my_retina["model_density"]
        self.rf_coverage_adjusted_to_1 = my_retina["rf_coverage_adjusted_to_1"]
        self.dd_regr_model = my_retina["dd_regr_model"]
        randomize_position = my_retina["randomize_position"]
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
            isinstance(ecc_limits, list) and len(ecc_limits) == 2
        ), "Wrong type or length of eccentricity, aborting"
        assert (
            isinstance(sector_limits, list) and len(sector_limits) == 2
        ), "Wrong type or length of sector_limits, aborting"
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

        self.eccentricity = ecc_limits
        self.ecc_lim_mm = np.asarray(
            [r / self.deg_per_mm for r in ecc_limits]
        )  # Turn list to numpy array and deg to mm
        self.visual_field_limit_for_dd_fit_mm = (
            visual_field_limit_for_dd_fit / self.deg_per_mm
        )
        self.polar_lim_deg = np.asarray(sector_limits)  # Turn list to numpy array
        self.randomize_position = randomize_position

        # Make or read fits
        self.fit.initialize(
            gc_type,
            response_type,
            fit_type="experimental",
            DoG_model=self.context.my_retina["DoG_model"],
        )
        (
            self.exp_stat_df,
            self.exp_cen_radius_mm,
            self.exp_sur_radius_mm,
            self.spat_DoG_fit_params,
        ) = self.fit.get_experimental_fits()

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
        Read re-digitized old literature data from mat files
        """

        print(
            "Reading density data from:",
            self.context.literature_data_files["gc_density_fullpath"],
        )
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

    def _fit_dd_vs_ecc(self, visual_field_limit_for_dd_fit_mm, dd_regr_model):
        """
        Fit dendritic field diameter with respect to eccentricity. Linear, quadratic and cubic fit.

        Parameters
        ----------
        self : object
            an instance of the class that this method belongs to

        Returns
        -------
        dict
            dictionary containing dendritic diameter parameters and related data for visualization
        """

        # Read dendritic field data and return linear fit with scipy.stats.linregress
        dendr_diam_parameters = {}

        dendr_diam1 = self.data_io.get_data(
            self.context.literature_data_files["dendr_diam1_fullpath"]
        )
        dendr_diam2 = self.data_io.get_data(
            self.context.literature_data_files["dendr_diam2_fullpath"]
        )

        # Quality control. Datasets separately for visualization
        data_set_1_x = np.squeeze(dendr_diam1["Xdata"])
        data_set_1_y = np.squeeze(dendr_diam1["Ydata"])
        data_set_2_x = np.squeeze(dendr_diam2["Xdata"])
        data_set_2_y = np.squeeze(dendr_diam2["Ydata"])

        # Both datasets together
        data_all_x = np.concatenate((data_set_1_x, data_set_2_x))
        data_all_y = np.concatenate((data_set_1_y, data_set_2_y))

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
        else:
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
            return root(equation, 1).x[0]

    def _fit_DoG_with_rf_coverage_one(self):
        """
        Create spatial receptive fields to model cells using coverage = 1.

        Places all ganglion cell spatial parameters to ganglion cell object dataframe self.gc_df

        At return, all units are in mm unless stated otherwise in the the column name
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

        # Change units to mm. Units are mm for the Chichilnisky data and they are at large eccentricity
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            self.gc_df["semi_xc"] = self.gc_df["semi_xc"] * data_microm_per_pix / 1000
            self.gc_df["semi_yc"] = self.gc_df["semi_yc"] * data_microm_per_pix / 1000

        elif self.context.my_retina["DoG_model"] == "circular":
            self.gc_df["rad_c"] = self.gc_df["rad_c"] * data_microm_per_pix / 1000

        # Calculate RF diameter scaling factor for all ganglion cells
        # Area of RF = gc_scaling_factors * Random_factor * Area of ellipse(semi_xc,semi_yc), solve gc_scaling_factors.
        # Units are pixels for the Chichilnisky data. We scale them to um2 at the actual eccentricity.
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            area_rfs_cen_mm2 = np.pi * self.gc_df["semi_xc"] * self.gc_df["semi_yc"]

        elif self.context.my_retina["DoG_model"] == "circular":
            area_rfs_cen_mm2 = np.pi * self.gc_df["rad_c"] ** 2

        """
        The area_of_rf contains area for all model units. Its sum must fill the whole area (coverage factor = 1).
        We do it separately for each ecc sector, step by step, to keep coverage factor at 1 despite changing gc density with ecc
        """
        area_scaling_factors_coverage1 = np.zeros(area_rfs_cen_mm2.shape)
        for index, surface_area_mm2 in enumerate(self.sector_surface_areas_mm2):
            scaling_for_coverage_1 = (surface_area_mm2) / np.sum(
                area_rfs_cen_mm2[self.gc_df["ecc_group_idx"] == index]
            )

            area_scaling_factors_coverage1[
                self.gc_df["ecc_group_idx"] == index
            ] = scaling_for_coverage_1

        radius_scaling_coverage_1 = np.sqrt(area_scaling_factors_coverage1)

        # Apply scaling factors to semi_xc and semi_yc.  Units are pixels
        # scale_random_distribution = 0.08  # Estimated by eye from Watanabe and Perry data.
        # Normal distribution with scale_random_distribution 0.08 cover about 25% above and below the mean value
        scale_random_distribution = 0.08
        random_normal_distribution1 = 1 + np.random.normal(
            scale=scale_random_distribution, size=n_cells
        )
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            semi_xc = (
                radius_scaling_coverage_1
                * self.gc_df["semi_xc"]
                * random_normal_distribution1
            )
            random_normal_distribution2 = 1 + np.random.normal(
                scale=scale_random_distribution, size=n_cells
            )  # second randomization

            semi_yc = (
                radius_scaling_coverage_1
                * self.gc_df["semi_yc"]
                * random_normal_distribution2
            )

            self.gc_df["semi_xc"] = semi_xc
            self.gc_df["semi_yc"] = semi_yc

        elif self.context.my_retina["DoG_model"] == "circular":
            rad_c = (
                radius_scaling_coverage_1
                * self.gc_df["rad_c"]
                * random_normal_distribution1
            )
            self.gc_df["rad_c"] = rad_c

    def _fit_DoG_with_rf_from_literature(self, lit_dd_vs_ecc_params, dd_regr_model):
        """
        Create spatial receptive fields to model cells according to eccentricity.

        Places all ganglion cell spatial parameters to ganglion cell object dataframe self.gc_df

        At return, all units are in mm unless stated otherwise in the the column name
        """

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
        um_per_pixel = self.context.apricot_metadata["data_microm_per_pix"]
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            # Scale semi_x to pix at its actual eccentricity
            self.gc_df["semi_xc"] = self.gc_df["semi_xc"] * gc_scaling_factors

            # Scale semi_x to mm
            self.gc_df["semi_xc"] = self.gc_df["semi_xc"] * um_per_pixel / 1000

            # Scale semi_y to pix at its actual eccentricity
            self.gc_df["semi_yc"] = self.gc_df["semi_yc"] * gc_scaling_factors

            # Scale semi_y to mm
            self.gc_df["semi_yc"] = self.gc_df["semi_yc"] * um_per_pixel / 1000

        elif self.context.my_retina["DoG_model"] == "circular":
            # Scale rad_c to pix at its actual eccentricity
            self.gc_df["rad_c"] = self.gc_df["rad_c"] * gc_scaling_factors

            # Scale rad_c to mm
            self.gc_df["rad_c"] = self.gc_df["rad_c"] * um_per_pixel / 1000

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

        # pdb.set_trace()
        # gc_density = np.mean(density_prop_all)
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

    def _boundary_polygon(self, ecc_lim_mm, polar_lim_deg, n_points=100):
        """
        Create a boundary polygon based on given eccentricity and polar angle limits.

        Parameters
        ----------
        ecc_lim_mm : np.ndarray
            An array representing the eccentricity limits in millimeters for
            left and right boundaries (shape: [2]).
        polar_lim_deg : np.ndarray
            An array representing the polar angle limits in degrees for
            bottom and top boundaries (shape: [2]).
        n_points : int
            Number of points to generate along each arc.

        Returns
        -------
        boundary_polygon : np.ndarray
            Array of Cartesian coordinates forming the vertices of the boundary polygon.
        """

        # Generate points for bottom and top polar angle limits
        bottom_x, bottom_y = self.pol2cart(
            np.full(n_points, ecc_lim_mm[0]),
            np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points),
        )
        top_x, top_y = self.pol2cart(
            np.full(n_points, ecc_lim_mm[1]),
            np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points),
        )

        # Generate points along the arcs for min and max eccentricities
        theta_range = np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points)
        min_ecc_x, min_ecc_y = self.pol2cart(
            np.full_like(theta_range, ecc_lim_mm[0]), theta_range
        )
        max_ecc_x, max_ecc_y = self.pol2cart(
            np.full_like(theta_range, ecc_lim_mm[1]), theta_range
        )

        # Combine them to form the vertices of the bounding polygon
        boundary_polygon = []

        # Add points from bottom arc
        for bx, by in zip(min_ecc_x, min_ecc_y):
            boundary_polygon.append((bx, by))

        # Add points from top arc (in reverse order)
        for tx, ty in reversed(list(zip(max_ecc_x, max_ecc_y))):
            boundary_polygon.append((tx, ty))

        return np.array(boundary_polygon)

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

    def _apply_force_based_layout(
        self,
        all_positions,
        gc_density,
    ):
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
        boundary_polygon = self._boundary_polygon(ecc_lim_mm, polar_lim_deg)
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
        optim_algorithm = self.context.my_retina["gc_placement_params"]["alorithm"]
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
        Scale center amplitude in the fitted pixel space to center volume of one.
        Scale surround amplitude also to center volume of one.
        Volume of 2D Gaussian = 2 * pi * sigma_x*sigma_y

        Second step of scaling is done before convolving with the stimulus.
        """
        if self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            cen_vol_pix3 = 2 * np.pi * df["semi_xc"] * df["semi_yc"]
        elif self.context.my_retina["DoG_model"] == "circular":
            cen_vol_pix3 = np.pi * df["rad_c"] ** 2

        df["relat_sur_ampl"] = df["ampl_s"] / df["ampl_c"]

        # This sets center volume (sum of all pixel values in data, after fitting) to one
        ampl_c_pix = 1 / cen_vol_pix3
        ampl_s_pix = df["relat_sur_ampl"] / cen_vol_pix3

        df["ampl_c"] = ampl_c_pix
        df["ampl_s"] = ampl_s_pix

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

    def _get_rf_masks(self, img_stack, mask_threshold=0.1):
        """
        Extracts the contours around the maximum of each receptive field in an image stack. The contour for a field is
        defined as the set of pixels with a value of at least 10% of the maximum pixel value in the field. Only the
        connected region of the contour that contains the maximum value is included.

        Parameters
        ----------
        img_stack : numpy.ndarray
            3D numpy array representing a stack of images. The shape of the array should be (N, H, W).
        mask_threshold : float between 0 and 1
            The threshold for the contour mask.

        Returns
        -------
        numpy.ndarray
            3D numpy array of boolean masks (N, H, W). In each mask, True indicates
            a pixel is part of the contour, and False indicates it is not.
        """
        assert (
            mask_threshold >= 0 and mask_threshold <= 1
        ), "mask_threshold must be between 0 and 1, aborting..."

        masks = []
        for img in img_stack:
            max_val = np.max(img)
            mask = img >= max_val * mask_threshold

            # Label the distinct regions in the mask
            labeled_mask, num_labels = ndimage.label(mask)

            # Find the label of the region that contains the maximum value
            max_label = labeled_mask[np.unravel_index(np.argmax(img), img.shape)]

            # Keep only the region in the mask that contains the maximum value
            mask = labeled_mask == max_label

            masks.append(mask)

        return np.array(masks)

    def _get_retina_with_rf_masks(
        self,
        rf_masks,
        rspace_pos_mm,
    ):
        pass

    def _get_rf_resampling_params(
        self,
        exp_sidelen,
        exp_um_per_pix,
        exp_dd_um,
        lit_dd_vs_ecc_params,
        gc_pos_ecc_mm,
    ):
        """
        Place rf images to pixel space
        First we calculate the dendritic diameter as a function of eccentricity for each gc unit.
        This dendritic diameter is then used to calculate the pixel size in um for each gc unit.
        The minimum pixel size is used to determine the new image stack sidelength.

        Later we will resample original VAE images to this space.
        """

        # Get gc_um_per_pix for all model cells
        # Determine the literature receptive field diameter based on eccentricity
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
        else:
            raise ValueError(
                f"Unknown dd_regr_model: {self.context.my_retina['dd_regr_model']}"
            )

        # Assuming the experimental data reflects the eccentricity for VAE generation
        gc_scaling_factors = lit_dd_at_gc_ecc_um / exp_dd_um

        gc_um_per_pix = gc_scaling_factors * exp_um_per_pix

        # Get min and max values of gc_um_per_pix
        min_um_per_pix = np.min(gc_um_per_pix)
        max_um_per_pix = np.max(gc_um_per_pix)

        # Get new img stack sidelength whose pixel size = min(gc_um_per_pix),
        # and sidelen =  ceil(max(gc_um_per_pix) / min(gc_um_per_pix)) * original sidelen)
        # new_pix_size = min_um_per_pix
        new_sidelen = int((max_um_per_pix / min_um_per_pix) * exp_sidelen)

        return gc_um_per_pix, new_sidelen, min_um_per_pix

    def _get_resampled_scaled_rfs(
        self,
        rfs,
        gc_um_per_pix,
        new_sidelen,
        min_um_per_pix,
    ):
        # Resample all images to new img stack. Use scipy.ndimage.zoom,
        # where zoom factor = gc_um_per_pix for this image / min(gc_um_per_pix)
        img_upsampled = np.zeros((len(rfs), new_sidelen, new_sidelen))
        for i, (img, this_gc_um_per_pix) in enumerate(zip(rfs, gc_um_per_pix)):
            zoom_factor = this_gc_um_per_pix / min_um_per_pix

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

            if 0:
                print(f"Original shape: {img.shape}")
                print(f"Padded shape: {img_padded.shape}")
                print(f"shape after zoom: {img_temp.shape}")
                print(f"shape after crop: {img_cropped.shape}")
                print(f"zoom_factor: {zoom_factor}")
                print(f"img_upsampled shape: {img_upsampled.shape}")
                ecc = self.gc_df["pos_ecc_mm"].values[i]
                print(f"ecc: {ecc}")
                print(f"new_sidelen: {new_sidelen}\n")

                plt.imshow(img_cropped)
                plt.show()

            img_upsampled[i] = img_cropped

        return img_upsampled

    def _get_dd_in_um(self, gc_df):
        # Add diameters to dataframe
        # Assumes rad_c, semi_xc and semi_yc are in millimeters
        if self.context.my_retina["DoG_model"] == "circular":
            den_diam_um_s = gc_df["rad_c"] * 2 * 1000
        elif self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            den_diam_um_s = pd.Series(
                self.ellipse2diam(
                    gc_df["semi_xc"].values * 1000,
                    gc_df["semi_yc"].values * 1000,
                )
            )

        return den_diam_um_s

    def _update_gc_vae_df(self, gc_vae_df_in, new_microm_per_pix):
        """
        Update gc_vae_df to have the same columns as gc_df with corresponding values.
        Update the remaining pixel values to mm, unless unit in is in the column name
        """

        gc_vae_df = gc_vae_df_in.reindex(columns=self.gc_df.columns)
        # These values come from _place_gc_units before _create_spatial_rfs in build()
        # They are independent from FIT.
        gc_vae_df["pos_ecc_mm"] = self.gc_df["pos_ecc_mm"]
        gc_vae_df["pos_polar_deg"] = self.gc_df["pos_polar_deg"]
        gc_vae_df["ecc_group_idx"] = self.gc_df["ecc_group_idx"]

        if self.context.my_retina["DoG_model"] == "circular":
            # Scale factor for rad_c and rad_s from pix to millimeters
            gc_vae_df["rad_c"] = gc_vae_df_in["rad_c"] * new_microm_per_pix / 1000  # mm
            gc_vae_df["rad_s"] = gc_vae_df_in["rad_s"] * new_microm_per_pix / 1000  # mm
            # dendritic diameter in micrometers
            gc_vae_df["den_diam_um"] = gc_vae_df["rad_c"] * 2 * 1000  # um

        elif self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            # Scale factor for semi_x and semi_y from pix to millimeters
            gc_vae_df["semi_xc"] = (
                gc_vae_df_in["semi_xc"] * new_microm_per_pix / 1000
            )  # mm
            gc_vae_df["semi_yc"] = (
                gc_vae_df_in["semi_yc"] * new_microm_per_pix / 1000
            )  # mm
            # dendritic diameter in micrometers
            gc_vae_df["den_diam_um"] = self.ellipse2diam(
                gc_vae_df["semi_xc"].values * 1000, gc_vae_df["semi_yc"].values * 1000
            )

            gc_vae_df["orient_cen_rad"] = gc_vae_df_in["orient_cen_rad"]

            gc_vae_df["xy_aspect_ratio"] = gc_vae_df["semi_yc"] / gc_vae_df["semi_xc"]

        gc_vae_df["ampl_c"] = gc_vae_df_in["ampl_c"]
        gc_vae_df["ampl_s"] = gc_vae_df_in["ampl_s"]

        gc_vae_df["xoc_pix"] = gc_vae_df_in["xoc"]
        gc_vae_df["yoc_pix"] = gc_vae_df_in["yoc"]

        return gc_vae_df

    def _get_full_retina_with_rf_images(
        self,
        ecc_lim_mm,
        polar_lim_deg,
        rf_img,
        df,
        gc_um_per_pix,
    ):
        """
        Build one retina image with all receptive fields. The retina sector is first rotated to
        be symmetric around the horizontal meridian. Then the image is cropped to the smallest
        rectangle that contains all receptive fields. The image is then rotated back to the original
        orientation.

        Parameters
        ----------
        ecc_lim_mm : numpy.ndarray
            1D numpy array with the eccentricity limits in mm.
        polar_lim_deg : numpy.ndarray
            1D numpy array with the polar angle limits in degrees.
        rf_img : numpy.ndarray
            3D numpy array of receptive field images. The shape of the array should be (N, H, W).
        df : pandas.DataFrame
            DataFrame with gc parameters.
        gc_um_per_pix : float
            The number of micrometers per pixel in the rf_img.
        """

        # First we need to get rotation angle of the mean meridian in degrees.
        rot_angle_deg = np.mean(polar_lim_deg)

        # Find corner coordinates of the retina image as [left upper, right_upper, left_lower, right lower]
        # Sector is now symmetrically around the horizontal meridian
        sector_limits_mm = np.zeros((4, 2))
        sector_limits_mm[0, :] = self.pol2cart(
            ecc_lim_mm[0], polar_lim_deg[1] - rot_angle_deg, deg=True
        )
        sector_limits_mm[1, :] = self.pol2cart(
            ecc_lim_mm[1], polar_lim_deg[1] - rot_angle_deg, deg=True
        )
        sector_limits_mm[2, :] = self.pol2cart(
            ecc_lim_mm[0], polar_lim_deg[0] - rot_angle_deg, deg=True
        )
        sector_limits_mm[3, :] = self.pol2cart(
            ecc_lim_mm[1], polar_lim_deg[0] - rot_angle_deg, deg=True
        )

        # Get the max extent for rectangular image
        min_x = np.min(sector_limits_mm[:, 0])
        max_x = np.max(sector_limits_mm[:, 0])
        min_y = np.min(sector_limits_mm[:, 1])
        max_y = np.max(sector_limits_mm[:, 1])

        # Check for max hor extent
        if np.max(ecc_lim_mm) > max_x:
            max_x = np.max(ecc_lim_mm)

        # Convert the rotation angle from degrees to radians
        theta_rad = np.radians(rot_angle_deg)

        # Find the max and min extents in rotated coordinates
        max_x_rot = np.max(
            np.repeat(np.max(ecc_lim_mm), sector_limits_mm.shape[0]) * np.cos(theta_rad)
            - sector_limits_mm[:, 1] * np.sin(theta_rad)
        )
        min_x_rot = np.min(
            np.repeat(np.min(ecc_lim_mm), sector_limits_mm.shape[0]) * np.cos(theta_rad)
            - sector_limits_mm[:, 1] * np.sin(theta_rad)
        )
        max_y_rot = np.max(
            np.repeat(np.max(ecc_lim_mm), sector_limits_mm.shape[0]) * np.sin(theta_rad)
            + sector_limits_mm[:, 1] * np.cos(theta_rad)
        )
        min_y_rot = np.min(
            np.repeat(np.min(ecc_lim_mm), sector_limits_mm.shape[0]) * np.sin(theta_rad)
            + sector_limits_mm[:, 1] * np.cos(theta_rad)
        )

        # Rotate back to original coordinates to get max and min extents
        max_x = max_x_rot * np.cos(theta_rad) + max_y_rot * np.sin(theta_rad)
        min_x = min_x_rot * np.cos(theta_rad) + min_y_rot * np.sin(theta_rad)
        max_y = max_y_rot * np.cos(theta_rad) - max_x_rot * np.sin(theta_rad)
        min_y = min_y_rot * np.cos(theta_rad) - min_x_rot * np.sin(theta_rad)

        # Pad with one full rf in each side. This prevents need to cutting the
        # rf imgs at the borders later on
        pad_size_x_mm = rf_img.shape[2] * gc_um_per_pix / 1000
        pad_size_y_mm = rf_img.shape[1] * gc_um_per_pix / 1000

        min_x = min_x - pad_size_x_mm
        max_x = max_x + pad_size_x_mm
        min_y = min_y - pad_size_y_mm
        max_y = max_y + pad_size_y_mm

        # Get retina image size in pixels
        img_size_x = int(np.ceil((max_x - min_x) * 1000 / gc_um_per_pix))
        img_size_y = int(np.ceil((max_y - min_y) * 1000 / gc_um_per_pix))

        ret_img = np.zeros((img_size_y, img_size_x))

        # Prepare numpy nd array to hold left upeer corner pixel coordinates for each rf image
        rf_lu_pix = np.zeros((df.shape[0], 2), dtype=int)

        pos_ecc_mm = df["pos_ecc_mm"].values
        pos_polar_deg = df["pos_polar_deg"].values

        # Locate left upper corner of each rf img an lay images onto retina image
        for i, row in df.iterrows():
            # Get the position of the rf image in mm
            x_mm, y_mm = self.pol2cart(
                pos_ecc_mm[i], pos_polar_deg[i] - rot_angle_deg, deg=True
            )
            # Get the position of the rf center in pixels
            x_pix_c = int(np.round((x_mm - min_x) * 1000 / gc_um_per_pix))
            y_pix_c = int(np.round((y_mm - min_y) * 1000 / gc_um_per_pix))

            # Get the position of the rf upper left corner in pixels
            # The xoc and yoc are in pixels
            x_pix = x_pix_c - int(row.xoc_pix)
            y_pix = y_pix_c - int(row.yoc_pix)

            # Get the rf image
            this_rf_img = rf_img[i, :, :]

            # Lay the rf image onto the retina image
            ret_img[
                y_pix : y_pix + this_rf_img.shape[0],
                x_pix : x_pix + this_rf_img.shape[1],
            ] += this_rf_img

            # Store the left upper corner pixel coordinates and width and height of each rf image.
            # The width and height are necessary because some are cut off at the edges of the retina image.
            rf_lu_pix[i, :] = [x_pix, y_pix]

        return ret_img, rf_lu_pix

    def _adjust_VAE_center_coverage_to_one(
        self,
        rfs,
        masks,
        img_ret,
        img_ret_mask,
        rf_lu_pix,
        tolerate_error=0.2,
        max_iters=100,
    ):
        """
        Iteratively adjust the receptive fields (RFs) to optimize their coverage of a given retina image.

        The RFs are updated using a pruning model that balances the coverage of the retina image with
        the growth and pruning rate of RFs. This process aims to converge the global coverage factor (GCF,
        sum over individual RFs) towards a target value (1.0 in this case).

        Parameters
        ----------
        rfs : numpy.ndarray
            Array representing the RFs, with shape (n_rfs, n_pixels, n_pixels).
        masks : numpy.ndarray
            Array of masks corresponding to the RFs, with the same shape as `rfs`.
        img_ret : numpy.ndarray
            The compiled retina image to be covered, with shape (h_pixels, w_pixels).
        rf_lu_pix : numpy.ndarray
            Array representing the coordinates of the upper-left pixel of each RF, with shape (n_rfs, 2).
        tolerate_error: float, optional
            The error tolerance between the GCF and the target value. The adjustment process will stop
            when the maximum absolute difference is less than this value. Default is 0.2.
        max_iters: int, optional
            The maximum number of iterations to prevent infinite loop. Default is 100.

        Returns
        -------
        rfs_adjusted : numpy.ndarray
            The adjusted RFs with the same shape as `rfs`.
        img_ret_adjusted : numpy.ndarray
            The adjusted retina image with the same shape as `img_ret`.
        """

        def _get_local_gain_mask(
            img, mask, min_midpoint, min_slope, max_midpoint, max_slope
        ):
            """
            Computes a mask that scales changes in RFs according to the piecewise logistic function.

            It ensures that the changes are applied only to the pixels within the given mask. The logistic
            function parameters can be adjusted to manipulate the gain mask.
            """

            # Piecewise logistic function
            def piecewise_logistic(x):
                # Negative side logistic function, user defined midpoint and slope
                def logistic_negative(x):
                    x = (x - min_midpoint) * min_slope
                    return 1 / (1 + np.exp(-x))

                # Positive side logistic function, user defined midpoint and slope
                def logistic_positive(x):
                    x = (x - max_midpoint) * max_slope
                    return 1 - (1 / (1 + np.exp(-x)))

                return np.piecewise(
                    x, [x < 0, x >= 0], [logistic_negative, logistic_positive]
                )

            # Apply the piecewise logistic transformation
            img_transformed = piecewise_logistic(img)
            if 0:
                # Visualize the function between -1 and 3 with 1000 points
                x = np.linspace(-1, 3, 1000)
                y = piecewise_logistic(x)
                plt.plot(x, y)
                # Put vertical lines to min_ret_value and max_ret_value
                plt.axvline(x=min_ret_value, color="r", linestyle="--")
                plt.axvline(x=max_ret_value, color="r", linestyle="--")
                plt.show()
                exit()

            # Apply mask. Necessary to avoid the rectangular grid to start evolving into RFs
            img_transformed_masked = img_transformed * mask

            return img_transformed_masked

        def _dendritic_adjustment_model(x, x_max, y_min, y_max, x_zero):
            """
            Computes the global delta using a linear model which represents the changes in RF.

            It determines the growth or pruning rate of an RF based on its current GCF and its
            deviation from the target value.
            """

            # Compute the slope and intercept of the line
            slope = y_min / (x_max - x_zero)  # negative slope from x_zero to x_max
            intercept = slope * x_zero

            # Compute the function values
            y = slope * x - intercept

            # Clip x values outside the range [y_min, y_max]
            y = np.clip(y, y_min, y_max)

            return y

        # For area to calculate the error later
        img_ret_mask = img_ret_mask.astype(bool)
        max_ret_value = np.max(img_ret[img_ret_mask])
        min_ret_value = np.min(img_ret[img_ret_mask])

        # Piecewise logistic parameters for local gain mask
        min_midpoint = min_ret_value / 2
        min_slope = 10 / abs(min_ret_value)
        max_midpoint = max_ret_value / 2
        max_slope = 10 / max_ret_value

        # Dendritic adjustment model parameters
        max_pruning = -0.1
        max_growth = 0.1
        converge_to = 1.0  # Global coverage factor (GCF) target value

        height = rfs.shape[1]
        width = rfs.shape[2]

        # Adjust rfs iteratively
        rfs_adjusted = rfs.copy()
        local_gain_masks = np.zeros(rfs.shape)
        error = np.inf
        iteration = 0
        img_ret_adjusted = img_ret
        min_rf_value = np.zeros((200, rfs.shape[0]))

        while error > tolerate_error and iteration < max_iters:
            iteration += 1

            # For each RF, get the global coverage factor and the local gain mask
            for i, rf in enumerate(rfs_adjusted):
                x_pix, y_pix = rf_lu_pix[i, :]

                # Get the coverage factor of the rf
                this_gcf = img_ret_adjusted[
                    y_pix : y_pix + height, x_pix : x_pix + width
                ]

                this_global_delta = _dendritic_adjustment_model(
                    this_gcf, max_ret_value, max_pruning, max_growth, converge_to
                )

                # Note that local gain mask does not cover all pixels,
                # thus the original negative surround of the RF is not affected
                local_gain_masks[i] = _get_local_gain_mask(
                    rf,
                    masks[i],
                    min_midpoint,
                    min_slope,
                    max_midpoint,
                    max_slope,
                )

                this_local_delta = this_global_delta * local_gain_masks[i]
                rfs_adjusted[i] += this_local_delta

                min_rf_value[iteration, i] = np.min(rfs_adjusted[i])

            # Re calculate the coverage factor of each pixel, and check if it has converged
            img_ret_adjusted = np.zeros(img_ret.shape)
            for i, rf in enumerate(rfs_adjusted):
                x_pix, y_pix = rf_lu_pix[i, :]

                # Get the coverage factor of the rf
                img_ret_adjusted[y_pix : y_pix + height, x_pix : x_pix + width] += rf

            # print(f"Maximum adjusted coverage: {img_ret_adjusted.max()}")
            # print(f"Minimum adjusted coverage: {img_ret_adjusted.min()}")

            # Calculate error as the maximum absolute difference from the converge_to value within the masked areas
            # error = np.max(np.abs(img_ret_adjusted * img_ret_mask - converge_to))

            # This creates a masked array where the "ignore" mask is True wherever img_ret_adjusted * img_ret_mask equals 0.
            # Thus we're effectively ignoring the zero values and considering only the non-zero values for calculations.
            masked_error = ma.masked_where(
                img_ret_adjusted * img_ret_mask == 0,
                img_ret_adjusted * img_ret_mask - converge_to,
            )

            # Calculate error as the difference between max img value and converge_to value within the non-masked (non-zero) areas
            # This does not guard against the lower bound of the img values. This should not be a problem, because we are working
            # with masked RF, and the mask is defined as above some positive value.
            error = ma.max(masked_error)

            # print(f"Iteration: {iteration}")
            # print(f"Error: {error}")

        return rfs_adjusted, img_ret_adjusted

    def _get_vae_rfs_with_good_fits(self, retina_vae, nsamples):
        """
        Get spatial receptive fields from the VAE.
        Discard any RFs that are not included in the good_idx_generated.
        Wthout this stage, we would have bad rf:s included into our vae rfs.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate from the VAE.

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
        nsamples_extra = int(nsamples * 1.5)  # 50% extra to account for outliers
        img_processed, img_raw = self._get_generated_rfs(
            retina_vae, n_samples=nsamples_extra
        )
        exp_um_per_pix = self.context.apricot_metadata["data_microm_per_pix"]

        # Fit elliptical gaussians to the generated receptive fields
        # If fit fails, or semi major or minor axis is >3 std from the mean,
        # replace with reserve RF
        self.fit.initialize(
            self.gc_type,
            self.response_type,
            fit_type="generated",
            DoG_model=self.context.my_retina["DoG_model"],
            spatial_data=img_processed,
            new_um_per_pix=exp_um_per_pix,
        )
        (
            _,
            _,
            _,
            gc_vae_df,
            good_idx_generated,
        ) = self.fit.get_generated_spatial_fits()

        # Replace bad rfs with the reserve rfs
        missing_indices = np.setdiff1d(np.arange(nsamples), good_idx_generated)
        available_idx_mask = good_idx_generated > nsamples
        for idx, this_miss in enumerate(missing_indices):
            # Get replacement index
            this_replace = np.where(available_idx_mask)[0][idx]
            # Replace the bad rf with the reserve rf
            img_processed[this_miss, :, :] = img_processed[this_replace, :, :]
            img_raw[this_miss, :, :] = img_raw[this_replace, :, :]
            gc_vae_df.loc[this_miss, :] = gc_vae_df.loc[this_replace, :]

        return img_processed, img_raw, gc_vae_df

    def _create_spatial_rfs(self):
        """
        Generation of spatial receptive fields (RFs) for the retinal ganglion cells (RGCs).

        RF become resampled, and the resolution will change if
        eccentricity is different from eccentricity of the original data.
        """

        # Get fit parameters for dendritic field diameter (um) with respect to eccentricity (mm).
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dd_regr_model = self.dd_regr_model  # "linear", "quadratic", "cubic"
        lit_dd_vs_ecc_params = self._fit_dd_vs_ecc(
            self.visual_field_limit_for_dd_fit_mm, dd_regr_model
        )

        # # Quality control: check that the fitted dendritic diameter is close to the original data
        # # Frechette_2005_JNeurophysiol datasets: 9.7 mm (45°); 9.0 mm (41°); 8.4 mm (38°)
        # # Estimate the orginal data eccentricity from the fit to full eccentricity range
        # exp_rad = self.exp_cen_radius_mm * 2 * 1000
        # dd_ecc_params_full = self._fit_dd_vs_ecc(np.inf, dd_regr_model)
        # data_ecc_mm = self._get_ecc_from_dd(dd_ecc_params_full, dd_regr_model, exp_rad)
        # data_ecc_deg = data_ecc_mm * self.deg_per_mm  # 37.7 deg

        # Endow cells with spatial elliptical receptive fields.
        # Units become mm unless specified in column names.
        if self.spatial_model == "FIT":
            # self.gc_df is updated silently
            if self.rf_coverage_adjusted_to_1 == True:
                # Assumes that the dendritic field diameter is proportional to the coverage
                self._fit_DoG_with_rf_coverage_one()
            elif self.rf_coverage_adjusted_to_1 == False:
                # Read the dendritic field diameter from literature data
                self._fit_DoG_with_rf_from_literature(
                    lit_dd_vs_ecc_params, dd_regr_model
                )
            # Add dendritic diameter for visualization, in micrometers
            self.gc_df["den_diam_um"] = self._get_dd_in_um(self.gc_df)

            # At this point the fitted ellipse spatial receptive fields are ready. All parameters are in self.gc_df.
            # The positions are in the columns 'pos_ecc_mm', 'pos_polar_deg', 'ecc_group_idx', 'xoc', 'yoc', and the rf parameters in either
            # (elliptical DoG) 'semi_xc', 'semi_yc', 'ampl_s', 'relat_sur_diam', 'offset', 'xy_aspect_ratio', 'orient_cen_rad', 'den_diam_um', 'relat_sur_ampl'
            # (circular DoG) 'ampl_c', 'rad_c', 'ampl_s', 'rad_s', 'offset', 'den_diam_um', 'relat_sur_ampl'.
            # If units are not mentioned, they are in mm space.

        if self.spatial_model == "VAE":
            # Fit or load variational autoencoder to generate receptive fields
            # self.gc_df is updated explicitly at the end of this if statement
            self.retina_vae = RetinaVAE(
                self.gc_type,
                self.response_type,
                self.training_mode,
                self.context,
                save_tuned_models=True,
            )

            # endow cells with spatial receptive fields using the generative variational autoencoder model
            nsamples = len(self.gc_df)

            # Get eccentricity in mm
            gc_pos_ecc_mm = np.array(self.gc_df.pos_ecc_mm.values)
            exp_um_per_pix = self.context.apricot_metadata["data_microm_per_pix"]
            # Mean fitted dendritic diameter for the original experimental data
            exp_dd_um = self.exp_cen_radius_mm * 2 * 1000  # in micrometers
            exp_sidelen = self.context.apricot_metadata["data_spatialfilter_height"]

            # Get resampling parameters
            gc_um_per_pix, new_sidelen, new_um_per_pix = self._get_rf_resampling_params(
                exp_sidelen,
                exp_um_per_pix,
                exp_dd_um,
                lit_dd_vs_ecc_params,
                gc_pos_ecc_mm,
            )
            self.updated_vae_um_per_pix = new_um_per_pix

            # Get 50% extra samples to cover the bad fits
            nsamples_extra = int(nsamples * 1.5)  # 50% extra to account for outliers
            img_processed_extra, img_raw_extra = self._get_generated_rfs(
                self.retina_vae, n_samples=nsamples_extra
            )

            idx_to_process = np.arange(nsamples)
            img_rfs = np.zeros((nsamples, new_sidelen, new_sidelen))
            available_idx_mask = np.ones(nsamples_extra, dtype=bool)
            available_idx_mask[idx_to_process] = False
            img_to_resample = img_processed_extra[idx_to_process, :, :]
            good_mask_compiled = np.zeros(nsamples, dtype=bool)
            for _ in range(100):
                # Upsample according to smallest rf diameter
                img_after_resample = self._get_resampled_scaled_rfs(
                    img_to_resample[idx_to_process, :, :],
                    gc_um_per_pix,
                    new_sidelen,
                    new_um_per_pix,
                )

                # Fit elliptical gaussians to the img[idx_to_process]
                self.fit.initialize(
                    self.gc_type,
                    self.response_type,
                    fit_type="generated",
                    DoG_model=self.context.my_retina["DoG_model"],
                    spatial_data=img_after_resample,
                    new_um_per_pix=new_um_per_pix,
                    mark_outliers_bad=True,
                )
                # Some fo the fits might be bad. Discard them.
                good_idx_this_iter = self.fit.good_idx_generated
                good_idx_generated = idx_to_process[good_idx_this_iter]
                # save the good rfs
                img_rfs[good_idx_generated, :, :] = img_after_resample[
                    good_idx_this_iter, :, :
                ]

                good_mask_compiled[good_idx_generated] = True
                # update idx_to_process
                idx_to_process = np.setdiff1d(idx_to_process, good_idx_generated)

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

            # Redo the good fits to get the stats out
            # Now there is no need to discard any rfs
            self.fit.initialize(
                self.gc_type,
                self.response_type,
                fit_type="generated",
                DoG_model=self.context.my_retina["DoG_model"],
                spatial_data=img_rfs,
                new_um_per_pix=new_um_per_pix,
                mark_outliers_bad=False,
            )
            (
                self.gen_stat_df,
                self.gen_spat_cen_sd,
                self.gen_spat_sur_sd,
                self.gc_vae_df,
                _,
            ) = self.fit.get_generated_spatial_fits()
            good_idx_compiled = np.where(good_mask_compiled)[0]

            # Update gc_vae_df to have the same columns as gc_df and to mm
            self.gc_vae_df = self._update_gc_vae_df(self.gc_vae_df, new_um_per_pix)
            # Get center masks for the generated spatial rfs
            # Mask threshold is relative to max value in the image
            img_rfs_mask = self._get_rf_masks(img_rfs, mask_threshold=0.1)

            # Save the generated receptive fields
            output_path = self.context.output_folder
            filename_stem = self.context.my_retina["spatial_rfs_file"]
            # Sum separate rf images onto one retina
            # Uses pos_ecc_mm, pos_polar_deg
            img_ret, rf_lu_pix = self._get_full_retina_with_rf_images(
                self.ecc_lim_mm,
                self.polar_lim_deg,
                img_rfs,
                self.gc_vae_df,
                new_um_per_pix,
            )

            img_ret_masked, _ = self._get_full_retina_with_rf_images(
                self.ecc_lim_mm,
                self.polar_lim_deg,
                img_rfs_mask,
                self.gc_vae_df,
                new_um_per_pix,
            )

            if self.rf_coverage_adjusted_to_1 is True:
                (
                    img_rfs_adjusted,
                    img_ret_adjusted,
                ) = self._adjust_VAE_center_coverage_to_one(
                    img_rfs,
                    img_rfs_mask,
                    img_ret,
                    img_ret_masked,
                    rf_lu_pix,
                    tolerate_error=0.01,
                )
                img_rfs_final = img_rfs_adjusted

            else:
                img_rfs_adjusted = np.zeros_like(img_rfs)
                img_ret_adjusted = np.zeros_like(img_ret)
                img_rfs_final = img_rfs

            # Save generated receptive fields
            self.data_io.save_generated_rfs(
                img_rfs_final, output_path, filename_stem=filename_stem
            )
            # Save masked retina to indicate center regions
            maskname_stem = Path(filename_stem).stem + "_center_mask.npy"
            self.data_io.save_generated_rfs(
                img_rfs_mask,
                output_path,
                filename_stem=maskname_stem,
            )

            # Set self attribute for later visualization of image histograms
            self.project_data.construct_retina["gen_spat_img"] = {
                "img_processed": img_processed_extra[good_idx_compiled, :, :],
                "img_raw": img_raw_extra[good_idx_compiled, :, :],
            }

            # Set self attributes for later visualization
            self.project_data.construct_retina["gen_rfs"] = {
                "img_rf": img_rfs,
                "img_rf_mask": img_rfs_mask,
                "img_rfs_adjusted": img_rfs_adjusted,
            }

            self.project_data.construct_retina["gen_ret"] = {
                "img_ret": img_ret,
                "img_ret_masked": img_ret_masked,
                "img_ret_adjusted": img_ret_adjusted,
            }

            # Save original df
            self.gc_df_original = self.gc_df.copy()
            # Apply the spatial VAE model to df
            self.gc_df = self.gc_vae_df

        # Scale center and surround amplitude: center Gaussian volume in pixel space becomes one
        # Surround amplitude is scaled relative to center volume of one
        self.gc_df = self._scale_both_amplitudes(self.gc_df)

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

        # -- Second, endow cells with spatial receptive fields
        self._create_spatial_rfs()

        # -- Third, endow cells with temporal receptive fields
        self._create_fixed_temporal_rfs()  # Chichilnisky data
        self._create_dynamic_temporal_rfs()  # Benardete & Kaplan data

        # -- Fourth, endow cells with tonic drive
        self._create_tonic_drive()

        n_rgc = len(self.gc_df)
        print(f"Built RGC mosaic with {n_rgc} cells")

        # Save the receptive field mosaic
        self.save_gc_csv()

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
