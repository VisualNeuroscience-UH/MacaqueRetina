# Numerical
import numpy as np
import numpy.ma as ma
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
from scipy.optimize import root
import pandas as pd
from scipy import ndimage

import torch
import torch.nn.functional as F

# from torch.utils.data import DataLoader

# from scipy.signal import convolve
# from scipy.interpolate import interp1d

# Image analysis
from skimage import measure
from skimage.transform import resize
import matplotlib.pyplot as plt

# Data IO
import cv2
from PIL import Image

# Viz
from tqdm import tqdm

# Comput Neurosci
# import brian2 as b2
# import brian2.units as b2u

# Local
from retina.fit_module import Fit
from retina.retina_math_module import RetinaMath
from retina.vae_module import RetinaVAE

# Builtin
from pathlib import Path
import pdb
from copy import deepcopy


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
    eccentricity_in_mm : list
        List of two floats, the eccentricity limits in mm
    theta : list
        Numpy array two floats, the sector limits in degrees
    randomize_position : bool
        Whether to randomize the position of the ganglion cells
    deg_per_mm : float
        Degrees per mm
    model_type : str
        Type of model, either "FIT"or  "VAE"
    gc_proportion : float
        Proportion of ganglion cells to be created
    gc_df : pd.DataFrame
        Dataframe containing the ganglion cell mosaic
    """

    _properties_list = [
        "path",
        "output_folder",
        "input_folder",
        "my_retina",
        "apricot_data_folder",
        "literature_data_folder",
        "dendr_diam1_file",
        "dendr_diam2_file",
        "gc_density_file",
        "apricot_metadata",
    ]

    def __init__(self, context, data_io, viz) -> None:
        # Dependency injection at ProjectManager construction
        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
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

    def _initialize(self, fits_from_file=None):
        """
        Initialize the ganglion cell mosaic.
            First: sets ConstructRetina instance parameters from conf file my_retina
            Second: creates empty gc_df to hold the final ganglion cell mosaics
            Third: gets gc creation model according to model_type
                Calls Fit or RetinaVAE classes

        See class attributes for more details.

        Parameters
        ----------
        fits_from_file : str
            Path to a file containing the fits. If None, fits are computed from scratch
        """

        my_retina = self.context.my_retina
        gc_type = my_retina["gc_type"]
        response_type = my_retina["response_type"]
        ecc_limits = my_retina["ecc_limits"]
        sector_limits = my_retina["sector_limits"]
        model_density = my_retina["model_density"]
        self.rf_coverage_adjusted_to_1 = my_retina["rf_coverage_adjusted_to_1"]
        self.dd_regr_model = my_retina["dd_regr_model"]
        randomize_position = my_retina["randomize_position"]
        self.deg_per_mm = my_retina["deg_per_mm"]

        self.gc_type = gc_type
        self.response_type = response_type

        self.model_type = my_retina["model_type"]
        if self.model_type in ["VAE"]:
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
        ), "Wrong type or length of theta, aborting"
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
        )  # Turn list to numpy array
        self.polar_lim_deg = np.asarray(sector_limits)  # Turn list to numpy array
        self.randomize_position = randomize_position

        # If study concerns visual field within 4 mm (20 deg) of retinal eccentricity, the cubic fit for dendritic diameters fails close to fovea. Better limit it to more central part of the data
        if np.max(self.ecc_lim_mm) <= 4:
            self.visual_field_fit_limit = 4
        else:
            self.visual_field_fit_limit = np.inf

        # Initialize pandas dataframe to hold the ganglion cells (one per row) and all their parameters in one place
        columns = [
            "pos_ecc_mm",
            "pos_polar_deg",
            "ecc_group_idx",
            "semi_xc",
            "semi_yc",
            "xy_aspect_ratio",
            "ampl_s",
            "relat_sur_diam",
            "orient_cen",
        ]
        self.gc_df = pd.DataFrame(columns=columns)

        # Current version needs Fit for all 'model_type's (FIT, VAE, etc.)
        # If surround is fixed, the surround position, semi_x, semi_y (aspect_ratio) and orientation are the same as center params. This appears to give better results.
        self.surround_fixed = 1

        # Make or read fits
        if fits_from_file is None:
            (
                self.exp_stat_df,
                self.good_data_idx,
                self.bad_data_idx,
                self.exp_spat_cen_sd_mm,
                self.exp_spat_sur_sd_mm,
                self.exp_temp_filt_to_viz,
                self.exp_spat_filt_to_viz,
                self.exp_spat_stat_to_viz,
                self.exp_temp_stat_to_viz,
                self.exp_tonic_dr_to_viz,
                self.apricot_data_resolution_hw,
            ) = Fit(
                self.context.apricot_data_folder,
                gc_type,
                response_type,
                fit_type="experimental",
            ).get_experimental_fits()
        else:
            # probably obsolete 230118 SV
            self.all_fits_df = pd.read_csv(
                fits_from_file, header=0, index_col=0
            ).fillna(0.0)

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
            The distribution to sample from. Supported distributions: "gamma", "vonmises", "skewnorm".

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
        ], "Distribution not supported"

        if distribution == "gamma":
            distribution_parameters = stats.gamma.rvs(
                a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )  # random_state is the seed
        elif distribution == "vonmises":
            distribution_parameters = stats.vonmises.rvs(
                kappa=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )
        elif distribution == "skewnorm":
            distribution_parameters = stats.skewnorm.rvs(
                a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )

        return distribution_parameters

    def _read_gc_density_data(self):
        """
        Read re-digitized old literature data from mat files
        """

        print("Reading density data from:", self.context.gc_density_file)
        gc_density = sio.loadmat(
            self.context.gc_density_file,
            variable_names=["Xdata", "Ydata"],
        )
        cell_eccentricity = np.squeeze(gc_density["Xdata"])
        cell_density = (
            np.squeeze(gc_density["Ydata"]) * 1e3
        )  # Cells are in thousands, thus the 1e3
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

    def _fit_dd_vs_ecc(self, visual_field_fit_limit, dd_regr_model):
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

        dendr_diam1 = sio.loadmat(
            self.context.dendr_diam1_file, variable_names=["Xdata", "Ydata"]
        )
        dendr_diam2 = sio.loadmat(
            self.context.dendr_diam2_file, variable_names=["Xdata", "Ydata"]
        )

        # Parasol fit
        gc_type = self.gc_type

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
        data_all_x_index = data_all_x <= visual_field_fit_limit
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
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": polynomials[1],
                "slope": polynomials[0],
            }
        elif dd_regr_model == "quadratic":
            polynomial_order = 2
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": polynomials[2],
                "slope": polynomials[1],
                "square": polynomials[0],
            }
        elif dd_regr_model == "cubic":
            polynomial_order = 3
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": polynomials[3],
                "slope": polynomials[2],
                "square": polynomials[1],
                "cube": polynomials[0],
            }

        dataset_name = f"All data {dd_regr_model} fit"

        self.dd_vs_ecc_to_viz = {
            "data_all_x": data_all_x,
            "data_all_y": data_all_y,
            "polynomials": polynomials,
            "dataset_name": dataset_name,
            "title": f"DF diam wrt ecc for {self.gc_type} type, {dataset_name} dataset",
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

    def _create_spatial_rfs_coverage(self):
        """
        Create spatial receptive fields to model cells using coverage = 1.
        Starting from 2D difference-of-gaussian parameters:
        'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'ampl_s','relat_sur_diam', 'orient_cen'

        Places all ganglion cell spatial parameters to ganglion cell object dataframe self.gc_df
        """

        # Set parameters for all cells
        n_cells = len(self.gc_df)
        spatial_df = self.exp_stat_df[self.exp_stat_df["domain"] == "spatial"]
        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # Calculate RF diameter scaling factor for all ganglion cells
        # Area of RF = Scaling_factor * Random_factor * Area of ellipse(semi_xc,semi_yc), solve Scaling_factor.
        area_of_ellipse = self.ellipse2area(
            self.gc_df["semi_xc"], self.gc_df["semi_yc"]
        )  # Units are pixels for the Chichilnisky data

        """
        The area_of_rf contains area for all model units. Its sum must fill the whole area (coverage factor = 1).
        We do it separately for each ecc sector, step by step, to keep coverage factor at 1 despite changing gc density with ecc
        """
        area_scaling_factors_coverage1 = np.zeros(area_of_ellipse.shape)
        for index, surface_area in enumerate(self.sector_surface_area_all):
            scaling_for_coverage_1 = (surface_area * 1e6) / np.sum(
                area_of_ellipse[self.gc_df["ecc_group_idx"] == index]
            )  # in micrometers2

            area_scaling_factors_coverage1[
                self.gc_df["ecc_group_idx"] == index
            ] = scaling_for_coverage_1

        # Apply scaling factors to semi_xc and semi_yc. Units are micrometers.
        # scale_random_distribution = 0.08  # Estimated by eye from Watanabe and Perry data.
        # Normal distribution with scale_random_distribution 0.08 cover about 25% above and below the mean value
        scale_random_distribution = 0.001
        random_normal_distribution1 = 1 + np.random.normal(
            scale=scale_random_distribution, size=n_cells
        )

        semi_xc = (
            np.sqrt(area_scaling_factors_coverage1)
            * self.gc_df["semi_xc"]
            * random_normal_distribution1
        )
        random_normal_distribution2 = 1 + np.random.normal(
            scale=scale_random_distribution, size=n_cells
        )  # second randomization

        semi_yc = (
            np.sqrt(area_scaling_factors_coverage1)
            * self.gc_df["semi_yc"]
            * random_normal_distribution2
        )

        # Scale from micrometers to millimeters and return to numpy matrix
        self.gc_df["semi_xc"] = semi_xc / 1000
        self.gc_df["semi_yc"] = semi_yc / 1000

        # self.gc_df["orient_cen"] = self.gc_df[
        #     "pos_polar_deg"
        # ]  # plus some noise here TODO. See Watanabe 1989 JCompNeurol section Dendritic field orietation

    def _create_spatial_rfs_ecc(self, dd_ecc_params, dd_regr_model):
        """
        Create spatial receptive fields to model cells according to eccentricity.
        Starting from 2D difference-of-gaussian parameters:
        'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'ampl_s','relat_sur_diam', 'orient_cen'

        Places all ganglion cell spatial parameters to ganglion cell object dataframe self.gc_df
        """

        # Get eccentricity data for all model cells
        gc_eccentricity = self.gc_df["pos_ecc_mm"].values

        # Get rf diameter vs eccentricity
        dict_key = "{0}_{1}".format(self.gc_type, dd_regr_model)
        diam_fit_params = dd_ecc_params[dict_key]

        if dd_regr_model == "linear":
            gc_diameters_um = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
            )  # Units are micrometers
        elif dd_regr_model == "quadratic":
            gc_diameters_um = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
                + diam_fit_params["square"] * gc_eccentricity**2
            )
        elif dd_regr_model == "cubic":
            gc_diameters_um = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
                + diam_fit_params["square"] * gc_eccentricity**2
                + diam_fit_params["cube"] * gc_eccentricity**3
            )

        # Set parameters for all cells
        n_cells = len(self.gc_df)
        spatial_df = self.exp_stat_df[self.exp_stat_df["domain"] == "spatial"]
        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # Scale factor for semi_x and semi_y from pix at data eccentricity to micrometers at the actual eccentricity
        # Units are pixels for the Chichilnisky data and they are at large eccentricity
        scaling_factor = self.context.apricot_metadata["data_microm_per_pix"] * (
            (gc_diameters_um / 2) / (self.exp_spat_cen_sd_mm * 1000)
        )

        # Scale semi_x to micrometers at its actual eccentricity and divide by 1000 to get mm
        self.gc_df["semi_xc"] = self.gc_df["semi_xc"] * scaling_factor / 1000

        # Scale semi_y to micrometers at its actual eccentricity
        self.gc_df["semi_yc"] = self.gc_df["semi_yc"] * scaling_factor / 1000

        # Apply scaling factors to semi_xc and semi_yc. Units are micrometers.
        scale_rand_distr = 0.001
        rnd_x = 1 + np.random.normal(scale=scale_rand_distr, size=n_cells)
        self.gc_df["semi_xc"] = self.gc_df["semi_xc"] * rnd_x
        rnd_y = 1 + np.random.normal(scale=scale_rand_distr, size=n_cells)
        self.gc_df["semi_yc"] = self.gc_df["semi_yc"] * rnd_y

    def _densfunc(self, r, d0, beta):
        return d0 * (1 + beta * r) ** (-2)

    def _place_gc_units(self, gc_density_func_params):
        """
        Place ganglion cell center positions to retina

        Creates self.gc_df: pandas.DataFrame with columns:
            pos_ecc_mm (mm), pos_polar_deg (deg), ecc_group_idx

        Parameters
        ----------
        gc_density_func_params: dict
            Dictionary with parameters for the density function
        """

        # Place cells inside one polar sector with density according to mid-ecc
        eccentricity_in_mm_total = self.ecc_lim_mm
        theta = self.polar_lim_deg
        randomize_position = self.randomize_position

        # Loop for reasonable delta ecc to get correct density in one hand and good cell distribution from the algo on the other
        # Lets fit close to 0.1 mm intervals, which makes sense up to some 15 deg. Thereafter longer jumps would do fine.
        fit_interval = 0.1  # mm
        n_steps = int(np.round(np.ptp(eccentricity_in_mm_total) / fit_interval))
        eccentricity_steps = np.linspace(
            eccentricity_in_mm_total[0], eccentricity_in_mm_total[1], 1 + n_steps
        )

        # Initialize position arrays
        matrix_polar_angle_randomized_all = np.asarray([])
        matrix_eccentricity_randomized_all = np.asarray([])
        gc_eccentricity_group_index = np.asarray([])

        true_eccentricity_end = []
        sector_surface_area_all = []
        for ecc_group_idx, current_step in enumerate(np.arange(int(n_steps))):
            if (
                true_eccentricity_end
            ):  # If the eccentricity has been adjusted below inside the loop
                eccentricity_in_mm = np.asarray(
                    [true_eccentricity_end, eccentricity_steps[current_step + 1]]
                )
            else:
                eccentricity_in_mm = np.asarray(
                    [
                        eccentricity_steps[current_step],
                        eccentricity_steps[current_step + 1],
                    ]
                )

            # fetch center ecc in mm
            center_ecc = np.mean(eccentricity_in_mm)

            # rotate theta to start from 0
            theta_rotated = theta - np.min(theta)
            angle = np.max(theta_rotated)  # The angle is now == max theta

            # Calculate area
            assert (
                eccentricity_in_mm[0] < eccentricity_in_mm[1]
            ), "Radii in wrong order, give [min max], aborting"
            sector_area_remove = self.sector2area(eccentricity_in_mm[0], angle)
            sector_area_full = self.sector2area(eccentricity_in_mm[1], angle)
            sector_surface_area = sector_area_full - sector_area_remove  # in mm2
            sector_surface_area_all.append(
                sector_surface_area
            )  # collect sector area for each ecc step

            # N cells for given ecc
            my_gaussian_fit = self.gauss_plus_baseline(
                center_ecc, *gc_density_func_params
            )  # leads to div by zero
            # my_gaussian_fit = self._densfunc(
            #     center_ecc, 5.32043939e05, 2.64289725
            # )  # deactivated SV 220531
            Ncells = sector_surface_area * my_gaussian_fit * self.gc_proportion

            # place cells in regular grid
            # Vector of cell positions in radial and polar directions. Angle in degrees.
            inner_arc_in_mm = (angle / 360) * 2 * np.pi * eccentricity_in_mm[0]
            delta_eccentricity_in_mm = eccentricity_in_mm[1] - eccentricity_in_mm[0]

            # By assuming that the ratio of the number of points in x and y direction respects
            # the sector's aspect ratio, ie.
            # n_segments_arc / n_segments_eccentricity = inner_arc_in_mm / delta_eccentricity_in_mm
            # we get:
            n_segments_arc = np.sqrt(
                Ncells * (inner_arc_in_mm / delta_eccentricity_in_mm)
            )
            n_segments_eccentricity = np.sqrt(
                Ncells * (delta_eccentricity_in_mm / inner_arc_in_mm)
            )
            # Because n_segments_arc and n_segments_eccentricity can be floats, we round them to integers
            int_n_segments_arc = int(round(n_segments_arc))
            int_n_segments_eccentricity = int(round(n_segments_eccentricity))

            # Recalc delta_eccentricity_in_mm given the n segments to avoid non-continuous cell densities
            true_n_cells = int_n_segments_arc * int_n_segments_eccentricity
            true_sector_area = true_n_cells / (my_gaussian_fit * self.gc_proportion)
            true_delta_eccentricity_in_mm = (
                int_n_segments_eccentricity / int_n_segments_arc
            ) * inner_arc_in_mm

            radius_segment_length = (
                true_delta_eccentricity_in_mm / int_n_segments_eccentricity
            )
            theta_segment_angle = (
                angle / int_n_segments_arc
            )  # Note that this is different from inner_arc_in_mm / int_n_segments_arc

            # Set the true_eccentricity_end
            true_eccentricity_end = (
                eccentricity_in_mm[0] + true_delta_eccentricity_in_mm
            )

            vector_polar_angle = np.linspace(theta[0], theta[1], int_n_segments_arc)
            vector_eccentricity = np.linspace(
                eccentricity_in_mm[0],
                true_eccentricity_end - radius_segment_length,
                int_n_segments_eccentricity,
            )

            # meshgrid and shift every second column to get good GC tiling
            matrix_polar_angle, matrix_eccentricity = np.meshgrid(
                vector_polar_angle, vector_eccentricity
            )
            matrix_polar_angle[::2] = matrix_polar_angle[::2] + (
                angle / (2 * n_segments_arc)
            )  # shift half the inter-cell angle

            # Randomization using normal distribution
            matrix_polar_angle_randomized = (
                matrix_polar_angle
                + theta_segment_angle
                * randomize_position
                * (
                    np.random.randn(
                        matrix_polar_angle.shape[0], matrix_polar_angle.shape[1]
                    )
                )
            )
            matrix_eccentricity_randomized = (
                matrix_eccentricity
                + radius_segment_length
                * randomize_position
                * (
                    np.random.randn(
                        matrix_eccentricity.shape[0], matrix_eccentricity.shape[1]
                    )
                )
            )

            matrix_polar_angle_randomized_all = np.append(
                matrix_polar_angle_randomized_all,
                matrix_polar_angle_randomized.flatten(),
            )
            matrix_eccentricity_randomized_all = np.append(
                matrix_eccentricity_randomized_all,
                matrix_eccentricity_randomized.flatten(),
            )

            assert true_n_cells == len(
                matrix_eccentricity_randomized.flatten()
            ), "N cells don't match, check the code"
            gc_eccentricity_group_index = np.append(
                gc_eccentricity_group_index,
                np.ones(true_n_cells) * ecc_group_idx,
            )

        # Save cell position data to current ganglion cell object
        self.gc_df["pos_ecc_mm"] = matrix_eccentricity_randomized_all
        self.gc_df["pos_polar_deg"] = matrix_polar_angle_randomized_all
        self.gc_df["ecc_group_idx"] = gc_eccentricity_group_index.astype(np.uint32)
        self.sector_surface_area_all = np.asarray(sector_surface_area_all)

        # Pass the GC object to self, because the Viz class is not inherited
        self.gc_density_func_params = gc_density_func_params

    def _create_temporal_receptive_fields(self):
        n_cells = len(self.gc_df)
        temporal_df = self.exp_stat_df[self.exp_stat_df["domain"] == "temporal"]
        for param_name, row in temporal_df.iterrows():
            shape, loc, scale, distribution, _ = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

    def _scale_both_amplitudes(self, gc_df):
        """
        Scale center and surround ampl_s so that the spatial RF volume is comparable to that of data.
        Second step of scaling is done before convolving with the stimulus.
        """

        # For each model cell, set center amplitude as data_cen_mean**2 / sigma_x * sigma_y
        # For each model cell, scale surround amplitude by data_sur_mean**2 / sur_sigma_x * sur_sigma_y
        # (Volume of 2D Gaussian = 2 * pi * sigma_x*sigma_y)

        n_rgc = len(gc_df)
        ampl_c = np.zeros(n_rgc)
        # ampl_s = np.zeros(n_rgc)

        for i in range(n_rgc):
            ampl_c[i] = self.exp_spat_cen_sd_mm**2 / (
                gc_df.iloc[i].semi_xc * gc_df.iloc[i].semi_yc
            )

        data_rel_sur_amplitude = gc_df["ampl_s"]
        gc_df["ampl_c"] = ampl_c
        gc_df["ampl_s"] = ampl_c * data_rel_sur_amplitude
        gc_df["relat_sur_ampl"] = gc_df["ampl_s"] / gc_df["ampl_c"]

        return gc_df

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

    def _get_generated_spatial_data(self, retina_vae, nsamples=10):
        # --- 1. make a probability density function of the latent space
        retina_vae = self.retina_vae

        latent_data = self.get_data_at_latent_space(retina_vae)

        # Make a probability density function of the latent_data
        # Both uniform and normal distr during learning is sampled
        # using gaussian kde estimate. The kde estimate is basically smooth histogram,
        # so it is not a problem that the data is not normal.
        latent_pdf = stats.gaussian_kde(latent_data.T)

        # --- 2. sample from the pdf
        n_samples = len(self.gc_df)
        # n_samples = 1000
        latent_samples = torch.tensor(latent_pdf.resample(n_samples).T).to(
            retina_vae.device
        )
        # Change the dtype to float32
        latent_samples = latent_samples.type(torch.float32)
        latent_dim = self.retina_vae.latent_dim

        self.gen_latent_space_to_viz = {
            "samples": latent_samples.to("cpu").numpy(),
            "dim": latent_dim,
            "data": latent_data,
        }

        # --- 3. decode the samples
        img_stack_np = self.retina_vae.vae.decoder(latent_samples)

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

    def _get_upsampled_scaled_rfs(
        self,
        rfs,
        dd_ecc_params,
        ret_pos_ecc_mm,
        data_um_per_pix,
        data_dd_um,
    ):
        """
        Place rf images to proximal pixel space. Upsample to original images.
        """

        # Assert that the vertical and horizontal sidelengths are equal
        assert (
            rfs.shape[-2] == rfs.shape[-1]
        ), "The receptive field images are not square, aborting..."

        # Get um_per_pix for all model cells
        # Determine the receptive field diameter based on eccentricity
        key = list(dd_ecc_params.keys())[0]
        parameters = dd_ecc_params[key]
        dd_um = np.polyval(
            [
                parameters.get("cube", 0),
                parameters.get("square", 0),
                parameters.get("slope", 0),
                parameters.get("intercept", 0),
            ],
            ret_pos_ecc_mm,
        )

        scaling_factors = dd_um / data_dd_um
        um_per_pix = scaling_factors * data_um_per_pix

        # Get min and max values of um_per_pix
        min_um_per_pix = np.min(um_per_pix)
        max_um_per_pix = np.max(um_per_pix)

        # Get new img stack sidelength whose pixel size = min(um_per_pix),
        # and sidelen =  ceil(max(um_per_pix) / min(um_per_pix)) * original sidelen)
        new_pix_size = min_um_per_pix
        old_sidelen = rfs.shape[-1]
        new_sidelen = int((max_um_per_pix / min_um_per_pix) * old_sidelen)

        # Resample all images to new img stack. Use scipy.ndimage.zoom,
        # where zoom factor = um_per_pix for this image / min(um_per_pix)
        img_upsampled = np.zeros((len(rfs), new_sidelen, new_sidelen))
        for i, (img, um_per_pix) in enumerate(zip(rfs, um_per_pix)):
            zoom_factor = um_per_pix / min_um_per_pix

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
                print(f"Original size: {img.shape}")
                print(f"Padded size: {img_padded.shape}")
                print(f"Size after zoom: {img_temp.shape}")
                print(f"Size after crop: {img_cropped.shape}")
                print(f"zoom_factor: {zoom_factor}\n")
                print(f"img_upsampled size: {img_upsampled.shape}")

            img_upsampled[i] = img_cropped

        return img_upsampled, min_um_per_pix

    def _get_dd_fit_for_viz(self, gc_df):
        # # Add diameters to dataframe
        # self.gc_df["den_diam_um"] = self.ellipse2diam(
        #     self.gc_df["semi_xc"].values * 1000, self.gc_df["semi_yc"].values * 1000
        # )
        # self.dd_vs_ecc_to_viz["dd_fit_x"] = self.gc_df[
        #     "pos_ecc_mm"
        # ].values
        # self.dd_vs_ecc_to_viz["dd_fit_y"] = self.gc_df[
        #     "den_diam_um"
        # ].values
        # Add diameters to dataframe
        gc_df["den_diam_um"] = self.ellipse2diam(
            gc_df["semi_xc"].values * 1000, gc_df["semi_yc"].values * 1000
        )
        dd_fit_x = gc_df["pos_ecc_mm"].values
        dd_fit_y = gc_df["den_diam_um"].values

        return gc_df, dd_fit_x, dd_fit_y

    def _update_gc_vae_df(self, gc_vae_df_in, new_microm_per_pix):
        """
        Update gc_vae_df to have the same columns as gc_df with corresponding values.
        """

        gc_vae_df = gc_vae_df_in.reindex(columns=self.gc_df.columns)
        gc_vae_df["pos_ecc_mm"] = self.gc_df["pos_ecc_mm"]
        gc_vae_df["pos_polar_deg"] = self.gc_df["pos_polar_deg"]
        gc_vae_df["ecc_group_idx"] = self.gc_df["ecc_group_idx"]

        # Scale factor for semi_x and semi_y from pix to micrometers and divide by 1000 to get mm
        gc_vae_df["semi_xc"] = gc_vae_df_in["semi_xc"] * new_microm_per_pix / 1000
        gc_vae_df["semi_yc"] = gc_vae_df_in["semi_yc"] * new_microm_per_pix / 1000

        gc_vae_df["den_diam_um"] = self.ellipse2diam(
            gc_vae_df["semi_xc"].values * 1000, gc_vae_df["semi_yc"].values * 1000
        )

        gc_vae_df["orient_cen"] = gc_vae_df_in["orient_cen"]

        gc_vae_df["xy_aspect_ratio"] = gc_vae_df_in["semi_yc"] / gc_vae_df_in["semi_xc"]

        gc_vae_df["ampl_c"] = gc_vae_df_in["ampl_c"]
        gc_vae_df["ampl_s"] = gc_vae_df_in["ampl_s"]

        gc_vae_df = self._scale_both_amplitudes(gc_vae_df)

        gc_vae_df["xoc_pix"] = gc_vae_df_in["xoc"]
        gc_vae_df["yoc_pix"] = gc_vae_df_in["yoc"]

        return gc_vae_df

    def _get_full_retina_with_rf_images(
        self, ecc_lim_mm, polar_lim_deg, rf_img, df, um_per_pix
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
        um_per_pix : float
            The number of micrometers per pixel in the rf_img.
        """

        # First we need to get rotation angle off the horizontal meridian in degrees.
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

        # Assuming `theta` is the rotation angle in degrees

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
        pad_size_x_mm = rf_img.shape[2] * um_per_pix / 1000
        pad_size_y_mm = rf_img.shape[1] * um_per_pix / 1000

        min_x = min_x - pad_size_x_mm
        max_x = max_x + pad_size_x_mm
        min_y = min_y - pad_size_y_mm
        max_y = max_y + pad_size_y_mm

        # Get retina image size in pixels
        img_size_x = int(np.ceil((max_x - min_x) * 1000 / um_per_pix))
        img_size_y = int(np.ceil((max_y - min_y) * 1000 / um_per_pix))

        ret_img = np.zeros((img_size_y, img_size_x))

        # Prepare numpy nd array to hold pixel coordinates for each rf image
        rf_lu_pix = np.zeros((df.shape[0], 2), dtype=int)

        # Locate left upper corner of each rf img an lay images onto retina image
        for i, row in df.iterrows():
            # Get the position of the rf image in mm
            x_mm, y_mm = self.pol2cart(
                row.pos_ecc_mm, row.pos_polar_deg - rot_angle_deg, deg=True
            )
            # Get the position of the rf center in pixels
            x_pix_c = int(np.round((x_mm - min_x) * 1000 / um_per_pix))
            y_pix_c = int(np.round((y_mm - min_y) * 1000 / um_per_pix))

            # Get the position of the rf upper left corner in pixels
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

    def _adjust_rf_coverage(
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
        min_rf_value = np.zeros((100, rfs.shape[0]))

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

        # Get fit parameters for dendritic field diameter (um) with respect to eccentricity (mm).
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dd_regr_model = self.dd_regr_model  # "linear", "quadratic", "cubic"
        dd_ecc_params = self._fit_dd_vs_ecc(self.visual_field_fit_limit, dd_regr_model)

        # # Quality control: check that the fitted dendritic diameter is close to the original data
        # # Frechette_2005_JNeurophysiol datasets: 9.7 mm (45°); 9.0 mm (41°); 8.4 mm (38°)
        # # Estimate the orginal data eccentricity from the fit to full eccentricity range
        # dd_ecc_params_full = self._fit_dd_vs_ecc(np.inf, dd_regr_model)
        # data_ecc_mm = self._get_ecc_from_dd(dd_ecc_params_full, dd_regr_model, dd)
        # data_ecc_deg = data_ecc_mm * self.deg_per_mm  # 38.4 deg

        # -- Second, endow cells with spatial receptive fields (units mm)
        if self.rf_coverage_adjusted_to_1 == True:
            # Assumes that the dendritic field diameter is proportional to the coverage
            self._create_spatial_rfs_coverage()
        elif self.rf_coverage_adjusted_to_1 == False:
            # Read the dendritic field diameter from literature data
            self._create_spatial_rfs_ecc(dd_ecc_params, dd_regr_model)
        # Add FIT:ed dendritic diameter for visualization
        (
            self.gc_df,
            self.dd_vs_ecc_to_viz["dd_fit_x"],
            self.dd_vs_ecc_to_viz["dd_fit_y"],
        ) = self._get_dd_fit_for_viz(self.gc_df)

        # Scale center and surround amplitude so that Gaussian volume is preserved
        self.gc_df = self._scale_both_amplitudes(
            self.gc_df
        )  # TODO - what was the purpose of this? Working retina uses ampl_c

        # At this point the fitted ellipse spatial receptive fields are ready. All parameters are in self.gc_df.
        # The positions are in the columns 'pos_ecc_mm', 'pos_polar_deg', 'ecc_group_idx', and the rf parameters in 'semi_xc',
        # 'semi_yc', 'xy_aspect_ratio', 'ampl_c', 'ampl_s', 'relat_sur_diam', 'relat_sur_ampl', 'orient_cen', 'den_diam_um'

        if self.model_type == "VAE":
            # Fit or load variational autoencoder to generate receptive fields
            self.retina_vae = RetinaVAE(
                self.gc_type,
                self.response_type,
                self.training_mode,
                self.context,
                save_tuned_models=True,
            )

            # -- Second, endow cells with spatial receptive fields using the generative variational autoencoder model
            nsamples = len(self.gc_df)
            img_processed, img_raw = self._get_generated_spatial_data(
                self.retina_vae, nsamples=nsamples
            )

            # Set self attribute for later visualization of image histograms
            self.gen_spat_img_to_viz = {
                "img_processed": img_processed,
                "img_raw": img_raw,
            }

            # Convert retinal positions (ecc, pol angle) to visual space positions in mm (x, y)
            ret_pos_ecc_mm = np.array(self.gc_df.pos_ecc_mm.values)
            # ret_pos_mm = self.pol2cart_df(self.gc_df)

            # Mean fitted dendritic diameter for the original experimental data
            data_dd_um = self.exp_spat_cen_sd_mm * 2 * 1000  # in micrometers
            data_um_per_pix = self.context.apricot_metadata["data_microm_per_pix"]

            # Upsample according to smallest rf diameter
            img_rfs, new_um_per_pix = self._get_upsampled_scaled_rfs(
                img_processed,
                dd_ecc_params,
                ret_pos_ecc_mm,
                data_um_per_pix,
                data_dd_um,
            )

            # Extract receptive field contours from the generated spatial data
            img_rfs_mask = self._get_rf_masks(img_rfs, mask_threshold=0.1)

            # Save the generated receptive fields
            output_path = self.context.output_folder
            self.data_io.save_generated_rfs(
                img_rfs, output_path, filename_stem="rf_values"
            )

            # Fit elliptical gaussians to the generated receptive fields
            (
                self.gen_stat_df,
                self.gen_spat_cen_sd,
                self.gen_spat_sur_sd,
                self.gen_spat_filt_to_viz,
                self.gen_spat_stat_to_viz,
                self.gc_vae_df,
            ) = Fit(
                self.context.apricot_data_folder,
                self.gc_type,
                self.response_type,
                spatial_data=img_rfs,
                fit_type="generated",
                new_um_per_pix=new_um_per_pix,
            ).get_generated_spatial_fits()

            # Update gc_vae_df to have the same columns as gc_df
            self.gc_vae_df = self._update_gc_vae_df(self.gc_vae_df, new_um_per_pix)

            # Add fitted VAE dendritic diameter for visualization
            (
                self.gc_vae_df,
                self.dd_vs_ecc_to_viz["dd_vae_x"],
                self.dd_vs_ecc_to_viz["dd_vae_y"],
            ) = self._get_dd_fit_for_viz(self.gc_vae_df)

            # Sum separate rf images onto one retina
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

            if self.rf_coverage_adjusted_to_1:
                img_rfs_adjusted, img_ret_adjusted = self._adjust_rf_coverage(
                    img_rfs,
                    img_rfs_mask,
                    img_ret,
                    img_ret_masked,
                    rf_lu_pix,
                    tolerate_error=0.01,
                )

                # Fit elliptical gaussians to the adjusted receptive fields
                (
                    self.gen_stat_df,
                    self.gen_spat_cen_sd,
                    self.gen_spat_sur_sd,
                    self.gen_spat_filt_to_viz,
                    self.gen_spat_stat_to_viz,
                    self.gc_vae_df,
                ) = Fit(
                    self.context.apricot_data_folder,
                    self.gc_type,
                    self.response_type,
                    spatial_data=img_rfs,
                    fit_type="generated",
                    new_um_per_pix=new_um_per_pix,
                ).get_generated_spatial_fits()

                # Update gc_vae_df to have the same columns as gc_df
                self.gc_vae_df = self._update_gc_vae_df(self.gc_vae_df, new_um_per_pix)

                # Add fitted VAE dendritic diameter for visualization
                (
                    self.gc_vae_df,
                    self.dd_vs_ecc_to_viz["dd_vae_x"],
                    self.dd_vs_ecc_to_viz["dd_vae_y"],
                ) = self._get_dd_fit_for_viz(self.gc_vae_df)

            else:
                img_rfs_adjusted = np.zeros_like(img_rfs)
                img_ret_adjusted = np.zeros_like(img_ret)

            self.gen_rfs_to_viz = {
                "img_rf": img_rfs,
                "img_rf_mask": img_rfs_mask,
                "img_rfs_adjusted": img_rfs_adjusted,
            }

            self.gen_ret_to_viz = {
                "img_ret": img_ret,
                "img_ret_masked": img_ret_masked,
                "img_ret_adjusted": img_ret_adjusted,
            }

            # Apply the spatial VAE model to df
            self.gc_df = self.gc_vae_df

        # -- Third, endow cells with temporal receptive fields
        self._create_temporal_receptive_fields()

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
            filepath = output_folder.joinpath(
                self.context.my_retina["mosaic_file_name"]
            )
        else:
            filepath = output_folder.joinpath(filename)

        print("Saving model mosaic to %s" % filepath)
        self.gc_df.to_csv(filepath)

    def show_exp_build_process(self, show_all_spatial_fits=False):
        """
        Show the process of building the mosaic
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_exp_build_process(
            self, show_all_spatial_fits=show_all_spatial_fits
        )

    def show_gen_exp_spatial_fit(self, n_samples=2):
        """
        Show the experimental (fitted) and generated spatial receptive fields
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_gen_exp_spatial_fit(self, n_samples=n_samples)

    def show_gen_exp_spatial_rf(self, ds_name="test_ds", n_samples=10):
        """
        Show the experimental (fitted) and generated spatial receptive fields
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_gen_exp_spatial_rf(self, ds_name=ds_name, n_samples=n_samples)

    def show_latent_tsne_space(self):
        """
        Show the latent space of the encoder
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_latent_tsne_space(self)

    def show_gen_spat_post_hist(self):
        """
        Show the original experimental spatial receptive fields and
        the generated spatial receptive fields before and after postprocessing
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_gen_spat_post_hist(self)

    def show_latent_space_and_samples(self):
        """
        Plot the latent samples on top of the estimated kde, one sublot
        for each successive two dimensions of latent_dim
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_latent_space_and_samples(self)

    def show_ray_experiment(self, ray_exp_name, this_dep_var, highlight_trial=None):
        """
        Show the ray experiment
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_ray_experiment(
            self, ray_exp_name, this_dep_var, highlight_trial=highlight_trial
        )

    def show_retina_img(self):
        """
        Show the VAE retina image
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_retina_img(self)

    def show_rf_imgs(self, n_samples=10):
        """
        Show the individual RFs of the VAE retina
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_rf_imgs(self, n_samples=n_samples)

    def show_rf_violinplot(self):
        """
        Show each RF and adjusted RF of the VAE retina as boxplots
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_rf_violinplot(self)
