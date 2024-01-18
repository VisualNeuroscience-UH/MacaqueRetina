# Numerical
import numpy as np
import numpy.ma as ma
import matplotlib.path as mplPath


import scipy.stats as stats
import scipy.optimize as opt
from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
from scipy.interpolate import griddata
from scipy.integrate import dblquad
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
from tqdm import tqdm

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
from dataclasses import dataclass


class Retina:
    """
    A class housing the retina-level parameters. Most values are defined in the project configuration file.

    Parameters
    ----------
    my_retina : dict
        A dictionary containing various parameters and settings for the retina.
        Defined in the project configuration file.

    Attributes
    ----------
    whole_ret_img : np.ndarray, computed
        An image representing the whole retina.
    whole_ret_lu_mm : np.ndarray, computed
        Coordinates of the left upper corner of the whole retina image in millimeters.
    cones_to_gcs_weights : np.ndarray, computed
        Weights mapping cones to ganglion cells.
    gc_placement_params : dict
        Parameters for placing ganglion cells.
    cone_placement_params : dict
        Parameters for placing cones.
    cone_general_params : dict
        Natural stimulus filtering parameters and cone to gcs connetcion parameters.
    rf_coverage_adjusted_to_1 : bool
        Indicates if receptive field coverage is adjusted to 1.
    dd_regr_model : object
        Regression model for dendritic diameter.
    deg_per_mm : float
        Degrees visual field per millimeter of the retina.
    ecc_lim_mm : np.ndarray
        Eccentricity limits in millimeters.
    ecc_limit_for_dd_fit_mm : float
        Eccentricity limit for dendritic density fit in millimeters.
    polar_lim_deg : np.ndarray
        Polar limits in degrees.
    gc_proportion : float
        Proportion of a specific type of ganglion cell based on type and response.

    Methods
    -------
    __init__(self, my_retina)
        Initializes the Retina instance with the given parameters.

    Raises
    ------
    ValueError
        If an unknown ganglion cell type is specified in `my_retina`.

    Notes
    -----
    - `my_retina` should contain keys like 'gc_placement_params', 'cone_placement_params',
      'cone_general_params', etc., with corresponding values.
    - The class includes several assertions to validate the input data types and lengths.
    - Proportions of different types of ganglion cells and their responses are calculated
      based on the provided parameters in `my_retina`.
    """

    def __init__(self, my_retina):
        # Computed downstream
        self.whole_ret_img = None
        self.whole_ret_lu_mm = None
        self.cones_to_gcs_weights = None

        self.gc_placement_params = my_retina["gc_placement_params"]
        self.cone_placement_params = my_retina["cone_placement_params"]
        self.cone_general_params = my_retina["cone_general_params"]

        self.rf_coverage_adjusted_to_1 = my_retina["rf_coverage_adjusted_to_1"]
        self.dd_regr_model = my_retina["dd_regr_model"]
        self.deg_per_mm = my_retina["deg_per_mm"]

        ecc_limits_deg = my_retina["ecc_limits_deg"]
        ecc_limit_for_dd_fit = my_retina["ecc_limit_for_dd_fit"]
        pol_limits_deg = my_retina["pol_limits_deg"]

        # Turn list to numpy array and deg to mm
        self.ecc_lim_mm = np.asarray(ecc_limits_deg) / self.deg_per_mm
        self.ecc_limit_for_dd_fit_mm = ecc_limit_for_dd_fit / self.deg_per_mm
        self.polar_lim_deg = np.asarray(pol_limits_deg)  # Turn list to numpy array

        # Assertions
        assert (
            isinstance(ecc_limits_deg, list) and len(ecc_limits_deg) == 2
        ), "Wrong type or length of eccentricity, aborting"
        assert (
            isinstance(pol_limits_deg, list) and len(pol_limits_deg) == 2
        ), "Wrong type or length of pol_limits_deg, aborting"

        model_density = my_retina["model_density"]
        assert model_density <= 1.0, "Density should be <=1.0, aborting"

        proportion_of_parasol_gc_type = my_retina["proportion_of_parasol_gc_type"]
        proportion_of_midget_gc_type = my_retina["proportion_of_midget_gc_type"]
        proportion_of_ON_response_type = my_retina["proportion_of_ON_response_type"]
        proportion_of_OFF_response_type = my_retina["proportion_of_OFF_response_type"]

        # Calculate self.gc_proportion from GC type specifications
        gc_type = my_retina["gc_type"].lower()
        response_type = my_retina["response_type"].lower()
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


@dataclass
class GanglionCellData:
    """
    A class to store and process data related to ganglion cell receptive fields.

    Attributes
    ----------
    gc_type : str
        Type of ganglion cell.
    response_type : str
        Type of response exhibited by the ganglion cell.
    spatial_model : str
        The spatial model used for the ganglion cell.
    temporal_model : str
        The temporal model used in the retina simulation.
    DoG_model : str
        Difference of Gaussian (DoG) model used.
    mask_threshold : float
        Threshold value for the receptive field center mask.
    n_units : int, computed
        The number of units.
    um_per_pix : float, computed
        Micrometers per pixel.
    pix_per_side : int, computed
        Number of pixels per side.
    um_per_side : float, computed
        Micrometers per side.
    img : np.ndarray, computed
        Receptive field image.
    img_mask : np.ndarray, computed
        Receptive field center mask.
    img_lu_pix : np.ndarray, computed
        Left upper corner of the receptive field image in pixels.
    X_grid_mm : np.ndarray, computed
        X grid in millimeters.
    Y_grid_mm : np.ndarray, computed
        Y grid in millimeters.
    cones_to_gcs_weights : np.ndarray, computed
        Weights mapping cones to ganglion cells.
    df : pandas.DataFrame
        DataFrame containing parameters of the ganglion cell mosaic.

        Columns present in all cases:
        - `pos_ecc_mm`: Eccentricity in mm
        - `pos_polar_deg`: Polar angle in degrees
        - `ecc_group_idx`: Eccentricity group index
        - `gc_scaling_factors`: Scaling factors for each eccentricity group
        - `zoom_factor`: Zoom factor for each eccentricity group
        - `xoc_pix`: X coordinate of center in pixels inside the rf image
        - `yoc_pix`: Y coordinate of center in pixels inside the rf image
        - `ampl_c`: Amplitude of center
        - `ampl_s`: Amplitude of surround
        - `den_diam_um`: Dendritic field diameter in micrometers
        - `center_mask_area_mm2`: Area of center mask in mm^2
        - `center_fit_area_mm2`: Area of center DoG fit in mm^2
        - `relat_sur_ampl`: Relative surround amplitude
        - `ampl_c_norm`: Normalized amplitude of center
        - `ampl_s_norm`: Normalized amplitude of surround
        - `tonic_drive` : Tonic drive of the ganglion cell

        In addition the following column names appear depending on the gc_type, spatial, temporal
        and DoG_models.

        Temporal parameters, fixed model
        - `n`: Order of the filters
        - `p1`: Normalization factor for the first filter
        - `p2`: Normalization factor for the second filter
        - `tau1`: Time constant of the first filter in ms
        - `tau2`: Time constant of the second filter in ms

        Temporal parameters, dynamic model.
        Midget has separate cen and sur filters.
        Parasol has gain control with params T0 and Chalf.
        - `A`, `A_cen`, `A_sur`: Gain of the model
        - `HS`, `HS_cen`, `HS_sur`: Strength of high-pass stage
        - `TS_cen`, `TS_sur`: Time constant of the high-pass stage
        - `TL`: Time constant of the low-pass stage
        - `NL`, `NL_cen`, `NL_sur`: Number of low-pass stages
        - `NLTL`, `NLTL_cen`, `NLTL_sur`: NL * TL
        - `deltaNLTL_sur`: NLTL_sur - NLTL_cen
        - `D`, `D_cen`: Initial delay before filtering
        - `Chalf`: Semi-saturation contrast of TS
        - `T0`: Time constant of the zero contrast

        Spatial parameters
        - `xos_pix`, `yos_pix`: X and Y coordinates of the surround in pixels
        - `offset`: Offset of the DoG model
        - `com_x_pix`, `com_y_pix`: X and Y coordinates of the center of mass in pixels
        - `orient_cen_rad`: Orientation of the center in radians
        - `orient_sur_rad`: Orientation of the surround in radians
        - `rad_c_mm`, `rad_c_pix`: Radius of the center in mm and pixels
        - `rad_s_mm`, `rad_s_pix`: Radius of the surround in mm and pixels
        - `relat_sur_diam`: Relative surround diameter
        - `semi_xc_mm`, `semi_xc_pix`: Semi-major axis of the center in mm and pixels
        - `semi_yc_mm`, `semi_yc_pix`: Semi-minor axis of the center in mm and pixels
        - `semi_xs_mm`, `semi_xs_pix`: Semi-major axis of the surround in mm and pixels
        - `semi_ys_mm`, `semi_ys_pix`: Semi-minor axis of the surround in mm and pixels
        - `xy_aspect_ratio`: Aspect ratio of the ellipse

    Methods
    -------
    __post_init__(self):
        Initializes the DataFrame with specified columns based on the ganglion cell properties.
    """

    gc_type: str
    response_type: str
    spatial_model: str
    temporal_model: str
    DoG_model: str
    mask_threshold: float

    # Computed values below
    n_units: int = None

    # Receptive field image related attributes
    um_per_pix: float = None
    pix_per_side: int = None
    um_per_side: float = None
    img: np.ndarray = None
    img_mask: np.ndarray = None
    img_lu_pix: np.ndarray = None
    X_grid_mm: np.ndarray = None
    Y_grid_mm: np.ndarray = None
    cones_to_gcs_weights: np.ndarray = None

    def __post_init__(self):
        columns = [
            "pos_ecc_mm",
            "pos_polar_deg",
            "ecc_group_idx",
            "gc_scaling_factors",
            "zoom_factor",
            "xoc_pix",
            "yoc_pix",
            "ampl_c",
            "ampl_s",
            "den_diam_um",
            "center_mask_area_mm2",
            "center_fit_area_mm2",
            "relat_sur_ampl",
            "ampl_c_norm",
            "ampl_s_norm",
            "tonic_drive",
        ]
        self.df = pd.DataFrame(columns=columns)


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

        self.device = self.context.device

        # Make or read fits
        my_retina = self.context.my_retina
        gc_type = my_retina["gc_type"]
        response_type = my_retina["response_type"]
        spatial_model = my_retina["spatial_model"]

        if spatial_model == "VAE":
            DoG_model = "ellipse_fixed"
        elif spatial_model == "FIT":
            DoG_model = my_retina["DoG_model"]

        self.fit.initialize(
            gc_type, response_type, fit_type="experimental", DoG_model=DoG_model
        )
        (
            self.exp_stat_df,
            self.exp_cen_radius_mm,
            self.exp_sur_radius_mm,
        ) = self.fit.get_experimental_fits(DoG_model)

        if "spatial_model" in my_retina and my_retina["spatial_model"] in ["VAE"]:
            self.training_mode = my_retina["training_mode"]

        self.spatial_rfs_file_filename = my_retina["spatial_rfs_file"]
        self.mosaic_filename = my_retina["mosaic_file"]
        self.rf_repulsion_params = my_retina["rf_repulsion_params"]

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

    def read_and_fit_unit_density_data(self, ret):
        """
        Read literature data from file and fit ganglion cell and cone density with respect to eccentricity.
        """

        def _sort_and_scale_density_data(eccentricity, density):
            """
            Sort and scale density data based on eccentricity.
            """
            index = np.argsort(eccentricity)
            return eccentricity[index], density[index] * 1e3  # Scale density

        def _process_density_data(filepaths):
            """
            Process density data from given filepaths.
            """
            unit_eccentricity = np.array([])
            unit_density = np.array([])
            for filepath in filepaths:
                density = self.data_io.get_data(filepath)
                _eccentricity = np.squeeze(density["Xdata"])
                _density = np.squeeze(density["Ydata"])
                unit_eccentricity = np.concatenate((unit_eccentricity, _eccentricity))
                unit_density = np.concatenate((unit_density, _density))

            # Sort and scale data
            unit_eccentricity, unit_density = _sort_and_scale_density_data(
                unit_eccentricity, unit_density
            )
            return unit_eccentricity, unit_density

        def _fit_density_data(eccentricity, density, unit_type):
            """
            Fit density data based on unit type.
            """
            if unit_type == "gc":
                this_function = self.gauss_plus_baseline_func
                p0 = [1000, 0, 2, np.min(density)]
            elif unit_type == "cone":
                this_function = self.double_exponential_func
                p0 = [0, -1, 0, 0]

            fit_parameters, _ = opt.curve_fit(
                this_function, eccentricity, density, p0=p0
            )

            # Save fit function and data for visualization
            setattr(self, f"{unit_type}_fit_function", this_function)
            self.project_data.construct_retina[f"{unit_type}_n_vs_ecc"] = {
                "fit_parameters": fit_parameters,
                "unit_eccentricity": eccentricity,
                "unit_density": density,
                "function": this_function,
            }

            return fit_parameters

        # Ganglion cell density data
        gc_filepaths = [self.context.literature_data_files["gc_density_fullpath"]]
        gc_eccentricity, gc_density = _process_density_data(gc_filepaths)
        gc_fit_parameters = _fit_density_data(gc_eccentricity, gc_density, "gc")

        # Cone density data
        cone_filepaths = [
            self.context.literature_data_files["cone_density1_fullpath"],
            self.context.literature_data_files["cone_density2_fullpath"],
        ]
        cone_eccentricity, cone_density = _process_density_data(cone_filepaths)
        cone_fit_parameters = _fit_density_data(cone_eccentricity, cone_density, "cone")

        ret.gc_density_params = gc_fit_parameters
        ret.cone_density_params = cone_fit_parameters

        return ret

    def _fit_dd_vs_ecc(self, ret, gc):
        """
        Fit dendritic field diameter with respect to eccentricity. Linear, quadratic and cubic fit.

        Returns
        -------
        dict
            dictionary containing dendritic diameter parameters and related data for visualization
        """

        dd_regr_model = ret.dd_regr_model
        ecc_limit_for_dd_fit_mm = ret.ecc_limit_for_dd_fit_mm

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

        # Quality control. Datasets separately for visualization
        assert dendr_diam_units["data1"] == ["mm", "um"]
        data_set_1_x = np.squeeze(dendr_diam1["Xdata"])
        data_set_1_y = np.squeeze(dendr_diam1["Ydata"])
        assert dendr_diam_units["data2"] == ["mm", "um"]
        data_set_2_x = np.squeeze(dendr_diam2["Xdata"])
        data_set_2_y = np.squeeze(dendr_diam2["Ydata"])
        assert dendr_diam_units["data3"] == ["deg", "um"]
        data_set_3_x = np.squeeze(dendr_diam3["Xdata"]) / ret.deg_per_mm
        data_set_3_y = np.squeeze(dendr_diam3["Ydata"])

        # Both datasets together
        data_all_x = np.concatenate((data_set_1_x, data_set_2_x, data_set_3_x))
        data_all_y = np.concatenate((data_set_1_y, data_set_2_y, data_set_3_y))

        # Limit eccentricities for central visual field studies to get better approximation at about 5 deg ecc (1mm)
        # x is eccentricity in mm
        # y is dendritic field diameter in micrometers
        data_all_x_index = data_all_x <= ecc_limit_for_dd_fit_mm
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
        dict_key = "{0}_{1}".format(gc.gc_type, dd_regr_model)

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

            # def linear_func(E, a, b):
            #     return a + E * b

            # fit_parameters, pcov = opt.curve_fit(
            #     linear_func, data_all_x, np.log10(data_all_y), p0=[1, 1]
            # )

            # a = fit_parameters[0]
            # b = fit_parameters[1]

            # a = (
            #     10**a
            # )  # Adjust the constant for later power fit of the form y = a * x^b

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
            "title": f"DF diam wrt ecc for {gc.gc_type} type, {dd_model_caption} dataset",
        }

        return dendr_diam_parameters

    def _fit_cone_noise_vs_freq(self, gc):
        """ """

        cone_noise = self.data_io.get_data(
            self.context.literature_data_files["cone_noise_fullpath"]
        )

        data_set_x = np.squeeze(cone_noise["Xdata"])
        data_set_y = np.squeeze(cone_noise["Ydata"])

        data_set_x_index = np.argsort(data_set_x)
        frequency_data = data_set_x[data_set_x_index]
        power_data = data_set_y[data_set_x_index]

        # # In linear scale their values are
        initial_guesses = [30, 0.002, 1.0, 0.04, 500, 30, 0.002]
        # Parameters are log_NL, log_TL, log_HS, log_TS, log_A0, log_M0, log_D.
        log_initial_guesses = [np.log(p) for p in initial_guesses]  # needs to be list

        # pdb.set_trace()
        # Log-transform the frequency and power data
        log_frequency_data = np.log(frequency_data)
        log_power_data = np.log(power_data)

        # Ensure no zero or negative values
        lower_bounds = [1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
        # Use np.inf if no upper bound is needed
        upper_bounds = [100, 0.1, 100, 0.1, 1000, 100, 0.1]

        # Take the log of bounds, except where the upper bound is np.inf
        log_lower_bounds = [np.log(low) for low in lower_bounds]
        log_upper_bounds = [np.log(up) for up in upper_bounds]

        bounds = (log_lower_bounds, log_upper_bounds)

        popt_log, pcov_log = opt.curve_fit(
            # RetinaMath.wrapper_log_space,
            self.wrapper_log_space,
            log_frequency_data,
            log_power_data,
            p0=log_initial_guesses,
            bounds=bounds,
        )

        gc.cone_noise_parameters = np.exp(popt_log)

        self.project_data.construct_retina["cone_noise_vs_freq"] = {
            "data_all_x": data_set_x,
            "data_all_y": data_set_y,
            "cone_noise_parameters": gc.cone_noise_parameters,
            "title": "cone_noise_vs_freq, asymmetric concave fit",
        }

        return gc

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

    def _generate_DoG_with_rf_coverage_one(self, ret, gc):
        """
        Generate Difference of Gaussians (DoG) model with full retinal field coverage.

        This function ensures full coverage of the retinal field (coverage factor = 1).
        It updates the `gc_df` dataframe with spatial parameters converted from pixels in
        orginal experimental data space to millimeters of final retina. It applies scaling
        for retinal coverage of one at the given eccentricity.
        """
        # Create all gc units from parameters fitted to experimental data
        n_cells = len(gc.df)
        data_microm_per_pix = self.context.apricot_metadata["data_microm_per_pix"]
        spatial_df = self.exp_stat_df[self.exp_stat_df["domain"] == "spatial"]
        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # Change units to mm. Here the scale reflects Chichilnisky data and they are at large eccentricity
        if gc.DoG_model in ["ellipse_independent", "ellipse_fixed"]:
            gc.df["semi_xc_mm"] = gc.df["semi_xc_pix"] * data_microm_per_pix / 1000
            gc.df["semi_yc_mm"] = gc.df["semi_yc_pix"] * data_microm_per_pix / 1000

        if gc.DoG_model == "ellipse_independent":
            # Add surround
            gc.df["semi_xs_mm"] = gc.df["semi_xs_pix"] * data_microm_per_pix / 1000
            gc.df["semi_ys_mm"] = gc.df["semi_ys_pix"] * data_microm_per_pix / 1000

        if gc.DoG_model == "circular":
            gc.df["rad_c_mm"] = gc.df["rad_c_pix"] * data_microm_per_pix / 1000
            gc.df["rad_s_mm"] = gc.df["rad_s_pix"] * data_microm_per_pix / 1000

        # Calculate RF diameter scaling factor for all ganglion cells. The surround in
        # ellipse_indenpendent model has the same scaling factor as the center.
        if gc.DoG_model in ["ellipse_independent", "ellipse_fixed"]:
            area_rfs_cen_mm2 = np.pi * gc.df["semi_xc_mm"] * gc.df["semi_yc_mm"]

        elif gc.DoG_model == "circular":
            area_rfs_cen_mm2 = np.pi * gc.df["rad_c_mm"] ** 2

        """
        The area_of_rf contains area for all model units. Its sum must fill the whole area (coverage factor = 1).
        We do it separately for each ecc sector, step by step, to keep coverage factor at 1 despite changing gc density with ecc
        r_scaled = sqrt( (area_scaled / area) * r^2 ) => r_scaling_factor = sqrt( (area_scaled / area) )
        """
        # Calculate area scaling factors for each eccentricity group
        area_scaling_factors_coverage1 = np.zeros(area_rfs_cen_mm2.shape)
        for index, sector_area_mm2 in enumerate(ret.sector_surface_areas_mm2):
            area_scaling_factor = (sector_area_mm2) / np.sum(
                area_rfs_cen_mm2[gc.df["ecc_group_idx"] == index]
            )

            area_scaling_factors_coverage1[
                gc.df["ecc_group_idx"] == index
            ] = area_scaling_factor

        radius_scaling_factors_coverage_1 = np.sqrt(area_scaling_factors_coverage1)

        # Save scaling factors for later working retina computations
        gc.df["gc_scaling_factors"] = radius_scaling_factors_coverage_1

        # Apply scaling factors.
        if gc.DoG_model in ["ellipse_independent", "ellipse_fixed"]:
            semi_xc = radius_scaling_factors_coverage_1 * gc.df["semi_xc_mm"]

            semi_yc = radius_scaling_factors_coverage_1 * gc.df["semi_yc_mm"]

            gc.df["semi_xc_mm"] = semi_xc
            gc.df["semi_yc_mm"] = semi_yc

        if gc.DoG_model == "ellipse_independent":
            # Add surround
            semi_xs = radius_scaling_factors_coverage_1 * gc.df["semi_xs_mm"]

            semi_ys = radius_scaling_factors_coverage_1 * gc.df["semi_ys_mm"]

            gc.df["semi_xs_mm"] = semi_xs
            gc.df["semi_ys_mm"] = semi_ys

        if gc.DoG_model == "circular":
            rad_c = radius_scaling_factors_coverage_1 * gc.df["rad_c_mm"]
            gc.df["rad_c_mm"] = rad_c

            rad_s = radius_scaling_factors_coverage_1 * gc.df["rad_s_mm"]
            gc.df["rad_s_mm"] = rad_s

        return gc

    def _generate_DoG_with_rf_from_literature(self, gc):
        """
        Generate Difference of Gaussians (DoG) model with dendritic field sizes from literature.

        Places all ganglion cell spatial parameters to ganglion cell object dataframe gc.df

        At return, all units are in mm unless stated otherwise in the the column name
        """

        gc_scaling_factors = gc.df["gc_scaling_factors"]

        # Create all gc units from parameters fitted to experimental data
        n_cells = len(gc.df)
        spatial_df = self.exp_stat_df[self.exp_stat_df["domain"] == "spatial"]
        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        um_per_pixel = self.context.apricot_metadata["data_microm_per_pix"]
        if gc.DoG_model in ["ellipse_independent", "ellipse_fixed"]:
            # Scale semi_x to virtual pix at its actual eccentricity
            semi_xc_pix_eccscaled = gc.df["semi_xc_pix"] * gc_scaling_factors

            # Scale semi_x to mm
            gc.df["semi_xc_mm"] = semi_xc_pix_eccscaled * um_per_pixel / 1000

            # Scale semi_y to pix at its actual eccentricity
            semi_yc_pix_eccscaled = gc.df["semi_yc_pix"] * gc_scaling_factors

            # Scale semi_y to mm
            gc.df["semi_yc_mm"] = semi_yc_pix_eccscaled * um_per_pixel / 1000
        if gc.DoG_model == "ellipse_independent":
            # Surround
            # Scale semi_x to pix at its actual eccentricity
            semi_xs_pix_eccscaled = gc.df["semi_xs_pix"] * gc_scaling_factors

            # Scale semi_x to mm
            gc.df["semi_xs_mm"] = semi_xs_pix_eccscaled * um_per_pixel / 1000

            # Scale semi_y to pix at its actual eccentricity
            semi_ys_pix_eccscaled = gc.df["semi_ys_pix"] * gc_scaling_factors

            # Scale semi_y to mm
            gc.df["semi_ys_mm"] = semi_ys_pix_eccscaled * um_per_pixel / 1000

        elif gc.DoG_model == "circular":
            # Scale rad_c to pix at its actual eccentricity
            rad_c_pix_eccscaled = gc.df["rad_c_pix"] * gc_scaling_factors

            # Scale rad_c to mm
            gc.df["rad_c_mm"] = rad_c_pix_eccscaled * um_per_pixel / 1000

            # Same for rad_s
            rad_s_pix_eccscaled = gc.df["rad_s_pix"] * gc_scaling_factors
            gc.df["rad_s_mm"] = rad_s_pix_eccscaled * um_per_pixel / 1000

        return gc

    def _densfunc(self, r, d0, beta):
        return d0 * (1 + beta * r) ** (-2)

    # GC placement functions
    def _initialize_positions_by_group(self, ret):
        """
        Initialize cell positions based on grouped eccentricities.

        Parameters
        ----------
        gc_density_params : tuple
            Parameters for the density function used to calculate cell density
            at a given eccentricity.

        Returns
        -------
        eccentricity_groups : list of ndarray
            A list of arrays where each array contains group indices representing
            eccentricity steps for cells.

        gc_initial_pos : list of ndarray
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
        gc_density_params = ret.gc_density_params
        cone_density_params = ret.cone_density_params

        # Loop for reasonable delta ecc to get correct density in one hand and good cell distribution from the algo on the other
        # Lets fit close to 0.1 mm intervals, which makes sense up to some 15 deg. Thereafter longer jumps would do fine.
        assert (
            ret.ecc_lim_mm[0] < ret.ecc_lim_mm[1]
        ), "Radii in wrong order, give [min max], aborting"
        eccentricity_in_mm_total = ret.ecc_lim_mm
        fit_interval = 0.1  # mm
        n_steps = math.ceil(np.ptp(eccentricity_in_mm_total) / fit_interval)
        eccentricity_steps = np.linspace(
            eccentricity_in_mm_total[0], eccentricity_in_mm_total[1], 1 + n_steps
        )

        angle_deg = np.ptp(ret.polar_lim_deg)  # The angle_deg is now == max theta_deg

        eccentricity_groups = []
        areas_all_mm2 = []
        gc_initial_pos = []
        gc_density_all = []
        cone_initial_pos = []
        cone_density_all = []
        for group_idx in range(len(eccentricity_steps) - 1):
            min_ecc = eccentricity_steps[group_idx]
            max_ecc = eccentricity_steps[group_idx + 1]
            avg_ecc = (min_ecc + max_ecc) / 2

            gc_density_group = self.gc_fit_function(avg_ecc, *gc_density_params)

            cone_density_group = self.cone_fit_function(avg_ecc, *cone_density_params)

            # Calculate area for this eccentricity group
            sector_area_remove = self.sector2area_mm2(min_ecc, angle_deg)
            sector_area_full = self.sector2area_mm2(max_ecc, angle_deg)
            sector_surface_area = sector_area_full - sector_area_remove  # in mm2

            # collect sector area for each ecc step
            areas_all_mm2.append(sector_surface_area)

            gc_units = math.ceil(
                sector_surface_area * gc_density_group * ret.gc_proportion
            )
            gc_positions = self._random_positions_within_group(
                min_ecc, max_ecc, gc_units, ret.polar_lim_deg
            )
            eccentricity_groups.append(np.full(gc_units, group_idx))
            gc_density_all.append(
                np.full(gc_units, gc_density_group * ret.gc_proportion)
            )
            gc_initial_pos.append(gc_positions)

            cone_units = math.ceil(sector_surface_area * cone_density_group)
            # cone_positions = self._random_positions_within_group(
            #     min_ecc, max_ecc, cone_units
            # )
            cone_positions = self._hexagonal_positions_within_group(
                min_ecc, max_ecc, cone_units, ret.polar_lim_deg
            )

            cone_density_all.append(np.full(cone_units, cone_density_group))
            cone_initial_pos.append(cone_positions)

        gc_density = np.concatenate(gc_density_all)
        cone_density = np.concatenate(cone_density_all)

        return (
            eccentricity_groups,
            areas_all_mm2,
            gc_initial_pos,
            gc_density,
            cone_initial_pos,
            cone_density,
        )

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

    def _apply_force_based_layout(
        self, ret, all_positions, unit_density, unit_placement_params
    ):
        """
        Apply a force-based layout on the given positions.

        Parameters
        ----------
        all_positions : list or ndarray
            Initial positions of nodes.
        unit_density : float
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

        n_iterations = unit_placement_params["n_iterations"]
        change_rate = unit_placement_params["change_rate"]
        unit_repulsion_stregth = unit_placement_params["unit_repulsion_stregth"]
        unit_distance_threshold = unit_placement_params["unit_distance_threshold"]
        diffusion_speed = unit_placement_params["diffusion_speed"]
        border_repulsion_stength = unit_placement_params["border_repulsion_stength"]
        border_distance_threshold = unit_placement_params["border_distance_threshold"]
        show_placing_progress = unit_placement_params["show_placing_progress"]
        show_skip_steps = unit_placement_params["show_skip_steps"]

        if show_placing_progress is True:
            # Init plotting
            fig_args = self.viz.show_unit_placement_progress(
                all_positions,
                ecc_lim_mm=ret.ecc_lim_mm,
                polar_lim_deg=ret.polar_lim_deg,
                init=True,
            )

        unit_distance_threshold = torch.tensor(unit_distance_threshold).to(self.device)
        unit_repulsion_stregth = torch.tensor(unit_repulsion_stregth).to(self.device)
        diffusion_speed = torch.tensor(diffusion_speed).to(self.device)
        n_iterations = torch.tensor(n_iterations).to(self.device)
        unit_density = torch.tensor(unit_density).to(self.device)

        rep = torch.tensor(border_repulsion_stength).to(self.device)
        dist_th = torch.tensor(border_distance_threshold).to(self.device)

        original_positions = deepcopy(all_positions)
        positions = torch.tensor(
            all_positions, requires_grad=True, dtype=torch.float64, device=self.device
        )
        change_rate = torch.tensor(change_rate).to(self.device)
        optimizer = torch.optim.Adam([positions], lr=change_rate, betas=(0.95, 0.999))

        ecc_lim_mm = torch.tensor(ret.ecc_lim_mm).to(self.device)
        polar_lim_deg = torch.tensor(ret.polar_lim_deg).to(self.device)
        boundary_polygon = self.viz.boundary_polygon(
            ecc_lim_mm.cpu().numpy(), polar_lim_deg.cpu().numpy()
        )

        # Adjust unit_distance_threshold and diffusion speed with density of the units
        # This is a technical trick to get good spread for different densities
        # The 1 mm ecc for parasol provides 952 units/mm2 density. This is the reference density.
        adjusted_distance_threshold = unit_distance_threshold * (952 / unit_density)
        adjusted_diffusion_speed = diffusion_speed * (952 / unit_density)

        for iteration in torch.range(0, n_iterations):
            optimizer.zero_grad()
            # Repulsive force between nodes
            diff = positions[None, :, :] - positions[:, None, :]
            dist = torch.norm(diff, dim=-1, p=2) + 1e-9

            # Clip minimum distance to avoid very high repulsion
            dist = torch.clamp(dist, min=0.00001)
            # Clip max to inf (zero repulsion) above a certain distance
            dist[dist > adjusted_distance_threshold] = torch.inf
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

            gc_diffusion_speed_reshaped = adjusted_diffusion_speed.view(-1, 1)
            new_data = (
                torch.randn_like(positions) * gc_diffusion_speed_reshaped
                + positions_delta
            )

            positions.data = positions + new_data

            if show_placing_progress is True:
                # Update the visualization every 100 iterations for performance (or adjust as needed)
                if iteration % show_skip_steps == 0:
                    positions_cpu = positions.detach().cpu().numpy()
                    self.viz.show_unit_placement_progress(
                        original_positions=original_positions,
                        positions=positions_cpu,
                        iteration=iteration,
                        boundary_polygon=boundary_polygon,
                        **fig_args,
                    )

        if show_placing_progress is True:
            plt.ioff()  # Turn off interactive mode

        return positions.detach().cpu().numpy()

    def _apply_voronoi_layout(self, ret, all_positions, unit_placement_params):
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
        n_iterations = unit_placement_params["n_iterations"]
        change_rate = unit_placement_params["change_rate"]
        show_placing_progress = unit_placement_params["show_placing_progress"]
        show_skip_steps = unit_placement_params["show_skip_steps"]

        if show_placing_progress:
            fig_args = self.viz.show_unit_placement_progress(
                all_positions,
                ecc_lim_mm=ret.ecc_lim_mm,
                polar_lim_deg=ret.polar_lim_deg,
                init=True,
            )

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

        ecc_lim_mm = ret.ecc_lim_mm
        polar_lim_deg = ret.polar_lim_deg
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
                torch.tensor(ret.ecc_lim_mm),
                torch.tensor(ret.polar_lim_deg),
            )

            positions = (positions_torch + position_deltas).numpy()

            if show_placing_progress and iteration % show_skip_steps == 0:
                self.viz.show_unit_placement_progress(
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

    def _random_positions_within_group(self, min_ecc, max_ecc, n_units, polar_lim_deg):
        eccs = np.random.uniform(min_ecc, max_ecc, n_units)
        angles = np.random.uniform(polar_lim_deg[0], polar_lim_deg[1], n_units)
        return np.column_stack((eccs, angles))

    def _hexagonal_positions_within_group(
        self, min_ecc, max_ecc, n_units, polar_lim_deg
    ):
        delta_ecc = max_ecc - min_ecc
        mean_ecc = (max_ecc + min_ecc) / 2

        # Calculate polar coords in mm at mean ecc
        x0, y0 = self.pol2cart(mean_ecc, polar_lim_deg[0])
        x1, y1 = self.pol2cart(mean_ecc, polar_lim_deg[1])
        delta_pol = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # n_pol = np.sqrt(n_units * delta_pol / (np.sqrt(3) * delta_ecc))
        n_pol = int(np.ceil(np.sqrt(n_units * (delta_pol / delta_ecc))))
        n_ecc = int(np.ceil(n_units / n_pol))

        # Generate evenly spaced values for eccentricity and angle
        eccs = np.linspace(min_ecc, max_ecc, n_ecc, endpoint=True)
        angles = np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_pol, endpoint=False)

        # Create a meshgrid of all combinations of eccs and angles
        eccs_grid, angles_grid = np.meshgrid(eccs, angles, indexing="ij")
        # Offset every other row by half the distance between columns
        for i in range(n_ecc):
            if i % 2 == 1:  # Check if the row is odd
                angles_grid[i, :] += (angles[1] - angles[0]) / 2

        # Reshape the grids and combine them into a single array
        positions = np.column_stack((eccs_grid.ravel(), angles_grid.ravel()))

        return positions[:n_units]

    def _optimize_positions(
        self, ret, initial_positions, unit_density, unit_placement_params
    ):
        # Merge the Groups
        all_positions = np.vstack(initial_positions)
        all_positions_tuple = self.pol2cart(all_positions[:, 0], all_positions[:, 1])
        all_positions_mm = np.column_stack(all_positions_tuple)

        # Optimize positions for ganglion cells
        optim_algorithm = unit_placement_params["algorithm"]
        if optim_algorithm == None:
            # Initial random placement.
            # Use this for testing/speed/nonvarying placements.
            optimized_positions = all_positions
            optimized_positions_mm = all_positions_mm
        else:
            if optim_algorithm == "force":
                # Apply Force Based Layout Algorithm with Boundary Repulsion
                optimized_positions_mm = self._apply_force_based_layout(
                    ret, all_positions_mm, unit_density, unit_placement_params
                )
            elif optim_algorithm == "voronoi":
                # Apply Voronoi-based Layout with Loyd's Relaxation
                optimized_positions_mm = self._apply_voronoi_layout(
                    ret, all_positions_mm, unit_placement_params
                )
            optimized_positions_tuple = self.cart2pol(
                optimized_positions_mm[:, 0], optimized_positions_mm[:, 1]
            )
            optimized_positions = np.column_stack(optimized_positions_tuple)

        return optimized_positions, optimized_positions_mm

    def _link_cone_noise_units_to_gcs(self, ret, gc):
        """
        Connect cones to ganglion cells for shared cone noise.
        """

        print("Connecting cones to ganglion cells for shared cone noise...")

        cone_pos_mm = ret.cone_optimized_positions_mm
        x_mm, y_mm = self.pol2cart(
            gc.df[["pos_ecc_mm"]].values, gc.df[["pos_polar_deg"]].values
        )
        gc_pos_mm = np.column_stack((x_mm, y_mm))

        # distances = distance.cdist(cone_pos_mm, gc_pos_mm, metric="euclidean")

        if gc.gc_type == "parasol":
            sd_cone = ret.cone_general_params["cone2gc_parasol"] / 1000
        elif gc.gc_type == "midget":
            sd_cone = ret.cone_general_params["cone2gc_midget"] / 1000
        cutoff_distance = ret.cone_general_params["cone2gc_cutoff_SD"] * sd_cone

        weights = np.zeros((len(cone_pos_mm), len(gc_pos_mm)))

        n_cones = cone_pos_mm.shape[0]
        n_gcs = gc_pos_mm.shape[0]

        # Normalize center activation to probability distribution
        img_cen = gc.img * gc.img_mask
        img_prob = img_cen / np.sum(img_cen, axis=(1, 2))[:, None, None]

        for i in tqdm(
            range(len(cone_pos_mm)),
            desc=f"Calculating {n_cones} x {n_gcs} probabilities",
        ):
            mean_cone = cone_pos_mm[i]
            dist_x_mtx = gc.X_grid_mm - mean_cone[0]
            dist_y_mtx = gc.Y_grid_mm - mean_cone[1]
            dist_mtx = np.sqrt(dist_x_mtx**2 + dist_y_mtx**2)

            # Drop weight as a Gaussian function of distance with sd = sd_cone
            probability = np.exp(-((dist_mtx / sd_cone) ** 2))
            probability[dist_mtx > cutoff_distance] = 0

            weights_mtx = probability * img_prob
            weights[i, :] = weights_mtx.sum(axis=(1, 2))

        self.project_data.construct_retina["cones_to_gcs"] = {
            "cone_pos_mm": cone_pos_mm,
            "gc_pos_mm": gc_pos_mm,
            "weights": weights,
            "X_grid_mm": gc.X_grid_mm,
            "Y_grid_mm": gc.Y_grid_mm,
            "gc_img_mask": gc.img_mask,
        }

        gc.cones_to_gcs_weights = weights

        return gc

    def _place_units(self, ret, gc):
        # Initial Positioning by Group
        print("\nPlacing units...\n")
        (
            eccentricity_groups,
            sector_surface_areas_mm2,
            gc_initial_pos,
            gc_density,
            cone_initial_pos,
            cone_density,
        ) = self._initialize_positions_by_group(ret)

        # Optimize positions
        gc_optimized_pos, gc_optimized_positions_mm = self._optimize_positions(
            ret, gc_initial_pos, gc_density, ret.gc_placement_params
        )
        cone_optimized_pos, cone_optimized_positions_mm = self._optimize_positions(
            ret, cone_initial_pos, cone_density, ret.cone_placement_params
        )

        # Assign ganglion cell positions to gc_df
        gc.df["pos_ecc_mm"] = gc_optimized_pos[:, 0]
        gc.df["pos_polar_deg"] = gc_optimized_pos[:, 1]
        gc.df["ecc_group_idx"] = np.concatenate(eccentricity_groups)

        ret.sector_surface_areas_mm2 = sector_surface_areas_mm2

        # Cones will be attached to gcs after the final position of gcs is known after
        # repulsion.
        ret.cone_optimized_positions_mm = cone_optimized_positions_mm

        return ret, gc

    # temporal filter and tonic drive functions
    def _create_fixed_temporal_rfs(self, gc):
        n_cells = len(gc.df)
        temporal_df = self.exp_stat_df[self.exp_stat_df["domain"] == "temporal"]
        for param_name, row in temporal_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        return gc

    def _read_temporal_statistics_benardete_kaplan(self, gc):
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

        cell_type = gc.gc_type

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
                "Mean",
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
                "Mean",
            ]

        col_names = ["Minimum", "Maximum", "Median", "Mean", "SD", "SEM"]
        distrib_params = np.zeros((len(temporal_model_parameters), 3))
        response_type = gc.response_type.upper()

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
            "suptitle": gc.gc_type + " " + gc.response_type,
            "all_data_fits_df": all_data_fits_df,
        }

        self.exp_stat_df = all_data_fits_df

        return temporal_exp_stat_df

    def _create_dynamic_temporal_rfs(self, gc):
        n_cells = len(gc.df)

        temporal_bk_stat_df = self._read_temporal_statistics_benardete_kaplan(gc)
        for param_name, row in temporal_bk_stat_df.iterrows():
            shape, loc, scale, distribution, *_ = row
            gc.df[param_name] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # For midget type, get snr-weighted average of A_cen and A_sur
        if gc.gc_type == "midget":
            snr_cen = temporal_bk_stat_df.loc["A_cen", "snr"]
            snr_sur = temporal_bk_stat_df.loc["A_sur", "snr"]
            weight_cen = snr_cen / (snr_cen + snr_sur)
            weight_sur = snr_sur / (snr_cen + snr_sur)
            gc.df["A"] = gc.df["A_cen"] * weight_cen + gc.df["A_sur"] * weight_sur

        return gc

    def _scale_both_amplitudes(self, gc):
        """
        Scale center amplitude to center volume of one.
        Scale surround amplitude also to center volume of one.
        Volume of 2D Gaussian = 2 * pi * sigma_x*sigma_y

        Second step of scaling is done before convolving with the stimulus.
        """
        if gc.DoG_model in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            cen_vol_mm3 = 2 * np.pi * gc.df["semi_xc_mm"] * gc.df["semi_yc_mm"]
        elif gc.DoG_model == "circular":
            cen_vol_mm3 = np.pi * gc.df["rad_c_mm"] ** 2

        gc.df["relat_sur_ampl"] = gc.df["ampl_s"] / gc.df["ampl_c"]

        # This sets center volume (sum of all pixel values in data, after fitting) to one
        ampl_c_norm = 1 / cen_vol_mm3
        ampl_s_norm = gc.df["relat_sur_ampl"] / cen_vol_mm3

        gc.df["ampl_c_norm"] = ampl_c_norm
        gc.df["ampl_s_norm"] = ampl_s_norm

        return gc

    def _create_tonic_drive(self, gc):
        """
        Create tonic drive for each cell.
        """
        tonic_df = self.exp_stat_df[self.exp_stat_df["domain"] == "tonic"]
        for param_name, row in tonic_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self._get_random_samples(
                shape, loc, scale, len(gc.df), distribution
            )

        return gc

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

    def _get_gc_img_params(self, ret, gc, ecc2dd_params):
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

        gc_pos_ecc_mm = np.array(gc.df.pos_ecc_mm.values)
        exp_um_per_pix = self.context.apricot_metadata["data_microm_per_pix"]
        # Mean fitted dendritic diameter for the original experimental data
        exp_dd_um = self.exp_cen_radius_mm * 2 * 1000  # in micrometers
        exp_pix_per_side = self.context.apricot_metadata["data_spatialfilter_height"]

        # Get rf diameter vs eccentricity
        dict_key = "{0}_{1}".format(gc.gc_type, ret.dd_regr_model)
        parameters = ecc2dd_params[dict_key]

        if ret.dd_regr_model in ["linear", "quadratic", "cubic"]:
            lit_dd_at_gc_ecc_um = np.polyval(
                [
                    parameters.get("cube", 0),
                    parameters.get("square", 0),
                    parameters.get("slope", 0),
                    parameters.get("intercept", 0),
                ],
                gc_pos_ecc_mm,
            )
        elif ret.dd_regr_model == "exponential":
            lit_dd_at_gc_ecc_um = parameters.get("constant", 0) + np.exp(
                gc_pos_ecc_mm / parameters.get("lamda", 0)
            )
        elif ret.dd_regr_model == "loglog":
            # Calculate dendritic diameter from the power law relationship
            # D = a * E^b, where E is the eccentricity and D is the dendritic diameter
            a = parameters["a"]
            b = parameters["b"]
            # Eccentricity in mm, dendritic diameter in um
            lit_dd_at_gc_ecc_um = a * np.power(gc_pos_ecc_mm, b)
        else:
            raise ValueError(f"Unknown dd_regr_model: {ret.dd_regr_model}")

        # Assuming the experimental data reflects the eccentricity for
        # VAE mtx generation
        gc_scaling_factors = lit_dd_at_gc_ecc_um / exp_dd_um
        gc_um_per_pix = gc_scaling_factors * exp_um_per_pix

        # Get min and max values of gc_um_per_pix
        new_um_per_pix = np.min(gc_um_per_pix)
        max_um_per_pix = np.max(gc_um_per_pix)

        # Get new img stack sidelength whose pixel size = min(gc_um_per_pix),
        new_pix_per_side = int(
            np.round((max_um_per_pix / new_um_per_pix) * exp_pix_per_side)
        )

        # Save scaling factors to gc_df for VAE model type
        gc.df["gc_scaling_factors"] = gc_scaling_factors
        # The pixel grid will be fixed for all units, but the unit eccentricities vary.
        # Thus we need to zoom units to the same size.
        gc.df["zoom_factor"] = gc_um_per_pix / new_um_per_pix

        # Set gc img parameters
        gc.um_per_pix = new_um_per_pix
        gc.pix_per_side = new_pix_per_side
        gc.um_per_side = new_um_per_pix * new_pix_per_side

        return gc

    def _get_resampled_scaled_gc_img(
        self,
        rfs,
        pix_per_side,
        zoom_factor,
    ):
        # Resample all images to new img stack. Use scipy.ndimage.zoom,
        img_upsampled = np.zeros((len(rfs), pix_per_side, pix_per_side))

        for i, img in enumerate(rfs):
            # Pad the image with zeros to achieve the new dimensions
            # If pix_per_side - img.shape[0] is even:
            if (pix_per_side - img.shape[0]) % 2 == 0:
                padding = int((pix_per_side - img.shape[0]) / 2)
            elif (pix_per_side - img.shape[0]) % 2 == 1:
                padding = (
                    int((pix_per_side - img.shape[0]) / 2),
                    int((pix_per_side - img.shape[0]) / 2) + 1,
                )  # (before, after)

            img_padded = np.pad(
                img, pad_width=padding, mode="constant", constant_values=0
            )

            # Upsample the padded image
            # img_temp = ndimage.zoom(img_padded, 1.0)
            img_temp = ndimage.zoom(img_padded, zoom_factor[i])

            # Crop the upsampled image to the new dimensions
            crop_length = pix_per_side / 2
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

    def _get_dd_in_um(self, gc):
        # Add diameters to dataframe
        if gc.DoG_model == "circular":
            den_diam_um_s = gc.df["rad_c_mm"] * 2 * 1000
        elif gc.DoG_model in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            den_diam_um_s = pd.Series(
                self.ellipse2diam(
                    gc.df["semi_xc_mm"].values * 1000,
                    gc.df["semi_yc_mm"].values * 1000,
                )
            )

        gc.df["den_diam_um"] = den_diam_um_s

        return gc

    def _update_vae_gc_df(
        self,
        ret,
        gc,
        gc_df_in,
    ):
        """
        Update gc_vae_df to have the same columns as gc_df with corresponding values.
        Update the remaining pixel values to mm, unless unit in is in the column name
        After repulsion have shifted the rfs, the eccentricity and polar angle of the receptive field center
        are calculated from the new_gc_img_lu_pix and whole_ret_lu_mm.
        """

        _df = gc_df_in.reindex(columns=gc.df.columns)

        # Calculate the eccentricity and polar angle of the receptive field center from the new_gc_img_lu_pix
        # and whole_ret_lu_mm
        xoc_mm = gc_df_in.xoc_pix * gc.um_per_pix / 1000
        yoc_mm = gc_df_in.yoc_pix * gc.um_per_pix / 1000
        rf_lu_mm = gc.img_lu_pix * gc.um_per_pix / 1000

        # Left upper corner of retina + left upper corner of receptive field relative to the left upper corner of retina
        # + offset of receptive field center from the left upper corner of receptive field
        x_mm = ret.whole_ret_lu_mm[0] + rf_lu_mm[:, 0] + xoc_mm
        y_mm = ret.whole_ret_lu_mm[1] - rf_lu_mm[:, 1] - yoc_mm
        (pos_ecc_mm, pos_polar_deg) = self.cart2pol(x_mm, y_mm)

        _df["pos_ecc_mm"] = pos_ecc_mm
        _df["pos_polar_deg"] = pos_polar_deg

        if gc.DoG_model == "ellipse_fixed":
            _df["relat_sur_diam"] = gc_df_in["relat_sur_diam"]

        if gc.DoG_model == "ellipse_independent":
            # Scale factor for semi_x and semi_y from pix to millimeters
            _df["semi_xs_mm"] = gc_df_in["semi_xs_pix"] * gc.um_per_pix / 1000
            _df["semi_ys_mm"] = gc_df_in["semi_ys_pix"] * gc.um_per_pix / 1000
            _df["orient_sur_rad"] = gc_df_in["orient_sur_rad"]
            _df["xos_pix"] = gc_df_in["xos_pix"]
            _df["yos_pix"] = gc_df_in["yos_pix"]

        if gc.DoG_model == "circular":
            # Scale factor for rad_c and rad_s from pix to millimetersq
            _df["rad_c_mm"] = gc_df_in["rad_c_pix"] * gc.um_per_pix / 1000
            _df["rad_s_mm"] = gc_df_in["rad_s_pix"] * gc.um_per_pix / 1000
            # dendritic diameter in micrometers
            _df["den_diam_um"] = _df["rad_c_mm"] * 2 * 1000  # um

        elif gc.DoG_model in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            # Scale factor for semi_x and semi_y from pix to millimeters
            _df["semi_xc_mm"] = gc_df_in["semi_xc_pix"] * gc.um_per_pix / 1000
            _df["semi_yc_mm"] = gc_df_in["semi_yc_pix"] * gc.um_per_pix / 1000
            # dendritic diameter in micrometers
            _df["den_diam_um"] = self.ellipse2diam(
                _df["semi_xc_mm"].values * 1000,
                _df["semi_yc_mm"].values * 1000,
            )

            _df["orient_cen_rad"] = gc_df_in["orient_cen_rad"]

            _df["xy_aspect_ratio"] = _df["semi_yc_mm"] / _df["semi_xc_mm"]

        _df["ampl_c"] = gc_df_in["ampl_c"]
        _df["ampl_s"] = gc_df_in["ampl_s"]

        _df["xoc_pix"] = gc_df_in["xoc_pix"]
        _df["yoc_pix"] = gc_df_in["yoc_pix"]

        gc.df = _df

        return gc

    def _get_full_retina_with_rf_images(self, ret, gc, gc_img):
        """
        Build one retina image with all receptive fields. The retina sector is first rotated to
        be symmetric around the horizontal meridian. Then the image is cropped to the smallest
        rectangle that contains all receptive fields. The image is then rotated back to the original
        orientation.

        Parameters
        ----------
        gc_img : numpy.ndarray
            3D numpy array of receptive field images. The shape of the array should be (N, H, W).
        um_per_pix : float
            The number of micrometers per pixel in the gc_img.
        df : pandas.DataFrame
            DataFrame with gc parameters.
        """
        # gc_img = gc.img
        um_per_pix = gc.um_per_pix
        df = gc.df

        ecc_lim_mm = ret.ecc_lim_mm
        polar_lim_deg = ret.polar_lim_deg

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
        # rf imgs at the borders later on. Assuming rectangular rf images.
        assert (
            gc_img.shape[1] == gc_img.shape[2]
        ), "rf images are not square, aborting..."
        pix_per_side = gc_img.shape[1]
        pad_size_mm = pix_per_side * um_per_pix / 1000

        min_x_mm_im = min_x_mm - pad_size_mm
        max_x_mm_im = max_x_mm + pad_size_mm
        min_y_mm_im = min_y_mm - pad_size_mm
        max_y_mm_im = max_y_mm + pad_size_mm

        # Get retina image size in pixels
        ret_pix_x = int(np.ceil((max_x_mm_im - min_x_mm_im) * 1000 / um_per_pix))
        ret_pix_y = int(np.ceil((max_y_mm_im - min_y_mm_im) * 1000 / um_per_pix))

        ret_img_pix = np.zeros((ret_pix_y, ret_pix_x))

        # Prepare numpy nd array to hold left upper corner pixel coordinates for each rf image
        gc_img_lu_pix = np.zeros((gc_img.shape[0], 2), dtype=int)

        pos_ecc_mm = df["pos_ecc_mm"].values
        pos_polar_deg = df["pos_polar_deg"].values

        # Locate left upper corner of each rf img and lay images onto retina image
        x_mm, y_mm = self.pol2cart(
            pos_ecc_mm.astype(np.float64),
            pos_polar_deg.astype(np.float64) - rot_deg,
            deg=True,
        )

        y_pix_c = (np.round((max_y_mm_im - y_mm) * 1000 / um_per_pix)).astype(np.int64)
        x_pix_c = (np.round((x_mm - min_x_mm_im) * 1000 / um_per_pix)).astype(np.int64)

        exp_pix_per_side = self.context.apricot_metadata["data_spatialfilter_height"]
        pix_scaler = pix_per_side / exp_pix_per_side

        for i, row in df.iterrows():
            # Get the position of the rf upper left corner in pixels
            # The xoc and yoc are the center of the rf image in the resampled data scale.
            # The pix_scaler is necessary because xoc and yoc are are shifted in the resampled data scale.
            y_pix_lu = y_pix_c[i] - int(row.yoc_pix * pix_scaler)
            x_pix_lu = x_pix_c[i] - int(row.xoc_pix * pix_scaler)

            # Get the rf image
            this_rf_img = gc_img[i, :, :]
            # Lay the rf image onto the retina image
            ret_img_pix[
                y_pix_lu : y_pix_lu + pix_per_side,
                x_pix_lu : x_pix_lu + pix_per_side,
            ] += this_rf_img
            # Store the left upper corner pixel coordinates and width and height of each rf image.
            # The width and height are necessary because some are cut off at the edges of the retina image.
            gc_img_lu_pix[i, :] = [x_pix_lu, y_pix_lu]

        gc.img_lu_pix = gc_img_lu_pix

        ret.whole_ret_lu_mm = np.array([min_x_mm_im, max_y_mm_im])

        return ret, gc, ret_img_pix

    def _get_vae_imgs_with_good_fits(self, gc, retina_vae):
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
        gc_vae_img : numpy.ndarray
            3D array of spatial receptive fields, shape (n_units, new_pix_per_side, new_pix_per_side).

        Notes
        -----
        The VAE generates a number of RFs that is larger than nsamples.
        This is to account for outliers that are not accepted.
        """

        nsamples = gc.n_units

        # Get samples. We take 50% extra samples to cover the bad fits
        nsamples_extra = int(nsamples * 1.5)  # 50% extra to account for outliers
        img_processed_extra, img_raw_extra = self._get_generated_rfs(
            retina_vae, n_samples=nsamples_extra
        )

        idx_to_process = np.arange(nsamples)
        gc_vae_img = np.zeros((nsamples, gc.pix_per_side, gc.pix_per_side))
        available_idx_mask = np.ones(nsamples_extra, dtype=bool)
        available_idx_mask[idx_to_process] = False
        img_to_resample = img_processed_extra[idx_to_process, :, :]
        good_mask_compiled = np.zeros(nsamples, dtype=bool)
        _gc_vae_df = pd.DataFrame(
            index=np.arange(nsamples),
            columns=["xoc_pix", "yoc_pix"],
        )
        zoom_factor = gc.df["zoom_factor"].values
        # Loop until there are no bad fits
        for _ in range(100):
            # Upsample according to smallest rf diameter
            img_after_resample = self._get_resampled_scaled_gc_img(
                img_to_resample[idx_to_process, :, :],
                gc.pix_per_side,
                zoom_factor[idx_to_process],
            )

            # Fit elliptical gaussians to the img[idx_to_process]
            # This is dependent metrics, not affecting the spatial RFs
            # other than quality assurance (below)
            # Fixed DoG model type excludes the model effect on unit selection
            # Note that this fits the img_after_resample and thus the
            # xoc_pix and yoc_pix are veridical for the upsampled data.
            self.fit.initialize(
                gc.gc_type,
                gc.response_type,
                fit_type="generated",
                DoG_model="ellipse_fixed",
                spatial_data=img_after_resample,
                um_per_pix=gc.um_per_pix,
                mark_outliers_bad=True,  # False to bypass bad fit check
            )

            # 6) Discard bad fits
            good_idx_this_iter = self.fit.good_idx_generated
            good_idx_generated = idx_to_process[good_idx_this_iter]
            # save the good rfs
            gc_vae_img[good_idx_generated, :, :] = img_after_resample[
                good_idx_this_iter, :, :
            ]

            good_df = self.fit.all_data_fits_df.loc[good_idx_this_iter, :]
            _gc_vae_df.loc[good_idx_generated, ["yoc_pix", "xoc_pix"]] = good_df.loc[
                :, ["yoc_pix", "xoc_pix"]
            ].values

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
        assert (
            len(good_idx_compiled) == nsamples
        ), "Bad fit loop did not remove all bad fits, aborting..."

        self.project_data.construct_retina["gen_spat_img"] = {
            "img_processed": img_processed_extra[good_idx_compiled, :, :],
            "img_raw": img_raw_extra[good_idx_compiled, :, :],
        }

        gc.df.loc[:, ["xoc_pix", "yoc_pix"]] = _gc_vae_df
        gc.img = gc_vae_img

        return gc

    def _apply_rf_repulsion(self, ret, gc):
        """
        Apply mutual repulsion to receptive fields (RFs) to ensure optimal coverage of a simulated retina.
        It involves multiple iterations to gradually move the RFs until they cover the retina
        with minimal overlapping, considering boundary effects and force gradients.

        Parameters:
        -----------
        img_ret_shape : tuple
            Shape of the retina image (height, width) in pixels.
        gc_img : numpy.ndarray
            3D array representing the RFs, shape (n_rfs, n_pixels, n_pixels).
        gc_img_mask : numpy.ndarray
            3D array of boolean masks for RF centers, shape (n_rfs, n_pixels, n_pixels).
        gc_img_lu_pix : numpy.ndarray
            2D array of the upper-left pixel coordinates of each RF, shape (n_rfs, 2).
        um_per_pix : float
            Scale factor representing micrometers per pixel in `gc_img`.

        Returns:
        --------
        new_gc_img: numpy.ndarray
            The updated RFs after repulsion and transformation, shape (n_rfs, n_pixels, n_pixels).
        new_gc_img_lu_pix: numpy.ndarray
            Updated upper-left pixel coordinates of each RF, shape (n_rfs, 2), each row indicating (x, y).
        final_retina: numpy.ndarray
            2D array representing the final state of the retina after RF adjustments, shape matching `img_ret_shape`.
        """

        img_ret_shape = ret.whole_ret_img.shape

        show_repulsion_progress = self.rf_repulsion_params["show_repulsion_progress"]
        change_rate = self.rf_repulsion_params["change_rate"]
        n_iterations = self.rf_repulsion_params["n_iterations"]
        show_skip_steps = self.rf_repulsion_params["show_skip_steps"]
        border_repulsion_stength = self.rf_repulsion_params["border_repulsion_stength"]
        cooling_rate = self.rf_repulsion_params["cooling_rate"]
        show_only_unit = self.rf_repulsion_params["show_only_unit"]

        n_units, H, W = gc.img.shape
        assert H == W, "RF must be square, aborting..."

        if show_repulsion_progress is True:
            # Init plotting
            fig_args = self.viz.show_repulsion_progress(
                np.zeros(img_ret_shape),
                np.zeros(img_ret_shape),
                ecc_lim_mm=ret.ecc_lim_mm,
                polar_lim_deg=ret.polar_lim_deg,
                init=True,
                um_per_pix=gc.um_per_pix,
                sidelen=H,
            )

        rf_positions = np.array(gc.img_lu_pix, dtype=float)
        rfs = np.array(gc.img, dtype=float)
        rfs_mask = np.array(gc.img_mask, dtype=bool)
        masked_rfs = rfs * rfs_mask
        sum_masked_rfs = np.sum(masked_rfs, axis=(1, 2))

        # Compute boundary effect
        boundary_polygon = self.viz.boundary_polygon(
            ret.ecc_lim_mm,
            ret.polar_lim_deg,
            um_per_pix=gc.um_per_pix,
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
                        um_per_pix=gc.um_per_pix,
                        sidelen=H,
                        **fig_args,
                    )

        # Resample to rectangular H, W resolution
        new_gc_img = np.zeros((n_units, H, W))
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
            new_gc_img[i, ...] = resampled_values.reshape(H, W)

        new_gc_img_lu_pix = np.array(
            [Xout[:, 0, 0], Yout[:, 0, 0]], dtype=np.int32
        ).T  # x, y
        com_x_local = com_x - new_gc_img_lu_pix[:, 0]
        com_y_local = com_y - new_gc_img_lu_pix[:, 1]

        new_retina = np.zeros(img_ret_shape)
        final_retina = np.zeros(img_ret_shape)
        for i in range(n_units):
            new_retina[Yt[i], Xt[i]] += rfs[i]
            final_retina[Yout[i], Xout[i]] += new_gc_img[i]

        if show_repulsion_progress is True:
            # Show one last time with the final interpolated result
            self.viz.show_repulsion_progress(
                reference_retina,
                center_mask,
                new_retina=final_retina,
                iteration=iteration,
                um_per_pix=gc.um_per_pix,
                sidelen=H,
                **fig_args,
            )
            plt.ioff()  # Turn off interactive mode

        gc.img = new_gc_img
        gc.img_lu_pix = new_gc_img_lu_pix

        gc.df["com_x_pix"] = com_x_local
        gc.df["com_y_pix"] = com_y_local

        # # TODO: consider solution with
        # gc.df["xoc_pix"] = com_x_local
        # gc.df["yoc_pix"] = com_y_local

        ret.whole_ret_img = final_retina

        return (
            ret,
            gc,
        )

    def _get_img_grid_mm(self, ret, gc):
        """
        Get the receptive field center x and y coordinate grids in mm.
        Necessary for downstream distance calculations.

        Parameters
        ----------
        gc_img_mask : np.ndarray
            3D array of boolean masks for RF centers, shape (n_rfs, n_pixels, n_pixels).
        um_per_pix : float or 1D array with length = n_rfs
            The number of micrometers per pixel in the rf_img.
        gc_img_lu_pix : np.ndarray
            2D array of the upper-left pixel coordinates of each RF, shape (n_rfs, 2), each row indicating (x, y).
        whole_ret_lu_mm : tuple
            Left upper corner of retina image position in mm, (min_x_mm, max_y_mm).
        """

        # Get the rf image size in pixels. gc.img is [N, H, W]
        rf_pix_y = gc.img_mask.shape[1]
        rf_pix_x = gc.img_mask.shape[2]

        X_grid, Y_grid = np.meshgrid(
            np.arange(rf_pix_x),
            np.arange(rf_pix_y),
            indexing="xy",
        )

        # if gc.um_per_pix is 1D numpy array, tile it to 3D for efficient numerical operations
        if isinstance(gc.um_per_pix, np.ndarray):
            gc.um_per_pix = np.tile(
                gc.um_per_pix[:, np.newaxis, np.newaxis], (1, rf_pix_y, rf_pix_x)
            )

        _X_grid = np.tile(X_grid, (gc.img_lu_pix.shape[0], 1, 1))
        _Y_grid = np.tile(Y_grid, (gc.img_lu_pix.shape[0], 1, 1))

        X_grid_local_mm = _X_grid * gc.um_per_pix / 1000
        Y_grid_local_mm = _Y_grid * gc.um_per_pix / 1000

        x_vec = gc.img_lu_pix[:, 0]
        _rf_lu_pix_x = np.tile(
            x_vec[:, np.newaxis, np.newaxis], (1, rf_pix_y, rf_pix_x)
        )
        y_vec = gc.img_lu_pix[:, 1]
        _rf_lu_pix_y = np.tile(
            y_vec[:, np.newaxis, np.newaxis], (1, rf_pix_y, rf_pix_x)
        )
        _rf_lu_mm_x = _rf_lu_pix_x * gc.um_per_pix / 1000
        _rf_lu_mm_y = _rf_lu_pix_y * gc.um_per_pix / 1000

        # x starts from the left
        X_grid_mm = ret.whole_ret_lu_mm[0] + _rf_lu_mm_x + X_grid_local_mm

        # y starts from the top
        Y_grid_mm = ret.whole_ret_lu_mm[1] - _rf_lu_mm_y - Y_grid_local_mm
        # pdb.set_trace()
        gc.X_grid_mm = X_grid_mm
        gc.Y_grid_mm = Y_grid_mm

        return gc

    def _add_center_mask_area_to_df(self, gc):
        """
        Get the area of the center mask for each RF in mm^2.

        Parameters
        ----------
        final_gc_img_mask : np.ndarray
            3D array of boolean masks for RF centers, shape (n_rfs, n_pixels, n_pixels).
        um_per_pix : float
            The number of micrometers per pixel in the rf_img.
        """
        # Get the area of the center mask for each RF in mm^2
        center_mask_area_mm2 = (
            np.sum(gc.img_mask, axis=(1, 2)) * gc.um_per_pix**2 / 1000**2
        )

        gc.df["center_mask_area_mm2"] = center_mask_area_mm2

        return gc

    def _add_center_fit_area_to_df(self, gc):
        if gc.DoG_model == "circular":
            gc.df["center_fit_area_mm2"] = np.pi * gc.df["rad_c_mm"] ** 2

        elif gc.DoG_model in ["ellipse_independent", "ellipse_fixed"]:
            gc.df["center_fit_area_mm2"] = (
                np.pi * gc.df["semi_xc_mm"] * gc.df["semi_yc_mm"]
            )

        return gc

    def _get_gc_fit_img(self, gc):
        """
        Make receptive field images from the generated FIT parameters.
        """

        n_cells = len(gc.df)
        num_pix_y = self.context.apricot_metadata["data_spatialfilter_height"]
        num_pix_x = self.context.apricot_metadata["data_spatialfilter_width"]
        assert num_pix_y == num_pix_x, "Exp data matrix must be square, aborting..."
        exp_pix_per_side = num_pix_y

        # Make fit to all cells
        grid_indices = np.linspace(0, exp_pix_per_side - 1, exp_pix_per_side)
        # the grid is (H, W) = (num_pix_y, num_pix_x)
        x_grid, y_grid = np.meshgrid(grid_indices, grid_indices)

        if gc.DoG_model == "ellipse_fixed":
            parameter_names = [
                "ampl_c",
                "xoc_pix",
                "yoc_pix",
                "semi_xc_pix",
                "semi_yc_pix",
                "orient_cen_rad",
                "ampl_s",
                "relat_sur_diam",
                "offset",
            ]

        elif gc.DoG_model == "ellipse_independent":
            parameter_names = [
                "ampl_c",
                "xoc_pix",
                "yoc_pix",
                "semi_xc_pix",
                "semi_yc_pix",
                "orient_cen_rad",
                "ampl_s",
                "xos_pix",
                "yos_pix",
                "semi_xs_pix",
                "semi_ys_pix",
                "orient_sur_rad",
                "offset",
            ]

        elif gc.DoG_model == "circular":
            parameter_names = [
                "ampl_c",
                "xoc_pix",
                "yoc_pix",
                "rad_c_pix",
                "ampl_s",
                "rad_s_pix",
                "offset",
            ]

        parameters = gc.df[parameter_names].values

        gc_fit_img = np.zeros((n_cells, exp_pix_per_side, exp_pix_per_side))

        for idx in range(n_cells):
            # Get DoG model fit parameters to popt
            popt = parameters[idx, :]

            # Ellipses for DoG2D_fixed_surround. Circular params are mapped to ellipse_fixed params
            if gc.DoG_model == "ellipse_fixed":
                gc_img_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)

            elif gc.DoG_model == "ellipse_independent":
                gc_img_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)

            elif gc.DoG_model == "circular":
                gc_img_fitted = self.DoG2D_circular((x_grid, y_grid), *popt)

            gc_fit_img[idx, :, :] = gc_img_fitted.reshape(
                exp_pix_per_side, exp_pix_per_side
            )

        return gc_fit_img

    def _create_spatial_rfs(self, ret, gc):
        """
        Generation of spatial receptive fields (RFs) for the retinal ganglion cells (RGCs).

        The RFs are generated using either a generative variational autoencoder (VAE) model or
        a fit to the data from the literature. The VAE model is trained on the data from
        the literature and generates RFs that are similar to the literature data.

        The RFs are generated in the following steps:
        1) Get the VAE model to generate receptive fields.
        2) "Bad fit loop", provides eccentricity-scaled vae rfs with good DoG fits (error < 3SD from mean).
        3) Get center masks.
        4) Sum separate rf images onto one retina pixel matrix.
        5) Apply repulsion adjustment to the receptive fields. Note that this will
        change the positions of the receptive fields.
        6) Redo the good fits for final statistics.


        RF become resampled, and the resolution will change if
        eccentricity is different from eccentricity of the original data.
        """

        # Get fit parameters for dendritic field diameter (dd) with respect to eccentricity (ecc).
        # Data from Watanabe_1989_JCompNeurol, Perry_1984_Neurosci and Goodchild_1996_JCompNeurol
        ecc2dd_params = self._fit_dd_vs_ecc(ret, gc)

        # # Quality control: check that the fitted dendritic diameter is close to the original data
        # # Frechette_2005_JNeurophysiol datasets: 9.7 mm (45°); 9.0 mm (41°); 8.4 mm (38°)
        # # Estimate the orginal data eccentricity from the fit to full eccentricity range
        # # TODO: move to integration tests
        # exp_rad = self.exp_cen_radius_mm * 2 * 1000
        # self.ecc_limit_for_dd_fit_mm = np.inf
        # dd_ecc_params_full = self._fit_dd_vs_ecc()
        # data_ecc_mm = self._get_ecc_from_dd(dd_ecc_params_full, dd_regr_model, exp_rad)
        # data_ecc_deg = data_ecc_mm * self.deg_per_mm  # 37.7 deg

        # Endow cells with spatial elliptical receptive fields.
        # Units become mm unless specified in column names.
        # self.gc_df and self.gc_vae_df may be updated silently below

        gc = self._get_gc_img_params(ret, gc, ecc2dd_params)

        if gc.spatial_model == "FIT":
            if ret.rf_coverage_adjusted_to_1 == True:
                # Assumes that the dendritic field diameter is proportional to the coverage
                gc = self._generate_DoG_with_rf_coverage_one(ret, gc)

            elif ret.rf_coverage_adjusted_to_1 == False:
                # Read the dendritic field diameter from literature data
                gc = self._generate_DoG_with_rf_from_literature(gc)

            # Add dendritic diameter to self.gc_df for visualization, in micrometers
            gc = self._get_dd_in_um(gc)

            # Create gc_img from DoG model
            print("\nGenerating RF images for FIT model...")
            gc.img = self._get_gc_fit_img(gc)

            # Resample and zoom according to smallest rf diameter
            gc.img = self._get_resampled_scaled_gc_img(
                gc.img, gc.pix_per_side, gc.df["zoom_factor"]
            )

            gc.img_mask = self.get_rf_masks(gc.img, mask_threshold=gc.mask_threshold)

            ret, gc, ret.whole_ret_img = self._get_full_retina_with_rf_images(
                ret, gc, gc.img
            )

            gc = self._get_img_grid_mm(ret, gc)

            # Add center mask area (mm^2) to gc_vae_df for visualization
            gc = self._add_center_mask_area_to_df(gc)

        elif gc.spatial_model == "VAE":
            # Endow cells with spatial receptive fields using the generative variational autoencoder model

            # 1) Get variational autoencoder to generate receptive fields
            print("\nGetting VAE model...")
            retina_vae = RetinaVAE(
                gc.gc_type,
                gc.response_type,
                self.training_mode,
                self.context,
                save_tuned_models=True,
            )

            # 2) "Bad fit loop", provides eccentricity-scaled vae rfs with good DoG fits (error < 3SD from mean).
            print("\nBad fit loop: Generating receptive fields with good DoG fits...")
            gc = self._get_vae_imgs_with_good_fits(gc, retina_vae)

            # 3) Get center masks
            gc.img_mask = self.get_rf_masks(gc.img, mask_threshold=gc.mask_threshold)

            viz_gc_vae_img = gc.img
            viz_gc_vae_img_mask = gc.img_mask

            # 4) Sum separate rf images onto one retina pixel matrix.
            # In the retina pixel matrix, for each rf get the upper left corner
            # pixel coordinates. Get the retina patch lu mm coordinates (padded).
            ret, gc, ret.whole_ret_img = self._get_full_retina_with_rf_images(
                ret, gc, gc.img
            )
            viz_whole_ret_img = ret.whole_ret_img

            # 5) Apply repulsion adjustment to the receptive fields. Note that this will
            # change the positions of the receptive fields.
            print("\nApplying repulsion between the receptive fields...")
            ret, gc = self._apply_rf_repulsion(ret, gc)

            # 6) Redo the good fits for final statistics
            print("\nFinal DoG fit to generated rfs...")
            self.fit.initialize(
                gc.gc_type,
                gc.response_type,
                fit_type="generated",
                DoG_model=gc.DoG_model,
                spatial_data=gc.img,
                um_per_pix=gc.um_per_pix,
                mark_outliers_bad=False,
            )

            (
                self.gen_stat_df,
                self.gen_spat_cen_sd,
                self.gen_spat_sur_sd,
                _gc_vae_df,
                _,
            ) = self.fit.get_generated_spatial_fits(gc.DoG_model)

            # 7) Update self.gc_vae_df to include new positions and DoG fits after repulsion
            # and convert units to to mm, where applicable
            print("\nUpdating ganglion cell dataframe...")
            gc = self._update_vae_gc_df(ret, gc, _gc_vae_df)

            # Check that all fits are good. If this starts creating problems, probably
            # the best solution is to remove the bad fit units totally from the self.gc_vae_df, self.gc_df,
            # final_gc_vae_img, new_gc_img_lu_pix, final_whole_ret_img, com_x, com_y, and update self.n_units
            assert gc.n_units == np.sum(
                _gc_vae_df["good_filter_data"]
            ), "Some final VAE fits are bad, aborting..."

            # 8) Get final center masks for the generated spatial rfs
            print("\nGetting final masked rfs and retina...")
            gc.img_mask = self.get_rf_masks(gc.img, mask_threshold=gc.mask_threshold)
            # Add center mask area (mm^2) to gc_vae_df for visualization
            gc = self._add_center_mask_area_to_df(gc)

            # 9) Sum separate rf center masks onto one retina pixel matrix.
            ret, gc, ret.whole_ret_img_mask = self._get_full_retina_with_rf_images(
                ret, gc, gc.img_mask
            )

            gc = self._get_img_grid_mm(ret, gc)

            # 10) Set vae data to project_data for later visualization
            self.project_data.construct_retina["retina_vae"] = retina_vae

            self.project_data.construct_retina["gen_rfs"] = {
                "gc_vae_img": viz_gc_vae_img,
                "gc_vae_img_mask": viz_gc_vae_img_mask,
                "final_gc_vae_img": gc.img,
                "centre_of_mass_x": gc.df["com_x_pix"],
                "centre_of_mass_y": gc.df["com_y_pix"],
            }

            self.project_data.construct_retina["gen_ret"] = {
                "img_ret": viz_whole_ret_img,
                "img_ret_masked": ret.whole_ret_img_mask,
                "img_ret_adjusted": ret.whole_ret_img,
            }

        # Add fitted DoG center area to gc_df for visualization
        gc = self._add_center_fit_area_to_df(gc)

        # Scale center and surround amplitude: center Gaussian volume in pixel space becomes one
        # Surround amplitude is scaled relative to center volume of one
        gc = self._scale_both_amplitudes(gc)

        # Set more project_data for later visualization
        self.project_data.construct_retina["dd_vs_ecc"][
            "dd_DoG_x"
        ] = gc.df.pos_ecc_mm.values
        self.project_data.construct_retina["dd_vs_ecc"][
            "dd_DoG_y"
        ] = gc.df.den_diam_um.values

        return ret, gc

        ### Generation of spatial receptive fields ends here ###
        #########################################################

    def _create_temporal_rfs(self, gc):
        if gc.temporal_model == "fixed":
            gc = self._create_fixed_temporal_rfs(gc)  # Chichilnisky data

            # For fixed model, we borrow the gain and mean firing rates from the Bnardete & Kaplan data
            gc_to_get_A = deepcopy(gc)
            gc_to_get_A = self._create_dynamic_temporal_rfs(
                gc_to_get_A
            )  # Benardete & Kaplan data
            gc.df["A"] = gc_to_get_A.df["A"]
            gc.df["Mean"] = gc_to_get_A.df["Mean"]

        elif gc.temporal_model == "dynamic":
            gc = self._create_dynamic_temporal_rfs(gc)  # Benardete & Kaplan data

        return gc

    def build(self):
        """
        Builds the receptive field mosaic. This is the main method to call.

        When ret or gc are updated, they are returned from the method.
        """
        my_retina = self.context.my_retina
        ret = Retina(my_retina)
        gc = GanglionCellData(
            my_retina["gc_type"],
            my_retina["response_type"],
            my_retina["spatial_model"],
            my_retina["temporal_model"],
            my_retina["DoG_model"],
            my_retina["center_mask_threshold"],
        )

        # -- First, place the ganglion cell midpoints (units mm)
        # Run GC and cone density fit to data, get func_params.
        # GC data from Perry_1984_Neurosci, cone data from Packer_1989_JCompNeurol
        ret = self.read_and_fit_unit_density_data(ret)

        # Place ganglion cells and cones to desired retina.
        ret, gc = self._place_units(ret, gc)
        gc.n_units = len(gc.df)

        # -- Second, endow cells with spatial receptive fields
        ret, gc = self._create_spatial_rfs(ret, gc)
        gc = self._link_cone_noise_units_to_gcs(ret, gc)
        gc = self._fit_cone_noise_vs_freq(gc)

        # -- Third, endow cells with temporal receptive fields
        gc = self._create_temporal_rfs(gc)

        # -- Fourth, endow cells with tonic drive
        gc = self._create_tonic_drive(gc)

        print(f"Built RGC mosaic with {gc.n_units} cells")

        # Save the receptive field images, associated metadata and cone noise data
        self.save_gc_img(gc)

        # Save the receptive field mosaic
        self.save_gc_csv(gc)

        # Save the project data
        # Attach data requested by other classes to project_data
        self.project_data.construct_retina["gc_df"] = gc.df

    def save_gc_img(self, gc):
        # Save the generated receptive field pix images, pix masks, and pixel locations in mm
        print("\nSaving data...")
        output_path = self.context.output_folder

        # Collate data for saving
        spatial_rfs_file = {
            "gc_img": gc.img,
            "gc_img_mask": gc.img_mask,
            "X_grid_mm": gc.X_grid_mm,
            "Y_grid_mm": gc.Y_grid_mm,
            "um_per_pix": gc.um_per_pix,
            "pix_per_side": gc.pix_per_side,
            "cones_to_gcs_weights": gc.cones_to_gcs_weights,
            "cone_noise_parameters": gc.cone_noise_parameters,
        }

        self.data_io.save_np_dict_to_npz(
            spatial_rfs_file, output_path, filename_stem=self.spatial_rfs_file_filename
        )

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

    def save_gc_csv(self, gc, filename=None):
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
            filepath = output_folder.joinpath(self.mosaic_filename)
        else:
            filepath = output_folder.joinpath(filename)

        print("Saving model mosaic to %s" % filepath)
        gc.df.to_csv(filepath)
