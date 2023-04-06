# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd

import torch
import torch.nn.functional as F

# from torch.utils.data import DataLoader

# from scipy.signal import convolve
# from scipy.interpolate import interp1d

# Data IO
# import cv2
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
from retina.gan_module import GAN

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
        Type of model, either "FIT", "VAE" or "GAN"
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
                Calls Fit, RetinaVAE or GAN classes

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
        randomize_position = my_retina["randomize_position"]
        self.deg_per_mm = my_retina["deg_per_mm"]

        self.gc_type = gc_type
        self.response_type = response_type

        self.model_type = my_retina["model_type"]
        if self.model_type in ["VAE", "GAN"]:
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
        self.eccentricity_in_mm = np.asarray(
            [r / self.deg_per_mm for r in ecc_limits]
        )  # Turn list to numpy array
        self.theta = np.asarray(sector_limits)  # Turn list to numpy array
        self.randomize_position = randomize_position

        # If study concerns visual field within 4 mm (20 deg) of retinal eccentricity, the cubic fit for dendritic diameters fails close to fovea. Better limit it to more central part of the data
        if np.max(self.eccentricity_in_mm) <= 4:
            self.visual_field_fit_limit = 4
        else:
            self.visual_field_fit_limit = np.inf

        # Initialize pandas dataframe to hold the ganglion cells (one per row) and all their parameters in one place
        columns = [
            "positions_eccentricity",
            "positions_polar_angle",
            "eccentricity_group_index",
            "semi_xc",
            "semi_yc",
            "xy_aspect_ratio",
            "amplitudes",
            "sur_ratio",
            "orientation_center",
        ]
        self.gc_df = pd.DataFrame(columns=columns)
        self.dendr_diam_model = "quadratic"  # 'linear' # 'quadratic' # cubic

        # Current version needs Fit for all 'model_type's (FIT, VAE, etc.)
        # If surround is fixed, the surround position, semi_x, semi_y (aspect_ratio) and orientation are the same as center params. This appears to give better results.
        self.surround_fixed = 1

        # Make or read fits
        if fits_from_file is None:
            (
                self.exp_stat_df,
                self.good_data_idx,
                self.bad_data_idx,
                self.exp_spat_cen_sd,
                self.exp_spat_sur_sd,
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

        :param shape:
        :param loc:
        :param scale:
        :param n_cells:
        :param distribution:

        :returns distribution_parameters
        """
        assert distribution in [
            "gamma",
            "beta",
            "skewnorm",
        ], "Distribution not supported"

        if distribution == "gamma":
            distribution_parameters = stats.gamma.rvs(
                a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
            )  # random_state is the seed
        elif distribution == "beta":
            distribution_parameters = stats.beta.rvs(
                a=shape[0],
                b=shape[1],
                loc=loc,
                scale=scale,
                size=n_cells,
                random_state=None,
            )  # random_state is the seed
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

    def _fit_dendritic_diameter_vs_eccentricity(self):
        """
        Dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
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
        data_all_x_index = data_all_x <= self.visual_field_fit_limit
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[
            data_all_x_index
        ]  # Don't forget to truncate values, too

        # Sort to ascending order
        data_all_x_index = np.argsort(data_all_x)
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[data_all_x_index]

        # Get rf diameter vs eccentricity
        dendr_diam_model = self.dendr_diam_model  # 'linear' # 'quadratic' # cubic
        dict_key = "{0}_{1}".format(self.gc_type, dendr_diam_model)

        if dendr_diam_model == "linear":
            polynomial_order = 1
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": polynomials[1],
                "slope": polynomials[0],
            }
        elif dendr_diam_model == "quadratic":
            polynomial_order = 2
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": polynomials[2],
                "slope": polynomials[1],
                "square": polynomials[0],
            }
        elif dendr_diam_model == "cubic":
            polynomial_order = 3
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": polynomials[3],
                "slope": polynomials[2],
                "square": polynomials[1],
                "cube": polynomials[0],
            }

        dataset_name = f"All data {dendr_diam_model} fit"
        self.dendrite_diam_vs_ecc_to_show = {
            "data_all_x": data_all_x,
            "data_all_y": data_all_y,
            "polynomials": polynomials,
            "dataset_name": dataset_name,
            "title": f"DF diam wrt ecc for {self.gc_type} type, {dataset_name} dataset",
        }

        return dendr_diam_parameters

    def _create_spatial_rfs(
        self,
        dendr_diam_vs_ecc_param_dict,
    ):
        """
        Create spatial receptive fields to model cells.
        Starting from 2D difference-of-gaussian parameters:
        'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'

        Places all ganglion cell spatial parameters to ganglion cell object dataframe self.gc_df
        """

        # Get eccentricity data for all model cells
        gc_eccentricity = self.gc_df["positions_eccentricity"].values

        # Get rf diameter vs eccentricity
        dendr_diam_model = self.dendr_diam_model  # from __init__ method
        dict_key = "{0}_{1}".format(self.gc_type, dendr_diam_model)
        diam_fit_params = dendr_diam_vs_ecc_param_dict[dict_key]

        if dendr_diam_model == "linear":
            gc_diameters = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
            )  # Units are micrometers
            polynomial_order = 1
        elif dendr_diam_model == "quadratic":
            gc_diameters = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
                + diam_fit_params["square"] * gc_eccentricity**2
            )
            polynomial_order = 2
        elif dendr_diam_model == "cubic":
            gc_diameters = (
                diam_fit_params["intercept"]
                + diam_fit_params["slope"] * gc_eccentricity
                + diam_fit_params["square"] * gc_eccentricity**2
                + diam_fit_params["cube"] * gc_eccentricity**3
            )
            polynomial_order = 3

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
                area_of_ellipse[self.gc_df["eccentricity_group_index"] == index]
            )  # in micrometers2

            area_scaling_factors_coverage1[
                self.gc_df["eccentricity_group_index"] == index
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

        self.gc_df["orientation_center"] = self.gc_df[
            "positions_polar_angle"
        ]  # plus some noise here TODO. See Watanabe 1989 JCompNeurol section Dendritic field orietation

    def _densfunc(self, r, d0, beta):
        return d0 * (1 + beta * r) ** (-2)

    def _place_gc_units(self, gc_density_func_params):
        """
        Place ganglion cell center positions to retina

        Creates self.gc_df: pandas.DataFrame with columns:
            positions_eccentricity, positions_polar_angle, eccentricity_group_index

        Parameters
        ----------
        gc_density_func_params: dict
            Dictionary with parameters for the density function
        """

        # Place cells inside one polar sector with density according to mid-ecc
        eccentricity_in_mm_total = self.eccentricity_in_mm
        theta = self.theta
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
        for eccentricity_group_index, current_step in enumerate(
            np.arange(int(n_steps))
        ):

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
            # my_gaussian_fit = self.gauss_plus_baseline(center_ecc, *gc_density_func_params) # leads to div by zero
            my_gaussian_fit = self._densfunc(
                center_ecc, 5.32043939e05, 2.64289725
            )  # deactivated SV 220531
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
                np.ones(true_n_cells) * eccentricity_group_index,
            )

        # Save cell position data to current ganglion cell object
        self.gc_df["positions_eccentricity"] = matrix_eccentricity_randomized_all
        self.gc_df["positions_polar_angle"] = matrix_polar_angle_randomized_all
        self.gc_df["eccentricity_group_index"] = gc_eccentricity_group_index.astype(
            np.uint32
        )
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

    def _scale_both_amplitudes(self):
        """
        Scale center and surround amplitudes so that the spatial RF volume is comparable to that of data.
        Second step of scaling is done before convolving with the stimulus.
        """

        # For each model cell, set center amplitude as data_cen_mean**2 / sigma_x * sigma_y
        # For each model cell, scale surround amplitude by data_sur_mean**2 / sur_sigma_x * sur_sigma_y
        # (Volume of 2D Gaussian = 2 * pi * sigma_x*sigma_y)

        n_rgc = len(self.gc_df)
        amplitudec = np.zeros(n_rgc)
        # amplitudes = np.zeros(n_rgc)

        for i in range(n_rgc):
            amplitudec[i] = self.exp_spat_cen_sd**2 / (
                self.gc_df.iloc[i].semi_xc * self.gc_df.iloc[i].semi_yc
            )

        data_rel_sur_amplitude = self.gc_df["amplitudes"]
        self.gc_df["amplitudec"] = amplitudec
        self.gc_df["amplitudes"] = amplitudec * data_rel_sur_amplitude
        self.gc_df["relative_sur_amplitude"] = (
            self.gc_df["amplitudes"] / self.gc_df["amplitudec"]
        )

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

    def build(self):
        """
        Builds the receptive field mosaic. This is the main method to call.
        """

        if self.initialized is False:
            self._initialize()

        # -- First, place the ganglion cell midpoints
        # Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
        gc_density_func_params = self._fit_gc_density_data()

        # Place ganglion cells to desired retina.
        self._place_gc_units(gc_density_func_params)

        # Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dendr_diam_vs_ecc_param_dict = self._fit_dendritic_diameter_vs_eccentricity()

        # -- Second, endow cells with spatial receptive fields
        self._create_spatial_rfs(dendr_diam_vs_ecc_param_dict)

        # Scale center and surround amplitude so that Gaussian volume is preserved
        self._scale_both_amplitudes()  # TODO - what was the purpose of this? Working retina uses amplitudec

        # At this point the spatial receptive fields are ready.
        # The positions are in gc_eccentricity, gc_polar_angle, and the rf parameters in gc_rf_models

        match self.model_type:
            case "FIT":
                pass
                # Everything at the moment is done above
                # This is just to check for model type

            case "VAE":

                # Fit or load variational autoencoder to generate receptive fields
                self.retina_vae = RetinaVAE(
                    self.gc_type,
                    self.response_type,
                    self.training_mode,
                    self.context.apricot_data_folder,
                    self.context.output_folder,
                )

                # -- Second, endow cells with spatial receptive fields using the generative variational autoencoder model
                # --- 1. make a probability density function of the latent space
                retina_vae = self.retina_vae
                latent_data = self.get_data_at_latent_space(retina_vae)

                # Make a probability density function of the latent_data
                latent_pdf = stats.gaussian_kde(latent_data.T)

                # --- 2. sample from the pdf
                n_samples = len(self.gc_df)
                # n_samples = 1000
                latent_samples = torch.tensor(latent_pdf.resample(n_samples).T).to(
                    retina_vae.device
                )
                # Change the dtype to float32
                latent_samples = latent_samples.type(torch.float32)
                match self.training_mode:
                    case "load_model":
                        latent_dim = self.retina_vae.vae.config["latent_dims"]
                    case "train_model":
                        latent_dim = self.retina_vae.latent_dim

                self.gen_latent_space_to_viz = {
                    "samples": latent_samples.to("cpu").numpy(),
                    "dim": latent_dim,
                    "data": latent_data,
                }

                # --- 3. decode the samples
                img_stack = self.retina_vae.vae.decoder(latent_samples)

                # Images were upsampled for VAE training.
                # Downsample generated images back to the Apricot size
                img_stack_downsampled = F.interpolate(
                    img_stack,
                    size=self.apricot_data_resolution_hw,
                    mode="bilinear",
                    align_corners=True,
                )

                img_stack_np = img_stack_downsampled.detach().cpu().numpy()

                # The shape of img_stack_np is (n_samples, 1, img_size, img_size)
                # Reshape to (n_samples, img_size, img_size)
                img_reshaped = np.reshape(
                    img_stack_np,
                    (n_samples, img_stack_np.shape[2], img_stack_np.shape[3]),
                )

                # Save the generated receptive fields
                output_path = self.context.output_folder

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

                # Set self attribute for later visualization of image histograms
                self.gen_spat_img_to_viz = {
                    "img_processed": img_flipped,
                    "img_raw": img_reshaped,
                }

                # TÄHÄN JÄIT:
                # SIIRRÄ VIZ => VIZ
                # INTEGROI VAE GENEROIDUT RF:T WORKING RETINAAN
                # SISÄLLYTÄ MUUTTUVA LR OPTIMOINTIIN?

                img_paths = self.save_generated_rfs(img_flipped, output_path)

                # Add image paths as a columnd to self.gc_df
                self.gc_df["img_path"] = img_paths

                (
                    self.gen_stat_df,
                    self.gen_spat_cen_sd,
                    self.gen_spat_sur_sd,
                    self.gen_spat_filt_to_viz,
                    self.gen_spat_stat_to_viz,
                ) = Fit(
                    self.context.apricot_data_folder,
                    self.gc_type,
                    self.response_type,
                    spatial_data=img_flipped,
                    fit_type="generated",
                ).get_generated_spatial_fits()

            case "GAN":
                # Use the generative adversarial network model to provide spatial and temporal receptive fields
                pass
            case other:
                raise ValueError("Model type not recognized")

        # -- Third, endow cells with temporal receptive fields
        self._create_temporal_receptive_fields()

        # -- Fourth, endow cells with tonic drive
        self._create_tonic_drive()

        n_rgc = len(self.gc_df)
        print(f"Built RGC mosaic with {n_rgc} cells")

        # Save the receptive field mosaic
        self.save_gc_csv()

    def plot_rfs_from_vae(self, img_stack, n_examples=4):
        """
        Show n_examples of the generated receptive fields
        """
        import matplotlib.pyplot as plt

        # Make a grid of subplots
        n_cols = 4
        n_rows = int(np.ceil(n_examples / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
        axes = axes.flatten()

        for i in range(n_examples):
            ax = axes[i]
            img = img_stack[i, 0, :, :].detach().cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"RF {i}")
            ax.axis("off")

        plt.show()

    def get_data_at_latent_space(self, retina_vae):
        """
        Get original image data as projected through encoder to the latent space
        """
        # Get the latent space data
        train_df = retina_vae.get_encoded_samples(ds_name="train_ds")
        valid_df = retina_vae.get_encoded_samples(ds_name="val_ds")
        test_df = retina_vae.get_encoded_samples(ds_name="test_ds")
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

    def save_generated_rfs(self, img_stack, output_path):
        """
        Saves a 3D image stack as a series of 2D image files using Pillow.

        Parameters
        ----------
            img_stack (numpy.ndarray): The 3D image stack to be saved, with shape (M, N, N).
            output_path (str or Path): The path to the output folder where the image files will be saved.
        """
        # Convert output_path to a Path object if it's a string
        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Create the output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Create pandas series object, which will hold the full paths to the generated images
        img_paths_s = pd.Series(index=range(img_stack.shape[0]))

        # Loop through each slice in the image stack
        for i in range(img_stack.shape[0]):
            # Rescale the pixel values to the range of 0 to 65535
            img_array = (img_stack[i, :, :] * 65535.0).astype(np.uint16)

            # Create a PIL Image object from the current slice
            img = Image.fromarray(img_array)

            # Save the image file with a unique name based on the slice index
            filename_full = output_path / f"slice_{i+1}.png"
            img.save(filename_full)

            # Add the full path to the image file to the pandas series object
            img_paths_s[i] = filename_full

        return img_paths_s

    def show_exp_build_process(self, show_all_spatial_fits=False):
        """
        Show the process of building the mosaic
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_exp_build_process(
            self, show_all_spatial_fits=show_all_spatial_fits
        )

    def show_gen_and_exp_spatial_rfs(self, n_samples=2):
        """
        Show the experimental (fitted) and generated spatial receptive fields
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_gen_and_exp_spatial_rfs(self, n_samples=n_samples)

    def show_gen_spat_postprocessing(self):
        """
        Show the original experimental spatial receptive fields and
        the generated spatial receptive fields before and after postprocessing
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_gen_spat_postprocessing(self)

    def show_latent_space_and_samples(self):
        """
        Plot the latent samples on top of the estimated kde, one sublot
        for each successive two dimensions of latent_dim
        self goes as argument, to be available for viz
        """

        # The argument "self" i.e. the construct_retina object becomes available in the Viz class as "mosaic"
        self.viz.show_latent_space_and_samples(self)
