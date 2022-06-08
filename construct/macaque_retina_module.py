# Numerical
from fileinput import filename
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d

# Viz
from tqdm import tqdm

# Comput Neurosci
import brian2 as b2
import brian2.units as b2u

# Local
from cxsystem2.core.tools import write_to_file, load_from_file
from construct.apricot_fitter_module import ApricotFits
from construct.construct_math_module import RetinaMath

# Builtin
import sys
from pathlib import Path
import os
from copy import deepcopy
import pdb


class ConstructRetina(RetinaMath):
    """
    Create the ganglion cell mosaic.
    All spatial parameters are saved to the dataframe *gc_df*
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

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        self._viz = viz

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def viz(self):
        return self._viz

    def initialize(self, fits_from_file=None):

        """
        Initialize the ganglion cell mosaic

        :param gc_type: 'parasol' or 'midget'
        :param fits_from_file: path to a file containing the fits
        """

        my_retina = self.context.my_retina
        gc_type = my_retina["gc_type"]
        response_type = my_retina["response_type"]
        ecc_limits = my_retina["ecc_limits"]
        sector_limits = my_retina["sector_limits"]
        model_density = my_retina["model_density"]
        randomize_position = my_retina["randomize_position"]
        self.deg_per_mm = my_retina["deg_per_mm"]

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

        # GC type specifications self.gc_proportion
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
            print("Unknown ganglion cell type, aborting")
            sys.exit()

        self.gc_type = gc_type
        self.response_type = response_type

        self.eccentricity = ecc_limits
        self.eccentricity_in_mm = np.asarray(
            [r / self.deg_per_mm for r in ecc_limits]
        )  # Turn list to numpy array
        self.theta = np.asarray(sector_limits)  # Turn list to numpy array
        self.randomize_position = randomize_position
        self.dendr_diam_model = "quadratic"  # 'linear' # 'quadratic' # cubic

        # If study concerns visual field within 4 mm (20 deg) of retinal eccentricity, the cubic fit for
        # dendritic diameters fails close to fovea. Better limit it to more central part of the data
        if np.max(self.eccentricity_in_mm) <= 4:
            self.visual_field_fit_limit = 4
        else:
            self.visual_field_fit_limit = np.inf

        # If surround is fixed, the surround position, semi_x, semi_y (aspect_ratio)
        # and orientation are are the same as center params. This appears to give better results.
        self.surround_fixed = 1

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

        # Set stimulus stuff
        self.stimulus_video = None

        # Make or read fits
        if fits_from_file is None:
            # init and call -- only connection to apricot_fitter_module
            (
                self.all_fits_df,
                self.temporal_filters_to_show,
                self.spatial_filters_to_show,
            ) = ApricotFits(
                self.context.apricot_data_folder, gc_type, response_type
            ).get_fits()
        else:
            self.all_fits_df = pd.read_csv(
                fits_from_file, header=0, index_col=0
            ).fillna(0.0)

        self.n_cells_data = len(self.all_fits_df)
        self.bad_data_indices = np.where((self.all_fits_df == 0.0).all(axis=1))[
            0
        ].tolist()
        self.good_data_indices = np.setdiff1d(
            range(self.n_cells_data), self.bad_data_indices
        )

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

    def _place_spatial_receptive_fields(
        self,
        spatial_statistics_dict,
        dendr_diam_vs_eccentricity_parameters_dict,
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
        diam_fit_params = dendr_diam_vs_eccentricity_parameters_dict[dict_key]

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
        n_cells = len(gc_eccentricity)
        n_parameters = len(spatial_statistics_dict.keys())
        gc_rf_models = np.zeros((n_cells, n_parameters))
        for index, key in enumerate(spatial_statistics_dict.keys()):
            shape = spatial_statistics_dict[key]["shape"]
            loc = spatial_statistics_dict[key]["loc"]
            scale = spatial_statistics_dict[key]["scale"]
            distribution = spatial_statistics_dict[key]["distribution"]
            gc_rf_models[:, index] = self._get_random_samples(
                shape, loc, scale, n_cells, distribution
            )

        # Calculate RF diameter scaling factor for all ganglion cells
        # Area of RF = Scaling_factor * Random_factor * Area of ellipse(semi_xc,semi_yc), solve Scaling_factor.
        area_of_ellipse = self.ellipse2area(
            gc_rf_models[:, 0], gc_rf_models[:, 1]
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
        # scale_random_distribution = 0.08  # Estimated by eye from Watanabe and Perry data. Normal distribution with scale_random_distribution 0.08 cover about 25% above and below the mean value
        scale_random_distribution = 0.001
        random_normal_distribution1 = 1 + np.random.normal(
            scale=scale_random_distribution, size=n_cells
        )
        semi_xc = (
            np.sqrt(area_scaling_factors_coverage1)
            * gc_rf_models[:, 0]
            * random_normal_distribution1
        )
        random_normal_distribution2 = 1 + np.random.normal(
            scale=scale_random_distribution, size=n_cells
        )  # second randomization
        semi_yc = (
            np.sqrt(area_scaling_factors_coverage1)
            * gc_rf_models[:, 1]
            * random_normal_distribution2
        )

        # Scale from micrometers to millimeters and return to numpy matrix
        gc_rf_models[:, 0] = semi_xc / 1000
        gc_rf_models[:, 1] = semi_yc / 1000

        # Save to ganglion cell dataframe. Keep it explicit to avoid unknown complexity
        self.gc_df["semi_xc"] = gc_rf_models[:, 0]
        self.gc_df["semi_yc"] = gc_rf_models[:, 1]
        self.gc_df["xy_aspect_ratio"] = gc_rf_models[:, 2]
        self.gc_df["amplitudes"] = gc_rf_models[:, 3]
        self.gc_df["sur_ratio"] = gc_rf_models[:, 4]
        # self.gc_df['orientation_center'] = gc_rf_models[:, 5]
        self.gc_df["orientation_center"] = self.gc_df[
            "positions_polar_angle"
        ]  # plus some noise here

    def _densfunc(self, r, d0, beta):
        return d0 * (1 + beta * r) ** (-2)

    def _place_gc_units(self, gc_density_func_params):
        """
        Place ganglion cell center positions to retina

        :param gc_density_func_params:
        :param show_build_process: True/False (default False)

        :returns matrix_eccentricity_randomized_all, matrix_orientation_surround_randomized_all
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
        self.gc_df["eccentricity_group_index"] = gc_eccentricity_group_index.astype(int)
        self.sector_surface_area_all = np.asarray(sector_surface_area_all)

        # Pass the GC object to self, because the Viz class is not inherited
        self.gc_density_func_params = gc_density_func_params

    def _fit_spatial_statistics(self):
        """
        Collect spatial statistics from Chichilnisky receptive field data
        """

        # parameter_names, data_all_viable_cells, bad_cell_indices = fitdata
        data_all_viable_cells = np.array(self.all_fits_df)
        bad_cell_indices = np.where((self.all_fits_df == 0.0).all(axis=1))[0].tolist()
        parameter_names = self.all_fits_df.columns.tolist()

        all_viable_cells = np.delete(data_all_viable_cells, bad_cell_indices, 0)

        chichilnisky_data_df = pd.DataFrame(
            data=all_viable_cells, columns=parameter_names
        )

        # Save stats description to gc object
        self.rf_datafit_description_series = chichilnisky_data_df.describe()

        # Calculate xy_aspect_ratio
        xy_aspect_ratio_pd_series = (
            chichilnisky_data_df["semi_yc"] / chichilnisky_data_df["semi_xc"]
        )
        xy_aspect_ratio_pd_series.rename("xy_aspect_ratio")
        chichilnisky_data_df["xy_aspect_ratio"] = xy_aspect_ratio_pd_series

        rf_parameter_names = [
            "semi_xc",
            "semi_yc",
            "xy_aspect_ratio",
            "amplitudes",
            "sur_ratio",
            "orientation_center",
        ]
        self.rf_parameter_names = rf_parameter_names  # For reference
        n_distributions = len(rf_parameter_names)
        shape = np.zeros(
            [n_distributions - 1]
        )  # orientation_center has two shape parameters, below alpha and beta
        loc = np.zeros([n_distributions])
        scale = np.zeros([n_distributions])
        ydata = np.zeros([len(all_viable_cells), n_distributions])
        x_model_fit = np.zeros([100, n_distributions])
        y_model_fit = np.zeros([100, n_distributions])

        # Create dict for statistical parameters
        spatial_statistics_dict = {}

        # Model 'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio' rf_parameter_names with a gamma function.
        for index, distribution in enumerate(rf_parameter_names[:-1]):
            # fit the rf_parameter_names, get the PDF distribution using the parameters
            ydata[:, index] = chichilnisky_data_df[distribution]
            shape[index], loc[index], scale[index] = stats.gamma.fit(
                ydata[:, index], loc=0
            )
            x_model_fit[:, index] = np.linspace(
                stats.gamma.ppf(
                    0.001, shape[index], loc=loc[index], scale=scale[index]
                ),
                stats.gamma.ppf(
                    0.999, shape[index], loc=loc[index], scale=scale[index]
                ),
                100,
            )
            y_model_fit[:, index] = stats.gamma.pdf(
                x=x_model_fit[:, index],
                a=shape[index],
                loc=loc[index],
                scale=scale[index],
            )

            # Collect parameters
            spatial_statistics_dict[distribution] = {
                "shape": shape[index],
                "loc": loc[index],
                "scale": scale[index],
                "distribution": "gamma",
            }

        # Model orientation distribution with beta function.
        index += 1
        ydata[:, index] = chichilnisky_data_df[rf_parameter_names[-1]]
        a_parameter, b_parameter, loc[index], scale[index] = stats.beta.fit(
            ydata[:, index], 0.6, 0.6, loc=0
        )  # initial guess for a_parameter and b_parameter is 0.6
        x_model_fit[:, index] = np.linspace(
            stats.beta.ppf(
                0.001, a_parameter, b_parameter, loc=loc[index], scale=scale[index]
            ),
            stats.beta.ppf(
                0.999, a_parameter, b_parameter, loc=loc[index], scale=scale[index]
            ),
            100,
        )
        y_model_fit[:, index] = stats.beta.pdf(
            x=x_model_fit[:, index],
            a=a_parameter,
            b=b_parameter,
            loc=loc[index],
            scale=scale[index],
        )
        spatial_statistics_dict[rf_parameter_names[-1]] = {
            "shape": (a_parameter, b_parameter),
            "loc": loc[index],
            "scale": scale[index],
            "distribution": "beta",
        }

        self.spatial_statistics_to_show = {
            "ydata": ydata,
            "spatial_statistics_dict": spatial_statistics_dict,
            "model_fit_data": (x_model_fit, y_model_fit),
        }

        # Return stats for RF creation
        return spatial_statistics_dict

    def _fit_tonic_drives(self):
        tonicdrive_array = np.array(
            self.all_fits_df.iloc[self.good_data_indices].tonicdrive
        )
        shape, loc, scale = stats.gamma.fit(tonicdrive_array)

        x_min, x_max = stats.gamma.ppf([0.001, 0.999], a=shape, loc=loc, scale=scale)
        xs = np.linspace(x_min, x_max, 100)
        pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
        title = self.gc_type + " " + self.response_type

        self.tonic_drives_to_show = {
            "xs": xs,
            "pdf": pdf,
            "tonicdrive_array": tonicdrive_array,
            "title": title,
        }

        return shape, loc, scale

    def _fit_temporal_statistics(self):
        temporal_filter_parameters = ["n", "p1", "p2", "tau1", "tau2"]
        distrib_params = np.zeros((len(temporal_filter_parameters), 3))

        for i, param_name in enumerate(temporal_filter_parameters):
            param_array = np.array(
                self.all_fits_df.iloc[self.good_data_indices][param_name]
            )
            shape, loc, scale = stats.gamma.fit(param_array)
            distrib_params[i, :] = [shape, loc, scale]

        self.temp_stat_to_show = {
            "temporal_filter_parameters": temporal_filter_parameters,
            "distrib_params": distrib_params,
            "suptitle": self.gc_type + " " + self.response_type,
            "all_fits_df": self.all_fits_df,
            "good_data_indices": self.good_data_indices,
        }

        return pd.DataFrame(
            distrib_params,
            index=temporal_filter_parameters,
            columns=["shape", "loc", "scale"],
        )

    def _create_temporal_filters(self, distrib_params_df, distribution="gamma"):

        n_rgc = len(self.gc_df)

        for param_name, row in distrib_params_df.iterrows():
            shape, loc, scale = row
            self.gc_df[param_name] = self._get_random_samples(
                shape, loc, scale, n_rgc, distribution
            )

    def _scale_both_amplitudes(self):
        """
        Scale center and surround amplitudes so that the spatial RF volume is comparable to that of data.
        Second step of scaling is done before convolving with the stimulus.
        :return:
        """

        df = self.all_fits_df.iloc[self.good_data_indices]
        data_pixel_len = 0.06  # in mm; pixel length 60 micrometers in dataset

        # Get mean center and surround RF size from data in millimeters
        mean_center_sd = np.mean(np.sqrt(df.semi_xc * df.semi_yc)) * data_pixel_len
        mean_surround_sd = (
            np.mean(np.sqrt((df.sur_ratio**2 * df.semi_xc * df.semi_yc)))
            * data_pixel_len
        )

        # For each model cell, set center amplitude as data_cen_mean**2 / sigma_x * sigma_y
        # For each model cell, scale surround amplitude by data_sur_mean**2 / sur_sigma_x * sur_sigma_y
        # (Volume of 2D Gaussian = 2 * pi * sigma_x*sigma_y)

        n_rgc = len(self.gc_df)
        amplitudec = np.zeros(n_rgc)
        # amplitudes = np.zeros(n_rgc)

        for i in range(n_rgc):
            amplitudec[i] = mean_center_sd**2 / (
                self.gc_df.iloc[i].semi_xc * self.gc_df.iloc[i].semi_yc
            )

        data_rel_sur_amplitude = self.gc_df["amplitudes"]
        self.gc_df["amplitudec"] = amplitudec
        self.gc_df["amplitudes"] = amplitudec * data_rel_sur_amplitude
        self.gc_df["relative_sur_amplitude"] = (
            self.gc_df["amplitudes"] / self.gc_df["amplitudec"]
        )

    def build(self):
        """
        Builds the receptive field mosaic
        :return:
        """
        # -- First, place the ganglion cell midpoints
        # Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
        gc_density_func_params = self._fit_gc_density_data()

        # Place ganglion cells to desired retina.
        self._place_gc_units(gc_density_func_params)

        # -- Second, endow cells with spatial receptive fields
        # Collect spatial statistics for receptive fields
        spatial_statistics_dict = self._fit_spatial_statistics()

        # Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dendr_diam_vs_eccentricity_parameters_dict = (
            self._fit_dendritic_diameter_vs_eccentricity()
        )

        # Construct spatial receptive fields. Centers are saved in the object
        self._place_spatial_receptive_fields(
            spatial_statistics_dict,
            dendr_diam_vs_eccentricity_parameters_dict,
        )

        # Scale center and surround amplitude so that Gaussian volume is preserved
        self._scale_both_amplitudes()  # TODO - what was the purpose of this?

        # At this point the spatial receptive fields are ready.
        # The positions are in gc_eccentricity, gc_polar_angle, and the rf parameters in gc_rf_models
        n_rgc = len(self.gc_df)

        # Summarize RF semi_xc and semi_yc as "RF radius" (geometric mean)
        self.gc_df["rf_radius"] = np.sqrt(self.gc_df.semi_xc * self.gc_df.semi_yc)

        # Finally, get non-spatial parameters
        temporal_statistics_df = self._fit_temporal_statistics()
        self._create_temporal_filters(temporal_statistics_df)

        td_shape, td_loc, td_scale = self._fit_tonic_drives()
        self.gc_df["tonicdrive"] = self._get_random_samples(
            td_shape, td_loc, td_scale, n_rgc, "gamma"
        )

        print("Built RGC mosaic with %d cells" % n_rgc)

    def save_mosaic(self, filename=None):

        output_folder = self.context.output_folder
        if filename is None:
            filepath = output_folder.joinpath(
                self.context.my_retina["mosaic_file_name"]
            )
        else:
            filepath = output_folder.joinpath(filename)

        print("Saving model mosaic to %s" % filepath)
        self.gc_df.to_csv(filepath)

    def show_build_process(self):
        """
        Show the process of building the mosaic
        self goes as argument, to be available for viz
        """

        self.viz.show_build_process(self, show_all_spatial_fits=False)

class WorkingRetina(RetinaMath):
    _properties_list = [
        "path",
        "output_folder",
        "my_retina",
        "my_stimulus_options",
    ]

    def __init__(self, context, data_io, viz) -> None:

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        # viz.client_object = self  # injecting client object pointer into viz object
        self._viz = viz

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def viz(self):
        return self._viz

    def initialize(self):
        """

        :param gc_dataframe: Ganglion cell parameters; positions are retinal coordinates; positions_eccentricity in mm, positions_polar_angle in degrees
        """

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

        # Metadata for Apricot dataset
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

        :param cell_index: int
        :return:
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

        :param cell_index: int
        :return:
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

    def _get_w_z_coords(self):
        """
        # Create w_coord, z_coord for cortical and visual coordinates, respectively
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

    def _save_for_cxsystem(self, spike_mons, filename=None, analog_signal=None):


        self.w_coord, self.z_coord = self._get_w_z_coords()

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

        :return: [xmin, xmax, ymin, ymax]
        """

        video_xmin_deg = self.stimulus_center.real - self.stimulus_width_deg / 2
        video_xmax_deg = self.stimulus_center.real + self.stimulus_width_deg / 2
        video_ymin_deg = self.stimulus_center.imag - self.stimulus_height_deg / 2
        video_ymax_deg = self.stimulus_center.imag + self.stimulus_height_deg / 2
        # left, right, bottom, top
        a = [video_xmin_deg, video_xmax_deg, video_ymin_deg, video_ymax_deg]

        return a

    def _initialize_digital_sampling(self):
        """
        Endows RGCs with stimulus/pixel space coordinates

        :return:
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

        :param cell_index: int
        :param reshape:
        :return:
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
            stimulus_cropped = stimulus_cropped.astype(np.int16)

        if reshape is True:
            sidelen = self.spatial_filter_sidelen
            n_frames = np.shape(self.stimulus_video.frames)[2]

            stimulus_cropped = np.reshape(stimulus_cropped, (sidelen**2, n_frames))

        return stimulus_cropped

    def load_stimulus(self, stimulus_video):
        """
        Loads stimulus video

        :param stimulus_video: VideoBaseClass, visual stimulus to project to the ganglion cell mosaic
        :return:
        """

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

        :param cell_index: int
        :return:
        """

        spatial_filter = self._create_spatial_filter(cell_index)
        s = self.spatial_filter_sidelen
        spatial_filter_1d = np.array([np.reshape(spatial_filter, s**2)]).T

        temporal_filter = self._create_temporal_filter(cell_index)

        spatiotemporal_filter = (
            spatial_filter_1d * temporal_filter
        )  # (Nx1) * (1xT) = NxT

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

        :param cell_index: int
        :return: array of length (stimulus timesteps)
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

        # pdb.set_trace()
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


# if __name__ == "__main__":
#     pass
