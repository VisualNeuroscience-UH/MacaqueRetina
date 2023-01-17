# Numerical
# from fileinput import filename
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd

# from scipy.signal import convolve
# from scipy.interpolate import interp1d

# Data IO
import cv2

# Viz
from tqdm import tqdm

# Comput Neurosci
# import brian2 as b2
# import brian2.units as b2u

# Local
# from cxsystem2.core.tools import write_to_file, load_from_file
from retina.apricot_fit_module import ApricotFit
from retina.retina_math_module import RetinaMath

# from retina.vae_module import ApricotVAE
from retina.vae_module import VAE
from retina.gan_module import GAN

# Builtin
# import sys
from pathlib import Path

# import os
# from copy import deepcopy
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
        -sets ConstructRetina instance parameters from conf file my_retina
        -inits gc_df to hold the final ganglion cell mosaics
        If the model_type if FIT:
            Calls ApricotFit to fit RF parameters to the data

        Parameters
        ----------
        fits_from_file : str
            Path to a file containing the fits. If None, fits are computed from scratch

        Returns
        -------
        None
        """

        my_retina = self.context.my_retina
        gc_type = my_retina["gc_type"]
        response_type = my_retina["response_type"]
        ecc_limits = my_retina["ecc_limits"]
        sector_limits = my_retina["sector_limits"]
        model_density = my_retina["model_density"]
        randomize_position = my_retina["randomize_position"]
        self.deg_per_mm = my_retina["deg_per_mm"]

        self.model_type = my_retina["model_type"]

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

        if self.model_type == "FIT":
            self.dendr_diam_model = "quadratic"  # 'linear' # 'quadratic' # cubic

            # If surround is fixed, the surround position, semi_x, semi_y (aspect_ratio) and orientation are the same as center params. This appears to give better results.
            self.surround_fixed = 1

            # # Set stimulus stuff
            # self.stimulus_video = None

            # Make or read fits
            if fits_from_file is None:
                # init and call -- only connection to apricot_fit_module
                (
                    self.all_fits_df,
                    self.temporal_filters_to_show,
                    self.spatial_filters_to_show,
                ) = ApricotFit(
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
            print("Back from FIT!")
        # elif self.model_type == "VAE":
        #     # Fit variational autoencoder to generate ganglion cells
        #     self.vae_model = ApricotVAE(
        #         self.context.apricot_data_folder, gc_type, response_type
        #     )
        #     print("Back to ConstructRetina!")
        elif self.model_type == "VAE":
            # Fit variational autoencoder to generate ganglion cells
            self.vae_model = VAE(
                self.context.apricot_data_folder, gc_type, response_type
            )
            print("Back from VAE!")
        elif self.model_type == "GAN":
            # Fit variational autoencoder to generate ganglion cells
            self.vae_model = GAN(
                self.context.apricot_data_folder, gc_type, response_type
            )
            print("Back from GAN!")

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

    def _create_spatial_receptive_fields(
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
        self.gc_df["eccentricity_group_index"] = gc_eccentricity_group_index.astype(
            np.uint32
        )
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
        Builds the receptive field mosaic. This is the main function to call.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.initialized is False:
            self._initialize()
        # return  # temporary jump back to config file -- this time for profiling

        # -- First, place the ganglion cell midpoints
        # Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
        gc_density_func_params = self._fit_gc_density_data()

        # Place ganglion cells to desired retina.
        self._place_gc_units(gc_density_func_params)

        if self.model_type == "FIT":
            # -- Second, endow cells with spatial receptive fields
            # Collect spatial statistics for receptive fields
            spatial_statistics_dict = self._fit_spatial_statistics()

            # Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
            # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
            dendr_diam_vs_eccentricity_parameters_dict = (
                self._fit_dendritic_diameter_vs_eccentricity()
            )

            # Create spatial receptive fields.
            self._create_spatial_receptive_fields(
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

            # Create temporal receptive fields
            self._create_temporal_filters(temporal_statistics_df, distribution="gamma")

            td_shape, td_loc, td_scale = self._fit_tonic_drives()

            # Create tonic drive
            self.gc_df["tonicdrive"] = self._get_random_samples(
                td_shape, td_loc, td_scale, n_rgc, "gamma"
            )
        elif self.model_type == "VAE":
            # Use the generative variational autoencoder model to provide spatial and temporal receptive fields
            pass
        else:
            raise ValueError("Model type not recognized")

        print("Built RGC mosaic with %d cells" % n_rgc)

    def save_mosaic(self, filename=None):
        """
        Save the mosaic to a csv file

        Parameters
        ----------
        filename : pathlib Path object, str or None

        Returns
        -------
        None
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

    def show_build_process(self, show_all_spatial_fits=False):
        """
        Show the process of building the mosaic
        self goes as argument, to be available for viz
        """

        self.viz.show_build_process(self, show_all_spatial_fits=show_all_spatial_fits)
