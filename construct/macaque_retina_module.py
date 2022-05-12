# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d

# Viz
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

# Comput Neurosci
import brian2 as b2
import brian2.units as b2u

# import neo
# from neo.io import NixIO
# import quantities as pq

# Local
# import utilities as ut # Where is this coming from?
from cxsystem2.core.tools import write_to_file, load_from_file
from construct import apricot_fitter_module as apricot
from viz.viz_module import Viz
from stimuli import visual_stimuli_module as vs
from vision_math.vision_math_module import Mathematics

# Builtin
import sys
from pathlib import Path
import os
from copy import deepcopy
import pdb


class MosaicConstructor(Mathematics, Viz):
    """
    Create the ganglion cell mosaic.
    All spatial parameters are saved to the dataframe *gc_df*
    """

    repo_path = Path(__file__).parent.parents[0]
    digitized_figures_path = repo_path / "construct/digitized_figures"
    # digitized_figures_path = repo_path

    def __init__(
        self,
        gc_type,
        response_type,
        ecc_limits,
        sector_limits,
        fits_from_file=None,
        model_density=1.0,
        randomize_position=0.7,
    ):
        """
        Initialize the ganglion cell mosaic

        :param gc_type: 'parasol' or 'midget'
        :param response_type: 'ON' or 'OFF'
        :param ecc_limits: [float, float], both in degrees
        :param sector_limits: [float, float], both in degrees
        :param model_density: float, arbitrary unit. Use to scale the desired number of cells.
        :param randomize_position: float, arbitrary unit. Use to scale the amount of randomization.
        """

        # Assertions
        assert (
            isinstance(ecc_limits, list) and len(ecc_limits) == 2
        ), "Wrong type or length of eccentricity, aborting"
        assert (
            isinstance(sector_limits, list) and len(sector_limits) == 2
        ), "Wrong type or length of theta, aborting"
        assert model_density <= 1.0, "Density should be <=1.0, aborting"

        # Proportion from all ganglion cells. Density of all ganglion cells is given later as a function of ecc from literature.
        proportion_of_parasol_gc_type = 0.08
        proportion_of_midget_gc_type = 0.64

        # Proportion of ON and OFF response type cells, assuming ON rf diameter = 1.2 x OFF rf diamter, and
        # coverage factor =1; Chichilnisky_2002_JNeurosci
        proportion_of_ON_response_type = 0.40
        proportion_of_OFF_response_type = 0.60

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

        self.deg_per_mm = (
            1 / 0.220
        )  # Turn deg2mm retina. One deg = 220um (Perry et al 1985). One mm retina is ~4.55 deg visual field.
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
            self.all_fits_df = apricot.ApricotFits(gc_type, response_type).get_fits()
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

    def get_random_samples(self, shape, loc, scale, n_cells, distribution):
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

    def read_gc_density_data(self):
        """
        Read re-digitized old literature data from mat files
        """
        digitized_figures_path = MosaicConstructor.digitized_figures_path
        print("Reading density data from:", digitized_figures_path)
        gc_density = sio.loadmat(
            digitized_figures_path / "Perry_1984_Neurosci_GCdensity_c.mat",
            variable_names=["Xdata", "Ydata"],
        )
        cell_eccentricity = np.squeeze(gc_density["Xdata"])
        cell_density = (
            np.squeeze(gc_density["Ydata"]) * 1e3
        )  # Cells are in thousands, thus the 1e3
        return cell_eccentricity, cell_density

    def fit_gc_density_data(self):
        """
        Fits a Gaussian to ganglion cell density (digitized data from Perry_1984).

        :returns a, x0, sigma, baseline (aka "gc_density_func_params")
        """

        cell_eccentricity, cell_density = self.read_gc_density_data()

        # Gaussian + baseline fit initial values for fitting
        scale, mean, sigma, baseline0 = 1000, 0, 2, np.min(cell_density)
        popt, pcov = opt.curve_fit(
            self.gauss_plus_baseline,
            cell_eccentricity,
            cell_density,
            p0=[scale, mean, sigma, baseline0],
        )

        return popt  # = gc_density_func_params

    def read_dendritic_fields_vs_eccentricity_data(self):
        """
        Read re-digitized old literature data from mat files
        """
        digitized_figures_path = MosaicConstructor.digitized_figures_path

        if self.gc_type == "parasol":
            dendr_diam1 = sio.loadmat(
                digitized_figures_path / "Perry_1984_Neurosci_ParasolDendrDiam_c.mat",
                variable_names=["Xdata", "Ydata"],
            )
            dendr_diam2 = sio.loadmat(
                digitized_figures_path
                / "Watanabe_1989_JCompNeurol_GCDendrDiam_parasol_c.mat",
                variable_names=["Xdata", "Ydata"],
            )
        elif self.gc_type == "midget":
            dendr_diam1 = sio.loadmat(
                digitized_figures_path / "Perry_1984_Neurosci_MidgetDendrDiam_c.mat",
                variable_names=["Xdata", "Ydata"],
            )
            dendr_diam2 = sio.loadmat(
                digitized_figures_path
                / "Watanabe_1989_JCompNeurol_GCDendrDiam_midget_c.mat",
                variable_names=["Xdata", "Ydata"],
            )

        return dendr_diam1, dendr_diam2

    def fit_dendritic_diameter_vs_eccentricity(self, viz_module=False):
        """
        Dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
        Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        """

        # Read dendritic field data and return linear fit with scipy.stats.linregress
        dendr_diam_parameters = {}

        dendr_diam1, dendr_diam2 = self.read_dendritic_fields_vs_eccentricity_data()

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

        if viz_module:
            # self.show_dendritic_diameter_vs_eccentricity(gc_type, data_all_x, data_all_y,
            # dataset_name='All data cubic fit', intercept=polynomials[3], slope=polynomials[2], square=polynomials[1], cube=polynomials[0])
            self.show_dendritic_diameter_vs_eccentricity(
                self.gc_type,
                data_all_x,
                data_all_y,
                polynomials,
                dataset_name="All data {0} fit".format(dendr_diam_model),
            )
            plt.show()

        return dendr_diam_parameters

    def densfunc(self, r, d0, beta):
        return d0 * (1 + beta * r) ** (-2)

    def place_gc_units(self, gc_density_func_params, viz_module=False):
        """
        Place ganglion cell center positions to retina

        :param gc_density_func_params:
        :param viz_module: True/False (default False)

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
            # my_gaussian_fit = self.gauss_plus_baseline(center_ecc, *gc_density_func_params)
            my_gaussian_fit = self.densfunc(center_ecc, 5.32043939e05, 2.64289725)
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

            # Randomize with respect to spacing
            # Randomization using uniform distribution [-0.5, 0.5]
            # matrix_polar_angle_randomized = matrix_polar_angle + theta_segment_angle * randomize_position \
            #                                 * (np.random.rand(matrix_polar_angle.shape[0],
            #                                                   matrix_polar_angle.shape[1]) - 0.5)
            # matrix_eccentricity_randomized = matrix_eccentricity + radius_segment_length * randomize_position \
            #                                  * (np.random.rand(matrix_eccentricity.shape[0],
            #                                                    matrix_eccentricity.shape[1]) - 0.5)
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

        # Visualize 2D retina with quality control for density
        # Pass the GC object to this guy, because the Visualize class is not inherited
        if viz_module:
            self.show_gc_positions_and_density(
                matrix_eccentricity_randomized_all,
                matrix_polar_angle_randomized_all,
                gc_density_func_params,
            )

    def fit_spatial_statistics(self, viz_module=False):
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

        # Quality control images
        if viz_module:
            self.show_spatial_statistics(
                ydata, spatial_statistics_dict, (x_model_fit, y_model_fit)
            )

        # Return stats for RF creation
        return spatial_statistics_dict

    def place_spatial_receptive_fields(
        self,
        spatial_statistics_dict,
        dendr_diam_vs_eccentricity_parameters_dict,
        viz_module=False,
    ):
        """
        Create spatial receptive fields to model cells.
        Starting from 2D difference-of-gaussian parameters:
        'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
        """

        # Get eccentricity data for all model cells
        # gc_eccentricity = self.gc_positions_eccentricity
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
            gc_rf_models[:, index] = self.get_random_samples(
                shape, loc, scale, n_cells, distribution
            )
        # Quality control images
        if viz_module:
            self.show_spatial_statistics(gc_rf_models, spatial_statistics_dict)

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
            # scaling_for_coverage_1 = (surface_area *1e6 ) / np.sum(area_of_rf[self.gc_df['eccentricity_group_index']==index])   # in micrometers2
            scaling_for_coverage_1 = (surface_area * 1e6) / np.sum(
                area_of_ellipse[self.gc_df["eccentricity_group_index"] == index]
            )  # in micrometers2

            # area_scaling_factors = area_of_rf / np.mean(area_of_ellipse)
            area_scaling_factors_coverage1[
                self.gc_df["eccentricity_group_index"] == index
            ] = scaling_for_coverage_1

        # area' = scaling factor * area
        # area_of_ellipse' = scaling_factor * area_of_ellipse
        # pi*a'*b' = scaling_factor * pi*a*b
        # a and b are the semi-major and semi minor axis, like radius
        # a'*a'*constant = scaling_factor * a * a * constant
        # a'/a = sqrt(scaling_factor)

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
        # semi_xc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:,0]
        # semi_yc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:,1]
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

        if viz_module:
            # Quality control for diameter distribution. In micrometers.
            gc_diameters = self.area2circle_diameter(
                self.ellipse2area(semi_xc, semi_yc)
            )

            polynomials = np.polyfit(gc_eccentricity, gc_diameters, polynomial_order)

            self.show_dendritic_diameter_vs_eccentricity(
                self.gc_type,
                gc_eccentricity,
                gc_diameters,
                polynomials,
                dataset_name="All data {0} fit".format(dendr_diam_model),
            )

            # gc_rf_models params: 'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
            # rho = self.gc_positions_eccentricity
            # phi = self.gc_positions_polar_angle
            rho = self.gc_df["positions_eccentricity"].values
            phi = self.gc_df["positions_polar_angle"].values

            self.show_gc_receptive_fields(
                rho, phi, gc_rf_models, surround_fixed=self.surround_fixed
            )

        # All ganglion cell spatial parameters are now saved to ganglion cell object dataframe gc_df

    def fit_tonic_drives(self, viz_module=False):
        tonicdrive_array = np.array(
            self.all_fits_df.iloc[self.good_data_indices].tonicdrive
        )
        shape, loc, scale = stats.gamma.fit(tonicdrive_array)

        if viz_module:
            x_min, x_max = stats.gamma.ppf(
                [0.001, 0.999], a=shape, loc=loc, scale=scale
            )
            xs = np.linspace(x_min, x_max, 100)
            plt.plot(xs, stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale))
            plt.hist(tonicdrive_array, density=True)
            plt.title(self.gc_type + " " + self.response_type)
            plt.xlabel("Tonic drive (a.u.)")
            plt.show()

        return shape, loc, scale

    def fit_temporal_statistics(self, viz_module=False):
        temporal_filter_parameters = ["n", "p1", "p2", "tau1", "tau2"]
        distrib_params = np.zeros((len(temporal_filter_parameters), 3))

        for i, param_name in enumerate(temporal_filter_parameters):
            param_array = np.array(
                self.all_fits_df.iloc[self.good_data_indices][param_name]
            )
            shape, loc, scale = stats.gamma.fit(param_array)
            distrib_params[i, :] = [shape, loc, scale]

        if viz_module:
            plt.subplots(2, 3)
            plt.suptitle(self.gc_type + " " + self.response_type)
            for i, param_name in enumerate(temporal_filter_parameters):
                plt.subplot(2, 3, i + 1)
                ax = plt.gca()
                shape, loc, scale = distrib_params[i, :]
                param_array = np.array(
                    self.all_fits_df.iloc[self.good_data_indices][param_name]
                )

                x_min, x_max = stats.gamma.ppf(
                    [0.001, 0.999], a=shape, loc=loc, scale=scale
                )
                xs = np.linspace(x_min, x_max, 100)
                ax.plot(xs, stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale))
                ax.hist(param_array, density=True)
                ax.set_title(param_name)

            plt.show()

        return pd.DataFrame(
            distrib_params,
            index=temporal_filter_parameters,
            columns=["shape", "loc", "scale"],
        )

    def create_temporal_filters(self, distrib_params_df, distribution="gamma"):

        n_rgc = len(self.gc_df)

        for param_name, row in distrib_params_df.iterrows():
            shape, loc, scale = row
            self.gc_df[param_name] = self.get_random_samples(
                shape, loc, scale, n_rgc, distribution
            )

    def scale_both_amplitudes(self):
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
            # amplitudes[i] = self.gc_df.iloc[i].amplitudes * (mean_surround_sd**2 / (self.gc_df.iloc[i].semi_xc * self.gc_df.iloc[i].semi_yc * self.gc_df.iloc[i].sur_ratio**2))

        data_rel_sur_amplitude = self.gc_df["amplitudes"]
        self.gc_df["amplitudec"] = amplitudec
        self.gc_df["amplitudes"] = amplitudec * data_rel_sur_amplitude
        self.gc_df["relative_sur_amplitude"] = (
            self.gc_df["amplitudes"] / self.gc_df["amplitudec"]
        )

    def visualize_mosaic(self):
        """
        Plots the full ganglion cell mosaic

        :return:
        """
        rho = self.gc_df["positions_eccentricity"].values
        phi = self.gc_df["positions_polar_angle"].values

        gc_rf_models = np.zeros((len(self.gc_df), 6))
        gc_rf_models[:, 0] = self.gc_df["semi_xc"]
        gc_rf_models[:, 1] = self.gc_df["semi_yc"]
        gc_rf_models[:, 2] = self.gc_df["xy_aspect_ratio"]
        gc_rf_models[:, 3] = self.gc_df["amplitudes"]
        gc_rf_models[:, 4] = self.gc_df["sur_ratio"]
        gc_rf_models[:, 5] = self.gc_df["orientation_center"]

        self.show_gc_receptive_fields(
            rho, phi, gc_rf_models, surround_fixed=self.surround_fixed
        )

    def build(self, viz_module=False):
        """
        Builds the receptive field mosaic
        :return:
        """
        # -- First, place the ganglion cell midpoints
        # Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
        gc_density_func_params = self.fit_gc_density_data()

        # Place ganglion cells to desired retina.
        self.place_gc_units(gc_density_func_params, viz_module=viz_module)

        # -- Second, endow cells with spatial receptive fields
        # Collect spatial statistics for receptive fields
        spatial_statistics_dict = self.fit_spatial_statistics(viz_module=viz_module)

        # Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dendr_diam_vs_eccentricity_parameters_dict = (
            self.fit_dendritic_diameter_vs_eccentricity(viz_module=viz_module)
        )

        # Construct spatial receptive fields. Centers are saved in the object
        self.place_spatial_receptive_fields(
            spatial_statistics_dict,
            dendr_diam_vs_eccentricity_parameters_dict,
            viz_module,
        )

        # Scale center and surround amplitude so that Gaussian volume is preserved
        self.scale_both_amplitudes()  # TODO - what was the purpose of this?

        # At this point the spatial receptive fields are ready.
        # The positions are in gc_eccentricity, gc_polar_angle, and the rf parameters in gc_rf_models
        n_rgc = len(self.gc_df)

        # Summarize RF semi_xc and semi_yc as "RF radius" (geometric mean)
        self.gc_df["rf_radius"] = np.sqrt(self.gc_df.semi_xc * self.gc_df.semi_yc)

        # Finally, get non-spatial parameters
        temporal_statistics_df = self.fit_temporal_statistics()
        self.create_temporal_filters(temporal_statistics_df)

        td_shape, td_loc, td_scale = self.fit_tonic_drives()
        self.gc_df["tonicdrive"] = self.get_random_samples(
            td_shape, td_loc, td_scale, n_rgc, "gamma"
        )

        print("Built RGC mosaic with %d cells" % n_rgc)

        if viz_module is True:
            plt.show()

    def save_mosaic(self, filepath):
        print("Saving model mosaic to %s" % filepath)
        self.gc_df.to_csv(filepath)


class FunctionalMosaic(Mathematics):
    def __init__(
        self,
        gc_dataframe,
        gc_type,
        response_type,
        stimulus_center=5 + 0j,
        stimulus_width_pix=240,
        stimulus_height_pix=240,
        pix_per_deg=60,
        fps=100,
    ):
        """

        :param gc_dataframe: Ganglion cell parameters; positions are retinal coordinates; positions_eccentricity in mm, positions_polar_angle in degrees
        """
        self.gc_type = gc_type
        self.response_type = response_type

        self.deg_per_mm = (
            1 / 0.220
        )  # Perry_1985_VisRes; 0.223 um/deg in the fovea, 169 um/deg at 90 deg ecc

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

        # Some settings related to plotting
        self.cmap_stim = "gray"
        self.cmap_spatial_filter = "bwr"

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

        self.initialize_digital_sampling()

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

    def get_extents_deg(self):
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

    def initialize_digital_sampling(self):
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

    def load_stimulus(self, stimulus_video, viz_module=False):
        """
        Loads stimulus video

        :param stimulus_video: VideoBaseClass, visual stimulus to project to the ganglion cell mosaic
        :param viz_module: True/False, show 1 frame of stimulus in pixel and visual coordinate systems (default False)
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
        xmin, xmax, ymin, ymax = self.get_extents_deg()
        for index, gc in self.gc_df_pixspace.iterrows():
            if (
                (gc.x_deg < xmin)
                | (gc.x_deg > xmax)
                | (gc.y_deg < ymin)
                | (gc.y_deg > ymax)
            ):
                self.gc_df.iloc[index] = 0.0  # all columns set as zero

        if viz_module is True:
            self.show_stimulus_with_gcs()

    def show_stimulus_with_gcs(self, frame_number=0, ax=None, example_gc=5):
        """
        Plots the 1SD ellipses of the RGC mosaic

        :param frame_number: int
        :param ax: matplotlib Axes object
        :return:
        """
        fig = plt.figure()
        ax = ax or plt.gca()
        ax.imshow(self.stimulus_video.frames[:, :, frame_number], vmin=0, vmax=255)
        ax = plt.gca()

        for index, gc in self.gc_df_pixspace.iterrows():
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
        locs = locs[locs < self.stimulus_height_pix]
        # locs=locs[locs>=0] # Including zero seems to shift center at least in deg
        locs = locs[locs > 0]

        # Set left y tick labels (pixels)
        left_y_labels = locs.astype(int)
        # plt.yticks(ticks=locs, labels=left_y_labels)
        plt.yticks(ticks=locs)
        ax.set_ylabel("pix")

        # Set x tick labels (degrees)
        xlocs = locs - np.mean(locs)
        down_x_labels = np.round(xlocs / self.pix_per_deg, decimals=2) + np.real(
            self.stimulus_center
        )
        plt.xticks(ticks=locs, labels=down_x_labels)
        ax.set_xlabel("deg")

        # Set right y tick labels (mm)
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.tick_params(axis="y")
        right_y_labels = np.round(
            (locs / self.pix_per_deg) / self.deg_per_mm, decimals=2
        )
        plt.yticks(ticks=locs, labels=right_y_labels)
        ax2.set_ylabel("mm")

        fig.tight_layout()

    def show_single_gc_view(self, cell_index, frame_number=0, ax=None):
        """
        Plots the stimulus frame cropped to RGC surroundings

        :param cell_index: int
        :param frame_number: int
        :param ax: matplotlib Axes object
        :return:
        """
        ax = ax or plt.gca()

        gc = self.gc_df_pixspace.iloc[cell_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)

        # Show stimulus frame cropped to RGC surroundings & overlay 1SD center RF on top of that
        ax.imshow(
            self.stimulus_video.frames[:, :, frame_number],
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

    def plot_tf_amplitude_response(self, cell_index, ax=None):

        ax = ax or plt.gca()

        tf = self._create_temporal_filter(cell_index)
        ft_tf = np.fft.fft(tf)
        timestep = self.data_filter_duration / len(tf) / 1000  # in seconds
        freqs = np.fft.fftfreq(tf.size, d=timestep)
        amplitudes = np.abs(ft_tf)

        ax.set_xscale("log")
        ax.set_xlim([0.1, 100])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        ax.plot(freqs, amplitudes, ".")

    def _create_postspike_filter(self, cell_index):
        raise NotImplementedError

    def create_spatiotemporal_filter(self, cell_index, viz_module=False):
        """
        Returns the outer product of the spatial and temporal filters

        :param cell_index: int
        :param viz_module: bool
        :return:
        """

        spatial_filter = self._create_spatial_filter(cell_index)
        s = self.spatial_filter_sidelen
        spatial_filter_1d = np.array([np.reshape(spatial_filter, s**2)]).T

        temporal_filter = self._create_temporal_filter(cell_index)

        spatiotemporal_filter = (
            spatial_filter_1d * temporal_filter
        )  # (Nx1) * (1xT) = NxT

        if viz_module is True:
            vmax = np.max(np.abs(spatial_filter))
            vmin = -vmax

            plt.subplots(1, 2, figsize=(10, 4))
            plt.suptitle(
                self.gc_type
                + " "
                + self.response_type
                + " / cell ix "
                + str(cell_index)
            )
            plt.subplot(121)
            plt.imshow(
                spatial_filter, cmap=self.cmap_spatial_filter, vmin=vmin, vmax=vmax
            )
            plt.colorbar()

            plt.subplot(122)
            plt.plot(range(self.temporal_filter_len), np.flip(temporal_filter))

            # plt.subplot(133)
            # plt.imshow(np.flip(spatiotemporal_filter, axis=1), aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
            # plt.colorbar()

            plt.tight_layout()

        return spatiotemporal_filter

    def get_cropped_video(self, cell_index, contrast=True, reshape=False):
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

        if reshape is True:
            s = self.spatial_filter_sidelen
            n_frames = np.shape(self.stimulus_video.frames)[2]

            stimulus_cropped = np.reshape(stimulus_cropped, (s**2, n_frames))

        return stimulus_cropped

    def plot_midpoint_contrast(self, cell_index, ax=None):
        """
        Plots the contrast in the mid-pixel of the stimulus cropped to RGC surroundings

        :param cell_index:
        :return:
        """
        stimulus_cropped = self.get_cropped_video(cell_index)

        midpoint_ix = (self.spatial_filter_sidelen - 1) // 2
        signal = stimulus_cropped[midpoint_ix, midpoint_ix, :]

        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        ax = ax or plt.gca()
        ax.plot(tvec, signal)
        ax.set_ylim([-1, 1])

    def plot_local_rms_contrast(self, cell_index, ax=None):
        """
        Plots local RMS contrast in the stimulus cropped to RGC surroundings.
        Note that is just a frame-by-frame computation, no averaging here

        :param cell_index:
        :return:
        """
        stimulus_cropped = self.get_cropped_video(
            cell_index, contrast=False
        )  # get stimulus intensities
        n_frames = self.stimulus_video.video_n_frames
        s = self.spatial_filter_sidelen
        signal = np.zeros(n_frames)

        for t in range(n_frames):
            frame_mean = np.mean(stimulus_cropped[:, :, t])
            squared_sum = np.sum((stimulus_cropped[:, :, t] - frame_mean) ** 2)
            signal[t] = np.sqrt(1 / (frame_mean**2 * b2u.s**2) * squared_sum)

        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        ax = ax or plt.gca()
        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

    def plot_local_michelson_contrast(self, cell_index, ax=None):
        """
        Plots local RMS contrast in the stimulus cropped to RGC surroundings.
        Note that is just a frame-by-frame computation, no averaging here

        :param cell_index:
        :return:
        """
        stimulus_cropped = self.get_cropped_video(
            cell_index, contrast=False
        )  # get stimulus intensities
        n_frames = self.stimulus_video.video_n_frames
        s = self.spatial_filter_sidelen
        signal = np.zeros(n_frames)

        for t in range(n_frames):
            frame_min = np.min(stimulus_cropped[:, :, t])
            frame_max = np.max(stimulus_cropped[:, :, t])
            signal[t] = (frame_max - frame_min) / (frame_max + frame_min)

        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        ax = ax or plt.gca()
        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

    def _generator_to_firing_rate(self, generator_potential):

        # firing_rate = np.exp(generator_potential)
        firing_rate = np.power(generator_potential, 2)

        return firing_rate

    def convolve_stimulus(self, cell_index, viz_module=False):
        """
        Convolves the stimulus with the stimulus filter

        :param cell_index: int
        :param viz_module: bool
        :return: array of length (stimulus timesteps)
        """
        # Get spatiotemporal filter
        spatiotemporal_filter = self.create_spatiotemporal_filter(
            cell_index, viz_module=False
        )
        # spatiotemporal_filter = self.create_spatiotemporal_filter(cell_index, viz_module=True)

        # Get cropped stimulus
        stimulus_cropped = self.get_cropped_video(cell_index, reshape=True)

        # Run convolution
        generator_potential = convolve(
            stimulus_cropped, spatiotemporal_filter, mode="valid"
        )
        generator_potential = generator_potential[0, :]

        # Add some padding to the beginning so that stimulus time and generator potential time match
        # (First time steps of stimulus are not convolved)
        video_dt = (1 / self.stimulus_video.fps) * b2u.second
        n_padding = int(
            self.data_filter_duration * b2u.ms / video_dt - 1
        )  # constant 49, comes from Apricot dataset. This might not be correct for short stimulus. Check SV 13.10.2020
        generator_potential = np.pad(
            generator_potential, (n_padding, 0), mode="constant", constant_values=0
        )

        tonic_drive = self.gc_df.iloc[cell_index].tonicdrive

        firing_rate = self._generator_to_firing_rate(generator_potential + tonic_drive)

        if viz_module is True:

            tvec = np.arange(0, len(generator_potential), 1) * video_dt

            plt.subplots(2, 1, sharex=True)
            plt.subplot(211)
            plt.plot(tvec, generator_potential + tonic_drive)
            plt.ylabel("Generator [a.u.]")

            plt.subplot(212)
            plt.plot(tvec, firing_rate)
            plt.xlabel("Time (s)]")
            plt.ylabel("Firing rate (Hz)]")

        # Return the 1-dimensional generator potential
        return generator_potential + tonic_drive

    def _old_style_visualization_for_run_cells(
        self,
        n_trials,
        n_cells,
        all_spiketrains,
        exp_generator_potential,
        duration,
        generator_potential,
        video_dt,
        tvec_new,
        viz_module=True,
    ):

        # Prepare data for manual visualization
        if n_trials > 1 and n_cells == 1:
            for_eventplot = np.array(all_spiketrains)
            for_histogram = np.concatenate(all_spiketrains)
            for_generatorplot = exp_generator_potential.flatten()
            n_samples = n_trials
            sample_name = "Trials"
        elif n_trials == 1 and n_cells > 1:
            for_eventplot = np.concatenate(all_spiketrains)
            for_histogram = np.concatenate(all_spiketrains[0])
            for_generatorplot = np.mean(exp_generator_potential, axis=1)
            n_samples = n_cells
            sample_name = "Cell #"
        else:
            viz_module = False
            print(
                "You attempted to viz_module gc activity, but you have either n_trials or n_cells must be 1, and the other > 1"
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

        # if spike_generator_model=='refractory':
        #     plt.subplots(2, 1, sharex=True)
        #     plt.subplot(211)
        #     plt.plot(   spiketrains[cell_index], np.ones(spiketrains[cell_index].shape) *
        #                 np.mean(state_monitor.lambda_ttlast[cell_index]), 'g+')
        #     # plt.plot(state_monitor.t, state_monitor.v[50])
        #     plt.plot(state_monitor.t, state_monitor.lambda_ttlast[cell_index])

        #     plt.xlim([0, duration / b2u.second])
        #     plt.ylabel('lambda_ttlast')

        #     plt.subplot(212)
        #     # Plot the generator and the average firing rate
        #     # plt.plot(state_monitor.t, state_monitor.ref[50])
        #     plt.plot(state_monitor.t, state_monitor.w[cell_index])
        #     plt.xlim([0, duration / b2u.second])
        #     plt.ylabel('w')

        #     plt.xlabel('Time (s)')

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
        # Andrew's answer: k=1/.082, a=. 077/.082
        a = 0.077 / 0.082  # ~ 0.94
        k = 1 / 0.082  # ~ 12.2
        w_coord = k * log(z_coord + a)

        return w_coord, z_coord

    def _save_for_cxsystem(self, spike_mons, filename=None, analog_signal=None):
        # Save to current working dir
        if filename is None:
            save_path = os.path.join(os.getcwd(), "most_recent_spikes")
        else:
            save_path = os.path.join(os.getcwd(), filename)

        self.output_file_extension = ".gz"

        self.w_coord, self.z_coord = self._get_w_z_coords()

        # Copied from CxSystem2\cxsystem2\core\stimuli.py The Stimuli class does not support reuse
        print(" -  Saving spikes, rgc coordinates and analog signal (if not None)...")
        self.generated_input_folder = save_path + self.output_file_extension
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

        write_to_file(save_path + self.output_file_extension, data_to_save)

    # def _save_for_neo(self, spike_mons, n_trials, n_cells, t_start, t_end, filename=None, analog_signal=None, analog_step=None):

    #     # Save to current working dir
    #     if filename is None:
    #         save_path = os.path.join(os.getcwd(),'most_recent_spikes_neo')
    #     else:
    #         save_path = os.path.join(os.getcwd(),filename)

    #     self.output_file_extension = '.h5' # NEO

    #     # Prep path
    #     print(" -  Saving spikes, rgc coordinates and analog signal (if not None)...")
    #     nix_fullpath = save_path + self.output_file_extension

    #     # create a new file overwriting any existing content
    #     nixfile = NixIO(filename=nix_fullpath, mode='ow') # modes 'ow' overwrite, 'rw' append?, 'ro' read only

    #     self.w_coord, self.z_coord = self._get_w_z_coords()
    #     # pdb.set_trace()
    #     # Prep Neo
    #     # Create Neo Block to contain all generated data
    #     block = neo.Block(name=filename)

    #     # # Create multiple Segments corresponding to trials
    #     # block.segments = [neo.Segment(index=i) for i in range(n_trials)]
    #     # Create one ChannelIndex (analog channels)
    #     block.channel_indexes = [neo.ChannelIndex(name='C%d' % i, index=i) for i in range(n_cells)]
    #     # Attach one Units (cells) to each ChannelIndex
    #     for idx, channel_idx in enumerate(block.channel_indexes):
    #         channel_idx.units = [neo.Unit('U%d' % i) for i in range(1)]
    #         channel_idx.index = np.array([idx])

    #     # Save spikes
    #     for idx2, channel_index in enumerate(block.channel_indexes):
    #         for idx, spike_monitor in zip(range(n_trials), spike_mons):
    #             spikes = spike_monitor.spike_trains()[idx2]
    #             train = neo.SpikeTrain( spikes,
    #                                     t_end[idx],
    #                                     t_start=t_start[idx],
    #                                     units='sec')
    #             train.name=f'Unit {idx2}, trial {idx}'
    #             # seg.spiketrains.append(train)
    #             channel_index.units[0].spiketrains.append(train)

    #         if analog_signal is not None:
    #             stepsize = (analog_step / b2u.second) * pq.s
    #             analog_sigarr = neo.AnalogSignal(   analog_signal[:,idx2],
    #                                             units="Hz",
    #                                             t_start=t_start[idx],
    #                                             sampling_period=stepsize)
    #             channel_index.analogsignals.append(analog_sigarr)

    #     # save nix to file
    #     nixfile.write_block(block)

    #     # close file
    #     nixfile.close()
    #     # pdb.set_trace()

    def run_cells(
        self,
        cell_index=None,
        n_trials=1,
        viz_module=False,
        save_data=False,
        spike_generator_model="refractory",
        return_monitor=False,
        filename=None,
    ):
        """
        Runs the LNP pipeline for a single ganglion cell (spiking by Brian2)

        :param cell_index: int or None. If None, run all cells
        :param n_trials: int
        :param viz_module: bool
        :param save_data: bool
        :param spike_generator_model: str, 'refractory' or 'poisson'
        :param return_monitor: bool, whether to return a raw Brian2 SpikeMonitor
        :param filename: str
        :return:
        """
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
                this_cell, viz_module=False
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
            # Create Brian PoissonGroup (inefficient implementation but nevermind)
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
            self._save_for_neo(
                spikemons,
                n_trials,
                n_cells,
                t_start,
                t_end,
                filename=filename,
                analog_signal=interpolated_rates_array,
                analog_step=poissongen_dt,
            )

            self._save_for_cxsystem(
                spikearrays, filename=filename, analog_signal=interpolated_rates_array
            )

        if viz_module is True:
            self._old_style_visualization_for_run_cells(
                n_trials,
                n_cells,
                all_spiketrains,
                exp_generator_potential,
                duration,
                generator_potential,
                video_dt,
                tvec_new,
            )

        if return_monitor is True:
            return spike_monitor
        else:
            return spiketrains, interpolated_rates_array.flatten()

    def run_all_cells(
        self, spike_generator_model="refractory", save_data=False, viz_module=False
    ):

        """
        Runs the LNP pipeline for all ganglion cells (legacy function)

        :param viz_module: bool
        :param spike_generator_model: str, 'refractory' or 'poisson'
        :param save_data: bool
        :return:
        """

        self.run_cells(
            cell_index=None,
            n_trials=1,
            spike_generator_model=spike_generator_model,
            save_data=save_data,
            viz_module=viz_module,
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
        spikes_df.to_csv(filename, index=False, header=False)

    def save_structure_csv(self, filename=None):
        """
        Saves x,y coordinates of model cells to a csv file (for use in ViSimpl).

        :param filename: str
        :return:
        """
        if filename is None:
            filename = self.gc_type + "_" + self.response_type + "_structure.csv"

        rgc_coords = self.gc_df[["x_deg", "y_deg"]].copy()
        rgc_coords["z_deg"] = 0.0

        rgc_coords.to_csv(filename, header=False, index=False)


# if __name__ == "__main__":
#     '''
#     Build and test your retina here, one gc type at a time. Temporal hemiretina of macaques.
#     '''
#     mosaic = MosaicConstructor(gc_type='parasol', response_type='on', ecc_limits=[4.8, 5.2],
#                                sector_limits=[-.4, .4], model_density=1.0, randomize_position=0.05)

#     mosaic.build()
#     mosaic.save_mosaic('parasol_on_single.csv')

#     testmosaic = pd.read_csv('parasol_on_single.csv', index_col=0)


#     ret = FunctionalMosaic(testmosaic, 'parasol', 'on', stimulus_center=5+0j,
#                            stimulus_width_pix=240, stimulus_height_pix=240)


#     stim = vs.ConstructStimulus(pattern='temporal_square_pattern', stimulus_form='circular',
#                                 temporal_frequency=0.1, spatial_frequency=1.0, stimulus_position=(-.06, 0.03),
#                                 duration_seconds=.4, image_width=240, image_height=240,
#                                 stimulus_size=.1, contrast=.99, baseline_start_seconds = 0.5,
#                                 baseline_end_seconds = 0.5, background=128, mean=128, phase_shift=0)  # np.pi+(np.pi/100)

#     stim.save_stimulus_to_videofile(filename='tmp')

#     # ret.load_stimulus(grating)
#     ret.load_stimulus(stim)

#     # # movie = vs.NaturalMovie('/home/henhok/nature4_orig35_fps100.avi', fps=100, pix_per_deg=60)
#     # movie = vs.NaturalMovie(r'C:\Users\Simo\Laskenta\Stimuli\videoita\naturevids\nature1.avi', fps=100, pix_per_deg=60)
#     # ret.load_stimulus(movie)

#     # ret.plot_midpoint_contrast(0)
#     # plt.show()
#     # ret.plot_local_rms_contrast(0)
#     # plt.show()
#     # ret.plot_local_michelson_contrast(0)
#     # plt.show()

#     example_gc=2 # int or 'None'
#     # ret.convolve_stimulus(example_gc, viz_module=True)
#     # plt.show()

#     # ret.run_single_cell(example_gc, n_trials=100, viz_module=True,
#     #                     spike_generator_model='poisson', save_example_data=True) # 'refractory'
#     # plt.show(block = False)

#     filenames = [f'Response_foo_{x}' for x in np.arange(1)]

#     for filename in filenames:

#         ret.run_cells(cell_index=example_gc, n_trials=200, viz_module=True, save_data=False,
#                         spike_generator_model='poisson', return_monitor=False, filename=filename)
#     # plt.show(block = False)

#     # # ret.run_all_cells(viz_module=True, spike_generator_model='refractory', reload_last=False)
#     # # plt.show(block = False)
#     # # ret.save_spikes_csv()

#     ret.show_stimulus_with_gcs(example_gc=example_gc, frame_number=51)
#     # ret.show_single_gc_view(cell_index=example_gc, frame_number=21)
#     # plt.show(block = False)

#     # # ret.show_analysis(filename='my_analysis', viz_module=True)
#     plt.show()

# '''
# This is code for building macaque retinal filters corresponding to midget and parasol cells' responses.
# We keep modular code structure, to be able to add new features at later phase.

# The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

# Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995) was approksimated with relation 5 deg/mm.
# This works fine up to 20 deg ecc, but underestimates the distance thereafter. If more peripheral representations are later
# necessary, the millimeters should be calculates by inverting the inverting the relation
# A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974; Dacey and Petersen, 1992)

# We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

# The density of many cell types is inversely proportional to dendritic field coverage,
# suggesting constant coverage factor (Perry_1984_Neurosci, Wassle_1991_PhysRev).
# Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, Lee_2010_ProgRetEyeRes).
# Parasol coverage factor is 3-4 close to fovea (Grunert_1993_VisRes); 2-7 according to Perry_1984_Neurosci.
# These include ON- and OFF-center cells, and perhaps other cell types.
# It is likely that coverage factor is 1 for midget and parasol ON- and OFF-center cells each,
# which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

# The spatiotemporal receptive fields for the four cell types (parasol & midget, ON & OFF) were modelled with double ellipsoid
# difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is
# described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci; Field_2010_Nature.

# Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells.
# He also shows that OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to implement
# if necessary).

# NOTE: bad cell indices hard coded from Chichilnisky apricot data. For another data set, viz_module fits, and change the bad cells.
# NOTE: If eccentricity stays under 20 deg, dendritic diameter data fitted up to 25 deg only (better fit close to fovea)

# -center-surround response ratio (in vivo, anesthetized, recorded from LGN; Croner_1995_VisRes) PC: ; MC: ;
# -Michelson contrast definition for sinusoidal gratings (Croner_1995_VisRes).
# -optical quality probably poses no major limit to behaviorally measured spatial vision (Williams_1981_IOVS).
# -spatial contrast sensitivity nonlinearity in the center subunits is omitted. This might reduce sensitivity to natural scenes Turner_2018_eLife.

# -quality control: compare to Watanabe_1989_JCompNeurol
#     -dendritic diameter scatter is on average (lower,upper quartile) 21.3% of the median diameter in the local area

#     Parasol dendritic field diameter: temporal retina 51.8 microm + ecc(mm) * 20.6 microm/mm,
#     nasal retina; 115.5 microm + ecc(mm) * 6.97 microm/mm

# '''
