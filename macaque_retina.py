import sys
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
from pathlib import Path
#import seaborn as sns
import visual_stimuli as vs
from visualize import Visualize
from vision_maths import Mathematics
from scipy.signal import fftconvolve, convolve
from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm
from mpl_toolkits import mplot3d
import brian2 as b2
from brian2.units import *
import apricot_fitter as apricot

# plt.rcParams['image.cmap'] = 'gray'


class MosaicConstructor(Mathematics, Visualize):
    '''
    Create the ganglion cell mosaic.
    All spatial parameters are saved to the dataframe *gc_df*
    '''

    script_path = Path(__file__).parent
    digitized_figures_path = script_path

    def __init__(self, gc_type, response_type, ecc_limits, sector_limits, fits_from_file=None, model_density=1.0, randomize_position=0.7):
        '''
        Initialize the ganglion cell mosaic

        :param gc_type: 'parasol' or 'midget'
        :param response_type: 'ON' or 'OFF'
        :param ecc_limits: [float, float], both in degrees
        :param sector_limits: [float, float], both in degrees
        :param model_density: float, arbitrary unit. Use to scale the desired number of cells.
        :param randomize_position: float, arbitrary unit. Use to scale the amount of randomization.
        '''

        # Assertions
        assert isinstance(ecc_limits, list) and len(ecc_limits) == 2, 'Wrong type or length of eccentricity, aborting'
        assert isinstance(sector_limits, list) and len(sector_limits) == 2, 'Wrong type or length of theta, aborting'
        assert model_density <= 1.0, 'Density should be <=1.0, aborting'

        # Proportion from all ganglion cells. Density of all ganglion cells is given later as a function of ecc from literature.
        proportion_of_parasol_gc_type = 0.1
        proportion_of_midget_gc_type = 0.8

        # Proportion of ON and OFF response type cells, assuming ON rf diameter = 1.2 x OFF rf diamter, and
        # coverage factor =1; Chichilnisky_2002_JNeurosci
        proportion_of_ON_response_type = 0.41
        proportion_of_OFF_response_type = 0.59

        # GC type specifications self.gc_proportion
        gc_type = gc_type.lower()
        response_type = response_type.lower()
        if all([gc_type == 'parasol', response_type == 'on']):
            self.gc_proportion = proportion_of_parasol_gc_type * proportion_of_ON_response_type * model_density
        elif all([gc_type == 'parasol', response_type == 'off']):
            self.gc_proportion = proportion_of_parasol_gc_type * proportion_of_OFF_response_type * model_density
        elif all([gc_type == 'midget', response_type == 'on']):
            self.gc_proportion = proportion_of_midget_gc_type * proportion_of_ON_response_type * model_density
        elif all([gc_type == 'midget', response_type == 'off']):
            self.gc_proportion = proportion_of_midget_gc_type * proportion_of_OFF_response_type * model_density
        else:
            print('Unkown ganglion cell type, aborting')
            sys.exit()

        self.gc_type = gc_type
        self.response_type = response_type

        self.deg_per_mm = 5  # Turn deg2mm retina. One mm retina is 5 deg visual field.
        self.eccentricity = ecc_limits
        self.eccentricity_in_mm = np.asarray([r / self.deg_per_mm for r in ecc_limits])  # Turn list to numpy array
        self.theta = np.asarray(sector_limits)  # Turn list to numpy array
        self.randomize_position = randomize_position
        self.dendr_diam_model = 'quadratic'  # 'linear' # 'quadratic' # cubic

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
        columns = ['positions_eccentricity', 'positions_polar_angle', 'eccentricity_group_index', 'semi_xc', 'semi_yc',
                   'xy_aspect_ratio', 'amplitudes', 'sur_ratio', 'orientation_center']
        self.gc_df = pd.DataFrame(columns=columns)

        # Set stimulus stuff
        self.stimulus_video = None

        # Make or read fits
        if fits_from_file is None:
            self.all_fits_df = apricot.ApricotFits(gc_type, response_type).get_fits()
        else:
            self.all_fits_df = pd.read_csv(fits_from_file, header=0, index_col=0).fillna(0.0)

        self.n_cells_data = len(self.all_fits_df)
        self.bad_data_indices = np.where((self.all_fits_df == 0.0).all(axis=1))[0].tolist()
        self.good_data_indices = np.setdiff1d(range(self.n_cells_data), self.bad_data_indices)


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
        assert distribution in ['gamma', 'beta', 'skewnorm'], "Distribution not supported"

        if distribution == 'gamma':
            distribution_parameters = stats.gamma.rvs(a=shape, loc=loc, scale=scale, size=n_cells,
                                                      random_state=None)  # random_state is the seed
        elif distribution == 'beta':
            distribution_parameters = stats.beta.rvs(a=shape[0], b=shape[1], loc=loc, scale=scale, size=n_cells,
                                                     random_state=None)  # random_state is the seed
        elif distribution == 'skewnorm':
            distribution_parameters = stats.skewnorm.rvs(a=shape, loc=loc, scale=scale, size=n_cells,
                                                         random_state=None)

        return distribution_parameters

    def read_gc_density_data(self):
        '''
        Read re-digitized old literature data from mat files
        '''
        digitized_figures_path = MosaicConstructor.digitized_figures_path

        gc_density = sio.loadmat(digitized_figures_path / 'Perry_1984_Neurosci_GCdensity_c.mat',
                                 variable_names=['Xdata', 'Ydata'])
        cell_eccentricity = np.squeeze(gc_density['Xdata'])
        cell_density = np.squeeze(gc_density['Ydata']) * 1e3  # Cells are in thousands, thus the 1e3
        return cell_eccentricity, cell_density

    def fit_gc_density_data(self):
        """
        Fits a Gaussian to ganglion cell density (digitized data from Perry_1984).

        :returns a, x0, sigma, baseline (aka "gc_density_func_params")
        """

        cell_eccentricity, cell_density = self.read_gc_density_data()

        # Gaussian + baseline fit initial values for fitting
        scale, mean, sigma, baseline0 = 1000, 0, 2, np.min(cell_density)
        popt, pcov = opt.curve_fit(self.gauss_plus_baseline, cell_eccentricity, cell_density,
                                   p0=[scale, mean, sigma, baseline0])

        return popt  # = gc_density_func_params

    def read_dendritic_fields_vs_eccentricity_data(self):
        '''
        Read re-digitized old literature data from mat files
        '''
        digitized_figures_path = MosaicConstructor.digitized_figures_path

        if self.gc_type == 'parasol':
            dendr_diam1 = sio.loadmat(digitized_figures_path / 'Perry_1984_Neurosci_ParasolDendrDiam_c.mat',
                                      variable_names=['Xdata', 'Ydata'])
            dendr_diam2 = sio.loadmat(digitized_figures_path / 'Watanabe_1989_JCompNeurol_GCDendrDiam_parasol_c.mat',
                                      variable_names=['Xdata', 'Ydata'])
        elif self.gc_type == 'midget':
            dendr_diam1 = sio.loadmat(digitized_figures_path / 'Perry_1984_Neurosci_MidgetDendrDiam_c.mat',
                                      variable_names=['Xdata', 'Ydata'])
            dendr_diam2 = sio.loadmat(digitized_figures_path / 'Watanabe_1989_JCompNeurol_GCDendrDiam_midget_c.mat',
                                      variable_names=['Xdata', 'Ydata'])

        return dendr_diam1, dendr_diam2

    def fit_dendritic_diameter_vs_eccentricity(self, visualize=False):
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
        data_set_1_x = np.squeeze(dendr_diam1['Xdata'])
        data_set_1_y = np.squeeze(dendr_diam1['Ydata'])
        data_set_2_x = np.squeeze(dendr_diam2['Xdata'])
        data_set_2_y = np.squeeze(dendr_diam2['Ydata'])

        # Both datasets together
        data_all_x = np.concatenate((data_set_1_x, data_set_2_x))
        data_all_y = np.concatenate((data_set_1_y, data_set_2_y))

        # Limit eccentricities for central visual field studies to get better approximation at about 5 deg ecc (1mm)
        data_all_x_index = data_all_x <= self.visual_field_fit_limit
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[data_all_x_index]  # Don't forget to truncate values, too

        # Sort to ascending order
        data_all_x_index = np.argsort(data_all_x)
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[data_all_x_index]

        # Get rf diameter vs eccentricity
        dendr_diam_model = self.dendr_diam_model  # 'linear' # 'quadratic' # cubic
        dict_key = '{0}_{1}'.format(self.gc_type, dendr_diam_model)

        if dendr_diam_model == 'linear':
            polynomial_order = 1
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {'intercept': polynomials[1], 'slope': polynomials[0]}
        elif dendr_diam_model == 'quadratic':
            polynomial_order = 2
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {'intercept': polynomials[2], 'slope': polynomials[1],
                                               'square': polynomials[0]}
        elif dendr_diam_model == 'cubic':
            polynomial_order = 3
            polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {'intercept': polynomials[3], 'slope': polynomials[2],
                                               'square': polynomials[1], 'cube': polynomials[0]}

        if visualize:
            # self.show_dendritic_diameter_vs_eccentricity(gc_type, data_all_x, data_all_y,
            # dataset_name='All data cubic fit', intercept=polynomials[3], slope=polynomials[2], square=polynomials[1], cube=polynomials[0])
            self.show_dendritic_diameter_vs_eccentricity(self.gc_type, data_all_x, data_all_y, polynomials,
                                                         dataset_name='All data {0} fit'.format(dendr_diam_model))
            plt.show()

        return dendr_diam_parameters

    def place_gc_units(self, gc_density_func_params, visualize=False):
        """
        Place ganglion cell center positions to retina

        :param gc_density_func_params:
        :param visualize: True/False (default False)

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
        eccentricity_steps = np.linspace(eccentricity_in_mm_total[0], eccentricity_in_mm_total[1], 1 + n_steps)

        # Initialize position arrays
        matrix_polar_angle_randomized_all = np.asarray([])
        matrix_eccentricity_randomized_all = np.asarray([])
        gc_eccentricity_group_index = np.asarray([])

        true_eccentricity_end = []
        sector_surface_area_all = []
        for eccentricity_group_index, current_step in enumerate(np.arange(int(n_steps))):

            if true_eccentricity_end:  # If the eccentricity has been adjusted below inside the loop
                eccentricity_in_mm = np.asarray([true_eccentricity_end, eccentricity_steps[current_step + 1]])
            else:
                eccentricity_in_mm = np.asarray(
                    [eccentricity_steps[current_step], eccentricity_steps[current_step + 1]])

            # fetch center ecc in mm
            center_ecc = np.mean(eccentricity_in_mm)

            # rotate theta to start from 0
            theta_rotated = theta - np.min(theta)
            angle = np.max(theta_rotated)  # The angle is now == max theta

            # Calculate area
            assert eccentricity_in_mm[0] < eccentricity_in_mm[1], 'Radii in wrong order, give [min max], aborting'
            sector_area_remove = self.sector2area(eccentricity_in_mm[0], angle)
            sector_area_full = self.sector2area(eccentricity_in_mm[1], angle)
            sector_surface_area = sector_area_full - sector_area_remove  # in mm2
            sector_surface_area_all.append(sector_surface_area)  # collect sector area for each ecc step

            # N cells for given ecc
            my_gaussian_fit = self.gauss_plus_baseline(center_ecc, *gc_density_func_params)
            Ncells = sector_surface_area * my_gaussian_fit * self.gc_proportion

            # place cells in regular grid
            # Vector of cell positions in radial and polar directions. Angle in degrees.
            inner_arc_in_mm = (angle / 360) * 2 * np.pi * eccentricity_in_mm[0]
            delta_eccentricity_in_mm = eccentricity_in_mm[1] - eccentricity_in_mm[0]

            # By assuming that the ratio of the number of points in x and y direction respects
            # the sector's aspect ratio, ie.
            # n_segments_arc / n_segments_eccentricity = inner_arc_in_mm / delta_eccentricity_in_mm
            # we get:
            n_segments_arc = np.sqrt(Ncells * (inner_arc_in_mm / delta_eccentricity_in_mm))
            n_segments_eccentricity = np.sqrt(Ncells * (delta_eccentricity_in_mm / inner_arc_in_mm))
            # Because n_segments_arc and n_segments_eccentricity can be floats, we round them to integers
            int_n_segments_arc = int(round(n_segments_arc))
            int_n_segments_eccentricity = int(round(n_segments_eccentricity))

            # Recalc delta_eccentricity_in_mm given the n segments to avoid non-continuous cell densities
            true_n_cells = int_n_segments_arc * int_n_segments_eccentricity
            true_sector_area = true_n_cells / (my_gaussian_fit * self.gc_proportion)
            true_delta_eccentricity_in_mm = (int_n_segments_eccentricity / int_n_segments_arc) * inner_arc_in_mm

            radius_segment_length = true_delta_eccentricity_in_mm / int_n_segments_eccentricity
            theta_segment_angle = angle / int_n_segments_arc  # Note that this is different from inner_arc_in_mm / int_n_segments_arc

            # Set the true_eccentricity_end
            true_eccentricity_end = eccentricity_in_mm[0] + true_delta_eccentricity_in_mm

            vector_polar_angle = np.linspace(theta[0], theta[1], int_n_segments_arc)
            vector_eccentricity = np.linspace(eccentricity_in_mm[0], true_eccentricity_end - radius_segment_length,
                                              int_n_segments_eccentricity)

            # meshgrid and shift every second column to get good GC tiling
            matrix_polar_angle, matrix_eccentricity = np.meshgrid(vector_polar_angle, vector_eccentricity)
            matrix_polar_angle[::2] = matrix_polar_angle[::2] + (
                        angle / (2 * n_segments_arc))  # shift half the inter-cell angle

            # Randomize with respect to spacing
            # Randomization using uniform distribution [-0.5, 0.5]
            # matrix_polar_angle_randomized = matrix_polar_angle + theta_segment_angle * randomize_position \
            #                                 * (np.random.rand(matrix_polar_angle.shape[0],
            #                                                   matrix_polar_angle.shape[1]) - 0.5)
            # matrix_eccentricity_randomized = matrix_eccentricity + radius_segment_length * randomize_position \
            #                                  * (np.random.rand(matrix_eccentricity.shape[0],
            #                                                    matrix_eccentricity.shape[1]) - 0.5)
            # Randomization using normal distribution
            matrix_polar_angle_randomized = matrix_polar_angle + theta_segment_angle * randomize_position \
                                            * (np.random.randn(matrix_polar_angle.shape[0],
                                                              matrix_polar_angle.shape[1]))
            matrix_eccentricity_randomized = matrix_eccentricity + radius_segment_length * randomize_position \
                                             * (np.random.randn(matrix_eccentricity.shape[0],
                                                               matrix_eccentricity.shape[1]))

            matrix_polar_angle_randomized_all = np.append(matrix_polar_angle_randomized_all,
                                                          matrix_polar_angle_randomized.flatten())
            matrix_eccentricity_randomized_all = np.append(matrix_eccentricity_randomized_all,
                                                           matrix_eccentricity_randomized.flatten())

            assert true_n_cells == len(matrix_eccentricity_randomized.flatten()), "N cells dont match, check the code"
            gc_eccentricity_group_index = np.append(gc_eccentricity_group_index,
                                                    np.ones(true_n_cells) * eccentricity_group_index)

        # Save cell position data to current ganglion cell object
        self.gc_df['positions_eccentricity'] = matrix_eccentricity_randomized_all
        self.gc_df['positions_polar_angle'] = matrix_polar_angle_randomized_all
        self.gc_df['eccentricity_group_index'] = gc_eccentricity_group_index.astype(int)
        self.sector_surface_area_all = np.asarray(sector_surface_area_all)

        # Visualize 2D retina with quality control for density
        # Pass the GC object to this guy, because the Visualize class is not inherited
        if visualize:
            self.show_gc_positions_and_density(matrix_eccentricity_randomized_all,
                                               matrix_polar_angle_randomized_all, gc_density_func_params)

    def fit_spatial_statistics(self, visualize=False):
        """
        Collect spatial statistics from Chichilnisky receptive field data
        """

        # parameter_names, data_all_viable_cells, bad_cell_indices = fitdata
        data_all_viable_cells = np.array(self.all_fits_df)
        bad_cell_indices = np.where((self.all_fits_df == 0.0).all(axis=1))[0].tolist()
        parameter_names = self.all_fits_df.columns.tolist()

        all_viable_cells = np.delete(data_all_viable_cells, bad_cell_indices, 0)

        chichilnisky_data_df = pd.DataFrame(data=all_viable_cells, columns=parameter_names)

        # Save stats description to gc object
        self.rf_datafit_description_series = chichilnisky_data_df.describe()

        # Calculate xy_aspect_ratio
        xy_aspect_ratio_pd_series = chichilnisky_data_df['semi_yc'] / chichilnisky_data_df['semi_xc']
        xy_aspect_ratio_pd_series.rename('xy_aspect_ratio')
        chichilnisky_data_df['xy_aspect_ratio'] = xy_aspect_ratio_pd_series

        rf_parameter_names = ['semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes', 'sur_ratio', 'orientation_center']
        self.rf_parameter_names = rf_parameter_names  # For reference
        n_distributions = len(rf_parameter_names)
        shape = np.zeros([n_distributions - 1])  # orientation_center has two shape parameters, below alpha and beta
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
            shape[index], loc[index], scale[index] = stats.gamma.fit(ydata[:, index], loc=0)
            x_model_fit[:, index] = np.linspace(
                stats.gamma.ppf(0.001, shape[index], loc=loc[index], scale=scale[index]),
                stats.gamma.ppf(0.999, shape[index], loc=loc[index], scale=scale[index]), 100)
            y_model_fit[:, index] = stats.gamma.pdf(x=x_model_fit[:, index], a=shape[index], loc=loc[index],
                                                    scale=scale[index])

            # Collect parameters
            spatial_statistics_dict[distribution] = {'shape': shape[index], 'loc': loc[index], 'scale': scale[index],
                                                     'distribution': 'gamma'}

        # Model orientation distribution with beta function.
        index += 1
        ydata[:, index] = chichilnisky_data_df[rf_parameter_names[-1]]
        a_parameter, b_parameter, loc[index], scale[index] = stats.beta.fit(ydata[:, index], 0.6, 0.6,
                                                                            loc=0)  # initial guess for a_parameter and b_parameter is 0.6
        x_model_fit[:, index] = np.linspace(
            stats.beta.ppf(0.001, a_parameter, b_parameter, loc=loc[index], scale=scale[index]),
            stats.beta.ppf(0.999, a_parameter, b_parameter, loc=loc[index], scale=scale[index]), 100)
        y_model_fit[:, index] = stats.beta.pdf(x=x_model_fit[:, index], a=a_parameter, b=b_parameter, loc=loc[index],
                                               scale=scale[index])
        spatial_statistics_dict[rf_parameter_names[-1]] = {'shape': (a_parameter, b_parameter), 'loc': loc[index],
                                                           'scale': scale[index], 'distribution': 'beta'}

        # Quality control images
        if visualize:
            self.show_spatial_statistics(ydata, spatial_statistics_dict, (x_model_fit, y_model_fit))

        # Return stats for RF creation
        return spatial_statistics_dict

    def place_spatial_receptive_fields(self, spatial_statistics_dict, dendr_diam_vs_eccentricity_parameters_dict,
                                       visualize=False):
        '''
        Create spatial receptive fields to model cells.
        Starting from 2D difference-of-gaussian parameters:
        'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
        '''

        # Get eccentricity data for all model cells
        # gc_eccentricity = self.gc_positions_eccentricity
        gc_eccentricity = self.gc_df['positions_eccentricity'].values

        # Get rf diameter vs eccentricity
        dendr_diam_model = self.dendr_diam_model  # from __init__ method
        dict_key = '{0}_{1}'.format(self.gc_type, dendr_diam_model)
        diam_fit_params = dendr_diam_vs_eccentricity_parameters_dict[dict_key]

        if dendr_diam_model == 'linear':
            gc_diameters = diam_fit_params['intercept'] + diam_fit_params[
                'slope'] * gc_eccentricity  # Units are micrometers
            polynomial_order = 1
        elif dendr_diam_model == 'quadratic':
            gc_diameters = diam_fit_params['intercept'] + diam_fit_params['slope'] * gc_eccentricity \
                           + diam_fit_params['square'] * gc_eccentricity ** 2
            polynomial_order = 2
        elif dendr_diam_model == 'cubic':
            gc_diameters = diam_fit_params['intercept'] + diam_fit_params['slope'] * gc_eccentricity \
                           + diam_fit_params['square'] * gc_eccentricity ** 2 \
                           + diam_fit_params['cube'] * gc_eccentricity ** 3
            polynomial_order = 3

        # Set parameters for all cells
        n_cells = len(gc_eccentricity)
        n_parameters = len(spatial_statistics_dict.keys())
        gc_rf_models = np.zeros((n_cells, n_parameters))
        for index, key in enumerate(spatial_statistics_dict.keys()):
            shape = spatial_statistics_dict[key]['shape']
            loc = spatial_statistics_dict[key]['loc']
            scale = spatial_statistics_dict[key]['scale']
            distribution = spatial_statistics_dict[key]['distribution']
            gc_rf_models[:, index] = self.get_random_samples(shape, loc, scale, n_cells,
                                                             distribution)
        # Quality control images
        if visualize:
            self.show_spatial_statistics(gc_rf_models, spatial_statistics_dict)

        # Calculate RF diameter scaling factor for all ganglion cells
        # Area of RF = Scaling_factor * Random_factor * Area of ellipse(semi_xc,semi_yc), solve Scaling_factor.
        area_of_ellipse = self.ellipse2area(gc_rf_models[:, 0],
                                            gc_rf_models[:, 1])  # Units are pixels for the Chichilnisky data

        '''
        The area_of_rf contains area for all model units. Its sum must fill the whole area (coverage factor = 1).
        We do it separately for each ecc sector, step by step, to keep coverage factor at 1 despite changing gc density with ecc
        '''
        area_scaling_factors_coverage1 = np.zeros(area_of_ellipse.shape)
        for index, surface_area in enumerate(self.sector_surface_area_all):
            # scaling_for_coverage_1 = (surface_area *1e6 ) / np.sum(area_of_rf[self.gc_df['eccentricity_group_index']==index])   # in micrometers2
            scaling_for_coverage_1 = (surface_area * 1e6) / \
                                     np.sum(area_of_ellipse[
                                                self.gc_df['eccentricity_group_index'] == index])  # in micrometers2

            # area_scaling_factors = area_of_rf / np.mean(area_of_ellipse)
            area_scaling_factors_coverage1[self.gc_df['eccentricity_group_index'] == index] \
                = scaling_for_coverage_1

        # area' = scaling factor * area
        # area_of_ellipse' = scaling_factor * area_of_ellipse
        # pi*a'*b' = scaling_factor * pi*a*b
        # a and b are the semi-major and semi minor axis, like radius
        # a'*a'*constant = scaling_factor * a * a * constant
        # a'/a = sqrt(scaling_factor)

        # Apply scaling factors to semi_xc and semi_yc. Units are micrometers.
        scale_random_distribution = 0.08  # Estimated by eye from Watanabe and Perry data. Normal distribution with scale_random_distribution 0.08 cover about 25% above and below the mean value
        random_normal_distribution1 = 1 + np.random.normal(scale=scale_random_distribution, size=n_cells)
        semi_xc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:, 0] * random_normal_distribution1
        random_normal_distribution2 = 1 + np.random.normal(scale=scale_random_distribution,
                                                           size=n_cells)  # second randomization
        semi_yc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:, 1] * random_normal_distribution2
        # semi_xc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:,0]
        # semi_yc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:,1]
        # Scale from micrometers to millimeters and return to numpy matrix
        gc_rf_models[:, 0] = semi_xc / 1000
        gc_rf_models[:, 1] = semi_yc / 1000

        # Save to ganglion cell dataframe. Keep it explicit to avoid unknown complexity
        self.gc_df['semi_xc'] = gc_rf_models[:, 0]
        self.gc_df['semi_yc'] = gc_rf_models[:, 1]
        self.gc_df['xy_aspect_ratio'] = gc_rf_models[:, 2]
        self.gc_df['amplitudes'] = gc_rf_models[:, 3]
        self.gc_df['sur_ratio'] = gc_rf_models[:, 4]
        # self.gc_df['orientation_center'] = gc_rf_models[:, 5]
        self.gc_df['orientation_center'] = self.gc_df['positions_polar_angle']  # plus some noise here

        if visualize:
            # Quality control for diameter distribution. In micrometers.
            gc_diameters = self.area2circle_diameter(self.ellipse2area(semi_xc, semi_yc))

            polynomials = np.polyfit(gc_eccentricity, gc_diameters, polynomial_order)

            self.show_dendritic_diameter_vs_eccentricity(self.gc_type, gc_eccentricity, gc_diameters, polynomials,
                                                         dataset_name='All data {0} fit'.format(dendr_diam_model))

            # gc_rf_models params: 'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
            # rho = self.gc_positions_eccentricity
            # phi = self.gc_positions_polar_angle
            rho = self.gc_df['positions_eccentricity'].values
            phi = self.gc_df['positions_polar_angle'].values

            self.show_gc_receptive_fields(rho, phi, gc_rf_models, surround_fixed=self.surround_fixed)

        # All ganglion cell spatial parameters are now saved to ganglion cell object dataframe gc_df

    def fit_tonic_drives(self, visualize=False):
        tonicdrive_array = np.array(self.all_fits_df.iloc[self.good_data_indices].tonicdrive)
        shape, loc, scale = stats.gamma.fit(tonicdrive_array)

        if visualize:
            x_min, x_max = stats.gamma.ppf([0.001, 0.999], a=shape, loc=loc, scale=scale)
            xs = np.linspace(x_min, x_max, 100)
            plt.plot(xs, stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale))
            plt.hist(tonicdrive_array, density=True)
            plt.title(self.gc_type + ' ' + self.response_type)
            plt.xlabel('Tonic drive (a.u.)')
            plt.show()

        return shape, loc, scale

    def fit_temporal_statistics(self, visualize=False):
        temporal_filter_parameters = ['n', 'p1', 'p2', 'tau1', 'tau2']
        distrib_params = np.zeros((len(temporal_filter_parameters), 3))

        for i, param_name in enumerate(temporal_filter_parameters):
            param_array = np.array(self.all_fits_df.iloc[self.good_data_indices][param_name])
            shape, loc, scale = stats.gamma.fit(param_array)
            distrib_params[i, :] = [shape, loc, scale]

        if visualize:
            plt.subplots(2,3)
            plt.suptitle(self.gc_type + ' ' + self.response_type)
            for i, param_name in enumerate(temporal_filter_parameters):
                plt.subplot(2,3,i+1)
                ax = plt.gca()
                shape, loc, scale = distrib_params[i, :]
                param_array = np.array(self.all_fits_df.iloc[self.good_data_indices][param_name])

                x_min, x_max = stats.gamma.ppf([0.001, 0.999], a=shape, loc=loc, scale=scale)
                xs = np.linspace(x_min, x_max, 100)
                ax.plot(xs, stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale))
                ax.hist(param_array, density=True)
                ax.set_title(param_name)

            plt.show()

        return pd.DataFrame(distrib_params, index=temporal_filter_parameters, columns=['shape', 'loc', 'scale'])

    def create_temporal_filters(self, distrib_params_df, distribution='gamma'):

        n_rgc = len(self.gc_df)

        for param_name, row in distrib_params_df.iterrows():
            shape, loc, scale = row
            self.gc_df[param_name] = self.get_random_samples(shape, loc, scale, n_rgc, distribution)

    def visualize_mosaic(self):
        """
        Plots the full ganglion cell mosaic

        :return:
        """
        rho = self.gc_df['positions_eccentricity'].values
        phi = self.gc_df['positions_polar_angle'].values

        gc_rf_models = np.zeros((len(self.gc_df), 6))
        gc_rf_models[:, 0] = self.gc_df['semi_xc']
        gc_rf_models[:, 1] = self.gc_df['semi_yc']
        gc_rf_models[:, 2] = self.gc_df['xy_aspect_ratio']
        gc_rf_models[:, 3] = self.gc_df['amplitudes']
        gc_rf_models[:, 4] = self.gc_df['sur_ratio']
        gc_rf_models[:, 5] = self.gc_df['orientation_center']

        self.show_gc_receptive_fields(rho, phi, gc_rf_models, surround_fixed=self.surround_fixed)


    def build(self, visualize=False):
        """
        Builds the receptive field mosaic
        :return:
        """
        # -- First, place the ganglion cell midpoints
        # Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
        gc_density_func_params = self.fit_gc_density_data()

        # Place ganglion cells to desired retina.
        self.place_gc_units(gc_density_func_params, visualize=visualize)

        # -- Second, endow cells with spatial receptive fields
        # Collect spatial statistics for receptive fields
        spatial_statistics_dict = self.fit_spatial_statistics(visualize=visualize)

        # Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dendr_diam_vs_eccentricity_parameters_dict = self.fit_dendritic_diameter_vs_eccentricity(
            visualize=visualize)

        # Construct spatial receptive fields. Centers are saved in the object
        self.place_spatial_receptive_fields(spatial_statistics_dict,
                                            dendr_diam_vs_eccentricity_parameters_dict, visualize)

        # At this point the spatial receptive fields are ready.
        # The positions are in gc_eccentricity, gc_polar_angle, and the rf parameters in gc_rf_models
        n_rgc = len(self.gc_df)

        # Summarize RF semi_xc and semi_yc as "RF radius" (geometric mean)
        self.gc_df['rf_radius'] = np.sqrt(self.gc_df.semi_xc * self.gc_df.semi_yc)

        # Finally, get non-spatial parameters
        temporal_statistics_df = self.fit_temporal_statistics()
        self.create_temporal_filters(temporal_statistics_df)

        td_shape, td_loc, td_scale = self.fit_tonic_drives()
        self.gc_df['tonicdrive'] = self.get_random_samples(td_shape, td_loc, td_scale, n_rgc, 'gamma')

        print("Built RGC mosaic with %d cells" % n_rgc)

        if visualize is True:
            plt.show()

    def save_mosaic(self, filepath):
        print('Saving model mosaic to %s' % filepath)
        self.gc_df.to_csv(filepath)


class FunctionalMosaic(Mathematics):

    def __init__(self, gc_dataframe, gc_type, response_type):
        """

        :param gc_dataframe: Ganglion cell parameters; positions are retinal coordinates; positions_eccentricity in mm, positions_polar_angle in degrees
        """
        self.gc_type = gc_type
        self.response_type = response_type

        self.deg_per_mm = 5
        self.stim_vmin = -0.5
        self.stim_vmax = 0.5

        self.cmap_stim = 'gray'
        self.cmap_spatial_filter = 'bwr'
        self.vmin_spatial_filter = -0.5
        self.vmax_spatial_filter = 0.5

        self.temporal_filter_fps = 120
        self.temporal_filter_timesteps = 15

        # Convert retinal positions (ecc, pol angle) to visual space positions in deg (azimuth, elev)
        vspace_pos = np.array([self.pol2cart(gc.positions_eccentricity, gc.positions_polar_angle)
                               for index, gc in gc_dataframe.iterrows()])
        vspace_pos = vspace_pos * self.deg_per_mm
        vspace_coords = pd.DataFrame({'x_deg': vspace_pos[:,0], 'y_deg': vspace_pos[:,1]})

        self.gc_df = pd.concat([gc_dataframe, vspace_coords], axis=1)

        # Convert RF center radii to degrees as well
        self.gc_df.semi_xc = self.gc_df.semi_xc * self.deg_per_mm
        self.gc_df.semi_yc = self.gc_df.semi_yc * self.deg_per_mm

        # Drop retinal positions from the df (so that they are not used by accident)
        self.gc_df = self.gc_df.drop(['positions_eccentricity', 'positions_polar_angle'], axis=1)


    def show_test_image(self, image_extents=[3.5, 6.5, -1.5, 1.5]):
        """
        Shows the mosaic overlayed on top of the test image.

        :param image_extents: image extents in visual space; given as [bottom, top, left, right] degrees
        :return:
        """
        image = cv2.imread('test_image.jpg', 0)
        plt.imshow(image, extent=image_extents, vmin=0, vmax=255)
        plt.title('Test image')
        plt.xlabel('X (deg)')
        plt.ylabel('Y (deg)')
        ax = plt.gca()

        for index, gc in self.gc_df.iterrows():
            circ = Ellipse((gc.x_deg, gc.y_deg), width=2 * gc.semi_xc, height=2 * gc.semi_yc,
                           angle=gc.orientation_center * (180 / np.pi), edgecolor='white', facecolor='None')
            ax.add_patch(circ)

    def _vspace_to_pixspace(self, x, y):
        """
        Converts visual space coordinates (in degrees; x=eccentricity, y=elevation) to pixel space coordinates.
        In pixel space, coordinates (q,r) correspond to matrix locations, ie. (0,0) is top-left.

        :param x: azimuth
        :param y: elevation
        :return:
        """
        video_width_px = self.stimulus_video.video_width
        video_height_px = self.stimulus_video.video_height
        pix_per_deg = self.stimulus_video.pix_per_deg

        # 1) Set the video center in visual coordinates as origin
        # 2) Scale to pixel space. Mirror+scale in y axis due to y-coordinate running top-to-bottom in pixel space
        # 3) Move the origin to video center in pixel coordinates
        q = pix_per_deg * (x - self.stimulus_video.video_center_vspace.real) + (video_width_px / 2)
        r = -pix_per_deg * (y - self.stimulus_video.video_center_vspace.imag) + (video_height_px / 2)

        return q, r

    def load_stimulus(self, stimulus_video, visualize=False, frame_number=0):
        """
        Loads stimulus video & endows RGCs with stimulus space coordinates

        :param stimulus_video: VideoBaseClass, visual stimulus to project to the ganglion cell mosaic
        :param visualize: True/False, show 1 frame of stimulus in pixel and visual coordinate systems (default False)
        :param frame_number: int, which frame of stimulus to show (default 0 = first frame)
        :return:
        """

        # Get parameters from the stimulus object
        from copy import deepcopy
        self.stimulus_video = deepcopy(stimulus_video)  # TODO - No copying plz
        stimulus_center = stimulus_video.video_center_vspace
        self.pix_per_deg = stimulus_video.pix_per_deg  # angular resolution (eg. van Hateren 1 arcmin/pix => 60 pix/deg)

        # Scale stimulus pixel values from 0-255 to [-0.5, 0.5]
        assert np.min(stimulus_video.frames) >= 0 and np.max(stimulus_video.frames) <= 255, \
            "Stimulus values must be between 0 and 255"
        self.stimulus_video.frames = (stimulus_video.frames / 255) - 0.5

        # Endow RGCs with pixel coordinates.
        # NB! Here we make a new dataframe where everything is in pixels
        pixspace_pos = np.array([self._vspace_to_pixspace(gc.x_deg, gc.y_deg)
                               for index, gc in self.gc_df.iterrows()])
        pixspace_coords = pd.DataFrame({'q_pix': pixspace_pos[:, 0], 'r_pix': pixspace_pos[:, 1]})

        self.gc_df_pixspace = pd.concat([self.gc_df, pixspace_coords], axis=1)

        # Scale RF axes to pixel space
        self.gc_df_pixspace.semi_xc = self.gc_df.semi_xc * self.pix_per_deg
        self.gc_df_pixspace.semi_yc = self.gc_df.semi_yc * self.pix_per_deg

        # Define spatial filter sidelength (based on angular resolution and widest semimajor axis)
        # Sidelength always odd number
        self.spatial_filter_sidelen = 2 * 3 * int(max(max(self.gc_df_pixspace.semi_xc), max(self.gc_df_pixspace.semi_yc))) + 1

        # Scale vmin_, vmax_spatial_filter based on filter sidelen
        v_spatial_filter_scaling = (13**2) / (self.spatial_filter_sidelen**2)  # 13**2 being the original dimensions
        self.vmin_spatial_filter = v_spatial_filter_scaling * self.vmin_spatial_filter
        self.vmax_spatial_filter = v_spatial_filter_scaling * self.vmax_spatial_filter

        # Drop RGCs whose center is not inside the stimulus
        xmin, xmax, ymin, ymax = self.stimulus_video.get_extents_deg()
        for index, gc in self.gc_df_pixspace.iterrows():
            if (gc.x_deg < xmin) | (gc.x_deg > xmax) | (gc.y_deg < ymin) | (gc.y_deg > ymax):
                self.gc_df.iloc[index] = 0.0  # all columns set as zero

        if visualize is True:
            self.show_stimulus_with_gcs()


    def show_stimulus_with_gcs(self, ax=None):
        frame_number = 0
        ax = ax or plt.gca()
        ax.imshow(self.stimulus_video.frames[:, :, frame_number])
        ax = plt.gca()

        for index, gc in self.gc_df_pixspace.iterrows():
            # When in pixel coordinates, positive value in Ellipse angle is clockwise. Thus minus here.
            circ = Ellipse((gc.q_pix, gc.r_pix), width=2 * gc.semi_xc, height=2 * gc.semi_yc,
                           angle=gc.orientation_center * (-180 / np.pi), edgecolor='white', facecolor='None')
            ax.add_patch(circ)


    def _get_crop_pixels(self, cell_index):
        """
        Get pixel coordinates for stimulus crop that is the same size as the spatial filter

        :param cell_index: int
        :return:
        """
        gc = self.gc_df_pixspace.iloc[cell_index]
        q_center = int(gc.q_pix)
        r_center = int(gc.r_pix)

        side_halflen = (self.spatial_filter_sidelen-1) // 2  # crops have width = height

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

        amplitudec = 1.0
        offset = 0.0
        s = self.spatial_filter_sidelen

        gc = self.gc_df_pixspace.iloc[cell_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)

        x_grid, y_grid = np.meshgrid(np.arange(qmin, qmax+1, 1),
                                     np.arange(rmin, rmax+1, 1))

        spatial_kernel = self.DoG2D_fixed_surround((x_grid, y_grid), amplitudec, gc.q_pix, gc.r_pix,
                                                gc.semi_xc, gc.semi_yc, gc.orientation_center, gc.amplitudes,
                                                gc.sur_ratio, offset)
        spatial_kernel = np.reshape(spatial_kernel, (s, s))

        # Scale to match data filter power
        # (simulated spatial filter has more pixels => convolution will have higher value, if not corrected)
        data_filtersum = self.gc_df.iloc[cell_index].filtersum
        scaling_factor = data_filtersum / np.sum(np.abs(spatial_kernel))

        # TODO - Figure out correct scaling here!
        scaled_spatial_kernel = scaling_factor * spatial_kernel

        return scaled_spatial_kernel

    def _create_temporal_filter(self, cell_index):
        """
        Creates the temporal component of the spatiotemporal filter

        :param cell_index: int
        :return:
        """
        # temporal_filter = apricot.ApricotFits(self.gc_type, self.response_type).get_mean_temporal_filter()
        # temporal_filter = np.flip(temporal_filter)
        temporal_filter = np.fromstring(self.gc_df.iloc[cell_index].temporal_filter[1:-1])
        temporal_filter = np.array([np.flip(temporal_filter)])

        # If stimulus can have some other fps than 120, then it needs scaling here

        return temporal_filter

    def _create_postspike_filter(self, cell_index):
        postspike_filter = np.fromstring(self.gc_df.iloc[cell_index].postspike_filter[1:-1], sep=' ')
        postspike_filter = np.array([np.flip(postspike_filter)])

        # If stimulus can have some other fps than 120, then it needs scaling here

        return postspike_filter

    def show_gc_view(self, cell_index, frame_number=0):
        """
        Plots the stimulus frame cropped to RGC surroundings, spatial kernel and
        elementwise multiplication of the two

        :param cell_index: int
        :param frame_number: int
        :return:
        """
        gc = self.gc_df_pixspace.iloc[cell_index]
        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)

        # 1) Show stimulus frame cropped to RGC surroundings & overlay 1SD center RF on top of that
        plt.subplot(131)
        plt.title('Cropped stimulus')
        plt.imshow(self.stimulus_video.frames[:, :, frame_number], cmap=self.cmap_stim)
        plt.xlim([qmin, qmax])
        plt.ylim([rmax, rmin])
        ax = plt.gca()

        # When in pixel coordinates, positive value in Ellipse angle is clockwise. Thus minus here.
        circ = Ellipse((gc.q_pix, gc.r_pix), width=2 * gc.semi_xc, height=2 * gc.semi_yc,
                       angle=gc.orientation_center * (-180 / np.pi), edgecolor='white', facecolor='None')
        ax.add_patch(circ)

        # 2) Show spatial kernel created for the stimulus resolution
        plt.subplot(132)
        plt.title('Spatial filter')
        spatial_kernel = self._create_spatial_filter(cell_index)
        plt.imshow(spatial_kernel, cmap=self.cmap_spatial_filter, vmin=self.vmin_spatial_filter, vmax=self.vmax_spatial_filter)

        # 3) Stimulus pixels multiplied elementwise with spatial filter ("keyhole view")
        plt.subplot(133)
        plt.title('Keyhole')

        # Pad the stimulus with zeros in case RGC is at the border
        the_frame = self.stimulus_video.frames[:, :, frame_number]
        padlen = (self.spatial_filter_sidelen - 1) // 2
        padded_frame = np.pad(the_frame, ((padlen, padlen),(padlen, padlen)),
                              mode='constant', constant_values=0)
        padded_frame_crop = padded_frame[padlen+rmin:padlen+rmax+1, padlen+qmin:padlen+qmax+1]

        # Then multiply elementwise
        keyhole_view = np.multiply(padded_frame_crop, spatial_kernel)
        plt.imshow(keyhole_view, cmap='bwr', vmin=-0.5, vmax=0.5)


    def create_spatiotemporal_filter(self, cell_index, visualize=False):
        """
        Returns the outer product of the spatial and temporal filters

        :param cell_index: int
        :param visualize: bool
        :return:
        """

        spatial_filter = self._create_spatial_filter(cell_index)
        s = self.spatial_filter_sidelen
        spatial_filter_1d = np.array([np.reshape(spatial_filter, s**2)]).T
        # TODO - Check that reshape doesn't mix dimensions
        temporal_filter = self._create_temporal_filter(cell_index)

        spatiotemporal_filter = spatial_filter_1d * temporal_filter  # (Nx1) * (1xT) = NxT
        # Scaling wrt experimental filter gain done in _create_spatial_filter()

        if visualize is True:
            plt.subplots(1, 3, figsize=(16, 4))
            plt.suptitle(self.gc_type + ' ' + self.response_type + ' / cell ix ' + str(cell_index))
            plt.subplot(131)
            plt.imshow(spatial_filter, cmap=self.cmap_spatial_filter,
                       vmin=self.vmin_spatial_filter, vmax=self.vmax_spatial_filter)
            plt.colorbar()

            plt.subplot(132)
            plt.plot(range(self.temporal_filter_timesteps), np.flip(temporal_filter[0,:]))
            plt.ylim([-2.5, 2.5])  # limits need to change if fps is changing

            plt.subplot(133)
            plt.imshow(np.flip(spatiotemporal_filter, axis=1), aspect='auto', cmap='bwr',
                       vmin=2*self.vmin_spatial_filter, vmax=2*self.vmax_spatial_filter)
            plt.colorbar()

            plt.tight_layout()

        return spatiotemporal_filter

    def get_cropped_video(self, cell_index, reshape=False):
        """
        Crops the video to RGC surroundings

        :param cell_index: int
        :param reshape:
        :return:
        """
        # Pad the stimulus with zeros in case RGC is at the border
        # the_frame = self.stimulus_video.frames[:, :, frame_number]
        # padlen = (self.spatial_filter_sidelen - 1) // 2
        # padded_frame = np.pad(the_frame, ((padlen, padlen),(padlen, padlen)),
        #                       mode='constant', constant_values=0)
        # padded_frame_crop = padded_frame[padlen+rmin:padlen+rmax+1, padlen+qmin:padlen+qmax+1]
        # TODO - Handle RGCs that are near the border
        # TODO - Check that reshape doesn't mix dimensions
        qmin, qmax, rmin, rmax = self._get_crop_pixels(cell_index)
        stimulus_cropped = self.stimulus_video.frames[rmin:rmax+1, qmin:qmax+1, :]

        if reshape is True:
            s = self.spatial_filter_sidelen
            n_frames = np.shape(self.stimulus_video.frames)[2]

            stimulus_cropped = np.reshape(stimulus_cropped, (s**2, n_frames))

        return stimulus_cropped

    def convolve_stimulus(self, cell_index, visualize=False):
        """
        Convolves the stimulus with the stimulus filter

        :param cell_index: int
        :return: array of length (stimulus timesteps)
        """
        # Get spatiotemporal filter
        spatiotemporal_filter = self.create_spatiotemporal_filter(cell_index)

        # Get cropped stimulus
        stimulus_cropped = self.get_cropped_video(cell_index, reshape=True)

        # Run convolution
        generator_potential = convolve(stimulus_cropped, spatiotemporal_filter, mode='valid')
        generator_potential = generator_potential[0, :]

        # Add some padding to the beginning so that stimulus time and generator potential time match
        # (First time steps of stimulus are not convolved)
        generator_potential = np.pad(generator_potential, (self.temporal_filter_timesteps-1, 0),
                                     mode='constant', constant_values=0)

        if visualize is True:
            tvec = np.arange(0, len(generator_potential), 1) * (1/self.temporal_filter_fps)
            plt.subplots(2, 1, sharex=True)
            plt.subplot(211)
            plt.plot(tvec, generator_potential)
            plt.xlabel('Time (s)')
            plt.ylabel('Generator potential (a.u.)')

            plt.subplot(212)
            tonic_drive = self.gc_df.iloc[cell_index].tonicdrive
            plt.plot(tvec, np.exp(generator_potential + tonic_drive))
            plt.xlabel('Time (s)')
            plt.ylabel('Instantaneous spike rate (a.u.)')

        # Return the 1-dimensional generator potential
        return generator_potential

    def run_single_cell(self, cell_index, n_trials=1, postspike_filter=False, visualize=False, return_monitor=False):
        """
        Runs the LNP pipeline for a single ganglion cell (spiking by Brian2)

        :param cell_index: int
        :param n_trials: int
        :param postspike_filter: bool
        :return:
        """
        if postspike_filter is True:
            raise NotImplementedError
        else:
            duration = self.stimulus_video.video_n_frames / self.stimulus_video.fps * second
            poissongen_dt = 1.0 * ms

            # Get instantaneous firing rate
            generator_potential = self.convolve_stimulus(cell_index)
            tonic_drive = self.gc_df.iloc[cell_index].tonicdrive
            exp_generator_potential = np.array(np.exp(generator_potential + tonic_drive))
            video_dt = (1 / self.stimulus_video.fps) * second

            # Let's interpolate the rate to 1ms intervals
            tvec_original = np.arange(0, len(exp_generator_potential)) * video_dt
            rates_func = interp1d(tvec_original, exp_generator_potential)

            tvec_new = np.arange(0, duration, poissongen_dt)
            interpolated_rates_array = np.array([rates_func(tvec_new)])  # This needs to be 2D array

            # Identical rates array for every trial; rows=time, columns=trial index
            inst_rates = b2.TimedArray(np.tile(interpolated_rates_array.T, (1, n_trials)) * Hz, poissongen_dt)

            # Create Brian PoissonGroup (inefficient implementation but nevermind)
            poisson_group = b2.PoissonGroup(n_trials, rates='inst_rates(t, i)')
            spike_monitor = b2.SpikeMonitor(poisson_group)
            net = b2.Network(poisson_group, spike_monitor)

            # duration = len(generator_potential) * video_dt
            net.run(duration)

            spiketrains = np.array(list(spike_monitor.spike_trains().values()))

        if visualize is True:
            plt.subplots(2, 1, sharex=True)
            plt.subplot(211)
            plt.eventplot(spiketrains)
            plt.xlim([0, duration/second])
            plt.xlabel('Time (s)')

            plt.subplot(212)
            # Plot the generator and the average firing rate
            tvec = np.arange(0, len(generator_potential), 1) * video_dt
            plt.plot(tvec, exp_generator_potential.flatten(), label='Generator')
            plt.xlim([0, duration / second])

            # # TODO - average firing rate here (should follow generator)
            # n_bins = int((duration/(1*ms)))
            # binned_spikes = np.histogram(spiketrains.flatten(), n_bins)[0] / n_trials
            #
            # plt.plot(np.arange(0, n_bins, 1)*1*ms, np.convolve(binned_spikes, [0.25, 0.5, 0.25], mode='same'))

        if return_monitor is True:
            return spike_monitor
        else:
            return spiketrains, interpolated_rates_array.flatten()


if __name__ == "__main__":
    mosaic = MosaicConstructor(gc_type='parasol', response_type='off', ecc_limits=[3, 6],
                               sector_limits=[-30.0, 30.0], model_density=1.0, randomize_position=0.05)

    mosaic.build()
    mosaic.visualize_mosaic()
    plt.show()
    # b = mosaic.fit_temporal_statistics(visualize=False)
    # mosaic.create_temporal_filters(b)
    # mosaic.build()
    # mosaic.fit_tonic_drives(visualize=True)
    # gc_density_func_params = mosaic.fit_gc_density_data()
    #
    # fitdata2 = load_dog_fits('results_temp/parasol_OFF_surfix.csv')
    # mosaic.build(visualize=True)
    # mosaic.visualize_mosaic()
    # plt.show()

    #mosaic.fit_dendritic_diameter_vs_eccentricity(visualize=True)
    # fitdata2 = load_dog_fits('results_temp/parasol_OFF_surfix.csv')
    # mosaic.build(fitdata2, visualize=True)
    # gc_density_func_params = mosaic.fit_gc_density_data()
    # mosaic.place_gc_units(gc_density_func_params, visualize=True)
    # plt.show()

    # You can fit data at runtime
    # import apricot_fitter
    #
    # x = apricot_fitter.ConstructReceptiveFields()
    # fitdata = x.fit_dog_to_sta_data('midget', 'OFF', surround_model=1,
    #                       semi_x_always_major=True)

    # ...or load premade fits
    # fitdata2 = load_dog_fits('results_temp/parasol_OFF_surfix.csv')
    #
    #
    # mosaic = GanglionCells(gc_type='parasol', response_type='OFF', ecc_limits=[4, 6],
    #                                   sector_limits=[-10.0, 10.0], model_density=1.0, randomize_position=0.6)
    # mosaic.build(fitdata2, visualize=False)
    # # Or you can load a previously built mosaic
    # #
    # # mosaic.visualize_mosaic()
    #
    # a = vs.ConstructStimulus(video_center_vspace=5 + 0j, pattern='sine_grating', temporal_frequency=2,
    #                          spatial_frequency=0.5,
    #                          duration_seconds=5, fps=120, orientation=45, image_width=90, image_height=90,
    #                          pix_per_deg=30, stimulus_size=0, contrast=0.7)
    #
    # mosaic.load_stimulus(a)

    #
    # mosaic.visualize_stimulus_and_grid(marked_cells=[2])
    # mosaic.visualize_rgc_view(100, show_block=True)

    # all_spikes = []
    # for i in range(10):
    #     # mosaic.create_spatiotemporal_kernel(i, visualize=True)
    #     # mosaic.simple_spiking(i)
    #     mosaic.visualize_rgc_view(i, show_block=True)
    #     #
    #     # filtered_stuff = mosaic.feed_stimulus_thru_filter(i)
    #     # n = len(filtered_stuff)
    #     # plt.plot(range(n), filtered_stuff)
    #     #
    #     tsp, Vmem, meanfr = mosaic.pillow_spiking(0)
    #     all_spikes.append(np.array(tsp) * (1/120))
    #
    # plt.eventplot(all_spikes)
    # plt.show()

    # Test experiment
    # contrasts = np.arange(0.1, 1.0, 0.1)
    # spatial_freqs = [1, 2, 4, 6, 12, 24]
    # results_table = np.zeros((len(contrasts), len(spatial_freqs)))
    # cont_grid, spatfreq_grid = np.meshgrid(contrasts, spatial_freqs)
    # response_threshold = 10
    # cell_ix = 0
    #
    # for j, spatfreq in enumerate(spatial_freqs):
    #     for i, contrast in enumerate(contrasts):
    #         a = vs.ConstructStimulus(video_center_vspace=5 + 0j, pattern='sine_grating', temporal_frequency=6,
    #                                  spatial_frequency=spatfreq,
    #                                  duration_seconds=5, fps=120, orientation=45, image_width=320, image_height=240,
    #                                  pix_per_deg=30, stimulus_size=5.0, contrast=contrast)
    #
    #         mosaic.load_stimulus(a)
    #         tsp, Vmem, meanfr = mosaic.pillow_spiking(cell_ix)
    #         results_table[i,j] = meanfr
    #
    # np.savetxt("results_temp/parasol_spat_tuning.csv", results_table, delimiter=',')

# Todo (Henri's)
# - RGC view: orientation is different in every friggin plot.... >.<
# - Normalize kernel energy => still sometimes cells with very high firing rates...
# - Saving/loading generated mosaic
# - Check scale of filter values in data
# - Fit temporal kernel
# - Fit postspike currents

# - Eccentricity and polar angle used sometimes as if they are the same thing
# - "center" has double meaning: RF center or center/surround
# - semi_x and semi_y... write somewhere that semi_x = horizontal, semi_y = vertical and orientation rotates ccw


# chichilnisky_fits = parasol_ON_object.fit_spatial_statistics(visualize=False)

# VisualImageArray(pattern='white_noise', stimulus_form='rectangular', duration_seconds=2,
#							 fps=30, pedestal =0, orientation=0, stimulus_position=(0,0), stimulus_size=4)

# # Define eccentricity and theta in degrees. Model_density is the relative density compared to true macaque values.
# ganglion_cell_object = GanglionCells(gc_type='parasol', responsetype='ON', eccentricity=[3,7], theta=[-30.0,30.0], density=1.0, randomize_position = 0.6)

# Operator.run_retina_construction(ganglion_cell_object, visualize=1)

# parasol_ON_object = GanglionCells(gc_type='parasol', responsetype='ON', eccentricity=[0.5,30], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

# Retina.run_retina_construction(parasol_ON_object, visualize=1)

# parasol_OFF_object = GanglionCells(gc_type='parasol', responsetype='OFF', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

# Operator.run_retina_construction(parasol_OFF_object, visualize=1)

# midget_ON_object = GanglionCells(gc_type='midget', responsetype='ON', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

# Operator.run_retina_construction(midget_ON_object, visualize=0)

# midget_OFF_object = GanglionCells(gc_type='midget', responsetype='OFF', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

# Operator.run_retina_construction(midget_OFF_object, visualize=0)

# sample_image_object = SampleImage()

# Operator.run_stimulus_sampling(sample_image_object, visualize=1)

# TODO (Simo's):

# Visual stimuli

#   -xy_aspcects_ratio show to some extent bimodal distribution. It should be convertable to all y_sigma > x_sigma, but attempt to do this failed. Fit quality decreased

#	-consider implementing significant correlations between spatial parameters

#   -model and apply time behaviour

#   -construct LGN. Probably a filter rather than spiking neurons. The latter dont make sense because we are interested in cx, not sub-cx.s


'''
This is code for building macaque retinal filters corresponding to midget and parasol cells' responses.
We keep modular code structure, to be able to add new features at later phase.

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

Visual angle ( A) in degrees from previous studies (Croner and Kaplan, 1995) was approksimated with relation 5 deg/mm.
This works fine up to 20 deg ecc, but undesestimates the distance thereafter. If more peripheral representations are later
necessary, the millimeters should be calculates by inverting the inverting the relation 
A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974; Dacey and Petersen, 1992)

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of meny cell types is inversely proportional to dendritic field coverage, 
suggesting constant coverage factor (Perry_1984_Neurosci, Wassle_1991_PhysRev).
Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, Lee_2010_ProgRetEyeRes).
Parasol coverage factor is 3-4 close to fovea (Grunert_1993_VisRes); 2-7 according to Perry_1984_Neurosci.
These include ON- and OFF-center cells, and perhaps other cell types.
It is likely that coverage factor is 1 for midget and parasol ON- and OFF-center cells each, 
which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four cell types (parasol & midget, ON & OFF) were modelled with double ellipsoid
difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is
described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci; Field_2010_Nature.

Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells. 
He also shows that OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to imeplement 
if necessary).

NOTE: bad cell indices hard coded from Chichilnisky apricot data. For another data set, visualize fits, and change the bad cells.
NOTE: If eccentricity stays under 20 deg, dendritic diameter data fitted up to 25 deg only (better fit close to fovea)

-center-surround response ratio (in vivo, anesthetized, recorded from LGN; Croner_1995_VisRes) PC: ; MC: ;
-Michelson contrast definition for sinusoidal gratings (Croner_1995_VisRes).
-Optical quality probably poses no major limit to behaviorally measured spatial vision (Williams_1981_IOVS).
-spatial contrast sensitivity nonlinearity in the center subunits is omitted. This might reduce senstivity to natural scenes Turner_2018_eLife.

-quality control: compare to Watanabe_1989_JCompNeurol
    -dendritic diameter scatter is on average (lower,upper quartile) 21.3% of the median diameter in the local area

    Parasol dendritic field diameter: temporal retina 51.8 microm + ecc(mm) * 20.6 microm/mm,
    nasal retina; 115.5 microm + ecc(mm) * 6.97 microm/mm

'''
