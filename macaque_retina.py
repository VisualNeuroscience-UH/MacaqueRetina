import os
import sys
import pdb
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from tqdm import tqdm
import cv2
from pathlib import Path
import quantities as pq
import elephant
import neo
import seaborn as sns
import visual_stimuli as vs
from visualize import Visualize
from vision_maths import Mathematics

plt.rcParams['image.cmap'] = 'gray'

class GanglionCells(Mathematics, Visualize):
    '''
    Create the ganglion cell mosaic.
    All spatial parameters are saved to the dataframe *gc_df*
    '''

    script_path = Path(__file__).parent
    digitized_figures_path = script_path

    def __init__(self, gc_type, response_type, ecc_limits, sector_limits, model_density=1.0, randomize_position=0.7):
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
        if all([gc_type == 'parasol', response_type == 'ON']):
            self.gc_proportion = proportion_of_parasol_gc_type * proportion_of_ON_response_type * model_density
        elif all([gc_type == 'parasol', response_type == 'OFF']):
            self.gc_proportion = proportion_of_parasol_gc_type * proportion_of_OFF_response_type * model_density
        elif all([gc_type == 'midget', response_type == 'ON']):
            self.gc_proportion = proportion_of_midget_gc_type * proportion_of_ON_response_type * model_density
        elif all([gc_type == 'midget', response_type == 'OFF']):
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
            # TODO - gc_density_func_params should not be arguments to the method
            my_gaussian_fit = self.gauss_plus_baseline(center_ecc, *gc_density_func_params)
            Ncells = sector_surface_area * my_gaussian_fit * self.gc_proportion

            # place cells in regular grid
            # Vector of cell positions in radial and polar directions. Angle in degrees.
            inner_arc_in_mm = (angle / 360) * 2 * np.pi * eccentricity_in_mm[0]
            delta_eccentricity_in_mm = eccentricity_in_mm[1] - eccentricity_in_mm[0]
            n_segments_arc = np.sqrt(Ncells * (
                        inner_arc_in_mm / delta_eccentricity_in_mm))  # note that the n_segments_arc and n_segments_eccentricity are floats
            n_segments_eccentricity = (delta_eccentricity_in_mm / inner_arc_in_mm) * n_segments_arc
            int_n_segments_arc = int(round(n_segments_arc))  # cells must be integers
            int_n_segments_eccentricity = int(round(n_segments_eccentricity))

            # Recalc delta_eccentricity_in_mm given the n segments to avoid non-continuous cell densities
            true_n_cells = int_n_segments_arc * int_n_segments_eccentricity
            true_sector_area = true_n_cells / (my_gaussian_fit * self.gc_proportion)
            true_delta_eccentricity_in_mm = (int_n_segments_eccentricity / int_n_segments_arc) * inner_arc_in_mm

            radius_segment_length = true_delta_eccentricity_in_mm / int_n_segments_eccentricity
            theta_segment_angle = angle / int_n_segments_arc

            # Set the true_eccentricity_end
            true_eccentricity_end = eccentricity_in_mm[0] + true_delta_eccentricity_in_mm

            vector_polar_angle = np.linspace(theta[0], theta[1], int_n_segments_arc)
            vector_eccentricity = np.linspace(eccentricity_in_mm[0], true_eccentricity_end - radius_segment_length,
                                              int_n_segments_eccentricity)
            # print vector_polar_angle
            # print '\n\n'
            # print vector_eccentricity
            # print '\n\n'

            # meshgrid and rotate every second to get good GC tiling
            matrix_polar_angle, matrix_eccentricity = np.meshgrid(vector_polar_angle, vector_eccentricity)
            matrix_polar_angle[::2] = matrix_polar_angle[::2] + (
                        angle / (2 * n_segments_arc))  # rotate half the inter-cell angle

            # randomize for given proportion
            matrix_polar_angle_randomized = matrix_polar_angle + theta_segment_angle * randomize_position \
                                            * (np.random.rand(matrix_polar_angle.shape[0],
                                                              matrix_polar_angle.shape[1]) - 0.5)
            matrix_eccentricity_randomized = matrix_eccentricity + radius_segment_length * randomize_position \
                                             * (np.random.rand(matrix_eccentricity.shape[0],
                                                               matrix_eccentricity.shape[1]) - 0.5)

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

    def read_dendritic_fields_vs_eccentricity_data(self):
        '''
        Read re-digitized old literature data from mat files
        '''
        digitized_figures_path = GanglionCells.digitized_figures_path

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

        # Limit eccentricities for central visual field studies to get better approximation at about 5 eg ecc (1mm)
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

        return dendr_diam_parameters

    def read_gc_density_data(self):
        '''
        Read re-digitized old literature data from mat files
        '''
        digitized_figures_path = GanglionCells.digitized_figures_path

        gc_density = sio.loadmat(digitized_figures_path / 'Perry_1984_Neurosci_GCdensity_c.mat',
                                 variable_names=['Xdata', 'Ydata'])
        cell_eccentricity = np.squeeze(gc_density['Xdata'])
        cell_density = np.squeeze(gc_density['Ydata']) * 1e3  # Cells are in thousands, thus the 1e3
        return cell_eccentricity, cell_density

    def fit_spatial_statistics(self, fitdata, visualize=False):
        """
        Collect spatial statistics from Chichilnisky receptive field data
        """

        # 2D DoG fit to Chichilnisky retina spike triggered average data. The visualize parameter will
        # show each DoG fit in order to search for bad cell fits and data.
        # parameter_names, data_all_viable_cells, bad_cell_indices = \
        #     self.fit_dog_to_sta_data(visualize=False, surround_model=self.surround_fixed)
        parameter_names, data_all_viable_cells, bad_cell_indices = fitdata


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

    def get_random_samples_from_known_distribution(self, shape, loc, scale, n_cells, distribution):
        """
        Create random samples from a model distribution.

        :param shape:
        :param loc:
        :param scale:
        :param n_cells:
        :param distribution:

        :returns distribution_parameters
        """
        assert distribution in ['gamma', 'beta'], "Distribution should be either gamma or beta"

        if distribution == 'gamma':
            distribution_parameters = stats.gamma.rvs(a=shape, loc=loc, scale=scale, size=n_cells,
                                                      random_state=None)  # random_state is the seed
        elif distribution == 'beta':
            distribution_parameters = stats.beta.rvs(a=shape[0], b=shape[1], loc=loc, scale=scale, size=n_cells,
                                                     random_state=None)  # random_state is the seed

        return distribution_parameters

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
            gc_rf_models[:, index] = self.get_random_samples_from_known_distribution(shape, loc, scale, n_cells,
                                                                                     distribution)
        # For semi_yc/semi_xc ratio, noise increases at index 327

        # Quality control images
        if visualize:
            self.show_spatial_statistics(gc_rf_models, spatial_statistics_dict)

        # Calculate RF diameter scaling factor for all ganglion cells
        # Area of RF = Scaling_factor * Random_factor * Area of ellipse(semi_xc,semi_yc), solve Scaling_factor.
        area_of_rf = self.circle_diameter2area(gc_diameters)  # All cells
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
        self.gc_df['orientation_center'] = gc_rf_models[:, 5]

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

    def build(self, fitdata, visualize=False):
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
        spatial_statistics_dict = self.fit_spatial_statistics(fitdata, visualize=visualize)

        # Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit.
        # Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
        dendr_diam_vs_eccentricity_parameters_dict = self.fit_dendritic_diameter_vs_eccentricity(
            visualize=visualize)

        # Construct spatial receptive fields. Centers are saved in the object
        self.place_spatial_receptive_fields(spatial_statistics_dict,
                                            dendr_diam_vs_eccentricity_parameters_dict, visualize)

        # At this point the spatial receptive fields are constructed.
        # The positions are in gc_eccentricity, gc_polar_angle, and the rf parameters in gc_rf_models

        if Visualize is True:
            plt.show()

    def save_mosaic(self, filepath, deg=False):
        if deg is True:
            pass  # Implement here unit conversion to degrees
        else:
            print('Saving model mosaic to %s' % filepath)
            self.gc_df.to_csv(filepath)

    def load_mosaic(self, filepath):  # TODO
        raise NotImplementedError

    def show_fitted_rf(self, cell_index, um_per_pixel=10, n_pixels=30):
        """
        Plots the spatial RF

        :param cell_index: int
        :param um_per_pixel: int, how many micrometers does one pixel cover (default 10)
        :param n_pixels: int, number of pixels per side
        :return:
        """
        # TODO - label axes
        gc = self.gc_df.iloc[cell_index]
        # print(gc)
        mm_per_pixel = um_per_pixel / 1000
        image_halfwidth_mm = (n_pixels / 2) * mm_per_pixel
        x_position_indices = np.linspace(gc.positions_eccentricity - image_halfwidth_mm,
                                         gc.positions_eccentricity + image_halfwidth_mm, n_pixels)
        y_position_indices = np.linspace(gc.positions_polar_angle - image_halfwidth_mm,
                                         gc.positions_polar_angle + image_halfwidth_mm, n_pixels)

        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)
        amplitudec = 1
        offset = 0

        fitted_data = self.DoG2D_fixed_surround((x_grid, y_grid), amplitudec, gc.positions_eccentricity,
                                                gc.positions_polar_angle,
                                                gc.semi_xc, gc.semi_yc, gc.orientation_center, gc.amplitudes,
                                                gc.sur_ratio, offset)

        sns.heatmap(np.reshape(fitted_data, (n_pixels, n_pixels)))
        plt.show()

    def load_stimulus(self, stimulus_video, visualize=False, frame_number=0):
        """
        Loads stimulus video/frames

        :param stimulus_video: VideoBaseClass, visual stimulus to project to the ganglion cell mosaic
        :param visualize: True/False, show 1 frame of stimulus in pixel and visual coordinate systems (default False)
        :param frame_number: int, which frame of stimulus to show (default 0 = first frame)
        :return:
        """
        # Check that video has values scaled between 0...255 etc.

        self.stimulus_video = stimulus_video

        if visualize is True:
            self.visualize_stimulus_and_grid(frame_number)

    # TODO - make visual space/pixel space correspondence perfect
    def visualize_stimulus_and_grid(self, frame_number=0, marked_cells=[]):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

        z = 8
        w = 3

        plt.subplot(121)
        plt.title('In pixel space')
        plt.imshow(self.stimulus_video.frames[:, :, frame_number], vmin=0, vmax=255)
        plt.xlabel('Horizontal coordinate (px)')
        plt.ylabel('Vertical coordinate (px)')

        z_px, w_px = self.vspace_to_pixspace(z, w)
        plt.scatter(z_px, w_px, c='blue')
        plt.scatter(60, 60, c='red')

        plt.subplot(122)
        plt.title('In visual space')

        # Change image coordinates to visual space coordinates
        vspace_extents = self.stimulus_video.get_extents_deg()
        plt.imshow(self.stimulus_video.frames[:, :, frame_number], extent=vspace_extents, vmin=0, vmax=255)

        # Plot all the ganglion cell RFs on top of the image
        ax = plt.gca()
        for i in range(len(self.gc_df)):
            the_gc = self.gc_df.loc[i]
            # print(the_gc)
            x, y = self.pol2cart(the_gc.positions_eccentricity, the_gc.positions_polar_angle)
            x = self.deg_per_mm * x
            y = self.deg_per_mm * y
            semi_x = self.deg_per_mm * the_gc.semi_xc
            semi_y = self.deg_per_mm * the_gc.semi_yc

            ellipse_fill = 'None'
            if i in marked_cells:
                ellipse_fill = 'blue'

            orientation_deg = the_gc.orientation_center * (180/np.pi)
            circ = Ellipse((x, y), width=2 * semi_x, height=2 * semi_y,
                           angle=orientation_deg, edgecolor='red', facecolor=ellipse_fill)
            ax.add_patch(circ)


        plt.scatter(z, w, c="blue")
        plt.xlabel('Eccentricity (deg)')
        plt.ylabel('Elevation (deg)')

        # plt.colorbar()
        plt.show(block=False)

    def vspace_to_pixspace(self, x, y):
        video_width_px = self.stimulus_video.video_width
        video_height_px = self.stimulus_video.video_height
        pix_per_deg = self.stimulus_video.pix_per_deg

        # 1) Set the video center in visual coordinates as origin
        # 2) Scale to pixel space. Mirror+scale in y axis due to y-coordinate running top-to-bottom in pixel space
        # 3) Move the origin to video center in pixel coordinates
        x_new = pix_per_deg * (x - self.stimulus_video.video_center_vspace.real) + (video_width_px / 2)
        y_new = -pix_per_deg * (y - self.stimulus_video.video_center_vspace.imag) + (video_height_px / 2)

        return x_new, y_new

    def pixspace_to_vspace(self, x, y):

        video_width_px = self.stimulus_video.video_width
        video_height_px = self.stimulus_video.video_height
        pix_per_deg = self.stimulus_video.pix_per_deg

        x_new = (1/pix_per_deg) * (x - (video_width_px / 2)) + self.stimulus_video.video_center_vspace.real
        y_new = (-1/pix_per_deg) * (y - (video_height_px / 2)) + self.stimulus_video.video_center_vspace.imag

        return x_new, y_new

    def visualize_rgc_view(self, cell_index, show_block=False):  # TODO - add global mosaic view as first plot
        frame_number = 0
        total_frames = len(self.stimulus_video.frames[0,0,:])

        the_gc = self.gc_df.iloc[cell_index]
        center_x, center_y = self.pol2cart(the_gc.positions_eccentricity, the_gc.positions_polar_angle)
        orientation_deg = the_gc.orientation_center * (180/np.pi)
        # Transform to degrees
        center_x = self.deg_per_mm * center_x
        center_y = self.deg_per_mm * center_y
        semi_x = self.deg_per_mm * the_gc.semi_xc
        semi_y = self.deg_per_mm * the_gc.semi_yc


        plt.subplots(2,2, figsize=(12,12))
        plt.suptitle("Cell index %d, video frame %d (of %d)" % (cell_index, frame_number, total_frames))

        plt.subplot(221)
        plt.title('RGC center 1SD ellipse wrt stimulus frame')
        plt.xlabel('Eccentricity (deg)')
        plt.ylabel('Elevation (deg)')

        ax = plt.gca()
        # 1) Show pixels limited to "3SD view" & overlay 1SD center RF on top of that
        pixspace_topleft, pixspace_bottomright = self.get_crop_extents(cell_index)

        # With this choice of vspace_extents, the position of the 1SD ellipse will be slightly shifted
        #vspace_extents = [vspace_ext_left, vspace_ext_right, vspace_ext_bottom, vspace_ext_top]

        vspace_topleft = self.pixspace_to_vspace(pixspace_topleft[0], pixspace_topleft[1])
        vspace_bottomright = self.pixspace_to_vspace(pixspace_bottomright[0], pixspace_bottomright[1])
        vspace_extents = [vspace_topleft[0], vspace_bottomright[0],
                          vspace_bottomright[1], vspace_topleft[1]]

        cropped_video = self.stimulus_video.frames[pixspace_topleft[0]:pixspace_bottomright[0] + 1,
                        pixspace_topleft[1]:pixspace_bottomright[1] + 1, :]
        plt.imshow(cropped_video[:,:, frame_number],
                   extent=vspace_extents, vmin=0, vmax=255)

        circ = Ellipse((center_x, center_y), width=2 * semi_x, height=2 * semi_y,
                       angle=orientation_deg, edgecolor='red', facecolor="None")
        ax.add_patch(circ)

        # 2) Show spatial kernel created for the stimulus resolution
        plt.subplot(222)
        plt.title("Spatial filter in stimulus resolution")
        ax = plt.gca()
        spatial_kernel = self.create_spatial_kernel(cell_index)
        plt.imshow(spatial_kernel)

        # 3) "Keyhole view"
        plt.subplot(223)
        plt.title("Stimulus x spatial filter (elementwise)")
        keyhole_view = np.multiply(cropped_video[:,:,frame_number], spatial_kernel)
        plt.imshow(keyhole_view)

        # 4) Integrated signal over frames

        plt.show(block=show_block)

    def get_crop_extents(self, cell_index, view_sd=5):  # TODO - cleanup this method
        the_gc = self.gc_df.iloc[cell_index]
        center_x, center_y = self.pol2cart(the_gc.positions_eccentricity, the_gc.positions_polar_angle)
        # Transform to degrees
        center_x = self.deg_per_mm * center_x
        center_y = self.deg_per_mm * center_y

        # Make a box with sidelength 2*view_sd*SD of the semimajor radius
        semimajor_radius = max(the_gc.semi_xc, the_gc.semi_yc) * self.deg_per_mm
        vspace_ext_left = center_x - semimajor_radius * view_sd
        vspace_ext_right = center_x + semimajor_radius * view_sd
        vspace_ext_bottom = center_y - semimajor_radius * view_sd
        vspace_ext_top = center_y + semimajor_radius * view_sd

        # Find the coordinates of the corresponding box in pixel space
        pixspace_topleft = self.vspace_to_pixspace(vspace_ext_left, vspace_ext_top)
        pixspace_bottomright = self.vspace_to_pixspace(vspace_ext_right, vspace_ext_bottom)
        pixspace_topleft = tuple([int(t) for t in pixspace_topleft])  # Round down
        pixspace_bottomright = tuple([int(t + 1) for t in pixspace_bottomright])  # Round up

        kernel_width = pixspace_bottomright[0] - pixspace_topleft[0] + 1
        kernel_height = pixspace_bottomright[1] - pixspace_topleft[1] + 1  # y-indexing runs top-to-bottom

        kernel_sidelen = max(kernel_height, kernel_width)
        pixspace_bottomright = (pixspace_topleft[0] + kernel_sidelen, pixspace_topleft[1] + kernel_sidelen)

        return pixspace_topleft, pixspace_bottomright

    def get_cropped_video(self, cell_index, reshape=False):
        pixspace_topleft, pixspace_bottomright = self.get_crop_extents(cell_index)
        crop_sidelen = pixspace_bottomright[0] - pixspace_topleft[0] + 1
        total_frames = len(self.stimulus_video.frames[0,0,:])

        cropped_video = self.stimulus_video.frames[pixspace_topleft[0]:pixspace_bottomright[0] + 1,
                        pixspace_topleft[1]:pixspace_bottomright[1] + 1, :]

        if reshape:
            cropped_video = np.reshape(cropped_video, (crop_sidelen**2, total_frames))

        return cropped_video


    def create_spatial_kernel(self, cell_index):
        """
        Creates the spatial kernel for one cell, respecting stimulus resolution

        :param cell_index: int
        :return:
        """
        the_gc = self.gc_df.iloc[cell_index]
        pixspace_topleft, pixspace_bottomright = self.get_crop_extents(cell_index)

        # Make the kernel the same size as the cropped stimulus video
        kernel_width = pixspace_bottomright[0] - pixspace_topleft[0] + 1
        kernel_height = pixspace_bottomright[1] - pixspace_topleft[1] + 1  # y-indexing runs top-to-bottom

        print("Cell index %d: kernel width x height %d x %d" % (cell_index, kernel_width, kernel_height))

        if kernel_width != kernel_height:
            print("Warning! Spatial filter width is different from height. The resulting filter is likely to be faulty.")

        # Get visual field coordinates for the cropped stimulus
        crop_xmin_deg, crop_ymin_deg = self.pixspace_to_vspace(pixspace_topleft[0], pixspace_bottomright[1])
        crop_xmax_deg, crop_ymax_deg = self.pixspace_to_vspace(pixspace_bottomright[0], pixspace_topleft[1])

        x_grid, y_grid = np.meshgrid(np.linspace(crop_xmin_deg, crop_xmax_deg, kernel_width),
                                     np.linspace(crop_ymin_deg, crop_ymax_deg, kernel_height))

        amplitudec = 1
        offset = 0
        gc_x, gc_y = self.pol2cart(the_gc.positions_eccentricity, the_gc.positions_polar_angle)
        gc_x = self.deg_per_mm * gc_x
        gc_y = self.deg_per_mm * gc_y
        semi_x = self.deg_per_mm * the_gc.semi_xc
        semi_y = self.deg_per_mm * the_gc.semi_yc
        ori = the_gc.orientation_center

        spatial_kernel = self.DoG2D_fixed_surround((x_grid, y_grid), amplitudec, gc_x, gc_y,
                                                semi_x, semi_y, ori, the_gc.amplitudes,
                                                the_gc.sur_ratio, offset)
        spatial_kernel = np.reshape(spatial_kernel[:,], (kernel_width, kernel_height))

        # spatial_kernel = np.array([self.DoG2D_fixed_surround((x_grid, y_grid), *fitted_spatial_params)]).T
        return spatial_kernel

    def create_spatiotemporal_kernel(self, cell_index, visualize=False):
        spatial_kernel = self.create_spatial_kernel(cell_index)
        spatial_kernel_sidelen = np.shape(spatial_kernel)[0]  # assuming square kernel
        spatial_kernel = [np.reshape(spatial_kernel, spatial_kernel_sidelen**2)]
        time_kernel = np.array([[0, 0.12, 0.12, 0.25, 0.25, 0.5, 0.5, 1, 1, 0]])
        spatiotemporal_kernel = spatial_kernel * time_kernel.T

        if visualize:
            plt.imshow(spatiotemporal_kernel)
            plt.xlabel('Space')
            plt.xlabel('Time (frames)')
            plt.show()

        return spatiotemporal_kernel

    def feed_stimulus(self, cell_index):

        # Create spatiotemporal kernel
        spatiotemporal_filter = self.create_spatiotemporal_kernel(cell_index)
        print("Created spatiotemporal kernel!")

        # Get video cropped to cell's receptive field
        cropped_video = self.get_cropped_video(cell_index, reshape=True)



# Obsolete - pre-OCNC code!
# def build_convolution_matrix(self, gc_index, stimulus_center, stimulus_width_px, stimulus_height_px):
# 	gc = self.gc_df.iloc[gc_index]
# 	n_pixels = 100
# 	x_position_indices = np.linspace(gc.positions_eccentricity - image_halfwidth_mm,
# 									 gc.positions_eccentricity + image_halfwidth_mm, n_pixels)
# 	y_position_indices = np.linspace(gc.positions_polar_angle - image_halfwidth_mm,
# 									 gc.positions_polar_angle + image_halfwidth_mm, n_pixels)
# 	fitted_data = self.DoG2D_fixed_surround((x_grid, y_grid), amplitudec, gc.positions_eccentricity, gc.positions_polar_angle,
# 				  			                 gc.semi_xc, gc.semi_yc, gc.orientation_center, gc.amplitudes, gc.sur_ratio, offset)
#
# 	x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)
#
# 	# Build the convolution matrix according to spatial & temporal properties
# 	# Linearize the spatial filter
# 	# Multiply spat x temporal = (S x 1) x (1 x T) = S x T matrix
# 	pass
#
# def generate_analog_response(self, visual_image):
# 	## Compute filtered response to stimulus
# 	# Create neo.AnalogSignal by convolving stimulus with the spatio-temporal filter
# 	# => point process conditional intensity
#
# 	# Need to take care of:
# 	#  - time in filter data vs stimulus fps
# 	#  - spatial filter size vs stimulus size
# 	# For each time point t in common_time:
# 	#   Convolved_data = Stim(t)^T (1xS) x
# 	#
# 	# data_sampling_rate = self.get_sampling_rate()
# 	# conv_data_signal = neo.AnalogSignal(convolved_data, units='1', sampling_rate = data_sampling_rate)
#
# 	## Compute interpolated h current (??)
# 	pass
#
# def generate_spike_train(self):
# 	## Static nonlinearity & spiking
# 	# Create spike train using elephant.spike_train_generation.inhomogeneous_poisson_process()
# 	# ...but need to take post-spike filter into account!!
# 	pass
#
# def create_temporal_filters(self):
# 	# Take in the temporal filter stats
# 	# Build a family of temporal filters, one for each GC
# 	pass
#
# def generate_gc_spike_trains(self):
# 	# Parallelize for faster computation :)
# 	pass


class SampleImage:
    '''
    This class gets one image at a time, and provides the cone response.
    After instantiation, the RGC group can get one frame at a time, and the system will give an impulse response.
    '''

    def __init__(self, micrometers_per_pixel=10, image_resolution=(100, 100), temporal_resolution=1):
        '''
        Instantiate new stimulus.
        '''
        self.millimeters_per_pixel = micrometers_per_pixel / 1000  # Turn to millimeters
        self.temporal_resolution = temporal_resolution
        self.optical_aberration = 2 / 60  # unit is degree
        self.deg_per_mm = 5

    def get_image(self, image_file_name='testi.jpg'):

        # Load stimulus
        image = cv2.imread(image_file_name, 0)  # The 0-flag calls for grayscale. Comes in as uint8 type

        # Normalize image intensity to 0-1, if RGB value
        if np.ptp(image) > 1:
            scaled_image = np.float32(image / 255)
        else:
            scaled_image = np.float32(image)  # 16 bit to save space and memory

        return scaled_image

    def blur_image(self, image):
        '''
        Gaussian smoothing from Navarro 1993: 2 arcmin FWHM under 20deg eccentricity.
        '''

        # Turn the optical aberration of 2 arcmin FWHM to Gaussian function sigma
        sigma_in_degrees = self.optical_aberration / (2 * np.sqrt(2 * np.log(2)))

        # Turn Gaussian function with defined sigma in degrees to pixel space
        sigma_in_mm = sigma_in_degrees / self.deg_per_mm
        sigma_in_pixels = sigma_in_mm / self.millimeters_per_pixel  # This is small, 0.28 pixels for 10 microm/pixel resolution

        # Turn
        kernel_size = (5, 5)  # Dimensions of the smoothing kernel in pixels, centered in the pixel to be smoothed
        image_after_optics = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_in_pixels)  # sigmaY = sigmaX

        return image_after_optics

    def aberrated_image2cone_response(self, image):

        # Compressing nonlinearity. Parameters are manually scaled to give dynamic cone ouput.
        # Equation, data from Baylor_1987_JPhysiol
        rm = 25  # pA
        k = 2.77e-4  # at 500 nm
        cone_sensitivity_min = 5e2
        cone_sensitivity_max = 1e4

        # Range
        response_range = np.ptp([cone_sensitivity_min, cone_sensitivity_max])

        # Scale
        image_at_response_scale = image * response_range  # Image should be between 0 and 1
        cone_input = image_at_response_scale + cone_sensitivity_min

        # Cone nonlinearity
        cone_response = rm * (1 - np.exp(-k * cone_input))

        return cone_response


class Operator:
    '''
    Operate the generation and running of retina here
    '''

    def run_stimulus_sampling(sample_image_object, visualize=False):
        one_frame = sample_image_object.get_image()
        one_frame_after_optics = sample_image_object.blur_image(one_frame)
        cone_response = sample_image_object.aberrated_image2cone_response(one_frame_after_optics)

        if visualize:
            fig, ax = plt.subplots(nrows=2, ncols=3)
            axs = ax.ravel()
            axs[0].hist(one_frame.flatten(), 20)
            axs[1].hist(one_frame_after_optics.flatten(), 20)
            axs[2].hist(cone_response.flatten(), 20)

            axs[3].imshow(one_frame, cmap='Greys')
            axs[4].imshow(one_frame_after_optics, cmap='Greys')
            axs[5].imshow(cone_response, cmap='Greys')

            plt.show()


# Obsolete: pre-OCNC code
# class VisualImageArray(vs.ConstructStimulus):
#
# 	def __init__(self, image_center, **kwargs):
# 		"""
# 		The visual stimulus, simple optics and cone responses
#
# 		:param image_center: x+yj, x for eccentricity and y for elevation (both in mm)
# 		:param kwargs:
# 		"""
# 		super(VisualImageArray, self).__init__(**kwargs)
# 		self.image_center = image_center

def load_dog_fits(csv_file_path):

    # All entries 0.0 => handpicked bad cell, all entries NaN => fitting failed
    dog_fits = pd.read_csv(csv_file_path, header=0, index_col=0).fillna(0.0)
    data_all_viable_cells = np.array(dog_fits)
    # Pick the rows where all the columns are zeros -> "bad data indices"
    bad_data_indices = np.where((dog_fits == 0.0).all(axis=1) )[0].tolist()
    param_names = dog_fits.columns.tolist()
    print('Loaded data for %d cells of which %d were bad' % (len(data_all_viable_cells), len(bad_data_indices)))

    return param_names, data_all_viable_cells, bad_data_indices



if __name__ == "__main__":
    # You can fit data at runtime
    # import apricot_fitter
    #
    # x = apricot_fitter.ConstructReceptiveFields()
    # fitdata = x.fit_dog_to_sta_data('midget', 'OFF', surround_model=1,
    #                       semi_x_always_major=True)

    # ...or load premade fits
    fitdata2 = load_dog_fits('results_temp/midget_OFF_surfix.csv')


    mosaic = GanglionCells(gc_type='parasol', response_type='ON', ecc_limits=[4, 6],
                                      sector_limits=[-10.0, 10.0], model_density=1.0, randomize_position=0.6)
    mosaic.build(fitdata2, visualize=False)
    # Or you can load a previously built mosaic
    #
    # mosaic.visualize_mosaic()

    a = vs.ConstructStimulus(video_center_vspace=5 + 0j, pattern='sine_grating', temporal_frequency=1,
                             spatial_frequency=0.5,
                             duration_seconds=3, fps=120, orientation=45, image_width=320, image_height=240,
                             pix_per_deg=10, stimulus_size=5.0)

    mosaic.load_stimulus(a)
    # mosaic.visualize_stimulus_and_grid(marked_cells=[2])
    # mosaic.visualize_rgc_view(2, show_block=True)

    for i in range(20):
        # mosaic.visualize_rgc_view(i, show_block=True)
        mosaic.feed_stimulus(i)
        # mosaic.create_spatiotemporal_kernel(i, visualize=True)

    plt.show()

# Targeting something like this:
# a = vs.ConstructStimulus(...)
# mosaic.load_stimulus(a)
# mosaic.run_single_cell_trials(5, 63, with_postspike=False)

# Todo (Henri's)

# - Is orientation CW or CCW in DOG fncs?
# - Eccentricity and polar angle used sometimes as if they are the same thing
# - "center" has double meaning: RF center or center/surround => change RF center to midpoint
# - semi_x and semi_y problematic: which one is major? does it describe 1SD ellipse? => change to sd_major/minor
# - orientation => orientation_ccw


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
