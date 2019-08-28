# This script fits spike-triggered average (STA) data from retinal ganglion cells (RGC) to functions expressed as
# the difference of two 2-dimensional elliptical Gaussians (DoG, Difference of Gaussians).
#
# The derived parameters are used to create artificial RGC mosaics and receptive fields (RFs).
#
# Data courtesy of The Chichilnisky Lab <http://med.stanford.edu/chichilnisky.html>
# Data paper: Field GD et al. (2010). Nature 467(7316):673-7.
# Only low resolution spatial RF maps are used here.


import sys
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse
from tqdm import tqdm
from pathlib import Path
from visualize import Visualize
from vision_maths import Mathematics

script_path = Path(__file__).parent
retina_data_path = script_path / 'apricot'
digitized_figures_path = script_path


class GetLiteratureData:
    '''
    Read data from external mat files. Data-specific definitions are isolated here.
    '''

    def read_gc_density_data(self):
        '''
        Read re-digitized old literature data from mat files
        '''

        gc_density = sio.loadmat(digitized_figures_path / 'Perry_1984_Neurosci_GCdensity_c.mat',
                                 variable_names=['Xdata', 'Ydata'])
        cell_eccentricity = np.squeeze(gc_density['Xdata'])
        cell_density = np.squeeze(gc_density['Ydata']) * 1e3  # Cells are in thousands, thus the 1e3
        return cell_eccentricity, cell_density

    def read_retina_glm_data(self, gc_type, responsetype):

        # Go to correct folder
        # cwd2 = os.getcwd()
        # retina_data_path = 'C:\\Users\\vanni\\OneDrive - University of Helsinki\\Work\\Simulaatiot\\Retinamalli\\Retina_GLM\\apricot'
        # retina_data_path = os.path.join(work_path, 'apricot')

        # Define filename
        if gc_type == 'parasol' and responsetype == 'ON':
            filename = 'Parasol_ON_spatial.mat'
            bad_data_indices=[15, 67, 71, 86, 89]   # Manually selected for Chichilnisky apricot data
            #bad_data_indices = []  # For debugging
        elif gc_type == 'parasol' and responsetype == 'OFF':
            filename = 'Parasol_OFF_spatial.mat'
            bad_data_indices = [6, 31, 73]
        elif gc_type == 'midget' and responsetype == 'ON':
            filename = 'Midget_ON_spatial.mat'
            bad_data_indices = [6, 13, 19, 23, 26, 28, 55, 74, 93, 99, 160, 162, 203, 220]
        elif gc_type == 'midget' and responsetype == 'OFF':
            filename = 'Midget_OFF_spatial.mat'
            bad_data_indices = [4, 5, 13, 23, 39, 43, 50, 52, 55, 58, 71, 72, 86, 88, 94, 100, 104, 119, 137,
                                154, 155, 169, 179, 194, 196, 224, 230, 234, 235, 239, 244, 250, 259, 278]
        else:
            print('Unkown celltype or responsetype, aborting')
            sys.exit()

        # Read data
        # filepath=os.path.join(retina_data_path,filename)
        filepath = retina_data_path / filename
        gc_spatial_data = sio.loadmat(filepath, variable_names=['c', 'stafit'])
        gc_spatial_data_array = gc_spatial_data['c']
        initial_center_values = gc_spatial_data['stafit']

        n_cells = len(gc_spatial_data_array[0,0,:])
        n_bad = len(bad_data_indices)
        print("Read %d cells from datafile and then removed %d bad cells (handpicked)" % (n_cells, n_bad))

        return gc_spatial_data_array, initial_center_values, bad_data_indices

    def read_dendritic_fields_vs_eccentricity_data(self):
        '''
        Read re-digitized old literature data from mat files
        '''
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


class ConstructReceptiveFields(GetLiteratureData, Visualize, Mathematics):
    """
    Methods for deriving spatial receptive field parameters from the apricot dataset (Field_2010)
    """

    def fit_dog_to_sta_data(self, gc_type, response_type, visualize=False, surround_fixed=False, save=None):
        '''
        Fits a function consisting of the difference of two 2-dimensional elliptical Gaussian functions to
        retinal spike triggered average (STA) data.
        The visualize parameter will show each DoG fit for search for bad cell fits and data.
        '''
        # gc_type = self.gc_type
        # response_type = self.response_type

        gc_spatial_data_array, initial_center_values, bad_data_indices = self.read_retina_glm_data(gc_type,
                                                                                                   response_type)

        n_cells = int(gc_spatial_data_array.shape[2])
        pixel_array_shape_y = gc_spatial_data_array.shape[0]  # Check indices: x horizontal, y vertical
        pixel_array_shape_x = gc_spatial_data_array.shape[1]

        # Make fit to all cells
        x_position_indices = np.linspace(1, pixel_array_shape_x,
                                         pixel_array_shape_x)  # Note: input coming from matlab, thus indexing starts from 1
        y_position_indices = np.linspace(1, pixel_array_shape_y, pixel_array_shape_y)
        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)

        all_viable_cells = np.setdiff1d(np.arange(n_cells), bad_data_indices)
        # Empty numpy matrix to collect fitted RFs
        if surround_fixed:
            parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes',
                               'sur_ratio', 'offset']
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = 'fixed'
        # if surround_fixed: # delta_semi_y
        # parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'delta_semi_y', 'orientation_center', 'amplitudes', 'sur_ratio', 'offset']
        # data_all_viable_cells = np.zeros(np.array([n_cells,len(parameter_names)]))
        # surround_status = 'fixed'
        else:
            parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes',
                               'xos', 'yos', 'semi_xs',
                               'semi_ys', 'orientation_surround', 'offset']
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))

            surround_status = 'independent'

        print(('Fitting DoG model, surround is {0}'.format(surround_status)))

        for cell_index in tqdm(all_viable_cells):
            # pbar(cell_index/n_cells)
            data_array = gc_spatial_data_array[:, :, cell_index]
            # Drop outlier cells

            # # Initial guess for center
            center_rotation_angle = float(initial_center_values[0, cell_index][4])
            if center_rotation_angle < 0:  # For negative angles, turn positive
                center_rotation_angle = center_rotation_angle + 2 * np.pi

            # Invert data arrays with negative sign for fitting and display. Fitting assumes that center peak is above mean
            if data_array.ravel()[np.argmax(np.abs(data_array))] < 0:
                data_array = data_array * -1

            # Set initial guess for fitting
            if surround_fixed:
                # Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset)
                p0 = np.array([1, 7, 7, 3, 3,
                               center_rotation_angle, 0.1, 3, 0])
                # boundaries=(np.array([.999, -np.inf, -np.inf, 0, 0, -2*np.pi, 0, 1, -np.inf]),
                # np.array([1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, 1, np.inf, np.inf]))
                boundaries = (np.array([.999, -np.inf, -np.inf, 0, 0, 0, 0, 1, -np.inf]),
                              np.array([1, np.inf, np.inf, np.inf, np.inf, 2 * np.pi, 1, np.inf, np.inf]))
            # if surround_fixed: # delta_semi_y
            # # Build initial guess for (amplitudec, xoc, yoc, semi_xc, delta_semi_y, orientation_center, amplitudes, sur_ratio, offset)
            # p0 = np.array([1, 7, 7, 3, 0,
            # center_rotation_angle, 0.1, 3, 0])
            # boundaries=(np.array([.999, -np.inf, -np.inf, 0, 0, 0, 0, 1, -np.inf]),
            # np.array([1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, 1, np.inf, np.inf]))
            else:
                # Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, xos, yos, semi_xs, semi_ys, orientation_surround, offset)
                p0 = np.array([1, 7, 7, 3, 3,
                               center_rotation_angle, 0.1, 7, 7,
                               3 * 3, 3 * 3, center_rotation_angle, 0])
                boundaries = (
                np.array([.999, -np.inf, -np.inf, 0, 0, -2 * np.pi, 0, -np.inf, -np.inf, 0, 0, -2 * np.pi, -np.inf]),
                np.array([1, np.inf, np.inf, np.inf, np.inf, 2 * np.pi, 1, np.inf, np.inf, np.inf, np.inf, 2 * np.pi,
                          np.inf]))

            try:
                if surround_fixed:
                    popt, pcov = opt.curve_fit(self.DoG2D_fixed_surround, (x_grid, y_grid), data_array.ravel(), p0=p0,
                                               bounds=boundaries)
                    data_all_viable_cells[cell_index, :] = popt
                else:
                    popt, pcov = opt.curve_fit(self.DoG2D_independent_surround, (x_grid, y_grid), data_array.ravel(),
                                               p0=p0, bounds=boundaries)
                    data_all_viable_cells[cell_index, :] = popt
            except:
                print(('Fitting failed for cell {0}'.format(str(cell_index))))
                data_all_viable_cells[cell_index, :] = np.nan
                bad_data_indices.append(cell_index)
                continue

            if visualize:
                # Visualize fits with data
                data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
                fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
                plt.suptitle(
                    'celltype={0}, responsetype={1}, cell N:o {2}'.format(gc_type, response_type, str(cell_index)),
                    fontsize=10)
                cen = ax1.imshow(data_array, vmin=-0.1, vmax=0.4, cmap=plt.cm.gray, origin='bottom',
                                 extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()))
                fig.colorbar(cen, ax=ax1)

                # # Ellipses for DoG2D_fixed_surround

                if self.surround_fixed:
                    e1 = ellipse((popt[np.array([1, 2])]), popt[3], popt[4], -popt[5] * 180 / np.pi, edgecolor='w',
                                 linewidth=2, fill=False)
                    e2 = ellipse((popt[np.array([1, 2])]), popt[7] * popt[3], popt[7] * popt[4], -popt[5] * 180 / np.pi,
                                 edgecolor='w', linewidth=2, fill=False, linestyle='--')
                    print(popt[0], popt[np.array([1, 2])], popt[3], popt[4], -popt[5] * 180 / np.pi)
                    print(popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8])
                # if surround_fixed: # delta_semi_y
                # # Build initial guess for (amplitudec, xoc, yoc, semi_xc, delta_semi_y, orientation_center, amplitudes, sur_ratio, offset)
                # e1=ellipse((popt[np.array([1,2])]),popt[3],popt[3]+popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False)
                # e2=ellipse((popt[np.array([1,2])]),popt[7]*popt[3],popt[7]*(popt[3]+popt[4]),-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')
                # print popt[0], popt[np.array([1,2])],'semi_xc=',popt[3], 'delta_semi_y=', popt[4],-popt[5]*180/np.pi
                # print popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8]
                else:
                    e1 = ellipse((popt[np.array([1, 2])]), popt[3], popt[4], -popt[5] * 180 / np.pi, edgecolor='w',
                                 linewidth=2, fill=False)
                    e2 = ellipse((popt[np.array([7, 8])]), popt[9], popt[10], -popt[11] * 180 / np.pi, edgecolor='w',
                                 linewidth=2, fill=False, linestyle='--')
                    print(popt[0], popt[np.array([1, 2])], popt[3], popt[4], -popt[5] * 180 / np.pi)
                    print(popt[6], popt[np.array([7, 8])], popt[9], popt[10], -popt[11] * 180 / np.pi)

                print('\n')

                ax1.add_artist(e1)
                ax1.add_artist(e2)

                sur = ax2.imshow(data_fitted.reshape(pixel_array_shape_y, pixel_array_shape_x), vmin=-0.1, vmax=0.4,
                                 cmap=plt.cm.gray, origin='bottom')
                fig.colorbar(sur, ax=ax2)

                plt.show()

            if save is not None:
                assert type(save) == str, "Use the parameter save to specify the output filename"
                fits_df = pd.DataFrame(data_all_viable_cells, columns=parameter_names)
                fits_df.to_csv(save)

            return parameter_names, data_all_viable_cells, bad_data_indices

    def fit_spatial_statistics(self, visualize=False):
        """
        Collect spatial statistics from Chichilnisky receptive field data
        """

        # 2D DoG fit to Chichilnisky retina spike triggered average data. The visualize parameter will
        # show each DoG fit for search for bad cell fits and data.
        parameter_names, data_all_viable_cells, bad_cell_indices = \
            self.fit_dog_to_sta_data(visualize=False, surround_fixed=self.surround_fixed)

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
        loc = np.zeros([n_distributions]);
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

    # return chichilnisky_data_df  -- chewing gum fix for a demo

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


if __name__ == '__main__':
    x = ConstructReceptiveFields()
    parameter_names, data_all_viable_cells, bad_data_indices = \
        x.fit_dog_to_sta_data('parasol', 'ON', save='results_temp/parasol_ON_surfix.csv', surround_fixed=True)
