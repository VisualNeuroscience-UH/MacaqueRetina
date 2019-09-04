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
from matplotlib.patches import Ellipse
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


class ConstructReceptiveFields(GetLiteratureData, Visualize, Mathematics):
    """
    Methods for deriving spatial receptive field parameters from the apricot dataset (Field_2010)
    """

    def fit_dog_to_sta_data(self, gc_type, response_type, visualize=False, surround_model=0, save=None, semi_x_always_major=False):
        """
        Fits a function consisting of the difference of two 2-dimensional elliptical Gaussian functions to
        retinal spike triggered average (STA) data.
        The visualize parameter will show each DoG fit for search for bad cell fits and data.

        :param gc_type: parasol/midget
        :param response_type: ON/OFF
        :param visualize: boolean, whether to visualize all fits
        :param surround_model: 0=fit center and surround separately, 1=surround midpoint same as center midpoint, 2=same as 1 but surround ratio fixed at 2 and no offset
        :param save: string, relative path to a csv file for saving
        :param semi_x_always_major: boolean, whether to rotate Gaussians so that semi_x is always the semimajor/longer axis
        :return:
        """

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

        # Create an empty matrix to collect fitted RFs
        # parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes',
        #                    'xos', 'yos', 'semi_xs', 'semi_ys', 'orientation_surround', 'offset']
        # data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))

        if surround_model == 1:
            parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes',
                               'sur_ratio', 'offset']
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = 'fixed'

        elif surround_model == 2:
            # Same parameter names as in "fixed surround" but amplitudec, sur_ratio or offset will not be fitted
            parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes',
                               'sur_ratio', 'offset']
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = 'fixed_double'

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
            if surround_model == 1:
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

            elif surround_model == 2:
                # Initial guess for
                # xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes
                p0 = np.array([7, 7, 1, 1, 0, 0.1])
                boundaries = (
                    np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0,      -np.inf]),
                    np.array([np.inf,   np.inf,  np.inf,  np.inf, 2*np.pi, np.inf])
                )

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
                if surround_model == 1:
                    popt, pcov = opt.curve_fit(self.DoG2D_fixed_surround, (x_grid, y_grid), data_array.ravel(), p0=p0,
                                               bounds=boundaries)
                    data_all_viable_cells[cell_index, :] = popt

                elif surround_model == 2:
                    popt, pcov = opt.curve_fit(self.DoG2D_fixed_double_surround, (x_grid, y_grid), data_array.ravel(), p0=p0,
                                               bounds=boundaries)
                    data_all_viable_cells[cell_index, 1:7] = popt
                    data_all_viable_cells[:, 0] = 1.0  # amplitudec
                    data_all_viable_cells[:, 7] = 2.0  # sur_ratio
                    data_all_viable_cells[:, 8] = 0.0  # offset


                else:
                    popt, pcov = opt.curve_fit(self.DoG2D_independent_surround, (x_grid, y_grid), data_array.ravel(),
                                               p0=p0, bounds=boundaries)
                    data_all_viable_cells[cell_index, :] = popt
            except:
                print(('Fitting failed for cell {0}'.format(str(cell_index))))
                data_all_viable_cells[cell_index, :] = np.nan
                bad_data_indices.append(cell_index)
                continue

            # Set rotation angle between 0 and pi
            data_all_viable_cells[cell_index, 5] = data_all_viable_cells[cell_index, 5] % np.pi

            # Rotate fit so that semi_x is always the semimajor axis (longer radius)
            if semi_x_always_major is True:
                if data_all_viable_cells[cell_index, 3] < data_all_viable_cells[cell_index, 4]:
                    sd_x = data_all_viable_cells[cell_index, 3]
                    sd_y = data_all_viable_cells[cell_index, 4]
                    rotation = data_all_viable_cells[cell_index, 5]

                    data_all_viable_cells[cell_index, 3] = sd_y
                    data_all_viable_cells[cell_index, 4] = sd_x
                    data_all_viable_cells[cell_index, 5] = (rotation + np.pi/2) % np.pi

                # Rotate also the surround if it is defined separately
                if surround_model == 0:
                    if data_all_viable_cells[cell_index, 9] < data_all_viable_cells[cell_index, 10]:
                        sd_x_sur = data_all_viable_cells[cell_index, 9]
                        sd_y_sur = data_all_viable_cells[cell_index, 10]
                        rotation = data_all_viable_cells[cell_index, 11]

                        data_all_viable_cells[cell_index, 9] = sd_y_sur
                        data_all_viable_cells[cell_index, 10] = sd_x_sur
                        data_all_viable_cells[cell_index, 11] = (rotation + np.pi / 2) % np.pi

            if visualize:
                # Visualize fits with data
                #data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)

                popt = data_all_viable_cells[cell_index, :]

                fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
                plt.suptitle(
                    'celltype={0}, responsetype={1}, cell N:o {2}'.format(gc_type, response_type, str(cell_index)),
                    fontsize=10)
                cen = ax1.imshow(data_array, vmin=-0.1, vmax=0.4, cmap=plt.cm.gray, origin='bottom',
                                 extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()))
                fig.colorbar(cen, ax=ax1)

                # # Ellipses for DoG2D_fixed_surround

                if surround_model == 1:
                    data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
                    # matplotlib.patches.Ellipse(xy, width, height, angle=0, **kwargs)
                    e1 = Ellipse((popt[np.array([1, 2])]), popt[3], popt[4], -popt[5] * 180 / np.pi, edgecolor='w',
                                 linewidth=2, fill=False)
                    e2 = Ellipse((popt[np.array([1, 2])]), popt[7] * popt[3], popt[7] * popt[4], -popt[5] * 180 / np.pi,
                                 edgecolor='w', linewidth=2, fill=False, linestyle='--')
                    print(popt[0], popt[np.array([1, 2])], popt[3], popt[4], -popt[5] * 180 / np.pi)
                    print(popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8])
                # if surround_fixed: # delta_semi_y
                # # Build initial guess for (amplitudec, xoc, yoc, semi_xc, delta_semi_y, orientation_center, amplitudes, sur_ratio, offset)
                # e1=ellipse((popt[np.array([1,2])]),popt[3],popt[3]+popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False)
                # e2=ellipse((popt[np.array([1,2])]),popt[7]*popt[3],popt[7]*(popt[3]+popt[4]),-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')
                # print popt[0], popt[np.array([1,2])],'semi_xc=',popt[3], 'delta_semi_y=', popt[4],-popt[5]*180/np.pi
                # print popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8]
                elif surround_model == 2:
                    data_fitted = self.DoG2D_fixed_double_surround((x_grid, y_grid), *popt[1:7])
                    e1 = Ellipse((popt[np.array([1, 2])]), popt[3], popt[4], -popt[5] * 180 / np.pi, edgecolor='w',
                                 linewidth=2, fill=False)
                    e2 = Ellipse((popt[np.array([1, 2])]), popt[7] * popt[3], popt[7] * popt[4], -popt[5] * 180 / np.pi,
                                 edgecolor='w', linewidth=2, fill=False, linestyle='--')
                    print(popt[0], popt[np.array([1, 2])], popt[3], popt[4], -popt[5] * 180 / np.pi)
                    print(popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8])

                else:
                    data_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)
                    e1 = Ellipse((popt[np.array([1, 2])]), popt[3], popt[4], -popt[5] * 180 / np.pi, edgecolor='w',
                                 linewidth=2, fill=False)
                    e2 = Ellipse((popt[np.array([7, 8])]), popt[9], popt[10], -popt[11] * 180 / np.pi, edgecolor='w',
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
            # FOR loop ends here

        if save is not None:
            assert type(save) == str, "Use the parameter save to specify the output filename"
            fits_df = pd.DataFrame(data_all_viable_cells, columns=parameter_names)
            fits_df.to_csv(save)

        return parameter_names, data_all_viable_cells, bad_data_indices


if __name__ == '__main__':
    x = ConstructReceptiveFields()
    x.fit_dog_to_sta_data('parasol', 'ON', surround_model=1, visualize=False,
                          semi_x_always_major=True, save='results_temp/parasol_ON_surfix.csv')
    x.fit_dog_to_sta_data('parasol', 'OFF', surround_model=1, visualize=False,
                      semi_x_always_major=True, save='results_temp/parasol_OFF_surfix.csv')
    x.fit_dog_to_sta_data('midget', 'ON', surround_model=1, visualize=False,
                      semi_x_always_major=True, save='results_temp/midget_ON_surfix.csv')
    x.fit_dog_to_sta_data('midget', 'OFF', surround_model=1, visualize=False,
                      semi_x_always_major=True, save='results_temp/midget_OFF_surfix.csv')

