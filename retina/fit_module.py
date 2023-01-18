""" 
These classes fit spike-triggered average (STA) data from retinal ganglion cells (RGC) to functions 
expressed as the difference of two 2-dimensional elliptical Gaussians (DoG, Difference of Gaussians).

The derived parameters are used to create artificial RGC mosaics and receptive fields (RFs).

Data courtesy of The Chichilnisky Lab <http://med.stanford.edu/chichilnisky.html>
Data paper: Field GD et al. (2010). Nature 467(7316):673-7.
Only low resolution spatial RF maps are used here.
"""

# Numerical
import numpy as np
import scipy.optimize as opt
from scipy.optimize import curve_fit
import pandas as pd

# Viz
from tqdm import tqdm

# Local
from retina.retina_math_module import RetinaMath
from retina.apricot_data_module import ApricotData

# Builtin
import pdb


class Fit(ApricotData, RetinaMath):
    """
    Methods for deriving spatial and temporal receptive field parameters from the apricot dataset (Field_2010)
    Call get_fits method to return the fits from the instance object self.all_fits
    """

    def __init__(self, apricot_data_folder, gc_type, response_type, _fit_all=True):

        super().__init__(apricot_data_folder, gc_type, response_type)
        if _fit_all is True:
            self._fit_all()

    def _fit_temporal_filters(self, normalize_before_fit=False):
        """
        Fits each temporal filter to a function consisting of the difference of two cascades of lowpass filters. This follows Chichilnisky&Kalmar 2002 JNeurosci.

        :return dataframe: concatenated parameters_df, error_df
        :set self.temporal_filters_to_show: dict of temporal filters to show with viz
        """
        # shape (n_cells, 15); 15 time points @ 30 Hz (500 ms)
        if normalize_before_fit is True:
            temporal_filters = self.read_temporal_filter_data(
                flip_negs=True, normalize=True
            )
        else:
            temporal_filters = self.read_temporal_filter_data(flip_negs=True)

        data_fps = self.metadata["data_fps"]
        data_n_samples = self.metadata["data_temporalfilter_samples"]

        good_indices = np.setdiff1d(np.arange(self.n_cells), self.bad_data_indices)
        parameter_names = ["n", "p1", "p2", "tau1", "tau2"]
        bounds = (
            [0, 0, 0, 0.1, 3],
            [np.inf, 10, 10, 3, 6],
        )  # bounds when time points are 0...14

        fitted_parameters = np.zeros((self.n_cells, len(parameter_names)))
        error_array = np.zeros(self.n_cells)
        max_error = -0.1

        xdata = np.arange(15)
        xdata_finer = np.linspace(0, max(xdata), 100)
        temporal_filters_to_show = {
            "xdata": xdata,
            "xdata_finer": xdata_finer,
            "title": f"{self.gc_type}_{self.response_type}",
        }

        for cell_ix in tqdm(good_indices, desc="Fitting temporal filters"):
            ydata = temporal_filters[cell_ix, :]

            try:
                popt, pcov = curve_fit(
                    self.diff_of_lowpass_filters, xdata, ydata, bounds=bounds
                )
                fitted_parameters[cell_ix, :] = popt
                error_array[cell_ix] = (1 / data_n_samples) * np.sum(
                    (ydata - self.diff_of_lowpass_filters(xdata, *popt)) ** 2
                )  # MSE error
            except:
                print("Fitting for cell index %d failed" % cell_ix)
                fitted_parameters[cell_ix, :] = np.nan
                error_array[cell_ix] = max_error
                continue

            temporal_filters_to_show[f"cell_ix_{cell_ix}"] = {
                "ydata": ydata,
                "y_fit": self.diff_of_lowpass_filters(xdata_finer, *popt),
            }

        parameters_df = pd.DataFrame(fitted_parameters, columns=parameter_names)
        # Convert taus to milliseconds
        parameters_df["tau1"] = parameters_df["tau1"] * (1 / data_fps) * 1000
        parameters_df["tau2"] = parameters_df["tau2"] * (1 / data_fps) * 1000

        error_df = pd.DataFrame(error_array, columns=["temporalfit_mse"])

        # For visualization in separate viz module
        self.temporal_filters_to_show = temporal_filters_to_show

        return pd.concat([parameters_df, error_df], axis=1)

    def _fit_spatial_filters(
        self,
        surround_model=1,
        semi_x_always_major=True,
    ):
        """
        Fits a function consisting of the difference of two 2-dimensional elliptical Gaussian functions to
        retinal spike triggered average (STA) data. The show_spatial_filter_response parameter will show
        each DoG fit in order to search for bad cell fits and data.

        Parameters
        ----------
        surround_model : int, optional, 0=fit center and surround separately, 1=surround midpoint same as center midpoint, by default 1
        semi_x_always_major : bool, optional, whether to rotate Gaussians so that semi_x is always the semimajor/longer axis, by default True

        Returns
        -------
        dataframe with spatial parameters and errors

        """

        (
            gc_spatial_data_array,
            initial_center_values,
            bad_data_indices,
        ) = self.read_spatial_filter_data()

        n_cells = int(gc_spatial_data_array.shape[2])
        pixel_array_shape_y = gc_spatial_data_array.shape[
            0
        ]  # Check indices: x horizontal, y vertical
        pixel_array_shape_x = gc_spatial_data_array.shape[1]

        # Make fit to all cells
        x_position_indices = np.linspace(
            1, pixel_array_shape_x, pixel_array_shape_x
        )  # Note: input coming from matlab, thus indexing starts from 1
        y_position_indices = np.linspace(1, pixel_array_shape_y, pixel_array_shape_y)
        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)

        all_viable_cells = np.setdiff1d(np.arange(n_cells), bad_data_indices)

        if surround_model == 1:
            parameter_names = [
                "amplitudec",
                "xoc",
                "yoc",
                "semi_xc",
                "semi_yc",
                "orientation_center",
                "amplitudes",
                "sur_ratio",
                "offset",
            ]
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = "fixed"

        else:
            parameter_names = [
                "amplitudec",
                "xoc",
                "yoc",
                "semi_xc",
                "semi_yc",
                "orientation_center",
                "amplitudes",
                "xos",
                "yos",
                "semi_xs",
                "semi_ys",
                "orientation_surround",
                "offset",
            ]
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))

            surround_status = "independent"

        # Create error & other arrays
        error_all_viable_cells = np.zeros((n_cells, 1))
        dog_filtersum_array = np.zeros((n_cells, 4))

        spatial_filters_to_show = {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "surround_model": surround_model,
            "pixel_array_shape_x": pixel_array_shape_x,
            "pixel_array_shape_y": pixel_array_shape_y,
        }

        # Go through all cells
        print(("Fitting DoG model, surround is {0}".format(surround_status)))
        for cell_index in tqdm(all_viable_cells, desc="Fitting spatial  filters"):
            # pbar(cell_index/n_cells)
            data_array = gc_spatial_data_array[:, :, cell_index]
            # Drop outlier cells

            # # Initial guess for center
            center_rotation_angle = float(initial_center_values[0, cell_index][4])
            if center_rotation_angle < 0:  # For negative angles, turn positive
                center_rotation_angle = center_rotation_angle + 2 * np.pi

            # Invert data arrays with negative sign for fitting and display.
            # Fitting assumes that center peak is above mean.
            if data_array.ravel()[np.argmax(np.abs(data_array))] < 0:
                data_array = data_array * -1

            # Set initial guess for fitting
            if surround_model == 1:
                # Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset)
                p0 = np.array([1, 7, 7, 3, 3, center_rotation_angle, 0.1, 3, 0])
                boundaries = (
                    np.array([0.999, -np.inf, -np.inf, 0, 0, 0, 0, 1, 0]),
                    np.array(
                        [1, np.inf, np.inf, np.inf, np.inf, 2 * np.pi, 1, np.inf, 0.001]
                    ),
                )
            else:
                # Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, xos, yos, semi_xs, semi_ys, orientation_surround, offset)
                p0 = np.array(
                    [
                        1,
                        7,
                        7,
                        3,
                        3,
                        center_rotation_angle,
                        0.1,
                        7,
                        7,
                        3 * 3,
                        3 * 3,
                        center_rotation_angle,
                        0,
                    ]
                )
                boundaries = (
                    np.array(
                        [
                            0.999,
                            -np.inf,
                            -np.inf,
                            0,
                            0,
                            -2 * np.pi,
                            0,
                            -np.inf,
                            -np.inf,
                            0,
                            0,
                            -2 * np.pi,
                            -np.inf,
                        ]
                    ),
                    np.array(
                        [
                            1,
                            np.inf,
                            np.inf,
                            np.inf,
                            np.inf,
                            2 * np.pi,
                            1,
                            np.inf,
                            np.inf,
                            np.inf,
                            np.inf,
                            2 * np.pi,
                            np.inf,
                        ]
                    ),
                )

            try:
                if surround_model == 1:
                    popt, pcov = opt.curve_fit(
                        self.DoG2D_fixed_surround,
                        (x_grid, y_grid),
                        data_array.ravel(),
                        p0=p0,
                        bounds=boundaries,
                    )
                    data_all_viable_cells[cell_index, :] = popt

                else:
                    popt, pcov = opt.curve_fit(
                        self.DoG2D_independent_surround,
                        (x_grid, y_grid),
                        data_array.ravel(),
                        p0=p0,
                        bounds=boundaries,
                    )
                    data_all_viable_cells[cell_index, :] = popt
            except:
                print(("Fitting failed for cell {0}".format(str(cell_index))))
                data_all_viable_cells[cell_index, :] = np.nan
                bad_data_indices.append(cell_index)
                continue

            # Set rotation angle between 0 and pi
            data_all_viable_cells[cell_index, 5] = (
                data_all_viable_cells[cell_index, 5] % np.pi
            )

            # Rotate fit so that semi_x is always the semimajor axis (longer radius)
            if semi_x_always_major is True:
                if (
                    data_all_viable_cells[cell_index, 3]
                    < data_all_viable_cells[cell_index, 4]
                ):
                    sd_x = data_all_viable_cells[cell_index, 3]
                    sd_y = data_all_viable_cells[cell_index, 4]
                    rotation = data_all_viable_cells[cell_index, 5]

                    data_all_viable_cells[cell_index, 3] = sd_y
                    data_all_viable_cells[cell_index, 4] = sd_x
                    data_all_viable_cells[cell_index, 5] = (
                        rotation + np.pi / 2
                    ) % np.pi

                # Rotate also the surround if it is defined separately
                if surround_model == 0:
                    if (
                        data_all_viable_cells[cell_index, 9]
                        < data_all_viable_cells[cell_index, 10]
                    ):
                        sd_x_sur = data_all_viable_cells[cell_index, 9]
                        sd_y_sur = data_all_viable_cells[cell_index, 10]
                        rotation = data_all_viable_cells[cell_index, 11]

                        data_all_viable_cells[cell_index, 9] = sd_y_sur
                        data_all_viable_cells[cell_index, 10] = sd_x_sur
                        data_all_viable_cells[cell_index, 11] = (
                            rotation + np.pi / 2
                        ) % np.pi

            # Set rotation angle between -pi/2 and pi/2 (otherwise hist bimodal)
            rotation = data_all_viable_cells[cell_index, 5]
            if rotation > np.pi / 2:
                data_all_viable_cells[cell_index, 5] = rotation - np.pi
            else:
                data_all_viable_cells[cell_index, 5] = rotation

            # Compute fitting error
            if surround_model == 1:
                data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
            else:
                data_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)

            data_fitted = data_fitted.reshape(pixel_array_shape_y, pixel_array_shape_x)
            fit_deviations = data_fitted - data_array
            data_mean = np.mean(data_array)
            # Normalized mean square error
            # Defn per https://se.mathworks.com/help/ident/ref/goodnessoffit.html without 1 - ...
            # 0 = perfect fit, infty = bad fit

            # MSE
            fit_error = np.sum(fit_deviations**2) / (13 * 13)
            error_all_viable_cells[cell_index, 0] = fit_error

            # Save DoG fit sums
            dog_filtersum_array[cell_index, 0] = np.sum(data_fitted[data_fitted > 0])
            dog_filtersum_array[cell_index, 1] = (-1) * np.sum(
                data_fitted[data_fitted < 0]
            )
            dog_filtersum_array[cell_index, 2] = np.sum(data_fitted)
            dog_filtersum_array[cell_index, 3] = np.sum(data_array[data_array > 0])

            # For visualization
            spatial_filters_to_show[f"cell_ix_{cell_index}"] = {
                "data_array": data_array,
                "suptitle": f"celltype={self.gc_type}, responsetype={self.response_type}, cell_ix={cell_index}",
            }

        # FOR loop ends here
        spatial_filters_to_show["data_all_viable_cells"] = data_all_viable_cells

        # Finally build a dataframe of the fitted parameters
        fits_df = pd.DataFrame(data_all_viable_cells, columns=parameter_names)
        aspect_ratios_df = pd.DataFrame(
            fits_df.semi_xc / fits_df.semi_yc, columns=["aspect_ratio"]
        ).fillna(0.0)
        dog_filtersum_df = pd.DataFrame(
            dog_filtersum_array,
            columns=[
                "dog_filtersum_cen",
                "dog_filtersum_sur",
                "dog_filtersum_total",
                "ctrl_filtersum_cen",
            ],
        )

        error_df = pd.DataFrame(error_all_viable_cells, columns=["spatialfit_mse"])
        good_indices = np.ones(len(data_all_viable_cells))
        for i in self.bad_data_indices:
            good_indices[i] = 0
        good_indices_df = pd.DataFrame(good_indices, columns=["good_filter_data"])

        # Save for later visualization
        self.spatial_filters_to_show = spatial_filters_to_show

        return pd.concat(
            [fits_df, aspect_ratios_df, dog_filtersum_df, error_df, good_indices_df],
            axis=1,
        )

    def _fit_all(self):
        """
        Fits spatial, temporal and tonic drive parameters to the ApricotData
        Returns the fits as self.all_fits which is an instance object attribute.
        """
        spatial_fits = self._fit_spatial_filters(
            surround_model=1,
            semi_x_always_major=True,
        )
        spatial_filter_sums = self.compute_spatial_filter_sums()

        temporal_fits = self._fit_temporal_filters()
        temporal_filter_sums = self.compute_temporal_filter_sums()

        tonicdrives = pd.DataFrame(self.read_tonicdrive(), columns=["tonicdrive"])

        # Collect everything into one big dataframe
        self.all_fits = pd.concat(
            [
                spatial_fits,
                spatial_filter_sums,
                temporal_fits,
                temporal_filter_sums,
                tonicdrives,
            ],
            axis=1,
        )

    def get_fits(self):
        return (
            self.all_fits,
            self.temporal_filters_to_show,
            self.spatial_filters_to_show,
        )

    def save(self, filepath):
        self.all_fits.to_csv(filepath)


# if __name__ == '__main__':

#     a = Fit('parasol', 'on')
#     a._fit_temporal_filters(show_temporal_filter_response=True, normalize_before_fit=True)
