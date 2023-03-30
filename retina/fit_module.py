""" 
These classes fit spike-triggered average (STA) data from retinal ganglion cells (RGC) to functions 
expressed as the difference of two 2-dimensional elliptical Gaussians (DoG, Difference of Gaussians).

The derived parameters are used to create artificial RGC mosaics and receptive fields (RFs).
"""

# Numerical
import numpy as np
import scipy.optimize as opt
from scipy.optimize import curve_fit
import scipy.stats as stats
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
    Call get_fits method to return the fits from the instance object self.all_data_fits_df
    """

    def __init__(self, apricot_data_folder, gc_type, response_type, _fit_all=True):

        super().__init__(apricot_data_folder, gc_type, response_type)
        if _fit_all is True:
            # Fit spatial and temporal filters and tonic drive values to experimental data.
            self._fit_all()

    def _fit_temporal_filters(self, normalize_before_fit=False):
        """
        Fits each temporal filter to a function consisting of the difference of two cascades of lowpass filters.
        This follows Chichilnisky&Kalmar 2002 JNeurosci. Uses retinal spike triggered average (STA) data.

        Parameters
        ----------
        normalize_before_fit : bool
            If True, normalize each temporal filter before fitting.
            If False, fit the raw temporal filters.

        Attributes
        ----------
        self.temporal_filters_to_show : dict
            Dictionary of temporal filters to show with viz

        Returns
        -------
        fitted_parameters : np.ndarray
            Array of shape (n_cells, 5) containing the fitted parameters for each cell.
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
        """
        Parameters
        ----------
        - n (float): Order of the filters.
        - p1 (float): Normalization factor for the first filter.
        - p2 (float): Normalization factor for the second filter.
        - tau1 (float): Time constant of the first filter.
        - tau2 (float): Time constant of the second filter.
        """
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
        Fit spatial filters to a difference of Gaussians (DoG) model. Uses retinal spike triggered average (STA) data.

        Parameters
        ----------
        surround_model : int, optional
            0=fit center and surround separately, 1=surround midpoint same as center midpoint, by default 1
        semi_x_always_major : bool, optional
            Whether to rotate Gaussians so that semi_x is always the semimajor/longer axis, by default True

        Attributes
        ----------
        self.spatial_filters_to_show : dict
            Dictionary of spatial filters to show with viz

        Returns
        -------
        dataframe with spatial parameters and errors for each cell (n_cells, 8)

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
            cen_rot_rad = float(initial_center_values[0, cell_index][4])
            if cen_rot_rad < 0:  # For negative angles, turn positive
                cen_rot_rad = cen_rot_rad + 2 * np.pi

            # Invert data arrays with negative sign for fitting and display.
            # Fitting assumes that center peak is above mean.
            if data_array.ravel()[np.argmax(np.abs(data_array))] < 0:
                data_array = data_array * -1

            # Set initial guess for fitting
            if surround_model == 1:
                # Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset)
                p0 = np.array([1, 7, 7, 3, 3, cen_rot_rad, 0.1, 3, 0])
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
                        cen_rot_rad,
                        0.1,
                        7,
                        7,
                        3 * 3,
                        3 * 3,
                        cen_rot_rad,
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
        Fits spatial, temporal and tonic drive parameters to the experimental data.
        Returns the fits as self.all_data_fits_df which is an instance object attribute.
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
        self.all_data_fits_df = pd.concat(
            [
                spatial_fits,
                spatial_filter_sums,
                temporal_fits,
                temporal_filter_sums,
                tonicdrives,
            ],
            axis=1,
        )

    def save(self, filepath):
        self.all_data_fits_df.to_csv(filepath)

    def _fit_spatial_statistics(self):
        """
        Fit spatial statistics of the spatial filter parameters. Returns gamma distribution parameters,
        except for orientation where it returns beta distribution parameters.

        Returns
        -------
        spatial_statistics_df : pd.DataFrame
            Dataframe with spatial statistics
        """

        # parameter_names, data_all_viable_cells, bad_cell_indices = fitdata
        data_all_viable_cells = np.array(self.all_data_fits_df)

        bad_cell_indices = np.where((self.all_data_fits_df == 0.0).all(axis=1))[
            0
        ].tolist()
        parameter_names = self.all_data_fits_df.columns.tolist()

        all_viable_cells = np.delete(data_all_viable_cells, bad_cell_indices, 0)

        spatial_data_df = pd.DataFrame(data=all_viable_cells, columns=parameter_names)

        # Save stats description to gc object
        self.rf_datafit_description_series = spatial_data_df.describe()

        # Calculate xy_aspect_ratio
        xy_aspect_ratio_pd_series = (
            spatial_data_df["semi_yc"] / spatial_data_df["semi_xc"]
        )
        xy_aspect_ratio_pd_series.rename("xy_aspect_ratio")
        spatial_data_df["xy_aspect_ratio"] = xy_aspect_ratio_pd_series

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
            ydata[:, index] = spatial_data_df[distribution]
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
        ydata[:, index] = spatial_data_df[rf_parameter_names[-1]]
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

        spatial_statistics_df = pd.DataFrame.from_dict(
            spatial_statistics_dict, orient="index"
        )
        spatial_statistics_df["domain"] = "spatial"

        # Return stats for RF creation
        return spatial_statistics_df

    def _fit_temporal_statistics(self):
        """
        Fit temporal statistics of the temporal filter parameters. Uses gamma distribution.

        Returns
        -------
        temporal_statistics_df : pd.DataFrame
            Dataframe with temporal statistics.
        """

        temporal_filter_parameters = ["n", "p1", "p2", "tau1", "tau2"]
        distrib_params = np.zeros((len(temporal_filter_parameters), 3))

        for i, param_name in enumerate(temporal_filter_parameters):
            param_array = np.array(
                self.all_data_fits_df.iloc[self.good_data_indices][param_name]
            )
            shape, loc, scale = stats.gamma.fit(param_array)
            distrib_params[i, :] = [shape, loc, scale]

        self.temporal_statistics_to_show = {
            "temporal_filter_parameters": temporal_filter_parameters,
            "distrib_params": distrib_params,
            "suptitle": self.gc_type + " " + self.response_type,
            "all_data_fits_df": self.all_data_fits_df,
            "good_data_indices": self.good_data_indices,
        }

        temporal_statistics_df = pd.DataFrame(
            distrib_params,
            index=temporal_filter_parameters,
            columns=["shape", "loc", "scale"],
        )

        temporal_statistics_df["distribution"] = "gamma"
        temporal_statistics_df["domain"] = "temporal"

        return temporal_statistics_df

    def _fit_tonicdrive_statistics(self):
        """
        Fit tonic drive statistics to tonic drive value fits. Uses gamma distribution.

        Returns
        -------
        tonicdrive_statistics_df : pandas.DataFrame
            DataFrame with tonic drive statistics.
        """

        tonicdrive_array = np.array(
            self.all_data_fits_df.iloc[self.good_data_indices].tonicdrive
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

        td_df = pd.DataFrame.from_dict(
            {
                "tonicdrive": {
                    "shape": shape,
                    "loc": loc,
                    "scale": scale,
                    "distribution": "gamma",
                    "domain": "tonic",
                }
            },
            orient="index",
        )

        return td_df

    def _get_center_surround_sd(self):
        """
        Get center and surround amplitudes so that the spatial RF volume scaling.

        Return
        ------
        mean_center_sd : float
            Mean center standard deviation in millimeters
        mean_surround_sd : float
            Mean surround standard deviation in millimeters
        """
        df = self.all_data_fits_df.iloc[self.good_data_indices]

        # Get mean center and surround RF size from data in millimeters
        mean_center_sd = np.mean(np.sqrt(df.semi_xc * df.semi_yc)) * self.DATA_PIXEL_LEN
        mean_surround_sd = (
            np.mean(np.sqrt((df.sur_ratio**2 * df.semi_xc * df.semi_yc)))
            * self.DATA_PIXEL_LEN
        )
        return mean_center_sd, mean_surround_sd

    def get_statistics(self):
        """
        Statistical receptive field model from data.

        Return
        ------
        statistics_df : pd.DataFrame
            Statistical model parameters for spatial, temporal, and tonic filters
            Indices are the parameter names
            Columns are shape, loc, scale, distribution ('gamma', 'beta'), domain ('spatial', 'temporal', 'tonic')
        good_data_indices
        bad_data_indices
        mean_center_sd : float
            Mean center standard deviation in millimeters
        mean_surround_sd : float
            Mean surround standard deviation in millimeters
        temporal_filters_to_show : dict
            Dictionary with temporal filter parameters and distributions
        spatial_filters_to_show : dict
            Dictionary with spatial filter parameters and distributions
        """
        self.n_cells_data = len(self.all_data_fits_df)
        self.bad_data_indices = np.where((self.all_data_fits_df == 0.0).all(axis=1))[
            0
        ].tolist()
        self.good_data_indices = np.setdiff1d(
            range(self.n_cells_data), self.bad_data_indices
        )

        # Get statistics for spatial filters of good data indices
        spatial_statistics_df = self._fit_spatial_statistics()

        # Get statistics for temporal filters of good data indices
        temporal_statistics_df = self._fit_temporal_statistics()

        # Get statistics for tonic drives of good data indices
        tonicdrive_statistics_df = self._fit_tonicdrive_statistics()

        # get center and surround sd
        mean_center_sd, mean_surround_sd = self._get_center_surround_sd()

        # Collect everything into one big dataframe
        statistics_df = pd.concat(
            [spatial_statistics_df, temporal_statistics_df, tonicdrive_statistics_df],
            axis=0,
        )
        return (
            statistics_df,
            self.good_data_indices,
            self.bad_data_indices,
            mean_center_sd,
            mean_surround_sd,
            self.temporal_filters_to_show,
            self.spatial_filters_to_show,
            self.spatial_statistics_to_show,
            self.temporal_statistics_to_show,
            self.tonic_drives_to_show,
        )


# if __name__ == '__main__':

#     a = Fit('parasol', 'on')
#     a._fit_temporal_filters(show_temporal_filter_response=True, normalize_before_fit=True)
