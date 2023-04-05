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
    Call get_experimental_fits or get_generated_spatial_fits method to return the fits from the instance object
    self.all_data_fits_df and other data to visualize the fits.
    """

    def __init__(
        self,
        apricot_data_folder,
        gc_type,
        response_type,
        spatial_data=None,
        fit_type="experimental",
    ):

        # Assigns to self the following attributes, which are necessary in this module:
        # gc_type, response_type, DATA_PIXEL_LEN, manually_picked_bad_data_idx, n_cells, metadata
        super().__init__(apricot_data_folder, gc_type, response_type)

        match fit_type:
            case "experimental":
                # Fit spatial and temporal filters and tonic drive values to experimental data.
                (
                    self.all_data_fits_df,
                    self.exp_spat_filt_to_viz,
                    self.exp_temp_filt_to_viz,
                    self.apricot_data_resolution_hw,
                    self.good_idx,
                ) = self._fit_experimental_data()
            case "generated":
                # Fit only spatial filters to generated data.
                # All filters are accepted, so no need to return good_idx.
                (
                    self.all_data_fits_df,
                    self.gen_spat_filt_to_viz,
                ) = self._fit_generated_data(spatial_data)

    def _fit_temporal_filters(self, good_idx, normalize_before_fit=False):
        """
        Fits each temporal filter to a function consisting of the difference of two cascades of lowpass filters.
        This follows the method described by Chichilnisky & Kalmar in their 2002 JNeurosci paper, using retinal spike
        triggered average (STA) data.

        Parameters
        ----------
        good_idx : array-like of int
            Indices of the cells to fit the temporal filters for.
        normalize_before_fit : bool, default False
            If True, normalize each temporal filter before fitting.
            If False, fit the raw temporal filters.

        Returns
        -------
        tuple
            A tuple containing:
            - fitted_parameters (pandas.DataFrame):
                DataFrame of shape (n_cells, 5) containing the fitted parameters for each cell. The columns are:
                - 'n': Order of the filters.
                - 'p1': Normalization factor for the first filter.
                - 'p2': Normalization factor for the second filter.
                - 'tau1': Time constant of the first filter in milliseconds.
                - 'tau2': Time constant of the second filter in milliseconds.
            - exp_temp_filt_to_viz (dict):
                Dictionary of temporal filters to show with viz. The keys are strings of the format 'cell_ix_{cell_idx}',
                where cell_idx is the index of the cell. Each value is a dictionary with the following keys:
                - 'ydata': The temporal filter data for the cell.
                - 'y_fit': The fitted temporal filter data for the cell.
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
        exp_temp_filt_to_viz = {
            "xdata": xdata,
            "xdata_finer": xdata_finer,
            "title": f"{self.gc_type}_{self.response_type}",
        }

        for cell_ix in tqdm(good_idx, desc="Fitting temporal filters"):
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

            exp_temp_filt_to_viz[f"cell_ix_{cell_ix}"] = {
                "ydata": ydata,
                "y_fit": self.diff_of_lowpass_filters(xdata_finer, *popt),
            }

        parameters_df = pd.DataFrame(fitted_parameters, columns=parameter_names)
        # Convert taus to milliseconds
        parameters_df["tau1"] = parameters_df["tau1"] * (1 / data_fps) * 1000
        parameters_df["tau2"] = parameters_df["tau2"] * (1 / data_fps) * 1000

        error_df = pd.DataFrame(error_array, columns=["temporalfit_mse"])

        return pd.concat([parameters_df, error_df], axis=1), exp_temp_filt_to_viz

    def _fit_spatial_filters(
        self,
        spat_data_array,
        cen_rot_rad_all=None,
        bad_idx_for_spatial_fit=None,
        surround_model=1,
        semi_x_always_major=True,
    ):
        """
        Fit difference of Gaussians (DoG) model spatial filters using retinal spike triggered average (STA) data.

        Parameters
        ----------
        spat_data_array : numpy.ndarray
            Array of shape `(n_cells, num_pix_y, num_pix_x)` containing the spatial data for each cell to fit.
        cen_rot_rad_all : numpy.ndarray or None, optional
            Array of shape `(n_cells,)` containing the rotation angle for each cell. If None, rotation is set to 0, by default None
        bad_idx_for_spatial_fit : numpy.ndarray or None, optional
            Indices of cells to exclude from fitting, by default None
        surround_model : int, optional
            Whether to fit center and surround separately (0) or to fix surround midpoint to be the same as center midpoint (1), by default 1
        semi_x_always_major : bool, optional
            Whether to rotate Gaussians so that semi_x is always the semimajor/longer axis, by default True

        Returns
        -------
        tuple
            A dataframe with spatial parameters and errors for each cell, and a dictionary of spatial filters to show with visualization.
            The dataframe has shape `(n_cells, 8)` and columns: ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center',
            'amplitudes', 'sur_ratio', 'offset'] if surround_model=1, or shape `(n_cells, 13)` and columns:
            ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes', 'xos', 'yos', 'semi_xs',
            'semi_ys', 'orientation_surround', 'offset'] if surround_model=0.
            The dictionary spat_filt_to_viz has keys:
                'x_grid': numpy.ndarray of shape `(num_pix_y, num_pix_x)`, X-coordinates of the grid points
                'y_grid': numpy.ndarray of shape `(num_pix_y, num_pix_x)`, Y-coordinates of the grid points
                'surround_model': int, the type of surround model used (0 or 1)
                'num_pix_x': int, the number of pixels in the x-dimension
                'num_pix_y': int, the number of pixels in the y-dimension
                'filters': numpy.ndarray of shape `(n_cells, num_pix_y, num_pix_x)`, containing the fitted spatial filters for each cell
            good_mask: numpy.ndarray of shape `(n_cells,)`, containing a boolean mask of cells that were successfully fitted

        Raises
        ------
        ValueError
            If the shape of spat_data_array is not `(n_cells, num_pix_y, num_pix_x)`.
        """

        n_cells = int(spat_data_array.shape[0])
        num_pix_y = spat_data_array.shape[1]  # Check indices: x horizontal, y vertical
        num_pix_x = spat_data_array.shape[2]

        # Make fit to all cells
        # Note: input coming from matlab, thus indexing starts from 1
        x_position_indices = np.linspace(1, num_pix_x, num_pix_x)
        y_position_indices = np.linspace(1, num_pix_y, num_pix_y)
        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)

        all_viable_cells = np.setdiff1d(np.arange(n_cells), bad_idx_for_spatial_fit)

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

        spat_filt_to_viz = {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "surround_model": surround_model,
            "num_pix_x": num_pix_x,
            "num_pix_y": num_pix_y,
        }

        # Set initial guess for fitting
        rot = 0.0
        if surround_model == 1:
            # Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset)
            p0 = np.array(
                [
                    1,
                    num_pix_y // 2,
                    num_pix_x // 2,
                    num_pix_y // 4,
                    num_pix_x // 4,
                    rot,
                    0.1,
                    3,
                    0,
                ]
            )
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
                    num_pix_y // 2,
                    num_pix_x // 2,
                    num_pix_y // 4,
                    num_pix_x // 4,
                    rot,
                    0.1,
                    num_pix_y // 2,
                    num_pix_x // 2,
                    (num_pix_y // 4) * 3,
                    (num_pix_x // 4) * 3,
                    rot,
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

        # Go through all cells
        print(("Fitting DoG model, surround is {0}".format(surround_status)))
        for cell_idx in tqdm(all_viable_cells, desc="Fitting spatial  filters"):
            this_rf = spat_data_array[cell_idx, :, :]

            rot = cen_rot_rad_all[cell_idx]

            # Invert data arrays with negative sign for fitting and display.
            # Fitting assumes that center peak is above mean.
            if this_rf.ravel()[np.argmax(np.abs(this_rf))] < 0:
                this_rf = this_rf * -1

            try:
                if surround_model == 1:
                    p0[5] = rot
                    popt, pcov = opt.curve_fit(
                        self.DoG2D_fixed_surround,
                        (x_grid, y_grid),
                        this_rf.ravel(),
                        p0=p0,
                        bounds=boundaries,
                    )
                    data_all_viable_cells[cell_idx, :] = popt
                else:
                    p0[5] = rot
                    p0[11] = rot
                    popt, pcov = opt.curve_fit(
                        self.DoG2D_independent_surround,
                        (x_grid, y_grid),
                        this_rf.ravel(),
                        p0=p0,
                        bounds=boundaries,
                    )
                    data_all_viable_cells[cell_idx, :] = popt
            except:
                print(("Fitting failed for cell {0}".format(str(cell_idx))))
                data_all_viable_cells[cell_idx, :] = np.nan
                bad_idx_for_spatial_fit.append(cell_idx)
                continue

            # Set rotation angle between 0 and pi
            data_all_viable_cells[cell_idx, 5] = (
                data_all_viable_cells[cell_idx, 5] % np.pi
            )

            # Rotate fit so that semi_x is always the semimajor axis (longer radius)
            if semi_x_always_major is True:
                if (
                    data_all_viable_cells[cell_idx, 3]
                    < data_all_viable_cells[cell_idx, 4]
                ):
                    sd_x = data_all_viable_cells[cell_idx, 3]
                    sd_y = data_all_viable_cells[cell_idx, 4]
                    rotation = data_all_viable_cells[cell_idx, 5]

                    data_all_viable_cells[cell_idx, 3] = sd_y
                    data_all_viable_cells[cell_idx, 4] = sd_x
                    data_all_viable_cells[cell_idx, 5] = (rotation + np.pi / 2) % np.pi

                # Rotate also the surround if it is defined separately
                if surround_model == 0:
                    if (
                        data_all_viable_cells[cell_idx, 9]
                        < data_all_viable_cells[cell_idx, 10]
                    ):
                        sd_x_sur = data_all_viable_cells[cell_idx, 9]
                        sd_y_sur = data_all_viable_cells[cell_idx, 10]
                        rotation = data_all_viable_cells[cell_idx, 11]

                        data_all_viable_cells[cell_idx, 9] = sd_y_sur
                        data_all_viable_cells[cell_idx, 10] = sd_x_sur
                        data_all_viable_cells[cell_idx, 11] = (
                            rotation + np.pi / 2
                        ) % np.pi

            # Set rotation angle between -pi/2 and pi/2 (otherwise hist bimodal)
            rotation = data_all_viable_cells[cell_idx, 5]
            if rotation > np.pi / 2:
                data_all_viable_cells[cell_idx, 5] = rotation - np.pi
            else:
                data_all_viable_cells[cell_idx, 5] = rotation

            # Compute fitting error
            if surround_model == 1:
                data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
            else:
                data_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)

            data_fitted = data_fitted.reshape(num_pix_y, num_pix_x)
            fit_deviations = data_fitted - this_rf
            data_mean = np.mean(this_rf)
            # Normalized mean square error
            # Defn per https://se.mathworks.com/help/ident/ref/goodnessoffit.html without 1 - ...
            # 0 = perfect fit, infty = bad fit

            # MSE
            fit_error = np.sum(fit_deviations**2) / (13 * 13)
            error_all_viable_cells[cell_idx, 0] = fit_error

            # Save DoG fit sums
            dog_filtersum_array[cell_idx, 0] = np.sum(data_fitted[data_fitted > 0])
            dog_filtersum_array[cell_idx, 1] = (-1) * np.sum(
                data_fitted[data_fitted < 0]
            )
            dog_filtersum_array[cell_idx, 2] = np.sum(data_fitted)
            dog_filtersum_array[cell_idx, 3] = np.sum(this_rf[this_rf > 0])

            # For visualization
            spat_filt_to_viz[f"cell_ix_{cell_idx}"] = {
                "spatial_data_array": this_rf,
                "suptitle": f"celltype={self.gc_type}, responsetype={self.response_type}, cell_ix={cell_idx}",
            }

        spat_filt_to_viz["data_all_viable_cells"] = data_all_viable_cells

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
        good_mask = np.ones(len(data_all_viable_cells))

        for i in bad_idx_for_spatial_fit:
            good_mask[i] = 0
        good_mask_df = pd.DataFrame(good_mask, columns=["good_filter_data"])

        return (
            pd.concat(
                [
                    fits_df,
                    aspect_ratios_df,
                    dog_filtersum_df,
                    error_df,
                    good_mask_df,
                ],
                axis=1,
            ),
            spat_filt_to_viz,
            good_mask,
        )

    def _fit_experimental_data(self):
        """
        Fits spatial, temporal and tonic drive parameters to the experimental data.

        Returns
        -------
        Tuple:
            all_data_fits_df : pandas.DataFrame
                DataFrame containing all the fitted data, including spatial and temporal filter
                parameters, sums of the spatial and temporal filters, and tonic drives.
            spat_filt_to_viz : numpy.ndarray
                Array of spatial filters in the format required for visualization.
            temp_filt_to_viz : numpy.ndarray
                Array of temporal filters in the format required for visualization.
            apricot_data_resolution_hw : Tuple[int, int]
                Tuple containing the height and width of the original Apricot data.
            good_idx : numpy.ndarray
                Array of indices of good data after manually picked bad data and
                failed spatial fit indeces have been removed.
        """

        # Read Apricot data and manually picked bad data indices
        (
            spatial_data,
            cen_rot_rad_all,
        ) = self.read_spatial_filter_data()

        # Get original Apricot data spatial resolution
        apricot_data_resolution_hw = spatial_data.shape[1:3]

        spatial_fits, spat_filt_to_viz, good_mask = self._fit_spatial_filters(
            spat_data_array=spatial_data,
            cen_rot_rad_all=cen_rot_rad_all,
            bad_idx_for_spatial_fit=self.manually_picked_bad_data_idx,
            surround_model=1,
            semi_x_always_major=True,
        )

        good_idx = np.where(good_mask == 1)[0]

        # Note that this ignores only manually picked bad data indices,
        # if remove_bad_data_idx=True.
        spatial_filter_sums = self.compute_spatial_filter_sums()

        temporal_fits, temp_filt_to_viz = self._fit_temporal_filters(good_idx)

        # Note that this ignores only manually picked bad data indices,
        # if remove_bad_data_idx=True.
        temporal_filter_sums = self.compute_temporal_filter_sums()

        tonicdrives = pd.DataFrame(self.read_tonicdrive(), columns=["tonicdrive"])

        # Collect everything into one big dataframe
        all_data_fits_df = pd.concat(
            [
                spatial_fits,
                spatial_filter_sums,
                temporal_fits,
                temporal_filter_sums,
                tonicdrives,
            ],
            axis=1,
        )

        return (
            all_data_fits_df,
            spat_filt_to_viz,
            temp_filt_to_viz,
            apricot_data_resolution_hw,
            good_idx,
        )

    def _fit_generated_data(self, spatial_data):
        """
        Fits spatial, temporal, and tonic drive parameters to the generated data.

        Parameters:
        -----------
        spatial_data : numpy.ndarray
            Array of shape (n_samples, height, width) containing the generated spatial data.

        Returns:
        --------
        all_data_fits_df : pandas.DataFrame
            A DataFrame containing the fitted parameters for the spatial filter.
        spat_filt_to_viz : numpy.ndarray
            Array of shape (n_samples, height, width) containing the visualized spatial filters.
        """

        cen_rot_rad_all = np.zeros(spatial_data.shape[0])

        spatial_fits, spat_filt_to_viz, _ = self._fit_spatial_filters(
            spat_data_array=spatial_data,
            cen_rot_rad_all=cen_rot_rad_all,
            bad_idx_for_spatial_fit=[],
            surround_model=1,
            semi_x_always_major=True,
        )

        # Collect everything into one big dataframe
        all_data_fits_df = pd.concat([spatial_fits], axis=1)

        return all_data_fits_df, spat_filt_to_viz

    def _fit_spatial_statistics(self, good_idx):
        """
        Fits gamma distribution parameters for the 'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes',
        and 'sur_ratio' RF parameters, and fits beta distribution parameters for the 'orientation_center' RF parameter.

        Parameters:
        -----------
        good_idx : array_like
            A list of indices indicating the selected good cells.

        Returns:
        --------
        spatial_stat_df : pandas DataFrame
            A DataFrame containing gamma distribution parameters for the RF parameters 'semi_xc', 'semi_yc',
            'xy_aspect_ratio', 'amplitudes', and 'sur_ratio', and beta distribution parameters for the 'orientation_center'.
        spat_stat_to_viz : dict
            A dictionary containing data that can be used to visualize the RF parameters' spatial statistics.
            Includes 'ydata', 'spatial_statistics_dict', and 'model_fit_data'.
        """

        data_all_cells = np.array(self.all_data_fits_df)
        all_viable_cells = data_all_cells[good_idx]

        parameter_names = self.all_data_fits_df.columns.tolist()
        spatial_data_df = pd.DataFrame(data=all_viable_cells, columns=parameter_names)

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

        spat_stat_to_viz = {
            "ydata": ydata,
            "spatial_statistics_dict": spatial_statistics_dict,
            "model_fit_data": (x_model_fit, y_model_fit),
        }

        spatial_stat_df = pd.DataFrame.from_dict(
            spatial_statistics_dict, orient="index"
        )
        spatial_stat_df["domain"] = "spatial"

        # Return stats for RF creation
        return spatial_stat_df, spat_stat_to_viz

    def _fit_temporal_statistics(self, good_idx):
        """
        Fit temporal statistics of the temporal filter parameters using the gamma distribution.

        Parameters
        ----------
        good_idx : ndarray
            Boolean index array indicating which rows of `self.all_data_fits_df` to use for fitting.

        Returns
        -------
        temporal_exp_stat_df : pd.DataFrame
            A DataFrame containing the temporal statistics of the temporal filter parameters, including the shape, loc,
            and scale parameters of the fitted gamma distribution, as well as the name of the distribution and the domain.

        temp_stat_to_viz : dict
            A dictionary containing information needed for visualization, including the temporal filter parameters, the
            fitted distribution parameters, the super title of the plot, `self.gc_type`, `self.response_type`, the
            `self.all_data_fits_df` DataFrame, and the `good_idx` Boolean index array.
        """

        temporal_filter_parameters = ["n", "p1", "p2", "tau1", "tau2"]
        distrib_params = np.zeros((len(temporal_filter_parameters), 3))

        for i, param_name in enumerate(temporal_filter_parameters):
            param_array = np.array(self.all_data_fits_df.iloc[good_idx][param_name])
            shape, loc, scale = stats.gamma.fit(param_array)
            distrib_params[i, :] = [shape, loc, scale]

        temp_stat_to_viz = {
            "temporal_filter_parameters": temporal_filter_parameters,
            "distrib_params": distrib_params,
            "suptitle": self.gc_type + " " + self.response_type,
            "all_data_fits_df": self.all_data_fits_df,
            "good_idx": good_idx,
        }

        temporal_exp_stat_df = pd.DataFrame(
            distrib_params,
            index=temporal_filter_parameters,
            columns=["shape", "loc", "scale"],
        )

        temporal_exp_stat_df["distribution"] = "gamma"
        temporal_exp_stat_df["domain"] = "temporal"

        return temporal_exp_stat_df, temp_stat_to_viz

    def _fit_tonicdrive_statistics(self, good_idx):
        """
        Fits tonic drive statistics to tonic drive value fits using gamma distribution.

        Parameters
        ----------
        good_idx : list of int
            List of indices of good data fits to be used for fitting the tonic drive statistics.

        Returns
        -------
        td_df : pandas.DataFrame
            DataFrame with tonic drive statistics, including shape, loc, and scale parameters for the gamma distribution
            as well as the distribution type (gamma) and domain (tonic).
        exp_tonic_dr_to_viz : dict
            Dictionary containing the following visualization data:
            - xs: an array of 100 x-values to plot the probability density function of the gamma distribution
            - pdf: an array of 100 y-values representing the probability density function of the gamma distribution
            - tonicdrive_array: a numpy array of tonic drive values used for fitting
            - title: a string representing the title of the plot, which includes the gc_type and response_type.
        """

        tonicdrive_array = np.array(self.all_data_fits_df.iloc[good_idx].tonicdrive)
        shape, loc, scale = stats.gamma.fit(tonicdrive_array)

        x_min, x_max = stats.gamma.ppf([0.001, 0.999], a=shape, loc=loc, scale=scale)
        xs = np.linspace(x_min, x_max, 100)
        pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
        title = self.gc_type + " " + self.response_type

        exp_tonic_dr_to_viz = {
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

        return td_df, exp_tonic_dr_to_viz

    def _get_center_surround_sd(self, good_idx):
        """
        Calculates mean center and surround standard deviations in millimeters for spatial RF volume scaling.

        Parameters
        ----------
        good_idx : ndarray or boolean mask
            Indices or boolean mask for selecting valid data fits.

        Returns
        -------
        mean_center_sd : float
            Mean center standard deviation in millimeters.
        mean_surround_sd : float
            Mean surround standard deviation in millimeters.
        """
        df = self.all_data_fits_df.iloc[good_idx]

        # Get mean center and surround RF size from data in millimeters
        mean_center_sd = np.mean(np.sqrt(df.semi_xc * df.semi_yc)) * self.DATA_PIXEL_LEN
        mean_surround_sd = (
            np.mean(np.sqrt((df.sur_ratio**2 * df.semi_xc * df.semi_yc)))
            * self.DATA_PIXEL_LEN
        )
        return mean_center_sd, mean_surround_sd

    def get_experimental_fits(self):
        """
        Statistical receptive field model from data.

        Returns
        -------
        exp_stat_df : pd.DataFrame
            DataFrame containing statistical model parameters for spatial, temporal, and tonic filters
            Indices are the parameter names
            Columns are shape, loc, scale, distribution ('gamma', 'beta'), domain ('spatial', 'temporal', 'tonic')
        good_data_fit_idx : list
            List of good data indices after spatial fit
        bad_data_fit_idx : list
            List of bad data indices after spatial fit
        exp_spat_cen_sd : float
            Mean center standard deviation in millimeters
        exp_spat_sur_sd : float
            Mean surround standard deviation in millimeters
        exp_temp_filt_to_viz : dict
            Dictionary with temporal filter parameters and distributions
        exp_spat_filt_to_viz : dict
            Dictionary with spatial filter parameters and distributions
        exp_spat_stat_to_viz : dict
            Dictionary with spatial filter statistics for visualization
        exp_temp_stat_to_viz : dict
            Dictionary with temporal filter statistics for visualization
        exp_tonic_dr_to_viz : dict
            Dictionary with tonic drive statistics for visualization
        apricot_data_resolution_hw : tuple
            Tuple containing the height and width of the Apricot dataset in pixels
        """

        # Get good and bad data indeces from all_data_fits_df. The spatial fit
        # may add bad indices to the data frame
        n_cells_data = len(self.all_data_fits_df)
        bad_data_fit_idx = np.where((self.all_data_fits_df == 0.0).all(axis=1))[
            0
        ].tolist()
        good_data_fit_idx = np.setdiff1d(range(n_cells_data), bad_data_fit_idx)

        # Get statistics for spatial filters of good data indices
        spatial_exp_stat_df, exp_spat_stat_to_viz = self._fit_spatial_statistics(
            good_data_fit_idx
        )

        # Get statistics for temporal filters of good data indices
        temporal_exp_stat_df, exp_temp_stat_to_viz = self._fit_temporal_statistics(
            good_data_fit_idx
        )

        # Get statistics for tonic drives of good data indices
        tonicdrive_exp_stat_df, exp_tonic_dr_to_viz = self._fit_tonicdrive_statistics(
            good_data_fit_idx
        )

        # get center and surround sd
        exp_spat_cen_sd, exp_spat_sur_sd = self._get_center_surround_sd(
            good_data_fit_idx
        )

        # Collect everything into one big dataframe
        exp_stat_df = pd.concat(
            [spatial_exp_stat_df, temporal_exp_stat_df, tonicdrive_exp_stat_df],
            axis=0,
        )
        return (
            exp_stat_df,
            good_data_fit_idx,
            bad_data_fit_idx,
            exp_spat_cen_sd,
            exp_spat_sur_sd,
            self.exp_temp_filt_to_viz,
            self.exp_spat_filt_to_viz,
            exp_spat_stat_to_viz,
            exp_temp_stat_to_viz,
            exp_tonic_dr_to_viz,
            self.apricot_data_resolution_hw,
        )

    def get_generated_spatial_fits(self):
        """
        Generate statistical receptive field model from simulated data.

        Return
        ------
        gen_stat_df : pd.DataFrame
            Statistical model parameters for spatial, temporal, and tonic filters
            Indices are the parameter names
            Columns are shape, loc, scale, distribution ('gamma', 'beta'), domain ('spatial', 'temporal', 'tonic')
        gen_mean_cen_sd : float
            Mean center standard deviation in millimeters
        gen_mean_sur_sd : float
            Mean surround standard deviation in millimeters
        gen_spat_filt_to_viz : dict
            Dictionary with spatial filter parameters and distributions
        gen_spat_stat_to_viz : dict
            Dictionary with spatial filter statistics and distributions
        """

        # For generated data, all data indeces are good
        good_idx = range(len(self.all_data_fits_df))

        # Get statistics for spatial filters
        gen_spat_stat_df, gen_spat_stat_to_viz = self._fit_spatial_statistics(good_idx)

        # get center and surround sd
        gen_mean_cen_sd, gen_mean_sur_sd = self._get_center_surround_sd(good_idx)

        # Collect everything into one big dataframe
        gen_stat_df = pd.concat(
            [gen_spat_stat_df],
            axis=0,
        )

        return (
            gen_stat_df,
            gen_mean_cen_sd,
            gen_mean_sur_sd,
            self.gen_spat_filt_to_viz,
            gen_spat_stat_to_viz,
        )


# if __name__ == '__main__':

#     a = Fit('parasol', 'on')
#     a._fit_temporal_filters(show_temporal_filter_response=True, normalize_before_fit=True)
