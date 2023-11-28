""" 
These classes fit spike-triggered average (STA) data from retinal ganglion cells (RGC) to functions 
expressed as the difference of two 2-dimensional elliptical Gaussians (DoG, Difference of Gaussians).

The derived parameters are used to create artificial RGC mosaics and receptive fields (RFs).
"""

# Numerical
import numpy as np
import scipy.optimize as opt
from scipy.optimize import curve_fit, minimize
import scipy.stats as stats
import pandas as pd

# Viz
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
from retina.retina_math_module import RetinaMath
from retina.apricot_data_module import ApricotData

# Builtin
import pdb


class Fit(RetinaMath):
    """
    This class contains methods for fitting elliptical and circularly symmetric
    difference of Gaussians (DoG) models to experimental  (Field_2010) and generated data.
    In addition, it contains methods for fitting experimental impulse response magnitude to
    a function consisting of cascade of two lowpass filters and for adding the tonic drive .
    """

    def __init__(self, context, data_io, project_data):
        # Dependency injection at ProjectManager construction
        self._context = context.set_context(self)
        self._data_io = data_io
        self._project_data = project_data

        self.metadata = self.context.apricot_metadata

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def project_data(self):
        return self._project_data

    def initialize(
        self,
        gc_type,
        response_type,
        fit_type="experimental",  # "experimental", "generated"
        DoG_model="ellipse_fixed",  # "ellipse_independent", "ellipse_fixed", "circular"
        spatial_data=None,
        um_per_pix=None,
        mark_outliers_bad=True,
    ):
        """
        Initialize the fit_dog object.

        Parameters
        ----------
        apricot_metadata : dict
            Metadata for the apricot dataset, including the data folder path.
        gc_type : str
            The type of ganglion cell.
        response_type : str
            The type of response.
        fit_type : str, optional
            The fit type, can be either 'experimental', 'generated' or "concentric_rings". Default is 'experimental'.
        spatial_data : array_like, optional
            The spatial data. Default is None.
        um_per_pix : float, optional
            The new micrometers per pixel value, required when fit_type is 'generated'.
            Default is None.

        Raises
        ------
        AssertionError
            If fit_type is 'generated' and um_per_pix is not provided.
        AssertionError
            If fit_type is 'generated' and spatial_data is not provided.
        """

        assert (
            fit_type == "experimental" or um_per_pix is not None
        ), "If fit_type is 'generated', um_per_pix must be provided"
        assert (
            fit_type == "experimental" or spatial_data is not None
        ), "If fit_type is 'generated', spatial_data must be provided"

        self.gc_type = gc_type
        self.response_type = response_type

        # Spatial DoG units are pixels at this point
        match fit_type:
            case "experimental":
                self.apricot_data = ApricotData(self.metadata, gc_type, response_type)
                self.bad_data_idx = self.apricot_data.manually_picked_bad_data_idx
                self.n_cells = self.apricot_data.n_cells
                (
                    self.all_data_fits_df,
                    self.exp_spat_filt,
                    self.exp_temp_filt,
                    self.good_idx_experimental,
                    self.spat_DoG_fit_params,
                ) = self._fit_experimental_data(DoG_model)
            case "generated":
                (
                    self.all_data_fits_df,
                    self.gen_spat_filt,
                    self.good_idx_generated,
                    self.spat_DoG_fit_params,
                ) = self._fit_DoG_generated_data(
                    DoG_model, spatial_data, mark_outliers_bad=mark_outliers_bad
                )

    def _fit_temporal_filters(self, good_idx_experimental, normalize_before_fit=False):
        """
        Fits each temporal filter to a function consisting of the difference of two cascades of lowpass filters.
        This follows the method described by Chichilnisky & Kalmar in their 2002 JNeurosci paper, using retinal spike
        triggered average (STA) data.

        Parameters
        ----------
        good_idx_experimental : array-like of int
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
            - exp_temp_filt (dict):
                Dictionary of temporal filters to show with viz. The keys are strings of the format 'cell_ix_{cell_idx}',
                where cell_idx is the index of the cell. Each value is a dictionary with the following keys:
                - 'ydata': The temporal filter data for the cell.
                - 'y_fit': The fitted temporal filter data for the cell.
        """

        # shape (n_cells, 15); 15 time points @ 30 Hz (500 ms)
        if normalize_before_fit is True:
            temporal_filters = self.apricot_data.read_temporal_filter_data(
                flip_negs=True, normalize=True
            )
        else:
            temporal_filters = self.apricot_data.read_temporal_filter_data(
                flip_negs=True
            )

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
        exp_temp_filt = {
            "xdata": xdata,
            "xdata_finer": xdata_finer,
            "title": f"{self.gc_type}_{self.response_type}",
        }

        for cell_ix in tqdm(good_idx_experimental, desc="Fitting temporal filters"):
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

            exp_temp_filt[f"cell_ix_{cell_ix}"] = {
                "ydata": ydata,
                "y_fit": self.diff_of_lowpass_filters(xdata_finer, *popt),
            }

        parameters_df = pd.DataFrame(fitted_parameters, columns=parameter_names)
        # Convert taus to milliseconds
        parameters_df["tau1"] = parameters_df["tau1"] * (1 / data_fps) * 1000
        parameters_df["tau2"] = parameters_df["tau2"] * (1 / data_fps) * 1000

        error_df = pd.DataFrame(error_array, columns=["temporalfit_mse"])

        return pd.concat([parameters_df, error_df], axis=1), exp_temp_filt

    def _get_fit_outliers(self, fits_df, bad_idx_for_spatial_fit, columns):
        """
        Finds the outliers of the spatial filters.
        """
        for col in columns:
            out_data = fits_df[col].values
            mean = np.mean(out_data)
            std_dev = np.std(out_data)
            threshold = 3 * std_dev
            mask = np.abs(out_data - mean) > threshold
            idx = np.where(mask)[0]
            bad_idx_for_spatial_fit += idx.tolist()
        bad_idx_for_spatial_fit.sort()

        return bad_idx_for_spatial_fit

    def _fit_spatial_filters(
        self,
        spat_data_array,
        cen_rot_rad_all=None,
        bad_idx_for_spatial_fit=None,
        DoG_model="ellipse_fixed",
        semi_x_always_major=True,
        mark_outliers_bad=True,
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
        DoG_model : str, optional
           ellipse_independent : fit center and surround anisotropic elliptical Gaussians independently,
           ellipse_fixed : fix anisotropic elliptical Gaussian surround midpoint and orientation to center (default),
           circular : fit isotropic circular Gaussians
        semi_x_always_major : bool, optional
            Whether to rotate Gaussians so that semi_x is always the semimajor/longer axis, by default True
        mark_outliers_bad : bool, optional
            Whether to mark outliers (> 3SD from mean) as bad, by default True

        Returns
        -------
        tuple
            A dataframe with spatial parameters and errors for each cell, and a dictionary of spatial filters to show with visualization.
            The dataframe has shape `(n_cells, 13)` and columns:
            ['ampl_c', 'xoc_pix', 'yoc_pix', 'semi_xc_pix', 'semi_yc_pix', 'orient_cen_rad', 'ampl_s', 'xos_pix', 'yos_pix', 'semi_xs_pix',
            'semi_ys_pix', 'orient_sur_rad', 'offset'] if DoG_model=ellipse_independent,
            or shape `(n_cells, 8)` and columns: ['ampl_c', 'xoc_pix', 'yoc_pix', 'semi_xc_pix', 'semi_yc_pix', 'orient_cen_rad',
            'ampl_s', 'relat_sur_diam', 'offset'] if DoG_model=ellipse_fixed.
            ['ampl_c', 'xoc_pix', 'yoc_pix', 'rad_c_pix', 'ampl_s', 'rad_s_pix', 'offset'] if DoG_model=circular.
            The dictionary spat_filt has keys:
                'x_grid': numpy.ndarray of shape `(num_pix_y, num_pix_x)`, X-coordinates of the grid points
                'y_grid': numpy.ndarray of shape `(num_pix_y, num_pix_x)`, Y-coordinates of the grid points
                'DoG_model': str, the type of DoG model used ('ellipse_independent', 'ellipse_fixed' or 'circular')
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

        if DoG_model == "ellipse_independent":
            parameter_names = [
                "ampl_c",
                "xoc_pix",
                "yoc_pix",
                "semi_xc_pix",
                "semi_yc_pix",
                "orient_cen_rad",
                "ampl_s",
                "xos_pix",
                "yos_pix",
                "semi_xs_pix",
                "semi_ys_pix",
                "orient_sur_rad",
                "offset",
            ]
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = "independent"
        elif DoG_model == "ellipse_fixed":
            parameter_names = [
                "ampl_c",
                "xoc_pix",
                "yoc_pix",
                "semi_xc_pix",
                "semi_yc_pix",
                "orient_cen_rad",
                "ampl_s",
                "relat_sur_diam",
                "offset",
            ]
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = "fixed"
        elif DoG_model == "circular":
            parameter_names = [
                "ampl_c",
                "xoc_pix",
                "yoc_pix",
                "rad_c_pix",
                "ampl_s",
                "rad_s_pix",
                "offset",
            ]
            data_all_viable_cells = np.zeros(np.array([n_cells, len(parameter_names)]))
            surround_status = "concentric"

        # Create error & other arrays
        error_all_viable_cells = np.zeros((n_cells, 1))
        dog_filtersum_array = np.zeros((n_cells, 4))
        aspect_ratios = np.zeros(n_cells)

        spat_filt = {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "DoG_model": DoG_model,
            "num_pix_x": num_pix_x,
            "num_pix_y": num_pix_y,
        }

        # Set initial guess for fitting
        rot = 0.0
        if DoG_model == "ellipse_independent":
            # Build initial guess for (ampl_c, xoc_pix, yoc_pix, semi_xc_pix, semi_yc_pix, orient_cen_rad, ampl_s, xos_pix, yos_pix, semi_xs_pix, semi_ys_pix, orient_sur_rad, offset)
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
        elif DoG_model == "ellipse_fixed":
            # Build initial guess for (ampl_c, xoc_pix, yoc_pix, semi_xc_pix, semi_yc_pix, orient_cen_rad, ampl_s, relat_sur_diam, offset)
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

        elif DoG_model == "circular":
            # Build initial guess for (ampl_c, xoc_pix, yoc_pix, rad_c_pix, ampl_s, rad_s_pix, offset)
            p0 = np.array(
                [
                    1,
                    num_pix_y // 2,
                    num_pix_x // 2,
                    num_pix_y // 4,
                    0.1,
                    num_pix_y // 2,
                    0,
                ]
            )
            boundaries = (
                np.array([0.999, -np.inf, -np.inf, 0, 0, 0, 0]),
                np.array([1, np.inf, np.inf, np.inf, 1, np.inf, np.inf]),
            )

        # Invert data arrays with negative sign for fitting and display.
        spat_data_array = self.flip_negative_spatial_rf(spat_data_array)

        # Go through all cells
        print(("Fitting DoG model, surround is {0}".format(surround_status)))
        for cell_idx in tqdm(all_viable_cells, desc="Fitting spatial  filters"):
            this_rf = spat_data_array[cell_idx, :, :]

            try:
                if DoG_model == "ellipse_independent":
                    rot = cen_rot_rad_all[cell_idx]
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
                elif DoG_model == "ellipse_fixed":
                    rot = cen_rot_rad_all[cell_idx]
                    p0[5] = rot
                    popt, pcov = opt.curve_fit(
                        self.DoG2D_fixed_surround,
                        (x_grid, y_grid),
                        this_rf.ravel(),
                        p0=p0,
                        bounds=boundaries,
                    )
                    data_all_viable_cells[cell_idx, :] = popt
                elif DoG_model == "circular":
                    popt, pcov = opt.curve_fit(
                        self.DoG2D_circular,
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

            if DoG_model in ["ellipse_independent", "ellipse_fixed"]:
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
                        data_all_viable_cells[cell_idx, 5] = (
                            rotation + np.pi / 2
                        ) % np.pi

                    # Rotate also the surround if it is defined separately
                    if DoG_model == "ellipse_independent":
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
                # Check position of semi_xc_pix and semi_yc_pix in parameter array
                semi_xc_idx = parameter_names.index("semi_xc_pix")
                semi_yc_idx = parameter_names.index("semi_yc_pix")
                aspect_ratios[cell_idx] = (
                    data_all_viable_cells[cell_idx, semi_xc_idx]
                    / data_all_viable_cells[cell_idx, semi_yc_idx]
                )

            # Compute fitting error
            if DoG_model == "ellipse_independent":
                data_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)
            elif DoG_model == "ellipse_fixed":
                data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
            elif DoG_model == "circular":
                data_fitted = self.DoG2D_circular((x_grid, y_grid), *popt)

            data_fitted = data_fitted.reshape(num_pix_y, num_pix_x)
            fit_deviations = data_fitted - this_rf

            # MSE
            fit_error = np.sum(fit_deviations**2) / np.prod(this_rf.shape)
            error_all_viable_cells[cell_idx, 0] = fit_error

            # Save DoG fit sums
            dog_filtersum_array[cell_idx, 0] = np.sum(data_fitted[data_fitted > 0])
            dog_filtersum_array[cell_idx, 1] = (-1) * np.sum(
                data_fitted[data_fitted < 0]
            )
            dog_filtersum_array[cell_idx, 2] = np.sum(data_fitted)
            dog_filtersum_array[cell_idx, 3] = np.sum(this_rf[this_rf > 0])

            # For visualization
            spat_filt[f"cell_ix_{cell_idx}"] = {
                "spatial_data_array": this_rf,
                "suptitle": f"celltype={self.gc_type}, responsetype={self.response_type}, cell_ix={cell_idx}",
            }

        # Fitted parameters are assigned to both a dictionary and a dataframe
        spat_filt["data_all_viable_cells"] = data_all_viable_cells

        # Add aspect ratios to parameter_names and data_all_viable_cells
        if DoG_model in ["ellipse_independent", "ellipse_fixed"]:
            parameter_names.append("xy_aspect_ratio")
            data_all_viable_cells = np.hstack(
                (data_all_viable_cells, aspect_ratios.reshape(n_cells, 1))
            )

        # Finally build a dataframe of the fitted parameters
        fits_df = pd.DataFrame(data_all_viable_cells, columns=parameter_names)

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

        if mark_outliers_bad == True:
            # identify outliers (> 3SD from mean) and mark them bad
            bad_idx_for_spatial_fit = self._get_fit_outliers(
                fits_df, bad_idx_for_spatial_fit, columns=fits_df.columns
            )

            for i in bad_idx_for_spatial_fit:
                good_mask[i] = 0
        elif mark_outliers_bad == False:
            # We need this check for failed fit in the case when
            # initialize is called with mark_outliers_bad=False
            self.nan_idx = fits_df[fits_df.isna().any(axis=1)].index.values
            if len(self.nan_idx) > 0:
                good_mask = np.where(good_mask)[0] != self.nan_idx
                good_mask = good_mask.astype(int)
                print(f"Removed units {self.nan_idx} with failed fits")

        good_mask_df = pd.DataFrame(good_mask, columns=["good_filter_data"])

        return (
            pd.concat(
                [
                    fits_df,
                    dog_filtersum_df,
                    error_df,
                    good_mask_df,
                ],
                axis=1,
            ),
            spat_filt,
            good_mask,
            parameter_names,
        )

    def _fit_experimental_data(self, DoG_model):
        """
        Fits spatial ellipse, temporal and tonic drive parameters to the experimental data.

        Returns
        -------
        Tuple:
            all_data_fits_df : pandas.DataFrame
                DataFrame containing all the fitted data, including spatial and temporal filter
                parameters, sums of the spatial and temporal filters, and tonic drives.
            spat_filt : numpy.ndarray
                Array of spatial filters in the format required for visualization.
            temp_filt : numpy.ndarray
                Array of temporal filters in the format required for visualization.
            good_idx_experimental : numpy.ndarray
                Array of indices of good data after manually picked bad data and
                failed spatial fit indeces have been removed.
        """

        # Read Apricot data and manually picked bad data indices
        (
            spatial_data,
            cen_rot_rad_all,
        ) = self.apricot_data.read_spatial_filter_data()

        # Check that original Apricot data spatial resolution match metadata given in project_conf_module.
        assert (
            spatial_data.shape[1] == self.metadata["data_spatialfilter_height"]
        ), "Spatial data height does not match metadata"
        assert (
            spatial_data.shape[2] == self.metadata["data_spatialfilter_width"]
        ), "Spatial data width does not match metadata"

        (
            exp_spat_fits_df,
            spat_filt,
            good_mask,
            spat_DoG_fit_params,
        ) = self._fit_spatial_filters(
            spat_data_array=spatial_data,
            cen_rot_rad_all=cen_rot_rad_all,
            bad_idx_for_spatial_fit=self.bad_data_idx,
            DoG_model=DoG_model,
        )

        spatial_filter_sums = self.apricot_data.compute_spatial_filter_sums()

        good_idx_experimental = np.where(good_mask == 1)[0]
        temp_fits_df, temp_filt = self._fit_temporal_filters(good_idx_experimental)

        # Note that this ignores only manually picked bad data indices,
        # if remove_bad_data_idx=True.
        temporal_filter_sums = self.apricot_data.compute_temporal_filter_sums()

        tonicdrives_df = pd.DataFrame(
            self.apricot_data.read_tonicdrive(), columns=["tonicdrive"]
        )

        # Collect everything into one big dataframe
        all_data_fits_df = pd.concat(
            [
                exp_spat_fits_df,
                spatial_filter_sums,
                temp_fits_df,
                temporal_filter_sums,
                tonicdrives_df,
            ],
            axis=1,
        )

        # Set all_data_fits_df rows which are not part of good_idx_experimental to zero
        all_data_fits_df.loc[~all_data_fits_df.index.isin(good_idx_experimental)] = 0.0

        return (
            all_data_fits_df,
            spat_filt,
            temp_filt,
            good_idx_experimental,
            spat_DoG_fit_params,
        )

    def _fit_DoG_generated_data(self, DoG_model, spatial_data, mark_outliers_bad=True):
        """
        Fits spatial DoG parameters to the generated data.

        Parameters:
        -----------
        DoG_model : str
            The type of DoG model used ('ellipse_independent', 'ellipse_fixed' or 'circular').
        spatial_data : numpy.ndarray
            Array of shape (n_samples, height, width) containing the generated spatial data.

        Returns:
        --------
        all_data_fits_df : pandas.DataFrame
            A DataFrame containing the fitted parameters for the spatial filter.
        spat_filt : numpy.ndarray
            Array of shape (n_samples, height, width) containing the visualized spatial filters.
        good_idx_generated : numpy.ndarray
            Array of indices of good data after failed spatial fit indeces have been removed.
        """

        cen_rot_rad_all = np.zeros(spatial_data.shape[0])

        (
            gen_spat_fits_df,
            spat_filt,
            good_mask,
            spat_DoG_fit_params,
        ) = self._fit_spatial_filters(
            spat_data_array=spatial_data,
            cen_rot_rad_all=cen_rot_rad_all,
            bad_idx_for_spatial_fit=[],
            DoG_model=DoG_model,
            mark_outliers_bad=mark_outliers_bad,
        )

        # Collect everything into one big dataframe
        all_data_fits_df = pd.concat([gen_spat_fits_df], axis=1)

        good_idx_generated = np.where(good_mask == 1)[0]
        # Set all_data_fits_df rows which are not part of good_idx_generated to zero
        all_data_fits_df.loc[~all_data_fits_df.index.isin(good_idx_generated)] = 0.0

        return all_data_fits_df, spat_filt, good_idx_generated, spat_DoG_fit_params

    def _fit_spatial_statistics(self, good_data_fit_idx):
        """
        Fits gamma distribution parameters for the 'semi_xc_pix', 'semi_yc_pix', 'xy_aspect_ratio', 'ampl_s',
        and 'relat_sur_diam' RF parameters, and fits vonmises distribution parameters for the 'orient_cen_rad' RF parameter.

        Parameters:
        -----------
        good_data_fit_idx : array_like
            A list of indices indicating the selected good cells.

        Returns:
        --------
        spatial_stat_df : pandas DataFrame
            A DataFrame containing distribution parameters for the RF parameters.
        spat_stat : dict
            A dictionary containing data that can be used to visualize the RF parameters' spatial statistics.
            Includes 'ydata', 'spatial_statistics_dict', and 'model_fit_data'.
        """

        # 1. Define the lists of parameters for gamma and vonmises distributions
        params_list = self.spat_DoG_fit_params
        vonmises_strings = ["orient_cen_rad", "orient_sur_rad"]
        vonmises_params = [param for param in params_list if param in vonmises_strings]
        # Create a list of parameters for gamma distribution for all the other strings in the params_list
        gamma_params = [param for param in params_list if param not in vonmises_strings]

        # 2. Create a dictionary with parameter name: distribution
        param_distribution_dict = {param: "gamma" for param in gamma_params}
        param_distribution_dict.update({param: "vonmises" for param in vonmises_params})

        data_all_cells = np.array(self.all_data_fits_df)
        all_viable_cells = data_all_cells[good_data_fit_idx]

        parameter_names = self.all_data_fits_df.columns.tolist()
        spatial_data_df = pd.DataFrame(data=all_viable_cells, columns=parameter_names)

        n_distributions = len(param_distribution_dict)
        loc = np.zeros([n_distributions])
        scale = np.zeros([n_distributions])
        ydata = np.zeros([len(all_viable_cells), n_distributions])
        x_model_fit = np.zeros([100, n_distributions])
        y_model_fit = np.zeros([100, n_distributions])

        # Create dict for statistical parameters
        spatial_statistics_dict = {}

        # 3. Refactor the remaining code
        for index, (param, dist) in enumerate(param_distribution_dict.items()):
            ydata[:, index] = spatial_data_df[param]

            if dist == "gamma":
                shape, loc[index], scale[index] = stats.gamma.fit(
                    ydata[:, index], loc=0
                )
                x_model_fit[:, index] = np.linspace(
                    stats.gamma.ppf(0.001, shape, loc=loc[index], scale=scale[index]),
                    stats.gamma.ppf(0.999, shape, loc=loc[index], scale=scale[index]),
                    100,
                )
                y_model_fit[:, index] = stats.gamma.pdf(
                    x=x_model_fit[:, index],
                    a=shape,
                    loc=loc[index],
                    scale=scale[index],
                )
                spatial_statistics_dict[param] = {
                    "shape": shape,
                    "loc": loc[index],
                    "scale": scale[index],
                    "distribution": "gamma",
                }

            elif dist == "vonmises":

                def neg_log_likelihood(params, data):
                    kappa, loc = params
                    return -np.sum(
                        stats.vonmises.logpdf(data, kappa, loc=loc, scale=np.pi)
                    )

                guess = [1.0, 0.0]  # kappa, loc
                result = minimize(neg_log_likelihood, guess, args=(ydata[:, index],))
                kappa, loc[index] = result.x
                scale[index] = np.pi  # fixed

                x_model_fit[:, index] = np.linspace(
                    stats.vonmises.ppf(
                        0.001, kappa, loc=loc[index], scale=scale[index]
                    ),
                    stats.vonmises.ppf(
                        0.999, kappa, loc=loc[index], scale=scale[index]
                    ),
                    100,
                )
                y_model_fit[:, index] = stats.vonmises.pdf(
                    x=x_model_fit[:, index],
                    kappa=kappa,
                    loc=loc[index],
                    scale=scale[index],
                )
                spatial_statistics_dict[param] = {
                    "shape": kappa,
                    "loc": loc[index],
                    "scale": scale[index],
                    "distribution": "vonmises",
                }

        spat_stat = {
            "ydata": ydata,
            "spatial_statistics_dict": spatial_statistics_dict,
            "model_fit_data": (x_model_fit, y_model_fit),
        }

        spatial_stat_df = pd.DataFrame.from_dict(
            spatial_statistics_dict, orient="index"
        )
        spatial_stat_df["domain"] = "spatial"

        # Return stats for RF creation
        return spatial_stat_df, spat_stat

    def _fit_temporal_statistics(self, good_data_fit_idx):
        """
        Fit temporal statistics of the temporal filter parameters using the gamma distribution.

        Parameters
        ----------
        good_data_fit_idx : ndarray
            Boolean index array indicating which rows of `self.all_data_fits_df` to use for fitting.

        Returns
        -------
        temporal_exp_stat_df : pd.DataFrame
            A DataFrame containing the temporal statistics of the temporal filter parameters, including the shape, loc,
            and scale parameters of the fitted gamma distribution, as well as the name of the distribution and the domain.

        temp_stat : dict
            A dictionary containing information needed for visualization, including the temporal filter parameters, the
            fitted distribution parameters, the super title of the plot, `self.gc_type`, `self.response_type`, the
            `self.all_data_fits_df` DataFrame, and the `good_data_fit_idx` Boolean index array.
        """

        temporal_filter_parameters = ["n", "p1", "p2", "tau1", "tau2"]
        distrib_params = np.zeros((len(temporal_filter_parameters), 3))

        for i, param_name in enumerate(temporal_filter_parameters):
            param_array = np.array(
                self.all_data_fits_df.iloc[good_data_fit_idx][param_name]
            )
            shape, loc, scale = stats.gamma.fit(param_array)
            distrib_params[i, :] = [shape, loc, scale]

        temp_stat = {
            "temporal_filter_parameters": temporal_filter_parameters,
            "distrib_params": distrib_params,
            "suptitle": self.gc_type + " " + self.response_type,
            "all_data_fits_df": self.all_data_fits_df,
            "good_idx_experimental": good_data_fit_idx,
        }

        temporal_exp_stat_df = pd.DataFrame(
            distrib_params,
            index=temporal_filter_parameters,
            columns=["shape", "loc", "scale"],
        )

        temporal_exp_stat_df["distribution"] = "gamma"
        temporal_exp_stat_df["domain"] = "temporal"

        return temporal_exp_stat_df, temp_stat

    def _fit_tonicdrive_statistics(self, good_data_fit_idx):
        """
        Fits tonic drive statistics to tonic drive value fits using gamma distribution.

        Parameters
        ----------
        good_data_fit_idx : list of int
            List of indices of good data fits to be used for fitting the tonic drive statistics.

        Returns
        -------
        td_df : pandas.DataFrame
            DataFrame with tonic drive statistics, including shape, loc, and scale parameters for the gamma distribution
            as well as the distribution type (gamma) and domain (tonic).
        exp_tonic_dr : dict
            Dictionary containing the following visualization data:
            - xs: an array of 100 x-values to plot the probability density function of the gamma distribution
            - pdf: an array of 100 y-values representing the probability density function of the gamma distribution
            - tonicdrive_array: a numpy array of tonic drive values used for fitting
            - title: a string representing the title of the plot, which includes the gc_type and response_type.
        """

        tonicdrive_array = np.array(
            self.all_data_fits_df.iloc[good_data_fit_idx].tonicdrive
        )
        shape, loc, scale = stats.gamma.fit(tonicdrive_array)

        x_min, x_max = stats.gamma.ppf([0.001, 0.999], a=shape, loc=loc, scale=scale)
        xs = np.linspace(x_min, x_max, 100)
        pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
        title = self.gc_type + " " + self.response_type

        exp_tonic_dr = {
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

        return td_df, exp_tonic_dr

    def _get_center_surround_sd(self, good_data_fit_idx, DoG_model):
        """
        Calculates mean center and surround standard deviations in millimeters for spatial RF volume scaling.

        Parameters
        ----------
        good_data_fit_idx : ndarray or boolean mask
            Indices or boolean mask for selecting valid data fits.

        Returns
        -------
        mean_cen_sd_mm : float
            Mean center standard deviation in millimeters.
        mean_sur_sd_mm : float
            Mean surround standard deviation in millimeters.
        """
        df = self.all_data_fits_df.iloc[good_data_fit_idx]

        data_mm_per_pix = self.metadata["data_microm_per_pix"] / 1000

        if DoG_model == "ellipse_fixed":
            # Get mean center and surround RF size from data in millimeters
            mean_cen_sd_mm = (
                np.mean(np.sqrt(df.semi_xc_pix * df.semi_yc_pix)) * data_mm_per_pix
            )
            mean_sur_sd_mm = (
                np.mean(
                    np.sqrt((df.relat_sur_diam**2 * df.semi_xc_pix * df.semi_yc_pix))
                )
                * data_mm_per_pix
            )
        elif DoG_model == "ellipse_independent":
            # Get mean center and surround RF size from data in millimeters
            mean_cen_sd_mm = (
                np.mean(np.sqrt(df.semi_xc_pix * df.semi_yc_pix)) * data_mm_per_pix
            )
            mean_sur_sd_mm = (
                np.mean(np.sqrt((df.semi_xs_pix * df.semi_ys_pix))) * data_mm_per_pix
            )
        elif DoG_model == "circular":
            # Get mean center and surround RF size from data in millimeters
            mean_cen_sd_mm = np.mean(df.rad_c_pix) * data_mm_per_pix
            mean_sur_sd_mm = np.mean(df.rad_s_pix) * data_mm_per_pix

        return mean_cen_sd_mm, mean_sur_sd_mm

    def get_experimental_fits(self, DoG_model):
        """
        Statistical receptive field model from data.

        Returns
        -------
        exp_stat_df : pd.DataFrame
            DataFrame containing statistical model parameters for spatial, temporal, and tonic filters
            Indices are the parameter names
            Columns are shape, loc, scale, distribution ('gamma', 'vonmises'), domain ('spatial', 'temporal', 'tonic')
        good_data_fit_idx : list
            List of good data indices after spatial fit
        bad_data_fit_idx : list
            List of bad data indices after spatial fit
        exp_cen_radius_mm : float
            Mean center standard deviation in millimeters
        exp_sur_radius_mm : float
            Mean surround standard deviation in millimeters
        exp_temp_filt : dict
            Dictionary with temporal filter parameters and distributions
        exp_spat_filt : dict
            Dictionary with spatial filter parameters and distributions
        exp_spat_stat : dict
            Dictionary with spatial filter statistics for visualization
        exp_temp_stat : dict
            Dictionary with temporal filter statistics for visualization
        exp_tonic_dr : dict
            Dictionary with tonic drive statistics for visualization
        """

        # Get good and bad data indeces from all_data_fits_df. The spatial fit
        # may add bad indices to the data frame
        n_cells_data = len(self.all_data_fits_df)
        bad_data_fit_idx = np.where((self.all_data_fits_df == 0.0).all(axis=1))[
            0
        ].tolist()
        good_data_fit_idx = np.setdiff1d(range(n_cells_data), bad_data_fit_idx)

        # Get statistics for spatial filters of good data indices
        spatial_exp_stat_df, exp_spat_stat = self._fit_spatial_statistics(
            good_data_fit_idx
        )

        # Get statistics for temporal filters of good data indices
        temporal_exp_stat_df, exp_temp_stat = self._fit_temporal_statistics(
            good_data_fit_idx
        )

        # Get statistics for tonic drives of good data indices
        tonicdrive_exp_stat_df, exp_tonic_dr = self._fit_tonicdrive_statistics(
            good_data_fit_idx
        )

        # get center and surround sd
        exp_cen_radius_mm, exp_sur_radius_mm = self._get_center_surround_sd(
            good_data_fit_idx, DoG_model
        )

        # Collect everything into one big dataframe
        exp_stat_df = pd.concat(
            [spatial_exp_stat_df, temporal_exp_stat_df, tonicdrive_exp_stat_df],
            axis=0,
        )

        self.project_data.fit["exp_spat_filt"] = self.exp_spat_filt
        self.project_data.fit["exp_spat_stat"] = exp_spat_stat
        self.project_data.fit["exp_temp_filt"] = self.exp_temp_filt
        self.project_data.fit["exp_temp_stat"] = exp_temp_stat
        self.project_data.fit["exp_tonic_dr"] = exp_tonic_dr

        return (
            exp_stat_df,
            exp_cen_radius_mm,
            exp_sur_radius_mm,
            self.spat_DoG_fit_params,
        )

    def get_generated_spatial_fits(self, DoG_model):
        """
        Generate statistical receptive field model from simulated data.

        Return
        ------
        gen_stat_df : pd.DataFrame
            Statistical model parameters for spatial, temporal, and tonic filters
            Indices are the parameter names
            Columns are shape, loc, scale, distribution ('gamma', 'vonmises'), domain ('spatial', 'temporal', 'tonic')
        gen_mean_cen_sd_mm : float
            Mean center standard deviation in millimeters
        gen_mean_sur_sd_mm : float
            Mean surround standard deviation in millimeters
        gen_spat_filt : dict
            Dictionary with spatial filter parameters and distributions
        gen_spat_stat : dict
            Dictionary with spatial filter statistics and distributions
        """

        good_idx_generated = self.good_idx_generated

        # Get statistics for spatial filters
        gen_spat_stat_df, gen_spat_stat = self._fit_spatial_statistics(
            good_idx_generated
        )

        # get center and surround sd
        gen_mean_cen_sd_mm, gen_mean_sur_sd_mm = self._get_center_surround_sd(
            good_idx_generated, DoG_model
        )

        # Collect everything into one big dataframe
        gen_stat_df = pd.concat(
            [gen_spat_stat_df],
            axis=0,
        )

        # Save data to project_data for vizualization
        self.project_data.fit["gen_spat_filt"] = self.gen_spat_filt
        self.project_data.fit["gen_spat_stat"] = gen_spat_stat

        return (
            gen_stat_df,
            gen_mean_cen_sd_mm,
            gen_mean_sur_sd_mm,
            self.all_data_fits_df,
            good_idx_generated,
        )


# if __name__ == '__main__':

#     a = Fit('parasol', 'on')
#     a._fit_temporal_filters(show_temporal_filter_response=True, normalize_before_fit=True)
