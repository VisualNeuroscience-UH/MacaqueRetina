""" 
These classes fit spike-triggered average (STA) data from retinal ganglion cells (RGC) to functions expressed as the difference of two 2-dimensional elliptical Gaussians (DoG, Difference of Gaussians).

The derived parameters are used to create artificial RGC mosaics and receptive fields (RFs).

Data courtesy of The Chichilnisky Lab <http://med.stanford.edu/chichilnisky.html>
Data paper: Field GD et al. (2010). Nature 467(7316):673-7.
Only low resolution spatial RF maps are used here.
"""

# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
from scipy.stats import norm, skewnorm, gamma
from scipy.optimize import curve_fit
import pandas as pd

# Viz
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

# Local
from viz.viz_module import Viz
from construct.construct_math_module import RetinaMath

# Builtin
import sys
from pathlib import Path
import pdb


repo_path = Path(__file__).parent
retina_data_path = repo_path / "apricot"
digitized_figures_path = repo_path / "digitized_figures"


class ApricotData:
    """
    Read data from external mat files.
    """

    def __init__(self, gc_type, response_type):
        gc_type = gc_type.lower()
        response_type = response_type.lower()
        self.gc_type = gc_type
        self.response_type = response_type

        # Define filenames
        # Spatial data are read from a separate mat file that have been derived from the originals.
        # Non-spatial data are read from the original data files.
        if gc_type == "parasol" and response_type == "on":
            self.spatial_filename = "Parasol_ON_spatial.mat"
            # self.bad_data_indices=[15, 67, 71, 86, 89]   # Simo's; Manually selected for Chichilnisky apricot (spatial) data
            self.bad_data_indices = [15, 71, 86, 89]

            self.filename_nonspatial = "mosaicGLM_apricot_ONParasol-1-mat.mat"

        elif gc_type == "parasol" and response_type == "off":
            self.spatial_filename = "Parasol_OFF_spatial.mat"
            # self.bad_data_indices = [6, 31, 73]  # Simo's
            self.bad_data_indices = [6, 31, 40, 76]

            self.filename_nonspatial = "mosaicGLM_apricot_OFFParasol-1-mat.mat"

        elif gc_type == "midget" and response_type == "on":
            self.spatial_filename = "Midget_ON_spatial.mat"
            # self.bad_data_indices = [6, 13, 19, 23, 26, 28, 55, 74, 93, 99, 160, 162, 203, 220]  # Simo's
            self.bad_data_indices = [13]
            self.filename_nonspatial = "mosaicGLM_apricot_ONMidget-1-mat.mat"

        elif gc_type == "midget" and response_type == "off":
            self.spatial_filename = "Midget_OFF_spatial.mat"
            # self.bad_data_indices = [4, 5, 13, 23, 39, 43, 50, 52, 55, 58, 71, 72, 86, 88, 94, 100, 104, 119, 137,
            #                     154, 155, 169, 179, 194, 196, 224, 230, 234, 235, 239, 244, 250, 259, 278]  # Simo's
            self.bad_data_indices = [39, 43, 50, 56, 109, 129, 137]
            self.filename_nonspatial = "mosaicGLM_apricot_OFFMidget-1-mat.mat"

        else:
            print("Unknown cell type or response type, aborting")
            sys.exit()

        filepath = retina_data_path / self.filename_nonspatial
        raw_data = sio.loadmat(filepath)  # , squeeze_me=True)
        self.data = raw_data["mosaicGLM"][0]
        self.n_cells = len(self.data)
        self.inverted_data_indices = self._get_inverted_indices()

        self.metadata = {
            "data_microm_per_pix": 60,
            "data_spatialfilter_width": 13,
            "data_spatialfilter_height": 13,
            "data_fps": 30,  # Uncertain - "30 or 120 Hz"
            "data_temporalfilter_samples": 15,
        }

    def _get_inverted_indices(self):
        """
        The rank-1 space and time matrices in the dataset have bumps in an inconsistent way, but the
        outer product always produces a positive deflection first irrespective of on/off polarity.
        This method tells which cell indices you need to flip to get a spatial filter with positive central component.

        :return: np.array
        """

        temporal_filters = self.read_temporal_filter_data(flip_negs=False)
        inverted_data_indices = np.argwhere(temporal_filters[:, 1] < 0).flatten()

        return inverted_data_indices

    def _read_postspike_filter(self):

        postspike_filter = np.array(
            [
                self.data[cellnum][0][0][0][0][0][2][0][0][0]
                for cellnum in range(self.n_cells)
            ]
        )
        return postspike_filter[:, :, 0]

    def _read_space_rk1(self):
        space_rk1 = np.array(
            [
                self.data[cellnum][0][0][0][0][0][3][0][0][2]
                for cellnum in range(self.n_cells)
            ]
        )
        return np.reshape(
            space_rk1, (self.n_cells, 13**2)
        )  # Spatial filter is 13x13 pixels in the Apricot dataset

    # Called from ApricotFits
    def read_spatial_filter_data(self):

        filepath = retina_data_path / self.spatial_filename
        gc_spatial_data = sio.loadmat(filepath, variable_names=["c", "stafit"])
        gc_spatial_data_array = gc_spatial_data["c"]
        initial_center_values = gc_spatial_data["stafit"]

        n_spatial_cells = len(gc_spatial_data_array[0, 0, :])
        n_bad = len(self.bad_data_indices)
        print("\n[%s %s]" % (self.gc_type, self.response_type))
        print(
            "Read %d cells from datafile and then removed %d bad cells (handpicked)"
            % (n_spatial_cells, n_bad)
        )

        return gc_spatial_data_array, initial_center_values, self.bad_data_indices

    def read_tonicdrive(self, remove_bad_data_indices=True):

        tonicdrive = np.array(
            [
                self.data[cellnum][0][0][0][0][0][1][0][0][0][0][0]
                for cellnum in range(self.n_cells)
            ]
        )
        if remove_bad_data_indices is True:
            tonicdrive[self.bad_data_indices] = 0.0

        return tonicdrive

    def read_temporal_filter_data(self, flip_negs=False, normalize=False):

        time_rk1 = np.array(
            [
                self.data[cellnum][0][0][0][0][0][3][0][0][3]
                for cellnum in range(self.n_cells)
            ]
        )
        temporal_filters = time_rk1[:, :, 0]

        # Flip temporal filters so that first deflection is always positive
        for i in range(self.n_cells):
            if temporal_filters[i, 1] < 0 and flip_negs is True:
                temporal_filters[i, :] = temporal_filters[i, :] * (-1)

        if normalize is True:
            assert (
                flip_negs is True
            ), "Normalization does not make sense without flip_negs"
            for i in range(self.n_cells):
                tf = temporal_filters[i, :]
                pos_sum = np.sum(tf[tf > 0])
                temporal_filters[i, :] = tf / pos_sum

        return temporal_filters

    def compute_spatial_filter_sums(self, remove_bad_data_indices=True):
        """
        Computes the pixelwise sum of the values in the rank-1 spatial filters.

        :param remove_bad_data_indices: bool
        :return:
        """
        space_rk1 = self._read_space_rk1()

        filter_sums = np.zeros((self.n_cells, 3))
        for i in range(self.n_cells):
            data_spatial_filter = np.array([space_rk1[i]])
            if i in self.inverted_data_indices:
                data_spatial_filter = (-1) * data_spatial_filter

            filter_sums[i, 0] = np.sum(data_spatial_filter[data_spatial_filter > 0])
            filter_sums[i, 1] = (-1) * np.sum(
                data_spatial_filter[data_spatial_filter < 0]
            )
            filter_sums[i, 2] = np.sum(data_spatial_filter)

        if remove_bad_data_indices is True:
            filter_sums[self.bad_data_indices, :] = 0

        return pd.DataFrame(
            filter_sums,
            columns=[
                "spatial_filtersum_cen",
                "spatial_filtersum_sur",
                "spatial_filtersum_total",
            ],
        )

    def compute_temporal_filter_sums(self, remove_bad_data_indices=True):

        temporal_filters = self.read_temporal_filter_data(
            flip_negs=True
        )  # 1st deflection positive, 2nd negative
        filter_sums = np.zeros((self.n_cells, 3))
        for i in range(self.n_cells):
            filter = temporal_filters[i, :]
            filter_sums[i, 0] = np.sum(filter[filter > 0])
            filter_sums[i, 1] = (-1) * np.sum(filter[filter < 0])
            filter_sums[i, 2] = np.sum(filter)

        if remove_bad_data_indices is True:
            filter_sums[self.bad_data_indices] = 0

        return pd.DataFrame(
            filter_sums,
            columns=[
                "temporal_filtersum_first",
                "temporal_filtersum_second",
                "temporal_filtersum_total",
            ],
        )


class ApricotFits(ApricotData, RetinaMath):
    """
    Methods for deriving spatial receptive field parameters from the apricot dataset (Field_2010)
    """

    def __init__(self, gc_type, response_type, _fit_all=True):

        super().__init__(gc_type, response_type)
        if _fit_all is True:
            self._fit_all()

    def _fit_temporal_filters(self, normalize_before_fit=False):
        """
        Fits each temporal filter to a function consisting of the difference of two
        cascades of lowpass filters. This follows Chichilnisky&Kalmar 2002 JNeurosci.

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
        show_spatial_filter_response=False,
    ):
        """
        Fits a function consisting of the difference of two 2-dimensional elliptical Gaussian functions to
        retinal spike triggered average (STA) data.
        The show_spatial_filter_response parameter will show each DoG fit in order to search for bad cell fits and data.

        :param show_spatial_filter_response: boolean, whether to visualize all fits
        :param surround_model: 0=fit center and surround separately, 1=surround midpoint same as center midpoint
        :param save: string, relative path to a csv file for saving
        :param semi_x_always_major: boolean, whether to rotate Gaussians so that semi_x is always the semimajor/longer axis
        :return:
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
            'x_grid' : x_grid,
            'y_grid' : y_grid,
            'surround_model' : surround_model,
            'pixel_array_shape_x' : pixel_array_shape_x,
            'pixel_array_shape_y' : pixel_array_shape_y,
        }

        # GO THROUGH ALL CELLS
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
        spatial_filters_to_show['data_all_viable_cells'] = data_all_viable_cells

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
        spatial_fits = self._fit_spatial_filters(
            surround_model=1,
            semi_x_always_major=True,
            show_spatial_filter_response=False,
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
        return self.all_fits, self.temporal_filters_to_show, self.spatial_filters_to_show

    def save(self, filepath):
        self.all_fits.to_csv(filepath)


# if __name__ == '__main__':

#     a = ApricotFits('parasol', 'on')
#     a._fit_temporal_filters(show_temporal_filter_response=True, normalize_before_fit=True)
