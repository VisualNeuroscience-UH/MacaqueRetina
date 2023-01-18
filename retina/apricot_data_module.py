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
import scipy.io as sio
import pandas as pd

# Local

# Builtin
import pdb


class ApricotData:
    """
    Read data from external mat files.
    """

    def __init__(self, apricot_data_folder, gc_type, response_type):

        self.apricot_data_folder = apricot_data_folder
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
            raise NotImplementedError("Unknown cell type or response type, aborting")

        # Read nonspatial data. Data type is numpy nd array, but it includes a lot of metadata.
        filepath = self.apricot_data_folder / self.filename_nonspatial
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

        Returns
        -------
        inverted_data_indices : np.ndarray
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

    # Called from Fit
    def read_spatial_filter_data(self):

        filepath = self.apricot_data_folder / self.spatial_filename
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

        Parameters
        ----------
        remove_bad_data_indices : bool
            If True, the sums for the bad data indices are set to zero.

        Returns
        -------
        filter_sums : pandas Dataframe from np.ndarray
            Array of shape (n_cells, 3) where each row is the sum of the positive, negative, and
            total values of the spatial filter for that cell.
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
