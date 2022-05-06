# Analysis
import numpy as np
import pandas as pd
from scipy.signal import resample, coherence, csd, correlate, welch

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Statistics
from statsmodels.tsa.stattools import grangercausalitytests as gc_test
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.diagnostic import kstest_normal, het_breuschpagan
import pyinform as pin

# Machine learning
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Computational neuroscience
import brian2.units as b2u

# This package
from analysis.statistics_module import Classifierbinomial
from analysis.analysis_base_module import AnalysisBase

# Builtin
from pathlib import Path
import os
import sys
import math
import time
import logging
import pdb
import warnings
import timeit

warnings.filterwarnings("ignore")

"""
Module for analysis of simulated electrophysiology data.

Developed by Simo Vanni 2020-2022
"""


class Analysis(AnalysisBase):

    # self.context. attributes
    _properties_list = [
        "path",
        "output_folder",
        "input_filename",
        "midpoint",
        "parameter",
        "NG_name",
        "t_idx_start",
        "t_idx_end",
    ]

    def __init__(self, context, data_io, cxparser, stat_tests, **kwargs) -> None:

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        self._cxparser = cxparser
        self._stat_tests = stat_tests

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def cxparser(self):
        return self._cxparser

    @property
    def stat_tests(self):
        return self._stat_tests

    # General analysis tools

    def scaler(self, data, scale_type="standard", feature_range=[-1, 1]):
        # From sklearn.preprocessing
        # Data is assumed to be [samples or time, features or regressors]
        if scale_type == "standard":
            # Standardize data by removing the mean and scaling to unit variance
            data_scaled = scale(data)
        elif scale_type == "minmax":
            # Transform features by scaling each feature to a given range.
            # If you put in matrix, note that scales each column (feature) independently
            if data.ndim > 1:
                minmaxscaler = MinMaxScaler(feature_range=feature_range)
                minmaxscaler.fit(data)
                data_scaled = minmaxscaler.transform(data)
            elif data.ndim == 1:  # Manual implementation for 1-D data
                feat_min, feat_max = feature_range
                data_std = (data - data.min()) / (data.max() - data.min())
                data_scaled = data_std * (feat_max - feat_min) + feat_min
        return data_scaled

    def _downsample(self, data, downsampling_factor=1, axis=0):
        # Note, using finite impulse response filter type
        # downsampled_data = decimate(data, downsampling_factor, axis=axis, ftype='fir',)
        N_time_points = data.shape[0]
        num = N_time_points // downsampling_factor
        downsampled_data = resample(data, num)

        return downsampled_data

    def _correlation_lags(self, in1_len, in2_len, mode="full"):
        # Copied from scipy.signal correlation_lags, mode full or same

        if mode == "full":
            lags = np.arange(-in2_len + 1, in1_len)
        elif mode == "same":
            # the output is the same size as `in1`, centered
            # with respect to the 'full' output.
            # calculate the full output
            lags = np.arange(-in2_len + 1, in1_len)
            # determine the midpoint in the full output
            mid = lags.size // 2
            # determine lag_bound to be used with respect
            # to the midpoint
            lag_bound = in1_len // 2
            # calculate lag ranges for even and odd scenarios
            if in1_len % 2 == 0:
                lags = lags[(mid - lag_bound) : (mid + lag_bound)]
            else:
                lags = lags[(mid - lag_bound) : (mid + lag_bound) + 1]

        return lags

    def get_coherence_of_two_signals(
        self, x, y, samp_freq=1.0, nperseg=64, high_cutoff=100
    ):
        """
        Coherence, cross spectral density, cross correlation

        From scipy-signal.coherence documentation: Cxy = abs(Pxy)**2/(Pxx*Pyy),
        where Pxx and Pyy are power spectral density estimates of X and Y, and
        Pxy is the cross spectral density estimate of X and Y.
        """
        x_scaled = self.scaler(x)
        y_scaled = self.scaler(y)

        # Coherence
        f, Cxy = coherence(
            x_scaled, y_scaled, fs=samp_freq, window="hann", nperseg=nperseg
        )

        # Cross spectral density
        f, Pxy = csd(
            x_scaled, y_scaled, fs=samp_freq, nperseg=nperseg, scaling="density"
        )

        # Cross correlation
        corr = correlate(y_scaled, x_scaled, mode="full")
        lags = self._correlation_lags(len(x_scaled), len(y_scaled))
        corr /= np.max(corr)

        # Power spectrum of input and output signals
        f, Pwelch_spec_x = welch(
            x_scaled, fs=samp_freq, nperseg=nperseg, scaling="density"
        )
        f, Pwelch_spec_y = welch(
            y_scaled, fs=samp_freq, nperseg=nperseg, scaling="density"
        )

        if high_cutoff is not None:
            indexes = f < high_cutoff
            f = f[indexes]
            Cxy = Cxy[indexes]
            Pxy = Pxy[indexes]
            Pwelch_spec_x = Pwelch_spec_x[indexes]
            Pwelch_spec_y = Pwelch_spec_y[indexes]

        coherence_sum = np.sum(Cxy)

        return (
            f,
            Cxy,
            Pwelch_spec_x,
            Pwelch_spec_y,
            Pxy,
            lags,
            corr,
            coherence_sum,
            x_scaled,
            y_scaled,
        )

    def pivot_to_2d_dataframe(
        self,
        long_df,
        index_column_name=None,
        column_column_name=None,
        value_column_name=None,
    ):

        assert all(
            [index_column_name, column_column_name, value_column_name]
        ), "Missing some column names, aborting..."
        wide_2d_df = long_df.pivot(
            index=index_column_name,
            columns=column_column_name,
            values=value_column_name,
        )

        return wide_2d_df

    def get_df_extremes(self, data_df, value_column_name, two_dim=False):

        if two_dim:
            # Get 2 dims for viz
            index_column_name = "Dimension-1 Value"
            column_column_name = "Dimension-2 Value"
            x_label = data_df["Dimension-2 Parameter"][0]
            y_label = data_df["Dimension-1 Parameter"][0]

            df_2d = self.pivot_to_2d_dataframe(
                data_df,
                index_column_name=index_column_name,
                column_column_name=column_column_name,
                value_column_name=value_column_name,
            )
            data_nd_array = df_2d.values
            x_values = df_2d.columns
            y_values = df_2d.index
        else:
            x_label = data_df["Dimension-1 Parameter"][0]
            data_nd_array = data_df[value_column_name].values
            x_values = data_df["Dimension-1 Value"].values

        min_value = np.amin(data_nd_array)
        min_idx = np.unravel_index(np.argmin(data_nd_array), data_nd_array.shape)
        max_value = np.amax(data_nd_array)
        max_idx = np.unravel_index(np.argmax(data_nd_array), data_nd_array.shape)

        if two_dim:
            return (
                data_nd_array,
                x_label,
                y_label,
                df_2d,
                x_values,
                y_values,
                min_value,
                min_idx,
                max_value,
                max_idx,
            )
        else:
            return (
                data_nd_array,
                x_label,
                x_values,
                min_value,
                min_idx,
                max_value,
                max_idx,
            )

    # Analysis functions used inside or outside
    def get_cross_corr_latency(self, lags, corr, dt):

        idx = lags >= 0
        corr_on_positive_lags = corr[idx]
        positive_lags = lags[idx]
        idx2 = np.argmax(corr_on_positive_lags)
        shift_in_samples = positive_lags[idx2]
        shift_in_seconds = shift_in_samples * dt

        return shift_in_seconds

    # Internal helper functions. Isolating complexity or iterative tasks.
    def _init_grcaus_regression_QA(self, iter_dict=None):
        # Init for folder specific regression errors. This needs to be in separate method, because in multiprocessing
        # the __init__ is run before the final output folder is defined.

        if iter_dict is not None:
            output_folder_full = iter_dict["_output_folder_full"]
        else:
            output_folder_full = self.context.output_folder  # CHECK

        self._tmp_regression_error_full_path = Path.joinpath(
            output_folder_full, "_tmp_regression_errors.txt"
        )
        self.regression_error_full_path = Path.joinpath(
            output_folder_full, "regression_errors.txt"
        )

        # Check for grcaus regression errors. The full path to failed data files, one per line, should be marked in regression_errors.txt file at datafolder
        regression_error_data_list = None
        if self.regression_error_full_path.is_file():
            with open(self.regression_error_full_path) as file:
                regression_error_data = file.readlines()
                regression_error_data_list = [
                    line.rstrip() for line in regression_error_data
                ]

        # Init txt file for Granger causality fit quality diagnostics
        save_gc_fit_dg_and_QA = self.extra_ana_args["GrCaus_args"][
            "save_gc_fit_dg_and_QA"
        ]
        if save_gc_fit_dg_and_QA is True:
            diag_filename = "grcaus_FitDiag.txt"
            self.gc_dg_filename = Path.joinpath(output_folder_full, diag_filename)
            # Flush existing file empty
            open(self.gc_dg_filename, "w").close()

        return regression_error_data_list

    def _get_passing(self, value, threshold, passing_goes="over"):
        assert passing_goes in [
            "under",
            "over",
            "both",
        ], 'Unknown option for passing_goes parameter, valid options are "over" and "under"'

        if np.isnan(value):
            passing = "FAIL"
        elif passing_goes == "over":
            if value <= threshold:
                passing = "FAIL"
            elif value > threshold:
                passing = "PASS"
        elif passing_goes == "under":
            if value > threshold:
                passing = "FAIL"
            elif value <= threshold:
                passing = "PASS"
        elif passing_goes == "both":
            if threshold[0] <= value <= threshold[1]:
                passing = "PASS"
            else:
                passing = "FAIL"

        return passing

    def _gc_model_diagnostics(
        self,
        gc_stat_dg_data,
        pre_idx,
        post_idx,
        show_figure=False,
        save_gc_fit_dg_and_QA=False,
        iter_dict=None,
    ):

        full_model = 1
        partial_model = 0
        current_model = full_model

        _results = gc_stat_dg_data[current_model]
        _residuals = _results.resid
        _exog = gc_stat_dg_data[current_model].model.exog
        _endog = gc_stat_dg_data[current_model].model.endog

        if show_figure is True:
            # plot preparations
            fig = plt.figure()
            plt.rc("figure", figsize=(18, 12))
            plt.rc("font", size=10)

        # Heteroscedasticity
        # Homoscedasticity is assumed as 0 hypothesis.
        p_th = 0.001
        stat_het, p_het, foo1, foo2 = het_breuschpagan(_endog, _exog)
        het_passing = self._get_passing(p_het, p_th, passing_goes="over")
        stat_het = self.round_to_n_significant(stat_het)
        p_het = self.round_to_n_significant(p_het)
        if show_figure is True:
            ax12 = plt.subplot(3, 2, (1, 2))
            # plt.plot(_exog[:,:-2],_residuals,'.b') # We skip the last regressor from exog, which is constant 1
            # We skip the last regressor from exog, which is constant 1
            plt.plot(_exog[:, :-2], _endog, ".b")
            ax12.set_title(
                f"""Heteroscedasticity {het_passing}\npre: {pre_idx}; post: {post_idx},\nstat: {stat_het}; p: {p_het}"""
            )

        # Cook's distance, Influence
        # cook_cutoff = 4 / _results.nobs
        cook_cutoff = 1
        Influence = OLSInfluence(_results)
        cooks_distance = Influence.cooks_distance[0]
        cook_passing = self._get_passing(
            np.max(cooks_distance), cook_cutoff, passing_goes="under"
        )
        if show_figure is True:
            ax3 = plt.subplot(3, 2, 3)
            Influence.plot_influence(ax=ax3, alpha=0.001)
            ax3.set_title(f"Influence plot (H leverage vs Studentized residuals)")
            ax3.set_xlabel("", fontsize=10)
            ax3.set_ylabel("Studentized residuals", fontsize=10)

            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(cooks_distance, ".b")
            ax4.hlines(
                cook_cutoff, 0, len(cooks_distance), colors="red", linestyles="dashed"
            )
            ax4.set_title(
                f"""Cook's distance {cook_passing}\nmax value: {np.max(cooks_distance)}"""
            )

        # # Normality
        # # If the pvalue is lower than threshold, we can reject the
        # # Null hypothesis that the sample comes from a normal distribution.
        p_th = 0.001
        stat_norm, p_norm = kstest_normal(_residuals)
        stat_norm = self.round_to_n_significant(stat_norm)
        p_norm = self.round_to_n_significant(p_norm)
        normality_passing = self._get_passing(p_norm, p_th, passing_goes="over")

        # print(f'Normality of error distribution: p_norm = {p_norm} -- {normality_passing}')
        if show_figure is True:
            ax5 = plt.subplot(3, 2, 5)
            sns.distplot(_residuals, ax=ax5)
            ax5.set_title(f"Distribution of residuals")
            ax6 = plt.subplot(3, 2, 6)
            qqplot(_residuals, line="s", ax=ax6)
            ax6.set_title(f"Normality of residuals {normality_passing}")

        # Serial autocorrelation
        limits = [1.5, 2.5]
        stat_acorr = durbin_watson(_residuals)
        acorr_passing = self._get_passing(stat_acorr, limits, passing_goes="both")
        # print(f'Autocorrelation of error distribution: stat = {str(stat_acorr)} -- {acorr_passing}')

        # Multicollinearity
        # Variance inflation factor
        VIF = np.zeros_like(_results.model.exog[1])
        for this_idx in np.arange(len(_results.model.exog[1])):
            VIF[this_idx] = variance_inflation_factor(_results.model.exog, this_idx)
        max_VIF = np.max(VIF)
        VIF_th = 1000
        vif_passing = self._get_passing(max_VIF, VIF_th, passing_goes="under")

        if show_figure is True:
            plt.show()

        # Results summary to txt file
        if save_gc_fit_dg_and_QA is True:
            filename_full_path = self.gc_dg_filename
            results_summary = _results.summary()
            with open(filename_full_path, "a") as text_file:
                text_file.write(
                    # f"\n{iter_dict['this_iter_data_file']}\npre {pre_idx} -- post {post_idx}\n"
                    f"\npre {pre_idx} -- post {post_idx}\n"
                )
                text_file.write("\n")
                text_file.write(
                    f"het {het_passing}, cook {cook_passing}, norm {normality_passing}, acorr {acorr_passing}, vif {vif_passing}"
                )
                text_file.write("\n\n")
                text_file.write(str(results_summary))
                text_file.write("\n")

        return het_passing, cook_passing, normality_passing, acorr_passing, vif_passing

    def _preprocess_for_info_analyzes(
        self,
        source_signal,
        target_signal,
        downsampling_factor,
        n_samples,
        t_idx_start=0,
        t_idx_end=None,
    ):
        """Downsample and 1-diff. Shift end idx from - to absolute"""
        assert (
            max(source_signal.shape) == source_signal.shape[0]
        ), "Source signal dimension mismatch at preprocessing--assumes axis 0 = data dim, aborting..."
        assert (
            max(target_signal.shape) == target_signal.shape[0]
        ), "Target signal dimension mismatch at preprocessing--assumes axis 0 = data dim, aborting..."

        source_signal_ds = self._downsample(
            source_signal, downsampling_factor=downsampling_factor, axis=0
        )
        target_signal_ds = self._downsample(
            target_signal, downsampling_factor=downsampling_factor, axis=0
        )
        # Preprocessing. Preferentially first-order differencing.
        diff_order = 1
        source_signal_pp = np.diff(source_signal_ds, n=diff_order, axis=0)
        target_signal_pp = np.diff(target_signal_ds, n=diff_order, axis=0)

        # Turn sample idx relative to end, such as -100, to actual end sample idx
        t_idx_end = self.end2idx(t_idx_end, n_samples)

        # Cut to requested length after preprocessing
        if source_signal.ndim == 2:
            source_signal_pp_cut = source_signal_pp[
                t_idx_start // downsampling_factor : t_idx_end // downsampling_factor, :
            ]
            target_signal_pp_cut = target_signal_pp[
                t_idx_start // downsampling_factor : t_idx_end // downsampling_factor, :
            ]
        elif source_signal.ndim == 1:
            source_signal_pp_cut = source_signal_pp[
                t_idx_start // downsampling_factor : t_idx_end // downsampling_factor
            ]
            target_signal_pp_cut = target_signal_pp[
                t_idx_start // downsampling_factor : t_idx_end // downsampling_factor
            ]

        return source_signal_pp_cut, target_signal_pp_cut

    def _get_spikes_with_leak(self, spikes, Lambda, dt):
        spikes_leak = np.zeros(spikes.shape)
        n_time_points = spikes.shape[0]
        decay = 1 - Lambda * dt

        # Filtered spike train from spikes
        time_vector = np.arange(n_time_points)
        for t in time_vector[1:]:
            spikes_leak[t - 1, :] += spikes[t - 1, :]

            # the filtered spike train has a leak time constant of Lambda (time points)
            # spikes_leak[t, :] = (1 - Lambda * dt) * spikes_leak[t - 1, :]
            spikes_leak[t, :] = decay * spikes_leak[t - 1, :]

        return spikes_leak

    def _get_input_with_leak(self, Input, Lambda, dt):
        # Original matlab code:
        # Get target output by leaky integration of the input %%
        # for t=2:TimeT
        #     xT(:,t) = (1 - lambda * dt) * xT(:, t-1) + dt * Input(:, t-1);
        # end
        input_leak = np.zeros(Input.shape)
        n_time_points = Input.shape[0]
        decay = 1 - Lambda * dt

        # Filtered input
        time_vector = np.arange(n_time_points)

        if Input.ndim == 2:
            for t in time_vector[1:]:
                # the filtered input has a leak time constant of Lambda (time points)
                input_leak[t, :] = decay * input_leak[t - 1, :] + dt * Input[t - 1, :]
        elif Input.ndim == 1:
            for t in time_vector[1:]:
                input_leak[t] = decay * input_leak[t - 1] + dt * Input[t - 1]

        return input_leak

    def _check_readout_group(self, simulation_engine, readout_group):

        if simulation_engine.lower() in ["cxsystem"]:
            if readout_group.lower() in ["e", "excitatory"]:
                readout_group = "NG1"
            elif readout_group.lower() in ["i", "inhibitory"]:
                readout_group = "NG2"
            elif "NG1" not in readout_group and "NG2" not in readout_group:
                raise ValueError(
                    f"{readout_group} is not valid readout_group for simulation_engine {simulation_engine}, aborting..."
                )
        elif simulation_engine.lower() in ["matlab"]:
            if readout_group.lower() in ["ng1", "excitatory"]:
                readout_group = "E"
            elif readout_group.lower() in ["ng2", "inhibitory"]:
                readout_group = "I"
            elif "E" not in readout_group and "I" not in readout_group:
                raise ValueError(
                    f"{readout_group} is not valid readout_group for simulation_engine {simulation_engine}, aborting..."
                )
        else:
            raise ValueError(
                f"{simulation_engine} is not a valid simulation_engine, aborting..."
            )

        return readout_group

    def _get_optimal_decoders(self, target_output, spikes_leak, decoding_method):

        # Computing optimal decoders
        # Decs = (spikes_leak' \ target_output')'; % target_output is the input smoothed with membrane leak; Note: (spikes_leak' \ target_output')' = target_output / spikes_leak
        # target_output = Decs * spikes_leak which can be solved for Decs: xb = a: solve b.T x.T = a.T instead
        # b: spikes_leak; a: target_output; x: Decs
        # From numpy for matlab users (numpy site)
        # Matlab: b/a; Python: Solve a.T x.T = b.T instead, which is a solution of x a = b for x

        if decoding_method in ["least_squares", "lstsq"]:
            Decs = np.linalg.lstsq(spikes_leak, target_output)[0]
        elif decoding_method in ["pseudoinverse", "pinv"]:
            Decs = np.dot(target_output.T, np.linalg.pinv(spikes_leak.T))
            Decs = Decs.T
        else:
            raise NotImplementedError(
                'Unknown method for decoding, try "least_squares", aborting...'
            )

        return Decs

    def _get_best_values(self, data_df, this_column, best_is="max"):
        # Get best value and corresponding parameter value(s)
        # from the given df and column

        at_param_list = [np.nan, np.nan]
        y_values = None
        min_idx_limit, max_idx_limit = [], []

        if "Dimension-2 Parameter" in data_df.columns:
            two_dim = True
            (
                data_nd_array,
                x_label,
                y_label,
                df_2d,
                x_values,
                y_values,
                min_value,
                min_idx,
                max_value,
                max_idx,
            ) = self.get_df_extremes(data_df, this_column, two_dim=True)
            if best_is == "max":
                # Note, in 2d matrix y is the row, ie the first index
                at_param_list = [x_values[max_idx[1]], y_values[max_idx[0]]]
                best_value = max_value
            elif best_is == "min":
                at_param_list = [x_values[min_idx[1]], y_values[min_idx[0]]]
                best_value = min_value
            axis_corners = [
                np.amin(x_values),
                np.amax(x_values),
                np.amin(y_values),
                np.amax(y_values),
            ]
        else:
            two_dim = False
            (
                data_nd_array,
                x_label,
                x_values,
                min_value,
                min_idx,
                max_value,
                max_idx,
            ) = self.get_df_extremes(data_df, this_column, two_dim=False)
            if best_is == "max":
                at_param_list[0] = x_values[max_idx]
                best_value = max_value
            elif best_is == "min":
                at_param_list[0] = x_values[min_idx]
                best_value = min_value
            axis_corners = [np.amin(x_values), np.amax(x_values)]

        return best_value, at_param_list, axis_corners, x_values, y_values

    def _get_normalized_error_variance(self, target_output, output):

        # Get MSE error according to Brendel_2020_PLoSComputBiol
        # we compute the variance of the error normalized by the variance of the target population
        Error = np.sum(np.var(target_output - output, axis=0)) / np.sum(
            np.var(target_output, axis=0)
        )

        return Error

    # Array analysis helper and subfunctions
    def _pin_transfer_entropy(self, target, source, embedding_vector, n_states=2):
        """
        Pyinform implementation of transfer_entropy. It gives the average transfer of entropy per timepoint.
        The local option enables looking at timepointwise transfer
        Start from https://elife-asu.github.io/PyInform/dist.html
        Ref: Moore et al. Front Rob AI, June 2018 | Volume 5 | Article 60
        Ref: J.T. Lizier et al. Information Sciences 208 (2012) 39–54
        """

        def _data2bins(data, n_states):
            # Base to bin edges
            bins = np.linspace(
                np.min(data), np.max(data), num=n_states + 1, endpoint=True
            )
            # Get digitized values
            # Note reducing to 16 bit integers to save memory
            inds = np.digitize(data, bins).astype("int64")
            return inds

        binned_source = _data2bins(source, n_states)
        binned_target = _data2bins(target, n_states)

        try:
            # Average transfer entropy. Note that you have the option of getting an array of te:s as a function of time,
            # for each input-output, if you set the local = True. With condition you can provide a confounding time series
            pin_te = pin.transferentropy.transfer_entropy(
                binned_source,
                binned_target,
                embedding_vector,
                condition=None,
                local=False,
            )
        # Memory allocation error , access violation writing on memory
        except (pin.error.InformError, OSError):
            print(
                f"\033[31m\nFailed transfer entropy due to memory allocation failure (history_epoch_samples = {embedding_vector}), setting transfer entropy to zero\033[97m"
            )
            pin_te = 0

        return pin_te

    def _vm_entropy(self, data, n_states=None, bins=None):
        """Computes entropy of data distribution."""

        # de-mean data
        data_demeaned = data - data.mean(axis=0)
        # flatten data
        _data = data_demeaned.flatten()
        n_labels = len(_data)

        if n_labels <= 1:
            return 0

        # Assuming Vm dynamic range about 30 mV. Assuming a 10-step resolution
        # Note, demeaned.
        bins = np.linspace(-15, 15, 11)

        if bins is not None:
            counts, foo = np.histogram(_data, bins=bins)
        else:
            value, counts = np.unique(_data, return_counts=True)

        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.0

        # Compute entropy
        n_states = math.e if n_states is None else n_states

        for i in probs:
            # ent change is 0 if PM(i) is 0, ie empty bin
            if i == 0.0:
                pass
            else:
                ent -= i * math.log(i, n_states)

        if 0:
            plt.hist(_data, bins=bins)
            plt.title(f"_vm_entropy = {ent}")
            plt.show()

        return ent

    def _get_MSE_1d(
        self,
        Input=None,
        simulated_output_vm=None,
        results_filename=None,
        data_dict=None,
        simulation_engine="CxSystem",
        readout_group="NG1",
        decoding_method="least_squares",
        output_type="estimated",
    ):
        """
        Get decoding error. Allows both direct call by default method on single files and with data dictionary for array analysis.
        """

        if Input is None:
            Input = self.data_io.read_input_matfile(
                filename=self.context.input_filename, variable="stimulus"
            )
        # n_input_time_points = Input.shape[0]

        # Check readoutgroup name, standardize
        readout_group = self._check_readout_group(simulation_engine, readout_group)

        # Get filtered spike train from simulation.
        if "matlab" in simulation_engine.lower():
            if data_dict is None:
                assert (
                    results_filename is not None
                ), 'For matlab, you need to provide the workspace data as "results_filename", aborting...'
                # Input scaling factor A is 2000 for matlab results
                data_dict = self.data_io.get_data(filename=results_filename)
            target_rO_name = f"rO{readout_group}"
            try:
                spikes_leak = data_dict[target_rO_name].T
            except KeyError:
                raise KeyError("rOE/rOI not found in matlab workspace, aborting...")
            Lambda = data_dict["lambda"]
            dt = data_dict["dt"]
            n_time_points = spikes_leak.shape[0]

        elif "cxsystem" in simulation_engine.lower():
            # Input scaling factor A is 15 for python results
            if data_dict is None:
                data_dict = self.data_io.get_data(
                    filename=results_filename, data_type="results"
                )
            n_time_points = self.cxparser.get_n_samples(data_dict)
            NG_name = [
                n for n in data_dict["spikes_all"].keys() if f"{readout_group}" in n
            ][0]
            n_neurons = data_dict["Neuron_Groups_Parameters"][NG_name][
                "number_of_neurons"
            ]

            # Get dt
            dt = self.cxparser.get_dt(data_dict)

            # Get Lambda, a.k.a. tau_soma, but in time points. Can be float.
            Lambda_unit = self.cxparser.get_namespace_variable(
                data_dict, readout_group, variable_name="taum_soma"
            )
            Lambda = int(Lambda_unit.base / dt)

            # Get spikes
            spike_idx = data_dict["spikes_all"][NG_name]["i"]
            spike_times = data_dict["spikes_all"][NG_name]["t"]
            spike_times_idx = np.array(spike_times / (dt * b2u.second), dtype=int)

            # Create spike vector
            spikes = np.zeros([n_time_points, n_neurons])
            spikes[spike_times_idx, spike_idx] = 1
            # Spikes with leak
            spikes_leak = self._get_spikes_with_leak(spikes, Lambda, dt)

        # Get input with leak, a.k.a. the target output
        input_leak = self._get_input_with_leak(Input, Lambda, dt)
        target_output = input_leak

        # Cut start and end time points by request
        # Cut spikes_leak and target_output
        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end
        spikes_leak_cut = spikes_leak[t_idx_start:t_idx_end]
        target_output_cut = target_output[t_idx_start:t_idx_end]

        # Get output
        assert output_type in [
            "estimated",
            "simulated",
        ], 'Unknown output type, should be "estimated" or "simulated", aborting...'
        if output_type == "estimated":
            # Get optimal decoders with analytical method. This is the best possible outputsignal, given the leaky spikes
            Decs = self._get_optimal_decoders(
                target_output_cut, spikes_leak_cut, decoding_method
            )
            # Get estimated output
            estimated_output = np.dot(Decs.T, spikes_leak_cut.T)
            output = estimated_output.T
        elif output_type == "simulated":
            # Get simulated vm values for target group. Only valid for cxsystem data.
            # both output and target data are scaled to -1,1 for comparison. Minmax keeps distribution histogram form intact.
            simulated_output = self.scaler(simulated_output_vm, scale_type="minmax")
            output = simulated_output
            target_output_cut = self.scaler(target_output_cut, scale_type="minmax")

        Error = self._get_normalized_error_variance(target_output_cut, output)

        return Error

    def _granger_causality(
        self,
        target_signal_pp,
        source_signal_pp,
        max_lag_seconds,
        ic,
        dt,
        downsampling_factor,
        signif=0.05,
        verbose=True,
    ):
        """
        Statsmodels VAR model.fit results.test_causality implementation
        :param target_signal_pp: numpy array, output signal
        :param source_signal_pp: numpy array, input signal
        :param  max_lag_seconds: float, parameter for VAR model fit
        :param ic: str,  either aic or bic, information criterion to select model duration
        :param dt: float, sampling frequency
        :param downsampling_factor: int
        :param signif: float [0,1], alpha level for statistical test of significance
        :param verbose: bool
        """

        max_lag_samples = int(max_lag_seconds / (dt * downsampling_factor))

        def get_column_names(data, prefix):
            try:
                name_list = [f"{prefix}{i}" for i in range(data.shape[1])]
            except IndexError:
                name_list = [f"{prefix}0"]
            return name_list

        target_name_list = get_column_names(target_signal_pp, "target")
        source_name_list = get_column_names(source_signal_pp, "source")

        # make df for explicit naming
        df = pd.DataFrame(
            data=np.column_stack([target_signal_pp, source_signal_pp]),
            columns=target_name_list + source_name_list,
        )
        model = VAR(df)
        try:
            results = model.fit(maxlags=max_lag_samples, ic=ic, trend="c")
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        history_epoch_samples = results.k_ar

        gc_results = results.test_causality(
            target_name_list, causing=source_name_list, kind="f", signif=signif
        )

        F_value = gc_results.test_statistic
        p_value = gc_results.pvalue

        if verbose is True:
            if p_value <= signif:
                print(f"\033[32m{gc_results.summary()})\033[97m")
            else:
                print(f"\033[31m{gc_results.summary()})\033[97m")

        return F_value, p_value, history_epoch_samples

    def _analyze_eicurrentdiff(self, data, NG):

        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        I_e, I_i, I_leak = self.cxparser.get_currents_by_interval(
            data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end
        )

        N_neurons = I_e.shape[1]

        # Calculate difference unit by unit, I_e all positive, I_i all negative, thus the difference is +
        EIdifference = I_e[t_idx_start:t_idx_end] + I_i[t_idx_start:t_idx_end]

        MeanEIdifference = np.sum(np.abs(EIdifference)) / N_neurons

        return MeanEIdifference

    def _analyze_meanfr(self, data, NG):

        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        n_samples = self.cxparser.get_n_samples(data)
        t_idx_end = self.end2idx(t_idx_end, n_samples)

        N_neurons, spikes, dt = self.cxparser._get_spikes_by_interval(
            data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end
        )

        MeanFR = spikes.size / (N_neurons * (t_idx_end - t_idx_start) * dt)

        return MeanFR

    def _analyze_meanvm(self, data, NG):

        # data_by_group = data["vm_all"][NG]
        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end
        vm = self.cxparser.get_vm_by_interval(
            data, NG=NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end
        )

        MeanVm = np.mean(vm)

        return MeanVm

    def _analyze_transfer_entropy(self, data, source_signal_original, dt, target_group):
        """
        Transfer entropy analysis. Reference in manuscript Artinano et al.
        :data: dictionary, output from CxSystem
        :source_signal_original: numpy array, input stimulus
        :dt: float, delta time i.e. sampling period
        :target_group: string, target group name
        :returns: float, transfer_entropy
        """
        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        max_time_lag_seconds = self.extra_ana_args["TE_args"]["max_time_lag_seconds"]
        downsampling_factor = self.extra_ana_args["TE_args"]["downsampling_factor"]
        embedding_vector = self.extra_ana_args["TE_args"]["embedding_vector"]
        n_states = self.extra_ana_args["TE_args"]["n_states"]

        assert isinstance(
            downsampling_factor, int
        ), "downsampling_factor must be integer, aborting..."

        # full length for fourier downsampling
        vm_unit = self.cxparser.get_vm_by_interval(
            data, target_group, t_idx_start=0, t_idx_end=None
        )
        target_signal = vm_unit / vm_unit.get_best_unit()
        n_samples = target_signal.shape[0]

        # Cut to requested length
        source_signal = source_signal_original[t_idx_start:t_idx_end, :]
        target_signal = target_signal[t_idx_start:t_idx_end, :]

        # Shift time according to cross correlation
        nperseg = n_samples // 6
        samp_freq = 1.0 / dt

        # recalc n_samples after cut
        n_samples = target_signal.shape[0]

        # Loop and take median.
        source_len = source_signal.shape[1]
        transfer_entropy_arr = np.zeros([source_len, 1]) * np.nan
        latency_arr = np.zeros([source_len, 1])
        for idx in np.arange(source_len):
            _source = source_signal[:, idx]
            _target = target_signal[:, idx]

            (
                f,
                Cxy,
                Pwelch_spec_x,
                Pwelch_spec_y,
                Pxy,
                lags,
                corr,
                coherence_sum,
                _source_scaled,
                _target_scaled,
            ) = self.get_coherence_of_two_signals(
                _source, _target, samp_freq=samp_freq, nperseg=nperseg
            )
            shift_in_seconds = self.get_cross_corr_latency(lags, corr, dt)
            latency_arr[idx] = shift_in_seconds

            if shift_in_seconds > max_time_lag_seconds:
                # If we exceed time lag, we assume that cross correlations peak is due to mainly noise
                continue

            shift_in_samples = int(shift_in_seconds // dt)

            # shift original source, anticipate k-1 in embedding vector and re-cut.
            # One sample in downsampled space equals the downsampling factor
            source_signal_1d = source_signal_original[
                t_idx_start
                - shift_in_samples
                + downsampling_factor : t_idx_end
                - shift_in_samples
                + downsampling_factor,
                idx,
            ]

            (
                source_signal_1d_pp,
                target_signal_1d_pp,
            ) = self._preprocess_for_info_analyzes(
                source_signal_1d,
                target_signal[:, idx],
                downsampling_factor,
                n_samples,
                t_idx_start=0,
                t_idx_end=None,
            )  # These are already cut

            transfer_entropy_arr[idx] = self._pin_transfer_entropy(
                target_signal_1d_pp,
                source_signal_1d_pp,
                embedding_vector,
                n_states=n_states,
            )

        if all(transfer_entropy_arr.flatten() == 0):
            transfer_entropy, MeanTransferEntropy_latency = 0, np.nan
        else:
            # Get median transfer entropy
            transfer_entropy = np.nanmean(transfer_entropy_arr)
            # # Get boolean array for indexing the latency
            # medIdx = transfer_entropy_arr == transfer_entropy
            # Get latency
            MeanTransferEntropy_latency = np.nanmean(latency_arr)

        return transfer_entropy, MeanTransferEntropy_latency

    def _analyze_GC_as_TE(
        self,
        target_signal,
        source_signal_original,
        ic,
        dt,
        downsampling_factor,
        max_time_lag_seconds,
    ):

        """
        Granger causality analysis as Transfer entropy, shifted and
        truncated to one pre and post time points.
        :target_signal numpy array:, output from CxSystem without units
        :source_signal_original numpy array:, input stimulus
        :ic str: information criterion to select correct model
        :dt float:, delta time i.e. sampling period
        :downsampling_factor int:
        :returns float: granger causality F-statistics
        """
        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        # # full length for fourier downsampling
        # vm_unit = self.cxparser.get_vm_by_interval(
        #     data, target_group, t_idx_start=0, t_idx_end=None
        # )
        # target_signal = vm_unit / vm_unit.get_best_unit()
        n_samples = target_signal.shape[0]

        # Cut to requested length
        source_signal = source_signal_original[t_idx_start:t_idx_end, :]
        target_signal = target_signal[t_idx_start:t_idx_end, :]

        # Shift time according to cross correlation
        nperseg = n_samples // 6
        samp_freq = 1.0 / dt

        # recalc n_samples after cut
        n_samples = target_signal.shape[0]

        # Loop and take median.
        source_len = source_signal.shape[1]
        gc_Fstat_arr = np.zeros([source_len, 1]) * np.nan

        for idx in np.arange(source_len):
            _source = source_signal[:, idx]
            _target = target_signal[:, idx]

            (
                f,
                Cxy,
                Pwelch_spec_x,
                Pwelch_spec_y,
                Pxy,
                lags,
                corr,
                coherence_sum,
                _source_scaled,
                _target_scaled,
            ) = self.get_coherence_of_two_signals(
                _source, _target, samp_freq=samp_freq, nperseg=nperseg
            )
            shift_in_seconds = self.get_cross_corr_latency(lags, corr, dt)

            if shift_in_seconds > max_time_lag_seconds:
                # If we exceed time lag, we assume that cross correlations peak is due to mainly noise
                continue

            shift_in_samples = int(shift_in_seconds // dt)

            # shift original source, anticipate k-1 in embedding vector and re-cut.
            # One sample in downsampled space equals the downsampling factor
            source_signal_1d = source_signal_original[
                t_idx_start
                - shift_in_samples
                + downsampling_factor : t_idx_end
                - shift_in_samples
                + downsampling_factor,
                idx,
            ]

            (
                source_signal_1d_pp,
                target_signal_1d_pp,
            ) = self._preprocess_for_info_analyzes(
                source_signal_1d,
                target_signal[:, idx],
                downsampling_factor,
                n_samples,
                t_idx_start=0,
                t_idx_end=None,
            )  # These are already cut

            max_lag_sec = (
                dt * downsampling_factor
            )  # One sample or zero sample, test , dt * downsampling_factor.

            gc_Fstat, foo1, foo2 = self._granger_causality(
                target_signal_1d_pp,
                source_signal_1d_pp,
                max_lag_sec,
                ic,
                dt,
                downsampling_factor,
                signif=0.05,
                verbose=True,
            )

            gc_Fstat_arr[idx] = gc_Fstat

        return gc_Fstat_arr

    def _analyze_grcaus(
        self,
        data,
        source_signal,
        dt,
        target_group,
        verbose=False,
        return_only_infomatrix=False,
        known_regression_failure=False,
        iter_dict=None,
    ):
        """
        Granger causality analysis between source and target of correctly classified units.
        If verbose, shows hypothesis test that source does not grcause target (H0) at 0.05 significance
        Assuming that the correctly classified units are on the diagonal indexes [0->0, 1->1 ..., n->n]
        :param data: dictionary, output from CxSystem
        :param source_signal: numpy array, input stimulus
        :param dt: float, delta time i.e. sampling period
        :param target_group: string, target group name
        :param verbose: bool, print fit result in color
        :param return_only_infomatrix: bool, for coherence, we need only infomatrix
        :param known_regression_failure: bool, if true, set return parameters to zero/nan and return
        :returns: GrCaus_information, GrCaus_p, GrCaus_latency, target_signal_entropy, MeanGrCaus_fitQA, GrCaus_InfoAsTE
        """

        # First, test if this data is known regression error data, if yes, return with zeros and nans.
        if known_regression_failure is True:
            print(
                f"\033[31m\nKnown bad data (earlier segmentation fault etc), setting all GC analysis values to zero or nan\033[97m"
            )
            if return_only_infomatrix is False:
                if logging.getLogger().hasHandlers():
                    logging.info(
                        f"Known bad data (earlier segmentation fault etc), setting all GC analysis values to zero or nan"
                    )
                # You might want to return without arguments (all are init to NaN:s in the invoking function) for iterations, i.e. nanmean/nanmedian calculations.
                # For single run visualization, nan did not work.
                return 0, 1, np.nan, 0, 0
            elif return_only_infomatrix is True:
                # If target signal shape changes, or classify starts assuming particular matrix size, this might fail
                return np.zeros((source_signal.shape[1], source_signal.shape[1]))

        max_time_lag_seconds = self.extra_ana_args["GrCaus_args"][
            "max_time_lag_seconds"
        ]

        # Barnett personal communication: sample-period should be kept reasonable in relation to information transfer.
        downsampling_factor = self.extra_ana_args["GrCaus_args"]["downsampling_factor"]
        save_gc_fit_dg_and_QA = self.extra_ana_args["GrCaus_args"][
            "save_gc_fit_dg_and_QA"
        ]
        show_figure = self.extra_ana_args["GrCaus_args"][
            "show_gc_fit_diagnostics_figure"
        ]

        ic = "aic"  # Akaike information criterion

        # full length for fourier downsampling
        vm_unit = self.cxparser.get_vm_by_interval(
            data, target_group, t_idx_start=0, t_idx_end=None
        )
        target_signal = vm_unit / vm_unit.get_best_unit()
        target_signal_entropy = self._vm_entropy(target_signal, n_states=2)

        assert isinstance(
            downsampling_factor, int
        ), "downsampling_factor must be integer, aborting..."

        # Pre-cutting slightly modifies the preprocessing, thus pre-cut here as with IxO analysis.
        source_signal_cut = source_signal[
            self.context.t_idx_start : self.end2idx(
                self.context.t_idx_end, source_signal.shape[0]
            ),
            :,
        ]
        target_signal_cut = target_signal[
            self.context.t_idx_start : self.end2idx(
                self.context.t_idx_end, target_signal.shape[0]
            ),
            :,
        ]

        source_signal_pp, target_signal_pp = self._preprocess_for_info_analyzes(
            source_signal_cut,
            target_signal_cut,
            downsampling_factor,
            vm_unit.shape[0],
            # t_idx_start=self.context.t_idx_start,
            t_idx_start=0,
            # t_idx_end=self.context.t_idx_end,
            t_idx_end=None,
        )

        pre_idx_array = np.arange(source_signal_pp.shape[1])
        post_idx_array = np.arange(target_signal_pp.shape[1])

        gc_matrix_np_F = np.full_like(
            np.empty((source_signal_pp.shape[1], target_signal_pp.shape[1])), 0
        )
        gc_fitQA = 0

        # Analyze 1-tp shifted inputs on diagonal
        F_value_arr = self._analyze_GC_as_TE(
            target_signal,
            source_signal,
            ic,
            dt,
            downsampling_factor,
            max_time_lag_seconds,
        )
        GrCaus_InfoAsTE = np.nanmean(np.log2(F_value_arr))  # First info, then mean

        # GC analysis one-by-one
        F_value_arr = np.zeros(source_signal_pp.shape[1])
        p_value_arr = np.zeros(source_signal_pp.shape[1])
        history_arr = np.zeros(source_signal_pp.shape[1])

        for this_idx in range(source_signal_pp.shape[1]):
            (
                F_value_arr[this_idx],
                p_value_arr[this_idx],
                history_arr[this_idx],
            ) = self._granger_causality(
                target_signal_pp[:, this_idx],
                source_signal_pp[:, this_idx],
                max_time_lag_seconds,
                ic,
                dt,
                downsampling_factor,
                verbose=verbose,
            )
        F_value = np.nanmean(F_value_arr)
        p_value = np.nanmean(p_value_arr)
        history_epoch_samples = int(np.round(np.nanmean(history_arr)))

        if F_value == 0:
            print(
                f"\033[31m\nFailed Granger Causality (fit failed), setting all GC analysis values to zero or nan\033[97m"
            )
            if logging.getLogger().hasHandlers():
                logging.error(
                    f"\nFailed Granger Causality (fit failed), setting all GC analysis values to zero or nan"
                )
            return 0, 1, np.nan, 0, 0
        # At timesteps of dt * downsampling factor
        latency = history_epoch_samples * dt * downsampling_factor

        # Get representative values
        GrCaus_information = np.nanmean(np.log2(F_value_arr))  # First to info, the ave
        GrCaus_p = p_value
        GrCaus_latency = latency
        MeanGrCaus_fitQA = 1

        # for each source signal, calculate all target signals.
        for pre_idx in pre_idx_array:
            for post_idx in post_idx_array:

                # The model order is the maximum number of lagged observations included in the model
                _source, _target = (
                    source_signal_pp[:, pre_idx],
                    target_signal_pp[:, post_idx],
                )
                signals = np.vstack([_target, _source]).T

                pairwise_gc_dict = gc_test(
                    signals, [history_epoch_samples], verbose=False
                )

                if return_only_infomatrix is False:
                    # GC quality control, several measures on error distribution
                    gc_stat_dg_data = pairwise_gc_dict[history_epoch_samples][1]
                    (
                        het_passing,
                        cook_passing,
                        normality_passing,
                        acorr_passing,
                        vif_passing,
                    ) = self._gc_model_diagnostics(
                        gc_stat_dg_data,
                        pre_idx,
                        post_idx,
                        show_figure=show_figure,
                        save_gc_fit_dg_and_QA=save_gc_fit_dg_and_QA,
                        iter_dict=iter_dict,
                    )
                    gc_fitQA += str(
                        [
                            het_passing,
                            cook_passing,
                            normality_passing,
                            acorr_passing,
                            vif_passing,
                        ]
                    ).count("PASS")

                # dict_keys(['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'])
                # test statistic, pvalues, degrees of freedom
                gc_test_type = "ssr_ftest"
                gc_matrix_np_F[pre_idx, post_idx] = pairwise_gc_dict[
                    history_epoch_samples
                ][0][gc_test_type][0]

        # Magnitude in base 2 log. If error distribution is gaussian, can be interpreted as bits
        # Multiplied with sample frequency, gives information transfer rate. Lionell Barnett, personal communication:
        # "Note that the log of the F-statistic for the nested VAR model may be interpreted as (Shannon) information
        # measured in bits (or nats, if natural logarithms are used). The information transfer rate -- a more meaningful
        # measure -- is then log(F) x f, with units of  bits (or nats) per second, where f is the sample frequency in Hz."
        gc_matrix_np_Info = np.log2(gc_matrix_np_F)

        if return_only_infomatrix is True:
            return gc_matrix_np_Info

        MeanGrCaus_fitQA = gc_fitQA / (pre_idx_array.size * post_idx_array.size)

        # Return one value per analysis (mean of best matching units), indicating GrCaus relation
        return (
            GrCaus_information,
            GrCaus_p,
            GrCaus_latency,
            target_signal_entropy,
            MeanGrCaus_fitQA,
            GrCaus_InfoAsTE,
        )

    def _analyze_coherence(self, data_dict, source_signal, dt, target_group):
        """
        Median sum of coherence spectrum between source and target of correctly classified units
        Assuming that the correctly classified units are on the diagonal indexes [0->0, 1->1 ..., n->n]
        :param data_dict: dictionary, output from CxSystem
        :param source_signal: numpy array, input stimulus
        :param dt: float, delta time i.e. sampling period
        :param target_group: string, target group name
        """
        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        vm_unit = self.cxparser.get_vm_by_interval(
            data_dict, target_group, t_idx_start=0, t_idx_end=None
        )
        target_signal = vm_unit / vm_unit.get_best_unit()

        nsamples = self.cxparser.get_n_samples(data_dict)
        nperseg = nsamples // 6
        samp_freq = 1.0 / dt

        # Cut to requested length
        source_signal = source_signal[t_idx_start:t_idx_end, :]
        target_signal = target_signal[t_idx_start:t_idx_end, :]

        # Init data matrix
        coherence_matrix_np_sum = np.full_like(
            np.empty((source_signal.shape[1], target_signal.shape[1])), 0
        )
        coherence_matrix_np_latency = np.full_like(
            np.empty((source_signal.shape[1], target_signal.shape[1])), 0
        )

        # Loop units. For each source signal, calculate all target signals.
        pre_idx_array = np.arange(source_signal.shape[1])
        post_idx_array = np.arange(target_signal.shape[1])

        for pre_idx in pre_idx_array:
            for post_idx in post_idx_array:
                _source, _target = source_signal[:, pre_idx], target_signal[:, post_idx]

                (
                    f,
                    Cxy,
                    Pwelch_spec_x,
                    Pwelch_spec_y,
                    Pxy,
                    lags,
                    corr,
                    coherence_sum,
                    _source_scaled,
                    _target_scaled,
                ) = self.get_coherence_of_two_signals(
                    _source, _target, samp_freq=samp_freq, nperseg=nperseg
                )
                shift_in_seconds = self.get_cross_corr_latency(lags, corr, dt)

                coherence_matrix_np_sum[pre_idx, post_idx] = coherence_sum
                coherence_matrix_np_latency[pre_idx, post_idx] = shift_in_seconds

        # Index to diagonal, ie "correctly" classified units and get median of values on diagonal
        eye_idx = np.eye(source_signal.shape[1], target_signal.shape[1], dtype=bool)
        coherences_on_diagonal = coherence_matrix_np_sum[eye_idx]
        MedianCoherenceSum = np.nanmedian(coherences_on_diagonal)
        latencies_on_diagonal = coherence_matrix_np_latency[eye_idx]
        MedianCoherenceLatency = np.nanmedian(latencies_on_diagonal)

        if 1:
            print(f"coherence_matrix_np_sum = {coherence_matrix_np_sum}")
            print(f"coherence_matrix_np_latency = {coherence_matrix_np_latency}")

        # return MedianCoherenceSum, MedianCoherenceLatency, MedianError, MedianErrorShifted
        return MedianCoherenceSum, MedianCoherenceLatency

    def _analyze_normerror(self, data_dict, source_signal):

        decoding_method = self.extra_ana_args["NormError_args"]["decoding_method"]
        do_only_simerror = self.extra_ana_args["NormError_args"]["do_only_simerror"]
        simulation_engine = "CxSystem"

        # #  Get input. This is always the project analog input file
        MeanEstimErr_E = np.nan
        MeanEstimErr_I = np.nan

        if do_only_simerror is not True:
            MeanEstimErr_E, target_output, output = self.get_MSE(
                source_signal,
                data_dict=data_dict,
                simulation_engine=simulation_engine,
                readout_group="excitatory",
                decoding_method=decoding_method,
                output_type="estimated",
            )
            MeanEstimErr_I, target_output, output = self.get_MSE(
                source_signal,
                data_dict=data_dict,
                simulation_engine=simulation_engine,
                readout_group="inhibitory",
                decoding_method=decoding_method,
                output_type="estimated",
            )
        MeanSimErr_O, target_output, output = self.get_MSE(
            source_signal, data_dict=data_dict, output_type="simulated"
        )

        return MeanEstimErr_E, MeanEstimErr_I, MeanSimErr_O

    def _analyze_classification_performance(self, y_data, y_true=None):
        """
        Calculate accuracy score from input matrix. If y_true is not given,
        assumes correct classification at identity mtx.
        :param y_data: numpy array, the input matrix to evaluate classification
        :param y_true: numpy array, the correct matches
        :returns: accuracy score
        """
        if y_true is None:
            try:
                eye_idx = np.eye(y_data.shape[0], y_data.shape[1], dtype=bool)
            except AttributeError:
                return 0
            y_true = eye_idx

        y_pred = np.zeros(y_data.shape)
        row_idx = np.arange(y_data.shape[0])
        column_idx = y_data.argmax(axis=1)
        y_pred[row_idx, column_idx] = 1

        Accuracy = accuracy_score(y_true, y_pred)

        return Accuracy

    def _analyze_euclid_dist(self, data, source_signal, target_group):
        """
        Analyze mean Euclidean distance between input and output in multiple dimensions.

        :param data: dictionary, output from CxSystem
        :param source_signal: numpy array, input stimulus
        :param target_group: string, target group name

        :returns MeanDist: float, mean distance over time points
        """

        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        vm_unit = self.cxparser.get_vm_by_interval(
            data, target_group, t_idx_start=t_idx_start, t_idx_end=t_idx_end
        )

        n_samples = self.cxparser.get_n_samples(data)
        t_idx_end = self.end2idx(t_idx_end, n_samples)
        target_signal_cut = vm_unit / vm_unit.get_best_unit()
        source_signal_cut = source_signal[t_idx_start:t_idx_end, :]

        # normalize to same space, cf MSE
        source_signal_norm = self.scaler(
            source_signal_cut, scale_type="minmax", feature_range=[0, 1]
        )  # minmax standard
        target_signal_norm = self.scaler(
            target_signal_cut, scale_type="minmax", feature_range=[0, 1]
        )

        # subtract
        difference = source_signal_norm - target_signal_norm

        # calculate distance timepoint-by-timepoint
        distances = np.linalg.norm(difference, axis=1)

        MeanDist_over_time = np.mean(distances)

        return MeanDist_over_time

    def _get_analyzed_array_as_df(self, data_df, analysisHR=None, iter_dict=None):
        """
        Call necessary analysis and build dataframes.
        :param data_df: pd dataframe, dataframe with independent variable names and values, which will hold the analyzed values
        :param analysisHR: string, analysis name
        :param iter_dict: dictionary, holds fields for parallel thread safe iterative analysis
        """

        # Get neuron group names
        filename_0 = data_df["Full path"].values[0]
        data = self.data_io.get_data(filename_0)
        NG_list = [
            n for n in data[self.map_data_types[analysisHR.lower()]].keys() if "NG" in n
        ]
        target_group = self.context.NG_name

        # Add neuron group columns
        if analysisHR.lower() in ["meanfr", "meanvm", "eicurrentdiff"]:
            for NG in NG_list:
                data_df[f"{analysisHR}_" + NG] = np.nan
        elif analysisHR.lower() in ["coherence"]:
            data_df[f"{analysisHR}_" + target_group + "_Sum"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_Latency"] = np.nan
        elif analysisHR.lower() in ["transferentropy"]:
            data_df[f"{analysisHR}_" + target_group + "_TransfEntropy"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_Latency"] = np.nan
        elif analysisHR.lower() in ["grcaus"]:
            data_df[f"{analysisHR}_" + target_group + "_Information"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_p"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_Latency"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_TargetEntropy"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_FitQuality"] = np.nan
            data_df[f"{analysisHR}_" + target_group + "_GCAsTE"] = np.nan
        elif analysisHR.lower() in ["normerror"]:
            do_only_simerror = self.extra_ana_args["NormError_args"]["do_only_simerror"]
            if do_only_simerror is not True:
                data_df[f"{analysisHR}" + "_ExcErr"] = np.nan
                data_df[f"{analysisHR}" + "_InhErr"] = np.nan
            data_df[f"{analysisHR}" + "_SimErr"] = np.nan
        elif analysisHR.lower() in ["classify"]:
            data_df[f"{analysisHR}" + "_Accuracy"] = np.nan
        elif analysisHR.lower() in ["edist"]:
            data_df[f"{analysisHR}_" + target_group + "_EDist"] = np.nan

        target_signal_dt = self.cxparser.get_dt(data)

        # Get input signal as reference
        if analysisHR.lower() in [
            "coherence",
            "transferentropy",
            "grcaus",
            "classify",
            "normerror",
            "edist",
        ]:
            if iter_dict is None:
                analog_input = self.data_io.get_data(
                    self.context.input_filename, data_type=None
                )
            else:
                analog_input = self.data_io.get_data(
                    self.context.input_filename,
                    data_type=None,
                    full_path=iter_dict["_input_file_full"],
                )

            source_signal = analog_input["stimulus"].T  # We want time x units
            # assuming input dt in milliseconds
            source_signal_dt = analog_input["frameduration"] / 1000

            assert (
                target_signal_dt == source_signal_dt
            ), "Different sampling rates in input and output has not been implemented, aborting..."

        # Check for regression errors. These should be manually marked in regression_errors.txt file at datafolder
        if iter_dict is None:
            regression_error_full_path = Path.joinpath(
                self.context.output_folder, "regression_errors.txt"
            )
        else:
            regression_error_full_path = Path.joinpath(
                iter_dict["_output_folder_full"], "regression_errors.txt"
            )

        regression_error_data_list = None
        if regression_error_full_path.is_file():
            with open(regression_error_full_path) as file:
                regression_error_data = file.readlines()
                regression_error_data_list = [
                    line.rstrip() for line in regression_error_data
                ]

        # Loop through datafiles
        for this_index, this_file in zip(data_df.index, data_df["Full path"].values):

            known_regression_failure = False
            if regression_error_data_list is not None and analysisHR.lower() in [
                "grcaus",
                "classify",
            ]:
                if this_file in regression_error_data_list:
                    known_regression_failure = True

            data = self.data_io.get_data(this_file)
            if iter_dict is not None:
                iter_dict["this_iter_data_file"] = Path(this_file)

            # Loop through neuron groups
            if analysisHR.lower() in ["meanfr", "meanvm", "eicurrentdiff"]:
                for NG in NG_list:
                    # _analyze_meanfr or _analyze_eicurrentdiff, analysis by single group
                    analyzed_results = eval(
                        f"self._analyze_{analysisHR.lower()}(data, NG)"
                    )
                    data_df.loc[this_index, f"{analysisHR}_" + NG] = analyzed_results

            elif analysisHR.lower() in ["coherence"]:
                MedianCoherenceSum, MedianCoherenceLatency = self._analyze_coherence(
                    data, source_signal, target_signal_dt, target_group
                )
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_Sum"
                ] = MedianCoherenceSum
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_Latency"
                ] = MedianCoherenceLatency

            # Information transfer
            elif analysisHR.lower() in ["transferentropy"]:
                try:
                    (
                        MeanTransferEntropy,
                        MeanTransferEntropy_latency,
                    ) = self._analyze_transfer_entropy(
                        data, source_signal, target_signal_dt, target_group
                    )
                except:
                    if logging.getLogger().hasHandlers():
                        logging.error(
                            f"\nFailed Transfer Entropy at get_analyzed_array_as_df, setting all TE analysis values to zero or nan"
                        )
                    MeanTransferEntropy, MeanTransferEntropy_latency = 0, np.nan
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_TransfEntropy"
                ] = MeanTransferEntropy
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_Latency"
                ] = MeanTransferEntropy_latency

            elif analysisHR.lower() in ["grcaus"]:  # Information transfer
                try:
                    (
                        GrCaus_information,
                        GrCaus_p,
                        GrCaus_latency,
                        target_entropy,
                        MeanGrCaus_fitQA,
                        GrCaus_InfoAsTE,
                    ) = self._analyze_grcaus(
                        data,
                        source_signal,
                        target_signal_dt,
                        target_group,
                        verbose=True,
                        known_regression_failure=known_regression_failure,
                        iter_dict=iter_dict,
                    )
                except:
                    if logging.getLogger().hasHandlers():
                        logging.error(
                            f"\nFailed Granger Causality at get_analyzed_array_as_df, setting all GC analysis values to zero or nan"
                        )
                    (
                        GrCaus_information,
                        GrCaus_p,
                        GrCaus_latency,
                        target_entropy,
                        MeanGrCaus_fitQA,
                        GrCaus_InfoAsTE,
                    ) = (0, 1, np.nan, 0, 0, 0)
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_Information"
                ] = GrCaus_information  # log(F)
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_p"
                ] = GrCaus_p
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_Latency"
                ] = GrCaus_latency
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_TargetEntropy"
                ] = target_entropy
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_FitQuality"
                ] = MeanGrCaus_fitQA
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_GCAsTE"
                ] = GrCaus_InfoAsTE

            elif analysisHR.lower() in ["normerror"]:  # Reconstruction error
                MeanEstimErr_E, MeanEstimErr_I, MeanSimErr_O = self._analyze_normerror(
                    data, source_signal
                )
                if do_only_simerror is not True:
                    data_df.loc[
                        this_index, f"{analysisHR}" + "_ExcErr"
                    ] = MeanEstimErr_E
                    data_df.loc[
                        this_index, f"{analysisHR}" + "_InhErr"
                    ] = MeanEstimErr_I
                data_df.loc[this_index, f"{analysisHR}" + "_SimErr"] = MeanSimErr_O

            # According to sensory information
            elif analysisHR.lower() in ["classify"]:
                # This is dependent on precalculated NormError, Coherence, TE and GC full IxO matrices
                gc_matrix_np_Info = self._analyze_grcaus(
                    data,
                    source_signal,
                    target_signal_dt,
                    target_group,
                    verbose=False,
                    return_only_infomatrix=True,
                    known_regression_failure=known_regression_failure,
                )
                Accuracy = self._analyze_classification_performance(gc_matrix_np_Info)
                data_df.loc[this_index, f"{analysisHR}_" + "Accuracy"] = Accuracy
            elif analysisHR.lower() in ["edist"]:  # Euclidean distance
                MeanDist = self._analyze_euclid_dist(data, source_signal, target_group)
                data_df.loc[
                    this_index, f"{analysisHR}_" + target_group + "_EDist"
                ] = MeanDist

        return data_df

    def _get_analyzed_full_array_as_np(
        self, data_df, analyzes_list=None, iter_dict=None
    ):
        """
        This is called either from analyze_IxO_array or from
        analyze_TEDrift because most boilerplate is the same.
        However, both types of analyzes cannot be done simultaneously.

        Call necessary analyzes and build numpy matrix
        :param data_df: pandas dataframe, filenames from the metadata file
        :param analyzes_list: list, list of requested analyzes
        """

        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        if analyzes_list is None:
            analyzes_list = ["Coherence", "TransferEntropy", "GrCaus", "NormError"]
        analyzes_lowercase_list = [t.lower() for t in analyzes_list]

        # Get n_samples, t_idx_end, target_group name, example_target_signal
        filename_0 = data_df["Full path"].values[0]
        data = self.data_io.get_data(filename_0)
        n_samples = self.cxparser.get_n_samples(data)
        t_idx_end = self.end2idx(t_idx_end, n_samples)
        target_group = self.context.NG_name
        vm_unit = self.cxparser.get_vm_by_interval(
            data, target_group, t_idx_start=0, t_idx_end=None
        )
        example_target_signal = vm_unit / vm_unit.get_best_unit()

        if iter_dict is None:
            analog_input = self.data_io.get_data(
                self.context.input_filename, data_type=None
            )
        else:
            analog_input = self.data_io.get_data(
                None,
                data_type=None,
                full_path=iter_dict["_input_file_full"],
            )

        # We want time x units
        source_signal_original = analog_input["stimulus"].T
        # assuming input dt in milliseconds
        source_signal_dt = analog_input["frameduration"] / 1000

        target_signal_dt = self.cxparser.get_dt(data)
        assert (
            target_signal_dt == source_signal_dt
        ), "Different sampling rates in input and output has not been implemented, aborting..."

        if set(["te_Latencies"]).issubset(analyzes_lowercase_list):
            assert not set(
                ["coherence", "transferentropy", "normerror", "grcaus"]
            ).intersection(
                analyzes_lowercase_list
            ), "Latency spectrum must be calculated in separate analysis info metrics, aborting..."

        # Init analyzed_data_dict with keys as future columns in data_df and each value
        # containing input x output x n_files numpy array of nan
        analyzed_data_dict = {}
        empty_np_array = np.empty(
            (
                source_signal_original.shape[1],
                example_target_signal.shape[1],
                data_df.shape[0],
            )
        )
        empty_np_array[:] = np.NaN

        nperseg = n_samples // 6
        samp_freq = 1.0 / target_signal_dt

        if set(["coherence"]).issubset(analyzes_lowercase_list):
            analyzed_data_dict[f"Coherence_{target_group}_Sum"] = empty_np_array.copy()
            analyzed_data_dict[
                f"Coherence_{target_group}_Latency"
            ] = empty_np_array.copy()
        if set(["transferentropy"]).issubset(analyzes_lowercase_list):
            te_max_time_lag_seconds = self.extra_ana_args["TE_args"][
                "max_time_lag_seconds"
            ]
            te_downsampling_factor = self.extra_ana_args["TE_args"][
                "downsampling_factor"
            ]
            te_embedding_vector = self.extra_ana_args["TE_args"]["embedding_vector"]
            te_n_states = self.extra_ana_args["TE_args"]["n_states"]
            assert isinstance(
                te_downsampling_factor, int
            ), "TransferEntropy downsampling_factor must be integer, aborting..."
            analyzed_data_dict[
                f"TransferEntropy_{target_group}_TransfEntropy"
            ] = empty_np_array.copy()
        if set(["tedrift"]).issubset(analyzes_lowercase_list):
            te_max_time_lag_seconds = self.extra_ana_args["TE_args"][
                "max_time_lag_seconds"
            ]
            te_downsampling_factor = self.extra_ana_args["TE_args"][
                "downsampling_factor"
            ]
            te_embedding_vector = self.extra_ana_args["TE_args"]["embedding_vector"]
            te_n_states = self.extra_ana_args["TE_args"]["n_states"]
            te_shift_start_time = self.extra_ana_args["TE_args"]["te_shift_start_time"]
            te_shift_end_time = self.extra_ana_args["TE_args"]["te_shift_end_time"]
            assert isinstance(
                te_downsampling_factor, int
            ), "TransferEntropy downsampling_factor must be integer, aborting..."
            te_shift_start_sample = int(
                te_shift_start_time // (te_downsampling_factor * target_signal_dt)
            )
            te_shift_end_sample = int(
                te_shift_end_time // (te_downsampling_factor * target_signal_dt)
            )
            te_sample_range = range(te_shift_start_sample, te_shift_end_sample)
            empty_np_array = np.empty(
                (
                    source_signal_original.shape[1],
                    example_target_signal.shape[1],
                    data_df.shape[0],
                    len(te_sample_range),
                )
            )
            empty_np_array[:] = np.NaN
            analyzed_data_dict[
                f"TEDrift_{target_group}_TE_Latencies"
            ] = empty_np_array.copy()
        if set(["normerror"]).issubset(analyzes_lowercase_list):
            do_only_simerror = self.extra_ana_args["NormError_args"]["do_only_simerror"]
            if do_only_simerror is not True:
                analyzed_data_dict[f"NormError_ExcErr"] = empty_np_array.copy()
                analyzed_data_dict[f"NormError_InhErr"] = empty_np_array.copy()
            analyzed_data_dict[f"NormError_SimErr"] = empty_np_array.copy()
        if set(["grcaus"]).issubset(analyzes_lowercase_list):
            regression_error_data_list = self._init_grcaus_regression_QA(
                iter_dict=iter_dict
            )
            gc_max_time_lag_seconds = self.extra_ana_args["GrCaus_args"][
                "max_time_lag_seconds"
            ]
            gc_downsampling_factor = self.extra_ana_args["GrCaus_args"][
                "downsampling_factor"
            ]
            assert isinstance(
                gc_downsampling_factor, int
            ), "GrCaus downsampling_factor must be integer, aborting..."
            analyzed_data_dict[
                f"GrCaus_{target_group}_Information"
            ] = empty_np_array.copy()
            analyzed_data_dict[f"GrCaus_{target_group}_p"] = empty_np_array.copy()
            analyzed_data_dict[f"GrCaus_{target_group}_Latency"] = empty_np_array.copy()
            analyzed_data_dict[
                f"GrCaus_{target_group}_FitQuality"
            ] = empty_np_array.copy()

        # Loop through datafiles
        for file_idx, this_file in zip(data_df.index, data_df["Full path"].values):

            this_file = Path(this_file)
            if set(["grcaus"]).issubset(analyzes_lowercase_list):
                # Automate regression failure detection for Granger causality. Problems ending in segmentation error and process termination prevent
                # postportem functions, thus we make here a premortem preparation, a "will", if you like. Takes about 1 ms.
                with open(self._tmp_regression_error_full_path, "w") as file:
                    file.write(this_file.as_posix())
                    file.write("\n")
                known_regression_failure = False

                if (
                    regression_error_data_list is not None
                    and "grcaus" in analyzes_lowercase_list
                ):
                    if this_file in regression_error_data_list:
                        known_regression_failure = True
                if known_regression_failure is True:
                    if logging.getLogger().hasHandlers():
                        logging.error(
                            f"\nKnown regression error, setting all GC analysis values to nan"
                        )
                    print(
                        f"\nKnown regression error, setting all GC analysis values to nan"
                    )

                # mark filename for optional gc quality control txt file
                if iter_dict is not None:
                    iter_dict["this_iter_data_file"] = this_file

            data_dict = self.data_io.get_data(this_file)
            vm_unit = self.cxparser.get_vm_by_interval(
                data_dict, target_group, t_idx_start=0, t_idx_end=None
            )
            target_signal = vm_unit / vm_unit.get_best_unit()

            # cut
            source_signal = source_signal_original[t_idx_start:t_idx_end, :]
            target_signal = target_signal[t_idx_start:t_idx_end, :]

            # run input x output data
            for source_idx in range(source_signal.shape[1]):
                for target_idx in range(target_signal.shape[1]):

                    _source, _target = (
                        source_signal[:, source_idx],
                        target_signal[:, target_idx],
                    )

                    if set(["coherence"]).issubset(analyzes_lowercase_list):
                        (
                            f,
                            Cxy,
                            Pwelch_spec_x,
                            Pwelch_spec_y,
                            Pxy,
                            lags,
                            corr,
                            coherence_sum,
                            _source_scaled,
                            _target_scaled,
                        ) = self.get_coherence_of_two_signals(
                            _source, _target, samp_freq=samp_freq, nperseg=nperseg
                        )
                        shift_in_seconds = self.get_cross_corr_latency(
                            lags, corr, target_signal_dt
                        )
                        analyzed_data_dict[f"Coherence_{target_group}_Sum"][
                            source_idx, target_idx, file_idx
                        ] = coherence_sum
                        analyzed_data_dict[f"Coherence_{target_group}_Latency"][
                            source_idx, target_idx, file_idx
                        ] = shift_in_seconds

                    if set(["transferentropy"]).issubset(
                        analyzes_lowercase_list
                    ):  # Information transfer
                        if (
                            "coherence_sum" not in locals()
                            or shift_in_seconds not in locals()
                        ):
                            (
                                f,
                                Cxy,
                                Pwelch_spec_x,
                                Pwelch_spec_y,
                                Pxy,
                                lags,
                                corr,
                                coherence_sum,
                                _source_scaled,
                                _target_scaled,
                            ) = self.get_coherence_of_two_signals(
                                _source, _target, samp_freq=samp_freq, nperseg=nperseg
                            )
                            shift_in_seconds = self.get_cross_corr_latency(
                                lags, corr, target_signal_dt
                            )

                        # Shift original input to match max coherence - downsampling_factor (to preserve k-1 embedding vector to comply with pyinform implementation)
                        if shift_in_seconds > te_max_time_lag_seconds:
                            transfer_entropy = np.nan
                        else:
                            shift_in_samples = int(shift_in_seconds // target_signal_dt)
                            source_signal_1d = source_signal_original[
                                t_idx_start
                                - shift_in_samples
                                + te_downsampling_factor : t_idx_end
                                - shift_in_samples
                                + te_downsampling_factor,
                                source_idx,
                            ]
                            (
                                source_signal_1d_pp,
                                target_signal_1d_pp,
                            ) = self._preprocess_for_info_analyzes(
                                source_signal_1d,
                                _target,
                                te_downsampling_factor,
                                n_samples,
                                t_idx_start=0,
                                t_idx_end=None,
                            )  # These are already cut
                            transfer_entropy = self._pin_transfer_entropy(
                                target_signal_1d_pp,
                                source_signal_1d_pp,
                                te_embedding_vector,
                                n_states=te_n_states,
                            )

                        # print(f'\ntransfer_entropy = {transfer_entropy}')
                        analyzed_data_dict[
                            f"TransferEntropy_{target_group}_TransfEntropy"
                        ][source_idx, target_idx, file_idx] = transfer_entropy

                    if set(["tedrift"]).issubset(analyzes_lowercase_list):

                        for shift_idx, shift_in_samples in enumerate(te_sample_range):
                            source_signal_1d = source_signal_original[
                                t_idx_start
                                - shift_in_samples
                                + te_downsampling_factor : t_idx_end
                                - shift_in_samples
                                + te_downsampling_factor,
                                source_idx,
                            ]
                            (
                                source_signal_1d_pp,
                                target_signal_1d_pp,
                            ) = self._preprocess_for_info_analyzes(
                                source_signal_1d,
                                _target,
                                te_downsampling_factor,
                                n_samples,
                                t_idx_start=0,
                                t_idx_end=None,
                            )  # These are already cut
                            transfer_entropy = self._pin_transfer_entropy(
                                target_signal_1d_pp,
                                source_signal_1d_pp,
                                te_embedding_vector,
                                n_states=te_n_states,
                            )
                            analyzed_data_dict[f"TEDrift_{target_group}_TE_Latencies"][
                                source_idx, target_idx, file_idx, shift_idx
                            ] = transfer_entropy
                        analyzed_data_dict["te_sample_range"] = te_sample_range
                        te_timeshifts_in_ms_np = (
                            np.array([*te_sample_range])
                            * (te_downsampling_factor * target_signal_dt)
                            * 1000
                        )
                        analyzed_data_dict[
                            "te_timeshifts_in_ms_np"
                        ] = te_timeshifts_in_ms_np.astype(int)

                    if set(["normerror"]).issubset(analyzes_lowercase_list):

                        decoding_method = self.extra_ana_args["NormError_args"][
                            "decoding_method"
                        ]
                        simulation_engine = "CxSystem"  # 'Matlab' or 'CxSystem'
                        if do_only_simerror is not True:
                            MeanEstimErr_E = self._get_MSE_1d(
                                Input=source_signal_original[:, source_idx],
                                simulated_output_vm=None,
                                data_dict=data_dict,
                                simulation_engine=simulation_engine,
                                readout_group="excitatory",
                                decoding_method=decoding_method,
                                output_type="estimated",
                            )
                            MeanEstimErr_I = self._get_MSE_1d(
                                Input=source_signal_original[:, source_idx],
                                simulated_output_vm=None,
                                data_dict=data_dict,
                                simulation_engine=simulation_engine,
                                readout_group="inhibitory",
                                decoding_method=decoding_method,
                                output_type="estimated",
                            )
                            analyzed_data_dict[f"NormError_ExcErr"][
                                source_idx, target_idx, file_idx
                            ] = MeanEstimErr_E
                            analyzed_data_dict[f"NormError_InhErr"][
                                source_idx, target_idx, file_idx
                            ] = MeanEstimErr_I
                        MeanSimErr_O = self._get_MSE_1d(
                            Input=source_signal_original[:, source_idx],
                            simulated_output_vm=_target,
                            data_dict=data_dict,
                            output_type="simulated",
                        )

                        analyzed_data_dict[f"NormError_SimErr"][
                            source_idx, target_idx, file_idx
                        ] = MeanSimErr_O

                    if set(["grcaus"]).issubset(analyzes_lowercase_list):

                        if known_regression_failure is True:
                            F_value, p_value, latency, gc_fitQA = (
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            )
                        else:
                            # Preprocess
                            (
                                source_signal_pp,
                                target_signal_pp,
                            ) = self._preprocess_for_info_analyzes(
                                _source,
                                _target,
                                gc_downsampling_factor,
                                vm_unit.shape[0],
                                t_idx_start=0,
                                t_idx_end=None,
                            )  # These are already cut
                            try:
                                (
                                    F_value,
                                    p_value,
                                    history_epoch_samples,
                                ) = self._granger_causality(
                                    target_signal_pp,
                                    source_signal_pp,
                                    gc_max_time_lag_seconds,
                                    "aic",
                                    target_signal_dt,
                                    gc_downsampling_factor,
                                    verbose=False,
                                )
                                latency = (
                                    history_epoch_samples
                                    * target_signal_dt
                                    * gc_downsampling_factor
                                )  # At timesteps of dt * downsampling factor

                                if (
                                    self.extra_ana_args["GrCaus_args"][
                                        "save_gc_fit_dg_and_QA"
                                    ]
                                    is True
                                ):
                                    signals = np.vstack(
                                        [target_signal_pp, source_signal_pp]
                                    ).T

                                    pairwise_gc_dict = gc_test(
                                        signals, [history_epoch_samples], verbose=False
                                    )

                                    # GC quality control, several measures on error distribution
                                    gc_stat_dg_data = pairwise_gc_dict[
                                        history_epoch_samples
                                    ][1]
                                    (
                                        het_passing,
                                        cook_passing,
                                        normality_passing,
                                        acorr_passing,
                                        vif_passing,
                                    ) = self._gc_model_diagnostics(
                                        gc_stat_dg_data,
                                        source_idx,
                                        target_idx,
                                        show_figure=False,
                                        save_gc_fit_dg_and_QA=True,
                                        iter_dict=iter_dict,
                                    )
                                    gc_fitQA = str(
                                        [
                                            het_passing,
                                            cook_passing,
                                            normality_passing,
                                            acorr_passing,
                                            vif_passing,
                                        ]
                                    ).count("PASS")
                                else:
                                    gc_fitQA = np.nan
                            except:
                                if logging.getLogger().hasHandlers():
                                    logging.error(
                                        f"\nFailed Granger Causality call, setting all GC analysis values to nan"
                                    )
                                F_value, p_value, latency, gc_fitQA = (
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                )

                        analyzed_data_dict[f"GrCaus_{target_group}_Information"][
                            source_idx, target_idx, file_idx
                        ] = np.log2(F_value)
                        analyzed_data_dict[f"GrCaus_{target_group}_p"][
                            source_idx, target_idx, file_idx
                        ] = p_value
                        analyzed_data_dict[f"GrCaus_{target_group}_Latency"][
                            source_idx, target_idx, file_idx
                        ] = latency
                        analyzed_data_dict[f"GrCaus_{target_group}_FitQuality"][
                            source_idx, target_idx, file_idx
                        ] = gc_fitQA

            # Remove the 'will' after successfull IxO loop
            if set(["grcaus"]).issubset(analyzes_lowercase_list) and Path.exists(
                self._tmp_regression_error_full_path
            ):
                Path.unlink(self._tmp_regression_error_full_path)

        data_dims = empty_np_array.shape
        analyzed_data_dict["data_dims"] = data_dims

        return analyzed_data_dict

    def _get_midpoint_parameter_dict(self, results_folder_suffix=""):
        """
        Collect a dictionary of folders containing the iterations for distinct inputs
        """

        # Get relevant midpoints, parameters and columns names
        coll_param_df = self.coll_mpa_dict["coll_param_df"]
        coll_mid_list = self.coll_mpa_dict["coll_mid_list"]

        # Get the full paths of analyzed iterations. If missing, abort
        midpoint_parameter_dict = {}
        not_found = []
        for midpoint in coll_mid_list:
            for parameter in coll_param_df.index.tolist():
                this_mid_par = midpoint + "_" + parameter
                full_paths_list = self.data_io.listdir_loop(
                    self.context.path, this_mid_par.lower(), None
                )
                # Include only folders and Remove earlier analyzes
                dir_full_paths_list = sorted(
                    [
                        f
                        for f in full_paths_list
                        if Path.is_dir(f) and not "_compiled_results" in str(f)
                    ]
                )

                # Exclude folders which do not terminate on numbers.
                # These are eg tests before iteration run.
                dir_full_paths_list_numsuffix = sorted(
                    [f for f in dir_full_paths_list if f.as_posix()[-1].isnumeric()]
                )

                # Folder names of iterations must end with a number after last underscore.
                # This enables results dictionaries in the same folder.
                # Exclude possible earlier created results_folder_suffix folder
                dir_full_paths_list_clean = [
                    f
                    for f in dir_full_paths_list_numsuffix
                    if this_mid_par + results_folder_suffix not in str(f)
                ]

                if dir_full_paths_list_clean:
                    midpoint_parameter_dict[this_mid_par] = dir_full_paths_list_clean
                else:
                    not_found.append(this_mid_par)

        # Report missing folder or abort
        assert (
            not not_found
        ), f"The following requested midpoint_parameter combinations were not found: {not_found}, aborting..."

        return midpoint_parameter_dict

    def _get_system_profile_metrics(self, data_df_compiled, independent_var_col_list):

        # Turn array analysis columns into system profile metrics

        def _column_comprehension(analysis, key_list=[]):

            relevant_columns = [
                col for col in data_df_compiled.columns if f"{analysis}" in col
            ]
            pruned_names = []
            if key_list is not []:
                key_columns = []
                for this_key in key_list:
                    this_key_column = [
                        col for col in relevant_columns if f"{this_key}" in col
                    ]
                    key_columns.extend(this_key_column)
                    pruned_names.append(f"{analysis}_{this_key}")
                relevant_columns = key_columns
            if pruned_names is []:
                pruned_names = [f"{analysis}"]
            return relevant_columns, pruned_names

        profile_metrics_columns_list = []

        # Init df_for_barplot with independent columns
        df_for_barplot = data_df_compiled[independent_var_col_list]

        ## Get energy metrics ##
        # From Attwell_2001_JCerBlFlMetab.pdf "As a simplification, all cells are treated as glutamatergic,
        # because excitatory neurons outnumber inhibitory cells by a factor of 9 to 1, and 90% of synapses
        # release glutamate (Abeles, 1991; Braitenberg and Schüz, 1998)."

        meanfr_column_list, pruned_names = _column_comprehension(
            "MeanFR", key_list=["NG1"]
        )
        selected_columns = data_df_compiled[meanfr_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        ## Get latency metrics ##
        latency_column_list, pruned_names = _column_comprehension(
            "Coherence", key_list=["Latency"]
        )
        selected_columns = data_df_compiled[latency_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        # latency_column_list, pruned_names = _column_comprehension('GrCaus', key_list=['latency'])
        # selected_columns = data_df_compiled[latency_column_list]
        # df_for_barplot[pruned_names] = selected_columns
        # profile_metrics_columns_list.extend(pruned_names)

        ## Get reconstruction metrics ##
        reco_column_list, pruned_names = _column_comprehension(
            "NormError", key_list=["SimErr"]
        )
        selected_columns = data_df_compiled[reco_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        # ## Get classification accuracy ##
        # classify_column_list, pruned_names = _column_comprehension('Classify', key_list=['Accuracy'])
        # selected_columns = data_df_compiled[classify_column_list]
        # df_for_barplot[pruned_names] = selected_columns
        # profile_metrics_columns_list.extend(pruned_names)

        ## Get information metrics ##
        info_column_list, pruned_names = _column_comprehension(
            "GrCaus", key_list=["Information"]
        )
        selected_columns = data_df_compiled[info_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        info_column_list, pruned_names = _column_comprehension(
            "TransferEntropy", key_list=["TransfEntropy"]
        )
        selected_columns = data_df_compiled[info_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        ## Get target entropy ##
        # This is missing after IxO analysis (secondary importance)
        try:
            target_entropy_column_list, pruned_names = _column_comprehension(
                "GrCaus", key_list=["TargetEntropy"]
            )
            selected_columns = data_df_compiled[target_entropy_column_list]
            df_for_barplot[pruned_names] = selected_columns
            profile_metrics_columns_list.extend(pruned_names)
        except ValueError:
            pass

        ## Get input-output coherence ##
        coherence_column_list, pruned_names = _column_comprehension(
            "Coherence", key_list=["Sum"]
        )
        selected_columns = data_df_compiled[coherence_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        min_values = df_for_barplot[profile_metrics_columns_list].min()
        max_values = df_for_barplot[profile_metrics_columns_list].max()

        return df_for_barplot, profile_metrics_columns_list, min_values, max_values

    # Main array analysis functions
    def get_PCA(
        self,
        data,
        n_components=2,
        col_names=None,
        extra_points=None,
        extra_points_at_edge_of_gamut=False,
    ):

        # scales to zero mean, sd of one
        values_np_scaled = self.scaler(data, scale_type="standard")

        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(values_np_scaled)

        if col_names is not None:
            index = col_names
        else:
            index = list(range(n_components))

        principal_axes_in_PC_space = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{pc}" for pc in range(n_components)],
            index=index,
        )
        explained_variance_ratio = pca.explained_variance_ratio_

        if extra_points is not None:
            pca_extra_points = pca.transform(extra_points)
        else:
            pca_extra_points = None

        if extra_points_at_edge_of_gamut is True:
            xmin, xmax, ymin, ymax = (
                np.min(pca_data[:, 0]),
                np.max(pca_data[:, 0]),
                np.min(pca_data[:, 1]),
                np.max(pca_data[:, 1]),
            )
            extra_points_np = np.zeros(pca_extra_points.shape)
            for rowidx, row in enumerate(pca_extra_points):
                x0, y0 = zip(row)
                # zip provides tuples, must get the floats
                x0, y0 = x0[0], y0[0]

                xyratio = np.abs(x0 / y0)
                if x0 < 0 and y0 >= 0:
                    xymaxratio = np.abs(xmin / ymax)
                    if xyratio >= xymaxratio:
                        xe = xmin
                        c = np.abs(xmin / x0)
                        ye = c * y0
                    else:
                        ye = ymax
                        c = np.abs(ymax / y0)
                        xe = c * x0
                if x0 >= 0 and y0 >= 0:
                    xymaxratio = np.abs(xmax / ymax)
                    if xyratio >= xymaxratio:
                        xe = xmax
                        c = np.abs(xmax / x0)
                        ye = c * y0
                    else:
                        ye = ymax
                        c = np.abs(ymax / y0)
                        xe = c * x0
                if x0 < 0 and y0 < 0:
                    xymaxratio = np.abs(xmin / ymin)
                    if xyratio >= xymaxratio:
                        xe = xmin
                        c = np.abs(xmin / x0)
                        ye = c * y0
                    else:
                        ye = ymin
                        c = np.abs(ymin / y0)
                        xe = c * x0
                if x0 >= 0 and y0 < 0:
                    xymaxratio = np.abs(xmax / ymin)
                    if xyratio >= xymaxratio:
                        xe = xmax
                        c = np.abs(xmax / x0)
                        ye = c * y0
                    else:
                        ye = ymin
                        c = np.abs(ymin / y0)
                        xe = c * x0
                extra_points_np[rowidx, :] = np.array([xe, ye])
            pca_extra_points = extra_points_np

        return (
            pca_data,
            principal_axes_in_PC_space,
            explained_variance_ratio,
            pca_extra_points,
        )

    def get_MSE(
        self,
        Input=None,
        results_filename=None,
        data_dict=None,
        simulation_engine="CxSystem",
        readout_group="NG1",
        decoding_method="least_squares",
        output_type="estimated",
    ):
        """
        Get decoding error. Allows both direct call by default method on single files and with data dictionary for array analysis.
        """

        if Input is None:
            Input = self.data_io.read_input_matfile(
                filename=self.context.input_filename, variable="stimulus"
            )
        # n_input_time_points = Input.shape[0]

        # Check readoutgroup name, standardize
        readout_group = self._check_readout_group(simulation_engine, readout_group)

        # Get filtered spike train from simulation.
        if "matlab" in simulation_engine.lower():  # Not in active use.
            if data_dict is None:
                assert (
                    results_filename is not None
                ), 'For matlab, you need to provide the workspace data as "results_filename", aborting...'
                # Input scaling factor A is 2000 for matlab results
                data_dict = self.data_io.get_data(filename=results_filename)
            # rOE missing in latest workspace, found rOEf instead
            target_rO_name = f"rO{readout_group}"
            spikes_leak = data_dict[target_rO_name].T
            Lambda = data_dict["lambda"]
            dt = data_dict["dt"]
            n_time_points = spikes_leak.shape[0]

        elif "cxsystem" in simulation_engine.lower():
            # Input scaling factor A is 15 for python results
            if data_dict is None:
                data_dict = self.data_io.get_data(
                    filename=results_filename, data_type="results"
                )
            n_time_points = self.cxparser.get_n_samples(data_dict)
            NG_name = [
                n for n in data_dict["spikes_all"].keys() if f"{readout_group}" in n
            ][0]
            n_neurons = data_dict["Neuron_Groups_Parameters"][NG_name][
                "number_of_neurons"
            ]

            # Get dt
            dt = self.cxparser.get_dt(data_dict)

            # Get Lambda, a.k.a. tau_soma, but in time points. Can be float.
            Lambda_unit = self.cxparser.get_namespace_variable(
                data_dict, readout_group, variable_name="taum_soma"
            )
            Lambda = Lambda_unit.base / dt

            # Get spikes
            spike_idx = data_dict["spikes_all"][NG_name]["i"]
            spike_times = data_dict["spikes_all"][NG_name]["t"]
            spike_times_idx = np.array(spike_times / (dt * b2u.second), dtype=int)

            # Create spike vector
            spikes = np.zeros([n_time_points, n_neurons])
            spikes[spike_times_idx, spike_idx] = 1
            # Spikes with leak
            spikes_leak = self._get_spikes_with_leak(spikes, Lambda, dt)

        # Get input with leak, a.k.a. the target output
        input_leak = self._get_input_with_leak(Input, Lambda, dt)
        target_output = input_leak

        # Cut start and end time points by request
        # Cut spikes_leak and target_output
        spikes_leak_cut = spikes_leak[self.context.t_idx_start : self.context.t_idx_end]
        target_output_cut = target_output[
            self.context.t_idx_start : self.context.t_idx_end
        ]

        # Get output
        assert output_type in [
            "estimated",
            "simulated",
        ], 'Unknown output type, should be "estimated" or "simulated", aborting...'
        if output_type == "estimated":
            # Get optimal decoders with analytical method. This is the best possible outputsignal, given the leaky spikes
            Decs = self._get_optimal_decoders(
                target_output_cut, spikes_leak_cut, decoding_method
            )
            # Get estimated output
            estimated_output = np.dot(Decs.T, spikes_leak_cut.T)
            output = estimated_output.T
        elif output_type == "simulated":
            # Get simulated vm values for target group. Only valid for cxsystem data.
            simulated_output_vm_unit = self.cxparser.get_vm_by_interval(
                data_dict,
                NG=self.context.NG_name,
                t_idx_start=self.context.t_idx_start,
                t_idx_end=self.context.t_idx_end,
            )
            simulated_output_vm = (
                simulated_output_vm_unit / simulated_output_vm_unit.get_best_unit()
            )
            # both output and target data are scaled to -1,1 for comparison. Minmax keeps distribution histogram form intact.
            simulated_output = self.scaler(simulated_output_vm, scale_type="minmax")
            output = simulated_output
            target_output_cut = self.scaler(target_output_cut, scale_type="minmax")

        Error = self._get_normalized_error_variance(target_output_cut, output)

        return Error, target_output_cut, output

    def analyze_arrayrun(self, metadata_filename=None, analysis=None, iter_dict=None):
        """
        Analyze simulation parameter array run. Create csv table. Needs a metadata file.
        :param metadata_filename: string
        :param analysis: string, analysis name keyword
        :param iter_dict, None or dict, if dict exists, this is parallel analysis
        Saves analysis results as [analysis name keyword]_timestamp.csv to data folder.
        """
        # Map to standard camelcase
        assert (
            analysis.lower() in self.map_ana_names.keys()
        ), "Analysis type not found, aborting..."
        analysisHR = self.map_ana_names[analysis.lower()]

        if iter_dict is None:
            # Get metadata and output filenames
            meta_full = self.data_io.parse_path(
                metadata_filename, data_type="metadata", exclude="cluster_metadata"
            )
        else:
            meta_full = iter_dict["_meta_fname_full"]
        metadata_filename, data_df = self.update_metadata(meta_full)

        metadataroot = Path.joinpath(meta_full.parent, meta_full.stem)
        filename_out = str(metadataroot).replace("metadata", analysisHR)
        filename_out = str(filename_out).replace("_updated", "")
        csv_name_out = Path(filename_out + ".csv")

        if "grcaus" in analysisHR.lower():
            self._init_grcaus_regression_QA(iter_dict=iter_dict)

        analyzed_data_df = self._get_analyzed_array_as_df(
            data_df, analysisHR=analysisHR, iter_dict=iter_dict
        )

        # Drop Full path column for concise printing
        analyzed_data_df = analyzed_data_df.drop(["Full path"], axis=1)

        # # Display values
        self.pp_df_full(analyzed_data_df)

        analyzed_data_df.to_csv(csv_name_out, index=False)

    def analyze_IxO_array(
        self, metadata_filename=None, analyzes_list=None, iter_dict=None
    ):
        """
        Analyze simulation parameter array run. Create csv table. Needs a metadata file.
        :param metadata_filename: string
        :param analyzes_list: list of strings, analysis name keywords
        :param iter_dict, None or dict, if dict exists, this is parallel analysis
        Saves results as [analysis #1 name_analysis #2 name_ ... keywords]_timestamp.gz to data folder.
        This holds the N input x N output numpy matrix
        """

        # Map from standard camelcase to lowercase.
        analysis_filename_prefix = "IxO_analysis_"

        # Validate IxO analyzes, log others for exclusion

        valid_ixo_ana_list = []
        for this_analysis in analyzes_list:
            if this_analysis.lower() in self.map_ixo_names.keys():
                valid_ixo_ana_list.append(this_analysis)
            else:
                logging.info(
                    f"SKIPPED ANALYSIS {this_analysis} -- not valid IxO analysis type"
                )

        # Get metadata and output filenames
        if iter_dict is None:
            # Get metadata and output filenames
            meta_full = self.data_io.parse_path(
                metadata_filename, data_type="metadata", exclude="cluster_metadata"
            )
        else:
            meta_full = iter_dict["_meta_fname_full"]

        metadata_filename, data_df = self.update_metadata(meta_full)

        metadataroot = Path.joinpath(meta_full.parent, meta_full.stem)
        filename_out = str(metadataroot).replace("metadata_", analysis_filename_prefix)
        IxO_dict_filename_out = Path(filename_out + ".gz")

        # Init txt file for Granger causality fit quality diagnostics
        if "grcaus" in analysis_filename_prefix.lower():
            self._init_grcaus_regression_QA(iter_dict=iter_dict)

        # if full analysis exists, load, otherwise calculate (slow)
        _do_analysis = True
        if IxO_dict_filename_out.is_file():
            analyzed_data_dict = self.data_io.load_from_file(IxO_dict_filename_out)
            existing_ana_list = [
                a
                for a in valid_ixo_ana_list
                for k in analyzed_data_dict.keys()
                if k.startswith(a)
            ]
            # if it does not match requested analyzes, redo them
            missing_ana_set = set(valid_ixo_ana_list).difference(set(existing_ana_list))
            if len(missing_ana_set) > 0:
                _do_analysis = True
                if logging.getLogger().hasHandlers():
                    logging.info(
                        f"\Reanalyzing despite existing {IxO_dict_filename_out}, because some analyses were missing..."
                    )
            else:
                _do_analysis = False
                if logging.getLogger().hasHandlers():
                    logging.info(
                        f"\nReloading analyzed_data_dict from existing {IxO_dict_filename_out}..."
                    )
        if _do_analysis is True:
            analyzed_data_dict = self._get_analyzed_full_array_as_np(
                data_df, analyzes_list=valid_ixo_ana_list, iter_dict=iter_dict
            )
            self.data_io.write_to_file(IxO_dict_filename_out, analyzed_data_dict)

        # Create distinct csv:s for each analysis in a loop
        data_dims = analyzed_data_dict["data_dims"][:2]
        idx_dim_0, idx_dim_1 = np.ogrid[: data_dims[0], : data_dims[1]]
        assert (
            data_dims[0] == data_dims[1]
        ), "Uneven input and output dimensions' analyzes are not implemented yet, aborting..."
        for analysisHR in valid_ixo_ana_list:

            # Drop Full path column for concise printing
            analyzed_data_df = data_df.drop(["Full path"], axis=1).copy()
            stat_dict = self.map_stat_types[analysisHR]
            # Add columns
            column_names = [n for n in analyzed_data_dict.keys() if analysisHR in n]
            for this_column in column_names:
                diagonal_data_np_3D = analyzed_data_dict[this_column][
                    idx_dim_1, idx_dim_1, :
                ]
                diagonal_data_np_2D = np.squeeze(diagonal_data_np_3D, 0)

                # Map stat types
                this_stat_list = [
                    stat for key, stat in stat_dict.items() if key in this_column
                ]
                if this_stat_list[0] == "median":
                    analyzed_data_df[this_column] = np.nanmedian(
                        diagonal_data_np_2D, axis=0
                    )
                elif this_stat_list[0] == "mean":
                    analyzed_data_df[this_column] = np.nanmean(
                        diagonal_data_np_2D, axis=0
                    )
                else:
                    raise NotImplementedError(
                        "Only 'median' and 'mean' are implemented, aborting..."
                    )

            # Save csv
            filename_out = str(metadataroot).replace("metadata", analysisHR)
            csv_name_out = Path(filename_out + ".csv")
            analyzed_data_df.to_csv(csv_name_out, index=False)

    def analyze_TE_drift(self, metadata_filename=None):
        """
        Analyze simulation parameter array run when input is shifting. Create csv table. Needs a metadata file.
        :param metadata_filename: string
        :param analyzes_list: list of strings, analysis name keywords
        Saves results as [analysis #1 name]_latencyShift_timestamp.csv to data folder.
        Saves results as [analysis #1 name].gz to data folder.
        This holds the N input x N output x N timeshifts x N param_variations numpy matrix
        """
        # Only TE analysis implemented, we call it tedrift here
        analyzes_list = ["TEDrift"]

        # Map from standard camelcase to lowercase.
        analysis_filename_prefix = "TEDrift_analysis_"

        for this_analysis in analyzes_list:
            assert (
                this_analysis.lower() in self.map_ana_names.keys()
            ), "Analysis type not found, aborting..."
            analysis_filename_prefix += self.map_ana_names[this_analysis.lower()] + "_"

        # Get metadata and output filenames
        meta_full = self.data_io.parse_path(
            metadata_filename, data_type="metadata", exclude="cluster_metadata"
        )

        metadata_filename, data_df = self.update_metadata(meta_full)

        metadataroot = Path.joinpath(meta_full.parent, meta_full.stem)
        filename_out = str(metadataroot).replace("metadata_", analysis_filename_prefix)
        TEDrift_dict_filename_out = Path(filename_out + ".gz")

        # if full analysis exists, load, otherwise calculate (slow)
        if TEDrift_dict_filename_out.is_file():
            analyzed_data_dict = self.data_io.load_from_file(TEDrift_dict_filename_out)
            if logging.getLogger().hasHandlers():
                logging.info(
                    f"\nReloading analyzed_data_dict from existing {TEDrift_dict_filename_out}"
                )
        else:
            t_idx_start = self.context.t_idx_start
            t_idx_end = self.context.t_idx_end
            analyzed_data_dict = self._get_analyzed_full_array_as_np(
                data_df, analyzes_list=analyzes_list
            )
            self.data_io.write_to_file(TEDrift_dict_filename_out, analyzed_data_dict)
        # Create distinct csv:s for each analysis in a loop
        data_dims = analyzed_data_dict["data_dims"][:2]
        te_timeshifts_in_ms_np = analyzed_data_dict["te_timeshifts_in_ms_np"]
        idx_dim_0, idx_dim_1 = np.ogrid[: data_dims[0], : data_dims[1]]
        assert (
            data_dims[0] == data_dims[1]
        ), "Uneven input and output dimensions' analyzes are not implemented yet, aborting..."
        for analysisHR in analyzes_list:
            # Drop Full path column for concise printing
            analyzed_data_df = data_df.drop(["Full path"], axis=1).copy()

            stat_dict = self.map_stat_types[analysisHR]
            column_names = [n for n in analyzed_data_dict.keys() if analysisHR in n]
            for shift_idx, shift_in_millisec in enumerate(te_timeshifts_in_ms_np):
                # Add columns
                for this_column in column_names:
                    diagonal_data_np_3D = analyzed_data_dict[this_column][
                        idx_dim_1, idx_dim_1, :, shift_idx
                    ]
                    diagonal_data_np_2D = np.squeeze(diagonal_data_np_3D, 0)

                    # Map stat types
                    col_name_w_shift = f"{this_column}_shift{str(shift_in_millisec)}ms"
                    this_stat_list = [
                        stat for key, stat in stat_dict.items() if key in this_column
                    ]
                    if this_stat_list[0] == "median":
                        analyzed_data_df[col_name_w_shift] = np.nanmedian(
                            diagonal_data_np_2D, axis=0
                        )
                    elif this_stat_list[0] == "mean":
                        analyzed_data_df[col_name_w_shift] = np.nanmean(
                            diagonal_data_np_2D, axis=0
                        )
                    else:
                        raise NotImplementedError(
                            "Only 'median' and 'mean' are implemented, aborting..."
                        )

                # Save csv
                filename_out = str(metadataroot).replace("metadata", analysisHR)
                csv_name_out = Path(filename_out + ".csv")
                analyzed_data_df.to_csv(csv_name_out, index=False)

    def collate_best_values(
        self,
        midpoint=None,
        parameter=None,
        analyzes=None,
        save_as_csv=True,
        output_folder=None,
    ):
        # Open given analyzes csv:s, identify correct columns, get best values
        # Write csv with best values and their coordinates in it
        if analyzes is None:
            analyzes = {
                "NormError_SimErr": "min",
                "TransferEntropy_NG3_L4_SS_L4_TransfEntropy": "max",
                "GrCaus_NG3_L4_SS_L4_Information": "max",
                "Coherence_NG3_L4_SS_L4_Sum": "max",
            }

        # Get midpoint, parameter, and update output folder if it exists
        if midpoint is None:
            midpoint = self.context.midpoint
        if parameter is None:
            parameter = self.context.parameter
        if output_folder is not None:
            self.context.output_folder = output_folder

        column_initials = ""

        # Init df for collated data
        collated_best_values_df = pd.DataFrame(
            columns=[
                "midpoint",
                "parameter",
                "analyzes",
                "best_param_x",
                "best_param_y",
                "best_value",
                "best_is",
                "axis_corners",
            ]
        )

        for idx, this_column in enumerate(analyzes.keys()):
            column_initials += this_column[0]
            first_underscore_idx = this_column.find("_")
            analysis_name = this_column[:first_underscore_idx]
            df = self.data_io.get_data(data_type=analysis_name)

            (
                best_value,
                at_param_list,
                axis_corners,
                x_values,
                y_values,
            ) = self._get_best_values(df, this_column, best_is=analyzes[this_column])
            collated_best_values_df.loc[idx] = [
                midpoint,
                parameter,
                this_column,
                at_param_list[0],
                at_param_list[1],
                best_value,
                analyzes[this_column],
                axis_corners,
            ]

        if save_as_csv is True:
            # Save df of collated data
            full_filename_out = Path.joinpath(
                self.context.output_folder, f"best_of_{column_initials}.csv"
            )
            collated_best_values_df.to_csv(full_filename_out, index=False)
        else:
            return collated_best_values_df, x_values, y_values

    def compile_analyzes_over_iterations(self, analysis_type="mean"):
        """
        Analyze replications of arrays using analyzed data from iteration data folders. Two types of analysis:
        :param analysis_type: str, 'mean' or 'IxO_accuracy'
        Analysis 'mean' uses csv files which already contain a scalar mean value for each data file
        Analysis 'IxO_accuracy' use the IxO_analysis*.gz files containing the IxOxN_data_files analysis matrices.
        """

        results_folder_suffix = "_compiled_results"
        midpoint_parameter_dict = self._get_midpoint_parameter_dict(
            results_folder_suffix=results_folder_suffix
        )

        coll_ana_df = self.coll_mpa_dict["coll_ana_df"]
        csv_col_list = coll_ana_df["csv_col_name"].values.tolist()

        # Collect iterations from mapped folders to one numpy array. First dim are iterations (different trials)
        for this_mid_par in midpoint_parameter_dict.keys():

            first_path = midpoint_parameter_dict[this_mid_par][0]
            basepath, folder_name = first_path.parent, first_path.name

            # Make output folders
            this_statpath = Path.joinpath(
                basepath, this_mid_par + results_folder_suffix
            )
            if not Path.exists(this_statpath):
                Path.mkdir(this_statpath, parents=True)

            # Load the first set of csv files to get a list of data analysis column names, number of iterations,
            # and init df with data from the independent parameter variation columns
            (
                data0_df,
                data_df_compiled,
                independent_var_col_list,
                dependent_var_col_list,
                time_stamp,
            ) = self.data_io.get_csv_as_df(folder_name=folder_name)
            data_columns_list = list(
                sorted(set(csv_col_list) & set(dependent_var_col_list))
            )
            # Capture TEDrift with multiple timeshifts in col names
            if not data_columns_list and "TEDrift" in csv_col_list[0]:
                data_columns_list = [
                    c for c in dependent_var_col_list for s in csv_col_list if s in c
                ]
            n_iter = len(midpoint_parameter_dict[this_mid_par])
            ini_stats_df = data_df_compiled[independent_var_col_list]

            if analysis_type in ["mean", "TEDrift"]:
                # init numpy array and final df
                init_mtx_np = data_df_compiled[data_columns_list].to_numpy()
                # This should allow both 1 and 2 dim searches
                dims_list = [x for x in init_mtx_np.shape]
                dims_list.insert(0, n_iter)

                collated_mtx_np = np.full(dims_list, np.nan)

                # Collect midpoint_parameter iteration data
                for iter_idx, this_path in enumerate(
                    midpoint_parameter_dict[this_mid_par]
                ):
                    (
                        data0_df,
                        data_df_compiled,
                        independent_var_col_list,
                        dependent_var_col_list,
                        time_stamp,
                    ) = self.data_io.get_csv_as_df(csv_path=this_path)
                    collated_mtx_np[iter_idx, :, :] = data_df_compiled[
                        data_columns_list
                    ].to_numpy()

                stat_df = ini_stats_df.copy()

                # Calculate statistics and save to csv:s
                # Mean
                stat_np = np.nanmean(collated_mtx_np, axis=0)
                data_columns_stat = [c + "_mean" for c in data_columns_list]
                stat_df[data_columns_stat] = stat_np

                # # Standard deviation
                # stat_np = np.nanstd(collated_mtx_np,axis=0)
                # data_columns_stat = [c + '_std' for c in data_columns_list]
                # stat_df[data_columns_stat] = stat_np

            elif analysis_type == "IxO_accuracy":

                first_ixo_full_filename = self.data_io.listdir_loop(
                    first_path, "IxO_analysis".lower(), None
                )
                first_ixo_dict = self.data_io.load_from_file(first_ixo_full_filename[0])
                first_ixo_mtx_np = first_ixo_dict[data_columns_list[0]]
                dims_ixo = first_ixo_mtx_np.shape[:2]
                n_data_files = first_ixo_mtx_np.shape[2]
                true_ixo = np.eye(dims_ixo[0], dims_ixo[1])
                ini_mtx_np = np.full([n_iter, dims_ixo[0], dims_ixo[1]], np.nan)
                # Make analyzes x data_files x iterations x ixo_dim0 x ixo_dim1 matrix
                all_data_mtx_np = np.full(
                    [
                        len(data_columns_list),
                        n_data_files,
                        n_iter,
                        dims_ixo[0],
                        dims_ixo[1],
                    ],
                    np.nan,
                )

                # loop iter folder full paths, get ixo_mtx_np data
                for this_iteration_idx, this_iteration_folder in enumerate(
                    midpoint_parameter_dict[this_mid_par]
                ):
                    data_type_str = "IxO_analysis"
                    ixo_full_filename_list = self.data_io.listdir_loop(
                        this_iteration_folder, data_type_str.lower(), None
                    )

                    if len(ixo_full_filename_list) > 1:
                        print(
                            f"\nFound {len(ixo_full_filename_list)} files in folder {this_iteration_folder}. Continuing with most recent..."
                        )
                        if logging.getLogger().hasHandlers():
                            logging.info(
                                f"\nFound {len(ixo_full_filename_list)} files in folder {this_iteration_folder}. Continuing with most recent..."
                            )
                        data_fullpath_filename = self.data_io.most_recent(
                            this_iteration_folder, data_type=data_type_str, exclude=None
                        )
                        ixo_full_filename_list = [data_fullpath_filename]

                    # There should be only one file starting IxO_analysis
                    ixo_dict = self.data_io.load_from_file(ixo_full_filename_list[0])
                    print(f"Loaded file {ixo_full_filename_list[0]}")
                    for this_analysis_idx, this_analysis in enumerate(
                        data_columns_list
                    ):
                        this_data = ixo_dict[this_analysis]
                        this_data_reshaped = np.moveaxis(this_data, 2, 0)
                        # Fill analyzes x data_files x iterations x ixo_dim0 x ixo_dim1 matrix
                        all_data_mtx_np[
                            this_analysis_idx, :, this_iteration_idx, :, :
                        ] = this_data_reshaped

                stat_df = ini_stats_df.copy()
                data_columns_ixo = []
                data_columns_ixo += [c + "_accuracy" for c in data_columns_list]
                data_columns_ixo += [c + "_p" for c in data_columns_list]
                data_columns_ixo += [c + "_conf_min" for c in data_columns_list]
                data_columns_ixo += [c + "_conf_max" for c in data_columns_list]

                # Init all data to NaN values
                for this_column in data_columns_ixo:
                    stat_df[this_column] = np.nan

                for this_analysis_idx, this_analysis in enumerate(data_columns_list):
                    for this_data_file_idx in range(n_data_files):

                        # this_mtx_np = ini_mtx_np.copy()
                        this_ixo_np = all_data_mtx_np[
                            this_analysis_idx, this_data_file_idx, :, :, :
                        ]

                        # Get IxO [0,1] according to best_is (min for error, max for info etc)
                        best_is = coll_ana_df.loc[
                            coll_ana_df["csv_col_name"] == this_analysis
                        ]["best_is"].values[0]
                        # Find best values row-wise
                        if best_is == "max":
                            if "TransfEntropy" in this_analysis:
                                this_ixo_np = np.nan_to_num(
                                    this_ixo_np, copy=True, nan=0.0
                                )
                            this_ixo_bool = (
                                this_ixo_np == this_ixo_np.max(axis=1)[:, None]
                            ).astype(int)
                        elif best_is == "min":
                            this_ixo_bool = (
                                this_ixo_np == this_ixo_np.min(axis=1)[:, None]
                            ).astype(int)
                        this_ixo_sum = this_ixo_bool.sum(axis=0)

                        classtest = Classifierbinomial(
                            this_ixo_sum, key_matrix=true_ixo
                        )
                        p_value = classtest.binomial_probability()
                        alpha_confidence = 0.05
                        (
                            accuracy,
                            lower_bound,
                            upper_bound,
                        ) = classtest.accuracy_confidence_interval(alpha_confidence)
                        this_analysis_columns_list = [
                            c for c in stat_df.columns if this_analysis in c
                        ]
                        stat_df.loc[this_data_file_idx, this_analysis_columns_list] = (
                            accuracy,
                            p_value,
                            lower_bound,
                            upper_bound,
                        )

            # Save to midpoint_parameter -named folders with names including analysis column title prefixes
            prefix_list = [ana[: ana.find("_")] + "_" for ana in data_columns_list]
            uniques_prefix_list = sorted(set(prefix_list))
            csv_name_prefix = "".join(
                ana[: ana.find("_")] + "_" for ana in uniques_prefix_list
            )

            full_csv_pathname_out = Path.joinpath(
                this_statpath, csv_name_prefix + analysis_type + ".csv"
            )
            stat_df.to_csv(full_csv_pathname_out, index=False)

    def optimal_value_analysis(self, analyze=None, delay=0, analog_input=None):
        """
        Analyze best possible value for each input.
        :param analyze: string, 'Coherence', 'Granger Causality', 'GC as TE', 'Transfer Entropy', 'Simulation Error'
        :param delay: int, delay in time steps
        """

        dt = 0.0001  # in seconds

        # NOTE whenever you request an array of delays (a.k.a. delays is a list),
        # you return from this if-statement.
        if isinstance(delay, list):
            delay_array = np.floor(np.linspace(delay[0], delay[1], delay[2])).astype(
                int
            )
            delay_array_in_milliseconds = delay_array * dt * 1000
            value_array = np.zeros(len(delay_array))
            noise_array = np.zeros(len(delay_array))
            if analog_input is None:
                analog_input = self.data_io.get_data(
                    self.context.input_filename, data_type=None
                )

            tic = time.time()
            for this_idx, this_delay in enumerate(delay_array):
                (
                    value_array[this_idx],
                    noise_array[this_idx],
                ) = self.optimal_value_analysis(
                    analyze=analyze, delay=this_delay, analog_input=analog_input
                )
            toc = time.time()
            delay_report = f"\nOptimal value analysis took\n{toc - tic:.2f} seconds"

            return (
                (delay_array_in_milliseconds, value_array, noise_array, delay_report),
                analyze,
                "value_vs_delay",
            )

        if analyze is None:
            raise ValueError("analyze must be specified")

        # This is why I hate recursion...
        if analog_input is None:
            analog_input = self.data_io.get_data(
                self.context.input_filename, data_type=None
            )

        source_signal = analog_input["stimulus"].T

        target_signal = source_signal

        # Delay target signal dimension 0 by delay
        target_signal = np.roll(target_signal, delay, axis=0)

        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        # Cut to requested length
        source_signal_cut = source_signal[t_idx_start:t_idx_end, :]
        target_signal_cut = target_signal[t_idx_start:t_idx_end, :]

        high_cutoff = 100  # Frequency in Hz
        nsamples = source_signal_cut.shape[0]
        nperseg = nsamples // 6
        samp_freq = 1.0 / dt

        value_np = (
            np.zeros((source_signal_cut.shape[1], target_signal_cut.shape[1])) * np.nan
        )
        # Use identity matrix as the index matrix.
        idx_mtx = np.eye(source_signal.shape[1], target_signal.shape[1], dtype=bool)

        # Get best possible value for each input
        if analyze == "Coherence":
            # Loop over source and target dimension 1.
            # Get coherence of the two signals.
            # Divide the coherence_sum by the length if Cxy.
            # optimal value is the nanmean of this divided sum.
            for source_idx in range(source_signal_cut.shape[1]):
                for target_idx in range(target_signal_cut.shape[1]):

                    (
                        f,
                        Cxy,
                        Pwelch_spec_x,
                        Pwelch_spec_y,
                        Pxy,
                        lags,
                        corr,
                        coherence_sum,
                        _source_scaled,
                        _target_scaled,
                    ) = self.get_coherence_of_two_signals(
                        source_signal_cut[:, source_idx],
                        target_signal_cut[:, target_idx],
                        samp_freq=samp_freq,
                        nperseg=nperseg,
                        high_cutoff=high_cutoff,
                    )

                    value_np[source_idx, target_idx] = np.sum(Cxy) / len(Cxy)

        elif analyze == "Granger Causality":

            ic = "aic"  # Akaike information criterion
            max_time_lag_seconds = self.extra_ana_args["GrCaus_args"][
                "max_time_lag_seconds"
            ]
            downsampling_factor = self.extra_ana_args["GrCaus_args"][
                "downsampling_factor"
            ]

            source_signal_pp, target_signal_pp = self._preprocess_for_info_analyzes(
                source_signal_cut,
                target_signal_cut,
                downsampling_factor,
                nsamples,
                t_idx_start=0,
                t_idx_end=None,
            )
            F_np = (
                np.zeros((source_signal_cut.shape[1], target_signal_cut.shape[1]))
                * np.nan
            )
            for source_idx in range(source_signal_cut.shape[1]):
                for target_idx in range(target_signal_cut.shape[1]):
                    (
                        F_np[source_idx, target_idx],
                        foo,
                        foo2,
                    ) = self._granger_causality(
                        target_signal_pp[:, target_idx],
                        source_signal_pp[:, source_idx],
                        max_time_lag_seconds,
                        ic,
                        dt,
                        downsampling_factor,
                        verbose=False,
                    )

            value_np = np.log2(F_np)  # From F-statistics to information

        elif analyze == "Transfer Entropy":

            max_time_lag_seconds = self.extra_ana_args["TE_args"][
                "max_time_lag_seconds"
            ]
            downsampling_factor = self.extra_ana_args["TE_args"]["downsampling_factor"]
            embedding_vector = self.extra_ana_args["TE_args"]["embedding_vector"]
            n_states = self.extra_ana_args["TE_args"]["n_states"]

            for source_idx in range(source_signal_cut.shape[1]):
                for target_idx in range(target_signal_cut.shape[1]):
                    (
                        f,
                        Cxy,
                        Pwelch_spec_x,
                        Pwelch_spec_y,
                        Pxy,
                        lags,
                        corr,
                        coherence_sum,
                        _source_scaled,
                        _target_scaled,
                    ) = self.get_coherence_of_two_signals(
                        source_signal_cut[:, source_idx],
                        target_signal_cut[:, target_idx],
                        samp_freq=samp_freq,
                        nperseg=nperseg,
                    )
                    shift_in_seconds = self.get_cross_corr_latency(lags, corr, dt)

                    # print(f"Shift in seconds: {shift_in_seconds}")
                    if shift_in_seconds > max_time_lag_seconds:
                        # If we exceed time lag, we assume that cross correlations peak is due to mainly noise
                        continue

                    shift_in_samples = int(shift_in_seconds // dt)

                    # shift original source, anticipate k-1 in embedding vector and re-cut.
                    # One sample in downsampled space equals the downsampling factor
                    source_signal_shifted = source_signal[
                        t_idx_start
                        - shift_in_samples
                        + downsampling_factor : t_idx_end
                        - shift_in_samples
                        + downsampling_factor,
                        source_idx,
                    ]

                    (
                        source_signal_pp,
                        target_signal_pp,
                    ) = self._preprocess_for_info_analyzes(
                        source_signal_shifted,
                        target_signal_cut[:, target_idx],
                        downsampling_factor,
                        nsamples,
                        t_idx_start=0,
                        t_idx_end=None,
                    )

                    value_np[source_idx, target_idx] = self._pin_transfer_entropy(
                        target_signal_pp,
                        source_signal_pp,
                        embedding_vector,
                        n_states,
                    )

        elif analyze == "GC as TE":
            ic = "aic"  # Akaike information criterion
            max_time_lag_seconds = self.extra_ana_args["GrCaus_args"][
                "max_time_lag_seconds"
            ]
            downsampling_factor = self.extra_ana_args["GrCaus_args"][
                "downsampling_factor"
            ]

            max_lag_sec = (
                dt * downsampling_factor
            )  # One sample or zero sample, test , dt * downsampling_factor.

            F_np = (
                np.zeros((source_signal_cut.shape[1], target_signal_cut.shape[1]))
                * np.nan
            )
            for source_idx in range(source_signal_cut.shape[1]):
                for target_idx in range(target_signal_cut.shape[1]):

                    (
                        f,
                        Cxy,
                        Pwelch_spec_x,
                        Pwelch_spec_y,
                        Pxy,
                        lags,
                        corr,
                        coherence_sum,
                        _source_scaled,
                        _target_scaled,
                    ) = self.get_coherence_of_two_signals(
                        source_signal_cut[:, source_idx],
                        target_signal_cut[:, target_idx],
                        samp_freq=samp_freq,
                        nperseg=nperseg,
                    )
                    shift_in_seconds = self.get_cross_corr_latency(lags, corr, dt)

                    if shift_in_seconds > max_time_lag_seconds:
                        # If we exceed time lag, we assume that cross correlations peak is due to mainly noise
                        continue

                    shift_in_samples = int(shift_in_seconds // dt)

                    # shift original source, anticipate k-1 in embedding vector and re-cut.
                    # One sample in downsampled space equals the downsampling factor
                    source_signal_shifted = source_signal[
                        t_idx_start
                        - shift_in_samples
                        + downsampling_factor : t_idx_end
                        - shift_in_samples
                        + downsampling_factor,
                        source_idx,
                    ]

                    (
                        source_signal_pp,
                        target_signal_pp,
                    ) = self._preprocess_for_info_analyzes(
                        source_signal_shifted,
                        target_signal_cut[:, target_idx],
                        downsampling_factor,
                        nsamples,
                        t_idx_start=0,
                        t_idx_end=None,
                    )

                    (
                        F_np[source_idx, target_idx],
                        foo,
                        foo2,
                    ) = self._granger_causality(
                        target_signal_pp,
                        source_signal_pp,
                        max_lag_sec,
                        ic,
                        dt,
                        downsampling_factor,
                        verbose=False,
                    )

            value_np = np.log2(F_np)  # From F-statistics to information

        elif analyze == "Simulation Error":
            for source_idx in range(source_signal_cut.shape[1]):
                for target_idx in range(target_signal_cut.shape[1]):

                    output = self.scaler(
                        target_signal_cut[:, target_idx], scale_type="minmax"
                    )
                    target = self.scaler(
                        source_signal_cut[:, source_idx], scale_type="minmax"
                    )

                    value_np[
                        source_idx, target_idx
                    ] = self._get_normalized_error_variance(target, output)

        # Take nanmean of the value_np matrix.
        optimal_value = np.nanmean(value_np[idx_mtx])
        nonoptimal_value = np.nanmean(value_np[np.invert(idx_mtx)])
        # print(f"optimal value for {analyze} is {optimal_value}")

        if isinstance(delay, int):
            # In value_np turn nan values to zeros.
            value_np[np.isnan(value_np)] = 0
            # In value_np turn inf values to zeros.
            value_np[np.isinf(value_np)] = 0
            # In value_np turn -inf values to zeros.
            value_np[np.isneginf(value_np)] = 0
            # In value_np turn -inf values to zeros.
            value_np[np.isposinf(value_np)] = 0

            return value_np, analyze, "full_mtx"

        return optimal_value, nonoptimal_value

    def describe_optimal_values(self,folderpath=None, savename=None):
        '''
        Get csv files from folderpath as dataframe.
        For the following analysis, use returned data_df_compiled
        Get description of the dataframe
        Save the description into folderpath
        '''

        # Get csv files from folderpath as dataframe.
        # get list of csv files in folderpath
        paths_list = self.data_io.listdir_loop(folderpath, data_type='csv', exclude='description')
        # get dataframes from full paths in the paths_list
        compiled_df = pd.DataFrame()
        for this_csv in paths_list:
            this_df = pd.read_csv(this_csv)
            compiled_df = pd.concat([compiled_df, this_df], axis=0, ignore_index=True)

        # Get description of the dataframe
        compiled_df_desc = compiled_df.describe()
        # Drop delay_in_ms
        compiled_df_desc = compiled_df_desc.drop(columns=['delay_in_ms'])

        # If savename is None, print to console
        if savename is None or savename == '':
            print(compiled_df_desc)
        # Save the description into folderpath
        elif folderpath is not None:
            compiled_df_desc.to_csv(
                Path.joinpath(folderpath, savename), index=True
            )
        else:
            folderpath = self.context.path
            compiled_df_desc.to_csv(
                Path.joinpath(folderpath, savename), index=True
            )


if __name__ == "__main__":

    # path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'

    # analysis = Analysis(path=path)
    # NG_name = 'NG3_L4_SS2_L4'

    # analysis.show_readout_on_input(NG_name, normalize=False, filename='Replica_test_results_20210114_1750000.gz')
    # analysis.show_spikes(filename='Replica_test_results_20210114_1750000.gz')

    # plt.show()
    pass
