"""
Analysis of retinal model ganglion cell spiking responses.
Contrast response function: Lee_1990_JOSA
Contrast sensitivity: Enroth-Cugell_1966_JPhysiol
Amplitude sensitivity: Lee_1990_JOSA
Receptive field: Chichilnisky_2001_Network
Fano factor: Uzzell_2004_JNeurophysiol
ISI analysis: : Uzzell_2004_JNeurophysiol
Temporal correlation: Greschner_2011_JPhysiol
Spatial correlation: Greschner_2011_JPhysiol
"""

# Numerical
import numpy as np
from scipy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt

# Comput Neurosci
import brian2.units as b2u

# Local
from cxsystem2.core.tools import write_to_file, load_from_file
from analysis.analysis_base_module import AnalysisBase

# Builtin
import pdb
import os
from pathlib import Path


class Analysis(AnalysisBase):
    # self.context. attributes
    _properties_list = [
        "path",
        "output_folder",
        "input_filename",
    ]

    def __init__(self, context, data_io, **kwargs) -> None:
        self._context = context.set_context(self._properties_list)
        self._data_io = data_io

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    def _get_spikes_by_interval(self, data, trial, t_start, t_end):
        key_name = f"spikes_{trial}"
        data_by_trial = data[key_name]

        idx_mask = np.logical_and(
            data_by_trial[1] > t_start * b2u.second,
            data_by_trial[1] < t_end * b2u.second,
        )

        spike_units = data_by_trial[0][idx_mask]
        spike_times = data_by_trial[1][idx_mask]

        return spike_units, spike_times

    def _analyze_meanfr(self, data, trial, t_start, t_end):
        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]
        mean_fr = times.size / (N_neurons * (t_end - t_start))

        return mean_fr

    def _analyze_unit_fr(self, data, trial, t_start, t_end):
        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]

        # Get firing rate for each neuron
        fr = np.zeros(N_neurons)
        for this_unit in range(N_neurons):
            unit_mask = units == this_unit
            times_unit = times[unit_mask]
            fr[this_unit] = times_unit.size / (t_end - t_start)

        return fr, N_neurons

    def _fourier_amplitude(
        self, data, trial, t_start, t_end, temp_freq, bins_per_cycle=32
    ):
        """
        Calculate the F1 and F2 amplitude (amplitude at the stimulus frequency and twice the stimulus frequency) of spike rates.

        Parameters
        ----------
        data : dict
            The data dictionary containing the spike information.
        trial : int
            The trial number to analyze.
        t_start : float
            The start time of the interval (in seconds) to analyze.
        t_end : float
            The end time of the interval (in seconds) to analyze.
        temp_freq : float
            The frequency (in Hz) of the stimulus.
        bins_per_cycle : int, optional
            The number of bins per cycle to use for the spike rate. The default is 32.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, int)
            The F1 and F2 amplitudes for each neuron, and the total number of neurons.

        """

        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]

        # Prepare for saving all spectra
        spectra = []

        # Get firing rate for each neuron
        amplitudes_F1 = np.zeros(N_neurons)
        amplitudes_F2 = np.zeros(N_neurons)  # Added to calculate F2
        cycle_length = 1 / temp_freq  # in seconds
        bins = np.arange(t_start, t_end, cycle_length / bins_per_cycle)
        for this_unit in range(N_neurons):
            unit_mask = units == this_unit
            times_unit = times[unit_mask]

            if len(times_unit) > 0:  # check if there are spikes for this unit
                # Bin spike rates
                spike_counts, _ = np.histogram(times_unit, bins=bins)
                # Convert spike counts to spike rates
                spike_rate = spike_counts / (cycle_length / bins_per_cycle)

                # Analyze Fourier amplitude
                # Compute the one-dimensional n-point discrete Fourier Transform for real input
                sp = np.fft.rfft(spike_rate)
                # Compute the frequencies corresponding to the coefficients
                freq = np.fft.rfftfreq(
                    len(spike_rate), d=(cycle_length / bins_per_cycle)
                )

                # Save the spectrum for plotting
                normalized_spectrum = np.abs(sp) / len(spike_rate) * 2
                spectra.append(normalized_spectrum)

                # Get F1 amplitude
                closest_freq_index = np.abs(freq - temp_freq).argmin()
                amplitudes_F1[this_unit] = (
                    np.abs(sp[closest_freq_index]) / len(spike_rate) * 2
                )

                # Get F2 amplitude
                closest_freq_index = np.abs(
                    freq - (2 * temp_freq)
                ).argmin()  # Change frequency to 2*temp_freq for F2
                amplitudes_F2[this_unit] = (
                    np.abs(sp[closest_freq_index]) / len(spike_rate) * 2
                )  # Store F2 amplitude in a different array

        return (
            amplitudes_F1,
            amplitudes_F2,
            N_neurons,
        )  # Return both F1 and F2 amplitudes

    def _fourier_amplitude_pooled(
        self, data, trial, t_start, t_end, temp_freq, bins_per_cycle=32
    ):
        """
        Calculate the F1 amplitude (amplitude at the stimulus frequency) of pooled spike rates.

        Parameters
        ----------
        data : dict
            The data dictionary containing the spike information.
        trial : int
            The trial number to analyze.
        t_start : float
            The start time of the interval (in seconds) to analyze.
        t_end : float
            The end time of the interval (in seconds) to analyze.
        temp_freq : float
            The frequency (in Hz) of the stimulus.
        bins_per_cycle : int, optional
            The number of bins per cycle to use for the spike rate. The default is 32.

        Returns
        -------
        float
            The F1 amplitude for the pooled neurons.
        """

        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]

        cycle_length = 1 / temp_freq  # in seconds
        bins = np.arange(t_start, t_end, cycle_length / bins_per_cycle)

        # Bin spike rates
        spike_counts, _ = np.histogram(times, bins=bins)
        # Convert spike counts to spike rates
        spike_rate = spike_counts / (cycle_length / bins_per_cycle)
        total_time = len(spike_rate) * (cycle_length / bins_per_cycle)

        # Analyze Fourier amplitude
        # Compute the one-dimensional n-point discrete Fourier Transform for real input
        sp = np.fft.rfft(spike_rate)
        # Compute the frequencies corresponding to the coefficients
        freq = np.fft.rfftfreq(len(spike_rate), d=(cycle_length / bins_per_cycle))

        # Normalize the spectrum. The factor of 2 is to account for the fact that we are
        # using half of the spectrum (the other half is the negative frequencies)
        normalized_spectrum = np.abs(sp) / len(spike_rate) * 2

        # Get spectrum per unit
        normalized_spectrum_per_unit = normalized_spectrum / N_neurons

        # Get F1 amplitude
        closest_freq_index = np.abs(freq - temp_freq).argmin()
        ampl_F1 = normalized_spectrum_per_unit[closest_freq_index]

        # Get F2 amplitude
        closest_freq_index = np.abs(freq - (2 * temp_freq)).argmin()
        ampl_F2 = normalized_spectrum_per_unit[closest_freq_index]

        return ampl_F1, ampl_F2

    def analyze_response(self, my_analysis_options):
        """
        R for firing rate
        F for Fourier amplitude
        """

        cond_names_string = "_".join(my_analysis_options["exp_variables"])
        filename = f"exp_metadata_{cond_names_string}.csv"
        data_folder = self.context.output_folder
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.columns.values
        t_start = my_analysis_options["t_start_ana"]
        t_end = my_analysis_options["t_end_ana"]
        n_trials_vec = pd.to_numeric(experiment_df.loc["n_trials", :].values)

        # Assert for equal number of trials
        assert np.all(
            n_trials_vec == n_trials_vec[0]
        ), "Not equal number of trials, aborting..."

        # Make dataframe with columns = conditions and index = trials
        R_popul_df = pd.DataFrame(index=range(n_trials_vec[0]), columns=cond_names)

        columns = cond_names.tolist()
        columns.extend(["trial", "F_peak"])
        # Make F1 and F2 dataframe
        F_popul_df = pd.DataFrame(index=range(n_trials_vec[0] * 2), columns=columns)

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = Path(data_folder) / ("Response_" + cond_name + ".gz")
            data_dict = self.data_io.get_data(filename)
            n_trials = n_trials_vec[idx]
            temp_freq = pd.to_numeric(
                experiment_df.loc["temporal_frequency", cond_name]
            )

            for this_trial in range(n_trials):
                mean_fr = self._analyze_meanfr(data_dict, this_trial, t_start, t_end)
                R_popul_df.loc[this_trial, cond_name] = mean_fr

                fr, N_neurons = self._analyze_unit_fr(
                    data_dict, this_trial, t_start, t_end
                )
                # If first trial, initialize R_unit_compiled and F_unit_compiled
                if idx == 0 and this_trial == 0:
                    R_unit_compiled = np.zeros((N_neurons, len(cond_names), n_trials))
                    F_unit_compiled = np.zeros(
                        (N_neurons, len(cond_names), n_trials, 2)
                    )
                R_unit_compiled[:, idx, this_trial] = fr

                # Amplitude spectra for pooled neurons, mean across units
                (ampl_F1, ampl_F2) = self._fourier_amplitude_pooled(
                    data_dict, this_trial, t_start, t_end, temp_freq
                )
                F_popul_df.loc[this_trial, "trial"] = this_trial
                F_popul_df.loc[this_trial, "F_peak"] = "F1"
                F_popul_df.loc[this_trial, cond_name] = ampl_F1
                F_popul_df.loc[this_trial + n_trials_vec[0], "trial"] = this_trial
                F_popul_df.loc[this_trial + n_trials_vec[0], "F_peak"] = "F2"
                F_popul_df.loc[this_trial + n_trials_vec[0], cond_name] = ampl_F2

                # Amplitude spectra for units
                (ampl_F1, ampl_F2, N_neurons) = self._fourier_amplitude(
                    data_dict, this_trial, t_start, t_end, temp_freq
                )

                F_unit_compiled[:, idx, this_trial, 0] = ampl_F1
                F_unit_compiled[:, idx, this_trial, 1] = ampl_F2

        # Set unit fr to dataframe, mean over trials
        R_unit_mean = np.mean(R_unit_compiled, axis=2)
        R_unit_df = pd.DataFrame(R_unit_mean, columns=cond_names)

        def create_dummy_variable(shape):
            dummy_var = np.empty(shape, dtype="<U12")
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        dummy_var[i, j, k] = f"n{i},c{j},f{k}"
            return dummy_var

        # Set unit F1 and F2 to dataframe, mean over trials
        F_unit_mean = np.mean(F_unit_compiled, axis=2)
        F_unit_mean_reshaped = np.concatenate(
            (F_unit_mean[:, :, 0], F_unit_mean[:, :, 1]), axis=0
        )

        F_peak = ["F1"] * N_neurons + ["F2"] * N_neurons
        unit = np.tile(np.arange(N_neurons), 2)
        F_unit_df = pd.DataFrame(F_unit_mean_reshaped, columns=cond_names.tolist())
        F_unit_df["unit"] = unit
        F_unit_df["F_peak"] = F_peak

        # Save results
        filename_out = f"{cond_names_string}_population_means.csv"
        csv_save_path = data_folder / filename_out
        R_popul_df.to_csv(csv_save_path)

        filename_out = f"{cond_names_string}_unit_means.csv"
        csv_save_path = data_folder / filename_out
        R_unit_df.to_csv(csv_save_path)

        filename_out = f"{cond_names_string}_F1F2_population_means.csv"
        csv_save_path = data_folder / filename_out
        F_popul_df.to_csv(csv_save_path)

        filename_out = f"{cond_names_string}_F1F2_unit_means.csv"
        csv_save_path = data_folder / filename_out
        F_unit_df.to_csv(csv_save_path)

    def amplitude_sensitivity(self):
        """ """
        pass

    def receptive_field(self):
        """ """
        pass

    def fano_factor(self):
        """ """
        pass

    def isi_analysis(self):
        """ """
        pass

    def temporal_correlation(self):
        """ """
        pass

    def spatial_correlation(self):
        """ """
        pass
