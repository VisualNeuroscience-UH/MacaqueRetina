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
        MeanFR = times.size / (N_neurons * (t_end - t_start))

        return MeanFR

    def _analyze_fr(self, data, trial, t_start, t_end):
        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]

        # Get firing rate for each neuron
        FR = np.zeros(N_neurons)
        for this_unit in range(N_neurons):
            unit_mask = units == this_unit
            times_unit = times[unit_mask]
            FR[this_unit] = times_unit.size / (t_end - t_start)

        return FR, N_neurons

    def _fourier_amplitude(
        self, data, trial, t_start, t_end, temp_freq, bins_per_cycle=32
    ):
        """
        Calculate the F1 amplitude (amplitude at the stimulus frequency) of spike rates.

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
        tuple of (numpy.ndarray, int)
            The F1 amplitudes for each neuron, and the total number of neurons.

        """

        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]

        # Prepare for saving all spectra
        spectra = []

        # Get firing rate for each neuron
        amplitudes = np.zeros(N_neurons)
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
                print(spike_rate)
                # Analyze Fourier amplitude
                # Compute the one-dimensional n-point discrete Fourier Transform for real input
                sp = np.fft.rfft(spike_rate)
                # Compute the frequencies corresponding to the coefficients
                freq = np.fft.rfftfreq(
                    len(spike_rate), d=(cycle_length / bins_per_cycle)
                )

                # Save the spectrum for plotting
                spectra.append(np.abs(sp))

                # Get F1 amplitude
                closest_freq_index = np.abs(freq - temp_freq).argmin()
                amplitudes[this_unit] = np.abs(sp[closest_freq_index])

        pdb.set_trace()
        # Creating subplots
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))

        # Plotting the raster plot
        axs[0].plot(times, units, ".b")
        axs[0].set_ylabel("Units")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_title("Spike Raster Plot")

        # Plotting the spectra
        for spectrum in spectra:
            axs[1].plot(freq, spectrum, color="gray", alpha=0.5)
        axs[1].plot(freq, np.mean(spectra, axis=0), color="black")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_title("Fourier Spectra")

        plt.tight_layout()
        plt.show()

        return amplitudes, N_neurons

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
        amplitude = normalized_spectrum_per_unit[closest_freq_index]

        # Creating subplots
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))

        # mask frequencies below 1 Hz
        freq_mask = freq > 1
        freq_masked = freq[freq_mask]
        normalized_spectrum_per_unit_masked = normalized_spectrum_per_unit[freq_mask]

        # Plotting the raster plot
        axs[0].plot(times, units, ".b")
        axs[0].set_ylabel("Units")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_title("Spike Raster Plot")

        # Plotting the spectrum
        axs[1].plot(freq_masked, normalized_spectrum_per_unit_masked, color="black")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_title("Fourier Spectrum")

        plt.tight_layout()
        plt.show()

        pdb.set_trace()

        # TÄHÄN JÄIT: AMPLITUDIEN SKAALAUS
        # MITEN NORMALISOIDAAN FOURIER SPEKTRIT?
        #

        return amplitude

    def contrast_respose(self, my_analysis_options):
        """
        Contrast response function: Lee_1990_JOSA
        """
        data_folder = self.context.output_folder
        experiment_df = self.data_io.get_data(filename="exp_metadata.csv")
        cond_names = experiment_df.columns.values
        t_start = my_analysis_options["t_start_ana"]
        t_end = my_analysis_options["t_end_ana"]
        n_trials_vec = pd.to_numeric(experiment_df.loc["n_trials", :].values)

        # Assert for equal number of trials
        assert np.all(
            n_trials_vec == n_trials_vec[0]
        ), "Not equal number of trials, aborting..."

        # Make dataframe with columns = conditions and index = trials
        data_df_population_means = pd.DataFrame(
            index=range(n_trials_vec[0]), columns=cond_names
        )

        temp_freqs = [0.231, 0.495, 1.023, 2.079, 4.191, 8.415, 16.863, 33.758]
        modulation_depth = [0, 0, 0, 0, 1, 0, 0, 0]

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = Path(data_folder) / ("Response_" + cond_name + ".gz")
            data_dict = self.data_io.get_data(filename)
            n_trials = n_trials_vec[idx]
            temp_freq = pd.to_numeric(
                experiment_df.loc["temporal_frequency", cond_name]
            )

            for this_trial in range(n_trials):
                MeanFR = self._analyze_meanfr(data_dict, this_trial, t_start, t_end)
                # Set results to dataframe
                data_df_population_means.loc[this_trial, cond_name] = MeanFR
                FR, N_neurons = self._analyze_fr(data_dict, this_trial, t_start, t_end)
                # If first trial, initialize FR dataframe
                if idx == 0 and this_trial == 0:
                    FR_compiled = np.zeros((N_neurons, len(cond_names), n_trials))
                # Set results to FR_compiled
                FR_compiled[:, idx, this_trial] = FR

                amplitudes, N_neurons = self._fourier_amplitude_pooled(
                    data_dict, this_trial, t_start, t_end, temp_freq
                )
                # kernels = self._first_order_kernel(
                #     data_dict, this_trial, t_start, t_end, temp_freqs, modulation_depth
                # )
                pdb.set_trace()

        # Set results to dataframe
        FR_compiled_mean = np.mean(FR_compiled, axis=2)
        data_df_units = pd.DataFrame(FR_compiled_mean, columns=cond_names)

        # Save results
        csv_save_path = data_folder / "contrast_population_means.csv"
        data_df_population_means.to_csv(csv_save_path)

        csv_save_path = data_folder / "contrast_unit_means.csv"
        data_df_units.to_csv(csv_save_path)

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
