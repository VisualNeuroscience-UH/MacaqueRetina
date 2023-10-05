"""
Analysis of retinal model ganglion cell spiking responses.
Contrast response function: Lee_1990_JOSA
Contrast sensitivity: Derrington 1984b JPhysiol
Temporal sensitivity: Lee_1990_JOSA
Receptive field: Chichilnisky_2001_Network
Fano factor: Uzzell_2004_JNeurophysiol
ISI analysis: : Uzzell_2004_JNeurophysiol
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

    def _analyze_peak2peak_fr(
        self, data, trial, t_start, t_end, temp_freq, bins_per_cycle=32
    ):
        # Analyze the peak-to-peak firing rate across units.
        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        times = times / b2u.second

        N_neurons = data["n_units"]

        cycle_length = 1 / temp_freq  # in seconds
        # Calculate N full cycles in the interval
        n_cycles = int(np.floor((t_end - t_start) / cycle_length))

        # Corrected time interval
        t_epoch = n_cycles * cycle_length

        # Change t_end to be the end of the last full cycle
        t_end_full = t_start + t_epoch

        # Remove spikes before t_start
        times = times[times > t_start]

        # Remove spikes after t_end_full
        times = times[times < t_end_full]

        # Calculate bins matching t_end_full - t_start
        bins = np.linspace(
            t_start, t_end_full, (n_cycles * bins_per_cycle) + 1, endpoint=True
        )

        bin_width = bins[1] - bins[0]
        # add one bin to the end
        spike_counts, _ = np.histogram(times, bins=bins)

        # Compute average cycle. Average across cycles.
        spike_counts_reshaped = np.reshape(
            spike_counts, (int(len(spike_counts) / bins_per_cycle), bins_per_cycle)
        )

        spike_counts_mean_across_cycles = np.mean(spike_counts_reshaped, axis=0)
        spike_count__unit_fr = spike_counts_mean_across_cycles / (N_neurons * bin_width)

        peak2peak_counts_all = np.max(spike_counts_mean_across_cycles) - np.min(
            spike_counts_mean_across_cycles
        )

        peak2peak_counts_unit_mean = peak2peak_counts_all / N_neurons
        # Convert to Hz: ptp mean across units / time for one bin
        peak2peak_fr = peak2peak_counts_unit_mean / bin_width

        return peak2peak_fr

    def _fourier_amplitude_and_phase(
        self, data, trial, t_start, t_end, temp_freq, phase_shift=0, bins_per_cycle=16
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
        phase_shift : float, optional
            The phase shift (in radians) to be applied. Default is 0.
        bins_per_cycle : int, optional
            The number of bins per cycle to use for the spike rate. The default is 32.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int)
            The F1 and F2 amplitudes, F1 and F2 phases for each neuron, and the total number of neurons.

        """

        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        N_neurons = data["n_units"]

        # Prepare for saving all spectra
        spectra = []

        # Get firing rate for each neuron
        amplitudes_F1 = np.zeros(N_neurons)  # Added to store F1
        amplitudes_F2 = np.zeros(N_neurons)  # Added to store F2

        phases_F1 = np.zeros(N_neurons)  # Added to store F1 phase
        phases_F2 = np.zeros(N_neurons)  # Added to store F2 phase

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

                # Get F1 amplitude and phase
                closest_freq_index = np.abs(freq - temp_freq).argmin()
                amplitudes_F1[this_unit] = (
                    np.abs(sp[closest_freq_index]) / len(spike_rate) * 2
                )
                phases_F1[this_unit] = np.angle(sp[closest_freq_index]) + phase_shift

                # Ensure phase remains in [-π, π]
                if phases_F1[this_unit] > np.pi:
                    phases_F1[this_unit] -= 2 * np.pi
                elif phases_F1[this_unit] < -np.pi:
                    phases_F1[this_unit] += 2 * np.pi

                # Get F2 amplitude and phase
                closest_freq_index = np.abs(freq - (2 * temp_freq)).argmin()
                amplitudes_F2[this_unit] = (
                    np.abs(sp[closest_freq_index]) / len(spike_rate) * 2
                )
                phases_F2[this_unit] = np.angle(sp[closest_freq_index]) + phase_shift

                # Ensure phase remains in [-π, π]
                if phases_F2[this_unit] > np.pi:
                    phases_F2[this_unit] -= 2 * np.pi
                elif phases_F2[this_unit] < -np.pi:
                    phases_F2[this_unit] += 2 * np.pi

        return (
            amplitudes_F1,
            amplitudes_F2,
            phases_F1,  # Return F1 phase
            phases_F2,  # Return F2 phase
            N_neurons,
        )

    def _generate_spikes(
        self,
        N_neurons,
        temp_freq,
        t_start,
        t_end,
        baseline_rate=10,
        modulation_depth=5,
    ):
        """
        A helper function to generate random spikes for N_neurons with sinusoidal modulation.

        Args:
        - N_neurons (int): Number of neurons.
        - temp_freq (float): Temporal frequency for sinusoidal modulation.
        - total_time (float): Total simulation time in seconds.
        - baseline_rate (float): Baseline firing rate in Hz.
        - modulation_depth (float): Depth of sinusoidal modulation in Hz.

        Returns:
        - spikes (list of np.ndarray): A list of spike times for each neuron.
        """

        sampling_rate = 10000  # Hz

        t = np.linspace(
            t_start, t_end, int((t_end - t_start) * sampling_rate)
        )  # 1 ms resolution
        modulating_signal_raw = (modulation_depth / 2) * np.sin(
            2 * np.pi * temp_freq * t
        )
        modulating_signal = baseline_rate + (
            modulating_signal_raw - np.min(modulating_signal_raw)
        )

        spikes = np.array([])

        for _ in range(N_neurons):
            neuron_spikes = []
            for i, rate in enumerate(modulating_signal):
                # For each time bin, decide whether to emit a spike based on rate
                if np.random.random() < (
                    rate / sampling_rate
                ):  # Convert Hz to rate per ms
                    neuron_spikes.append(t[i])
            spikes = np.concatenate((spikes, np.array(neuron_spikes)), axis=0)

        return np.sort(spikes)

    def _fourier_amplitude_pooled(
        self,
        data,
        trial,
        t_start,
        t_end,
        temp_freq,
        bins_per_cycle=32,
    ):
        units, times = self._get_spikes_by_interval(data, trial, t_start, t_end)
        times = times / b2u.second
        N_neurons = data["n_units"]

        cycle_length = 1 / temp_freq

        # Due to scalloping loss artefact causing spectral leakage, we need to
        # match the sampling points in frequency space to stimulation frequency.

        # Find the integer number of full cycles matching t_end - t_start
        n_cycles = int(np.floor((t_end - t_start) / cycle_length))

        # Corrected time interval
        t_epoch = n_cycles * cycle_length

        # Change t_end to be the end of the last full cycle
        t_end_full = t_start + t_epoch

        # Remove spikes before t_start
        times = times[times > t_start]

        # Remove spikes after t_end_full
        times = times[times < t_end_full]

        # Calculate bins matching t_end_full - t_start
        bins = np.linspace(
            t_start, t_end_full, (n_cycles * bins_per_cycle) + 1, endpoint=True
        )

        bin_width = bins[1] - bins[0]

        # Bin spike rates
        spike_counts, _ = np.histogram(times, bins=bins)
        spike_rate = spike_counts / bin_width

        # Compute Fourier Transform and associated frequencies
        sp = np.fft.rfft(spike_rate)
        freq = np.fft.rfftfreq(len(spike_rate), d=bin_width)

        # the np.fft.rfft function gives the positive frequency components
        # for real-valued inputs. This is half the total amplitude.
        # To adjust for this, we multiply the amplitude by 2:
        normalized_spectrum = 2 * np.abs(sp) / len(spike_rate)
        normalized_spectrum_per_unit = normalized_spectrum / N_neurons

        # Extract the F1 and F2 amplitudes
        closest_freq_index = np.abs(freq - temp_freq).argmin()
        ampl_F1 = normalized_spectrum_per_unit[closest_freq_index]

        closest_freq_index = np.abs(freq - (2 * temp_freq)).argmin()
        ampl_F2 = normalized_spectrum_per_unit[closest_freq_index]

        return ampl_F1, ampl_F2

    def _normalize_phase(self, phase_np, experiment_df, exp_variables):
        """
        Reset the phase so that the slowest temporal frequency is 0
        """
        assert (
            phase_np.shape[1] == experiment_df.shape[1]
        ), "Number of conditions do not match, aborting..."
        assert len(exp_variables) < 3, "More than 2 variables, aborting..."

        # Make df with index = conditions and columns = levels
        cond_value_df = pd.DataFrame(
            index=experiment_df.columns.values, columns=exp_variables
        )

        # Make new columns with conditions' levels
        for cond_idx, cond in enumerate(exp_variables):
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            cond_value_df[cond] = levels_s

        # Find row indeces of the slowest temporal frequency
        slowest_temp_freq = cond_value_df.loc[:, "temporal_frequency"].min()
        slowest_temp_freq_idx = cond_value_df[
            cond_value_df["temporal_frequency"] == slowest_temp_freq
        ].index.values

        len_second_dim = 0
        if len(exp_variables) == 2:
            # get number of levels for the second variable
            for cond in exp_variables:
                if cond != "temporal_frequency":
                    second_cond_levels = cond_value_df.loc[:, cond].unique()
                    len_second_dim = int(phase_np.shape[1] / len(second_cond_levels))

        phase_np_reset = np.zeros_like(phase_np)
        for this_cond in slowest_temp_freq_idx:
            # From experiment_df.columns.values, get the index of the slowest temporal frequency
            slowest_idx = np.where(experiment_df.columns.values == this_cond)[0][0]
            all_idx = np.arange(slowest_idx, slowest_idx + len_second_dim)
            phase_to_subtract = np.expand_dims(phase_np[:, slowest_idx], axis=1)
            phase_np_reset[:, all_idx] = phase_np[:, all_idx] - phase_to_subtract

        # If phase is over pi, subtract 2pi
        phase_np_reset[phase_np_reset > np.pi] -= 2 * np.pi

        return phase_np_reset

    def analyze_response(self, my_analysis_options):
        """
        R for firing rate
        F for Fourier amplitude
        """

        exp_variables = my_analysis_options["exp_variables"]
        cond_names_string = "_".join(exp_variables)
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
            phase_shift = pd.to_numeric(experiment_df.loc["phase_shift", cond_name])

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
                        (N_neurons, len(cond_names), n_trials, 4)
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
                (
                    ampl_F1,
                    ampl_F2,
                    phase_F1,
                    phase_F2,
                    N_neurons,
                ) = self._fourier_amplitude_and_phase(
                    data_dict, this_trial, t_start, t_end, temp_freq, phase_shift
                )

                F_unit_compiled[:, idx, this_trial, 0] = ampl_F1
                F_unit_compiled[:, idx, this_trial, 1] = ampl_F2
                F_unit_compiled[:, idx, this_trial, 2] = phase_F1
                F_unit_compiled[:, idx, this_trial, 3] = phase_F2

        # Set unit fr to dataframe, mean over trials
        R_unit_mean = np.mean(R_unit_compiled, axis=2)
        R_unit_df = pd.DataFrame(R_unit_mean, columns=cond_names)

        # Set unit F1 and F2 to dataframe, mean over trials
        F_unit_mean = np.mean(F_unit_compiled, axis=2)
        F_unit_mean_ampl_reshaped = np.concatenate(
            (F_unit_mean[:, :, 0], F_unit_mean[:, :, 1]), axis=0
        )

        F_peak = ["F1"] * N_neurons + ["F2"] * N_neurons
        unit = np.tile(np.arange(N_neurons), 2)
        F_unit_ampl_df = pd.DataFrame(
            F_unit_mean_ampl_reshaped, columns=cond_names.tolist()
        )
        F_unit_ampl_df["unit"] = unit
        F_unit_ampl_df["F_peak"] = F_peak

        F_unit_mean_phase_reshaped = np.concatenate(
            (F_unit_mean[:, :, 2], F_unit_mean[:, :, 3]), axis=0
        )

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

        filename_out = f"{cond_names_string}_F1F2_unit_ampl_means.csv"
        csv_save_path = data_folder / filename_out
        F_unit_ampl_df.to_csv(csv_save_path)

        if "temporal_frequency" in exp_variables:
            # Normalize phase -- resets phase to slowest temporal frequency
            F_unit_mean_phase_reshaped_norm = self._normalize_phase(
                F_unit_mean_phase_reshaped, experiment_df, exp_variables
            )

            F_unit_phase_df = pd.DataFrame(
                F_unit_mean_phase_reshaped_norm, columns=cond_names.tolist()
            )
            F_unit_phase_df["unit"] = unit
            F_unit_phase_df["F_peak"] = F_peak

            filename_out = f"{cond_names_string}_F1F2_unit_phase_means.csv"
            csv_save_path = data_folder / filename_out
            F_unit_phase_df.to_csv(csv_save_path)

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
