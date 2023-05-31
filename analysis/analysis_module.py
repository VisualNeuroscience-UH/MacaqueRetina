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
import matplotlib.pyplot as plt
import pandas as pd

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

    def contrast_respose(self, my_analysis_options):
        """
        Contrast response function: Lee_1990_JOSA
        """
        data_folder = self.context.output_folder
        experiment_df = pd.read_csv(data_folder / "exp_metadata.csv", index_col=0)
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

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = Path(data_folder) / ("Response_" + cond_name + ".gz")
            data_dict = self.data_io.get_data(filename)
            n_trials = n_trials_vec[idx]

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


# if __name__ == "__main__":
#     root_path = r"C:\Users\Simo\Laskenta\SimuOut"

#     cell_type = "parasol"
#     response_type = "on"

#     # data_folder = cell_type + '_' + response_type.upper() + '_c12tf0'
#     data_folder_path = os.path.join(
#         root_path, cell_type + "_" + response_type.upper() + "_c13"
#     )
#     R = Analysis(data_folder_path)
#     R.contrast_respose()
