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

    def _show_rasterplot(self, spiketrain_list, title):
        for i, spiketrain in enumerate(spiketrain_list):
            t = spiketrain.rescale(b2u.ms)
            plt.plot(t, i * np.ones_like(t), "k.", markersize=2)
        plt.axis("tight")
        plt.xlim(0, 7000)
        plt.xlabel("Time (ms)", fontsize=16)
        plt.ylabel("Spike Train Index", fontsize=16)
        plt.gca().tick_params(axis="both", which="major", labelsize=14)
        plt.title(title)
        plt.show()

    def _show_rasterplot_from_df(self, spiketrain_df, unit_idx=0, title=""):
        plt.figure()
        unit_data_df = spiketrain_df.loc[spiketrain_df["unit_idx"] == unit_idx]
        plt.plot(unit_data_df["spike_time"], unit_data_df["trial"], "k.", markersize=2)
        plt.axis("tight")
        # plt.xlim(0, 7000)
        plt.xlabel("Time (ms)", fontsize=16)
        plt.ylabel("Spike Train Index", fontsize=16)
        plt.gca().tick_params(axis="both", which="major", labelsize=14)
        plt.title(title)

    def _get_spike_trains(self, fullpath):
        """
        Return pandas dataframe with columns=['trial', 'unit_idx', 'spike_time']
        Successive trials are appended to the end of df
        """

        file_type = self.file_type

        assert (
            file_type == "cxsystem"
        ), "Sorry, nix is defunc at the moment, u need to use cxsystem and gz filetype"

        data = load_from_file(fullpath)
        trial_name_list = [name for name in data.keys() if "spikes" in name]

        # build pandas df
        spiketrains_df = pd.DataFrame(columns=["trial", "unit_idx", "spike_time"])
        for trial_idx, trial_key in enumerate(trial_name_list):
            it_list = data[trial_key]
            trial_df = pd.DataFrame(it_list[0], columns=["unit_idx"])
            trial_df["spike_time"] = it_list[1] / b2u.second
            trial_df["trial"] = trial_idx
            spiketrains_df = spiketrains_df.append(trial_df, ignore_index=True)

        return spiketrains_df

    # Imported and modified from SystemTools
    # TÄHÄN JÄIT: SEURAAVAT KOLME MENETELMÄÄ SOVITETAAN TÄHÄN

    def get_n_samples(self, data):
        nsamples = len(data["time_vector"])
        assert (
            nsamples == data["time_vector"].shape[0]
        ), "time_vector shape inconsistency, aborting..."
        return nsamples

    def _get_spikes_by_interval(self, data, NG, t_idx_start, t_idx_end):
        data_by_group = data["spikes_all"][NG]

        # Get and mark MeanFR to df
        N_neurons = data_by_group["count"].size

        # spikes by interval needs seconds, thus we need to multiply with dt
        dt = self.get_dt(data)

        t_start = t_idx_start * dt
        t_end = t_idx_end * dt

        spikes = data_by_group["t"][
            np.logical_and(
                data_by_group["t"] > t_start * b2u.second,
                data_by_group["t"] < t_end * b2u.second,
            )
        ]

        return N_neurons, spikes, dt

    def _analyze_meanfr(self, data, NG):
        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end

        n_samples = self.get_n_samples(data)
        t_idx_end = self.end2idx(t_idx_end, n_samples)

        N_neurons, spikes, dt = self._get_spikes_by_interval(
            data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end
        )

        MeanFR = spikes.size / (N_neurons * (t_idx_end - t_idx_start) * dt)

        return MeanFR

    def contrast_respose(self):
        """
        Contrast response function: Lee_1990_JOSA
        """
        output_folder = self.context.output_folder
        load_path = output_folder / "exp_metadata.gz"
        [cond_metadata_key, cond_names, cond_options] = load_from_file(load_path)
        # experiment_df = pd.read_csv(output_folder / "exp_metadata.csv", index_col=0)
        data_folder = self.context.output_folder

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = Path(data_folder) / ("Response_" + cond_name + ".gz")
            data_dict = self.data_io.get_data(filename)
            pdb.set_trace()

        # get spike trains
        csv_save_path = output_folder / "contrast_df.csv"
        data_df.to_csv(csv_save_path)

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
