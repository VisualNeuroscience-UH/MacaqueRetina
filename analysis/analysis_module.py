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
# import neo
# import quantities as pq
# from neo.io import NixIO
# import elephant as el
import brian2.units as b2u

# Local
from cxsystem2.core.tools import write_to_file, load_from_file
from analysis.analysis_base_module import AnalysisBase

# Builtin
import pdb

# import time
import os


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

        # nix spiketrains is defunc at the moment, because
        # multiple trials cannot be handled with current syntax
        # if file_type=='nix':
        #     nixfile = NixIO(filename=fullpath, mode='ro')
        #     block = nixfile.read_block()
        #     nixfile.close()
        #     spiketrains = el.neo_tools.get_all_spiketrains(block)
        # elif file_type=='cxsystem':
        #   pass
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

    def contrast_respose(self):
        """ """
        filenames = self.filenames
        path = self.context.path

        """
        MODULARISOI RESPONSSIANALYYSI
        MINIMOI ASIAT JOTKA TÄYTYY VAAN TIETÄÄ
        MAKSIMOI ASIAT JOTKA NÄKYVÄT YHDELLÄ SILMÄYKSELLÄ
        PSEUDOCODE
        1. YLEINEN OSA, GET METADATA, GET CONDITIONS
        2. ALUSTA TARVITTAVAT TULOS DF:T
        -MEAN DF
        -PROBABILITY DISTRIBUTION
        --GET TIME INTERVAL
        --SET RESPONSE DYNAMIC RANGE/BIN EDGES
        3. GET SPIKE COUNTS, SET DF VALUES
        4. VISUALIZE DF DATA
        """

        # Loop files
        metadata_folder = ""
        load_path = os.path.join(path, metadata_folder, "metadata_conditions.gz")
        metadata_conditions = load_from_file(load_path)
        # Create dict of cond_names : row_col_idxs
        row_col_idxs = np.column_stack(
            (metadata_conditions[2][0], metadata_conditions[2][1])
        ).tolist()
        cond_idx = dict(zip(metadata_conditions[1], row_col_idxs))

        # Create pd.DataFrame with corresponding data, fill the data below
        # Get conditions
        contrasts = np.round(metadata_conditions[0]["contrast"] * 100, 1)
        temporal_frequencies = np.round(metadata_conditions[0]["temporal_frequency"], 1)
        data_df = pd.DataFrame(index=contrasts, columns=temporal_frequencies)

        # for this_file in filenames:
        unit_idx = 2  # Representative unit
        for cond_idx_key, idx in zip(cond_idx.keys(), cond_idx.values()):
            this_file_list = [i for i in filenames if cond_idx_key in i]
            assert len(this_file_list) == 1, "Not unique filename, aborting..."
            this_file = this_file_list[0]
            fullpath = os.path.join(path, this_file)

            # Get spiketrains dataframe by condition
            spiketrains_df = self._get_spike_trains(fullpath)

            self._show_rasterplot_from_df(
                spiketrains_df, unit_idx=unit_idx, title=cond_idx_key
            )

        plt.show()
        csv_save_path = os.path.join(path + metadata_folder, "contrast_df.csv")
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


if __name__ == "__main__":
    root_path = r"C:\Users\Simo\Laskenta\SimuOut"

    cell_type = "parasol"
    response_type = "on"

    # data_folder = cell_type + '_' + response_type.upper() + '_c12tf0'
    data_folder_path = os.path.join(
        root_path, cell_type + "_" + response_type.upper() + "_c13"
    )
    R = Analysis(data_folder_path)
    R.contrast_respose()
