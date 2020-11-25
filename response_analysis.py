'''
Analysis of retinal model ganglion cell spiking responses.
Contrast response function: Lee_1990_JOSA
Amplitude sensitivity: Lee_1990_JOSA
Receptive field: Shah_2020_eLife
Fano factor: Uzzell_2004_JNeurophysiol
ISI analysis: : Uzzell_2004_JNeurophysiol
Temporal correlation: Greschner_2011_JPhysiol
Spatial correlation: Greschner_2011_JPhysiol
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import neo
import quantities as pq
from neo.io import NixIO
import elephant as el

from cxsystem2.core.tools import write_to_file, load_from_file

import pdb


class ResponseAnalysis():

    def __init__(self, path):
        '''
        '''
        # Check if folder exists
        fullpath = os.path.join(path)
        assert os.path.isdir(fullpath), f'Did not find {fullpath}'

        # Get filenames for path
        file_prefix = 'Response'
        file_extension = 'h5'
        filenames = []
        with os.scandir(fullpath) as it:
            for entry in it:
                if  entry.name.startswith(file_prefix) and entry.name.endswith(file_extension):
                    filenames.append(entry.name)

        self.path = path
        self.filenames = filenames

    def _show_rasterplot(self, spiketrain_list, title):
        for i, spiketrain in enumerate(spiketrain_list):
            t = spiketrain.rescale(pq.ms)
            plt.plot(t, i * np.ones_like(t), 'k.', markersize=2)
        plt.axis('tight')
        plt.xlim(0, 7000)
        plt.xlabel('Time (ms)', fontsize=16)
        plt.ylabel('Spike Train Index', fontsize=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)
        plt.title(title)
        plt.show()

    def contrast_respose(self):
        '''
        '''
        filenames = self.filenames
        path = self.path
        # TÄHÄN JÄIT: LOOP TO GET MEAN RATES, MAKE MATRIX, PLOT 
        # Loop files
        metadata_folder = ''
        load_path = os.path.join(path + metadata_folder, 'metadata_conditions.gz')
        metadata_conditions = load_from_file(load_path)
        # Create dict of cond_names : row_col_idxs
        row_col_idxs = np.column_stack((metadata_conditions[2][0],metadata_conditions[2][1])).tolist()
        cond_idx = dict(zip(metadata_conditions[1], row_col_idxs))
        # Create pd.DataFrame with corresponding data, fill the data below
        contrasts = np.round(metadata_conditions[0]['contrast'] * 100,1)
        temporal_frequencies = np.round(metadata_conditions[0]['temporal_frequency'],1)
        data_df = pd.DataFrame(index=contrasts, columns=temporal_frequencies)

        # TÄHÄN JÄIT: FUNKTIOI SMOOTH, SEN JÄLKEEN TEE FOURIER PSD, SIITÄ QUANT

        # for this_file in filenames:
        for cond_idx_key, idx in zip(cond_idx.keys(), cond_idx.values()):
            this_file_list = [i for i in filenames if cond_idx_key in i] 
            assert len(this_file_list) == 1, 'Not unique filename, aborting...'
            this_file=this_file_list[0]
            nix_fullpath = os.path.join(path,this_file)
            nixfile = NixIO(filename=nix_fullpath, mode='ro')
            block = nixfile.read_block()
            spiketrains = el.neo_tools.get_all_spiketrains(block)
            # self._show_rasterplot(spiketrains, cond_idx_key)
            binned_spike_trains = el.conversion.BinnedSpikeTrain(   spiketrains,
                                                                    t_start=0.5*pq.s, 
                                                                    t_stop=6.5*pq.s, 
                                                                    bin_size=6*pq.s)
            # pdb.set_trace()            
            spike_counts = binned_spike_trains.to_array()
            # stimulus_time = 6 * pq.s
            stimulus_time = 6
            mean_rate = np.mean(spike_counts / stimulus_time)
            # idx = 
            data_df.iloc[idx[0], idx[1]] = mean_rate
            nixfile.close()

        csv_save_path = os.path.join(path + metadata_folder, 'contrast_df.csv')
        data_df.to_csv(csv_save_path)
        # pdb.set_trace()

        

    def amplitude_sensitivity(self):
        '''
        '''
        pass

    def receptive_field(self):
        '''
        '''
        pass

    def fano_factor(self):
        '''
        '''
        pass

    def isi_analysis(self):
        '''
        '''
        pass

    def temporal_correlation(self):
        '''
        '''
        pass

    def spatial_correlation(self):
        '''
        '''
        pass



if __name__ == "__main__":

    root_path = r'C:\Users\Simo\Laskenta\SimuOut'

    cell_type = 'parasol'
    response_type = 'off'
 
    # data_folder = cell_type + '_' + response_type.upper() + '_c12tf0'
    data_folder_path = os.path.join(root_path,cell_type + '_' + response_type.upper() + '_snr')
    R = ResponseAnalysis(data_folder_path)
    R.contrast_respose()

    