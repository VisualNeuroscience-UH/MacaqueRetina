'''
Analysis of retinal model ganglion cell spiking responses.
Contrast response function: Lee_1990_JOSA
Contrast sensitivity: Enroth-Cugell_1966_JPhysiol
Amplitude sensitivity: Lee_1990_JOSA
Receptive field: Chichilnisky_2001_Network
Fano factor: Uzzell_2004_JNeurophysiol
ISI analysis: : Uzzell_2004_JNeurophysiol
Temporal correlation: Greschner_2011_JPhysiol
Spatial correlation: Greschner_2011_JPhysiol
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import neo
import quantities as pq
from neo.io import NixIO
import elephant as el
import brian2.units as b2u 

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
        file_extension = 'gz' # 'gz' , 'h5'
        file_type = 'cxsystem' # 'cxsystem', 'nix'
        filenames = []
        with os.scandir(fullpath) as it:
            for entry in it:
                if  entry.name.startswith(file_prefix) and entry.name.endswith(file_extension):
                    filenames.append(entry.name)

        self.path = path
        self.filenames = filenames
        self.file_type = file_type

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

    def _show_rasterplot_from_df(self, spiketrain_df, unit_idx=0,title=''):
        # pdb.set_trace()
        # for i, spiketrain in enumerate(spiketrain_list):
        #     t = spiketrain.rescale(pq.ms)
        #     plt.plot(t, i * np.ones_like(t), 'k.', markersize=2)
        plt.figure()
        unit_data_df = spiketrain_df.loc[spiketrain_df['unit_idx'] == unit_idx]
        plt.plot(unit_data_df['spike_time'], unit_data_df['trial'], 'k.', markersize=2)
        plt.axis('tight')
        # plt.xlim(0, 7000)
        plt.xlabel('Time (ms)', fontsize=16)
        plt.ylabel('Spike Train Index', fontsize=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)
        plt.title(title)
        # plt.show()

    def _get_spike_trains(self, fullpath):
        '''
        Return pandas dataframe with columns=['trial', 'unit_idx', 'spike_time']
        Successive trials are appended to the end of df
        '''

        file_type = self.file_type

        # nix spiketrains is defunc at the moment, because 
        # multiple trials cannot be handled with current syntax
        # if file_type=='nix':
        #     nixfile = NixIO(filename=fullpath, mode='ro')
        #     block = nixfile.read_block()
        #     nixfile.close()
        #     spiketrains = el.neo_tools.get_all_spiketrains(block)
        #     pdb.set_trace()
        # elif file_type=='cxsystem':
        #     pdb.set_trace()

        assert file_type=='cxsystem', 'Sorry, nix is defunc at the moment, u need to use cxsystem and gz filetype'
        
        data = load_from_file(fullpath)
        trial_name_list = [name for name in data.keys() if 'spikes' in name]

        # build pandas df
        spiketrains_df = pd.DataFrame(columns=['trial', 'unit_idx', 'spike_time'])
        for trial_idx, trial_key in enumerate(trial_name_list):
            it_list = data[trial_key]
            trial_df = pd.DataFrame(it_list[0], columns=['unit_idx'])
            trial_df['spike_time'] = it_list[1] / b2u.second
            trial_df['trial'] = trial_idx
            spiketrains_df = spiketrains_df.append(trial_df, ignore_index=True)

        return spiketrains_df

    # def _get_mean_spike_trains(self, spiketrains_df):

    #     pdb.set_trace()
    #     spiketrains_df.groupby(['trial']).mean()
    #     return spiketrains_mean_df

    def contrast_respose(self):
        '''
        '''
        filenames = self.filenames
        path = self.path
        
        '''
        TÄHÄN JÄIT: MODULARISOI RESPONSSIANALYYSI
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
        '''    
        
        # Loop files
        metadata_folder = ''
        load_path = os.path.join(path, metadata_folder, 'metadata_conditions.gz')
        metadata_conditions = load_from_file(load_path)
        # Create dict of cond_names : row_col_idxs
        row_col_idxs = np.column_stack((metadata_conditions[2][0],metadata_conditions[2][1])).tolist()
        cond_idx = dict(zip(metadata_conditions[1], row_col_idxs))

        # Create pd.DataFrame with corresponding data, fill the data below
        # Get conditions
        contrasts = np.round(metadata_conditions[0]['contrast'] * 100,1)
        temporal_frequencies = np.round(metadata_conditions[0]['temporal_frequency'],1)
        data_df = pd.DataFrame(index=contrasts, columns=temporal_frequencies)

        # for this_file in filenames:
        unit_idx = 2 # Representative unit
        for cond_idx_key, idx in zip(cond_idx.keys(), cond_idx.values()):
            this_file_list = [i for i in filenames if cond_idx_key in i] 
            assert len(this_file_list) == 1, 'Not unique filename, aborting...'
            this_file=this_file_list[0]
            fullpath = os.path.join(path,this_file)

            # Get spiketrains dataframe by condition
            spiketrains_df = self._get_spike_trains(fullpath)
            
            # Get firing rate for each trial for a specified time interval
            # -get time interval
            # -set response dynamic range, bin edges
            # -count spikes from time interval,
            #separately for each trial
            # -set df value
            # -show joint probability distribution

            # Get mean spiketrains across trials
            # spiketrains_mean_df = self._get_mean_spike_trains(spiketrains_df)


            self._show_rasterplot_from_df(spiketrains_df,unit_idx=unit_idx, title=cond_idx_key)
            # pdb.set_trace()

        plt.show()
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
    response_type = 'on'
 
    # data_folder = cell_type + '_' + response_type.upper() + '_c12tf0'
    data_folder_path = os.path.join(root_path,cell_type + '_' + response_type.upper() + '_c13')
    R = ResponseAnalysis(data_folder_path)
    R.contrast_respose()

    