import neo
import quantities as pq
import os
import elephant
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from macaque_retina import MosaicConstructor, FunctionalMosaic
import visual_stimuli as vs
from cxsystem2.core.tools import write_to_file, load_from_file

import pdb

class Experiment():
    '''
    Run your experiment here
    '''


    def __init__(self, input_options={}):
        '''
        '''
        # Get VideoBaseClass options
        VBC = vs.VideoBaseClass()
        self.options = VBC.options

        # Replace with input options
        self._replace_options(input_options)

    def _replace_options(self, input_options):

        # Replace with input options
        for this_key in input_options.keys():
            self.options[this_key] = input_options[this_key]
        

    def _meshgrid_conditions(self, conditions_to_meshgrid, *varargs ):

        assert len(conditions_to_meshgrid) == len(varargs), 'N conditions does not match N 1D vectors of values, cannot meshgrid'
        
        # Get meshgrid of all values
        values = np.meshgrid(*varargs)

        # flatten arrays for easier indexing later
        values_flat = [v.flatten() for v in values]

        # Get conditions_idx
        conditions_metadata_idx = np.meshgrid(*[np.arange(len(v)) for v in varargs])
        conditions_metadata_idx_flat = [v.flatten() for v in conditions_metadata_idx]
        # Get conditions
        # list with N dicts, N = N experiments to run. Each of the N dicts contains all 
        # condition:value pairs
        conditions = []
        # conditions_idx contains list of strings with 1st letters of condition names and 
        # a running idx to value
        conditions_idx = []
        n_dicts = len(values_flat[0])
        for dict_idx in range(n_dicts):
            this_dict={}
            this_str = ''
            other_letters = ''
            for condition_idx, this_condition in enumerate(conditions_to_meshgrid):
                this_dict[this_condition] = values_flat[condition_idx][dict_idx]
                str_idx = this_condition.find('_')
                if str_idx > 0:
                    other_letters = this_condition[str_idx + 1]
                this_str =  this_str + \
                            this_condition[0] + \
                            other_letters + \
                            str(conditions_metadata_idx_flat[condition_idx][dict_idx])
            conditions.append(this_dict)
            conditions_idx.append(this_str)

        # Get conditions_metadata_key
        conditions_metadata_key = dict(zip(conditions_to_meshgrid, varargs))

        return conditions, conditions_metadata_key, conditions_idx, conditions_metadata_idx_flat
    
    def contrast_respose(self, contrast_min = .98, contrast_max = .98, contrast_steps = 1):
        '''
        Setup
        '''
        # contrast_min = 0.02
        # contrast_max = .98
        # contrast_steps = 13
        # contrast_min = .98
        # contrast_max = .98
        # contrast_steps = 1

        contrasts = np.logspace(np.log10(contrast_min),np.log10(contrast_max),contrast_steps)

        # mean_min = 128
        # mean_max = 128
        # mean_steps = 1

        # means = np.logspace(np.log10(mean_min),np.log10(mean_max),mean_steps)

        # temporal_frequencies = np.array([1.22, 9.76, 39.1])
        temporal_frequencies = np.array([1.22])

        # Calculate voltage values, assuming voltage = Td
        # meshgrid with mean lum and freq
        #Return conditions -- a dict with all keywords matching visual_stimuli.ConstructStimulus
        # conditions_to_meshgrid = ['contrast', 'mean', 'temporal_frequency']
        conditions_to_meshgrid = ['contrast', 'temporal_frequency']

        conditions, conditions_metadata_key, conditions_idx, conditions_metadata_idx_flat = \
            self._meshgrid_conditions(conditions_to_meshgrid, contrasts, temporal_frequencies)
                
        return conditions, conditions_metadata_key, conditions_idx, conditions_metadata_idx_flat

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

    def run(self,   ret, conditions, metadata, conditions_idx, conditions_metadata_idx_flat, 
                    n_trials=1, data_folder='', save_only_metadata=False):
        '''
        Unpack and run all conditions
        '''
        for this_key in metadata.keys():
            assert this_key in self.options.keys(), 'Missing {this_key} in visual stimuli options, check stim param name'
        
        # Test data_folder, create if missing
        os.makedirs(data_folder, exist_ok=True)


        save_path = os.path.join(data_folder,'metadata_conditions.gz')
        write_to_file(save_path,[metadata, conditions_idx, conditions_metadata_idx_flat, self.options])

        if save_only_metadata:
            return

        # Replace with input options
        for idx, input_options in enumerate(conditions):
            self._replace_options(input_options)
            stim = vs.ConstructStimulus(**self.options)

            stim.save_to_file(filename=os.path.join(data_folder,'Stim_' + conditions_idx[idx]))

            ret.load_stimulus(stim)

            example_gc=None # int or 'None'

            filename = os.path.join(data_folder,'Response_' + conditions_idx[idx])

            ret.run_cells(cell_index=example_gc, n_trials=n_trials, visualize=False, save_data=True, 
                            spike_generator_model='poisson', return_monitor=False, filename=filename) # spike_generator_model='refractory' or 'poisson'
        

if __name__ == "__main__":

    root_path = r'C:\Users\Simo\Laskenta\SimuOut'
    # root_path = ''
    
    cell_type = 'parasol'
    response_type = 'on'

    n_trials = 200

    options = {}
    options["duration_seconds"] = 0.4  # seconds
    options["mean"] = 128  # intensity mean
    options["contrast"] = .9
    options["image_width"] = 240  # Image width in pixels
    options["image_height"] = 240  # Image height in pixels
    options["background"] = 128
    options["mean"] = 128  # intensity mean


    # Valid options sine_grating; square_grating; colored_temporal_noise; white_gaussian_noise; 
    # natural_images; natural_video; phase_scrambled_video; temporal_sine_pattern; temporal_square_pattern
    options["pattern"] = 'temporal_square_pattern'
    options["phase_shift"] = 0 # 0 - 2pi, to have grating or temporal oscillation phase shifted
    options["stimulus_form"] = 'circular'  # Valid options circular, rectangular, annulus
    options["stimulus_position"] = (-.06, 0.03)  # (0, 0) Stimulus center position in degrees inside the video. (0,0) is the center.

    # In degrees. Radius for circle and annulus, half-width for rectangle. 0 gives smallest distance from image borders, ie max radius
    options["stimulus_size"] = .1

    # Init optional arguments
    options["spatial_frequency"] = 1.0
    options["temporal_frequency"] = 1.0
    options["spatial_band_pass"] = None
    options["temporal_band_pass"] = None
    options["orientation"] = 0.0  # No rotation or vertical
    options["size_inner"] = None
    options["size_outer"] = None
    options["on_proportion"]  = 0.5 # between 0 and 1, proportion of stimulus-on time
    options["direction"] = 'increment' # or 'decrement'

    options["baseline_start_seconds"] = 0.4
    options["baseline_end_seconds"] = 0.2

    E=Experiment(options)
    
    # Get retina
    testmosaic = pd.read_csv(f'{cell_type}_{response_type}_single.csv', index_col=0)

    # ret = FunctionalMosaic(testmosaic, cell_type, response_type, stimulus_center=5.03-0.01j,
    #                        stimulus_width_pix=240, stimulus_height_pix=240)
    ret = FunctionalMosaic(testmosaic, cell_type, response_type, stimulus_center=5+0j,
                           stimulus_width_pix=240, stimulus_height_pix=240)

    # Get all conditions to run
    conditions, metadata, idx,conditions_metadata_idx_flat = E.contrast_respose(
        contrast_min = 0.02, 
        contrast_max = .98, 
        contrast_steps = 13) 
        # contrast_min = 0.98, 
        # contrast_max = .98, 
        # contrast_steps = 1) 

    # data_folder = cell_type + '_' + response_type.upper() + '_c12tf0'
    data_folder_path = os.path.join(root_path,cell_type + '_' + response_type.upper() + '_c13b')
    # data_folder = cell_type + '_' + response_type.upper() + '_metadata'
    E.run(  ret, conditions, metadata, idx, conditions_metadata_idx_flat, 
            n_trials=n_trials, data_folder=data_folder_path, save_only_metadata=False)



