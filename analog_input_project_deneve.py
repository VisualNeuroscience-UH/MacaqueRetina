# External
import scipy.io as sio
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Builtin
import os

# Own
# import utilities as ut
# from system_utilities import SystemUtilities as ut

# Construction
import time
import sys
import pdb

# sys.path.append(r'C:\Users\Simo\Laskenta\Git_Repos\MacaqueRetina_Git') # temp
sys.path.append(r'/opt2/Laskenta_ssd/Git_Repos/MacaqueRetina') # temp
from macaque_retina import MosaicConstructor, FunctionalMosaic


class AnalogInput():
    '''
    Creates analog input in CxSystem compatible video mat file format.

    frameduration assumes milliseconds
    '''
    def __init__(   self, 
                    Nrequested_units = 3, 
                    Ntime = 10000, 
                    filename_out = 'my_video.mat', 
                    input_type = 'quadratic_oscillation',
                    coord_type = 'dummy',
                    Ncycles = 2,
                    frameduration = 15):


        # get Input
        if input_type == 'noise':
            Input = self.create_noise_input(Nx = Nrequested_units, Ntime = Ntime)
        elif input_type == 'quadratic_oscillation':
            if Nrequested_units != 2:
                print(f'NOTE: You requested {input_type} input type, setting excessive units to 0 value')
            Input = self.create_quadratic_oscillation_input(Nx = Nrequested_units, Ntime = Ntime, Ncycles = Ncycles)
        elif input_type == 'step_current':
            Input = self.create_step_input(Nx = Nrequested_units, Ntime = Ntime)
        # if current_injection is True:
        #     Input = self.create_current_injection(Input)
        #     filename_out = filename_out[:-4] + '_ci.mat'

        # get coordinates
        if coord_type == 'dummy':
            w_coord, z_coord = self.get_dummy_coordinates(Nx = Nrequested_units)
        elif coord_type == 'real':
            w_coord, z_coord = self.get_real_coordinates(Nx = Nrequested_units)

        assert 'w_coord' in locals(), 'coord_type not set correctly, check __init__, aborting'
        w_coord = np.expand_dims(w_coord, 1)
        z_coord = np.expand_dims(z_coord, 1)

        self.save_video(filename_out = filename_out, 
                        Input = Input, 
                        z_coord = z_coord, 
                        w_coord = w_coord,
                        frameduration = frameduration)

    def _lineplot(self, data):
        data_df = pd.DataFrame(data)
        sns.lineplot(data=data_df)
        plt.show()

    def _gaussian_filter(self):

        sigma = 30 # was abs(30)
        w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp( -1 * np.power(np.arange(1000) - 500,2) / (2 * np.power(sigma,2)))
        w=w/np.sum(w)
        return w

    def _normalize(self, Input):
        # Scale to interval [0, 1]
        Input = Input - min(np.ravel(Input))
        Input = Input / max(np.ravel(Input))
        return Input

    def create_noise_input(self, Nx = 0, Ntime = None):   

        assert Nx != 0, 'N units not set, aborting...'
        assert Ntime is not None, 'N timepoints not set, aborting...'
        Input=(np.random.multivariate_normal(np.zeros([Nx]), np.eye(Nx), Ntime)).T

        # Get gaussian filter, apply
        w = self._gaussian_filter()
        A = 15 # Deneve project was 2000, from their Learning.py file
        for d in np.arange(Nx):
            Input[d,:] = A * np.convolve(Input[d,:],w,'same')

        return Input


    def create_quadratic_oscillation_input(self, Nx = 0, Ntime = None, Ncycles = 0):   

        assert Nx != 0, 'N units not set, aborting...'
        assert Ncycles != 0, 'N cycles not set, aborting...'
        assert Ntime is not None, 'N timepoints not set, aborting...'

        freq = Ncycles * 2 * np.pi * 1/Ntime # frequency, this gives Ncycles over all time points
        time = np.arange(Ntime)
        A = 5 # Deneve project was 2000, from their Learning.py file

        sine_wave = np.sin(freq * time)
        cosine_wave = np.cos(freq * time)

        if Nx > 2:
            Input = A * np.array([sine_wave, cosine_wave])
            unit_zero_input = np.zeros(sine_wave.shape)
            stack_to_add = np.tile(unit_zero_input, (Nx - 2, 1))
            zero_padded_input_stack = np.vstack((Input, stack_to_add))
            Input = zero_padded_input_stack

        return Input

    def create_step_input(self, Nx = 0, Ntime = None):   

        assert Nx != 0, 'N units not set, aborting...'
        assert Ntime is not None, 'N timepoints not set, aborting...'

        # Create your input here. Zeros and ones at this point.
        # Create matrix of zeros with shape of Input
        Input = (np.concatenate((np.zeros((Ntime//3,), dtype=int), np.ones((Ntime//3), dtype=int), np.zeros((Ntime//3), dtype=int)), axis=None))
        Input = (np.concatenate((Input, np.zeros((Ntime-np.size((Input),0),), dtype=int)), axis=None))
        Input = (np.tile(Input.T, (Nx,1)))

        A = 5 # Amplification, Units = ?
        Input = A * Input

        minI = np.min(Input)
        maxI = np.max(Input)
        print(f'minI = {minI}')
        print(f'maxI = {maxI}')
        return Input

    def get_dummy_coordinates(self, Nx = 0):
        # Create dummy coordinates for CxSystem format video input.
        # NOTE: You are safer with local mode on in CxSystem to use these

        assert Nx != 0, 'N units not set, aborting...'

        # N units btw 4 and 6 deg ecc
        z_coord = np.linspace(4.8, 5.2, Nx)
        z_coord = z_coord + 0j # Add second dimension

        # Copied from macaque retina, to keep w and z coords consistent
        a = .077 / .082 # ~ 0.94
        k = 1 / .082 # ~ 12.2
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord 

    def get_real_coordinates(self, Nx = 0):
        # For realistic coordinates, we use Macaque retina module

        assert Nx != 0, 'N units not set, aborting...'

        # Get gc mosaic
        mosaic = MosaicConstructor(gc_type='parasol', response_type='on', ecc_limits=[4.8, 5.2],
                                    sector_limits=[-.4, .4], model_density=1.0, randomize_position=0.05)

        mosaic.build()
        mosaic.save_mosaic('deneve_test_mosaic.csv')


        testmosaic = pd.read_csv('deneve_test_mosaic.csv', index_col=0)

        ret = FunctionalMosaic(testmosaic, 'parasol', 'on', stimulus_center=5+0j,
                                   stimulus_width_pix=240, stimulus_height_pix=240)
        w_coord, z_coord = FunctionalMosaic._get_w_z_coords(ret)

        # Get random sample sized Nrequested_units, assert for too small sample

        Nmosaic_units = w_coord.size
        assert Nx <= Nmosaic_units, 'Too few units in mosaic, increase ecc and / or sector limits in get_real_coordinates method'
        idx = np.random.choice(Nmosaic_units, size=Nx, replace=False)
        w_coord, z_coord = w_coord[idx], z_coord[idx]

        return w_coord, z_coord 

    def save_video(self, filename_out = None, Input = None, z_coord = None, w_coord = None, frameduration = None):

        assert all([filename_out is not None, Input is not None, z_coord is not None, w_coord is not None, frameduration is not None]), \
            'Some input missing from save_video, aborting...'

        total_duration = Input.shape[1] * frameduration / 1000
        # mat['stimulus'].shape should be (Nunits, Ntimepoints)
        mat_out_dict = {'z_coord': z_coord, 
                        'w_coord': w_coord, 
                        'stimulus':Input, 
                        'frameduration':frameduration,
                        'stimulus_duration_in_seconds':total_duration}
        sio.savemat(filename_out, mat_out_dict)
        print(f'Duration of stimulus is {total_duration} seconds')


if __name__ == "__main__":

    if sys.platform == 'linux':
        root_path = r'/opt3/Laskenta/Models/Deneve/in'
    elif sys.platform == 'win32':
        root_path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test\Replica_in'

    for idx in np.arange(5,20):
        # filename_out = 'input_noise_210408.mat'
        filename_out = f'noise_210916_SingleSpike_{str(idx)}.mat'
        full_filename_out = os.path.join(root_path, filename_out)
        Nrequested_units = 3
        Ntime = 20000
        input_type = 'noise' # 'quadratic_oscillation' or 'noise' or 'step_current'
        Ncycles = 4
        dt = 0.1 # IMPORTANT: assuming milliseconds

        AnalogInput(
            Nrequested_units = Nrequested_units, 
            Ntime = Ntime, 
            filename_out = full_filename_out, 
            input_type = input_type,
            Ncycles = Ncycles,
            frameduration = dt)

