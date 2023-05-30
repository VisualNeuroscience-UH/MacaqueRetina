# Analysis
import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt

# Local
from stimuli.visual_stimulus_module import VideoBaseClass, ConstructStimulus

# Builtin
import pdb
import os
from pathlib import Path


class Experiment(VideoBaseClass):
    """
    Build your experiment here
    """

    _properties_list = [
        "path",
        "output_folder",
        "input_folder",
        "my_stimulus_options",
        "my_stimulus_metadata",
    ]

    def __init__(self, context, data_io):
        super().__init__()

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    def _replace_options(self, input_options):
        # Replace with input options
        for this_key in input_options.keys():
            self.options[this_key] = input_options[this_key]

    def _meshgrid_conditions(self, conditions_to_meshgrid, *varargs):
        assert len(conditions_to_meshgrid) == len(
            varargs
        ), "N conditions does not match N 1D vectors of values, cannot meshgrid"

        # Get meshgrid of all values
        values = np.meshgrid(*varargs)

        # flatten arrays for easier indexing later
        values_flat = [v.flatten() for v in values]

        # Get cond_names
        conditions_metadata_idx = np.meshgrid(*[np.arange(len(v)) for v in varargs])
        cond_array_list = [v.flatten() for v in conditions_metadata_idx]
        # Get conditions to replace the corresponding options in the stimulus
        # list with N dicts, N = N experiments to run. Each of the N dicts contains all
        # condition:value pairs
        cond_options = []
        # cond_names contains list of strings with 1st letters of condition names and
        # a running idx to value
        cond_names = []
        n_dicts = len(values_flat[0])
        for dict_idx in range(n_dicts):
            this_dict = {}
            this_str = ""
            other_letters = ""
            for condition_idx, this_condition in enumerate(conditions_to_meshgrid):
                this_dict[this_condition] = values_flat[condition_idx][dict_idx]
                str_idx = this_condition.find("_")
                if str_idx > 0:
                    other_letters = this_condition[str_idx + 1]
                this_str = (
                    this_str
                    + this_condition[0]
                    + other_letters
                    + str(cond_array_list[condition_idx][dict_idx])
                )
            cond_options.append(this_dict)
            cond_names.append(this_str)

        # Get cond_metadata_key
        cond_metadata_key = dict(zip(conditions_to_meshgrid, varargs))

        return (
            cond_options,
            cond_metadata_key,
            cond_names,
            cond_array_list,
        )

    def contrast_respose(self, contrast_min=0.98, contrast_max=0.98, contrast_steps=1):
        """
        Setup
        """

        contrasts = np.logspace(
            np.log10(contrast_min), np.log10(contrast_max), contrast_steps
        )

        temporal_frequencies = np.array([1.22])

        # Calculate voltage values, assuming voltage = Td
        # Return cond_options -- a dict with all keywords matching visual_stimulus_module.ConstructStimulus
        conditions_to_meshgrid = ["contrast", "temporal_frequency"]

        (
            cond_options,
            cond_metadata_key,
            cond_names,
            cond_array_list,
        ) = self._meshgrid_conditions(
            conditions_to_meshgrid, contrasts, temporal_frequencies
        )

        return (
            cond_options,
            cond_metadata_key,
            cond_names,
            cond_array_list,
        )

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

    def run(
        self,
        cond_options,
        metadata,
        cond_names,
        cond_array_list,
        n_trials=1,
        save_only_metadata=False,
    ):
        """
        Unpack and run all conditions
        """
        for this_key in metadata.keys():
            assert (
                this_key in self.options.keys()
            ), "Missing {this_key} in visual stimuli options, check stim param name"

        # Update options to match my_stimulus_options in conf file
        self._replace_options(self.context.my_stimulus_options)

        # Test data_folder, create if missing, write metadata
        data_folder = self.context.output_folder
        save_path = data_folder / "metadata_conditions.gz"
        self.data_io.write_to_file(
            save_path,
            [metadata, cond_names, cond_array_list, self.options],
        )

        if save_only_metadata:
            return

        # Replace with input options
        for idx, input_options in enumerate(cond_options):
            stimulus_video_name = "Stim_" + cond_names[idx]
            input_options["stimulus_video_name"] = stimulus_video_name
            self._replace_options(input_options)

            stim = self.stimulate.make_stimulus_video(self.options)
            stimulus_video_name_full = Path(data_folder) / stimulus_video_name
            self.data_io.save_stimulus_to_videofile(stimulus_video_name_full, stim)

            self.working_retina.load_stimulus(stim)

            example_gc = None  # int or 'None'

            filename = Path(data_folder) / ("Response_" + cond_names[idx])

            self.working_retina.run_cells(
                cell_index=example_gc,
                n_trials=n_trials,
                save_data=True,
                spike_generator_model="poisson",
                return_monitor=False,
                filename=filename,
            )  # spike_generator_model='refractory' or 'poisson'


if __name__ == "__main__":
    pass
    # root_path = r"C:\Users\Simo\Laskenta\SimuOut"
    # # root_path = ''

    # cell_type = "parasol"
    # response_type = "on"

    # n_trials = 200

    # options = {}
    # options["duration_seconds"] = 0.4  # seconds
    # options["contrast"] = 0.9
    # options["image_width"] = 240  # Image width in pixels
    # options["image_height"] = 240  # Image height in pixels
    # options["background"] = 128
    # options["mean"] = 128  # intensity mean

    # Valid options sine_grating; square_grating; colored_temporal_noise; white_gaussian_noise;
    # natural_images; natural_video; phase_scrambled_video; temporal_sine_pattern; temporal_square_pattern
    # options["pattern"] = "temporal_square_pattern"
    # options[
    #     "phase_shift"
    # ] = 0  # 0 - 2pi, to have grating or temporal oscillation phase shifted
    # options[
    #     "stimulus_form"
    # ] = "circular"  # Valid options circular, rectangular, annulus
    # options["stimulus_position"] = (
    #     -0.06,
    #     0.03,
    # )  # (0, 0) Stimulus center position in degrees inside the video. (0,0) is the center.

    # In degrees. Radius for circle and annulus, half-width for rectangle. 0 gives smallest distance from image borders, ie max radius
    # options["stimulus_size"] = 0.1

    # Init optional arguments
    # options["spatial_frequency"] = 1.0
    # options["temporal_frequency"] = 1.0
    # options["size_inner"] = None
    # options["size_outer"] = None
    # options["on_proportion"] = 0.5  # between 0 and 1, proportion of stimulus-on time
    # options["direction"] = "increment"  # or 'decrement'

    # options["baseline_start_seconds"] = 0.4
    # options["baseline_end_seconds"] = 0.2

    # E = Experiment(options)

    # Get retina
    # testmosaic = pd.read_csv(f"{cell_type}_{response_type}_single.csv", index_col=0)

    # ret = WorkingRetina(testmosaic, cell_type, response_type, stimulus_center=5.03-0.01j,
    #                        stimulus_width_pix=240, stimulus_height_pix=240)
    # ret = WorkingRetina(
    #     testmosaic,
    #     cell_type,
    #     response_type,
    #     stimulus_center=5 + 0j,
    #     stimulus_width_pix=240,
    #     stimulus_height_pix=240,
    # )

    # # Get all conditions to run
    # conditions, metadata, idx, cond_array_list = E.contrast_respose(
    #     # contrast_min = 0.02,
    #     # contrast_max = .98,
    #     # contrast_steps = 13)
    #     contrast_min=0.98,
    #     contrast_max=0.98,
    #     contrast_steps=1,
    # )

    # # data_folder = cell_type + '_' + response_type.upper() + '_c12tf0'
    # data_folder_path = os.path.join(
    #     root_path, cell_type + "_" + response_type.upper() + "_c1tmp4"
    # )
    # # data_folder = cell_type + '_' + response_type.upper() + '_metadata'
    # E.run(
    #     ret,
    #     conditions,
    #     metadata,
    #     idx,
    #     cond_array_list,
    #     n_trials=n_trials,
    #     data_folder=data_folder_path,
    #     save_only_metadata=False,
    # )
