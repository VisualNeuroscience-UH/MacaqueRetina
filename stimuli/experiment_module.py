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
        )

    def contrast_respose(self, contrast_min=0.98, contrast_max=0.98, contrast_steps=1):
        """
        Setup
        """

        contrasts = np.logspace(
            np.log10(contrast_min), np.log10(contrast_max), contrast_steps
        )

        # Calculate voltage values, assuming voltage = Td
        # Return cond_options -- a dict with all keywords matching visual_stimulus_module.ConstructStimulus
        # conditions_to_meshgrid = ["contrast", "temporal_frequency"]
        conditions_to_meshgrid = ["contrast"]

        (
            cond_options,
            cond_metadata_key,
            cond_names,
        ) = self._meshgrid_conditions(
            conditions_to_meshgrid,
            contrasts,
        )

        # Return a nice list with all conditions to run
        contrast_experiment = {
            "cond_options": cond_options,  # list of dicts
            "cond_metadata_key": cond_metadata_key,  # dict
            "cond_names": cond_names,  # list of strings
        }

        return contrast_experiment

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

    def _create_dataframe(self, metadata, cond_names, options, video_name_list):
        df = pd.DataFrame(index=options.keys(), columns=cond_names)

        # Set all values equal to options.values()
        for key, value in options.items():
            df.loc[key] = value

        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()  # Convert numpy array to list

            df.loc[key] = value

        # Update stimulus_video_name
        df.loc["stimulus_video_name"] = video_name_list

        return df

    def run(
        self,
        this_experiment,
        n_trials=1,
        save_only_metadata=False,
    ):
        """
        Unpack and run all conditions
        """

        cond_options = this_experiment["cond_options"]
        metadata = this_experiment["cond_metadata_key"]
        cond_names = this_experiment["cond_names"]

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
            [metadata, cond_names, self.options],
        )

        if save_only_metadata:
            return

        video_name_list = []
        # Replace with input options
        for idx, input_options in enumerate(cond_options):
            stimulus_video_name = "Stim_" + cond_names[idx]
            input_options["stimulus_video_name"] = stimulus_video_name
            video_name_list.append(stimulus_video_name)
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

        result_df = self._create_dataframe(
            metadata, cond_names, self.options, video_name_list
        )
        # Save dataframe
        save_path = data_folder / "metadata_conditions.csv"
        result_df.to_csv(save_path)


if __name__ == "__main__":
    pass
