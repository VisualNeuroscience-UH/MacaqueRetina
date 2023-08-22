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

    def _meshgrid_conditions(self, options):
        # Get all conditions to meshgrid
        conditions_to_meshgrid = list(options.keys())

        # Get all values to meshgrid
        values_to_meshgrid = [
            options[condition] for condition in conditions_to_meshgrid
        ]

        # Get meshgrid of all values
        values = np.meshgrid(*values_to_meshgrid)

        # flatten arrays for easier indexing later
        values_flat = [v.flatten() for v in values]

        # Get cond_names
        conditions_metadata_idx = np.meshgrid(
            *[np.arange(len(v)) for v in values_to_meshgrid]
        )
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

        return (
            cond_options,
            cond_names,
        )

    def _build(self, experiment_dict):
        """
        Setup
        """

        exp_variables = experiment_dict["exp_variables"]
        min_max_values = experiment_dict["min_max_values"]
        n_steps = experiment_dict["n_steps"]
        logaritmic = experiment_dict["logaritmic"]  # True or False

        # Create a dictionary with all options to vary. The keys are the options to vary,
        # the values include n_steps between the corresponding min_max_values. The steps
        # can be linear or logaritmic
        cond_metadata_key = {}
        for idx, option in enumerate(exp_variables):
            if logaritmic[idx]:
                cond_metadata_key[option] = np.logspace(
                    np.log10(min_max_values[idx][0]),
                    np.log10(min_max_values[idx][1]),
                    n_steps[idx] + 1,
                )
            else:
                cond_metadata_key[option] = np.linspace(
                    min_max_values[idx][0], min_max_values[idx][1], n_steps[idx]
                )

        # Calculate voltage values, assuming voltage is linearly associated with photopic Td
        # Return cond_options -- a dict with all keywords matching visual_stimulus_module.ConstructStimulus
        # and values being a list of values to replace the corresponding keyword in the stimulus
        (
            cond_options,
            cond_names,
        ) = self._meshgrid_conditions(cond_metadata_key)

        # Return a nice list with all conditions to run
        conditions_dict = {
            "cond_options": cond_options,  # list of dicts
            "cond_metadata_key": cond_metadata_key,  # dict
            "cond_names": cond_names,  # list of strings
        }

        return conditions_dict

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

    def _create_dataframe(self, cond_options, cond_names, options):
        df = pd.DataFrame(index=options.keys(), columns=cond_names)
        n_columns = len(cond_names)

        # Set all values equal to options.values()
        for key, value in options.items():
            if isinstance(value, tuple):
                repeated_tuple = tuple([value] * n_columns)
                df.loc[key] = repeated_tuple
            else:
                try:
                    df.loc[key] = value
                except:
                    print(key, value)
                    pdb.set_trace()

        for idx, this_dict in enumerate(cond_options):
            for key, value in this_dict.items():
                if isinstance(value, tuple):
                    repeated_tuple = tuple([value] * n_columns)
                    df.loc[key][idx] = repeated_tuple
                else:
                    df.loc[key][idx] = value

        return df

    def build_and_run(
        self,
        experiment_dict,
        n_trials=1,
        build_without_run=False,
    ):
        conditions_dict = self._build(experiment_dict)

        """
        Unpack and run all conditions
        """

        # Get parameters to vary in this experiment
        cond_options = conditions_dict["cond_options"]
        metadata = conditions_dict["cond_metadata_key"]
        cond_names = conditions_dict["cond_names"]

        # First check that experiment metadata keys are valid stimulus options
        for this_key in metadata.keys():
            assert (
                this_key in self.options.keys()
            ), "Missing {this_key} in visual stimuli options, check stim param name"

        # Update options to match my_stimulus_options in conf file
        self._replace_options(self.context.my_stimulus_options)

        # Replace filename with None. If don't want to save the stimulus, None is valid,
        # but if want to save, then filename will be generated in the loop below
        self.options["stimulus_video_name"] = None

        # Update simulation options
        spike_generator_model = self.context.my_run_options["spike_generator_model"]
        simulation_dt = self.context.my_run_options["simulation_dt"]

        data_folder = self.context.output_folder

        if not build_without_run:
            # Replace with input options
            for idx, input_options in enumerate(cond_options):
                # Create stimulus video name. Note, this updates the cond_options dict
                stimulus_video_name = "Stim_" + cond_names[idx]
                input_options["stimulus_video_name"] = stimulus_video_name

                # Replace options with input_options
                self._replace_options(input_options)

                # Try loading existing file, if not found, create stimulus
                try:
                    stim = self.data_io.load_stimulus_from_videofile(
                        stimulus_video_name
                    )
                except:
                    stim = self.stimulate.make_stimulus_video(self.options)
                    # Raw intensity is stimulus specific
                    self.options["raw_intensity"] = stim.options["raw_intensity"]

                self.working_retina.load_stimulus(stim)

                example_gc = None  # int or 'None'

                filename = Path(data_folder) / ("Response_" + cond_names[idx])

                self.working_retina.run_cells(
                    cell_index=example_gc,
                    n_trials=n_trials,
                    save_data=True,
                    spike_generator_model=spike_generator_model,
                    return_monitor=False,
                    filename=filename,
                    simulation_dt=simulation_dt,
                )  # Run simulation

        # Write metadata to csv to log current experiment to its output folder
        self.options["n_trials"] = n_trials
        if len(cond_options[0].keys()) > 1:
            self.options["logaritmic"] = tuple(experiment_dict["logaritmic"])
        else:
            self.options["logaritmic"] = experiment_dict["logaritmic"]
        result_df = self._create_dataframe(cond_options, cond_names, self.options)
        cond_names_string = "_".join(experiment_dict["exp_variables"])
        filename_df = f"exp_metadata_{cond_names_string}.csv"
        save_path = data_folder / filename_df
        # Check if path exists, create parents if not
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_path)


if __name__ == "__main__":
    pass
