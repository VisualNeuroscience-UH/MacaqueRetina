# Builtins
from pathlib import Path
import pdb
from typing import Type
from copy import deepcopy

from context.context_base_module import ContextBase


class Context(ContextBase):
    """
    Set context (paths, filenames, etc) for the project.
    Each module orders self.context.property name from context by calling set_context()
    Paths and folders relative to main project path become pathlib.Path object in construction (__init__)
    Variable names with either 'file' or 'folder' in the name become pathlib.Path object in construction (__init__)
    """

    def __init__(self, all_properties) -> None:
        self.validated_properties = self._validate_properties(all_properties)

    def set_context(self, class_instance):
        """
        Each module orders class_instance.context.property name from context by calling set_context().
        Empty list provides all properties.

        The context object is separate for each processed class_instance, but the class_instance.context.attribute
        is always the same as context.attribute. Thus it will always point to the project manager context
        (input arguments to the project manager init).

        Parameters
        ----------
        class_instance : object
            Class instance that calls set_context().
        """

        if hasattr(class_instance, "_properties_list"):
            _properties_list = class_instance._properties_list
        else:
            _properties_list = []

        if isinstance(_properties_list, list):
            pass
        elif isinstance(_properties_list, str):
            _properties_list = [_properties_list]
        else:
            raise TypeError("properties list must be a list or a string, aborting...")

        # Make a copy of the context object, so that you do not always return the same object
        _context = deepcopy(self)

        for attr, val in self.validated_properties.items():
            if len(_properties_list) > 0 and attr in _properties_list:
                setattr(_context, attr, val)
            elif len(_properties_list) > 0 and attr not in _properties_list:
                pass
            else:
                setattr(_context, attr, val)

        return _context

    def _validate_properties(self, all_properties):
        validated_properties = {}

        # Validate main project path
        if "path" not in all_properties.keys():
            raise KeyError('"path" key is missing, aborting...')
        elif not Path(all_properties["path"]).is_dir:
            raise KeyError('"path" key is not a valid path, aborting...')
        elif not Path(all_properties["path"]).is_absolute():
            raise KeyError('"path" is not absolute path, aborting...')

        path = Path(all_properties["path"])
        validated_properties["path"] = path

        # Check input and output folders
        if "output_folder" not in all_properties.keys():
            raise KeyError('"output_folder" key is missing, aborting...')
        output_folder = all_properties["output_folder"]
        if Path(output_folder).is_relative_to(path):
            validated_properties["output_folder"] = output_folder
        elif path.joinpath(output_folder).is_dir:
            validated_properties["output_folder"] = path.joinpath(output_folder)

        if "stimulus_folder" not in all_properties.keys():
            raise KeyError('"stimulus_folder" key is missing, aborting...')
        stimulus_folder = all_properties["stimulus_folder"]
        if Path(stimulus_folder).is_relative_to(path):
            validated_properties["stimulus_folder"] = stimulus_folder
        elif path.joinpath(stimulus_folder).is_dir:
            validated_properties["stimulus_folder"] = path.joinpath(stimulus_folder)

        if "input_folder" not in all_properties.keys():
            raise KeyError('"input_folder" key is missing, aborting...')
        input_folder = all_properties["input_folder"]
        if Path(input_folder).is_relative_to(path):
            validated_properties["input_folder"] = input_folder
        elif path.joinpath(input_folder).is_dir:
            validated_properties["input_folder"] = path.joinpath(input_folder)

        # Create the output, stimulus and input folders if they don't exist
        if not validated_properties["output_folder"].is_dir():
            validated_properties["output_folder"].mkdir(parents=True)
        if not validated_properties["stimulus_folder"].is_dir():
            validated_properties["stimulus_folder"].mkdir(parents=True)
        if not validated_properties["input_folder"].is_dir():
            validated_properties["input_folder"].mkdir(parents=True)

        # Remove validated keys before the loop
        for k in ["path", "input_folder", "output_folder", "stimulus_folder"]:
            all_properties.pop(k, None)

        for attr, val in all_properties.items():
            if val is None:
                validated_properties[attr] = val
            elif isinstance(val, int):
                validated_properties[attr] = val
            elif isinstance(val, dict):
                validated_properties[attr] = val
            elif Path(val).is_relative_to(path):
                validated_properties[attr] = Path(val)
            elif "file" in attr:
                validated_properties[attr] = Path(val)
            elif "folder" in attr:
                validated_properties[attr] = Path(val)
            elif "path" in attr:
                validated_properties[attr] = Path(val)
            elif isinstance(val, str):
                validated_properties[attr] = val

        return validated_properties
