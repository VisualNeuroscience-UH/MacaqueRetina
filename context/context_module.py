# Builtins
import pdb
from pathlib import Path
from typing import Type
from copy import deepcopy
import hashlib
import json
import datetime

import numpy as np

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

    def generate_hash(self, my_dict):
        """
        Generate a hash from the input dictionary with all values converted to a format suitable for JSON serialization and
        return a hash string trimmed to the necessary length as determined by internal logic.

        Parameters
        ----------
        my_dict : dict
            A dictionary containing parameters to be hashed. This dictionary can include complex data types such as complex numbers,
            datetime objects, numpy arrays, sets, and tuples.

        Returns
        -------
        str
            A portion of the SHA-256 hash of the serialized input dictionary as a hexadecimal string, with the length based on
            the calculated necessary hash length.

        Raises
        ------
        TypeError
            If an object in the dictionary cannot be serialized to JSON and does not match any of the specified types in the helper
            function.

        Notes
        -----
        Uses helper functions `_serialize_nonhashable_object` to convert non-serializable objects and `_calculate_hash_length` to
        determine the length of the hash to return based on the content of the dictionary.
        """

        def _serialize_nonhashable_object(obj):
            if isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                try:
                    return obj.__dict__
                except AttributeError:
                    raise TypeError(
                        f"Object of type {obj.__class__.__name__} is not JSON serializable"
                    )

        def _calculate_hash_length(data):
            """Calculate the necessary hash length based on the complexity and diversity of the data."""
            # Example complexity measure: count of keys times average string length of values
            avg_value_length = sum(len(str(value)) for value in data.values()) / len(
                data
            )
            return int(
                min(64, max(12, avg_value_length / 2))
            )  # Return between 12 and 64 characters

        # Serialize the parameters and encode them to bytes
        params_str = json.dumps(
            my_dict, default=_serialize_nonhashable_object, sort_keys=True
        ).encode()

        # Generate the hash
        hash_object = hashlib.sha256(params_str)
        hash_digest = hash_object.hexdigest()

        # Calculate the necessary hash length and return the required portion of the hash
        necessary_length = _calculate_hash_length(my_dict)
        return hash_digest[:necessary_length]
