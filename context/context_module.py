# Builtins
from pathlib import Path
import pdb
from typing import Type

from context.context_module_base import ContextBase


class Context(ContextBase):
    '''
    Set context (paths, filenames, etc) for the project.
    Each module orders self.context.property name from context by calling set_context()
    Paths and folders become pathlib.Path object in construction (__init__)
    '''

    def __init__(self, all_properties) -> None:

        self.validated_properties = self._validate_properties(all_properties)

    def set_context(self, _properties_list=[]):
        '''
         Each module orders object_instance.context.property name from context by calling set_context(). Empty list provides all properties.
         '''
        
        
        if isinstance(_properties_list,list):
            pass
        elif isinstance(_properties_list,str):
            _properties_list = [_properties_list]        
        else:
            raise TypeError('properties list must be a list or a string, aborting...')

        for attr, val in self.validated_properties.items():

            if len(_properties_list) > 0 and attr in _properties_list:
                setattr(self, attr, val) 
            elif len(_properties_list) > 0 and attr not in _properties_list:
                pass
            else:
                setattr(self, attr, val) 

        return self

    def _validate_properties(self, all_properties):

        validated_properties = {}

        # Validate main project path
        if 'path' not in all_properties.keys():
            raise KeyError('"path" key is missing, aborting...')
        elif not Path(all_properties['path']).is_dir:
            raise KeyError('"path" key is not a valid path, aborting...')
        elif not Path(all_properties['path']).is_absolute(): 
            raise KeyError('"path" is not absolute path, aborting...')

        path = Path(all_properties['path'])
        validated_properties['path'] = path

        # Check input and output folders
        if 'input_folder' not in all_properties.keys():
            raise KeyError('"input_folder" key is missing, aborting...')
        input_folder = all_properties['input_folder']
        if Path(input_folder).is_relative_to(path):
            validated_properties['input_folder'] = input_folder
        elif path.joinpath(input_folder).is_dir:    
            validated_properties['input_folder'] = path.joinpath(input_folder)

        if 'output_folder' not in all_properties.keys():
            raise KeyError('"output_folder" key is missing, aborting...')
        output_folder = all_properties['output_folder']
        if Path(input_folder).is_relative_to(path):
            validated_properties['output_folder'] = output_folder
        elif path.joinpath(output_folder).is_dir:    
            validated_properties['output_folder'] = path.joinpath(output_folder)

        # Remove validated keys before the loop
        for k in  ['path','input_folder','output_folder']:
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
            elif 'file' in attr:
                validated_properties[attr] = Path(val)
            elif 'folder' in attr:
                validated_properties[attr] = Path(val)
            elif isinstance(val, str):
                validated_properties[attr] = val

        return validated_properties