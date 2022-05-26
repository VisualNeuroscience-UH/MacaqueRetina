# Builtin
from pathlib import Path
from argparse import ArgumentError
import copy
import pdb
import sys

# Analysis
import numpy as np
import pandas as pd




class ProjectUtilities():
    '''
    Utilities for ProjectManager class. This class is not instantiated. It serves as a container for project independent helper functions.
    '''
    
    def round_to_n_significant(self, value_in, significant_digits=3):

        boolean_test = value_in != 0

        if boolean_test and not np.isnan(value_in):
            int_to_subtract = significant_digits - 1
            value_out = round(value_in, -int(np.floor(np.log10(np.abs(value_in))) - int_to_subtract))
        else:
            value_out = value_in
            
        return value_out    

    def destroy_from_folders(self, path=None, dict_key_list=None):
        '''
        Run destroy_data from root folder, deleting selected variables from data files one level towards leafs.
        '''

        if path is None:
            p = Path('.')
        elif isinstance(path, Path):
            p = path
        elif isinstance(path, str):
            p = Path(path)
            if not p.is_dir():
                raise ArgumentError(f'path argument is not valid path, aborting...')

        folders = [x for x in p.iterdir() if x.is_dir()]
        metadata_full = []
        for this_folder in folders:
            for this_file in list(this_folder.iterdir()):
                if 'metadata' in str(this_file):
                    metadata_full.append(this_file.resolve())

        for this_metadata in metadata_full:
            try:
                print(f'Updating {this_metadata}')
                updated_meta_full, foo_df = self.update_metadata(this_metadata)
                self.destroy_data(updated_meta_full, dict_key_list=dict_key_list)
            except FileNotFoundError:
                print(f'No files for {this_metadata}, nothing changed...')

    def destroy_data(self, meta_fname, dict_key_list=None ):
        """
        Sometimes you have recorded too much and you want to reduce the filesize by removing some data.

        For not manipulating accidentally data in other folders (from path context), this method works only either at the metadata folder or with full path.

        :param meta_fname: str or pathlib object, metadata file name or full path
        :param dict_key_list: list, list of dict keys to remove from the file.
        example dict_key_list={'vm_all' : ['NG1_L4_CI_SS_L4', 'NG2_L4_CI_BC_L4']}

        Currently specific to destroying the second level of keys, as in above example.
        """
        if dict_key_list is None:
            raise ArgumentError(dict_key_list, 'dict_key_list is None - nothing to do, aborting...')
        
        if Path(meta_fname).is_file() and 'metadata' in str(meta_fname):
            meta_df = self.data_io.get_data(meta_fname)
        else:
            raise FileNotFoundError('The first argument must be valid metadata file name in current folder, or full path to metadata file')

        def format(filename, dict_key_list):
            # This will destroy the selected data
            data_dict = self.data_io.get_data(filename)
            for key in dict_key_list.keys():
                for key2 in dict_key_list[key]:
                    try:
                        del data_dict[key][key2]
                    except KeyError:
                        print(f'Key {key2} not found, assuming removed, nothing changed...')
                        return
            self.data_io.write_to_file(filename, data_dict)

        for filename in meta_df['Full path']:
            if Path(filename).is_file():
                format(filename, dict_key_list)

    def pp_df_full(self, df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1): 
            print(df)

    def end2idx(self, t_idx_end, n_samples):

        if t_idx_end is None:
            t_idx_end = n_samples
        elif t_idx_end < 0:
            t_idx_end = n_samples + t_idx_end
        return t_idx_end

    def metadata_manipulator(self, meta_full=None, filename=None, multiply_rows=1, replace_dict={}):
        '''
        Replace strings in a metadata file.
        :param path: str or pathlib object
        :param filename: str, metadata filename, if empty, search most recent in path
        :param replace_dict: dict, 
            keys: 'columns', 'find' and 'replace'
            values: lists of same length  
            key: 'rows'
            values: list of row index values (as in df.loc) for the changes to apply
        '''

        if meta_full is None:
            raise ArgumentError('Need full path to metadatafile, aborting...')

        if not replace_dict:
            raise ArgumentError('Missing replace dict, aborting...')

        data_df = self.data_io.load_from_file(meta_full)

        # multiply rows by factor multiply_rows
        multiply_rows = 2
        new_df = pd.DataFrame(np.repeat(data_df.values, multiply_rows, axis=0), columns = data_df.columns)

        for this_row in replace_dict['rows']:
            for col_idx, this_column in enumerate(replace_dict['columns']):
                f = replace_dict['find'][col_idx]
                r = replace_dict['replace'][col_idx]
                print(f'Replacing {f=} for {r=}, for {this_row=}, {this_column=}')
                new_df.loc[this_row][this_column] = (
                str(new_df.loc[this_row][this_column]).replace(f,r) # str method
                )

        self.pp_df_full(new_df)
        new_meta_full = self._write_updated_metadata_to_file(meta_full, new_df)
        print(f'Created {new_meta_full}')

    # Debugging
    def pp_df_memory(self, df):
        BYTES_TO_MB_DIV = 0.000001
        mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
        print("Memory usage is " + str(mem) + " MB")

    def pp_obj_size(self, obj):
        from IPython.lib.pretty import pprint 
        pprint(obj)
        print(f'\nObject size is {sys.getsizeof(obj)} bytes')

    def get_added_attributes(self, obj1, obj2):

        XOR_attributes = set(dir(obj1)).symmetric_difference(dir(obj2))
        unique_attributes_list = [n for n in XOR_attributes if not n.startswith('_')]
        return unique_attributes_list

    def pp_attribute_types(self, obj, attribute_list=[]):

        if not attribute_list:
            attribute_list = dir(obj)

        for this_attribute in attribute_list:
            attribute_type = eval(f'type(obj.{this_attribute})')
            print(f'{this_attribute}: {attribute_type}')

    def countlines(self, start, lines=0, header=True, begin_start=None):
        # Counts lines in folder .py files.
        # From https://stackoverflow.com/questions/38543709/count-lines-of-code-in-directory-using-python
        if header:
            print('{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'FILE'))
            print('{:->11}|{:->11}|{:->20}'.format('', '', ''))

        for thing in Path.iterdir(start):
            thing = Path.joinpath(start, thing)
            if thing.is_file():
                if thing.endswith('.py'):
                    with open(thing, 'r') as f:
                        newlines = f.readlines()
                        newlines = len(newlines)
                        lines += newlines

                        if begin_start is not None:
                            reldir_of_thing = '.' + thing.replace(begin_start, '')
                        else:
                            reldir_of_thing = '.' + thing.replace(start, '')

                        print('{:>10} |{:>10} | {:<20}'.format(
                                newlines, lines, reldir_of_thing))


        for thing in Path.iterdir(start):
            thing = Path.joinpath(start, thing)
            if Path.is_dir(thing):
                lines = self.countlines(thing, lines, header=False, begin_start=start)

        return lines

