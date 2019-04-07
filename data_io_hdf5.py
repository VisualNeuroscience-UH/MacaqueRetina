import numpy as np
import h5py
import os
import pdb

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)

def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, tuple, float)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif item is None:
            h5file[path + key] = ''
        else:
            pdb.set_trace()
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/')

def _recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def save_array_to_hdf5(array, filename):
    """
    ....
    """
    assert isinstance(array, np.ndarray), f'Cannot save {type(array)} type'
    with h5py.File(filename, 'w') as hdf5_file_handle:
        dset = hdf5_file_handle.create_dataset("array", data=array)     

def load_array_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as hdf5_file_handle:
       array = hdf5_file_handle['array'][...]
    return array
    
if __name__ == '__main__':

    data = {'x': 'astring',
            'y': np.arange(10),
            'd': {'z': np.ones((2,3)),
                  'b': b'bytestring'}}
    print(data)
    filename = 'test.h5'
    save_dict_to_hdf5(data, filename)
    dd = load_dict_from_hdf5(filename)
    print(dd)
    # should test for bad type