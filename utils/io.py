import numpy as np
import numpy.ma as ma
from pathlib import Path
from attridict import AttriDict as Dict
import yaml
import h5py
import tarfile
import os

def load_config(file):
    """
    Load configuration from YAML file.
    
    """
    with open(file, "r") as f:
        conf_dict = Dict(yaml.safe_load(f))
    return conf_dict


def convert_uint_to_float(data, datamin, datamax, depth, mask_val=0):
    """
    Convert uint array to float array.
    
    """
    datamin = float(datamin)
    datamax = float(datamax)
    
    mask = data == mask_val
    ma_data = ma.masked_array(data, mask=mask)
    ma_data = (ma_data - 1) / (2 ** depth - 2) * (datamax - datamin) + datamin
    return ma_data

def load_timestamps(file: Path):
    """
    Load timestamps written to txt file.
    
    """
    array = []
    with open(file) as f:
            f = f.readlines()

    for line in f:
        array.append(line.split(' ')[0])
    
    return array

def convert_float_to_uint(data, datamin, datamax, depth, target_type):
    """
    Convert float array to uint array.
    
    """
    datamin = float(datamin)
    datamax = float(datamax)
    
    maskmin = np.nonzero(data < datamin)
    maskmax = np.nonzero(data > datamax)
    
    data[maskmin] = datamin
    data[maskmax] = datamax
    
    new_data = np.round(((data - datamin) / (datamax - datamin) * (2 ** depth - 2)) + 1)
    new_data[np.isnan(new_data)] = 0
    new_data = new_data.astype(target_type)
    
    return new_data


def write_attrs(group: h5py.Group, attrs: dict):
    """
    Write dict of attributes to h5 file.
    
    """
    for k, val in attrs.items():
        group.attrs[k] = val

        
def write_image(group: h5py.Group, ds_name: str, data: np.ndarray, what_attrs: dict):
    """
    Write np.array as h5 dataset to h5 group in h5 file.
    
    """
    try:
        del group[ds_name]
    except:
        pass
    dataset = group.create_dataset(
        ds_name, data=data, dtype="uint8", compression="gzip", compression_opts=9
    )

    ds_what = group.require_group("what")
    for k, val in what_attrs.items():
        ds_what.attrs[k] = val
        
def get_members(tf, subfolder_name="YYYY-MM-DD/"):
    """
    Yields members of given tar file.
    
    """
    l = len(subfolder_name)
    for member in tf.getmembers():
        if member.path.endswith(".vol"):
            member.path = member.path[l:]
            yield member
            
def extract_all(tf, out_dir):
    """
    Extracts all members of given tar file.
    
    """
    for item in tf:
        if item.name.endswith('.vol'):
            item.name = os.path.basename(item.name)
            tf.extract(item, str(out_dir))
    
            
def directory_is_empty(directory) -> bool:
    """
    Check if directory of Path type is empty.
    
    """
    return not any(directory.iterdir())

def rmtree(top):
    """
    Remove all files from directory of Path type.
    
    """
    for root, dirs, files in top.walk(top_down=False):
        for name in files:
            (root / name).unlink()
        for name in dirs:
            (root / name).rmdir()