import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pyart
import wradlib as wrl
import h5py
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import yaml

### config/parameters ### TODO create a config file and map to variables instead of this
# root directory - at this point we iterate only through one folder with all files (tested on data from 2016-01-11) - TODO
rootdir = r'C:\Users\marti\kinit\radar-dataset-preprocess\2016-01-11'
# radar pictures related parameters
image_capture_interval = 300 # interval between two radar image captures in seconds
grid_shape = [1, 336, 336] # TODO map this to numbers in code
grid_limits = ((2000, 2000), (-170000.0, 170000.0), (-170000.0, 170000.0)) # TODO map this to numbers in code + possibly change in config to separate z+y+x fields
a = 200.0 # parameter a of the Z/R relationship Standard value according to Marshall-Palmer is a=200
b = 1.6 # parameter b of the Z/R relationship Standard value according to Marshall-Palmer is b=1.6
rainy_pxl_threshold = 0.05 # threshold for determining whether the pixel of radar picture is rainy or not (mm of rain per capture interval)
rainy_img_threshold = 0.2 # threshold for determining whether the radar picture is rainy or not (percentage of rainy pixels in image)
# parameters related to training of NN
input_length = 6 # how many images we want to base our prediction on
target_length = 6 # how many images we want to forecast into future
test_size = 0.15 # percentage split of the full dataset used for the test set. Should be a float in the range (0, 1).
val_size = 0.15 # percentage split of the training set used for the validation set. Should be a float in the range [0, 1).
batch_size = 8 # how many samples per batch to load in DataLoader
shuffle = False # whether to shuffle the train/validation indices
pin_memory = False # whether to copy tensors into CUDA pinned memory. Set it to True if using GPU
target_timestamp = False # returns target observation timestamp if set to true
###-------------------###

# get paths to all files
files = glob.glob(rootdir + '/*.hdf', recursive=True)

# ratios array
ratios = np.array([])
# timestamps datetime array
timestamps = np.array([], dtype=np.datetime64)

for file in files:
    # read raw radar data from hdf file
    radar = pyart.aux_io.read_odim_h5(file)
    
    # perform Cartesian mapping, limit to the reflectivity field.
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(1, 336, 336), # TODO change numbers to configureable params
        grid_limits=((2000, 2000), (-170000.0, 170000.0), (-170000.0, 170000.0)), # TODO change numbers to configureable params
        fields=['reflectivity_horizontal'])
    
    # data is in dBZ (decibel of the reflectivity factor Z) + convert to np.array
    dBZ = np.array(grid.fields['reflectivity_horizontal']['data'][0])

    # convert from reflectivity to reflectivity factor Z (units mm^6/h^3)
    Z = wrl.trafo.idecibel(dBZ)
    # convert to rainfall intensity (unit: mm/h) using the Marshall-Palmer Z(R) parameters
    R = wrl.zr.z_to_r(Z, a=a, b=b)
    # convert to rainfall depth (mm) assuming a rainfall duration of five minutes (i.e. 300 seconds)
    depth = wrl.trafo.r_to_depth(R, image_capture_interval)
    # compute ratio of rainy to all pixels in the precipitation map
    ratio = len(depth[depth > rainy_pxl_threshold]) / depth.size
    ratios = np.append(ratios, ratio)
    # timestamp of precipitation map
    timestamp = np.datetime64(datetime.strptime(file.split('_')[-1].split('.')[0], '%Y%m%d%H%M%S')) # TODO be able to configure dateformat and split symbols to support different file names
    timestamps = np.append(timestamps, timestamp)
    print(file, f"Loaded {len(ratios)}/{len(files)} files")

print("File load done.")

# which of the maps are above a certain threshold
maps_above_thres = ratios > rainy_img_threshold

# indices of target observations
target_obs_idx = maps_above_thres.nonzero()[0]

# datetimes of target observations
target_obs_idx = maps_above_thres.nonzero()[0]

# check if between target and lead observation are not missing observations
# what is the time delta between target and lead
deltas = timestamps[target_obs_idx] - timestamps[np.subtract(target_obs_idx, target_length + input_length)]
# if delta is different from the time that should be between target and lead remove it
target_obs_idx = target_obs_idx[(deltas == timedelta(seconds=((target_length+input_length)*image_capture_interval)))]

# final image masks - which images are chosen for final dataset
final_image_mask = np.zeros(len(ratios), dtype=bool)
# target image masks - which images are target images
target_image_mask = np.zeros(len(ratios), dtype=bool)

# set chosen indices to True
target_image_mask[target_obs_idx] = True

# choose images that are away from target observation for a set range
for shift in range(0, target_length + input_length):
    final_image_mask[np.subtract(target_obs_idx, shift)] = True

# empty array for saving chosen precipitation maps
prec_maps = np.empty((sum(final_image_mask),336,336), dtype=np.single) # TODO change numbers to configureable params - same numbers somewhere in code

# indices of filtered observations
obs_idx = final_image_mask.nonzero()[0]

for i in range(sum(final_image_mask)):
    # file path
    file = files[final_image_mask.nonzero()[0][i]]
    
    # read raw radar data from hdf file
    radar = pyart.aux_io.read_odim_h5(file)
    
    # perform Cartesian mapping, limit to the reflectivity field.
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(1, 336, 336), # TODO change numbers to configureable params - duplicate
        grid_limits=((2000, 2000), (-170000.0, 170000.0), (-170000.0, 170000.0)), # TODO change numbers to configureable params - duplicate
        fields=['reflectivity_horizontal'])
    
    # data is in dBZ (decibel of the reflectivity factor Z) + convert to np.array
    dBZ = np.array(grid.fields['reflectivity_horizontal']['data'][0])
    
    # adds to array
    prec_maps[i,:,:] = dBZ[None,:]
    print(file, f"Appended {i+1}/{len(range(sum(final_image_mask)))} suitable files")

print("Suitable files appended.")

# shift target indices such that first target is equal to target_length+input_length-1
shift_target_obs_idx = np.array(list(map(lambda x: np.where(obs_idx == x)[0][0], target_obs_idx)))

# save "after cleaning" variables as dict for easier work
clean_data = {'prec_maps': prec_maps, 'target_idx': shift_target_obs_idx, 'timestamps': timestamps[obs_idx]}

# custom dataset class for torch
class prec_maps_torch_dataset(Dataset):
    def __init__(self, input_data, num_input_images=input_length, num_output_images=target_length, target_timestamp=target_timestamp):
        # inherit methods and properties from parent class
        super(prec_maps_torch_dataset, self).__init__()
        
        # does user want to get timestamp of target obs?
        self.target_timestamp = target_timestamp
        # length of input (how many radar maps to take as an input)
        self.num_input = num_input_images
        # length of output (how many radar maps to take as an output)
        self.num_output = num_output_images
        
        # save target timestamps if user wants them
        if self.target_timestamp:
            self.timestamps = input_data['timestamps']
        
        # save precipitation maps and target obs indices
        self.prec_maps = input_data['prec_maps']
        self.targets = input_data['target_idx']
        
        # length as count of target observations
        self.length = len(self.targets)

    def __getitem__(self, index):
        # save a num_input+num_output window at certain index
        imgs = self.prec_maps[(self.targets[index]-(self.num_input+self.num_output)+1):(self.targets[index]+1)]
        # return timestamp as datetime object
        if self.target_timestamp:
            timestamp = self.timestamps[self.targets[index]].astype(object)
            timestamp = np.array([timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute])
        
        # slice img to input and target
        input_img = imgs[:self.num_input]
        target_img = imgs[-self.num_output:]
        
        # return results
        if self.target_timestamp:
            return timestamp, input_img, target_img
        else:
            return input_img, target_img

    def __len__(self):
        return self.length
    
data_for_torch = prec_maps_torch_dataset(input_data=clean_data, target_timestamp=True)

# Based on: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
def get_loaders(data,
                batch_size=batch_size,
                num_input_images=input_length,
                num_output_images=target_length,
                target_timestamp=target_timestamp,
                test_size=test_size,
                valid_size=val_size,
                shuffle=shuffle,
                pin_memory=pin_memory):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the precipitation radar images dataset.
    If using CUDA, set pin_memory to True.
    Params
    ------
    - data: data of custom torch Dataset class.
    - batch_size: how many samples per batch to load.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - test_loader: test set iterator.
    """
    error_msg = "[!] valid_size should be in the range <0, 1)."
    assert ((valid_size >= 0) and (valid_size < 1)), error_msg
    error_msg = "[!] test_size should be in the range (0, 1)."
    assert ((test_size > 0) and (test_size < 1)), error_msg

    # load the dataset
    dataset = data
    
    # get length, indices and split int for train/test
    dataset_length = len(dataset)
    indices = list(range(dataset_length))
    split = int(np.floor((1 - test_size) * dataset_length))
    
    # indices of train/test split
    full_train_idx, test_idx = indices[:split], indices[split:]
    
    # subset elements from given indices
    test_sampler = SubsetRandomSampler(test_idx)
    
    # if valid_size provided, split train into train/val
    if valid_size > 0:
        split = int(np.floor((1 - valid_size) * len(full_train_idx)))
        if shuffle:
            np.random.shuffle(full_train_idx)
        train_idx, valid_idx = full_train_idx[:split], full_train_idx[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
    else:
        train_sampler = SubsetRandomSampler(full_train_idx)
    
    # load train into DataLoader 
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=pin_memory
    )
    
    # load test/val into DataLoader
    if valid_size > 0:
        valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=pin_memory
        )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler, pin_memory=pin_memory
    )
    
    # return DataLoaders
    if valid_size > 0:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader

print("Saving radar_dataloader.pth to", os.getcwd())

# save dataloader to current dir
torch.save(get_loaders(data_for_torch), 'radar_dataloader.pth')

print("Done.")
