### WORK IN PROGRESS ###
# should convert h5 files to torch dataloader in the future #
import os
import time
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
    - valid_size: percentage split of the training set used for the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
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