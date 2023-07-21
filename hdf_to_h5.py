import os
import time
from datetime import datetime, timedelta
import numpy as np
import pyart
import wradlib as wrl
import h5py
import glob
import yaml

# for returning time elapsed running the script
tic = time.time()

### config/parameters ### TODO create a config file and map to variables instead of this
# root directory - at this point we iterate only through one folder with all files (tested on data from 2016-01-11) - TODO
rootdir = r'C:\Users\marti\kinit\radar-dataset-preprocess\2016-01-11'
# radar pictures related parameters
image_capture_interval = 300 # interval between two radar image captures in seconds
grid_shape = [1, 340, 340] # TODO map this to numbers in code
grid_limits = ((2000, 2000), (-170000.0, 170000.0), (-170000.0, 170000.0)) # TODO map this to numbers in code + possibly change in config to separate z+y+x fields
a = 200.0 # parameter a of the Z/R relationship. Standard value according to Marshall-Palmer is a=200
b = 1.6 # parameter b of the Z/R relationship. Standard value according to Marshall-Palmer is b=1.6
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
target_timestamp = False # returns target observation timestamp if set to True
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

if len(range(sum(final_image_mask))) == 0:
    print("No suitable files found based on provided parameters.")
else:
    print("Suitable files appended.")

# shift target indices such that first target is equal to target_length+input_length-1
shift_target_obs_idx = np.array(list(map(lambda x: np.where(obs_idx == x)[0][0], target_obs_idx)))

# save "after cleaning" variables as dict for easier work
clean_data = {'prec_maps': prec_maps, 'target_idx': shift_target_obs_idx, 'timestamps': timestamps[obs_idx]}

# time elapsed running the script
toc = round(time.time() - tic, 2)

print(f"Done. Elapsed {toc} seconds.")
