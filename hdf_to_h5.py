import os
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pyart
import wradlib as wrl
import h5py
import glob
import yaml

# for returning time elapsed running the script
tic = time.time()

# load config file
config_file = open("config_hdf_to_h5.yaml", "r")
config = yaml.safe_load(config_file)

# config/parameters
rootdir = config['rootdir'] # root directory - at this point we iterate only through one folder with all files - TODO
outdir = config['outdir'] # output directory
# radar pictures related parameters
image_capture_interval = config['image_capture_interval'] # interval between two radar image captures in seconds
grid_shape =  tuple(config['grid_shape'])
grid_limits = tuple([tuple(config['grid_limits'][0]), tuple(config['grid_limits'][1]), tuple(config['grid_limits'][2])])
a = config['a'] # parameter a of the Z/R relationship. Standard value according to Marshall-Palmer is a=200
b = config['b'] # parameter b of the Z/R relationship. Standard value according to Marshall-Palmer is b=1.6
rainy_pxl_threshold = config['rainy_pxl_threshold'] # threshold for determining whether the pixel of radar picture is rainy or not (mm of rain per capture interval)
rainy_img_threshold = config['rainy_img_threshold'] # threshold for determining whether the radar picture is rainy or not (percentage of rainy pixels in image)
input_length = config['input_length'] # how many images we want to base our prediction on
target_length = config['target_length'] # how many images we want to forecast into future

# get paths to all files
files = glob.glob(rootdir + '/*.hdf', recursive=True)

# ratios array
ratios = np.array([])
# timestamps datetime array
timestamps = np.array([], dtype=np.datetime64)

try:
    for file in files:
        # read raw radar data from hdf file
        radar = pyart.aux_io.read_odim_h5(file)
        
        # perform Cartesian mapping, limit to the reflectivity field.
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=grid_shape,
            grid_limits=grid_limits,
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
except:
    logging.exception(f"An exception occured while processing {file}.")

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
prec_maps = np.empty((sum(final_image_mask),grid_shape[1],grid_shape[2]), dtype=np.single)

# indices of filtered observations
obs_idx = final_image_mask.nonzero()[0]

# shift target indices such that first target is equal to target_length+input_length-1
shift_target_obs_idx = np.array(list(map(lambda x: np.where(obs_idx == x)[0][0], target_obs_idx)))

try:
    for i in range(sum(final_image_mask)):
        # file path
        file = files[final_image_mask.nonzero()[0][i]]
        
        # read raw radar data from hdf file
        radar = pyart.aux_io.read_odim_h5(file)
        
        # perform Cartesian mapping, limit to the reflectivity field.
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=grid_shape,
            grid_limits=grid_limits,
            fields=['reflectivity_horizontal'])
        
        # data is in dBZ (decibel of the reflectivity factor Z) + convert to np.array
        dBZ = np.array(grid.fields['reflectivity_horizontal']['data'][0])
        
        # creates and saves new h5 file to outdir
        hf = h5py.File(os.path.join(outdir, file.split('_')[-1].split('.')[0][0:12] + '.h5'), 'w') # TODO outdir as config parameter
        hf.create_dataset('precipitation_map', data=dBZ, chunks=True)
        hf.close()
        print(file, f"Appended {i+1}/{len(range(sum(final_image_mask)))} suitable files")
except:
    logging.exception(f"An exception occured while processing {file}.")

if len(range(sum(final_image_mask))) == 0:
    print("No suitable files found based on provided parameters.")
else:
    print("Suitable files appended.")

# save meta data for further usage # TODO - export them to h5 metafile
meta_data = {'target_idx': shift_target_obs_idx, 'timestamps': timestamps[obs_idx]} # TODO add to metafile + change these such that after "except" they only keep the obseravtions before except

# time elapsed running the script
toc = round(time.time() - tic, 2)

print(f"Done. Elapsed {toc} seconds.")
