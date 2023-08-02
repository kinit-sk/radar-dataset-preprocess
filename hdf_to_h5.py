import os
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pyart
import h5py
import yaml

def convert_hdf_to_h5(configuration_file: str):
    # for returning time elapsed running the script
    tic = time.time()

    # load config file
    config_file = open(configuration_file, "r")
    config = yaml.safe_load(config_file)

    # config/parameters 
    outdir = config['outdir'] # output directory
    # radar pictures related parameters
    image_capture_interval = config['image_capture_interval'] # interval between two radar image captures in seconds
    grid_shape =  tuple(config['grid_shape']) # number of points in the grid (z, y, x)
    grid_limits = tuple([tuple(config['grid_limits'][0]), tuple(config['grid_limits'][1]), tuple(config['grid_limits'][2])]) # minimum and maximum grid location (inclusive) in meters for the z, y, x coordinates
    aggregate = config['aggregate'] # how to aggregate precipitation from radar maps - CAPPI/CMAX supported (grid parameters must be configured appropriately)
    rainy_img_threshold = config['rainy_img_threshold'] # threshold for determining whether the radar picture is rainy or not (percentage of rainy pixels in image)
    input_length = config['input_length'] # how many images we want to base our prediction on
    target_length = config['target_length'] # how many images we want to forecast into future

    # load ratios file from outdir
    hf = h5py.File(os.path.join(outdir, 'ratios.h5'), 'r')
    # save them to variables
    ratios = np.array(hf['ratios'])
    timestamps = np.array([np.datetime64(datetime.strptime(item.decode(), '%Y-%m-%dT%H:%M:%S.%fZ')) for item in hf['timestamps']])
    filepaths = [item.decode() for item in hf['filepaths']]

    hf.close()

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

    # indices of filtered observations
    obs_idx = final_image_mask.nonzero()[0]

    # shift target indices such that first target is equal to target_length+input_length-1
    shift_target_obs_idx = np.array(list(map(lambda x: np.where(obs_idx == x)[0][0], target_obs_idx)))

    try:
        for i in range(sum(final_image_mask)):
            # file path
            file = filepaths[final_image_mask.nonzero()[0][i]]
            
            # read raw radar data from hdf file
            radar = pyart.aux_io.read_odim_h5(file)
            
            # perform Cartesian mapping, limit to the reflectivity field.
            grid = pyart.map.grid_from_radars(
                (radar,),
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                fields=['reflectivity_horizontal'])
            
            # data is in dBZ (decibel of the reflectivity factor Z) + convert to np.array
            # choice based on inputed aggregate method
            if aggregate == 'CAPPI':
                dBZ = np.array(grid.fields['reflectivity_horizontal']['data'][0])
            elif aggregate == 'CMAX':
                dBZ = np.array(grid.fields['reflectivity_horizontal']['data']).max(axis=0)
            else:
                print('Unsupported aggregate method provided in config file, using CMAX instead.')
                dBZ = np.array(grid.fields['reflectivity_horizontal']['data']).max(axis=0)
            
            # creates and saves new h5 file to outdir
            hf = h5py.File(os.path.join(outdir, file.split('_')[-1].split('.')[0][0:12] + '.h5'), 'w')
            hf.create_dataset('precipitation_map', data=dBZ, chunks=True)
            hf.close()
            print(file, f"Appended {i+1}/{len(range(sum(final_image_mask)))} suitable files")
    except:
        logging.exception(f"An exception occured during appending of files while processing {file}.")
        print("Appending stopped.")

        # keep meta data only about currently appended files
        stop_index = i
        # save meta data for further usage
        meta_data = {'target_idx': shift_target_obs_idx, 'timestamps': timestamps[obs_idx]}
        # subset meta data
        meta_data['target_idx'] = meta_data['target_idx'][np.where(meta_data['target_idx'] < stop_index)]
        meta_data['timestamps'] = meta_data['timestamps'][range(stop_index)]

        # write metadata as h5 file
        mf = h5py.File(os.path.join(outdir, 'metadata.h5'), 'w')
        # datetime to string with utf8 encoding
        utc_str_arr = np.array([np.datetime_as_string(n,timezone='UTC').encode('utf-8') for n in meta_data['timestamps']])
        # save to file
        mf.create_dataset('target_idx', data=meta_data['target_idx'], chunks=True)
        mf.create_dataset('timestamps', data=utc_str_arr, chunks=True)

        mf.close()
    else:
        if len(range(sum(final_image_mask))) == 0:
            print("No suitable files found based on provided parameters.")
        else:
            print("Suitable files appended successfully.")

        # save meta data for further usage
        meta_data = {'target_idx': shift_target_obs_idx, 'timestamps': timestamps[obs_idx]}

        # write metadata as h5 file
        mf = h5py.File(os.path.join(outdir, 'metadata.h5'), 'w')
        # datetime to string with utf8 encoding
        utc_str_arr = np.array([np.datetime_as_string(n,timezone='UTC').encode('utf-8') for n in meta_data['timestamps']])
        # save to file
        mf.create_dataset('target_idx', data=meta_data['target_idx'], chunks=True)
        mf.create_dataset('timestamps', data=utc_str_arr, chunks=True)

        mf.close()

    with open(os.path.join(outdir, 'config_hdf_to_h5.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # time elapsed running the script
    toc = round(time.time() - tic, 2)

    print(f"Done. Elapsed {toc} seconds.")

if __name__ == '__main__':
    convert_hdf_to_h5('config.yaml')
