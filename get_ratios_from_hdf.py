import os
import time
import logging
from datetime import datetime
import numpy as np
import pyart
import wradlib as wrl
import h5py
import yaml

def get_ratios_from_hdf(configuration_file: str):
    # for returning time elapsed running the script
    tic = time.time()

    # load config file
    config_file = open(configuration_file, "r")
    config = yaml.safe_load(config_file)

    # config/parameters 
    rootdir = config['rootdir'] # root directory
    outdir = config['outdir'] # output directory
    # radar pictures related parameters
    image_capture_interval = config['image_capture_interval'] # interval between two radar image captures in seconds
    grid_shape =  tuple(config['grid_shape']) # number of points in the grid (z, y, x)
    grid_limits = tuple([tuple(config['grid_limits'][0]), tuple(config['grid_limits'][1]), tuple(config['grid_limits'][2])]) # minimum and maximum grid location (inclusive) in meters for the z, y, x coordinates
    a = config['a'] # parameter a of the Z/R relationship. Standard value according to Marshall-Palmer is a=200
    b = config['b'] # parameter b of the Z/R relationship. Standard value according to Marshall-Palmer is b=1.6
    rainy_pxl_threshold = config['rainy_pxl_threshold'] # threshold for determining whether the pixel of radar picture is rainy or not (mm of rain per capture interval)

    # check if ratios file exists in outdir
    # if yes, open it and start appending where you left
    # else start from scratch
    if os.path.exists(os.path.join(outdir, 'ratios.h5')):
        print('File already exists. Loading only missing data.')

        hf = h5py.File(os.path.join(outdir, 'ratios.h5'), 'r+')
        try:
            for idx in range(len(hf['filepaths'])):
                file = hf['filepaths'][idx].decode()
                # when hf is empty, add yet to be added data
                if hf['timestamps'][idx] == b'':
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
                    # timestamp of precipitation map
                    timestamp = np.datetime64(datetime.strptime(file.split('_')[-1].split('.')[0], '%Y%m%d%H%M%S'))

                    # append to file
                    hf["ratios"][idx] = ratio
                    hf["timestamps"][idx] = np.datetime_as_string(timestamp,timezone='UTC').encode('utf-8')

                    print(file, f"Loaded {idx+1}/{len(hf['filepaths'])} files.")
                else:
                    print(file, f"File {idx+1}/{len(hf['filepaths'])} already loaded.")
        except:
            logging.exception(f"An exception occured during loading of files while processing {file}.")
            print('Loading stopped.')
        else:
            print("File load successfully done.")
        finally:
            hf.close()
    else:
        # create h5 file
        hf = h5py.File(os.path.join(outdir, 'ratios.h5'), 'a')

        # get paths to all hdf files in rootdir and its subdirectories
        filepaths = []
        for dirname, dirs, files in os.walk(rootdir):
            for filename in files:
                filename_without_extension, extension = os.path.splitext(filename)
                if extension == '.hdf':
                    filepaths.append(os.path.join(dirname, filename))

        # append filepaths to ratios dataset
        hf.create_dataset('filepaths', data=filepaths, chunks=True)

        try:
            for file in filepaths:
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
                # timestamp of precipitation map
                timestamp = np.datetime64(datetime.strptime(file.split('_')[-1].split('.')[0], '%Y%m%d%H%M%S'))

                # which file are we at
                idx = filepaths.index(file)
                if idx == 0:
                    # this is the first element, create new empty dataset
                    hf.create_dataset('ratios', (len(filepaths),), dtype=type(ratio))
                    hf.create_dataset('timestamps', (len(filepaths),), dtype=h5py.string_dtype(encoding='utf-8'))

                # append to file
                hf["ratios"][idx] = ratio
                hf["timestamps"][idx] = np.datetime_as_string(timestamp,timezone='UTC').encode('utf-8')

                print(file, f"Loaded {idx+1}/{len(filepaths)} files")
        except:
            logging.exception(f"An exception occured during loading of files while processing {file}.")
            print('Loading stopped.')
        else:
            print("File load successfully done.")
        finally:
            hf.close()

    with open(os.path.join(outdir, 'config_get_ratios_from_hdf.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # time elapsed running the script
    toc = round(time.time() - tic, 2)

    print(f"Done. Elapsed {toc} seconds.")

if __name__ == '__main__':
    get_ratios_from_hdf('config.yaml')