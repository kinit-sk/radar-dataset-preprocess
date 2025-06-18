import io
import sys
import gc
import logging
import argparse
import datetime as dt
import numpy as np
import h5py
import dask
import os
from dask.diagnostics import ProgressBar
from pathlib import Path
from threading import RLock
import wradlib.classify as classify
from wradlib.trafo import decibel
from wradlib.zr import r_to_z
from collections import defaultdict
from utils.wradlib_io import read_rainbow_wrl_custom
from utils.io import directory_is_empty, load_timestamps, rmtree, convert_float_to_uint, load_config, write_image
# simple text trap to get rid of pyart welcome messages
text_trap = io.StringIO()
sys.stdout = text_trap
import pyart
sys.stdout = sys.__stdout__


# rainbow field names for referencing
RAINBOW_FIELD_NAMES = {
    "W": "spectrum_width",
    "Wv": "spectrum_width_vv",  # non standard name
    "Wu": "unfiltered_spectrum_width",  # non standard name
    "Wvu": "unfiltered_spectrum_width_vv",  # non standard name
    "V": "velocity",
    "Vv": "velocity_vv",  # non standard name
    "Vu": "unfiltered_velocity",  # non standard name
    "Vvu": "unfiltered_velocity_vv",  # non standard name
    "dBZ": "reflectivity",
    "dBZv": "reflectivity_vv",  # non standard name
    "dBuZ": "unfiltered_reflectivity",  # non standard name
    "dBuZv": "unfiltered_reflectivity_vv",  # non standard name
    "ZDR": "differential_reflectivity",
    "ZDRu": "unfiltered_differential_reflectivity",  # non standard name
    "RhoHV": "cross_correlation_ratio",
    "RhoHVu": "unfiltered_cross_correlation_ratio",  # non standard name
    "PhiDP": "differential_phase",
    "uPhiDP": "uncorrected_differential_phase",  # non standard name
    "uPhiDPu": "uncorrected_unfiltered_differential_phase",  # non standard name
    "KDP": "specific_differential_phase",
    "uKDP": "uncorrected_specific_differential_phase",  # non standard name
    "uKDPu": "uncorrected_unfiltered_specific_differential_phase",  # non standard name
    "SQI": "signal_quality_index",  # non standard name
    "SQIv": "signal_quality_index_vv",  # non standard name
    "SQIu": "unfiltered_signal_quality_index",  # non standard name
    "SQIvu": "unfiltered_signal_quality_index_vv",  # non standard name
    "TEMP": "temperature",  # non standard name
    "ISO0": "iso0",  # non standard name
}

# typical maximum values for products as listed in Rainbow5 format manual        
RAINBOW_DATAMAXES = {
    "dBZ": 95.5,
    "dBZv": 95.5,
    "V": 30.0,
    "Vv": 30.0,
    "W": 15.0,
    "ZDR": 12.0,
    "RhoHV": 1.0,
    "PhiDP": 360.0,
    "KDP": 36.0,
}

# typical minimum values for products as listed in Rainbow5 format manual        
RAINBOW_DATAMINS = {
    "dBZ": -31.5,
    "dBZv": -31.5,
    "V": -30.0,
    "Vv": -30.0,
    "W": 0.0,
    "ZDR": -8.0,
    "RhoHV": 0.0,
    "PhiDP": 0.0,
    "KDP": -18.0,
}

# typical units for products as listed in Rainbow5 format manual 
RAINBOW_UNITS = {
    "dBZ": 'dB',
    "dBZv": 'dB',
    "V": 'm/s',
    "Vv": 'm/s',
    "W": 'm/s',
    "ZDR": 'dB',
    "RhoHV": '',
    "PhiDP": 'degree',
    "KDP": 'degree/km',
}
        

def convert_rnbw_to_h5(conf, all_files, outfiles, datetimes_str, rain_thres):
    """
    Function for converting rainbow type files of multiple products to one h5 datasets.

    """
    
    # check the last observation if it contains enough rain
    dbz_radars = []
    
    for file in all_files[0]['dBZ']:
        dbz_radars.append(read_rainbow_wrl_custom(file))
        
    dbz_radars = tuple(dbz_radars)
    
    # perform Cartesian mapping of Radar class, limit to the reflectivity field.
    grid = pyart.map.grid_from_radars(
        (dbz_radars),
        grid_shape=(1, 517, 755),
        grid_limits=((2000, 2000), (-517073/2, 517073/2), (-(789412+720621)/4, (789412+720621)/4)), # CAPPI 2km, limits based on size by SHMU - TODO - add this params to config
        grid_origin=((46.05+50.7)/2,(13.6+23.8)/2),
        fields=[RAINBOW_FIELD_NAMES['dBZ']],
    )
    
    # to np.array from Grid object
    dbz_data = grid.fields[RAINBOW_FIELD_NAMES['dBZ']]["data"][0]
    del grid
    gc.collect()
            
    # remove clutter from image
    clmap = classify.filter_gabella(dbz_data, rm_nans=False, cartesian=True)
    dbz_data[np.nonzero(clmap)] = np.nan
    
    # ratio by which we compare selected radar obs to threshold
    rainy_pxls_ratio = np.sum(dbz_data >= rain_thres.rate)/np.prod(dbz_data.shape)
    
    if rainy_pxls_ratio >= rain_thres.fraction:
        for i, files in enumerate(all_files):
            for product in files.keys():
                if i == 0 and product == 'dBZ':
                    data = dbz_data
                else:
                    radars = []
                    # read selected radar images as Radar class to 1 tuple                    
                    for file in files[product]:
                        radars.append(read_rainbow_wrl_custom(file))
                    radars = tuple(radars)
                    
                    # perform Cartesian mapping of Radar class, limit to the reflectivity field.
                    grid = pyart.map.grid_from_radars(
                        (radars),
                        grid_shape=(8, 517, 755),
                        grid_limits=((1000.0,8000), (-517073/2, 517073/2), (-(789412+720621)/4, (789412+720621)/4)), # CAPPI 2km, limits based on size by SHMU - TODO - add this params to config
                        grid_origin=((46.05+50.7)/2,(13.6+23.8)/2),
                        fields=[RAINBOW_FIELD_NAMES[product]],
                    )

                    # to np.array from Grid object
                    data = grid.fields[RAINBOW_FIELD_NAMES[product]]["data"][0]
                    del grid
                    gc.collect()

                    # remove clutter from image
                    if product in ['dBZ', 'dBZv']:
                        clmap = classify.filter_gabella(data, rm_nans=False, cartesian=True)
                        data[np.nonzero(clmap)] = np.nan

                # save parameters for compression
                datamin = RAINBOW_DATAMINS[product]
                datamax = RAINBOW_DATAMAXES[product]
                datadepth = 8 #TODO customizable parameter
                
                # convert radar to uint
                data = convert_float_to_uint(
                    data,
                    datamin=datamin,
                    datamax=datamax,
                    depth=datadepth,
                    target_type=np.uint8,
                )
                
                # create metadata attributes about file
                what_attrs = {
                    "date": np.string_(datetimes_str[i][:8]),
                    "time": np.string_(datetimes_str[i][8:]),
                    "object": np.string_("RADARIMG"),
                    "source": np.string_("SHMU"),
                    "datamin": datamin,
                    "datamax": datamax,
                    "datadepth": datadepth,
                    "productcode": np.string_(product),
                    "product": np.string_(RAINBOW_FIELD_NAMES[product]),
                    "units": np.string_(RAINBOW_UNITS[product]),
                    "resolution": np.string_("1KM"),
                    }
                
                
                with h5py.File(outfiles[i], 'a') as f:
                    # create groups and write image to h5 file
                    group = f.require_group(product)
                    ds_group = group.require_group('data')
                    
                    write_image(
                                group=ds_group,
                                ds_name='data',
                                data=data,
                                what_attrs=what_attrs,
                                )
                    
                del data
                gc.collect()
        
        complete_path = Path(conf.log_path) / 'completed_timestamps.txt'
        complete_path.touch()
        
        with RLock():
            with open(complete_path, 'a') as cf:    
                for datetime_str in datetimes_str:
                    cf.write(datetime_str + ' ' + '\n')
            
    else:
        nonrainy_path = Path(conf.log_path) / 'nonrainy_timestamps.txt'
        nonrainy_path.touch()
            
        with RLock():
            with open(nonrainy_path, 'a') as nf:    
                for datetime_str in datetimes_str:
                    nf.write(datetime_str + ' ' + '\n')
        
    gc.collect()
                    
        
def main(conf, restarted):
    # setup paths
    input_path = Path(conf.input_path)
    output_path = Path(conf.output_path)
    inc_ts_path = Path(conf.log_path) / 'incomplete_timestamps.txt'
    com_ts_path = Path(conf.log_path) / 'completed_timestamps.txt'
    non_ts_path = Path(conf.log_path) / 'nonrainy_timestamps.txt'
    com_days_path = Path(conf.log_path) / 'completed_days.txt'
    
    # create outputdir if doesnt exist
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    # setup logging
    log_path = Path(conf.log_path) / 'logs.log'
    logging.basicConfig(filename=log_path, 
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    logging.info('Starting script...')
    
    # date format in file - TODO - maybe part of config
    date_format = '%Y%m%d%H%M'
    
    # get rain threshold array to np.array and convert to dbz
    rain_threshold = conf.rain_threshold
    rain_threshold.rate = r_to_z(rain_threshold.rate, a=200., b=1.6)
    rain_threshold.rate = decibel(rain_threshold.rate)
    
    # get observation and "jump" intervals
    jump = conf.check_rain_interval
    interval = conf.obs_interval
        
    # if this is restarted run, then check which incomplete timestamps (discarded observations) were already checked and delete files which were interrupted during transformation (because they are incomplete)
    incomplete_timestamps = []
    complete_timestamps = []
    nonrainy_timestamps = []
    complete_days = []
    date_str = None
    
    if restarted:
        # get incomplete timestamps
        if inc_ts_path.exists():
            logging.info(f'Restarted run will use incomplete timestamps from file: {inc_ts_path}.')
            incomplete_timestamps = load_timestamps(inc_ts_path)
            incomplete_timestamps = np.unique(incomplete_timestamps)
        else:
            logging.info(f'Did not find incomplete timestamps file at {inc_ts_path}. Did it ever exist or is it just moved?')
            
        # get complete timestamps
        if com_ts_path.exists():
            logging.info(f'Restarted run will use completed timestamps from file: {com_ts_path}.') 
            complete_timestamps = load_timestamps(com_ts_path)
        else:
            logging.info(f'Did not find complete timestamps file at {com_ts_path}. Did it ever exist or is it just moved?')
            
        # get nonrainy timestamps
        if non_ts_path.exists():
            logging.info(f'Restarted run will use nonrainy timestamps from file: {non_ts_path}.')
            nonrainy_timestamps = load_timestamps(non_ts_path)
        else:
            logging.info(f'Did not find nonrainy timestamps file at {non_ts_path}. Did it ever exist or is it just moved?')
            
        if com_days_path.exists():
            logging.info(f'Restarted run will use complete days from file: {com_days_path}.')
            complete_days = load_timestamps(com_days_path)
        else:
            logging.info(f'Did not find complete days file at {com_days_path}. Did it ever exist or is it just moved?')
        
        # delete observations from output that didn't close properly
        for path_object in output_path.rglob('*'):
            if path_object.is_file() and path_object.suffix == '.h5':
                    if path_object.stem not in complete_timestamps and path_object.stem[8:] not in complete_days:
                        logging.info(f'{path_object} not found in completed_timestamps.txt or completed_days.txt. Deleting...')
                            
                        path_object.unlink()
                        logging.info(f'Deleted.')

        # check if temp directory is empty, if not, get its date
        if not directory_is_empty(temp_path):
            for item in temp_path.iterdir():
                date_str = item.stem[:8]
    
    # create iterator from tar files
    iterator = [ Path(f.path) for f in os.scandir(input_path) if f.is_dir() ]
              
    if date_str is not None:
        iterator.insert(0, date_str)
    
    for j, tarfile_path in enumerate(iterator):
        # if selected tarfile is not completed, unpack him with all the other radars for the given day
        if j==0 and date_str is not None:
            logging.info(f'{date_str} already unpacked.')
            date_str = tarfile_path
        else:
            if tarfile_path.stem[:8] not in complete_days:
                date_str = tarfile_path.stem[:8]

        if j==0 or tarfile_path.stem[:8] not in complete_days:
            res_files = []
            res_outfile = []
            res_datetime = []
            
            logging.info('Starting check of observations suitability.')
            
            # iterate over subdirectories for each day
            start_datetime = dt.datetime.strptime(date_str, '%Y%m%d')
            # iterate over one day
            for lag in range(0, 24*60, interval):
                datetime = start_datetime + dt.timedelta(minutes=lag)
                datetime_str = datetime.strftime(date_format)
                print(datetime_str)
                # outfile path
                output_subpath = output_path / date_str
                outfile = output_subpath / (datetime_str + '.h5')
                
                # append to dask only timestamps that arent already transformed or discarded from data
                if datetime_str not in incomplete_timestamps and datetime_str not in complete_timestamps and datetime_str not in nonrainy_timestamps:
                    selected_files_dict = defaultdict(list)
                    
                    # find paths to selected timestamp and given product
                    for product in conf.products:
                        selected_files = []

                        # get paths to 4 radar station images of 1 product and append to one array
                        for radar_code in conf.radar_codes:
                            subpaths = Path(Path(str(tarfile_path)[0:-2] + radar_code) / datetime.strftime("%Y-%m-%d")).glob('*')
                            for subpath_object in subpaths:
                                filepath = str(subpath_object)
                                if datetime_str in filepath and filepath.endswith(product+'.vol'):
                                    selected_files.append(filepath)
                        
                        # check if all radars have observation for the day
                        if len(selected_files) == len(conf.radar_codes):
                            selected_files_dict[product] = selected_files
                        else:
                            inc_ts_path.touch()
                            with open(inc_ts_path, 'a') as tf:
                                tf.write(datetime_str + ' ' + product + '\n')
                    
                    # dict has to be the same size as number of products for it to make sense to append the task
                    if len(selected_files_dict) == len(conf.products):
                        if not output_subpath.exists():
                            output_subpath.mkdir(parents=True)
                        res_files.append(selected_files_dict)
                        res_outfile.append(outfile)
                        res_datetime.append(datetime)
                    else:
                        inc_ts_path.touch()
                        with open(inc_ts_path, 'a') as tf:
                            tf.write(datetime_str + ' ' + product + '\n')
                                    
            logging.info('Check done.')
            
            incomplete_timestamps = []

            if inc_ts_path.exists():
                incomplete_timestamps = load_timestamps(inc_ts_path)
                incomplete_timestamps = np.unique(incomplete_timestamps)
            
            final_res_files = None
            final_res_outfile = None
            final_res_datetime = None
            res = []
            
            logging.info(f'Appending suitable tasks to dask...')
            
            # iterate only over larger intervals (in rainbow_to_h5 function it will be checked whether or not these will be appended to final dataset based on rain threshold)      
            # initialize start and end datetimes
            start_datetime = dt.datetime.strptime(date_str, '%Y%m%d') + dt.timedelta(minutes=jump) - dt.timedelta(minutes=interval)
            end_datetime = dt.datetime.strptime(date_str, '%Y%m%d') + dt.timedelta(hours=24) - dt.timedelta(minutes=interval)
            
            for lag in range(0, end_datetime.hour * 60 + end_datetime.minute + 1, jump):
                datetime = start_datetime + dt.timedelta(minutes=lag)
                temp_datetimes = []
                incomplete_bool = False
                complete_bool = False
                nonrainy_bool = False
                
                # get only intervals that are full of observations
                for i in range(0, jump, interval):
                    temp_datetime = datetime - dt.timedelta(minutes=i)
                    temp_datetime_str = temp_datetime.strftime(date_format)
                    
                    if temp_datetime_str in complete_timestamps:
                        complete_bool = True
                    
                    elif temp_datetime_str in nonrainy_timestamps:
                        nonrainy_bool = True
                        
                    elif temp_datetime_str in incomplete_timestamps:
                        incomplete_bool = True
                        temp_datetimes.append(temp_datetime)
                    else:
                        temp_datetimes.append(temp_datetime)
                
                # get indices of datetimes so we can append other stuff
                if incomplete_bool:
                    inc_ts_path.touch()     
                    with open(inc_ts_path, 'a') as tf:
                        for temp_datetime in temp_datetimes:
                            temp_datetime_str = temp_datetime.strftime(date_format)
                            if temp_datetime_str not in incomplete_timestamps:
                                tf.write(temp_datetime_str + ' ' + 'not_full_interval' + '\n')

                if not incomplete_bool and not complete_bool and not nonrainy_bool:
                    idx = []
                    for temp_datetime in temp_datetimes:
                        for i in range(len(res_datetime)):
                            if res_datetime[i] == temp_datetime:
                                idx.append(i)
                    
                    # append to dask
                    final_res_files = [res_files[i] for i in idx]
                    final_res_outfile = [res_outfile[i] for i in idx]
                    final_res_datetimes_str = [temp_datetime.strftime(date_format) for temp_datetime in temp_datetimes]
                    res.append(dask.delayed(convert_rnbw_to_h5)(conf, final_res_files, final_res_outfile, final_res_datetimes_str, rain_threshold))
            
            logging.info('Append done.')
            
            logging.info(f"Creating {len(res)} dask tasks! {int(jump/interval)} observations per task.")
            
            with ProgressBar(minimum = 10, dt = 60):
                scheduler = "processes" if conf.nworkers > 1 else "single-threaded"
                res = dask.compute(*res, num_workers=conf.nworkers, scheduler=scheduler)
            
            # delete temporary radar files from temp_path 
            rmtree(temp_path)
            
            # write date_str as complete day
            com_days_path.touch()
            with open(com_days_path, 'a') as f:
                    f.write(date_str + ' ' + '\n')
            
            complete_days.append(date_str)
            
            logging.info(f'Done transforming data from {date_str}.')
               
    logging.info("Everything done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--config', '--c', required=True, help='Path to the config file.')
    parser.add_argument('--restarted', '--r', action='store_true', help='Flag to indicate if script is being rerun/restarted due to interruption.')

    args = parser.parse_args()
    conf = load_config(Path(args.config))
    
    main(conf, args.restarted)
    