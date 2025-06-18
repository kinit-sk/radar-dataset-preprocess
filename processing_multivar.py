import io
import sys
import gc
import numpy as np
import h5py
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path
import warnings
import pandas as pd
from utils.wradlib_io import read_rainbow_wrl_custom
from utils.io import convert_float_to_uint, write_image
# simple text trap to get rid of pyart welcome messages
text_trap = io.StringIO()
sys.stdout = text_trap
import pyart
sys.stdout = sys.__stdout__
from tqdm import tqdm


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
        

def convert_rnbw_to_h5(files, outfile, datetime):
    """
    Function for converting rainbow type files of multiple products to one h5 datasets.

    """
    try:
        for product in files.keys():
            radars = []
            # read selected radar images as Radar class to 1 tuple                    
            for file in files[product]:
                radars.append(read_rainbow_wrl_custom(file))
            radars = tuple(radars)
            
            # perform Cartesian mapping of Radar class, limit to the reflectivity field.
            grid = pyart.map.grid_from_radars(
                (radars),
                grid_shape=(16, 517, 755),
                grid_limits=((500.0,8000), (-517073/2, 517073/2), (-(789412+720621)/4, (789412+720621)/4)), # CAPPI 2km, limits based on size by SHMU - TODO - add this params to config
                grid_origin=((46.05+50.7)/2,(13.6+23.8)/2),
                fields=[RAINBOW_FIELD_NAMES[product]],
            )

            # to np.array from Grid object
            data = grid.fields[RAINBOW_FIELD_NAMES[product]]["data"].data
            data[grid.fields[RAINBOW_FIELD_NAMES[product]]["data"].mask] = np.nan
            del grid
            gc.collect()

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
                "date": np.string_(datetime[:8]),
                "time": np.string_(datetime[8:]),
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
            
            
            with h5py.File(outfile, 'a') as f:
                # create groups and write image to h5 file
                group = f.require_group(product)
                ds_group = group.require_group('data')
                
                write_image(
                            group=ds_group,
                            ds_name='data',
                            data=data,
                            what_attrs=what_attrs,
                            )
    except:
        print("An exception occurred processing ", datetime)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    final_df = pd.DataFrame()

    products = ["dBZ", "dBZv", "PhiDP", "KDP", "RhoHV", "W", "ZDR"]

    print("Looking for files to process")
    folder = Path("/home/projects/p709-24-2/datasets/SHMU_Rainbow")
    for product in products:
        result = list(folder.rglob(f"*0{product}.vol"))

        files_df = pd.DataFrame(result, columns=['full_path'])
        files_df['name'] = [x.name for x in result]
        files_df['time'] = [x.name[0:12] for x in result]
        files_df['time_parsed'] = pd.to_datetime(files_df['time'])
        files_df['product'] = product

        final_df = pd.concat([final_df, files_df])

    out_folder = Path("/home/projects/p709-24-2/datasets/SHMU_4_New")
    processed = list(out_folder.rglob(f"*.h5"))
    processed = [item.name.split('.')[0] for item in processed]

    print("Dropping already processed")
    final_df = final_df[~final_df['time'].isin(processed)]

    nworkers = 8
    res = []

    print("Preparing dusk jobs")

    for time in tqdm(final_df.time.unique()):
        files_disc = {}
        temp_df = final_df[final_df.time == time]
        for product in products:
            files_disc[product] = temp_df[temp_df["product"] == product].full_path.apply(str).to_list()
        res.append(dask.delayed(convert_rnbw_to_h5)(files_disc, '/home/projects/p709-24-2/datasets/SHMU_4_New/' + time + '.h5', time))

    with ProgressBar():
        scheduler = "processes" if nworkers > 1 else "single-threaded"
        res = dask.compute(*res, num_workers=nworkers, scheduler=scheduler)
