from pathlib import Path
import pandas as pd
from utils.wradlib_io import read_rainbow_wrl_custom
from utils.io import directory_is_empty, load_timestamps, rmtree, convert_float_to_uint, load_config, write_image
import numpy as np
import pyart
import dask
from dask.diagnostics import ProgressBar
import warnings
import h5py

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

def dbztommh(data):
    data = 10 ** (data * 0.1)
    data = (data / 200) ** (1 / 1.6)
    return data

def process_radars(datetime):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        dbz_radars = []

        for file in files_df[files_df.time == datetime].full_path.astype('str'):
            dbz_radars.append(read_rainbow_wrl_custom(file))
            
        dbz_radars = tuple(dbz_radars)

        grid = pyart.map.grid_from_radars(
            (dbz_radars),
            grid_shape=(8, 517, 755),
            grid_limits=((1000.0,8000), (-517073/2, 517073/2), (-(789412+720621)/4, (789412+720621)/4)),
            grid_origin=((46.05+50.7)/2,(13.6+23.8)/2),
            fields=[RAINBOW_FIELD_NAMES['dBZ']],
            min_radius=1750.0,
            weighting_function="cressman",
        )

        product = 'dBZ'

        dbz_data = grid.fields[RAINBOW_FIELD_NAMES[product]]["data"].data
        cmax = np.nanmax(dbz_data, axis=0)
        cmax_r = dbztommh(cmax)

        metrics_dict = {
        'sum': np.nansum(cmax),
        'square_sum': np.nansum(np.power(cmax, 2)),
        'light': np.nansum(cmax > 20),
        'moderate': np.nansum(cmax > 30),
        'heavy': np.nansum(cmax > 45),
        'sum_rr': np.nansum(cmax_r),
        'square_sum_rr': np.nansum(np.power(cmax_r, 2)),
        }

        metrics = pd.DataFrame(metrics_dict, index=[datetime])
        metrics /= (cmax.shape[0] * cmax.shape[1])

        metrics.to_csv('D:/processed_reflectivity/metrics.csv', mode='a', header=False)

        product = 'dBZ'

        # save parameters for compression
        datamin = RAINBOW_DATAMINS[product]
        datamax = RAINBOW_DATAMAXES[product]
        datadepth = 8

        target_type = np.uint8

        dbz_data[grid.fields[RAINBOW_FIELD_NAMES[product]]["data"].mask] = np.nan

        # convert radar to uint
        data = convert_float_to_uint(
            dbz_data,
            datamin=datamin,
            datamax=datamax,
            depth=datadepth,
            target_type=target_type,
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

        with h5py.File(f'D:/processed_reflectivity/{datetime}.h5', 'a') as f:
            # create groups and write image to h5 file
            group = f.require_group(product)
            ds_group = group.require_group('data')
            
            write_image(
                group=ds_group,
                ds_name='data',
                data=data,
                what_attrs=what_attrs
            )
    except:
        print("An exception occurred processing ", datetime)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    folder = Path("D:/extracted_volumes")
    result = list(folder.rglob("*dBZ.vol"))

    files_df = pd.DataFrame(result, columns=['full_path'])
    files_df['name'] = [x.name for x in result]
    files_df['time'] = [x.name[0:12] for x in result]
    files_df['time_parsed'] = pd.to_datetime(files_df['time'])

    datetimes = files_df['time'].unique()
    datetimes.sort()

    metrics = pd.read_csv("D:/processed_reflectivity/metrics.csv")
    datetimes = list(set(datetimes).difference(metrics.datetime.astype(str)))

    res = []
    nworkers = 7

    for datetime in datetimes:
        res.append(dask.delayed(process_radars)(datetime))

    with ProgressBar():
        scheduler = "processes" if nworkers > 1 else "single-threaded"
        res = dask.compute(*res, num_workers=nworkers, scheduler=scheduler)

