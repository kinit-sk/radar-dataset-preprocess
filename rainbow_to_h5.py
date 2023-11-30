import os
import io
import sys
import gc
import logging
import argparse
import datetime as dt
import numpy as np
import numpy.ma as ma
import h5py
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path
from addict import Dict
import yaml
import wradlib.classify as classify
from wradlib.trafo import decibel
from wradlib.zr import r_to_z
from collections import defaultdict

text_trap = io.StringIO()
sys.stdout = text_trap
import pyart
from pyart.core.radar import Radar
sys.stdout = sys.__stdout__


# TODO - move functions to utils folder
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
        
"""
Routines for reading RAINBOW files (Used by SELEX) using the wradlib library

"""

# specific modules for this function
import os

try:
    import wradlib  # noqa

    _WRADLIB_AVAILABLE = True
    # `read_rainbow` as of wradlib version 1.0.0
    try:
        from wradlib.io import read_Rainbow as read_rainbow
    except ImportError:
        from wradlib.io import read_rainbow
except:
    _WRADLIB_AVAILABLE = False


import datetime
import warnings
import numpy as np

text_trap = io.StringIO()
sys.stdout = text_trap
from pyart.config import FileMetadata, get_fillvalue
from pyart.core.radar import Radar
from pyart.exceptions import MissingOptionalDependency
from pyart.io.common import _test_arguments, make_time_unit_str
sys.stdout = sys.__stdout__

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


def read_rainbow_wrl_custom(
    filename,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    **kwargs
):
    """
    Read a RAINBOW file.
    This routine has been tested to read rainbow5 files version 5.22.3,
    5.34.16 and 5.35.1.
    Since the rainbow file format is evolving constantly there is no guaranty
    that it can work with other versions.
    If necessary, the user should adapt to code according to its own
    file version and raise an issue upstream.

    Data types read by this routine:
    Reflectivity: dBZ, dBuZ, dBZv, dBuZv
    Velocity: V, Vu, Vv, Vvu
    Spectrum width: W, Wu, Wv, Wvu
    Differential reflectivity: ZDR, ZDRu
    Co-polar correlation coefficient: RhoHV, RhoHVu
    Co-polar differential phase: PhiDP, uPhiDP, uPhiDPu
    Specific differential phase: KDP, uKDP, uKDPu
    Signal quality parameters: SQI, SQIu, SQIv, SQIvu
    Temperature: TEMP
    Position of the range bin respect to the ISO0: ISO0

    Parameters
    ----------
    filename : str
        Name of the RAINBOW file to read.
    field_names : dict, optional
        Dictionary mapping RAINBOW field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    file_field_names : bool, optional
        True to use the MDV data type names for the field names. If this
        case the field_names parameter is ignored. The field dictionary will
        likely only have a 'data' key, unless the fields are defined in
        `additional_metadata`.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.

    Returns
    -------
    radar : Radar
        Radar object containing data from RAINBOW file.

    """

    # check that wradlib is available
    if not _WRADLIB_AVAILABLE:
        raise MissingOptionalDependency(
            "wradlib is required to use read_rainbow_wrl but is not installed"
        )

    # test for non empty kwargs
    _test_arguments(kwargs)

    # check if it is the right file. Open it and read it
    bfile = os.path.basename(filename)
    supported_file = (
        bfile.endswith(".vol") or bfile.endswith(".azi") or bfile.endswith(".ele")
    )
    if not supported_file:
        raise ValueError(
            "Only data files with extension .vol, .azi or .ele are supported"
        )

    # create metadata retrieval object
    if field_names is None:
        field_names = RAINBOW_FIELD_NAMES
    filemetadata = FileMetadata(
        "RAINBOW",
        field_names,
        additional_metadata,
        file_field_names,
        exclude_fields,
        include_fields,
    )

    rbf = read_rainbow(filename, loaddata=True)

    # check the number of slices
    nslices = int(rbf["volume"]["scan"]["pargroup"]["numele"])
    if nslices > 1:
        single_slice = False
        common_slice_info = rbf["volume"]["scan"]["slice"][0]
    else:
        single_slice = True
        common_slice_info = rbf["volume"]["scan"]["slice"]

    # check the data type
    # all slices should have the same data type
    datatype = common_slice_info["slicedata"]["rawdata"]["@type"]
    field_name = filemetadata.get_field_name(datatype)
    if field_name is None:
        raise ValueError("Field Name Unknown")

    # get definitions from filemetadata class
    latitude = filemetadata("latitude")
    longitude = filemetadata("longitude")
    altitude = filemetadata("altitude")
    metadata = filemetadata("metadata")
    sweep_start_ray_index = filemetadata("sweep_start_ray_index")
    sweep_end_ray_index = filemetadata("sweep_end_ray_index")
    sweep_number = filemetadata("sweep_number")
    sweep_mode = filemetadata("sweep_mode")
    fixed_angle = filemetadata("fixed_angle")
    elevation = filemetadata("elevation")
    _range = filemetadata("range")
    azimuth = filemetadata("azimuth")
    _time = filemetadata("time")
    field_dic = filemetadata(field_name)

    # other metadata
    frequency = filemetadata("frequency")

    # get general file information

    # position and radar frequency
    if "sensorinfo" in rbf["volume"].keys():
        latitude["data"] = np.array(
            [rbf["volume"]["sensorinfo"]["lat"]], dtype="float64"
        )
        longitude["data"] = np.array(
            [rbf["volume"]["sensorinfo"]["lon"]], dtype="float64"
        )
        altitude["data"] = np.array(
            [rbf["volume"]["sensorinfo"]["alt"]], dtype="float64"
        )
        frequency["data"] = np.array(
            [3e8 / float(rbf["volume"]["sensorinfo"]["wavelen"])], dtype="float64"
        )
    elif "radarinfo" in rbf["volume"].keys():
        latitude["data"] = np.array(
            [rbf["volume"]["radarinfo"]["@lat"]], dtype="float64"
        )
        longitude["data"] = np.array(
            [rbf["volume"]["radarinfo"]["@lon"]], dtype="float64"
        )
        altitude["data"] = np.array(
            [rbf["volume"]["radarinfo"]["@alt"]], dtype="float64"
        )
        frequency["data"] = np.array(
            [3e8 / float(rbf["volume"]["radarinfo"]["wavelen"])], dtype="float64"
        )

    # antenna speed
    if "antspeed" in common_slice_info:
        ant_speed = float(common_slice_info["antspeed"])
    else:
        ant_speed = 10.0
        print(
            "WARNING: Unable to read antenna speed. Default value of "
            + str(ant_speed)
            + " deg/s will be used"
        )

    # angle step
    angle_step = float(common_slice_info["anglestep"])

    # sweep_number (is the sweep index)
    sweep_number["data"] = np.arange(nslices, dtype="int32")

    # get number of rays and number of range bins per sweep
    rays_per_sweep = np.empty(nslices, dtype="int32")

    if single_slice:
        rays_per_sweep[0] = int(common_slice_info["slicedata"]["rawdata"]["@rays"])
        nbins = int(common_slice_info["slicedata"]["rawdata"]["@bins"])
        ssri = np.array([0], dtype="int32")
        seri = np.array([rays_per_sweep[0] - 1], dtype="int32")
    else:
        # number of range bins per ray in sweep
        nbins_sweep = np.empty(nslices, dtype="int32")
        for i in range(nslices):
            slice_info = rbf["volume"]["scan"]["slice"][i]
            # number of rays per sweep
            rays_per_sweep[i] = int(slice_info["slicedata"]["rawdata"]["@rays"])

            # number of range bins per ray in sweep
            nbins_sweep[i] = int(slice_info["slicedata"]["rawdata"]["@bins"])

        # all sweeps have to have the same number of range bins
        mask_missing_bins = np.zeros(nslices, dtype=bool)
        if any(nbins_sweep != nbins_sweep[0]):
            #warnings.warn('Number of range bins changes between sweeps.')
            #raise ValueError("number of range bins changes between sweeps")
            for i in range(1, nslices):
                if nbins_sweep[i] != nbins_sweep[0]:
                    mask_missing_bins[i] = True
            
        nbins = nbins_sweep[0]
        ssri = np.cumsum(np.append([0], rays_per_sweep[:-1])).astype("int32")
        seri = np.cumsum(rays_per_sweep).astype("int32") - 1

    # total number of rays and sweep start ray index and end
    total_rays = sum(rays_per_sweep)
    sweep_start_ray_index["data"] = ssri
    sweep_end_ray_index["data"] = seri

    # range
    r_res = float(common_slice_info["rangestep"]) * 1000.0
    if "start_range" in common_slice_info.keys():
        start_range = float(common_slice_info["start_range"]) * 1000.0
    else:
        start_range = 0.0
    _range["data"] = np.linspace(
        start_range + r_res / 2.0, float(nbins - 1.0) * r_res + r_res / 2.0, nbins
    ).astype("float32")

    # containers for data
    t_fixed_angle = np.empty(nslices, dtype="float64")
    moving_angle = np.empty(total_rays, dtype="float64")
    static_angle = np.empty(total_rays, dtype="float64")
    time_data = np.empty(total_rays, dtype="float64")
    fdata = np.ma.zeros(
        (total_rays, nbins), dtype="float32", fill_value=get_fillvalue()
    )

    # read data from file
    if bfile.endswith(".vol") or bfile.endswith(".azi"):
        scan_type = "ppi"
        sweep_mode["data"] = np.array(nslices * ["azimuth_surveillance"])
    else:
        scan_type = "rhi"
        sweep_mode["data"] = np.array(["elevation_surveillance"])

    # read data from file:
    for i in range(nslices):
        if single_slice:
            slice_info = common_slice_info
        else:
            slice_info = rbf["volume"]["scan"]["slice"][i]

        # fixed angle
        t_fixed_angle[i] = float(slice_info["posangle"])

        # fixed angle (repeated for each ray)
        static_angle[ssri[i] : seri[i] + 1] = t_fixed_angle[i]

        # moving angle
        moving_angle[ssri[i] : seri[i] + 1], angle_start, angle_stop = _get_angle(
            slice_info["slicedata"]["rayinfo"],
            angle_step=angle_step,
            scan_type=scan_type,
        )

        # time
        time_data[ssri[i] : seri[i] + 1], sweep_start_epoch = _get_time(
            slice_info["slicedata"]["@date"],
            slice_info["slicedata"]["@time"],
            angle_start[0],
            angle_stop[-1],
            angle_step,
            rays_per_sweep[i],
            ant_speed,
            scan_type=scan_type,
        )

        if i == 0:
            volume_start_epoch = sweep_start_epoch + 0.0
            start_time = datetime.datetime.fromtimestamp(volume_start_epoch, tz=datetime.timezone.utc)

        # data
        fdata[ssri[i] : seri[i] + 1, :] = _get_data(
            slice_info["slicedata"]["rawdata"], rays_per_sweep[i], nbins, mask_missing_bins[i]
        )

    if bfile.endswith(".vol") or bfile.endswith(".azi"):
        azimuth["data"] = moving_angle
        elevation["data"] = static_angle
    else:
        azimuth["data"] = static_angle
        elevation["data"] = moving_angle

    fixed_angle["data"] = t_fixed_angle

    _time["data"] = time_data - volume_start_epoch
    _time["units"] = make_time_unit_str(start_time)

    # fields
    fields = {}
    # create field dictionary
    field_dic["_FillValue"] = get_fillvalue()
    field_dic["data"] = fdata
    fields[field_name] = field_dic

    # metadata
    # metadata['instrument_name'] = radar_id

    # instrument_parameters
    instrument_parameters = dict()
    instrument_parameters.update({"frequency": frequency})

    return Radar(
        _time,
        _range,
        fields,
        metadata,
        scan_type,
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth,
        elevation,
        instrument_parameters=instrument_parameters,
    )


def _get_angle(ray_info, angle_step=None, scan_type="ppi"):
    """
    obtains the ray angle start, stop and center

    Parameters
    ----------
    ray_info : dictionary of dictionaries
        contains the ray info
    angle_step : float
        Optional. The angle step. Used in case there is no information of
        angle stop. Otherwise ignored.
    scan_type : str
        Default ppi. scan_type. Either ppi or rhi.

    Returns
    -------
    moving_angle : numpy array
        the central point of the angle [Deg]
    angle_start :
        the starting point of the angle [Deg]
    angle_stop :
        the end point of the angle [Deg]

    """
    bin_to_deg = 360.0 / 65536.0

    def _extract_angles(data):
        angle = np.array(data * bin_to_deg, dtype="float64")
        if scan_type == "rhi":
            ind = (angle > 225.0).nonzero()
            angle[ind] -= 360.0
        return angle

    try:
        angle_start = _extract_angles(ray_info["data"])
        if angle_step is None:
            raise ValueError("Unknown angle step")
        angle_stop = angle_start + angle_step
    except TypeError:
        angle_start = _extract_angles(ray_info[0]["data"])
        angle_stop = _extract_angles(ray_info[1]["data"])

    moving_angle = np.angle(
        (np.exp(1.0j * np.deg2rad(angle_start)) + np.exp(1.0j * np.deg2rad(angle_stop)))
        / 2.0,
        deg=True,
    )
    moving_angle[moving_angle < 0.0] += 360.0  # [0, 360]

    return moving_angle, angle_start, angle_stop


def _get_data(rawdata, nrays, nbins, mask_missing_bins):
    """
    Obtains the raw data

    Parameters
    ----------
    rawdata : dictionary of dictionaries
        contains the raw data information
    nrays : int
        Number of rays in sweep
    nbins : int
        Number of bins in ray
    mask_missing_bins : bool
        True/False boolean representing if slice has incompatible bin size

    Returns
    -------
    data : numpy array
        the data

    """
    databin = rawdata["data"]
    datamin = float(rawdata["@min"])
    datamax = float(rawdata["@max"])
    datadepth = float(rawdata["@depth"])
    datatype = rawdata["@type"]

    databin = np.pad(databin, ((0, 0), (0, (nbins-databin.shape[1]))), mode='constant', constant_values=0)

    data = np.array(
        datamin + databin * (datamax - datamin) / 2**datadepth, dtype="float32"
    )
    
    #data = np.pad(data, ((0, 0), (0, (nbins-data.shape[1]))), mode='constant', constant_values=datamin)

    # fill invalid data with fill value
    #mask = data == datamin
    mask = databin == 0
    data[mask.nonzero()] = get_fillvalue()

    # put phidp data in the range [-180, 180]
    if (datatype == "PhiDP") or (datatype == "uPhiDP") or (datatype == "uPhiDPu"):
        is_above_180 = data > 180.0
        data[is_above_180.nonzero()] -= 360.0

    data = np.reshape(data, [nrays, nbins])
    mask = np.reshape(mask, [nrays, nbins])

    masked_data = np.ma.array(data, mask=mask, fill_value=get_fillvalue())

    return masked_data


def _get_time(
    date_sweep,
    time_sweep,
    first_angle_start,
    last_angle_stop,
    angle_step,
    nrays,
    ant_speed,
    scan_type="ppi",
):
    """
    Computes the time at the center of each ray

    Parameters
    ----------
    date_sweep, time_sweep : str
        the date and time of the sweep
    first_angle_start : float
        The starting point of the first angle in the sweep
    last_angle_stop : float
        The end point of the last angle in the sweep
    nrays : int
        Number of rays in sweep
    ant_speed : float
        antenna speed [deg/s]
    scan_type : str
        Default ppi. scan_type. Either ppi or rhi.

    Returns
    -------
    time_data : numpy array
        the time of each ray
    sweep_start_epoch : float
        sweep start time in seconds since 1.1.1970

    """
    datetime_sweep = datetime.datetime.strptime(
        date_sweep + " " + time_sweep, "%Y-%m-%d %H:%M:%S"
    )
    sweep_start_epoch = (datetime_sweep - datetime.datetime(1970, 1, 1)).total_seconds()
    if scan_type == "ppi":
        if (last_angle_stop > first_angle_start) and (
            (last_angle_stop - first_angle_start) / nrays > angle_step
        ):
            sweep_duration = (last_angle_stop - first_angle_start) / ant_speed
        else:
            sweep_duration = (last_angle_stop + 360.0 - first_angle_start) / ant_speed
    else:
        if last_angle_stop > first_angle_start:
            sweep_duration = (last_angle_stop - first_angle_start) / ant_speed
        else:
            sweep_duration = (first_angle_start - last_angle_stop) / ant_speed

    time_angle = sweep_duration / nrays

    sweep_end_epoch = sweep_start_epoch + sweep_duration

    time_data = np.linspace(
        sweep_start_epoch + time_angle / 2.0,
        sweep_end_epoch - time_angle / 2.0,
        num=nrays,
    )

    return time_data, sweep_start_epoch

"""
The end of rainbow reading routines.

"""

def convert_rnbw_to_h5(conf, all_files, outfiles, datetimes_str, rain_thres):
    """
    Function for converting rainbow type files of multiple products to one h5 datasets.

    """
    
    # check preselected observation if it contains enough rain
    dbz_radar = None
    
    for file in all_files[0]['dBZ']:
        if conf.rainy_radar_code in file:
            dbz_radar = read_rainbow_wrl_custom(file)
    
    # perform Cartesian mapping of Radar class, limit to the reflectivity field. - TODO - remake so it takes all radars
    grid = pyart.map.grid_from_radars(
    (dbz_radar,),
    grid_shape=(1, 340, 340),
    grid_limits=((2000, 2000), (-170000.0, 170000.0), (-170000.0, 170000.0)),
    fields=[RAINBOW_FIELD_NAMES['dBZ']])
    
    # to np.array from Grid object
    data = grid.fields[RAINBOW_FIELD_NAMES['dBZ']]["data"][0]
            
    # remove clutter from image
    
    clmap = classify.filter_gabella(data, rm_nans=False, cartesian=True)
    data[np.nonzero(clmap)] = np.nan
    clmap = classify.filter_gabella(data, rm_nans=False, cartesian=True)
    data[np.nonzero(clmap)] = np.nan
    
    # ratio by which we compare selected radar obs to threshold
    rainy_pxls_ratio = np.sum(data >= rain_thres.rate)/np.prod(data.shape)
    
    if rainy_pxls_ratio >= rain_thres.fraction:
        for i, files in enumerate(all_files):
            for product in files.keys():
                radars = []
                # read selected radar images as Radar class to 1 tuple                    
                for file in files[product]:
                    if i == 0 and product == 'dBZ' and conf.rainy_radar_code in file:
                        radars.append(dbz_radar)
                    else:
                        radars.append(read_rainbow_wrl_custom(file))
                radars = tuple(radars)
                
                # perform Cartesian mapping of Radar class, limit to the reflectivity field.
                grid = pyart.map.grid_from_radars(
                    (radars),
                    grid_shape=(1, 517, 755),
                    grid_limits=((2000, 2000), (-517073/2, 517073/2), (-(789412+720621)/4, (789412+720621)/4)), # CAPPI 2km, limits based on size by SHMU - TODO - add this params to config
                    grid_origin=((46.05+50.7)/2,(13.6+23.8)/2),
                    fields=[RAINBOW_FIELD_NAMES[product]],
                )

                # to np.array from Grid object
                data = grid.fields[RAINBOW_FIELD_NAMES[product]]["data"][0]

                # save parameters for compression
                datamin = RAINBOW_DATAMINS[product]
                datamax = RAINBOW_DATAMAXES[product]
                datadepth = 8 #TODO customizable parameter

                # remove clutter from image
                if product in ['dBZ', 'dBZv']:
                    clmap = classify.filter_gabella(data, rm_nans=False, cartesian=True)
                    data[np.nonzero(clmap)] = np.nan
                    clmap = classify.filter_gabella(data, rm_nans=False, cartesian=True)
                    data[np.nonzero(clmap)] = np.nan
                
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
        
        complete_path = Path(conf.log_path) / 'completed_timestamps.txt'
        complete_path.touch()
            
        for datetime_str in datetimes_str:
            with open(complete_path, 'a') as cf:
                cf.write(datetime_str + ' ' + '\n')
            
    else:
        nonrainy_path = Path(conf.log_path) / 'nonrainy_timestamps.txt'
        nonrainy_path.touch()
            
        for datetime_str in datetimes_str:
            with open(nonrainy_path, 'a') as nf:
                nf.write(datetime_str + ' ' + '\n')
        
    del data
    del grid
    gc.collect()
                    
        
def main(conf, restarted):
    # setup paths
    input_path = Path(conf.input_path)
    output_path = Path(conf.output_path)
    inc_ts_path = Path(conf.log_path) / 'incomplete_timestamps.txt'
    com_ts_path = Path(conf.log_path) / 'completed_timestamps.txt'
    non_ts_path = Path(conf.log_path) / 'nonrainy_timestamps.txt'
    
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
        
        # delete observations from output that didn't close properly
        for path_object in output_path.rglob('*'):
            if path_object.is_file() and path_object.suffix == '.h5':
                    if path_object.stem not in complete_timestamps:
                        logging.info(f'{path_object} not found in completed_timestamps.txt. Deleting...')
                            
                        path_object.unlink()
                        logging.info(f'Deleted.')
    
    res_files = []
    res_outfile = []
    res_datetime = []
    
    logging.info('Starting check of observations suitability.')
    
    # iterate over subdirectories for each day
    for path_object in input_path.glob('*'):
        if path_object.is_dir():
            date_str = str(path_object)[-8:]
            start_datetime = dt.datetime.strptime(date_str, '%Y%m%d')
            # iterate over one day
            for lag in range(0, 24*60, interval):
                datetime = start_datetime + dt.timedelta(minutes=lag)
                datetime_str = datetime.strftime(date_format)
                # outfile path
                output_subpath = output_path / date_str
                outfile = output_subpath / (datetime_str + '.h5')
                input_subpath = input_path / date_str
                
                # append to dask only timestamps that arent already transformed or discarded from data
                if datetime_str not in incomplete_timestamps and datetime_str not in complete_timestamps and datetime_str not in nonrainy_timestamps:
                    selected_files_dict = defaultdict(list)
                    
                    # find paths to selected timestamp and given product
                    for product in conf.products:
                        selected_files = []

                        # get paths to 4 radar station images of 1 product and append to one array
                        for radar_code in conf.radar_codes:
                            for subpath_object in Path(input_subpath / radar_code).glob('*'):
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
    for path_object in input_path.glob('*'):
        if path_object.is_dir():
            
            # initialize start and end datetimes
            date_str = str(path_object)[-8:]
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
    
    logging.info(f'Append done.')
    
    logging.info(f"Creating {len(res)} dask tasks! {int(jump/interval)} observations per task.")
    
    with ProgressBar(minimum = 10, dt = 60):
        scheduler = "processes" if conf.nworkers > 1 else "single-threaded"
        res = dask.compute(*res, scheduler=scheduler)
                
    logging.info(f"All done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--config', '--c', required=True, help='Path to the config file.')
    parser.add_argument('--restarted', '--r', action='store_true', help='Flag to indicate if script is being rerun/restarted due to interruption.')

    args = parser.parse_args()
    conf = load_config(Path(args.config))
    
    main(conf, args.restarted)
    