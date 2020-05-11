import datetime
from datetime import timedelta
import json
import math
import os
import typing
import warnings
import cv2 as cv
import h5py
import lz4.frame
import matplotlib.dates
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import OrderedDict
from nptyping import Array
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import time
from tqdm import tqdm
import logging
import wandb
import pickle


class FIFOCacheDictionary(OrderedDict):
    'Limit size, evicting the least recently looked-up key when full'

    def __init__(self, maxsize, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

class WandbExtraLogs(tf.keras.callbacks.Callback):

    def __init__(self):
        self.epoch_timer = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_timer = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({'epoch_time':time.perf_counter()-self.epoch_timer}, step=epoch)
        self.epoch_timer = 0

def impute_img(image_cache, timestamp):
    """Returns the  missing channel by imputing from two different time steps if possible

    Parameters:
    image cache
    missing timestamp

    Returns:
    numpy array
    """
    earliest_t = max((k for k in image_cache if k < timestamp), default=None)
    after_t = min((k for k in image_cache if k > timestamp), default=None)

    if not earliest_t or not after_t:
        if earliest_t:
            return image_cache[earliest_t]
        if after_t:
            return image_cache[after_t]

    alpha = 0.5  # 0.5 for now since we just check the previous frame and after in the cache
    beta = (1.0 - alpha)

    return cv.addWeighted(image_cache[earliest_t], alpha, image_cache[after_t], beta, 0.0)

def benchmark(dataset, num_sample=1000, whole_epoch=False):
    counter = 0
    start_time = time.perf_counter()
    for i, sample in tqdm(enumerate(dataset), total=num_sample):
        # Performing a training step
        counter += 1
        if not whole_epoch and i == num_sample:
            break
    tf.print(f'Execution time: {time.perf_counter() - start_time}. Imgs per second: {1/((time.perf_counter() - start_time)/min(counter,num_sample))}')

def get_all_imgs(stations, df_paths, region_size, n_channels, training, prints=False):
    if training:
        filepaths = np.unique(df_paths['hdf5_8bit_path'])
        pickle_folder = f'./data/imgs_cropped/{region_size}/'
        if not os.path.isdir(pickle_folder):
            os.mkdir(pickle_folder)
        file_start_timestamp = min(df_paths.index)
        file_end_timestamp = file_start_timestamp + datetime.timedelta(days=14)
        pickle_filename = f'./data/imgs_cropped/{region_size}/{file_start_timestamp}_{file_end_timestamp}.pickle'
    else:
        filepaths_offsets = defaultdict(list)
        filepaths = []
        for filepath, offset in df_paths:
            filepaths_offsets[filepath] = offset
            filepaths.append(filepath)

    images = {}

    for filepath in tqdm(filepaths, desc="Generating crops for filepaths (all stations at once)"):
        if not os.path.isfile(filepath):
            print(f"Invalid hdf5 path: {filepath}")
            continue
        with h5py.File(filepath, "r") as h5_data:
            global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
            global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
            archive_lut_size = global_end_idx - global_start_idx
            global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
            global_end_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_end_time"], "%Y.%m.%d.%H%M")
            lut_timestamps = [global_start_time + idx * datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
            station_coords = get_station_coords(h5_data, stations, range(len(lut_timestamps)))
            if not station_coords:
                logging.info(f'No station coords found in file => skipping it. \nfilepath:{filepath}')
                continue

            ### Get images
            if training:
                for i, timestamp in enumerate(lut_timestamps):
                    imgs_crops = get_sat_img_crops(station_coords, h5_data, i, region_size)
                    images[timestamp] = imgs_crops

                ### Save file if completed 14 days interval
                if global_end_time >= file_end_timestamp:
                    with open(pickle_filename, 'wb') as pickle_file:
                        pickle.dump(images, pickle_file)
                    images = {}
                    file_start_timestamp = file_end_timestamp
                    file_end_timestamp = file_start_timestamp + datetime.timedelta(days=14)
                    pickle_filename = f'./data/imgs_cropped/{region_size}/{file_start_timestamp}_{file_end_timestamp}.pickle'
            else:
                for offset in filepaths_offsets[filepath]:
                    if offset == None:
                        continue
                    imgs_crops = get_sat_img_crops(station_coords, h5_data, offset, region_size)
                    images[lut_timestamps[offset]] = imgs_crops
    if training:
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(images, pickle_file)
    else:
        return images

def get_filepath_timestamps_file_offsets(
    df: pd.DataFrame,
    t0: datetime.datetime,
    backward_offsets: typing.List[datetime.timedelta]
) -> typing.List[typing.Tuple[typing.AnyStr, typing.List[typing.Tuple[datetime.datetime, int]]]]:
    """Returns filepath and list of timestamp + offset for every timestamp of the sequence

    Parameters:
    df (pd.Dataframe): Dataframe containing GHI values
    t0 (datetime.datetime): Starting timestamp for sequence
    backward_offsets (list[datetime.timedelta]): List of offset for input sequence

    Returns:
    tuple: Tuple of filepath and list of timestamp + offset
    """
    output = []

    ## value is either a int/str or a series(in which case all values are the same and we can just take the 1st element)
    if type(df.at[t0,'hdf5_8bit_path']) == str:
        filepath = df.at[t0,'hdf5_8bit_path']
    else:
        filepath = df.at[t0,'hdf5_8bit_path'][0]

    if type(df.at[t0,'hdf5_8bit_offset']) == int:
        timestamps_offsets = [df.at[t0,'hdf5_8bit_offset']]
    else:
        timestamps_offsets = [df.at[t0,'hdf5_8bit_offset'][0]]


    for offset in backward_offsets:
        t = (t0 - offset)

        if t in df.index:

            ## value is either a int/str or a series(in which case all values are the same and we can just take the 1st element)
            if type(df.at[t,'hdf5_8bit_path']) == str:
                filepath_t = df.at[t,'hdf5_8bit_path']
            else:
                filepath_t = df.at[t,'hdf5_8bit_path'][0]

            if type(df.at[t,'hdf5_8bit_offset']) == int:
                timestamps_offset_t = df.at[t,'hdf5_8bit_offset']
            else:
                timestamps_offset_t = df.at[t,'hdf5_8bit_offset'][0]


            #filepath_t = filepath
            if filepath != filepath_t:
                #now we need to change file, so writing previous cumulative infos before re-init
                output.append((filepath, timestamps_offsets))

                #start of a new file and timestamps_offsets
                filepath = filepath_t
                timestamps_offsets = []

            timestamps_offsets.append(timestamps_offset_t)

        else:
            timestamps_offsets.append(None)
    output.append((filepath, timestamps_offsets))
    return output

def get_files_timestamps_offset_for_batch(
    df: pd.DataFrame,
    target_datetimes: typing.List[datetime.datetime],
    step_size=30, steps=3
):

    filepaths_times_offset = []
    input_times_seq = []
    for t0 in target_datetimes:
        step = timedelta(minutes=step_size)
        input_times = [t0]
        filepaths_times_offset.extend(df[df.index==input_times[-1]].hdf5_8bit_path.tolist())
        for _ in range(steps-1):
            input_times.append(input_times[-1] - step)
            filepaths_times_offset.extend(df[df.index==input_times[-1]].hdf5_8bit_path.tolist())
        input_times_seq.append(input_times)
    return filepaths_times_offset

def get_station_coords(h5_data, stations_id_postion, offsets):
    """Returns a list of station image coordinates.
    Returns None if coord could not be found in file at any offset in given list.

    Parameters:
    h5_data (hdf5 file): File content of hdf5 file.
    stations_id_postion (list(float,float,float))): List of latitude/longitude/elevation for each station.
    offsets (list(int)): List of offset to look for lat/long infor in file.

    Returns:
    list: List of station image coordinates.
    """
    stations_coords = []
    # assume lats/lons stay identical throughout all frames
    lats, lons = None, None
    for offset in offsets:
        if not offset:
            continue
        lats, lons = fetch_hdf5_sample("lat", h5_data, offset), fetch_hdf5_sample("lon", h5_data, offset)
    if lats is None or lons is None:
        logging.info('Could not fetch lats/lons arrays (hdf5 might be empty).')
        return None
    for reg, coords in stations_id_postion.items():
        station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
        stations_coords.append(station_coords)
    return stations_coords


def get_sat_img_crops(
    stations_postion: typing.List[typing.Tuple[float, float, float]],
    h5_data,
    hdf5_offset: int,
    region_size: int,
    channels=['ch1','ch2','ch3','ch4','ch6']
) -> Array[int, float, float, int]:
    """Returns array of satelite image crops in a tensor of shape (station idx, width, height, channel idx)

    Parameters:
    stations_id_postion (list((float,float,float))): List of latitude/longitude/elevation.
    h5_data (hdf5 file): File handler.
    hdf5_offset (int): Offset of image in hdf5 file.
    region_size (int): Size of the region of the image to retreived. Cropped region is square and centered on each station.
    channels (list[str]): List of channels' name to retreive

    Returns:
    array: Array of croppped images for all station at given offset in given file.
    """
    # Get relevant satelite image crops of current file for all stations
    imgs_crops = np.zeros((len(stations_postion), region_size, region_size, len(channels)), dtype=np.uint8)
    for channel_idx, channel_name in enumerate(channels):
        assert channel_name in h5_data, f"Missing channel: {channel_name}"
        norm_min = h5_data[channel_name].attrs.get("orig_min", None)
        norm_max = h5_data[channel_name].attrs.get("orig_max", None)
        channel_data = fetch_hdf5_sample(channel_name, h5_data, hdf5_offset)
        assert channel_data is None or channel_data.shape == (650, 1500), f'One of the saved channels({channel_name}) had an expected dimension. Expected (650, 1500), got {array.shape}'
        last_valid_array_idx = None
        if channel_data is None:
            continue
        channel_data = (((channel_data.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
        for station_idx, station_coords in enumerate(stations_postion):
            # Extract crop from satelite image and add to tensor
            margin_xy = (region_size//2, region_size//2)
            bounds_x = (int(station_coords[0] - margin_xy[0]), int(station_coords[0] + margin_xy[0]))
            bounds_y = (int(station_coords[1] - margin_xy[1]), int(station_coords[1] + margin_xy[1]))
            clipped_bounds_x = (max(0,bounds_x[0]), min(bounds_x[1], 650))
            clipped_bounds_y = (max(0,bounds_y[0]), min(bounds_y[1], 1500))
            delta_x = (bounds_x[0]-clipped_bounds_x[0], bounds_x[1]-clipped_bounds_x[1])
            delta_y = (bounds_y[0]-clipped_bounds_y[0], bounds_y[1]-clipped_bounds_y[1])
            imgs_crops[station_idx, -delta_x[0]:region_size-delta_x[1], -delta_y[0]:region_size-delta_y[1], channel_idx] = channel_data[clipped_bounds_x[0]:clipped_bounds_x[1], clipped_bounds_y[0]:clipped_bounds_y[1]]
    return imgs_crops

def get_sat_imgs_crops(
    stations_id_postion: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    filepaths_times_offset: typing.Dict[typing.AnyStr, typing.Tuple[datetime.datetime, int]],
    region_size: int,
    channels=['ch1','ch2','ch3','ch4','ch6']
) -> typing.Dict[datetime.datetime,Array[int, int, float, float]]:
    """Returns dictionary indexed by timestamp of satelite image crops in a tensor of shape (station idx, channel idx, width, height)

    Returns an empty dictionary if no filepaths/timestamps provided.

    Parameters:
    stations_id_postion (dict(str:(float,float,float))): Dictionary of latitude/longitude/elevation indexed by station id
    filepaths_times_offset (dict(str:(datetime,int))): Dictionary indexed by filepath of datetime and offset found in each hdf5 file
    region_size (int): size of the region of the image to retreived. Cropped region is square and centered on each station.
    channels (list[str]): List of channels' name to retreive

    Returns:
    dict: Dictionary of croppped images for all station indexed by timestamp
    """
    sat_img_dict = defaultdict(list)
    stations_coords = None
    stations_id = list(stations_id_postion.keys())

    for hdf5_path, timestamp_offset_list in filepaths_times_offset.items():
        assert os.path.isfile(hdf5_path), f"Invalid hdf5 path: {hdf5_path}"
        with h5py.File(hdf5_path, "r") as h5_data:
            timestamps, hdf5_offsets = zip(*timestamp_offset_list)

            # Get stations coords (relative to satelite images)
            if not stations_coords:
                stations_coords = get_station_coords(h5_data, stations_id_postion)

            # Verify timestamps match current hdf5 file
            global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
            global_end_time = global_start_time + timedelta(hours=24)
            for t in timestamps:
                assert global_start_time <= t and t <= global_end_time, f"Timestamp {t} does not belong to file {hdf5_path}"

            # Get relevant satelite image crops of current file for all stations
            imgs_crops = np.zeros((len(hdf5_offsets), len(stations_coords.keys()), len(channels), region_size, region_size), dtype=np.uint8)
            for channel_idx, channel_name in enumerate(channels):
                assert channel_name in h5_data, f"Missing channel: {channel_name}"
                norm_min = h5_data[channel_name].attrs.get("orig_min", None)
                norm_max = h5_data[channel_name].attrs.get("orig_max", None)
                channel_data = [fetch_hdf5_sample(channel_name, h5_data, idx) for idx in hdf5_offsets]
                assert all([array is None or array.shape == (650, 1500) for array in channel_data]), f'One of the saved channels({channel_name}) had an expected dimension. Expected (650, 1500), got {array.shape}'
                last_valid_array_idx = None
                for idx in range(len(hdf5_offsets)):
                    array = channel_data[idx]
                    if array is None:
                        continue
                    array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                    for station_idx, station_coords in enumerate(stations_coords.values()):
                        # Extract crop from satelite image and add to tensor
                        margin_xy = (region_size//2, region_size//2)
                        bounds_x = (max(0,station_coords[0] - margin_xy[0]), min(station_coords[0] + margin_xy[0], 650))
                        bounds_y = (max(0,station_coords[1] - margin_xy[1]), min(station_coords[1] + margin_xy[1], 1500))
                        imgs_crops[idx, station_idx, channel_idx, :, :] = array[bounds_x[0]:bounds_x[1]+1, bounds_y[0]:bounds_y[1]+1]

            # Fill dictionary with img_crop tensor by timestamp
            for i in range(len(hdf5_offsets)):
                sat_img_dict[timestamps[i]].append(imgs_crops[i, :, :, :, :])
    return sat_img_dict

def get_ghi(
    df,
    stations_id: typing.List[str],
    timestamps: typing.List[datetime.datetime],
    target_time_offsets: typing.List[datetime.timedelta],
):
    """Returns ghi values for all stations in station_ids at every timestamp in timestamps

    Returns None if any input list is empty

    Parameters:
    df (pd.Dataframe): Dataframe containing GHI values
    station_ids (list[str]): List of station identifiers
    timestamps (list[datetime]): List of timestamps to capture GHI value at

    Returns:
    dict: Dictionary of ghi values at each time offset indexed by station id
    """
    if not stations_id or not timestamps:
        return None

    stations_ghi = {}
    for station_id in stations_id:
        stations_ghi[station_id] = np.array([[df.at[pd.Timestamp(t+delta), station_id + "_GHI"] for delta in target_time_offsets] for t in timestamps])
    return stations_ghi

def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""
    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0
    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


def get_label_html_color_code(idx):
    """Returns the PASCAL VOC HTML color code for a given label index."""
    color_array = get_label_color_mapping(idx)
    return f"#{color_array[0]:02X}{color_array[1]:02X}{color_array[2]:02X}"


def fig2array(fig):
    """Transforms a pyplot figure into a numpy-compatible BGR array.

    The reason why we flip the channel order (RGB->BGR) is for OpenCV compatibility. Feel free to
    edit this function if you wish to use it with another display library.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf[..., ::-1]


def compress_array(
        array: np.ndarray,
        compr_type: typing.Optional[str] = "auto",
) -> bytes:
    """Compresses the provided numpy array according to a predetermined strategy.

    If ``compr_type`` is 'auto', the best strategy will be automatically selected based on the input
    array type. If ``compr_type`` is an empty string (or ``None``), no compression will be applied.
    """
    assert compr_type is None or compr_type in ["lz4", "float16+lz4", "uint8+jpg",
                                                "uint8+jp2", "uint16+jp2", "auto", ""], \
        f"unrecognized compression strategy '{compr_type}'"
    if compr_type is None or not compr_type:
        return array.tobytes()
    if compr_type == "lz4":
        return lz4.frame.compress(array.tobytes())
    if compr_type == "float16+lz4":
        assert np.issubdtype(array.dtype, np.floating), "no reason to cast to float16 is not float32/64"
        return lz4.frame.compress(array.astype(np.float16).tobytes())
    if compr_type == "uint8+jpg":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jpg compression via tensorflow requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8, "jpg compression requires uint8 array"
        return tf.io.encode_jpeg(array).numpy()
    if compr_type == "uint8+jp2" or compr_type == "uint16+jp2":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jp2 compression via opencv requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8 or array.dtype == np.uint16, "jp2 compression requires uint8/16 array"
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        retval, buffer = cv.imencode(".jp2", array)
        assert retval, "JPEG2000 encoding failed"
        return buffer.tobytes()
    # could also add uint16 png/tiff via opencv...
    if compr_type == "auto":
        # we cheat for auto-decompression by prefixing the strategy in the bytecode
        if array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)):
            if array.dtype == np.uint8:
                return b"uint8+jpg" + compress_array(array, compr_type="uint8+jpg")
            if array.dtype == np.uint16:
                return b"uint16+jp2" + compress_array(array, compr_type="uint16+jp2")
        return b"lz4" + compress_array(array, compr_type="lz4")


def decompress_array(
        buffer: typing.Union[bytes, np.ndarray],
        compr_type: typing.Optional[str] = "auto",
        dtype: typing.Optional[typing.Any] = None,
        shape: typing.Optional[typing.Union[typing.List, typing.Tuple]] = None,
) -> np.ndarray:
    """Decompresses the provided numpy array according to a predetermined strategy.

    If ``compr_type`` is 'auto', the correct strategy will be automatically selected based on the array's
    bytecode prefix. If ``compr_type`` is an empty string (or ``None``), no decompression will be applied.

    This function can optionally convert and reshape the decompressed array, if needed.
    """
    compr_types = ["lz4", "float16+lz4", "uint8+jpg", "uint8+jp2", "uint16+jp2"]
    assert compr_type is None or compr_type in compr_types or compr_type in ["", "auto"], \
        f"unrecognized compression strategy '{compr_type}'"
    assert isinstance(buffer, bytes) or buffer.dtype == np.uint8, "invalid raw data buffer type"
    if isinstance(buffer, np.ndarray):
        buffer = buffer.tobytes()
    if compr_type == "lz4" or compr_type == "float16+lz4":
        buffer = lz4.frame.decompress(buffer)
    if compr_type == "uint8+jpg":
        # tf.io.decode_jpeg often segfaults when initializing parallel pipelines, let's avoid it...
        # buffer = tf.io.decode_jpeg(buffer).numpy()
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type.endswith("+jp2"):
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type == "auto":
        decompr_buffer = None
        for compr_code in compr_types:
            if buffer.startswith(compr_code.encode("ascii")):
                decompr_buffer = decompress_array(buffer[len(compr_code):], compr_type=compr_code,
                                                  dtype=dtype, shape=shape)
                break
        assert decompr_buffer is not None, "missing auto-decompression code in buffer"
        buffer = decompr_buffer
    array = np.frombuffer(buffer, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array


def fetch_hdf5_sample(
        dataset_name: str,
        reader: h5py.File,
        sample_idx: int,
) -> typing.Any:
    """Decodes and returns a single sample from an HDF5 dataset.

    Args:
        dataset_name: name of the HDF5 dataset to fetch the sample from using the reader. In the context of
            the GHI prediction project, this may be for example an imagery channel name (e.g. "ch1").
        reader: an HDF5 archive reader obtained via ``h5py.File(...)`` which can be used for dataset indexing.
        sample_idx: the integer index (or offset) that corresponds to the position of the sample in the dataset.

    Returns:
        The sample. This function will automatically decompress the sample if it was compressed. It the sample is
        unavailable because the input was originally masked, the function will return ``None``. The sample itself
        may be a scalar or a numpy array.
    """
    dataset_lut_name = dataset_name + "_LUT"
    if dataset_lut_name in reader:
        sample_idx = reader[dataset_lut_name][sample_idx]
        if sample_idx == -1:
            return None  # unavailable
    dataset = reader[dataset_name]
    if "compr_type" not in dataset.attrs:
        # must have been compressed directly (or as a scalar); return raw output
        return dataset[sample_idx]
    compr_type, orig_dtype, orig_shape = dataset.attrs["compr_type"], None, None
    if "orig_dtype" in dataset.attrs:
        orig_dtype = dataset.attrs["orig_dtype"]
    if "orig_shape" in dataset.attrs:
        orig_shape = dataset.attrs["orig_shape"]
    if "force_cvt_uint8" in dataset.attrs and dataset.attrs["force_cvt_uint8"]:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint8, shape=orig_shape)
        orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
        array = ((array.astype(np.float32) / 255) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
    elif "force_cvt_uint16" in dataset.attrs and dataset.attrs["force_cvt_uint16"]:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint16, shape=orig_shape)
        orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
        array = ((array.astype(np.float32) / 65535) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
    else:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=orig_dtype, shape=orig_shape)
    return array
