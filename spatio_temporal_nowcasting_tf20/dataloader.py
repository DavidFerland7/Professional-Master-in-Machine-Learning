import argparse
import datetime
from datetime import timedelta
import json
import os
import typing
import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import utils as util
import h5py
import logging
import copy
from preprocessing import preprocess_dataframe
from collections import defaultdict
import time
import pickle
import random


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """function to prepare & return your own data loader.

    Note that you can use either the netCDF or HDF5 data. Each iteration over your data loader should return a
    2-element tuple containing the tensor that should be provided to the model as input, and the target values. In
    this specific case, you will not be able to provide the latter since the dataframe contains no GHI, and we are
    only interested in predictions, not training. Therefore, you must return a placeholder (or ``None``) as the second
    tuple element.

    Reminder: the dataframe contains imagery paths for every possible timestamp requested in ``target_datetimes``.
    However, it is expected to use "past" imagery (i.e. imagery at T<=0) for any T in
    ``target_datetimes``, but not rely on "future" imagery to generate predictions (for T>0).

    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
        target_datetimes: a list of timestamps the data loader should use to provide imagery for model.
            The ordering of this list is important, as each element corresponds to a sequence of GHI values
            to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
            which are added to each timestamp (T=0) in this datetimes list.
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """

    def get_clearsky_ghi(df, config):
        df_feat = df.loc[:, [col for col in df if col.startswith("CLEARSKY_GHI_")]]
        return df_feat

    def get_azimuth(df, config):
        df_feat = df.loc[:, [col for col in df if col.startswith("AZIMUTH_")]]
        return df_feat

    def get_time_of_day(df, config):
        df_feat = df.loc[:, [col for col in df if col.startswith("FEAT_TIME_OF_DAY_")]]
        return df_feat

    def get_biweekly(df, config):
        df_feat = df.loc[:, [col for col in df if col.startswith("FEAT_BIWEEKLY_")]]
        return df_feat

    def get_targets_and_mask(df, config):
        df_masked_targets = df.loc[:, [f'GHI_{target}' for target in config['data']['features_from_df']['TARGET']] + [f'MASK_{target}' for target in config['data']['features_from_df']['TARGET']]]
        return df_masked_targets

    ### Preprocess dataframe ###
    df_paths, df = preprocess_dataframe(dataframe, config, {'stations': stations})
    print("first index={}; last index={}".format(min(df.index), max(df.index)))

    backward_offsets = config['data_generator']['backward_offsets']
    if not backward_offsets:
        assert 'step_size' in config['data']['imgs'].keys() and 'time_steps' in config['data']['imgs'].keys(), 'No backward offset or timestep and step size given in config.'
        step_size = config['data']['imgs']['step_size']
        step_count = config['data']['imgs']['time_steps']
        backward_offsets = [datetime.timedelta(minutes=i * step_size) for i in range(step_count)]
    else:
        backward_offsets = [datetime.timedelta(minutes=offset) for offset in backward_offsets]
        step_count = len(backward_offsets)

    start_time_ = time.perf_counter()
    # Get cropped images (load if exist or create/save otherwise) ###
    region_size = config['data']['imgs']['region_size']
    pickle_folder = f'/project/cq-training-1/project1/teams/team11/data/imgs_cropped/{region_size}/'

    min_requested_date = min(df.index)
    max_requested_date = max(df.index)
    min_available_date = max_requested_date
    max_available_date = min_requested_date
    image_cache_files = []

    # Build files list and check is requested interval is available
    if not os.path.isdir(pickle_folder):
        raise Exception(f'Requested crop size ({region_size}) does not exist in cache')
    for f in os.listdir(pickle_folder):
        bounds = f.split('_')
        start_timestamp, end_timestamp = datetime.datetime.strptime(bounds[0], "%Y-%m-%d %H:%M:%S"), datetime.datetime.strptime(bounds[1].split('.')[0], "%Y-%m-%d %H:%M:%S")
        # Add file to list if one of it's bound is within the requested range
        if start_timestamp <= max_requested_date and end_timestamp >= min_requested_date:
            image_cache_files.append(pickle_folder + f)
        min_available_date = min(start_timestamp, min_available_date)
        max_available_date = max(end_timestamp, max_available_date)
    if min_requested_date < min_available_date or max_requested_date > max_available_date:
        raise Exception(f'Requested range [{min_requested_date} - {max_requested_date}] is wider than available range [{min_available_date} - {max_available_date}]')

    print("\ntotal time to util.get_all_imgs:{}".format(time.perf_counter() - start_time_))

    # Get clearsky_ghi
    clearsky_ghi_features = get_clearsky_ghi(df, config)
    # Get azimuth
    azimuth_features = get_azimuth(df, config)
    # Get time of day
    time_of_day_features = get_time_of_day(df, config)
    # Get biweekly
    biweekly_features = get_biweekly(df, config)
    # Get targets and mask
    masked_targets = get_targets_and_mask(df, config)

    available_stations_dict = defaultdict(list)
    for t in np.unique(df.index.tolist()):
        t_stations = np.array([df.at[t, 'station']])
        if len(t_stations.shape) > 1:
            t_stations = np.squeeze(t_stations)

        for station in t_stations:
            idx = list(stations.keys()).index(station)
            available_stations_dict[t].append(idx)

    def data_generator():
        #     """Yields a tuple of tensors.
        #     The first tensor contains the input data and is of shape (batch size * number of stations, number of time steps, number of channels, width, height).
        #     The second tensor contains the target GHIs and is of shape (batch size * number of stations, number of targets)
        #     """
        stations_id = stations.keys()
        n_stations = len(stations_id)
        tensor_dim = (
            n_stations,
            step_count,
            config['data']['imgs']['region_size'],
            config['data']['imgs']['region_size'],
            config['data']['imgs']['n_channels']
        )

        # Overwrite targetdatetime
        if config['globals']['training']:
            target_datetimes = df.index.to_pydatetime()

            # Shuffle filepath lists
            if config['data_generator']['shuffle_target_datetimes']:
                random.seed(1)
                random.shuffle(image_cache_files)

        for f in image_cache_files:
            with open(f, 'rb') as pickled_image_cache:
                imgs_cache = pickle.load(pickled_image_cache)
                for t0 in list(imgs_cache.keys()):
                    if t0 not in target_datetimes:
                        continue
                    if t0 not in masked_targets.index:
                        print(f'Skipping t0={t0}. No targets available in df using user_config dates.')
                        break
                    if t0 not in clearsky_ghi_features.index:
                        print(f'Skipping t0={t0}. No clearsky_ghi feature available in df using user_config dates.')
                        break
                    if t0 not in azimuth_features.index:
                        print(f'Skipping t0={t0}. No azimuth feature available in df using user_config dates.')
                        break
                    images = np.zeros(shape=tensor_dim)
                    targets = masked_targets.loc[t0]
                    clearsky_ghi_feats = clearsky_ghi_features.loc[t0]
                    azimuth_feats = azimuth_features.loc[t0]
                    time_of_day_feats = time_of_day_features.loc[t0]
                    biweekly_feats = biweekly_features.loc[t0]
                    if len(targets.shape) == 1:
                        targets = np.array([targets])
                        clearsky_ghi_feats = np.array([clearsky_ghi_feats])
                        azimuth_feats = np.array([azimuth_feats])
                        time_of_day_feats = np.array([time_of_day_feats])
                        biweekly_feats = np.array([biweekly_feats])
                    else:
                        targets = np.array(targets)
                        clearsky_ghi_feats = np.array(clearsky_ghi_feats)
                        azimuth_feats = np.array(azimuth_feats)
                        time_of_day_feats = np.array(time_of_day_feats)
                        biweekly_feats = np.array(biweekly_feats)

                    # Get available stations at t0
                    available_stations = available_stations_dict[t0]

                    # Get past and current images
                    past_timestamps = [t0 - offset for offset in backward_offsets]
                    for i, timestamp in enumerate(past_timestamps):
                        if timestamp in imgs_cache:
                            images[:, i, ...] = imgs_cache[timestamp]
                        else:
                            logging.info(f'Missing offset for {timestamp} using interpolation instead.')
                            # Interpolate with last image in cache
                            imgs_cache[timestamp] = util.impute_img(imgs_cache, timestamp)
                            images[:, i, ...] = imgs_cache[timestamp]

                    # Package and yield valid data per available station
                    for i, station_idx in enumerate(available_stations):
                        imgs = tf.convert_to_tensor(images[station_idx, ...])
                        clearsky_ghi_feat = tf.convert_to_tensor(clearsky_ghi_feats[i, :])
                        azimuth_feat = tf.convert_to_tensor(azimuth_feats[i, :])
                        time_of_day_feat = tf.convert_to_tensor(time_of_day_feats[i, :])
                        biweekly_feat = tf.convert_to_tensor(biweekly_feats[i, :])

                        target = tf.convert_to_tensor(targets[i, :])

                        assert imgs.shape == (step_count, config['data']['imgs']['region_size'], config['data']['imgs']['region_size'], config['data']['imgs']['n_channels']), f'Sample shape error. Got {imgs.shape}, '
                        assert target.shape == (2 * len(target_time_offsets),), f'Target shape error. Got {target.shape}, expected ({2*len(target_time_offsets)},).'

                        input_dict = {'imgs': imgs, 'CLEARSKY_GHI': clearsky_ghi_feat, 'azimuth': azimuth_feat, 'timeofday': time_of_day_feat, 'biweekly': biweekly_feat}
                        yield input_dict, target

    data_loader = tf.data.Dataset.from_generator(
        data_generator,
        output_types=({'imgs': tf.float32, 'CLEARSKY_GHI': tf.float32, 'azimuth': tf.float32, 'timeofday': tf.float32, 'biweekly': tf.float32}, tf.float32),
        output_shapes=(
            {
                'imgs': (step_count, config['data']['imgs']['region_size'], config['data']['imgs']['region_size'], config['data']['imgs']['n_channels']),
                'CLEARSKY_GHI': (len(config['data']['features_from_df']['CLEARSKY_GHI']),),
                'azimuth': (len(config['data']['features_from_df']['AZIMUTH']),),
                'timeofday': (time_of_day_features.shape[1],),
                'biweekly': (biweekly_features.shape[1],),
            },
            (len(target_time_offsets) * 2,)
        )
    )

    return data_loader
