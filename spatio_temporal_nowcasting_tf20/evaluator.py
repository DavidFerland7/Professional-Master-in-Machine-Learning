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
from collections import defaultdict
import time
import pickle
import random
from preprocessing import preprocess_dataframe


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

    stations_id = list(stations.keys())

    ### Preprocess dataframe ###
    df_paths, df = preprocess_dataframe(dataframe, config, {'stations': stations})

    # INFERENCE: Keep only stations requested
    df = df[df['station'].isin(stations_id)]

    print("first index={}; last index={}".format(min(df.index), max(df.index)))

    # Get clearsky_ghi
    clearsky_ghi_features = get_clearsky_ghi(df, config)
    # Get azimuth
    azimuth_features = get_azimuth(df, config)
    # Get time of day
    time_of_day_features = get_time_of_day(df, config)
    # Get biweekly
    biweekly_features = get_biweekly(df, config)

    backward_offsets = config['data_generator']['backward_offsets']
    if not backward_offsets:
        assert 'step_size' in config['data']['imgs'].keys() and 'time_steps' in config['data']['imgs'].keys(), 'No backward offset or timestep and step size given in config.'
        step_size = config['data']['imgs']['step_size']
        step_count = config['data']['imgs']['time_steps']
        backward_offsets = [datetime.timedelta(minutes=i * step_size) for i in range(step_count)]
    else:
        backward_offsets = [datetime.timedelta(minutes=offset) for offset in backward_offsets]
        step_count = len(backward_offsets)

    targets_input = target_datetimes
    target_datetimes_safe = target_datetimes

    available_stations_dict = defaultdict(list)
    for t in list(set(target_datetimes_safe)):
        t_stations = np.array([df.at[t, 'station']])
        if len(t_stations.shape) > 1:
            t_stations = np.squeeze(t_stations)

        for station in t_stations:
            idx = stations_id.index(station)
            available_stations_dict[t].append(idx)

    def data_generator():
        #     """Yields a tuple of tensors.
        #     The first tensor contains the input data and is of shape (batch size * number of stations, number of time steps, number of channels, width, height).
        #     The second tensor contains the target GHIs and is of shape (batch size * number of stations, number of targets)
        #     """

        n_stations = len(stations_id)
        tensor_dim = (
            n_stations,
            step_count,
            config['data']['imgs']['region_size'],
            config['data']['imgs']['region_size'],
            config['data']['imgs']['n_channels']
        )

        for t0 in targets_input:

            # INFERENCE -> get images for the t0 requested
            #image_cache_files_safe = util.get_files_timestamps_offset_for_batch(df_paths, [t0], config['data']['imgs']['step_size'], config['data']['imgs']['time_steps'])
            image_cache_files_safe = util.get_filepath_timestamps_file_offsets(df_paths, t0, backward_offsets)
            imgs_cache = util.get_all_imgs(stations, image_cache_files_safe, config['data']['imgs']['region_size'], config['data']['imgs']['n_channels'], config['globals']['prints'], config['globals']['training'])

            if t0 not in target_datetimes_safe:
                raise Exception("Error -> in Inference mode, all t0 should be in target datetimes list")
            if t0 not in clearsky_ghi_features.index:
                print(f'Skipping t0={t0}. No clearsky_ghi feature available in df using user_config dates.')
                break
            if t0 not in azimuth_features.index:
                print(f'Skipping t0={t0}. No azimuth feature available in df using user_config dates.')
                break
            images = np.zeros(shape=tensor_dim)

            clearsky_ghi_feats = clearsky_ghi_features.loc[t0]
            azimuth_feats = azimuth_features.loc[t0]
            time_of_day_feats = time_of_day_features.loc[t0]
            biweekly_feats = biweekly_features.loc[t0]
            if len(clearsky_ghi_feats.shape) == 1:
                clearsky_ghi_feats = np.array([clearsky_ghi_feats])
                azimuth_feats = np.array([azimuth_feats])
                time_of_day_feats = np.array([time_of_day_feats])
                biweekly_feats = np.array([biweekly_feats])
            else:
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
                    pass
                    # Decided to not impute images anymore and to set them to zeros tensors instead
                    # This can be interpreted as noise injection and has regularization effect

                    # logging.info(f'Missing offset for {timestamp} using interpolation instead.')
                    # # Interpolate with last image in cache
                    # imgs_cache[timestamp] = util.impute_img(imgs_cache, timestamp)
                    # images[:,i,...] = imgs_cache[timestamp]

            # Package and yield valid data per available station
            for i, station_idx in enumerate(available_stations):
                imgs = tf.convert_to_tensor(images[station_idx, ...])
                clearsky_ghi_feat = tf.convert_to_tensor(clearsky_ghi_feats[i, :])
                azimuth_feat = tf.convert_to_tensor(azimuth_feats[i, :])
                time_of_day_feat = tf.convert_to_tensor(time_of_day_feats[i, :])
                biweekly_feat = tf.convert_to_tensor(biweekly_feats[i, :])

                # hardcode targets to zeros for inference
                target = tf.zeros((2 * len(target_time_offsets),))

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


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    """function to prepare & return prediction model.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
    """

    from train import load_model
    loaded_model = load_model(config)

    return loaded_model


def generate_all_predictions_custom(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any],
) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""

    data_loader = prepare_dataloader(dataframe, target_datetimes, target_stations, target_time_offsets, user_config)
    data_loader = data_loader.prefetch(tf.data.experimental.AUTOTUNE).batch(user_config['data']['imgs']['batch_size'])
    model = prepare_model(target_stations, target_time_offsets, user_config)

    predictions = model.predict(data_loader)
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()

    predictions_by_stations = np.zeros_like(predictions)
    assert predictions.shape[0] / len(list(target_stations.keys())) == len(target_datetimes), 'total # of preds/#stations does not match len(target datetimes)'

    # reshape timestamps x station -> station x timestamps
    for s in range(len(list(target_stations.keys()))):
        for t in range(len(target_datetimes)):
            predictions_by_stations[s * len(target_datetimes) + t] = predictions[t * len(list(target_stations.keys())) + s]

    return predictions_by_stations


def parse_gt_ghi_values(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station GHI values from the provided dataframe for the evaluation of predictions."""
    gt = []
    for station_idx, station_name in enumerate(target_stations):
        station_ghis = dataframe[station_name + "_GHI"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_ghis.index:
                    seq_vals.append(station_ghis.iloc[station_ghis.index.get_loc(index)])
                else:
                    seq_vals.append(float("nan"))
            gt.append(seq_vals)
    return np.concatenate(gt, axis=0)


def parse_nighttime_flags(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_DAYTIME"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(station_flags.iloc[station_flags.index.get_loc(index)] > 0)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        stats_output_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

        # This file should be run in inference only (setting globals/training to False automatically)
        user_config['globals']['training'] = False

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]

    target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]

    if "bypass_predictions_path" in admin_config and admin_config["bypass_predictions_path"]:
        # re-open cached output if possible (for 2nd pass eval)
        assert os.path.isfile(preds_output_path), f"invalid preds file path: {preds_output_path}"
        with open(preds_output_path, "r") as fd:
            predictions = fd.readlines()
        assert len(predictions) == len(target_datetimes) * len(target_stations), \
            "predicted ghi sequence count mistmatch wrt target datetimes x station count"
        assert len(predictions) % len(target_stations) == 0
        predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    else:
        predictions = generate_all_predictions_custom(target_stations, target_datetimes,
                                                      target_time_offsets, dataframe, user_config)
        with open(preds_output_path, "w") as fd:
            for pred in predictions:
                fd.write(",".join([f"{v:0.03f}" for v in pred.tolist()]) + "\n")

    if any([s + "_GHI" not in dataframe for s in target_stations]):
        print("station GHI measures missing from dataframe, skipping stats output")
        return

    assert not np.isnan(predictions).any(), "user predictions should NOT contain NaN values"
    predictions = predictions.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    gt = parse_gt_ghi_values(target_stations, target_datetimes, target_time_offsets, dataframe)
    gt = gt.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    day = parse_nighttime_flags(target_stations, target_datetimes, target_time_offsets, dataframe)
    day = day.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))

    squared_errors = np.square(predictions - gt)
    stations_rmse = np.sqrt(np.nanmean(squared_errors, axis=(1, 2)))
    for station_idx, (station_name, station_rmse) in enumerate(zip(target_stations, stations_rmse)):
        print(f"station '{station_name}' RMSE = {station_rmse:.02f}")
    horizons_rmse = np.sqrt(np.nanmean(squared_errors, axis=(0, 1)))
    for horizon_idx, (horizon_offset, horizon_rmse) in enumerate(zip(target_time_offsets, horizons_rmse)):
        print(f"horizon +{horizon_offset} RMSE = {horizon_rmse:.02f}")
    overall_rmse = np.sqrt(np.nanmean(squared_errors))
    print(f"overall RMSE = {overall_rmse:.02f}")

    if stats_output_path is not None:
        # we remove nans to avoid issues in the stats comparison script, and focus on daytime predictions
        squared_errors = squared_errors[~np.isnan(gt) & day]
        with open(stats_output_path, "w") as fd:
            for err in squared_errors.reshape(-1):
                fd.write(f"{err:0.03f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("-s", "--stats_output_path", type=str, default=None,
                        help="path where the prediction stats should be saved (for benchmarking)")
    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        stats_output_path=args.stats_output_path,
    )
