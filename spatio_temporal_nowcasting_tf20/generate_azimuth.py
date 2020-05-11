import datetime
import pandas as pd
from datetime import timedelta
import tensorflow as tf
import numpy as np
import typing
import logging
import argparse
import json
import os
import sys
import copy
import itertools
import time
from pysolar.solar import get_azimuth
import pytz

def preprocess_dataframe(config, admin_config):

    index = pd.date_range(start=datetime.datetime(2010,1,1,0,0), end=datetime.datetime(2017,1,2,0,0), freq='15T', name='iso-datetime')
    index.freq = None
    print('index: ', index)
    df_data = pd.DataFrame(None, index=index)
    stations = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']
    df_data = df_data.assign(
        **{
            **dict( (station + '_LAT', admin_config['stations'][station][0]) for station in stations),
            **dict( (station + '_LONG', admin_config['stations'][station][1]) for station in stations),
            **dict( (station + '_ALT', admin_config['stations'][station][2]) for station in stations),
            **{'time_utc_index': df_data.index},
        }
    )

    ### FOR TEST
    #df_data = df_data[:100]

    start_time_ = time.perf_counter()
    ## GET AZMIMUTH
    df_data = df_data.assign(
        **{
            **dict(
                (
                    station + '_AZIMUTH',
                    df_data.apply(lambda x: get_azimuth(x[station + '_LAT'], x[station + '_LONG'], x['time_utc_index'].to_pydatetime().replace(tzinfo=pytz.UTC)), axis=1)
                ) for station in stations
            )
        }
    )
    print("\ntotal time to get ghi for {} obs :{}".format(df_data.shape[0], time.perf_counter() - start_time_))

    print("df_data.columns:, ",df_data.columns)
    path = './data/azimuth/'
    df_data.to_csv(path + "azimuth_20100101-20170102.csv", columns = [col for col in df_data.columns if col.endswith('AZIMUTH')])

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


    ####################### adhoc tests on preprocessing_backup.py file ###########################
    preprocess_dataframe(user_config, admin_config)

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

