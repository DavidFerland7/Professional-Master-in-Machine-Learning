# TODO:
# 1) create dummy validation set + monitor training on validation set (for early stopping and log the metrics on valiation set)
# 2) connect training runs to weights and biaises
# 3) define the hyperparameter search workflow
# 4) create dummy test set + test prediction function to be evaluated by TA's evaluation function
# 5) add callback: checkpoints
# 6) Create a .sh file (bash file) to launch training script on compute note + try the whole training process on gpu with dummy small dataset


# TODO: make sure master branch is seeded  ref: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)

import argparse
import datetime
import json
import typing
import sys
import os
import re
import pandas as pd
import numpy as np
import importlib
import copy
from datetime import timedelta
import utils as util


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        region_size: typing.SupportsInt,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        stats_output_path: typing.Optional[typing.AnyStr] = None
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

    # Get all images if they don't exist in cache
    #region_size = user_config['data']['imgs']['region_size']
    cache_folder = f'./data/imgs_cropped/{region_size}/'
    if not os.path.isdir(cache_folder):
        util.get_all_imgs(target_stations, dataframe.dropna(subset=['ncdf_path']), region_size, user_config['data']['imgs']['n_channels'], user_config['globals']['prints'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("--region_size", type=int)
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("-s", "--stats_output_path", type=str, default=None,
                        help="path where the prediction stats should be saved (for benchmarking)")

    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        region_size=args.region_size,
        user_config_path=args.user_cfg_path,
        stats_output_path=args.stats_output_path,
    )
