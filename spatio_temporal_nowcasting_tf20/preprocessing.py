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
#from pysolar.solar import get_azimuth, get_altitude, get_radiation_direct
#from pysolar.solar import get_azimuth
#import pytz

def preprocess_dataframe(df, config, admin_config):

    df.index = pd.to_datetime(df.index)
    path_vars = ["ncdf_path", "hdf5_8bit_path","hdf5_8bit_offset", "hdf5_16bit_path",	"hdf5_16bit_offset"]

    #1) Create df with paths to be returned
    df_paths = copy.deepcopy(df.loc[:,path_vars])
    if config['globals']['prints']: print("df_paths (orig) shape:{}".format(df_paths.shape))

    df_paths.loc[df_paths['ncdf_path'] == 'nan', 'ncdf_path'] = np.NaN
    df_paths = df_paths.dropna(subset=['ncdf_path'])
    if config['globals']['prints']: print("df_paths shape:{}".format(df_paths.shape))

    #2) get dense features and mask
    df_data = copy.deepcopy(df).drop(path_vars, axis=1)
    if config['globals']['prints']: print("df_data shape (orig) :{}".format(df_data.shape))

    ### 2b) merge azimuth
    azimuth_df = pd.read_csv('./data/azimuth/' + 'azimuth_20100101-20170102.csv', index_col=0)
    df_data = df_data.merge(azimuth_df, how='left', left_index=True, right_index=True)


    stations = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']
    # NOTE: leave commented 'time_utc_index' -> its in case we have to generate azimuth at runtime
    df_data = df_data.assign(
        **{
            **dict( (station + '_DAYTIME_' + str(offset), df_data[station + '_DAYTIME'].shift(offset)) for station, offset in list(itertools.product(stations, config['data']['features_from_df']['TARGET']))),
            **dict( (station + '_CLEARSKY_GHI_' + str(offset), df_data[station + '_CLEARSKY_GHI'].shift(offset)) for station, offset in list(itertools.product(stations, config['data']['features_from_df']['CLEARSKY_GHI']))),
            **dict( (station + '_AZIMUTH_' + str(offset), df_data[station + '_AZIMUTH'].shift(offset)) for station, offset in list(itertools.product(stations, config['data']['features_from_df']['AZIMUTH']))),
            **dict( (station + '_LAT', admin_config['stations'][station][0]) for station in stations),
            **dict( (station + '_LONG', admin_config['stations'][station][1]) for station in stations),
            **dict( (station + '_ALT', admin_config['stations'][station][2]) for station in stations),
            #**{'time_utc_index': df_data.index},
        }
    )

    #3) get targets
    if config['globals']['training']:
        df_data = df_data.assign(
            **{
                **dict( (station + '_GHI_' + str(offset), df_data[station + '_GHI'].shift(offset)) for station, offset in list(itertools.product(stations, config['data']['features_from_df']['TARGET']))),
            }
        )

    # * save head before dropping/manipulating rows to visualize data
    if config['globals']['training']:
        df_data.head(100).to_csv('misc/head_df_BEFORE.csv')

    #4) delete rows for some periods of time
    if config['globals']['training']:
        df_data = df_data.sort_index()

        s1 = datetime.datetime.strptime("2010-01-01", "%Y-%m-%d")
        e1 = datetime.datetime.strptime("2010-04-15", "%Y-%m-%d")

        s2 = datetime.datetime.strptime("2012-09-24", "%Y-%m-%d")
        e2 = datetime.datetime.strptime("2012-10-17", "%Y-%m-%d")

        s3 = datetime.datetime.strptime("2013-05-22", "%Y-%m-%d")
        e3 = datetime.datetime.strptime("2013-06-09", "%Y-%m-%d")

        df_data = df_data.loc[df_data.index.difference(df_data.index[df_data.index.slice_indexer(s1, e1)])]
        df_data = df_data.loc[df_data.index.difference(df_data.index[df_data.index.slice_indexer(s2, e2)])]
        df_data = df_data.loc[df_data.index.difference(df_data.index[df_data.index.slice_indexer(s3, e3)])]
        if config['globals']['prints']: print("df_data shape (delete trou alex):{}".format(df_data.shape))


    #5) Additional features (Time of day & bi-weekly encoding)
    day_of_time = pd.get_dummies(df_data.index.to_series().dt.hour.astype('str'), prefix='FEAT_TIME_OF_DAY')
    biweekly = pd.get_dummies((df_data.index.to_series().dt.week//2).astype('str'), prefix='FEAT_BIWEEKLY')
    df_data = pd.concat([df_data, day_of_time, biweekly], axis=1)

    #6) concat stations in rows
    df_data = pd.concat(
        [
            df_data[[col for col in df_data if col.startswith('BND_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('BND_')[-1]).assign(station = 'BND'),
            df_data[[col for col in df_data if col.startswith('TBL_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('TBL_')[-1]).assign(station = 'TBL'),
            df_data[[col for col in df_data if col.startswith('DRA_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('DRA_')[-1]).assign(station = 'DRA'),
            df_data[[col for col in df_data if col.startswith('FPK_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('FPK_')[-1]).assign(station = 'FPK'),
            df_data[[col for col in df_data if col.startswith('GWN_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('GWN_')[-1]).assign(station = 'GWN'),
            df_data[[col for col in df_data if col.startswith('PSU_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('PSU_')[-1]).assign(station = 'PSU'),
            df_data[[col for col in df_data if col.startswith('SXF_') or col.startswith('FEAT_')]].rename(columns = lambda x: x.split('SXF_')[-1]).assign(station = 'SXF'),
        ]
    )
    df_data.drop(columns=['CLEARSKY_GHI', 'AZIMUTH', 'GHI', 'DAYTIME'], axis=1, inplace=True)

    if config['globals']['prints']: print("df_data shape (concat station in rows):{}".format(df_data.shape))

    #7) delete rows with daytime_t0=0 OR all targets equals to NaN
    if config['globals']['training']:
        df_data = df_data.loc[df_data['DAYTIME_0'] ==1]
        if config['globals']['prints']: print("df_data shape (delete rows (daytime_0=0)):{}".format(df_data.shape))

        df_data = df_data.dropna(subset=[f'GHI_{target}' for target in config['data']['features_from_df']['TARGET']], how='all')
        if config['globals']['prints']: print("df_data shape (delete rows all nan)):{}".format(df_data.shape))

        if config['globals']['prints']: print("\ncount nan for each column (BEFORE NaN REPLACEMENT):\n{}".format(df_data.shape[0]-df_data.count()))

    #8) create mask
    for target in config['data']['features_from_df']['TARGET']:
        if config['globals']['training']:
            mask = ~((df_data[f'GHI_{target}'].isna().reset_index(drop=True))  | (df_data[f'DAYTIME_{target}'].isna().reset_index(drop=True))  | ( (df_data[f'DAYTIME_{target}']==0).reset_index(drop=True)))
            mask_name = f'MASK_{target}'
            mask.index = df_data.index
            df_data[mask_name] = mask.astype(int)
        if config['globals']['prints']: print("\ndescribe mask:\n{}".format(df_data.shape[0]-df_data.loc[:,[f'MASK_{target}' for target in config['data']['features_from_df']['TARGET']]].sum()))
        else:
            #TODO: validate we have to create the mask anyways at 1 for inference
            df_data[f'MASK_{target}'] = 1

    #9) delete rows with all masks==0 (can happen if ghi_0 is NaN AND other t+x masks are NaN for other reasons than all of their ghi being NaN (e.g. at least of them have non NaN ghi, but have daytime==0)

    if config['globals']['training']:
        all_mask_0 = (df_data[[f'MASK_{target}' for target in config['data']['features_from_df']['TARGET']]] == 0).all(axis=1)
        df_data = df_data[~all_mask_0]
        if config['globals']['prints']: print("\ndf_data shape (delete rows with all masks==0)):{}".format(df_data.shape))

        #10) stats
        if config['globals']['prints']: print("\nstats:")
        if config['globals']['prints']: print("0) all masks==0: {}".format(all_mask_0.sum()))
        if config['globals']['prints']: print("1) daytime_t_0==0: {}".format((df_data['DAYTIME_0']==0).sum()))
        if config['globals']['prints']: print("2) mask_t_0==0 [->2379 obs have ghi_t_0==NaN on row with not all ghi == NaN AND also 828 have all masks==0]: {}".format((df_data['MASK_0']==0).sum()))
        if config['globals']['prints']: print("3) mask patterns frequencies:\n{}".format(df_data.groupby([f'MASK_{target}' for target in config['data']['features_from_df']['TARGET']]).size().reset_index().rename(columns={0:'freq'})))
        # print("3g) mask pattern==[0 1 1 0]: {}".format(((df_data['MASK']==0) & (df_data['MASK_4']==1) & (df_data['MASK_12']==1) & (df_data['MASK_24']==0)).sum()))


    #11) replace nan
    df_data.loc[
        :,[f'GHI_{target}' for target in config['data']['features_from_df']['TARGET'] if config['globals']['training']] +
          [f'DAYTIME_{target}' for target in config['data']['features_from_df']['TARGET']] +
          [col for col in df_data if col.startswith('CLEARSKY_GHI_')] +
          [col for col in df_data if col.startswith('AZIMUTH_')]
    ]=\
    df_data.loc[
        :,[f'GHI_{target}' for target in config['data']['features_from_df']['TARGET'] if config['globals']['training']] +
          [f'DAYTIME_{target}' for target in config['data']['features_from_df']['TARGET']] +
          [col for col in df_data if col.startswith('CLEARSKY_GHI_')] +
          [col for col in df_data if col.startswith('AZIMUTH_')]
    ].fillna(value=0)

    # * save head after dropping/manipulating rows to visualize data
    if config['globals']['training']:
        if config['globals']['prints']: print("\ncount nan for each column (AFTER NaN REPLACEMENT):\n{}".format(df_data.shape[0]-df_data.count()))
        df_data.head(100).to_csv('misc/head_df_AFTER.csv')
    return df_paths, df_data


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
    df = pd.read_pickle('./data/catalog.20100101-20160101.pkl')
    df = df[:1000]
    df_paths, df_data = preprocess_dataframe(df, user_config, admin_config)

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

