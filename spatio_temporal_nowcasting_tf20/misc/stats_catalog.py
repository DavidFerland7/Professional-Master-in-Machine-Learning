##################################################
# Compressing and decompressing numpy arrays
#import utils
import numpy as np
import h5netcdf
import json
import os
import h5py
import pandas as pd
import datetime

########### Stats catalog ############
OBS = None

# tidy data
df = pd.read_pickle("/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl")[:OBS]
#df = pd.read_csv("../catalog.csv", index_col=0, nrows=OBS)
#print(df.shape)
print("df.shape before concatenation of stations: {}".format(df.shape))
print(df.head())

id_vars = ["ncdf_path", "hdf5_8bit_path", "hdf5_8bit_offset", "hdf5_16bit_path", "hdf5_16bit_offset"]
df2 = pd.concat(
    [
        # df[id_vars + [col for col in df if col.startswith('BND_')]].rename(columns = lambda x: x.lstrip('BND').strip("_")).insert(0, 'station', 'BND'),
        # df[id_vars + [col for col in df if col.startswith('TBL_')]].rename(columns = lambda x: x.lstrip('TBL').strip("_")).insert(0, 'station', 'TBL')
        #
        df[id_vars + [col for col in df if col.startswith('BND_')]].rename(columns=lambda x: x.lstrip('BND').strip("_")).assign(station='BND'),
        df[id_vars + [col for col in df if col.startswith('TBL_')]].rename(columns=lambda x: x.lstrip('TBL').strip("_")).assign(station='TBL'),
        df[id_vars + [col for col in df if col.startswith('DRA_')]].rename(columns=lambda x: x.lstrip('DRA').strip("_")).assign(station='DRA'),
        df[id_vars + [col for col in df if col.startswith('FPK_')]].rename(columns=lambda x: x.lstrip('FPK').strip("_")).assign(station='FPK'),
        df[id_vars + [col for col in df if col.startswith('GWN_')]].rename(columns=lambda x: x.lstrip('GWN').strip("_")).assign(station='GWN'),
        df[id_vars + [col for col in df if col.startswith('PSU_')]].rename(columns=lambda x: x.lstrip('PSU').strip("_")).assign(station='PSU'),
        df[id_vars + [col for col in df if col.startswith('SFX_')]].rename(columns=lambda x: x.lstrip('SFX').strip("_")).assign(station='SFX')
    ]
)
print(df2.head())
print(df2.shape)
print("df.shape before daytime=1 filter: {}".format(df2.shape))

print(list(df2.columns))

# filter daytime =1 only
df2 = df2.loc[df2['DAYTIME'] == 1]
print("df.shape after daytime=1 filter: {}".format(df2.shape))

# filter ghi
df_ghi_nan = pd.isnull(df2.GHI)
print(df_ghi_nan.sum())
print("nan rows:\n{}".format(df2.loc[df_ghi_nan, ['CLEARSKY_GHI', "GHI"]]))

df2 = df2.loc[pd.notnull(df2.GHI)]
print("df.shape after daytime=1 and ghi not nan filters: {}".format(df2.shape))

# define month
df2['month'] = pd.to_datetime(df2.index.to_series()).dt.month
print(df2['month'].head())

# get average
avg = df2.groupby(['station', 'month'], as_index=False).agg({'CLEARSKY_GHI': ['mean', 'std'], 'GHI': ['mean', 'std']})
print(avg)
avg.columns = avg.columns.to_flat_index()
print(avg)

#avg.to_csv("../mean_ghi_by_station_and_month.csv", index=False)
