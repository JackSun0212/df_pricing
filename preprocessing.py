'''
Useful functions
'''
from utils import *

import os
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



days = set(np.arange(1, 32))
months = set(np.arange(1, 13))
years = set(np.arange(2009, 2019))

def transform_to_datetime(x):
    date = x.split(' ')[0]
    m,d,y = date.split('/')

    # consistency check
    if (int(m) in months) & (int(d) in days) & (int(y) in years):
        return pd.Timestamp(year=int(y), month=int(m), day=int(d), unit='D')
    else:
        print('Problem with date : {}'.format(x))
        return np.nan



def get_date_frame(data_name):

    path = get_data_path(data_name)
    df = pd.read_csv(path)

    time_columns= ['WORKORDERKEY', 'WOCATEGORY','ACTUALSTART','ACTUALFINISH']
    # Note : Each WO has only one'WOCATEGORY', thus, we can just drop duplicates
    df_date = df[time_columns].drop_duplicates()

    if df_date.groupby(['WORKORDERKEY']).count().values.max() !=1 :
        raise Error('Different info for the same work order key')

    df_date['start'] = df_date.ACTUALSTART.apply(transform_to_datetime)
    df_date['finish'] = df_date.ACTUALFINISH.apply(transform_to_datetime)
    df_date.drop(['ACTUALSTART', 'ACTUALFINISH'], axis=1, inplace=True)
    df_date.dropna(subset=['start', 'finish'], inplace=True) # nan are inconsistent values

    df_date['length_of_time'] = df_date.finish - df_date.start

    df_date.drop(df_date[df_date.length_of_time < '0 days 00:00:00'].index, inplace=True)

    df_date['length_of_time_in_days'] = df_date['length_of_time'].dt.days
    df_date.drop(['length_of_time'], axis=1, inplace=True)

    os.makedirs(os.path.join('prep_data', 'date_frame'), exist_ok=True)
    save_path = os.path.join('prep_data', 'date_frame', data_name)
    df_date.to_csv(save_path, index=False, date_format= '%Y/%m/%d' )

    print('Date_frame saved here : {}'.format(save_path))

def load_date_frame(data_name):

    path = os.path.join('prep_data', 'date_frame', data_name)
    if not os.path.exists(path):
        print("Could not find preprocessed data.")
        print("Starting preprocessing ...")
        get_date_frame(data_name)
        print("Loading data...")


    df = pd.read_csv(path, parse_dates=['start', 'finish'], infer_datetime_format=True)
    print("Data loaded")
    return df
