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


def get_frame_for_prediction(data_name):

    path = get_data_path(data_name)
    df = pd.read_csv(path)

    columns_of_interest = ['WORKORDERKEY', 'WOCATEGORY', 'ORIGINATINGSQUAWK', 'ITEMNUMBER',
                           # squawk
                           'TOTALSQUAWKCOST', 'TOTALSQUAWKREVENUE', # these 2 are the targets variables
                           'TOTALSQUAWKESTIMATEDCOST','TOTALSQUAWKESTIMATEDREVENUE',
                           # labor
                           'TOTALLABORESTIMATEDCOST', 'TOTALLABORESTIMATEDREVENUE',
                           # parts
                           'TOTALPARTSESTIMATEDCOST', 'TOTALPARTSESTIMATEDREVENUE',
                            ]

    df = df[columns_of_interest]
    df = df.loc[df.WOCATEGORY.isin(['C-2 (Maintenance only)',
                                    'B-3 (Inspections, Modifications, Maintenance)',
                                    'C-1 (Maintenance, Modifications)',
                                    'B-4 (Inspections, Maintenance)'])]

    df['is_WA']= (df.ORIGINATINGSQUAWK!=0.0)


    df_group = df.groupby(['WORKORDERKEY', 'WOCATEGORY', 'ITEMNUMBER', 'is_WA']).sum().reset_index(level=(1,2,3))\
                                                                                .drop('ORIGINATINGSQUAWK', axis=1)

    #for WA only keep 2 features (to predict)
    df_wa = df_group.loc[df_group.is_WA==True, ['TOTALSQUAWKCOST', 'TOTALSQUAWKREVENUE']].add_prefix('WA_')
    df_wa = df_wa.reset_index(level=0).groupby('WORKORDERKEY').sum()


    #for original squawk, same : sum over all items
    df_origin = df_group.loc[df_group.is_WA==False].drop('is_WA', axis=1)

    df_origin_tot = df_origin.drop(['WOCATEGORY','ITEMNUMBER'], axis=1)
    df_origin_tot = df_origin_tot.reset_index(level=0).groupby('WORKORDERKEY').sum()


    frame_to_concat = [df_group['WOCATEGORY'].reset_index().drop_duplicates().set_index('WORKORDERKEY')]
    frame_to_concat.append(df_origin_tot)
    frame_to_concat.append(df_wa)


    # for original squawk, we collect the info for each item:
    for item in df_origin.ITEMNUMBER.unique():
        frame_to_concat.append(df_origin.loc[df_origin.ITEMNUMBER==item] \
                                        .drop(['WOCATEGORY','ITEMNUMBER','TOTALSQUAWKCOST','TOTALSQUAWKREVENUE'],axis=1) \
                                        .add_prefix('I{}_'.format(item)))


    df_full = pd.concat(frame_to_concat, axis=1)

    print('Final shape : {}'.format(df_full.shape))
    print('Imputing missing values with zeros')
    n_nan = df_full.isnull().values.sum()
    n_val = df_full.shape[0]*df_full.shape[1]
    print('{} missing values imputted ({}%)'.format(df_full.isnull().values.sum(), np.round(100*n_nan/n_val, 2)))

    df_full.fillna(0,inplace=True)

    os.makedirs(os.path.join('prep_data', 'model_frame'), exist_ok=True)
    save_path = os.path.join('prep_data', 'model_frame', data_name)
    df_full.to_csv(save_path)

    print('Model_frame saved here : {}'.format(save_path))


def load_frame_for_prediction(data_name):

    path = os.path.join('prep_data', 'model_frame', data_name)
    if not os.path.exists(path):
        print("Could not find preprocessed data.")
        print("Starting preprocessing ...")
        get_frame_for_prediction(data_name)
        print("Loading data...")


    df = pd.read_csv(path)
    print("Data loaded")
    return df.set_index('WORKORDERKEY')
