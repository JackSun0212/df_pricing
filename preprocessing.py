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

    #for WA only keep 2 features (to predict) and their estimates(as baselines)
    df_wa = df_group.loc[df_group.is_WA==True,
                    ['TOTALSQUAWKCOST', 'TOTALSQUAWKREVENUE',
                     'TOTALSQUAWKESTIMATEDCOST','TOTALSQUAWKESTIMATEDREVENUE']].add_prefix('WA_')
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

    print('Imputing missing values with zeros')
    n_nan = df_full.isnull().values.sum()
    n_val = df_full.shape[0]*df_full.shape[1]
    print('{} missing values imputted ({}%)'.format(df_full.isnull().values.sum(), np.round(100*n_nan/n_val, 2)))

    df_full.fillna(0,inplace=True)

    # adding aircraft info:
    lookup = load_('wo_lookup')
    lookup = lookup.drop(['ORG','AIRCRAFTSERIALNUMBER'], axis=1).set_index('WORKORDERKEY')
    df_lu = df_full.join(lookup)

    # removing rows with negative costs
    print('Removing rows with negative costs')
    col_neg_val = []
    test_negative_values = (df_lu<0).sum()
    columns_with_neg_val = test_negative_values.iloc[test_negative_values.nonzero()].index
    for col in columns_with_neg_val:
        if 'cost' in col.lower():
            col_neg_val.append(col)

    df_final = df_lu.iloc[((df_lu[col_neg_val]<0).sum(axis=1)==0).nonzero()]

    print('Encoding categorical features')
    df_final['wo_category'] = df_final.WOCATEGORY.apply(encode_categories)
    df_final['aircraft_model'] = df_final.AIRCRAFTMODELNUMBER.apply(encode_aircraft_model)
    df_final.drop(['WOCATEGORY', 'AIRCRAFTMODELNUMBER'], axis=1, inplace=True)

    os.makedirs(os.path.join('prep_data', 'model_frame'), exist_ok=True)
    save_path = os.path.join('prep_data', 'model_frame', data_name)
    df_final.to_csv(save_path)

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


def get_iqr_bound(x, coef=2):
    q1,q3 = x.quantile([.25,.75])
    iqr = q3-q1
    return q3+coef*iqr


def get_outliers(data):
    outliers=set()

    #drop when revenue or expected revenue is 0
    outliers.update(data[data.TOTALSQUAWKREVENUE==0].index)
    outliers.update(data[data.TOTALSQUAWKESTIMATEDREVENUE==0].index)

    #drop when cost or estimate cost is very low
    min_= data.TOTALSQUAWKCOST.quantile(0.03)
    outliers.update(data[data.TOTALSQUAWKCOST<min_].index)
    outliers.update(data[data.TOTALSQUAWKESTIMATEDCOST<min_].index)

    # With 2IQR rule, drop high values for:
    # cost, expected cost and work arising cost
    # revenue, expected revenue and work arising revenue
    max_ = get_iqr_bound(data.TOTALSQUAWKCOST)
    outliers.update(data[data.TOTALSQUAWKCOST>max_].index)
    max_ = get_iqr_bound(data.TOTALSQUAWKESTIMATEDCOST)
    outliers.update(data[data.TOTALSQUAWKESTIMATEDCOST>max_].index)
    max_ = get_iqr_bound(data.WA_TOTALSQUAWKCOST)
    outliers.update(data[data.WA_TOTALSQUAWKCOST>max_].index)
    max_ = get_iqr_bound(data.TOTALSQUAWKREVENUE)
    outliers.update(data[data.TOTALSQUAWKREVENUE>max_].index)
    max_ = get_iqr_bound(data.TOTALSQUAWKESTIMATEDREVENUE)
    outliers.update(data[data.TOTALSQUAWKESTIMATEDREVENUE>max_].index)
    max_ = get_iqr_bound(data.WA_TOTALSQUAWKREVENUE)
    outliers.update(data[data.WA_TOTALSQUAWKREVENUE>max_].index)

    print('Outliers represents {}% of the dataset'.format(np.round(100*len(outliers)/len(data),1)))
    return outliers


def remove_outliers(df):
    df_1 = df[df.wo_category==1].drop('wo_category', axis=1)
    df_2 = df[df.wo_category==2].drop('wo_category', axis=1)
    df_3 = df[df.wo_category==3].drop('wo_category', axis=1)
    df_4 = df[df.wo_category==4].drop('wo_category', axis=1)

    clean_1 = df_1.drop(get_outliers(df_1))
    clean_2 = df_2.drop(get_outliers(df_2))
    clean_3 = df_3.drop(get_outliers(df_3))
    clean_4 = df_4.drop(get_outliers(df_4))

    return [clean_1,clean_2, clean_3, clean_4]
