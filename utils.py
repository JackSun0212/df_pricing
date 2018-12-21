'''
Useful functions
'''

import os
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_data_path(data_name):
    main_dir = os.path.realpath('')[:-11]
    data_dir = os.path.join(main_dir, 'High Volume Discounts', 'Data')
    if data_name == 'wilmington':
        path = os.path.join(data_dir, 'WilmingtonWorkOrderData.csv')
    elif data_name == 'reno':
        path = os.path.join(data_dir, 'RenoWorkOrderData.csv')
    elif data_name == 'littlerock':
        path = os.path.join(data_dir, 'LittleRockWorkOrderData.csv')
    elif data_name == 'fleet':
        path = os.path.join(data_dir, 'Fleet size.csv')
    elif data_name == 'wo_lookup':
        path = os.path.join(data_dir, 'WorkOrderKey_AircraftLookup.xlsx')
    else:
        raise Error("Invalid name, please use : wilmington, reno, littlerock, fleet, wo_lookup" )
    return path

def load_(data_name):
    path = get_data_path(data_name)
    if data_name == 'fleet':
        return pd.read_csv(path, encoding='latin-1')
    elif data_name == 'wo_lookup':
        return pd.read_excel(path)



days = set(np.arange(1, 32))
months = set(np.arange(1, 13))
years = set(np.arange(2009, 2019))

def transform_to_datetime(x):
    if type(x)==float:
        if np.isnan(x):
            return np.nan
    date = x.split(' ')[0]
    m,d,y = date.split('/')

    # consistency check
    if (int(m) in months) & (int(d) in days) & (int(y) in years):
        return pd.Timestamp(year=int(y), month=int(m), day=int(d), unit='D')
    else:
        print('Problem with date : {}'.format(x))
        return np.nan


def encode_categories(x):
    if x=='C-1 (Maintenance, Modifications)':
        return(1)
    elif x=='C-2 (Maintenance only)':
        return(2)
    elif x=='B-3 (Inspections, Modifications, Maintenance)':
        return(3)
    elif x=='B-4 (Inspections, Maintenance)':
        return(4)


def encode_aircraft_model(x):
    try:
        if x[:2]=='F2': #'F20','F200','F2000','F2000DX','F2000EX','F2000EXEASy','F2000LX','F2000LXS','F2000S'
            return(4)
        elif x[:2]=='F7': #'F7X'
            return(3)
        elif x[:2]=='F9': #'F900','F900A','F900B','F900C','F900DX', 'F900EX', 'F900LX', 'F900EXEASy'
            return(2)
        else:
            return(1)
    except: # missing values
        return(0)
