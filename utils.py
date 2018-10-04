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
        path = os.path.join(data_dir, 'WorkOrderKey_AircraftLookup.csv')
    else:
        raise Error("Invalid name, please use : wilmington, reno, littlerock, fleet, wo_lookup" )

    return path
