#!/usr/bin/env python

"""
    subset-data.py
"""

from __future__ import print_function

from rsub import *
from matplotlib import pyplot as plt

import sys
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import KDTree

ox.config(log_file=True, log_console=True, use_cache=True)
pd.set_option('display.width', 200)

# --
# Helpers

def parse_dublin(bbox_str):
    bbox = map(lambda x: float(x.split('=')[1]), bbox_str.split(';'))
    return dict(zip(['west', 'south', 'east', 'north'], bbox))

def filter_dublin(bbox_str, lat, lon):
    bbox = parse_dublin(bbox_str)
    return (
        (lat > bbox['south']) & 
        (lat < bbox['north']) & 
        (lon < bbox['east']) & 
        (lon > bbox['west'])
    )

# --
# Load taxi

def load_taxi(path):
    return pd.read_csv(path, usecols=[
        'pickup_datetime', 'dropoff_datetime', 
        'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude',
    ], dtype={
        'pickup_datetime' : str,
        'dropoff_datetime' : str,
        'pickup_longitude' : float,
        'pickup_latitude' : float,
        'dropoff_longitude' : float,
        'dropoff_latitude' : float,
    })


paths = [
    '../yellow_tripdata_2011-05.csv',
    '../yellow_tripdata_2011-06.csv',
]

df = pd.concat(map(load_taxi, paths))

# --
# Drop invalid locations

# NYC bbox
bbox_str = "westlimit=-74.087334; southlimit=40.641833; eastlimit=-73.858681; northlimit=40.884708"
df = df[filter_dublin(bbox_str, df.dropoff_latitude, df.dropoff_longitude)]
df = df[filter_dublin(bbox_str, df.pickup_latitude, df.pickup_longitude)]

# --
# Clean dates

pickup_datetime = pd.to_datetime(df.pickup_datetime)
df['date']      = df.pickup_datetime.apply(lambda x: x.split(' ')[0])
df['dayofweek'] = pickup_datetime.dt.dayofweek
df['hour']      = pickup_datetime.dt.hour
df['target']    = df.date == '2011-06-26'

# --
# Subset times

parade = df[(df.dayofweek == 6) & (df.hour >= 12) & (df.hour <= 13)]
sunday = df[(df.dayofweek == 6)]

parade.to_csv('../parade.tsv', sep='\t', index=False)
sunday.to_csv('../sunday.tsv', sep='\t', index=False)
