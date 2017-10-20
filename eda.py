#!/usr/bin/env python

"""
    gtf-taxi.py
"""

from rsub import *
from matplotlib import pyplot as plt

import sys
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd

ox.config(log_file=True, log_console=True, use_cache=True)
pd.set_option('display.width', 200)

# --
# Helpers

def parse_dublin(bbox):
    bbox = map(lambda x: float(x.split('=')[1]), bbox.split(';'))
    return dict(zip(['west', 'south', 'east', 'north'], bbox))


def latlon2cartesian_vec(latlon, d=1):
    latlon = np.radians(latlon)
    return np.array([
        d * np.cos(latlon[:,0]) * np.cos(-latlon[:,1]), # x
        d * np.cos(latlon[:,0]) * np.sin(-latlon[:,1]), # y
        d * np.sin(latlon[:,0]),                        # z
    ]).T


def haversine_distance_vec(v1, v2, radius=6371):
    v1, v2 = np.radians(v1), np.radians(v2)
    
    latitude1, longitude1 = v1[:,0],v1[:,1]
    latitude2, longitude2 = v2[:,0],v2[:,1]
    
    dlongitude = longitude2 - longitude1 
    dlatitude = latitude2 - latitude1 
    a = np.sin(dlatitude/2) ** 2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(dlongitude/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = radius * c
    return km


def get_nearest_nodes(G, coords_q):
    # !! Should maybe be worried about underflow
    node_ids  = np.array(G.nodes())
    graph_coords = np.array([[data['y'], data['x']] for node, data in G.nodes(data=True)])
    
    graph_coords_c = latlon2cartesian(graph_coords)
    coords_q_c = latlon2cartesian(coords_q)
    
    print('computing sims', file=sys.stderr)
    sims = graph_coords_c.dot(coords_q_c.T)
    print('computing nns', file=sys.stderr)
    nns  = sims.argmax(axis=0)
    print('computing dists', file=sys.stderr)
    dists = haversine_distance_v(graph_coords[nns], coords_q)
    return node_ids[nns], dists * 1000


# --
# Load taxi

def load_taxi(path):
    return pd.read_csv(path, low_memory=False, usecols=[
        'pickup_datetime', 'dropoff_datetime', 
        'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude',
    ])


paths = [
    'yellow_tripdata_2011-05.csv',
    'yellow_tripdata_2011-06.csv',
]

df = pd.concat(map(load_taxi, paths))

# --
# Drop invalid locations

bbox = parse_dublin("westlimit=-74.087334; southlimit=40.641833; eastlimit=-73.858681; northlimit=40.884708")
df = df[(
    (df.pickup_latitude > bbox['south']) & 
    (df.pickup_latitude < bbox['north']) & 
    (df.pickup_longitude < bbox['east']) & 
    (df.pickup_longitude > bbox['west'])
)]

# --
# Clean dates

pickup_datetime = pd.to_datetime(df.pickup_datetime)
df['date']      = df.pickup_datetime.apply(lambda x: x.split(' ')[0])
df['dayofweek'] = pickup_datetime.dt.dayofweek
df['hour']      = pickup_datetime.dt.hour
df['target']    = df.date == '2011-06-26'

# --
# Subset times

sub = df[(df.dayofweek == 6) & (df.hour >= 12) & (df.hour <= 13)]
sub = sub.sort_values('pickup_datetime')

# sub.to_csv('./sub', sep='\t', index=False)
# sub = pd.read_csv('./sub', sep='\t')


# ==============================
# Graph stuff


# Smell check -- can we see the parade in raw data?

bbox = parse_dublin('westlimit=-74.023561; southlimit=40.698405; eastlimit=-73.968158; northlimit=40.754962')

pos = sub[sub.target]
neg = sub[~sub.target]

_ = plt.scatter(pos.pickup_longitude, pos.pickup_latitude, s=1, alpha=0.03)
_ = plt.xlim(bbox['west'], bbox['east'])
_ = plt.ylim(bbox['south'], bbox['north'])
show_plot()

sel = np.random.choice(neg.shape[0], pos.shape[0], replace=False)
_ = plt.scatter(neg.pickup_longitude.iloc[sel], neg.pickup_latitude.iloc[sel], s=1, alpha=0.03)
_ = plt.xlim(bbox['west'], bbox['east'])
_ = plt.ylim(bbox['south'], bbox['north'])
show_plot()


# --
# Load street graph

G = ox.graph_from_place('Manhattan Island, New York City, New York, USA', network_type='drive')
Gp = ox.project_graph(G)
fig, ax = ox.plot_graph(Gp)
show_plot()

# --
# Map each dropoff to nearest neighbor in graph

sub['pickup_nearest_node'], sub['pickup_nearest_dist'] =\
    get_nearest_nodes(G, np.array(sub[['pickup_latitude', 'pickup_longitude']]))

sub['dropoff_nearest_node'], sub['dropoff_nearest_dist'] =\
    get_nearest_nodes(G, np.array(sub[['dropoff_latitude', 'dropoff_longitude']]))


# Smell test
i = 300
ox.get_nearest_node(G, (sub.pickup_latitude.iloc[i], sub.pickup_longitude.iloc[i]), return_dist=True)
tuple(sub.iloc[i][['pickup_nearest_node', 'pickup_nearest_dist']])


# --
# Join data to graph

node_ids = np.array(G.nodes())
graph_coords = np.array([[data['y'], data['x']] for node, data in G.nodes(data=True)])

neg = sub[~sub.target]
pos = sub[sub.target]

pos_counts = pd.value_counts(np.concatenate([np.array(pos.pickup_nearest_node), np.array(pos.dropoff_nearest_node)]))
neg_counts = pd.value_counts(np.concatenate([np.array(neg.pickup_nearest_node), np.array(neg.dropoff_nearest_node)]))

counts = pd.DataFrame({
    "pos" : pos_counts.loc[node_ids],
    "neg" : neg_counts.loc[node_ids] / 9, # number of other days
}).fillna(0)
counts['d'] = counts.neg - counts.pos
counts = counts.loc[node_ids]

# Can we see parade in graph?
bbox = parse_dublin('westlimit=-74.013605; southlimit=40.715355; eastlimit=-73.954124; northlimit=40.772092')
ax = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=counts.d.loc[node_ids], s=5)
_ = plt.xlim(bbox['west'], bbox['east'])
_ = plt.ylim(bbox['south'], bbox['north'])
show_plot()

# --

np.save('edges', np.array(G.edges)[:,:2])
pd.DataFrame(np.array(G.edges)[:,:2]).to_csv('./edges.tsv', sep='\t', index=False, header=False)
counts[['neg', 'pos']].reset_index().to_csv('./nodes.tsv', sep='\t', index=False)
np.save('coords', graph_coords)
np.save('node_ids', node_ids)
# # >>

# ind = np.array(pd.Series(node_ids).isin(nodes))

# scounts = counts.loc[node_ids]

# # bbox = parse_dublin('westlimit=-74.013605; southlimit=40.715355; eastlimit=-73.954124; northlimit=40.772092')
# ax = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(scounts.d) * np.sqrt(np.abs(scounts.d * ind)), s=5, cmap='seismic')
# # _ = plt.xlim(bbox['west'], bbox['east'])
# # _ = plt.ylim(bbox['south'], bbox['north'])
# show_plot()





