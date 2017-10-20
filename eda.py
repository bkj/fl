#!/usr/bin/env python

"""
    format-parade.py
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
    
    graph_coords_c = latlon2cartesian_vec(graph_coords)
    coords_q_c = latlon2cartesian_vec(coords_q)
    
    kd_tree = KDTree(graph_coords_c, 2, metric='euclidean')
    
    dists, nns = kd_tree.query(coords_q_c)
    dists = dists.squeeze()
    
    return node_ids[nns], np.arcsin(dists.clip(max=1)) * 6371 * 1000

# --

df = pd.read_csv('../parade.tsv', sep='\t')

# Smell check -- can we see the parade in raw data?
bbox = parse_dublin('westlimit=-74.023561; southlimit=40.698405; eastlimit=-73.968158; northlimit=40.754962')

_ = plt.scatter(df[df.target].pickup_longitude, df[df.target].pickup_latitude, s=1, alpha=0.03, c='blue')
_ = plt.xlim(bbox['west'], bbox['east'])
_ = plt.ylim(bbox['south'], bbox['north'])
show_plot()

tmp = df[~df.target]
sel = np.random.choice(tmp.shape[0], df.target.sum(), replace=False)
_ = plt.scatter(tmp.pickup_longitude.iloc[sel], tmp.pickup_latitude.iloc[sel], s=1, alpha=0.03, c='red')
_ = plt.xlim(bbox['west'], bbox['east'])
_ = plt.ylim(bbox['south'], bbox['north'])
show_plot()


# --
# Load street graph

G = ox.graph_from_place('Manhattan Island, New York City, New York, USA', network_type='drive')
graph_coords = np.array([[data['y'], data['x']] for node, data in G.nodes(data=True)])

# --
# Map each dropoff to nearest neighbor in graph

df['pickup_nearest_node'], df['pickup_nearest_dist'] =\
    get_nearest_nodes(G, np.array(df[['pickup_latitude', 'pickup_longitude']]))

df['dropoff_nearest_node'], df['dropoff_nearest_dist'] =\
    get_nearest_nodes(G, np.array(df[['dropoff_latitude', 'dropoff_longitude']]))


# Smell test
i = np.random.choice(df.shape[0])
a = ox.get_nearest_node(G, (df.pickup_latitude.iloc[i], df.pickup_longitude.iloc[i]), return_dist=True)
b = tuple(df.iloc[i][['pickup_nearest_node', 'pickup_nearest_dist']])
assert a[0] == b[0]
assert np.abs(a[1] - b[1]) < 1

# --
# Join data to graph

node_ids = np.array(G.nodes())

neg = df[~df.target]
pos = df[df.target]

pos_counts = pd.value_counts(np.concatenate([np.array(pos.pickup_nearest_node), np.array(pos.dropoff_nearest_node)]))
neg_counts = pd.value_counts(np.concatenate([np.array(neg.pickup_nearest_node), np.array(neg.dropoff_nearest_node)]))

nneg = neg.date.unique().shape[0]
counts = pd.DataFrame({
    "pos" : pos_counts.loc[node_ids],
    "neg" : neg_counts.loc[node_ids] / nneg, # number of other days
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

pd.DataFrame(np.array(G.edges)[:,:2]).to_csv('../parade-edges.tsv', sep='\t', index=False, header=False)
counts.reset_index().to_csv('../parade-nodes.tsv', sep='\t', index=False)
np.save('../parade-coords', graph_coords)

# # >>

# ind = np.array(pd.Series(node_ids).isin(nodes))

# scounts = counts.loc[node_ids]

# # bbox = parse_dublin('westlimit=-74.013605; southlimit=40.715355; eastlimit=-73.954124; northlimit=40.772092')
# ax = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(scounts.d) * np.sqrt(np.abs(scounts.d * ind)), s=5, cmap='seismic')
# # _ = plt.xlim(bbox['west'], bbox['east'])
# # _ = plt.ylim(bbox['south'], bbox['north'])
# show_plot()





