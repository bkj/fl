#!/usr/bin/env python

"""
    format-sunday.py
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

df = pd.read_csv('../sunday.tsv', sep='\t')

# --
# Load street graph

G = ox.graph_from_place('Manhattan Island, New York City, New York, USA', network_type='drive')
graph_coords = np.array([[data['y'], data['x']] for node, data in G.nodes(data=True)])
node_ids = np.array(G.nodes())

# --
# Map each dropoff to nearest neighbor in graph

df['pickup_nearest_node'], df['pickup_nearest_dist'] =\
    get_nearest_nodes(G, np.array(df[['pickup_latitude', 'pickup_longitude']]))

df['dropoff_nearest_node'], df['dropoff_nearest_dist'] =\
    get_nearest_nodes(G, np.array(df[['dropoff_latitude', 'dropoff_longitude']]))

# --
# Join data to graph

# Drop points too far away
df = df[(df.pickup_nearest_dist < 250) & (df.dropoff_nearest_dist < 250)]

# Add id for spacetime graph node
df['pickup_stnode'] = df.pickup_nearest_node.astype(str) + "-" + df.hour.astype(str)
df['dropoff_stnode'] = df.dropoff_nearest_node.astype(str) + "-" + df.hour.astype(str)

neg = df[~df.target]
pos = df[df.target]

nneg = neg.date.unique().shape[0]
counts = pd.DataFrame({
    "pos_pickup"  : pos.groupby('pickup_stnode').target.count(),
    "pos_dropoff" : pos.groupby('dropoff_stnode').target.count(),
    "neg_pickup"  : neg.groupby('pickup_stnode').target.count() / nneg,
    "neg_dropoff" : neg.groupby('dropoff_stnode').target.count() / nneg,
}).fillna(0)

counts['pos'] = counts.pos_pickup + counts.pos_dropoff
counts['neg'] = counts.neg_pickup + counts.neg_dropoff
counts['d']   = counts.neg - counts.pos

# --
# Make graph -- geo graph replicated in time

edges = pd.DataFrame(np.array(G.edges)[:,:2])

all_edges = []
for i in range(24):
    all_edges.append(edges.astype(str) + '-%d' % i)

unodes = pd.Series(np.unique(edges)).astype(str)
for i in range(1, 24):
    all_edges.append(pd.DataFrame({
        0 : (unodes + '-%d' % i),
        1 : (unodes + '-%d' % (i - 1)),
    }))

all_edges = pd.concat(all_edges).sort_values(0)

# Add observations for all nodes in network
uedges = set(np.unique(all_edges))
counts = counts.reindex(uedges).fillna(0)

all_edges.to_csv('../sunday-edges.tsv', sep='\t', index=False, header=False)
counts.reset_index().to_csv('../sunday-nodes.tsv', sep='\t', index=False)

id_lookup = pd.Series(np.arange(node_ids.shape[0]), index=node_ids)
tmp = pd.Series(counts.index).apply(lambda x: x.split('-')[0])
tmp = np.array(id_lookup.loc[np.array(tmp).astype(int)])
tmp = graph_coords[tmp]
np.save('../sunday-coords', tmp)

