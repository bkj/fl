#!/usr/bin/env python

"""
    sunday-supervx.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from time import time

from rsub import *
from matplotlib import pyplot as plt

from supervx import SuperVX, SuperGraph

# --
# Args

E = 4
S = 0.5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-sparsity', type=float, default=S * E)
    parser.add_argument('--reg-edge', type=float, default=E)
    return parser.parse_args()

args = parse_args()

# --
# IO

edges = pd.read_csv('../sunday-edges.tsv', sep='\t', header=None)
nodes = pd.read_csv('../sunday-nodes.tsv', sep='\t')
graph_coords = np.load('../sunday-coords.npy')

# >>

hrs = map(str, range(10, 15))
# Subset edges to relevant timeslice
sel = (
    edges[0].apply(lambda x: x.split('-')[1] in hrs) & 
    edges[1].apply(lambda x: x.split('-')[1] in hrs)
)
edges = edges[sel]
# edges[0] = edges[0].apply(lambda x: x.split('-')[0])
# edges[1] = edges[1].apply(lambda x: x.split('-')[0])
edges = edges.drop_duplicates().reset_index(drop=True)

sel = nodes['index'].apply(lambda x: x.split('-')[1] in hrs)
nodes = nodes[sel].reset_index(drop=True)
graph_coords = graph_coords[np.array(sel)]

# nodes['index'] = nodes['index'].apply(lambda x: x.split('-')[0])
# nodes['row'] = np.arange(nodes.shape[0])
# graph_coords = graph_coords[np.array(nodes.groupby('index').row.min())]
# nodes = nodes.groupby('index')[['neg', 'pos']].sum().reset_index()
# nodes['d'] = nodes.neg - nodes.pos

# <<

edges = edges[edges[0] != edges[1]]
nodes['uid'] = np.arange(nodes.shape[0])
node_lookup = nodes[['index', 'uid']].set_index('index')

# Clean
feats = np.array(nodes.d)

# Dedupe edges
edges = np.hstack([
    np.array(node_lookup.loc[edges[0]]), 
    np.array(node_lookup.loc[edges[1]]),
])
sel = edges[:,0] >= edges[:,1]
edges[sel] = edges[sel,::-1]
edges = edges[pd.DataFrame(edges).drop_duplicates().index]


# --
# Run


if not os.path.exists('partition.npy'):
    supergraph = SuperGraph(edges, feats)
    np.save('partition', supergraph.partition)
else:
    supergraph = SuperGraph(edges, feats, np.load('partition.npy'))

svx = SuperVX(supergraph.supernodes, supergraph.superedges, reg_sparsity=args.reg_sparsity, reg_edge=args.reg_edge)
t = time()
svx.solve(UseADMM=True, Verbose=True)
time() - t
fitted = supergraph.unpack(svx.values)

# --
# Plot raw signal

# cmax = np.sqrt(np.abs(nodes.d)).max()
# hours = np.array(nodes['index'].apply(lambda x: x.split('-')[1])).astype('int')

# f, axs = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(10, 10))

# axs = np.hstack(axs)
# for h in np.unique(hours):
#     ax = axs[h]
#     sel = hours == h
#     c = np.sign(nodes.d[sel]) * np.sqrt(np.abs(nodes.d[sel]))
#     _ = ax.scatter(graph_coords[sel,1], graph_coords[sel,0], c=c, vmin=-cmax, vmax=cmax + 1, s=1, cmap='seismic')
#     _ = ax.set_title('hour=%d' % h)
#     _ = ax.axis('off')

# show_plot()


# --
# Plot results

cmax = np.sqrt(np.abs(fitted)).max()
hours = np.array(nodes['index'].apply(lambda x: x.split('-')[1])).astype('int')

f, axs = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(15, 15))

axs = np.hstack(axs)
_ = [ax.axis('off') for ax in axs]
for h in np.unique(hours):
    ax = axs[h]
    sel = hours == h
    c = np.sign(fitted[sel]) * np.sqrt(np.abs(fitted[sel]))
    _ = ax.scatter(graph_coords[sel,1], graph_coords[sel,0], c=c, vmin=-cmax, vmax=cmax + 1, s=1, cmap='seismic')
    _ = ax.set_title('hour=%d' % h)

show_plot()



