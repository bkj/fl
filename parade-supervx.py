#!/usr/bin/env python

"""
    parade-supervx.py
"""

import argparse
import numpy as np
import pandas as pd
from time import time

from rsub import *
from matplotlib import pyplot as plt

from supervx import SuperVX, SuperGraph

np.random.seed(123)

# --
# Args

E = 16
S = 0.3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-sparsity', type=float, default=S * E)
    parser.add_argument('--reg-edge', type=float, default=E)
    return parser.parse_args()

args = parse_args()

# --
# IO

edges = pd.read_csv('../parade-edges.tsv', sep='\t', header=None)
nodes = pd.read_csv('../parade-nodes.tsv', sep='\t')
graph_coords = np.load('../parade-coords.npy')

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

supergraph = SuperGraph(edges, feats, partition=None)
svx = SuperVX(supergraph.supernodes, supergraph.superedges, reg_sparsity=args.reg_sparsity, reg_edge=args.reg_edge)
svx.solve(UseADMM=True, Verbose=True, EpsAbs=1e-6, EpsRel=1e-6, MaxIters=50)
fitted = supergraph.unpack(svx.values)

# --
# Plot results

# c = np.sign(fitted) * np.sqrt(np.abs(fitted))
# cmax = np.abs(c).max()
# _ = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=c, vmin=-cmax, vmax=cmax + 1, s=3, cmap='seismic')
# _ = plt.title("gamma=%d | lambda=%d" % (args.reg_sparsity, args.reg_edge))
# _ = plt.colorbar()
# show_plot()
