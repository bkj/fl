#!/usr/bin/env python

"""
    taxi-snap.py
"""

import sys
sys.path.append('/home/bjohnson/software/snap/snap')
sys.path.append('/home/bjohnson/software/snap/snapvx')

from time import time
import numpy as np
import pandas as pd
from snapvx import *

from rsub import *
from matplotlib import pyplot as plt

# --
# Helpers

def parse_dublin(bbox):
    bbox = map(lambda x: float(x.split('=')[1]), bbox.split(';'))
    return dict(zip(['west', 'south', 'east', 'north'], bbox))


# --
# IO

edges = pd.read_csv('./edges.tsv', sep='\t', header=None)
nodes = pd.read_csv('./nodes.tsv', sep='\t')
graph_coords = np.load('./coords.npy')

nodes['d']   = nodes.neg - nodes.pos
nodes['uid'] = np.arange(nodes.shape[0])

node_lookup = nodes[['index', 'uid']].set_index('index')

# _ = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(nodes.d) * np.sqrt(np.abs(nodes.d)), s=5, cmap='seismic')
# _ = plt.title('raw data')
# show_plot()


# --
# Run SNAPVX solver

# Format data
y = np.array(nodes.d)

# Dedupe edges
uedges = np.hstack([
    np.array(node_lookup.loc[edges[0]]), 
    np.array(node_lookup.loc[edges[1]]),
])
sel = uedges[:,0] > uedges[:,1]
uedges[sel] = uedges[sel,::-1]
uedges = uedges[pd.DataFrame(uedges).drop_duplicates().index]

# Run

gamma = 5
lambda_ = 10

snap_graph = PUNGraph.New()
_ = [snap_graph.AddNode(i) for i in range(len(y))]
for n1, n2 in np.array(uedges):
    _ = snap_graph.AddEdge(n1, n2)

gvx = TGraphVX(snap_graph)

X = []
for i in range(len(y)):
    x = Variable(1, name='x')
    gvx.SetNodeObjective(i, 0.5 * sum_squares(x - y[i]) + gamma * norm1(x))
    X.append(x)

def edge_objective(src, dst, data):
    return lambda_ * norm1(src['x'] - dst['x']), []

gvx.AddEdgeObjectives(edge_objective)

t = time.time()
gvx.Solve(UseADMM=True, Verbose=True)
time.time() - t

X = np.vstack([np.asarray(x.value).squeeze() for x in X])
X = X.squeeze()

ax = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(X) * np.sqrt(np.abs(X)), s=5, cmap='seismic')
show_plot()
