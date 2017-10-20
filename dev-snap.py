#!/usr/bin/env python

"""
    taxi-snap.py
"""

import sys
# sys.path.append('/home/bjohnson/software/snap/snap')
# sys.path.append('/home/bjohnson/software/snap/snapvx')

from time import time
import numpy as np
import pandas as pd
from snapvx import *
import snap

from rsub import *
from matplotlib import pyplot as plt

# --
# Helpers

def parse_dublin(bbox):
    bbox = map(lambda x: float(x.split('=')[1]), bbox.split(';'))
    return dict(zip(['west', 'south', 'east', 'north'], bbox))


# --
# IO

edges = pd.read_csv('../parade-edges.tsv', sep='\t', header=None)
nodes = pd.read_csv('../parade-nodes.tsv', sep='\t')
graph_coords = np.load('../parade-coords.npy')

edges = edges[edges[0] != edges[1]]

nodes['uid'] = np.arange(nodes.shape[0])
node_lookup = nodes[['index', 'uid']].set_index('index')

_ = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(nodes.d) * np.sqrt(np.abs(nodes.d)), s=5, cmap='seismic')
_ = plt.title('raw data')
show_plot()

# --
# Prep

gamma = 10
lambda_ = 10

# Format data
y = np.array(nodes.d)

# Dedupe edges
uedges = np.hstack([
    np.array(node_lookup.loc[edges[0]]), 
    np.array(node_lookup.loc[edges[1]]),
])
sel = uedges[:,0] >= uedges[:,1]
uedges[sel] = uedges[sel,::-1]
uedges = uedges[pd.DataFrame(uedges).drop_duplicates().index]

# --
# Fast way, straight CVXPY

XX = Variable(y.shape[0])

mse = 0.5 * sum_squares(XX - y)
spr = gamma * norm1(XX)
edg = lambda_ * norm1(XX[uedges[:,0]] - XX[uedges[:,1]])

prob = Problem(Minimize(mse + spr + edg))
t = time.time()
prob.solve()
time.time() - t

XX = np.asarray(XX.value).squeeze()
c = np.sign(XX) * np.sqrt(np.abs(XX))
_ = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=c, vmin=-np.abs(c).max(), vmax=np.abs(c).max() + 1, s=5, cmap='seismic')
_ = plt.title("gamma=%d | lambda=%d" % (gamma, lambda_))
_ = plt.colorbar()
show_plot()

# --
# Manual clustering

import networkx as nx
from community import community_louvain

# Partition graph
G = nx.from_edgelist(uedges)
partition = community_louvain.best_partition(G)
partition = np.array([p[1] for p in sorted(partition.items(), key=lambda x: x[0])])

# Create supergraph
superedges = set([])
sgraph_info = {}
g_nodes = np.array(G.nodes())
for p in np.unique(partition):
    psel = partition == p
    
    partition_nodes = g_nodes[psel]
    
    # Get interior and exterior edges
    int_edges = []
    ext_edges = []
    for node in partition_nodes:
        for neib in G.neighbors(node):
            if neib in partition_nodes:
                if node < neib:
                    int_edges.append((node, neib))
            else:
                if p < partition[neib]:
                    ext_edges.append((node, neib))
                    superedges.add((p, partition[neib]))
    
    unodes = np.unique(int_edges)
    node_map = dict(zip(unodes, np.arange(unodes.shape[0])))
    
    sgraph_info[p] = {
        "feats"     : y[psel],
        "int_edges" : np.array(int_edges),
        "ext_edges" : np.array(ext_edges),
        "node_map"  : node_map,
    }

# Construct supergraph
sgraph = PUNGraph.New()
_ = [sgraph.AddNode(sid) for sid in sgraph_info.keys()]
_ = [sgraph.AddEdge(e1, e2) for e1, e2 in superedges]

gvx = TGraphVX(sgraph)

# Node objectives
V = {}
for sid, snode in sgraph_info.items():
    v = Variable(snode['feats'].shape[0], name='x')
    
    mse = 0.5 * sum_squares(v - snode['feats'])
    spr = gamma * norm1(v)
    internal_edges = lambda_ * norm1(
        v[np.array([snode['node_map'][i] for i in snode['int_edges'][:,0]])] - 
        v[np.array([snode['node_map'][i] for i in snode['int_edges'][:,1]])]
    )
    
    gvx.SetNodeObjective(sid, mse + spr + internal_edges)
    V[sid] = v


# Edge objectives
for sid1, sid2 in superedges:
    snode1 = sgraph_info[sid1]
    snode2 = sgraph_info[sid2]
    
    idx1, idx2 = [], []
    
    for in1, out1 in snode1['ext_edges']:
        if partition[out1] == sid2:
            idx1.append(snode1['node_map'][in1])
            idx2.append(snode2['node_map'][out1])
    
    if len(idx1):
        obj = lambda_ * norm1(V[sid1][idx1] - V[sid2][idx2])
        gvx.SetEdgeObjective(sid1, sid2, obj)

t = time.time()
gvx.Solve(UseADMM=True, Verbose=True)
gvx.value
print time.time() - t

V_ = dict([(k, np.asarray(v.value).squeeze()) for k,v in V.items()])
Vf = np.zeros(y.shape[0])

for i in np.arange(y.shape[0]):
    p = partition[i]
    Vf[i] = V_[p][sgraph_info[p]['node_map'][i]]

_ = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(Vf) * np.sqrt(np.abs(Vf)), s=5, cmap='seismic')
show_plot()

# This way:
#
# No ADMM - 2s
# ADMM - 20s
#
# Old way
# No ADMM - 30s