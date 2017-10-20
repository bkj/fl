#!/usr/bin/env python

"""
    supervx.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict

from snapvx import *
import networkx as nx
from community import community_louvain

from rsub import *
from matplotlib import pyplot as plt

# --
# SuperVX

class SuperVX(object):
    
    def __init__(self, supernodes, superedges, reg_sparsity=5, reg_edge=10):
        self.reg_sparsity = reg_sparsity
        self.reg_edge = reg_edge
        
        self.gvx = self._make_gvx(supernodes, superedges)
        self.values = self._add_objectives(self.gvx, supernodes, superedges)
                
    def _make_gvx(self, supernodes, superedges):
        # Construct supergraph
        sgraph = PUNGraph.New()
        _ = [sgraph.AddNode(l) for l in supernodes.keys()]
        _ = [sgraph.AddEdge(e1, e2) for e1, e2 in superedges.keys() if e1 != e2]
        return TGraphVX(sgraph)
    
    def _add_objectives(self, gvx, supernodes, superedges):
        
        # Change these functions to change behavior
        def node_obj(v, sfeats, int_edges):
           return (
                0.5 * sum_squares(v - sfeats) + 
                self.reg_sparsity * norm1(v) + 
                self.reg_edge * norm1(v[int_edges[:,0]] - v[int_edges[:,1]])
            )
        
        def edge_obj(values, ext_edges):
            return (
                self.reg_edge * norm1(values[sid1][ext_edges[:,0]] - values[sid2][ext_edges[:,1]])
            )
        
        # Node objectives
        values = {}
        for sid, sfeats in supernodes.items():            
            v = Variable(sfeats.shape[0], name='x')
            gvx.SetNodeObjective(sid, node_obj(v, sfeats, np.array(superedges[sid, sid])))
            values[sid] = v
        
        # Edge objectives
        for (sid1, sid2), ext_edges in superedges.items():
            if sid1 != sid2:
                gvx.SetEdgeObjective(sid1, sid2, edge_obj(values, np.array(superedges[sid1, sid2])))
        
        return values
    
    def solve(self, **kwargs):
        return self.gvx.Solve(**kwargs)


class SuperGraph(object):
    
    def __init__(self, edges, feats):
        
        # Partition graph
        G = nx.from_edgelist(edges)
        partition = community_louvain.best_partition(G)
        partition = np.array([p[1] for p in sorted(partition.items(), key=lambda x: x[0])])
        
        self.lookup, self.supernodes = self._make_supernodes(feats, partition)
        self.superedges = self._make_superedges(G, partition, self.lookup)
        
    def _make_supernodes(self, feats, partition):
        lookup = {}
        supernodes = {}
        for p in np.unique(partition):
            psel = np.where(partition == p)[0]
            lookup[p] = dict(zip(psel, range(len(psel))))
            supernodes[p] = feats[psel]
        
        return lookup, supernodes
    
    def _make_superedges(self, G, partition, lookup):
        superedges = defaultdict(list)
        for node in G.nodes():
            p_node = partition[node]
            for neib in G.neighbors(node):
                p_neib = partition[neib]
                
                if (p_node == p_neib) and (node >= neib):
                    continue
                elif p_node > p_neib:
                    continue
                
                superedge = (lookup[p_node][node], lookup[p_neib][neib])
                superedges[(p_node, p_neib)].append(superedge)
        
        return superedges
    
    def unpack(self, values):
        values = dict([(k, np.asarray(v.value).squeeze()) for k,v in values.items()])
        
        reverse_lookup = [[(i[0], (sk, i[1])) for i in v.items()] for sk,v in supergraph.lookup.items()]
        reverse_lookup = dict(reduce(lambda a,b: a + b, reverse_lookup))
        
        unpacked = np.zeros(len(reverse_lookup))
        for idx, (p, int_idx) in reverse_lookup.items():
            unpacked[idx] = values[p][int_idx]
        
        return unpacked





# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-sparsity', type=float, default=5)
    parser.add_argument('--reg-edge', type=float, default=10)
    return parser.parse_args()

args = parse_args()

# --
# IO

edges = pd.read_csv('../edges.tsv', sep='\t', header=None)
edges = edges[edges[0] != edges[1]]

nodes = pd.read_csv('../nodes.tsv', sep='\t')


nodes['d']   = nodes.neg - nodes.pos
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

supergraph = SuperGraph(edges, feats)
svx = SuperVX(supergraph.supernodes, supergraph.superedges, reg_sparsity=args.reg_sparsity, reg_edge=args.reg_edge)
svx.solve(UseADMM=False, Verbose=True)
fitted = supergraph.unpack(svx.values)

graph_coords = np.load('../coords.npy')
_ = plt.scatter(graph_coords[:,1], graph_coords[:,0], c=np.sign(fitted) * np.sqrt(np.abs(fitted)), s=5, cmap='seismic')
show_plot()
