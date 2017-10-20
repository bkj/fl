#!/usr/bin/env R

# fusedlasso.R

require(genlasso)

# --
# IO

# Load nodes
nodes = read.csv('./nodes.tsv', sep='\t')
nodes$id = as.numeric(as.factor(nodes$index))
nodes = nodes[order(nodes$id),]
rownames(nodes) = NULL

# Load edges
edges = orig_edges = read.csv('./edges.tsv', sep='\t', header=FALSE)
edges = unlist(t(edges), use.names=F)
edges = edges[1:prod(dim(edges))]
edges = as.numeric(as.factor(edges))

# --
# Fused lasso

gr = graph(edges=edges, directed=FALSE)
D  = getDgSparse(gr)
y  = nodes$neg - nodes$pos

fl = fusedlasso(y, D=D, verbose=TRUE, gamma=0.2)
B = fl$beta

# --
# Inspect results

z = colSums(B != 0)
names(z) = NULL
z

ind = 500
B[,ind][which(B[,ind] != 0)]

# Print nodes w/ nonzero weights
cat(paste(nodes[nodes$id %in% which(B[,ind] != 0),]$index, collapse=',\n'), )
