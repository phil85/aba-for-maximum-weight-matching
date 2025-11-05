# © 2025, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

def greedy_max_weight_matching(G):
    matched = set()
    matching = set()
    for u, v, w in sorted(G.edges(data="weight"), key=lambda x: -x[2]):
        if u not in matched and v not in matched:
            matching.add((u, v))
            matched.add(u)
            matched.add(v)
    return matching


def run_greedy(X):

    # Create a graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(X.shape[0]))

    # Compute distances
    distances = squareform(pdist(X, metric='sqeuclidean'))

    # Add edges with weights
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            G.add_edge(i, j, weight=distances[i, j])

    # Find maximum weight matching
    matching = greedy_max_weight_matching(G)

    # Get labels
    labels = np.zeros(X.shape[0], dtype=int)
    for i, (u, v) in enumerate(matching):
        labels[u] = labels[v] = i

    return labels
