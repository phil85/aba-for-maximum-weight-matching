import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from networkx.algorithms.matching import max_weight_matching


def run_networkx(X):

    # Create a graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(X.shape[0]))

    try: 

        # Compute distances
        distances = squareform(pdist(X, metric='sqeuclidean'))

        # Add edges with weights
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                G.add_edge(i, j, weight=distances[i, j])

        # Find maximum weight matching
        matching = max_weight_matching(G)

    except:
        print('Run out of memory error!')
        return []

    # Get labels
    labels = np.zeros(X.shape[0], dtype=int)
    for i, (u, v) in enumerate(matching):
        labels[u] = labels[v] = i

    return labels
