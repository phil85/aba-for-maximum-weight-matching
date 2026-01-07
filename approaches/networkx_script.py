# © 2025, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from networkx.algorithms.matching import max_weight_matching


def run_networkx(X):

    # Start timer
    start_time = time.perf_counter()

    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))

    try:
        distances = squareform(pdist(X, metric="sqeuclidean"))

        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                G.add_edge(i, j, weight=distances[i, j])

        matching = max_weight_matching(G)

        # End timer
        end_time = time.perf_counter()
        cpu_time = end_time - start_time

    except MemoryError:
        print("Run out of memory error!")
        return np.array([]), None   

    labels = np.zeros(X.shape[0], dtype=int)
    for i, (u, v) in enumerate(matching):
        labels[u] = labels[v] = i

    return labels, cpu_time

def main():
    parser = argparse.ArgumentParser(description="Run NetworkX max-weight matching")
    parser.add_argument("input", help="Input data file (.npy or .csv)")
    args = parser.parse_args()

    # Load dataset
    X = pd.read_csv(args.input, header=None).astype(float).values        

    labels, cpu = run_networkx(X)

    print(np.array2string(labels, threshold=np.inf), "Elapsed_time = ", cpu)


if __name__ == "__main__":
    main()