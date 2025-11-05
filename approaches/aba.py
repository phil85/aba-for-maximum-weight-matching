import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def run_aba(X):

    # Get number of objects
    n_objects = X.shape[0]

    # Get number of anticlusters
    n_clusters = int(np.ceil(n_objects / 2))

    # Initialize labels
    labels = np.full(n_objects, -1)

    # Compute distances to global centroid
    global_centroid = X.mean(axis=0)
    distances = cdist(X, [global_centroid], 'sqeuclidean')

    # Sort objects in descending distance from global centroid
    sorted_objects = np.argsort(-distances[:, 0])

    # Get v1 (all odd indexed objects) and v2 (all even indexed objects)
    v1 = sorted_objects[::2]
    v2 = sorted_objects[1::2]

    # Assign first objects to centers
    labels[v1] = np.arange(n_clusters)

    # Compute distances between v1 and v2
    distances = cdist(X[v2, :], X[v1, :], 'sqeuclidean')

    # Solve assignment problem to maximize total distance
    _, col_ind = linear_sum_assignment(distances, maximize=True)

    # Update labels
    labels[v2] = col_ind

    return labels



