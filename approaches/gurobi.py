# © 2025, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import numpy as np
import gurobipy as gb
from scipy.spatial.distance import pdist, squareform


def run_gurobi(X, time_limit=None):

    # Get object ids
    object_ids = np.arange(X.shape[0])

    try: # To catch memory errors 

        # Compute distances
        distances = squareform(pdist(X, metric='sqeuclidean'))
        dist = {(i, j): distances[i, j] for i in object_ids for j in object_ids if i < j}

        # Create model
        m = gb.Model()

        # Add variables
        x = m.addVars(dist.keys(), vtype=gb.GRB.BINARY, obj=dist)

        # Maximize objective
        m.modelSense = gb.GRB.MAXIMIZE
        
        # Add constraints
        m.addConstrs(gb.quicksum(x[i, j] for j in object_ids if i < j) + gb.quicksum(x[j, i] for j in object_ids if j < i) <= 1 for i in object_ids)
        
        # Set Parameters
        if time_limit is not None:
            m.setParam('TimeLimit', time_limit)

        # Set output flag
        m.setParam('OutputFlag', 0)

        # Optimize model
        m.optimize()

    except: 
        print('Run out of memory error!')
        return [], np.nan
    
    # Get solution
    n = X.shape[0]
    labels = np.full(n, -1)

    # Extract pairs
    pairs = {(i, j) for i in range(n) for j in range(i+1, n) if x[i, j].X > 0.5}
    for label, (i, j) in enumerate(pairs):
        labels[i] = labels[j] = label        

    return labels, m.MIPGap