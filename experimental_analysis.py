# © 2025, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from approaches.aba import run_aba
from approaches.gurobi import run_gurobi
from approaches.random import run_random
from approaches.networkx import run_networkx
from approaches.greedy import run_greedy

# Define results file name
results_file_name = 'results.csv'

# Define path to datasets
datasets_path = 'datasets/'

# Set time limit
time_limit = 7200

# Select approaches
approaches = [
    'gurobi',
    'networkx',
    'aba',
    'greedy',
    'random-1',
    'random-2',
    'random-3',
]

# Select datasets
datasets = [
            'abalone-2k',
            'adult-2k',
            'bank-2k',
            'creditcard-2k',
            'electric-2k',
            'facebook-2k',
            'frogs-2k',
            'plants-2k',
            'pulsar-2k',
            'travel-2k',
            'travel',
            'facebook',
            'electric',
            'npi',
            'pulsar',
            'creditcard'
            ]

# Initialize results
results = pd.DataFrame()

for dataset in datasets:

    # Load dataset
    X = pd.read_csv(datasets_path + dataset + '.csv', header=None).astype(float).values

    for approach in approaches:

            # Print progress
            print(dataset, approach)

            # Initilize new results
            new_results = pd.Series()

            # Add approach and dataset information
            new_results['dataset'] = dataset
            new_results['n_objects'] = X.shape[0]
            new_results['n_features'] = X.shape[1]
            new_results['approach'] = approach

            # Start stopwatch
            tic = time.perf_counter()

            # Run approach
            if approach == 'gurobi':

                # Run gurobi approach
                labels, mip_gap = run_gurobi(X, time_limit=time_limit)

                # Get results
                new_results['time_limit'] = time_limit
                new_results['mip_gap'] = mip_gap

            elif approach == 'networkx':
                labels = run_networkx(X)
            elif approach == 'aba':
                labels = run_aba(X)
            elif approach == 'greedy':
                labels = run_greedy(X)
            else: 
                # Get random seed
                approach_name, seed = approach.split('-')

                # Run random approach
                labels = run_random(X, random_seed=int(seed))

                # Add approach information 
                new_results['approach'] = approach_name
                new_results['random_seed'] = seed

            # Get runtime
            runtime = time.perf_counter() - tic
            new_results['runtime'] = runtime
              
            # Compute total sum of distances within the clusters
            if len(labels) > 0:
                sum_distances_within = 0.0
                for label in np.unique(labels):
                    objects = X[labels == label]
                    if len(objects) >= 2:
                        dists = pdist(objects, metric='sqeuclidean')  
                        sum_distances_within += np.sum(dists)
                new_results['sum_distances_within'] = sum_distances_within
            else:
                new_results['sum_distances_within'] = np.nan

            if results.empty:
                results = pd.DataFrame(new_results).T
            else:
                results = pd.concat([results, pd.DataFrame(new_results).T])

            # Save results into CSV-file
            results.to_csv(results_file_name, index=False)
