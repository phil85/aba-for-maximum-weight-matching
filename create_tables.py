import pandas as pd

# Load results
df = pd.read_csv('results copy.csv')
# Get best result for each dataset

best_results = df.groupby('dataset')['sum_distances_within'].max()
n_values = df.groupby('dataset')['n_objects'].first()
d_values = df.groupby('dataset')['n_features'].first()

gurobi_values = 100 * ((best_results - df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['sum_distances_within'].first()) / best_results)
networkx_values = 100 * ((best_results - df.loc[df['approach'] == 'networkx', :].groupby('dataset')['sum_distances_within'].first()) / best_results)    
aba_values = 100 * ((best_results - df.loc[df['approach'] == 'aba', :].groupby('dataset')['sum_distances_within'].first()) / best_results)
papenberg_values = 100 * ((best_results - df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['sum_distances_within'].mean()) / best_results)
greedy_values = 100 * ((best_results - df.loc[df['approach'] == 'greedy', :].groupby('dataset')['sum_distances_within'].first()) / best_results)
random_values = 100 * ((best_results - df.loc[df['approach'] == 'random', :].groupby('dataset')['sum_distances_within'].mean()) / best_results)

# Prepare table
table = pd.DataFrame(columns=['Dataset', 'n', 'd', 'best', 'gurobi', 'networkx', 'aba', 'papenberg', 'greedy', 'random'], index=best_results.index)

table['Dataset'] = best_results.index
table['n'] = n_values
table['d'] = d_values
table['best'] = best_results.round(2)
table['gurobi'] = gurobi_values.round(2)
table['networkx'] = networkx_values.round(2)
table['aba'] = aba_values.round(2)
table['papenberg'] = papenberg_values.round(2)
table['greedy'] = greedy_values.round(2)
table['random'] = random_values.round(2)

# Use styler
styler = table.style
styler = styler.format(na_rep='\\NA', precision=2)
styler = styler.hide(axis='index')

latex_str = styler.to_latex()

# 

# Save to file
with open('results_table.tex', 'w') as f:
    f.write(latex_str)
