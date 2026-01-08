import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results copy.csv')

# Replace names of datasets
df['dataset'] = df['dataset'].replace({
    'abalone-2k': 'Abalone-2k',
    'adult-2k': 'Adult-2k',
    'bank-2k': 'Bank-2k',
    'creditcard-2k': 'Creditcard-2k',
    'electric-2k': 'Electric-2k',
    'facebook-2k': 'Facebook-2k',
    'frogs-2k': 'Frogs-2k',
    'plants-2k': 'Plants-2k',
    'pulsar-2k': 'Pulsar-2k',
    'travel-2k': 'Travel-2k',
    'travel': 'Travel',
    'facebook': 'Facebook',
    'electric': 'Electric',
    'npi': 'Npi',
    'pulsar': 'Pulsar',
    'creditcard': 'Creditcard'
})

best_results = df.groupby('dataset')['sum_distances_within'].max()
n_values = df.groupby('dataset')['n_objects'].first()
d_values = df.groupby('dataset')['n_features'].first()

# --------------------------------------------------------------
# 1) Create table with objective function values
# --------------------------------------------------------------

# Get dataset for which an optimal solution has been found
idx = (df['approach'] == 'gurobi') & (df['mip_gap'] < 0.0001) & (df['runtime'] < 7200)
dataset_for_which_an_optimal_solution_has_been_found = df.loc[idx, 'dataset'].unique()

gurobi_values = 100 * ((best_results - df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['sum_distances_within'].first()) / best_results)
networkx_values = 100 * ((best_results - df.loc[df['approach'] == 'networkx', :].groupby('dataset')['sum_distances_within'].first()) / best_results)    
aba_values = 100 * ((best_results - df.loc[df['approach'] == 'aba', :].groupby('dataset')['sum_distances_within'].first()) / best_results)
papenberg_values = 100 * ((best_results - df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['sum_distances_within'].mean()) / best_results)
greedy_values = 100 * ((best_results - df.loc[df['approach'] == 'greedy', :].groupby('dataset')['sum_distances_within'].first()) / best_results)
random_values = 100 * ((best_results - df.loc[df['approach'] == 'random', :].groupby('dataset')['sum_distances_within'].mean()) / best_results)

# Prepare table
table = pd.DataFrame(columns=['Dataset', 'n', 'd', 'best', 'gurobi', 'networkx', 'aba', 'papenberg', 'greedy', 'random'], index=n_values.sort_values().index)

table['Dataset'] = table.index
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
styler = styler.format(na_rep='\\NA', precision=2, thousands=',')
styler = styler.set_properties(subset=(dataset_for_which_an_optimal_solution_has_been_found, ["best"]),**{"font-weight": "bold"})
styler = styler.hide(axis='index')

latex_str = styler.to_latex(convert_css=True)

# Replace substrings
latex_str = latex_str.replace("Dataset & n & d & best & gurobi & networkx & aba & papenberg & greedy & random \\\\", 
                              "\\toprule Dataset & $n$ & $d$ & Best & \% DEV & \% DEV & \% DEV & \% DEV & \% DEV & \% DEV \\\\\n & & & objective & Gurobi & NetworkX & \ABA & Papenberg & Greedy & Random \\\\ \midrule")

latex_str = latex_str.replace("\\end{tabular}", "\\bottomrule\n\\end{tabular}")

# Save to file
with open('results_table_ofv.tex', 'w') as f:
    f.write(latex_str)

# --------------------------------------------------------------
# 2) Create table with running times
# --------------------------------------------------------------

gurobi_values = df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['runtime'].first()
networkx_values = df.loc[df['approach'] == 'networkx', :].groupby('dataset')['runtime'].first()
aba_values = df.loc[df['approach'] == 'aba', :].groupby('dataset')['runtime'].first()
papenberg_values = df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['runtime'].mean()
greedy_values = df.loc[df['approach'] == 'greedy', :].groupby('dataset')['runtime'].first()
random_values = df.loc[df['approach'] == 'random', :].groupby('dataset')['runtime'].mean()

# Remove runtime for datasets where no solution has been found
idx = df.loc[df['approach'] == 'networkx', :].groupby('dataset')['sum_distances_within'].first().isna()
networkx_values.loc[idx] = pd.NA

idx = df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['sum_distances_within'].first().isna()
gurobi_values.loc[idx] = pd.NA

idx = df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['sum_distances_within'].first().isna()
papenberg_values.loc[idx] = pd.NA

# Prepare table
table = pd.DataFrame(columns=['Dataset', 'n', 'd', 'gurobi', 'networkx', 'aba', 'papenberg', 'greedy', 'random'], index=n_values.sort_values().index)

table['Dataset'] = table.index
table['n'] = n_values
table['d'] = d_values
table['gurobi'] = gurobi_values.round(2)
table['networkx'] = networkx_values.round(2)
table['aba'] = aba_values.round(2)
table['papenberg'] = papenberg_values.round(2)
table['greedy'] = greedy_values.round(2)
table['random'] = random_values.round(2)

# Use styler
styler = table.style
styler = styler.format(na_rep='\\NA', precision=2, thousands=',')
styler = styler.hide(axis='index')

latex_str = styler.to_latex(convert_css=True)

# Replace substrings
latex_str = latex_str.replace("Dataset & n & d & gurobi & networkx & aba & papenberg & greedy & random \\\\", 
                              "\\toprule Dataset & $n$ & $d$ & CPU & CPU & CPU & CPU & CPU & CPU \\\\\n & & & Gurobi & NetworkX & \ABA & Papenberg & Greedy & Random \\\\ \midrule")

latex_str = latex_str.replace("\\end{tabular}", "\\bottomrule\n\\end{tabular}")

latex_str = latex_str.replace(" & 0.00 \\", " & $<$0.01 \\")

# Save to file
with open('results_table_cpu.tex', 'w') as f:
    f.write(latex_str)

# --------------------------------------------------------------
# 3) Create table with slowdown factors
# --------------------------------------------------------------

aba_values = df.loc[df['approach'] == 'aba', :].groupby('dataset')['runtime'].first()
gurobi_values = df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['runtime'].first() / aba_values
networkx_values = df.loc[df['approach'] == 'networkx', :].groupby('dataset')['runtime'].first() / aba_values
papenberg_values = df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['runtime'].mean() / aba_values
greedy_values = df.loc[df['approach'] == 'greedy', :].groupby('dataset')['runtime'].first() / aba_values

# Remove runtime for datasets where no solution has been found
idx = df.loc[df['approach'] == 'networkx', :].groupby('dataset')['sum_distances_within'].first().isna()
networkx_values.loc[idx] = pd.NA

idx = df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['sum_distances_within'].first().isna()
gurobi_values.loc[idx] = pd.NA

idx = df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['sum_distances_within'].first().isna()
for entry in idx.index[idx]:
    papenberg_values.loc[entry] = pd.NA

# Prepare table
table = pd.DataFrame(columns=['Dataset', 'n', 'd', 'aba', 'gurobi', 'networkx', 'papenberg', 'greedy'], index=n_values.sort_values().index)

table['Dataset'] = table.index
table['n'] = n_values
table['d'] = d_values
table['aba'] = aba_values.round(2)
table['gurobi'] = gurobi_values.round(2)
table['networkx'] = networkx_values.round(2)
table['papenberg'] = papenberg_values.round(2)
table['greedy'] = greedy_values.round(2)

# Use styler
styler = table.style
styler = styler.format(na_rep='\\NA', precision=2, thousands=',')
styler = styler.hide(axis='index')

styler = styler.format(
    subset=['gurobi', 'networkx', 'papenberg', 'greedy'],
    formatter="$\\times$ {:,}"
)

latex_str = styler.to_latex(convert_css=True)

# Replace substrings
latex_str = latex_str.replace("Dataset & n & d & aba & gurobi & networkx & papenberg & greedy \\\\", 
                              "\\toprule Dataset & $n$ & $d$ & \\ABA\ CPU [s] & \\multicolumn{4}{c}{Slow-down factor from \\textsf{ABA}} \\\\ \cmidrule(lr){5-8} \n & & & & Gurobi/\ABA & NetworkX/\ABA & Papenberg/\ABA & Greedy/\ABA \\\\ \midrule")

latex_str = latex_str.replace("\\end{tabular}", "\\bottomrule\n\\end{tabular}")
latex_str = latex_str.replace(" $\\times$ nan", "\\NA")

# Save to file
with open('results_table_slowdown.tex', 'w') as f:
    f.write(latex_str)

# --------------------------------------------------------------
# 4) Create figure with slowdown factors
# --------------------------------------------------------------

aba_values = df.loc[df['approach'] == 'aba', :].groupby('dataset')['runtime'].first()
gurobi_values = df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['runtime'].first() / aba_values
networkx_values = df.loc[df['approach'] == 'networkx', :].groupby('dataset')['runtime'].first() / aba_values
papenberg_values = df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['runtime'].mean() / aba_values
greedy_values = df.loc[df['approach'] == 'greedy', :].groupby('dataset')['runtime'].first() / aba_values

# Remove runtime for datasets where no solution has been found
idx = df.loc[df['approach'] == 'networkx', :].groupby('dataset')['sum_distances_within'].first().isna()
networkx_values.loc[idx] = pd.NA

idx = df.loc[df['approach'] == 'gurobi', :].groupby('dataset')['sum_distances_within'].first().isna()
gurobi_values.loc[idx] = pd.NA

idx = df.loc[df['approach'] == 'anticlust', :].groupby('dataset')['sum_distances_within'].first().isna()
for entry in idx.index[idx]:
    papenberg_values.loc[entry] = pd.NA


fig, ax = plt.subplots(figsize=(10, 4))

instances = n_values.sort_values().index

# Add bars for each approach with slowdown factors
bar_width = 0.15
x = range(len(instances))
ax.set_axisbelow(True)
ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
ax.bar([i - 2*bar_width for i in x], gurobi_values.loc[instances], width=bar_width, label='Gurobi', color='C0')
ax.bar([i - bar_width for i in x], networkx_values.loc[instances], width=bar_width, label='NetworkX', color='C1')
ax.bar(x, papenberg_values.loc[instances], width=bar_width, label='Papenberg', color='C2')
ax.bar([i + bar_width for i in x], greedy_values.loc[instances], width=bar_width, label='Greedy', color='C3')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(instances, rotation=45, ha='right')
ax.set_ylabel('Slow-down factor (log scale)')
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False
)

import numpy as np

ax.set_yscale("log")

ax.relim()
ax.autoscale_view()

ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)

def add_nan_bars(x_positions, values, offset, color):
    for i, v in enumerate(values):
        if np.isnan(v):
            ax.bar(
                x_positions[i] + offset,
                height=ymax,
                bottom=ymin,
                width=bar_width,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                linewidth=1.2,
            )
            label = None

x_list = list(x)

add_nan_bars(x_list, gurobi_values.loc[instances], -2 * bar_width, "C0")
add_nan_bars(x_list, networkx_values.loc[instances], -bar_width, "C1")
add_nan_bars(x_list, papenberg_values.loc[instances], 0, "C2")
add_nan_bars(x_list, greedy_values.loc[instances], bar_width, "C3")

plt.tight_layout()
plt.savefig('slowdown_factors.pdf')
plt.close()

