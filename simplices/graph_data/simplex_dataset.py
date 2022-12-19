from flagser_count import *
import numpy as np 
from pathlib import Path
import os
import torch 
from tqdm import tqdm





def save(W0_graph, global_simplex_count, neuron_simplex_count, data_path, seed, count):

    np.savez_compressed(
        data_path/Path(f"{count}.npz"),
        W0 = W0_graph,
        global_simplex = global_simplex_count,
        neuron_simplex = neuron_simplex_count,
        seed = seed,
    )





def simplex_dataset(n_neurons, n_datasets, seed):

    data_path = f"neurons_{n_neurons}"
    if not os.path.isdir(data_path):
                os.mkdir(data_path)


    for i in tqdm(range(n_datasets), desc = "Creating dataset", leave=False, colour="#52D1DC"):
        rng = seed + i
        upper = nx.to_numpy_array(nx.watts_strogatz_graph(n_neurons, k = n_neurons//3, p = 0.3, seed = rng))  #This is the upper triangular part of the matrix
        lower = nx.to_numpy_array(nx.watts_strogatz_graph(n_neurons, k = n_neurons//3, p = 0.3, seed = rng + 1))  #This is the lower triangular part of the matrix
        
        out = np.zeros((n_neurons, n_neurons))
        out[np.triu_indices(n_neurons)] = upper[np.triu_indices(n_neurons)]
        out[np.tril_indices(n_neurons)] = lower[np.tril_indices(n_neurons)]
        W0_graph = out  #nx.from_numpy_array(out, create_using=nx.DiGraph)


        Hasse_simplex = flagser_count_unweighted(W0_graph)

        global_simplex_count = Hasse_simplex.simplex_counter()
        neuron_simplex_count = []

        for neuron in Hasse_simplex.level_0:
            neuron_simplex_count.append(neuron.simplex_count)

        save(W0_graph, global_simplex_count, neuron_simplex_count, data_path, seed, i)

        #print("Max dim: ", len(Hasse_simplex.levels_id))

        



"""
Define parameters
"""

n_neurons = 15
n_datasets = 500
seed = 123

simplex_dataset(n_neurons, n_datasets, seed)



# filename = "neurons_20/0.npz"

# all_info = np.load(filename, allow_pickle=True)

# print(all_info["global_simplex"])
# print(all_info["W0"])
# print(len(all_info["neuron_simplex"]))