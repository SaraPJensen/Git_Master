from neuro_ml.dataset import DatasetParams 
from neuro_ml.models import (
    ModelParams,
    SimplexModelParams,
    OuterModel,
    Simplicial_GCN,
    Simplicial_MPN,
    Simplicial_MPN_simple,
    Simplicial_MPN_speedup,
    Outer_Simplicial
)

from neuro_ml.fit import fit, test_model
import torch
import sys
from typing_extensions import dataclass_transform
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)



def simplex(dataset_params, mode, model_epoch = "best"):
    model_params = SimplexModelParams(n_neurons=dataset_params.n_neurons_remaining, output_dim = dataset_params.output_dim)
    
    model_name = Outer_Simplicial

    if mode == "train": 
    # Fit the model
        fit(
            model_name,
            model_is_classifier=False,
            model_params=model_params,
            dataset_params=dataset_params,
            device=device,
        )

    else: test_model(model_name, epoch=model_epoch, dataset_params=dataset_params, model_params = model_params, model_is_classifier=False, device=device)




def edge_regressor(dataset_params, mode, model_epoch = "best"):
    # Set the number of time steps we want to calculate co-firing rates for and the number of neurons
    model_params = ModelParams(n_shifts=10, n_neurons=dataset_params.n_neurons_remaining, output_dim = dataset_params.output_dim)
    
    model_name = EdgeRegressor

    if mode == "train": 
    # Fit the model
        fit(
            model_name,
            model_is_classifier=False,
            model_params=model_params,
            dataset_params=dataset_params,
            device=device,
        )

    else: test_model(model_name, epoch=model_epoch, dataset_params=dataset_params, model_params = model_params, model_is_classifier=False, device=device)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the simulation, window size, and number of files to use
    network_type = "small_world"
    cluster_sizes = [20] #[30] #[40] #[10, 10] #[30] #[24, 10, 12] #[20]
    n_clusters = len(cluster_sizes)
    n_neurons=sum(cluster_sizes)
    n_timesteps = 100000
    timestep_bin_length = 100000
    number_of_files = 500
    output_dim = 5    #Max simplex dimension

    neurons_remove = 0
    n_neurons_remaining = n_neurons - neurons_remove

    mode = "train"
    #mode = "test"

    print("Cluster sizes: ", cluster_sizes)
    #print(f"Removing {neurons_remove} neurons")
    
    dataset_params = DatasetParams(
        network_type,
        n_clusters,
        cluster_sizes,
        n_neurons,
        n_neurons_remaining,
        neurons_remove,
        n_timesteps,
        timestep_bin_length,
        number_of_files,
        output_dim,
    )

    simplex(dataset_params, mode)
