from neuro_ml.dataset import DatasetParams #SimulationEnum
from neuro_ml.models import (
    EdgeRegressor,
    EdgeRegressorParams,
    EdgeClassifier,
    EdgeClassifierParams,
    OuterModel
)
from neuro_ml.fit import fit, test_model
import torch
import sys
from typing_extensions import dataclass_transform
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)

def fit_edge_classifier(dataset_params):
    edge_classifer_params = EdgeClassifierParams(n_shifts=20, n_classes=3)

    fit(
        EdgeClassifier,
        model_is_classifier=True,
        model_params=edge_classifer_params,
        dataset_params=dataset_params,
        device=device,
    )


def edge_regressor(dataset_params, mode, model_epoch = "best"):
    # Set the number of time steps we want to calculate co-firing rates for and the number of neurons
    edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons_remaining, output_dim = dataset_params.time_dep)

    model_name = EdgeRegressor
    
    # if dataset_params.dim == "high":
    #     model_name = EdgeRegressor
    # else: 
    #     model_name = OuterModel

    if mode == "train": 
    # Fit the model
        fit(
            model_name,
            model_is_classifier=False,
            model_params=edge_regressor_params,
            dataset_params=dataset_params,
            device=device,
        )

    else: test_model(model_name, epoch=model_epoch, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the simulation, window size, and number of files to use
    network_type = "small_world"
    cluster_sizes = [10, 10] #[30] #[40] #[10, 10] #[30] #[24, 10, 12] #[20]
    n_clusters = len(cluster_sizes)
    n_neurons=sum(cluster_sizes)
    time_dep = 1
    n_timesteps = 100000
    timestep_bin_length = 100000
    number_of_files = 150 

    neurons_remove = 0
    n_neurons_remaining = n_neurons - neurons_remove

    mode = "train"
    #mode = "test"

    print("Cluster sizes: ", cluster_sizes)
    print(f"Removing {neurons_remove} neurons")
    
    dataset_params = DatasetParams(
        network_type,
        n_clusters,
        cluster_sizes,
        n_neurons,
        n_neurons_remaining,
        neurons_remove,
        time_dep,
        n_timesteps,
        timestep_bin_length,
        number_of_files,
    )

    edge_regressor(dataset_params, mode)
