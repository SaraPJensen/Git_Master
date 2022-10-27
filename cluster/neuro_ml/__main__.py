from neuro_ml.dataset import SimulationEnum, DatasetParams
from neuro_ml.models import (
    EdgeRegressor,
    EdgeRegressorParams,
    EdgeClassifier,
    EdgeClassifierParams,
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
    edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons, output_dim = dataset_params.output_dim)

    if mode == "train": 
    # Fit the model
        fit(
            EdgeRegressor,
            model_is_classifier=False,
            model_params=edge_regressor_params,
            dataset_params=dataset_params,
            device=device,
        )

    else: test_model(EdgeRegressor, epoch=model_epoch, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device)





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the simulation, window size, and number of files to use
    network_type = "small_world"
    cluster_sizes = [20, 30, 50] #[30] #[40] #[10, 10] #[30] #[24, 10, 12] #[20]
    n_clusters = len(cluster_sizes)
    n_neurons=sum(cluster_sizes)
    n_timesteps = 200000
    timestep_bin_length = 200000
    number_of_files = 200 
    dim = "high"

    mode = "train"
    #mode = "test"

    if dim == "high":
        output_dim = n_neurons
    elif params.dim == "low":
        output_dim = n_clusters

    print("Cluster sizes: ", cluster_sizes)

    
    dataset_params = DatasetParams(
        network_type,
        n_clusters,
        cluster_sizes,
        n_neurons,
        n_timesteps,
        timestep_bin_length,
        number_of_files,
        output_dim
    )

    edge_regressor(dataset_params, mode)

    #edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons)
    #test_model(EdgeRegressor, epoch=100, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device)
