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


def fit_edge_regressor(dataset_params):
    # Set the number of time steps we want to calculate co-firing rates for and the number of neurons
    edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons)

    # Fit the model
    fit(
        EdgeRegressor,
        model_is_classifier=False,
        model_params=edge_regressor_params,
        dataset_params=dataset_params,
        device=device,
    )



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the simulation, window size, and number of files to use
    cluster_sizes = [30] #[30] #[40] #[10, 10] #[30] #[24, 10, 12] #[20]
    n_clusters = len(cluster_sizes)
    n_neurons=sum(cluster_sizes)
    n_timesteps=100001
    timestep_bin_length=100001
    number_of_files= 100 
    network_type = "small_world"

    print("Cluster sizes: ", cluster_sizes)

    dataset_params = DatasetParams(
        network_type,
        n_clusters,
        cluster_sizes,
        n_neurons,
        n_timesteps,
        timestep_bin_length,
        number_of_files,
    )

    fit_edge_regressor(dataset_params)


    #edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons)
    #test_model(EdgeRegressor, epoch=100, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device)
