from neuro_ml.dataset import DatasetParams #SimulationEnum
from neuro_ml.models import (
    EdgeRegressor,
    EdgeRegressorParams,
    EdgeClassifier,
    EdgeClassifierParams,
    OuterModel
)
from neuro_ml.fit import fit, test_model, simplex_test
import torch
import sys
from typing_extensions import dataclass_transform
from IPython.core import ultratb
from neuro_ml import config

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
    edge_regressor_params = EdgeRegressorParams(n_shifts=dataset_params.n_shifts, n_neurons=dataset_params.n_neurons_remaining, output_dim = dataset_params.n_neurons_remaining)
    model_name = EdgeRegressor

    if mode == "train": 
    # Fit the model
        fit(
            model_name,
            model_is_classifier=False,
            model_params=edge_regressor_params,
            dataset_params=dataset_params,
            device=device,
        )

    elif mode == "simplex_test":
        simplex_test(model_name, epoch=model_epoch, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device, max_threshold = 7)

    else: test_model(model_name, epoch=model_epoch, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device)






if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set the simulation, window size, and number of files to use
    network_type = "small_world"
    cluster_sizes = [70] #[30] #[40] #[10, 10] #[30] #[24, 10, 12] #[20]
    n_clusters = len(cluster_sizes)
    n_neurons=sum(cluster_sizes)
    n_timesteps = 200000
    number_of_files = 200
    n_shifts = 10

    simplex_threshold = 0   # Number of neurons in the smallest simplex kept
    weight_threshold = 0 # only keep edges with absolute value above this threshold 
    # For small_world, [30], 0, 0.25 and 0.5 basically gave the same results

    neurons_remove = 0
    n_neurons_remaining = n_neurons - neurons_remove

    torch.cuda.manual_seed_all(config.SEED)
    neurons_remove = 0
    n_neurons_remaining = n_neurons - neurons_remove
    #mode = "train"
    mode = "test"
    #mode = "simplex_test"

    scaled_loss = False

    print()
    print("Cluster sizes: ", cluster_sizes)
    print(f"Removing {neurons_remove} neurons")
    print(f"Only keeping simplices containing at least {simplex_threshold} neurons")
    
    dataset_params = DatasetParams(
        network_type,
        n_clusters,
        cluster_sizes,
        n_neurons,
        n_neurons_remaining,
        neurons_remove,
        n_timesteps,
        number_of_files,
        weight_threshold,
        simplex_threshold,
        scaled_loss,
        n_shifts
    )

    edge_regressor(dataset_params, mode)


    # simplex_list = [0, 3, 4, 5, 6]  

    # for simplex_threshold in simplex_list:
    #     torch.cuda.manual_seed_all(config.SEED)
    #     neurons_remove = 0
    #     n_neurons_remaining = n_neurons - neurons_remove

    #     print()
    #     print("Cluster sizes: ", cluster_sizes)
    #     print(f"Removing {neurons_remove} neurons")
    #     print(f"Only keeping simplices containing at least {simplex_threshold} neurons")
        
    #     dataset_params = DatasetParams(
    #         network_type,
    #         n_clusters,
    #         cluster_sizes,
    #         n_neurons,
    #         n_neurons_remaining,
    #         neurons_remove,
    #         n_timesteps,
    #         number_of_files,
    #         weight_threshold,
    #         simplex_threshold,
    #         scaled_loss,
    #         n_shifts
    #     )

    #     edge_regressor(dataset_params, mode)

    # n_neurons = 25

    # for neurons_remove in range(1, 10):

    #     mode = "train"
    #     #mode = "test"
    #     n_neurons_remaining = n_neurons - neurons_remove

    #     print()
    #     print("Cluster sizes: ", cluster_sizes)
    #     print(f"Removing {neurons_remove} neurons")
    #     print(f"Only keeping simplices containing at least {simplex_threshold} neurons")
        
    #     dataset_params = DatasetParams(
    #         network_type,
    #         n_clusters,
    #         cluster_sizes,
    #         n_neurons,
    #         n_neurons_remaining,
    #         neurons_remove,
    #         n_timesteps,
    #         number_of_files,
    #         weight_threshold,
    #         simplex_threshold,
    #         scaled_loss
    #     )

    #     edge_regressor(dataset_params, mode)




    # simplex_threshold = 4   # Number of neurons in the smallest simplex kept
    # weight_threshold = 0 # only keep edges with absolute value above this threshold 
    # # For small_world, [30], 0, 0.25 and 0.5 basically gave the same results

    # neurons_remove = 0
    # n_neurons_remaining = n_neurons - neurons_remove

    # all_sizes = [[20], [25], [30], [40], [50], [60], [70]]
    # #simplex_threshold = 2

    # for cluster_sizes in all_sizes:
    #     n_clusters = len(cluster_sizes)
    #     n_neurons=sum(cluster_sizes)

    #     torch.cuda.manual_seed_all(config.SEED)
    #     neurons_remove = 0
    #     n_neurons_remaining = n_neurons - neurons_remove
    #     #mode = "train"
    #     #mode = "test"

    #     print()
    #     print("Cluster sizes: ", cluster_sizes)
    #     print(f"Removing {neurons_remove} neurons")
    #     print(f"Only keeping simplices containing at least {simplex_threshold} neurons")

    #     dataset_params = DatasetParams(
    #         network_type,
    #         n_clusters,
    #         cluster_sizes,
    #         n_neurons,
    #         n_neurons_remaining,
    #         neurons_remove,
    #         n_timesteps,
    #         number_of_files,
    #         weight_threshold,
    #         simplex_threshold,
    #         scaled_loss,
    #         n_shifts
    #     )

    #     edge_regressor(dataset_params, mode)
