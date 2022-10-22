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


def fit_lstm(dataset_params):
    lstm_params = LSTMParams(
        timestep_bin_length=dataset_params.timestep_bin_length,
        n_nodes=dataset_params.n_neurons,
    )
    fit(
        LSTM,
        False,
        lstm_params,
        dataset_params,
        device=device,
    )


def fit_graph_transformer(dataset_params):
    graph_transformer_params = GraphTransformerParams(
        timestep_bin_length=dataset_params.timestep_bin_length,
        n_neurons=dataset_params.n_neurons,
        n_latent_features=100,
        n_eigen_values=18,
        n_heads=9,
        n_edges=180,
        batch_size=32,
        device=device,
        num_layers=5,
    )
    fit(
        GraphTransformer,
        model_is_classifier=False,
        model_params=graph_transformer_params,
        dataset_params=dataset_params,
        device=device,
    )


def fit_gcn(dataset_params):
    gcn_params = GCNParams(
        timestep_bin_length=dataset_params.timestep_bin_length,
        n_neurons=dataset_params.n_neurons,
    )

    fit(
        GCN,
        False,
        model_params=gcn_params,
        dataset_params=dataset_params,
        device=device,
    )


def fit_simple_transformer(dataset_params):
    transformer_params = SimpleTransformerPrams(
        d_model=dataset_params.timestep_bin_length, n_head=5
    )
    fit(
        SimpleTransformer,
        model_is_classifier=False,
        model_params=transformer_params,
        dataset_params=dataset_params,
        device=device,
    )


def fit_simple(dataset_params):
    simple_params = SimpleParams(n_shifts=10)
    fit(
        Simple,
        model_is_classifier=False,
        model_params=simple_params,
        dataset_params=dataset_params,
        device=device,
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
    n_clusters = 1
    cluster_sizes = [24, 10, 12] #[20]
    n_neurons=sum(cluster_sizes)
    n_timesteps=100000
    timestep_bin_length=100000
    number_of_files=150
    network_type = "small_world"

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
    #fit_edge_classifier(dataset_params)

    #edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons)
    #test_model(EdgeRegressor, epoch=30, dataset_params=dataset_params, model_params = edge_regressor_params, model_is_classifier=False, device=device)
