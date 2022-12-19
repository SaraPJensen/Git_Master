import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils.sparse import dense_to_sparse
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset
import os
import torch.nn as nn

from neuro_ml.models.edge_regressor import EdgeRegressor, ModelParams
from neuro_ml.models.simplicial import Simplicial_GCN, Simplicial_MPN, SimplexModelParams, Simplicial_MPN_simple



class OuterModel(nn.Module):
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset
    NAME = "simplex_finder"

    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons
        self.output_dim = params.output_dim
        self.params = params

        self.EdgeRegressor = EdgeRegressor(params)
        self.simplicial = Simplicial_GCN(params)
        

    def forward(self, x, edge_index):
        connectivity = self.EdgeRegressor(x, edge_index)
        edge_index = connectivity.nonzero().t()

        pred = self.simplicial(connectivity)
  
        return pred


    def makedirs(self, network_type, save_folder, n_remove):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{save_folder}"):
                os.mkdir(f"saved_models/{self.NAME}/{save_folder}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{save_folder}/remove_{n_remove}"):
                os.mkdir(f"saved_models/{self.NAME}/{save_folder}/remove_{n_remove}") 

        return f"saved_models/{self.NAME}/{save_folder}/remove_{n_remove}"


    def save(self, filename):   
        # Saves the model to file
        torch.save(self.state_dict(), filename)