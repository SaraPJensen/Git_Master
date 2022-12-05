import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils.sparse import dense_to_sparse
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset
import os
import torch.nn as nn

@dataclass
class ModelParams:
    n_shifts: int 
    n_neurons: int    #The number of remaining neurons
    output_dim: int




class Simplicial_MPN(MessagePassing): # Or `Module`

    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MP_iterations = 2
        self.output_dim = params.output_dim

        self.mlp1 = Seq(
            Linear(self.MP_iterations, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 1)
        )

        self.mlp2 = Seq(
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, 5)
        )


    def forward(self, edge_index, x): 
        # This is the actual message-passing step

        propagated = self.propagate(edge_index, x)


    def propagate(self, edge_index, x):
        # process arguments and create *_kwargs

        for i in range(self.MP_iterations):
            # Message
            messages = self.message(x, x, edge_index)

            # Aggregate
            aggregated = self.aggregate(messages)

            # Update
            output = self.update(aggregated)

        #Maybe add a third MLP layer which updates after the two iterations?? 
        
        return output

    def message(self, x_sender, x_receiver, edge_index):
        return self.mlp1(x_receiver)

    def aggregate(self, messages):
        aggregation = torch.sum(messages)
        return aggregation


    def update(self, aggregated, **kwargs):    #Returns something of the same size
        output = self.mlp2(aggregated)
        return output














class Simplicial_GCN(nn.Module):
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset
    NAME = "simplicial"

    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons
        self.output_dim = params.output_dim
        self.params = params
        self.hidden_dim = 30

        self.simplex1 = GCNConv(self.n_neurons, self.hidden_dim)
        self.simplex2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.simplex3 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.linear = Linear(self.hidden_dim, self.output_dim)
        
        
    def forward(self, x):
        #print("Connectivity shape: ", connectivity.shape)
        #print("Edge index shape: ", edge_indices.shape)
        edge_index = x.nonzero().t()

        simplices = self.simplex1(x, edge_index)
        simplices = simplices.relu()
        simplices = self.simplex2(simplices, edge_index)
        simplices = simplices.relu()
        simplices = self.simplex3(simplices, edge_index)
        simplices = global_mean_pool(simplices, None)


        #print("Simplices shape: ", simplices.shape)
        pred = self.linear(simplices).squeeze()
        #print("Pred shape: ", pred.shape)
        #print()

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
