import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv, TransformerConv, GATv2Conv
from torch_geometric.utils.sparse import dense_to_sparse
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import W0_to_simplex_Dataset
import os
import torch.nn as nn
from torch_geometric.utils import add_self_loops


@dataclass
class SimplexModelParams:
    n_neurons: int    #The number of remaining neurons
    output_dim: int



class Simple_MLP(MessagePassing): 

    DATASET = W0_to_simplex_Dataset 
    NAME = "W0_to_simplex_simple_mlp"


    def __init__(self, params):
        super(Simple_MLP, self).__init__()  


        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim


        self.mlp = Seq(
            Linear(self.n_neurons * self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)   #Can only find simplices of one dimension
        )


    def forward(self, edge_index, x): 
        input = x.flatten()
        output = self.mlp(input)

        return output





    def makedirs(self, network_type):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}") 

        return f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"


    
    def save(self, filename):   
        # Saves the model to file
        torch.save(self.state_dict(), filename)