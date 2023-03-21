import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset
import os
import torch.nn as nn

@dataclass
class EdgeRegressorParams:
    n_shifts: int 
    n_neurons: int 
    output_dim: int


class EdgeRegressor(MessagePassing):
    """
    Calculates x_i' using the formula:
    x_i' = MLP_2(||_{j \in \mathcal{N}(i) MLP_1(1/N ||_{t=1}^M x_i * P^t x_j))
    """

    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset    #Variables defined outside the init belong to all instances of the class, and cannot be changed
    NAME = "edge_regressor"

    def __init__(self, params):
        super().__init__()
        self.n_shifts = params.n_shifts # M, number of time steps for which we consider the influence of i  on j forward in time
        self.n_neurons = params.n_neurons #_remaining # N, number of neurons in the network
        #self.n_clusters = params.n_clusters # Number of clusters in the network
        self.selection_matrix = (torch.eye(self.n_neurons)
                .repeat_interleave(self.n_neurons, dim=0)) # Matrix that selects the j-th neuron in the MLP_1 output
                #repeat_interleave repeats each element n_neurons times

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dim = params.output_dim

        self.mlp1 = Seq(
            Linear(params.n_shifts, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 1)
            ) # First MLP
        
        self.mlp2 = Seq(
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.output_dim)
        ) # Second MLP

        with torch.no_grad():
            nn.init.kaiming_normal_(self.mlp1[0].weight)
            nn.init.kaiming_normal_(self.mlp1[2].weight)
            nn.init.kaiming_normal_(self.mlp1[4].weight)

            nn.init.kaiming_normal_(self.mlp2[0].weight)
            nn.init.kaiming_normal_(self.mlp2[2].weight)


    def forward(self, x, edge_index):
        #print("Calling forward")
        # x has shape [N, n_shitfs]
        # edge_index has shape [2, E], one layer for each direction
        high_dim = self.propagate(edge_index, x=x)   #This calls message and update, shape of answer is [n_neurons, n_neurons], this is just the output from MLP2 
        #print("High dim shape: ", high_dim.shape)

        return high_dim

    def message(self, x_i, x_j):
        #print("Calling message")
        # x_i has shape [E, in_channels], E is the number of edges, n_neurons*n_neurons
        # x_j has shape [E, in_channels], in_channels is the number of timesteps
        batch_size = int(x_i.shape[0] / self.n_neurons**2)

        # Calculate the influence of i on j forward in time
        inner_products = [torch.sum(x_i*self.shift(x_j, -(t+1)), dim=1).unsqueeze(dim=1) for t in range(self.n_shifts)]  
        #List of length 10, since 10 shifts considered, each element is a tensor of shape [n_neurons*n_neurons, 1], i.e. [E, 1], giving the influence at that timestep

        # Concatenate the inner products to get vectors of length M
        tmp = torch.cat(inner_products, dim=1)
        tmp = tmp / (x_i.shape[1]/100) # Normalize the inner products to get co-firings per 100 time steps, shape: 400, 10

        batch_selection_matrix = self.selection_matrix.repeat(batch_size, 1).to(self.device) # Repeat the selection matrix for the entire batch, shape: 400, 20
        #Output of MLP_1, shape: [n_neurons*n_neurons, 1]
        #Batch selection matrix shape: [n_neurons*n_neurons, n_neurons]

        #print("Output of mlp1 shape: ", self.mlp1(tmp).shape)
        #print("batch_selection_matrix shape: ", batch_selection_matrix.shape)

        return_val = self.mlp1(tmp)*batch_selection_matrix   #This has shape [n_neurons*n_neurons, n_neurons]

        #print("Return shape from message: ", return_val.shape)

        return return_val #self.mlp1(tmp)*batch_selection_matrix # Apply the first MLP to the entire batch

    def update(self, inputs):  #Shape of input is [n_neurons, n_neurons]
        #print("Input shape to update: ", inputs.shape)
        #print("Calling update")
        mlp2_out = self.mlp2(inputs)
        #print("MLP2 output shape: ", mlp2_out.shape)
        return mlp2_out #self.mlp2(inputs) # Apply the second MLP, output shape: [n_neurons, n_neurons]

    def shift(self, x, n):
        #print("Calling shift")
        # Shifts the time series x by n time steps to the left, called by message, once for each timestep
        result = torch.zeros_like(x)
        if n < 0:
            result[:, :n] = x[:, -n:]
        elif n > 0:
            result[:, -n:] = x[:, :n]
        else:
            result = x
        return result


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





class OuterModel(nn.Module):
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset
    NAME = "edge_regressor"

    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons

        self.GNN = EdgeRegressor(params)
        self.dim_red = Seq(
            Linear(self.n_neurons*self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, params.output_dim*params.output_dim)  
        )
        self.params = params


    def forward(self, x, edge_index):
        high_dim = self.GNN(x, edge_index).flatten()
        #print("High dim shape: ", high_dim.shape)

        flat_low_dim = self.dim_red(high_dim)
        #print("Flat low dim shape: ", flat_low_dim.shape)

        low_dim = torch.reshape(flat_low_dim, (self.params.output_dim, self.params.output_dim))
        #print("Low dim shape: ", low_dim.shape)

        return low_dim


    def makedirs(self, network_type, save_folder, n_remove):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{save_folder}"):
                os.mkdir(f"saved_models/{self.NAME}/{save_folder}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{save_folder}/_remove_{n_remove}"):
                os.mkdir(f"saved_models/{self.NAME}/{save_folder}/_remove_{n_remove}") 

        return f"saved_models/{self.NAME}/{save_folder}_remove_{n_remove}"


    def save(self, network_type, save_folder, filename):   #No longer in use
        # Saves the model to file
        torch.save(self.state_dict(), f"saved_models/{self.NAME}/{filename}")

    
        

