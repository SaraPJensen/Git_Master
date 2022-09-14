import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset
import os 

@dataclass
class EdgeRegressorParams:
    n_shifts: int = 10
    n_neurons: int = 20

class EdgeRegressor(MessagePassing):
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset
    NAME = "edge_regressor"

    def __init__(self, params):
        super().__init__()
        self.n_shifts = params.n_shifts
        self.selection_matrix = torch.eye(20).repeat_interleave(20, dim=0)
        self.mlp1 = Seq(
            Linear(params.n_shifts, params.n_neurons),
            ReLU(),
            Linear(params.n_neurons, 10*params.n_neurons),
            ReLU(),
            Linear(10*params.n_neurons, params.n_neurons),
            ReLU(),
            Linear(params.n_neurons, 1)
            )
        self.mlp2 = Seq(
            Linear(params.n_neurons, 10*params.n_neurons),
            ReLU(),
            Linear(10*params.n_neurons, params.n_neurons)
        )

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        inner_products = [torch.sum(x_i*self.shift(x_j, -(t+1)), dim=1).unsqueeze(dim=1) for t in range(self.n_shifts)]

        tmp = torch.cat(inner_products, dim=1)

        tmp = tmp / (x_i.shape[1]/100) #F.normalize(tmp, dim=1)

        return self.mlp1(tmp)*(self.selection_matrix.to(x_i.device).repeat(int(x_i.shape[0] / 400), 1))

    def update(self, inputs):
        return self.mlp2(inputs)

    def shift(self, x, n):
        result = torch.zeros_like(x)
        if n < 0:
            result[:, :n] = x[:, -n:]
        elif n > 0:
            result[:, -n:] = x[:, :n]
        else:
            result = x
        return result

    def save(self, filename):
        print("Trying to save in edge regressor, filename: ", filename)
        savepath = f"models/{self.NAME}"
        if not os.path.exists(savepath):  ##This is where it craches, test further tomorrow
            print("Current position: ", os.getcwd())
            os.makedirs(savepath)
            print("Made directory")

        torch.save(self.state_dict(), f"models/{self.NAME}/{filename}")
        print("Saved model in edge regressor") 
