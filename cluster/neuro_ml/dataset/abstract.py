import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy.sparse import coo_matrix


class AbstractDataset(Dataset):
    def __init__(
        self,
        filenames,
        dataset_params,
        model_is_classifier,
    ) -> None:
        super().__init__()

        self._load_x_and_y(   #Defines self.X and self.y
            filenames,
            dataset_params,
            model_is_classifier,
        )

    def __len__(self):
        raise NotImplementedError(
            "Length of dataset is not implemented for abstract dataset"
        )

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Getting item is not implemented for abstract dataset"
        )

    def _load_x_and_y(
        self,
        filenames,
        dataset_params,
        model_is_classifier,
    ):
        """
        Load the dataset
        """
        self.X = []
        self.y = []

        for filename in tqdm(
            filenames,
            unit="files",
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):
            # Convert sparse X to dense
            raw_data = np.load(filename, allow_pickle=True)
            raw_x = raw_data["X_sparse"].item()
            W0 = raw_data["W0"] 

            W0_hubs = raw_data["W0_hubs"]
            edge_index_hubs = raw_data["edge_index_hubs"]

            # for key in raw_data.keys():
            #     print(key)                  

            coo = coo_matrix(raw_x)
            values = coo.data
            indices = np.vstack((coo.row, coo.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape

            X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step

            # If model is a classifier, one-hot encode the weight matrix (the connectivity matrix, the ground truth), shape [n_neurons, n_neurons]
            y = (    
                self.one_hot(torch.tensor(W0))
                if model_is_classifier
                else torch.tensor(W0)
            )

            # print("X shape: ", X.shape)
            # print("W shape: ", y.shape)
            # print("W type: ", type(y))

            # Cut X into windows of length timestep_bin_length and append
            for i in range(
                math.floor(
                    dataset_params.n_timesteps / dataset_params.timestep_bin_length   #These are usually kept the same
                )
            ):
                x_slice = X[
                    :,
                    dataset_params.timestep_bin_length
                    * i: dataset_params.timestep_bin_length
                    * (i + 1),
                ]
                if x_slice.any():
                    self.X.append(x_slice.float())
                    self.y.append(y.float())

    def _create_edge_indices(self, n_neurons):
        """
        For each simulation in the dataset create an edge index based on the non-zero elements of W_0
        """
        self.edge_index = []

        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            y = y.reshape(n_neurons, n_neurons)   #Presumably its already of this shape? 
            edge_index = torch.nonzero(y)  #Returns the indices of the non-zero elements

            self.edge_index.append(edge_index.T)  #Initial hypothesis for W_0, this is G' 

    def _create_fully_connected_edge_index(self, n_neurons):
        """
        For each simulation in the dataset create a fully connected edge index
        """
        #this is of length 70 (no. of training samples), each element is a 2xN tensor, where N is the number of edges in the graph, one layer for each direction
        #So all nodes are bidirectionally connected to all nodes 
        self.edge_index = []   
        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            edge_index = torch.ones(n_neurons, n_neurons)  #Set all the connections to 1
            self.edge_index.append(edge_index.nonzero().T) #Returns the indices of the non-zero elements, i.e. all of them in this case
        
        # print("Initial edge index length: ", len(self.edge_index))
        # print("edge index slice shape: ", self.edge_index[0].shape)
        # print("Edge index slice: ", self.edge_index[0])

    def create_geometric_data(self):   #Somehow in use...
        """
        Create a list of torch_geometric.data.Data objects from the dataset
        """
        data = []
        for i in range(len(self)):
            inputs, y = self[i].values()
            data_i = Data(inputs["X"], inputs["edge_index"], y=y)
            data.append(data_i)
        return data

    def to_binary(self, y): 
        """
        Create a binary representation of the weight matrix
        """
        zeros = torch.zeros(len(y))
        ones = torch.ones(len(y))
        y = torch.where(y == 0, zeros, ones)
        return y

    def one_hot(self, y):  
        """
        Create a one-hot representation of the weight matrix
        """
        return F.one_hot((y.sign() + 1).to(torch.int64))
