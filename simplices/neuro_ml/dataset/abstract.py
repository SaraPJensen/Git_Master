import math
import torch
import torch.nn.functional as F
import numpy as np
from pyflagser import flagser_weighted, flagser_unweighted
from neuro_ml.dataset.transforms import Subset
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

        self.output_dim = dataset_params.output_dim
        self.n_remaining = dataset_params.n_neurons_remaining
        self.n_initial = dataset_params.n_neurons
        self.transform = Subset(dataset_params.neurons_remove, dataset_params.n_neurons)

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

            #y = np.zeros((20))  #Elements 0-9 the betti numbers, elements 10-19 the simplex counts
            y = np.zeros((self.output_dim))   #For simplicity, only use the first 5 simplices

            test = np.array([[-1, 0, 0, 0],
                            [1, 1, 1, 0],
                            [1, 0, 0.8, 0],
                            [0, -1, 1, 1]])

            W0[W0 != 0] = 1    #Set all non-zero values of W0 to 1, binarise
            
            flag_dict = flagser_unweighted(W0, max_dimension = self.output_dim-1)     #np.rint(W0).astype(int)


            y[0:len(flag_dict["cell_count"])] = flag_dict["cell_count"]
            
            #y[0:len(flag_dict["betti"])] = flag_dict["betti"]
            #y[10:10+len(flag_dict["cell_count"])] = flag_dict["cell_count"]    #They all have the same number of edges

            y = torch.from_numpy(y)

            # print()
            # print("Betti numbers: ")
            # print(flag_dict["betti"])
            # print()

            # print("Simplices: ")
            #print(flag_dict["cell_count"])
            # print()

            # print("Euler characteristic: ", flag_dict["euler"])
            # print()
                
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
                    #x, y = self.transform(x_slice.float(), y.float())
                    #self.X.append(x_slice.float())

                    self.X.append(torch.from_numpy(W0).float())  #use this as input instead to find a network able to learn from the connectivity matrix
                    self.y.append(y.float())


    def _create_edge_indices(self, neurons_remaining):   #Not in use, this would be cheating
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
            y = y.reshape(neurons_remaining, neurons_remaining)   #Presumably its already of this shape? 
            edge_index = torch.nonzero(y)  #Returns the indices of the non-zero elements

            self.edge_index.append(edge_index.T)  #Initial hypothesis for W_0, this is G' 


    def _create_fully_connected_edge_index(self, neurons_remaining):   #neurons_remaining should be n_neurons
        """
        For each simulation in the dataset create a fully connected edge index
        """
        #this is of length 70 (no. of training samples), each element is a 2xN tensor, where N is the number of edges in the graph, one layer for each direction
        #So all nodes are initially assumed to be bidirectionally connected to all nodes
        
        # The edge index must include all the neurons

        self.edge_index = []   
        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            edge_index = torch.ones(neurons_remaining, neurons_remaining)  #Set all the connections to 1
            self.edge_index.append(edge_index.nonzero().T) #Returns the indices of the non-zero elements, i.e. all of them in this case


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
