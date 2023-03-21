import math
import torch
import torch.nn.functional as F
import numpy as np
from neuro_ml.dataset.transforms import Subset, Simplex_filtering, Weight_filtering
from neuro_ml.dataset.torch_flagser import flagser_count_unweighted 
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from neo.core import SpikeTrain
import pickle



class AbstractDataset(Dataset):
    def __init__(
        self,
        filenames,
        dataset_params,
        model_is_classifier,
    ) -> None:
        super().__init__()

        self.n_remaining = dataset_params.n_neurons_remaining
        self.n_initial = dataset_params.n_neurons

        self.subset = Subset(dataset_params.neurons_remove, dataset_params.n_neurons)
        self.w_filtering = Weight_filtering(dataset_params.weight_threshold)   
        self.s_filtering = Simplex_filtering(dataset_params.simplex_threshold)   

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
        self.max_complex_edges = []

        for filename in tqdm(
            filenames,
            unit="files",
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):
            # Convert sparse X to dense
            with open(filename, "rb") as f:
                stuff = pickle.load(f)   #stuff = dictionary

            X_sparse = stuff["X_sparse"]

            coo = coo_matrix(X_sparse)
            values = coo.data
            indices = np.vstack((coo.row, coo.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape

            X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step
            y = stuff["W0"].float()   #W0 has shape (n_neurons, n_neurons)

            global_simplex_count = stuff["global_simplex_count"]   #Number of simplices above a certain dimension
            neuron_simplex_count = stuff["neuron_simplex_count"]   #Number of simplices above a certain dimension for each neuron
            Maximal_complex_edges = stuff["Maximal_complex_edges"]   #Number of edges in the maximal complex

            x, y = self.subset(X, y)   #Remove neurons from X and W0
            y = self.w_filtering(y)    #Remove edges with a weight below a certain threshold
            y = self.s_filtering(y, Maximal_complex_edges)    #Remove edges which are not in the simplex of a certain dimension

            self.X.append(x)
            self.y.append(y)
            self.max_complex_edges.append(Maximal_complex_edges)





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


    def _create_fully_connected_edge_index(self, n_neurons):   #n_neurons should be n_neurons
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
            edge_index = torch.ones(n_neurons, n_neurons)  #Set all the connections to 1
            self.edge_index.append(edge_index.nonzero().T) #Returns the indices of the non-zero elements, i.e. all of them in this case


    def create_geometric_data(self):  
        """
        Create a list of torch_geometric.data.Data objects from the dataset
        """
        data = []
        for i in range(len(self)):
            inputs, y, max_complex_edges = self[i].values()
            data_i = Data(inputs["X"], inputs["edge_index"], y=y, max_complex_edges=max_complex_edges)
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
