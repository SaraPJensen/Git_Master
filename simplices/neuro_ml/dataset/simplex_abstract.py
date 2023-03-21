import math
import torch
import torch.nn.functional as F
import numpy as np
from pyflagser import flagser_weighted, flagser_unweighted
from neuro_ml.dataset.transforms import Subset
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data




class SimplexAbstract(Dataset):
    def __init__(
        self,   
        filenames,
        dataset_params
    ):

        super().__init__()

        self.neurons = dataset_params.n_neurons
        self.output_dim = dataset_params.output_dim

        self._load_x_and_y(
            filenames,
            dataset_params
        )

    def _load_x_and_y(
        self,
        filenames,
        dataset_params
    ):

        self.W0 = []
        self.edge_index = []
        self.y = []

        for filename in tqdm(
            filenames,
            unit="files",
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):

            all_data = np.load(filename, allow_pickle=True)
            W0 = torch.from_numpy(all_data["W0"]).float()
            global_simplex = all_data["global_simplex"]
            neuron_simplex = all_data["neuron_simplex"]


            if dataset_params.advanced:
            #This is the more complex version, where each node is associated with the number of simplicies in which it is source, sink and mediator
                neuron_simplex = [torch.from_numpy(simplex).flatten() for simplex in neuron_simplex]
                #print(neuron_simplex)

                simplex_tensor = torch.zeros((self.neurons, (self.output_dim + 1) * 3), dtype=torch.float32)
                current_dim = neuron_simplex[0].shape[0]   #highest simplex order * 3

                #print(current_dim)
                
                if current_dim <= self.output_dim + 1:
                    for i in range(self.neurons):
                        simplex_tensor[i, 0:current_dim] = neuron_simplex[i]

                else:
                    for i in range(self.neurons):
                        simplex_tensor[i, 0:(self.output_dim+1)*3] = neuron_simplex[i][0:(self.output_dim+1)*3]

                new = simplex_tensor[:, 3:]   #Ignore the edges


            elif not dataset_params.advanced:   #+2 to only train on the 3-simplicies
                #For simplicity, only take the total number of simplicies each neurons is part of, regardless of role
                simple_neuron_simplex = [torch.sum(torch.from_numpy(simplex), dim = 1) for simplex in neuron_simplex]
                simple_simplex_tensor = torch.zeros((self.neurons, self.output_dim +2))

                current_dim = simple_neuron_simplex[0].shape[0] 

                if current_dim <= self.output_dim +2:
                    for i in range(self.neurons):
                        simple_simplex_tensor[i, 0:current_dim] = simple_neuron_simplex[i]

                else:
                    for i in range(self.neurons):
                        simple_simplex_tensor[i, 0:self.output_dim +2] = simple_neuron_simplex[i][0:self.output_dim +2]

                new = simple_simplex_tensor[:, 2:]   #Ignore the edges


            self.W0.append(W0)
            self.edge_index.append(W0.nonzero().t())
            self.y.append(new)



    def create_geometric_data(self):
        """
        Create a list of torch_geometric.data.Data objects from the dataset
        """
        data = []
        for i in range(len(self)):
            x, edge_index, y = self[i].values()
            data_i = Data(x, edge_index, y=y)
            data.append(data_i)
        return data

        


