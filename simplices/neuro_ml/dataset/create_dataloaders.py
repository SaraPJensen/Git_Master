import json
import os
import torch
import torch_geometric
from zenlog import log
from pathlib import Path
from neuro_ml import config
from torch.utils.data import dataloader, random_split, Dataset
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from tqdm import tqdm
from pyflagser import flagser_unweighted


def create_dataloaders(
    Dataset,
    #model_is_classifier,
    dataset_params,
    train_val_test_size=config.TRAIN_VAL_TEST_SIZE,
    seed=config.SEED,
    batch_size=config.BATCH_SIZE,
):
    # Find the dataset path
    dataset_path = config.dataset_path / dataset_params.foldername

    # Load the filenames
    all_filenames = [
        dataset_path / f"{seed}.npz" for seed in range(dataset_params.number_of_files)
    ]

    # for filename in all_filenames:   #Remove datasets where the network exploded, frequency over 50Hz
    #     raw_data = np.load(filename, allow_pickle=True)
    #     raw_x = raw_data["X_sparse"].item()
    #     coo = coo_matrix(raw_x)
    #     values = coo.data
    #     indices = np.vstack((coo.row, coo.col))
    #     i = torch.LongTensor(indices)
    #     v = torch.FloatTensor(values)
    #     shape = coo.shape
    #     X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step
    #     tot_secs = dataset_params.n_timesteps/1000
    #     frequency = torch.sum(X)/tot_secs/sum(dataset_params.cluster_sizes)

    #     if frequency > 50: 
    #         all_filenames.remove(filename)
    
    # print("Datasets remaining after removing exploding datasets: ", len(all_filenames))

    # Split into train, val and test using the train_val_test_size from the config
    train_filenames, val_filenames, test_filenames = random_split(
        all_filenames,
        [
            int(len(all_filenames) * train_val_test_size[0]),
            int(len(all_filenames) * train_val_test_size[1]),
            len(all_filenames)
            - int(len(all_filenames) * train_val_test_size[0])
            - int(len(all_filenames) * train_val_test_size[1]),
        ],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = Dataset(
        train_filenames,
        dataset_params,
        #model_is_classifier,
    )

    val_dataset = Dataset(
        val_filenames,
        dataset_params,
        #model_is_classifier,
    )

    test_dataset = Dataset(
        test_filenames,
        dataset_params,
        #model_is_classifier,
    )

    # Pytorch geometric has a different data loader so if the dataset is geometric we use that
    if Dataset.IS_GEOMETRIC:
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset.create_geometric_data(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

        val_loader = torch_geometric.loader.DataLoader(
            val_dataset.create_geometric_data(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False)

        test_loader = torch_geometric.loader.DataLoader(
            test_dataset.create_geometric_data(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False)


    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    log.info("Data loaders created successfully")

    return (
        train_loader,
        val_loader,
        test_loader,
    )





class Connectivity_Dataset(Dataset):
    def __init__(self, data, output_dim):

        self.output_dim = output_dim
        self._load_x_and_y(data)  #Defines self.X and self.y
        
    
    def _load_x_and_y(self, data):
        """
        Load the dataset
        """
        self.X = []
        self.y = []

        for element in tqdm(
            data,
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):
            W0 = element
            y = np.zeros((self.output_dim))   #For simplicity, only use the first 5 simplices

            flag_dict = flagser_unweighted(W0, max_dimension = self.output_dim-1)     #np.rint(W0).astype(int)

            y[0:len(flag_dict["cell_count"])] = flag_dict["cell_count"]
            
            y = torch.from_numpy(y)

            self.X.append(torch.from_numpy(W0).float())
            self.y.append(y.float())

    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
       





def connectivity_dataloaders(    #Need to add something on how the data is returned 
    dataset_params,
    train_val_test_size=config.TRAIN_VAL_TEST_SIZE,
    seed=config.SEED,
    batch_size=config.BATCH_SIZE):

    connectivity_matrices = []

    for i in range(dataset_params.number_of_files):
        rng = torch.Generator().manual_seed(i)
        upper = nx.to_numpy_array(nx.watts_strogatz_graph(dataset_params.n_neurons, k = dataset_params.n_neurons//3, p = 0.3, seed = rng.seed()))  #This is the upper triangular part of the matrix
        lower = nx.to_numpy_array(nx.watts_strogatz_graph(dataset_params.n_neurons, k = dataset_params.n_neurons//3, p = 0.3, seed = rng.seed()))  #This is the lower triangular part of the matrix
        
        out = np.zeros((dataset_params.n_neurons, dataset_params.n_neurons))
        out[np.triu_indices(dataset_params.n_neurons)] = upper[np.triu_indices(dataset_params.n_neurons)]
        out[np.tril_indices(dataset_params.n_neurons)] = lower[np.tril_indices(dataset_params.n_neurons)]
        W0_graph = nx.from_numpy_array(out, create_using=nx.DiGraph)

        connectivity_matrices.append(nx.to_numpy_array(W0_graph))

    
    train_dataset, val_dataset, test_dataset = random_split(
        connectivity_matrices,
        [int(len(connectivity_matrices) * train_val_test_size[0]),
        int(len(connectivity_matrices) * train_val_test_size[1]),
        len(connectivity_matrices) - int(len(connectivity_matrices) * train_val_test_size[0])- int(len(connectivity_matrices) * train_val_test_size[1]),],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = Connectivity_Dataset(train_dataset, dataset_params.output_dim)
    val_dataset = Connectivity_Dataset(val_dataset, dataset_params.output_dim)
    test_dataset = Connectivity_Dataset(test_dataset, dataset_params.output_dim)


    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


    return (
        train_loader,
        val_loader,
        test_loader,
    )




    