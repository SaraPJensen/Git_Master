import json
import os
import torch
import torch_geometric
from zenlog import log
from pathlib import Path
from neuro_ml import config
from torch.utils.data import dataloader, random_split
import numpy as np
from scipy.sparse import coo_matrix
import pickle


def create_dataloaders(
    Dataset,
    model_is_classifier,
    dataset_params,
    train_val_test_size=config.TRAIN_VAL_TEST_SIZE,
    seed=config.SEED,
    batch_size=config.BATCH_SIZE,
):
    # Find the dataset path
    dataset_path = config.dataset_path / dataset_params.foldername

    # Load the filenames
    all_filenames = [
        dataset_path / f"{seed}.pkl" for seed in range(dataset_params.number_of_files)
    ]

    for filename in all_filenames:   #Remove datasets where the network exploded, frequency over 50Hz
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
        tot_secs = dataset_params.n_timesteps/1000
        frequency = torch.sum(X)/tot_secs/sum(dataset_params.cluster_sizes)

        if frequency > 50: 
            all_filenames.remove(filename)

        #print("Frequency: ", frequency)
    
    print("Datasets remaining after removing exploding datasets: ", len(all_filenames))

    # Split into train, val and test using the train_val_test_size from the config
    torch.manual_seed(config.SEED)
    train_filenames, val_filenames, test_filenames = random_split(
        all_filenames,
        [
            int(len(all_filenames) * train_val_test_size[0]),
            int(len(all_filenames) * train_val_test_size[1]),
            len(all_filenames)
            - int(len(all_filenames) * train_val_test_size[0])
            - int(len(all_filenames) * train_val_test_size[1]),
        ])

    train_dataset = Dataset(
        train_filenames,
        dataset_params,
        model_is_classifier,
    )

    val_dataset = Dataset(
        val_filenames,
        dataset_params,
        model_is_classifier,
    )

    test_dataset = Dataset(
        test_filenames,
        dataset_params,
        model_is_classifier,
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


