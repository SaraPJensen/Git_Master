import json
import os
import torch
import torch_geometric
from zenlog import log
from pathlib import Path
from neuro_ml import config
from torch.utils.data import dataloader, random_split


def _parse_json_simulation_params(dataset_path, max_number_of_files):
    with open(dataset_path / "simulation-params.json") as f:
        params = json.load(f)

    number_of_neurons = params["n_neurons"]
    total_number_of_timesteps = params["n_timesteps"]

    return (
        number_of_neurons,
        total_number_of_timesteps,
    )


def _train_val_test_split_filenames(all_filenames, train_val_test_size, seed):
    assert sum(train_val_test_size) == 1, "Sum of the train/val/test size must be 0"
    assert all(
        0 <= split_size for split_size in train_val_test_size
    ), "All objects of train/val/test size must be non-zero"

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

    return train_filenames, val_filenames, test_filenames


def create_train_val_dataloaders(
    Dataset,
    model_is_classifier,
    dataset_params,
    train_val_test_size=config.TRAIN_VAL_TEST_SIZE,
    seed=config.SEED,
    batch_size=config.BATCH_SIZE,
):
    dataset_path = config.dataset_path / dataset_params.foldername

    all_filenames = [
        dataset_path / f"{seed}.npz" for seed in range(dataset_params.number_of_files)
    ]

    train_filenames, val_filenames, _ = random_split(
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
        model_is_classifier,
    )

    val_dataset = Dataset(
        val_filenames,
        dataset_params,
        model_is_classifier,
    )

    if Dataset.IS_GEOMETRIC:
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset.create_geometric_data(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = torch_geometric.loader.DataLoader(
            val_dataset.create_geometric_data(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

    log.info("Data loaders created successfully")

    return (
        train_loader,
        val_loader,
    )


def create_test_dataloader(
    Dataset,
    dataset_params,
    model_is_classifier,
    train_val_test_size=config.TRAIN_VAL_TEST_SIZE,
    seed=config.SEED,
    batch_size=config.BATCH_SIZE,
):
    dataset_path = config.dataset_path / dataset_params.foldername

    all_filenames = [
        dataset_path / f"{seed}.npz" for seed in range(len(os.listdir(dataset_path)))
    ]

    _, _, test_filenames = random_split(
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

    test_dataset = Dataset(
        test_filenames,
        dataset_params,
        model_is_classifier,
    )

    if Dataset.IS_GEOMETRIC:
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset.create_geometric_data(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
    return test_loader
