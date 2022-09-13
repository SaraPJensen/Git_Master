import os
import torch
import torch.nn.functional as F
from zenlog import log
from neuro_ml import config
from neuro_ml.fit.val import val
from neuro_ml.fit.train import train
from neuro_ml.dataset import create_train_val_dataloaders, create_test_dataloader


def find_mean_variance(data_loader):
    mean = 0
    variance = 0
    for batch in data_loader:
        y = batch[-1]
        mean += y.mean().sum()

    mean /= len(data_loader)

    for batch in data_loader:
        y = batch[-1]
        variance += ((y - mean).pow(2)).sum() / len(y)

    variance /= len(data_loader)

    return mean, variance


def fit(
    model,
    model_is_classifier,
    model_params,
    dataset_params,
    device,
    epochs=config.EPOCHS,
    learing_rate=config.LEARNING_RATE,
):
    train_loader, val_loader = create_train_val_dataloaders(
        model.DATASET, model_is_classifier, dataset_params
    )

    model = model(model_params).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learing_rate)
    criterion = (
        torch.nn.CrossEntropyLoss()
        if model_is_classifier == True
        else torch.nn.MSELoss()
    )

    log.debug(f"Fitting {model.__class__.__name__} on {device}")

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = val(model, val_loader, criterion, device)
        log.info(
            f"{epoch + 1}) train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
        if epoch % 10 == 0:
            model.save(
                f"{dataset_params.n_neurons}_neurons_{dataset_params.timestep_bin_length}_timesteps_{epoch}.pt")
