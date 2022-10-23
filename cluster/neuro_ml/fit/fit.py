import os
import torch
import torch.nn.functional as F
from zenlog import log
from neuro_ml import config
from neuro_ml.fit.val import val
from neuro_ml.fit.train import train
from neuro_ml.dataset import create_train_val_dataloaders, create_test_dataloader



def fit(
    model,
    model_is_classifier,
    model_params,
    dataset_params,
    device,
    epochs=config.EPOCHS,
    learing_rate=config.LEARNING_RATE,
):
    # Prepare data loaders for training and validation
    train_loader, val_loader = create_train_val_dataloaders(
        model.DATASET, model_is_classifier, dataset_params
    )

    # Initialize model
    model = model(model_params).to(device)

    # Set optimizer and criterion
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learing_rate)
    criterion = (
        torch.nn.CrossEntropyLoss()
        if model_is_classifier == True
        else torch.nn.MSELoss()
    )

    log.debug(f"Fitting {model.__class__.__name__} on {device}")

    loss_name = model.makedirs(dataset_params.network_type, dataset_params.foldername)
    file = open(f"{loss_name}/loss.csv", "w")
    file.write("epoch,train_loss,val_loss\n")
    file.close()

    # For each epoch calculate training and validation loss
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = val(model, val_loader, criterion, device)
        log.info(
            f"{epoch}) train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        file = open(f"{loss_name}/loss.csv", "a")
        file.write(f"{epoch},{train_loss},{val_loss}\n")
        file.close()

        if (epoch % 20) == 0 or epoch == epochs:
            filename = dataset_params.foldername + f"/epoch_{epoch}.pt"
            model.save(dataset_params.network_type, dataset_params.foldername, filename)

