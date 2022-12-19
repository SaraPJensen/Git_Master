import os
import torch
import torch.nn.functional as F
from zenlog import log
from neuro_ml import config
from neuro_ml.fit.val import val
from neuro_ml.fit.train import train
from neuro_ml.dataset import create_dataloaders, connectivity_dataloaders



def fit(
    model,
    model_is_classifier,
    model_params,
    dataset_params,
    device,
    epochs=config.EPOCHS,
    learing_rate=config.LEARNING_RATE,
):
    #Prepare data loaders for training and validation
    train_loader, val_loader, _ = create_dataloaders(
        model.DATASET, dataset_params    #This is the only one which needs the initial number of neurons
    )

    #train_loader, val_loader, _ = connectivity_dataloaders(dataset_params)

    # Initialize model
    model = model(model_params).to(device)

    # Set optimizer and criterion
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learing_rate)
    criterion = (
        torch.nn.CrossEntropy()
        if model_is_classifier == True
        else torch.nn.MSELoss()
    )

    #print(model)

    log.debug(f"Fitting {model.__class__.__name__} on {device}")

    #loss_folder = model.makedirs(dataset_params.save_folder)
    loss_folder = model.makedirs(dataset_params.network_type)
    file = open(f"{loss_folder}/loss.csv", "w")
    file.write("epoch,train_loss,val_loss\n")
    file.close()

    best_loss = 20000

    # For each epoch calculate training and validation 
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = val(model, val_loader, criterion, device)
        log.info(
            f"{epoch}) train : {train_loss:.4f}, val : {val_loss:.4f}")

        file = open(f"{loss_folder}/loss.csv", "a")
        file.write(f"{epoch},{train_loss},{val_loss}\n")
        file.close()

        best_filename = loss_folder + f"/epoch_best.pt"
        #f"saved_models/{self.NAME}/{save_folder}"

        if val_loss < best_loss:
            #torch.save(model.state_dict(), f"{loss_folder}/epoch_best.pt")
            model.save(best_filename)
            best_epoch = epoch
            best_loss = val_loss

        # if (epoch % 20) == 0 or epoch == epochs:
        #     #torch.save(model.state_dict(), f"{loss_folder}/epoch_{epoch}.pt")

        #     filename = loss_folder + f"/epoch_{epoch}.pt"
        #     model.save(filename)

    print(f"Best validation obtained in epoch {best_epoch} with {best_loss:.4f}.")
