import os
import torch
import torch.nn.functional as F
from zenlog import log
from neuro_ml import config
from neuro_ml.fit.val import val
from neuro_ml.fit.train import train
from neuro_ml.dataset import create_dataloaders



class ScaledMSELoss(torch.nn.Module):
    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        non_zero = target != 0
        non_zero = non_zero.float()
        masked = pred * non_zero
        mse_add = torch.mean((masked - target) ** 2)
        loss = torch.sum(mse + 0.5*mse_add)  #0.5 is the weight of the additional loss, arbitrarily chosen to be 0.5

        return loss




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
    train_loader, val_loader, test_loader = create_dataloaders(
        model.DATASET, model_is_classifier, dataset_params    #This is the only one which needs the initial number of neurons
    )

    # Initialize model
    torch.manual_seed(config.SEED)
    model = model(model_params).to(device)

    # Set optimizer and criterion
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learing_rate)
    criterion = (
        torch.nn.CrossEntropyLoss()
        if model_is_classifier == True
        else ScaledMSELoss()
        if dataset_params.scaled_loss == True
        else torch.nn.MSELoss()
    )

    log.debug(f"Fitting {model.__class__.__name__} on {device}")

    loss_folder = model.makedirs(dataset_params.network_type, dataset_params.save_folder, dataset_params.neurons_remove)
    file = open(f"{loss_folder}/loss.csv", "w")
    file.write("epoch,train_loss,train_loss_std,val_loss,val_loss_std,test_loss,test_loss_std\n")
    file.close()

    best_loss = 1

    # For each epoch calculate training and validation loss
    for epoch in range(1, epochs+1):
        train_loss, train_loss_std = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_loss_std = val(model, val_loader, criterion, device)

        test_loss, test_loss_std = val(model, test_loader, criterion, device)

        log.info(
            f"{epoch}) train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        file = open(f"{loss_folder}/loss.csv", "a")
        file.write(f"{epoch},{train_loss},{train_loss_std},{val_loss},{val_loss_std},{test_loss},{test_loss_std}\n")
        file.close()

        if val_loss < best_loss:
            torch.save(model.state_dict(), f"{loss_folder}/epoch_best.pt")
            best_epoch = epoch
            best_loss = val_loss
            #print("Saved model in epoch ", best_epoch)

        if epoch == epochs:
            torch.save(model.state_dict(), f"{loss_folder}/epoch_{epoch}.pt")

    print(f"Best validation loss obtained in epoch {best_epoch} with {best_loss:.4f}.")
