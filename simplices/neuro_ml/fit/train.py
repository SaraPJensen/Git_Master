from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device, simplex_batch_to_device
import sklearn.metrics
import torchmetrics
import torch
import numpy as np
from zenlog import log
import scipy.signal


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    avg_loss = 0

    # For each batch in the data loader calculate the loss and update the model
    for batch_idx, batch in enumerate(
        (t := tqdm(data_loader, leave=False, colour="#7FEFBD"))
    ):  

        x, edge_index, y = simplex_batch_to_device(batch, device)

        optimizer.zero_grad()
        y_hat = model(edge_index, x)

        # print("Pred type: ", y_hat.dtype)
        # print("Ground truth type: ", y.dtype)
        # exit()

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        t.set_description(f"Train loss: {loss:.4f}/({avg_loss/(batch_idx + 1):.4f})")

        # if batch_idx == 100: 
        #     print("Prediction: ", y_hat.int())
        #     print("Truth: ", y.int())
        #     print()



    return avg_loss / len(data_loader)
