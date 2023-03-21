import torch
from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device, simplex_batch_to_device
import sklearn.metrics


def val(model, data_loader, criterion, device, output_dim):
    model.eval()
    avg_loss = 0
    avg_loss_2s = 0
    avg_loss_3s = 0
    with torch.no_grad():
        # For each batch in the data loader calculate the validation loss
        for batch_idx, batch in enumerate(
            (t := tqdm(data_loader, leave=False, colour="#FF5666"))
        ):
            #x, other_inputs, y = batch_to_device(batch, device)
            x, edge_index, y = simplex_batch_to_device(batch, device)

            y_hat = model(edge_index, x)

            y_hat = torch.round(y_hat)   #We're looking for integers, so makes sense to round the validation predictions

            if output_dim == 2:
                y_hat_2s = y_hat[:,0]
                y_hat_3s = y_hat[:,1]

                y_2s = y[:,0]
                y_3s = y[:,1]

                loss_2s = criterion(y_hat_2s, y_2s)
                loss_3s = criterion(y_hat_3s, y_3s)

                avg_loss_2s += loss_2s.item()
                avg_loss_3s += loss_3s.item()

                t.set_description(f"Val_2s loss: {loss_2s:.4f}/({avg_loss_2s/(batch_idx + 1):.4f}), Val_3s loss: {loss_3s:.4f}/({avg_loss_3s/(batch_idx + 1):.4f})")


            else: 
                loss = criterion(y_hat, y)
                avg_loss += loss.item()
                t.set_description(f"Val loss: {loss:.4f}/({avg_loss/(batch_idx + 1):.4f})")


    if output_dim == 2:
        return (avg_loss_2s+avg_loss_3s)/len(data_loader), avg_loss_2s/len(data_loader), avg_loss_3s/len(data_loader)

    else:
        return avg_loss / len(data_loader)
    