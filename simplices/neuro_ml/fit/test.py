from neuro_ml.dataset import create_dataloaders
import torch
import torch
from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device, simplex_batch_to_device
from zenlog import log
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_graph(self) -> None:    #Use this to plot the graph
        """Plots the graph of the connectivity filter"""
        data = Data(num_nodes=self.W0.shape[0], edge_index=self._edge_index)
        graph = to_networkx(data, remove_self_loops=True)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
        nx.draw(graph, pos, with_labels=True, node_size=20, node_color='red', edge_color='black', arrowsize=5)
        plt.show()


def plot_connectivity(W0):
        """Plots the connectivity filter"""
        palette = seaborn.color_palette("vlag", as_cmap=True)
        W0_scaled = np.tanh(W0.cpu().numpy())
        seaborn.heatmap(W0_scaled, cmap = palette, center = 0, linecolor='black')
        plt.show()


def plot_pred(W0, W0_pred, path, epoch, dataset_params, loss):
        """Plots the connectivity filter"""
        palette = seaborn.color_palette("vlag", as_cmap=True)

        diff = W0 - W0_pred

        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(30, 7)
        ax[0].set_title("True", fontsize=20)
        ax[1].set_title("Predicted", fontsize=20) 
        ax[2].set_title("Difference", fontsize=20)

        fig.suptitle(f"{dataset_params.cluster_sizes}, epoch: {epoch}, loss: {loss:.4f}", fontsize=24)
        seaborn.heatmap(W0.cpu(), cmap = palette, center = 0, ax=ax[0])
        seaborn.heatmap(W0_pred.cpu(), cmap = palette, center = 0, ax=ax[1])
        seaborn.heatmap(diff.cpu(), cmap = palette, center = 0, ax=ax[2])
        plt.savefig(f"{path}/pred_epoch_{epoch}.png") 



def test_model(model, epoch, dataset_params, model_params, model_is_classifier, device):  
    _, _, test_loader = create_dataloaders(
        model.DATASET, dataset_params    
    )

    aggr = "add"

    model = model(model_params)
    model.load_state_dict(torch.load(f"saved_models/{model.NAME}/{dataset_params.network_type}/neurons_{dataset_params.n_neurons}/{aggr}/epoch_{epoch}.pt", map_location=torch.device(device)))
    model.to(device)
    criterion = torch.nn.MSELoss()

    #path = f"saved_models/{model.NAME}/{dataset_params.save_folder}/remove_{dataset_params.neurons_remove}"

    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            (t := tqdm(test_loader, leave=False, colour="#FF5666"))):

            #x, other_inputs, y = batch_to_device(batch, device)

            x, edge_index, y = simplex_batch_to_device(batch, device)

            y_hat = torch.round(model(edge_index, x)) 

            loss = criterion(y_hat, y)
            avg_loss += loss.item()

            t.set_description(f"Test loss: {loss:.4f}/({avg_loss/(batch_idx + 1):.4f})")

            # print(y)
            # print()
            # print(torch.sum(y).item())
            # print()
            # print(y_hat)
            # print()
            # print()
            # print()

        #plot_pred(y, y_hat, path, epoch, dataset_params, loss.item())

    avg_test_loss = avg_loss / len(test_loader)
    log.info(f"Avg test loss: {avg_test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
