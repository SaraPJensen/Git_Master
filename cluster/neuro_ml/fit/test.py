from neuro_ml.dataset import create_test_dataloader
import torch
import torch
from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device
from zenlog import log
import seaborn
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(self) -> None:    #Use this to plot the graph!!! 
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


#This is never used 
def test_model(model, epoch, dataset_params, model_params, model_is_classifier, device):  
    test_loader = create_test_dataloader(
        model.DATASET,
        dataset_params,
        model_is_classifier,
    )

    model = model(model_params)
    model.load_state_dict(torch.load(f"models/{model.NAME}/{dataset_params.foldername}_{epoch}.pt"))
    model.to(device)
    criterion = torch.nn.MSELoss()

    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            (t := tqdm(test_loader, leave=False, colour="#FF5666"))
        ):
            x, other_inputs, y = batch_to_device(batch, device)

            y_hat = model(x, other_inputs) 

            loss = criterion(y_hat, y)

            avg_loss += loss.item()

    plot_connectivity(y)
    plot_connectivity(y_hat)

    avg_test_loss = avg_loss / len(test_loader)
    log.info(f"Avg test loss: {avg_test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
