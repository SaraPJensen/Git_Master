from neuro_ml.dataset import create_dataloaders
import torch
import torch
from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device
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

        network_name = dataset_params.network_type

        diff = W0 - W0_pred

        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(30, 7)
        ax[0].set_title("True", fontsize=20)
        ax[1].set_title("Predicted", fontsize=20) 
        ax[2].set_title("Difference", fontsize=20)

        fig.suptitle(f"{network_name} network, {dataset_params.cluster_sizes} neurons, epoch: {epoch}, loss: {loss:.4f}", fontsize=24)
        seaborn.heatmap(W0.cpu(), cmap = palette, center = 0, ax=ax[0])
        seaborn.heatmap(W0_pred.cpu(), cmap = palette, center = 0, ax=ax[1])
        seaborn.heatmap(diff.cpu(), cmap = palette, center = 0, ax=ax[2])
        plt.savefig(f"{path}/{network_name}_{dataset_params.cluster_sizes}_pred_epoch_{epoch}.png") 






def test_model(model, epoch, dataset_params, model_params, model_is_classifier, device):  
    _, _, test_loader = create_dataloaders(
        model.DATASET, 
        model_is_classifier,
        dataset_params, 
    )

    model = model(model_params)
    model.load_state_dict(torch.load(f"saved_models/{model.NAME}/{dataset_params.save_folder}/remove_{dataset_params.neurons_remove}/epoch_{epoch}.pt", map_location=torch.device(device)))
    model.to(device)
    criterion = torch.nn.MSELoss()

    path = f"saved_models/{model.NAME}/{dataset_params.save_folder}/remove_{dataset_params.neurons_remove}"

    model.eval()
    avg_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            (t := tqdm(test_loader, leave=False, colour="#FF5666"))):

            # print(batch)
            # exit()

            x, other_inputs, y, max_complex_edges = batch_to_device(batch, device)

            y_hat = model(x, other_inputs) 

            loss = criterion(y_hat, y)
            avg_loss += loss.item()

            t.set_description(f"Test loss: {loss:.4f}/({avg_loss/(batch_idx + 1):.4f})")

        plot_pred(y, y_hat, path, epoch, dataset_params, loss.item())

    avg_test_loss = avg_loss / len(test_loader)
    log.info(f"Avg test loss: {avg_test_loss}")








class Simplex_filtering(object):
    """
    Remove edges which do not form part of any simplex above a certain dimension
    """

    def __init__(self, simplex_threshold):
        self.threshold = simplex_threshold

    def __call__(self, W0, Maximal_complex_edges):

        if self.threshold < 2:
            return W0

        W0_filter = torch.zeros((W0.shape[0], W0.shape[1]))

        if len(Maximal_complex_edges) > self.threshold - 2:  #First element are the 1-simplices, second element are the 2-simplices etc. To keep the 2-simplices (threshold = 3), start at index 1
            remaining = Maximal_complex_edges[self.threshold - 2:]
            for dim in remaining:
                for edge in dim:
                    W0_filter[edge[0], edge[1]] = W0[edge[0], edge[1]]

        return W0_filter
            



def simplex_filter(threshold, max_complex_edges, pred, true):
    pred_filter = torch.zeros((pred.shape[0], pred.shape[1]))
    true_filter = torch.zeros((true.shape[0], true.shape[1]))

    if threshold < 2:
            return pred, true

    if len(max_complex_edges) > threshold - 2:  #First element are the 1-simplices, second element are the 2-simplices etc. To keep the 2-simplices (threshold = 3), start at index 1
        remaining = max_complex_edges[threshold - 2:]
        for dim in remaining:
            for edge in dim:
                pred_filter[edge[0], edge[1]] = pred[edge[0], edge[1]]
                true_filter[edge[0], edge[1]] = true[edge[0], edge[1]]

    return pred_filter, true_filter



def MSE_simplex(pred, true):
    loss = torch.sum((pred - true)**2)
    #print(true)
    non_zero = torch.sum(true != 0)

    if non_zero == 0:
        loss = torch.tensor(0.0, dtype=torch.float32)
        non_zero = torch.tensor(1.0, dtype=torch.float32)

    #print(f"loss: {loss/non_zero}, non_zero: {non_zero}")

    return loss/non_zero







def simplex_test(model, epoch, dataset_params, model_is_classifier, model_params, device, max_threshold):
    _, _, test_loader = create_dataloaders(
        model.DATASET,
        model_is_classifier,
        dataset_params,
    )

    model = model(model_params)
    model.load_state_dict(torch.load(f"saved_models/{model.NAME}/{dataset_params.save_folder}/remove_{dataset_params.neurons_remove}/epoch_{epoch}.pt", map_location=torch.device(device)))
    model.to(device)

    path = f"saved_models/{model.NAME}/{dataset_params.save_folder}/remove_{dataset_params.neurons_remove}"

    model.eval()
    avg_loss = np.zeros((max_threshold, 2))


    with torch.no_grad():
        for batch_idx, batch in enumerate(
            (t := tqdm(test_loader, leave=False, colour="#FF5666"))):

            x, other_inputs, y, max_complex_edges = batch_to_device(batch, device)

            y_hat = model(x, other_inputs)

            for threshold in range(max_threshold):
                #print("Threshold: ", threshold)
                #print(y)
                y_hat_filter, y_filter = simplex_filter(threshold, max_complex_edges, y_hat, y)
                #print(y_filter)
                #input()
                loss = MSE_simplex(y_hat_filter, y_filter)
                avg_loss[threshold, 0] += loss.item()
                if loss.item()  != 0:
                    avg_loss[threshold, 1] += 1  #Count number of non-zero losses, to ignore the zero losses when calculating the average
    

            #t.set_description(f"Test loss: {loss:.4f}/({avg_loss/(batch_idx + 1):.4f})")

    for threshold in range(max_threshold):
        avg_loss[threshold, 0] = avg_loss[threshold, 0] / avg_loss[threshold, 1]  
        print(f"Avg test loss for threshold {threshold}: {avg_loss[threshold, 0]}")

    print()
    print(avg_loss)

        #plot_pred(y, y_hat, path, epoch, dataset_params, loss.item())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
