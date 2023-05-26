import numpy as np
import pandas as pd
import plotly as py

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, path

def add_opacity(color, opacity):
    return f'rgba{color[3:-1]}, {opacity})'

my_colours = []

for colour in reversed(px.colors.qualitative.Bold):
    my_colours.append(colour)

for colour in reversed(px.colors.qualitative.Vivid):
    my_colours.append(colour)

translucent_colours = [add_opacity(c, 0.15) for c in my_colours]




def plot_loss_remove(network, cluster_size: int, neurons_removed: int, simplex_filter: int, epochs: int, error: bool):

    fig = go.Figure()

    iteration = neurons_removed
    if error:
        iteration = reversed(iteration)

    for idx, removed in enumerate(iteration):

        data_path = f"../saved_models/edge_regressor/{network}/cluster_sizes_[{cluster_size}]_n_steps_200000_simplex_threshold_{simplex_filter}/remove_{removed}/loss.csv"

        df = pd.read_csv(data_path)

        train_loss = df["train_loss"][0:epochs]
        val_loss = df["val_loss"][0:epochs]

        if error: 
            val_loss_std = df["val_loss_std"][0:epochs]/np.sqrt(30)

        fig.add_trace(go.Scatter(y=val_loss, name=f"{removed}", line=dict(width=3, color=my_colours[idx+6])))

        if error:
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = val_loss + val_loss_std,
                mode = 'lines',
                marker = dict(color=my_colours[idx+6]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                name = "Lower bound",
                y = val_loss - val_loss_std,
                marker = dict(color=my_colours[idx+6]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = add_opacity(my_colours[idx+6], 0.3),
                fill = 'tonexty',
                showlegend=False
            ))

    if network == "random":
        network_name = "random"
    elif network == "small_world":
        network_name = "small-world"

    fig.update_layout(
        title=f"Validation loss for {network_name} network,<br>{cluster_size} neurons, with neurons removed",
        legend_title = "Removed",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        font_family = "Garamond",
        font_size = 15)

    fig.write_image(f"figures/{network}_{cluster_size}_neurons_removed_error_{error}.pdf")

neurons_removed = range(0, 10)


plot_loss_remove("random", 30, neurons_removed, 0, 15, True)
plot_loss_remove("small_world", 25, neurons_removed, 0, 15, True)