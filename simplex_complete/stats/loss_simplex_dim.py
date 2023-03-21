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




def plot_loss_filtering(network, cluster_size: int, neurons_removed: int, simplex_threshold: list, epochs: int, error: bool):

    fig = go.Figure()

    iteration = simplex_threshold
    if error:
        iteration = reversed(iteration)
    
    for idx, threshold in enumerate(iteration):
        data_path = f"../saved_models/edge_regressor/{network}/cluster_sizes_[{cluster_size}]_n_steps_200000_simplex_threshold_{threshold}/remove_{neurons_removed}/loss.csv"

        df = pd.read_csv(data_path)

        train_loss = df["train_loss"][0:epochs]
        val_loss = df["val_loss"][0:epochs]

        train_loss_std = df["train_loss_std"][0:epochs]
        val_loss_std = df["val_loss_std"][0:epochs]

        fig.add_trace(go.Scatter(y=train_loss, name=f"Train - {threshold}", line=dict(width=3, color=my_colours[idx+6], dash="dash")))

        fig.add_trace(go.Scatter(y=val_loss, name=f"Val. - {threshold}", line=dict(width=3, color=my_colours[idx+6])))

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
                    fillcolor = translucent_colours[idx+6], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )


            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = train_loss + train_loss_std,
                mode = 'lines',
                marker = dict(color=my_colours[idx+6]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = train_loss - train_loss_std,
                    marker = dict(color=my_colours[idx+6]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[idx+6], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    if network == "random":
        network_name = "random"
    elif network == "small_world":
        network_name = "small-world"

    fig.update_layout(
        title=f"Loss for {network_name} network, {cluster_size} neurons,<br>with simplex thresholding",
        legend_title = "Loss - Threshold",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        font_family = "Garamond",
        font_size = 15)

    fig.write_image(f"figures/{network}_{cluster_size}_simplex_threshold_error_{error}.pdf")



filter_list = [0, 3, 4, 5, 6]

plot_loss_filtering("random", 60, 0, filter_list, 15, True)
plot_loss_filtering("small_world", 60, 0, filter_list, 15, True)







def plot_validation_filtering(network, cluster_size: int, neurons_removed: int, simplex_threshold: list, epochs: int, error: bool):

    fig = go.Figure()
    iteration = simplex_threshold
    if error:
        iteration = reversed(iteration)
    
    for idx, threshold in enumerate(iteration):
        data_path = f"../saved_models/edge_regressor/{network}/cluster_sizes_[{cluster_size}]_n_steps_200000_simplex_threshold_{threshold}/remove_{neurons_removed}/loss.csv"

        df = pd.read_csv(data_path)

        val_loss = df["val_loss"][0:epochs]
        val_loss_std = df["val_loss_std"][0:epochs]

        fig.add_trace(go.Scatter(y=val_loss, name=f"{threshold}", line=dict(width=3, color=my_colours[idx+6])))

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
                    fillcolor = translucent_colours[idx+6], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    if network == "random":
        network_name = "random"
    elif network == "small_world":
        network_name = "small-world"

    fig.update_layout(
        title=f"Validation loss for {network_name} network,<br>{cluster_size} neurons, with simplex thresholding",
        legend_title = "Threshold",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        font_family = "Garamond",
        font_size = 15)

    fig.write_image(f"figures/{network}_{cluster_size}_validation_simplex_threshold_error_{error}.pdf")


#plot_loss_filtering_error("random", 60, 0, filter_list, 20)
#plot_loss_filtering_error("small_world", 70, 0, filter_list, 20)


# filter_list = [0, 3, 4, 5]
# plot_loss_filtering_error("random", 30, 0, filter_list, 20)

filter_list = [0, 3, 4, 5, 6]

plot_validation_filtering("random", 60, 0, filter_list, 15, True)
plot_validation_filtering("small_world", 60, 0, filter_list, 15, True)