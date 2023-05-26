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




def plot_loss(network, cluster_size: list, neurons_removed: int, simplex_filter: int, epochs: int, error: bool):

    fig = go.Figure()
    
    iteration = cluster_size
    if error:
        iteration = reversed(iteration)

    for idx, size in enumerate(iteration):
        data_path = f"../saved_models/edge_regressor/{network}/cluster_sizes_[{size}]_n_steps_200000_simplex_threshold_{simplex_filter}/remove_{neurons_removed}/loss.csv"

        df = pd.read_csv(data_path)

        # train_loss = df["train_loss"][0:epochs]
        # train_loss_std = df["train_loss_std"][0:epochs]

        val_loss = df["val_loss"][0:epochs]
        val_loss_std = df["val_loss_std"][0:epochs]/np.sqrt(30)

        # fig.add_trace(go.Scatter(y=train_loss, name=f"Train - {size}", line=dict(width=3, color=my_colours[idx], dash="dash")))

        # if error:
        #     fig.add_trace(
        #         go.Scatter(
        #         name = "Upper bound",
        #         y = train_loss + train_loss_std,
        #         mode = 'lines',
        #         marker = dict(color=my_colours[idx]),
        #         line = dict(width = 0),
        #         showlegend=False
        #     ))

        #     fig.add_trace(
        #         go.Scatter(
        #             name = "Lower bound",
        #             y = train_loss - train_loss_std,
        #             marker = dict(color=my_colours[idx]),
        #             line = dict(width = 0),
        #             mode = 'lines',
        #             fillcolor = translucent_colours[idx], 
        #             fill = 'tonexty',
        #             showlegend=False
        #         )
        #     )

        fig.add_trace(go.Scatter(y=val_loss, name=f"{size}", line=dict(width=3, color=my_colours[idx])))

        if error: 
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = val_loss + val_loss_std,
                mode = 'lines',
                marker = dict(color=my_colours[idx]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = val_loss - val_loss_std,
                    marker = dict(color=my_colours[idx]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[idx], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )


    if network == "random":
        network_name = "random"
    elif network == "small_world":
        network_name = "small-world"

    fig.update_layout(
        title=f"Validation loss for {network_name} networks of different sizes",
        legend_title='Neurons',
        xaxis_title="Epochs",
        yaxis_title="Loss",
        font_family = "Garamond",
        font_size = 15)

    fig.write_image(f"figures/{network}_val_loss_epochs_error_{error}.pdf")


sm_size_list = [10, 15, 20, 25, 30, 40, 50, 60, 70]
rm_size_list = [10, 15, 20, 25, 30, 40, 50, 60, 70]


plot_loss("random", rm_size_list, 0, 0, 15, error=True)

plot_loss("small_world", sm_size_list, 0, 0, 15, error=True)



def best_loss(network_type, cluster_size, neurons_removed: int, simplex_filter: int):

    best_loss = np.zeros(len(cluster_size))
    best_loss_std = np.zeros(len(cluster_size))
    x = cluster_size

    for idx, size in enumerate(cluster_size):
        if type(size) == int:
            data_path = f"../saved_models/edge_regressor/{network_type}/cluster_sizes_[{size}]_n_steps_200000_simplex_threshold_{simplex_filter}/remove_{neurons_removed}/loss.csv"
        else:
            data_path = f"../saved_models/edge_regressor/{network_type}/cluster_sizes_{size}_n_steps_200000_simplex_threshold_{simplex_filter}/remove_{neurons_removed}/loss.csv"

        df = pd.read_csv(data_path)

        val_loss = df["val_loss"]
        val_loss_std = df["val_loss_std"]

        best_loss[idx] = np.min(val_loss)
        best_loss_std[idx] = val_loss_std[np.argmin(val_loss)]/np.sqrt(30)


    fig = go.Figure([
        go.Scatter(
            name = "Best loss",
            x = x, 
            y=best_loss, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
                  
        go.Scatter(
            name = "Upper bound",
            x = x,
            y = best_loss + best_loss_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = x,
            y = best_loss - best_loss_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)', 
            fill = 'tonexty',
            showlegend=False
        )
        ])
    
    if network_type == "random":
        network_name = "random"
    elif network_type == "small_world":
        network_name = "small-world"

    fig.update_layout(title=f'Best validation loss for {network_name} networks of different sizes', 
                xaxis_title='Neurons',
                yaxis_title='Best validation loss',
                font_family = "Garamond",
                font_size = 15)
    
    fig.write_image(f"figures/{network_type}_best_loss.pdf")

    

sm_size_list = [10, 15, 20, 25, 30, 40, 50, 60, 70]
rm_size_list = [10, 15, 20, 25, 30, 40, 50, 60, 70]

best_loss("random", rm_size_list, 0, 0)
best_loss("small_world", sm_size_list, 0, 0)




