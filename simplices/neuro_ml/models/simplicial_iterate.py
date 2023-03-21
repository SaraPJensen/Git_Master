import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv, TransformerConv, GATv2Conv
from torch_geometric.utils.sparse import dense_to_sparse
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import W0_to_simplex_Dataset
import os
import torch.nn as nn
from torch_geometric.utils import add_self_loops

@dataclass
class SimplexModelParams:
    n_neurons: int    #The number of remaining neurons
    output_dim: int





class Simplicial_MPN_iterate(MessagePassing):

    DATASET = W0_to_simplex_Dataset 
    NAME = "W0_to_simplex_iterate_4dim"


    def __init__(self, params):
        super(Simplicial_MPN_iterate, self).__init__(aggr='add')   #Options: add*, min, max, mean*, sum*
        #super().__init__()

        self.aggr = "add"

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim


        self.messages = nn.ModuleList(Seq(
            Linear(self.n_neurons * dim, 10*self.n_neurons),  
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons * dim)  #What should the output dimension be??? 
            ) for dim in range(1, self.output_dim + 2))



        self.updates = nn.ModuleList(Seq(
            Linear(self.n_neurons * dim, 10*self.n_neurons),  
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
            ) for dim in range(1, self.output_dim + 2))


        self.mlp3 = Seq(
            Linear(self.n_neurons * (self.output_dim+2), 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.output_dim)
        )


        # with torch.no_grad():
        #     nn.init.kaiming_normal_(self.mlp1[0].weight)
        #     nn.init.kaiming_normal_(self.mlp1[2].weight)
        #     nn.init.kaiming_normal_(self.mlp1[4].weight)
        #     nn.init.kaiming_normal_(self.mlp1[6].weight)

        #     nn.init.kaiming_normal_(self.mlp2[0].weight)
        #     nn.init.kaiming_normal_(self.mlp2[2].weight)
        #     nn.init.kaiming_normal_(self.mlp2[4].weight)

        #     nn.init.kaiming_normal_(self.mlp3[0].weight)
        #     nn.init.kaiming_normal_(self.mlp3[2].weight)
        #     nn.init.kaiming_normal_(self.mlp3[4].weight)



    def forward(self, edge_index, x): 
        #x has shape [n_neurons, neuron_features], so features are the outgoing connections from each node

        #Add self loops to preserve the information about the current state 
        edge_index, _ = add_self_loops(edge_index, num_nodes = self.n_neurons)

        old_hidden = x

        for message_mlp, update_mlp in zip(self.messages, self.updates):
            #print("Hidden shape: ", old_hidden.shape)
            new_hidden = self.propagate(edge_index, x=old_hidden, message_mlp=message_mlp, update_mlp=update_mlp)
            old_hidden = torch.concat((old_hidden, new_hidden), dim = 1)
            #print("New hidden shape: ", old_hidden.shape)
        
        final = self.mlp3(old_hidden)
        # print(final.shape)
        # exit()

        return final



    def message(self, x_j, message_mlp):  #x_j is the features of the sender, x_i the features of the receiver
        #print("Message input shape: ", x_j.shape)
        message = message_mlp(x_j)
        #print("Message output shape: ", message.shape)


        return message
    

    def update(self, aggregated, x, update_mlp):    #Make different update function for each iteration
        #print("x shape to update: ", x.shape)
        #print("Aggregation shape to update: ", aggregated.shape)
        #input = torch.concat((x, aggregated))    #Do I need to do this explicitly?? XXX
        output = update_mlp(aggregated)
        #print(update_mlp)
        #print("Update output shape: ", output.shape)

        return output




    def makedirs(self, network_type):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}") 
        
        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}/{self.aggr}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}/{self.aggr}") 

        return f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}/{self.aggr}"


    
    def save(self, filename):   
        # Saves the model to file
        torch.save(self.state_dict(), filename)