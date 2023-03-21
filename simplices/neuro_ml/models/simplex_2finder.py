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
from torch_geometric.nn import aggr

@dataclass
class SimplexModelParams:
    n_neurons: int    #The number of remaining neurons
    output_dim: int





class MPN_forward(MessagePassing): 
    DATASET = W0_to_simplex_Dataset 

    def __init__(self, params):
        super(MPN_forward, self).__init__(aggr="add", flow = "source_to_target")   #Options: add*, min, max, mean*, sum*
        #super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        

        self.mlp_sources = Seq(
            Linear(self.n_neurons , self.n_neurons),   
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  
        )

        self.mlp_targets = Seq(
            Linear(self.n_neurons , self.n_neurons),   
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  
        )


        self.mlp2 = Seq(
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)
        )


        with torch.no_grad():
            nn.init.kaiming_normal_(self.mlp_sources[0].weight)
            nn.init.kaiming_normal_(self.mlp_sources[2].weight)
            nn.init.kaiming_normal_(self.mlp_sources[4].weight)
            nn.init.kaiming_normal_(self.mlp_sources[6].weight)

            nn.init.kaiming_normal_(self.mlp_targets[0].weight)
            nn.init.kaiming_normal_(self.mlp_targets[2].weight)
            nn.init.kaiming_normal_(self.mlp_targets[4].weight)
            nn.init.kaiming_normal_(self.mlp_targets[6].weight)

            nn.init.kaiming_normal_(self.mlp2[0].weight)
            nn.init.kaiming_normal_(self.mlp2[2].weight)
            nn.init.kaiming_normal_(self.mlp2[4].weight)



    def forward(self, edge_index, x): 
        #x has shape [n_neurons, neuron_features], so features are the outgoing connections from each node

        #Add self loops to preserve the information about the current state 
        #edge_index, _ = add_self_loops(edge_index, num_nodes = self.n_neurons)

        output_1 = self.propagate(edge_index, x=x, info = "targets")  #Send message about your targets to targets, get the targets of your sources
        output_2 = self.propagate(edge_index, x=x.T, info = "sources") #Send message about your sources to targets, get sources of your sources
        
        return output_1, output_2



    def message(self, x_j, x_i, info):  #x_j is the features of the sender, x_i the features of the receiver

        input = torch.mul(x_i, x_j)

        if info == "targets":
            message = self.mlp_targets(input)

        elif info == "sources":
            message = self.mlp_sources(input)

        return message
    

    def update(self, aggregated):    #Returns something of the same size
        output = self.mlp2(aggregated)

        return output



class MPN_backwards(MessagePassing): 

    DATASET = W0_to_simplex_Dataset 

    def __init__(self, params):
        super(MPN_backwards, self).__init__(aggr="add", flow = "target_to_source")   #Options: add*, min, max, mean*, sum*
        #super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        
        self.mlp_sources = Seq(
            Linear(self.n_neurons , self.n_neurons),   
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  
        )

        self.mlp_targets = Seq(
            Linear(self.n_neurons , self.n_neurons),   
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  
        )



        self.mlp2 = Seq(
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)
        )


        with torch.no_grad():
            nn.init.kaiming_normal_(self.mlp_sources[0].weight)
            nn.init.kaiming_normal_(self.mlp_sources[2].weight)
            nn.init.kaiming_normal_(self.mlp_sources[4].weight)
            nn.init.kaiming_normal_(self.mlp_sources[6].weight)

            nn.init.kaiming_normal_(self.mlp_targets[0].weight)
            nn.init.kaiming_normal_(self.mlp_targets[2].weight)
            nn.init.kaiming_normal_(self.mlp_targets[4].weight)
            nn.init.kaiming_normal_(self.mlp_targets[6].weight)

            nn.init.kaiming_normal_(self.mlp2[0].weight)
            nn.init.kaiming_normal_(self.mlp2[2].weight)
            nn.init.kaiming_normal_(self.mlp2[4].weight)



    def forward(self, edge_index, x): 
        #x has shape [n_neurons, neuron_features], so features are the outgoing connections from each node

        #Add self loops to preserve the information about the current state 
        #edge_index, _ = add_self_loops(edge_index, num_nodes = self.n_neurons)

        output_1 = self.propagate(edge_index, x=x, info = "targets")  #Send message about your targets to sources, get the targets of your targets
        output_2 = self.propagate(edge_index, x=x.T, info = "sources") #Send message about your sources to sources, get sources of your targets    #Problem: This flips the features of both x_i and x_j!! 
        
        return output_1, output_2



    def message(self, x_j, x_i, info):  #x_j is the features of the sender, x_i the features of the receiver

        input = torch.mul(x_i, x_j)

        if info == "targets":
            message = self.mlp_targets(input)

        elif info == "sources":
            message = self.mlp_sources(input)

        return message

    

    def update(self, aggregated):    #Returns something of the same size
        output = self.mlp2(aggregated)

        return output






class Outer_Simplicial_2finder(nn.Module):
    DATASET = W0_to_simplex_Dataset
    NAME = "outer_simplicial_1dim_working"

    def __init__(self, params):
        super().__init__()

        self.aggr = "add"

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        #self.x_learn = torch.empty((self.n_neurons, self.output_dim)).to(self.device)   #This is what we want to learn. Each row corresponds to the simplicial features of the node

        self.MPN_forward = MPN_forward(params)
        self.MPN_backwards = MPN_backwards(params)


        self.mlp_2s = Seq(
            Linear(self.n_neurons * 4, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 1)
        )


        with torch.no_grad():
            nn.init.kaiming_normal_(self.mlp_2s[0].weight)
            nn.init.kaiming_normal_(self.mlp_2s[2].weight)
            nn.init.kaiming_normal_(self.mlp_2s[4].weight)




    def forward(self, edge_index, x):
        output_1, output_2 = self.MPN_forward(edge_index, x)
        output_3, output_4 = self.MPN_backwards(edge_index, x)

        concat = torch.concat((output_1, output_2, output_3, output_4), dim = 1)#, output_3, output_4), dim = 1)    

        dim2 = self.mlp_2s(concat)

        return dim2

    
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
