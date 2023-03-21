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
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
        )

        self.mlp_targets = Seq(
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
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

        #input = torch.cat((x_i, x_j), dim = 1)    #This is the complex version
        input = torch.mul(x_i, x_j)

        if info == "targets":
            message = self.mlp_targets(input)

        elif info == "sources":
            message = self.mlp_sources(input)

        return message
    

    def update(self, aggregated):    #Returns something of the same size
        #print("Aggregation shape to update: ", aggregated.shape)
        output = self.mlp2(aggregated)
       #output=aggregated

        #print("Update output shape: ", output.shape)

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
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
        )

        self.mlp_targets = Seq(
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
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

        #input = torch.cat((x_i, x_j), dim = 1)    #This is the complex version

        input = torch.mul(x_i, x_j)

        if info == "targets":
            message = self.mlp_targets(input)

        elif info == "sources":
            message = self.mlp_sources(input)

        return message

    

    def update(self, aggregated):    #Returns something of the same size
        #print("Aggregation shape to update: ", aggregated.shape)
        output = self.mlp2(aggregated)
        #output = aggregated

        #print("Update output shape: ", output.shape)

        return output






class MPN_forward_3s(MessagePassing): 
    DATASET = W0_to_simplex_Dataset 

    def __init__(self, params):
        super(MPN_forward_3s, self).__init__(aggr="add", flow = "source_to_target")   #Options: add*, min, max, mean*, sum*
        #super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        

        self.mlp_sources = Seq(
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
        )

        self.mlp_targets = Seq(
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
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

        #input = torch.cat((x_i, x_j), dim = 1)    #This is the complex version
        input = torch.mul(x_i, x_j)

        if info == "targets":
            message = self.mlp_targets(input)

        elif info == "sources":
            message = self.mlp_sources(input)

        return message
    

    def update(self, aggregated):    #Returns something of the same size
        #print("Aggregation shape to update: ", aggregated.shape)
        output = self.mlp2(aggregated)
       #output=aggregated

        #print("Update output shape: ", output.shape)

        return output



class MPN_backwards_3s(MessagePassing): 

    DATASET = W0_to_simplex_Dataset 

    def __init__(self, params):
        super(MPN_backwards_3s, self).__init__(aggr="add", flow = "target_to_source")   #Options: add*, min, max, mean*, sum*
        #super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        
        self.mlp_sources = Seq(
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
        )

        self.mlp_targets = Seq(
            Linear(self.n_neurons , self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)  #What should the output dimension be??? 
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

        #input = torch.cat((x_i, x_j), dim = 1)    #This is the complex version

        input = torch.mul(x_i, x_j)

        if info == "targets":
            message = self.mlp_targets(input)

        elif info == "sources":
            message = self.mlp_sources(input)

        return message

    

    def update(self, aggregated):    #Returns something of the same size
        #print("Aggregation shape to update: ", aggregated.shape)
        output = self.mlp2(aggregated)
        #output = aggregated

        #print("Update output shape: ", output.shape)

        return output










class Outer_Simplicial(nn.Module):
    DATASET = W0_to_simplex_Dataset
    NAME = "outer_simplicial_2dim_testing"

    def __init__(self, params):
        super().__init__()

        self.aggr = "add_message_dotprod"

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
 
        self.MPN_forward = MPN_forward(params)
        self.MPN_backwards = MPN_backwards(params)

        self.MPN_forward_3s = MPN_forward_3s(params)
        self.MPN_backwards_3s = MPN_backwards_3s(params)

        self.mlp_2s = Seq(
            Linear(self.n_neurons * 4, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 1)
        )

        self.mlp_3s = Seq(
            Linear(self.n_neurons * 4, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 1)
        )


        self.reduce = Seq(
            Linear(self.n_neurons * 4, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)
        )


        with torch.no_grad():
            nn.init.kaiming_normal_(self.mlp_2s[0].weight)
            nn.init.kaiming_normal_(self.mlp_2s[2].weight)
            nn.init.kaiming_normal_(self.mlp_2s[4].weight)

            nn.init.kaiming_normal_(self.mlp_3s[0].weight)
            nn.init.kaiming_normal_(self.mlp_3s[2].weight)
            nn.init.kaiming_normal_(self.mlp_3s[4].weight)



    def forward(self, edge_index, x):
        # output_1, output_2 = self.MPN_forward(edge_index, x)
        # output_3, output_4 = self.MPN_backwards(edge_index, x)

        # concat_2s = torch.concat((output_1, output_2, output_3, output_4), dim = 1)#, output_3, output_4), dim = 1)    

        # dim2 = self.mlp_2s(concat_2s)

        # reduced = self.reduce(concat_2s)

        output_1, output_2 = self.MPN_forward_3s(edge_index, x)   
        output_3, output_4 = self.MPN_backwards_3s(edge_index, x)

        concat_3s = torch.concat((output_1, output_2, output_3, output_4), dim = 1)  

        dim3 = self.mlp_3s(concat_3s)  

        #final = torch.concat((dim2, dim3), dim = 1)

        return dim3

    
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





"""
Notes

As of now, it is unable to find the 3-simplicies. The iterative scheme performs far worse, giving a val error of over 200. 
Using the message passing scheme or only once gives a val loss of around 32. 

"""