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




class Simplicial_MPN(MessagePassing): 

    DATASET = W0_to_simplex_Dataset    #Variables defined outside the init belong to all instances of the class, and cannot be changed
    NAME = "W0_to_simplex"


    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.MP_iterations = 2
        self.output_dim = params.output_dim
        self.x_learn = torch.empty((self.n_neurons, 3 * self.output_dim))   #This is what we want to learn. Each row corresponds to the simplicial features of the node

        self.mlp1 = Seq(
            Linear(self.n_neurons * 2, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.output_dim)  #What should the output dimension be??? 
        )

        self.mlp2 = Seq(
            Linear(self.output_dim, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons)
        )


        self.mlp3 = Seq(
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.output_dim * 3)
        )


    def senders(self, receiver_id, edge_index):
        #Get a list of the id's of the neurons who have outgoing connections to the node in question
        senders = []
        for pair in edge_index.T:
            if pair[1] == receiver_id:
                senders.append(pair[0])
        return torch.tensor(senders)


    def receivers(self, sender_id, edge_index):
        #Get a list of the id's of the neurons who have incoming connections from the node in question
        receivers = []
        for pair in edge_index.T:
            if pair[0] == sender_id:
                receivers.append(pair[1])
        return torch.tensor(receivers)



    def forward(self, edge_index, x): 
        # This is the actual message-passing step

        x_tmp = torch.zeros_like(x)

        for neuron_id in range(self.n_neurons):
            own_connections = x[:, neuron_id]
            senders = self.senders(neuron_id, edge_index)

            # print("Node id: ", neuron_id)
            # print("Number of senders: ", senders.shape)
            # print()

            sender_messages = torch.stack([self.message(sender_id, own_connections, x) for sender_id in senders])  
            #Shape: [num_senders, output_dim]
            #print(sender_messages.shape)

            aggregation = self.aggregate(sender_messages)
            x_tmp[:, neuron_id] = self.update(aggregation)
        
        x = x_tmp

        x_tmp = torch.zeros_like(x)
        for neuron_id in range(self.n_neurons):
            own_connections = x[:, neuron_id]
            receivers = self.receivers(neuron_id, edge_index)

            # print("Node id: ", neuron_id)
            # print("Number of receivers: ", receivers.shape)
            # print()

            receiver_messages = torch.stack([self.message(receiver_id, own_connections, x) for receiver_id in receivers])  
            #Shape: [num_receivers, output_dim]
            #print(receiver_messages.shape)

            aggregation = self.aggregate(receiver_messages)

            x_tmp[:, neuron_id] = self.update(aggregation)
        
        x = x_tmp


        for node in range(self.n_neurons):
            output = self.mlp3(x)
            self.x_learn = output    #Do this for every node in the network

        return self.x_learn



    def message(self, sender_id, own_connections, x):
        incoming_connections = x[:, sender_id]   #Each incoming and outgoing node sends the information about which nodes they have incoming connections from

        input = torch.cat((own_connections, incoming_connections))  
        #print("Input shape: ", input.shape)
        #Want to learn something about the relation between the two 

        message = self.mlp1(input)
        #print("Message shape: ", message.shape)

        return message
        


    def aggregate(self, messages):
        aggregation = torch.sum(messages, dim = 0)  #Shape: [output_dim]
        return aggregation


    def update(self, aggregated):    #Returns something of the same size
        #Problem: how to prevent the aggregated vector's size from being dependent on the number of incoming and outgoing nodes?
        output = self.mlp2(aggregated)
        #print("mlp2 output shape: ", output.shape)    #Shape: [num_neurons]

        return output



    def makedirs(self, network_type):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}") 

        return f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"


    
    def save(self, filename):   
        # Saves the model to file
        torch.save(self.state_dict(), filename)



#Tested: GCNConv, GATConv (no), GATv2Conv, TransformerConv, 
#None of them seem to learn much


class Simplicial_GCN(nn.Module):
    DATASET = W0_to_simplex_Dataset
    NAME = "simplicial_GATv2Conv"

    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons
        self.output_dim = params.output_dim
        self.params = params
        self.hidden_dim = 30

        self.simplex1 = GATv2Conv(self.n_neurons, self.hidden_dim, heads = 2)
        self.simplex2 = GATv2Conv(self.hidden_dim * 2, self.hidden_dim, heads = 1)
        self.simplex3 = GATv2Conv(self.hidden_dim, self.hidden_dim, heads = 2)
        self.linear = Linear(self.hidden_dim * 2, self.output_dim)
        
        
    def forward(self, edge_index, x):
        #print("Connectivity shape: ", connectivity.shape)
        #print("Edge index shape: ", edge_indices.shape)

        simplices = self.simplex1(x, edge_index)
        simplices = simplices.relu()
        simplices = self.simplex2(simplices, edge_index)
        simplices = simplices.relu()
        simplices = self.simplex3(simplices, edge_index)
        #simplices = global_mean_pool(simplices, None)


        #print("Simplices shape: ", simplices.shape)

        pred = self.linear(simplices)
        #print("Pred shape: ", pred.shape)

        return pred

    def makedirs(self, network_type):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}") 

        return f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"


    def save(self, filename):   
        # Saves the model to file
        torch.save(self.state_dict(), filename)





class Simplicial_MPN_simple(MessagePassing): # Or `Module`

    DATASET = W0_to_simplex_Dataset #TimeSeriesAndEdgeIndicesToWeightsDataset    #Variables defined outside the init belong to all instances of the class, and cannot be changed
    NAME = "W0_to_simplex_simple"


    def __init__(self, params):
        super().__init__()

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        self.x_learn = torch.empty((self.n_neurons, self.output_dim))   #This is what we want to learn. Each row corresponds to the simplicial features of the node

        self.mlp1 = Seq(
            Linear(self.n_neurons * 2, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.output_dim)  #What should the output dimension be??? 
        )

        self.mlp2 = Seq(
            Linear(self.output_dim, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons)
        )


        self.mlp3 = Seq(
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.output_dim)
        )


    def senders(self, receiver_id, edge_index):
        #Get a list of the id's of the neurons who have outgoing connections to the node in question
        senders = []
        for pair in edge_index.T:
            if pair[1] == receiver_id:
                senders.append(pair[0])
        return torch.tensor(senders)


    def receivers(self, sender_id, edge_index):
        #Get a list of the id's of the neurons who have incoming connections from the node in question
        receivers = []
        for pair in edge_index.T:
            if pair[0] == sender_id:
                receivers.append(pair[1])
        return torch.tensor(receivers)



    def forward(self, edge_index, x): 
        # This is the actual message-passing step

        x_tmp = torch.zeros_like(x)

        for neuron_id in range(self.n_neurons):
            own_connections = x[:, neuron_id]
            senders = self.senders(neuron_id, edge_index)
            sender_messages = torch.stack([self.message(sender_id, own_connections, x) for sender_id in senders])  

            aggregation = self.aggregate(sender_messages)
            x_tmp[:, neuron_id] = self.update(aggregation)
        
        x = x_tmp

        x_tmp = torch.zeros_like(x)
        for neuron_id in range(self.n_neurons):
            own_connections = x[:, neuron_id]
            receivers = self.receivers(neuron_id, edge_index)
            receiver_messages = torch.stack([self.message(receiver_id, own_connections, x) for receiver_id in receivers])  

            aggregation = self.aggregate(receiver_messages)

            x_tmp[:, neuron_id] = self.update(aggregation)
        
        x = x_tmp


        for node in range(self.n_neurons):
            output = self.mlp3(x)
            self.x_learn = output    #Do this for every node in the network

        return self.x_learn



    def message(self, sender_id, own_connections, x):
        incoming_connections = x[:, sender_id]   #Each incoming and outgoing node sends the information about which nodes they have incoming connections from

        input = torch.cat((own_connections, incoming_connections))  
        message = self.mlp1(input)

        return message
        


    def aggregate(self, messages):
        aggregation = torch.sum(messages, dim = 0)  #Shape: [output_dim]
        return aggregation


    def update(self, aggregated):    #Returns something of the same size
        output = self.mlp2(aggregated)

        return output




    def makedirs(self, network_type):
        if not os.path.isdir(f"saved_models/{self.NAME}"):
                os.mkdir(f"saved_models/{self.NAME}")

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}") 

        if not os.path.exists(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"):
                os.mkdir(f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}") 

        return f"saved_models/{self.NAME}/{network_type}/neurons_{self.n_neurons}"


    
    def save(self, filename):   
        # Saves the model to file
        torch.save(self.state_dict(), filename)







#Current best val loss: 37ish

class Simplicial_MPN_speedup(MessagePassing): # Or `Module`

    DATASET = W0_to_simplex_Dataset #TimeSeriesAndEdgeIndicesToWeightsDataset    #Variables defined outside the init belong to all instances of the class, and cannot be changed
    NAME = "W0_to_simplex_complex_4dim"


    def __init__(self, params):
        super(Simplicial_MPN_speedup, self).__init__(aggr='add')   #Options: add*, min, max, mean*, sum*
        #super().__init__()

        self.aggr = "add"

        self.n_neurons = params.n_neurons #N, number of (remaining) neurons in the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = params.output_dim
        self.x_learn = torch.empty((self.n_neurons, self.output_dim))   #This is what we want to learn. Each row corresponds to the simplicial features of the node

        self.mlp1 = Seq(
            Linear(self.n_neurons * 2, self.n_neurons),   #Input dim is *2 if message is concatenation of features
            ReLU(),
            Linear(self.n_neurons, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.output_dim * 3)  #What should the output dimension be??? 
        )

        self.mlp2 = Seq(
            Linear(self.output_dim * 3, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.n_neurons)
        )


        self.mlp3 = Seq(
            Linear(self.n_neurons * 4, 10*self.n_neurons),
            ReLU(),
            Linear(10*self.n_neurons, self.n_neurons),
            ReLU(),
            Linear(self.n_neurons, self.output_dim)
        )

        with torch.no_grad():
            nn.init.kaiming_normal_(self.mlp1[0].weight)
            nn.init.kaiming_normal_(self.mlp1[2].weight)
            nn.init.kaiming_normal_(self.mlp1[4].weight)
            nn.init.kaiming_normal_(self.mlp1[6].weight)

            nn.init.kaiming_normal_(self.mlp2[0].weight)
            nn.init.kaiming_normal_(self.mlp2[2].weight)
            nn.init.kaiming_normal_(self.mlp2[4].weight)

            nn.init.kaiming_normal_(self.mlp3[0].weight)
            nn.init.kaiming_normal_(self.mlp3[2].weight)
            nn.init.kaiming_normal_(self.mlp3[4].weight)



    def forward(self, edge_index, x): 
        #x has shape [n_neurons, neuron_features], so features are the outgoing connections from each node

        #Add self loops to preserve the information about the current state 
        edge_index, _ = add_self_loops(edge_index, num_nodes = self.n_neurons)

        #x = x.T  #Let the features be the incoming connections from each node, this made no difference... 

        output_1 = self.propagate(edge_index, x=x)#, flow = "source_to_target")   #Send message about your targets to targets, get the targets of your sources
        #Flip the edge index and repeat

        #print("Output1 shape: ", output_1.shape)

        flipped = torch.flip(edge_index, [0])

        #output_2 = self.propagate(flipped, x = output_1)

        #final = self.mlp3(output_2)
        
        output_2 = self.propagate(flipped, x=x)#, flow = "target_to_source")   #Send message about your tagets to sources, get the targets of your sources
        output_3 = self.propagate(edge_index, x=x.T)#, flow = "source_to_target")   #Send message about your sources to targets, get sources of your sources
        output_4 = self.propagate(flipped, x=x.T)#, flow = "target_to_source")  #Send message about your sources to sources, get the sources of your targets

        concat = torch.concat((output_1, output_2, output_3, output_4), dim = 1)
        #print("Concat shape: ", concat.shape)

        final = self.mlp3(concat)
        #print("Final shape: ", final.shape)

        return final



    def message(self, x_j, x_i):  #x_j is the features of the sender, x_i the features of the receiver
        # print("x_i shape: ", x_i.shape)
        #print("x_j shape: ", x_j.shape)
        # print(x_i[0])
        # print(x_j[0])
        input = torch.cat((x_i, x_j), dim = 1)    #This is the complex version

        #print("Message input shape: ", input.shape)

        message = self.mlp1(input)

        #print("Message output shape: ", message.shape)

        return message
    

    def update(self, aggregated):    #Returns something of the same size
        #print("Aggregation shape to update: ", aggregated.shape)
        output = self.mlp2(aggregated)   #This doesn't make sense when you input the transpose!! XXX 

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