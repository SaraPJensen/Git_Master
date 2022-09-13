from data_generate_same import *

print("Started working")


dataloader = Dataloaders(d_points = 100, neurons = 5, timesteps = 100, cause = 2, effect = 1, batchsize_train = 20, batchsize_val=10, train_size = 0.7, val_size = 0.1)


class Simple_RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, seq_length, cell_type='RNN'):
        """
        :param input_size: Input features, 2 in this case
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """

        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.seq_length = seq_length

        self.rnn = nn.RNN(input_size, hidden_size, num_rnn_layers, nonlinearity = 'tahn', bias = True, batch_first = True, dropout = False, bidirectional = False)
        
        self.output_layer = nn.Sequential(nn.Linear(hidden_state_size, input_size),
            nn.Sigmoid()
        )

    #In training loop, must set first initial state to zero, then use the current_hidden_state as input for later epochs
    def forward(self, input_seq, hidden_state):

        #Might want to do this permutation outside of the forward pass for easier comparison when evaluating
        input = torch.permute(input_seq, (0, 2, 1))   #Input shape must be [batch_size, seq_len, input_features]

        output = self.rnn(input.float(), hidden_state)

        hidden_state = output[1]
        predictions = self.final_layer(output[0])

        return predictions, hidden_state






class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, seq_length, cell_type='RNN'):
        """
        :param input_size: Input features, 2 in this case
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """

        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.seq_length = seq_length

        '''   #Check whether it works with just using this instead
        self.rnn = nn.RNN(input_size, hidden_size, num_rnn_layers, nonlinearity = 'tahn', bias = True, batch_first = True, dropout = False, bidirectional = False)
        self.output_layer = nn.Sequential(nn.Linear(hidden_state_size, input_size),
            nn.Sigmoid()
        )
        '''


        input_size_list = []
        input_size_list.append(self.input_size)

        for i in range(1, num_rnn_layers):
            input_size_list.append(self.hidden_state_size) 
        
        #input_size_list[-1] = self.input_size   #Make sure the output is of the same dimension as the input

        self.cell_options = {
            'RNN': RNNsimpleCell,   #Could just use the nn.RNN here, and just add the output layer
            'GRU': GRUCell,
            'LSTM': LSTMCell       
        }

        self.cell_type = self.cell_options[cell_type]

        self.cells = nn.ModuleList([])

        for input_size in input_size_list:
            self.cells.append(self.cell_type(hidden_state_size, input_size))

        self.output_layer = nn.Sequential(nn.Linear(hidden_state_size, input_size),
            nn.Sigmoid()
        )
        


    def forward(self, input_seq, initial_hidden_state, is_train = True):

        current_hidden_state = initial_hidden_state

        pred_seq = []   #prediction sequence

        input = input_seq[:, 0, :]  #First time step

        for i in range(self.seq_length):
            new_hidden_state = torch.zeros_like(current_hidden_state)


            #input has shape [batch_size, seq_length, features]
            for l in range(0, self.num_rnn_layers):
                new_hidden_state[l] = self.cells[l](input, current_hidden_state[l])
                input = new_hidden_state[l, :, :]

            current_hidden_state = new_hidden_state

            pred = self.output_layer(current_hidden_state[-1]) 

            pred_seq.append(pred)

        
        # Get the input tokens for the next step in the sequence
            if i < self.seq_length - 1:
                if is_train:
                    input = input_seq[:, i+1, :]
                else:
                    input = pred

        
        predictions = torch.stack(pred_seq, dim=1)  # Convert the sequence of logits to a tensor

        return predictions, current_hidden_state

                                    



        

    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.

        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """

        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length
        else:
            sequence_length = 40  # Max sequence length to generate

        # Get embeddings for the whole sequence
        embeddings = embedding_layer(input=tokens)  # Should have shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state

        input_tokens = embeddings[:,0,:]  # Should have shape (batch_size, embedding_size)

        for i in range(sequence_length):
            new_hidden_state = torch.zeros_like(current_hidden_state)

            input_for_the_first_layer = torch.cat((input_tokens,processed_cnn_features), dim=1)
            new_hidden_state[0] = self.cells[0](input_for_the_first_layer, current_hidden_state[0])

            for l in range(1, self.num_rnn_layers):
                new_hidden_state[l] = self.cells[l](new_hidden_state[l-1,:,:self.hidden_state_size], current_hidden_state[l])                 

            current_hidden_state = new_hidden_state

            logits_i = output_layer(new_hidden_state[-1,:,:self.hidden_state_size])

            logits_sequence.append(logits_i)
            # Find the next predicted output element
            predictions = torch.argmax(logits_i, dim=1)

            # Get the input tokens for the next step in the sequence
            if i < sequence_length - 1:
                if is_train:
                    input_tokens = embeddings[:, i+1, :]
                else:
                    input_tokens = embedding_layer(predictions)

        logits = torch.stack(logits_sequence, dim=1)  # Convert the sequence of logits to a tensor

        return logits, current_hidden_state






class RNNsimpleCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            input_size: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes + input_size, hidden_state_sizes]. Initialized
                         using variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 
        """
        super(RNNsimpleCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        self.weight = nn.Parameter(
            torch.randn(input_size + hidden_state_size, hidden_state_size) / np.sqrt(input_size + hidden_state_size))
        self.bias = nn.Parameter(torch.zeros(1, hidden_state_size))


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]  
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]
        """
        x2 = torch.cat((x, state_old), dim=1)
        state_new = torch.tanh(torch.mm(x2, self.weight) + self.bias)

        return state_new







input_size = 2
hidden_size = 100
num_layers = 3
nonlinearity = 'relu'
bias = True
batch_first = True   #Batch_size is the first dimension of the input 
dropout = 0

lr = 0.001
momentum = 0.9

rnn = nn.Sequential(
    nn.RNN(input_size, hidden_size, num_layers, nonlinearity = nonlinearity, bias = bias, batch_first = batch_first, dropout = dropout, bidirectional = False),
    
    nn.RNN(hidden_size, input_size, 1, nonlinearity = nonlinearity, bias = bias, batch_first = batch_first, dropout = dropout, bidirectional = False))


train_loader = dataloader['Train']
val_loader = dataloader['Val']
test_loader = dataloader['Test']
print()



print("Got datasets")



def Train(model, train_data, criterion, optimiser):
    model.train(True)
    model.train()
    losses = []
    accuracies = []

    for batch_idx, data in enumerate(train_data):
        input = torch.permute(data['positions'], (0, 2, 1))   #Input shape must be [batch_size, seq_len, input_features]

        print("Input size: ", input.size())
        #input = data['positions']

        optimiser.zero_grad()

        output = rnn(input.float())

        #print("Output size: ", len(output))


criterion = nn.MSELoss()
optimiser = optim.SGD(rnn.parameters(), lr,  momentum)

Train(rnn, train_loader, criterion, optimiser)

'''

def Train_test(model, train_data, test_data, criterion, optimiser, scheduler, num_epochs):
    
    for e in range num_epochs:
'''