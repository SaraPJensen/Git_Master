from data_generate_same import *
from plot_probs import *
import csv

print("Started")


class Simple_RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, seq_length, rnn):
        """
        :param input_size: Input features, 2 in this case
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """
        super(Simple_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.seq_length = seq_length

        self.rnn = rnn

        self.output_layer = nn.Linear(hidden_state_size, input_size)   #add a softmax layer to make it a classification problem instead


    #In training loop, must set first initial state to zero, then use the current_hidden_state as input for later epochs
    def forward(self, input_seq, hidden_state):

        output = self.rnn(input_seq, hidden_state)

        hidden_state = output[1]
        predictions = self.output_layer(output[0]) 

        return predictions, hidden_state




d_points = 500
input_size = 2
hidden_state_size = 50
num_rnn_layers = 1
seq_length = 100
nonlinearity = 'relu'
batchsize_train = 20
batchsize_test = 10
bias = True
batch_first = True   #Batch_size is the first dimension of the input 
dropout = 0
epochs = 10

neurons = 6
cause = 2
effect = 1   #move upwards

lr = 0.001
momentum = 0.9


dataloader = Dataloaders(d_points = d_points, neurons = neurons, timesteps = seq_length, cause = cause, effect = effect, batchsize_train = batchsize_train, batchsize_val=batchsize_test, train_size = 0.7, test_size = 0.3)


train_loader = dataloader['Train']
test_loader = dataloader['Test']
print()


pytorch_rnn = nn.RNN(input_size, hidden_state_size, num_rnn_layers, nonlinearity = nonlinearity, bias = bias, batch_first = True, dropout = dropout, bidirectional = False)

my_rnn = Simple_RNN(input_size, hidden_state_size, num_rnn_layers, seq_length, pytorch_rnn)


criterion = nn.MSELoss()
optimiser = optim.SGD(my_rnn.parameters(), lr,  momentum)

print("Starting training")

def Train(model, train_data, criterion, optimiser, current_hidden_state = None):
    model.train(True)
    model.train()
    losses = []

    batches = len(train_data)

    torch.autograd.set_detect_anomaly(True)


    for batch_idx, data in enumerate(train_data):

        input = data['positions']
        input_p = torch.permute(input, (0, 2, 1))   #Input shape must be [batch_size, seq_len, input_features]

        print("Input shape: ", input_p.shape)
        
        optimiser.zero_grad()

        if current_hidden_state is None: 
                initial_hidden_state = torch.zeros(num_rnn_layers, batchsize_train, hidden_state_size)
        else: 
            initial_hidden_state = current_hidden_state.detach()
        
        (output, current_hidden_state) = model(input_p.float(), initial_hidden_state)   #Assume the hidden state should be pass between batches as well

        pred = output[:, :-1 :]
        compare = input_p[:, 1:, :]

        loss = criterion(pred.double(), compare.double())   #Need to shift the input relative to the output, otherwise it will just learn to spit out what I put in
        #ignore the first point in the input and the last point in the output

        loss.backward()
        
        optimiser.step()   #Problem occurs when this is included
        losses.append(loss.detach())


    return np.mean(losses), current_hidden_state



def Test(model, test_data, criterion, optimiser, current_hidden_state = None):
    model.train(False)
    losses = []

    
    file_preds = open("test_preds.csv", 'w')
    #file_preds.write("Predicted positions of the two particles over time, two rows per sample, one for each particle \n")
    file_preds.close()

    file_true = open("test_true.csv", 'w')
    #file_true.write("True positions of the two particles over time, two rows per sample, one for each particle \n")
    file_true.close()
    

    initial_hidden_state = torch.zeros(num_rnn_layers, batchsize_test, hidden_state_size)
    
    for batch_idx, data in enumerate(test_data):
        input = data['positions']
        input_p = torch.permute(input, (0, 2, 1))   #Input shape must be [batch_size, seq_len, input_features]

        (output, current_hidden_state) = model(input_p.float(), initial_hidden_state)   #Assume the hidden state should be pass between batches as well
        initial_hidden_state = current_hidden_state
        pred = output[:, :-1 :]   #Skip the last one
        compare = input_p[:, 1:, :]   #Skip the first one

        loss = criterion(pred.double(), compare.double())   #Need to shift the input relative to the output, otherwise it will just learn to spit out what I put in
     
        losses.append(loss.detach())

        file_preds = open("test_preds.csv", 'a')
        pred_writer = csv.writer(file_preds)

        file_true = open("test_true.csv", 'a')
        true_writer = csv.writer(file_true)

        output = output.detach().numpy()
        input_p = input_p.detach().numpy()

        pred = pred.detach().numpy()
        compare = compare.detach().numpy()
 
        for batch in range(len(input[:, 0, 0])):
            #pred_writer.writerows(np.round(output[batch, :, :].transpose()))
            #true_writer.writerows(input_p[batch, :, :].transpose())
            pred_writer.writerows(np.round(pred[batch, :, :].transpose()))
            true_writer.writerows(compare[batch, :, :].transpose())

        file_preds.close()
        file_true.close()

    print("Average test loss: ", np.mean(losses))





def Train_Test(my_rnn, train_loader, test_loader, criterion, optimiser, epochs):
    avg_loss = []

    current_hidden_state = None

    file_loss = open("train_loss.csv", 'w')
    file_loss.write("Average training loss by epoch\n")
    file_loss.write("Epoch,Loss\n")
    file_loss.close()
    
    for e in range(epochs):
        loss, current_hidden_state = Train(my_rnn, train_loader, criterion, optimiser, current_hidden_state)

        avg_loss.append(loss)
        print(f"At epoch {e}, average loss: {loss}")

        file_loss = open("train_loss.csv", 'a')
        file_loss.write(f"{e},{loss}\n")
        file_loss.close()
    
    Test(my_rnn, test_loader, criterion, optimiser, current_hidden_state)



Train_Test(my_rnn, train_loader, test_loader, criterion, optimiser, epochs)


print("True values")
plotting("test_true.csv", neurons)

print()

print("Predicted values")
plotting("test_preds.csv", neurons)

#plotting(filename, neurons)