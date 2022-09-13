from data_generate_same import *

print("Started working")


dataloader = Dataloaders(d_points = 100, neurons = 5, timesteps = 100, cause = 2, effect = 1, batchsize_train = 20, batchsize_val=10, train_size = 0.7, val_size = 0.1)


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




 
    