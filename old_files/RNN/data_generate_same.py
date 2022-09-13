import numpy as np 
import torch
from torch import nn
import random
import torch.utils.data as data
import torch.optim as optim



def data_generate(d_points, N, T, cause, effect):
    '''
    Arguments:
        N = number of neurons, i.e. no. of possible positions
        T = number of time-steps
        d_points = datapoints
        cause = position of P1 which causes P2 to move upwards
        effect = does P2 move up or down when P1 is in the position cause? 1 for up, -1 for down
    '''
    vals = [-1, 1]

    #Chance of the other particle of moving upwards in the second timestep when this particle is found in this position
    #First N numbers for p1 at each position, second 2N numbers for p2 at each position
    probabilities = np.full((d_points, 1, 2*N), 0.5) 
    positions = np.zeros((d_points, 2, T))

    for point in range(d_points):

        #Initial position: random number between 0 and N
        positions[point, 0, 0] = np.random.randint(0, N)   
        positions[point, 1, 0] = np.random.randint(0, N)

        probabilities[point, 0, cause] = 0.5 + effect/2
        
        for i in range(1, T):
            positions[point, 0, i] = (positions[point, 0, i-1]+np.random.choice(vals)) % N

            if positions[point, 0, i-1] == cause:
                positions[point, 1, i] = (positions[point, 1, i-1] + effect) % N
            
            else: 
                positions[point, 1, i] = (positions[point, 1, i-1] + np.random.choice(vals)) % N

    return positions, probabilities



class Dataset:
    def __init__(self, d_points, neurons, timesteps, cause, effect):
        super().__init__()   
        '''
        Arguments:
            neurons = number of neurons, i.e. no. of possible positions
            timesteps = number of time-steps
            d_points = datapoints
            cause = position of P1 which causes P2 to move upwards
            effect = does P2 move up or down when P1 is in the position cause? 1 for up, -1 for down
        '''
        self.d_points = d_points

        self.positions, self.probabilities = data_generate(d_points, neurons, timesteps, cause, effect)

    def __len__(self):
        return len(self.positions[:,0,0])

    def __getitem__(self, idx):
        pos = torch.from_numpy(self.positions[idx])
        prob = torch.from_numpy(self.probabilities[idx])

        sample = {'positions': pos,
                  'probabilities': prob}

        return sample




def Dataloaders(d_points, neurons, timesteps, cause, effect, batchsize_train, batchsize_val, train_size = 0.7, test_size = 0.3):
    full_dataset = Dataset(d_points, neurons, timesteps, cause, effect)

    train_size = int(0.7 * len(full_dataset))
    test_size = int(0.3 * len(full_dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = data.DataLoader(train_dataset, batch_size = batchsize_train, shuffle=True, pin_memory=True, drop_last=True)  
    test_loader  = data.DataLoader(test_dataset, batch_size = batchsize_val, shuffle = False, drop_last=True)   #can add num_workers as last argument here if using a gpu

    '''
    for batch_idx, data in enumerate(train_loader):
        print((data['positions']).shape)
        #torch.permute(data['positions'], (0, 2, 1)).size()
    '''

    dataloaders = {'Train' : train_loader, 
                'Test': test_loader}

    return dataloaders



#dataloader = Dataloaders(d_points = 100, neurons = 5, timesteps = 1000, cause = 2, effect = 1, batchsize_train = 20, batchsize_val=10, train_size = 0.7, val_size = 0.1)




