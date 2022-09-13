import numpy as np 
import torch
from torch import nn
import random
import torch.utils.data as data



def data_generate(d_points, N, T):
    '''
    Arguments:
        N = number of neurons, i.e. no. of possible positions
        T = number of time-steps
        d_points = datapoints
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

        cause = np.random.randint(0, N)   #Which position must the first particle be in for the second to move with certainty
        effect = np.random.choice(vals)   #Does p2 move up or down when p1 is in the specific position
        probabilities[point, 0, cause] = 0.5 + effect/2
        
        for i in range(1, T):
            positions[point, 0, i] = (positions[point, 0, i-1]+np.random.choice(vals)) % N

            if positions[point, 0, i-1] == cause:
                positions[point, 1, i] = (positions[point, 1, i-1] + effect) % N
            
            else: 
                positions[point, 1, i] = (positions[point, 1, i-1] + np.random.choice(vals)) % N

    return positions, probabilities



class Dataset:
    def __init__(self, d_points, neurons, timesteps):
        super().__init__()   
        '''
        Arguments:
            N = number of neurons, i.e. no. of possible positions
            T = number of time-steps
            d_points = datapoints
        '''
        self.d_points = d_points

        self.positions, self.probabilities = data_generate(d_points, neurons, timesteps)

    def __len__(self):
        return len(self.positions[:,0,0])

    def __getitem__(self, idx):
        pos = torch.from_numpy(self.positions[idx])
        prob = torch.from_numpy(self.probabilities[idx])

        sample = {'positions': pos,
                  'probabilities': prob}

        return sample




def Dataloaders(d_points, neurons, timesteps, batchsize_train, batchsize_val, train_size = 0.7, val_size = 0.1):
    full_dataset = Dataset(d_points, neurons, timesteps)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset)-train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size,val_size, test_size])

    train_loader = data.DataLoader(train_dataset, batch_size = batchsize_train, shuffle=True, pin_memory=True)  
    val_loader   = data.DataLoader(val_dataset, batch_size = batchsize_val, shuffle = False)
    test_loader  = data.DataLoader(test_dataset, batch_size = batchsize_val, shuffle = False)   #can add num_workers as last argument here if using a gpu

    dataloaders = {'Train' : train_loader, 
                'Val' : val_loader, 
                'Teas': test_loader}

    return dataloaders



dataloader = Dataloaders(d_points = 100, neurons = 20, timesteps= 100, batchsize_train = 20, batchsize_val=10, train_size = 0.7, val_size = 0.1)


train = dataloader['Train']

