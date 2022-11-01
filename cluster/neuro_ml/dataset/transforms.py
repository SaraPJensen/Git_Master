import torch
import numpy as np
import random


class Subset(object):
    """
    Ramdomly remove n_remove of the nodes/neurons from the dataset, both from X and W0
    """

    def __init__(self, n_remove, n_initial):
        self.n_remove = n_remove      #Number of neurons to remove
        self.n_initial = n_initial    #The initial number of neurons

    def __call__(self, X, W0):
        
        delete = np.random.choice(range(self.n_initial), size= (self.n_remove), replace = False)

        all = np.arange(0, self.n_initial)

        keep = [neuron for neuron in all if neuron not in delete]

        # print("delete: ", delete)
        # print("keep: ", keep)

        X = X[keep, :]
        W0 = W0[:, keep]
        W0 = W0[keep, :]

        # print("X shape: ", X.shape)
        # print("W0 shape: ", W0.shape)

        return X, W0



'''
n_neurons = 10
timesteps = 12

testobject = Subset(2, n_neurons)

X_init = torch.arange(n_neurons*timesteps).reshape((n_neurons, timesteps))

W0_init = torch.arange(n_neurons * n_neurons).reshape((n_neurons, n_neurons))

print("X_init shape: ", X_init.shape)


X, W0 = testobject(X_init, W0_init)

print(X_init)
print()
print(X)
print()
print()
print()
print(W0_init)
print()
print(W0)
'''