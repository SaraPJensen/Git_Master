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
        #new_size = self.n_initial - self.n_remove

        delete = np.random.choice(range(self.n_initial), size= (self.n_remove), replace = False)
        all = np.arange(0, self.n_initial)
        keep = [neuron for neuron in all if neuron not in delete]

        W_subset = torch.zeros((self.n_initial, self.n_initial))

        X = X[keep, :]
        W0 = W0[:, keep]
        W0 = W0[keep, :]

        return X, W0



class Weight_filtering(object):
    """
    Remove edges with a weight below a certain threshold
    """

    def __init__(self, weight_threshold):
        self.threshold = weight_threshold

    def __call__(self, W0):

        if self.threshold == 0:
            return W0

        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                if torch.abs(W0[i, j]) < self.threshold:
                    W0[i, j] = 0

        return W0



class Simplex_filtering(object):
    """
    Remove edges which do not form part of any simplex above a certain dimension
    """

    def __init__(self, simplex_threshold):
        self.threshold = simplex_threshold

    def __call__(self, W0, Maximal_complex_edges):

        if self.threshold < 2:
            return W0

        W0_filter = torch.zeros((W0.shape[0], W0.shape[1]))

        if len(Maximal_complex_edges) > self.threshold - 2:  #First element are the 1-simplices, second element are the 2-simplices etc. To keep the 2-simplices (threshold = 3), start at index 1
            remaining = Maximal_complex_edges[self.threshold - 2:]
            for dim in remaining:
                for edge in dim:
                    W0_filter[edge[0], edge[1]] = W0[edge[0], edge[1]]

        return W0_filter
            
