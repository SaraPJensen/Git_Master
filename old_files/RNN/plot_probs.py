from cmath import nan
import pandas as pd 
import numpy as np

def plotting(filename, neurons):

    data=pd.read_csv(filename, header = None)

    p1_pos = data.iloc[::2, :]
    p2_pos = data.iloc[1::2,:]

    p1_pos = p1_pos.to_numpy()
    p2_pos = p2_pos.to_numpy()

    d_points = len(p2_pos[:, 0])
    timesteps = len(p2_pos[0, :])

    p2_diff = np.zeros_like(p2_pos)

    for row in range(d_points):
        for col in range (timesteps - 1):
            p2_diff[row, col] = (p2_pos[row, col + 1] - p2_pos[row, col]) 

            if p2_diff[row, col] == neurons - 1:
                p2_diff[row, col] = -1

            elif p2_diff[row, col] == - (neurons - 1):
                p2_diff[row, col] = 1
            

    zeros = np.zeros_like(p2_diff)

    p2_probs = np.maximum(p2_diff, zeros)  #set -1 to 0 to turn them into the probability of moving upwards (it always moves up or down)

    print()
    print("Average probability of p2 moving upwards")
    print(np.mean(p2_probs))

    

    all_cond_probs = np.empty([d_points, neurons])

    for row in range(d_points):
        for i in range(neurons):
            indices = np.where(p1_pos[row, :] == i)[0]
            
            count = len(indices)
            prob = 0
            for idx in indices:
                prob += p2_probs[row, idx]

            if count > 0:
                all_cond_probs[row, i] = prob/count 
            else: 
                all_cond_probs[row, i] = nan

            
    
    cond_probs = []
    for i in range(neurons):
        cond_probs.append(np.nanmean(all_cond_probs[:, i]))

    #print(cond_probs)

    print("Chance of particle 2 moving upwards conditional on the position of particle 1")
    print("Pos.   ---    Probability")
    for i in range(neurons):
        print(f"{i}      ---    {cond_probs[i]:5f}")

    
    




if __name__ == '__main__':
    filename = "test_true.csv" 
    neurons = 6

    plotting(filename, neurons)