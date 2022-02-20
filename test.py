import numpy as np
import torch




seq_len = 10

sample_size = 5



array = np.empty((sample_size, seq_len))


for i in range (sample_size):
    start = np.random.randint(0, 10)

    #start = 9

    for j in range(seq_len):
        array[i, j] = start

        start *= 2


'''
print(array.shape)

print(array)


data = torch.from_numpy(array)

print(data.size()[0])


print()

print(array[0])

print()

'''
priya = np.array([np.sort(np.random.choice(np.arange(1,10000), seq_len, replace=False, p= (1/9999)*np.ones(9999) )) for i in np.arange(sample_size)])

print(priya)

print()
print(priya[0])

print()

idx = 3

inp_data = priya[idx]
                        
# predict the next number (adding 1)
change = inp_data - np.roll(inp_data, 1)
change[0] = 0 
labels = inp_data + change

labels = np.roll(inp_data, 1)

#print(change)
print(inp_data)
print(labels)

