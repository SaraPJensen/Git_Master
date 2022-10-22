import numpy as np

# data = load('0.npz')
# lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])

data = np.load('0.npz')        # data contains x = [1,2,3,4,5]

for key in data.keys():
    print(key)                        # x
    #print(data[key])                  # [1,2,3,4,5]