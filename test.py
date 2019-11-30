import numpy as np


x_train = np.array([
    [1, 2],
    [3, 4],
    [4, 5]])

K = np.zeros((x_train[:,0].size, x_train[0,:].size))

print(x_train[:,0].size)
print(K)