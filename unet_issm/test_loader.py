import torch
import numpy as np
from utilities import DataNormalizer
import matplotlib.pyplot as plt

i = 23

X = np.load('test_data/X_{}.npy'.format(i))
Y = np.load('test_data/Y_{}.npy'.format(i))

xx, yy = np.meshgrid(np.arange(X.shape[3], dtype=float), np.arange(X.shape[2], dtype=float))
xx /= 32.
yy /= 32.


X = torch.tensor(X, dtype=torch.float).permute(0,2,3,1)
Y = torch.tensor(Y, dtype=torch.float).permute(0,2,3,1)
X1 = np.zeros((X.shape[0], X.shape[1], X.shape[2], 2))
X1[:, :, :, 0] = xx
X1[:, :, :, 1] = yy
X1 = torch.tensor(X1, dtype=torch.float)
X = torch.concatenate([X, X1], axis=-1)

ntrain = int(0.85*X.shape[0])
ntest = X.shape[0] - ntrain

x_normalizer = DataNormalizer([1e3, 1., 1.,1e9,1., 1.])
y_normalizer = DataNormalizer([1e2, 1e2])

# Normalize inputs
X = x_normalizer.encode(X).permute(0,3,1,2)
Y = y_normalizer.encode(Y).permute(0,3,1,2)
#X = X[:, :, 0:136, 0:128]
#Y = Y[:, :, 0:136, 0:128]

#print(X.shape)
#print(Y.shape)
#quit()

test_dataset = torch.utils.data.TensorDataset(X, Y)
