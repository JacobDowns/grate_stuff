
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt

X = np.load('data/X.npy')
Y = np.load('data/Y.npy')

# Resolution
nx = X.shape[1]
ny = X.shape[2]
# Number of training examples
N = X.shape[0]

X = torch.tensor(X)
Y = torch.tensor(Y)

x_mean = X.mean()
x_std = X.std()


# Gaussian normalizer for each channel
class GaussianNormalizer(object):
    def __init__(self, x_mean, x_std):
        super(GaussianNormalizer, self).__init__()
        self.mean = x_mean
        self.std = x_std

    def encode(self, x):
        x = (x - self.mean) / self.std
        return x

    def decode(self, x, sample_idx=None):
        x = (x * self.std) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class MyDataset(torch.utils.data.TensorDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        return {'x': x, 'y': y}
    
    def __len__(self):
        return len(self.x)


normalizer = GaussianNormalizer(x_mean, x_std)

X = normalizer.encode(X)
Y = normalizer.encode(Y)
add_grid = True

if add_grid:
    # Append x, y coordinate grid
    xx, yy = np.meshgrid(
        np.linspace(0.,1.,nx),
        np.linspace(0.,1.,ny)
    )
    xx = torch.tensor(xx.astype(np.float32))
    yy = torch.tensor(yy.astype(np.float32))

    xx = xx.unsqueeze(0).repeat(X.shape[0],1,1).unsqueeze(-1)
    yy = yy.unsqueeze(0).repeat(X.shape[0],1,1).unsqueeze(-1)

    X = torch.cat((X, xx, yy), dim=-1)


X = X.permute(0,3,1,2)
Y = Y.permute(0,3,1,2)


ntrain = int(0.75 * N)

X_train = X[0:ntrain]
Y_train = Y[0:ntrain] 

X_test = X[(ntrain):]
Y_test = Y[(ntrain):]

np.save('data/X_test.npy', X_test.numpy())
np.save('data/Y_test.npy', Y_test.numpy())

train_dataset = MyDataset(X_train, Y_train)
test_dataset = MyDataset(X_test, Y_test)
