import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
from torch import nn



def load_dataset(i):
    base_dir = '/home/jake/grate/training_random3/training_data/'
    data = xr.load_dataset(base_dir + str(i) + '.nc')

    # Inputs
    H = data.Thickness.data
    sx = data.Sx.data
    sy = data.Sy.data
    C = data.C.data
    X = np.stack([H, sx, sy, C]).astype(np.float32)

    # Outputs
    vx = data.Vx.data
    vy = data.Vy.data
    Y = np.stack([vx, vy]).astype(np.float32)

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    X = X.permute(1,0,2,3)
    Y = Y.permute(1,0,2,3)
    
    np.save('test_data/X_{}.npy'.format(i), X)
    np.save('test_data/Y_{}.npy'.format(i), Y)
    
    
    # Stupid way of generating patches
    kx = 32
    ky = 32
    u = nn.Unfold(kernel_size = (kx, ky), stride=16)

    X0 = u(X)
    X0 = X0.reshape(X.shape[0], 4, kx, ky, X0.shape[-1])
    X0 = X0.permute(0,4,2,3,1)
    X0 = X0.reshape(X0.shape[0]*X0.shape[1], kx, ky, 4)

    Y0 = u(Y)
    Y0 = Y0.reshape(Y.shape[0], 2, kx, ky, Y0.shape[-1])
    Y0 = Y0.permute(0,4,2,3,1)
    Y0 = Y0.reshape(Y0.shape[0]*Y0.shape[1], kx, ky, 2)

    indexes = torch.amax(X0[:,:,:,0], dim=(1,2)) > 0.1
    X0 = X0[indexes]
    Y0 = Y0[indexes]

    # Add x, y coords
    xx, yy = np.meshgrid(np.linspace(0.,1.,kx), np.linspace(0.,1.,ky))
    X1 = np.zeros((X0.shape[0], X0.shape[1], X0.shape[2], 2))
    X1[:, :, :, 0] = xx
    X1[:, :, :, 1] = yy
    X1 = torch.tensor(X1)
    X0 = torch.concatenate([X0, X1], axis=-1)

    return X0, Y0

def load_datasets():

    X = []
    Y = []
    for i in range(24):
        print(i)
        x, y = load_dataset(i)
        X.append(x)
        Y.append(y)        

    X = torch.cat(X)
    Y = torch.cat(Y)
    print(X.shape)
    print(Y.shape)

    np.save('X.npy', X.numpy().astype(np.float32))
    np.save('Y.npy', Y.numpy().astype(np.float32))

load_datasets()
