
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt


data = loadmat('data/ns_train_data.mat')
U = data['u']
V = data['v']
W = data['w']

#quit()

# Number of time steps to input to the model
k = 10
# Spatial resolution
nx = U.shape[1]
ny = U.shape[2]
# Number of simulated timesteps
T = U.shape[-1]
indexes0 = np.arange(k)[np.newaxis, :] + np.arange(T-k)[:,np.newaxis]
indexes1 = np.arange(k, T)


def get_field(A):
    A0 = U[:,:,:,indexes0]
    A1 = U[:,:,:,indexes1]



    A0 = np.array(A0).astype(np.float32)
    A1 = np.array(A1).astype(np.float32)

    A0 = np.transpose(A0, (0,3,1,2,4))
    A1 = np.transpose(A1, (0,-1,1,2))


    A0 = np.concatenate(A0, axis=0)
    A1 = np.concatenate(A1, axis=0)
  
    return A0, A1
    
U0, U1 = get_field(U)
V0, V1 = get_field(V)


X = np.concatenate((U0, V0), axis=-1)
Y = np.stack((U1, V1), axis=-1)
np.save('data/X.npy', X)
np.save('data/Y.npy', Y)




for i in range(20):
    #plt.subplot)2,1,1)
    plt.imshow(X[0,:,:,i])
    plt.colorbar()

    #plt.subplot(2,1,2)
    #plt.imshow(U1[0,:,:,0,i])
    #plt.colorbar()

    plt.show()

quit()
