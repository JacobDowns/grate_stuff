import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer
from fourier_neural_operator import FNO2d
from utilities import DataNormalizer, LpLoss
import random

X = np.load('data/X_test.npy')
Y = np.load('data/Y_test.npy')
X = torch.tensor(X)
Y = torch.tensor(Y)

x_normalizer = DataNormalizer(np.array([1e3, 1., 1.]).astype(np.float32))
y_normalizer = DataNormalizer(np.array([1e3, 1e3]).astype(np.float32))
X = x_normalizer.encode(X)
Y = y_normalizer.encode(Y)

batch_size = 15
test_dataset = torch.utils.data.TensorDataset(X, Y)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

"""
Loop through examples from a glacier that hasn't been seen in training 
and see if the emulator is any good.
"""

modes = 15
width = 30
model = FNO2d(modes, modes, width).cuda()
model.load_state_dict(torch.load('data/state.pt'))
ntest = X.shape[0]
myloss = LpLoss(size_average=False)
y_normalizer.cuda()
model.eval()

test_l2 = 0.0
with torch.no_grad():
     
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        out = model(x)
        out = y_normalizer.decode(out)        
        y = y_normalizer.decode(y)

        x = x.cpu()
        out = out.cpu()
        y  = y.cpu()

        test_l2 += myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1)).item()

        for i in range(x.shape[0]):
            plt.subplot(4,2,1)
            plt.title('Ice Thickness (m)')
            plt.imshow(x[i,:,:,0]*1e3)
            plt.colorbar()
    
            plt.subplot(4,2,3)
            plt.title('u (m/a)')
            plt.imshow(y[i,:,:,0])
            plt.colorbar()

            plt.subplot(4,2,4)
            plt.title('u emulator (m/a)')
            plt.imshow(out[i,:,:,0])
            plt.colorbar()
            
            plt.subplot(4,2,5)
            plt.title('v (m/a)')
            plt.imshow(y[i,:,:,1])
            plt.colorbar()

            plt.subplot(4,2,6)
            plt.title('v emulator (m/a)')
            plt.imshow(out[i,:,:,1])
            plt.colorbar()

            plt.subplot(4,2,7)
            plt.title('u mismatch (m/a)')
            plt.imshow(y[i,:,:,0] - out[i,:,:,0])
            plt.colorbar()


            plt.subplot(4,2,8)
            plt.title('v mismatch (m/a)')            
            plt.imshow(y[i,:,:,1] - out[i,:,:,1], vmin = -50., vmax = 50.)
            plt.colorbar()
            
            plt.show()

    test_l2 /= ntest
    print(test_l2)
