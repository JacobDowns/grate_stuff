import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer
from fourier_neural_operator import FNO2d
from utilities import DataNormalizer, LpLoss
import random
import scipy.ndimage

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

        x = scipy.ndimage.zoom(x, zoom = (1,2,2,1), order=1)[0:2]
        y = scipy.ndimage.zoom(y, zoom = (1,2,2,1), order=1)[0:2]
        x = torch.tensor(x)
        y = torch.tensor(y)


        x, y = x.cuda(), y.cuda()
    
        
        out = model(x)
        out = y_normalizer.decode(out)        
        y = y_normalizer.decode(y)

        x = x.cpu()
        out = out.cpu()
        y  = y.cpu()

        test_l2 += myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1)).item()

        cmap = plt.get_cmap('viridis', 24)

        for i in range(x.shape[0]):
            plt.subplot(2,1,1)
            plt.title('u (m/a)')
            plt.imshow(y[i,:,:,0], vmin = y[i,:,:,0].min(), vmax = y[i,:,:,0].max(), cmap = cmap)
            plt.colorbar()

            plt.subplot(2,1,2)
            plt.title('u emulator (m/a)')
            plt.imshow(out[i,:,:,0],  vmin = y[i,:,:,0].min(), vmax = y[i,:,:,0].max(), cmap = cmap)
            #plt.imshow(out[i,:,:,0] - y[i,:,:,0], cmap = cmap)

            plt.colorbar()

            plt.tight_layout()
            plt.show()
            quit()

    test_l2 /= ntest
    print(test_l2)
