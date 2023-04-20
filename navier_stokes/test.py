import torch
from neuralop.models import TFNO
from data_loader import train_dataset, test_dataset
from neuralop import LpLoss, H1Loss
from neuralop import Trainer
from timeit import default_timer
import numpy as np
import matplotlib.pyplot as plt


batch_size = 50

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

model = TFNO(n_modes=(16, 16), hidden_channels=32,
            in_channels=22,
            out_channels=2,
            factorization='tucker',
            implementation='factorized',
            rank=0.05,
            n_layers=4
)

model.load_state_dict(torch.load('data/state.pt'))
model = model.cuda()
model.eval()

myloss = LpLoss(d=2, p=2)


test_l2 = 0.0
ntest = 0
with torch.no_grad():
    for sample in test_loader:
        x = sample['x']
        y = sample['y']
        x, y = x.cuda(), y.cuda()

        out = model(x)

        x = x.cpu()
        y = y.cpu()
        out = out.cpu()
        
        for i in range(len(y)):
            plt.subplot(3,2,1)
            plt.imshow(y[i,0,:,:])
            plt.colorbar()

            plt.subplot(3,2,2)
            plt.imshow(out[i,0,:,:])
            plt.colorbar()

            plt.subplot(3,2,3)
            plt.imshow(y[i,1,:,:])
            plt.colorbar()

            plt.subplot(3,2,4)
            plt.imshow(out[i,1,:,:])
            plt.colorbar()


            plt.subplot(4,2,5)
            plt.imshow(y[i,0,:,:] - out[i,0,:,:])
            plt.colorbar()

            plt.subplot(3,2,6)
            plt.imshow(y[i,1,:,:] - out[i,1,:,:])
            plt.colorbar()

            
            """
            Qx = np.gradient(out[i,0,:,:], axis=0) / 256.
            Vy = np.gradient(out[i,1,:,:], axis=1) / 256. 

            plt.subplot(4,2,7)
            plt.imshow(Qx + Vy)
            plt.colorbar()
            """
            plt.show()

        test_l2 += myloss(out,y).sum().item()
        test_errors.append(test_l2)
        ntest += 1

test_l2 /= ntest
print('Test err: ', test_l2)

