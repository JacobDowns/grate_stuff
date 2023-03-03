import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import UNet
from test_loader import test_dataset, y_normalizer
from utilities import LpLoss
import matplotlib.pyplot as plt
batch_size = 2

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)



model = UNet(2, in_channels=6, depth=3, merge_mode='concat')
model.load_state_dict(torch.load('data/state.pt'))
model.cuda()
model.eval()
myloss = LpLoss(size_average=False)
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        print(x.shape)
        print(y.shape)
        #quit()
        
        out = model(x)
        #out = y_normalizer.decode(out)        
        #y = y_normalizer.decode(y)
        #test_l2 += myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1)).item()

        x = x.cpu()
        out = out.cpu()
        y  = y.cpu()

        print(x.shape)
        print(y.shape)
        print(out.shape)
        #quit()

        for i in range(len(x)):
            plt.subplot(3,1,1)
            plt.imshow(x[i,0,:,:])
            plt.colorbar()

            plt.subplot(3,1,2)
            plt.imshow(y[i,0,:,:])
            plt.colorbar()

            plt.subplot(3,1,3)
            plt.imshow(out[i,0,:,:])
            plt.colorbar()

            plt.show()
