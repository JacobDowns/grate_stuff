import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import UNet
from utilities import LpLoss
from timeit import default_timer
import matplotlib.pyplot as plt
from data_loader import train_dataset, test_dataset, y_normalizer
import random

batch_size = 2500
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Number of times to iterate through all data
epochs = 1500
model = UNet(2, in_channels=6, depth=3, merge_mode='concat')
model.cuda()

learning_rate = 0.001
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
#y_normalizer.cuda()

myloss = LpLoss(size_average=False)
#y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    k = 0
    ntrain = 0
    for x, y in train_loader:

        #print(k)
        x, y = x.cuda(), y.cuda()
        grid = x[:,:,:,[-2,-1]].clone()

        j = random.randint(0,3)
        #print(j)
        if j == 1:
            x = torch.flip(x, [2])
            y = torch.flip(y, [2])
        elif j == 2:
            x = torch.flip(x, [3])
            y = torch.flip(y, [3])
        elif j == 3:
            x = torch.rot90(x, 1, (2,3))
            y = torch.rot90(y, 1, (2,3))

        x[:,:,:,[-2,-1]] = grid    
        optimizer.zero_grad()        

        batch_size = x.shape[0]
        out = model(x)
        
        """
        for i in range(10):
            plt.subplot(2,1,1)
            plt.imshow(out[i,0,:,:].detach().cpu())
            plt.colorbar()

            plt.subplot(2,1,2)
            plt.imshow(y[i,0,:,:].detach().cpu())
            plt.colorbar()
            plt.show()
        """

        loss = myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()
        ntrain += 1
        k += 1

    if ep % 25 == 0:
        torch.save(model.state_dict(), 'data/state.pt')

        # This test loop just helps monitor how well the emulator is generalizing
        model.eval()
        test_l2 = 0.0
        ntest = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                
                out = model(x)

                """
                for i in range(10):
                    plt.subplot(2,1,1)
                    plt.imshow(out[i,0,:,:].cpu())
                    plt.colorbar()

                    plt.subplot(2,1,2)
                    plt.imshow(y[i,0,:,:].cpu())
                    plt.colorbar()
                    plt.show()
                """
                #out = y_normalizer.decode(out)        
                #y = y_normalizer.decode(y)
                test_l2 += myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1)).item()
            
                ntest += 1

        test_l2 /= ntest
        print('Test err: ', test_l2)
        
    train_l2/= ntrain
    t2 = default_timer()
    print(ep, t2-t1, train_l2)
