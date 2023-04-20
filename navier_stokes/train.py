import torch
from neuralop.models import TFNO
from data_loader import train_dataset, test_dataset
from neuralop import LpLoss, H1Loss
from neuralop import Trainer
from timeit import default_timer
import numpy as np
import random
import matplotlib.pyplot as plt

device = 'cuda'

batch_size = 50
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

epochs = 2000
model = TFNO(n_modes=(16, 16), hidden_channels=32,
            in_channels=22,
            out_channels=2,
            factorization='tucker',
            implementation='factorized',
            rank=0.05,
            n_layers=4
)
model = model.cuda()
#model.load_state_dict(torch.load('data/state1.pt'))

learning_rate = 0.001
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

myloss = LpLoss(d=2, p=2)

train_errors = []
test_errors = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    k = 0
    ntrain = 0
    for sample in train_loader:
        x = sample['x']
        y = sample['y']
    
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()        
        batch_size = x.shape[0]
        out = model(x)
        
        loss = (myloss(out, y)**2).sum()
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()
        ntrain += 1
        k += 1

    if ep % 25 == 0:
        # This test loop just helps monitor how well the emulator is generalizing
        torch.save(model.state_dict(), 'data/state.pt')
        model.eval()
        test_l2 = 0.0
        ntest = 0
        with torch.no_grad():
            for sample in test_loader:
                x = sample['x']
                y = sample['y']
                x, y = x.cuda(), y.cuda()
                
                out = model(x)
                test_l2 += (myloss(out,y)**2).sum().item()
                test_errors.append(test_l2)
                ntest += 1

        test_l2 /= ntest
        print('Test err: ', test_l2)
        
    train_l2/= ntrain
    train_errors.append(train_l2)
    t2 = default_timer()
    print(ep, t2-t1, train_l2)


test_errors = np.array(test_errors)
train_errors = np.array(train_errors)
np.save('data/train_errors.npy', train_errors)
np.save('data/test_errors.npy', test_errors)
torch.save(model.state_dict(), 'data/state1.pt')
