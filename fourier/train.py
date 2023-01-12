import xarray as xr
import numpy as np
import torch
from timeit import default_timer
from fourier_neural_operator import FNO2d
from utilities import DataNormalizer, LpLoss
import random


"""
Trains a Fourier neural operator with example inputs of ice thickness + surface gradient,
and outputs of surface velocity gradients.
"""

# Divide data into training and test sets. This isn't completely necessary.
train = 0.8
X = np.load('data/X.npy')
Y = np.load('data/Y.npy')
X = torch.tensor(X)
Y = torch.tensor(Y)
ntrain = int(0.8*X.shape[0])
ntest = X.shape[0] - ntrain

x_normalizer = DataNormalizer(np.array([1e3, 1., 1.]).astype(np.float32))
y_normalizer = DataNormalizer(np.array([1e3, 1e3]).astype(np.float32))
X = x_normalizer.encode(X)
Y = y_normalizer.encode(Y)

dataset = torch.utils.data.TensorDataset(X, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [ntrain, ntest])

batch_size = 15
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
epochs = 7500
iterations = epochs*(ntrain//batch_size)
modes = 15
width = 30

model = FNO2d(modes, modes, width).cuda()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
y_normalizer.cuda()

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        """
        Apply a random transformation to input and output. 
        Helps ensure the emulator isn't overfitting.
        """
        
        j = random.randint(0,3)
        if j == 1:
            x = torch.flip(x, [1])
            y = torch.flip(y, [1])
        elif j == 2:
            x = torch.flip(x, [2])
            y = torch.flip(y, [2])
        elif j == 3:
            x = torch.rot90(x, 1, (1,2))
            y = torch.rot90(y, 1, (1,2))
            
        optimizer.zero_grad()

        batch_size = x.shape[0]
        out = model(x)
        out = y_normalizer.decode(out)        
        y = y_normalizer.decode(y)
        
        loss = myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()


    # This test loop just helps monitor how well the emulator is generalizing
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            j = random.randint(0,3)

            if j == 1:
                x = torch.flip(x, [1])
                y = torch.flip(y, [1])
            elif j == 2:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])
            elif j == 3:
                x = torch.rot90(x, 1, (1,2))
                y = torch.rot90(y, 1, (1,2))
            
            out = model(x)
            out = y_normalizer.decode(out)        
            y = y_normalizer.decode(y)
            test_l2 += myloss(out.reshape(out.shape[0],-1), y.reshape(out.shape[0],-1)).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)

# Save the trained model parameters
torch.save(model.state_dict(), 'data/state.pt')
