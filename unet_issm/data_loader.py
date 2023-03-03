import torch
import numpy as np
from utilities import DataNormalizer
import matplotlib.pyplot as plt

X = np.load('training_data/X.npy')
Y = np.load('training_data/Y.npy')

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

#print(torch.std(X[:,3,:,:], dim=(0,1,2)))
#print(torch.max(X[:,3,:,:]))

ntrain = int(0.85*X.shape[0])
ntest = X.shape[0] - ntrain

x_normalizer = DataNormalizer([1e3, 1., 1.,1e9,1., 1.])
y_normalizer = DataNormalizer([1e2, 1e2])

# Normalize inputs
X = x_normalizer.encode(X).permute(0,3,1,2)
Y = y_normalizer.encode(Y).permute(0,3,1,2)

# Split into train / test data
X_train = X[0:ntrain]
Y_train = Y[0:ntrain]

X_test = X[ntrain:]
Y_test = Y[ntrain:]

"""
print(X_train.shape)
import matplotlib.pyplot as plt
plt.imshow(X_train[2000,4,:,:])
plt.colorbar()
plt.show()

plt.imshow(Y_train[2000,0,:,:])
plt.colorbar()
plt.show()
quit()
"""

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
