import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob

"""
X_train = np.load('X_train_masked.npy')
Y_train = np.load('Y_train_masked.npy')

print(X_train.shape)
print(Y_train.shape)

k = 190

plt.subplot(4,1,1)
plt.imshow(Y_train[k,:,:,0], vmin = -50, vmax = 50.)
plt.colorbar()

plt.subplot(4,1,2)
plt.imshow(Y_train[k,:,:,1], vmin = -50., vmax = 50.)
plt.colorbar()

plt.subplot(4,1,3)
plt.imshow(Y_train[k,:,:,2])
plt.colorbar()

plt.subplot(4,1,4)
plt.imshow(X_train[k,:,:,0])
plt.colorbar()

plt.show()
quit()
"""

training_files = glob.glob('train/*.nc')
test_files = glob.glob('test/*.nc')

def get_xy(file_name):
    data = xr.open_dataset(file_name)
    print(data)

   
    """
    Collect all necessary model inputs for the emulator.
    """
    n = 2
    mask = data.MaskIceLevelset.data[::n]
    mask[mask > 0.] = 0.
    mask[mask < 0.] = 1.

    # Thickness
    H = data.Thickness.data[::n]
    H[H == 10.] = 0.
    # Surface gradient
    Sx = data.Sx.data[::n]
    Sy = data.Sy.data[::n]
    # Basal traction coefficient
    beta = data.beta.data
    beta = np.tile(beta, (len(H), 1, 1))
    # Ice hardness
    B = data.rheology_B.data
    B = np.tile(B, (len(H), 1, 1))
    # Full array of inputs
    x = np.stack([H, Sx, Sy, beta, B])

    """
    Assemble all necessary outputs for emulator.
    """
    
    # Surface velocity vector
    Vx = data.Vx.data[::n]*mask
    Vy = data.Vy.data[::n]*mask
    #Vx = data.Vx.data[::n]
    #Vy = data.Vy.data[::n]
    # Fraction of velocity due to sliding
    VelSurf = data.Vel.data[::n]
    VelBase = data.VelBase.data[::n]
    sliding_frac = np.clip(VelBase / (VelSurf + 1e-3), 0., 1.)
    y = np.stack([Vx, Vy, sliding_frac])

    x = np.transpose(x, (1,2,3,0))
    y = np.transpose(y, (1,2,3,0))
    return x, y

    
def get_XY(file_names):

    xs = []
    ys = []

    nmax = 0
    kmax = 0
    for file_name in file_names:
        x, y = get_xy(file_name)

        nmax = max(nmax, x.shape[1])
        kmax = max(kmax, x.shape[2])

        xs.append(x)
        ys.append(y)

    
    X = []
    Y = []

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        px = nmax - x.shape[1]
        py = kmax - y.shape[2]

        x = np.pad(x, ((0,0), (0,px), (0,py), (0,0)))
        x = np.roll(x, int(px/2), axis = 1)
        x = np.roll(x, int(py/2), axis = 2)

        y = np.pad(y, ((0,0), (0,px), (0,py), (0,0)))
        y = np.roll(y, int(px/2), axis = 1)
        y = np.roll(y, int(py/2), axis = 2)

        X.append(x)
        Y.append(y)

    X = np.concatenate(X, axis=0).astype(np.float32)
    Y = np.concatenate(Y, axis=0).astype(np.float32)

    return X, Y

X_train, Y_train = get_XY(training_files)
np.save('X_train_masked.npy', X_train)
np.save('Y_train_masked.npy', Y_train)

X_test, Y_test = get_XY(test_files)
np.save('X_test_masked.npy', X_test)
np.save('Y_test_masked.npy', Y_test)
