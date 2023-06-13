import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def gen_field(nx, ny, correlation_scale):

    # Create the smoothing kernel
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    # Generate random noise and smooth it
    noise = np.random.randn(nx, ny) 
    z = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        
    # Normalize so its in 0-1 range
    z -= z.min()
    z /= z.max()

    return z

z = gen_field(250, 250, 100)
plt.imshow(z)
plt.colorbar()
plt.show()
