import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def gen_field(nx, ny, correlation_scale):

    z = xr.zeros_like((nx, ny))

    # Create the smoothing kernel
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    # Smooth the random noise
    noise = np.random.randn(*H.data[0].shape) 
    z.data[0] = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        
    # Normalize so its in 0-1 range
    z.data[0] -= z.data[0].min()
    z.data[0] /= z.data[0].max()

    #plt.imshow(z.data[0]**2)
    #plt.colorbar()
    #plt.show()
    return z



z = gen_field(250, 250, 100)
plt.imshow(z)
plt.colorbar()
plt.show()
