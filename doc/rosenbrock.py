import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.close('all')
fig = plt.figure()

ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = 100* (Y - X**2)**2+(X-1)**2


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, color='b',linewidth=0)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()
