
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.close('all')

#a = 1
###cos
fig = plt.figure()

ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)

Z = np.cos(np.pi * X)*np.cos(np.pi * Y)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, color='b',linewidth=0)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()

###exp

fig = plt.figure()

ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)

Z = np.exp(X)*np.exp(Y)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, color='b',linewidth=0)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()


###tnah

fig = plt.figure()

ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)

Z = np.tanh(X)*np.tanh(Y)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, color='b',linewidth=0)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()

###Gaussian

fig = plt.figure()

ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)

Z = np.exp(-2*X**2)*np.cos(-2*Y**2)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, color='b',linewidth=0)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()
