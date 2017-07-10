import numpy as np
from smt.problems import Sphere
from smt.sampling import LHS
from GEKPLS_QKG import provide_data

# Initialization of the problem
ndim = 4
ndoe = int(5*ndim)

# Define the function
fun = Sphere(ndim = ndim)

# Construction of the DOE
sampling = LHS(xlimits=fun.xlimits,criterion = 'm')
x = sampling(ndoe)

# Compute the output
y = fun(x)

# Compute the gradient
yd = np.zeros((ndoe,ndim))
for i in range(ndim):
    yd[:,i] = fun(x,kx=i).reshape((1,ndoe))

est_theta, x_new, y_new = provide_data(x,y,yd,fun.xlimits)
print est_theta
print x_new
print y_new.T
