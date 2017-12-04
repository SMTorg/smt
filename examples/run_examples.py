'''
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich>
        Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
'''

from __future__ import print_function, division
import numpy as np
from scipy import linalg
from smt.utils import compute_rms_error

from smt.problems import Sphere, NdimRobotArm
from smt.sampling import LHS
from smt.methods import LS, QP, KPLS, KRG, KPLSK, GEKPLS
try:
    from smt.methods import IDW, RBF, RMTC, RMTB
    compiled_available = True
except:
    compiled_available = False

try:
    import matplotlib.pyplot as plt
    plot_status = True
except:
    plot_status = False

########### Initialization of the problem, construction of the training and validation points

ndim = 10
ndoe = int(10*ndim)
# Define the function
fun = Sphere(ndim = ndim)

# Construction of the DOE
sampling = LHS(xlimits=fun.xlimits,criterion = 'm')
xt = sampling(ndoe)
# Compute the output
yt = fun(xt)
# Compute the gradient
for i in range(ndim):
    yd = fun(xt,kx=i)
    yt = np.concatenate((yt,yd),axis=1)

# Construction of the validation points
ntest = 500
sampling = LHS(xlimits=fun.xlimits)
xtest = sampling(ntest)
ytest = fun(xtest)
ydtest = np.zeros((ntest,ndim))
for i in range(ndim):
    ydtest[:,i] = fun(xtest,kx=i).T

########### The LS model

# Initialization of the model
t = LS(print_prediction = False)
# Add the DOE
t.set_training_values(xt,yt[:,0])

# Train the model
t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('LS,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('LS, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


########### The QP model

t = QP(print_prediction = False)
t.set_training_values(xt,yt[:,0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('QP,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('QP, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


########### The Kriging model

# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[1e-2]*ndim,print_prediction = False)
t.set_training_values(xt,yt[:,0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('Kriging,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('Kriging, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


# Variability of the model for any x
variability = t.predict_variances(xtest)

########### The KPLS model
# The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
# an integer in [1,ndim[ and a list of length n_comp, respectively. Here is an
# an example using 2 principal components.

t = KPLS(n_comp=2, theta0=[1e-2,1e-2],print_prediction = False)
t.set_training_values(xt,yt[:,0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('KPLS,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('KPLS, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


# Variability of the model for any x
variability = t.predict_variances(xtest)

########### The KPLSK model
# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.

t = KPLSK(n_comp=2, theta0=[1e-2,1e-2],print_prediction = False)
t.set_training_values(xt,yt[:,0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('KPLSK,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('KPLSK, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


# Variability of the model for any x
variability = t.predict_variances(xtest)

########### The GEKPLS model using 1 approximating points
# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.

t = GEKPLS(n_comp=1, theta0=[1e-2], xlimits=fun.xlimits,delta_x=1e-4,extra_points= 1,print_prediction = False)
t.set_training_values(xt,yt[:,0])
# Add the gradient information
for i in range(ndim):
    t.set_training_derivatives(xt,yt[:, 1+i].reshape((yt.shape[0],1)),i)

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('GEKPLS1,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('GEKPLS1, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


# Variability of the model for any x
variability = t.predict_variances(xtest)

########### The GEKPLS model using 2 approximating points
# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.

t = GEKPLS(n_comp=1, theta0=[1e-2], xlimits=fun.xlimits,delta_x=1e-4,
           extra_points= 2,print_prediction = False)
t.set_training_values(xt,yt[:,0])
# Add the gradient information
for i in range(ndim):
    t.set_training_derivatives(xt,yt[:, 1+i].reshape((yt.shape[0],1)),i)

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('GEKPLS2,  err: '+str(compute_rms_error(t,xtest,ytest)))
if plot_status:
    plt.figure()
    plt.plot(ytest,ytest,'-.')
    plt.plot(ytest,y,'.')
    plt.xlabel(r'$y$ True')
    plt.ylabel(r'$y$ prediction')
    plt.show()

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
    print('GEKPLS2, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
    if plot_status:
        plt.figure()
        plt.plot(ydtest[:,i],ydtest[:,i],'-.')
        plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
        plt.xlabel(r'$y$ derivative True')
        plt.ylabel(r'$y$ derivative prediction')
        plt.show()


# Variability of the model for any x
variability = t.predict_variances(xtest)

if compiled_available:
    ########### The IDW model

    t = IDW(print_prediction = False)
    t.set_training_values(xt,yt[:,0])

    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('IDW,  err: '+str(compute_rms_error(t,xtest,ytest)))
    if plot_status:
        plt.figure()
        plt.plot(ytest,ytest,'-.')
        plt.plot(ytest,y,'.')
        plt.xlabel(r'$y$ True')
        plt.ylabel(r'$y$ prediction')
        plt.show()

    ########### The RBF model

    t = RBF(print_prediction = False,poly_degree = 0)
    t.set_training_values(xt,yt[:,0])

    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('RBF,  err: '+str(compute_rms_error(t,xtest,ytest)))
    if plot_status:
        plt.figure()
        plt.plot(ytest,ytest,'-.')
        plt.plot(ytest,y,'.')
        plt.xlabel(r'$y$ True')
        plt.ylabel(r'$y$ prediction')
        plt.show()

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest,ndim))
    for i in range(ndim):
        yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
        print('RBF, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
        if plot_status:
            plt.figure()
            plt.plot(ydtest[:,i],ydtest[:,i],'-.')
            plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
            plt.xlabel(r'$y$ derivative True')
            plt.ylabel(r'$y$ derivative prediction')
            plt.show()


    ########### The RMTB and RMTC models are suitable for low-dimensional problems
    # Initialization of the problem
    ndim = 3
    ndoe = int(250*ndim)
    # Define the function
    fun = NdimRobotArm(ndim=ndim)

    # Construction of the DOE
    sampling = LHS(xlimits=fun.xlimits)
    xt = sampling(ndoe)

    # Compute the output
    yt = fun(xt)
    # Compute the gradient
    for i in range(ndim):
        yd = fun(xt,kx=i)
        yt = np.concatenate((yt,yd),axis=1)

    # Construction of the validation points
    ntest = 500
    sampling = LHS(xlimits=fun.xlimits)
    xtest = sampling(ntest)
    ytest = fun(xtest)

    ########### The RMTB model

    t = RMTB(xlimits=fun.xlimits, min_energy=True, nonlinear_maxiter=20,print_prediction = False)
    t.set_training_values(xt,yt[:,0])
    # Add the gradient information
    for i in range(ndim):
        t.set_training_derivatives(xt,yt[:, 1+i].reshape((yt.shape[0],1)),i)
    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('RMTB,  err: '+str(compute_rms_error(t,xtest,ytest)))
    if plot_status:
        plt.figure()
        plt.plot(ytest,ytest,'-.')
        plt.plot(ytest,y,'.')
        plt.xlabel(r'$y$ True')
        plt.ylabel(r'$y$ prediction')
        plt.show()

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest,ndim))
    for i in range(ndim):
        yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
        print('RMTB, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
        if plot_status:
            plt.figure()
            plt.plot(ydtest[:,i],ydtest[:,i],'-.')
            plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
            plt.xlabel(r'$y$ derivative True')
            plt.ylabel(r'$y$ derivative prediction')
            plt.show()

    ########### The RMTC model

    t = RMTC(xlimits=fun.xlimits, min_energy=True, nonlinear_maxiter=20,print_prediction = False)
    t.set_training_values(xt,yt[:,0])
    # Add the gradient information
    for i in range(ndim):
        t.set_training_derivatives(xt,yt[:, 1+i].reshape((yt.shape[0],1)),i)

    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('RMTC,  err: '+str(compute_rms_error(t,xtest,ytest)))
    if plot_status:
        plt.figure()
        plt.plot(ytest,ytest,'-.')
        plt.plot(ytest,y,'.')
        plt.xlabel(r'$y$ True')
        plt.ylabel(r'$y$ prediction')
        plt.show()

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest,ndim))
    for i in range(ndim):
        yd_prediction[:,i] = t.predict_derivatives(xtest,kx=i).T
        print('RMTC, err of the '+str(i)+'-th derivative: '+ str(compute_rms_error(t,xtest,ydtest[:,i],kx=i)))
        if plot_status:
            plt.figure()
            plt.plot(ydtest[:,i],ydtest[:,i],'-.')
            plt.plot(ydtest[:,i],yd_prediction[:,i],'.')
            plt.xlabel(r'$y$ derivative True')
            plt.ylabel(r'$y$ derivative prediction')
            plt.show()
