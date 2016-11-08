# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 15:24:43 2015

@author: MohamedAmine
"""

from __future__ import division

import os
import time
import numpy as np
from scipy.misc import derivative

def p1(x):
    '''
    xlimits[:, 0] = [5,0.1]
    xlimits[:, 1] = [10,1]
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1):
        return 2.1952/(x0**3*x1)
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        y[i,0] = func(x0,x1)
        point = [x0,x1]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p2(x):
    '''
    xlimits[:, 0] = [5,0.1,0.125,5]
    xlimits[:, 1] = [10,1,1,10]
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3):
        tau1 = 6000/(np.sqrt(2)*x2*x3)
        tau2 = 6000*(14+0.5*x3)*np.sqrt(0.25*(x3**2+(x2+x0)**2))/(2*(0.707*x2*x3*(x3/12+0.25*(x2+x0)**2)))
        return np.sqrt(tau1**2+tau2**2+x3*tau1*tau2/np.sqrt(0.25*(x3**2+(x2+x0)**2)))
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        y[i,0] = func(x0,x1,x2,x3)
        point = [x0,x1,x2,x3]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p3(x):
    '''
    xlimits[:, 0] = [5,0.1]
    xlimits[:, 1] = [10,1]
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1):
        return 504000/(x0**3*x1)
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        y[i,0] = func(x0,x1)
        point = [x0,x1]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p4(x):
    '''
    xlimits[:, 0] = [0.05,100,63070,990,63.1,700,1120,9855]
    xlimits[:, 1] = [0.15,50000,115600,1110,116,820,1680,12045]

    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7):
        return 2*np.pi*x2*(x3-x5)/(np.log(x1/x0)*(1+2*x6*x2/(np.log(x1/x0)*x0**2*x7)+x2/x4))
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7)
        point = [x0,x1,x2,x3,x4,x5,x6,x7]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p5(x):
    '''
    xlimits[:, 0] = 0
    xlimits[:, 1] = [1,1,1,1,2*np.pi,2*np.pi,2*np.pi,2*np.pi]

    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7):
        x = [x0,x1,x2,x3,x4,x5,x6,x7]
        s1,s2 = 0,0
        for i in range(4):
            s3 = 0
            for j in range(4,5+i):
                s3 += x[j]
            s1 += x[i]*np.cos(s3)
            s2 += x[i]*np.sin(s3)
        return (s1**2+s2**2)**0.5
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7)
        point = [x0,x1,x2,x3,x4,x5,x6,x7]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p6(x):
    '''
    xlimits[:, 0] = [150,220,6,-10,16,0.5,0.08,2.5,1700,0.025]
    xlimits[:, 1] = [200,300,10,10,45,1,0.18,6,2500,0.08]
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9):
        return 0.036*x0**0.758*x1**0.0035*(x2/np.cos(np.deg2rad(x3))**2)*x4**0.006*x5**0.04*(100*x6/np.cos(np.deg2rad(x3)))**(-0.3)*(x7*x8)**0.49+x0*x9
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        x8 = x[i,8]
        x9 = x[i,9]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9)
        point = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p7(x):
    '''
    xlimits[:, 0] = [1.8,9,0.252,7.2,0.09,10.8,1.6425,10.8,0.144,2.025,2.7,0.252,12.6,3.6,0.09]
    xlimits[:, 1] = [2.2,11,0.308,8.8,0.11,13.2,2.0075,13.2,0.176,2.475,3.3,0.308,15.4,4.4,0.11]
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14):
        return x2*np.pi*x1*(x0/2)**2+x8*np.pi*x7*(x6/2)**2+x4*np.pi*x3*(x9/2)**2+x11*np.pi*x10*(x5/2)**2+x14*np.pi*x13*(x12/2)**2
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        x8 = x[i,8]
        x9 = x[i,9]
        x10 = x[i,10]
        x11 = x[i,11]
        x12 = x[i,12]
        x13 = x[i,13]
        x14 = x[i,14]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14)
        point = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p8(x):
    '''
    xlimits[:, 0] = [1.8,9,10530000,7.2,3510000,10.8,1.6425,10.8,5580000,2.025,2.7,0.252,12.6,3.6,0.09]
    xlimits[:, 1] = [2.2,11,12870000,8.8,4290000,13.2,2.0075,13.2,6820000,2.475,3.3,0.308,15.4,4.4,0.11]
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14):
        K1 = np.pi*x2*x0/(32*x1)
        K2 = np.pi*x8*x6/(32*x7)
        K3 = np.pi*x4*x9/(32*x3)
        M1 = x11*np.pi*x10*x5/(4*9.80665)
        M2 = x14*np.pi*x13*x12/(4*9.80665)
        J1 = 0.5*M1*(x5/2)**2
        J2 = 0.5*M2*(x12/2)**2
        a = 1
        b = -((K1+K2)/J1+(K2+K3)/J2)
        c = (K1*K2+K2*K3+K3*K1)/(J1*J2)
        return np.sqrt((-b-np.sqrt(b**2-4*a*c))/(2*a))/(2*np.pi)
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        x8 = x[i,8]
        x9 = x[i,9]
        x10 = x[i,10]
        x11 = x[i,11]
        x12 = x[i,12]
        x13 = x[i,13]
        x14 = x[i,14]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14)
        point = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def p9(x):
    '''
    xlimits[:, 0] = [0.01]*17+[0.3]*17+[0.5]*17
    xlimits[:, 1] = [0.05]*17+[0.65]*17+[1]*17
    '''
    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50):
        x = np.hstack((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50))
        s = 0
        for i in range(17):
            s1,s2 = 0,0
            for j in range(i,17):
                s1 += x[34+j]
            for j in range(i+1,17):
                s2 += x[34+j]
            s += 12/(x[i]*x[17+i]**3)*(s1**3-s2**3)
        return 50/(3*200)*s
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        x8 = x[i,8]
        x9 = x[i,9]
        x10 = x[i,10]
        x11 = x[i,11]
        x12 = x[i,12]
        x13 = x[i,13]
        x14 = x[i,14]
        x15 = x[i,5]
        x16 = x[i,6]
        x17 = x[i,7]
        x18 = x[i,8]
        x19 = x[i,9]
        x20 = x[i,0]
        x21 = x[i,1]
        x22 = x[i,2]
        x23 = x[i,3]
        x24 = x[i,4]
        x25 = x[i,5]
        x26 = x[i,6]
        x27 = x[i,7]
        x28 = x[i,8]
        x29 = x[i,9]
        x30 = x[i,0]
        x31 = x[i,1]
        x32 = x[i,2]
        x33 = x[i,3]
        x34 = x[i,4]
        x35 = x[i,5]
        x36 = x[i,6]
        x37 = x[i,7]
        x38 = x[i,8]
        x39 = x[i,9]
        x40 = x[i,0]
        x41 = x[i,1]
        x42 = x[i,2]
        x43 = x[i,3]
        x44 = x[i,4]
        x45 = x[i,5]
        x46 = x[i,6]
        x47 = x[i,7]
        x48 = x[i,8]
        x49 = x[i,9]
        x50 = x[i,0]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50)
        point = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd

def rana(x):
    """
    x in [-5.12,5.12]
    """
    n,dim = x.shape
    y = np.zeros((n,1))
    for i in range(n):
        for j in range(dim-1):
            y[i,0] += x[i,j]*np.cos(np.sqrt(np.abs(x[i,j+1]+x[i,j]+1)))*np.sin(np.sqrt(np.abs(x[i,j+1]-x[i,j]+1)))+(1+x[i,j+1])*np.sin(np.sqrt(np.abs(x[i,j+1]+x[i,j]+1)))*np.cos(np.sqrt(np.abs(x[i,j+1]-x[i,j]+1)))

    return y ,{}

def rosenbrock(x):
	"""
	Evaluate objective and constraints for the Rosenbrock test case:	
	"""
	n,dim = x.shape

	#parameters:
	Opt =[]
	Opt_point_scalar = 1
	#construction of O vector
	for i in range(0, dim):
		Opt.append(Opt_point_scalar)
	
	#Construction of Z vector
	Z= np.zeros((n,dim))
	for i in range(0,dim):
		Z[:,i] = (x[:,i]-Opt[i]+1)
	
	#Sum
	sum1 = np.zeros((n,1))
	for i in range(0,dim-1):
		sum1[:,0] += 100*(((Z[:,i]**2)-Z[:,i+1])**2)+((Z[:,i]-1)**2)
		
	return sum1,{}	

def carre(x):
    """
    Fonction carre:
    Entrees x (n_evals,dimension)
        dim
    y = sum x(i)^2
         i
    """
    n, dim = x.shape
    y = np.zeros((n,1))
    y[:,0] = np.sum(x**2,1).T
    yd = np.zeros((n,dim))
    for i in range(dim):
        yd[:,i] = 2*x[:,i]
    return y,yd

def carre1(x):
    """
    Fonction carre:
    Entrees x (n_evals,dimension)
        dim
    y = sum x(i)^2
         i
    """
    n, dim = x.shape
    y = np.zeros((n,1))
    y[:,0] = np.sum(x[:,1:]**2,1).T+(x[:,0]**3)
    yd = np.zeros((n,dim))
    for i in range(dim):
        if i != 0:
            yd[:,i] = 2*x[:,i]
        else:
            yd[:,i] =3*x[:,i]**2
    return y,yd

def ackley(x):
    """
    Fonction Ackley:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape
    a = 20.
    b = 0.2
    c = 2*np.pi
    s1 = array2d(np.sum(x**2,1)).T
    s2 = array2d(np.sum(np.cos(c*x),1)).T
    return -a * np.exp(-b*np.sqrt(1./dim*s1))-np.exp(1./dim*s2)+a+np.exp(1),{}

def dixonPrice(x):
    """
    Fonction Dixon & Price:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim < 2:
        print "\n ****Dimension of inputs must be greter than one for Dixon & Price function**** \n"
        return
    s1 = np.zeros((n,1))
    y = np.zeros((n,1))
    for i in range(n):
        for j in range(1,dim):
            s1[i] += (j+1.) * (2. * x[i,j]**2- x[i,j-1])**2
        y[i] = s1[i] + (x[i,0] - 1.)**2

    return y

def griewank(x):
    """
    Fonction Griewank:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape
    fr = 4000.
    s = np.zeros((n,1))
    p = np.ones((n,1))
    s = array2d(np.sum(x**2,1)).T
    p = array2d(np.prod(np.cos(x/np.sqrt(array2d(range(1,dim+1)))),axis = 1)).T
    return s/fr - p + 1.,{}

def branin(x):
    """
    Fonction Brinin:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 2:
        print "dimension must be equal 2"
        raise
    return array2d((x[:,1]-5.1/(4.*(np.pi)**2)*x[:,0]**2+5./np.pi *x[:,0]-6)**2+10.*(1.-1./(8.*np.pi))*np.cos(x[:,0])+10).T,{}

def tird(x):
    """
    Fonction Tird:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape

    s1 = np.zeros((n,1))
    s2 = np.zeros((n,1))
    s1 = array2d(np.sum((x-1)**2,1)).T
    for i in range(1,dim):
        s2 = s2 + array2d(x[:,i-1]*x[:,i]).T

    return array2d(s1-s2),{}

def G07(x):
    """
    Fonction G07 with y[:,traints:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 10:
        print "dimension must be equal 10"
        raise

    y = np.zeros((n,9))
    y[:,0] = x[:,0]**2+x[:,1]**2+x[:,0]*x[:,1]-14*x[:,0]-16*x[:,1]+ \
        (x[:,2]-10)**2+4*(x[:,3]-5)**2+(x[:,4]-3)**2+2*(x[:,5]-1)**2+5* \
        x[:,6]**2+7*(x[:,7]-11)**2+2*(x[:,8]-10)**2+(x[:,9]-7)**2+45

    y[:,1] = 4*x[:,0]+5*x[:,1]-3*x[:,6]+9*x[:,7]-105
    y[:,2] = 10*x[:,0]-8*x[:,1]-17*x[:,6]+2*x[:,7]
    y[:,3] = -8*x[:,0]+2*x[:,1]+5*x[:,8]-2*x[:,9]-12
    y[:,4] = 3*(x[:,0]-2)**2+4*(x[:,1]-3)**2+2*x[:,2]**2-7*x[:,3]-120
    y[:,5] = 5*x[:,0]**2+8*x[:,1]+(x[:,2]-6)**2-2*x[:,3]-40
    y[:,7] = 0.5*(x[:,0]-8)**2+2*(x[:,1]-4)**2+3*x[:,4]**2-x[:,5]-30
    y[:,6] = x[:,0]**2+2*(x[:,1]-2)**2-2*x[:,0]*x[:,1]+14*x[:,4]-6*x[:,5]
    y[:,8] = -3*x[:,0]+6*x[:,1]+12*(x[:,8]-8)**2-7*x[:,9]

    return array2d(y[:,0]).T,{}

def G07MOD(x):
    """
    Modification fonction G07 with y[:,traints:
    Entrees x (n_evals,dimension)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 10:
        print "dimension must be equal 10"
        raise

    y = np.zeros((n,9))
    y[:,0] = x[:,0]**2+x[:,1]**2+x[:,0]*x[:,1]-14*x[:,0]-16*x[:,1]+ \
        (x[:,2]-10)**2+4*(x[:,3]-5)**2+(x[:,4]-3)**2+2*(x[:,5]-1)**2+5* \
        x[:,6]**2+7*(x[:,7]-11)**2+2*(x[:,8]-10)**2+(x[:,9]-7)**2+45

    y[:,1] = (4*x[:,0]+5*x[:,1]-3*x[:,6]+9*x[:,7]-105)/105.
    y[:,2] = (10*x[:,0]-8*x[:,1]-17*x[:,6]+2*x[:,7])/370.
    y[:,3] = (-8*x[:,0]+2*x[:,1]+5*x[:,8]-2*x[:,9]-12)/158.
    y[:,4] = (3*(x[:,0]-2)**2+4*(x[:,1]-3)**2+2*x[:,2]**2-7*x[:,3]-120)/1258.
    y[:,5] = (5*x[:,0]**2+8*x[:,1]+(x[:,2]-6)**2-2*x[:,3]-40)/816.
    y[:,6] = (0.5*(x[:,0]-8)**2+2*(x[:,1]-4)**2+3*x[:,4]**2-x[:,5]-30)/834.
    y[:,7] = (x[:,0]**2+2*(x[:,1]-2)**2-2*x[:,0]*x[:,1]+14*x[:,4]-6*x[:,5])/788.
    y[:,8] = (-3*x[:,0]+6*x[:,1]+12*(x[:,8]-8)**2-7*x[:,9])/4048.

    return y,{}

def G7(x):

    def partial_derivative(function, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)

    def func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9):
        return x0**2+x1**2+x0*x1-14*x0-16*x1+(x2-10)**2+4*(x3-5)**2+(x4-3)**2+2*(x5-1)**2+5*x6**2+7*(x7-11)**2+2*(x8-10)**2+(x9-7)**2+45
        
    n, dim = x.shape
    y = np.zeros((n,1))
    yd = np.zeros((n,dim))
    for i in range(n):
        x0 = x[i,0]
        x1 = x[i,1]
        x2 = x[i,2]
        x3 = x[i,3]
        x4 = x[i,4]
        x5 = x[i,5]
        x6 = x[i,6]
        x7 = x[i,7]
        x8 = x[i,8]
        x9 = x[i,9]
        y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9)
        point = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
        for j in range(dim):
            yd[i,j] = partial_derivative(func,var=j,point=point)

    return y, yd


def G07GEK(x):
    """
    Modification fonction G07 with y[:,traints:
    Entrees x (n_evals,dimension)
    """
    n, dim = x.shape
    if dim != 10:
        print "dimension must be equal 10"
        raise

    y = np.zeros((n,1))
    y[:,0] = x[:,0]**2+x[:,1]**2+x[:,0]*x[:,1]-14*x[:,0]-16*x[:,1]+ \
        (x[:,2]-10)**2+4*(x[:,3]-5)**2+(x[:,4]-3)**2+2*(x[:,5]-1)**2+5* \
        x[:,6]**2+7*(x[:,7]-11)**2+2*(x[:,8]-10)**2+(x[:,9]-7)**2+45
    yd = np.zeros((n,10))
    yd[:,0] = 2*x[:,0]+x[:,1]-14
    yd[:,1] = 2*x[:,1]+x[:,0]-16
    yd[:,2] = 2*(x[:,2]-10)
    yd[:,3] = 8*(x[:,3]-5)
    yd[:,4] = 2*(x[:,4]-3)
    yd[:,5] =  4*(x[:,5]-1)
    yd[:,6] =  10*x[:,6]
    yd[:,7] =  14*(x[:,7]-11)
    yd[:,8] =  4*(x[:,8]-10)
    yd[:,9] =  2*(x[:,9]-7)
    
    return y,yd


def mystery(x):
    x = array2d(x)
    n, dim = x.shape
    if dim != 2:
        print "dimension must be equal 2"
        raise

    y = np.zeros((n,2))
    y[:,0] = 2 + 0.01 * (x[:,1]-x[:,0]**2)**2 + (1-x[:,0])**2 + 2* \
        (2-x[:,1])**2 + 7 * np.sin(0.5 * x[:,0]) * np.sin(0.7*x[:,0]*x[:,1])

    y[:,1] = - np.sin(x[:,0]-x[:,1]-np.pi/8.)

    return y,{}

def function2(x):
    # x_i in [0,1]^2 for i = 1,2
    x = array2d(x)
    n, dim = x.shape
    if dim != 2:
        print "dimension must be equal 2"
        raise

    y = np.zeros((n,4))
    y[:,0] = - (x[:,0]-1)**2 - (x[:,1] - 0.5)**2
    y[:,1] = (x[:,0]-3)**2 + (x[:,1] + 2)**2*np.exp(-x[:,1]**7) - 12
    y[:,2] = 10 * x[:,0] + x[:,1] - 7
    y[:,3] = (x[:,0]-0.5)**2 + (x[:,1]-0.5)**2-0.2

    return y,{}

def newBranin(x):
    # xi in [-5,10]*[0,15]
    x = array2d(x)
    n, dim = x.shape
    if dim != 2:
        print "dimension must be equal 2"
        raise

    y = np.zeros((n,2))
    y[:,0] = - (x[:,0]-10)**2-(x[:,1]-15)**2
    y[:,1] = (x[:,1] - 5.1/(4*np.pi**2)*x[:,0]**2+5./np.pi*x[:,0]-6)**2 +10 * \
        (1-1./(8*np.pi)) * np.cos(x[:,0]) - 5

    return y,{}

def g2(x):
    """
    binf = [0]
    bsup = [10]
    for i in range(dim):
        binf.append(0)
        bsup.append(10)

    """
    x = array2d(x)
    n, dim = x.shape
    y = np.zeros((n,3))
    ii = np.zeros((1,dim))
    for i in range(dim):
        ii[0,i]= i+1

    y[:,0] = -np.abs((np.sum(np.cos(x)**4,1)-2.*np.prod(np.cos(x)**2,1))/np.sqrt(np.sum(ii * x**2,1)))
    for i in range(n):
        z = -np.prod(x[i,:])+0.75
        if z >= 0:
            y[i,1] = np.log(1+z)
        else:
            y[i,1] = -np.log(1-z)

    y[:,2] = (np.sum(x,1)-7.5*dim)/(2.5*dim)

    return y,{}

def g1(x):
    x = array2d(x)
    n, dim = x.shape
    y = np.zeros((n,10))

    y[:,0] = 5*np.sum(x[:,0:4],1)-5*np.sum(x[:,0:4]**2,1)-np.sum(x[:,4:],1)
    y[:,1] = 2*x[:,0]+2*x[:,1]+x[:,9]+x[:,10]-10
    y[:,2] = 2*x[:,0]+2*x[:,2]+x[:,9]+x[:,11]-10
    y[:,3] = 2*x[:,1]+2*x[:,2]+x[:,10]+x[:,11]-10
    y[:,4] = -8*x[:,0]+x[:,9]
    y[:,5] = -8*x[:,1]+x[:,10]
    y[:,6] = -8*x[:,2]+x[:,11]
    y[:,7] = -2*x[:,3]-x[:,4]+x[:,9]
    y[:,8] = -2*x[:,5]-x[:,6]+x[:,10]
    y[:,9] = -2*x[:,7]-x[:,8]+x[:,11]
    return y,{}

def WWF(X):
    """
    Wing Weight Function

    binf=[150]
    binf.append(220)
    binf.append(6)
    binf.append(-10)
    binf.append(16)
    binf.append(0.5)
    binf.append(0.08)
    binf.append(2.5)
    binf.append(1700)
    binf.append(0.025)

    bsup = [200]
    bsup.append(300)
    bsup.append(10)
    bsup.append(10)
    bsup.append(45)
    bsup.append(1)
    bsup.append(0.18)
    bsup.append(6)
    bsup.append(2500)
    bsup.append(0.08)
    """
    X = array2d(X)
    n, dim = X.shape
    if dim != 10:
        print "dimension must be equal 10"
        raise

#    if (X[:,0] < 150) or (X[:,0] > 200) or (X[:,1] < 220) or (X[:,1] > 300) or (X[:,2] < 6) or (X[:,2] > 10) or (X[:,3] < -10) or (X[:,3] > 10) or (X[:,4] < 16) or (X[:,4] > 45) or (X[:,5] < 0.5) or (X[:,5] > 1) or (X[:,6] < 0.08) or (X[:,6] > 0.18) or (X[:,7] < 2.5) or (X[:,7] > 6) or (X[:,8] < 1700) or (X[:,8] > 2500) or (X[:,9] < 0.025) or (X[:,9] > 0.08):
#        print "Bounds of variables are not respected"
#        raise

    return 0.036*X[:,0]**0.758*X[:,1]**0.0035*(X[:,2]/np.cos(np.radians(X[:,3]))**2)**0.6*X[:,4]**0.006*X[:,5]**0.04*(100*X[:,6]/np.cos(np.radians(X[:,3])))**(-0.3)*(X[:,7]*X[:,8])**0.49+X[:,0]*X[:,9],{}

def WB4(x):
    """
    Welded Beam
    Best solution : f = 1.728226
    x* = [0.20564426101885,3.47257874213172,9.03662391018928,0.20572963979791]

    binf=[0.125]
    binf.append(0.1)
    binf.append(0.1)
    binf.append(0.1)

    bsup = [10]
    bsup.append(10)
    bsup.append(10)
    bsup.append(10)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 4:
        print "dimension must be equal 4"
        raise

    #Data
    P = 6000.
    L = 14.
    E = 30*1e6
    G = 12*1e6
    tmax = 13600.
    smax = 30000.
    xmax = 10.
    dmax = 0.25
    M = P*(L+x[:,1]/2.)
    R = np.sqrt(0.25*(x[:,1]**2+(x[:,0]+x[:,2])**2))
    J = np.sqrt(2)*x[:,0]*x[:,1]*(x[:,1]**2/12.+0.25*(x[:,0]+x[:,2])**2)
    Pc = 4.013*E/(6*L**2)*x[:,2]*x[:,3]**3*(1-0.25*x[:,2]*np.sqrt(E/G)/L)
    t1 = P/(np.sqrt(2)*x[:,0]*x[:,1])
    t2 = M*R/J
    t = np.sqrt(t1**2+t1*t2*x[:,1]/R+t2**2)
    s = 6*P*L/(x[:,3]*x[:,2]**2)
    d = 4*P*L**3/(E*x[:,3]*x[:,2]**3)

    y = np.zeros((n,7))
    y[:,0] = 1.10471*x[:,0]**2*x[:,1]+0.04811*x[:,2]*x[:,3]*(14+x[:,1])
    y[:,1] = (t-tmax)/tmax
    y[:,2] = (s-smax)/smax
    y[:,3] = (x[:,0]-x[:,3])/xmax
    y[:,4] = (0.10471*x[:,0]**2+0.04811*x[:,2]*x[:,3]*(14+x[:,1])-5)/5.
    y[:,5] = (d-dmax)/dmax
    y[:,6] = (P-Pc)/P

    return y,{}

def PVD4(x):
    """
    Pressure Vessel Design
    binf = [0]
    binf.append(0)
    binf.append(0)
    binf.append(0)

    bsup = [1]
    bsup.append(1)
    bsup.append(50)
    bsup.append(240)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 4:
        print "dimension must be equal 4"
        raise

    y = np.zeros((n,4))
    y[:,0] = 0.6224*x[:,0]*x[:,2]*x[:,3]+1.7781*x[:,1]*x[:,2]**2+3.1661*x[:,0]**2*x[:,3]+19.84*x[:,0]**2*x[:,2]
    y[:,1] = -x[:,0]+0.0193*x[:,2]
    y[:,2] = -x[:,1]+0.00954*x[:,2]
    for i in range(n):
        z = -np.pi*x[i,2]**2*x[i,3]-4./3*np.pi*x[i,2]**3+1296000
        if z >= 0:
            y[i,3] = np.log(1 +z)
        else:
            y[i,3] = -np.log(1 - z)

    return y,{}

def SR7(x):
    """
    Spead reducer
    binf = [2.6]
    binf.append(0.7)
    binf.append(17.)
    binf.append(7.3)
    binf.append(7.3)
    binf.append(2.9)
    binf.append(5.)

    bsup = [3.6]
    bsup.append(0.8)
    bsup.append(28)
    bsup.append(8.3)
    bsup.append(8.3)
    bsup.append(3.9)
    bsup.append(5.5)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 7:
        print "dimension must be equal 7"
        raise

    y = np.zeros((n,12))
    A = 3.3333*x[:,2]**2+14.9334*x[:,2]-43.0934
    B = x[:,5]**2+x[:,6]**2
    C = x[:,5]**3+x[:,6]**3
    D = x[:,3]*x[:,5]**2+x[:,4]*x[:,6]**2
    A1 = ((745*x[:,3]/(x[:,1]*x[:,2]))**2+(16.91*1e6))**0.5
    B1 = 0.1*x[:,5]**3
    A2 = ((745*x[:,4]/(x[:,1]*x[:,2]))**2+(157.5*1e6))**0.5
    B2 = 0.1*x[:,6]**3
    y[:,0] = 0.7854*x[:,0]*x[:,1]**2*A-1.508*x[:,0]*B+7.477*C+0.7854*D
    y[:,1] = (27-x[:,0]*x[:,1]**2*x[:,2])/27.
    y[:,2] = (397.5-x[:,0]*x[:,1]**2*x[:,2]**2)/397.5
    y[:,3] = (1.93-(x[:,1]*x[:,5]**4*x[:,2])/x[:,3]**3)/1.93
    y[:,4] = (1.93-(x[:,1]*x[:,6]**4*x[:,2])/x[:,4]**3)/1.93
    y[:,5] = ((A1/B1)-1100)/1100.
    y[:,6] = ((A2/B2)-850)/850.
    y[:,7] = (x[:,1]*x[:,2]-40)/40.
    y[:,8] = (5-(x[:,0]/x[:,1]))/5.
    y[:,9] = ((x[:,0]/x[:,1])-12)/12.
    y[:,10] = (1.9+1.5*x[:,5]-x[:,3])/1.9
    y[:,11] = (1.9+1.1*x[:,6]-x[:,4])/1.9

    return y,{}

def GTCD(x):
    """
    Gas Transmission Design
    binf = [20]
    binf.append(1)
    binf.append(20)
    binf.append(0.1)

    bsup = [50]
    bsup.append(10)
    bsup.append(50)
    bsup.append(60)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 4:
        print "dimension must be equal 4"
        raise

    y = np.zeros((n,2))
    y[:,0] = (8.61*1e5)*x[:,0]**(0.5)*x[:,1]*x[:,2]**(-2/3)*x[:,3]**(-0.5)+(3.69*1e4)*x[:,2]+(7.72*1e8)*x[:,0]**-1*x[:,1]**0.219-(765.43*1e6)*x[:,0]**-1
    y[:,1] = x[:,3]*x[:,1]**-2+x[:,1]**-2-1

    return y,{}

def G3MOD(x):
    """
    G3MOD
    binf = [0]
    bsup = [1]
    for i in range(dim):
        binf.append(0)
        bsup.append(1)
    """
    x = array2d(x)
    n, dim = x.shape

    y = np.zeros((n,2))
    for i in range(n):
        z = np.sqrt(dim)**dim*np.prod(x[i,:])
        if z >= 0:
            y[i,0] = - np.log(1+z)
        else:
            y[i,0] = np.log(1-z)
    y[:,1] = np.sum(x**2,1)-1

    return y,{}

def G4(x):
    """
    G4
    binf = [78]
    binf.append(33)
    binf.append(27)
    binf.append(27)
    binf.append(27)

    bsup = [102]
    bsup.append(45)
    bsup.append(45)
    bsup.append(45)
    bsup.append(45)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 5:
        print "dimension must be equal 5"
        raise

    y = np.zeros((n,7))
    u = 85.334407+0.0056858*x[:,1]*x[:,4]+0.0006262*x[:,0]*x[:,3]-0.0022053*x[:,2]*x[:,4]
    v = 80.51249+0.0071317*x[:,1]*x[:,4]+0.0029955*x[:,0]*x[:,1]+0.0021813*x[:,2]**2
    w = 9.300961+0.0047026*x[:,2]*x[:,4]+0.0012547*x[:,0]*x[:,2]+0.0019085*x[:,2]*x[:,3]
    y[:,0] = 5.3578547*x[:,2]**2+0.8356891*x[:,0]*x[:,4]+37.293239*x[:,0]-40792.141
    y[:,1] = -u
    y[:,2] = u-92
    y[:,3] = -v+90
    y[:,4] = v-110
    y[:,5] = -w +20
    y[:,6] = w-25

    return y,{}

def G5MOD(x):
    """
    G5MOD
    binf = [0]
    binf.append(0)
    binf.append(-0.55)
    binf.append(-0.55)

    bsup = [1200]
    bsup.append(1200)
    bsup.append(0.55)
    bsup.append(0.55)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 4:
        print "dimension must be equal 4"
        raise

    y = np.zeros((n,6))

    y[:,0] = 3*x[:,0]+1e-6*x[:,0]**3+2*x[:,1]+(2*1e-6/3.)*x[:,1]**3
    y[:,1] = x[:,2]-x[:,3]-0.55
    y[:,2] = x[:,3]-x[:,2]-0.55
    y[:,3] = 1000*np.sin(-x[:,2]-0.25)+1000*np.sin(-x[:,3]-0.25)+894.8-x[:,0]
    y[:,4] = 1000*np.sin(x[:,2]-0.25)+1000*np.sin(x[:,2]-x[:,3]-0.25)+894.8-x[:,1]
    y[:,5] = 1000*np.sin(x[:,3]-0.25)+1000*np.sin(x[:,3]-x[:,2]-0.25)+1294.8

    return y,{}

def G9(x):
    """
    G9
    binf = [-10]
    bsup = [10]
    for i in range(dim):
        binf.append(-10)
        bsup.append(10)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 7:
        print "dimension must be equal 7"
        raise

    y = np.zeros((n,5))
    y[:,0] = (x[:,0]-10)**2+5*(x[:,1]-12)**2+x[:,2]**4+3*(x[:,3]-11)**2+10*x[:,4]**6+7*x[:,5]**2+x[:,6]**4-4*x[:,5]*x[:,6]-10*x[:,5]-8*x[:,6]
    y[:,1] = (2*x[:,0]**2+3*x[:,1]**4+x[:,2]+4*x[:,3]**2+5*x[:,4]-127)/127.
    y[:,2] = (7*x[:,0]+3*x[:,1]+10*x[:,2]**2+x[:,3]-x[:,4]-282)/282.
    y[:,3] = (23*x[:,0]+x[:,1]**2+6*x[:,5]**2-8*x[:,6]-196)/196.
    y[:,4] = 4*x[:,0]**2+x[:,1]**2-3*x[:,0]*x[:,1]+2*x[:,2]**2+5*x[:,5]-11*x[:,6]
    return y,{}

def G10(x):
    """
    G10
    binf = [100]
    binf.append(1000)
    binf.append(1000)
    binf.append(10)
    binf.append(10)
    binf.append(10)
    binf.append(10)
    binf.append(10)

    bsup = [1e4]
    bsup.append(1e4)
    bsup.append(1e4)
    bsup.append(1e3)
    bsup.append(1e3)
    bsup.append(1e3)
    bsup.append(1e3)
    bsup.append(1e3)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 8:
        print "dimension must be equal 8"
        raise

    y = np.zeros((n,7))
    y[:,0] = x[:,0]+x[:,1]+x[:,2]
    y[:,1] = -1 + 0.0025*(x[:,3]+x[:,5])
    y[:,2] = -1 + 0.0025*(-x[:,3]+x[:,4]+x[:,6])
    y[:,3] = -1 + 0.01*(-x[:,4]+x[:,7])
    for i in range(n):
        z = 100*x[i,0]-x[i,0]*x[i,5]+833.33252*x[i,3]-83333.333
        if z >= 0:
            y[i,4] = np.log(1+z)
        else:
            y[i,4] = - np.log(1-z)

        z = x[i,1]*x[i,3]-x[i,1]*x[i,6]-1250*x[i,3]+1250*x[i,4]
        if z >= 0:
            y[i,5] = np.log(1+z)
        else:
            y[i,5] = - np.log(1-z)

        z = x[i,2]*x[i,4]-x[i,2]*x[i,7]-2500*x[i,4]+1250000
        if z >= 0:
            y[i,6] = np.log(1+z)
        else:
            y[i,6] = - np.log(1-z)

    return y,{}

def Hesse(x):
    """
    Hesse
    binf = [0]
    binf.append(0)
    binf.append(1)
    binf.append(0)
    binf.append(1)
    binf.append(0)

    bsup = [5]
    bsup.append(4)
    bsup.append(5)
    bsup.append(6)
    bsup.append(5)
    bsup.append(10)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 6:
        print "dimension must be equal 6"
        raise

    y = np.zeros((n,7))
    y[:,0] = -25*(x[:,0]-2)**2-(x[:,1]-2)**2-(x[:,2]-1)**2-(x[:,3]-4)**2-(x[:,4]-1)**2-(x[:,5]-4)**2
    y[:,1] = (2-x[:,0]-x[:,1])/2.
    y[:,2] = (x[:,0]+x[:,1]-6)/6.
    y[:,3] = (-x[:,0]+x[:,1]-2)/2.
    y[:,4] = (x[:,0]-3*x[:,1]-2)/2.
    y[:,5] = (4-(x[:,2]-3)**2-x[:,3])/4.
    y[:,6] = (4-(x[:,4]-3)**2-x[:,5])/4.

    return y,{}
    
def G18(x):
    """
    binf =[-10]
    bsup = [10]
    for i in range(7):
        binf.append(-10)
        bsup.append(10)

    binf.append(0)
    bsup.append(20)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 9:
        print "dimension must be equal 9"
        raise

    y = np.zeros((n,14))

    y[:,0] = -0.5*(x[:,0]*x[:,3]-x[:,1]*x[:,2]+x[:,2]*x[:,8]-x[:,4]*x[:,8] \
        + x[:,4]*x[:,7]-x[:,5]*x[:,6])
    y[:,1] = x[:,2]**2 + x[:,3]**2-1
    y[:,2] = x[:,8]**2-1
    y[:,3] = x[:,4]**2+x[:,5]**2-1
    y[:,4] = x[:,0]**2+(x[:,1]-x[:,8])**2-1
    y[:,5] = (x[:,0]-x[:,4])**2+(x[:,1]-x[:,5])**2-1
    y[:,6] = (x[:,0]-x[:,6])**2+(x[:,1]-x[:,7])**2-1
    y[:,7] = (x[:,2]-x[:,4])**2+(x[:,3]-x[:,5])**2-1
    y[:,8] = (x[:,2]-x[:,6])**2+(x[:,3]-x[:,7])**2-1
    y[:,9] = x[:,6]**2+(x[:,7]-x[:,8])**2-1
    y[:,10] = x[:,1]*x[:,2]-x[:,0]*x[:,3]
    y[:,11] = -x[:,2]*x[:,8]
    y[:,12] = x[:,4]*x[:,8]
    y[:,13] = x[:,5]*x[:,6]-x[:,4]*x[:,7]
    return y,{}

def G19(x):
    """
    binf = [0]
    bsup = [10]
    for i in range(14):
        binf.append(0)
        bsup.append(10)
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 15:
        print "dimension must be equal 15"
        raise

    y = np.zeros((n,6))
    a = np.array([[-16,2,0,1,0],[0,-2,0,0.4,2],[-3.5,0,2,0,0],[0,-2,0,-4,-1], \
        [0,-9,-2,1,-2.8],[2,0,-4,0,0],[-1,-1,-1,-1,-1],[-1,-2,-3,-2,-1], \
        [1,2,3,4,5],[1,1,1,1,1]])
    b = np.array([[-40,-2,-0.25,-4,-4,-1,-40,-60,5,1]]).T
    c = np.array([[30,-20,-10,32,-10],[-20,39,-6,-31,32],[-10,-6,10,-6,-10], \
        [32,-31,-6,39,-20],[-10,32,-10,-20,30]])
    d = np.array([[4,8,10,6,2]]).T
    e = np.array([[-15,-27,-36,-18,-12]]).T

    for k in range(n):
        s1 = 0
        s2 = 0
        s3 = 0
        for i in range(5):
            for j in range(5):
                s1 += c[i,j]*x[k,10+i]*x[k,10+j]

        for j in range(5):
            s2 += d[j]*x[k,10+j]**3

        for i in range(10):
            s3 += -b[i]*x[k,i]

        y[k,0] = s1 + s2 + s3

        for j in range(5):
            s4 = 0
            s5 = 0
            for i in range(5):
                s4 += -2*c[i,j]*x[k,10+i]

            for i in range(10):
                s5 += a[i,j]*x[k,i]

            y[k,j+1] = s4 - e[j] + s5

    return y,{}
    
def ex8_2_1(x):
	"""
	binf = [160]
	bsup = [163.75280616399999]
	binf.append(160)
	bsup.append(163.75280616399999)
	binf.append(160)
	bsup.append(163.75280616399999)
	binf.append(160)
	bsup.append(163.75280616399999)
	binf.append(160)
	bsup.append(163.75280616399999)
	binf.append(160)
	bsup.append(178.46122759599999)
	binf.append(160)
	bsup.append(178.46122759599999)
	binf.append(160)
	bsup.append(178.46122759599999)
	binf.append(160)
	bsup.append(178.46122759599999)
	binf.append(160)
	bsup.append(178.46122759599999)
	binf.append(160)
	bsup.append(200)
	binf.append(160)
	bsup.append(200)
	binf.append(160)
	bsup.append(200)
	binf.append(160)
	bsup.append(200)
	binf.append(160)
	bsup.append(200)
	binf.append(160)
	bsup.append(221.53877240400001)
	binf.append(160)
	bsup.append(221.53877240400001)
	binf.append(160)
	bsup.append(221.53877240400001)
	binf.append(160)
	bsup.append(221.53877240400001)
	binf.append(160)
	bsup.append(221.53877240400001)
	binf.append(160)
	bsup.append(236.24719383600001)
	binf.append(160)
	bsup.append(236.24719383600001)
	binf.append(160)
	bsup.append(236.24719383600001)
	binf.append(160)
	bsup.append(236.24719383600001)
	binf.append(160)
	bsup.append(236.24719383600001)
	binf.append(60)
	bsup.append(63.752806163999999)
	binf.append(60)
	bsup.append(78.461227596000001)
	binf.append(60)
	bsup.append(100)
	binf.append(60)
	bsup.append(121.538772404)
	binf.append(60)
	bsup.append(136.24719383600001)
	binf.append(60)
	bsup.append(63.752806163999999)
	binf.append(60)
	bsup.append(78.461227596000001)
	binf.append(60)
	bsup.append(100)
	binf.append(60)
	bsup.append(121.538772404)
	binf.append(60)
	bsup.append(136.24719383600001)
	binf.append(60)
	bsup.append(63.752806163999999)
	binf.append(60)
	bsup.append(78.461227596000001)
	binf.append(60)
	bsup.append(100)
	binf.append(60)
	bsup.append(121.538772404)
	binf.append(60)
	bsup.append(136.24719383600001)
	binf.append(60)
	bsup.append(63.752806163999999)
	binf.append(60)
	bsup.append(78.461227596000001)
	binf.append(60)
	bsup.append(100)
	binf.append(60)
	bsup.append(121.538772404)
	binf.append(60)
	bsup.append(136.24719383600001)
	binf.append(60)
	bsup.append(63.752806163999999)
	binf.append(60)
	bsup.append(78.461227596000001)
	binf.append(60)
	bsup.append(100)
	binf.append(60)
	bsup.append(121.538772404)
	binf.append(60)
	bsup.append(136.24719383600001)
	binf.append(4.8283137373022997)
	bsup.append(7.0255383146385197)
	binf.append(4.4228486291941396)
	bsup.append(6.6200732065303596)
	binf.append(6.2146080984221896)
	bsup.append(8.4118326757584096)
	binf.append(6.2146080984221896)
	bsup.append(8.4118326757584096)
	binf.append(6.2146080984221896)
	bsup.append(8.4118326757584096)
 
     solution
     x =   array([[163.7528061640,163.7528061640,163.7528061640,163.7528061640,163.7528061640, \
        178.4612275960,178.4612275960,178.4612275960,178.4612275960,178.4612275960,200,200.,200., \
        200.,200.,221.5387724040,221.5387724040,221.5387724040,221.5387724040,221.5387724040, \
        236.2471938360,236.2471938360,236.2471938360,236.2471938360,236.2471938360,63.7528061640, \
        78.4612275960,100.,121.5387724040,122.6545295120,63.7528061640,78.4612275960,100., \
        113.4617331710,113.4618966620,  63.7528061640,78.4612275960,100.,100.0000021750, \
        100.0000043630,63.7528061640,78.4612275960,86.5382683028,86.5382929070,86.5388135233, \
        63.7528061640,77.3455196976,77.3455082053,77.3460345151,77.3460345151,6.8023947633, \
        6.1092475828,7.4955419439,7.9010070520,8.1886891244]])
        1e-05 sont acceptes
	"""
	x = array2d(x)
	n, dim = x.shape
	if dim != 55:
		print "dimension must be equal 55"
		raise
		
	y = np.zeros((n,32))
	y[:,0] = 3 * np.exp(0.59999999999999998 * x[:,52]) + 3 * np.exp(0.59999999999999998 * x[:,53]) + \
	   3 * np.exp(0.59999999999999998 * x[:,54]) - 1.5471103391371599e-06 * x[:,0] - \
	   0.00021904031699053399 * x[:,1] - 0.0026481311826779398 * x[:,2] - 0.00021904031699053399 * x[:,3] \
	   - 1.5471103391371599e-06 * x[:,4] - 0.00021904031699053299 * x[:,5] - 0.0310117896917886 * x[:,6] \
	   - 0.374923157717238 * x[:,7] - 0.0310117896917886 * x[:,8] - 0.00021904031699053201 * x[:,9] - \
	   0.0026481311826779298 * x[:,10] - 0.374923157717237 * x[:,11] - 4.5327075795914 * x[:,12] - 0.374923157717237 * \
	   x[:,13] - 0.0026481311826779099 * x[:,14] - 0.00021904031699053201 * x[:,15] - 0.031011789691788399 * x[:,16] \
	   - 0.374923157717236 * x[:,17] - 0.031011789691788399 * x[:,18] - 0.00021904031699053101 * x[:,19] - \
	   1.5471103391371301e-06 * x[:,20] - 0.000219040316990529 * x[:,21] - 0.0026481311826778899 * x[:,22] - \
	   0.000219040316990529 * x[:,23] - 1.5471103391371199e-06 * x[:,24] - 1.9690495225382001e-06 * x[:,25] - \
	   0.00027877858526067898 * x[:,26] - 0.0033703487779537401 * x[:,27] - 0.00027877858526067898 * x[:,28] - \
	   1.9690495225382001e-06 * x[:,29] - 0.00027877858526067898 * x[:,30] - 0.0394695505168218 * x[:,31] - \
	   0.47717492800375799 * x[:,32] - 0.0394695505168218 * x[:,33] - 0.00027877858526067698 * x[:,34] - \
	   0.0033703487779537201 * x[:,35] - 0.47717492800375599 * x[:,36] - 5.7689005558436 * x[:,37] - \
	   0.47717492800375599 * x[:,38] - 0.0033703487779537101 * x[:,39] - 0.00027877858526067698 * x[:,40] - \
	   0.039469550516821598 * x[:,41] - 0.47717492800375499 * x[:,42] - 0.039469550516821598 * x[:,43] - \
	   0.000278778585260676 * x[:,44] - 1.9690495225381598e-06 * x[:,45] - 0.000278778585260674 * x[:,46] - \
	   0.0033703487779536698 * x[:,47] - 0.000278778585260674 * x[:,48] - 1.9690495225381598e-06 * x[:,49]
	  
	y[:,1]=x[:,51] -x[:,54] + 1.09861228866811
	y[:,2]=x[:,51] -x[:,53] + 1.7917594692280501
	y[:,3]=x[:,51] -x[:,52] + 1.3862943611198899
	y[:,4]=x[:,50] -x[:,54] + 1.3862943611198899
	y[:,5]=x[:,50] -x[:,53] + 1.09861228866811
	y[:,6]=x[:,50] -x[:,52] + 0.69314718055994495
	y[:,7]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,24] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,49] - 8
	y[:,8]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,23] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,48] - 8
	y[:,9]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,0] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,25] - 8
	y[:,10]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,22] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,47] - 8
	y[:,11]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,21] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,46] - 8
	y[:,12]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,1] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,26] - 8
	y[:,13]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,20] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,45] - 8
	y[:,14]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,19] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,44] - 8
	y[:,15]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,18] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,43] - 8
	y[:,16]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,2] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,27] - 8
	y[:,17]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,17] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,42] - 8
	y[:,18]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,16] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,41] - 8
	y[:,19]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,15] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,40] - 8
	y[:,20]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,14] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,39] - 8
	y[:,21]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,3] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,28] - 8
	y[:,22]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,13] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,38] - 8
	y[:,23]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,12] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,37] - 8
	y[:,24]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,11] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,36] - 8
	y[:,25]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,4] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,29] - 8
	y[:,26]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,10] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,35] - 8
	y[:,27]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,9] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,34] - 8
	y[:,28]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,8] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,33] - 8
	y[:,29]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,5] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,30] - 8
	y[:,30]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,7] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,32] - 8
	y[:,31]=np.exp((- x[:,50]) + 2.99573227355399) * x[:,6] + np.exp((- x[:,51]) + 2.7725887222397798) * x[:,31] - 8
	
	return y, {}
  
def ex2_1_10(x):
    """
    binf = [0]
    bsup = [70]
    for i in range(19):
        binf.append(0)
        bsup.append(70)
        
    solution
    x = np.zeros((1,20))
    x[0,3]=62.6086956522
    x[0,15] = 4.3478260870
    tolerance 1e-10
        
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 20:
        print "dimension must be equal 20"
        raise
        
    y = np.zeros((n,11))
    y[:,0] = (-31.5) * np.power((x[:,0]) + 19, 2) - 7.5 * np.power((x[:,1]) + 27, 2) - 22 * np.power((x[:,2]) + 23, 2) - \
        45.5 * np.power((x[:,3]) + 53, 2) - 22.5 * np.power((x[:,4]) + 42, 2) - 25 * np.power((x[:,5]) - 26, 2) -  \
        44.5 * np.power((x[:,6]) + 33, 2) - 29 * np.power((x[:,7]) + 23, 2) - 43 * np.power((x[:,8]) - 41, 2) - \
        41 * np.power((x[:,9]) - 19, 2) + 21 * np.power((x[:,10]) + 52, 2) + 49 * np.power((x[:,11]) + 3, 2) + \
        24 * np.power((x[:,12]) - 81, 2) + 45.5 * np.power((x[:,13]) - 30, 2) + 5.5 * np.power((x[:,14]) + 85, 2) + \
        31.5 * np.power((x[:,15]) - 68, 2) + 30.5 * np.power((x[:,16]) - 27, 2) + 30.5 * np.power((x[:,17]) + 81, 2) + \
        19 * np.power((x[:,18]) - 97, 2) + 13 * np.power((x[:,19]) + 73, 2)

    y[:,1]=x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4] + x[:,5] + x[:,6] + x[:,7] + x[:,8] + x[:,9] + x[:,10] + x[:,11] + \
        x[:,12] + x[:,13] + x[:,14] + x[:,15] + x[:,16] + x[:,17] + x[:,18] + x[:,19] - 200
    y[:,2]=8 * x[:,0] + 5 * x[:,1] + 2 * x[:,2] + 5 * x[:,3] + 3 * x[:,4] + 8 * x[:,5] + x[:,6] + 3 * x[:,7] + 3 * x[:,8] + \
        5 * x[:,9] + 4 * x[:,10] + 5 * x[:,11] + 5 * x[:,12] + 6 * x[:,13] + x[:,14] + 7 * x[:,15] + x[:,16] + 2 * x[:,17] + \
        2 * x[:,18] + 4 * x[:,19] - 400
    y[:,3]=x[:,0] + 2 * x[:,1] + x[:,2] + 7 * x[:,3] + 8 * x[:,4] + 7 * x[:,5] + 6 * x[:,6] + 5 * x[:,7] + 8 * x[:,8] + \
        7 * x[:,9] + 2 * x[:,10] + 3 * x[:,11] + 5 * x[:,12] + 5 * x[:,13] + 4 * x[:,14] + 5 * x[:,15] + 4 * x[:,16] + \
        2 * x[:,17] + 2 * x[:,18] + 8 * x[:,19] - 460
    y[:,4]=3 * x[:,0] + 6 * x[:,1] + 6 * x[:,2] + 3 * x[:,3] + x[:,4] + 6 * x[:,5] + x[:,6] + 6 * x[:,7] + 7 * x[:,8] + \
        x[:,9] + 4 * x[:,10] + 3 * x[:,11] + x[:,12] + 4 * x[:,13] + 3 * x[:,14] + 6 * x[:,15] + 4 * x[:,16] + 6 * x[:,17] + \
        5 * x[:,18] + 4 * x[:,19] - 400
    y[:,5]=5 * x[:,0] + 5 * x[:,1] + 2 * x[:,2] + x[:,3] + 3 * x[:,4] + 5 * x[:,5] + 5 * x[:,6] + 7 * x[:,7] + 4 * x[:,8] + \
        3 * x[:,9] + 4 * x[:,10] + x[:,11] + 7 * x[:,12] + 3 * x[:,13] + 8 * x[:,14] + 3 * x[:,15] + x[:,16] + 6 * x[:,17] + \
        2 * x[:,18] + 8 * x[:,19] - 415
    y[:,6]=6 * x[:,0] + 6 * x[:,1] + 6 * x[:,2] + 4 * x[:,3] + 5 * x[:,4] + 2 * x[:,5] + 2 * x[:,6] + 4 * x[:,7] + 3 * x[:,8] + \
        2 * x[:,9] + 7 * x[:,10] + 5 * x[:,11] + 3 * x[:,12] + 6 * x[:,13] + 7 * x[:,14] + 5 * x[:,15] + 8 * x[:,16] + 4 * x[:,17] + \
        6 * x[:,18] + 3 * x[:,19] - 470
    y[:,7]=3 * x[:,0] + 2 * x[:,1] + 6 * x[:,2] + 3 * x[:,3] + 2 * x[:,4] + x[:,5] + 6 * x[:,6] + x[:,7] + 7 * x[:,8] + 3 * x[:,9] + \
        7 * x[:,10] + 7 * x[:,11] + 8 * x[:,12] + 2 * x[:,13] + 3 * x[:,14] + 4 * x[:,15] + 5 * x[:,16] + 8 * x[:,17] + x[:,18] + \
        2 * x[:,19] - 405
    y[:,8]=x[:,0] + 5 * x[:,1] + 2 * x[:,2] + 4 * x[:,3] + 7 * x[:,4] + 3 * x[:,5] + x[:,6] + 5 * x[:,7] + 7 * x[:,8] + 6 * x[:,9] + \
        x[:,10] + 7 * x[:,11] + 2 * x[:,12] + 4 * x[:,13] + 7 * x[:,14] + 5 * x[:,15] + 3 * x[:,16] + 4 * x[:,17] + x[:,18] + \
        2 * x[:,19] - 385
    y[:,9]=5 * x[:,0] + 4 * x[:,1] + 5 * x[:,2] + 4 * x[:,3] + x[:,4] + 4 * x[:,5] + 4 * x[:,6] + 2 * x[:,7] + 5 * x[:,8] + 2 * x[:,9] + \
        3 * x[:,10] + 6 * x[:,11] + x[:,12] + 7 * x[:,13] + 7 * x[:,14] + 5 * x[:,15] + 8 * x[:,16] + 7 * x[:,17] + 2 * x[:,18] + x[:,19] - 415
    y[:,10]=3 * x[:,0] + 5 * x[:,1] + 5 * x[:,2] + 6 * x[:,3] + 4 * x[:,4] + 4 * x[:,5] + 5 * x[:,6] + 6 * x[:,7] + 4 * x[:,8] + 4 * x[:,9] + \
        8 * x[:,10] + 4 * x[:,11] + 2 * x[:,12] + x[:,13] + x[:,14] + x[:,15] + 2 * x[:,16] + x[:,17] + 7 * x[:,18] + 3 * x[:,19] - 380
        
    return y,{}

def ex2_1_7(x):
    """
    binf = [0]
    bsup = [20]
    for i in range(19):
        binf.append(0)
        bsup.append(20)
        
    solution
    x = np.zeros((1,20))
    x[0,2]=1.0428999241
    x[0,10] = 1.7467437901
    x[0,12] = 0.4314708838
    x[0,15] = 4.4330502738
    x[0,17] = 15.8589317580
    x[0,19] = 16.4869033700
    tolerance 1e-10
        
    """
    x = array2d(x)
    n, dim = x.shape
    if dim != 20:
        print "dimension must be equal 20"
        raise
        
    y = np.zeros((n,11))
    y[:,0]=-np.power((x[:,1]) - 2, 2) - 0.5 * np.power((x[:,0]) - 2, 2) - 1.5 * np.power((x[:,2]) - 2, 2) - 2 * np.power((x[:,3]) - 2, 2) - \
        2.5 * np.power((x[:,4]) - 2, 2) - 3 * np.power((x[:,5]) - 2, 2) - 3.5 * np.power((x[:,6]) - 2, 2) - 4 * np.power((x[:,7]) - 2, 2) - \
        4.5 * np.power((x[:,8]) - 2, 2) - 5 * np.power((x[:,9]) - 2, 2) - 5.5 * np.power((x[:,10]) - 2, 2) - 6 * np.power((x[:,11]) - 2, 2) - \
        6.5 * np.power((x[:,12]) - 2, 2) - 7 * np.power((x[:,13]) - 2, 2) - 7.5 * np.power((x[:,14]) - 2, 2) - 8 * np.power((x[:,15]) - 2, 2) - \
        8.5 * np.power((x[:,16]) - 2, 2) - 9 * np.power((x[:,17]) - 2, 2) - 9.5 * np.power((x[:,18]) - 2, 2) - 10 * np.power((x[:,19]) - 2, 2)
    y[:,1]=x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4] + x[:,5] + x[:,6] + x[:,7] + x[:,8] + x[:,9] + x[:,10] + x[:,11] + x[:,12] + x[:,13] + \
        x[:,14] + x[:,15] + x[:,16] + x[:,17] + x[:,18] + x[:,19] - 40
    y[:,2]=-x[:,0] - x[:,1] - 9 * x[:,2] + 3 * x[:,3] + 5 * x[:,4] + x[:,7] + 7 * x[:,8] - 7 * x[:,9] - 4 * x[:,10] - 6 * x[:,11] - 3 * x[:,12] + \
        7 * x[:,13] - 5 * x[:,15] + x[:,16] + x[:,17] + 2 * x[:,19] - 9
    y[:,3]=2 * x[:,0] - x[:,1] - x[:,2] - 9 * x[:,3] + 3 * x[:,4] + 5 * x[:,5] + x[:,8] + 7 * x[:,9] - 7 * x[:,10] - 4 * x[:,11] - 6 * x[:,12] - \
        3 * x[:,13] + 7 * x[:,14] - 5 * x[:,16] + x[:,17] + x[:,18] 
    y[:,4]=2 * x[:,1] - x[:,2] - x[:,3] - 9 * x[:,4] + 3 * x[:,5] + 5 * x[:,6] + x[:,9] + 7 * x[:,10] - 7 * x[:,11] - 4 * x[:,12] - 6 * x[:,13] - \
        3 * x[:,14] + 7 * x[:,15] - 5 * x[:,17] + x[:,18] + x[:,19] +1
    y[:,5]=x[:,0] + 2 * x[:,2] - x[:,3] - x[:,4] - 9 * x[:,5] + 3 * x[:,6] + 5 * x[:,7] + x[:,10] + 7 * x[:,11] - 7 * x[:,12] - 4 * x[:,13] - 6 * x[:,14] - \
        3 * x[:,15] + 7 * x[:,16] - 5 * x[:,18] + x[:,19] - 4
    y[:,6]=x[:,0] + x[:,1] + 2 * x[:,3] - x[:,4] - x[:,5] - 9 * x[:,6] + 3 * x[:,7] + 5 * x[:,8] + x[:,11] + 7 * x[:,12] - 7 * x[:,13] - 4 * x[:,14] - \
        6 * x[:,15] - 3 * x[:,16] + 7 * x[:,17] - 5 * x[:,19] - 5
    y[:,7]=(-5) * x[:,0] + x[:,1] + x[:,2] + 2 * x[:,4] - x[:,5] - x[:,6] - 9 * x[:,7] + 3 * x[:,8] + 5 * x[:,9] + x[:,12] + 7 * x[:,13] - 7 * x[:,14] - \
        4 * x[:,15] - 6 * x[:,16] - 3 * x[:,17] + 7 * x[:,18] +3
    y[:,8]=(-5) * x[:,1] + x[:,2] + x[:,3] + 2 * x[:,5] - x[:,6] - x[:,7] - 9 * x[:,8] + 3 * x[:,9] + 5 * x[:,10] + x[:,13] + 7 * x[:,14] - 7 * x[:,15] - \
        4 * x[:,16] - 6 * x[:,17] - 3 * x[:,18] + 7 * x[:,19] +1
    y[:,9]=7 * x[:,0] - 5 * x[:,2] + x[:,3] + x[:,4] + 2 * x[:,6] - x[:,7] - x[:,8] - 9 * x[:,9] + 3 * x[:,10] + 5 * x[:,11] + x[:,14] + 7 * x[:,15] - \
        7 * x[:,16] - 4 * x[:,17] - 6 * x[:,18] - 3 * x[:,19] - 2
    y[:,10]=(-3) * x[:,0] + 7 * x[:,1] - 5 * x[:,3] + x[:,4] + x[:,5] + 2 * x[:,7] - x[:,8] - x[:,9] - 9 * x[:,10] + 3 * x[:,11] + 5 * x[:,12] + x[:,15] + \
        7 * x[:,16] - 7 * x[:,17] - 4 * x[:,18] - 6 * x[:,19] +5
        
    return y,{}


    def sobol_fun(x):
        
        n,dim = x.shape
        y = np.ones((n,1))
        for i in range(dim):
            y = y * (np.abs(4*x[i,:]-2)+i)/(1+i)
            
        return y,{}
