# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 15:24:43 2015

@author: MohamedAmine
"""

from __future__ import division
import numpy as np

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