# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:20:11 2018

@author: m.meliani
"""

import numpy as np
import matplotlib.pyplot as plt


def cheap(Xc):
    return 0.5*((Xc*6-2)**2)*np.sin((Xc*6-2)*2)+(Xc-0.5)*10. - 5

def expensive(Xe):
    return ((Xe*6-2)**2)*np.sin((Xe*6-2)*2)

Xe = np.array([[0],[0.5],[1]])
Xc = np.vstack((np.array([[0.2],[0.5],[0.8]]),Xe))

n_doe = Xe.size
n_cheap = Xc.size

ye = expensive(Xe)
yc = cheap(Xc)

Xr = np.linspace(0,1, 100)
Yr = expensive (Xr)
from smt.extensions import MFK

sm = MFK(theta0=np.array(Xe.shape[1]*[1.]), print_global = False)


sm.set_training_values(Xc, yc, name =0) #low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(Xe, ye) #high-fidelity dataset without name
sm.train()
x = np.linspace(0, 1, 101, endpoint = True).reshape(-1,1)

y = sm.predict_values(x)
MSE = sm.predict_variances(x)

plt.figure()

plt.plot(Xr, expensive(Xr), label ='reference')
plt.plot(x, y, label ='mean_gp')
plt.plot(Xe, ye, 'ko', label ='expensive doe')
plt.plot(Xc, yc, 'g*', label ='cheap doe')

plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(MSE)),np.ravel(y+3*np.sqrt(MSE)), facecolor ="lightgrey", edgecolor="g" ,label ='tolerance +/- 3*sigma')
plt.legend(loc=0)
plt.ylim(-10,17)
plt.xlim(-0.1,1.1)
plt.show()