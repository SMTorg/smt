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

Xe = np.array([[0],[0.4],[1]])
Xc = np.vstack((np.array([[0.1],[0.2],[0.3],[0.5],[0.6],[0.7],[0.8],[0.9]]),Xe))

ye = expensive(Xe)
yc = cheap(Xc)

Xr = np.linspace(0,1, 100)
from smt.extensions import MFK
from sklearn.gaussian_process.correlation_models import squared_exponential

sm = MFK(theta0=np.array(Xe.shape[1]*[1.]))


sm.set_training_values(Xc, yc, name =0) #low-fidelity dataset
sm.set_training_values(Xe, ye, name =1) #high-fidelity dataset
sm.train()
x = np.linspace(0, 1, 101, endpoint = True).reshape(-1,1)
y = sm.predict_values(x)
MSE = sm.predict_variances(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(np.ravel(x), np.ravel(y-10*np.sqrt(MSE)),np.ravel(y+10*np.sqrt(MSE)), facecolor ="grey", edgecolor="g" ,label ='tolerance +/- 3*sigma')
ax.scatter(Xe, ye, label ='expensive')
ax.scatter(Xc, yc, label ='cheap')
ax.plot(x, y, label ='surrogate')
ax.plot(Xr, expensive(Xr), label ='reference')

ax.legend(fontsize ='x-large')

plt.show()