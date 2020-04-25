# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:06:25 2020

@author: rouxemi
"""

import numpy as np
import six
from smt.applications import EGO_para
from smt.sampling_methods import FullFactorial

import sklearn
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

def function_test_1d(x):
    # function xsinx
    import numpy as np

    x = np.reshape(x, (-1,))
    y = np.zeros(x.shape)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    return y.reshape((-1, 1))

n_iter = 2
n_par = 3
xlimits = np.array([[0.0, 25.0]])
xdoe = np.atleast_2d([0, 7, 25]).T
n_doe = xdoe.size

criterion = "EI"  #'EI' or 'SBO' or 'UCB'

ego = EGO_para(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits, n_par=n_par)

x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe, krig = ego.optimize(
    fun=function_test_1d
)
print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(x_opt), float(y_opt)))

x_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
y_plot = function_test_1d(x_plot)

fig = plt.figure(figsize=[10, 10])
for i in range(n_iter):
    k = n_doe + i*(n_par)
    x_data_k = x_data[0:k]
    y_data_k = y_data[0:k]
    for p in range(n_par):
        x_data_k = x_data[0:k+p]
        y_data_k = y_data[0:k+p]
        y_data_k[k+p-1] = ego.gpr.predict_values(x_data_k[k+p-1])
        ego.gpr.set_training_values(x_data_k, y_data_k)
        ego.gpr.train()
    
        y_gp_plot = ego.gpr.predict_values(x_plot)
        y_gp_plot_var = ego.gpr.predict_variances(x_plot)
        y_ei_plot = -ego.EI(x_plot, y_data_k)
    
        ax = fig.add_subplot(n_iter, n_par, i*(n_par) + p + 1)
        ax1 = ax.twinx()
        ei, = ax1.plot(x_plot, y_ei_plot, color="red")
    
        true_fun, = ax.plot(x_plot, y_plot)
        data, = ax.plot(
            x_data_k, y_data_k, linestyle="", marker="o", color="orange"
        )
        if i < n_iter - 1:
            opt, = ax.plot(
                x_data[k], y_data[k], linestyle="", marker="*", color="r"
            )
        gp, = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
        sig_plus = y_gp_plot + 3 * y_gp_plot_var
        sig_moins = y_gp_plot - 3 * y_gp_plot_var
        un_gp = ax.fill_between(
            x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
        )
        lines = [true_fun, data, gp, un_gp, opt, ei]
        fig.suptitle("EGO optimization of $f(x) = x \sin{x}$")
        fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
        ax.set_title("iteration {}.{}".format(i + 1,p + 1))
        fig.legend(
            lines,
            [
                "f(x)=xsin(x)",
                "Given data points",
                "Kriging prediction",
                "Kriging 99% confidence interval",
                "Next point to evaluate",
                "Expected improvment function",
            ],
        )
plt.show()