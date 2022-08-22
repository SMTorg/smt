#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:48:01 2021

@author: psaves
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from smt.surrogate_models import KRG
from smt.applications.mixed_integer import (
    MixedIntegerSurrogateModel,
    ENUM,
    ORD,
    FLOAT,
    HOMO_GAUSSIAN,
    GOWER_MAT,
    HOMO_HYP,
)


def f1(x1):
    return x1**2


def f2(x2, x3):
    return -(x2**2) + 0.3 * x3


def f(X):
    y = []
    for x in X:
        if x[0] == 0:
            y.append(f1(x[1]))
        elif x[0] == 1:
            y.append(f2(x[1], x[2]))
    return np.array(y)


print(f(np.atleast_2d([0, 2])))
print(f(np.atleast_2d([1, 2, 1])))

xdoe1 = np.zeros((41, 2))
xdoe1[:, 1] = np.linspace(-5, 5, 41)
ydoe1 = f(xdoe1)

xdoe2 = np.zeros((121, 3))
xdoe2[:, 0] = np.ones(121)

x_cont = np.linspace(-5, 5, 11)
u = []
v = []
for (xi, yi) in itertools.product(x_cont, x_cont):
    u.append(xi)
    v.append(yi)
x_cont = np.concatenate(
    (np.asarray(v).reshape(-1, 1), np.asarray(u).reshape(-1, 1)), axis=1
)
xdoe2[:, 1:3] = x_cont
ydoe2 = f(xdoe2)


xdoe1 = np.zeros((41, 4))
xdoe1[:, 1] = np.linspace(-5, 5, 41)
xdoe2 = np.zeros((121, 4))
xdoe2[:, 0] = np.ones(121)

x_cont = np.linspace(-5, 5, 11)
u = []
v = []
for (xi, yi) in itertools.product(x_cont, x_cont):
    u.append(xi)
    v.append(yi)
x_cont = np.concatenate(
    (np.asarray(v).reshape(-1, 1), np.asarray(u).reshape(-1, 1)), axis=1
)
xdoe2[:, 2:4] = x_cont

Xt = np.concatenate((xdoe1, xdoe2), axis=0)
Yt = np.concatenate((ydoe1, ydoe2), axis=0)
xlimits = [["Blue", "Red"], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
xtypes = [(ENUM, 2), FLOAT, FLOAT, FLOAT]

# Surrogate
sm = MixedIntegerSurrogateModel(
    categorical_kernel=HOMO_GAUSSIAN,
    xtypes=xtypes,
    xlimits=xlimits,
    surrogate=KRG(theta0=[1e-2], n_start=30, corr="abs_exp"),
)
sm.set_training_values(Xt, Yt)
sm.train()
print(sm._surrogate.optimal_theta)
