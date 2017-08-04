from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

from smt.problems import Sphere
from smt.problems import NdimCantileverBeam, NdimRobotArm
from smt.sampling import LHS
from smt.methods import RMTC, RMTB, RBF
from smt.utils import compute_rms_error

ndim = 3
prob = Sphere(ndim=ndim)

prob = NdimCantileverBeam(ndim=ndim)
prob = NdimRobotArm(ndim=ndim)

sampling = LHS(xlimits=prob.xlimits)

ndim = prob.options['ndim']

sm = RMTC(xlimits=prob.xlimits)#, min_energy=False, nln_max_iter=0)
sm = RMTB(xlimits=prob.xlimits, min_energy=True, nln_max_iter=20)
sm = RBF(poly_degree=0)

nt = 1000
ne = int(nt / 2)

np.random.seed(0)
xt = sampling(nt)
yt = prob(xt)
dyt = {}
for kx in range(ndim):
    dyt[kx] = prob(xt, kx)

np.random.seed(1)
xe = sampling(ne)
ye = prob(xe)

sm.set_training_values( xt, yt)
if 0:
    for kx in range(ndim):
        sm.set_training_values( xt, dyt[kx], kx)
sm.train()

print(compute_rms_error(sm))
print(compute_rms_error(sm, xe, ye))

nplot = 50
a = 1.0

nr = 2
nc = 2

xe = np.zeros((nplot, ndim))
for ix in range(ndim):
    for kx in range(ndim):
        xe[:, kx] = 0.2 * prob.xlimits[kx, 0] + 0.8 * prob.xlimits[kx, 1]

    xe[:, ix] = np.linspace(a*prob.xlimits[ix, 0], a*prob.xlimits[ix, 1], nplot)
    ye = prob(xe)
    ye2 = sm.predict_values(xe)
    plt.subplot(nr, nc, ix + 1)
    plt.plot(xe[:, ix], ye, '-or')
    plt.plot(xe[:, ix], ye2, '-ob')

_ = sm.predict_derivatives(xe, 0)

plt.show()
