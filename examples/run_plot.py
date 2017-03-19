from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

from smt.problems import CantileverBeam, Carre, ReducedProblem, RobotArm, Rosenbrock
from smt.problems import TensorProduct, TorsionVibration, WaterFlow, WeldedBeam, WingWeight
from smt.problems import NdimCantileverBeam, NdimRobotArm, NdimRosenbrock, NdimStepFunction
from smt.sampling import LHS, Random, FullFactorial, Clustered
from smt import LS, PA2, KPLS, IDW, RBF, RMTS, RMTB


ndim = 4
# prob = Carre(ndim=ndim)
# prob = TensorProduct(ndim=ndim, func='cos', width=1.)
# prob = WeldedBeam(ndim=3)
# prob = CantileverBeam(ndim=3*ndim)
# prob = Rosenbrock(ndim=ndim)
# prob = RobotArm(ndim=2*ndim)

prob = NdimCantileverBeam(ndim=ndim)
# prob = NdimRobotArm(ndim=ndim)
# prob = NdimRosenbrock(ndim=ndim)
# prob = NdimStepFunction(ndim=ndim)

sampling = LHS(xlimits=prob.xlimits)
# sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
# sampling = Random(xlimits=prob.xlimits)

# sampling = Clustered(kernel=sampling)


ndim = prob.options['ndim']

# sm = RMTS(xlimits=prob.xlimits, min_energy=False, nln_max_iter=0)
sm = RMTB(xlimits=prob.xlimits, min_energy=False, nln_max_iter=0)
# sm = IDW()
# sm = KPLS(name='KRG', n_comp=ndim, theta0=[1.0]*ndim)
# sm = RBF(d0=1e0, poly_degree=1)

nt = 50000
ne = 1000 * ndim

np.random.seed(0)
xt = sampling(nt)
yt = prob(xt)
dyt = {}
for kx in range(ndim):
    dyt[kx] = prob(xt, kx)

np.random.seed(1)
xe = sampling(ne)
ye = prob(xe)

sm.add_training_pts('exact', xt, yt)
if 0:
    for kx in range(ndim):
        sm.add_training_pts('exact', xt, dyt[kx], kx)
sm.train()

print(sm.compute_rms_error())
print(sm.compute_rms_error(xe, ye))

nplot = 50
xe1 = np.zeros((nplot, ndim))
xe2 = np.zeros((nplot, ndim))
for kx in range(ndim):
    xe1[:, kx] = 0.25 * prob.xlimits[kx, 0] + 0.75 * prob.xlimits[kx, 1]
    xe2[:, kx] = 0.25 * prob.xlimits[kx, 0] + 0.75 * prob.xlimits[kx, 1]

a = 1.0

xe1[:, 0] = np.linspace(a*prob.xlimits[0, 0], a*prob.xlimits[0, 1], nplot)
ye1 = prob(xe1)
y1 = sm.predict(xe1)
plt.subplot(2, 1, 1)
plt.plot(xe1[:, 0], ye1, '-or')
plt.plot(xe1[:, 0], y1, '-ob')

xe2[:, -1] = np.linspace(a*prob.xlimits[-1, 0], a*prob.xlimits[-1, 1], nplot)
ye2 = prob(xe2)
y2 = sm.predict(xe2)
plt.subplot(2, 1, 2)
plt.plot(xe2[:, -1], ye2, '-or')
plt.plot(xe2[:, -1], y2, '-ob')

plt.show()
