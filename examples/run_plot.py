from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

from smt.problems import CantileverBeam, Carre, ReducedProblem, RobotArm, Rosenbrock
from smt.problems import TensorProduct, TorsionVibration, WaterFlow, WeldedBeam, WingWeight
from smt.problems import NdimCantileverBeam, NdimRobotArm, NdimRosenbrock, NdimStepFunction
from smt.sampling import LHS, Random, FullFactorial, Clustered
from smt import LS, PA2, KPLS, IDW, RBF, RMTS, RMTB


ndim = 3
prob = Carre(ndim=ndim)
# prob = TensorProduct(ndim=ndim, func='cos', width=1.)
# prob = WeldedBeam(ndim=3)
# prob = CantileverBeam(ndim=3*ndim)
# prob = Rosenbrock(ndim=ndim)
# prob = RobotArm(ndim=2*ndim)

prob = NdimCantileverBeam(ndim=ndim)
prob = NdimRobotArm(ndim=ndim)
# prob = NdimRosenbrock(ndim=ndim)
# prob = NdimStepFunction(ndim=ndim)

sampling = LHS(xlimits=prob.xlimits)
# sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
# sampling = Random(xlimits=prob.xlimits)

# sampling = Clustered(kernel=sampling)


ndim = prob.options['ndim']

sm = RMTS(xlimits=prob.xlimits)#, min_energy=False, nln_max_iter=0)
sm = RMTB(xlimits=prob.xlimits, min_energy=True, nln_max_iter=20)
# sm = IDW()
# sm = KPLS(name='KRG', n_comp=ndim, theta0=[1.0]*ndim)
# sm = RBF(d0=1e0, poly_degree=1)

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

sm.add_training_pts('exact', xt, yt)
if 0:
    for kx in range(ndim):
        sm.add_training_pts('exact', xt, dyt[kx], kx)
sm.train()

print(sm.compute_rms_error())
print(sm.compute_rms_error(xe, ye))

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
    ye2 = sm.predict(xe)
    plt.subplot(nr, nc, ix + 1)
    plt.plot(xe[:, ix], ye, '-or')
    plt.plot(xe[:, ix], ye2, '-ob')

plt.show()
