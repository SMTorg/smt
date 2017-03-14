from __future__ import print_function, division
import numpy as np
import pylab

from smt.problems import CantileverBeam, Carre, ReducedProblem, RobotArm, Rosenbrock
from smt.problems import TensorProduct, TorsionVibration, WaterFlow, WeldedBeam, WingWeight
from smt.sampling import LHS, Random, FullFactorial, Clustered
from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS
from smt.idw import IDW
from smt.rmts import RMTS
from smt.rmtb import RMTB


# prob = Carre(ndim=4)
prob = TensorProduct(ndim=3, func='tanh', width=10.)
# prob = WeldedBeam(ndim=3)
# prob = CantileverBeam(ndim=3)
# prob = Rosenbrock(ndim=4)
# prob = RobotArm(ndim=4)


sampling = LHS(xlimits=prob.xlimits)
sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
# sampling = Random(xlimits=prob.xlimits)

sampling = Clustered(kernel=sampling)


ndim = prob.options['ndim']

sm = RMTS({'name':'RMTS','num_elem':[4]*ndim, 'smoothness':[1.0]*ndim,
    'xlimits':prob.xlimits, 'approx_norm': 4, 'min_energy': False,
    'reg_dv': 1e-10, 'reg_cons': 1e-10, 'save_solution': False,
    'mg_factors': [4], 'solver': 'krylov', 'max_nln_iter': 10,
    'line_search': 'backtracking', 'max_print_depth': 5,
}, {})
sm = RMTB({'name':'RMTB', 'order':[3]*ndim, 'num_ctrl_pts':[30]*ndim, 'xlimits':prob.xlimits,
    'max_nln_iter': 0, 'max_print_depth': 5, 'min_energy': True, 'save_solution': False,
    'line_search': 'null', 'reg_dv': 1e-15, 'reg_cons': 1e-4})
# sm = IDW({'name':'IDW'},{'global':False})
# sm = KPLS({'name':'KRG', 'n_comp':ndim, 'theta0': [1e-2]*ndim},{})

nt = 1000 * ndim
ne = 100 * ndim

np.random.seed(0)
xt = sampling(nt)
yt = prob(xt)

np.random.seed(1)
xe = sampling(ne)
ye = prob(xe)

sm.add_training_pts('exact', xt, yt)
sm.train()

print(sm.compute_rms_error())
print(sm.compute_rms_error(xe, ye))

nplot = 50
xe1 = np.zeros((nplot, ndim))
xe2 = np.zeros((nplot, ndim))
for kx in range(ndim):
    xe1[:, kx] = 0.2 * prob.xlimits[kx, 0] + 0.8 * prob.xlimits[kx, 1]
    xe2[:, kx] = 0.2 * prob.xlimits[kx, 0] + 0.8 * prob.xlimits[kx, 1]

a = 1.0

xe1[:, 0] = np.linspace(a*prob.xlimits[0, 0], a*prob.xlimits[0, 1], nplot)
ye1 = prob(xe1)
y1 = sm.predict(xe1)
pylab.subplot(2, 1, 1)
pylab.plot(xe1[:, 0], ye1, '-or')
pylab.plot(xe1[:, 0], y1, '-ob')

xe2[:, -1] = np.linspace(a*prob.xlimits[-1, 0], a*prob.xlimits[-1, 1], nplot)
ye2 = prob(xe2)
y2 = sm.predict(xe2)
pylab.subplot(2, 1, 2)
pylab.plot(xe2[:, -1], ye2, '-or')
pylab.plot(xe2[:, -1], y2, '-ob')

pylab.show()
