from __future__ import print_function, division
import numpy as np

from smt.problems.carre import Carre
from smt.problems.tensor_product import TensorProduct
from smt.sampling.lhs import lhs_center
from smt.sampling.random import random
from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS
from smt.idw import IDW
from smt.rmts import RMTS
from smt.rmtb import RMTB


ndim = 3
# prob = Carre(ndim=ndim)
prob = TensorProduct(ndim=ndim, func='tanh', width=5.)
sampling = lhs_center

sm = RMTS({'name':'RMTS','num_elem':[8]*ndim, 'smoothness':[1.0]*ndim,
    'xlimits':prob.xlimits, 'approx_norm': 4,
    'reg_dv': 1e-10, 'reg_cons': 1e-10, 'save_solution': False,
    'mg_factors': [4], 'solver': 'krylov', 'max_nln_iter': 15,
    'line_search': 'backtracking', 'max_print_depth': 4,
}, {})
sm = RMTB({'name':'RMTB', 'order':[4]*ndim, 'num_ctrl_pts':[20]*ndim, 'xlimits':prob.xlimits,
    'max_nln_iter': 15, 'max_print_depth': 4, 'save_solution': False})
# sm = IDW({'name':'IDW'},{'global':False})
# sm = KPLS({'name':'KRG', 'n_comp':ndim, 'theta0': [1e-2]*ndim},{})

nt = 5000 * ndim
ne = 100 * ndim

np.random.seed(0)
xt = sampling(prob.xlimits, nt)
yt = prob(xt)

np.random.seed(1)
xe = sampling(prob.xlimits, ne)
ye = prob(xe)

sm.add_training_pts('exact', xt, yt)
sm.train()

print(sm.compute_rms_error())
print(sm.compute_rms_error(xe, ye))

xe = np.zeros((50, ndim))
for kx in range(ndim):
    xe[:, kx] = 0.25 * prob.xlimits[kx, 0] + 0.75 * prob.xlimits[kx, 1]
xe[:, 0] = np.linspace(1.2*prob.xlimits[0, 0], 1.2*prob.xlimits[0, 1], 50)
ye = prob(xe)
y = sm.predict(xe)
import pylab
pylab.plot(xe[:, 0], ye, '-or')
pylab.plot(xe[:, 0], y, '-ob')
pylab.show()
