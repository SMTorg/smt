from __future__ import division
import numpy as np
from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS
from smt.idw import IDW
from smt.rmts import RMTS
from smt.mbr import MBR
from scipy import linalg
import tools_benchmark as fun
from tools_doe import trans
import doe_lhs

np.random.seed(0)

# Initialization of the problem
dim = 4
ndoe = 100*dim

# Upper and lower bounds of the problem
xlimits = np.zeros((dim, 2))
xlimits[:, 0] = -10
xlimits[:, 1] = 10

# Construction of the DOE in [0,1]
xt = doe_lhs.lhs(dim,ndoe,'m')

# Transform the DOE in [LB,UB]
xt = trans(xt,xlimits[:, 0],xlimits[:, 1])

# Compute the output (+ the gradient)
yt,yd = fun.carre(xt)

# Construction of the validation points
ntest = 500
xtest = doe_lhs.lhs(dim, ntest)
xtest = trans(xtest,xlimits[:, 0],xlimits[:, 1])
ytest,ydtest = fun.carre(xtest)
ntest = xtest.shape[0]

########### The LS model

# Initialization of the model
t = LS({'name':'LS'},{})
# Add the DOE
t.add_training_pts('exact',xt,yt)
# Train the model
t.train()
# Prediction of the validation points
y = t.predict(xtest)

print 'LS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))

########### The PA2 model
t = PA2({'name':'PA2'},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'PA2,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))

########### The IDW model
t = IDW({'name':'IDW'},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'IDW,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))

########### The RMTS model
t = RMTS({'name':'RMTS','num_elem':[4]*dim, 'smoothness':[1.0]*dim, 'xlimits':xlimits,
    'mode': 'approx', 'solver_mg': [4], 'solver_type': 'krylov-mg', 'solver_pc': 'lu',
    'solver_krylov': 'fgmres', 'solver_rtol': 1e-10,
},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'RMTS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))


exit()
import RMTSlib
import scipy.sparse
elem_lists = [
    np.array(t.sm_options['num_elem']),
    np.array(t.sm_options['num_elem']) / 2.]
nx = 1
num = {'term': 4}
mg_full_uniq2coeff = t._compute_uniq2coeff(1, elem_lists[-1],
    np.prod(elem_lists[-1]), num['term'], np.prod(elem_lists[-1] + 1))

ne = np.prod(elem_lists[-2] + 1) * 2 ** nx
nnz = ne * num['term']
num_coeff = num['term'] * np.prod(elem_lists[-1])
data, rows, cols = RMTSlib.compute_jac_interp(
    nnz, nx, elem_lists[-1], elem_lists[-2] + 1, xlimits)
mg_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(ne, num_coeff))
mg_matrix = mg_jac * mg_full_uniq2coeff

nnz = np.prod(elem_lists[-2] + 1) * 4 ** nx
nrows = np.prod(elem_lists[-2] + 1) * 2 ** nx
ncols = np.prod(elem_lists[-1] + 1) * 2 ** nx
data, rows, cols = RMTSlib.compute_mg_interp(nx, nnz, elem_lists[-1])
mg_matrix = scipy.sparse.csc_matrix((data, (rows, cols)),
                                    shape=(nrows, ncols))

np.set_printoptions(precision=2)
print(mg_jac.todense())
print(mg_full_uniq2coeff.todense())
print(mg_matrix.todense())
print(mg_matrix.T.dot(mg_matrix).todense())
print(t.sol[:2*elem_lists[-2][0]+2])

n1 = elem_lists[-2][0] + 1
n2 = elem_lists[-1][0] + 1
x1 = np.linspace(-10, 10, n1)
x2 = np.linspace(-10, 10, n2)

y1 = np.sin(x1) # * 6.14/20)
dydx1 = np.cos(x1) # * 6.14/20) * 6.14/20
u1 = np.zeros(2*n1)
u1[:n1] = y1
u1[n1:] = dydx1

print(n1, n2, u1.shape, mg_matrix.shape)
u2 = mg_matrix.T.dot(u1)
y2 = u2[:n2]
dydx2 = u2[n2:]

u3 = mg_matrix.dot(u2)
y3 = u3[:n1]
dydx3 = u3[n1:]

print(mg_matrix.T.dot(mg_matrix).dot(np.ones(2*n2)))

import scipy.sparse.linalg
mtx = t.tmp
lu = scipy.sparse.linalg.splu(mtx)
inv = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=lu.solve)
w1, v = scipy.sparse.linalg.eigs(mtx)
w2, v = scipy.sparse.linalg.eigs(inv)
print(w1)
print(w2)
exit(0)

import pylab
pylab.subplot(2, 1, 1)
pylab.plot(x1, y1, 'b-o')
pylab.plot(x2, y2+0.1, 'r-o')
pylab.plot(x1, y3, 'k-o')
pylab.subplot(2, 1, 2)
pylab.plot(x1, dydx1, 'b-o')
pylab.plot(x2, dydx2, 'r-o')
pylab.plot(x1, dydx3, 'k-o')
pylab.show()

exit(0)




########### The MBR model
t = MBR({'name':'MBR','order':[4]*dim,'num_ctrl_pts':[5]*dim,
    'xlimits':xlimits
},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'MBR,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))
