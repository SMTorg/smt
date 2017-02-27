from __future__ import division
import numpy as np
from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS
from smt.idw import IDW
from scipy import linalg
import tools_benchmark as fun
from tools_doe import  trans
import doe_lhs

# Initialization of the problem
dim = 10
ndoe = 10*dim

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
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)

########### The PA2 model
t = PA2({'name':'PA2'},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'PA2,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)

########### The IDW model
t = IDW({'name':'IDW'},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'IDW,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)

########### The Kriging model
# The variables 'name', 'ncomp' and 'theta0' must be equal to 'Kriging',
# dim and a list of length dim, respectively.
t = KPLS({'name':'KRG','n_comp':dim,'theta0': [1e-2]*dim},{})
t.add_training_pts('exact',xt,yt)

t.train()
y = t.predict(xtest)

print 'Kriging,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)


########### The KPLS model
# The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
# an integer in [1,dim[ and a list of length n_comp, respectively. Here is an
# an example using 1 principal component.
t = KPLS({'name':'KPLS','n_comp':2,'theta0': [1e-2,1e-2]},{})
t.add_training_pts('exact',xt,yt)

t.train()
y = t.predict(xtest)

print 'KPLS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)


########### The KPLSK model
# The variables 'name' must be equal to 'KPLSK'. 'n_comp' and 'theta0' must be
# an integer in [1,dim[ and a list of length n_comp, respectively.
t = KPLS({'name':'KPLSK','n_comp':2,'theta0': [1e-2,1e-2]},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'KPLSK,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)

########### The GEKPLS model
# The variables 'name' must be equal to 'GEKPLS'. 'n_comp' and 'theta0' must be
# an integer in [1,dim[ and a list of length n_comp, respectively.
t = KPLS({'name':'GEKPLS','n_comp':2,'theta0': [1e-2,1e-2],'xlimits':xlimits},{})
t.add_training_pts('exact',xt,yt)
# Add the gradient information
for i in range(dim):
    t.add_training_pts('exact',xt,yd[:, i].reshape((yt.shape[0],1)),kx=i)

t.train()
y = t.predict(xtest)

print 'GEKPLS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))*100)
