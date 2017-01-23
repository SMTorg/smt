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
dim = 3
ndoe = 500*dim

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
t = RMTS({'name':'RMTS','num_elem':[2]*dim, 'smoothness':[1.0]*dim,
    'xlimits':xlimits, 'mode': 'approx'},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'RMTS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))

########### The MBR model
t = MBR({'name':'MBR','order':[4]*dim,'num_ctrl_pts':[5]*dim,
    'xlimits':xlimits},{})
t.add_training_pts('exact',xt,yt)
t.train()
y = t.predict(xtest)

print 'MBR,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1))))
