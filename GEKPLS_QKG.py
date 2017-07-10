import numpy as np
from smt.methods import GEKPLS


def provide_data(x,y,yd,xlimits,delta_x = 1e-3,nb_extra_pts = 1):

    ndoe,ndim = np.shape(x)
    t = GEKPLS(n_comp=1, theta0=[1e-2], xlimits=xlimits,
               delta_x=delta_x,extra_points= nb_extra_pts)

    t.add_training_points('exact',x,y)
    # Add the gradient information
    for i in range(ndim):        
        t.add_training_points('exact',x,yd[:, i].reshape((ndoe,1)),kx=i)

    t.train()

    est_hyperp = t.coeff_pls**2 * t.optimal_theta
    y = t.y_norma*t.y_std+t.y_mean
    x = t.X_norma*t.X_std+t.X_mean

    return est_hyperp, x, y
