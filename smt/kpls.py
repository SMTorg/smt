"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)


TODO
Add additional points GEKPLS1, GEKPLS2 and so on
"""

from __future__ import division
import warnings

import numpy as np
from scipy import linalg, optimize
from pyDOE import *

from smt.sm import SM
from smt.pairwise import manhattan_distances
from smt.pls import pls as _pls

def standardization(X,y,copy=False):

    """
    We substract the mean from each variable. Then, we divide the values of each
    variable by its standard deviation.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    y: np.ndarray [n_obs, 1]
            - The output variable.

    copy: bool
            - A copy of matrices X and y will be used (copy = True).
            - Matrices X and y will be used. The matrices X and y will be
              normalized (copy = False).
            - (copy = False by default).

    Returns
    -------

    X: np.ndarray [n_obs, dim]
          The standardized input matrix.

    y: np.ndarray [n_obs, 1]
          The standardized output vector.

    X_mean: list(dim)
            The mean of each input variable.

    y_mean: list(1)
            The mean of the output variable.

    X_std:  list(dim)
            The standard deviation of each input variable.

    y_std:  list(1)
            The standard deviation of the output variable.

    """
    X_mean = np.mean(X, axis=0)
    X_std = X.std(axis=0,ddof=1)
    y_mean = np.mean(y, axis=0)
    y_std = y.std(axis=0,ddof=1)
    X_std[X_std == 0.] = 1.
    y_std[y_std == 0.] = 1.

    # center and scale X
    if copy:
        Xr = (X.copy() - X_mean) / X_std
        yr = (y.copy() - y_mean) / y_std
        return Xr, yr, X_mean, y_mean, X_std, y_std

    else:
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
        return X, y, X_mean, y_mean, X_std, y_std


def l1_cross_distances(X):

    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    Returns
    -------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The L1 cross-distances between the vectors in X.

    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """

    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij.astype(np.int)


def componentwise_distance(D,corr,n_comp,dim,coeff_pls,limit=int(1e4)):

    """
    Computes the nonzero componentwise cross-spatial-correlation-distance
    between the vectors in X.

    Parameters
    ----------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The L1 cross-distances between the vectors in X.

    corr: str
            - Name of the correlation function used.
              squar_exp or abs_exp.

    n_comp: int
            - Number of principal components used.

    dim: int
            - Number of dimension.

    coeff_pls: np.ndarray [dim, n_comp]
            - The PLS-coefficients.

    limit: int
            - Manage the memory.

    Returns
    -------

    D_corr: np.ndarray [n_obs * (n_obs - 1) / 2, n_comp]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.

    """

    D_corr = np.zeros((D.shape[0],n_comp))
    i,nb_limit  = 0,int(limit)

    if n_comp == dim:
        # Kriging
        while True:
            if i * nb_limit > D_corr.shape[0]:
                return D_corr
            else:
                if corr == 'squar_exp':
                    D_corr[i*nb_limit:(i+1)*nb_limit,:] = D[i*nb_limit:(i+1)*
                                                             nb_limit,:]**2
                else:
                    # abs_exp
                    D_corr[i*nb_limit:(i+1)*nb_limit,:] = np.abs(D[i*nb_limit:
                                                            (i+1)*nb_limit,:])
                i+=1
    else:
        # KPLS or GEKPLS
        while True:
            if i * nb_limit > D_corr.shape[0]:
                return D_corr
            else:
                if corr == 'squar_exp':
                    D_corr[i*nb_limit:(i+1)*nb_limit,:] = np.dot(D[i*nb_limit:
                                    (i+1)*nb_limit,:]** 2,coeff_pls**2)
                else:
                    # abs_exp
                    D_corr[i*nb_limit:(i+1)*nb_limit,:] = np.dot(np.abs(D[i*
                        nb_limit:(i+1)*nb_limit,:]),np.abs(coeff_pls))
                i+=1


def compute_pls(X,y,n_comp,pts=None,delta_x=None,xlimits=None,extra_pts=0,
                opt=0):

    """
    Computes the PLS-coefficients.

    Parameters
    ----------

    X: np.ndarray [n_obs,dim]
            - - The input variables.

    y: np.ndarray [n_obs,1]
            - The output variable

    n_comp: int
            - Number of principal components used.

    pts: dict()
            - The gradient values.

    delta_x: real
            - The step used in the FOTA.

    xlimits: np.ndarray[dim, 2]
            - The upper and lower var bounds.

    extra_pts: int
            - The number of extra points per each training point.

    opt: int
            - opt = 0: using the KPLS model.
            - opt = 1: using the GEKPLS model.

    Returns
    -------

    Coeff_pls: np.ndarray[dim, n_comp]
            - The PLS-coefficients.

    XX: np.ndarray[extra_pts*nt, dim]
            - Extra points added (only when extra_pts > 0)

    yy: np.ndarray[extra_pts*nt, 1]
            - Extra points added (only when extra_pts > 0)

    """
    nt,dim = X.shape
    XX = np.empty(shape = (0,dim))
    yy = np.empty(shape = (0,1))
    pls = _pls(n_comp)

    if opt == 0:
        #KPLS
        pls.fit(X,y)
        return np.abs(pls.x_rotations_), XX, yy
    elif opt == 1:
        #GEKPLS-KPLS
        coeff_pls = np.zeros((nt,dim,n_comp))
        for i in range(nt):
            if dim >= 3:
                sign = np.roll(bbdesign(dim,center=1),1,axis=0)
                _X = np.zeros((sign.shape[0],dim))
                _y = np.zeros((sign.shape[0],1))
                sign = sign * delta_x*(xlimits[:,1]-xlimits[:,0])
                _X = X[i,:]+ sign
                for j in range(1,dim+1):
                    sign[:,j-1] = sign[:,j-1]*pts['exact'][j][1][i,0]
                _y = y[i,:]+ np.sum(sign,axis=1).reshape((sign.shape[0],1))
            else:
                _X = np.zeros((9,dim))
                _y = np.zeros((9,1))
                # center
                _X[:,:] = X[i,:].copy()
                _y[0,0] = y[i,0].copy()
                # right
                _X[1,0] +=delta_x*(xlimits[0,1]-xlimits[0,0])
                _y[1,0] = _y[0,0].copy()+ pts['exact'][1][1][i,0]*delta_x*(
                    xlimits[0,1]-xlimits[0,0])
                # up
                _X[2,1] +=delta_x*(xlimits[1,1]-xlimits[1,0])
                _y[2,0] = _y[0,0].copy()+ pts['exact'][2][1][i,0]*delta_x*(
                    xlimits[1,1]-xlimits[1,0])
                # left
                _X[3,0] -=delta_x*(xlimits[0,1]-xlimits[0,0])
                _y[3,0] = _y[0,0].copy()- pts['exact'][1][1][i,0]*delta_x*(
                    xlimits[0,1]-xlimits[0,0])
                # down
                _X[4,1] -=delta_x*(xlimits[1,1]-xlimits[1,0])
                _y[4,0] = _y[0,0].copy()-pts['exact'][2][1][i,0]*delta_x*(
                    xlimits[1,1]-xlimits[1,0])
                # right up
                _X[5,0] +=delta_x*(xlimits[0,1]-xlimits[0,0])
                _X[5,1] +=delta_x*(xlimits[1,1]-xlimits[1,0])
                _y[5,0] = _y[0,0].copy()+ pts['exact'][1][1][i,0]*delta_x*(
                    xlimits[0,1]-xlimits[0,0])+pts['exact'][2][1][i,0]*delta_x*(
                    xlimits[1,1]-xlimits[1,0])
                # left up
                _X[6,0] -=delta_x*(xlimits[0,1]-xlimits[0,0])
                _X[6,1] +=delta_x*(xlimits[1,1]-xlimits[1,0])
                _y[6,0] = _y[0,0].copy()- pts['exact'][1][1][i,0]*delta_x*(
                    xlimits[0,1]-xlimits[0,0])+pts['exact'][2][1][i,0]*delta_x*(
                    xlimits[1,1]-xlimits[1,0])
                # left down
                _X[7,0] -=delta_x*(xlimits[0,1]-xlimits[0,0])
                _X[7,1] -=delta_x*(xlimits[1,1]-xlimits[1,0])
                _y[7,0] = _y[0,0].copy()- pts['exact'][1][1][i,0]*delta_x*(
                    xlimits[0,1]-xlimits[0,0])-pts['exact'][2][1][i,0]*delta_x*(
                    xlimits[1,1]-xlimits[1,0])
                # right down
                _X[3,0] +=delta_x*(xlimits[0,1]-xlimits[0,0])
                _X[3,1] -=delta_x*(xlimits[1,1]-xlimits[1,0])
                _y[3,0] = _y[0,0].copy()+ pts['exact'][1][1][i,0]*delta_x*(
                    xlimits[0,1]-xlimits[0,0])-pts['exact'][2][1][i,0]*delta_x*(
                    xlimits[1,1]-xlimits[1,0])

            pls.fit(_X.copy(),_y.copy())
            coeff_pls[i,:,:] = pls.x_rotations_
            #Add additional points
            if extra_pts != 0:
                max_coeff = np.argsort(np.abs(coeff_pls[i,:,0]))[-extra_pts:]
                for ii in max_coeff:
                    XX = np.vstack((XX,X[i,:]))
                    XX[-1,ii] += delta_x*(xlimits[ii,1]-xlimits[ii,0])
                    yy = np.vstack((yy,y[i,0]))
                    yy[-1,0] += pts['exact'][1+ii][1][i,0]*delta_x*(
                        xlimits[ii,1]-xlimits[ii,0])

        return np.abs(coeff_pls).mean(axis=0), XX, yy

"""
The kpls-correlation models subroutine.
"""

def abs_exp(theta, d):

    """
    Absolute exponential autocorrelation model.
    (Ornstein-Uhlenbeck stochastic process)::

    Parameters
    ----------
    theta : list[ncomp]
        the autocorrelation parameter(s).

    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        - |d_i * coeff_pls_i| if PLS is used, |d_i| otherwise

    Returns
    -------
    r : np.ndarray[n_obs * (n_obs - 1) / 2,1]
        - An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0],1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i,nb_limit  = 0,int(1e4)
    while True:
        if i * nb_limit > d.shape[0]:
            return r
        else:
            r[i*nb_limit:(i+1)*nb_limit,0] = np.exp(-np.sum(theta.reshape(1,
                    n_components) * d[i*nb_limit:(i+1)*nb_limit,:], axis=1))
            i+=1


def squar_exp(theta, d):

    """
    Squared exponential correlation model.

    Parameters
    ----------
    theta : list[ncomp]
        the autocorrelation parameter(s).

    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
            - |d_i * coeff_pls_i| if PLS is used, |d_i| otherwise

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
            - An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0],1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i,nb_limit  = 0,int(1e4)

    while True:
        if i * nb_limit > d.shape[0]:
            return r
        else:
            r[i*nb_limit:(i+1)*nb_limit,0] = np.exp(-np.sum(theta.reshape(1,
                    n_components) * d[i*nb_limit:(i+1)*nb_limit,:], axis=1))
            i+=1


"""
The built-in regression models subroutine for the KPLS module.
"""

def constant(x):

    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    Parameters
    ----------
    x: np.ndarray[n_obs,dim]
            - An array giving the locations x at which the regression model
              should be evaluated.

    Returns
    -------
    f: np.ndarray[n_obs,p]
            - An array with the values of the regression model.
    """

    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])

    return f


def linear(x):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x: np.ndarray[n_obs,dim]
            - An array giving the locations x at which the regression model
              should be evaluated.

    Returns
    -------
    f: np.ndarray[n_obs,p]
            - An array with the values of the regression model.
    """

    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])

    return f


def quadratic(x):

    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x: np.ndarray[n_obs,dim]
            - An array giving the locations x at which the regression model
              should be evaluated.

    Returns
    -------
    f: np.ndarray[n_obs,p]
            - An array with the values of the regression model.
    """

    x = np.asarray(x, dtype=np.float)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f

"""
The KPLS class.
"""

class KPLS(SM):

    '''
    - Ordinary kriging
    - KPLS(-K)
    - GEKPLS
    '''

    _regression_types = {
        'constant': constant,
        'linear': linear,
        'quadratic': quadratic}

    _correlation_types = {
        'abs_exp': abs_exp,
        'squar_exp': squar_exp}

    def _set_default_options(self):

        '''
        Constructor.

        Arguments
        ---------
        sm_options : dict
            Model-related options, listed below

        printf_options : dict
            Output printing options, listed below
        '''

        sm_options = {
            'name': 'KPLS',               # KRG for Standard kriging if n_comp
                                          # = dimension
                                          # KPLS for Kriging combined Partial
                                          # Least Squares
                                          # KPLSK for Kriging combined Partial
                                          # Least Squares + local optim Kriging
                                          # GEKPLS for Gradient Enhanced KPLS
            'n_comp': 1,                  # Number of principal components
            'theta0': [1e-2],             # Initial hyperparameters
            'delta_x': 1e-4,              # Step used in the FOTA
            'xlimits': None,              # np.ndarray[nx, 2]: upper and lower
                                          # var bounds
            'extra_pts': 0,               # Number of extra points per each
                                          # training point
            'poly' : 'constant',          # Regression term
            'corr' :  'squar_exp',        # Type of the correlation function
        }
        printf_options = {
            'global': True,               # Overriding option to print output
            'time_eval': True,            # Print evaluation times
            'time_train': True,           # Print assembly and solution time summary
            'problem': True,              # Print problem information
        }

        sm_options['best_iteration_fail'] = None
        sm_options['nb_ill_matrix'] = 5
        self.sm_options = sm_options
        self.printf_options = printf_options


    ############################################################################
    # Model functions
    ############################################################################


    def fit(self):

        """
        Train the model
        """

        self._check_param()

        # Compute PLS coefficients
        X = self.training_pts['exact'][0][0]
        y = self.training_pts['exact'][0][1]

        if 0 in self.training_pts['exact']:
            #GEKPLS
            if 1 in self.training_pts['exact'] and self.sm_options['name'] == 'GEKPLS':
                self.coeff_pls, XX, yy = compute_pls(X.copy(),y.copy(),
                    self.sm_options['n_comp'],self.training_pts,
                    self.sm_options['delta_x'],self.sm_options['xlimits'],
                                            self.sm_options['extra_pts'],1)
                if self.sm_options['extra_pts'] != 0:
                    self.nt *= (self.sm_options['extra_pts']+1)
                    X = np.vstack((X,XX))
                    y = np.vstack((y,yy))
            #KPLS
            elif (self.sm_options['name'] == 'KPLS' or self.sm_options['name']
                  == 'KPLSK') and self.sm_options['n_comp'] < self.dim:
                self.coeff_pls, XX, yy = compute_pls(X.copy(),y.copy(), \
                   self.sm_options['n_comp'])
            #Kriging
            else:
                self.coeff_pls = None

        # Center and scale X and y
        self.X_norma, self.y_norma, self.X_mean, self.y_mean, self.X_std, \
            self.y_std = standardization(X,y)

        # Calculate matrix of distances D between samples
        D, self.ij = l1_cross_distances(self.X_norma)
        if (np.min(np.sum(D, axis=1)) == 0.):
            raise Exception("Multiple input features cannot have the same value.")

        # Regression matrix and parameters
        self.F = self.sm_options['poly'](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F,p)

        # Optimization
        self.optimal_rlf_value, self.optimal_par, self.optimal_theta = \
                self._optimize_hyperparam(D)

        del self.y_norma, self.D


    def _reduced_likelihood_function(self, theta):

        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta: list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        reduced_likelihood_function_value: real
            - The value of the reduced likelihood function associated to the
              given autocorrelation parameters theta.

        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:

            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or given beta0 for Ordinary
            Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            G
            QR decomposition of the matrix Ft.
        """
        # Initialize output
        reduced_likelihood_function_value = - np.inf
        par = {}

        # Set up R
        r = self.sm_options['corr'](theta, self.D)
        MACHINE_EPSILON = np.finfo(np.double).eps
        nugget = 10.*MACHINE_EPSILON
        R = np.eye(self.nt) * (1. + nugget)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:,0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:,0]

        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError:
            return reduced_likelihood_function_value, par

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode='economic')
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]

        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]

            if condF > 1e15:
                raise Exception("F is too ill conditioned. Poor combination "
                                "of regression model and observations.")

            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par
        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)
        sigma2 = (rho ** 2.).sum(axis=0) / (self.nt)

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2. / self.nt)).prod()

        # Compute/Organize output
        reduced_likelihood_function_value = - sigma2.sum() * detR
        par['sigma2'] = sigma2 * self.y_std ** 2.
        par['beta'] = beta
        par['gamma'] = linalg.solve_triangular(C.T, rho)
        par['C'] = C
        par['Ft'] = Ft
        par['G'] = G

        # A particular case when f_min_cobyla fail
        if ( self.sm_options['best_iteration_fail'] is not None) and \
            (not np.isinf(reduced_likelihood_function_value)):

            if (reduced_likelihood_function_value >  self.sm_options[
                    'best_iteration_fail']):
                 self.sm_options['best_iteration_fail'] = \
                    reduced_likelihood_function_value
                 self._thetaMemory = theta

        elif ( self.sm_options['best_iteration_fail'] is None) and \
            (not np.isinf(reduced_likelihood_function_value)):
             self.sm_options['best_iteration_fail'] = \
                    reduced_likelihood_function_value
             self._thetaMemory = theta

        return reduced_likelihood_function_value, par

    def evaluate(self, x, kx):
        """
        Evaluate the surrogate model at x.

        Parameters
        ----------
        x: np.ndarray[n_eval,dim]
        An array giving the point(s) at which the prediction(s) should be made.
        kx : int or None
        None if evaluation of the interpolant is desired.
        int  if evaluation of derivatives of the interpolant is desired
             with respect to the kx^{th} input variable (kx is 0-based).

        Returns
        -------
        y : np.ndarray[n_eval,1]
        - An array with the output values at x.
        """

        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        y = np.zeros(n_eval)

        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=
                                 False)
        d = componentwise_distance(dx,self.sm_options['corr'].__name__,
                                        self.sm_options['n_comp'],self.dim,
                                        self.coeff_pls,1e2)

        # Get regression function and correlation
        f = self.sm_options['poly'](x)
        r = self.sm_options['corr'](self.optimal_theta, d).reshape(n_eval,
                                        self.nt)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par['beta']) + np.dot(r,
                    self.optimal_par['gamma'])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()

        return y


    def _optimize_hyperparam(self,D):

        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        D: np.ndarray [n_obs * (n_obs - 1) / 2, n_comp]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.

        Returns
        -------
        best_optimal_rlf_value: real
            - The value of the reduced likelihood function associated to the
              best autocorrelation parameters theta.


        best_optimal_par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters.


        best_optimal_theta: list(n_comp)
            - The best hyperparameters found by the optimization.
        """

        # Initialize the hyperparameter-optimization
        def minus_reduced_likelihood_function(log10t):
                return - self._reduced_likelihood_function(
                    theta=10.**log10t)[0]

        key, limit, _rhobeg = True, 10*self.sm_options['n_comp'], 0.5

        for ii in range(self.sm_options['kriging-step']+1):
            best_optimal_theta, best_optimal_rlf_value, best_optimal_par, \
                constraints = [], [], [], []

            for i in range(self.sm_options['n_comp']):
                constraints.append(lambda log10t,i=i:
                                   log10t[i] - np.log10(1e-6))
                constraints.append(lambda log10t,i=i:
                                   np.log10(10) - log10t[i])

            # Compute D which is the componentwise distances between locations
            #  x and x' at which the correlation model should be evaluated.
            self.D = componentwise_distance(D,
                                        self.sm_options['corr'].__name__,
                                        self.sm_options['n_comp'],self.dim,
                                        self.coeff_pls)

            # Initialization
            k, incr, stop, best_optimal_rlf_value = 0, 0, 1, -1e20
            while (k < stop):
                # Use specified starting point as first guess
                theta0 = self.sm_options['theta0']
                try:
                    optimal_theta = 10. ** optimize.fmin_cobyla(
                        minus_reduced_likelihood_function, np.log10(theta0),
                        constraints, rhobeg= _rhobeg, rhoend = 1e-4,iprint=0,
                        maxfun=limit)

                    optimal_rlf_value, optimal_par = \
                        self._reduced_likelihood_function(theta=optimal_theta)

                    # Compare the new optimizer to the best previous one
                    if k > 0:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                            if incr != 0:
                                return
                        else:
                            if optimal_rlf_value >= self.sm_options[
                                    'best_iteration_fail'] :
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta
                                else:
                                    if  self.sm_options['best_iteration_fail'] \
                                        > best_optimal_rlf_value:
                                        best_optimal_theta = self._thetaMemory
                                        best_optimal_rlf_value , best_optimal_par = \
                                          self._reduced_likelihood_function(\
                                          theta= best_optimal_theta)
                    else:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                        else:
                            if optimal_rlf_value >=  self.sm_options[
                                    'best_iteration_fail']:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta

                            else:
                                if  self.sm_options['best_iteration_fail'] > \
                                    best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory.copy()
                                    best_optimal_rlf_value , best_optimal_par = \
                                        self._reduced_likelihood_function( \
                                            theta=best_optimal_theta)
                    k += 1
                except ValueError as ve:
                    # If iteration is max when fmin_cobyla fail is not reached
                    if (self.sm_options['nb_ill_matrix'] > 0):
                        self.sm_options['nb_ill_matrix'] -= 1
                        k += 1
                        stop += 1
                        # One evaluation objectif function is done at least
                        if ( self.sm_options['best_iteration_fail'] is not
                             None):
                            if  self.sm_options['best_iteration_fail'] > \
                                best_optimal_rlf_value:
                                best_optimal_theta = self._thetaMemory
                                best_optimal_rlf_value , best_optimal_par = \
                                    self._reduced_likelihood_function(theta=
                                    best_optimal_theta)
                    # Optimization fail
                    elif best_optimal_par == [] :
                        print("Optimization failed. Try increasing the ``nugget``")
                        raise ve
                    # Break the while loop
                    else:
                        k = stop + 1
                        print("fmin_cobyla failed but the best value is retained")

            if self.sm_options['kriging-step']:
                # Next iteration is to do a kriging model starting from the
                # given by the KPLS or GEKPLS models
                if key:
                    if self.sm_options['corr'].__name__ == 'squar_exp':
                        self.sm_options['theta0'] = (best_optimal_theta *
                                                 self.coeff_pls**2).sum(1)
                    else:
                        self.sm_options['theta0'] = (best_optimal_theta *
                                                 np.abs(self.coeff_pls)).sum(1)
                    self.sm_options['n_comp'] = self.dim
                    key, limit, _rhobeg,self.sm_options[
                        'best_iteration_fail'] = False, 3*self.sm_options[
                        'n_comp'], 0.05, None

        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta


    def _check_param(self):

        """
        This function check some parameters of the model.
        """

        # Check regression model
        if not callable(self.sm_options['poly']):
            if self.sm_options['poly'] in self._regression_types:
                self.sm_options['poly'] = self._regression_types[
                    self.sm_options['poly']]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given." % (self._regression_types.keys(
                                 ), self.sm_options['poly']))


        if not(self.sm_options['name'] in ['KRG','KPLS','GEKPLS','KPLSK']):
            raise Exception("The %s model is not found in ['KRG','KPLS','GEKPLS','KPLSK']."
                               %(self.sm_options['name']))
        else:
            if self.sm_options['name'] == 'KRG':
                if self.sm_options['n_comp'] != self.dim:
                    raise Exception('The number of principal components must be equal to the number of dimension for using the kriging model.')

                self.sm_options['kriging-step'] = 0

            elif self.sm_options['name'] == 'GEKPLS':
                if not(1 in self.training_pts['exact']):
                    raise Exception('Derivative values are needed for using the GEKPLS model.')
                self.sm_options['kriging-step'] = 0

            elif self.sm_options['name'] == 'KPLSK':
                self.sm_options['kriging-step'] = 1

            else:
                self.sm_options['kriging-step'] = 0

        if self.dim == self.sm_options['n_comp']:
            if self.sm_options['name'] != 'KRG':
                warnings.warn('Kriging is used instead of the KPLS model!')
                self.sm_options['name'] = 'KRG'

        if len(self.sm_options['theta0']) != self.sm_options['n_comp']:
            raise('Number of principal components must be equal to the number of theta0.')

        if not callable(self.sm_options['corr']):
            if self.sm_options['corr'] in self._correlation_types:
                self.sm_options['corr'] = self._correlation_types[self.sm_options['corr']]

            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.sm_options['corr']))


    def _check_F(self,n_samples_F,p):

        """
        This function check the F-parameters of the model.
        """

        if n_samples_F != self.nt:
            raise Exception("Number of rows in F and X do not match. Most "
                            "likely something is going wrong with the "
                            "regression model.")
        if p > n_samples_F:
            raise Exception(("Ordinary least squares problem is undetermined "
                             "n_samples=%d must be greater than the "
                             "regression model size p=%d.") % (self.nt, p))
