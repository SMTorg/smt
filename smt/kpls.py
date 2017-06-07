"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)


TODO:
- Add additional points GEKPLS1, GEKPLS2 and so on

- define outputs['sol'] = self.sol

- debug _train: self_pkl = pickle.dumps(obj)
                           cPickle.PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed

"""

from __future__ import division
import warnings

import numpy as np
from scipy import linalg, optimize
from pyDOE import *
from types import FunctionType
from smt.utils.caching import cached_operation

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


def compute_pls(X,y,n_comp,pts=None,delta_x=None,xlimits=None,extra_points=0,
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

    extra_points: int
            - The number of extra points per each training point.

    opt: int
            - opt = 0: using the KPLS model.
            - opt = 1: using the GEKPLS model.

    Returns
    -------

    Coeff_pls: np.ndarray[dim, n_comp]
            - The PLS-coefficients.

    XX: np.ndarray[extra_points*nt, dim]
            - Extra points added (only when extra_points > 0)

    yy: np.ndarray[extra_points*nt, 1]
            - Extra points added (only when extra_points > 0)

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
            if extra_points != 0:
                max_coeff = np.argsort(np.abs(coeff_pls[i,:,0]))[-extra_points:]
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

    def _declare_options(self):
        super(KPLS, self)._declare_options()
        declare = self.options.declare

        declare('name', 'KPLS', types=str,
                desc='KRG for Standard kriging if n_comp = dimension ' +
                     'KPLS for Kriging with Partial Least Squares ' +
                     'KPLSK for Kriging with Partial Least Squares + local optim Kriging ' +
                     'GEKPLS for Gradient Enhanced KPLS')
        declare('xlimits', types=np.ndarray,
                desc='Lower/upper bounds in each dimension - ndarray [nx, 2]')
        declare('n_comp', 1, types=int, desc='Number of principal components')
        declare('theta0', [1e-2], types=(list, np.ndarray), desc='Initial hyperparameters')
        declare('delta_x', 1e-4, types=(int, float), desc='Step used in the FOTA')
        declare('extra_points', 0, types=int, desc='Number of extra points per training point')
        declare('poly', 'constant', values=('constant', 'linear', 'quadratic'), types=FunctionType,
                desc='regr. term')
        declare('corr', 'squar_exp', values=('abs_exp', 'squar_exp'), types=FunctionType,
                desc='type of corr. func.')
        declare('best_iteration_fail', None)
        declare('nb_ill_matrix', 5)
        declare('kriging-step')
        declare('data_dir', values=None, types=str,
                desc='Directory for loading / saving cached data; None means do not save or load')

    ############################################################################
    # Model functions
    ############################################################################


    def _new_train(self):

        """
        Train the model
        """

        self._check_param()

        # Compute PLS coefficients
        X = self.training_points['exact'][0][0]
        y = self.training_points['exact'][0][1]

        if 0 in self.training_points['exact']:
            #GEKPLS
            if 1 in self.training_points['exact'] and self.options['name'] == 'GEKPLS':
                self.coeff_pls, XX, yy = compute_pls(X.copy(),y.copy(),
                    self.options['n_comp'],self.training_points,
                    self.options['delta_x'],self.options['xlimits'],
                                            self.options['extra_points'],1)
                if self.options['extra_points'] != 0:
                    self.nt *= (self.options['extra_points']+1)
                    X = np.vstack((X,XX))
                    y = np.vstack((y,yy))
            #KPLS
            elif (self.options['name'] == 'KPLS' or self.options['name']
                  == 'KPLSK') and self.options['n_comp'] < self.dim:
                self.coeff_pls, XX, yy = compute_pls(X.copy(),y.copy(), \
                   self.options['n_comp'])
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
        self.F = self.options['poly'](self.X_norma)
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

    def _train(self):
        """
        Train the model
        """
        """
        inputs = {'self': self}
        with cached_operation(inputs, self.options['data_dir']) as outputs:
            if outputs:
                self.sol = outputs['sol']
            else:
                self._new_train()
                #outputs['sol'] = self.sol
        """
        self._new_train()

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
        r = self.options['corr'](theta, self.D)
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
        if ( self.options['best_iteration_fail'] is not None) and \
            (not np.isinf(reduced_likelihood_function_value)):

            if (reduced_likelihood_function_value >  self.options[
                    'best_iteration_fail']):
                 self.options['best_iteration_fail'] = \
                    reduced_likelihood_function_value
                 self._thetaMemory = theta

        elif ( self.options['best_iteration_fail'] is None) and \
            (not np.isinf(reduced_likelihood_function_value)):
             self.options['best_iteration_fail'] = \
                    reduced_likelihood_function_value
             self._thetaMemory = theta

        return reduced_likelihood_function_value, par

    def _predict(self, x, kx):
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
        - An array with the output values at x if dx = 0.
        - An array with the i-th partial derivative at x if dx != 0
        """

        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=
                                 False)
        d = componentwise_distance(dx,self.options['corr'].__name__,
                                   self.options['n_comp'],self.dim,
                                   self.coeff_pls)
        # Compute the correlation function
        r = self.options['corr'](self.optimal_theta, d).reshape(n_eval,
                                        self.nt)
        # Output prediction
        if kx == 0:
            y = np.zeros(n_eval)

            # Compute the regression function
            f = self.options['poly'](x)
            
            # Scaled predictor
            y_ = np.dot(f, self.optimal_par['beta']) + np.dot(r,
                        self.optimal_par['gamma'])
            # Predictor
            y = (self.y_mean + self.y_std * y_).ravel()

            return y
        # Gradient prediction
        else:            
            if self.options['corr'].__name__ != 'squar_exp':
                raise ValueError(
                'The derivative is only available for square exponential kernel')
            # Beta and gamma = R^-1(y-FBeta)
            beta = self.optimal_par['beta']
            gamma = self.optimal_par['gamma']
            
            if self.options['poly'].__name__ == 'constant':
                df = np.array([0])
            elif self.options['poly'].__name__ == 'linear':
                df = np.zeros((self.dim + 1, self.dim))
                df[1:,:] = 1
            else:
                raise ValueError(
                    'The derivative is only available for ordinary kriging or '+
                    'universal kriging using a linear trend')
            df_dx = np.dot(df.T, beta)
            d_dx=x[:,kx-1].reshape((n_eval,1))-self.X_norma[:,kx-1].reshape((1,self.nt))
            if self.options['name'] == 'KPLSK' or self.options['name'] == 'KRG':
                return (df_dx[0]-2*self.optimal_theta[kx-1]*np.dot(d_dx*r,gamma))* \
                       self.y_std/self.X_std[kx-1]
            else:
                # PLS-based models
                theta = np.sum(self.optimal_theta * self.coeff_pls**2,axis=1)
                return (df_dx[0]-2*theta[kx-1]*np.dot(d_dx*r,gamma))* \
                       self.y_std/self.X_std[kx-1]




            

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

        key, limit, _rhobeg = True, 10*self.options['n_comp'], 0.5

        for ii in range(self.options['kriging-step']+1):
            best_optimal_theta, best_optimal_rlf_value, best_optimal_par, \
                constraints = [], [], [], []

            for i in range(self.options['n_comp']):
                constraints.append(lambda log10t,i=i:
                                   log10t[i] - np.log10(1e-6))
                constraints.append(lambda log10t,i=i:
                                   np.log10(10) - log10t[i])

            # Compute D which is the componentwise distances between locations
            #  x and x' at which the correlation model should be evaluated.
            self.D = componentwise_distance(D,
                                        self.options['corr'].__name__,
                                        self.options['n_comp'],self.dim,
                                        self.coeff_pls)

            # Initialization
            k, incr, stop, best_optimal_rlf_value = 0, 0, 1, -1e20
            while (k < stop):
                # Use specified starting point as first guess
                theta0 = self.options['theta0']
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
                            if optimal_rlf_value >= self.options[
                                    'best_iteration_fail'] :
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta
                                else:
                                    if  self.options['best_iteration_fail'] \
                                        > best_optimal_rlf_value:
                                        best_optimal_theta = self._thetaMemory
                                        best_optimal_rlf_value , best_optimal_par = \
                                          self._reduced_likelihood_function(\
                                          theta= best_optimal_theta)
                    else:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                        else:
                            if optimal_rlf_value >=  self.options[
                                    'best_iteration_fail']:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta

                            else:
                                if  self.options['best_iteration_fail'] > \
                                    best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory.copy()
                                    best_optimal_rlf_value , best_optimal_par = \
                                        self._reduced_likelihood_function( \
                                            theta=best_optimal_theta)
                    k += 1
                except ValueError as ve:
                    # If iteration is max when fmin_cobyla fail is not reached
                    if (self.options['nb_ill_matrix'] > 0):
                        self.options['nb_ill_matrix'] -= 1
                        k += 1
                        stop += 1
                        # One evaluation objectif function is done at least
                        if ( self.options['best_iteration_fail'] is not
                             None):
                            if  self.options['best_iteration_fail'] > \
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

            if self.options['kriging-step']:
                # Next iteration is to do a kriging model starting from the
                # given by the KPLS or GEKPLS models
                if key:
                    if self.options['corr'].__name__ == 'squar_exp':
                        self.options['theta0'] = (best_optimal_theta *
                                                 self.coeff_pls**2).sum(1)
                    else:
                        self.options['theta0'] = (best_optimal_theta *
                                                 np.abs(self.coeff_pls)).sum(1)
                    self.options['n_comp'] = self.dim
                    key, limit, _rhobeg,self.options[
                        'best_iteration_fail'] = False, 3*self.options[
                        'n_comp'], 0.05, None

        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta


    def _check_param(self):

        """
        This function check some parameters of the model.
        """

        # Check regression model
        if not callable(self.options['poly']):
            if self.options['poly'] in self._regression_types:
                self.options['poly'] = self._regression_types[
                    self.options['poly']]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given." % (self._regression_types.keys(
                                 ), self.options['poly']))


        if not(self.options['name'] in ['KRG','KPLS','GEKPLS','KPLSK']):
            raise Exception("The %s model is not found in ['KRG','KPLS','GEKPLS','KPLSK']."
                               %(self.options['name']))
        else:
            if self.options['name'] == 'KRG':
                if self.options['n_comp'] != self.dim:
                    raise Exception('The number of principal components must be equal to the number of dimension for using the kriging model.')

                self.options['kriging-step'] = 0

            elif self.options['name'] == 'GEKPLS':
                if not(1 in self.training_points['exact']):
                    raise Exception('Derivative values are needed for using the GEKPLS model.')
                self.options['kriging-step'] = 0

            elif self.options['name'] == 'KPLSK':
                self.options['kriging-step'] = 1

            else:
                self.options['kriging-step'] = 0

        if self.dim == self.options['n_comp']:
            if self.options['name'] != 'KRG':
                warnings.warn('Kriging is used instead of the KPLS model!')
                self.options['name'] = 'KRG'

        if len(self.options['theta0']) != self.options['n_comp']:
            raise Exception('Number of principal components must be equal to the number of theta0.')

        if not callable(self.options['corr']):
            if self.options['corr'] in self._correlation_types:
                self.options['corr'] = self._correlation_types[self.options['corr']]

            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.options['corr']))


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
