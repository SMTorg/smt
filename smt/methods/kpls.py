"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)

TODO:
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

from smt.methods.sm import SM
from smt.utils.pairwise import manhattan_distances
from smt.utils.pls import pls as _pls

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


def componentwise_distance(D,corr,n_comp,coeff_pls):

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

    coeff_pls: np.ndarray [dim, n_comp]
            - The PLS-coefficients.

    Returns
    -------

    D_corr: np.ndarray [n_obs * (n_obs - 1) / 2, n_comp]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.

    """
    #Manage the memory.
    limit=int(1e4)

    D_corr = np.zeros((D.shape[0],n_comp))
    i,nb_limit  = 0,int(limit)

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


def compute_pls(X,y,n_comp):

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

    Returns
    -------

    Coeff_pls: np.ndarray[dim, n_comp]
            - The PLS-coefficients.

    """
    nt,dim = X.shape
    pls = _pls(n_comp)

    pls.fit(X,y)
    return np.abs(pls.x_rotations_)

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
    - KPLS
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
                desc='KPLS for Kriging with Partial Least Squares')
        declare('n_comp', 1, types=int, desc='Number of principal components')
        declare('theta0', [1e-2], types=(list, np.ndarray), desc='Initial hyperparameters')
        declare('poly', 'constant', values=('constant', 'linear', 'quadratic'), types=FunctionType,
                desc='regr. term')
        declare('corr', 'squar_exp', values=('abs_exp', 'squar_exp'), types=FunctionType,
                desc='type of corr. func.')
        declare('best_iteration_fail', None)
        declare('nb_ill_matrix', 5)
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

        self.coeff_pls = compute_pls(X.copy(),y.copy(),self.options['n_comp'])

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
            Universal Kriging or for Ordinary Kriging.
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

    def _predict_value(self, x):
        '''
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        '''
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=
                                 False)
        d = componentwise_distance(dx,self.options['corr'].__name__,self.options['n_comp'],
                                   self.coeff_pls)
        # Compute the correlation function
        r = self.options['corr'](self.optimal_theta, d).reshape(n_eval,self.nt)

        y = np.zeros(n_eval)

        # Compute the regression function
        f = self.options['poly'](x)

        # Scaled predictor
        y_ = np.dot(f, self.optimal_par['beta']) + np.dot(r,
                    self.optimal_par['gamma'])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()

        return y

    def _predict_derivative(self, x, kx):
        '''
        Evaluates the derivatives at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        '''
        kx += 1

        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=
                                 False)
        d = componentwise_distance(dx,self.options['corr'].__name__,self.options['n_comp'],
                                   self.coeff_pls)
        # Compute the correlation function
        r = self.options['corr'](self.optimal_theta, d).reshape(n_eval,self.nt)

        if self.options['corr'].__name__ != 'squar_exp':
            raise ValueError(
            'The derivative is only available for square exponential kernel')

        if self.options['poly'].__name__ == 'constant':
            df = np.array([0])
        elif self.options['poly'].__name__ == 'linear':
            df = np.zeros((self.dim + 1, self.dim))
            df[1:,:] = 1
        else:
            raise ValueError(
                'The derivative is only available for ordinary kriging or '+
                'universal kriging using a linear trend')

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par['beta']
        gamma = self.optimal_par['gamma']

        df_dx = np.dot(df.T, beta)
        d_dx=x[:,kx-1].reshape((n_eval,1))-self.X_norma[:,kx-1].reshape((1,self.nt))
        theta = np.sum(self.optimal_theta * self.coeff_pls**2,axis=1)

        return (df_dx[0]-2*theta[kx-1]*np.dot(d_dx*r,gamma))*self.y_std/self.X_std[kx-1]

    def _predict_variance(self, x):
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=
                                 False)
        d = componentwise_distance(dx,self.options['corr'].__name__,self.options['n_comp'],
                                   self.coeff_pls)
        # Compute the correlation function
        r = self.options['corr'](self.optimal_theta, d).reshape(n_eval,self.nt)

        C = self.optimal_par['C']
        rt = linalg.solve_triangular(self.optimal_par['C'], r.T, lower=True)

        u = linalg.solve_triangular(self.optimal_par['G'].T,np.dot(self.optimal_par['Ft'].T, rt) -
                             self.options['poly'](x).T)

        MSE = self.optimal_par['sigma2']*(1.-(rt ** 2.).sum(axis=0)+(u ** 2.).sum(axis=0))
        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        MSE[MSE < 0.] = 0.
        return MSE

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
            return - self._reduced_likelihood_function(theta=10.**log10t)[0]

        limit, _rhobeg = 10*self.options['n_comp'], 0.5

        best_optimal_theta, best_optimal_rlf_value, best_optimal_par, \
                constraints = [], [], [], []

        for i in range(self.options['n_comp']):
            constraints.append(lambda log10t,i=i:log10t[i] - np.log10(1e-6))
            constraints.append(lambda log10t,i=i:np.log10(10) - log10t[i])

        # Compute D which is the componentwise distances between locations
        #  x and x' at which the correlation model should be evaluated.
        self.D = componentwise_distance(D,self.options['corr'].__name__,
                            self.options['n_comp'],self.coeff_pls)

        # Initialization
        k, incr, stop, best_optimal_rlf_value = 0, 0, 1, -1e20
        while (k < stop):
            # Use specified starting point as first guess
            theta0 = self.options['theta0']
            try:
                optimal_theta = 10. ** optimize.fmin_cobyla(
                    minus_reduced_likelihood_function, np.log10(theta0),constraints,
                    rhobeg= _rhobeg, rhoend = 1e-4,iprint=0,maxfun=limit)

                optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                    theta=optimal_theta)

                # Compare the new optimizer to the best previous one
                if k > 0:
                    if np.isinf(optimal_rlf_value):
                        stop += 1
                        if incr != 0:
                            return
                    else:
                        if optimal_rlf_value >= self.options['best_iteration_fail']:
                            if optimal_rlf_value > best_optimal_rlf_value:
                                best_optimal_rlf_value = optimal_rlf_value
                                best_optimal_par = optimal_par
                                best_optimal_theta = optimal_theta
                            else:
                                if self.options['best_iteration_fail'] \
                                   > best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory
                                    best_optimal_rlf_value , best_optimal_par = \
                                        self._reduced_likelihood_function(\
                                          theta= best_optimal_theta)
                else:
                    if np.isinf(optimal_rlf_value):
                        stop += 1
                    else:
                        if optimal_rlf_value >=  self.options['best_iteration_fail']:
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
                    if (self.options['best_iteration_fail'] is not None):
                        if self.options['best_iteration_fail'] > \
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
                                 "%s was given." % (self._regression_types.keys(),
                                self.options['poly']))

        if len(self.options['theta0']) != self.options['n_comp']:
            raise Exception('Number of principal components must be equal to the number of theta0.')

        if not callable(self.options['corr']):
            if self.options['corr'] in self._correlation_types:
                self.options['corr'] = self._correlation_types[self.options['corr']]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."% (self._correlation_types.keys(),
                                self.options['corr']))


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
