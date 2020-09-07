"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)

This package is distributed under New BSD license.

TODO:
- fail_iteration and nb_iter_max to remove from options
- define outputs['sol'] = self.sol

"""

from __future__ import division

# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
from scipy import linalg, optimize
from types import FunctionType
from smt.utils.caching import cached_operation

from smt.surrogate_models.surrogate_model import SurrogateModel
from sklearn.metrics.pairwise import manhattan_distances
from smt.utils.kriging_utils import constant, linear, quadratic
from smt.utils.kriging_utils import (
    squar_exp,
    abs_exp,
    standardization,
    l1_cross_distances,
)

from scipy.optimize import minimize


class KrgBased(SurrogateModel):

    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _correlation_types = {"abs_exp": abs_exp, "squar_exp": squar_exp}

    def _initialize(self):
        super(KrgBased, self)._initialize()
        declare = self.options.declare
        supports = self.supports
        declare(
            "poly",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type",
        )
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp"),
            desc="Correlation function type",
        )
        declare(
            "data_dir",
            types=str,
            desc="Directory for loading / saving cached data; None means do not save or load",
        )
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        declare(
            "vartype", types=list , desc="For mixed integer : variables types between continuous: \"cont\", integer: \"int\", and categorial with n levels: (\"cate\",n) "
        )

        self.name = "KrigingBased"
        self.best_iteration_fail = None
        self.nb_ill_matrix = 5
        supports["derivatives"] = True
        supports["variances"] = True

    ############################################################################
    # Model functions
    ############################################################################

    def _new_train(self):

        """
        Train the model
        """
        self._check_param()

        # Sampling points X and y
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]

        # Compute PLS-coefficients (attr of self) and modified X and y (if GEKPLS is used)
        if self.name != "Kriging":
            X, y = self._compute_pls(X.copy(), y.copy())

        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_mean,
            self.y_mean,
            self.X_std,
            self.y_std,
        ) = standardization(X, y)

        # Calculate matrix of distances D between samples
        D, self.ij = l1_cross_distances(self.X_norma)
        ###
        if np.min(np.sum(D, axis=1)) == 0.0 and self.options["vartype"] is None:
            raise Exception("Multiple input features cannot have the same value.")
        ####
        # Regression matrix and parameters
        self.F = self._regression_types[self.options["poly"]](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F, p)

        # Optimization
        (
            self.optimal_rlf_value,
            self.optimal_par,
            self.optimal_theta,
        ) = self._optimize_hyperparam(D)
        if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
            if self.options["eval_noise"]:
                self.optimal_theta = self.optimal_theta[:-1]
        del self.y_norma, self.D

    def _train(self):
        """
        Train the model
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
        reduced_likelihood_function_value = -np.inf
        par = {}
        # Set up R
        MACHINE_EPSILON = np.finfo(np.double).eps
        nugget = 10.0 * MACHINE_EPSILON
        if self.name == "MFK":
            if self._lvl != self.nlvl:
                # in the case of multi-fidelity optimization
                # it is very probable that lower-fidelity correlation matrix
                # becomes ill-conditionned
                nugget = 10.0 * nugget
        noise = 0.0
        tmp_var = theta
        if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
            if self.options["eval_noise"]:
                theta = tmp_var[:-1]
                noise = tmp_var[-1]

        r = self._correlation_types[self.options["corr"]](theta, self.D).reshape(-1, 1)

        R = np.eye(self.nt) * (1.0 + nugget + noise)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            return reduced_likelihood_function_value, par

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )

            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par

        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / self.nt)).prod()

        # Compute/Organize output
        if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
            n_samples = self.nt
            p = self.p
            q = self.q
            sigma2 = (rho ** 2.0).sum(axis=0) / (n_samples - p - q)
            reduced_likelihood_function_value = -(n_samples - p - q) * np.log10(
                sigma2
            ) - n_samples * np.log10(detR)
        else:
            sigma2 = (rho ** 2.0).sum(axis=0) / (self.nt)
            reduced_likelihood_function_value = -sigma2.sum() * detR
        par["sigma2"] = sigma2 * self.y_std ** 2.0
        par["beta"] = beta
        par["gamma"] = linalg.solve_triangular(C.T, rho)
        par["C"] = C
        par["Ft"] = Ft
        par["G"] = G

        # A particular case when f_min_cobyla fail
        if (self.best_iteration_fail is not None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):

            if reduced_likelihood_function_value > self.best_iteration_fail:
                self.best_iteration_fail = reduced_likelihood_function_value
                self._thetaMemory = np.array(tmp_var)

        elif (self.best_iteration_fail is None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):
            self.best_iteration_fail = reduced_likelihood_function_value
            self._thetaMemory = np.array(tmp_var)

        return reduced_likelihood_function_value, par

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        if not (self.options["vartype"] is None):
            x = self._project_values(x)

        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)
        y = np.zeros(n_eval)
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](x)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(r, self.optimal_par["gamma"])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()

        return y

    def _predict_derivatives(self, x, kx):
        """
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
        """
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)

        if self.options["corr"] != "squar_exp":
            raise ValueError(
                "The derivative is only available for square exponential kernel"
            )
        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par["beta"]
        gamma = self.optimal_par["gamma"]
        df_dx = np.dot(df.T, beta)
        d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma[:, kx].reshape((1, self.nt))
        if self.name != "Kriging" and "KPLSK" not in self.name:
            theta = np.sum(self.optimal_theta * self.coeff_pls ** 2, axis=1)
        else:
            theta = self.optimal_theta
        y = (
            (df_dx[kx] - 2 * theta[kx] * np.dot(d_dx * r, gamma))
            * self.y_std
            / self.X_std[kx]
        )
        return y

    def _predict_variances(self, x):

        # Initialization
        if not (self.options["vartype"] is None):
            x = self._project_values(x)

        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)

        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)

        C = self.optimal_par["C"]
        rt = linalg.solve_triangular(C, r.T, lower=True)

        u = linalg.solve_triangular(
            self.optimal_par["G"].T,
            np.dot(self.optimal_par["Ft"].T, rt)
            - self._regression_types[self.options["poly"]](x).T,
        )

        A = self.optimal_par["sigma2"]
        B = 1.0 - (rt ** 2.0).sum(axis=0) + (u ** 2.0).sum(axis=0)
        MSE = np.einsum("i,j -> ji", A, B)

        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        MSE[MSE < 0.0] = 0.0
        return MSE

    def _project_values(self, x):
        """
        This function project continuously relaxed values 
        to their closer assessable values.
        --------
        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Continuous evaluation point input variable values 
      
        Returns
        -------
        y : np.ndarray
            Feasible evaluation point input variable values.
        """
        if self.options["vartype"] is None:
            return x
        
        if type(self.options["vartype"]) is list and not hasattr(self, 'vartype'):
            dim = np.shape(x)[1]
            self.vartype = self._transform_vartype(dim)      
        vartype = self.vartype 
        for j in range(0, np.shape(x)[0]):
            i = 0
            while i < np.shape(x[j])[0]:
                if i < np.shape(x[j])[0] and vartype[i] == 0:
                    i = i + 1
                    ##Continuous : Do nothing
                elif i < np.shape(x[j])[0] and vartype[i] == 1:
                    x[j][i] = np.round(x[j][i])
                    i = i + 1
                    ##Integer : Round
                elif i < np.shape(x[j])[0] and vartype[i] > 1:
                    k = []
                    i0 = i
                    ind = vartype[i]
                    while (i < np.shape(x[j])[0]) and (vartype[i] == ind):
                        k.append(x[j][i])
                        i = i + 1
                    y = np.zeros(np.shape(k))
                    y[np.argmax(k)] = 1
                    x[j][i0:i] = y
                    ##Categorial : The biggest level is selected.
        return x

    def _transform_vartype(self, dim):
        """
        This function unfold vartype list to a coded array with
        0 for continuous variables, 1 for integers and n>1 for each 
        level of the n-th categorical variable.
        Each level correspond to a new continuous dimension.
        
        --------
        Arguments
        ---------
        dim : int
            The number of dimension
            
        Returns
        -------
        vartype : np.ndarray
            The type of the each dimension. 
        """
        vartype = self.options["vartype"]
        if vartype is None:
            vartype = np.zeros(dim)
        if isinstance(vartype, list):
            temp = []
            ind_cate = 2
            for i in vartype:
                if i == "cont":
                    temp.append(0)
                    ##new continuous dimension : append 0
                elif i == "int":
                    temp.append(1)
                    ##new integer dimension : append 1
                elif i[0] == "cate":
                    for j in range(i[1]):
                        temp.append(ind_cate)
                    ##For each level
                    ##new categorical dimension : append n
                    ind_cate = ind_cate + 1
                else:
                    raise Exception ("type_error")
            temp = np.array(temp)
            vartype = temp
        ## Assign 0 to continuous variables, 1 for int and n>1 for each
        # categorical variable.

        return vartype

    def _relax_limits(self, xlimits,dim=0):
        """
        This function unfold xlimits to add contiuous dimensions
        Each level correspond to a new continuous dimension in [0,1].
        Integer dimensions are relaxed continuously.
        
        --------
        Arguments
        ---------
        xlimits : np.ndarray
        The bounds of the each original dimension and their labels .
        dim : int
        The number of dimension
    
        Returns
        -------
        xlimits : np.ndarray
        The bounds of the each original dimension  (cont, int or cate).
        """
        
        #Continuous optimization : do nothing
        if (self.options["vartype"] is None ):
            return xlimits
        
        xlim=xlimits
        if type(self.options["vartype"]) is list and not hasattr(self, 'vartype'):
            if dim==0 : 
                try: 
                    dim = np.shape(x)[1]
                except NameError :
                    raise Exception ("Missing dimension to unfold xlimits")
            self.vartype = self._transform_vartype(dim)
        vt = self.vartype 
        
        #continuous or integer => no cate
        if (isinstance(xlim[0][0], np.float64)) :
            for ty in vt :
                if not(ty==0 or ty==1) : 
                    raise Exception ("xlimits used an incorrect type")
       #cate not found => error
        elif (isinstance(xlim[0][0], np.str_) or isinstance(xlim[0], list) ) :
            rais=1
            xlim=np.zeros((np.size(vt),2))
            ind_r=0
            ind_o=-1
            tmp=0        
            count=0
            for ty in vt :
                #if cate : add dimensions
                if not(ty==0 or ty==1) : 
                        rais=0
                        xlim[ind_r]=[0,1]
 
                #if (cate,n) we should have n labels
                        if ty==tmp :
                            count=count+1
                        else:
                            ind_o=ind_o+1
                            tmp=ty
                            count=0
                        try : 
                            err=(xlimits[ind_o][count])
                        except:
                            raise Exception ("missing labels in xlimits")                                 
 
                else:
                #if not cate : recopy bounds
                    no_cate=0
                    while no_cate==0:
                        try: 
                            ind_o=ind_o+1
                            xlim[ind_r]=xlimits[ind_o]
                            no_cate=1
                        except:
                             if (ind_o == np.size(vt)+1):
                                    raise Exception ("xlimits used an incorrect type")                                 
                ind_r=ind_r+1        
            if rais==1 :
                raise Exception ("xlimits used an incorrect type")
             
        return(xlim)    
    
    def _assign_labels(self, x, xlimits):
        """
        This function reduce inputs from relaxed space to original space by 
        assigning labels to categorical variables.
                
        --------
        Arguments
        ---------
         x : np.ndarray [n_evals, dim]
        Continuous evaluation point input variable values 
        xlimits : np.ndarray
        The bounds of the each original dimension and their labels .
        
        Returns
        -------
        x_labeled : np.ndarray [n_evals, dim]
        Evaluation point input variable values and corresponding labels
        """
        
        #Continuous optimization : do nothing
        if (self.options["vartype"] is None ):
            return x
        
     
        if type(self.options["vartype"]) is list and not hasattr(self, 'vartype'):
            dim = np.shape(x)[1]
            self.vartype = self._transform_vartype(dim)
        vt = self.vartype 
        
        xlim=xlimits
        x2=np.copy(x)
        nbpt=(np.shape(x)[0])
           
        #continuous or integer => no cate
        if (isinstance(xlim[0][0], np.float64)) :
            for ty in vt :
                if not(ty==0 or ty==1) : 
                    raise Exception ("xlimits used an incorrect type")
     
        #cate => to label
        elif (isinstance(xlim[0][0], np.str_) or isinstance(xlim[0], list) ) :
            dim_out_cate= int(max(0,np.max(vt)-1))
            dim_out= (vt == 0).sum()+(vt == 1).sum()+dim_out_cate
            x2=np.array(np.zeros((nbpt,dim_out)),dtype=np.str_)
            
            for p in range(nbpt):
                j=0
                tmp=0
                cpt=0
                for i in range(np.shape(x)[1]):
                  if vt[i]==0 or vt[i]==1 :
                      x2[p][j]=x[p][i]
                      j=j+1
                  else :
                     tmp2= vt[i]
                     if tmp2 == tmp :
                         tmp=tmp2
                         if x[p][i] > 0.999:
                             x2[p][j]=xlimits[j][cpt]
                             j=j+1
                     else:
                         tmp=tmp2
                         cpt=0
                         if x[p][i] > 0.999:
                             x2[p][j]=xlimits[j][cpt]
                             j=j+1

                     cpt=cpt+1

                         
                    
        
        return(x2)    




    def _optimize_hyperparam(self, D):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
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


        best_optimal_theta: list(n_comp) or list(dim)
            - The best hyperparameters found by the optimization.
        """
        # reinitialize optimization best values
        self.best_iteration_fail = None
        self._thetaMemory = None
        # Initialize the hyperparameter-optimization
        def minus_reduced_likelihood_function(log10t):
            return -self._reduced_likelihood_function(theta=10.0 ** log10t)[0]

        limit, _rhobeg = 10 * len(self.options["theta0"]), 0.5
        exit_function = False
        if "KPLSK" in self.name:
            n_iter = 1
        else:
            n_iter = 0

        for ii in range(n_iter, -1, -1):
            (
                best_optimal_theta,
                best_optimal_rlf_value,
                best_optimal_par,
                constraints,
            ) = (
                [],
                [],
                [],
                [],
            )

            for i in range(len(self.options["theta0"])):
                constraints.append(lambda log10t, i=i: log10t[i] - np.log10(1e-6))
                constraints.append(lambda log10t, i=i: np.log10(100) - log10t[i])

            self.D = self._componentwise_distance(D, opt=ii)

            # Initialization
            k, incr, stop, best_optimal_rlf_value = 0, 0, 1, -1e20
            while k < stop:
                # Use specified starting point as first guess
                theta0 = self.options["theta0"]
                if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
                    if self.options["eval_noise"]:
                        theta0 = np.concatenate(
                            [theta0, np.array([self.options["noise0"]])]
                        )
                        constraints.append(lambda log10t: log10t[-1] + 16)
                        constraints.append(lambda log10t: 10 - log10t[-1])
                try:
                    optimal_theta = 10.0 ** optimize.fmin_cobyla(
                        minus_reduced_likelihood_function,
                        np.log10(theta0),
                        constraints,
                        rhobeg=_rhobeg,
                        rhoend=1e-4,
                        maxfun=limit,
                    )
                    optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                        theta=optimal_theta
                    )
                    # Compare the new optimizer to the best previous one
                    if k > 0:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                            if incr != 0:
                                return
                        else:
                            if optimal_rlf_value >= self.best_iteration_fail:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta
                                else:
                                    if (
                                        self.best_iteration_fail
                                        > best_optimal_rlf_value
                                    ):
                                        best_optimal_theta = self._thetaMemory
                                        (
                                            best_optimal_rlf_value,
                                            best_optimal_par,
                                        ) = self._reduced_likelihood_function(
                                            theta=best_optimal_theta
                                        )
                    else:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                        else:
                            if optimal_rlf_value >= self.best_iteration_fail:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta

                            else:
                                if self.best_iteration_fail > best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory.copy()
                                    (
                                        best_optimal_rlf_value,
                                        best_optimal_par,
                                    ) = self._reduced_likelihood_function(
                                        theta=best_optimal_theta
                                    )
                    k += 1
                except ValueError as ve:
                    # If iteration is max when fmin_cobyla fail is not reached
                    if self.nb_ill_matrix > 0:
                        self.nb_ill_matrix -= 1
                        k += 1
                        stop += 1
                        # One evaluation objectif function is done at least
                        if self.best_iteration_fail is not None:
                            if self.best_iteration_fail > best_optimal_rlf_value:
                                best_optimal_theta = self._thetaMemory
                                (
                                    best_optimal_rlf_value,
                                    best_optimal_par,
                                ) = self._reduced_likelihood_function(
                                    theta=best_optimal_theta
                                )
                    # Optimization fail
                    elif best_optimal_par == []:
                        print("Optimization failed. Try increasing the ``nugget``")
                        raise ve
                    # Break the while loop
                    else:
                        k = stop + 1
                        print("fmin_cobyla failed but the best value is retained")

            if "KPLSK" in self.name:
                if self.name == "MFKPLSK" and self.options["eval_noise"]:
                    # best_optimal_theta contains [theta, noise] if eval_noise = True
                    theta = best_optimal_theta[:-1]
                else:
                    # best_optimal_theta contains [theta] if eval_noise = False
                    theta = best_optimal_theta

                if exit_function:
                    return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

                if self.options["corr"] == "squar_exp":
                    self.options["theta0"] = (theta * self.coeff_pls ** 2).sum(1)
                else:
                    self.options["theta0"] = (theta * np.abs(self.coeff_pls)).sum(1)

                self.options["n_comp"] = int(self.nx)
                limit = 10 * self.options["n_comp"]
                self.best_iteration_fail = None
                exit_function = True

        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

    def _check_param(self):
        """
        This function check some parameters of the model.
        """

        # FIXME: _check_param should be overriden in corresponding subclasses
        if self.name in ["KPLS", "KPLSK", "GEKPLS", "MFKPLS", "MFKPLSK"]:

            d = self.options["n_comp"]
        else:
            d = self.nx

        if len(self.options["theta0"]) != d:
            if len(self.options["theta0"]) == 1:
                self.options["theta0"] *= np.ones(d)
            else:
                raise ValueError(
                    "the number of dim %s should be equal to the length of theta0 %s."
                    % (d, len(self.options["theta0"]))
                )

        if self.supports["training_derivatives"]:
            if not (1 in self.training_points[None]):
                raise Exception(
                    "Derivative values are needed for using the GEKPLS model."
                )

    def _check_F(self, n_samples_F, p):
        """
        This function check the F-parameters of the model.
        """

        if n_samples_F != self.nt:
            raise Exception(
                "Number of rows in F and X do not match. Most "
                "likely something is going wrong with the "
                "regression model."
            )
        if p > n_samples_F:
            raise Exception(
                (
                    "Ordinary least squares problem is undetermined "
                    "n_samples=%d must be greater than the "
                    "regression model size p=%d."
                )
                % (self.nt, p)
            )
