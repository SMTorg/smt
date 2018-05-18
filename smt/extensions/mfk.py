# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>

Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of 
order 1 (AR1)
"""

from __future__ import division
import numpy as np
from smt.surrogate_models.krg_based import KrgBased
from types import FunctionType
from smt.utils.kriging_utils import l1_cross_distances, componentwise_distance
from scipy.linalg import solve_triangular
from scipy import linalg
from sklearn.metrics.pairwise import manhattan_distances

"""
The MFK class.
"""


class MFK(KrgBased):

    """
    - MFK
    """
 
    def _initialize(self):
        super(MFK, self)._initialize()
        declare = self.options.declare
        
        
        declare('rho_regr', 'constant',types=FunctionType,\
                values=('constant', 'linear', 'quadratic'), desc='regr. term')
        declare('theta0', None, types=(np.ndarray),\
                desc='Initial hyperparameters')
        self.name = 'MFK'
    

   

    def _check_list_structure(self, X, y):

        

        if type(X) is not list:
            nlevel = 1
            X = [X]
        else:
            nlevel = len(X)


        if type(y) is not list:
            y = [y]

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")


        n_samples = np.zeros(nlevel, dtype = int)
        n_features = np.zeros(nlevel, dtype = int)
        n_samples_y = np.zeros(nlevel, dtype = int)
        for i in range(nlevel):
            n_samples[i], n_features[i] = X[i].shape
            if i>1 and n_features[i] != n_features[i-1]:
                raise ValueError("All X must have the same number of columns.")
            y[i] = np.asarray(y[i]).ravel()[:, np.newaxis]
            n_samples_y[i] = y[i].shape[0]
            if n_samples[i] != n_samples_y[i]:
                raise ValueError("X and y must have the same number of rows.")


        self.nx = n_features[0]
        self.nt_all = n_samples
        self.nlvl = nlevel
        self.ny = y[0].shape[1]
        self.X = X[:]
        self.y = y[:]
        

        return
    
    def _new_train(self):
        """
        Overrides KrgBased implementation
        Train the model
        """
        
        xt =[]
        yt = []
        i=0
        while(self.training_points.get(i, None) is not None):
            xt.append(self.training_points[i][0][0])
            yt.append(self.training_points[i][0][1])
            i = i+1
        xt.append(self.training_points[None][0][0])
        yt.append(self.training_points[None][0][1])
            
        
        self._check_list_structure(xt, yt)
        self._check_param()
        X = self.X
        y = self.y
        ## TODO : normilze
        self.X_mean, self.y_mean, self.X_std, \
            self.y_std = np.array([0]), np.array([0]), np.array([1]), np.array([1])
        nlevel = self.nlvl
        n_samples = self.nt_all

        # initialize lists
        
        self.D_all = nlevel*[0]
        self.F_all = nlevel*[0]
        self.p_all = nlevel*[0]
        self.q_all = nlevel*[0]
        self.optimal_rlf_value = nlevel*[0]
        self.optimal_par = nlevel*[{}]
        self.optimal_theta = nlevel*[0]


        for lvl in range(nlevel):

            # Calculate matrix of distances D between samples
            self.D_all[lvl] = l1_cross_distances(X[lvl])
            

            # Regression matrix and parameters
            self.F_all[lvl] = self.options['poly'](X[lvl])
            self.p_all[lvl] = self.F_all[lvl].shape[1]

            # Concatenate the autoregressive part for levels > 0
            if lvl > 0:
                F_rho = self.options['rho_regr'](X[lvl])
                self.q_all[lvl] = F_rho.shape[1]
                self.F_all[lvl] = np.hstack((F_rho*np.dot((self.y[lvl-1])[-n_samples[lvl]:],
                                              np.ones((1,self.q_all[lvl]))), self.F_all[lvl]))
            else:
                self.q_all[lvl] = 0

            n_samples_F_i = self.F_all[lvl].shape[0]

            if n_samples_F_i != n_samples[lvl]:
                raise Exception("Number of rows in F and X do not match. Most "
                                "likely something is going wrong with the "
                                "regression model.")

            if int(self.p_all[lvl] + self.q_all[lvl]) >= n_samples_F_i:
                raise Exception(("Ordinary least squares problem is undetermined "
                                 "n_samples=%d must be greater than the regression"
                                 " model size p+q=%d.")
                                 % (n_samples[i], self.p_all[lvl]+self.q_all[lvl]))

       
        for lvl in range(nlevel):
            # Determine Gaussian Process model parameters
            self.F = self.F_all[lvl]
            D, self.ij = self.D_all[lvl]
#            D = self.D_all[lvl]
            self.nt = self.nt_all[lvl]
            self.y_norma = self.y[lvl]
            self.X_norma = self.X[lvl]
            self.q = self.q_all[lvl]
            self.p = self.p_all[lvl]
            self.optimal_rlf_value[lvl], self.optimal_par[lvl], self.optimal_theta[lvl] = \
                self._optimize_hyperparam(D)
            
            del self.y_norma, self.D

    def _componentwise_distance(self,dx,opt=0):
        d = componentwise_distance(dx,self.options['corr'].__name__,
                                   self.nx)
        return d

    def _predict_values(self, X):
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
        
        # Initialization X = atleast_2d(X)
        nlevel = self.nlvl
        n_eval, n_features_X = X.shape
#        if n_features_X != self.n_features:
#            raise ValueError("Design must be an array of n_features columns.")

        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, nlevel))
#        if self.normalize:
#            X = (X - self.X_mean) / self.X_std
##                X = (X - self.X_mean[0]) / self.X_std[0]
        f = self.options['poly'](X)
        f0 = self.options['poly'](X)
        dx = manhattan_distances(X, Y=self.X[0], sum_over_features=False)

        # Get regression function and correlation
        F = self.F_all[0]
        C = self.optimal_par[0]['C']

        beta = self.optimal_par[0]['beta']
        Ft = solve_triangular(C, F, lower=True)
        yt = solve_triangular(C, self.y[0], lower=True)
        r_ = self.options['corr'](self.optimal_theta[0], dx).reshape(n_eval, self.nt_all[0])
        gamma = solve_triangular(C.T, yt - np.dot(Ft,beta), lower=False)

        # Scaled predictor
        mu[:,0]= (np.dot(f, beta) + np.dot(r_,gamma)).ravel()

        # Calculate recursively kriging mean and variance at level i
        for i in range(1,nlevel):
            F = self.F_all[i]
            C = self.optimal_par[i]['C']
            g = self.options['rho_regr'](X)
            dx = manhattan_distances(X, Y=self.X[i], sum_over_features=False)
            r_ = self.options['corr'](self.optimal_theta[i], dx).reshape(n_eval, self.nt_all[i])
            f = np.vstack((g.T*mu[:,i-1], f0.T))

            Ft = solve_triangular(C, F, lower=True)
            #TODO: consider different regressions?
            yt = solve_triangular(C, self.y[i], lower=True)
            r_t = solve_triangular(C,r_.T, lower=True)
            beta = self.optimal_par[i]['beta']

            # scaled predictor
            mu[:,i] = (np.dot(f.T, beta) \
                       + np.dot(r_t.T, yt - np.dot(Ft,beta))).ravel()

        # scaled predictor
        for i in range(nlevel):# Predictor
            mu[:,i] = self.y_mean + self.y_std * mu[:,i]
           
        self.mu_all = mu
        return mu[:,-1].reshape((n_eval,1))
        


    def _predict_variances(self, X):
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
        # Initialization X = atleast_2d(X)
        nlevel = self.nlvl
        n_eval, n_features_X = X.shape
#        if n_features_X != self.n_features:
#            raise ValueError("Design must be an array of n_features columns.")

        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, nlevel))
#        if self.normalize:
#            X = (X - self.X_mean) / self.X_std
##                X = (X - self.X_mean[0]) / self.X_std[0]
        f = self.options['poly'](X)
        f0 = self.options['poly'](X)
        dx = manhattan_distances(X, Y=self.X[0], sum_over_features=False)

        # Get regression function and correlation
        F = self.F_all[0]
        C = self.optimal_par[0]['C']

        beta = self.optimal_par[0]['beta']
        Ft = solve_triangular(C, F, lower=True)
        yt = solve_triangular(C, self.y[0], lower=True)
        r_ = self.options['corr'](self.optimal_theta[0], dx).reshape(n_eval, self.nt_all[0])
        gamma = solve_triangular(C.T, yt - np.dot(Ft,beta), lower=False)

        # Scaled predictor
        mu[:,0]= (np.dot(f, beta) + np.dot(r_,gamma)).ravel()

    
        self.sigma2_rho = nlevel*[None]
        MSE = np.zeros((n_eval,nlevel))
        r_t = solve_triangular(C, r_.T, lower=True)
        G = self.optimal_par[0]['G']

        u_ = solve_triangular(G.T, f.T - np.dot(Ft.T, r_t), lower=True)
        MSE[:,0] = self.optimal_par[0]['sigma2'] * (1 \
                            - (r_t**2).sum(axis=0) + (u_**2).sum(axis=0))

        # Calculate recursively kriging mean and variance at level i
        for i in range(1,nlevel):
            F = self.F_all[i]
            C = self.optimal_par[i]['C']
            g = self.options['rho_regr'](X)
            dx = manhattan_distances(X, Y=self.X[i], sum_over_features=False)
            r_ = self.options['corr'](self.optimal_theta[i], dx).reshape(n_eval, self.nt_all[i])
            f = np.vstack((g.T*mu[:,i-1], f0.T))

            Ft = solve_triangular(C, F, lower=True)
            #TODO: consider different regressions?
            yt = solve_triangular(C, self.y[i], lower=True)
            r_t = solve_triangular(C,r_.T, lower=True)
            G = self.optimal_par[i]['G']
            beta = self.optimal_par[i]['beta']

            # scaled predictor
            

        
            sigma2 = self.optimal_par[i]['sigma2'] 
            q = self.q_all[i]
            p = self.p_all[i]
            Q_ = (np.dot((yt-np.dot(Ft,beta)).T, yt-np.dot(Ft,beta)))[0,0]
            u_ = solve_triangular(G.T, f - np.dot(Ft.T, r_t), lower=True)
            sigma2_rho = np.dot(g, \
                sigma2*linalg.inv(np.dot(G.T,G))[:q,:q] \
                    + np.dot(beta[:q], beta[:q].T))
            sigma2_rho = (sigma2_rho * g).sum(axis=1)
            MSE[:,i] = sigma2_rho * MSE[:,i-1] \
                            + Q_/(2*(self.nt_all[i]-p-q)) \
                            * (1 - (r_t**2).sum(axis=0)) \
                            + sigma2 * (u_**2).sum(axis=0)

                

        # scaled predictor
        for i in range(nlevel):# Predictor
            MSE[:,i] = self.y_std**2 * MSE[:,i]
      
        self.MSE_all = MSE
        return MSE[:,-1].reshape((n_eval,1))
        