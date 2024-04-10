"""
Author: Hugo Reimeringer <@>

This package is distributed under New BSD license.
"""

#------------------------------------------Imports------------------------------------------
from sklearn.decomposition import PCA
from typing import Optional
import numpy as np


from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.surrogate_models import KRG

import warnings


class PODGP(SurrogateModel):
    """
    Class for Proper Orthogonal Decomposition and Gaussian Processes (POD+GP) based surrogate model.

    Attributes
    ----------
    

    Examples
    --------
    >>> from smt.surrogate_models import PODGP
    >>> sm = PODGP(print_training=False)
    >>> sm.options['print_prediction'] = False
    """
    
    name = "POD+GP"

    def _initialize(self):
        """
        Description
        """
        super()._initialize()
        supports = self.supports
        
        supports["variances"] = True
        supports["derivatives"] = True
    
    @staticmethod
    def choice_n_mods_tol(EV_list: np.ndarray, tol: float) -> (int, float) :
        """
        Calculates the required number of kept mods to explain the wanted ratio of variance.

        Parameters
        ----------
        EV_list : np.ndarray
            Description
        tol : float
            Description

        Returns
        -------
        n_mods : int
            Description
        ev_ratio : float
            Description
        """
        
        sum_tot = sum(EV_list)
        sum_ = 0
        for i in range(len(EV_list)):
            sum_ += EV_list[i]
            EV_ratio = sum_/sum_tot
            if sum_/sum_tot >= tol:
                return i+1, EV_ratio

    def POD(self, database = np.array([]), tol = None, n_mods = None, random_state = None) -> None:
        #kwargs ou argument normaux ?
        """
        Performs the POD 

        Parameters
        ----------
        database : np.ndarray[ny, nt]
            Description
        tol : float
            Wanted tolerance for the pod (if n_mods not set)
        n_mods : int
            Wanted number of kept mod for the pod (if tol not set)
        random_state : int
            Description
        
        Examples
        --------
        >>> from smt.surrogate_models import PODGP
        >>> sm = PODGP(print_training=False)
        """
        choice_svd = None

        svd = PCA(svd_solver = 'randomized', random_state = random_state)
        
        if n_mods != None:
            self.n_mods = n_mods
            choice_svd = "mod"
            
        if tol != None:
            if choice_svd != None:
                raise ValueError(
                    "pod can't use both arguments 'n_mods' and 'tol' at the same time"
                )
            else:
                choice_svd = "tol"
                
        if choice_svd == None:
            raise ValueError(
                "either one of the arguments 'n_mods' and 'tol' must be specified"
            )
            
        if database.size == 0:
            raise ValueError(
                "'database' argument must be specified"
            )
        self.n_snapshot = database.shape[1]
        self.ny = database.shape[0]
            
        svd.fit(database.T)
        self.U = svd.components_.T
        self.S = svd.singular_values_
        EV_list = svd.explained_variance_
        
        if choice_svd == "tol":
            self.n_mods, self.EV_ratio = PODGP.choice_n_mods_tol(EV_list, tol)
        elif choice_svd == "n_mods":
            if self.n_mods > self.n_snapshot:
                raise ValueError(
                    "the number of kept mods can't be superior to the number of data values (snapshots)"
                )
            self.EV_ratio = sum(EV_list[:self.n_mods])/sum(EV_list)
            
        self.mean = np.atleast_2d(database.mean(axis=1)).T
        self.basis = np.array(self.U[:, :self.n_mods])
        self.coeff = np.dot(self.basis.T, database - self.mean).T
        self.pod_done = True
        self.training_values_set = False
        
        self.sm_list = []
        for i in range(self.n_mods):
            self.sm_list.append(KRG(print_global = False))
        
    def get_left_basis(self) -> np.ndarray :
        """
        Getter for the left_basis of the POD.

        Returns
        -------
        left_basis : np.ndarray
            Description
        """
        return self.U
    
    def get_singular_values(self) -> np.ndarray :
        """
        Getter for the singular values from the Sigma matrix of the POD.

        Returns
        -------
        singular_values : np.ndarray
            Description
        """
        return self.S

    def get_ev_ratio(self) -> float :
        """
        Getter for the explained variance ratio with the kept mods.

        Returns
        -------
        ev_ratio : float
            Description
        """
        return self.EV_ratio
    
    def get_n_mods(self) -> int :
        """
        Getter for the number of mods kept during the POD.

        Returns
        -------
        n_mods : int
            Description
        """
        return self.n_mods
                
    def set_GP_options(self, GP_options_list = [{}]) -> None :
        """
        Set the options for the GP surrogate models used.

        Parameters
        ----------
        GP_options_list : list[dict]
            Optional Parameter.
            List containing dictionnaries for the options. The k-th dictionnary corresponds to the options of the k-th GP model.
            If the options are commun to all the surogate models, a single dictionnary can be used in the list.
            The available options are the same as the kriging one's.
        
        Example
        --------
        >>> dict1 = {'corr' : 'matern52', 'theta0' : [1e-2]}
        >>> dict2 = {'poly' : 'quadratic'}
        >>> GP_options_list = [dict1, dict2]
        >>> sm.set_GP_options(GP_options_list)
        """
        
        if not(self.pod_done):
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying the setting the GP options."    
            )
        if len(GP_options_list) == 1:
            mod_options = "global"
        elif len(GP_options_list) != self.n_mods:
            raise ValueError(
                f"expected GP_options_list of size n_mods = {self.n_mods}, but got {len(GP_options_list)} instead."
            )
        else:
            mod_options = "local"
        
        for i in range(self.n_mods):
            if mod_options == 'local':
                index = i
            elif mod_options == 'global':
                index = 0
            for key in GP_options_list[index].keys():
                self.sm_list[i].options[key] = GP_options_list[index][key]    
            
    def set_training_values(self, xt, name = None) -> None : #fonction personnalisée ?
        """
        Set training data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        
        xt = xt.T
        if not(self.pod_done):
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying the 'GP' method"    
            )
        self.nt = xt.shape[1]
        self.dim_snapshot = xt.shape[0]
        self.nx = xt.shape[0]
        if self.nt != self.n_snapshot:
            raise ValueError(
                f"there must be the same amount of train values than data values (snapshots), {self.nt} != {self.n_snapshot}"    
            )
        
        for i in range(self.n_mods):
            self.sm_list[i].set_training_values(xt.T, self.coeff[:, i])
            
        self.training_values_set = True
        self.train_done = False
    
    def train(self) -> None :
        """
        Performs the training of the model. 
        """
        
        if not self.training_values_set:
            raise RuntimeError(
                "the training values should have been set before trying to train the model"    
            )
       
        for i in range(self.n_mods):
            self.sm_list[i].train()
        self.train_done = True
        
    def get_gp_coef(self) -> np.ndarray:
        """
        Getter for the list of the GP surrogate models used 

        Returns
        -------
        sm_list : np.ndarray
            Description
        """
        
        return self.sm_list
    
    def _predict_values(self, xn) -> np.ndarray:
        """
        Predict the output values at a set of points.

        Parameters
        ----------
        xn : np.ndarray
            Input values for the prediction points.

        Returns
        -------
        yn : np.ndarray
            Output values at the prediction points.
        """
        
        xn = xn.T
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"    
            )
        
        self.dim_new = xn.shape[0]
        
        if self.dim_new != self.nx:
            raise ValueError(
                f"the data values (snapshots) and the new values where to make a prediction must be the same size, {self.dim_new} != {self.nx}"    
            )
        
        self.n_new = xn.shape[1]
        mean_coeff_gp = np.zeros((self.n_new, self.n_mods))
        
        for i in range(self.n_mods):
            mu_i = self.sm_list[i].predict_values(xn.T)
            mean_coeff_gp[:,i] = mu_i[:,0]
        
        mean_x_new = self.mean + np.dot(mean_coeff_gp, self.basis.T).T
        
        return mean_x_new
    
    def _predict_variances(self, xn) -> np.ndarray :
        """
        Predict the output values at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.

        Returns
        -------
        y : np.ndarray[nt, ny]
            Output values at the prediction points.
        """
        
        xn = xn.T
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"    
            )
        
        dim_new = xn.shape[0]
        
        if dim_new != self.nx:
            raise ValueError(
                f"the data values (snapshots) and the new values where to make a prediction must be the same size, {self.dim_new} != {self.nx}"    
            )
        
        self.n_new = xn.shape[1]
        var_coeff_gp = np.zeros((self.n_new, self.n_mods))
        
        for i in range(self.n_mods):
            sigma_i_square = self.sm_list[i].predict_variances(xn.T)
            var_coeff_gp[:,i] = sigma_i_square[:,0]
        
        var_x_new = np.dot(var_coeff_gp, (self.basis**2).T).T
        
        return var_x_new
    
    def _predict_derivatives(self, xn, kx):
        xn = xn.T
        d = kx
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"    
            )
        
        dim_new = xn.shape[0]
        
        if dim_new != self.nx:
            raise ValueError(
                f"the data values (snapshots) and the new values where to make a prediction must be the same size, {self.dim_new} != {self.nx}"    
            )
        
        if kx >= self.nx:
            raise ValueError(
                "the number of the desired derivatives must correspond to an existing dimension of the data"    
            )
        
        n_new = xn.shape[1]
        deriv_coeff_gp = np.zeros((n_new, self.n_mods))
        
        for i in range(self.n_mods):
            deriv_coeff_gp[:,i] = self.sm_list[i].predict_derivatives(xn.T, d)[:,0]
        
        deriv_x_new = np.dot(deriv_coeff_gp, self.basis.T).T
        
        return deriv_x_new