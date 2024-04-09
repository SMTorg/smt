"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from typing import Optional
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.surrogate_models import KRG
from sklearn.decomposition import PCA
import numpy as np
import warnings


class PODGP(SurrogateModel):
    name = "POD+GP"

    def _initialize(self):
        super()._initialize()
        self.random_state = None
    
    def set_up_dico(self):
        
        return {}
        
    
    def _predict_derivatives(x, kx):
        return None
        
    def _predict_output_derivatives(x):
        return None
        
    def _predict_values(x):
        return None
    
    def _predict_variance_derivatives(x, kx):
        return None	
    
    def _predict_variances(x):
        return None
    
    def _set_training_derivatives(self, xt: np.ndarray, dyt_dxt: np.ndarray, kx: int, name: Optional[str] = None):
        return None
    
    
    
    def _train():
        return None
    
    def choice_n_mods_tol(self, EV_list):
        tol = self.tol
        
        sum_tot = sum(EV_list)
        sum_ = 0
        for i in range(len(EV_list)):
            sum_ += EV_list[i]
            EV_ratio = sum_/sum_tot
            if sum_/sum_tot >= tol:
                return i+1, EV_ratio

    def POD(self, **kwargs):
        dico = kwargs
        choice_svd = None
        
        if "random_state" in dico.keys():
            self.random_state = dico["random_state"]
        self.svd = PCA(svd_solver = 'randomized', random_state = self.random_state)
        
        if "n_mods" in dico.keys():
            self.n_mods = dico["n_mods"]
            choice_svd = "mod"
            
        if "tol" in dico.keys():
            if choice_svd != None:
                raise ValueError(
                    "pod can't use both arguments 'n_mods' and 'tol' at the same time"
                )
            else:
                self.tol = dico["tol"]
                choice_svd = "tol"
                
        if choice_svd == None:
            raise ValueError(
                "either one of the arguments 'n_mods' and 'tol' must be specified"
            )
            
        if "database" not in dico.keys():
            raise ValueError(
                "'database' argument must be specified"
            )
        self.database = dico["database"]
        self.n_snapshot = self.database.shape[1]
        self.dim_snapshot = self.database.shape[0]
            
        self.svd.fit(self.database.T)
        self.U = self.svd.components_.T
        self.S = self.svd.singular_values_
        EV_list = self.svd.explained_variance_
        
        if choice_svd == "tol":
            self.n_mods, self.EV_ratio = self.choice_n_mods_tol(EV_list)
        elif choice_svd == "n_mods":
            if self.n_mods > self.n_snapshot:
                raise ValueError(
                    "the number of kept mods can't be superior to the number of data values (snapshots)"
                )
            self.EV_ratio = sum(EV_list[:self.n_mods])/sum(EV_list)
            
        self.mean = np.atleast_2d(self.database.mean(axis=1)).T
        self.basis = np.array(self.U[:, :self.n_mods])
        self.coeff = np.dot(self.basis.T, self.database - self.mean).T
        self.pod_done = True
        self.training_values_set = False
        
        self.sm_list = []
        for i in range(self.n_mods):
            self.sm_list.append(KRG(print_global = False))
        
    def get_left_basis(self):
        return self.U
    
    def get_singular_values(self):
        return self.S

    def get_ev_ratio(self):
        return self.EV_ratio
    
    def get_n_mods(self):
        return self.n_mods
                
    def set_GP_options(self, GP_options_list):
        
        if not(self.pod_done):
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying the 'GP' method"    
            )
        if len(GP_options_list) == 1:
            mod_options = "global"
        elif len(GP_options_list) != self.n_mods:
            raise ValueError(
                f"expected GP_options_list of size n_mods = {self.n_mods}, but got {len(GP_options_list)} instead"
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
            
    def set_training_values(self, xt, name=None):
        if not(self.pod_done):
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying the 'GP' method"    
            )
        self.n_train = xt.shape[1]
        if self.n_train != self.n_snapshot:
            raise ValueError(
                f"there must be the same amount of train values than data values (snapshots), {self.n_train} != {self.n_snapshot}"    
            )
        
        for i in range(self.n_mods):
            self.sm_list[i].set_training_values(xt.T, self.coeff[:, i])
            
        self.training_values_set = True
        self.train_done = False
    
    def train(self):
        if not self.training_values_set:
            raise RuntimeError(
                "the training values should have been set before trying to train the model"    
            )
       
        for i in range(self.n_mods):
            self.sm_list[i].train()
        self.train_done = True
    
    def predict_values(self, xn):
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"    
            )
        
        self.dim_new = xn.shape[0]
        
        if self.dim_new != self.dim_snapshot:
            raise ValueError(
                f"the data values (snapshots) and the new values where to make a prediction must be the same size, {self.dim_new} != {self.dim_snapshot}"    
            )
        
        self.n_new = xn.shape[1]
        mean_coeff_gp = np.zeros((self.n_new, self.n_mods))
        
        for i in range(self.n_mods):
            mu_i = self.sm_list[i].predict_values(xn.T)
            mean_coeff_gp[:,i] = mu_i[:,0]
        
        mean_x_new = self.mean + np.dot(mean_coeff_gp, self.basis.T).T
        
        return mean_x_new
    
    def predict_variances(self, xn):
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"    
            )
        
        self.dim_new = xn.shape[0]
        
        if self.dim_new != self.dim_snapshot:
            raise ValueError(
                f"the data values (snapshots) and the new values where to make a prediction must be the same size, {self.dim_new} != {self.dim_snapshot}"    
            )
        
        self.n_new = xn.shape[1]
        mean_coeff_gp = np.zeros((self.n_new, self.n_mods))
        
        for i in range(self.n_mods):
            mu_i = self.sm_list[i].predict_values(xn.T)
            mean_coeff_gp[:,i] = mu_i[:,0]
        
        mean_x_new = self.mean + np.dot(mean_coeff_gp, self.basis.T).T
        
        return mean_x_new