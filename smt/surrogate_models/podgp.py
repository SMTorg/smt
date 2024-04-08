"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from typing import Optional
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.surrogate_models import KRG
from smt.utils.kriging import componentwise_distance
from sklearn.decomposition import PCA
import numpy as np
import warnings


class PODGP(SurrogateModel):
    name = "POD+GP"

    def _initialize(self):
        super()._initialize()
        self.basis = np.array([])
        self.coeff = np.array([])
        self.sm_list = []
        self.KRG_options = []
        self.x_train = np.array([])
        self.database = np.array([])
        self.n_mods = 0
        self.tol = 0
        self.svd = PCA(svd_solver = 'randomized')
        # self.azerty = kwargs
        
        # print(self.azerty.items())
        
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
    
    def _set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None) -> None:
        return None
    
    def _train():
        return None

    def POD(self, **kwargs):
        dico = kwargs
        choice_svd = None
        self.database = kwargs['database']
        if "n_mods" in dico.keys():
            self.n_mods = kwargs["n_mods"]
            choic_svd = "mod"
        if "tol_svd" in dico.keys():
            if choice_svd != None:
                raise ValueError(
                    "svd can't use both arguments n_mods and tol_svd at the same time."
                )
            self.tol_svd = kwargs["tol_svd"]
            choic_svd = "mod"
        
            
        self.svd.fit(self.database.T)
        
        U = self.svd.components_.T
        S = self.svd.singular_values_
        
        
    def GP(self, **kwargs):
        dico = kwargs
