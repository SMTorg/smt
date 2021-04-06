# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:08:54 2021

@author: robin
"""

#import sys
#sys.path.insert(0,'C:/Users/robin/bayesian-optim')

#%% imports

import numpy as np
from random import randint
import matplotlib.pyplot as plt
from types import FunctionType

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from smt.utils.options_dictionary import OptionsDictionary
from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KPLS, KRG, KPLSK, MGP
from smt.sampling_methods import LHS

#%% Optimization loop incrementing the surrogates

class MOO(SurrogateBasedApplication):
    
    def _initialize(self):
        
        super(MOO, self)._initialize()
        declare = self.options.declare

        declare("fun", None, types=FunctionType, desc="Function to minimize")
        declare("criterion","GA",types=str,values=["EI", "GA"],
            desc="criterion for next evaluation point determination: Expected Improvement, \
            Surrogate-Based Optimization or genetic algo point",)
        declare("n_iter", 10, types=int, desc="Number of optimizer steps")
        declare("n_max_optim",20,types=int,
                desc="Maximum number of internal optimizations",)
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("n_start", 20, types=int, desc="Number of optimization start points")
        declare("n_parallel",1,types=int,
            desc="Number of parallel samples to compute using qEI criterion",)
        declare("surrogate",KRG(print_global=False),types=(KRG, KPLS, KPLSK, MGP),
            desc="SMT kriging-based surrogate model used internaly",
        )#ne pas utiliser ou adapter au multiobj qu'on aie bien des modees indep pour chaque objectif
        declare("pop_size",100,types=int,
            desc="number of individuals for the genetic algorithm",)
        declare("n_gen",100,types=int,
            desc="number generations for the genetic algorithm",)
        
    def optimize(self, fun):
        x_data, y_data = self._setup_optimizer(fun)
        #n_parallel = self.options["n_parallel"]
        n_gen = self.options["n_gen"]
        pop_size = self.options["pop_size"]
        self.ny = len(y_data)
        self.ndim = self.options["xlimits"].shape[0]
        
        #obtention des modeles de chaque objectif en liste
        self.modelize(x_data, y_data)
        self.probleme= self.def_prob()
        
        for k in range(self.options["n_iter"]):
            
            print("iteration ",k+1)

            # find next best x-coord point to evaluate
            new_x = self._find_best_point()                
            new_y = fun(np.array([new_x]))
            
            #update model with the new point
            for i in range(len(y_data)):            
                y_data[i] = np.atleast_2d(np.append(y_data[i],new_y[i],axis=0))
            x_data = np.atleast_2d(np.append(x_data,np.array([new_x]),axis=0))
            
            self.modelize(x_data, y_data)
            
        print("Model is well refined, NSGA2 is running...")
        self.result = minimize(self.probleme,NSGA2(pop_size=pop_size),("n_gen",n_gen),verbose=False)
        print("Optimizatin done, get the front with .result.F and the set with .result.X")
        
    def _find_best_point(self): #pour l'instant j'en prends juste un au hasard
        # à terme faire en fonction de leur distance ? ou autre ?
        res = minimize(self.probleme,
               NSGA2(pop_size=self.options["pop_size"]),
               ("n_gen", self.options["n_gen"]),
               verbose=False)
        X = res.X
        #Y = result.F
        i = randint(0,X.shape[0]-1)
        return X[i,:]#, Y[i,:]
    
    def modelize(self,xt,yt):
        self.modeles = []
        for iny in range(self.ny):   
            t= KRG(print_global=False)
            t.set_training_values(xt,yt[iny])        
            t.train()
            self.modeles.append(t)

    def def_prob(self):
        n_obj = self.ny
        n_var = self.ndim
        xbounds = self.options["xlimits"]
        modelizations = self.modeles
        class MyProblem(Problem):
        
            def __init__(self):
                super().__init__(n_var=n_var,
                                 n_obj=n_obj,
                                 n_constr=0,
                                 xl=np.asarray([i[0] for i in xbounds]),
                                 xu=np.asarray([i[1] for i in xbounds]),
                                 elementwise_evaluation=True)
        
            def _evaluate(self, x, out, *args, **kwargs):
                xx = np.asarray(x).reshape(1, -1) #le modèle prend un array en entrée
                out["F"] = [i.predict_values(xx)[0][0] for i in modelizations]
                
        
        return MyProblem()                
        
    def _setup_optimizer(self,fun):
        sampling = LHS(xlimits=self.options["xlimits"])
        xt = sampling(self.options["n_start"])
        yt = fun(xt)
        return xt, yt