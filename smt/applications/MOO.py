# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:08:54 2021

@author: robin
"""

# import sys
# sys.path.insert(0,'C:/Users/robin/bayesian-optim')

#%% imports

import numpy as np
from random import randint, uniform

# import matplotlib.pyplot as plt
from types import FunctionType

from scipy.stats import norm
from scipy.optimize import minimize as minimize1D

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

# from pymoo.visualization.scatter import Scatter

# from smt.utils.options_dictionary import OptionsDictionary
from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KPLS, KRG, KPLSK, MGP
from smt.sampling_methods import LHS

#%% Optimization loop incrementing the surrogates


class MOO(SurrogateBasedApplication):
    def _initialize(self):

        super(MOO, self)._initialize()
        declare = self.options.declare

        declare("fun", None, types=FunctionType, desc="Function to minimize")
        declare(
            "criterion",
            "PI",
            types=str,
            values=["PI", "GA"],
            desc="criterion for next evaluation point determination: Expected Improvement, \
            Surrogate-Based Optimization or genetic algo point",
        )
        declare("n_iter", 10, types=int, desc="Number of optimizer steps")
        declare(
            "n_max_optim",
            20,
            types=int,
            desc="Maximum number of internal optimizations",
        )
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("n_start", 20, types=int, desc="Number of optimization start points")
        declare(
            "n_parallel",
            1,
            types=int,
            desc="Number of parallel samples to compute using qEI criterion",
        )
        declare(
            "surrogate",
            KRG(print_global=False),
            types=(KRG, KPLS, KPLSK, MGP),
            desc="SMT kriging-based surrogate model used internaly",
        )  # ne pas utiliser ou adapter au multiobj qu'on aie bien des modees indep pour chaque objectif
        declare(
            "pop_size",
            100,
            types=int,
            desc="number of individuals for the genetic algorithm",
        )
        declare(
            "n_gen",
            100,
            types=int,
            desc="number generations for the genetic algorithm",
        )
        declare(
            "q",
            0.5,
            types=float,
            desc="importance ration of desgn space in comparation to objective space when chosing a point with GA",
        )
        declare("verbose", False, types=bool, desc="Print computation information")

    def optimize(self, fun):
        """
        Optimize the multi-objective function fun. At the end, the object's item
        .modeles is a SMT surrogate_model object with the most precise fun's model
        .result is the result of its optimization thanks to NSGA2

        Parameters
        ----------
        fun : function
            function taking x=ndarray[ne,ndim],
            returning y = [ndarray[ne, 1],ndarray[ne, 1],...]
            where y[i][j][0] = fi(xj).
        """
        x_data, y_data = self._setup_optimizer(fun)
        # n_parallel = self.options["n_parallel"]
        n_gen = self.options["n_gen"]
        pop_size = self.options["pop_size"]
        self.ny = len(y_data)
        self.ndim = self.options["xlimits"].shape[0]

        # obtention des modeles de chaque objectif en liste
        self.modelize(x_data, y_data)
        self.probleme = self.def_prob()

        if type(y_data) != list:
            y_data = list(y_data)

        for k in range(self.options["n_iter"]):

            self.log(str("iteration " + str(k + 1)))

            # find next best x-coord point to evaluate
            new_x = self._find_best_point()
            new_y = fun(np.array([new_x]))

            # update model with the new point
            for i in range(len(y_data)):
                y_data[i] = np.atleast_2d(np.append(y_data[i], new_y[i], axis=0))
            x_data = np.atleast_2d(np.append(x_data, np.array([new_x]), axis=0))

            self.modelize(x_data, y_data)

        self.log("Model is well refined, NSGA2 is running...")
        self.result = minimize(
            self.probleme, NSGA2(pop_size=pop_size), ("n_gen", n_gen), verbose=False
        )
        self.log(
            "Optimization done, get the front with .result.F and the set with .result.X"
        )

    # retourner x, f ?

    def _setup_optimizer(self, fun):
        """
        Parameters
        ----------
        fun : objective function

        Returns
        -------
        xt : array of arrays
            sampling points in the design space.
        yt : list of arrays
            yt[i] = f1(xt).

        """
        sampling = LHS(xlimits=self.options["xlimits"])
        xt = sampling(self.options["n_start"])
        yt = fun(xt)
        return xt, yt

    def modelize(self, xt, yt):
        self.modeles = []
        for iny in range(self.ny):
            t = KRG(print_global=False)
            t.set_training_values(xt, yt[iny])
            t.train()
            self.modeles.append(t)

    def def_prob(self):
        """
        Creates the pymoo Problem object with the surrogate as objective

        Returns
        -------
        MyProblem : pymoo.problem
        """
        n_obj = self.ny
        n_var = self.ndim
        xbounds = self.options["xlimits"]
        modelizations = self.modeles

        class MyProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=0,
                    xl=np.asarray([i[0] for i in xbounds]),
                    xu=np.asarray([i[1] for i in xbounds]),
                    elementwise_evaluation=True,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                xx = np.asarray(x).reshape(1, -1)  # le modèle prend un array en entrée
                out["F"] = [i.predict_values(xx)[0][0] for i in modelizations]

        return MyProblem()

    def _find_best_point(self):
        """
        Selects the best point to refine the model, according to the chosen method

        Returns
        -------
        ndarray
            next point for the model update.
        """
        criterion = self.options["criterion"]

        if criterion == "GA":
            res = minimize(
                self.probleme,
                NSGA2(pop_size=self.options["pop_size"]),
                ("n_gen", self.options["n_gen"]),
                verbose=False,
            )
            X = res.X  # Y = result.F
            ydata = np.transpose(
                np.asarray([mod.training_points[None][0][1] for mod in self.modeles])
            )[0]
            xdata = np.transpose(
                np.asarray([mod.training_points[None][0][0] for mod in self.modeles])
            )[0]
            # MOBOpt criterion
            q = self.options["q"]
            n = ydata.shape[1]
            d_l_x = [sum([np.linalg.norm(xj - xi) for xj in xdata]) / n for xi in xdata]
            d_l_f = [sum([np.linalg.norm(yj - yi) for yj in ydata]) / n for yi in ydata]
            µ_x = np.mean(d_l_x)
            µ_f = np.mean(d_l_f)
            var_x, var_f = np.var(d_l_x), np.var(d_l_f)
            # i = randint(0,X.shape[0]-1)#pour l'instant j'en prends juste un au hasard
            dispersion = [
                q * (d_l_x[j] - µ_x) / var_x + (1 - q) * (d_l_f[j] - µ_f) / var_f
                for j in range(n)
            ]
            i = dispersion.index(max(dispersion))
            return X[i, :]  # , Y[i,:]

        if criterion == "PI":
            self.obj_k = lambda x: -self.PI(x)

        xstart = np.zeros(self.ndim)
        bounds = self.options["xlimits"]
        for i in range(self.ndim):
            xstart[i] = uniform(*bounds[i])
        return minimize1D(self.obj_k, xstart, bounds=bounds).x

    def pareto(self, Y):
        """
        Parameters
        ----------
        Y : list of arrays
            liste of the points to compare.

        Returns
        -------
        index : list
            list of the indexes in Y of the Pareto-optimal points.
        """
        index = []  # indexes of the best points (Pareto)
        n = len(Y)
        dominated = [False] * n
        for y in range(n):
            if not dominated[y]:
                for y2 in range(y + 1, n):
                    if not dominated[
                        y2
                    ]:  # if y2 is dominated (by y0), we already compared y0 to y
                        y_domine_y2, y2_domine_y = self.dominate_min(Y[y], Y[y2])

                        if y_domine_y2:
                            dominated[y2] = True
                        if y2_domine_y:
                            dominated[y] = True
                            break
                if not dominated[y]:
                    index.append(y)
        return index

    # retourne a-domine-b , b-domine-a !! for minimization !!
    def dominate_min(self, a, b):
        """
        Parameters
        ----------
        a : array or list
            coordinates in the objective space.
        b : array or list
            same thing than a.

        Returns
        -------
        bool
            a dominates b (in terms of minimization !).
        bool
            b dominates a (in terms of minimization !).
        """
        a_bat_b = False
        b_bat_a = False
        for i in range(len(a)):
            if a[i] < b[i]:
                a_bat_b = True
                if b_bat_a:
                    return False, False  # same front
            if a[i] > b[i]:
                b_bat_a = True
                if a_bat_b:
                    return False, False
        if a_bat_b and (not b_bat_a):
            return True, False
        if b_bat_a and (not a_bat_b):
            return False, True
        return False, False  # same points

    # Caution !!!! 2-d design space only for the moment !!!
    def PI(self, x):
        """
        Parameters
        ----------
        x : list
            coordonnées du point à évaluer.
        pareto_front : liste
            liste des valeurs dans l'espace objectif des points optimaux
            du modèle actuel.
        moyennes : list
            liste des fonctions moyennes du modèle par objectif à approximer.
        variances : list
            liste des variances des modèles sur chaque objectif.

        Returns
        -------
        pi_x : float
            PI(x) : probabilité que x soit une amélioration € [0,1]
        """

        ydata = np.transpose(
            np.asarray([mod.training_points[None][0][1] for mod in self.modeles])
        )[0]
        pareto_index = self.pareto(ydata)
        pareto_front = [ydata[i] for i in pareto_index]
        moyennes = [mod.predict_values for mod in self.modeles]
        variances = [
            mod.predict_variances for mod in self.modeles
        ]  # racine d'une fction ne marche pas,je fais donc en 2 temps pour ecart-type
        x = np.asarray(x).reshape(1, -1)
        sig1, sig2 = variances[0](x)[0][0] ** 0.5, variances[1](x)[0][0] ** 0.5
        moy1, moy2 = moyennes[0](x)[0][0], moyennes[1](x)[0][0]
        # print(x)
        # print("sigma1&2",sig1, sig2)
        # print("pareto front point 1" , pareto_front)
        # print("moyenne 1 puis 2 ", moy1, moy2)
        m = len(pareto_front)
        try:
            pi_x = norm.cdf((pareto_front[0][0] - moy1) / sig1)
            for i in range(1, m - 1):
                pi_x += (
                    norm.cdf((pareto_front[i + 1][0] - moy1) / sig1)
                    - norm.cdf((pareto_front[i][0] - moy1) / sig1)
                ) * norm.cdf((pareto_front[i + 1][1] - moy2) / sig2)
            pi_x += (1 - norm.cdf((pareto_front[m - 1][0] - moy1) / sig1)) * norm.cdf(
                (pareto_front[m - 1][1] - moy2) / sig2
            )
            # print("pi_x = ",pi_x)
            return pi_x
        except:  # for training points -> having variances = 0
            print("training x called : ", x)
            return 0

    def log(self, msg):
        if self.options["verbose"]:
            print(msg)
