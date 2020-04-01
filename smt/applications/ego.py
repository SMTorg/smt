"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli

This package is distributed under New BSD license.

Saves Paul branch :  v3
"""
# TODO : documentation

from __future__ import division
import six
import numpy as np

from types import FunctionType

from scipy.stats import norm
from scipy.optimize import minimize

from smt.utils.options_dictionary import OptionsDictionary
from smt.applications.application import SurrogateBasedApplication
from smt.utils.misc import compute_rms_error

from smt.surrogate_models import KPLS, KRG, KPLSK
from smt.sampling_methods import LHS


class EGO(SurrogateBasedApplication):
    def _initialize(self):
        super(EGO, self)._initialize()
        declare = self.options.declare

        declare("fun", None, types=FunctionType, desc="Function to minimize")
        declare(
            "criterion",
            "EI",
            types=str,
            values=["EI", "SBO", "UCB"],
            desc="criterion for next evaluation point determination: Expected Improvement, \
            Surrogate-Based Optimization or Upper Confidence Bound",
        )
        declare("n_iter", None, types=int, desc="Number of optimizer steps")
        declare(
            "n_max_optim",
            20,
            types=int,
            desc="Maximum number of internal optimizations",
        )
        declare("n_start", 20, types=int, desc="Number of optimization start points")
        declare(
            "n_doe",
            None,
            types=int,
            desc="Number of points of the initial LHS doe, only used if xdoe is not given",
        )
        declare("xdoe", None, types=np.ndarray, desc="Initial doe inputs")
        declare("ydoe", None, types=np.ndarray, desc="Initial doe outputs")
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("verbose", False, types=bool, desc="Print computation information")

    def optimize(self, fun):
        """
        Optimizes fun

        Parameters
        ----------

        fun: function to optimize: ndarray[n, nx] or ndarray[n] -> ndarray[n, 1]

        Returns
        -------

        [nx, 1]: x optimum
        [1, 1]: y optimum
        int: index of optimum in data arrays 
        [ndoe + n_iter, nx]: coord-x data
        [ndoe + n_iter, 1]: coord-y data
        [ndoe, nx]: coord-x initial doe
        [ndoe, 1]: coord-y initial doe
        """
        xlimits = self.options["xlimits"]
        sampling = LHS(xlimits=xlimits, criterion="ese")

        xdoe = self.options["xdoe"]
        if xdoe is None:
            self.log("Build initial DOE with LHS")
            n_doe = self.options["n_doe"]
            x_doe = sampling(n_doe)
        else:
            self.log("Initial DOE given")
            x_doe = np.atleast_2d(xdoe)

        ydoe = self.options["ydoe"]
        if ydoe is None:
            y_doe = fun(x_doe)
        else: # to save time if y_doe is already given to EGO
            y_doe = ydoe

        # to save the initial doe
        x_data = x_doe
        y_data = y_doe

        self.gpr = KRG(print_global=False)

        bounds = xlimits

        criterion = self.options["criterion"]
        n_iter = self.options["n_iter"]
        n_start = self.options["n_start"]
        n_max_optim = self.options["n_max_optim"]

        for k in range(n_iter):

            self.gpr.set_training_values(x_data, y_data)
            self.gpr.train()

            if criterion == "EI":
                self.obj_k = lambda x: -self.EI(np.atleast_2d(x), y_data)
            elif criterion == "EI_penalized":
                self.obj_k = lambda x: -self.EI_penalized(np.atleast_2d(x),x_data,y_data)
            elif criterion == "EI_penalized_cate":
                self.obj_k = lambda x: -self.EI_penalized_cate(np.atleast_2d(x), x_data, y_data)
            elif criterion == "SBO":
                self.obj_k = lambda x: self.SBO(np.atleast_2d(x))
            elif criterion == "UCB":
                self.obj_k = lambda x: self.UCB(np.atleast_2d(x))

            success = False
            n_optim = 1  # in order to have some success optimizations with SLSQP
            while not success and n_optim <= n_max_optim:
                opt_all = []
                x_start = sampling(n_start)

                for ii in range(n_start):
                    if criterion=="EI_penalized":
                        #construct the bounds in the form of constraints
                        cons = []
                        for factor in range(len(bounds)):
                            lower, upper = bounds[factor]
                            l = {'type': 'ineq','fun': lambda x, lb=lower, i=factor: x[i] - lb}
                            u = {'type': 'ineq', 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
                            cons.append(l)
                            cons.append(u)  
                        opt_all.append((minimize(self.obj_k,x_start[ii, :],method="COBYLA",constraints=cons,options={"maxiter": 200})))
                    else : 
                        opt_all.append(minimize(self.obj_k,x_start[ii, :],method="SLSQP",bounds=bounds,options={"maxiter": 200}))

                opt_all = np.asarray(opt_all)

                opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
                obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
                success = obj_success.size != 0
                if not success:
                    self.log("New start point for the internal optimization")
                    n_optim += 1

            if n_optim >= n_max_optim:
                self.log("Internal optimization failed at EGO iter = {}".format(k))
                break
            elif success:
                self.log("Internal optimization succeeded at EGO iter = {}".format(k))

            ind_min = np.argmin(obj_success)
            opt = opt_success[ind_min]
            x_et_k = np.atleast_2d(opt["x"])
            if criterion=="EI_penalized":
                x_et_k = np.round(x_et_k)
            if criterion == "EI_penalized_cate":

                  y=np.zeros(np.shape(x_et_k))
                  y[0][np.argmax(x_et_k[0])]=1
                  x_et_k = y
            y_et_k = fun(x_et_k)

            y_data = np.atleast_2d(np.append(y_data, y_et_k)).T
            x_data = np.atleast_2d(np.append(x_data, x_et_k, axis=0))

        ind_best = np.argmin(y_data)        
        x_opt = x_data[ind_best]
        y_opt = y_data[ind_best]

        return x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe

    def log(self, msg):
        if self.options["verbose"]:
            print(msg)

    def EI(self, points, y_data):
        """ Expected improvement """
        f_min = np.min(y_data)
        pred = self.gpr.predict_values(points)
        sig = np.sqrt(self.gpr.predict_variances(points))
        args0 = (f_min - pred) / sig
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = sig * norm.pdf(args0)
        if sig.size == 1 and sig == 0.0:  # can be use only if one point is computed
            return 0.0

        ei = args1 + args2
        return ei

    def EI_penalized(self, points, x_data,y_data):
        """ Expected improvement """
        f_min = np.min(y_data)
        pred = self.gpr.predict_values(points)
        sig = np.sqrt(self.gpr.predict_variances(points))
        args0 = (f_min - pred) / sig
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = sig * norm.pdf(args0)
        if sig.size == 1 and sig == 0.0:  # can be use only if one point is computed
            return 0.0
        ei = args1 + args2
        for i in range(len(points)) :
            p =np.atleast_2d(points[i])
            for x in x_data:
                x =np.atleast_2d(x)
               #if np.abs(p-x)<1:
                   #ei[i]=ei[i]*np.reciprocal(1+100*np.exp(-np.reciprocal(1-np.square(p-x))))
                pena=( (self.EI(p,y_data)-self.EI(x,y_data))/np.power(np.linalg.norm(p-x),4) )
                if pena > 0 :
                    ei[i]= ei[i]- pena
                ei[i]=max(ei[i],0)
        return ei

    def EI_penalized_cate(self, points, x_data, y_data):
        """ Expected improvement """
        f_min = np.min(y_data)
        pred = self.gpr.predict_values(points)
        sig = np.sqrt(self.gpr.predict_variances(points))
        args0 = (f_min - pred) / sig
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = sig * norm.pdf(args0)
        if sig.size == 1 and sig == 0.0:  # can be use only if one point is computed
            return 0.0
        ei = args1 + args2
        for i in range(len(points)):
            p = np.atleast_2d(points[i])
            for x in x_data:
                x = np.atleast_2d(x)
                # if np.abs(p-x)<1:
                # ei[i]=ei[i]*np.reciprocal(1+100*np.exp(-np.reciprocal(1-np.square(p-x))))
                pena = ((self.EI(p, y_data) - self.EI(x, y_data)) / np.power(np.linalg.norm(p - x), 4))
                if pena > 0:
                    ei[i] = ei[i] - pena
                ei[i] = max(ei[i], 0)
        return ei


    def SBO(self, point):
        """ Surrogate based optimization: min the surrogate model by suing the mean mu """
        res = self.gpr.predict_values(point)
        return res

    def UCB(self, point):
        """ Upper confidence bound optimization: minimize by using mu - 3*sigma """
        pred = self.gpr.predict_values(point)
        var = self.gpr.predict_variances(point)
        res = pred - 3.0 * np.sqrt(var)
        return res
