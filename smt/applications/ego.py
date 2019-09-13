"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli

This package is distributed under New BSD license.

"""
# TODO : documentation

from __future__ import division
import six
import numpy as np
import warnings

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
            desc="criterion for next evaluaition point",
        )
        declare("niter", None, types=int, desc="Number of iterations")
        declare(
            "nmax_optim", 20, types=int, desc="Maximum number of internal optimizations"
        )
        declare("nstart", 20, types=int, desc="Number of start")
        declare("ndoe", None, types=int, desc="Number of points of the initial doe")
        declare("xdoe", None, types=np.ndarray, desc="Initial doe inputs")
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("verbose", False, types=bool, desc="Print computation information")

    def optimize(self, fun):
        xlimits = self.options["xlimits"]
        sampling = LHS(xlimits=xlimits, criterion="ese")

        doe = self.options["xdoe"]
        if doe is None:
            self.log("Build initial DOE with LHS")
            ndoe = self.options["ndoe"]
            x_doe = sampling(ndoe)
        else:
            self.log("Initial DOE given")
            x_doe = np.atleast_2d(doe)

        y_doe = fun(x_doe)

        # to save the initial doe
        x_data = x_doe
        y_data = y_doe

        gpr = KRG(print_global=False)

        bounds = xlimits

        criterion = self.options["criterion"]
        niter = self.options["niter"]
        nstart = self.options["nstart"]
        nmax_optim = self.options["nmax_optim"]

        for k in range(niter):

            f_min_k = np.min(y_data)
            gpr.set_training_values(x_data, y_data)
            gpr.train()

            if criterion == "EI":
                obj_k = lambda x: -EGO.EI(gpr, np.atleast_2d(x), f_min_k)
            elif criterion == "SBO":
                obj_k = lambda x: EGO.SBO(gpr, np.atleast_2d(x))
            elif criterion == "UCB":
                obj_k = lambda x: EGO.UCB(gpr, np.atleast_2d(x))

            success = False
            noptim = 1  # in order to have some success optimizations with SLSQP
            while not success and noptim <= nmax_optim:
                opt_all = []
                x_start = sampling(nstart)
                for ii in range(nstart):
                    opt_all.append(
                        minimize(
                            obj_k,
                            x_start[ii, :],
                            method="SLSQP",
                            bounds=bounds,
                            options={"maxiter": 200},
                        )
                    )

                opt_all = np.asarray(opt_all)

                opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
                obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
                success = obj_success.size != 0
                if not success:
                    self.log("New start point for the internal optimization")
                    noptim += 1

            if noptim >= nmax_optim:
                self.log("Internal optimization failed at EGO iter = {}".format(k))
                break
            elif success:
                self.log("Internal optimization succeeded at EGO iter = {}".format(k))

            ind_min = np.argmin(obj_success)
            opt = opt_success[ind_min]
            x_et_k = np.atleast_2d(opt["x"])
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

    @staticmethod
    def EI(gpr, points, f_min):
        """ Expected improvement """
        pred = gpr.predict_values(points)
        var = gpr.predict_variances(points)
        args0 = (f_min - pred) / var
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = var * norm.pdf(args0)
        if var == 0.0:  # can be use only if one point is computed
            return 0.0

        ei = args1 + args2
        return ei

    @staticmethod
    def SBO(gpr, point):
        """ Surrogate based optimization: min the surrogate model by suing the mean mu """
        res = gpr.predict_values(point)
        return res

    @staticmethod
    def UCB(gpr, point):
        """ Upper confidence bound optimization: minimize by using mu - 3*sigma """
        pred = gpr.predict_values(point)
        var = gpr.predict_variances(point)
        res = pred - 3.0 * var
        return res
