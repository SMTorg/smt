"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli

This package is distributed under New BSD license.

"""
# TODO : documentation

from __future__ import division
import six
import numpy as np
import warnings

from scipy.stats import norm
from scipy.optimize import minimize

from smt.utils.options_dictionary import OptionsDictionary
from smt.extensions.extensions import Extensions
from smt.utils.misc import compute_rms_error

from smt.surrogate_models import KPLS, KRG, KPLSK
from smt.sampling_methods import LHS


class EGO(object):
    def optimize(self, fun, n_iter, criterion, ndim, ndoe, xlimits):

        sampling = LHS(xlimits=xlimits, criterion="ese")
        x_doe = sampling(ndoe)
        y_doe = fun(x_doe)

        # to save the initial doe
        x_data = x_doe
        y_data = y_doe

        gpr = KRG(theta0=[1e-2] * ndim, print_global=False)

        for _ in range(n_iter):
            x_start = np.atleast_2d(np.random.rand(20) * 25).T
            f_min_k = np.min(y_data)
            gpr.set_training_values(x_data, y_data)
            gpr.train()
            if criterion == "EI":
                obj_k = lambda x: -EGO.EI(gpr, np.atleast_2d(x), f_min_k)
            elif criterion == "SBO":
                obj_k = lambda x: EGO.SBO(gpr, np.atleast_2d(x))
            elif criterion == "UCB":
                obj_k = lambda x: EGO.UCB(gpr, np.atleast_2d(x))

            opt_all = np.array(
                [
                    minimize(obj_k, x_st, method="SLSQP", bounds=[(0, 25)])
                    for x_st in x_start
                ]
            )
            opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
            obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
            ind_min = np.argmin(obj_success)
            opt = opt_success[ind_min]
            x_et_k = opt["x"]

            y_et_k = fun(x_et_k)

            y_data = np.atleast_2d(np.append(y_data, y_et_k)).T
            x_data = np.atleast_2d(np.append(x_data, x_et_k)).T

        ind_best = np.argmin(y_data)
        x_opt = x_data[ind_best]
        y_opt = y_data[ind_best]

        return x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe

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
