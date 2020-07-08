"""
Authors: Nathalie Bartoli, Remy Priem, Remi Lafage, Emile Roux <emile.roux@univ-smb.fr>

This package is distributed under New BSD license.

"""

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


class Evaluator(object):
    """
    An interface for evaluation of a function at x points (nsamples of dimension nx).
    User can derive this interface and override the run() method to implement custom multiprocessing.
    """

    def run(self, fun, x):
        """
        Evaluates fun at x.

        Parameters
        ---------
        fun : function to evaluate: (nsamples, nx) -> (nsample, 1)

        x : np.ndarray[nsamples, nx]
            nsamples points of nx dimensions.

        Returns
        -------
        np.ndarray[nsample, 1]
            fun evaluations at the nsamples points.

        """
        return fun(x)


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
            "n_parallel",
            1,
            types=int,
            desc="Number of parallel samples to compute using qEI criterion",
        )
        declare(
            "qEI",
            "KBLB",
            types=str,
            values=["KB", "KBLB", "KBUB", "KBRand", "CLmin"],
            desc="Approximated q-EI maximization strategy",
        )
        declare(
            "evaluator",
            default=Evaluator(),
            types=Evaluator,
            desc="Object used to run function fun to optimize at x points (nsamples, nxdim)",
        )
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
        # Set the bounds of the optimization problem
        xlimits = self.options["xlimits"]

        # Build initial DOE
        self._sampling = LHS(xlimits=xlimits, criterion="ese")
        self._evaluator = self.options["evaluator"]

        xdoe = self.options["xdoe"]
        if xdoe is None:
            self.log("Build initial DOE with LHS")
            n_doe = self.options["n_doe"]
            x_doe = self._sampling(n_doe)
        else:
            self.log("Initial DOE given")
            x_doe = np.atleast_2d(xdoe)

        ydoe = self.options["ydoe"]
        if ydoe is None:
            y_doe = self._evaluator.run(fun, x_doe)
        else:  # to save time if y_doe is already given to EGO
            y_doe = ydoe

        # to save the initial doe
        x_data = x_doe
        y_data = y_doe

        self.gpr = KRG(print_global=False)

        n_iter = self.options["n_iter"]
        n_parallel = self.options["n_parallel"]

        for k in range(n_iter):
            # Virtual enrichement loop
            for p in range(n_parallel):
                x_et_k, success = self._find_points(x_data, y_data)
                if not success:
                    self.log(
                        "Internal optimization failed at EGO iter = {}.{}".format(k, p)
                    )
                    break
                elif success:
                    self.log(
                        "Internal optimization succeeded at EGO iter = {}.{}".format(
                            k, p
                        )
                    )
                # Set temporaly the y_data to the one predicted by the kringin metamodel
                y_et_k = self.set_virtual_point(np.atleast_2d(x_et_k), y_data)

                # Update y_data with predicted value
                y_data = np.atleast_2d(np.append(y_data, y_et_k)).T
                x_data = np.atleast_2d(np.append(x_data, x_et_k, axis=0))

            # Compute the real values of y_data
            x_to_compute = np.atleast_2d(x_data[-n_parallel:])
            y = self._evaluator.run(fun, x_to_compute)
            y_data[-n_parallel:] = y

        # Find the optimal point
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

    def _find_points(self, x_data, y_data):
        """
        Function that analyse a set of x_data and y_data and give back the 
        more interesting point to evaluates according to the selected criterion
        
        Inputs: 
            - x_data and y_data
        Outputs:
            - x_et_k : the points to evaluate
            - success bool : boolean succes flag to interupte
                the main loop if need
        
        """
        self.gpr.set_training_values(x_data, y_data)
        self.gpr.train()

        criterion = self.options["criterion"]
        n_start = self.options["n_start"]
        n_max_optim = self.options["n_max_optim"]
        bounds = self.options["xlimits"]

        if criterion == "EI":
            self.obj_k = lambda x: -self.EI(np.atleast_2d(x), y_data)
        elif criterion == "SBO":
            self.obj_k = lambda x: self.SBO(np.atleast_2d(x))
        elif criterion == "UCB":
            self.obj_k = lambda x: self.UCB(np.atleast_2d(x))

        success = False
        n_optim = 1  # in order to have some success optimizations with SLSQP
        while not success and n_optim <= n_max_optim:
            opt_all = []
            x_start = self._sampling(n_start)
            for ii in range(n_start):
                opt_all.append(
                    minimize(
                        lambda x: float(self.obj_k(x)),
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
                n_optim += 1

        if n_optim >= n_max_optim:
            # self.log("Internal optimization failed at EGO iter = {}".format(k))
            return np.atleast_2d(0), False

        ind_min = np.argmin(obj_success)
        opt = opt_success[ind_min]
        x_et_k = np.atleast_2d(opt["x"])
        return x_et_k, True

    def set_virtual_point(self, x, y_data):
        qEI = self.options["qEI"]

        if qEI == "CLmin":
            return np.min(y_data)

        if qEI == "KB":
            return self.gpr.predict_values(x)

        if qEI == "KBUB":
            conf = 3.0

        if qEI == "KBLB":
            conf = -3.0

        if qEI == "KBRand":
            conf = np.random.randn()

        pred = self.gpr.predict_values(x)
        var = self.gpr.predict_variances(x)
        return pred + conf * np.sqrt(var)
