"""
Authors: Nathalie Bartoli, Remy Priem, Remi Lafage, Emile Roux <emile.roux@univ-smb.fr>

This package is distributed under New BSD license.

"""

from types import FunctionType

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from smt.applications.application import SurrogateBasedApplication
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
)
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS, KPLS, KPLSK, KRG, MGP, GPX
from smt.utils.design_space import (
    BaseDesignSpace,
    DesignSpace,
)


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
            values=["EI", "SBO", "LCB"],
            desc="criterion for next evaluation point determination: Expected Improvement, \
            Surrogate-Based Optimization or Lower Confidence Bound",
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
        declare("verbose", False, types=bool, desc="Print computation information")
        declare(
            "enable_tunneling",
            False,
            types=bool,
            desc="Enable the penalization of points that have been already evaluated in EI criterion",
        )
        declare(
            "surrogate",
            KRG(print_global=False),
            types=(KRG, KPLS, KPLSK, GEKPLS, MGP, GPX),
            desc="SMT kriging-based surrogate model used internaly",
        )
        self.options.declare(
            "random_state",
            types=(type(None), int, np.random.RandomState),
            desc="Numpy RandomState object or seed number which controls random draws",
        )

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
        """
        x_data, y_data = self._setup_optimizer(fun)
        n_iter = self.options["n_iter"]
        n_parallel = self.options["n_parallel"]

        for k in range(n_iter):
            # Virtual enrichement loop
            for p in range(n_parallel):
                # find next best x-coord point to evaluate
                x_et_k, success = self._find_best_point(
                    x_data, y_data, self.options["enable_tunneling"]
                )
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
                # Set temporaly the y-coord point based on the kriging prediction
                x_et_k = np.atleast_2d(x_et_k)
                if self.mixint:
                    x_et_k, _ = self.design_space.correct_get_acting(x_et_k)
                y_et_k = self._get_virtual_point(x_et_k, y_data)

                # Update y_data with predicted value
                y_data = y_data.reshape(y_data.shape[0], self.gpr.ny)
                y_data = np.vstack((y_data, y_et_k))
                x_data = np.atleast_2d(np.append(x_data, x_et_k, axis=0))

            # Compute the real values of y_data
            x_to_compute = np.atleast_2d(x_data[-n_parallel:])
            y = self._evaluator.run(fun, x_to_compute)
            y_data[-n_parallel:] = y

        # Find the optimal point
        ind_best = np.argmin(y_data if y_data.ndim == 1 else y_data[:, 0])
        x_opt = x_data[ind_best]
        y_opt = y_data[ind_best]
        return x_opt, y_opt, ind_best, x_data, y_data

    def log(self, msg):
        if self.options["verbose"]:
            print(msg)

    def EI(self, points, enable_tunneling=False, x_data=None):
        """Expected improvement"""
        y_data = np.atleast_2d(self.gpr.training_points[None][0][1])
        f_min = y_data[np.argmin(y_data[:, 0])]
        pred = self.gpr.predict_values(points)
        sig = np.sqrt(self.gpr.predict_variances(points))
        args0 = (f_min - pred) / sig
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = sig * norm.pdf(args0)
        if sig.size == 1 and sig == 0.0:  # can be use only if one point is computed
            return 0.0
        ei = args1 + args2
        # penalize the points already evaluated with tunneling
        if enable_tunneling:
            for i in range(len(points)):
                p = np.atleast_2d(points[i])
                EIp = self.EI(p, enable_tunneling=False)
                for x in x_data:
                    x = np.atleast_2d(x)
                    # if np.abs(p-x)<1:
                    # ei[i]=ei[i]*np.reciprocal(1+100*np.exp(-np.reciprocal(1-np.square(p-x))))
                    pena = (EIp - self.EI(x, enable_tunneling=False)) / (
                        1e-9 + np.power(np.linalg.norm(p - x), 4)
                    )
                    if pena > 0:
                        ei[i] = ei[i] - pena
                    ei[i] = max(ei[i], 0)
        return ei

    def SBO(self, point):
        """Surrogate based optimization: min the surrogate model by suing the mean mu"""
        res = self.gpr.predict_values(point)
        return res

    def LCB(self, point):
        """Lower confidence bound optimization: minimize by using mu - 3*sigma"""
        pred = self.gpr.predict_values(point)
        var = self.gpr.predict_variances(point)
        res = pred - 3.0 * np.sqrt(var)
        return res

    def _setup_optimizer(self, fun):
        """
        Instanciate internal surrogate used for optimization
        and setup function evaluator wrt options

        Parameters
        ----------

        fun: function to optimize: ndarray[n, nx] or ndarray[n] -> ndarray[n, 1]

        Returns
        -------

        ndarray: initial coord-x doe
        ndarray: initial coord-y doe = fun(xdoe)

        """
        # Set the model
        self.gpr = self.options["surrogate"]
        self.design_space: BaseDesignSpace = self.gpr.design_space
        if isinstance(self.design_space, DesignSpace):
            self.design_space.seed = self.options["random_state"]

        # Handle mixed integer optimization
        is_continuous = self.design_space.is_all_cont
        if not is_continuous:
            self.categorical_kernel = self.gpr.options["categorical_kernel"]
            self.mixint = MixedIntegerContext(
                self.design_space,
                work_in_folded_space=True,
            )

            underlying_gpr = self.gpr
            self.gpr = self.mixint.build_kriging_model(self.gpr)

            self.categorical_kernel = underlying_gpr.options["categorical_kernel"]
            self.mixint = MixedIntegerContext(
                self.design_space,
                work_in_folded_space=True,
            )
            self._sampling = self.mixint.build_sampling_method(
                random_state=self.options["random_state"],
            )

        else:
            self.mixint = None
            sampling = MixedIntegerSamplingMethod(
                LHS,
                self.design_space,
                criterion="ese",
                random_state=self.options["random_state"],
            )
            self._sampling = lambda n: sampling(n)

            self.categorical_kernel = None

        # Build DOE
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

        return x_doe, y_doe

    def _train_gpr(self, x_data, y_data):
        self.gpr.set_training_values(x_data, y_data)
        if self.gpr.supports["training_derivatives"]:
            for kx in range(self.gpr.nx):
                self.gpr.set_training_derivatives(
                    x_data, y_data[:, 1 + kx].reshape((y_data.shape[0], 1)), kx
                )
        self.gpr.train()

    def _find_best_point(self, x_data=None, y_data=None, enable_tunneling=False):
        """
        Function that analyse a set of x_data and y_data and give back the
        more interesting point to evaluates according to the selected criterion

        Parameters
        ----------

        x_data: ndarray(n_points, nx)
        y_data: ndarray(n_points, 1)

        Returns
        -------

        ndarray(nx, 1): the next best point to evaluate
        boolean: success flag

        """
        self._train_gpr(x_data, y_data)

        criterion = self.options["criterion"]
        n_start = self.options["n_start"]
        n_max_optim = self.options["n_max_optim"]
        method = "SLSQP"
        options = {"maxiter": 200}
        if self.mixint:
            bounds = self.design_space.get_num_bounds()
            cons = []
            for j in range(len(bounds)):
                lower, upper = bounds[j]
                lo = {"type": "ineq", "fun": lambda x, lb=lower, i=j: x[i] - lb}
                up = {"type": "ineq", "fun": lambda x, ub=upper, i=j: ub - x[i]}
                cons.append(lo)
                cons.append(up)
            bounds = None
            options = {"maxiter": 300}
        else:
            bounds = self.design_space.get_num_bounds()
            cons = ()

        if criterion == "EI":
            self.obj_k = lambda x: -self.EI(np.atleast_2d(x), enable_tunneling, x_data)
        elif criterion == "SBO":
            self.obj_k = lambda x: self.SBO(np.atleast_2d(x))
        elif criterion == "LCB":
            self.obj_k = lambda x: self.LCB(np.atleast_2d(x))

        success = False
        n_optim = 1  # in order to have some success optimizations with SLSQP
        while not success and n_optim <= n_max_optim:
            opt_all = []
            x_start = self._sampling(n_start)
            for ii in range(n_start):
                try:
                    opt_all.append(
                        minimize(
                            lambda x: float(np.array(self.obj_k(x)).flat[0]),
                            x_start[ii, :],
                            method=method,
                            bounds=bounds,
                            constraints=cons,
                            options=options,
                        )
                    )

                except ValueError:  # in case "x0 violates bound constraints" error
                    print("warning: `x0` violates bound constraints")
                    print("x0={}".format(x_start[ii, :]))
                    print("bounds={}".format(bounds))
                    opt_all.append({"success": False})

            opt_all = np.asarray(opt_all)
            for opt_i in opt_all:
                if (
                    opt_i["message"]
                    == "Maximum number of function evaluations has been exceeded."
                ):
                    opt_i["success"] = True
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

    def _get_virtual_point(self, x, y_data):
        """
        Depending on the qEI attribute return a predicted value at given point x

        Parameters
        ----------

        x: ndarray(1, 1) the x-coord point where to forecast the y-coord virtual point
        y_data: current y evaluation list only used when qEI is CLmin

        Returns
        -------

        ndarray(1, 1): the so-called virtual y-coord point

        """
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
