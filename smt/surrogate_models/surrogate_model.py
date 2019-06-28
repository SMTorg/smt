"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

from __future__ import division

import numpy as np
from collections import defaultdict

from smt.utils.printer import Printer
from smt.utils.options_dictionary import OptionsDictionary
from smt.utils.checks import check_support, check_nx, check_2d_array


class SurrogateModel(object):
    """
    Base class for all surrogate models.

    Attributes
    ----------
    options : OptionsDictionary
        Dictionary of options. Options values can be set on this attribute directly
        or they can be passed in as keyword arguments during instantiation.
    supports : dict
        Dictionary containing information about what this surrogate model supports.

    Examples
    --------
    >>> from smt.surrogate_models import RBF
    >>> sm = RBF(print_training=False)
    >>> sm.options['print_prediction'] = False
    """

    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.

        For the list of options, see the documentation for the surrogate model being used.

        Parameters
        ----------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.

        Examples
        --------
        >>> from smt.surrogate_models import RBF
        >>> sm = RBF(print_global=False)
        """
        self.options = OptionsDictionary()

        self.supports = supports = {}
        supports["training_derivatives"] = False
        supports["derivatives"] = False
        supports["output_derivatives"] = False
        supports["adjoint_api"] = False
        supports["variances"] = False

        declare = self.options.declare

        declare(
            "print_global",
            True,
            types=bool,
            desc="Global print toggle. If False, all printing is suppressed",
        )
        declare(
            "print_training",
            True,
            types=bool,
            desc="Whether to print training information",
        )
        declare(
            "print_prediction",
            True,
            types=bool,
            desc="Whether to print prediction information",
        )
        declare(
            "print_problem",
            True,
            types=bool,
            desc="Whether to print problem information",
        )
        declare(
            "print_solver", True, types=bool, desc="Whether to print solver information"
        )

        self._initialize()
        self.options.update(kwargs)
        self.training_points = defaultdict(dict)
        self.printer = Printer()

    def set_training_values(self, xt, yt, name=None):
        """
        Set training data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        yt : np.ndarray[nt, ny] or np.ndarray[nt]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        xt = check_2d_array(xt, "xt")
        yt = check_2d_array(yt, "yt")

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        self.nx = xt.shape[1]
        self.ny = yt.shape[1]
        kx = 0
        self.training_points[name][kx] = [np.array(xt), np.array(yt)]

    def update_training_values(self, yt, name=None):
        """
        Update the training data (values) at the previously set input values.

        Parameters
        ----------
        yt : np.ndarray[nt, ny] or np.ndarray[nt]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        yt = check_2d_array(yt, "yt")

        kx = 0

        if kx not in self.training_points[name]:
            raise ValueError(
                "The training points must be set first with set_training_values "
                + "before calling update_training_values."
            )

        xt = self.training_points[name][kx][0]
        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "The number of training points does not agree with the earlier call of "
                + "set_training_values."
            )

        self.training_points[name][kx][1] = np.array(yt)

    def set_training_derivatives(self, xt, dyt_dxt, kx, name=None):
        """
        Set training data (derivatives).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        dyt_dxt : np.ndarray[nt, ny] or np.ndarray[nt]
            The derivatives values for the nt training points.
        kx : int
            0-based index of the derivatives being set.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        check_support(self, "training_derivatives")

        xt = check_2d_array(xt, "xt")
        dyt_dxt = check_2d_array(dyt_dxt, "dyt_dxt")

        if xt.shape[0] != dyt_dxt.shape[0]:
            raise ValueError(
                "the first dimension of xt and dyt_dxt must have the same length"
            )

        if not isinstance(kx, int):
            raise ValueError("kx must be an int")

        self.training_points[name][kx + 1] = [np.array(xt), np.array(dyt_dxt)]

    def update_training_derivatives(self, dyt_dxt, kx, name=None):
        """
        Update the training data (values) at the previously set input values.

        Parameters
        ----------
        dyt_dxt : np.ndarray[nt, ny] or np.ndarray[nt]
            The derivatives values for the nt training points.
        kx : int
            0-based index of the derivatives being set.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        check_support(self, "training_derivatives")

        dyt_dxt = check_2d_array(dyt_dxt, "dyt_dxt")

        if kx not in self.training_points[name]:
            raise ValueError(
                "The training points must be set first with set_training_values "
                + "before calling update_training_values."
            )

        xt = self.training_points[name][kx][0]
        if xt.shape[0] != dyt_dxt.shape[0]:
            raise ValueError(
                "The number of training points does not agree with the earlier call of "
                + "set_training_values."
            )

        self.training_points[name][kx + 1][1] = np.array(dyt_dxt)

    def train(self):
        """
        Train the model
        """
        n_exact = self.training_points[None][0][0].shape[0]

        self.printer.active = self.options["print_global"]
        self.printer._line_break()
        self.printer._center(self.name)

        self.printer.active = (
            self.options["print_global"] and self.options["print_problem"]
        )
        self.printer._title("Problem size")
        self.printer("   %-25s : %i" % ("# training points.", n_exact))
        self.printer()

        self.printer.active = (
            self.options["print_global"] and self.options["print_training"]
        )
        if self.name == "MixExp":
            # Mixture of experts model
            self.printer._title("Training of the Mixture of experts")
        else:
            self.printer._title("Training")

        # Train the model using the specified model-method
        with self.printer._timed_context("Training", "training"):
            self._train()

    def predict_values(self, x):
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
        x = check_2d_array(x, "x")
        check_nx(self.nx, x)
        n = x.shape[0]
        self.printer.active = (
            self.options["print_global"] and self.options["print_prediction"]
        )

        if self.name == "MixExp":
            # Mixture of experts model
            self.printer._title("Evaluation of the Mixture of experts")
        else:
            self.printer._title("Evaluation")
        self.printer("   %-12s : %i" % ("# eval points.", n))
        self.printer()

        # Evaluate the unknown points using the specified model-method
        with self.printer._timed_context("Predicting", key="prediction"):
            y = self._predict_values(x)

        time_pt = self.printer._time("prediction")[-1] / n
        self.printer()
        self.printer("Prediction time/pt. (sec) : %10.7f" % time_pt)
        self.printer()
        return y.reshape((n, self.ny))

    def predict_derivatives(self, x, kx):
        """
        Predict the dy_dx derivatives at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        dy_dx : np.ndarray[nt, ny]
            Derivatives.
        """
        check_support(self, "derivatives")
        x = check_2d_array(x, "x")
        check_nx(self.nx, x)
        n = x.shape[0]
        self.printer.active = (
            self.options["print_global"] and self.options["print_prediction"]
        )

        if self.name == "MixExp":
            # Mixture of experts model
            self.printer._title("Evaluation of the Mixture of experts")
        else:
            self.printer._title("Evaluation")
        self.printer("   %-12s : %i" % ("# eval points.", n))
        self.printer()

        # Evaluate the unknown points using the specified model-method
        with self.printer._timed_context("Predicting", key="prediction"):
            y = self._predict_derivatives(x, kx)

        time_pt = self.printer._time("prediction")[-1] / n
        self.printer()
        self.printer("Prediction time/pt. (sec) : %10.7f" % time_pt)
        self.printer()

        return y.reshape((n, self.ny))

    def predict_output_derivatives(self, x):
        """
        Predict the derivatives dy_dyt at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.

        Returns
        -------
        dy_dyt : dict of np.ndarray[nt, nt]
            Dictionary of output derivatives.
            Key is None for derivatives wrt yt and kx for derivatives wrt dyt_dxt.
        """
        check_support(self, "output_derivatives")
        check_nx(self.nx, x)

        dy_dyt = self._predict_output_derivatives(x)
        return dy_dyt

    def predict_variances(self, x):
        """
        Predict the variances at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.

        Returns
        -------
        s2 : np.ndarray[nt, ny]
            Variances.
        """
        check_support(self, "variances")
        check_nx(self.nx, x)
        n = x.shape[0]
        s2 = self._predict_variances(x)
        return s2.reshape((n, self.ny))

    def _initialize(self):
        """
        Implemented by surrogate models to declare options and declare what they support (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        self.supports['derivatives'] = True
        """
        pass

    def _train(self):
        """
        Implemented by surrogate models to perform training (optional, but typically implemented).
        """
        pass

    def _predict_values(self, x):
        """
        Implemented by surrogate models to predict the output values.

        Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.

        Returns
        -------
        y : np.ndarray[nt, ny]
            Output values at the prediction points.
        """
        raise Exception("This surrogate model is incorrectly implemented")

    def _predict_derivatives(self, x, kx):
        """
        Implemented by surrogate models to predict the dy_dx derivatives (optional).

        If this method is implemented, the surrogate model should have

        ::
            self.supports['derivatives'] = True

        in the _initialize() implementation.

        Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        dy_dx : np.ndarray[nt, ny]
            Derivatives.
        """
        check_support(self, "derivatives", fail=True)

    def _predict_output_derivatives(self, x):
        """
        Implemented by surrogate models to predict the dy_dyt derivatives (optional).

        If this method is implemented, the surrogate model should have

        ::
            self.supports['output_derivatives'] = True

        in the _initialize() implementation.

        Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.

        Returns
        -------
        dy_dyt : dict of np.ndarray[nt, nt]
            Dictionary of output derivatives.
            Key is None for derivatives wrt yt and kx for derivatives wrt dyt_dxt.
        """
        check_support(self, "output_derivatives", fail=True)

    def _predict_variances(self, x):
        """
        Implemented by surrogate models to predict the variances at a set of points (optional).

        If this method is implemented, the surrogate model should have

        ::
            self.supports['variances'] = True

        in the _initialize() implementation.

        Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.

        Returns
        -------
        s2 : np.ndarray[nt, ny]
            Variances.
        """
        check_support(self, "variances", fail=True)
