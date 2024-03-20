"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Base class for benchmarking/test problems.
"""

from typing import Optional

import numpy as np

from smt.utils.checks import ensure_2d_array
from smt.utils.design_space import BaseDesignSpace, DesignSpace
from smt.utils.options_dictionary import OptionsDictionary


class Problem:
    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.

        For the list of options, see the documentation for the problem being used.

        Parameters
        ----------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.

        Examples
        --------
        >>> from smt.problems import Sphere
        >>> prob = Sphere(ndim=3)
        """
        self.options = OptionsDictionary()
        self.options.declare("ndim", 1, types=int)
        self.options.declare("return_complex", False, types=bool)
        self._initialize()
        self.options.update(kwargs)

        self.xlimits = np.zeros((self.options["ndim"], 2))
        self._design_space = None

        self.eval_x = None
        self.eval_is_acting = None

        self._setup()

    def _initialize(self) -> None:
        """
        Implemented by problem to declare options (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass

    def _setup(self) -> None:
        pass

    def _set_design_space(self, design_space: BaseDesignSpace):
        """
        Set the design space definition (best is to use the smt.utils.design_space.DesignSpace class directly) of
        this problem from the _setup function. If used, there is no need to set xlimits.
        """
        self._design_space = design_space
        self.options["ndim"] = len(design_space.design_variables)
        self.xlimits = design_space.get_num_bounds()

    @property
    def design_space(self) -> BaseDesignSpace:
        """Gets the design space definitions as an instance of BaseDesignSpace"""
        if self._design_space is None:
            self._design_space = DesignSpace(self.xlimits)
        return self._design_space

    def sample(self, n):
        x, _ = self.design_space.sample_valid_x(n)
        return x

    def __call__(self, x: np.ndarray, kx: Optional[int] = None) -> np.ndarray:
        """
        Evaluate the function.
        The input vectors might be corrected if it is a hierarchical design space. You can get the corrected x and
        information about which variables are acting from: problem.eval_x and problem.eval_is_acting

        Parameters
        ----------
        x : ndarray[n, nx] or ndarray[n]
            Evaluation points where n is the number of evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[n, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        x = ensure_2d_array(x, "x")

        if x.shape[1] != self.options["ndim"]:
            raise ValueError(
                "The second dimension of x should be %i" % self.options["ndim"]
            )

        if kx is not None:
            if not isinstance(kx, int) or kx < 0:
                raise TypeError("kx should be None or a non-negative int.")

        # Correct the design vector and get information about which design variables are active
        x_corr, self.eval_is_acting = self.design_space.correct_get_acting(x)
        self.eval_x = x_corr

        y = self._evaluate(x_corr, kx)

        if self.options["return_complex"]:
            return y
        else:
            return np.real(y)

    def _evaluate(self, x: np.ndarray, kx: Optional[int] = None) -> np.ndarray:
        """
        Implemented by surrogate models to evaluate the function.

        Parameters
        ----------
        x : ndarray[n, nx]
            Evaluation points where n is the number of evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[n, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        raise Exception("This problem has not been implemented correctly")
